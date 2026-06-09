"""
ChunkedASRStream — adapts any chunked `ASRBackend` (Qwen, Whisper, NeMo,
Gemini, Google STT batch) to the streaming `ASRStream` protocol.

Strategy:

- Per-utterance audio buffer accumulates frames.
- Every `interim_interval_ms`, transcribe the **last `interim_window_sec`
  seconds** of the buffer (not the whole thing) — keeps interim ASR cost
  bounded so slow backends like Whisper-on-CPU don't pile up.
- If the backend is still busy with a previous interim, skip this tick
  rather than queueing — prevents tail latency from compounding.
- On `end_utterance()`, transcribe the **full** utterance for an accurate
  final, then preserve any frames that arrived during the ASR call so they
  start the NEXT utterance instead of being thrown away.

This trades a small amount of interim-text instability (each interim sees
only the latest few seconds and may not have full context) for backend
portability + bounded latency. True streaming-native backends in Phase 2
won't need this adapter.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np

from ..asr_backends import ASRBackend
from ..asr_stream import StreamingResult, new_utterance_id, now_ms
from ..audio_utils import detect_silence
from ..logger import get_logger

log = get_logger(__name__)


@dataclass
class ChunkedASRStream:
    """Wraps an `ASRBackend` and exposes the streaming protocol."""

    backend: ASRBackend
    sample_rate: int = 16000
    interim_interval_ms: int = 500
    max_utterance_sec: float = 20.0
    min_utterance_sec: float = 0.5
    silence_threshold: float = 0.004
    # Sliding window for INTERIM transcriptions only — keeps cost bounded.
    # Final still uses the whole utterance buffer.
    interim_window_sec: float = 5.0

    name: str = "chunked"
    model_name: str = ""

    _utterance_pcm: list[np.ndarray] = None  # type: ignore[assignment]
    _utterance_samples: int = 0
    _utterance_start_offset: float = 0.0
    _utterance_id: str = ""
    _last_interim_ms: float = 0.0
    _last_interim_text: str = ""
    _absolute_offset: float = 0.0
    _busy: bool = False  # True while a backend call is in flight

    def __post_init__(self) -> None:
        self.name = f"chunked-{self.backend.name}"
        self.model_name = self.backend.model_name
        self._utterance_pcm = []
        self._reset_utterance()

    # ── Protocol ────────────────────────────────────────────────────────

    async def feed(self, pcm: np.ndarray) -> list[StreamingResult]:
        """Append PCM, run a (sliding-window) interim transcription if it's
        time and the backend isn't busy. Auto-flush the utterance as final
        if it grows past max_utterance_sec without an explicit end."""
        if pcm.size == 0:
            return []
        if pcm.dtype != np.float32:
            pcm = pcm.astype(np.float32)
        if pcm.ndim > 1:
            pcm = pcm.flatten()

        self._utterance_pcm.append(pcm)
        self._utterance_samples += pcm.size
        emissions: list[StreamingResult] = []
        utterance_dur = self._utterance_samples / float(self.sample_rate)

        if utterance_dur < self.min_utterance_sec:
            return emissions

        # Safety valve — VAD endpointer never fired for this utterance.
        if utterance_dur >= self.max_utterance_sec:
            log.info("Stream %s force-flushing utterance after %.1fs",
                     self._utterance_id, utterance_dur)
            emissions.extend(await self.end_utterance())
            return emissions

        # Interim tick? Only if the prior call has finished (no pile-up).
        now = now_ms()
        time_since_last = now - self._last_interim_ms
        if time_since_last < self.interim_interval_ms:
            return emissions
        if self._busy:
            # Backend is still processing the last interim. Skip this tick
            # entirely — wait for the next interval to try again. This
            # prevents the audio queue from filling up behind a slow ASR.
            return emissions

        self._last_interim_ms = now

        # Skip ASR on silent windows to avoid hallucinating "Okay." etc.
        if self._utterance_is_silent():
            return emissions

        text = await self._transcribe_window(self.interim_window_sec)
        if text and text != self._last_interim_text:
            self._last_interim_text = text
            emissions.append(StreamingResult(
                id=self._utterance_id,
                text=text,
                is_final=False,
                start=self._utterance_start_offset,
                duration=utterance_dur,
            ))

        return emissions

    async def end_utterance(self) -> list[StreamingResult]:
        """Run a FINAL transcription on the full utterance buffer, emit it,
        and start a fresh utterance — preserving any frames that arrived
        during the ASR call so they don't get thrown away."""
        if self._utterance_samples == 0:
            return []

        if self._utterance_is_silent():
            # Whole utterance is silent → discard, no final
            self._reset_utterance()
            return []

        # Snapshot the current buffer's COUNT before awaiting. New frames
        # arriving during the await will be appended past this index.
        snapshot_count = len(self._utterance_pcm)
        snapshot_samples = self._utterance_samples
        snapshot_id = self._utterance_id
        snapshot_start = self._utterance_start_offset

        # Concatenate just the snapshot — frames that arrive after this
        # point go to _utterance_pcm[snapshot_count:] and survive the reset
        # below.
        snapshot_pcm = np.concatenate(self._utterance_pcm[:snapshot_count])

        text = await self._transcribe_audio(snapshot_pcm)

        snapshot_dur = snapshot_samples / float(self.sample_rate)
        result = StreamingResult(
            id=snapshot_id,
            text=text or "",
            is_final=True,
            start=snapshot_start,
            duration=snapshot_dur,
        )

        # Frames that came in during the await — keep them, restart utterance
        leftover = self._utterance_pcm[snapshot_count:]
        leftover_samples = sum(p.size for p in leftover)

        self._absolute_offset = snapshot_start + snapshot_dur
        self._utterance_pcm = leftover
        self._utterance_samples = leftover_samples
        self._utterance_id = new_utterance_id()
        self._utterance_start_offset = self._absolute_offset
        self._last_interim_ms = 0.0
        self._last_interim_text = ""

        return [result]

    async def close(self) -> None:
        return None

    # ── Internals ───────────────────────────────────────────────────────

    def _utterance_is_silent(self) -> bool:
        """Energy-based silence check on the most-recent ~1.5s of the buffer.
        When True, the ASR call is skipped — saves compute and avoids most
        hallucinations on pure silence. (Energy-only — no VAD model loaded.)
        """
        if self._utterance_samples == 0:
            return True
        sample_window = int(1.5 * self.sample_rate)
        last = self._utterance_pcm[-1]
        if sample_window <= last.size:
            recent = last
        else:
            recent = np.concatenate(self._utterance_pcm)[-sample_window:]
        return detect_silence(recent, self.silence_threshold)

    def _reset_utterance(self) -> None:
        self._utterance_pcm = []
        self._utterance_samples = 0
        self._utterance_id = new_utterance_id()
        self._utterance_start_offset = self._absolute_offset
        self._last_interim_ms = 0.0
        self._last_interim_text = ""

    async def _transcribe_window(self, window_sec: float) -> str:
        """Transcribe ONLY the last `window_sec` of the utterance buffer.
        Used for interims — keeps ASR cost bounded as the utterance grows."""
        if self._utterance_samples == 0:
            return ""
        n = int(window_sec * self.sample_rate)
        # Pick from the tail end — last `n` samples
        if self._utterance_samples <= n:
            audio = np.concatenate(self._utterance_pcm)
        else:
            audio = np.concatenate(self._utterance_pcm)[-n:]
        return await self._transcribe_audio(audio)

    async def _transcribe_audio(self, audio: np.ndarray) -> str:
        """Run the backend on the given audio, off the asyncio thread."""
        if audio.size == 0:
            return ""
        self._busy = True
        try:
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(
                None, self.backend.transcribe, audio
            )
        except Exception:
            log.exception("ChunkedASRStream %s transcribe failed",
                          self._utterance_id)
            text = ""
        finally:
            self._busy = False
        return (text or "").strip()
