"""
Streaming ASR abstractions.

Production-style streaming: the session pushes audio frames in continuously,
the backend emits *interim* hypotheses as words arrive, and *final* results
when the VAD detects the end of an utterance. This decouples ASR from
chunk-based polling and matches the protocol production systems use
(Deepgram, AssemblyAI, Google Live Captions, etc).

Phase 1: the only stream implementation is `ChunkedASRStream`, which adapts
any existing `ASRBackend` (Qwen, Whisper, NeMo, Gemini, Google STT batch) to
this protocol by re-transcribing the rolling utterance buffer every N ms.
True streaming-native backends (`GoogleSTTStream`, `GeminiLiveStream`,
`NemoStreamingStream`) come in Phase 2.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol

import numpy as np

from .audio_utils import detect_silence
from .logger import get_logger

log = get_logger(__name__)


# ───────────────────────── Public types ──────────────────────────────────

@dataclass
class StreamingResult:
    """One emission from an ASRStream.

    Attributes
    ----------
    id : str
        Stable identifier for the *utterance* this result belongs to. Both
        interim and final emissions for the same utterance share this id, so
        the UI can update the same row in place.
    text : str
        Best-current hypothesis.
    is_final : bool
        True once the utterance is locked (VAD-detected endpoint, or
        max-utterance timeout). The text won't change after this.
    start : float
        Absolute start time of the utterance in seconds, relative to the
        session's audio offset.
    duration : float
        Duration of the audio so far in this utterance, in seconds.
    speaker : str | None
        Speaker label assigned by diarization. Only set on `is_final=True`
        results — interim emissions don't carry speaker info because
        diarization runs on the full utterance audio.
    confidence : float | None
        0..1 confidence if the backend provides it; None otherwise.
    """
    id: str
    text: str
    is_final: bool
    start: float
    duration: float
    speaker: str | None = None
    confidence: float | None = None


class ASRStream(Protocol):
    """Streaming ASR protocol. Implementations may be native-streaming
    (gRPC bidirectional, etc.) or wrappers over a chunked `ASRBackend`."""

    name: str
    model_name: str

    async def feed(self, pcm: np.ndarray) -> list[StreamingResult]:
        """Push a PCM chunk into the stream. Returns 0+ emissions
        (typically 0 most ticks, 1 interim every interim_interval, and 1
        final when end_utterance is triggered)."""
        ...

    async def end_utterance(self) -> list[StreamingResult]:
        """Caller (typically the VAD endpointer) signals that the current
        utterance is done. Stream flushes and emits a single final result.
        Resets internal state so the next feed() begins a new utterance."""
        ...

    async def close(self) -> None:
        """Tear down any open connections / cancel pending tasks."""
        ...


# ───────────────────────── VAD endpoint detector ─────────────────────────

class VADEndpointer:
    """
    Detects end-of-utterance from a rolling audio stream using simple energy
    thresholds. Cheap, no model load, runs on every audio frame.

    State machine:
        idle      ── audio above threshold ──▶  speaking
        speaking  ── audio below threshold for `min_silence_ms` ──▶  endpoint!

    The endpointer reports an "endpoint" event ONCE per speaking-to-idle
    transition, then resets to idle until the next speech burst.

    For higher-quality speech/non-speech discrimination (e.g., to ignore
    keyboard clicks), swap `_is_speech_frame` for a NeMo VAD call. We use
    energy here because it runs in microseconds and we call it on every
    audio frame.
    """

    def __init__(self, sample_rate: int,
                 silence_threshold: float = 0.004,
                 min_silence_ms: int = 600,
                 min_speech_ms: int = 200):
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms

        self._state: str = "idle"           # 'idle' | 'speaking'
        self._speech_ms: float = 0.0        # time in 'speaking' so far
        self._silence_ms: float = 0.0       # time of trailing silence within speaking

    def reset(self) -> None:
        self._state = "idle"
        self._speech_ms = 0.0
        self._silence_ms = 0.0

    def feed(self, pcm: np.ndarray) -> bool:
        """
        Feed one PCM chunk. Returns True if an utterance just ended (a single
        rising-edge event), False otherwise.

        The caller's pipeline is then responsible for calling
        `stream.end_utterance()` to flush a final result.
        """
        if pcm.size == 0:
            return False
        chunk_ms = (pcm.size / float(self.sample_rate)) * 1000.0
        is_speech = self._is_speech_frame(pcm)

        if self._state == "idle":
            if is_speech:
                self._state = "speaking"
                self._speech_ms = chunk_ms
                self._silence_ms = 0.0
            return False

        # state == "speaking"
        if is_speech:
            self._speech_ms += chunk_ms
            self._silence_ms = 0.0
            return False

        # silent frame within a speaking utterance
        self._silence_ms += chunk_ms
        if (self._silence_ms >= self.min_silence_ms and
                self._speech_ms >= self.min_speech_ms):
            # Endpoint reached — reset to idle, return True ONCE
            self._state = "idle"
            self._speech_ms = 0.0
            self._silence_ms = 0.0
            return True
        return False

    def is_speaking(self) -> bool:
        return self._state == "speaking"

    def _is_speech_frame(self, pcm: np.ndarray) -> bool:
        """Energy-based speech detection. RMS above threshold = speech."""
        return not detect_silence(pcm, self.silence_threshold)


# ───────────────────────── Utterance-id helper ───────────────────────────

def new_utterance_id() -> str:
    """Short stable id for an utterance — used as a key by the UI to
    overwrite the same active row with each interim emission."""
    return uuid.uuid4().hex[:10]


def now_ms() -> float:
    return time.perf_counter() * 1000.0
