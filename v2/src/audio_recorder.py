"""
Audio capture persistence.

Two pieces:

- `RollingMP3Writer` — append PCM frames from a hot audio path; encode in a
  background thread (so the audio callback isn't blocked by ffmpeg); roll over
  to a new file when the current part exceeds `max_bytes`. Falls back to WAV
  if MP3 encoding fails (e.g., ffmpeg missing). Used by realtime CLI mode and
  by the per-WebSocket Session in the web server.

- `chunk_audio_file(path, window_sec, sample_rate)` — generator that streams
  a long audio file in fixed-duration windows. Used by batch mode to keep
  memory bounded for hour-long inputs.
"""

from __future__ import annotations

import io
import queue
import threading
import wave
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from pydub import AudioSegment

from .audio_utils import load_audio, segment_to_float32
from .logger import get_logger

log = get_logger(__name__)

# Sentinel pushed onto the queue to ask the writer thread to flush + exit.
_STOP = object()


class RollingMP3Writer:
    """
    Append-only audio recorder that rolls over at a configured size.

    Lifecycle:
        rec = RollingMP3Writer(Path("audio/realtime_run"), sample_rate=16000)
        rec.write(pcm_chunk_a)   # callable from audio callback (non-blocking)
        rec.write(pcm_chunk_b)
        rec.close()              # flush, join, return final part path

    The encoder runs in a daemon thread; `write()` only enqueues. If the queue
    fills (encoder slower than producer), the producer briefly blocks — better
    than losing audio. On close, all pending frames are flushed to disk.
    """

    def __init__(self, base_path: Path, sample_rate: int,
                 max_bytes: int = 10 * 1024 * 1024,
                 bitrate_kbps: int = 128,
                 fmt: str = "mp3"):
        self.base_path = Path(base_path)
        self.base_path.parent.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.max_bytes = max_bytes
        self.bitrate = f"{bitrate_kbps}k"
        self.fmt = fmt.lower()
        if self.fmt not in ("mp3", "wav"):
            raise ValueError(f"Unsupported recording format: {fmt}")
        self._parts: list[Path] = []
        self._part_idx = 0
        self._closed = False

        # Pending PCM samples in the *current* part (still in memory until flush).
        # We flush a part when its in-memory PCM has reached an estimated size
        # (after MP3 encoding) of `max_bytes`. The estimate uses the configured
        # bitrate — for WAV we use raw 16-bit mono size.
        self._pending: list[np.ndarray] = []
        self._pending_samples = 0

        # Background thread + queue for non-blocking encode.
        self._q: "queue.Queue[object]" = queue.Queue(maxsize=256)
        self._thread = threading.Thread(
            target=self._run, name=f"rec-{self.base_path.name}", daemon=True
        )
        self._thread.start()

    # ----- Public API ----------------------------------------------------

    def write(self, pcm_float32: np.ndarray) -> None:
        """Enqueue a PCM chunk. Non-blocking from the producer's perspective."""
        if self._closed:
            return
        if pcm_float32.size == 0:
            return
        if pcm_float32.dtype != np.float32:
            pcm_float32 = pcm_float32.astype(np.float32, copy=False)
        if pcm_float32.ndim > 1:
            pcm_float32 = pcm_float32.flatten()
        # Copy so the caller is free to mutate the source buffer.
        self._q.put(pcm_float32.copy())

    def close(self) -> Optional[Path]:
        """Flush remaining audio, stop the writer thread, return last part."""
        if self._closed:
            return self._parts[-1] if self._parts else None
        self._closed = True
        self._q.put(_STOP)
        self._thread.join(timeout=15.0)
        if self._thread.is_alive():
            log.warning("RollingMP3Writer: encoder thread did not exit in time")
        return self._parts[-1] if self._parts else None

    @property
    def parts(self) -> list[Path]:
        return list(self._parts)

    # ----- Background thread --------------------------------------------

    def _run(self) -> None:
        try:
            while True:
                item = self._q.get()
                if item is _STOP:
                    self._flush_part(final=True)
                    return
                self._pending.append(item)
                self._pending_samples += len(item)
                if self._estimated_bytes() >= self.max_bytes:
                    self._flush_part(final=False)
        except Exception:
            log.exception("RollingMP3Writer: encoder thread crashed")

    def _estimated_bytes(self) -> int:
        """Estimate the on-disk size of the currently-pending audio."""
        seconds = self._pending_samples / float(self.sample_rate)
        if self.fmt == "wav":
            return int(seconds * self.sample_rate * 2)  # 16-bit mono
        # MP3: bytes ≈ bitrate_bps * seconds / 8
        bitrate_bps = int(self.bitrate.rstrip("k")) * 1000
        return int(bitrate_bps * seconds / 8)

    def _flush_part(self, final: bool) -> None:
        if self._pending_samples == 0:
            return
        self._part_idx += 1
        part_path = self.base_path.with_name(
            f"{self.base_path.name}_{self._part_idx:03d}.{self.fmt}"
        )

        # Concatenate pending frames into a single int16 buffer
        pcm_float = np.concatenate(self._pending)
        self._pending.clear()
        self._pending_samples = 0

        int16 = (np.clip(pcm_float, -1.0, 1.0) * 32767.0).astype(np.int16)

        try:
            if self.fmt == "mp3":
                self._write_mp3(part_path, int16)
            else:
                self._write_wav(part_path, int16)
            self._parts.append(part_path)
            log.info("Recorder wrote %s (%.1fs, %d bytes)",
                     part_path.name,
                     len(int16) / float(self.sample_rate),
                     part_path.stat().st_size)
        except Exception as e:
            # Fallback to WAV if MP3 fails (likely ffmpeg missing).
            if self.fmt == "mp3":
                log.warning("MP3 export failed (%s) — falling back to WAV", e)
                wav_path = part_path.with_suffix(".wav")
                try:
                    self._write_wav(wav_path, int16)
                    self._parts.append(wav_path)
                    self.fmt = "wav"
                except Exception:
                    log.exception("WAV fallback also failed")
            else:
                log.exception("Recorder flush failed for %s", part_path)

    def _write_wav(self, path: Path, int16: np.ndarray) -> None:
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(int16.tobytes())

    def _write_mp3(self, path: Path, int16: np.ndarray) -> None:
        seg = AudioSegment(
            int16.tobytes(),
            frame_rate=self.sample_rate,
            sample_width=2,
            channels=1,
        )
        seg.export(str(path), format="mp3", bitrate=self.bitrate)


def recording_base_path(prefix: str, asr_backend_name: str,
                        asr_model_name: str, recordings_dir: Path,
                        timestamp_format: str = "%d-%m-%y_%H-%M-%S") -> Path:
    """
    Build a recording base path like
        <dir>/<prefix>_<DD-MM-YY_HH-MM-SS>
    The actual files written by RollingMP3Writer will be
        <base>_001.mp3, <base>_002.mp3, ...

    `asr_backend_name` and `asr_model_name` are accepted for API
    compatibility but no longer included in the filename — the corresponding
    transcript file's header records that info. Keeps audio filenames short
    and chronological.
    """
    ts = datetime.now().strftime(timestamp_format)
    recordings_dir.mkdir(parents=True, exist_ok=True)
    return recordings_dir / f"{prefix}_{ts}"


# ---------------------------------------------------------------------------
# Batch-mode windowing
# ---------------------------------------------------------------------------

def chunk_audio_file(path: Path, window_sec: float,
                     sample_rate: int) -> Iterator[tuple[float, np.ndarray]]:
    """
    Stream an audio file in fixed-duration windows (in seconds).

    Yields (window_start_seconds, pcm_float32). The caller is expected to
    add `window_start_seconds` to any segment timestamps it produces, so
    the final transcript has correct absolute times.
    """
    audio = load_audio(path, sample_rate=sample_rate)
    total_ms = len(audio)
    window_ms = int(window_sec * 1000)
    if window_ms <= 0:
        raise ValueError("window_sec must be > 0")

    for start_ms in range(0, total_ms, window_ms):
        end_ms = min(start_ms + window_ms, total_ms)
        slice_seg = audio[start_ms:end_ms]
        pcm = segment_to_float32(slice_seg)
        yield start_ms / 1000.0, pcm
