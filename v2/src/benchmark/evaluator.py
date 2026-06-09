"""
Benchmark evaluator — runs an ASRBackend over a DatasetLoader, computes WER
and friends per-sample, and caches transcripts so reruns are fast.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..asr_backends import ASRBackend
from ..audio_utils import load_audio, segment_to_float32
from ..logger import get_logger
from .datasets import DatasetLoader, Sample

log = get_logger(__name__)


@dataclass
class SampleResult:
    """Per-sample evaluation outcome."""
    id: str
    reference: str
    hypothesis: str
    wer: float       # word error rate
    cer: float       # character error rate
    mer: float       # match error rate
    wil: float       # word information lost
    duration: float  # seconds
    latency_ms: float  # transcribe call wall time


def _normalize(text: str) -> str:
    """Light normalization for fair WER comparison."""
    import re
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)  # punctuation → space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _score(reference: str, hypothesis: str) -> tuple[float, float, float, float]:
    """Returns (WER, CER, MER, WIL) using jiwer."""
    import jiwer
    ref = _normalize(reference)
    hyp = _normalize(hypothesis)
    if not ref:
        # No reference → can't score; return 1.0 across the board for clarity
        return (1.0, 1.0, 1.0, 1.0)
    if not hyp:
        return (1.0, 1.0, 1.0, 1.0)
    measures = jiwer.compute_measures(ref, hyp)
    return (
        float(measures["wer"]),
        float(jiwer.cer(ref, hyp)),
        float(measures["mer"]),
        float(measures["wil"]),
    )


def _cache_key(backend_name: str, model_name: str, sample: Sample) -> str:
    """Stable hash of (backend, model, sample id, file size, mtime)."""
    try:
        st = sample.audio_path.stat()
        size = st.st_size
        mtime = int(st.st_mtime)
    except OSError:
        size = -1
        mtime = -1
    s = f"{backend_name}|{model_name}|{sample.id}|{size}|{mtime}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


class Evaluator:
    """Run a backend over a dataset, scoring with jiwer; cache to disk."""

    def __init__(self, backend: ASRBackend, cache_dir: Path,
                 sample_rate: int = 16000, use_cache: bool = True):
        self.backend = backend
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.use_cache = use_cache

    def run(self, loader: DatasetLoader,
            limit: int | None = None) -> list[SampleResult]:
        results: list[SampleResult] = []
        for sample in loader.iter_samples(limit=limit):
            try:
                hyp, latency_ms = self._transcribe_cached(sample)
            except Exception:
                log.exception("Eval failed on %s; recording empty hypothesis",
                              sample.id)
                hyp, latency_ms = "", 0.0

            wer, cer, mer, wil = _score(sample.reference, hyp)
            r = SampleResult(
                id=sample.id,
                reference=sample.reference,
                hypothesis=hyp,
                wer=wer, cer=cer, mer=mer, wil=wil,
                duration=sample.duration,
                latency_ms=latency_ms,
            )
            results.append(r)
            log.info("[%s] WER=%.3f CER=%.3f dur=%.1fs lat=%.0fms ref=%r hyp=%r",
                     sample.id, wer, cer, sample.duration, latency_ms,
                     sample.reference[:60], hyp[:60])
        return results

    # ----- Internals -----------------------------------------------------

    def _transcribe_cached(self, sample: Sample) -> tuple[str, float]:
        key = _cache_key(self.backend.name, self.backend.model_name, sample)
        cache_file = self.cache_dir / f"{key}.json"
        if self.use_cache and cache_file.exists():
            try:
                payload = json.loads(cache_file.read_text())
                log.debug("cache hit %s", cache_file.name)
                return payload["hypothesis"], float(payload.get("latency_ms", 0.0))
            except Exception:
                log.warning("cache read failed for %s; re-transcribing",
                            cache_file.name)

        # Load audio + transcribe
        audio = load_audio(sample.audio_path, self.sample_rate)
        pcm = segment_to_float32(audio)
        t0 = time.perf_counter()
        hyp = self.backend.transcribe(pcm)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if self.use_cache:
            try:
                cache_file.write_text(json.dumps({
                    "id": sample.id,
                    "backend": self.backend.name,
                    "model": self.backend.model_name,
                    "hypothesis": hyp,
                    "latency_ms": latency_ms,
                    "reference": sample.reference,
                    "duration": sample.duration,
                }, indent=2))
            except Exception:
                log.exception("cache write failed for %s", cache_file.name)
        return hyp, latency_ms
