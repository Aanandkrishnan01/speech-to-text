"""
Boundary-aware chunking helpers for streaming transcription.

The realtime + web pipelines need to cut audio into pieces small enough for
ASR latency targets, without slicing through the middle of a word. Two
strategies are supported:

- **VAD-aware** (default): given the VAD segments already produced by the
  diarization pass, pick the latest "speech ended" point that's at least
  `min_chunk_s` from the start. If no such point exists within `max_chunk_s`,
  force-cut at `max_chunk_s` so the pipeline doesn't stall on continuous
  noise/speech.
- **Overlap + dedup**: take a fixed window with `overlap_s` of overlap with
  the previous window; dedupe transcripts by suffix/prefix matching using
  RapidFuzz.

A third value (`"none"`) preserves the original behavior exactly — useful as
a regression-safety opt-out.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable

import numpy as np


class BoundaryStrategy(str, Enum):
    VAD = "vad"
    OVERLAP = "overlap"
    NONE = "none"


def find_endpoint(
    pcm_length_samples: int,
    sample_rate: int,
    vad_segments: Iterable[tuple[float, float]],
    min_chunk_s: float = 1.0,
    max_chunk_s: float = 4.0,
    min_silence_ms: int = 200,
) -> int | None:
    """
    Pick a sample index where it's safe to cut, based on VAD output.

    The VAD already returned `(start, end)` tuples in seconds. We pick the
    LATEST `end` that satisfies:
        min_chunk_s ≤ end ≤ max_chunk_s
        AND there's at least `min_silence_ms` of silence after it (i.e., the
            next VAD segment starts later than `end + min_silence_ms`, OR
            this is the final VAD segment AND there's enough buffered audio
            after it).

    Args:
        pcm_length_samples: total length of the buffered window we're cutting
        sample_rate: audio sample rate
        vad_segments: list of (start_s, end_s) tuples from VAD
        min_chunk_s: don't return a cut earlier than this
        max_chunk_s: don't return a cut later than this; caller force-cuts
                     here if this function returns None and the buffer is full
        min_silence_ms: required silence after the candidate end

    Returns:
        sample index where we can safely cut, or None if no safe cut found.
    """
    segs = sorted(vad_segments, key=lambda se: se[0])
    if not segs:
        return None

    pcm_total_s = pcm_length_samples / float(sample_rate)
    sil_s = min_silence_ms / 1000.0

    # Candidates: each segment end. Walk in reverse so we prefer the latest
    # acceptable cut point (more audio per chunk = better ASR context).
    for i in range(len(segs) - 1, -1, -1):
        end_s = segs[i][1]
        if end_s < min_chunk_s or end_s > max_chunk_s:
            continue
        # Verify there's sufficient silence after this end.
        if i + 1 < len(segs):
            next_start_s = segs[i + 1][0]
            gap = next_start_s - end_s
        else:
            # Last VAD segment — gap is "everything between end and pcm_total".
            gap = pcm_total_s - end_s
        if gap >= sil_s:
            return int(end_s * sample_rate)
    return None


def dedupe_overlap(prev_text: str, new_text: str,
                   min_ratio: int = 88, max_trim_words: int = 12) -> str:
    """
    Trim the leading words of `new_text` that overlap with the trailing words
    of `prev_text`, using RapidFuzz partial matching.

    Returns `new_text` with the overlap removed. Capped at `max_trim_words`
    to avoid over-trimming when the speaker repeats themselves intentionally
    (e.g., "very very very").

    If `prev_text` is empty, returns `new_text` unchanged.
    """
    if not prev_text or not new_text:
        return new_text or ""
    try:
        from rapidfuzz import fuzz
    except ImportError:
        return new_text

    prev_words = prev_text.split()
    new_words = new_text.split()
    if not prev_words or not new_words:
        return new_text

    # Walk the prefix of new_words (up to max_trim_words) and find the
    # longest one that fuzzy-matches a tail of prev_words at >= min_ratio.
    best_trim = 0
    upper = min(max_trim_words, len(prev_words), len(new_words))
    for k in range(1, upper + 1):
        prefix = " ".join(new_words[:k])
        suffix = " ".join(prev_words[-k:])
        if fuzz.ratio(prefix, suffix) >= min_ratio:
            best_trim = k

    if best_trim == 0:
        return new_text
    return " ".join(new_words[best_trim:])
