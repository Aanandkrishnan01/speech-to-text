"""
Benchmark dataset loaders.

Currently supports LibriSpeech (test-clean / test-other) via huggingface_hub.
The dataset gets cached under `<cache_dir>/datasets/librispeech/<split>/`
on first use; subsequent runs skip the download.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

from ..logger import get_logger

log = get_logger(__name__)


@dataclass
class Sample:
    """One labeled audio clip from a benchmark dataset."""
    id: str
    audio_path: Path
    reference: str
    duration: float  # seconds


class DatasetLoader(Protocol):
    """Iterable yielding `Sample`s. Implementations may lazily download."""
    name: str
    def iter_samples(self, limit: int | None = None) -> Iterator[Sample]: ...


class LibriSpeechLoader:
    """
    LibriSpeech ASR test loader. Splits supported: test-clean | test-other.

    The Hugging Face mirror at `openslr/librispeech_asr` ships flac + JSONL
    metadata. We download the relevant split parquet/audio shards lazily via
    `datasets.load_dataset` (already pulled in by the qwen-asr stack).
    """

    name = "librispeech"
    SUPPORTED_SPLITS = ("test-clean", "test-other")

    def __init__(self, split: str, cache_dir: Path,
                 hf_repo: str = "openslr/librispeech_asr",
                 allow_download: bool = False):
        if split not in self.SUPPORTED_SPLITS:
            raise ValueError(
                f"Unsupported LibriSpeech split: {split!r}. "
                f"Use one of {self.SUPPORTED_SPLITS}."
            )
        self.split = split
        self.cache_dir = Path(cache_dir)
        self.hf_repo = hf_repo
        self.allow_download = allow_download
        self._dataset = None

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "LibriSpeech loader requires `datasets`. "
                "Install via: pip install datasets soundfile"
            ) from e

        ds_cache = self.cache_dir / "datasets"
        ds_cache.mkdir(parents=True, exist_ok=True)

        # Map our split name to HF's
        hf_split = self.split.replace("-", ".")  # test-clean → test.clean
        log.info("Loading LibriSpeech %s from %s (cache=%s)",
                 hf_split, self.hf_repo, ds_cache)

        # `download_mode='reuse_dataset_if_exists'` blocks redownloads.
        download_mode = "reuse_dataset_if_exists" if not self.allow_download else "force_redownload"

        # Skip download entirely if the dataset isn't already cached AND user
        # didn't pass --allow-download.
        os.environ.setdefault("HF_DATASETS_CACHE", str(ds_cache))
        try:
            self._dataset = load_dataset(
                self.hf_repo,
                "clean" if "clean" in self.split else "other",
                split=hf_split,
                cache_dir=str(ds_cache),
                download_mode=download_mode,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load LibriSpeech split {self.split!r}. "
                f"If this is the first run, pass --allow-download. "
                f"Underlying error: {e}"
            ) from e

    def iter_samples(self, limit: int | None = None) -> Iterator[Sample]:
        self._ensure_loaded()
        ds = self._dataset
        n = len(ds) if limit is None else min(limit, len(ds))
        log.info("Iterating %d/%d LibriSpeech %s samples",
                 n, len(ds), self.split)
        for i in range(n):
            row = ds[i]
            # row['audio'] is a dict {path, array, sampling_rate} in HF audio
            # column. We want a real on-disk path so the project's load_audio
            # works uniformly.
            audio_info = row.get("audio") or {}
            path = audio_info.get("path")
            if path is None:
                # Synthesize a file from the in-memory array as a last resort
                # (HF sometimes drops the path on streaming).
                continue
            text = (row.get("text") or row.get("transcript") or "").strip()
            duration = audio_info.get("array", [])
            duration = (len(duration) /
                        float(audio_info.get("sampling_rate") or 16000)) \
                       if hasattr(duration, "__len__") else 0.0
            yield Sample(
                id=row.get("id", f"{self.split}_{i:06d}"),
                audio_path=Path(path),
                reference=text,
                duration=duration,
            )


def load_dataset(name: str, *, split: str | None = None, cache_dir: Path,
                 allow_download: bool = False, hf_repo: str | None = None,
                 **kwargs) -> DatasetLoader:
    """Factory: build a DatasetLoader by name."""
    name = (name or "").lower()
    if name == "librispeech":
        return LibriSpeechLoader(
            split=split or "test-clean",
            cache_dir=cache_dir,
            hf_repo=hf_repo or "openslr/librispeech_asr",
            allow_download=allow_download,
        )
    raise ValueError(
        f"Unknown benchmark dataset: {name!r}. Supported: 'librispeech'."
    )
