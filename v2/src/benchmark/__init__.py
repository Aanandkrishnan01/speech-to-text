"""Benchmark harness — runs an ASR backend over a labeled dataset and produces WER + gap reports."""

from .datasets import DatasetLoader, LibriSpeechLoader, Sample, load_dataset
from .evaluator import Evaluator, SampleResult
from .report import write_json, write_markdown_gap_assessment

__all__ = [
    "DatasetLoader", "LibriSpeechLoader", "Sample", "load_dataset",
    "Evaluator", "SampleResult",
    "write_json", "write_markdown_gap_assessment",
]
