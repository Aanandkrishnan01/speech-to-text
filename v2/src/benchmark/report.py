"""Benchmark report writers — JSON detail + Markdown gap assessment."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .evaluator import SampleResult


def write_json(results: list[SampleResult], path: Path,
               meta: dict[str, Any]) -> None:
    """Write a machine-readable JSON dump. `meta` carries backend/model/dataset/etc."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "meta": meta,
        "summary": _summary(results),
        "samples": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(payload, indent=2))


def write_markdown_gap_assessment(results: list[SampleResult], path: Path,
                                  meta: dict[str, Any],
                                  worst_n: int = 10) -> None:
    """Write a human-readable Markdown report focused on accuracy gaps."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = _summary(results)

    lines: list[str] = []
    lines.append(f"# ASR Gap Assessment\n")
    lines.append(f"_Generated {datetime.now().isoformat(timespec='seconds')}_\n")
    lines.append("## Setup\n")
    for k, v in meta.items():
        lines.append(f"- **{k}**: `{v}`")
    lines.append("")

    lines.append("## Aggregate scores\n")
    lines.append("| metric | value |")
    lines.append("| ------ | ----- |")
    lines.append(f"| samples | {summary['count']} |")
    lines.append(f"| total audio (s) | {summary['total_duration']:.1f} |")
    lines.append(f"| WER (mean) | {summary['mean_wer']:.4f} |")
    lines.append(f"| CER (mean) | {summary['mean_cer']:.4f} |")
    lines.append(f"| MER (mean) | {summary['mean_mer']:.4f} |")
    lines.append(f"| WIL (mean) | {summary['mean_wil']:.4f} |")
    lines.append(f"| latency p50 (ms) | {summary['p50_latency']:.0f} |")
    lines.append(f"| latency p90 (ms) | {summary['p90_latency']:.0f} |")
    lines.append("")

    lines.append("## Error-band distribution\n")
    bands = summary["bands"]
    lines.append("| WER band | count | %% |")
    lines.append("| -------- | ----- | -- |")
    for band, n in bands.items():
        pct = (100.0 * n / max(1, summary["count"]))
        lines.append(f"| {band} | {n} | {pct:.1f} |")
    lines.append("")

    lines.append(f"## Worst {worst_n} samples\n")
    worst = sorted(results, key=lambda r: r.wer, reverse=True)[:worst_n]
    if not worst:
        lines.append("_No samples to display._")
    else:
        lines.append("| id | WER | duration | reference → hypothesis |")
        lines.append("| -- | ---:| -------:| ----------------------- |")
        for r in worst:
            ref = (r.reference or "")[:80].replace("|", "/")
            hyp = (r.hypothesis or "")[:80].replace("|", "/")
            lines.append(
                f"| `{r.id}` | {r.wer:.3f} | {r.duration:.1f}s | "
                f"**ref:** {ref!r}<br>**hyp:** {hyp!r} |"
            )

    path.write_text("\n".join(lines) + "\n")


def _summary(results: list[SampleResult]) -> dict[str, Any]:
    if not results:
        return {
            "count": 0, "total_duration": 0.0,
            "mean_wer": 0.0, "mean_cer": 0.0, "mean_mer": 0.0, "mean_wil": 0.0,
            "p50_latency": 0.0, "p90_latency": 0.0,
            "bands": {"0-5%": 0, "5-15%": 0, "15-30%": 0, ">30%": 0},
        }

    def mean(xs: Iterable[float]) -> float:
        xs = list(xs)
        return sum(xs) / max(1, len(xs))

    def percentile(xs: list[float], p: float) -> float:
        if not xs:
            return 0.0
        xs = sorted(xs)
        k = max(0, min(len(xs) - 1, int(round(p / 100.0 * (len(xs) - 1)))))
        return xs[k]

    bands = {"0-5%": 0, "5-15%": 0, "15-30%": 0, ">30%": 0}
    for r in results:
        if r.wer < 0.05: bands["0-5%"] += 1
        elif r.wer < 0.15: bands["5-15%"] += 1
        elif r.wer < 0.30: bands["15-30%"] += 1
        else: bands[">30%"] += 1

    latencies = [r.latency_ms for r in results]
    return {
        "count": len(results),
        "total_duration": sum(r.duration for r in results),
        "mean_wer": mean(r.wer for r in results),
        "mean_cer": mean(r.cer for r in results),
        "mean_mer": mean(r.mer for r in results),
        "mean_wil": mean(r.wil for r in results),
        "p50_latency": percentile(latencies, 50),
        "p90_latency": percentile(latencies, 90),
        "bands": bands,
    }
