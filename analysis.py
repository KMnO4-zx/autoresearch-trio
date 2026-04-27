#!/usr/bin/env python3
"""Summarize autoresearch runs and plot experiment progress.

Default usage:

    uv run analysis.py

This reads results.csv, skips crash rows without metrics, and writes a progress
plot to outputs/autoresearch_progress.png.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_RESULTS_CSV = Path("results.csv")
DEFAULT_OUTPUT = Path("outputs/autoresearch_progress.png")

HIGHER_IS_BETTER = {
    "ex",
    "soft_f1",
    "r_ves",
    "valid_sql_rate",
}

LOWER_IS_BETTER = {
    "unsafe_sql_rate",
    "latency_ms",
}

METRIC_LABELS = {
    "ex": "Execution Accuracy / EX",
    "soft_f1": "Soft F1",
    "r_ves": "R-VES",
    "valid_sql_rate": "Valid SQL Rate",
    "unsafe_sql_rate": "Unsafe SQL Rate",
    "latency_ms": "Latency (ms)",
}


@dataclass
class RunRow:
    experiment: int
    commit: str
    status: str
    description: str
    metric: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot autoresearch-trio experiment progress")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--metric",
        default="ex",
        choices=sorted(HIGHER_IS_BETTER | LOWER_IS_BETTER),
        help="metric to plot; default is the primary EX metric",
    )
    parser.add_argument("--title", default=None, help="override plot title")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--show-discard-labels", action="store_true")
    return parser.parse_args()


def is_better(metric: str, value: float, best: float) -> bool:
    if metric in LOWER_IS_BETTER:
        return value < best
    return value > best


def load_rows(path: Path, metric: str) -> list[RunRow]:
    rows: list[RunRow] = []
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")

    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            status = row.get("status", "")
            if status not in {"baseline", "keep", "discard"}:
                continue
            try:
                value = float(row.get(metric) or "")
            except ValueError:
                continue
            rows.append(
                RunRow(
                    experiment=len(rows) + 1,
                    commit=str(row.get("commit", "")),
                    status=status,
                    description=str(row.get("description", "")),
                    metric=value,
                )
            )

    if not rows:
        raise RuntimeError(f"No plottable rows with metric {metric!r} in {path}")
    return rows


def running_best_values(rows: list[RunRow], metric: str) -> list[float]:
    best = float("inf") if metric in LOWER_IS_BETTER else float("-inf")
    values: list[float] = []
    for row in rows:
        if is_better(metric, row.metric, best):
            best = row.metric
        values.append(best)
    return values


def best_row(rows: list[RunRow], metric: str) -> RunRow:
    key = (lambda row: -row.metric) if metric in LOWER_IS_BETTER else (lambda row: row.metric)
    return max(rows, key=key)


def y_limits(values: list[float]) -> tuple[float, float]:
    ymin = min(values)
    ymax = max(values)
    span = ymax - ymin
    pad = max(0.02, span * 0.22)
    lo = max(0.0, ymin - pad)
    hi = min(1.0, ymax + pad) if ymax <= 1.0 else ymax + pad
    if hi - lo < 0.08:
        mid = (hi + lo) / 2
        lo = max(0.0, mid - 0.04)
        hi = min(1.0, mid + 0.04) if ymax <= 1.0 else mid + 0.04
    return lo, hi


def compact_label(description: str) -> str:
    labels = {
        "baseline": "baseline",
        "shuffle training examples": "shuffle train order",
        "lower learning rate": "LR 1e-4->5e-5",
        "increase lora rank": "LoRA rank 16->32",
        "increase batch size to 8": "batch size 4->8",
        "lower learning rate further": "LR 5e-5->2.5e-5",
        "test intermediate learning rate": "LR 5e-5->7.5e-5",
        "reduce lora rank": "LoRA rank 16->8",
        "tighten sql prompt rules": "tighten prompt rules",
        "require semicolon output": "require semicolon",
    }
    return labels.get(description, description[:42])


def plot_progress(
    rows: list[RunRow],
    metric: str,
    output: Path,
    title: str | None,
    dpi: int,
    show_discard_labels: bool,
) -> None:
    kept = [row for row in rows if row.status == "keep"]
    green = [row for row in rows if row.status in {"baseline", "keep"}]
    discarded = [row for row in rows if row.status == "discard"]
    running_best = running_best_values(rows, metric)
    best = best_row(rows, metric)
    direction = "lower is better" if metric in LOWER_IS_BETTER else "higher is better"

    fig, ax = plt.subplots(figsize=(16, 7.2), dpi=dpi)

    if discarded:
        ax.scatter(
            [row.experiment for row in discarded],
            [row.metric for row in discarded],
            s=26,
            c="#cfcfcf",
            alpha=0.55,
            label="Discarded",
            zorder=2,
        )

    if green:
        ax.scatter(
            [row.experiment for row in green],
            [row.metric for row in green],
            s=70,
            c="#2ecc71",
            edgecolors="#166534",
            linewidths=1.2,
            label="Kept",
            zorder=4,
        )

    ax.step(
        [row.experiment for row in rows],
        running_best,
        where="post",
        color="#5ec98c",
        linewidth=2.2,
        label="Running best",
        zorder=3,
    )

    label_rows = green + discarded if show_discard_labels else green
    for i, row in enumerate(label_rows):
        color = "#2f8f55" if row.status in {"baseline", "keep"} else "#777777"
        ax.annotate(
            compact_label(row.description),
            xy=(row.experiment, row.metric),
            xytext=(8, 10 + (i % 2) * 8),
            textcoords="offset points",
            rotation=28,
            color=color,
            fontsize=10,
            ha="left",
            va="bottom",
        )

    plot_title = title or (
        f"Autoresearch-TRIO Progress: {len(rows)} Experiments, "
        f"{len(kept)} Kept Improvements"
    )
    ax.set_title(plot_title, fontsize=16)
    ax.set_xlabel("Experiment #", fontsize=13)
    ax.set_ylabel(f"{METRIC_LABELS[metric]} ({direction})", fontsize=13)
    ax.grid(True, color="#e9e9e9", linewidth=0.8)
    ax.set_xlim(0.5, len(rows) + 0.5)
    ax.set_ylim(*y_limits([row.metric for row in rows]))
    ax.legend(loc="best", frameon=True)
    ax.text(
        0.995,
        0.02,
        f"Best: {best.commit} / {metric}={best.metric:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#666",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def print_summary(rows: list[RunRow], metric: str, output: Path) -> None:
    best = best_row(rows, metric)
    kept_count = sum(row.status == "keep" for row in rows)
    print(f"experiments: {len(rows)}")
    print(f"kept: {kept_count}")
    print(f"best_commit: {best.commit}")
    print(f"best_{metric}: {best.metric:.6f}")
    print(f"best_description: {best.description}")
    print(f"plot: {output}")


def main() -> int:
    args = parse_args()
    rows = load_rows(args.results, args.metric)
    plot_progress(
        rows=rows,
        metric=args.metric,
        output=args.output,
        title=args.title,
        dpi=args.dpi,
        show_discard_labels=args.show_discard_labels,
    )
    print_summary(rows, args.metric, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
