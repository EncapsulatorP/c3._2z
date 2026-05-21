from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np

_MPLCONFIGDIR = Path("/tmp/mplconfig")
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_RECAMAN_TERMS = 120
DEFAULT_COLLATZ_SEED = 27
DEFAULT_COLLATZ_LIMIT = 200
DEFAULT_OUTPUT_DIR = Path("applications/outputs/shadow_thresholds")


def recaman_sequence(n_terms: int) -> list[int]:
    values = [0]
    seen = {0}
    for step in range(1, n_terms):
        candidate = values[-1] - step
        if candidate > 0 and candidate not in seen:
            next_value = candidate
        else:
            next_value = values[-1] + step
        values.append(next_value)
        seen.add(next_value)
    return values


def collatz_sequence(seed: int, max_terms: int) -> list[int]:
    values = [int(seed)]
    current = int(seed)
    while current != 1 and len(values) < max_terms:
        if current % 2 == 0:
            current //= 2
        else:
            current = 3 * current + 1
        values.append(current)
    return values


def log_levels(value: int, depth: int = 3) -> list[float]:
    current = float(value)
    levels = []
    for _ in range(depth):
        current = math.log1p(current)
        levels.append(current)
    return levels


def anomaly_rows(name: str, values: Sequence[int], depth: int = 3) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    previous_levels: list[float] | None = None
    previous_value: int | None = None

    for index, value in enumerate(values):
        levels = log_levels(value, depth=depth)
        deltas = [0.0] * depth if previous_levels is None else [levels[i] - previous_levels[i] for i in range(depth)]
        raw_delta = 0.0 if previous_value is None else float(value - previous_value)
        jump_score = max(abs(delta) for delta in deltas)
        row: dict[str, float | int | str] = {
            "sequence": name,
            "index": index,
            "value": value,
            "raw_delta": raw_delta,
            "jump_score": jump_score,
        }
        for level_index, level_value in enumerate(levels, start=1):
            row[f"L{level_index}"] = level_value
        for level_index, delta_value in enumerate(deltas, start=1):
            row[f"dL{level_index}"] = delta_value
        rows.append(row)
        previous_levels = levels
        previous_value = value
    return rows


def save_rows(rows: Sequence[dict[str, float | int | str]], output_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def top_anomalies(rows: Sequence[dict[str, float | int | str]], top_k: int = 8) -> list[dict[str, float | int | str]]:
    filtered = [row for row in rows if int(row["index"]) > 0]
    return sorted(filtered, key=lambda row: float(row["jump_score"]), reverse=True)[:top_k]


def plot_sequences(recaman_rows: Sequence[dict[str, float | int | str]], collatz_rows: Sequence[dict[str, float | int | str]], output_dir: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    layouts = [
        ("Recaman values", recaman_rows, "value", axes[0, 0], "tab:blue"),
        ("Recaman jump score", recaman_rows, "jump_score", axes[0, 1], "tab:red"),
        ("Collatz values", collatz_rows, "value", axes[1, 0], "tab:green"),
        ("Collatz jump score", collatz_rows, "jump_score", axes[1, 1], "tab:orange"),
    ]
    for title, rows, key, axis, color in layouts:
        xs = np.array([int(row["index"]) for row in rows], dtype=int)
        ys = np.array([float(row[key]) for row in rows], dtype=float)
        axis.plot(xs, ys, color=color)
        if key == "value":
            axis.set_yscale("symlog")
        axis.set_title(title)
        axis.set_xlabel("index")
        axis.set_ylabel(key)
        axis.grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_dir / "shadow_thresholds.png", dpi=180)
    plt.close(figure)


def build_summary(recaman_rows: Sequence[dict[str, float | int | str]], collatz_rows: Sequence[dict[str, float | int | str]], output_dir: Path) -> str:
    recaman_top = top_anomalies(recaman_rows)
    collatz_top = top_anomalies(collatz_rows)
    lines = []
    lines.append("Shadow Threshold Application")
    lines.append("")
    lines.append("Setup")
    lines.append("- Recaman and Collatz are analyzed with multi-level log coordinates `L1`, `L2`, `L3`.")
    lines.append("- A shadow-threshold score is defined as `max(|dL1|, |dL2|, |dL3|)` per step.")
    lines.append("")
    lines.append("Top Recaman anomalies")
    for row in recaman_top:
        lines.append(
            f"- index={int(row['index'])}, value={int(row['value'])}, raw_delta={float(row['raw_delta']):.6e}, jump_score={float(row['jump_score']):.6e}"
        )
    lines.append("")
    lines.append("Top Collatz anomalies")
    for row in collatz_top:
        lines.append(
            f"- index={int(row['index'])}, value={int(row['value'])}, raw_delta={float(row['raw_delta']):.6e}, jump_score={float(row['jump_score']):.6e}"
        )
    lines.append("")
    lines.append("Interpretation")
    lines.append(
        "- The log-level jump score highlights steps where the sequence changes scale class abruptly, while suppressing raw magnitude blow-ups."
    )
    lines.append(
        "- In practice this is the use-case: mark comparability-class transitions first, then inspect the raw step only at the flagged indices."
    )
    summary = "\n".join(lines) + "\n"
    (output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Application example: shadow-threshold anomaly detection.")
    parser.add_argument("--recaman-terms", type=int, default=DEFAULT_RECAMAN_TERMS)
    parser.add_argument("--collatz-seed", type=int, default=DEFAULT_COLLATZ_SEED)
    parser.add_argument("--collatz-limit", type=int, default=DEFAULT_COLLATZ_LIMIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    recaman_rows = anomaly_rows("recaman", recaman_sequence(args.recaman_terms))
    collatz_rows = anomaly_rows("collatz", collatz_sequence(args.collatz_seed, args.collatz_limit))
    save_rows(recaman_rows, output_dir / "recaman.csv")
    save_rows(collatz_rows, output_dir / "collatz.csv")
    plot_sequences(recaman_rows, collatz_rows, output_dir)
    summary = build_summary(recaman_rows, collatz_rows, output_dir)
    print(summary, end="")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
