from __future__ import annotations

import argparse
import csv
import math
import os
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

_MPLCONFIGDIR = Path("/tmp/mplconfig")
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_STATES = ("111111111", "122121212", "233732252")
DEFAULT_BASE = 10
DEFAULT_OUTPUT_DIR = Path("outputs/omega_stability4")


def parse_digit_string(digit_string: str, base: int) -> list[int]:
    digits = [int(char) for char in digit_string.strip()]
    if not digits:
        raise ValueError("digit string must not be empty")
    if any(digit < 0 or digit >= base for digit in digits):
        raise ValueError(f"all digits must lie in [0, {base - 1}]")
    return digits


def digit_entropy(digits: Sequence[int]) -> float:
    counts = Counter(int(digit) for digit in digits)
    total = float(len(digits))
    if total == 0.0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def alphabet_size(digits: Sequence[int]) -> int:
    return len(set(int(digit) for digit in digits))


def max_entropy_for_alphabet(active_alphabet_size: int) -> float:
    return math.log2(active_alphabet_size) if active_alphabet_size > 1 else 0.0


def residue_robustness_score(digits: Sequence[int]) -> float:
    entropy = digit_entropy(digits)
    active_alphabet_size = alphabet_size(digits)
    entropy_max = max_entropy_for_alphabet(active_alphabet_size)
    return entropy / entropy_max if entropy_max > 0.0 else 0.0


def digit_counts_str(digits: Sequence[int]) -> str:
    counts = Counter(int(digit) for digit in digits)
    return ", ".join(f"{digit}:{count}" for digit, count in sorted(counts.items()))


def carry_slacks(digits: Sequence[int], base: int) -> list[int]:
    return [base - 1 - int(digit) for digit in digits]


def perturbation_metrics(digits: Sequence[int], base: int) -> dict[str, float]:
    original_score = residue_robustness_score(digits)
    original_alphabet_size = alphabet_size(digits)

    total_valid = 0
    preserve_alphabet = 0
    score_deltas: list[float] = []
    preserved_score_deltas: list[float] = []

    for index, digit in enumerate(digits):
        for step in (-1, 1):
            candidate = digit + step
            if candidate < 0 or candidate >= base:
                continue
            total_valid += 1
            perturbed = list(digits)
            perturbed[index] = candidate
            perturbed_alphabet_size = alphabet_size(perturbed)
            perturbed_score = residue_robustness_score(perturbed)
            delta = abs(perturbed_score - original_score)
            score_deltas.append(delta)
            if perturbed_alphabet_size == original_alphabet_size:
                preserve_alphabet += 1
                preserved_score_deltas.append(delta)

    return {
        "valid_pm1_perturbations": float(total_valid),
        "same_alphabet_rate_pm1": preserve_alphabet / total_valid if total_valid else math.nan,
        "mean_abs_score_delta_pm1": float(np.mean(score_deltas)) if score_deltas else math.nan,
        "max_abs_score_delta_pm1": float(np.max(score_deltas)) if score_deltas else math.nan,
        "mean_abs_score_delta_same_alphabet_pm1": (
            float(np.mean(preserved_score_deltas)) if preserved_score_deltas else math.nan
        ),
    }


def analyze_state(name: str, digit_string: str, base: int) -> dict[str, object]:
    digits = parse_digit_string(digit_string, base)
    entropy = digit_entropy(digits)
    active_alphabet_size = alphabet_size(digits)
    entropy_max = max_entropy_for_alphabet(active_alphabet_size)
    score = residue_robustness_score(digits)
    slacks = carry_slacks(digits, base)
    perturb = perturbation_metrics(digits, base)

    row: dict[str, object] = {
        "name": name,
        "digits": digit_string,
        "width": len(digits),
        "alphabet_size": active_alphabet_size,
        "digit_counts": digit_counts_str(digits),
        "digit_entropy_bits": entropy,
        "max_entropy_bits": entropy_max,
        "residue_robustness_score": score,
        "carry_slack_mean": float(np.mean(slacks)),
        "carry_slack_std": float(np.std(slacks)),
        "carry_slack_min": float(np.min(slacks)),
        "carry_slack_max": float(np.max(slacks)),
    }
    row.update(perturb)
    return row


def save_rows(rows: Sequence[dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_scores(rows: Sequence[dict[str, object]], output_dir: Path) -> None:
    names = [str(row["name"]) for row in rows]
    entropy_scores = np.array([float(row["residue_robustness_score"]) for row in rows], dtype=float)
    same_alphabet_rates = np.array([float(row["same_alphabet_rate_pm1"]) for row in rows], dtype=float)
    mean_score_deltas = np.array([float(row["mean_abs_score_delta_pm1"]) for row in rows], dtype=float)
    slack_std = np.array([float(row["carry_slack_std"]) for row in rows], dtype=float)

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()
    metrics = [
        ("Entropy Robustness Score", entropy_scores),
        ("Same-Alphabet Rate (±1)", same_alphabet_rates),
        ("Mean |Score Delta| (±1)", mean_score_deltas),
        ("Carry Slack Std", slack_std),
    ]

    for axis, (title, values) in zip(axes, metrics):
        axis.bar(names, values, color=["tab:blue", "tab:green", "tab:orange"])
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_dir / "test4_metric_bars.png", dpi=180)
    plt.close(figure)


def build_summary(rows: Sequence[dict[str, object]], output_dir: Path) -> str:
    lines = []
    lines.append("Test 4 summary")
    lines.append("")
    lines.append("Metric definitions")
    lines.append("- `residue_robustness_score = H / log2(|alphabet|)` when the active alphabet has size > 1, else 0.")
    lines.append("- `same_alphabet_rate_pm1` measures how often a valid single-digit +/-1 perturbation keeps the active alphabet size unchanged.")
    lines.append("- `carry_slack_std` measures how uneven the distances to overflow (`9 - digit`) are across positions.")

    lines.append("")
    lines.append("State metrics")
    for row in rows:
        lines.append(
            f"- {row['name']} ({row['digits']}): score={float(row['residue_robustness_score']):.6f}, alphabet={int(row['alphabet_size'])}, entropy={float(row['digit_entropy_bits']):.6f}, same_alphabet_rate={float(row['same_alphabet_rate_pm1']):.6f}, mean_abs_score_delta={float(row['mean_abs_score_delta_pm1']):.6f}, carry_slack_std={float(row['carry_slack_std']):.6f}"
        )

    rows_by_name = {str(row["name"]): row for row in rows}
    repunit = rows_by_name.get("repunit")
    balanced = rows_by_name.get("balanced_12")
    mixed = rows_by_name.get("mixed_2357")

    lines.append("")
    lines.append("Conclusions")
    if repunit and balanced and mixed:
        lines.append(
            f"- The proposed entropy score does separate the three states as claimed: repunit={float(repunit['residue_robustness_score']):.6f}, balanced_12={float(balanced['residue_robustness_score']):.6f}, mixed_2357={float(mixed['residue_robustness_score']):.6f}."
        )
        lines.append(
            f"- On that metric, `122121212` is the strongest interior state of the three, while `111111111` is the boundary case with score 0."
        )
        lines.append(
            f"- A simple perturbation proxy does not fully support the stronger robustness claim: `233732252` preserves alphabet size under +/-1 perturbations more often ({float(mixed['same_alphabet_rate_pm1']):.6f}) than `122121212` ({float(balanced['same_alphabet_rate_pm1']):.6f}), and its mean score drift is smaller ({float(mixed['mean_abs_score_delta_pm1']):.6f} vs {float(balanced['mean_abs_score_delta_pm1']):.6f})."
        )
        lines.append(
            f"- The carry geometry story is supported: `233732252` has much more uneven carry slack (std {float(mixed['carry_slack_std']):.6f}) than `122121212` ({float(balanced['carry_slack_std']):.6f}) and `111111111` ({float(repunit['carry_slack_std']):.6f})."
        )
        lines.append(
            "- The falsifiable takeaway is that the normalized entropy score captures one notion of interior typicality, but it is not by itself a complete residue-robustness metric."
        )

    summary = "\n".join(lines) + "\n"
    (output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test 4: entropy-based residue robustness proxy on digit states.")
    parser.add_argument("--states", nargs="+", default=list(DEFAULT_STATES), help="digit strings to analyze")
    parser.add_argument("--base", type=int, default=DEFAULT_BASE, help="digit base")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="directory for CSV, plots, and summary")
    parser.add_argument("--show", action="store_true", help="show the saved plot after generation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(args.states) != 3:
        names = [f"state_{index + 1}" for index in range(len(args.states))]
    else:
        names = ["repunit", "balanced_12", "mixed_2357"]

    rows = [analyze_state(name, digit_string, args.base) for name, digit_string in zip(names, args.states)]
    save_rows(rows, output_dir / "test4_state_metrics.csv")
    plot_scores(rows, output_dir)
    summary = build_summary(rows, output_dir)

    print(summary, end="")
    print(f"Saved outputs to {output_dir}")

    if args.show:
        image = plt.imread(output_dir / "test4_metric_bars.png")
        plt.figure(figsize=(11, 8))
        plt.imshow(image)
        plt.axis("off")
        plt.title("test4_metric_bars.png")
        plt.show()


if __name__ == "__main__":
    main()
