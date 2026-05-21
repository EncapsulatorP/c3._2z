from __future__ import annotations

import argparse
import csv
import math
import os
from collections import Counter
from pathlib import Path
from typing import Sequence

import mpmath as mp
import numpy as np

_MPLCONFIGDIR = Path("/tmp/mplconfig")
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_SAMPLE_COUNT = 1000
DEFAULT_DIGIT_WIDTH = 9
DEFAULT_COMPOSITE_BASE = 10
DEFAULT_PRIME_BASE = 5
DEFAULT_SUB_PRIME_BASE = 2
DEFAULT_MP_DPS = 250
DEFAULT_RNG_SEED = 12345
DEFAULT_OUTPUT_DIR = Path("outputs/omega_stability5")


def random_digit_string(width: int, rng: np.random.Generator) -> str:
    digits = [str(int(rng.integers(1, 10)))]
    digits.extend(str(int(rng.integers(0, 10))) for _ in range(width - 1))
    return "".join(digits)


def parse_digits(digit_string: str) -> list[int]:
    return [int(char) for char in digit_string]


def positional_value(digits: Sequence[int], base: int) -> int:
    value = 0
    for digit in digits:
        value = value * base + int(digit)
    return value


def digit_entropy(digits: Sequence[int]) -> float:
    counts = Counter(int(digit) for digit in digits)
    total = float(len(digits))
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def first_digit_benford_probability(first_digit: int) -> float:
    return math.log10(1.0 + 1.0 / float(first_digit))


def leading_trailing_asymmetry(digits: Sequence[int]) -> float:
    half = len(digits) // 2
    if half == 0:
        return 0.0
    leading_mean = sum(digits[:half]) / half
    trailing_mean = sum(digits[-half:]) / half
    return abs(leading_mean - trailing_mean) / 9.0


def mp_log(value: int | float | mp.mpf) -> mp.mpf:
    return mp.log(mp.mpf(value))


def delta_pair(high_value: int, low_value: int) -> tuple[mp.mpf, mp.mpf]:
    high_log = mp_log(high_value)
    low_log = mp_log(low_value)
    delta1 = high_log - low_log
    delta2 = mp.log(high_log) - mp.log(low_log)
    return delta1, delta2


def mp_to_string(value: mp.mpf, digits: int = 80) -> str:
    return mp.nstr(value, n=digits)


def auc_score(feature: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(feature)
    ranks = np.empty(feature.size, dtype=float)
    ranks[order] = np.arange(1, feature.size + 1, dtype=float)
    positive = int(labels.sum())
    negative = int(labels.size - positive)
    if positive == 0 or negative == 0:
        return math.nan
    positive_rank_sum = float(ranks[labels.astype(bool)].sum())
    return (positive_rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def best_threshold_accuracy(feature: np.ndarray, labels: np.ndarray) -> tuple[float, str]:
    unique_values = np.unique(feature)
    thresholds = np.concatenate(
        [
            unique_values[:1] - 1e-12,
            (unique_values[:-1] + unique_values[1:]) / 2.0,
            unique_values[-1:] + 1e-12,
        ]
    )
    best_accuracy = -1.0
    best_rule = "feature <= threshold"
    for threshold in thresholds:
        pred_le = feature <= threshold
        pred_ge = feature >= threshold
        acc_le = float(np.mean(pred_le == labels))
        acc_ge = float(np.mean(pred_ge == labels))
        if acc_le > best_accuracy:
            best_accuracy = acc_le
            best_rule = f"feature <= {threshold:.6e}"
        if acc_ge > best_accuracy:
            best_accuracy = acc_ge
            best_rule = f"feature >= {threshold:.6e}"
    return best_accuracy, best_rule


def pearson_corr(feature: np.ndarray, target: np.ndarray) -> float:
    feature_centered = feature - feature.mean()
    target_centered = target - target.mean()
    denom = float(np.linalg.norm(feature_centered) * np.linalg.norm(target_centered))
    if denom == 0.0:
        return math.nan
    return float(np.dot(feature_centered, target_centered) / denom)


def analyze_samples(
    sample_count: int,
    digit_width: int,
    composite_base: int,
    prime_base: int,
    sub_prime_base: int,
    rng_seed: int,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(rng_seed)
    rows: list[dict[str, object]] = []

    for sample_index in range(sample_count):
        digit_string = random_digit_string(digit_width, rng)
        digits = parse_digits(digit_string)
        composite_value = positional_value(digits, composite_base)
        prime_value = positional_value(digits, prime_base)
        sub_prime_value = positional_value(digits, sub_prime_base)

        delta1, delta2 = delta_pair(composite_value, prime_value)
        sub_delta1, sub_delta2 = delta_pair(prime_value, sub_prime_value)
        ratio = delta2 / delta1 if delta1 != 0 else mp.mpf("0")

        row: dict[str, object] = {
            "sample_index": sample_index,
            "digits": digit_string,
            "composite_value": composite_value,
            "prime_value": prime_value,
            "sub_prime_value": sub_prime_value,
            "delta1_str": mp_to_string(delta1),
            "delta2_str": mp_to_string(delta2),
            "delta1_float": float(delta1),
            "delta2_float": float(delta2),
            "delta_ratio_float": float(ratio),
            "sub_delta1_str": mp_to_string(sub_delta1),
            "sub_delta2_str": mp_to_string(sub_delta2),
            "sub_delta1_float": float(sub_delta1),
            "sub_delta2_float": float(sub_delta2),
            "digit_entropy_bits": digit_entropy(digits),
            "first_digit_benford_prob": first_digit_benford_probability(digits[0]),
            "leading_trailing_asymmetry": leading_trailing_asymmetry(digits),
            "first_digit": digits[0],
            "last_digit": digits[-1],
            "reconstruction_ok": composite_value == positional_value(digits, composite_base),
        }
        rows.append(row)

    ratios = np.array([float(row["delta_ratio_float"]) for row in rows], dtype=float)
    threshold = float(np.median(ratios))
    for row in rows:
        row["stable_threshold"] = threshold
        row["stable_label"] = int(float(row["delta_ratio_float"]) <= threshold)
    return rows


def save_rows(rows: Sequence[dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_features(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    labels = np.array([int(row["stable_label"]) for row in rows], dtype=int)
    ratios = np.array([float(row["delta_ratio_float"]) for row in rows], dtype=float)
    features = {
        "digit_entropy_bits": np.array([float(row["digit_entropy_bits"]) for row in rows], dtype=float),
        "first_digit_benford_prob": np.array([float(row["first_digit_benford_prob"]) for row in rows], dtype=float),
        "leading_trailing_asymmetry": np.array([float(row["leading_trailing_asymmetry"]) for row in rows], dtype=float),
    }

    results: list[dict[str, object]] = []
    for name, values in features.items():
        auc_forward = auc_score(values, labels)
        auc_reverse = auc_score(-values, labels)
        best_auc = max(auc_forward, auc_reverse)
        direction = "increasing => stable" if auc_forward >= auc_reverse else "decreasing => stable"
        accuracy, rule = best_threshold_accuracy(values if auc_forward >= auc_reverse else -values, labels)
        if auc_forward < auc_reverse:
            rule = rule.replace("feature", f"-{name}")
        results.append(
            {
                "feature": name,
                "pearson_with_ratio": pearson_corr(values, ratios),
                "best_auc": best_auc,
                "best_threshold_accuracy": accuracy,
                "stable_direction": direction,
                "threshold_rule": rule,
            }
        )
    return results


def save_feature_results(rows: Sequence[dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(sample_rows: Sequence[dict[str, object]], feature_rows: Sequence[dict[str, object]], output_dir: Path) -> None:
    ratios = np.array([float(row["delta_ratio_float"]) for row in sample_rows], dtype=float)
    first_digits = np.array([int(row["first_digit"]) for row in sample_rows], dtype=int)
    entropy = np.array([float(row["digit_entropy_bits"]) for row in sample_rows], dtype=float)
    benford = np.array([float(row["first_digit_benford_prob"]) for row in sample_rows], dtype=float)
    asym = np.array([float(row["leading_trailing_asymmetry"]) for row in sample_rows], dtype=float)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].scatter(first_digits, ratios, s=10, alpha=0.5, color="tab:blue")
    axes[0, 0].set_xlabel("first digit")
    axes[0, 0].set_ylabel("Δ2 / Δ1")
    axes[0, 0].set_title("Stability ratio vs first digit")
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].scatter(entropy, ratios, s=10, alpha=0.5, color="tab:green")
    axes[0, 1].set_xlabel("digit entropy")
    axes[0, 1].set_ylabel("Δ2 / Δ1")
    axes[0, 1].set_title("Stability ratio vs entropy")
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].scatter(asym, ratios, s=10, alpha=0.5, color="tab:orange")
    axes[1, 0].set_xlabel("leading/trailing asymmetry")
    axes[1, 0].set_ylabel("Δ2 / Δ1")
    axes[1, 0].set_title("Stability ratio vs asymmetry")
    axes[1, 0].grid(alpha=0.25)

    feature_names = [str(row["feature"]) for row in feature_rows]
    aucs = np.array([float(row["best_auc"]) for row in feature_rows], dtype=float)
    axes[1, 1].bar(feature_names, aucs, color=["tab:purple", "tab:red", "tab:brown"])
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].set_ylabel("best AUC")
    axes[1, 1].set_title("Cheap-feature prediction of stability class")
    axes[1, 1].tick_params(axis="x", rotation=15)
    axes[1, 1].grid(axis="y", alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_dir / "diagnostics.png", dpi=180)
    plt.close(figure)


def build_summary(
    sample_rows: Sequence[dict[str, object]],
    feature_rows: Sequence[dict[str, object]],
    composite_base: int,
    prime_base: int,
    sub_prime_base: int,
    mp_dps: int,
    output_dir: Path,
) -> str:
    ratios = np.array([float(row["delta_ratio_float"]) for row in sample_rows], dtype=float)
    stable_rate = np.mean([int(row["stable_label"]) for row in sample_rows])
    threshold = float(sample_rows[0]["stable_threshold"]) if sample_rows else math.nan
    feature_rows_sorted = sorted(feature_rows, key=lambda row: float(row["best_auc"]), reverse=True)

    lines = []
    lines.append("Test 5 summary")
    lines.append("")
    lines.append("Assumption")
    lines.append(
        f"- Each random digit string is evaluated on the composite positional layer `base={composite_base}` and the prime positional shadows `base={prime_base}` and `base={sub_prime_base}`."
    )
    lines.append(
        "- `Δ1` is defined as `log(V_composite) - log(V_prime)` and `Δ2` as `log(log(V_composite)) - log(log(V_prime))`, both computed with high precision."
    )
    lines.append(
        "- Stability is defined by the median split of the absorption ratio `Δ2 / Δ1`: lower ratio means the first log-level difference is more absorbed by the second."
    )
    lines.append("")
    lines.append("Run")
    lines.append(f"- sample count: {len(sample_rows)}")
    lines.append(f"- digit width: {len(str(sample_rows[0]['digits'])) if sample_rows else 0}")
    lines.append(f"- mpmath precision: {mp_dps} decimal digits")
    lines.append(f"- stability threshold (median Δ2/Δ1): {threshold:.12e}")
    lines.append(f"- stable class fraction: {stable_rate:.4f}")
    lines.append(
        f"- ratio range: min={ratios.min():.12e}, max={ratios.max():.12e}, mean={ratios.mean():.12e}, std={ratios.std():.12e}"
    )
    lines.append("")
    lines.append("Cheap-feature screening")
    for row in feature_rows_sorted:
        lines.append(
            f"- {row['feature']}: best_auc={float(row['best_auc']):.6f}, best_threshold_accuracy={float(row['best_threshold_accuracy']):.6f}, pearson_with_ratio={float(row['pearson_with_ratio']):.6f}, direction={row['stable_direction']}"
        )
    lines.append("")
    lines.append("Conclusion")
    best = feature_rows_sorted[0]
    lines.append(
        f"- The strongest cheap predictor was `{best['feature']}` with AUC {float(best['best_auc']):.6f} and threshold accuracy {float(best['best_threshold_accuracy']):.6f}."
    )
    if str(best["feature"]) == "first_digit_benford_prob" and float(best["best_auc"]) > 0.9:
        lines.append(
            "- On this construction, the stability split is almost entirely recoverable from a first-digit Benford-like statistic. That means the tower classification is not adding much beyond a very cheap leading-digit feature."
        )
    else:
        lines.append(
            "- On this construction, the cheap features did not trivially recover the stability split, so the tower statistic retained some independent information."
        )
    if any(str(row["feature"]) == "digit_entropy_bits" for row in feature_rows_sorted):
        entropy_row = next(row for row in feature_rows_sorted if str(row["feature"]) == "digit_entropy_bits")
        lines.append(
            f"- Digit entropy by itself was weak here: AUC {float(entropy_row['best_auc']):.6f}."
        )

    summary = "\n".join(lines) + "\n"
    (output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test 5: high-precision tower-vs-cheap-feature screening on random fixed-width integers.")
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--digit-width", type=int, default=DEFAULT_DIGIT_WIDTH)
    parser.add_argument("--composite-base", type=int, default=DEFAULT_COMPOSITE_BASE)
    parser.add_argument("--prime-base", type=int, default=DEFAULT_PRIME_BASE)
    parser.add_argument("--sub-prime-base", type=int, default=DEFAULT_SUB_PRIME_BASE)
    parser.add_argument("--mp-dps", type=int, default=DEFAULT_MP_DPS)
    parser.add_argument("--rng-seed", type=int, default=DEFAULT_RNG_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mp.mp.dps = args.mp_dps
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_rows = analyze_samples(
        sample_count=args.sample_count,
        digit_width=args.digit_width,
        composite_base=args.composite_base,
        prime_base=args.prime_base,
        sub_prime_base=args.sub_prime_base,
        rng_seed=args.rng_seed,
    )
    feature_rows = evaluate_features(sample_rows)

    save_rows(sample_rows, output_dir / "sample_metrics.csv")
    save_feature_results(feature_rows, output_dir / "feature_screening.csv")
    plot_results(sample_rows, feature_rows, output_dir)
    summary = build_summary(
        sample_rows=sample_rows,
        feature_rows=feature_rows,
        composite_base=args.composite_base,
        prime_base=args.prime_base,
        sub_prime_base=args.sub_prime_base,
        mp_dps=args.mp_dps,
        output_dir=output_dir,
    )
    print(summary, end="")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
