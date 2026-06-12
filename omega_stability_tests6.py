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
DEFAULT_BASES = (10, 5, 2)
DEFAULT_MP_DPS = 250
DEFAULT_MAX_LEVELS = 8
DEFAULT_RECAMAN_TERMS = 64
DEFAULT_RNG_SEED = 12345
DEFAULT_OUTPUT_DIR = Path("outputs/omega_stability6")
DEFAULT_EXAMPLES = (
    ("light_speed", "299792458"),
    ("balanced_12", "122121212"),
    ("mixed_2357", "233732252"),
    ("repunit", "111111111"),
)
DIGIT_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def random_digit_string(width: int, rng: np.random.Generator) -> str:
    digits = [str(int(rng.integers(1, 10)))]
    digits.extend(str(int(rng.integers(0, 10))) for _ in range(width - 1))
    return "".join(digits)


def digit_entropy(digit_string: str) -> float:
    counts = Counter(digit_string)
    total = float(len(digit_string))
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def first_digit_benford_probability(digit_string: str) -> float:
    first_digit = 0
    for char in digit_string:
        first_digit = int(char)
        if first_digit != 0:
            break
    if first_digit == 0:
        return 0.0
    return math.log10(1.0 + 1.0 / float(first_digit))


def leading_trailing_asymmetry(digit_string: str) -> float:
    digits = [int(char) for char in digit_string]
    half = len(digits) // 2
    if half == 0:
        return 0.0
    leading_mean = sum(digits[:half]) / half
    trailing_mean = sum(digits[-half:]) / half
    return abs(leading_mean - trailing_mean) / 9.0


def positional_shadow_value(digit_string: str, base: int) -> int:
    value = 0
    for char in digit_string:
        value = value * int(base) + int(char)
    return value


def canonicalize_coefficients(digit_string: str, base: int) -> str:
    coefficients = [int(char) for char in digit_string]
    digits_rev: list[int] = []
    carry = 0

    for coefficient in reversed(coefficients):
        total = coefficient + carry
        digits_rev.append(total % base)
        carry = total // base

    while carry > 0:
        digits_rev.append(carry % base)
        carry //= base

    canonical = "".join(DIGIT_ALPHABET[digit] for digit in reversed(digits_rev)).lstrip("0")
    return canonical or "0"


def iterated_logs(value: int, base: int, max_levels: int) -> list[mp.mpf]:
    current = mp.mpf(value)
    levels: list[mp.mpf] = []
    for _ in range(max_levels):
        if current <= 0:
            break
        current = mp.log(current, base)
        if not mp.isfinite(current):
            break
        levels.append(current)
    return levels


def closest_anchor(value: mp.mpf) -> tuple[str, mp.mpf, mp.mpf]:
    nearest_integer = mp.mpf(int(mp.nint(value)))
    candidates = (
        ("one", mp.mpf("1")),
        ("e", mp.e),
        ("integer", nearest_integer),
    )
    best_class = "generic"
    best_anchor = value
    best_distance = mp.inf
    for anchor_class, anchor_value in candidates:
        distance = abs(value - anchor_value)
        if distance < best_distance:
            best_class = anchor_class
            best_anchor = anchor_value
            best_distance = distance
    return best_class, best_anchor, best_distance


def _mp_str(value: mp.mpf | None, digits: int = 40) -> str:
    if value is None:
        return ""
    return mp.nstr(value, n=digits)


def _level_spectrum_to_str(values: Sequence[mp.mpf]) -> str:
    parts = []
    for index, value in enumerate(values, start=1):
        parts.append(f"L{index}:{mp.nstr(value, n=16)}")
    return "|".join(parts)


def _int_or_zero(value: int | None) -> int:
    return 0 if value is None else int(value)


def layer_obstruction(
    digit_string: str,
    base: int,
    max_levels: int,
    mp_dps: int,
) -> dict[str, object]:
    legal_digit_string = canonicalize_coefficients(digit_string, base)
    mirror_string = legal_digit_string[::-1]
    value = positional_shadow_value(legal_digit_string, base)
    mirror_value = positional_shadow_value(mirror_string, base)
    forward_levels = iterated_logs(value, base=base, max_levels=max_levels)
    mirror_levels = iterated_logs(mirror_value, base=base, max_levels=max_levels)
    valid_depth = min(len(forward_levels), len(mirror_levels))

    score_floor = mp.power(10, -min(40, max(12, mp_dps // 4)))
    decoherence_floor = mp.power(10, -min(80, max(20, mp_dps // 2)))

    delta_levels: list[mp.mpf] = []
    score_levels: list[mp.mpf] = []
    center_levels: list[mp.mpf] = []
    anchor_levels: list[str] = []

    best_index: int | None = None
    best_score = mp.mpf("0")
    best_anchor_class = "symmetric"
    best_anchor_distance = mp.mpf("0")
    best_center = mp.mpf("0")
    first_noncontracting_level: int | None = None
    depth_to_decoherence = 0

    previous_delta: mp.mpf | None = None
    for index in range(valid_depth):
        forward = forward_levels[index]
        mirror = mirror_levels[index]
        delta = abs(forward - mirror)
        center = (forward + mirror) / 2
        anchor_class, _, anchor_distance = closest_anchor(center)
        score = mp.log10(1 + delta / max(anchor_distance, score_floor))

        delta_levels.append(delta)
        score_levels.append(score)
        center_levels.append(center)
        anchor_levels.append(anchor_class)

        if delta > decoherence_floor:
            depth_to_decoherence = index + 1

        if previous_delta is not None and previous_delta > 0 and first_noncontracting_level is None:
            if delta / previous_delta > 1:
                first_noncontracting_level = index

        if delta > 0 and (best_index is None or score > best_score):
            best_index = index
            best_score = score
            best_anchor_class = anchor_class
            best_anchor_distance = anchor_distance
            best_center = center

        previous_delta = delta

    if best_index is None:
        signature = f"b{base}:sym"
        return {
            "base": base,
            "input_digit_string": digit_string,
            "legal_digit_string": legal_digit_string,
            "mirror_digit_string": mirror_string,
            "value": value,
            "mirror_value": mirror_value,
            "valid_depth": valid_depth,
            "depth_to_decoherence": 0,
            "first_noncontracting_level": 0,
            "h_star": 0,
            "anchor_class": "symmetric",
            "anchor_distance_str": _mp_str(mp.mpf("0")),
            "center_at_h_star_str": _mp_str(mp.mpf("0")),
            "peak_score_log10_str": _mp_str(mp.mpf("0")),
            "peak_score_log10_float": 0.0,
            "delta_at_h_star_str": _mp_str(mp.mpf("0")),
            "delta_levels_str": _level_spectrum_to_str(delta_levels),
            "score_levels_str": _level_spectrum_to_str(score_levels),
            "center_levels_str": _level_spectrum_to_str(center_levels),
            "anchor_levels_str": "|".join(f"L{index + 1}:{anchor}" for index, anchor in enumerate(anchor_levels)),
            "signature": signature,
        }

    h_star = best_index + 1
    signature = f"b{base}:h{h_star}:{best_anchor_class}:d{depth_to_decoherence}"
    return {
        "base": base,
        "input_digit_string": digit_string,
        "legal_digit_string": legal_digit_string,
        "mirror_digit_string": mirror_string,
        "value": value,
        "mirror_value": mirror_value,
        "valid_depth": valid_depth,
        "depth_to_decoherence": depth_to_decoherence,
        "first_noncontracting_level": _int_or_zero(first_noncontracting_level),
        "h_star": h_star,
        "anchor_class": best_anchor_class,
        "anchor_distance_str": _mp_str(best_anchor_distance),
        "center_at_h_star_str": _mp_str(best_center),
        "peak_score_log10_str": _mp_str(best_score),
        "peak_score_log10_float": float(best_score),
        "delta_at_h_star_str": _mp_str(delta_levels[best_index]),
        "delta_levels_str": _level_spectrum_to_str(delta_levels),
        "score_levels_str": _level_spectrum_to_str(score_levels),
        "center_levels_str": _level_spectrum_to_str(center_levels),
        "anchor_levels_str": "|".join(f"L{index + 1}:{anchor}" for index, anchor in enumerate(anchor_levels)),
        "signature": signature,
    }


def analyze_digit_string(
    name: str,
    digit_string: str,
    bases: Sequence[int],
    max_levels: int,
    mp_dps: int,
) -> dict[str, object]:
    layer_rows = [layer_obstruction(digit_string, base=base, max_levels=max_levels, mp_dps=mp_dps) for base in bases]
    signatures = [str(row["signature"]) for row in layer_rows]
    core_signatures = [
        "sym" if int(row["h_star"]) == 0 else f"h{int(row['h_star'])}:{str(row['anchor_class'])}"
        for row in layer_rows
    ]
    h_stars = [int(row["h_star"]) for row in layer_rows]
    anchor_classes = [str(row["anchor_class"]) for row in layer_rows]
    disagreement = int(len(set(core_signatures)) > 1)
    late_prime = int(any(int(row["h_star"]) > h_stars[0] for row in layer_rows[1:]))
    total_peak_score = float(sum(float(row["peak_score_log10_float"]) for row in layer_rows))

    row: dict[str, object] = {
        "name": name,
        "digits": digit_string,
        "width": len(digit_string),
        "mirror_digits": digit_string[::-1],
        "digit_entropy_bits": digit_entropy(digit_string),
        "first_digit_benford_prob": first_digit_benford_probability(digit_string),
        "leading_trailing_asymmetry": leading_trailing_asymmetry(digit_string),
        "distinct_layer_signatures": len(set(signatures)),
        "distinct_layer_cores": len(set(core_signatures)),
        "prime_signature_disagreement": disagreement,
        "late_prime_obstruction": late_prime,
        "total_peak_score_log10": total_peak_score,
        "layer_signature_vector": "||".join(signatures),
        "layer_core_vector": "|".join(core_signatures),
        "anchor_class_vector": "|".join(anchor_classes),
        "h_star_vector": "|".join(str(value) for value in h_stars),
    }
    for layer in layer_rows:
        base = int(layer["base"])
        row[f"base_{base}_signature"] = layer["signature"]
        row[f"base_{base}_legal_digits"] = layer["legal_digit_string"]
        row[f"base_{base}_mirror_digits"] = layer["mirror_digit_string"]
        row[f"base_{base}_h_star"] = int(layer["h_star"])
        row[f"base_{base}_anchor_class"] = layer["anchor_class"]
        row[f"base_{base}_depth_to_decoherence"] = int(layer["depth_to_decoherence"])
        row[f"base_{base}_peak_score_log10"] = float(layer["peak_score_log10_float"])
        row[f"base_{base}_first_noncontracting_level"] = int(layer["first_noncontracting_level"])
        row[f"base_{base}_delta_at_h_star_str"] = layer["delta_at_h_star_str"]
        row[f"base_{base}_center_at_h_star_str"] = layer["center_at_h_star_str"]
        row[f"base_{base}_anchor_distance_str"] = layer["anchor_distance_str"]
        row[f"base_{base}_delta_levels_str"] = layer["delta_levels_str"]
        row[f"base_{base}_score_levels_str"] = layer["score_levels_str"]
    return row


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


def evaluate_binary_target(rows: Sequence[dict[str, object]], target_key: str) -> list[dict[str, object]]:
    labels = np.array([int(row[target_key]) for row in rows], dtype=int)
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
        forward_is_better = auc_forward >= auc_reverse
        direction = "increasing => label=1" if forward_is_better else "decreasing => label=1"
        accuracy, rule = best_threshold_accuracy(values if forward_is_better else -values, labels)
        if not forward_is_better:
            rule = rule.replace("feature", f"-{name}")
        results.append(
            {
                "target": target_key,
                "feature": name,
                "label_rate": float(labels.mean()),
                "pearson_with_label": pearson_corr(values, labels.astype(float)),
                "best_auc": best_auc,
                "best_threshold_accuracy": accuracy,
                "direction": direction,
                "threshold_rule": rule,
            }
        )
    return results


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


def matched_random_walk(n_terms: int, rng: np.random.Generator) -> list[int]:
    values = [0]
    for step in range(1, n_terms):
        direction = -1 if int(rng.integers(0, 2)) == 0 else 1
        candidate = values[-1] + direction * step
        if candidate < 0:
            candidate = values[-1] + step
        values.append(candidate)
    return values


def analyze_sequence(
    name: str,
    values: Sequence[int],
    bases: Sequence[int],
    max_levels: int,
    mp_dps: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    previous_signature = None

    for index, value in enumerate(values):
        digit_string = str(int(value))
        row = analyze_digit_string(
            name=f"{name}_{index}",
            digit_string=digit_string,
            bases=bases,
            max_levels=max_levels,
            mp_dps=mp_dps,
        )
        row["sequence"] = name
        row["index"] = index
        row["value"] = int(value)
        row["sequence_width"] = len(digit_string)
        row["signature_changed"] = int(previous_signature is not None and row["layer_signature_vector"] != previous_signature)
        previous_signature = str(row["layer_signature_vector"])
        rows.append(row)
    return rows


def save_rows(rows: Sequence[dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_random_results(rows: Sequence[dict[str, object]], feature_rows: Sequence[dict[str, object]], output_dir: Path) -> None:
    benford = np.array([float(row["first_digit_benford_prob"]) for row in rows], dtype=float)
    disagreement = np.array([float(row["prime_signature_disagreement"]) for row in rows], dtype=float)
    late_prime = np.array([float(row["late_prime_obstruction"]) for row in rows], dtype=float)
    total_score = np.array([float(row["total_peak_score_log10"]) for row in rows], dtype=float)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].scatter(benford, total_score, s=10, alpha=0.45, color="tab:red")
    axes[0, 0].set_xlabel("first-digit Benford probability")
    axes[0, 0].set_ylabel("total peak score")
    axes[0, 0].set_title("Aggregate obstruction score vs leading-digit proxy")
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].hist(total_score, bins=24, color="tab:blue", alpha=0.8)
    axes[0, 1].set_xlabel("total peak score")
    axes[0, 1].set_ylabel("count")
    axes[0, 1].set_title("Random-band obstruction score distribution")
    axes[0, 1].grid(alpha=0.25)

    target_order = ["prime_signature_disagreement", "late_prime_obstruction"]
    bars = []
    labels = []
    for target in target_order:
        best_row = max((row for row in feature_rows if str(row["target"]) == target), key=lambda row: float(row["best_auc"]))
        labels.append(target.replace("_", "\n"))
        bars.append(float(best_row["best_auc"]))
    axes[1, 0].bar(labels, bars, color=["tab:green", "tab:orange"])
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].set_ylabel("best AUC")
    axes[1, 0].set_title("Best cheap-feature recovery of obstruction labels")
    axes[1, 0].grid(axis="y", alpha=0.25)

    signature_counts = Counter(str(row["layer_signature_vector"]) for row in rows)
    top_signatures = signature_counts.most_common(5)
    axes[1, 1].bar(range(len(top_signatures)), [count for _, count in top_signatures], color="tab:purple")
    axes[1, 1].set_xticks(range(len(top_signatures)))
    axes[1, 1].set_xticklabels([f"s{index + 1}" for index in range(len(top_signatures))])
    axes[1, 1].set_ylabel("count")
    axes[1, 1].set_title("Top cross-layer signatures")
    axes[1, 1].grid(axis="y", alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_dir / "random_obstruction_diagnostics.png", dpi=180)
    plt.close(figure)


def plot_sequence_results(recaman_rows: Sequence[dict[str, object]], random_rows: Sequence[dict[str, object]], output_dir: Path) -> None:
    recaman_index = np.array([int(row["index"]) for row in recaman_rows], dtype=int)
    random_index = np.array([int(row["index"]) for row in random_rows], dtype=int)
    recaman_score = np.array([float(row["total_peak_score_log10"]) for row in recaman_rows], dtype=float)
    random_score = np.array([float(row["total_peak_score_log10"]) for row in random_rows], dtype=float)
    recaman_disagreement = np.array([float(row["prime_signature_disagreement"]) for row in recaman_rows], dtype=float)
    random_disagreement = np.array([float(row["prime_signature_disagreement"]) for row in random_rows], dtype=float)

    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(recaman_index, recaman_score, marker="o", linewidth=1.2, markersize=3, label="Recaman", color="tab:blue")
    axes[0].plot(random_index, random_score, marker="o", linewidth=1.0, markersize=2.5, label="matched random walk", color="tab:orange")
    axes[0].set_ylabel("total peak score")
    axes[0].set_title("Sequence obstruction scores")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].step(recaman_index, recaman_disagreement, where="mid", label="Recaman", color="tab:blue")
    axes[1].step(random_index, random_disagreement, where="mid", label="matched random walk", color="tab:orange")
    axes[1].set_xlabel("sequence index")
    axes[1].set_ylabel("prime disagreement")
    axes[1].set_title("Prime-layer disagreement along the sequence")
    axes[1].set_yticks([0, 1])
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_dir / "sequence_obstruction_diagnostics.png", dpi=180)
    plt.close(figure)


def build_summary(
    random_rows: Sequence[dict[str, object]],
    feature_rows: Sequence[dict[str, object]],
    example_rows: Sequence[dict[str, object]],
    recaman_rows: Sequence[dict[str, object]],
    random_walk_rows: Sequence[dict[str, object]],
    bases: Sequence[int],
    mp_dps: int,
    output_dir: Path,
) -> str:
    signature_counts = Counter(str(row["layer_signature_vector"]) for row in random_rows)
    prime_disagreement_rate = float(np.mean([int(row["prime_signature_disagreement"]) for row in random_rows]))
    late_prime_rate = float(np.mean([int(row["late_prime_obstruction"]) for row in random_rows]))
    best_by_target = {
        target: max((row for row in feature_rows if str(row["target"]) == target), key=lambda row: float(row["best_auc"]))
        for target in {str(row["target"]) for row in feature_rows}
    }

    def sequence_summary(rows: Sequence[dict[str, object]]) -> tuple[float, int, float]:
        disagreement_rate = float(np.mean([int(row["prime_signature_disagreement"]) for row in rows]))
        signature_changes = int(sum(int(row["signature_changed"]) for row in rows))
        mean_score = float(np.mean([float(row["total_peak_score_log10"]) for row in rows]))
        return disagreement_rate, signature_changes, mean_score

    recaman_disagreement, recaman_changes, recaman_score = sequence_summary(recaman_rows)
    random_disagreement, random_changes, random_score = sequence_summary(random_walk_rows)

    lines = []
    lines.append("Test 6 summary")
    lines.append("")
    lines.append("Setup")
    lines.append(f"- sample count: {len(random_rows)} random {random_rows[0]['width'] if random_rows else 0}-digit integers")
    lines.append(f"- positional layers: {', '.join(str(base) for base in bases)}")
    lines.append(f"- mpmath precision: {mp_dps} decimal digits")
    lines.append(
        "- Each decimal coefficient string is first canonicalized into a legal base-b digit string by carrying in base b. The obstruction test is then applied to that legal string and its mirror on each positional layer."
    )
    lines.append("")
    lines.append("Random-band obstruction spectrum")
    lines.append(f"- prime-signature disagreement rate: {prime_disagreement_rate:.4f}")
    lines.append(f"- late-prime-obstruction rate: {late_prime_rate:.4f}")
    for signature, count in signature_counts.most_common(5):
        lines.append(f"- signature `{signature}` occurred {count} times")
    lines.append("")
    lines.append("Cheap-feature recovery")
    for target in ("prime_signature_disagreement", "late_prime_obstruction"):
        row = best_by_target[target]
        lines.append(
            f"- {target}: best cheap feature was `{row['feature']}` with AUC {float(row['best_auc']):.6f}, threshold accuracy {float(row['best_threshold_accuracy']):.6f}, direction {row['direction']}"
        )
    lines.append("")
    lines.append("Named examples")
    for row in example_rows:
        lines.append(
            f"- {row['name']} ({row['digits']}): signatures `{row['layer_signature_vector']}`, total peak score {float(row['total_peak_score_log10']):.6f}, disagreement={int(row['prime_signature_disagreement'])}, late_prime={int(row['late_prime_obstruction'])}"
        )
        legal_bits = []
        for base in bases:
            legal_bits.append(f"b{base}={row[f'base_{base}_legal_digits']}")
        lines.append(f"  legal digits: {'; '.join(legal_bits)}")
    lines.append("")
    lines.append("Sequence comparison")
    lines.append(
        f"- Recaman: disagreement rate {recaman_disagreement:.4f}, signature changes {recaman_changes}, mean total peak score {recaman_score:.6f}"
    )
    lines.append(
        f"- matched random walk: disagreement rate {random_disagreement:.4f}, signature changes {random_changes}, mean total peak score {random_score:.6f}"
    )

    light_speed_row = next((row for row in example_rows if str(row["name"]) == "light_speed"), None)
    if light_speed_row is not None:
        lines.append("")
        lines.append("Interpretation")
        lines.append(
            f"- After carry-canonicalization, the light-speed seed still produces a nontrivial cross-layer signature: `{light_speed_row['layer_signature_vector']}`. That means the obstruction view is not collapsing it to the same legal object on every positional layer."
        )
    if float(best_by_target["late_prime_obstruction"]["best_auc"]) < 0.9:
        lines.append(
            "- Cheap digit summaries do not fully recover the late-prime obstruction label. That is weaker than a theorem, but stronger than the earlier Test 5 result where the split was almost purely a first-digit effect."
        )
    else:
        lines.append(
            "- The late-prime obstruction label is still strongly exposed to a cheap first-digit proxy, so the new invariant is only partially independent."
        )
    if recaman_changes > random_changes:
        lines.append(
            "- On this finite window the obstruction signature changes more often on Recaman than on the matched random walk, which is the direction you would want from an anomaly detector."
        )
    else:
        lines.append(
            "- On this finite window Recaman does not change obstruction signature more often than the matched random walk, so the sequence claim should stay provisional."
        )

    summary = "\n".join(lines) + "\n"
    (output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test 6: obstruction-spectrum analysis under positional prime layers.")
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--digit-width", type=int, default=DEFAULT_DIGIT_WIDTH)
    parser.add_argument("--bases", type=int, nargs="+", default=list(DEFAULT_BASES))
    parser.add_argument("--mp-dps", type=int, default=DEFAULT_MP_DPS)
    parser.add_argument("--max-levels", type=int, default=DEFAULT_MAX_LEVELS)
    parser.add_argument("--recaman-terms", type=int, default=DEFAULT_RECAMAN_TERMS)
    parser.add_argument("--rng-seed", type=int, default=DEFAULT_RNG_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mp.mp.dps = args.mp_dps
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.rng_seed)

    random_rows = [
        analyze_digit_string(
            name=f"random_{index}",
            digit_string=random_digit_string(args.digit_width, rng),
            bases=args.bases,
            max_levels=args.max_levels,
            mp_dps=args.mp_dps,
        )
        for index in range(args.sample_count)
    ]
    feature_rows = evaluate_binary_target(random_rows, "prime_signature_disagreement")
    feature_rows.extend(evaluate_binary_target(random_rows, "late_prime_obstruction"))

    example_rows = [
        analyze_digit_string(name=name, digit_string=digit_string, bases=args.bases, max_levels=args.max_levels, mp_dps=args.mp_dps)
        for name, digit_string in DEFAULT_EXAMPLES
    ]

    recaman_rows = analyze_sequence(
        name="recaman",
        values=recaman_sequence(args.recaman_terms),
        bases=args.bases,
        max_levels=args.max_levels,
        mp_dps=args.mp_dps,
    )
    random_walk_rows = analyze_sequence(
        name="matched_random_walk",
        values=matched_random_walk(args.recaman_terms, rng),
        bases=args.bases,
        max_levels=args.max_levels,
        mp_dps=args.mp_dps,
    )

    save_rows(random_rows, output_dir / "random_obstruction_metrics.csv")
    save_rows(feature_rows, output_dir / "random_feature_screening.csv")
    save_rows(example_rows, output_dir / "example_invariants.csv")
    save_rows(recaman_rows, output_dir / "recaman_obstruction.csv")
    save_rows(random_walk_rows, output_dir / "matched_random_walk_obstruction.csv")

    plot_random_results(random_rows, feature_rows, output_dir)
    plot_sequence_results(recaman_rows, random_walk_rows, output_dir)
    summary = build_summary(
        random_rows=random_rows,
        feature_rows=feature_rows,
        example_rows=example_rows,
        recaman_rows=recaman_rows,
        random_walk_rows=random_walk_rows,
        bases=args.bases,
        mp_dps=args.mp_dps,
        output_dir=output_dir,
    )
    print(summary, end="")
    print(f"Saved outputs to {output_dir}")

    if args.show:
        for image_path in sorted(output_dir.glob("*.png")):
            image = plt.imread(image_path)
            plt.figure(figsize=(10, 6))
            plt.imshow(image)
            plt.axis("off")
            plt.title(image_path.name)
        plt.show()


if __name__ == "__main__":
    main()
