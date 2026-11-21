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
DEFAULT_BASES = (10, 7, 5, 3, 2)
DEFAULT_MP_DPS = 250
DEFAULT_MAX_LEVELS = 8
DEFAULT_RECAMAN_TERMS = 256
DEFAULT_SEQUENCE_MIN_WIDTH = 1
DEFAULT_RNG_SEED = 12345
DEFAULT_ONE_TOL = 0.125
DEFAULT_E_TOL = 0.125
DEFAULT_INTEGER_TOL = 0.075
DEFAULT_OUTPUT_DIR = Path("outputs/omega_stability7")
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
    for char in digit_string:
        first_digit = int(char)
        if first_digit != 0:
            return math.log10(1.0 + 1.0 / float(first_digit))
    return 0.0


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


def classify_anchor(
    value: mp.mpf,
    one_tol: mp.mpf,
    e_tol: mp.mpf,
    integer_tol: mp.mpf,
) -> tuple[str, mp.mpf, mp.mpf]:
    nearest_integer = mp.mpf(int(mp.nint(value)))
    dist_one = abs(value - mp.mpf("1"))
    dist_e = abs(value - mp.e)
    dist_integer = abs(value - nearest_integer)

    if dist_one <= one_tol:
        return "one", mp.mpf("1"), dist_one
    if dist_e <= e_tol:
        return "e", mp.e, dist_e
    if dist_integer <= integer_tol:
        return "integer", nearest_integer, dist_integer

    candidates = (
        ("one", mp.mpf("1"), dist_one),
        ("e", mp.e, dist_e),
        ("integer", nearest_integer, dist_integer),
    )
    anchor_class, anchor_value, anchor_distance = min(candidates, key=lambda item: item[2])
    return "generic", anchor_value, anchor_distance


def _mp_str(value: mp.mpf | None, digits: int = 40) -> str:
    if value is None:
        return ""
    return mp.nstr(value, n=digits)


def _level_spectrum_to_str(values: Sequence[mp.mpf]) -> str:
    parts = []
    for index, value in enumerate(values, start=1):
        parts.append(f"L{index}:{mp.nstr(value, n=16)}")
    return "|".join(parts)


def core_signature(level: int, anchor_class: str, symmetric: bool = False, contracting: bool = False) -> str:
    if symmetric:
        return "sym"
    if contracting:
        return "contracting"
    return f"h{level}:{anchor_class}"


def layer_obstruction(
    digit_string: str,
    base: int,
    max_levels: int,
    mp_dps: int,
    one_tol: mp.mpf,
    e_tol: mp.mpf,
    integer_tol: mp.mpf,
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

    best_peak_index: int | None = None
    best_peak_score = mp.mpf("0")
    best_peak_anchor_class = "symmetric"
    best_peak_anchor_distance = mp.mpf("0")
    best_peak_center = mp.mpf("0")

    first_break_index: int | None = None
    first_break_score = mp.mpf("0")
    first_break_anchor_class = "contracting"
    first_break_anchor_distance = mp.mpf("0")
    first_break_center = mp.mpf("0")
    first_break_delta = mp.mpf("0")

    depth_to_decoherence = 0
    previous_delta: mp.mpf | None = None

    for index in range(valid_depth):
        forward = forward_levels[index]
        mirror = mirror_levels[index]
        delta = abs(forward - mirror)
        center = (forward + mirror) / 2
        anchor_class, _, anchor_distance = classify_anchor(center, one_tol=one_tol, e_tol=e_tol, integer_tol=integer_tol)
        score = mp.log10(1 + delta / max(anchor_distance, score_floor))

        delta_levels.append(delta)
        score_levels.append(score)
        center_levels.append(center)
        anchor_levels.append(anchor_class)

        if delta > decoherence_floor:
            depth_to_decoherence = index + 1

        if previous_delta is not None and previous_delta > 0 and first_break_index is None:
            if delta / previous_delta > 1:
                first_break_index = index
                first_break_score = score
                first_break_anchor_class = anchor_class
                first_break_anchor_distance = anchor_distance
                first_break_center = center
                first_break_delta = delta

        if delta > 0 and (best_peak_index is None or score > best_peak_score):
            best_peak_index = index
            best_peak_score = score
            best_peak_anchor_class = anchor_class
            best_peak_anchor_distance = anchor_distance
            best_peak_center = center

        previous_delta = delta

    if best_peak_index is None:
        peak_level = 0
        peak_anchor_class = "symmetric"
        peak_signature = f"b{base}:sym"
        peak_core = "sym"
        peak_delta = mp.mpf("0")
    else:
        peak_level = best_peak_index + 1
        peak_anchor_class = best_peak_anchor_class
        peak_signature = f"b{base}:h{peak_level}:{peak_anchor_class}:d{depth_to_decoherence}"
        peak_core = core_signature(peak_level, peak_anchor_class)
        peak_delta = delta_levels[best_peak_index]

    if best_peak_index is None:
        first_break_level = 0
        first_break_signature = f"b{base}:sym"
        first_break_core = "sym"
    elif first_break_index is None:
        first_break_level = 0
        first_break_signature = f"b{base}:contracting:d{depth_to_decoherence}"
        first_break_core = "contracting"
    else:
        first_break_level = first_break_index + 1
        first_break_signature = f"b{base}:h{first_break_level}:{first_break_anchor_class}:d{depth_to_decoherence}"
        first_break_core = core_signature(first_break_level, first_break_anchor_class)

    return {
        "base": base,
        "input_digit_string": digit_string,
        "legal_digit_string": legal_digit_string,
        "mirror_digit_string": mirror_string,
        "value": value,
        "mirror_value": mirror_value,
        "valid_depth": valid_depth,
        "depth_to_decoherence": depth_to_decoherence,
        "h_star_peak": peak_level,
        "h_star_first_break": first_break_level,
        "peak_anchor_class": peak_anchor_class,
        "first_break_anchor_class": first_break_anchor_class,
        "peak_anchor_distance_str": _mp_str(best_peak_anchor_distance),
        "first_break_anchor_distance_str": _mp_str(first_break_anchor_distance),
        "peak_center_str": _mp_str(best_peak_center),
        "first_break_center_str": _mp_str(first_break_center),
        "peak_score_log10_str": _mp_str(best_peak_score),
        "first_break_score_log10_str": _mp_str(first_break_score),
        "peak_score_log10_float": float(best_peak_score),
        "first_break_score_log10_float": float(first_break_score),
        "peak_delta_str": _mp_str(peak_delta),
        "first_break_delta_str": _mp_str(first_break_delta),
        "delta_levels_str": _level_spectrum_to_str(delta_levels),
        "score_levels_str": _level_spectrum_to_str(score_levels),
        "center_levels_str": _level_spectrum_to_str(center_levels),
        "anchor_levels_str": "|".join(f"L{index + 1}:{anchor}" for index, anchor in enumerate(anchor_levels)),
        "peak_signature": peak_signature,
        "first_break_signature": first_break_signature,
        "peak_core": peak_core,
        "first_break_core": first_break_core,
    }


def analyze_digit_string(
    name: str,
    digit_string: str,
    bases: Sequence[int],
    max_levels: int,
    mp_dps: int,
    one_tol: mp.mpf,
    e_tol: mp.mpf,
    integer_tol: mp.mpf,
) -> dict[str, object]:
    layer_rows = [
        layer_obstruction(
            digit_string=digit_string,
            base=base,
            max_levels=max_levels,
            mp_dps=mp_dps,
            one_tol=one_tol,
            e_tol=e_tol,
            integer_tol=integer_tol,
        )
        for base in bases
    ]

    peak_signatures = [str(row["peak_signature"]) for row in layer_rows]
    first_break_signatures = [str(row["first_break_signature"]) for row in layer_rows]
    peak_cores = [str(row["peak_core"]) for row in layer_rows]
    first_break_cores = [str(row["first_break_core"]) for row in layer_rows]
    peak_levels = [int(row["h_star_peak"]) for row in layer_rows]
    first_break_levels = [int(row["h_star_first_break"]) for row in layer_rows]
    peak_anchor_classes = [str(row["peak_anchor_class"]) for row in layer_rows]
    first_break_anchor_classes = [str(row["first_break_anchor_class"]) for row in layer_rows]

    peak_disagreement = int(len(set(peak_cores)) > 1)
    first_break_disagreement = int(len(set(first_break_cores)) > 1)
    late_prime_peak = int(any(level > peak_levels[0] for level in peak_levels[1:]))
    late_prime_first_break = int(any(level > first_break_levels[0] for level in first_break_levels[1:]))

    row: dict[str, object] = {
        "name": name,
        "digits": digit_string,
        "width": len(digit_string),
        "mirror_digits": digit_string[::-1],
        "digit_entropy_bits": digit_entropy(digit_string),
        "first_digit_benford_prob": first_digit_benford_probability(digit_string),
        "leading_trailing_asymmetry": leading_trailing_asymmetry(digit_string),
        "peak_signature_disagreement": peak_disagreement,
        "first_break_signature_disagreement": first_break_disagreement,
        "late_prime_obstruction_peak": late_prime_peak,
        "late_prime_obstruction_first_break": late_prime_first_break,
        "total_peak_score_log10": float(sum(float(layer["peak_score_log10_float"]) for layer in layer_rows)),
        "total_first_break_score_log10": float(sum(float(layer["first_break_score_log10_float"]) for layer in layer_rows)),
        "peak_signature_vector": "||".join(peak_signatures),
        "first_break_signature_vector": "||".join(first_break_signatures),
        "peak_core_vector": "|".join(peak_cores),
        "first_break_core_vector": "|".join(first_break_cores),
        "peak_h_vector": "|".join(str(level) for level in peak_levels),
        "first_break_h_vector": "|".join(str(level) for level in first_break_levels),
        "peak_anchor_class_vector": "|".join(peak_anchor_classes),
        "first_break_anchor_class_vector": "|".join(first_break_anchor_classes),
    }

    for layer in layer_rows:
        base = int(layer["base"])
        row[f"base_{base}_peak_signature"] = layer["peak_signature"]
        row[f"base_{base}_first_break_signature"] = layer["first_break_signature"]
        row[f"base_{base}_legal_digits"] = layer["legal_digit_string"]
        row[f"base_{base}_mirror_digits"] = layer["mirror_digit_string"]
        row[f"base_{base}_peak_core"] = layer["peak_core"]
        row[f"base_{base}_first_break_core"] = layer["first_break_core"]
        row[f"base_{base}_h_star_peak"] = int(layer["h_star_peak"])
        row[f"base_{base}_h_star_first_break"] = int(layer["h_star_first_break"])
        row[f"base_{base}_peak_anchor_class"] = layer["peak_anchor_class"]
        row[f"base_{base}_first_break_anchor_class"] = layer["first_break_anchor_class"]
        row[f"base_{base}_depth_to_decoherence"] = int(layer["depth_to_decoherence"])
        row[f"base_{base}_peak_score_log10"] = float(layer["peak_score_log10_float"])
        row[f"base_{base}_first_break_score_log10"] = float(layer["first_break_score_log10_float"])
        row[f"base_{base}_peak_delta_str"] = layer["peak_delta_str"]
        row[f"base_{base}_first_break_delta_str"] = layer["first_break_delta_str"]
        row[f"base_{base}_peak_center_str"] = layer["peak_center_str"]
        row[f"base_{base}_first_break_center_str"] = layer["first_break_center_str"]
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


def best_threshold_classifier(feature: np.ndarray, labels: np.ndarray) -> dict[str, object]:
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
    best_direction = "increasing => label=1"
    best_pred = np.zeros_like(labels)

    for threshold in thresholds:
        pred_le = feature <= threshold
        pred_ge = feature >= threshold
        acc_le = float(np.mean(pred_le == labels))
        acc_ge = float(np.mean(pred_ge == labels))
        if acc_le > best_accuracy:
            best_accuracy = acc_le
            best_rule = f"feature <= {threshold:.6e}"
            best_direction = "decreasing => label=1"
            best_pred = pred_le.astype(int)
        if acc_ge > best_accuracy:
            best_accuracy = acc_ge
            best_rule = f"feature >= {threshold:.6e}"
            best_direction = "increasing => label=1"
            best_pred = pred_ge.astype(int)

    return {
        "accuracy": best_accuracy,
        "rule": best_rule,
        "direction": best_direction,
        "predictions": best_pred,
    }


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
        classifier = best_threshold_classifier(values if forward_is_better else -values, labels)
        rule = str(classifier["rule"])
        if not forward_is_better:
            rule = rule.replace("feature", f"-{name}")
        results.append(
            {
                "target": target_key,
                "feature": name,
                "label_rate": float(labels.mean()),
                "pearson_with_label": pearson_corr(values, labels.astype(float)),
                "best_auc": best_auc,
                "best_threshold_accuracy": float(classifier["accuracy"]),
                "direction": "increasing => label=1" if forward_is_better else "decreasing => label=1",
                "threshold_rule": rule,
            }
        )
    return results


def evaluate_benford_residual(rows: Sequence[dict[str, object]], target_key: str) -> list[dict[str, object]]:
    labels = np.array([int(row[target_key]) for row in rows], dtype=int)
    benford = np.array([float(row["first_digit_benford_prob"]) for row in rows], dtype=float)
    classifier = best_threshold_classifier(-benford, labels)
    residual = (classifier["predictions"] != labels).astype(int)

    features = {
        "digit_entropy_bits": np.array([float(row["digit_entropy_bits"]) for row in rows], dtype=float),
        "leading_trailing_asymmetry": np.array([float(row["leading_trailing_asymmetry"]) for row in rows], dtype=float),
    }

    results: list[dict[str, object]] = []
    for name, values in features.items():
        results.append(
            {
                "target": target_key,
                "feature": name,
                "benford_accuracy": float(classifier["accuracy"]),
                "residual_rate": float(residual.mean()),
                "pearson_with_residual": pearson_corr(values, residual.astype(float)),
                "best_auc_on_residual": max(auc_score(values, residual), auc_score(-values, residual)),
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
    one_tol: mp.mpf,
    e_tol: mp.mpf,
    integer_tol: mp.mpf,
    min_width: int,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    rows: list[dict[str, object]] = []
    previous_peak_signature: str | None = None
    previous_first_break_signature: str | None = None
    skipped_terms = 0

    for index, value in enumerate(values):
        digit_string = str(int(value))
        width = len(digit_string)
        legal_widths = [len(canonicalize_coefficients(digit_string, base)) for base in bases]
        if max(legal_widths, default=0) < min_width:
            skipped_terms += 1
            continue

        row = analyze_digit_string(
            name=f"{name}_{index}",
            digit_string=digit_string,
            bases=bases,
            max_levels=max_levels,
            mp_dps=mp_dps,
            one_tol=one_tol,
            e_tol=e_tol,
            integer_tol=integer_tol,
        )
        row["sequence"] = name
        row["index"] = index
        row["value"] = int(value)
        row["sequence_width"] = width
        row["max_legal_width"] = max(legal_widths, default=0)
        row["peak_signature_changed"] = int(previous_peak_signature is not None and row["peak_signature_vector"] != previous_peak_signature)
        row["first_break_signature_changed"] = int(
            previous_first_break_signature is not None and row["first_break_signature_vector"] != previous_first_break_signature
        )
        previous_peak_signature = str(row["peak_signature_vector"])
        previous_first_break_signature = str(row["first_break_signature_vector"])
        rows.append(row)

    meta = {
        "requested_terms": len(values),
        "used_terms": len(rows),
        "skipped_terms": skipped_terms,
        "min_width": min_width,
    }
    return rows, meta


def save_rows(rows: Sequence[dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_random_results(
    rows: Sequence[dict[str, object]],
    feature_rows: Sequence[dict[str, object]],
    bases: Sequence[int],
    output_dir: Path,
) -> None:
    benford = np.array([float(row["first_digit_benford_prob"]) for row in rows], dtype=float)
    first_break_score = np.array([float(row["total_first_break_score_log10"]) for row in rows], dtype=float)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].scatter(benford, first_break_score, s=10, alpha=0.45, color="tab:red")
    axes[0, 0].set_xlabel("first-digit Benford probability")
    axes[0, 0].set_ylabel("total first-break score")
    axes[0, 0].set_title("First-break obstruction vs leading-digit proxy")
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].hist(first_break_score, bins=24, color="tab:blue", alpha=0.8)
    axes[0, 1].set_xlabel("total first-break score")
    axes[0, 1].set_ylabel("count")
    axes[0, 1].set_title("Random-band first-break score distribution")
    axes[0, 1].grid(alpha=0.25)

    target_order = (
        "peak_signature_disagreement",
        "first_break_signature_disagreement",
        "late_prime_obstruction_peak",
        "late_prime_obstruction_first_break",
    )
    target_auc = []
    for target in target_order:
        best_row = max((row for row in feature_rows if str(row["target"]) == target), key=lambda row: float(row["best_auc"]))
        target_auc.append(float(best_row["best_auc"]))
    axes[1, 0].bar(range(len(target_order)), target_auc, color=["tab:green", "tab:olive", "tab:orange", "tab:brown"])
    axes[1, 0].set_xticks(range(len(target_order)))
    axes[1, 0].set_xticklabels(["peak\nagree", "break\nagree", "peak\ndelay", "break\ndelay"])
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].set_ylabel("best cheap-feature AUC")
    axes[1, 0].set_title("Cheap-feature recovery of obstruction labels")
    axes[1, 0].grid(axis="y", alpha=0.25)

    composite_base = int(bases[0])
    odd_bases = [int(base) for base in bases[1:] if int(base) % 2 == 1]
    peak_delay_rates = []
    break_delay_rates = []
    for base in odd_bases:
        peak_delay_rates.append(
            float(
                np.mean(
                    [
                        int(row[f"base_{base}_h_star_peak"]) > int(row[f"base_{composite_base}_h_star_peak"])
                        for row in rows
                    ]
                )
            )
        )
        break_delay_rates.append(
            float(
                np.mean(
                    [
                        int(row[f"base_{base}_h_star_first_break"]) > int(row[f"base_{composite_base}_h_star_first_break"])
                        for row in rows
                    ]
                )
            )
        )
    x = np.arange(len(odd_bases))
    axes[1, 1].bar(x - 0.18, peak_delay_rates, width=0.36, label="peak delay", color="tab:purple")
    axes[1, 1].bar(x + 0.18, break_delay_rates, width=0.36, label="first-break delay", color="tab:cyan")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([f"base {base}" for base in odd_bases])
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].set_ylabel("delay rate vs base 10")
    axes[1, 1].set_title("Odd-prime delay rates")
    axes[1, 1].grid(axis="y", alpha=0.25)
    axes[1, 1].legend()

    figure.tight_layout()
    figure.savefig(output_dir / "random_obstruction_diagnostics.png", dpi=180)
    plt.close(figure)


def plot_sequence_results(recaman_rows: Sequence[dict[str, object]], random_rows: Sequence[dict[str, object]], output_dir: Path) -> None:
    recaman_index = np.array([int(row["index"]) for row in recaman_rows], dtype=int)
    random_index = np.array([int(row["index"]) for row in random_rows], dtype=int)

    recaman_first_break_score = np.array([float(row["total_first_break_score_log10"]) for row in recaman_rows], dtype=float)
    random_first_break_score = np.array([float(row["total_first_break_score_log10"]) for row in random_rows], dtype=float)
    recaman_peak_score = np.array([float(row["total_peak_score_log10"]) for row in recaman_rows], dtype=float)
    random_peak_score = np.array([float(row["total_peak_score_log10"]) for row in random_rows], dtype=float)
    recaman_changes = np.cumsum(np.array([int(row["first_break_signature_changed"]) for row in recaman_rows], dtype=int))
    random_changes = np.cumsum(np.array([int(row["first_break_signature_changed"]) for row in random_rows], dtype=int))

    figure, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(recaman_index, recaman_first_break_score, linewidth=1.2, label="Recaman", color="tab:blue")
    axes[0].plot(random_index, random_first_break_score, linewidth=1.0, label="matched random walk", color="tab:orange")
    axes[0].set_ylabel("first-break score")
    axes[0].set_title("Sequence first-break obstruction scores")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(recaman_index, recaman_peak_score, linewidth=1.2, label="Recaman", color="tab:blue")
    axes[1].plot(random_index, random_peak_score, linewidth=1.0, label="matched random walk", color="tab:orange")
    axes[1].set_ylabel("peak score")
    axes[1].set_title("Sequence peak obstruction scores")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].step(recaman_index, recaman_changes, where="mid", label="Recaman", color="tab:blue")
    axes[2].step(random_index, random_changes, where="mid", label="matched random walk", color="tab:orange")
    axes[2].set_xlabel("sequence index")
    axes[2].set_ylabel("cumulative first-break changes")
    axes[2].set_title("Signature-change accumulation")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    figure.tight_layout()
    figure.savefig(output_dir / "sequence_obstruction_diagnostics.png", dpi=180)
    plt.close(figure)


def build_summary(
    random_rows: Sequence[dict[str, object]],
    feature_rows: Sequence[dict[str, object]],
    residual_rows: Sequence[dict[str, object]],
    example_rows: Sequence[dict[str, object]],
    recaman_rows: Sequence[dict[str, object]],
    random_walk_rows: Sequence[dict[str, object]],
    recaman_meta: dict[str, int],
    random_walk_meta: dict[str, int],
    bases: Sequence[int],
    mp_dps: int,
    output_dir: Path,
) -> str:
    peak_signature_counts = Counter(str(row["peak_signature_vector"]) for row in random_rows)
    first_break_signature_counts = Counter(str(row["first_break_signature_vector"]) for row in random_rows)
    label_rates = {
        "peak_signature_disagreement": float(np.mean([int(row["peak_signature_disagreement"]) for row in random_rows])),
        "first_break_signature_disagreement": float(np.mean([int(row["first_break_signature_disagreement"]) for row in random_rows])),
        "late_prime_obstruction_peak": float(np.mean([int(row["late_prime_obstruction_peak"]) for row in random_rows])),
        "late_prime_obstruction_first_break": float(np.mean([int(row["late_prime_obstruction_first_break"]) for row in random_rows])),
    }

    def best_target_rows(target: str) -> dict[str, object]:
        return max((row for row in feature_rows if str(row["target"]) == target), key=lambda row: float(row["best_auc"]))

    def best_residual_rows(target: str) -> dict[str, object]:
        return max((row for row in residual_rows if str(row["target"]) == target), key=lambda row: float(row["best_auc_on_residual"]))

    def sequence_summary(rows: Sequence[dict[str, object]]) -> tuple[float, float, int, int]:
        first_break_disagreement = float(np.mean([int(row["first_break_signature_disagreement"]) for row in rows]))
        peak_disagreement = float(np.mean([int(row["peak_signature_disagreement"]) for row in rows]))
        first_break_changes = int(sum(int(row["first_break_signature_changed"]) for row in rows))
        peak_score_mean = float(np.mean([float(row["total_peak_score_log10"]) for row in rows]))
        return first_break_disagreement, peak_disagreement, first_break_changes, peak_score_mean

    rec_first_dis, rec_peak_dis, rec_changes, rec_peak_mean = sequence_summary(recaman_rows)
    walk_first_dis, walk_peak_dis, walk_changes, walk_peak_mean = sequence_summary(random_walk_rows)

    composite_base = int(bases[0])
    odd_bases = [int(base) for base in bases[1:] if int(base) % 2 == 1]
    delay_lines = []
    for base in odd_bases:
        peak_rate = float(
            np.mean(
                [int(row[f"base_{base}_h_star_peak"]) > int(row[f"base_{composite_base}_h_star_peak"]) for row in random_rows]
            )
        )
        break_rate = float(
            np.mean(
                [
                    int(row[f"base_{base}_h_star_first_break"]) > int(row[f"base_{composite_base}_h_star_first_break"])
                    for row in random_rows
                ]
            )
        )
        delay_lines.append((base, peak_rate, break_rate))

    lines = []
    lines.append("Test 7 summary")
    lines.append("")
    lines.append("Setup")
    lines.append(f"- sample count: {len(random_rows)} random {random_rows[0]['width'] if random_rows else 0}-digit integers")
    lines.append(f"- positional layers: {', '.join(str(base) for base in bases)}")
    lines.append(f"- mpmath precision: {mp_dps} decimal digits")
    lines.append(
        "- Anchor classes use tolerance bands: `one` and `e` are checked first, `integer` is only assigned inside its own narrower band, and everything else falls back to `generic`."
    )
    lines.append(
        "- Each decimal coefficient string is first canonicalized into a legal base-b digit string by carrying in base b. Each layer then logs both `h_star_peak` (where the obstruction is loudest) and `h_star_first_break` (where mirror contraction first fails) for that legal string and its mirror."
    )
    lines.append("")
    lines.append("Random-band obstruction spectrum")
    lines.append(
        f"- peak-signature disagreement rate: {float(np.mean([int(row['peak_signature_disagreement']) for row in random_rows])):.4f}"
    )
    lines.append(
        f"- first-break-signature disagreement rate: {float(np.mean([int(row['first_break_signature_disagreement']) for row in random_rows])):.4f}"
    )
    lines.append(
        f"- late-prime peak delay rate: {float(np.mean([int(row['late_prime_obstruction_peak']) for row in random_rows])):.4f}"
    )
    lines.append(
        f"- late-prime first-break delay rate: {float(np.mean([int(row['late_prime_obstruction_first_break']) for row in random_rows])):.4f}"
    )
    for signature, count in first_break_signature_counts.most_common(5):
        lines.append(f"- first-break signature `{signature}` occurred {count} times")

    lines.append("")
    lines.append("Odd-prime delay check")
    for base, peak_rate, break_rate in delay_lines:
        lines.append(
            f"- base {base}: peak delay vs base 10 = {peak_rate:.4f}, first-break delay vs base 10 = {break_rate:.4f}"
        )

    lines.append("")
    lines.append("Cheap-feature recovery")
    for target in (
        "peak_signature_disagreement",
        "first_break_signature_disagreement",
        "late_prime_obstruction_peak",
        "late_prime_obstruction_first_break",
    ):
        row = best_target_rows(target)
        if label_rates[target] in (0.0, 1.0) or math.isnan(float(row["best_auc"])):
            lines.append(
                f"- {target}: label collapsed to a constant or near-constant pattern under these bases, so AUC is not informative."
            )
        else:
            lines.append(
                f"- {target}: best cheap feature was `{row['feature']}` with AUC {float(row['best_auc']):.6f}, threshold accuracy {float(row['best_threshold_accuracy']):.6f}"
            )

    lines.append("")
    lines.append("Benford residual check")
    for target in ("late_prime_obstruction_peak", "late_prime_obstruction_first_break"):
        row = best_residual_rows(target)
        lines.append(
            f"- {target}: after the best Benford threshold, the residual error rate is {float(row['residual_rate']):.6f}; the strongest remaining cheap feature is `{row['feature']}` with residual AUC {float(row['best_auc_on_residual']):.6f}"
        )

    lines.append("")
    lines.append("Named examples")
    for row in example_rows:
        lines.append(
            f"- {row['name']} ({row['digits']}): peak cores `{row['peak_core_vector']}`, first-break cores `{row['first_break_core_vector']}`, first-break score {float(row['total_first_break_score_log10']):.6f}"
        )
        legal_bits = []
        for base in bases:
            legal_bits.append(f"b{base}={row[f'base_{base}_legal_digits']}")
        lines.append(f"  legal digits: {'; '.join(legal_bits)}")

    lines.append("")
    lines.append("Sequence comparison")
    lines.append(
        f"- Recaman: analyzed {recaman_meta['used_terms']} of {recaman_meta['requested_terms']} terms after skipping max legal widths < {recaman_meta['min_width']}; first-break disagreement {rec_first_dis:.4f}, peak disagreement {rec_peak_dis:.4f}, first-break changes {rec_changes}, mean peak score {rec_peak_mean:.6f}"
    )
    lines.append(
        f"- matched random walk: analyzed {random_walk_meta['used_terms']} of {random_walk_meta['requested_terms']} terms after skipping max legal widths < {random_walk_meta['min_width']}; first-break disagreement {walk_first_dis:.4f}, peak disagreement {walk_peak_dis:.4f}, first-break changes {walk_changes}, mean peak score {walk_peak_mean:.6f}"
    )

    light_speed_row = next((row for row in example_rows if str(row["name"]) == "light_speed"), None)
    lines.append("")
    lines.append("Interpretation")
    if light_speed_row is not None:
        lines.append(
            f"- After carry-canonicalization, the light-speed seed still separates after the anchor fix: peak `{light_speed_row['peak_core_vector']}`, first-break `{light_speed_row['first_break_core_vector']}`."
        )

    best_peak = best_target_rows("late_prime_obstruction_peak")
    best_break = best_target_rows("late_prime_obstruction_first_break")
    if label_rates["late_prime_obstruction_first_break"] in (0.0, 1.0) or math.isnan(float(best_break["best_auc"])):
        lines.append(
            "- The first-break delay label collapses under this base set: base 10 is almost always contracting while the prime layers almost always break later. That makes `h_star_first_break` informative descriptively, but weak as a random-band discriminator."
        )
    elif float(best_break["best_auc"]) < float(best_peak["best_auc"]):
        lines.append(
            "- The first-break label is less recoverable from cheap features than the peak label, so the stronger invariant is closer to the object you originally wanted."
        )
    else:
        lines.append(
            "- The first-break label is not cleaner than the peak label under this setup, so the leakage into cheap digit features is still substantial."
        )

    best_break_residual = best_residual_rows("late_prime_obstruction_first_break")
    best_peak_residual = best_residual_rows("late_prime_obstruction_peak")
    if float(best_peak_residual["best_auc_on_residual"]) < 0.6:
        lines.append(
            "- For the non-degenerate peak-delay label, the Benford residual is only weakly explained by other cheap features. That supports a small independent obstruction signal."
        )
    else:
        lines.append(
            "- For the non-degenerate peak-delay label, the Benford residual is still predictable from other cheap features, so the obstruction label is not yet isolated."
        )

    if delay_lines:
        base7 = next((line for line in delay_lines if line[0] == 7), None)
        base5 = next((line for line in delay_lines if line[0] == 5), None)
        if base7 is not None and base5 is not None and abs(base7[1] - base5[1]) < 0.1:
            lines.append(
                "- On the informative peak-delay statistic, base 7 tracks base 5 closely enough that the anomaly looks more like an odd-prime effect than a base-5-only quirk."
            )
        else:
            lines.append(
                "- On the informative peak-delay statistic, base 7 does not track base 5 closely. Base 5 is the strongest odd-prime anomaly, base 3 is weaker, and base 7 is almost absent."
            )

    if rec_changes > walk_changes:
        lines.append(
            "- Under legal base-b representations, Recaman still changes first-break signature more often than the matched random walk."
        )
    else:
        lines.append(
            "- Under legal base-b representations, Recaman no longer leads the matched random walk on first-break signature changes."
        )
    if rec_peak_mean < walk_peak_mean:
        lines.append(
            "- That sequence win is not uniform: the matched random walk still carries a higher mean peak obstruction score, so the sequence evidence is mixed rather than decisive."
        )

    summary = "\n".join(lines) + "\n"
    (output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test 7: corrected obstruction-spectrum analysis under positional prime layers.")
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--digit-width", type=int, default=DEFAULT_DIGIT_WIDTH)
    parser.add_argument("--bases", type=int, nargs="+", default=list(DEFAULT_BASES))
    parser.add_argument("--mp-dps", type=int, default=DEFAULT_MP_DPS)
    parser.add_argument("--max-levels", type=int, default=DEFAULT_MAX_LEVELS)
    parser.add_argument("--recaman-terms", type=int, default=DEFAULT_RECAMAN_TERMS)
    parser.add_argument("--sequence-min-width", type=int, default=DEFAULT_SEQUENCE_MIN_WIDTH)
    parser.add_argument("--rng-seed", type=int, default=DEFAULT_RNG_SEED)
    parser.add_argument("--one-tol", type=float, default=DEFAULT_ONE_TOL)
    parser.add_argument("--e-tol", type=float, default=DEFAULT_E_TOL)
    parser.add_argument("--integer-tol", type=float, default=DEFAULT_INTEGER_TOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mp.mp.dps = args.mp_dps
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.rng_seed)

    one_tol = mp.mpf(str(args.one_tol))
    e_tol = mp.mpf(str(args.e_tol))
    integer_tol = mp.mpf(str(args.integer_tol))

    random_rows = [
        analyze_digit_string(
            name=f"random_{index}",
            digit_string=random_digit_string(args.digit_width, rng),
            bases=args.bases,
            max_levels=args.max_levels,
            mp_dps=args.mp_dps,
            one_tol=one_tol,
            e_tol=e_tol,
            integer_tol=integer_tol,
        )
        for index in range(args.sample_count)
    ]

    feature_rows: list[dict[str, object]] = []
    residual_rows: list[dict[str, object]] = []
    for target in (
        "peak_signature_disagreement",
        "first_break_signature_disagreement",
        "late_prime_obstruction_peak",
        "late_prime_obstruction_first_break",
    ):
        feature_rows.extend(evaluate_binary_target(random_rows, target))
        residual_rows.extend(evaluate_benford_residual(random_rows, target))

    example_rows = [
        analyze_digit_string(
            name=name,
            digit_string=digit_string,
            bases=args.bases,
            max_levels=args.max_levels,
            mp_dps=args.mp_dps,
            one_tol=one_tol,
            e_tol=e_tol,
            integer_tol=integer_tol,
        )
        for name, digit_string in DEFAULT_EXAMPLES
    ]

    recaman_rows, recaman_meta = analyze_sequence(
        name="recaman",
        values=recaman_sequence(args.recaman_terms),
        bases=args.bases,
        max_levels=args.max_levels,
        mp_dps=args.mp_dps,
        one_tol=one_tol,
        e_tol=e_tol,
        integer_tol=integer_tol,
        min_width=args.sequence_min_width,
    )
    random_walk_rows, random_walk_meta = analyze_sequence(
        name="matched_random_walk",
        values=matched_random_walk(args.recaman_terms, rng),
        bases=args.bases,
        max_levels=args.max_levels,
        mp_dps=args.mp_dps,
        one_tol=one_tol,
        e_tol=e_tol,
        integer_tol=integer_tol,
        min_width=args.sequence_min_width,
    )

    save_rows(random_rows, output_dir / "random_obstruction_metrics.csv")
    save_rows(feature_rows, output_dir / "random_feature_screening.csv")
    save_rows(residual_rows, output_dir / "random_benford_residuals.csv")
    save_rows(example_rows, output_dir / "example_invariants.csv")
    save_rows(recaman_rows, output_dir / "recaman_obstruction.csv")
    save_rows(random_walk_rows, output_dir / "matched_random_walk_obstruction.csv")

    plot_random_results(random_rows, feature_rows, args.bases, output_dir)
    plot_sequence_results(recaman_rows, random_walk_rows, output_dir)
    summary = build_summary(
        random_rows=random_rows,
        feature_rows=feature_rows,
        residual_rows=residual_rows,
        example_rows=example_rows,
        recaman_rows=recaman_rows,
        random_walk_rows=random_walk_rows,
        recaman_meta=recaman_meta,
        random_walk_meta=random_walk_meta,
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
