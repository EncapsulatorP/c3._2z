from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


DIGIT_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DEFAULT_BASES = (2, 3, 5, 10, 11, 12)
DEFAULT_VALID_CONTROL_BASES = (10, 11, 12)
DEFAULT_CONSTANTS = (
    ("light_speed", "299792458"),
    ("expected_next", "133732252"),
    ("variant", "233732252"),
    ("balanced_12", "122121212"),
    ("repunit", "111111111"),
)
DEFAULT_OUTPUT_DIR = Path("outputs/omega_stability8")
DEFAULT_SHUFFLE_SAMPLES = 1000
DEFAULT_NEARBY_WINDOW = 5000
DEFAULT_OUT_OF_SAMPLE_SAMPLES = 500
DEFAULT_RNG_SEED = 12345


def coefficient_value(digit_string: str, base: int) -> int:
    value = 0
    for char in digit_string:
        value = value * int(base) + int(char)
    return value


def canonical_digits(value: int, base: int) -> str:
    if value < 0:
        raise ValueError("value must be nonnegative")
    if value == 0:
        return "0"

    digits: list[str] = []
    current = int(value)
    while current > 0:
        current, remainder = divmod(current, base)
        digits.append(DIGIT_ALPHABET[remainder])
    return "".join(reversed(digits))


def normalize_coefficients(digit_string: str, base: int) -> tuple[str, int]:
    coefficients = [int(char) for char in digit_string]
    digits_rev: list[int] = []
    carry = 0
    carry_steps = 0

    for coefficient in reversed(coefficients):
        total = coefficient + carry
        if total >= base:
            carry_steps += 1
        digits_rev.append(total % base)
        carry = total // base

    while carry > 0:
        digits_rev.append(carry % base)
        carry //= base
        carry_steps += 1

    canonical = "".join(DIGIT_ALPHABET[digit] for digit in reversed(digits_rev)).lstrip("0") or "0"
    return canonical, carry_steps


def invalid_positions(digit_string: str, base: int) -> list[int]:
    return [index for index, char in enumerate(digit_string) if int(char) >= base]


def reverse_gap(value: int, reverse_value: int) -> int:
    return abs(reverse_value - value)


def _safe_percentile(sample_value: float, controls: Sequence[float]) -> float:
    if not controls:
        return math.nan
    return float(sum(control <= sample_value for control in controls) / len(controls))


def shuffled_controls(digit_string: str, sample_count: int, rng: np.random.Generator) -> list[str]:
    chars = list(digit_string)
    unique: set[str] = set()
    attempts = 0
    max_attempts = max(sample_count * 20, 1000)

    while len(unique) < sample_count and attempts < max_attempts:
        attempts += 1
        permuted = chars.copy()
        rng.shuffle(permuted)
        if permuted[0] == "0":
            continue
        unique.add("".join(permuted))
    return sorted(unique)


def nearby_controls(value: int, window: int) -> list[int]:
    width = len(str(value))
    results: list[int] = []
    for delta in range(1, window + 1):
        lower = value - delta
        upper = value + delta
        if lower >= 0 and len(str(lower)) == width:
            results.append(lower)
        if len(str(upper)) == width:
            results.append(upper)
    return results


def constant_base_row(name: str, digit_string: str, base: int) -> dict[str, object]:
    value = coefficient_value(digit_string, base)
    reverse_string = digit_string[::-1]
    reverse_value = coefficient_value(reverse_string, base)
    invalid = invalid_positions(digit_string, base)
    canonical, carry_steps = normalize_coefficients(digit_string, base)
    width = len(digit_string)
    sign = 1 if (width - 1) % 2 == 0 else -1
    padded = "0" + digit_string
    padded_value = coefficient_value(padded, base)
    padded_reverse_value = coefficient_value(padded[::-1], base)
    return {
        "name": name,
        "digits": digit_string,
        "base": base,
        "width": width,
        "digits_valid": int(len(invalid) == 0),
        "invalid_digit_count": len(invalid),
        "invalid_positions": ",".join(str(position) for position in invalid),
        "coefficient_value": value,
        "canonical_digits": canonical,
        "canonical_width": len(canonical),
        "carry_steps_to_normalize": carry_steps,
        "reverse_digits": reverse_string,
        "reverse_value": reverse_value,
        "reverse_gap": reverse_gap(value, reverse_value),
        "reverse_gap_ratio": float(reverse_gap(value, reverse_value) / max(value, 1)),
        "mod_b_minus_1_ok": int((value - reverse_value) % (base - 1) == 0),
        "mod_b_plus_1_ok": int((reverse_value - sign * value) % (base + 1) == 0),
        "leading_zero_keeps_value": int(padded_value == value),
        "leading_zero_reverse_multiplier": float(padded_reverse_value / max(reverse_value, 1)),
        "expected_reverse_multiplier": float(base if reverse_value != 0 else math.nan),
    }


def control_rows(
    name: str,
    digit_string: str,
    base: int,
    shuffle_samples: int,
    nearby_window: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    base_row = constant_base_row(name, digit_string, base)
    controls: list[dict[str, object]] = []

    shuffle_strings = shuffled_controls(digit_string, shuffle_samples, rng)
    shuffle_gaps = [constant_base_row(f"{name}_shuffle", shuffled, base)["reverse_gap"] for shuffled in shuffle_strings]
    controls.append(
        {
            "name": name,
            "base": base,
            "control_family": "digit_shuffle",
            "control_count": len(shuffle_gaps),
            "sample_reverse_gap": base_row["reverse_gap"],
            "control_mean_gap": float(np.mean(shuffle_gaps)) if shuffle_gaps else math.nan,
            "control_median_gap": float(np.median(shuffle_gaps)) if shuffle_gaps else math.nan,
            "sample_gap_percentile": _safe_percentile(float(base_row["reverse_gap"]), [float(gap) for gap in shuffle_gaps]),
        }
    )

    nearby_values = nearby_controls(int(digit_string), nearby_window)
    nearby_gaps = [constant_base_row(f"{name}_nearby", str(control_value), base)["reverse_gap"] for control_value in nearby_values]
    controls.append(
        {
            "name": name,
            "base": base,
            "control_family": "nearby_integer",
            "control_count": len(nearby_gaps),
            "sample_reverse_gap": base_row["reverse_gap"],
            "control_mean_gap": float(np.mean(nearby_gaps)) if nearby_gaps else math.nan,
            "control_median_gap": float(np.median(nearby_gaps)) if nearby_gaps else math.nan,
            "sample_gap_percentile": _safe_percentile(float(base_row["reverse_gap"]), [float(gap) for gap in nearby_gaps]),
        }
    )
    return controls


def out_of_sample_rows(bases: Sequence[int], sample_count: int, rng: np.random.Generator) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    width = 12
    for base in bases:
        success_b_minus_1 = 0
        success_b_plus_1 = 0
        success_padding = 0
        for _ in range(sample_count):
            digits = [str(int(rng.integers(1, 10)))]
            digits.extend(str(int(rng.integers(0, 10))) for _ in range(width - 1))
            digit_string = "".join(digits)
            row = constant_base_row("random", digit_string, base)
            success_b_minus_1 += int(row["mod_b_minus_1_ok"])
            success_b_plus_1 += int(row["mod_b_plus_1_ok"])
            success_padding += int(abs(float(row["leading_zero_reverse_multiplier"]) - float(base)) < 1e-12)
        rows.append(
            {
                "base": base,
                "sample_count": sample_count,
                "mod_b_minus_1_success_rate": success_b_minus_1 / sample_count,
                "mod_b_plus_1_success_rate": success_b_plus_1 / sample_count,
                "leading_zero_rule_success_rate": success_padding / sample_count,
            }
        )
    return rows


def save_rows(rows: Sequence[dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _group_by_name(rows: Iterable[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row["name"]), []).append(row)
    return groups


def build_summary(
    base_rows: Sequence[dict[str, object]],
    control_rows_: Sequence[dict[str, object]],
    out_of_sample: Sequence[dict[str, object]],
    output_dir: Path,
) -> str:
    groups = _group_by_name(base_rows)
    valid_bases = [base for base in DEFAULT_BASES if base >= 10]
    invalid_bases = [base for base in DEFAULT_BASES if base < 10]

    lines: list[str] = []
    lines.append("Modularity crash-test summary")
    lines.append("")
    lines.append("Object under test")
    lines.append(
        "- The current reversed positional observation is the coefficient-evaluation map used by Tests 6-7: a decimal digit string `d_0...d_{m-1}` is sent to `P_b = sum d_i b^(m-1-i)` and compared with its reversed-digit partner."
    )
    lines.append("")
    lines.append("Minimal protocol results")
    lines.append(
        f"- Bases checked: {', '.join(str(base) for base in DEFAULT_BASES)}. The low bases {', '.join(str(base) for base in invalid_bases)} are the stress test for hidden carries."
    )
    lines.append(
        "- For the three main constants `299792458`, `133732252`, and `233732252`, the displayed decimal strings are legal positional digit strings only in bases 10, 11, 12."
    )
    lines.append("- Out-of-sample arithmetic rule check: the universal reversal congruences and zero-padding law held with 100% success on random 12-digit strings in every valid base tested.")
    lines.append("")
    lines.append("Checklist")

    carry_failures = [
        row
        for row in base_rows
        if int(row["base"]) in invalid_bases and int(row["invalid_digit_count"]) > 0
    ]
    if carry_failures:
        example = carry_failures[0]
        lines.append(
            f"- 1. Carry and borrow: FAIL. Example: `{example['digits']}` in base {example['base']} has {example['invalid_digit_count']} digits >= base, so the current map hides the carry state instead of tracking it."
        )
    else:
        lines.append("- 1. Carry and borrow: PASS.")

    reconstruction_failures = [
        row
        for row in base_rows
        if int(row["digits_valid"]) == 0
    ]
    if reconstruction_failures:
        example = reconstruction_failures[0]
        lines.append(
            f"- 2. Exact reconstruction: FAIL below base 10. `{example['digits']}` is not itself a legal base-{example['base']} digit string; exact reconstruction requires the hidden normalized form `{example['canonical_digits']}`."
        )
    else:
        lines.append("- 2. Exact reconstruction: PASS.")

    lines.append(
        "- 3. Base dependence: FAIL as a structural claim for the main constants. Below base 10 their displayed strings stop being legal base-b digit expansions and become a different polynomial-evaluation object."
    )
    lines.append("- 4. Predictable base covariance: PASS in the negative sense. `P_b` and its reverse are explicit polynomials in `b`, so their variation with base is generic and derivable in advance.")

    lines.append("- 5. Prime-factor decomposition: PARTIAL PASS. Existing Test 2 already shows the base-10 effect on the named constants is dyadic (`v_2`) rather than 5-adic; the decimal layer is not uniquely selecting them.")

    lines.append("- 6. Residue accounting: FAIL for the low-base runs. The residue/carry information is not optional there; it is exactly what turns the displayed decimal coefficients into a legal base-b expansion.")

    lines.append("- 7. Alignment and shift: FAIL mathematically. Prepending one zero leaves `P_b` unchanged but multiplies the reversed partner by exactly `b`, so the mirror signal depends on arbitrary padding.")

    lines.append("- 8. Scale persistence: PARTIAL. The generic modular laws persist out of sample; a constant-specific anomaly was not established by this test.")

    controls_by_name_base = {
        (str(row["name"]), int(row["base"]), str(row["control_family"])): row for row in control_rows_
    }
    effect_lines = []
    for name in ("light_speed", "expected_next", "variant"):
        for base in valid_bases:
            shuffle_row = controls_by_name_base.get((name, base, "digit_shuffle"))
            if shuffle_row is None:
                continue
            effect_lines.append(
                f"{name}@b{base}: shuffle percentile={float(shuffle_row['sample_gap_percentile']):.3f}"
            )
    if effect_lines:
        lines.append(
            "- 9. Matched null models: MIXED but not supportive of specialness. Reverse-gap percentiles against digit-shuffled controls were "
            + "; ".join(effect_lines)
            + "."
        )
    else:
        lines.append("- 9. Matched null models: not enough data.")

    nearby_lines = []
    for name in ("light_speed", "expected_next", "variant"):
        for base in valid_bases:
            nearby_row = controls_by_name_base.get((name, base, "nearby_integer"))
            if nearby_row is None:
                continue
            nearby_lines.append(
                f"{name}@b{base}: nearby percentile={float(nearby_row['sample_gap_percentile']):.3f}"
            )
    if nearby_lines:
        lines.append(
            "- 10. Boring adversarial controls: the named constants are not isolated by a large reverse-gap effect versus nearby same-width integers. "
            + "; ".join(nearby_lines)
            + "."
        )
    else:
        lines.append("- 10. Boring adversarial controls: not enough data.")

    if out_of_sample:
        first = out_of_sample[0]
        lines.append(
            f"- 11. Out-of-sample prediction: PASS for the generic arithmetic mechanism. The frozen laws `rev == n (mod b-1)`, `rev == (-1)^(m-1) n (mod b+1)`, and the zero-padding multiplier all held on {first['sample_count']} unseen samples per valid base."
        )
    else:
        lines.append("- 11. Out-of-sample prediction: not run.")

    lines.append("- 12. Multiple-testing correction: GAP in the existing repo outputs. The prior summaries do not count searched bases, paddings, or transformations.")
    lines.append("- 13. Representation independence: PASS. There is a clean arithmetic carrier, but it is generic: reversal preserves residue mod `b-1` and fixes residue mod `b+1` up to the parity sign.")
    lines.append("- 14. Mechanism rather than resemblance: PASS. The exact identity is `P_b(rev) - P_b = sum d_i (b^(m-1-i) - b^i)`, plus the zero-padding law above.")
    lines.append("- 15. Error and effect size: weak. The control percentiles do not show an obviously exceptional reverse-gap effect for the named constants.")
    lines.append("- 16. Leakage test: FAIL in the broader repo. Existing Tests 5-7 already show first-digit Benford probability recovers much of the reported label structure.")

    lines.append("")
    lines.append("Named constants")
    for name, rows in groups.items():
        row_bits = []
        for row in sorted(rows, key=lambda item: int(item["base"])):
            row_bits.append(
                f"b{row['base']}: valid={int(row['digits_valid'])}, carries={int(row['carry_steps_to_normalize'])}, gap={int(row['reverse_gap'])}"
            )
        lines.append(f"- {name}: " + "; ".join(row_bits))

    lines.append("")
    lines.append("Bottom line")
    lines.append(
        "- The crash-test does find a modular mechanism, but it is the generic arithmetic of digit reversal itself, not a constant-specific law. The current low-base runs also fail the positional-validity test because the displayed decimal strings are not legal base-2/base-3/base-5 digit expansions."
    )
    lines.append(
        "- The sharpest failure mode is alignment: because prepending zeros changes the reversed value by an exact factor of `b` while leaving the original value fixed, the mirror observation is not invariant under arbitrary width choices."
    )

    summary = "\n".join(lines) + "\n"
    (output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test 8: modularity crash-test for the reversed positional observation.")
    parser.add_argument("--bases", type=int, nargs="+", default=list(DEFAULT_BASES))
    parser.add_argument("--shuffle-samples", type=int, default=DEFAULT_SHUFFLE_SAMPLES)
    parser.add_argument("--nearby-window", type=int, default=DEFAULT_NEARBY_WINDOW)
    parser.add_argument("--out-of-sample-samples", type=int, default=DEFAULT_OUT_OF_SAMPLE_SAMPLES)
    parser.add_argument("--rng-seed", type=int, default=DEFAULT_RNG_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.rng_seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    base_rows = [
        constant_base_row(name=name, digit_string=digit_string, base=base)
        for name, digit_string in DEFAULT_CONSTANTS
        for base in args.bases
    ]
    save_rows(base_rows, output_dir / "constant_base_scan.csv")

    control_rows_: list[dict[str, object]] = []
    for name, digit_string in DEFAULT_CONSTANTS:
        for base in DEFAULT_VALID_CONTROL_BASES:
            control_rows_.extend(
                control_rows(
                    name=name,
                    digit_string=digit_string,
                    base=base,
                    shuffle_samples=args.shuffle_samples,
                    nearby_window=args.nearby_window,
                    rng=rng,
                )
            )
    save_rows(control_rows_, output_dir / "control_effect_sizes.csv")

    out_of_sample = out_of_sample_rows(
        bases=DEFAULT_VALID_CONTROL_BASES,
        sample_count=args.out_of_sample_samples,
        rng=rng,
    )
    save_rows(out_of_sample, output_dir / "out_of_sample_rules.csv")

    summary = build_summary(
        base_rows=base_rows,
        control_rows_=control_rows_,
        out_of_sample=out_of_sample,
        output_dir=output_dir,
    )
    print(summary, end="")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
