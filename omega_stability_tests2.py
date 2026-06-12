from __future__ import annotations

import argparse
import csv
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

_MPLCONFIGDIR = Path("/tmp/mplconfig")
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DIGIT_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DEFAULT_BASES = (8, 10, 16)
DEFAULT_SEED = 299792458
DEFAULT_EXPECTED_NEXT = 233732252
DEFAULT_RANDOM_SAMPLES = 1000
DEFAULT_MAX_STEPS = 60
DEFAULT_RNG_SEED = 123457
DEFAULT_OUTPUT_DIR = Path("outputs/omega_stability2")
OPERATOR_LABEL = "example operator: cyclic adjacent sum modulo base"
DEFAULT_VALUATION_BASE = 10
DEFAULT_LOG_BASE = 10.0
DEFAULT_LOG_DEPTH = 3
DEFAULT_VARIANT_VALUE = 233732252


# Replace this one-line operator with the actual rule before treating the results as evidence about your chain.
def T(digits: Sequence[int], base: int) -> list[int]:
    return [((left + right) % base) for left, right in zip(digits, list(digits[1:]) + [digits[0]])]


@dataclass
class ChainResult:
    seed_value: int
    base: int
    width: int
    status: str
    repeat_start: int
    cycle_length: int
    states: list[tuple[int, ...]]

    @property
    def steps_taken(self) -> int:
        return max(0, len(self.states) - 1)

    @property
    def terminal_state(self) -> tuple[int, ...]:
        return self.states[-1]

    @property
    def cycle_states(self) -> list[tuple[int, ...]]:
        if self.status == "max_steps":
            return []
        return self.states[self.repeat_start : self.repeat_start + self.cycle_length]


def _validate_base(base: int) -> int:
    base = int(base)
    if base < 2 or base > len(DIGIT_ALPHABET):
        raise ValueError(f"base must be in [2, {len(DIGIT_ALPHABET)}], got {base}")
    return base


def _validate_width(width: int) -> int:
    width = int(width)
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}")
    return width


def _natural_width(value: int, base: int) -> int:
    if value == 0:
        return 1
    width = 0
    current = value
    while current > 0:
        current //= base
        width += 1
    return width


def int_to_digits(value: int, base: int, width: int | None = None) -> list[int]:
    base = _validate_base(base)
    if value < 0:
        raise ValueError(f"value must be >= 0, got {value}")

    digits: list[int]
    if value == 0:
        digits = [0]
    else:
        digits = []
        current = value
        while current > 0:
            digits.append(current % base)
            current //= base
        digits.reverse()

    if width is None:
        return digits

    width = _validate_width(width)
    if len(digits) > width:
        raise ValueError(f"value {value} needs {len(digits)} digits in base {base}, exceeds width {width}")
    return [0] * (width - len(digits)) + digits


def digits_to_int(digits: Sequence[int], base: int) -> int:
    base = _validate_base(base)
    value = 0
    for digit in digits:
        digit = int(digit)
        if digit < 0 or digit >= base:
            raise ValueError(f"digit {digit} is out of range for base {base}")
        value = value * base + digit
    return value


def digits_to_str(digits: Sequence[int]) -> str:
    return "".join(DIGIT_ALPHABET[digit] for digit in digits)


def prime_factorization(value: int) -> list[tuple[int, int]]:
    value = int(value)
    if value < 1:
        raise ValueError(f"value must be >= 1, got {value}")

    remaining = value
    factors: list[tuple[int, int]] = []
    exponent = 0
    while remaining % 2 == 0:
        remaining //= 2
        exponent += 1
    if exponent:
        factors.append((2, exponent))

    divisor = 3
    while divisor * divisor <= remaining:
        exponent = 0
        while remaining % divisor == 0:
            remaining //= divisor
            exponent += 1
        if exponent:
            factors.append((divisor, exponent))
        divisor += 2

    if remaining > 1:
        factors.append((remaining, 1))
    return factors


def factorization_to_str(factors: Sequence[tuple[int, int]]) -> str:
    parts = []
    for prime, exponent in factors:
        if exponent == 1:
            parts.append(str(prime))
        else:
            parts.append(f"{prime}^{exponent}")
    return " * ".join(parts) if parts else "1"


def base_prime_factors(base: int) -> list[int]:
    return [prime for prime, _ in prime_factorization(base)]


def p_adic_valuation(value: int, prime: int) -> int:
    if value == 0:
        raise ValueError("p-adic valuation is not defined for 0")
    if prime <= 1:
        raise ValueError(f"prime must be > 1, got {prime}")

    exponent = 0
    remaining = int(value)
    while remaining % prime == 0:
        remaining //= prime
        exponent += 1
    return exponent


def split_base_part(value: int, base: int) -> tuple[int, int, dict[int, int]]:
    base = _validate_base(base)
    remaining = int(value)
    valuations: dict[int, int] = {}
    base_part = 1
    for prime in base_prime_factors(base):
        exponent = p_adic_valuation(remaining, prime)
        valuations[prime] = exponent
        if exponent:
            factor = prime ** exponent
            base_part *= factor
            remaining //= factor
    return base_part, remaining, valuations


def iterated_log_levels(value: int | float, log_base: float, depth: int) -> list[float]:
    current = float(value)
    levels: list[float] = []
    base_ln = math.log(float(log_base))
    for _ in range(depth):
        if current <= 0.0:
            break
        current = math.log(current) / base_ln
        levels.append(current)
    return levels


def _levels_to_row(levels: Sequence[float], depth: int) -> list[float]:
    row = list(levels[:depth])
    while len(row) < depth:
        row.append(math.nan)
    return row


def digit_entropy(digits: Sequence[int]) -> float:
    counts = Counter(int(digit) for digit in digits)
    total = float(len(digits))
    return -sum((count / total) * math.log(count / total, 2.0) for count in counts.values())


def is_repunit(digits: Sequence[int]) -> bool:
    return len(digits) > 0 and all(digit == 1 for digit in digits)


def is_repdigit(digits: Sequence[int]) -> bool:
    return len(digits) > 0 and all(digit == digits[0] for digit in digits)


def classify_cycle(cycle_states: Sequence[Sequence[int]]) -> str:
    if len(cycle_states) == 0:
        return "max_steps"
    if len(cycle_states) == 1:
        state = cycle_states[0]
        if is_repunit(state):
            return "repunit_fixed_point"
        if is_repdigit(state):
            return "repdigit_fixed_point"
        return "other_fixed_point"
    if any(is_repunit(state) for state in cycle_states):
        return "repunit_cycle"
    if any(is_repdigit(state) for state in cycle_states):
        return "repdigit_cycle"
    return "nontrivial_cycle"


def apply_operator(state: Sequence[int], base: int) -> tuple[int, ...]:
    next_state = tuple(int(digit) for digit in T(state, base))
    if len(next_state) != len(state):
        raise ValueError(f"operator changed width from {len(state)} to {len(next_state)}")
    if any(digit < 0 or digit >= base for digit in next_state):
        raise ValueError(f"operator produced digits outside base {base}: {next_state}")
    return next_state


def iterate_chain(seed_digits: Sequence[int], base: int, max_steps: int) -> ChainResult:
    base = _validate_base(base)
    seed_state = tuple(int(digit) for digit in seed_digits)
    seen = {seed_state: 0}
    states = [seed_state]

    for _ in range(max_steps):
        next_state = apply_operator(states[-1], base)
        if next_state in seen:
            states.append(next_state)
            repeat_start = seen[next_state]
            cycle_length = len(states) - 1 - repeat_start
            status = "fixed_point" if cycle_length == 1 else "cycle"
            return ChainResult(
                seed_value=digits_to_int(seed_state, base),
                base=base,
                width=len(seed_state),
                status=status,
                repeat_start=repeat_start,
                cycle_length=cycle_length,
                states=states,
            )
        seen[next_state] = len(states)
        states.append(next_state)

    return ChainResult(
        seed_value=digits_to_int(seed_state, base),
        base=base,
        width=len(seed_state),
        status="max_steps",
        repeat_start=-1,
        cycle_length=0,
        states=states,
    )


def random_seed_digits(base: int, width: int, rng: np.random.Generator) -> list[int]:
    base = _validate_base(base)
    width = _validate_width(width)
    digits = rng.integers(0, base, size=width, dtype=np.int64).tolist()
    digits[0] = int(rng.integers(1, base, dtype=np.int64))
    return [int(digit) for digit in digits]


def save_chain_csv(result: ChainResult, output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "digits", "value_base10", "entropy_bits"])
        for step, state in enumerate(result.states):
            writer.writerow([step, digits_to_str(state), digits_to_int(state, result.base), f"{digit_entropy(state):.12f}"])


def plot_chain_heatmap(result: ChainResult, output_path: Path) -> None:
    matrix = np.array(result.states, dtype=float)
    plt.figure(figsize=(9, 4.8))
    image = plt.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=result.base - 1)
    plt.colorbar(image, label="digit value")
    plt.xlabel("digit position")
    plt.ylabel("step")
    plt.title(f"Seed chain in base {result.base}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def run_null_model(
    base: int,
    width: int,
    random_samples: int,
    max_steps: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, object]], Counter]:
    rows: list[dict[str, object]] = []
    label_counts: Counter = Counter()

    for sample_index in range(random_samples):
        seed_digits = random_seed_digits(base, width, rng)
        result = iterate_chain(seed_digits, base=base, max_steps=max_steps)
        label = classify_cycle(result.cycle_states)
        label_counts[label] += 1
        rows.append(
            {
                "sample_index": sample_index,
                "base": base,
                "width": width,
                "seed_digits": digits_to_str(seed_digits),
                "seed_value_base10": digits_to_int(seed_digits, base),
                "status": result.status,
                "terminal_label": label,
                "steps_taken": result.steps_taken,
                "cycle_length": result.cycle_length,
                "terminal_digits": digits_to_str(result.terminal_state),
                "terminal_value_base10": digits_to_int(result.terminal_state, base),
            }
        )
    return rows, label_counts


def save_null_rows(rows: Sequence[dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_terminal_rates(
    seed_summaries: Sequence[dict[str, object]],
    null_label_counts: dict[int, Counter],
    random_samples: int,
    output_path: Path,
) -> None:
    categories = sorted({summary["terminal_label"] for summary in seed_summaries} | {label for counts in null_label_counts.values() for label in counts})
    x = np.arange(len(categories))
    width = 0.8 / max(1, len(seed_summaries))

    plt.figure(figsize=(11, 5.5))
    for offset, summary in enumerate(seed_summaries):
        base = int(summary["base"])
        probabilities = [null_label_counts[base][category] / random_samples for category in categories]
        plt.bar(x + offset * width, probabilities, width=width, label=f"base {base}")

    plt.xticks(x + width * (len(seed_summaries) - 1) / 2, categories, rotation=25, ha="right")
    plt.ylabel("probability under null model")
    plt.title("Null-model terminal label rates by base")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_steps_histogram(
    seed_summaries: Sequence[dict[str, object]],
    null_rows_by_base: dict[int, list[dict[str, object]]],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(len(seed_summaries), 1, figsize=(9, 3.6 * len(seed_summaries)), sharex=False)
    if len(seed_summaries) == 1:
        axes = [axes]

    for axis, summary in zip(axes, seed_summaries):
        base = int(summary["base"])
        steps = np.array([int(row["steps_taken"]) for row in null_rows_by_base[base]], dtype=int)
        axis.hist(steps, bins=min(25, max(5, steps.max(initial=1) + 1)), color="tab:blue", alpha=0.75)
        axis.axvline(int(summary["steps_taken"]), color="tab:red", linewidth=1.4, label="seed")
        axis.set_title(f"Steps to repeat under null model, base {base}")
        axis.set_xlabel("steps taken")
        axis.set_ylabel("count")
        axis.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _width_for_base(seed_value: int, base: int, requested_width: int | None) -> int:
    if requested_width is not None:
        return _validate_width(requested_width)
    return _natural_width(seed_value, base)


def _build_seed_summary(result: ChainResult, label: str, expected_next: int | None) -> dict[str, object]:
    next_digits = result.states[1] if len(result.states) > 1 else result.states[0]
    next_value = digits_to_int(next_digits, result.base)
    expected_match = None if expected_next is None or result.base != 10 else (next_value == expected_next)
    return {
        "base": result.base,
        "width": result.width,
        "seed_digits": digits_to_str(result.states[0]),
        "next_digits": digits_to_str(next_digits),
        "next_value_base10": next_value,
        "terminal_digits": digits_to_str(result.terminal_state),
        "terminal_value_base10": digits_to_int(result.terminal_state, result.base),
        "terminal_label": label,
        "status": result.status,
        "steps_taken": result.steps_taken,
        "cycle_length": result.cycle_length,
        "expected_next_match": expected_match,
    }


def _same_or_lower_fraction(values: Sequence[int], probe: int) -> float:
    if not values:
        return math.nan
    return float(sum(value <= probe for value in values) / len(values))


def build_valuation_rows(values: Sequence[int], valuation_base: int, log_base: float, log_depth: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for value in values:
        factors = prime_factorization(value)
        base_part, coprime_part, valuations = split_base_part(value, valuation_base)
        raw_levels = iterated_log_levels(value, log_base=log_base, depth=log_depth)
        coprime_levels = iterated_log_levels(coprime_part, log_base=log_base, depth=log_depth)
        row: dict[str, object] = {
            "value": value,
            "factorization": factorization_to_str(factors),
            "base_part": base_part,
            "coprime_part": coprime_part,
        }
        for prime in base_prime_factors(valuation_base):
            row[f"v_{prime}"] = valuations.get(prime, 0)
        for index, level in enumerate(_levels_to_row(raw_levels, log_depth), start=1):
            row[f"raw_log_level_{index}"] = level
        for index, level in enumerate(_levels_to_row(coprime_levels, log_depth), start=1):
            row[f"coprime_log_level_{index}"] = level
        rows.append(row)
    return rows


def build_pairwise_delta_rows(
    named_values: Sequence[tuple[str, int]],
    valuation_base: int,
    log_base: float,
    log_depth: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (left_name, left_value), (right_name, right_value) in zip(named_values[:-1], named_values[1:]):
        left_base_part, left_coprime, _ = split_base_part(left_value, valuation_base)
        right_base_part, right_coprime, _ = split_base_part(right_value, valuation_base)
        comparisons = [
            ("raw", left_value, right_value, left_value, right_value),
            ("coprime", left_coprime, right_coprime, left_coprime, right_coprime),
        ]
        for mode, left_operand, right_operand, _, _ in comparisons:
            left_levels = iterated_log_levels(left_operand, log_base=log_base, depth=log_depth)
            right_levels = iterated_log_levels(right_operand, log_base=log_base, depth=log_depth)
            for level_index, (left_level, right_level) in enumerate(zip(left_levels, right_levels), start=1):
                rows.append(
                    {
                        "pair": f"{left_name}_to_{right_name}",
                        "mode": mode,
                        "level": level_index,
                        "left_value": left_operand,
                        "right_value": right_operand,
                        "left_log": left_level,
                        "right_log": right_level,
                        "delta": right_level - left_level,
                    }
                )
    return rows


def save_dict_rows(rows: Sequence[dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_pairwise_deltas(pairwise_rows: Sequence[dict[str, object]], output_path: Path) -> None:
    pairs = sorted({str(row["pair"]) for row in pairwise_rows})
    figure, axes = plt.subplots(len(pairs), 1, figsize=(9, 3.6 * len(pairs)), sharex=False)
    if len(pairs) == 1:
        axes = [axes]

    for axis, pair_name in zip(axes, pairs):
        pair_rows = [row for row in pairwise_rows if str(row["pair"]) == pair_name]
        raw_rows = [row for row in pair_rows if str(row["mode"]) == "raw"]
        coprime_rows = [row for row in pair_rows if str(row["mode"]) == "coprime"]
        raw_levels = np.array([int(row["level"]) for row in raw_rows], dtype=int)
        raw_deltas = np.array([float(row["delta"]) for row in raw_rows], dtype=float)
        coprime_levels = np.array([int(row["level"]) for row in coprime_rows], dtype=int)
        coprime_deltas = np.array([float(row["delta"]) for row in coprime_rows], dtype=float)

        axis.plot(raw_levels, raw_deltas, marker="o", color="tab:blue", label="raw")
        axis.plot(coprime_levels, coprime_deltas, marker="o", color="tab:orange", label="coprime-only")
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set_title(pair_name.replace("_", " "))
        axis.set_xlabel("iterated log level")
        axis.set_ylabel("delta")
        axis.grid(alpha=0.25)
        axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def build_valuation_summary(
    valuation_rows: Sequence[dict[str, object]],
    pairwise_rows: Sequence[dict[str, object]],
    valuation_base: int,
    log_base: float,
    output_dir: Path,
) -> str:
    lines = []
    lines.append("Base-pollution strip test summary")
    lines.append("")
    lines.append(f"Positional base: {valuation_base}")
    lines.append(f"Archimedean log base: {log_base:g}")

    lines.append("")
    lines.append("Valuation split")
    for row in valuation_rows:
        prime_bits = ", ".join(
            f"v_{key.split('_')[1]}={row[key]}" for key in row.keys() if key.startswith("v_")
        )
        lines.append(
            f"- {row['value']}: {row['factorization']}; base_part={row['base_part']}; coprime_part={row['coprime_part']}; {prime_bits}"
        )

    lines.append("")
    lines.append("Pairwise archimedean shifts")
    pair_names = sorted({str(row['pair']) for row in pairwise_rows})
    for pair_name in pair_names:
        pair_rows = [row for row in pairwise_rows if str(row["pair"]) == pair_name]
        lines.append(f"- {pair_name}:")
        for mode in ("raw", "coprime"):
            mode_rows = [row for row in pair_rows if str(row["mode"]) == mode]
            deltas = ", ".join(
                f"L{int(row['level'])}={float(row['delta']):.6e}" for row in mode_rows
            )
            lines.append(f"  {mode}: {deltas}")

    lines.append("")
    lines.append("Interpretation")
    expected_variant_rows = [
        row for row in pairwise_rows if str(row["pair"]) == "expected_to_variant" and str(row["mode"]) == "raw"
    ]
    expected_variant_coprime_rows = [
        row for row in pairwise_rows if str(row["pair"]) == "expected_to_variant" and str(row["mode"]) == "coprime"
    ]
    if expected_variant_rows and expected_variant_coprime_rows:
        level1_raw = float(expected_variant_rows[0]["delta"])
        level1_coprime = float(expected_variant_coprime_rows[0]["delta"])
        lines.append(
            f"- For 133732252 vs 233732252, stripping the base-{valuation_base} part leaves the level-1 shift unchanged ({level1_raw:.6e} vs {level1_coprime:.6e}) because both numbers have the same base-part."
        )
        if len(expected_variant_rows) > 1 and len(expected_variant_coprime_rows) > 1:
            level2_raw = float(expected_variant_rows[1]["delta"])
            level2_coprime = float(expected_variant_coprime_rows[1]["delta"])
            lines.append(
                f"- At level 2 the coprime-only shift is slightly larger ({level2_coprime:.6e}) than the raw shift ({level2_raw:.6e}), so the leading-digit perturbation does not disappear after stripping."
            )

    seed_expected_rows = [
        row for row in pairwise_rows if str(row["pair"]) == "seed_to_expected" and str(row["mode"]) == "raw"
    ]
    seed_expected_coprime_rows = [
        row for row in pairwise_rows if str(row["pair"]) == "seed_to_expected" and str(row["mode"]) == "coprime"
    ]
    if seed_expected_rows and seed_expected_coprime_rows:
        lines.append(
            "- For 299792458 vs 133732252, the coprime-only deltas are larger in magnitude than the raw deltas because the two numbers do not share the same 2-adic depth."
        )

    lines.append(
        "- On these examples, 5 contributes nothing: every tested value has v_5 = 0. The only base-10 pollution present is dyadic."
    )

    summary = "\n".join(lines) + "\n"
    (output_dir / "valuation_summary.txt").write_text(summary, encoding="utf-8")
    return summary


def build_summary(
    seed_summaries: Sequence[dict[str, object]],
    null_rows_by_base: dict[int, list[dict[str, object]]],
    null_label_counts: dict[int, Counter],
    expected_next: int | None,
    output_dir: Path,
) -> str:
    lines = []
    lines.append("Digit-symbolic null-model summary")
    lines.append("")
    lines.append(f"Operator: {OPERATOR_LABEL}")
    lines.append("Warning: edit T(...) near the top of the file if this is not your actual chain operator.")

    base10_summary = next((summary for summary in seed_summaries if int(summary["base"]) == 10), None)
    if expected_next is not None and base10_summary is not None:
        match = base10_summary["expected_next_match"]
        lines.append("")
        lines.append("Operator check")
        lines.append(
            f"- T(299792458) produced {base10_summary['next_value_base10']} in base 10; expected {expected_next}; match={'yes' if match else 'no'}"
        )
        if not match:
            lines.append("- The current operator does not reproduce the quoted chain, so the null-model outputs below are about the placeholder rule only.")

    lines.append("")
    lines.append("Seed across bases")
    for summary in seed_summaries:
        lines.append(
            f"- base {summary['base']}: width={summary['width']}, seed={summary['seed_digits']}, next={summary['next_digits']}, terminal={summary['terminal_digits']}, label={summary['terminal_label']}, steps={summary['steps_taken']}, cycle_length={summary['cycle_length']}"
        )

    lines.append("")
    lines.append("Null model")
    for summary in seed_summaries:
        base = int(summary["base"])
        rows = null_rows_by_base[base]
        counts = null_label_counts[base]
        same_label_rate = counts[str(summary["terminal_label"])] / len(rows)
        repunit_rate = (counts["repunit_fixed_point"] + counts["repunit_cycle"]) / len(rows)
        repdigit_rate = (
            counts["repunit_fixed_point"]
            + counts["repunit_cycle"]
            + counts["repdigit_fixed_point"]
            + counts["repdigit_cycle"]
        ) / len(rows)
        step_percentile = _same_or_lower_fraction([int(row["steps_taken"]) for row in rows], int(summary["steps_taken"]))
        lines.append(
            f"- base {base}: P(same terminal label)={same_label_rate:.4f}, P(repunit endpoint)={repunit_rate:.4f}, P(repdigit endpoint)={repdigit_rate:.4f}, step percentile={step_percentile:.4f}"
        )

    terminal_labels = {str(summary["terminal_label"]) for summary in seed_summaries}
    terminal_digits = {str(summary["terminal_digits"]) for summary in seed_summaries}
    lines.append("")
    lines.append("Cross-base consistency")
    lines.append(f"- unique terminal labels across tested bases: {len(terminal_labels)}")
    lines.append(f"- unique terminal digit strings across tested bases: {len(terminal_digits)}")
    if len(terminal_labels) == 1 and len(terminal_digits) == 1:
        lines.append("- The tested seed landed in the same visible endpoint class across bases.")
    else:
        lines.append("- The tested seed changed endpoint class or terminal digits across bases, so any claimed structure is representation-dependent under the current T.")

    summary = "\n".join(lines) + "\n"
    (output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run digit-symbolic chain checks with a null model and base comparison.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="seed value to analyze")
    parser.add_argument("--bases", type=int, nargs="+", default=list(DEFAULT_BASES), help="bases to test")
    parser.add_argument("--width", type=int, default=None, help="optional fixed digit width for every base")
    parser.add_argument("--expected-next", type=int, default=DEFAULT_EXPECTED_NEXT, help="expected base-10 next value for the operator check")
    parser.add_argument("--random-samples", type=int, default=DEFAULT_RANDOM_SAMPLES, help="null-model sample count per base")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="maximum chain steps before truncation")
    parser.add_argument("--rng-seed", type=int, default=DEFAULT_RNG_SEED, help="seed for the random null model")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="directory for csv, plots, and summary")
    parser.add_argument("--valuation-base", type=int, default=DEFAULT_VALUATION_BASE, help="base whose prime factors are stripped before the auxiliary valuation test")
    parser.add_argument("--log-base", type=float, default=DEFAULT_LOG_BASE, help="log base for the auxiliary archimedean comparison")
    parser.add_argument("--log-depth", type=int, default=DEFAULT_LOG_DEPTH, help="number of iterated log levels in the auxiliary comparison")
    parser.add_argument(
        "--variant-value",
        type=int,
        default=DEFAULT_VARIANT_VALUE,
        help="comparison value for the leading-digit variant test",
    )
    parser.add_argument("--show", action="store_true", help="show the saved plots interactively after generation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_summaries: list[dict[str, object]] = []
    null_rows_by_base: dict[int, list[dict[str, object]]] = {}
    null_label_counts: dict[int, Counter] = {}
    rng = np.random.default_rng(args.rng_seed)

    for base in args.bases:
        width = _width_for_base(args.seed, base, args.width)
        seed_digits = int_to_digits(args.seed, base, width=width)
        seed_result = iterate_chain(seed_digits, base=base, max_steps=args.max_steps)
        seed_label = classify_cycle(seed_result.cycle_states)
        seed_summary = _build_seed_summary(seed_result, label=seed_label, expected_next=args.expected_next)
        seed_summaries.append(seed_summary)

        base_prefix = output_dir / f"base_{base}"
        save_chain_csv(seed_result, base_prefix.with_name(base_prefix.name + "_seed_chain.csv"))
        plot_chain_heatmap(seed_result, base_prefix.with_name(base_prefix.name + "_seed_chain.png"))

        null_rows, label_counts = run_null_model(
            base=base,
            width=width,
            random_samples=args.random_samples,
            max_steps=args.max_steps,
            rng=rng,
        )
        null_rows_by_base[base] = null_rows
        null_label_counts[base] = label_counts
        save_null_rows(null_rows, base_prefix.with_name(base_prefix.name + "_null_model.csv"))

    plot_terminal_rates(seed_summaries, null_label_counts, args.random_samples, output_dir / "null_terminal_rates.png")
    plot_steps_histogram(seed_summaries, null_rows_by_base, output_dir / "null_steps_histogram.png")
    summary = build_summary(
        seed_summaries=seed_summaries,
        null_rows_by_base=null_rows_by_base,
        null_label_counts=null_label_counts,
        expected_next=args.expected_next,
        output_dir=output_dir,
    )

    valuation_values = [args.seed, args.expected_next, args.variant_value]
    valuation_rows = build_valuation_rows(
        values=valuation_values,
        valuation_base=args.valuation_base,
        log_base=args.log_base,
        log_depth=args.log_depth,
    )
    valuation_pairwise_rows = build_pairwise_delta_rows(
        named_values=[("seed", args.seed), ("expected", args.expected_next), ("variant", args.variant_value)],
        valuation_base=args.valuation_base,
        log_base=args.log_base,
        log_depth=args.log_depth,
    )
    save_dict_rows(valuation_rows, output_dir / "valuation_decomposition.csv")
    save_dict_rows(valuation_pairwise_rows, output_dir / "valuation_pairwise_deltas.csv")
    plot_pairwise_deltas(valuation_pairwise_rows, output_dir / "valuation_pairwise_deltas.png")
    valuation_summary = build_valuation_summary(
        valuation_rows=valuation_rows,
        pairwise_rows=valuation_pairwise_rows,
        valuation_base=args.valuation_base,
        log_base=args.log_base,
        output_dir=output_dir,
    )

    print(summary, end="")
    print()
    print(valuation_summary, end="")
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
