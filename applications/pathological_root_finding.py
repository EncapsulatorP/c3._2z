from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np

_MPLCONFIGDIR = Path("/tmp/mplconfig")
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_TARGET_L2 = 5.0
DEFAULT_MODULATION = 0.05
DEFAULT_FREQUENCY = 5.0
DEFAULT_MAX_ITERS = 25
DEFAULT_TOL = 1e-12
DEFAULT_OUTPUT_DIR = Path("applications/outputs/pathological_root_finding")


def safe_exp(value: float) -> float:
    if value > 700.0:
        return math.inf
    return math.exp(value)


def x_from_l2(l2_value: float) -> float:
    return safe_exp(safe_exp(l2_value))


def l2_from_x(x_value: float) -> float:
    if x_value <= 1.0:
        raise ValueError("x must be > 1 for L2 = log(log(x))")
    return math.log(math.log(x_value))


def transformed_equation(l2_value: float, target_l2: float, modulation: float, frequency: float) -> float:
    return l2_value + modulation * math.sin(frequency * l2_value) - target_l2


def transformed_equation_prime(l2_value: float, modulation: float, frequency: float) -> float:
    return 1.0 + modulation * frequency * math.cos(frequency * l2_value)


def raw_equation(x_value: float, target_l2: float, modulation: float, frequency: float) -> float:
    l2_value = l2_from_x(x_value)
    return transformed_equation(l2_value, target_l2=target_l2, modulation=modulation, frequency=frequency)


def raw_equation_prime(x_value: float, modulation: float, frequency: float) -> float:
    l2_value = l2_from_x(x_value)
    numerator = transformed_equation_prime(l2_value, modulation=modulation, frequency=frequency)
    return numerator / (x_value * math.log(x_value))


def run_newton_in_raw_space(
    x0: float,
    target_l2: float,
    modulation: float,
    frequency: float,
    max_iters: int,
    tol: float,
) -> list[dict[str, float | int]]:
    history: list[dict[str, float | int]] = []
    x_value = float(x0)

    for iteration in range(max_iters):
        fx = raw_equation(x_value, target_l2=target_l2, modulation=modulation, frequency=frequency)
        derivative = raw_equation_prime(x_value, modulation=modulation, frequency=frequency)
        step = fx / derivative
        next_x = x_value - step

        if not math.isfinite(next_x) or next_x <= 1.0:
            next_x = max(1.0000001, x_value - math.copysign(abs(x_value) * 0.5, step))

        history.append(
            {
                "iteration": iteration,
                "x": x_value,
                "l2": l2_from_x(x_value),
                "f": fx,
                "derivative": derivative,
                "step": step,
                "abs_f": abs(fx),
            }
        )

        if abs(fx) < tol:
            break
        x_value = next_x

    return history


def run_newton_in_l2_space(
    l2_0: float,
    target_l2: float,
    modulation: float,
    frequency: float,
    max_iters: int,
    tol: float,
) -> list[dict[str, float | int]]:
    history: list[dict[str, float | int]] = []
    l2_value = float(l2_0)

    for iteration in range(max_iters):
        fx = transformed_equation(l2_value, target_l2=target_l2, modulation=modulation, frequency=frequency)
        derivative = transformed_equation_prime(l2_value, modulation=modulation, frequency=frequency)
        step = fx / derivative
        history.append(
            {
                "iteration": iteration,
                "x": x_from_l2(l2_value),
                "l2": l2_value,
                "f": fx,
                "derivative": derivative,
                "step": step,
                "abs_f": abs(fx),
            }
        )
        if abs(fx) < tol:
            break
        l2_value = l2_value - step

    return history


def save_history(rows: list[dict[str, float | int]], output_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_histories(raw_rows: list[dict[str, float | int]], l2_rows: list[dict[str, float | int]], output_dir: Path) -> None:
    raw_iters = np.array([int(row["iteration"]) for row in raw_rows], dtype=int)
    raw_residuals = np.array([float(row["abs_f"]) for row in raw_rows], dtype=float)
    l2_iters = np.array([int(row["iteration"]) for row in l2_rows], dtype=int)
    l2_residuals = np.array([float(row["abs_f"]) for row in l2_rows], dtype=float)

    figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=False)
    axes[0].plot(raw_iters, raw_residuals, marker="o", label="raw x-space Newton", color="tab:blue")
    axes[0].plot(l2_iters, l2_residuals, marker="o", label="L2-space Newton", color="tab:green")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("|f|")
    axes[0].set_title("Residual decay")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    raw_l2 = np.array([float(row["l2"]) for row in raw_rows], dtype=float)
    l2_l2 = np.array([float(row["l2"]) for row in l2_rows], dtype=float)
    axes[1].plot(raw_iters, raw_l2, marker="o", label="raw x-space Newton", color="tab:blue")
    axes[1].plot(l2_iters, l2_l2, marker="o", label="L2-space Newton", color="tab:green")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("L2 coordinate")
    axes[1].set_title("Coordinate trajectory")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_dir / "comparison.png", dpi=180)
    plt.close(figure)


def build_summary(raw_rows: list[dict[str, float | int]], l2_rows: list[dict[str, float | int]], output_dir: Path) -> str:
    raw_last = raw_rows[-1]
    l2_last = l2_rows[-1]
    lines = []
    lines.append("Pathological Root-Finding Application")
    lines.append("")
    lines.append("Setup")
    lines.append("- Solve `L2 + 0.05 sin(5 L2) = 5`, where `L2 = log(log(x))`.")
    lines.append("- Compare Newton directly in `x` against Newton in `L2` space.")
    lines.append("")
    lines.append("Results")
    lines.append(
        f"- Raw x-space Newton: {len(raw_rows)} iterations, final |f|={float(raw_last['abs_f']):.6e}, final x={float(raw_last['x']):.6e}, final L2={float(raw_last['l2']):.6e}"
    )
    lines.append(
        f"- L2-space Newton: {len(l2_rows)} iterations, final |f|={float(l2_last['abs_f']):.6e}, final x={float(l2_last['x']):.6e}, final L2={float(l2_last['l2']):.6e}"
    )
    lines.append("")
    lines.append("Interpretation")
    lines.append(
        "- The raw solver pays the `1 / (x log x)` conditioning penalty, so steps are numerically awkward in raw coordinates."
    )
    lines.append(
        "- The L2-space solver works on the natural comparability class of the equation and therefore converges in a much better-conditioned coordinate system."
    )
    summary = "\n".join(lines) + "\n"
    (output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Application example: tower-preconditioned root finding.")
    parser.add_argument("--target-l2", type=float, default=DEFAULT_TARGET_L2)
    parser.add_argument("--modulation", type=float, default=DEFAULT_MODULATION)
    parser.add_argument("--frequency", type=float, default=DEFAULT_FREQUENCY)
    parser.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS)
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    l2_root_guess = args.target_l2
    raw_x0 = x_from_l2(args.target_l2 - 0.75)
    l2_0 = args.target_l2 - 0.75

    raw_rows = run_newton_in_raw_space(
        x0=raw_x0,
        target_l2=args.target_l2,
        modulation=args.modulation,
        frequency=args.frequency,
        max_iters=args.max_iters,
        tol=args.tol,
    )
    l2_rows = run_newton_in_l2_space(
        l2_0=l2_0,
        target_l2=args.target_l2,
        modulation=args.modulation,
        frequency=args.frequency,
        max_iters=args.max_iters,
        tol=args.tol,
    )

    save_history(raw_rows, output_dir / "raw_newton.csv")
    save_history(l2_rows, output_dir / "l2_newton.csv")
    plot_histories(raw_rows, l2_rows, output_dir)
    summary = build_summary(raw_rows, l2_rows, output_dir)
    print(summary, end="")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
