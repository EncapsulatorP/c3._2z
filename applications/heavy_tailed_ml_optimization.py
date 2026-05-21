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


DEFAULT_TARGET_L2 = 3.0
DEFAULT_RAW_LR = 5e11
DEFAULT_L2_LR = 0.6
DEFAULT_STEPS = 40
DEFAULT_OUTPUT_DIR = Path("applications/outputs/heavy_tailed_ml_optimization")


def safe_exp(value: float) -> float:
    if value > 700.0:
        return math.inf
    return math.exp(value)


def x_from_l2(l2_value: float) -> float:
    return safe_exp(safe_exp(l2_value))


def l2_from_x(x_value: float) -> float:
    if x_value <= 1.0:
        raise ValueError("x must be > 1")
    return math.log(math.log(x_value))


def loss_from_x(x_value: float, target_l2: float) -> float:
    l2_value = l2_from_x(x_value)
    return 0.5 * (l2_value - target_l2) ** 2


def gradient_from_x(x_value: float, target_l2: float) -> float:
    l2_value = l2_from_x(x_value)
    return (l2_value - target_l2) / (x_value * math.log(x_value))


def loss_from_l2(l2_value: float, target_l2: float) -> float:
    return 0.5 * (l2_value - target_l2) ** 2


def gradient_from_l2(l2_value: float, target_l2: float) -> float:
    return l2_value - target_l2


def run_raw_gradient_descent(x0: float, target_l2: float, learning_rate: float, steps: int) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    x_value = float(x0)
    for step in range(steps):
        loss = loss_from_x(x_value, target_l2=target_l2)
        gradient = gradient_from_x(x_value, target_l2=target_l2)
        rows.append(
            {
                "step": step,
                "x": x_value,
                "l2": l2_from_x(x_value),
                "loss": loss,
                "gradient": gradient,
            }
        )
        next_x = x_value - learning_rate * gradient
        x_value = max(1.0000001, next_x)
    return rows


def run_l2_gradient_descent(l2_0: float, target_l2: float, learning_rate: float, steps: int) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    l2_value = float(l2_0)
    for step in range(steps):
        loss = loss_from_l2(l2_value, target_l2=target_l2)
        gradient = gradient_from_l2(l2_value, target_l2=target_l2)
        rows.append(
            {
                "step": step,
                "x": x_from_l2(l2_value),
                "l2": l2_value,
                "loss": loss,
                "gradient": gradient,
            }
        )
        l2_value = l2_value - learning_rate * gradient
    return rows


def save_rows(rows: list[dict[str, float | int]], output_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_runs(raw_rows: list[dict[str, float | int]], l2_rows: list[dict[str, float | int]], output_dir: Path) -> None:
    raw_steps = np.array([int(row["step"]) for row in raw_rows], dtype=int)
    raw_losses = np.array([float(row["loss"]) for row in raw_rows], dtype=float)
    l2_steps = np.array([int(row["step"]) for row in l2_rows], dtype=int)
    l2_losses = np.array([float(row["loss"]) for row in l2_rows], dtype=float)

    figure, axes = plt.subplots(2, 1, figsize=(9, 8))
    axes[0].plot(raw_steps, raw_losses, label="raw gradient descent", color="tab:blue")
    axes[0].plot(l2_steps, l2_losses, label="L2-coordinate descent", color="tab:green")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss decay on a synthetic heavy-tailed plateau")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    raw_l2 = np.array([float(row["l2"]) for row in raw_rows], dtype=float)
    l2_l2 = np.array([float(row["l2"]) for row in l2_rows], dtype=float)
    axes[1].plot(raw_steps, raw_l2, label="raw gradient descent", color="tab:blue")
    axes[1].plot(l2_steps, l2_l2, label="L2-coordinate descent", color="tab:green")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("L2 coordinate")
    axes[1].set_title("Progress in the natural tower coordinate")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_dir / "optimization.png", dpi=180)
    plt.close(figure)


def build_summary(raw_rows: list[dict[str, float | int]], l2_rows: list[dict[str, float | int]], output_dir: Path) -> str:
    raw_last = raw_rows[-1]
    l2_last = l2_rows[-1]
    lines = []
    lines.append("Heavy-Tailed Optimization Application")
    lines.append("")
    lines.append("Setup")
    lines.append("- Optimize `0.5 * (log(log(x)) - 3)^2` from a large initial scale.")
    lines.append("- Compare raw gradient descent in `x` against gradient descent in `L2 = log(log(x))`.")
    lines.append("")
    lines.append("Results")
    lines.append(
        f"- Raw gradient descent: {len(raw_rows)} steps, final loss={float(raw_last['loss']):.6e}, final x={float(raw_last['x']):.6e}, final L2={float(raw_last['l2']):.6e}"
    )
    lines.append(
        f"- L2-coordinate descent: {len(l2_rows)} steps, final loss={float(l2_last['loss']):.6e}, final x={float(l2_last['x']):.6e}, final L2={float(l2_last['l2']):.6e}"
    )
    lines.append("")
    lines.append("Interpretation")
    lines.append(
        "- In raw space the gradient is suppressed by `1 / (x log x)`, so the optimizer crawls across the plateau."
    )
    lines.append(
        "- In L2 space the optimizer sees the true quadratic geometry directly, so it descends the heavy-tailed landscape much more efficiently."
    )
    summary = "\n".join(lines) + "\n"
    (output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Application example: tower-coordinate optimization.")
    parser.add_argument("--target-l2", type=float, default=DEFAULT_TARGET_L2)
    parser.add_argument("--raw-lr", type=float, default=DEFAULT_RAW_LR)
    parser.add_argument("--l2-lr", type=float, default=DEFAULT_L2_LR)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    x0 = x_from_l2(args.target_l2 + 2.0)
    l2_0 = args.target_l2 + 2.0

    raw_rows = run_raw_gradient_descent(
        x0=x0,
        target_l2=args.target_l2,
        learning_rate=args.raw_lr,
        steps=args.steps,
    )
    l2_rows = run_l2_gradient_descent(
        l2_0=l2_0,
        target_l2=args.target_l2,
        learning_rate=args.l2_lr,
        steps=args.steps,
    )

    save_rows(raw_rows, output_dir / "raw_gd.csv")
    save_rows(l2_rows, output_dir / "l2_gd.csv")
    plot_runs(raw_rows, l2_rows, output_dir)
    summary = build_summary(raw_rows, l2_rows, output_dir)
    print(summary, end="")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
