from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

_MPLCONFIGDIR = Path("/tmp/mplconfig")
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_EPS = 1e-6
DEFAULT_Z_MIN = 0.1
DEFAULT_Z_MAX = 2.0
DEFAULT_Z_POINTS = 500
DEFAULT_C_MIN = 1.01
DEFAULT_C_MAX = 2.5
DEFAULT_C_POINTS = 500
DEFAULT_DELTA_MAX = 0.5
DEFAULT_DELTA_POINTS = 200
MAX_FLOAT_LOG = math.log(sys.float_info.max)
EXP_UNDERFLOW_CUTOFF = -745.0


def phi(z: float) -> float:
    return 2.0 * z


def _validate_base(base: float) -> float:
    base = float(base)
    if base <= 1.0:
        raise ValueError(f"base must be > 1.0, got {base}")
    return base


def _pow_or_inf(base: float, exponent: float) -> float:
    if math.isinf(exponent):
        return math.inf
    log_value = exponent * math.log(base)
    if log_value > MAX_FLOAT_LOG:
        return math.inf
    return math.exp(log_value)


def _log1p_exp(shift: float) -> float:
    if shift == -math.inf or shift < EXP_UNDERFLOW_CUTOFF:
        return 0.0
    return math.log1p(math.exp(shift))


def _exact_log_chain_values(bases: Sequence[float], u: float) -> list[float]:
    tail_values = [u]
    for base in reversed(bases[1:]):
        tail_values.append(_pow_or_inf(base, tail_values[-1]))
    return list(reversed(tail_values))


def residue_from_bases(bases: Sequence[float], z: float, eps: float = DEFAULT_EPS) -> float:
    if eps < 0.0:
        raise ValueError(f"eps must be >= 0, got {eps}")

    bases = tuple(_validate_base(base) for base in bases)
    u = phi(float(z))
    if len(bases) < 1:
        raise ValueError("at least one base is required")

    exact_values = _exact_log_chain_values(bases, u)
    if not exact_values:
        return 0.0

    if eps == 0.0:
        delta = 0.0
    else:
        ln_top = exact_values[0] * math.log(bases[0])
        delta = _log1p_exp(math.log(eps) - ln_top) / math.log(bases[0])

    for index in range(1, len(bases)):
        exact_value = exact_values[index - 1]
        if math.isinf(exact_value):
            delta = 0.0
            continue
        delta = math.log1p(delta / exact_value) / math.log(bases[index])
    return delta


def tower3(c: float, u: float) -> float:
    c = _validate_base(c)
    return c ** (c ** (c ** u))


def tower3_varied(c1: float, c2: float, c3: float, u: float) -> float:
    c1 = _validate_base(c1)
    c2 = _validate_base(c2)
    c3 = _validate_base(c3)
    return c1 ** (c2 ** (c3 ** u))


def residue(c: float, z: float, eps: float = DEFAULT_EPS) -> float:
    return residue_from_bases((c, c, c), z, eps=eps)


def residue_varied(c1: float, c2: float, c3: float, z: float, eps: float = DEFAULT_EPS) -> float:
    return residue_from_bases((c1, c2, c3), z, eps=eps)


def residue_direct(c: float, z: float, eps: float = DEFAULT_EPS) -> float:
    c = _validate_base(c)
    u = phi(z)
    try:
        x_value = tower3(c, u) + eps
        l1 = math.log(x_value) / math.log(c)
        l2 = math.log(l1) / math.log(c)
        l3 = math.log(l2) / math.log(c)
    except (OverflowError, ValueError):
        return math.nan
    return l3 - u


def _array_from_iter(values: Iterable[float]) -> np.ndarray:
    return np.fromiter(values, dtype=float)


def _save_csv(path: Path, columns: Sequence[str], matrix: np.ndarray) -> None:
    header = ",".join(columns)
    np.savetxt(path, matrix, delimiter=",", header=header, comments="")


def _plot_residue_profile(z_values: np.ndarray, c_values: Sequence[float], eps: float, output_dir: Path) -> np.ndarray:
    plt.figure(figsize=(9, 5.5))
    result = np.empty((z_values.size, len(c_values) + 1), dtype=float)
    result[:, 0] = z_values

    for offset, c in enumerate(c_values, start=1):
        residues = _array_from_iter(residue(c, z, eps=eps) for z in z_values)
        result[:, offset] = residues
        plt.plot(z_values, residues, label=f"c={c:.4g}")

    plt.axhline(0.0, color="black", linewidth=0.7)
    plt.xlabel("z")
    plt.ylabel("omega")
    plt.title(r"Test 1: Residue profile $\omega$ vs $z$")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "test1_residue_vs_z.png", dpi=180)
    plt.close()

    headers = ["z"] + [f"omega_c_{c:.6f}" for c in c_values]
    _save_csv(output_dir / "test1_residue_vs_z.csv", headers, result)
    return result


def _plot_base_sweep(c_range: np.ndarray, z0: float, eps: float, output_dir: Path) -> np.ndarray:
    residues = _array_from_iter(residue(c, z0, eps=eps) for c in c_range)
    abs_residues = np.abs(residues)
    c_star_index = int(np.nanargmin(abs_residues))
    c_star = c_range[c_star_index]
    omega_star = residues[c_star_index]

    plt.figure(figsize=(9, 5.5))
    plt.plot(c_range, residues, color="tab:blue")
    plt.scatter([c_star], [omega_star], color="tab:red", s=28, label=f"min |omega| at c={c_star:.4f}")
    plt.axhline(0.0, color="black", linewidth=0.7)
    plt.xlabel("c")
    plt.ylabel("omega")
    plt.title(r"Test 2: Residue $\omega$ vs base $c$ at fixed $z$")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "test2_residue_vs_c.png", dpi=180)
    plt.close()

    matrix = np.column_stack([c_range, residues])
    _save_csv(output_dir / "test2_residue_vs_c.csv", ["c", "omega"], matrix)
    return matrix


def _plot_symmetry_breaking(base: float, z0: float, deltas: np.ndarray, eps: float, output_dir: Path) -> np.ndarray:
    symmetric_residue = residue(base, z0, eps=eps)
    broken_residues = _array_from_iter(
        residue_varied(base, base, base + delta, z0, eps=eps) for delta in deltas
    )

    plt.figure(figsize=(9, 5.5))
    plt.plot(deltas, broken_residues, color="tab:green")
    plt.axhline(
        symmetric_residue,
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
        label=f"symmetric residue={symmetric_residue:.3e}",
    )
    plt.xlabel("delta")
    plt.ylabel("omega")
    plt.title(r"Test 3: Residue under symmetry breaking $c_3 \rightarrow c_3 + \delta$")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "test3_symmetry_breaking.png", dpi=180)
    plt.close()

    matrix = np.column_stack([deltas, broken_residues])
    _save_csv(output_dir / "test3_symmetry_breaking.csv", ["delta", "omega"], matrix)
    return matrix


def _zero_crossings(x_values: np.ndarray, y_values: np.ndarray) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for left, right, y_left, y_right in zip(x_values[:-1], x_values[1:], y_values[:-1], y_values[1:]):
        if np.isnan(y_left) or np.isnan(y_right):
            continue
        if y_left == 0.0:
            intervals.append((left, left))
        elif y_left * y_right < 0.0:
            intervals.append((left, right))
    if y_values[-1] == 0.0:
        intervals.append((x_values[-1], x_values[-1]))
    return intervals


def _sign_flips(values: np.ndarray, atol: float = 1e-15) -> int:
    filtered = values[np.abs(values) > atol]
    if filtered.size < 2:
        return 0
    signs = np.sign(filtered)
    return int(np.sum(signs[1:] != signs[:-1]))


def _validate_stable_vs_direct(output_dir: Path, eps: float) -> tuple[int, float]:
    c_values = np.linspace(1.01, 1.2, 10)
    z_values = np.linspace(0.1, 0.8, 10)
    comparisons: list[tuple[float, float, float, float, float]] = []

    for c in c_values:
        for z in z_values:
            direct = residue_direct(float(c), float(z), eps=eps)
            if math.isnan(direct):
                continue
            stable = residue(float(c), float(z), eps=eps)
            comparisons.append((float(c), float(z), direct, stable, abs(direct - stable)))

    if not comparisons:
        return 0, math.nan

    matrix = np.array(comparisons, dtype=float)
    _save_csv(
        output_dir / "validation_direct_vs_stable.csv",
        ["c", "z", "direct", "stable", "abs_error"],
        matrix,
    )
    return matrix.shape[0], float(np.max(matrix[:, -1]))


def _build_summary(
    profile_data: np.ndarray,
    profile_c_values: Sequence[float],
    base_data: np.ndarray,
    bifurcation_data: np.ndarray,
    symmetric_residue: float,
    validation_count: int,
    validation_max_error: float,
    output_dir: Path,
) -> str:
    lines = []
    lines.append("Omega perturbation stability summary")
    lines.append("")
    lines.append("Test 1: residue profile vs z")
    for column, c in enumerate(profile_c_values, start=1):
        residues = profile_data[:, column]
        lines.append(
            f"- c={c:.6f}: min={np.min(residues):.6e}, max={np.max(residues):.6e}, max|omega|={np.max(np.abs(residues)):.6e}"
        )

    c_range = base_data[:, 0]
    residues = base_data[:, 1]
    abs_residues = np.abs(residues)
    c_star_index = int(np.nanargmin(abs_residues))
    c_star = c_range[c_star_index]
    omega_star = residues[c_star_index]
    zero_crossings = _zero_crossings(c_range, residues)
    abs_residue_flips = _sign_flips(np.diff(abs_residues))
    min_on_boundary = c_star_index in (0, len(c_range) - 1)

    lines.append("")
    lines.append("Test 2: residue stability vs base c")
    lines.append(f"- c* candidate (min |omega|): c={c_star:.6f}, omega={omega_star:.6e}")
    lines.append(f"- sweep range: min omega={np.min(residues):.6e}, max omega={np.max(residues):.6e}")
    lines.append(f"- min sits on sweep boundary: {'yes' if min_on_boundary else 'no'}")
    lines.append(f"- sign flips in d|omega|/dc: {abs_residue_flips}")
    lines.append(f"- zero crossing intervals found: {len(zero_crossings)}")
    for left, right in zero_crossings[:5]:
        if left == right:
            lines.append(f"  exact zero at c={left:.6f}")
        else:
            lines.append(f"  sign change between c={left:.6f} and c={right:.6f}")
    if len(zero_crossings) > 5:
        lines.append(f"  ... {len(zero_crossings) - 5} more intervals")

    broken_residues = bifurcation_data[:, 1]
    max_jump = float(np.max(np.abs(np.diff(broken_residues)))) if broken_residues.size > 1 else 0.0
    lines.append("")
    lines.append("Test 3: symmetry breaking")
    lines.append(f"- symmetric residue: {symmetric_residue:.6e}")
    lines.append(
        f"- broken sweep: min omega={np.min(broken_residues):.6e}, max omega={np.max(broken_residues):.6e}"
    )
    lines.append(f"- max adjacent jump: {max_jump:.6e}")

    lines.append("")
    lines.append("Validation")
    if validation_count == 0:
        lines.append("- no direct-evaluation points stayed finite in the validation window")
    else:
        lines.append(
            f"- direct vs stable comparisons: {validation_count} points, max abs error={validation_max_error:.6e}"
        )

    summary = "\n".join(lines) + "\n"
    (output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run omega perturbation stability sweeps.")
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS, help="absolute perturbation epsilon")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/omega_stability"), help="directory for plots and csv")
    parser.add_argument("--z-min", type=float, default=DEFAULT_Z_MIN, help="minimum z for Test 1")
    parser.add_argument("--z-max", type=float, default=DEFAULT_Z_MAX, help="maximum z for Test 1")
    parser.add_argument("--z-points", type=int, default=DEFAULT_Z_POINTS, help="number of z samples for Test 1")
    parser.add_argument("--z0", type=float, default=1.0, help="fixed z value for Tests 2 and 3")
    parser.add_argument("--c-min", type=float, default=DEFAULT_C_MIN, help="minimum c for Test 2")
    parser.add_argument("--c-max", type=float, default=DEFAULT_C_MAX, help="maximum c for Test 2")
    parser.add_argument("--c-points", type=int, default=DEFAULT_C_POINTS, help="number of c samples for Test 2")
    parser.add_argument(
        "--profile-c-values",
        type=float,
        nargs="+",
        default=[1.2, 1.5, math.e, 2.0],
        help="bases to overlay in Test 1",
    )
    parser.add_argument("--symmetry-base", type=float, default=1.5, help="base c for Test 3")
    parser.add_argument("--delta-max", type=float, default=DEFAULT_DELTA_MAX, help="maximum delta for Test 3")
    parser.add_argument("--delta-points", type=int, default=DEFAULT_DELTA_POINTS, help="number of delta samples for Test 3")
    parser.add_argument("--show", action="store_true", help="open plots interactively after saving them")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    z_values = np.linspace(args.z_min, args.z_max, args.z_points)
    c_range = np.linspace(args.c_min, args.c_max, args.c_points)
    deltas = np.linspace(0.0, args.delta_max, args.delta_points)

    profile_data = _plot_residue_profile(z_values, args.profile_c_values, args.eps, output_dir)
    base_data = _plot_base_sweep(c_range, args.z0, args.eps, output_dir)
    bifurcation_data = _plot_symmetry_breaking(args.symmetry_base, args.z0, deltas, args.eps, output_dir)

    symmetric_residue = residue(args.symmetry_base, args.z0, eps=args.eps)
    validation_count, validation_max_error = _validate_stable_vs_direct(output_dir, eps=args.eps)
    summary = _build_summary(
        profile_data=profile_data,
        profile_c_values=args.profile_c_values,
        base_data=base_data,
        bifurcation_data=bifurcation_data,
        symmetric_residue=symmetric_residue,
        validation_count=validation_count,
        validation_max_error=validation_max_error,
        output_dir=output_dir,
    )

    print(summary, end="")
    print(f"Saved outputs to {output_dir}")

    if args.show:
        for image_name in (
            "test1_residue_vs_z.png",
            "test2_residue_vs_c.png",
            "test3_symmetry_breaking.png",
        ):
            image = plt.imread(output_dir / image_name)
            plt.figure(figsize=(10, 6))
            plt.imshow(image)
            plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
