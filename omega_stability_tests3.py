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


DEFAULT_BASES = (2.0, 2.01, math.sqrt(2.0), 1.5, math.e, math.pi)
DEFAULT_TARGET = 8.0
DEFAULT_TARGET_DEPTH = 3.0
DEFAULT_X_MIN = 4.0
DEFAULT_X_MAX = 12.0
DEFAULT_X_POINTS = 600
DEFAULT_BASE_SWEEP_MIN = 1.1
DEFAULT_BASE_SWEEP_MAX = 4.0
DEFAULT_BASE_SWEEP_POINTS = 800
DEFAULT_RECAMAN_TERMS = 64
DEFAULT_RANDOM_TRIALS = 4000
DEFAULT_WINDOW = 4
DEFAULT_RNG_SEED = 12345
DEFAULT_OUTPUT_DIR = Path("outputs/omega_stability3")


def _validate_base(base: float) -> float:
    base = float(base)
    if base <= 1.0:
        raise ValueError(f"base must be > 1.0, got {base}")
    return base


def log_base(x_value: float, base: float) -> float:
    base = _validate_base(base)
    x_value = float(x_value)
    if x_value <= 0.0:
        return math.nan
    return math.log(x_value) / math.log(base)


def depth_residual(base: float, x_value: float, target_depth: float) -> float:
    projected = log_base(x_value, base)
    if math.isnan(projected):
        return math.nan
    return projected - float(target_depth)


def integer_alignment_residual(base: float, x_value: float) -> float:
    projected = log_base(x_value, base)
    if math.isnan(projected):
        return math.nan
    return abs(projected - round(projected))


def compression_factor(base: float, x_value: float) -> float:
    base = _validate_base(base)
    x_value = float(x_value)
    if x_value <= 0.0:
        return math.nan
    return 1.0 / (x_value * math.log(base))


def _format_base(base: float) -> str:
    if math.isclose(base, math.e, rel_tol=0.0, abs_tol=1e-12):
        return "e"
    if math.isclose(base, math.pi, rel_tol=0.0, abs_tol=1e-12):
        return "pi"
    if math.isclose(base, math.sqrt(2.0), rel_tol=0.0, abs_tol=1e-12):
        return "sqrt2"
    return f"{base:.4g}"


def _csv_name(prefix: str, base: float) -> str:
    label = _format_base(base).replace(".", "_")
    return f"{prefix}_{label}"


def _save_csv(path: Path, headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(headers))
        writer.writerows(rows)


def recaman_sequence(n_terms: int) -> list[int]:
    if n_terms < 1:
        raise ValueError(f"n_terms must be >= 1, got {n_terms}")

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
    if n_terms < 1:
        raise ValueError(f"n_terms must be >= 1, got {n_terms}")

    values = [0]
    for step in range(1, n_terms):
        direction = -1 if int(rng.integers(0, 2)) == 0 else 1
        candidate = values[-1] + direction * step
        if candidate < 0:
            candidate = values[-1] + step
        values.append(candidate)
    return values


def first_hit_index(values: Sequence[int], target: int) -> int | None:
    for index, value in enumerate(values):
        if value == target:
            return index
    return None


def absolute_residual_window(
    values: Sequence[int],
    center_index: int,
    base: float,
    target_depth: float,
    window: int,
) -> np.ndarray:
    offsets = range(-window, window + 1)
    profile = []
    for offset in offsets:
        index = center_index + offset
        if 0 <= index < len(values):
            profile.append(abs(depth_residual(base, values[index], target_depth)))
        else:
            profile.append(math.nan)
    return np.array(profile, dtype=float)


def _plot_local_residuals(
    x_values: np.ndarray,
    bases: Sequence[float],
    target_depth: float,
    target_x: float,
    output_dir: Path,
) -> np.ndarray:
    plt.figure(figsize=(9.5, 5.8))
    matrix = np.empty((x_values.size, len(bases) + 1), dtype=float)
    matrix[:, 0] = x_values

    for column, base in enumerate(bases, start=1):
        residuals = np.array([depth_residual(base, x_value, target_depth) for x_value in x_values], dtype=float)
        matrix[:, column] = residuals
        plt.plot(x_values, residuals, label=f"c={_format_base(base)}")

    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.axvline(target_x, color="tab:red", linewidth=0.8, linestyle="--", label=f"x={target_x:g}")
    plt.xlabel("x")
    plt.ylabel(r"$\omega_3(c, x) = \log_c(x) - 3$")
    plt.title("Test 1: Depth-3 residual around the dyadic target")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "test1_depth3_residual_vs_x.png", dpi=180)
    plt.close()

    headers = ["x"] + [_csv_name("omega3", base) for base in bases]
    _save_csv(output_dir / "test1_depth3_residual_vs_x.csv", headers, matrix.tolist())
    return matrix


def _plot_base_sweep(
    base_values: np.ndarray,
    target_x: float,
    target_depth: float,
    highlighted_bases: Sequence[float],
    output_dir: Path,
) -> np.ndarray:
    omega_values = np.array([depth_residual(base, target_x, target_depth) for base in base_values], dtype=float)
    compression_values = np.array([compression_factor(base, target_x) for base in base_values], dtype=float)
    alignment_values = np.array([integer_alignment_residual(base, target_x) for base in base_values], dtype=float)

    figure, axes = plt.subplots(2, 1, figsize=(9.5, 8.2), sharex=True)
    axes[0].plot(base_values, omega_values, color="tab:blue")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel(r"$\omega_3(c, 8)$")
    axes[0].set_title("Test 2: Blind-spot strength and compression at x=8")
    axes[0].grid(alpha=0.25)

    axes[1].plot(base_values, compression_values, color="tab:green")
    axes[1].set_xlabel("base c")
    axes[1].set_ylabel(r"$1 / (8 \ln c)$")
    axes[1].grid(alpha=0.25)

    for base in highlighted_bases:
        if base_values[0] <= base <= base_values[-1]:
            omega_here = depth_residual(base, target_x, target_depth)
            compression_here = compression_factor(base, target_x)
            axes[0].scatter([base], [omega_here], s=28, color="tab:red")
            axes[1].scatter([base], [compression_here], s=28, color="tab:red")
            axes[0].annotate(_format_base(base), (base, omega_here), textcoords="offset points", xytext=(5, 5), fontsize=8)

    figure.tight_layout()
    figure.savefig(output_dir / "test2_blind_spot_vs_base.png", dpi=180)
    plt.close(figure)

    matrix = np.column_stack([base_values, omega_values, compression_values, alignment_values])
    _save_csv(
        output_dir / "test2_blind_spot_vs_base.csv",
        ["base", "omega3", "compression", "integer_alignment_residual"],
        matrix.tolist(),
    )
    return matrix


def _save_recaman_sequence(values: Sequence[int], bases: Sequence[float], target_depth: float, output_dir: Path) -> None:
    headers = ["index", "value"] + [_csv_name("omega3", base) for base in bases]
    rows = []
    for index, value in enumerate(values):
        row = [index, value]
        row.extend(depth_residual(base, value, target_depth) for base in bases)
        rows.append(row)
    _save_csv(output_dir / "test3_recaman_sequence.csv", headers, rows)


def _plot_hit_index_histogram(
    hit_indices: Sequence[int],
    recaman_hit_index: int,
    output_dir: Path,
) -> None:
    plt.figure(figsize=(9.5, 5.2))
    plt.hist(hit_indices, bins=min(20, max(4, len(set(hit_indices)))), color="tab:blue", alpha=0.8)
    plt.axvline(recaman_hit_index, color="tab:red", linewidth=1.3, label=f"Recaman hit index={recaman_hit_index}")
    plt.xlabel("first hit index of 8")
    plt.ylabel("count")
    plt.title("Test 3A: Random-walk first-hit distribution for the target value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "test3_hit_index_histogram.png", dpi=180)
    plt.close()


def _plot_aligned_profiles(
    offsets: np.ndarray,
    recaman_profiles: dict[float, np.ndarray],
    null_profiles: dict[float, list[np.ndarray]],
    output_dir: Path,
) -> None:
    bases = list(recaman_profiles.keys())
    columns = 2
    rows = math.ceil(len(bases) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(11, 3.6 * rows), sharex=True, sharey=False)
    axes_flat = np.atleast_1d(axes).ravel()

    for axis, base in zip(axes_flat, bases):
        recaman_profile = recaman_profiles[base]
        profile_list = null_profiles.get(base, [])
        if profile_list:
            stack = np.vstack(profile_list)
            mean_profile = np.nanmean(stack, axis=0)
            std_profile = np.nanstd(stack, axis=0)
            axis.plot(offsets, mean_profile, color="tab:blue", label="null mean")
            axis.fill_between(offsets, mean_profile - std_profile, mean_profile + std_profile, color="tab:blue", alpha=0.18)
        axis.plot(offsets, recaman_profile, color="tab:red", linewidth=1.6, label="Recaman")
        axis.axvline(0, color="black", linewidth=0.7, linestyle="--")
        axis.set_title(f"base { _format_base(base) }")
        axis.set_xlabel("offset from first hit of 8")
        axis.set_ylabel(r"$|\omega_3|$")
        axis.grid(alpha=0.22)
        axis.legend(fontsize=8)

    for axis in axes_flat[len(bases) :]:
        axis.axis("off")

    figure.suptitle("Test 3B: Aligned residual window around the first hit of 8", fontsize=12)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    figure.savefig(output_dir / "test3_aligned_hit_profiles.png", dpi=180)
    plt.close(figure)


def _same_or_lower_fraction(values: Sequence[float], probe: float) -> float:
    if not values:
        return math.nan
    return float(sum(value <= probe for value in values) / len(values))


def build_summary(
    bases: Sequence[float],
    target_x: float,
    target_depth: float,
    local_matrix: np.ndarray,
    base_sweep_matrix: np.ndarray,
    recaman_values: Sequence[int],
    recaman_hit_index: int,
    null_hit_indices: Sequence[int],
    recaman_scores: dict[float, float],
    null_scores: dict[float, list[float]],
    random_trials: int,
    output_dir: Path,
) -> str:
    lines = []
    lines.append("Framework v1.3 summary")
    lines.append("")
    lines.append("Object")
    lines.append("- This runner measures local depth-3 log alignment around x=8 and compares Recaman against a matched random-walk null model.")
    lines.append("- Base 2 is included as the blind-spot control; non-dyadic bases are the actual discriminators.")

    lines.append("")
    lines.append("Test 1: local residual around x=8")
    for column, base in enumerate(bases, start=1):
        residual_at_target = depth_residual(base, target_x, target_depth)
        lines.append(f"- base { _format_base(base) }: omega3(8)={residual_at_target:.6e}")

    base_values = base_sweep_matrix[:, 0]
    omega_values = base_sweep_matrix[:, 1]
    compression_values = base_sweep_matrix[:, 2]
    minimum_index = int(np.nanargmin(np.abs(omega_values)))
    lines.append("")
    lines.append("Test 2: base sweep at x=8")
    lines.append(
        f"- min |omega3(c, 8)| on the sweep occurs at c={base_values[minimum_index]:.6f} with omega3={omega_values[minimum_index]:.6e}"
    )
    lines.append(
        f"- compression at c=2 is {compression_factor(2.0, target_x):.6e}; compression falls to {compression_values[-1]:.6e} at the top of the sweep"
    )

    lines.append("")
    lines.append("Test 3: Recaman vs matched random walk")
    lines.append(f"- Recaman first hits {int(target_x)} at index {recaman_hit_index}")
    lines.append(f"- random-hit rate within the first {len(recaman_values)} terms: {len(null_hit_indices) / random_trials:.4f}")
    if null_hit_indices:
        hit_percentile = _same_or_lower_fraction(null_hit_indices, recaman_hit_index)
        lines.append(
            f"- among random walks that do hit {int(target_x)}, the fraction hitting by index {recaman_hit_index} is {hit_percentile:.4f}"
        )
        lines.append(f"- mean first-hit index under the null model: {np.mean(null_hit_indices):.4f}")
    else:
        lines.append("- none of the random walks hit the target value, so the hit-window comparison is empty")

    for base in bases:
        score = recaman_scores[base]
        null_base_scores = null_scores.get(base, [])
        if null_base_scores:
            percentile = _same_or_lower_fraction(null_base_scores, score)
            lines.append(
                f"- base { _format_base(base) }: Recaman mean |omega3| in the hit window={score:.6e}, null mean={np.mean(null_base_scores):.6e}, percentile={percentile:.4f}"
            )
        else:
            lines.append(f"- base { _format_base(base) }: no null hit-window scores available")

    summary = "\n".join(lines) + "\n"
    (output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blind-spot diagnostics for the x=8 dyadic alignment claim.")
    parser.add_argument("--bases", type=float, nargs="+", default=list(DEFAULT_BASES), help="bases to compare")
    parser.add_argument("--target-x", type=float, default=DEFAULT_TARGET, help="dyadic target value")
    parser.add_argument("--target-depth", type=float, default=DEFAULT_TARGET_DEPTH, help="depth used in omega3(c, x)=log_c(x)-depth")
    parser.add_argument("--x-min", type=float, default=DEFAULT_X_MIN, help="minimum x for the local residual sweep")
    parser.add_argument("--x-max", type=float, default=DEFAULT_X_MAX, help="maximum x for the local residual sweep")
    parser.add_argument("--x-points", type=int, default=DEFAULT_X_POINTS, help="sample count for the local residual sweep")
    parser.add_argument("--base-sweep-min", type=float, default=DEFAULT_BASE_SWEEP_MIN, help="minimum base for the blind-spot sweep")
    parser.add_argument("--base-sweep-max", type=float, default=DEFAULT_BASE_SWEEP_MAX, help="maximum base for the blind-spot sweep")
    parser.add_argument("--base-sweep-points", type=int, default=DEFAULT_BASE_SWEEP_POINTS, help="sample count for the blind-spot sweep")
    parser.add_argument("--recaman-terms", type=int, default=DEFAULT_RECAMAN_TERMS, help="number of Recaman terms to generate")
    parser.add_argument("--random-trials", type=int, default=DEFAULT_RANDOM_TRIALS, help="number of matched random walks")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="half-width of the aligned hit window")
    parser.add_argument("--rng-seed", type=int, default=DEFAULT_RNG_SEED, help="seed for the random null model")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="directory for plots and csv files")
    parser.add_argument("--show", action="store_true", help="show plots after saving them")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    bases = tuple(_validate_base(base) for base in args.bases)
    x_values = np.linspace(args.x_min, args.x_max, args.x_points)
    base_values = np.linspace(args.base_sweep_min, args.base_sweep_max, args.base_sweep_points)

    local_matrix = _plot_local_residuals(
        x_values=x_values,
        bases=bases,
        target_depth=args.target_depth,
        target_x=args.target_x,
        output_dir=output_dir,
    )
    base_sweep_matrix = _plot_base_sweep(
        base_values=base_values,
        target_x=args.target_x,
        target_depth=args.target_depth,
        highlighted_bases=bases,
        output_dir=output_dir,
    )

    recaman_values = recaman_sequence(args.recaman_terms)
    recaman_hit_index = first_hit_index(recaman_values, int(args.target_x))
    if recaman_hit_index is None:
        raise RuntimeError(f"Recaman did not hit target {args.target_x:g} within {args.recaman_terms} terms")

    _save_recaman_sequence(recaman_values, bases=bases, target_depth=args.target_depth, output_dir=output_dir)

    offsets = np.arange(-args.window, args.window + 1)
    recaman_profiles = {
        base: absolute_residual_window(recaman_values, recaman_hit_index, base, args.target_depth, args.window)
        for base in bases
    }
    recaman_scores = {base: float(np.nanmean(profile)) for base, profile in recaman_profiles.items()}

    rng = np.random.default_rng(args.rng_seed)
    null_hit_indices: list[int] = []
    null_profiles: dict[float, list[np.ndarray]] = {base: [] for base in bases}
    null_scores: dict[float, list[float]] = {base: [] for base in bases}

    for _ in range(args.random_trials):
        values = matched_random_walk(args.recaman_terms, rng)
        hit_index = first_hit_index(values, int(args.target_x))
        if hit_index is None:
            continue
        null_hit_indices.append(hit_index)
        for base in bases:
            profile = absolute_residual_window(values, hit_index, base, args.target_depth, args.window)
            null_profiles[base].append(profile)
            null_scores[base].append(float(np.nanmean(profile)))

    hit_rows = []
    for base in bases:
        base_scores = null_scores[base]
        hit_rows.append(
            [
                _format_base(base),
                len(null_hit_indices),
                args.random_trials,
                len(null_hit_indices) / args.random_trials,
                recaman_hit_index,
                recaman_scores[base],
                float(np.mean(base_scores)) if base_scores else math.nan,
                float(np.std(base_scores)) if base_scores else math.nan,
                _same_or_lower_fraction(base_scores, recaman_scores[base]) if base_scores else math.nan,
            ]
        )
    _save_csv(
        output_dir / "test3_hit_window_summary.csv",
        [
            "base",
            "null_hit_count",
            "random_trials",
            "null_hit_rate",
            "recaman_first_hit_index",
            "recaman_mean_abs_omega3",
            "null_mean_abs_omega3",
            "null_std_abs_omega3",
            "recaman_percentile",
        ],
        hit_rows,
    )

    if null_hit_indices:
        _plot_hit_index_histogram(null_hit_indices, recaman_hit_index, output_dir)
    _plot_aligned_profiles(offsets, recaman_profiles, null_profiles, output_dir)

    summary = build_summary(
        bases=bases,
        target_x=args.target_x,
        target_depth=args.target_depth,
        local_matrix=local_matrix,
        base_sweep_matrix=base_sweep_matrix,
        recaman_values=recaman_values,
        recaman_hit_index=recaman_hit_index,
        null_hit_indices=null_hit_indices,
        recaman_scores=recaman_scores,
        null_scores=null_scores,
        random_trials=args.random_trials,
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
