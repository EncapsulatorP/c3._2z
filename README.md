# Omega Stability — Iterated Logarithm Framework

![Omega Stability Banner](assets/omega-readme-banner.svg)

This repository implements and tests an **analytic tower framework** built around iterated exponential and logarithmic maps, with applications to numerical optimization, root-finding, sequence analysis, and digit-symbolic number theory.

---

## Core Definitions

### The Tower and Residue

| Symbol | Definition | Description |
|---|---|---|
| `phi(z)` | `2z` | Seed function |
| `tower3(c, u)` | `c^(c^(c^u))` | Depth-3 power tower with base `c` |
| `residue(c, z, eps)` | see below | Perturbation residue through the log-chain |
| `L2(x)` | `log(log(x))` | Double-log coordinate ("comparability class") |
| `omega3(c, x)` | `log_c(x) − 3` | Depth-3 log alignment residual |

**Residue (`omega`):** Given a small perturbation `eps` at the top of the tower, the residue measures how that perturbation propagates after being passed back through the full chain via iterated logarithms. Formally it is computed by:

$$\delta_0 = \frac{\ln(1 + \varepsilon / x_0)}{\ln c_0}, \qquad \delta_{k+1} = \frac{\ln(1 + \delta_k / x_k)}{\ln c_{k+1}}$$

where `x_k` are the exact forward values in the tower. A small residue means the tower is stable: the top-level perturbation is absorbed before propagating to the base.

### L2-Coordinate (Comparability Class)

The double-log map `L2 = log(log(x))` reparametrizes a heavy-tailed space into a well-conditioned quadratic one. The key identity is:

$$\nabla_x \mathcal{L}(L_2(x)) = \frac{L_2'(x)}{x \ln x} \cdot \nabla_{L_2} \mathcal{L}$$

In raw `x`-space this gradient is suppressed by `1/(x ln x)`, making descent numerically degenerate at large scale. Working directly in `L2` removes this conditioning penalty.

### Digit-Chain Operator

A configurable map `T(digits, base)` acts on the digit representation of an integer in a given base. The repository uses a **placeholder operator** (cyclic adjacent sum mod base) as a scaffolding stand-in; the scientific object is a specific one-line rule that maps the seed `299792458` (speed of light, m/s) to `133732252`. Results from tests using the placeholder are treated as null-model characterizations only.

### Positional Layers and Obstruction Signature

For a digit string `n`, the **obstruction signature** is computed per base `b` by evaluating the mirror-contraction depth of an iterated tower anchored at the positional value of `n` in base `b`. Each layer reports:
- `h_star_peak` — the level at which the obstruction is loudest
- `h_star_first_break` — the level where mirror contraction first fails
- **Anchor class** — whether the peak aligns near `1`, `e`, an integer, or is generic

The combined multi-base signature (e.g. `b10:h1:integer:d3||b5:h4:integer:d4||b2:h1:integer:d5`) encodes the cross-layer obstruction structure.

---

## Repository Structure

```
omega_stability_tests.py        # Test 1 — tower residue stability sweep
omega_stability_tests2.py       # Test 2 — digit-chain null model, multi-base
omega_stability_tests3.py       # Test 3 — blind spots, Recaman vs null walk
omega_stability_tests4.py       # Test 4 — digit entropy state metrics
omega_stability_tests5.py       # Test 5 — multi-precision positional layer stability
omega_stability_tests6.py       # Test 6 — obstruction spectrum (bases 10, 5, 2)
omega_stability_tests7.py       # Test 7 — extended obstruction spectrum (bases 10, 7, 5, 3, 2)

applications/
  heavy_tailed_ml_optimization.py   # Gradient descent in L2 vs raw x
  pathological_root_finding.py      # Newton in L2 vs raw x
  shadow_thresholds.py              # Log-level jump-score anomaly detector

outputs/
  omega_stability/      # Test 1 results (CSV + conclusions)
  omega_stability2/     # Test 2 results
  omega_stability3/     # Test 3 results
  omega_stability4/     # Test 4 results
  omega_stability5/     # Test 5 results
  omega_stability6/     # Test 6 results
  omega_stability7/     # Test 7 results

applications/outputs/
  heavy_tailed_ml_optimization/
  pathological_root_finding/
  shadow_thresholds/
```

---

## Goals

1. **Measure analytic stability** of the depth-3 power tower under input perturbations and verify the residue computation is numerically reliable.
2. **Search for a canonical base** — a value `c*` where the residue profile has a non-trivial interior minimum, indicating a structurally selected coordinate.
3. **Test the digit-chain construction** — determine whether a specific symbolic operator on digit representations produces a base-invariant chain starting from `299792458`.
4. **Characterize the Recaman sequence** relative to a matched random-walk null model using the log-alignment and obstruction frameworks.
5. **Demonstrate applied utility** of the L2 reparametrization for heavy-tailed optimization and pathological root-finding.
6. **Build a falsifiable digit obstruction metric** — a multi-base obstruction signature that goes beyond cheap statistics like first-digit Benford probability.

---

## Named Examples Used Across Tests

| Label | Value | Notes |
|---|---|---|
| `light_speed` | `299792458` | Speed of light (m/s); primary digit-chain seed |
| `mixed_2357` | `233732252` | Candidate "next chain value" variant; digit set {2,3,7} |
| `balanced_12` | `122121212` | Maximally balanced two-symbol string (entropy reference) |
| `repunit` | `111111111` | Boundary state; single-symbol, zero entropy score |

---

## Dependencies

- Python 3.9+
- `numpy`
- `matplotlib`
- `mpmath` (Tests 5, 6, 7 — 250 decimal digit precision)

---

## Running the Tests

Each script is self-contained and writes its outputs to the corresponding `outputs/omega_stabilityN/` directory.

```powershell
python omega_stability_tests.py
python omega_stability_tests2.py
python omega_stability_tests3.py
python omega_stability_tests4.py
python omega_stability_tests5.py
python omega_stability_tests6.py
python omega_stability_tests7.py

python applications/heavy_tailed_ml_optimization.py
python applications/pathological_root_finding.py
python applications/shadow_thresholds.py
```

All scripts accept `--help` for configurable parameters (sample counts, bases, precision, output directory, etc.).

---

## Key Null Models

Every quantitative claim is benchmarked against one of:
- **Residue sweep baseline** — monotone behavior with no interior minimum
- **Matched random walk** — a random walk with the same step-magnitude distribution as Recaman, used for sequence comparisons in Tests 3, 6, 7
- **Random digit null model** — 1000 uniformly-sampled 9-digit integers, used to calibrate the obstruction spectrum in Tests 5, 6, 7
