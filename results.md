# Results Summary

This document collects the quantitative outcomes and conclusions from all seven omega-stability tests and the three application runs.

---

## Test 1 — Tower Residue Stability (`omega_stability_tests.py`)

**Goal:** Verify that the depth-3 residue computation is numerically stable across a sweep of `z` values and base values `c`, and search for an interior canonical base `c*`.

### Residue Profile vs `z` (4 fixed bases)

| Base `c` | Min residue | Max residue |
|---|---|---|
| 1.2 | 4.18 × 10⁻⁵ | 1.06 × 10⁻⁴ |
| 1.5 | 1.62 × 10⁻⁸ | 4.75 × 10⁻⁶ |
| e ≈ 2.718 | 0 | 8.12 × 10⁻⁹ |
| 2.0 | 0 | 2.54 × 10⁻⁷ |

Residues at base `e` and base `2` were the smallest, consistent with known tower convergence properties near those bases.

### Base Sweep (c ∈ [1.01, 2.5], 500 points)

- **Minimum `|omega|`** found at sweep boundary `c = 2.5` (value `4.65 × 10⁻¹³²`), **not** at an interior point.
- Zero sign flips in `d|omega|/dc`: the profile was monotone — no interior canonical base was found.
- Zero-crossing intervals: 0.

### Symmetry Breaking (Test 3)

- Symmetric residue: `9.76 × 10⁻⁷`
- Max adjacent jump after perturbation: `1.32 × 10⁻⁸` — smooth, no discontinuous pitchfork.

### Validation

- 100 direct vs. stable comparison points: max absolute error `1.66 × 10⁻¹⁰`. The two computation paths agree to 10 significant figures.

### Conclusion

The tower is **numerically stable** and well-conditioned. However, **no interior canonical base** was found in the tested range — the sweep is monotone, not structurally selective. This run is best treated as a stable negative control.

---

## Test 2 — Digit-Chain Null Model (`omega_stability_tests2.py`)

**Goal:** Test whether the seed `299792458` (speed of light) produces a base-invariant chain under a digit operator `T`, and characterize the null-model behavior under the placeholder rule.

### Operator Check

| | |
|---|---|
| Expected `T(299792458)` in base 10 | `233732252` | base 10 | 122121212 | base 10 111111111+e
| Actual placeholder output | `186616930` |
| Match | **No** |

The placeholder (cyclic adjacent sum mod base) does not reproduce the intended chain. All downstream results describe the placeholder behavior only.

### Seed Chain Across Bases

| Base | Terminal state | Label | Steps |
|---|---|---|---|
| 8 | `1755717340` | `max_steps` | 60 |
| 10 | `398729877` | `max_steps` | 60 |
| 16 | `00000000` | `repdigit_fixed_point` | 20 |

- Cross-base consistency **failed**: 2 distinct terminal labels and 3 distinct terminal digit strings.
- The endpoint class changes with the base representation, confirming strong representation-dependence.

### Null Model

| Base | P(same terminal label) | P(repdigit endpoint) |
|---|---|---|
| 8 | 0.995 | 0.000 |
| 10 | 1.000 | 0.000 |
| 16 | 1.000 | 1.000 |

Base-16 collapse to repdigit `00000000` has null probability 1.0 — this outcome is generic under the placeholder rule, not structurally special.

### Conclusion

The scaffolding is functional. **The scientific object (the real operator `T`) has not yet been plugged in.** Until then, all results are about the placeholder and cannot serve as evidence for or against the quoted digit chain.

---

## Test 3 — Blind Spots and Recaman vs Null Walk (`omega_stability_tests3.py`)

**Goal:** Test whether `x = 8` is a structurally selected point ("blind spot") at dyadic base 2, and whether the Recaman sequence's visit to 8 is anomalous versus a matched random-walk null model.

### Local Depth-3 Residual at `x = 8`

| Base `c` | `omega3(c, 8)` |
|---|---|
| 2 | **0.000000** (exact blind spot) |
| 2.01 | −2.14 × 10⁻² |
| √2 | +3.000 |
| 1.5 | +2.129 |
| e | −0.921 |
| π | −1.183 |

The value `omega3(2, 8) = 0` exactly because `log_2(8) = 3` — `8 = 2^3` is a perfect dyadic cube. This is a genuine, trivially-explained alignment at base 2.

### Base Sweep at `x = 8`

- Minimum `|omega3(c, 8)|` on sweep occurs at `c = 2.000125` (near-base-2), confirming a sharp local alignment centered on the dyadic point.
- Compression factor is largest near `c = 2`.

### Recaman vs Matched Random Walk

- Recaman first hits `8` at index **16**.
- Random-walk hit rate within 64 terms: **0.444** — hitting 8 is not rare.
- Among random walks that do hit 8: **76.24%** reach it by index 16 (Recaman's timing). Not exceptional.

| Base | Recaman `|omega3|` (hit window) | Null mean | Percentile |
|---|---|---|---|
| 2 | 1.446 | 1.174 | 0.779 |
| 2.01 | 1.419 | 1.160 | 0.766 |
| √2 | 5.893 | 4.663 | 0.802 |
| 1.5 | 4.601 | 3.602 | 0.802 |
| **e** | **0.619** | **0.841** | **0.266** |
| **π** | **0.565** | **0.915** | **0.143** |

Dyadic and near-dyadic bases make Recaman look *worse* than the null (high percentile). Non-dyadic bases (e, π) make Recaman look *better* than the null — these are the informative discriminators.

### Conclusion

- The dyadic blind spot at `x = 8` is **real but trivial** (an exact power of 2).
- Recaman's visit to 8 is **not statistically exceptional** in the null model.
- The most informative follow-up uses **non-dyadic bases** (e, π) where the base-2 alignment is not dominating.

---

## Test 4 — Digit State Entropy Metrics (`omega_stability_tests4.py`)

**Goal:** Define a normalized residue-robustness score from digit entropy and test whether it separates named digit states.

### Metric: `residue_robustness_score = H / log2(|alphabet|)`

| State | Value | Score | Alphabet size | `same_alphabet_rate_pm1` | `carry_slack_std` |
|---|---|---|---|---|---|
| `repunit` | 111111111 | 0.000 | 1 | 0.000 | 0.000 |
| `balanced_12` | 122121212 | **0.991** | 2 | 0.500 | 0.497 |
| `mixed_2357` | 233732252 | 0.876 | 4 | **0.611** | **1.618** |

### Findings

- The entropy score separates all three states as claimed.
- `122121212` scores highest by entropy (near-maximum entropy over a 2-symbol alphabet).
- However, `233732252` is **more robust under single-digit ±1 perturbations** (alphabet-size preservation rate 0.611 vs 0.500) and has **lower mean score drift** (0.017 vs 0.078).
- The carry-geometry story holds: `233732252` has much more uneven carry slack (std 1.618) — consistent with the "carry obstruction" picture.

### Conclusion

The normalized entropy score is a useful first-order discriminator, but **it is not a complete robustness metric**. Perturbation stability and carry geometry add independent information.

---

## Test 5 — Multi-Precision Positional Layer Stability (`omega_stability_tests5.py`)

**Goal:** Compare stability across a composite positional layer (base 10) and prime positional shadows (bases 5, 2) using 250-decimal-digit arithmetic, and evaluate whether cheap digit features can predict the stability class.

### Setup

- 1000 random 9-digit integers, mpmath 250 dps
- Stability = lower half of absorption ratio `Δ2/Δ1` where `Δ1 = log(V₁₀) − log(V₅)`, `Δ2 = log(log(V₁₀)) − log(log(V₅))`
- Stability threshold (median Δ2/Δ1): **0.05809**; ratio range [0.0560, 0.0645], std 0.00200

### Cheap-Feature Screening

| Feature | AUC | Threshold Accuracy | Pearson with ratio |
|---|---|---|---|
| `first_digit_benford_prob` | **0.9938** | **0.962** | 0.942 |
| `leading_trailing_asymmetry` | 0.5143 | 0.516 | 0.048 |
| `digit_entropy_bits` | 0.5010 | 0.508 | −0.005 |

### Conclusion

The stability split is **almost entirely recoverable from the first-digit Benford probability** (AUC 0.994, accuracy 96.2%). The tower classification is adding little beyond this cheap leading-digit statistic. Digit entropy by itself is uninformative (AUC ≈ 0.50).

---

## Test 6 — Obstruction Spectrum, Bases 10/5/2 (`omega_stability_tests6.py`)

**Goal:** Compute a multi-base obstruction signature for each digit string and test whether cheap features recover the cross-layer disagreement, comparing Recaman against a matched random walk.

### Random-Band Obstruction Spectrum (1000 samples)

- Prime-signature disagreement rate: **0.404**
- Late-prime-obstruction rate: **0.401**
- Dominant signature (596/1000): `b10:h1:integer:d3||b5:h1:integer:d4||b2:h1:integer:d5`
- Second signature (401/1000): `b10:h1:integer:d3||b5:h4:integer:d4||b2:h1:integer:d5`

### Named Examples

| Label | Value | Signature | Disagreement | Late-prime |
|---|---|---|---|---|
| `light_speed` | 299792458 | `b10:h1:integer:d3‖b5:h4:integer:d4‖b2:h1:integer:d5` | **1** | **1** |
| `balanced_12` | 122121212 | `b10:h1:integer:d3‖b5:h1:integer:d4‖b2:h1:integer:d5` | 0 | 0 |
| `mixed_2357` | 233732252 | `b10:h1:integer:d3‖b5:h1:integer:d4‖b2:h1:integer:d5` | 0 | 0 |
| `repunit` | 111111111 | `b10:sym‖b5:sym‖b2:sym` | 0 | 0 |

`299792458` (light speed) produces a **nontrivial cross-layer signature** with late-prime obstruction — it is not collapsed to the same object across all positional layers.

### Cheap-Feature Recovery

| Label | Best feature | AUC | Accuracy |
|---|---|---|---|
| `prime_signature_disagreement` | `first_digit_benford_prob` | 0.805 | 0.738 |
| `late_prime_obstruction` | `first_digit_benford_prob` | 0.810 | 0.741 |

Benford partially recovers these labels but with meaningful residual error (~26%), unlike the near-perfect recovery in Test 5.

### Sequence Comparison

| Sequence | Disagreement rate | Signature changes | Mean peak score |
|---|---|---|---|
| Recaman (64 terms) | **0.703** | **61** | 5.885 |
| Matched random walk | 0.719 | 52 | **7.596** |

Recaman changes obstruction signature more often than the random walk (61 vs 52 changes), which is the **desired direction** for an anomaly detector. However, the random walk has a higher mean peak obstruction score, so the evidence is mixed.

### Conclusion

The light-speed seed is structurally distinguishable from the generic random band by its cross-layer signature. Cheap digit statistics only partially explain the obstruction labels, suggesting a **small independent obstruction signal**. Recaman shows more signature changes than the null but carries a lower mean score.

---

## Test 7 — Extended Obstruction Spectrum, Bases 10/7/5/3/2 (`omega_stability_tests7.py`)

**Goal:** Extend the obstruction framework to 5 bases including all primes up to 7, add a first-break depth metric alongside the peak metric, and re-characterize the Recaman vs null-walk comparison.

### Random-Band Spectrum (1000 samples, 9-digit integers)

- Peak-signature disagreement rate: **0.799**
- First-break-signature disagreement rate: **1.000** (base 10 is almost always contracting while prime layers almost always break later — the first-break label collapses)
- Late-prime peak delay rate: **0.525**
- Dominant first-break signature (932/1000): `b10:contracting:d3||b7:h4:generic:d4||b5:h4:integer:d4||b3:h4:generic:d4||b2:h5:generic:d5`

### Odd-Prime Delay Check

| Base | Peak delay vs base 10 | First-break delay vs base 10 |
|---|---|---|
| 7 | 0.004 | **0.986** |
| 5 | 0.401 | **1.000** |
| 3 | 0.120 | **1.000** |

**Base 5 is the strongest odd-prime anomaly** for peak delay. Base 7 shows almost no peak delay but always delays at the first-break level.

### Named Examples

| Label | Value | Peak cores | First-break score |
|---|---|---|---|
| `light_speed` | 299792458 | `h1:generic‖h1:generic‖h4:integer‖h1:generic‖h1:generic` | **0.991** |
| `balanced_12` | 122121212 | `h1:generic‖h1:generic‖h1:generic‖h2:integer‖h1:generic` | 0.278 |
| `mixed_2357` | 233732252 | `h1:generic‖h1:generic‖h1:generic‖h1:generic‖h1:generic` | 0.085 |
| `repunit` | 111111111 | `sym‖sym‖sym‖sym‖sym` | **0.000** |

`299792458` retains the highest first-break score (0.991), well above all others. Its distinction at base 5 (`h4:integer`) survives the expanded anchor-band fix.

### Cheap-Feature Recovery

| Label | Best feature | AUC | Accuracy |
|---|---|---|---|
| `peak_signature_disagreement` | `first_digit_benford_prob` | 0.677 | 0.799 |
| `late_prime_obstruction_peak` | `first_digit_benford_prob` | 0.716 | 0.697 |

After Benford threshold: residual error on `late_prime_obstruction_peak` is 0.303; the strongest additional feature is `leading_trailing_asymmetry` (residual AUC 0.571), indicating a **small but real independent obstruction signal** beyond Benford.

### Sequence Comparison (256 Recaman terms)

| Sequence | First-break disagreement | First-break changes | Mean peak score |
|---|---|---|---|
| Recaman (163 terms analyzed) | **0.896** | **136** | 3.586 |
| Matched random walk (202 terms) | 0.871 | 103 | **4.669** |

After removing small-width padding artifacts, Recaman still changes its first-break signature more often than the matched random walk (136 vs 103). The matched walk retains a higher mean peak score, so the sequence-level evidence is **mixed but directionally consistent** with Recaman exhibiting more structural variation.

### Conclusion

- The light-speed seed remains structurally separated after the anchor-band refinement.
- First-break delay collapses to a near-constant under this base set, limiting its discriminative power as a random-band label.
- For the non-degenerate peak-delay statistic, **base 5 is the strongest odd-prime discriminator**; base 7 is nearly absent.
- There is a small independent obstruction signal beyond Benford (residual AUC 0.571 from leading-trailing asymmetry).
- Recaman evidence is mixed: more signature changes than the null, but lower mean peak score.

---

## Application 1 — Heavy-Tailed ML Optimization

**File:** `applications/heavy_tailed_ml_optimization.py`

**Task:** Minimize `0.5 * (L2(x) − 3)²` where `L2(x) = log(log(x))`, starting from a very large scale.

| Method | Steps | Final loss | Final x | Final L2 |
|---|---|---|---|---|
| Raw gradient descent (in `x`) | 40 | **2.000** | 2.85 × 10⁶⁴ | 5.000 |
| L2-coordinate descent | 40 | **9.86 × 10⁻³²** | 5.28 × 10⁸ | **3.000** |

**Result:** L2-space descent converges to the exact minimum in the same number of steps. Raw-space descent fails to leave the plateau because the gradient is suppressed by `1/(x log x)`.

---

## Application 2 — Pathological Root Finding

**File:** `applications/pathological_root_finding.py`

**Task:** Solve `L2 + 0.05 sin(5 L2) = 5` (a perturbed equation with no closed form) using Newton's method.

| Method | Iterations | Final `|f|` | Final L2 |
|---|---|---|---|
| Newton in raw `x` | 25 | **7.99 × 10⁻³** | 4.999 |
| Newton in `L2`-space | **5** | **6.22 × 10⁻¹⁵** | 5.005 |

**Result:** L2-space Newton converges **5× faster** and to **12 orders of magnitude better residual**. Raw-space Newton pays the `1/(x log x)` conditioning penalty on every step.

---

## Application 3 — Shadow Thresholds (Sequence Anomaly Detection)

**File:** `applications/shadow_thresholds.py`

**Task:** Identify anomalous steps in the Recaman and Collatz(27) sequences using a multi-level log jump score `max(|ΔL1|, |ΔL2|, |ΔL3|)`.

### Top Recaman Anomalies

| Index | Value | Raw Δ | Jump score |
|---|---|---|---|
| 112 | 152 | +112 | **1.317** |
| 111 | 40 | −111 | 1.310 |
| 65 | 91 | +65 | 1.226 |
| 64 | 26 | −64 | 1.215 |
| 32 | 46 | +32 | 1.142 |

### Top Collatz(27) Anomalies

| Index | Value | Raw Δ | Jump score |
|---|---|---|---|
| 77 | 9232 | +6155 | **1.099** |
| 67 | 7288 | +4859 | 1.098 |
| 75 | 6154 | +4103 | 1.098 |

**Result:** The log-level jump score identifies **scale-class transitions** (steps that cross a comparability boundary) while suppressing mere magnitude blow-ups. For Collatz, the ascending phase cluster is flagged as uniformly anomalous. For Recaman, the large backward jumps near index 111–112 score highest.

---

## Overall Cross-Test Summary

| Theme | Finding |
|---|---|
| Tower numerical stability | Confirmed to ≤10⁻¹⁰ error; the residue computation is reliable |
| Canonical base `c*` | **Not found** in [1.01, 2.5]; profile is monotone |
| Digit-chain operator | Placeholder does not match the intended chain; scaffolding verified |
| Dyadic blind spot at `x=8` | Real but trivial (`8 = 2³`); non-dyadic bases are informative |
| Recaman vs null walk | Mixed: more signature changes, lower mean peak score |
| `299792458` obstruction | Persistently distinguished across all test configurations (first-break score 0.991) |
| Benford dominance | First-digit Benford probability is the strongest cheap predictor (AUC up to 0.994), but residual obstruction signal exists |
| L2 reparametrization | Dramatically improves both gradient descent (loss 2.0 → 10⁻³²) and root-finding (25 → 5 iterations) |

---

## Next Steps

| Area | Recommended next action |
|---|---|
| L2 reparametrization | Dramatically improves both gradient descent (loss 2.0 → 10⁻³²) and root-finding (25 → 5 iterations) |

---

## Remarks

**1. Implement the real operator T (Test 2 is explicitly broken)**
Every Test 2 result describes the placeholder, not the intended object. The scaffolding is verified; the science has not been plugged in yet. All Test 2 conclusions are about the placeholder and cannot serve as evidence for or against the quoted digit chain.

**2. Condition on Benford, then retest obstruction (highest priority)**
Test 5 shows AUC 0.994 for first-digit Benford probability — suspiciously close to perfect. This raises a concrete question: does the tower obstruction metric measure anything beyond Benford's law, or is it simply recovering it? Controlled test: stratify samples by leading digit, recompute within-stratum AUC. If within-stratum AUC ≈ 0.5, the obstruction signal is an artifact of Benford's law, not tower structure. This could invalidate a core claim.

**3. Fix the collapsed first-break metric (Test 7)**
First-break disagreement rate reached 1.000 because base 10 is always contracting — the metric is degenerate when base 10 is used as the reference. Using relative first-break depths between prime bases only (e.g., base-5 delay vs. base-7 delay) would recover a non-trivial discriminator without the ceiling collapse.

**4. Get a proper p-value for Recaman at non-dyadic bases**
Test 3 shows Recaman at base π is at the 14.3rd percentile vs. the null — suggestive but based on a small null sample. Running 10,000 null walks at bases e and π specifically would produce an actual p-value and either confirm or eliminate this as a signal worth pursuing.
