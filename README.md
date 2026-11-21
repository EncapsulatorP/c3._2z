# c3.2z ·  Power tower composition in physical constants like the speed of  light. Prime power tower composition Forms. Omega wave power towers

<p align="center">
  <a href="https://encapsulatorp.github.io/c3.2z/">
    <img src="assets/c32z-walkthrough.svg" alt="c3.2z 5-Scene Interactive Walkthrough" width="100%">
  </a>
</p>

<p align="center">
  <strong><a href="https://encapsulatorp.github.io/c3.2z/">Launch Interactive Lab →</a></strong>
  &nbsp;·&nbsp;
  <code>0 Dependencies</code>
  &nbsp;·&nbsp;
  <code>TypedArray Wall &amp; Dedup Kernels</code>
  &nbsp;·&nbsp;
  <code>8 Live Invariants</code>
</p>

<details>
<summary><strong>Technical Basics &amp; Non-Claims Contract (Click to expand)</strong></summary>

- **01 · 3D Coarse-Graining Lattice:** 32×32 RG relief averaging discrete prime-factor heights and Recamán wall densities.
- **02 · Recamán Wall Trees &amp; Dedup:** O(1) prime-sieve wall classifier comparing standard collisions (`a₂₀ = a₂₄ = 42`) against Strict Dedup (`0` duplicates).
- **03 · Odd Mirror Knot &amp; Prime Rosette:** Antisymmetric cubic map `r·ξ − e·ξ³` paired with a 5-fold prime-letter rosette.
- **04 · Ω = W(1) Prime Layers &amp; Optical Loop:** Prime pencil `Tₚ(x) = Ω·p^(Ω−x)` and Beer–Lambert feedback `xₙ₊₁ = exp(−xₙ)`.
- **05 · Hopf Detector:** Supercritical limit cycle across IQ, Coherent Mirror Sum, and Power-only projections.
- **Non-Claims:** Decimal digits of `c = 299,792,458 m/s` serve strictly as a deterministic symbolic test alphabet, not a physical derivation of `c`.

</details>



![Omega Stability](assets/omega-readme-banner.svg)

This project studies **how perturbations travel through power towers and what survives a change of representation**. The analytic framework uses iterated logarithms to measure how a perturbation at the output of a tower propagates back through its layers. Prime Forms extends that inquiry by changing the units used to write the speed of light and reversing the order of prime-indexed maps.

## Analytic tower framework

Start with the seed $u=2z$ and the depth-three tower $c^{c^{c^u}}$. Taking a logarithm transfers an additive perturbation back one layer. For a layer with base $b$ and unperturbed output $x$, the update is exact wherever both logarithms are defined.

$$
\log_b(x+\delta)-\log_b(x)=\frac{\ln(1+\delta/x)}{\ln b}.
$$

The script applies this update through the tower to recover the perturbation at the seed. It also sweeps the base to ask whether the response selects an interior minimum. In the recorded sweep at $z=1$, over $1.01\le c\le 2.5$, the smallest absolute remainder was at the upper endpoint. [Sweep results and validation](outputs/omega_stability/conclusions.md)

The earlier work also uses $L_2(x)=\ln\ln x$ as a coordinate for numerical optimization and root-finding. The [research archive](RESEARCH_ARCHIVE.md) preserves the original README, experiment inventory, and null models.

## Two constructions in Prime Forms

**A change of units changes the digit tower.** Write the vacuum speed of light as `299792458` in metres per second. Counting prime factors with multiplicity in each digit gives `122121213`. Capping each count at two gives `122121212`. The only change is the third factor of $2$ in $8=2^3$. You can reproduce the calculation with the unit selector and tower controls. The speed is the same physical quantity throughout, and the encoding depends on the displayed number.

**Reversing two layers preserves the slope and changes the wave.** Let $\Omega=W(1)$ and set $T_p(x)=\Omega p^{\Omega-x}$ for prime $p$. Both compositions $T_q\circ T_p$ and $T_p\circ T_q$ fix $\Omega$ and have derivative $\Omega^2\ln p\ln q$ there. For distinct primes their second derivatives differ. With a small sinusoidal input around $\Omega$, the mean-subtracted half-difference of the two outputs therefore has a leading second harmonic. The Ω towers screen computes the comparison, and [The Order of Waves](assets/omega-tower-waves.png) draws it.

## Explore the implementation

Open [index.html](index.html) from a local checkout. The gallery combines a draggable lattice, prime forms, and feedback experiments. Neo developed the visual layout and the helper that exports the current settings and live checks as a Markdown specification.

| Screen | What you can examine |
| --- | --- |
| [Optical phase slots](index.html#phase-slots) | An ideal phase-modulation spectrum driven by digit-factor layers |
| [Logistic fork](index.html#logistic-fork) | Even and odd mirror responses as the strands are swapped or biased |
| [Hopf detector](index.html#dimensions) | Projecting an orbit to a detector and varying the pump setting |
| [Gain sweep](index.html#gain-title) | Stability thresholds across a family of towers driven by one gain |

## Reproduce the analytic sweep

```bash
python -m pip install numpy matplotlib
python omega_stability_tests.py --output-dir /tmp/omega-stability
```

The [reference outputs](outputs/omega_stability/) include the base sweep, the perturbation profiles, and a comparison against direct evaluation. The [applications](applications/) show the double-log coordinate in optimization and root-finding.

## From the original thread

The project grew from Zolton’s question about which patterns in the digits of the speed of light could survive a change in representation. The [original walkthrough](https://github.com/EncapsulatorP/c3._2z/blob/1ced156cc8c9f3e8c27d770afb229fc26c146d69/index.html) follows that question through the Goldbach and Collatz examples, the logistic quarter-turn, Catalan’s disk, and the prime-layer wave experiment.

[![Animated walkthrough of the unit tree, prime towers, and Ω wave plate](assets/prime-forms-walkthrough.png)](assets/prime-forms-walkthrough.png)

[Still preview](assets/prime-forms-walkthrough-still.png) · [Animated SVG](assets/prime-forms-walkthrough.svg) · [Earlier gallery](pages/prime-forms.html)
