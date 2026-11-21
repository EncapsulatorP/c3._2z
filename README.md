# c3._2z · Omega Stability

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

**A change of units changes the digit tower.** Write the vacuum speed of light as `299792458` in metres per second. Counting prime factors with multiplicity in each digit gives `122121213`. Capping each count at two gives `122121212`. The only change is the third factor of $2$ in $8=2^3$. The [Unit tree](index.html#unit-tree) lets you reproduce that calculation across unit choices. The speed is the same physical quantity throughout, and the encoding depends on the displayed number.

**Reversing two layers preserves the slope and changes the wave.** Let $\Omega=W(1)$ and set $T_p(x)=\Omega p^{\Omega-x}$ for prime $p$. Both compositions $T_q\circ T_p$ and $T_p\circ T_q$ fix $\Omega$ and have derivative $\Omega^2\ln p\ln q$ there. For distinct primes their second derivatives differ. With a small sinusoidal input around $\Omega$, the mean-subtracted half-difference of the two outputs therefore has a leading second harmonic. [Ω towers](index.html#omega-towers) computes the comparison, and [The Order of Waves](index.html#order-of-waves) draws it.

## Explore the implementation

Open [index.html](index.html) from a local checkout to run the 13-stop tour. Every screen is also accessible from the main menu. The phase, lattice, and feedback screens use explicit mathematical models with adjustable inputs.

| Screen | What you can examine |
| --- | --- |
| [Unit tree](index.html#unit-tree) and [optical phase slots](index.html#phase-slots) | Exact unit conversions, digit-factor layers, and an ideal phase-modulation spectrum |
| [Prime Forms](index.html#prime-forms) and [logistic fork](index.html#logistic-fork) | Even and odd mirror responses as the strands are swapped or biased |
| [3D lattice](index.html#lattice) and [Hopf detector](index.html#dimensions) | Coarse-graining a digit-seeded height field and projecting an orbit to a detector |
| [Gain sweep](index.html#gain-title) and [research controls](index.html#research-controls) | Stability thresholds across a family of towers driven by one gain |
| [Ω towers](index.html#omega-towers) and [Ω light loop](index.html#omega-light) | Reversed compositions and a model of optical feedback |
| [Model checks](index.html#model-checks) | Live numerical checks and a copyable record of the current settings |

[![Animated walkthrough of the unit tree, prime towers, and Ω wave plate](assets/prime-forms-walkthrough.png)](index.html)

[Still preview](assets/prime-forms-walkthrough-still.png) · [Animated SVG](assets/prime-forms-walkthrough.svg) · [Earlier gallery](pages/prime-forms.html)

## Reproduce the analytic sweep

```bash
python -m pip install numpy matplotlib
python omega_stability_tests.py --output-dir /tmp/omega-stability
```

The [reference outputs](outputs/omega_stability/) include the base sweep, the perturbation profiles, and a comparison against direct evaluation. The [applications](applications/) show the double-log coordinate in optimization and root-finding.

## From the original thread

The project grew from Zolton’s question about which patterns in the digits of the speed of light could survive a change in representation. [From the original thread](index.html#thread-notes) follows that question through the Goldbach and Collatz examples, the logistic quarter-turn, Catalan’s disk, and the prime-layer wave experiment.
