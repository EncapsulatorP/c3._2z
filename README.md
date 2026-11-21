# c3._2z · Prime Forms

**What can you build from the digits of the speed of light?** Start with the exact vacuum speed in a chosen unit, factor its displayed digits, then use the resulting heights to drive a geometric form or a phase-slot model.

[![An animated tour of exact units, prime towers, phase slots, and reversed Omega layers](assets/prime-forms-walkthrough.png)](index.html)

[Open the main index](index.html) · [Still preview](assets/prime-forms-walkthrough-still.png) · [Animated SVG source](assets/prime-forms-walkthrough.svg)

## Follow the implementation

| Start here | What to try |
| --- | --- |
| [Unit tree](index.html#unit-tree) | Compare exact unit conversions. In m/s the final digit is $8=2^3$. Toggle the cap or edit a digit rule and watch the stack change. |
| [Optical phase slots](index.html#phase-slots) | Turn each tower height into quarter-turns of phase. Compare the carrier and sidebands in the ideal modulation model. |
| [Prime Forms](index.html#prime-forms) and [logistic fork](index.html#logistic-fork) | Swap or bias the two strands. The knot, ribbon, rosette and loom respond to these controls. |
| [3D lattice](index.html#lattice) and [Hopf detector](index.html#dimensions) | Orbit a digit-seeded height field and average adjacent cells. Then compare a loop, its detector projection, and the fixed point produced at a different pump setting. |
| [Recamán walls](index.html#obs-recaman) and [photonic blockade](index.html#obs-photonic) | Click the number line to forbid a value and watch the Recamán arcs reroute. Then place blockade cavities in the path of light pulses and compare the two kinds of exclusion. |
| [Gain sweep](index.html#gain-title) and [research controls](index.html#research-controls) | Change one shared gain for a family of exponential towers. Compare their thresholds, then compare nearby offsets and equal-gain mirror responses. |
| [Ω towers](index.html#omega-towers), [light loop](index.html#omega-light) and [wave plate](index.html#order-of-waves) | Reverse two prime layers that share a fixed point and a slope. Their half-difference reveals the order through its leading second harmonic. |
| [Model checks](index.html#model-checks) | Inspect the live residuals and copy the current settings. |

The index opens as one page with a direct link to every screen. For a local preview, open **index.html** in a browser. [Earlier Prime Forms gallery](pages/prime-forms.html)

## Where the threads meet

The speed-of-light thread supplies the input, $c=299\,792\,458\ \mathrm{m/s}$, and the exact unit comparisons. The digit rule supplies the prime letters. Mirrors and logistic forks then turn those letters into paired signals, and the coarse-graining view adds a way to see what survives averaging.

The earlier Catalan and CM discussion supplied the quarter-turn motif. At the logistic two-cycle flip, the linearized two-phase map squares to $-I$. Reading one coordinate gives $\chi_4(n)=\sin(\pi n/2)$. Summing those signs with weights $1/n^2$ gives Catalan’s constant. At odd primes with $\chi_4(p)=-1$, the Gaussian CM curve $y^2=x^3-x$ has vanishing Frobenius trace.

The implemented Ω study asks a different cancellation question. Let $\Omega=W(1)$ and $T_p(x)=\Omega p^{\Omega-x}$. The compositions $T_7\circ T_5$ and $T_5\circ T_7$ have the same slope at $\Omega$. Their difference starts at second order, which is the amber trace in [The Order of Waves](index.html#order-of-waves).

Neon (Gemini) contributed the visual design, the RG lattice, the Hopf detector and the obstructions lab. The [research archive](RESEARCH_ARCHIVE.md) preserves the earlier notebook and links to its scripts and results.
