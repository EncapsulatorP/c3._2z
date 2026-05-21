# Conclusions

## Bottom Line

This run supports the narrow claim that `8` is a **local blind spot at dyadic base 2**, but it does **not** support the broader claim that Recaman's visit to `8` is uniquely structured relative to a matched random null model.

## What The Run Established

- The blind-spot control behaved exactly as expected: `omega3(2, 8) = 0`.
- Nearby dyadic base `2.01` already moved off zero (`-2.143229e-02`), and the non-dyadic bases were much farther away.
- The base sweep placed the minimum `|omega3(c, 8)|` at `c = 2.000125`, which is consistent with a sharp local alignment centered on base 2.
- The compression factor was largest near base 2 in the scanned range, which supports the “local signal-compression / blind-spot” reading.

## What The Run Did Not Establish

- Recaman hitting `8` is not rare by itself in the matched random-walk null model.
- In the first 64 terms, the random walks hit `8` with probability `0.4440`.
- Among random walks that did hit `8`, `76.24%` reached it by index `16`, while Recaman reached it at index `16`. That is not exceptional timing.

## Sequence Interpretation

- At dyadic and near-dyadic bases (`2`, `2.01`, `sqrt(2)`, `1.5`), Recaman's hit-window residual was **worse** than the null mean, so those bases do not make the Recaman event look special.
- At non-dyadic bases (`e`, `pi`), Recaman's hit-window residual was **better** than the null mean, so those bases separate Recaman from the null more effectively.

## Interpretation

The strongest statement justified by this run is:

- `8` is a genuine alignment point for the `log_c(x) - 3` projection at base 2.

The stronger statement **not** justified is:

- “Recaman's absorption of `8` is uniquely tower-structured.”

At the moment, the local dyadic blind spot looks real, but the sequence-level uniqueness claim remains unproven.

## Practical Next Step

Use base 2 as a control only. If the framework is meant to detect sequence structure rather than trivial dyadic alignment, the meaningful follow-up is to compare Recaman and other sequences under non-dyadic bases such as `e` and `pi`, where the base-2 blind spot is not dominating the measurement.
