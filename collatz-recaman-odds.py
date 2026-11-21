#!/usr/bin/env python3
"""Exact finite counts for Collatz-driven Recamán jumps.

Draw uniformly from the inclusive integer bounds. Each flight is one trial
and only its last attempted subtraction is scored. Reverse order is a
control on the same jump list. A blocked subtraction scores zero.
"""

import argparse
from collections import Counter
import json

STEP_LIMIT = 5_000


def shortcut(n):
    """Perform one shortcut Collatz step."""
    return (3 * n + 1) // 2 if n % 2 else n // 2


def full_flight(seed):
    """Return the positive states before the first arrival at 1."""
    states = []
    n = seed
    for _ in range(STEP_LIMIT):
        if n == 1:
            break
        states.append(n)
        n = shortcut(n)
    if n != 1:
        raise RuntimeError(f"Seed {seed} exceeded the step limit of {STEP_LIMIT}")
    return states


def repunit_flight(k):
    """Return the k odd states before the first even landing."""
    n = (1 << k) - 1
    states = []
    for _ in range(k):
        states.append(n)
        n = shortcut(n)
    return states


def recaman(jumps):
    """Use supplied jump lengths and score the final attempted subtraction."""
    seen = {0}
    position = 0
    obstructed = False
    for jump in jumps:
        down = position - jump
        obstructed = down < 0 or down in seen
        position = position + jump if obstructed else down
        seen.add(position)
    return position, 0 if obstructed else (1 if position % 2 else -1)


def repunit_score(k):
    """Closed form of the forward repunit score."""
    return int(k % 3 == 0) - 2 * int(k % 6 == 0)


def tally(flights):
    forward = Counter()
    reverse = Counter()
    for jumps in flights:
        parity = sum(jumps) % 2
        for order, counts in ((jumps, forward), (reversed(jumps), reverse)):
            position, sign = recaman(order)
            assert position % 2 == parity
            counts[sign] += 1
    return {
        label: {str(sign): histogram[sign] for sign in (1, -1, 0)}
        for label, histogram in (("forward", forward), ("reverse", reverse))
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-seed", type=int, default=100_000)
    parser.add_argument("--max-repunit-length", type=int, default=1_025)
    args = parser.parse_args()
    if min(args.max_seed, args.max_repunit_length) < 2:
        parser.error("both upper bounds must be at least 2")
    assert recaman([7, 11, 17]) == (1, 1)
    assert shortcut(repunit_flight(3)[-1]) == 3**3 - 1
    for k in range(2, args.max_repunit_length + 1):
        assert recaman(repunit_flight(k))[1] == repunit_score(k)
    results = {
        "all_seeds": {
            "range": [2, args.max_seed],
            **tally(full_flight(n) for n in range(2, args.max_seed + 1)),
        },
        "binary_repunit_lengths": {
            "range": [2, args.max_repunit_length],
            **tally(repunit_flight(k) for k in range(2, args.max_repunit_length + 1)),
        },
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
