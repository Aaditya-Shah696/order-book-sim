"""Deterministic seeding: one master seed determines an entire run."""

from __future__ import annotations

import random


def spawn_rngs(seed: int, n: int) -> list[random.Random]:
    """Derive ``n`` independent ``random.Random`` streams from one master seed.

    The same ``(seed, n)`` always produces the same streams in the same order,
    regardless of what else has run in the process.
    """
    master = random.Random(seed)
    return [random.Random(master.getrandbits(64)) for _ in range(n)]
