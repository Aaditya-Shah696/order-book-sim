"""FundamentalValue: a jump random walk noise traders anchor to and an informed trader knows."""

from __future__ import annotations

import random


class FundamentalValue:
    """Integer-tick fundamental value following a jump random walk.

    Each tick, with probability ``p_jump``, the value moves by a uniform
    ``1..max_jump_ticks`` in a random direction; otherwise it stays put. The
    path is generated lazily from its own ``rng`` and cached, so ``value_at(t)``
    is deterministic for a given seed no matter which agent asks first or how
    far ahead it asks. The value never drops below 1.
    """

    def __init__(
        self,
        *,
        initial: int,
        rng: random.Random,
        p_jump: float,
        max_jump_ticks: int,
    ) -> None:
        if initial <= 0:
            raise ValueError(f"initial must be positive, got {initial}")
        if not (0.0 <= p_jump <= 1.0):
            raise ValueError(f"p_jump must be in [0, 1], got {p_jump}")
        if max_jump_ticks <= 0:
            raise ValueError(f"max_jump_ticks must be positive, got {max_jump_ticks}")
        self._rng = rng
        self._p_jump = p_jump
        self._max_jump_ticks = max_jump_ticks
        self._path: list[int] = [initial]

    def value_at(self, t: int) -> int:
        """Fundamental value at tick ``t`` (extends the cached path as needed)."""
        if t < 0:
            raise ValueError(f"t must be non-negative, got {t}")
        while len(self._path) <= t:
            value = self._path[-1]
            if self._rng.random() < self._p_jump:
                direction = self._rng.choice((-1, 1))
                value += direction * self._rng.randint(1, self._max_jump_ticks)
            self._path.append(max(1, value))
        return self._path[t]

    def path(self, n_ticks: int) -> list[int]:
        """Values at ticks ``0 .. n_ticks-1``."""
        if n_ticks <= 0:
            return []
        self.value_at(n_ticks - 1)
        return self._path[:n_ticks]
