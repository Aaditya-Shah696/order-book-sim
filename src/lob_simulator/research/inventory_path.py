"""Inventory through the session, not just at its end.

Avellaneda-Stoikov's inventory skew is ``gamma * sigma^2 * (T - t)``: it is
largest at the open and exactly zero at the horizon. A strategy compared on
*terminal* inventory alone is therefore judged at the one tick where A-S is
built to stop caring. This module records ``|inventory|`` at every tick so the
two can be compared along the whole path.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from ..agents.quoting import QuotingAgent
from ..market import MarketSpec
from .monte_carlo import mean_fill_markout, run_session


@dataclass(frozen=True, kw_only=True)
class InventoryPaths:
    """Per-seed results for one strategy: ``inventory[i, t]`` is seed ``i``'s inventory
    after tick ``t``."""

    strategy_name: str
    seeds: tuple[int, ...]
    inventory: np.ndarray
    terminal_pnl: np.ndarray
    mean_markout: np.ndarray

    @property
    def n_ticks(self) -> int:
        return int(self.inventory.shape[1])

    def mean_abs_inventory(self) -> np.ndarray:
        """``mean_i |inventory[i, t]|`` for every ``t``."""
        result: np.ndarray = np.mean(np.abs(self.inventory), axis=0)
        return result

    def mean_abs_inventory_at(self, t: int) -> float:
        return float(self.mean_abs_inventory()[t])

    def mean_abs_inventory_over(self, start: int, stop: int) -> float:
        """Mean of ``|inventory|`` over seeds and over ticks ``start .. stop-1``."""
        return float(np.mean(np.abs(self.inventory[:, start:stop])))


def run_inventory_paths(
    mm_factory: Callable[[], QuotingAgent],
    *,
    strategy_name: str,
    spec: MarketSpec,
    seeds: Sequence[int],
    n_ticks: int,
) -> InventoryPaths:
    """Run ``mm_factory`` once per seed and keep every tick's inventory."""
    inventory = np.zeros((len(seeds), n_ticks), dtype=int)
    terminal_pnl = np.zeros(len(seeds))
    markouts = np.zeros(len(seeds))
    for i, seed in enumerate(seeds):
        mm = mm_factory()
        session = run_session(mm, spec=spec, seed=seed, n_ticks=n_ticks)
        engine = session.engine
        rows = [s for s in engine.agent_state_log if s.agent_id == mm.agent_id]
        inventory[i, :] = [s.inventory for s in rows]
        terminal_pnl[i] = mm.pnl(session.mark_for_pnl)
        markouts[i] = mean_fill_markout(engine, mm.agent_id)
    return InventoryPaths(
        strategy_name=strategy_name,
        seeds=tuple(seeds),
        inventory=inventory,
        terminal_pnl=terminal_pnl,
        mean_markout=markouts,
    )
