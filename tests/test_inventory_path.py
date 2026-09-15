"""Inventory-through-the-session tests, and the horizon effect itself."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from lob_simulator.agents.quoting import AvellanedaStoikovMM, InventorySkewMM
from lob_simulator.market import INFORMED, SUBJECT_AGENT_ID
from lob_simulator.research.inventory_path import run_inventory_paths

N_TICKS = 1000
SPEC = replace(INFORMED, n_noise=15)


def _skew() -> InventorySkewMM:
    return InventorySkewMM(
        agent_id=SUBJECT_AGENT_ID,
        cash=1_000_000,
        inventory=0,
        quote_qty=5,
        half_spread_ticks=2.0,
        skew_k=0.05,
    )


def _as(gamma: float = 2e-5, sigma: float = 3.2, k: float = 0.6) -> AvellanedaStoikovMM:
    """gamma * sigma^2 * T = 0.2 ticks/unit at the open: four times Skew's 0.05, zero at T."""
    return AvellanedaStoikovMM(
        agent_id=SUBJECT_AGENT_ID,
        cash=1_000_000,
        inventory=0,
        quote_qty=5,
        gamma=gamma,
        sigma=sigma,
        k=k,
        horizon_t=float(N_TICKS),
    )


class TestShape:
    def test_one_row_per_seed_one_column_per_tick(self) -> None:
        paths = run_inventory_paths(
            _skew, strategy_name="Skew", spec=SPEC, seeds=range(3), n_ticks=200
        )
        assert paths.inventory.shape == (3, 200)
        assert paths.terminal_pnl.shape == (3,)
        assert paths.mean_markout.shape == (3,)
        assert paths.n_ticks == 200
        assert paths.mean_abs_inventory().shape == (200,)

    def test_inventory_starts_from_zero_and_is_reproducible(self) -> None:
        a = run_inventory_paths(_skew, strategy_name="Skew", spec=SPEC, seeds=[5], n_ticks=100)
        b = run_inventory_paths(_skew, strategy_name="Skew", spec=SPEC, seeds=[5], n_ticks=100)
        assert (a.inventory == b.inventory).all()
        assert abs(a.inventory[0, 0]) <= 10  # at most one fill either side on tick 0

    def test_mean_abs_inventory_over_matches_direct_computation(self) -> None:
        paths = run_inventory_paths(
            _skew, strategy_name="Skew", spec=SPEC, seeds=range(2), n_ticks=100
        )
        expected = np.mean(np.abs(paths.inventory[:, 20:60]))
        assert paths.mean_abs_inventory_over(20, 60) == expected
        assert paths.mean_abs_inventory_at(99) == paths.mean_abs_inventory()[99]


class TestHorizonEffect:
    """A-S's skew is gamma*sigma^2*(T-t): strong early, gone at T."""

    def test_as_skew_per_unit_decays_to_zero_at_the_horizon(self) -> None:
        mm = _as()
        assert mm.skew_per_unit(0) > 0.05
        assert mm.skew_per_unit(N_TICKS // 2) > 0
        assert mm.skew_per_unit(N_TICKS) == 0.0
        assert mm.skew_per_unit(N_TICKS + 50) == 0.0

    def test_as_controls_inventory_mid_session_and_lets_go_at_the_end(self) -> None:
        seeds = range(12)
        skew = run_inventory_paths(
            _skew, strategy_name="Skew", spec=SPEC, seeds=seeds, n_ticks=N_TICKS
        )
        as_paths = run_inventory_paths(
            _as, strategy_name="A-S", spec=SPEC, seeds=seeds, n_ticks=N_TICKS
        )
        mid = (N_TICKS // 4, 3 * N_TICKS // 4)
        last = (N_TICKS - 25, N_TICKS)

        as_mid = as_paths.mean_abs_inventory_over(*mid)
        as_end = as_paths.mean_abs_inventory_over(*last)
        skew_mid = skew.mean_abs_inventory_over(*mid)

        assert as_mid < skew_mid, f"A-S mid-session |inv| {as_mid:.1f} vs Skew {skew_mid:.1f}"
        assert as_end > 2 * as_mid, (
            f"A-S should let inventory go as its skew vanishes: end {as_end:.1f}, mid {as_mid:.1f}"
        )
