"""Monte Carlo batch runner tests.

Trial counts here are far below the 500 used for reported results. These test
the runner; scripts/run_monte_carlo.py produces the numbers.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from helpers import ScriptedAgent
from lob_simulator.agents.quoting import InventorySkewMM, NaiveMM
from lob_simulator.engine import Engine
from lob_simulator.market import INFORMED, SUBJECT_AGENT_ID, UNINFORMED
from lob_simulator.research.monte_carlo import (
    bootstrap_mean_ci,
    fill_markout_by_counterparty,
    mean_fill_markout,
    run_monte_carlo,
    run_session,
    run_trial,
)
from lob_simulator.types import Place, Side

N_TICKS = 500
SPEC = replace(UNINFORMED, n_noise=15)
INFORMED_SPEC = replace(INFORMED, n_noise=15)


def _naive_factory() -> NaiveMM:
    return NaiveMM(agent_id=0, cash=1_000_000, inventory=0, quote_qty=5, half_spread_ticks=2.0)


def _skew_factory() -> InventorySkewMM:
    return InventorySkewMM(
        agent_id=0, cash=1_000_000, inventory=0, quote_qty=5, half_spread_ticks=2.0, skew_k=0.05
    )


class TestSingleTrial:
    def test_pnl_decomposes_into_realized_plus_unrealized(self) -> None:
        result = run_trial(_naive_factory, seed=1, n_ticks=N_TICKS, spec=SPEC)
        assert result.terminal_pnl == pytest.approx(result.realized_pnl + result.unrealized_pnl)

    def test_fill_count_is_positive_over_a_real_run(self) -> None:
        result = run_trial(_naive_factory, seed=1, n_ticks=N_TICKS, spec=SPEC)
        assert result.fill_count > 0
        assert result.passive_units > 0
        assert result.passive_units + result.aggressor_units >= result.fill_count

    def test_had_two_sided_tick_true_for_a_healthy_run(self) -> None:
        result = run_trial(_naive_factory, seed=1, n_ticks=N_TICKS, spec=SPEC)
        assert result.had_two_sided_tick is True

    def test_same_seed_gives_identical_trial_result(self) -> None:
        a = run_trial(_naive_factory, seed=7, n_ticks=N_TICKS, spec=SPEC)
        b = run_trial(_naive_factory, seed=7, n_ticks=N_TICKS, spec=SPEC)
        assert a == b


class TestMarkoutByCounterparty:
    """Hand-checkable tape: one NaiveMM and two scripted takers, no noise.

    Tick 0: the MM quotes 98/102 around the reference mark of 100. Agent 7
    ("informed") lifts the ask for 3 at 102; agent 8 ("noise") hits the bid
    for 2 at 98. The book stays two-sided on the MM's remainders, mark 100.
    Tick 1: the MM re-quotes 98/102; agent 7 rests a bid at 101, so the mark
    becomes 101.5. Tick 2: the MM re-quotes 99/104 around 101.5; with agent
    7's bid still resting at 101 the mark is 102.5.
    """

    INFORMED_ID = 7
    NOISE_ID = 8

    def _run(self) -> tuple[Engine, int]:
        mm = NaiveMM(
            agent_id=SUBJECT_AGENT_ID, cash=1_000_000, inventory=0, quote_qty=5,
            half_spread_ticks=2.0,
        )
        informed = ScriptedAgent(
            agent_id=self.INFORMED_ID,
            cash=1_000_000,
            inventory=100,
            script=[
                [Place(agent_id=self.INFORMED_ID, side=Side.BUY, price=102, qty=3)],
                [Place(agent_id=self.INFORMED_ID, side=Side.BUY, price=101, qty=1)],
            ],
        )
        noise = ScriptedAgent(
            agent_id=self.NOISE_ID,
            cash=1_000_000,
            inventory=100,
            script=[[Place(agent_id=self.NOISE_ID, side=Side.SELL, price=98, qty=2)]],
        )
        engine = Engine([mm, informed, noise], reference_price=100.0)
        engine.run(3)
        return engine, mm.agent_id

    def test_tape_is_what_the_docstring_says(self) -> None:
        engine, _ = self._run()
        assert [(t.price, t.qty, t.aggressor_agent) for t in engine.trade_log] == [
            (102, 3, self.INFORMED_ID),
            (98, 2, self.NOISE_ID),
        ]
        assert [m.mark for m in engine.mark_log] == [100.0, 101.5, 102.5]

    def test_split_at_horizon_zero(self) -> None:
        engine, mm_id = self._run()
        split = fill_markout_by_counterparty(
            engine, mm_id, counterparties={self.INFORMED_ID}, horizon=0
        )
        assert split.horizon == 0
        assert (split.counterparty_units, split.other_units, split.total_units) == (3, 2, 5)
        assert split.counterparty_unit_share == pytest.approx(0.6)
        # Sold 3 at 102 against a mark of 100: +2/unit. Bought 2 at 98: +2/unit.
        assert split.counterparty_markout == pytest.approx(2.0)
        assert split.other_markout == pytest.approx(2.0)
        assert split.total_markout == pytest.approx(2.0)

    def test_split_at_horizon_one_separates_the_two_classes(self) -> None:
        engine, mm_id = self._run()
        split = fill_markout_by_counterparty(
            engine, mm_id, counterparties={self.INFORMED_ID}, horizon=1
        )
        # Mark one tick later is 101.5: the sale at 102 earns +0.5, the buy at 98 earns +3.5.
        assert split.counterparty_markout == pytest.approx(0.5)
        assert split.other_markout == pytest.approx(3.5)
        assert split.total_markout == pytest.approx((3 * 0.5 + 2 * 3.5) / 5)

    def test_horizon_is_clipped_at_the_end_of_the_tape(self) -> None:
        engine, mm_id = self._run()
        far = fill_markout_by_counterparty(
            engine, mm_id, counterparties={self.INFORMED_ID}, horizon=1_000
        )
        near = fill_markout_by_counterparty(
            engine, mm_id, counterparties={self.INFORMED_ID}, horizon=2
        )
        assert far == replace(near, horizon=1_000)

    def test_empty_counterparty_set_puts_everything_in_other(self) -> None:
        engine, mm_id = self._run()
        split = fill_markout_by_counterparty(engine, mm_id, counterparties=(), horizon=1)
        assert split.counterparty_units == 0
        assert math.isnan(split.counterparty_markout)
        assert split.other_units == 5
        assert split.other_markout == pytest.approx(split.total_markout)

    def test_mean_fill_markout_is_the_total_of_the_split(self) -> None:
        engine, mm_id = self._run()
        split = fill_markout_by_counterparty(
            engine, mm_id, counterparties={self.INFORMED_ID}, horizon=1
        )
        assert mean_fill_markout(engine, mm_id, horizon=1) == split.total_markout

    def test_agent_with_no_passive_fills_is_all_nan(self) -> None:
        engine, _ = self._run()
        split = fill_markout_by_counterparty(engine, self.NOISE_ID, counterparties=(), horizon=1)
        assert split.total_units == 0
        assert math.isnan(split.total_markout)
        assert math.isnan(split.counterparty_unit_share)

    def test_rejects_negative_horizon(self) -> None:
        engine, mm_id = self._run()
        with pytest.raises(ValueError):
            fill_markout_by_counterparty(engine, mm_id, counterparties=(), horizon=-1)


class TestCounterpartyFieldsInTrials:
    def test_uninformed_market_reports_none_for_the_informed_split(self) -> None:
        """No informed trader means "not applicable", which is None, not NaN:
        NaN would read as "had one, no fills" and would break trial equality."""
        session = run_session(_naive_factory(), spec=SPEC, seed=1, n_ticks=50)
        assert session.market.informed_agent_id is None
        result = run_trial(_naive_factory, seed=1, n_ticks=N_TICKS, spec=SPEC)
        assert result.informed_unit_share is None
        assert result.markout_vs_informed is None
        assert result.markout_vs_noise is None
        assert math.isfinite(result.mean_markout)

    def test_informed_market_split_is_consistent_with_the_total(self) -> None:
        session = run_session(_naive_factory(), spec=INFORMED_SPEC, seed=3, n_ticks=N_TICKS)
        informed = session.market.informed_agent_id
        assert informed == INFORMED_SPEC.n_noise + 1
        split = fill_markout_by_counterparty(session.engine, 0, counterparties={informed})
        assert split.counterparty_units > 0, "the informed trader never hit the MM in 500 ticks"
        assert split.other_units > 0
        weighted = (
            split.counterparty_units * split.counterparty_markout
            + split.other_units * split.other_markout
        ) / split.total_units
        assert split.total_markout == pytest.approx(weighted)

        result = run_trial(_naive_factory, seed=3, n_ticks=N_TICKS, spec=INFORMED_SPEC)
        assert result.informed_unit_share == pytest.approx(split.counterparty_unit_share)
        assert result.markout_vs_informed == pytest.approx(split.counterparty_markout)
        assert result.markout_vs_noise == pytest.approx(split.other_markout)
        assert result.passive_units == split.total_units

    def test_summary_carries_the_split_means(self) -> None:
        summary = run_monte_carlo(
            _naive_factory, strategy_name="NaiveMM", seeds=range(5), n_ticks=N_TICKS,
            spec=INFORMED_SPEC,
        )
        assert 0.0 < summary.informed_unit_share < 1.0
        assert math.isfinite(summary.markout_vs_informed)
        assert math.isfinite(summary.markout_vs_noise)
        assert summary.mean_passive_units > 0
        assert summary.mean_aggressor_units >= 0
        shares = [t.informed_unit_share for t in summary.trials]
        assert all(s is not None for s in shares)
        assert summary.informed_unit_share == pytest.approx(
            np.mean([s for s in shares if s is not None])
        )

    def test_uninformed_summary_split_is_nan_without_warnings(self) -> None:
        summary = run_monte_carlo(
            _naive_factory, strategy_name="NaiveMM", seeds=range(3), n_ticks=200, spec=SPEC
        )
        assert math.isnan(summary.informed_unit_share)
        assert math.isnan(summary.markout_vs_informed)
        assert math.isnan(summary.markout_vs_noise)


class TestBootstrapCI:
    def test_ci_brackets_the_sample_mean_for_a_large_enough_sample(self) -> None:
        rng = np.random.default_rng(1)
        values = rng.normal(loc=100.0, scale=10.0, size=200)
        lower, upper = bootstrap_mean_ci(values, n_bootstrap=1000, seed=0)
        assert lower < values.mean() < upper

    def test_ci_narrows_as_sample_size_grows(self) -> None:
        rng = np.random.default_rng(2)
        small = rng.normal(loc=0.0, scale=10.0, size=20)
        large = rng.normal(loc=0.0, scale=10.0, size=2000)
        lo_s, hi_s = bootstrap_mean_ci(small, n_bootstrap=1000, seed=0)
        lo_l, hi_l = bootstrap_mean_ci(large, n_bootstrap=1000, seed=0)
        assert (hi_l - lo_l) < (hi_s - lo_s)

    def test_deterministic_given_a_fixed_bootstrap_seed(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        a = bootstrap_mean_ci(values, n_bootstrap=500, seed=3)
        b = bootstrap_mean_ci(values, n_bootstrap=500, seed=3)
        assert a == b


class TestMonteCarloSummary:
    def test_summary_always_carries_mean_variance_skew_percentiles_and_ci(self) -> None:
        summary = run_monte_carlo(
            _naive_factory,
            strategy_name="NaiveMM",
            seeds=range(30),
            n_ticks=N_TICKS,
            spec=SPEC,
        )
        assert summary.n_trials == 30
        assert isinstance(summary.mean_pnl, float)
        assert isinstance(summary.var_pnl, float)
        assert isinstance(summary.skew_pnl, float)
        assert summary.p5_pnl <= summary.mean_pnl <= summary.p95_pnl
        lower, upper = summary.mean_pnl_ci95
        assert lower < summary.mean_pnl < upper

    def test_inventory_std_uses_the_same_ddof_as_pnl_variance(self) -> None:
        summary = run_monte_carlo(
            _naive_factory, strategy_name="NaiveMM", seeds=range(6), n_ticks=200, spec=SPEC
        )
        inventories = [t.terminal_inventory for t in summary.trials]
        assert summary.std_terminal_inventory == pytest.approx(np.std(inventories, ddof=1))
        assert summary.var_pnl == pytest.approx(
            np.var([t.terminal_pnl for t in summary.trials], ddof=1)
        )

    def test_two_sided_tick_fraction_at_least_99_percent(self) -> None:
        summary = run_monte_carlo(
            _naive_factory,
            strategy_name="NaiveMM",
            seeds=range(40),
            n_ticks=N_TICKS,
            spec=SPEC,
        )
        assert summary.two_sided_tick_fraction >= 0.99, (
            f"only {summary.two_sided_tick_fraction:.1%} of trials had a two-sided tick"
        )

    def test_same_seed_set_reproducible_across_runs(self) -> None:
        seeds = range(15)
        a = run_monte_carlo(
            _naive_factory, strategy_name="NaiveMM", seeds=seeds, n_ticks=N_TICKS, spec=SPEC
        )
        b = run_monte_carlo(
            _naive_factory, strategy_name="NaiveMM", seeds=seeds, n_ticks=N_TICKS, spec=SPEC
        )
        assert a.mean_pnl == b.mean_pnl
        assert a.trials == b.trials

    def test_inventory_skew_has_tighter_terminal_inventory_than_naive(self) -> None:
        """The same claim as test_inventory_skew.py, through the runner end to end."""
        seeds = range(25)
        naive = run_monte_carlo(
            _naive_factory, strategy_name="NaiveMM", seeds=seeds, n_ticks=N_TICKS, spec=SPEC
        )
        skew = run_monte_carlo(
            _skew_factory,
            strategy_name="InventorySkewMM",
            seeds=seeds,
            n_ticks=N_TICKS,
            spec=SPEC,
        )

        naive_mean_abs_inv = sum(abs(t.terminal_inventory) for t in naive.trials) / naive.n_trials
        skew_mean_abs_inv = sum(abs(t.terminal_inventory) for t in skew.trials) / skew.n_trials
        assert skew_mean_abs_inv < naive_mean_abs_inv
