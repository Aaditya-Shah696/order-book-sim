"""Monte Carlo batch runner tests.

Trial counts here are far below the 500 used for reported results. These test
the runner; scripts/run_monte_carlo.py produces the numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from lob_simulator.agents.quoting import InventorySkewMM, NaiveMM
from lob_simulator.research.monte_carlo import bootstrap_mean_ci, run_monte_carlo, run_trial

N_TICKS = 500
N_NOISE = 15


def _naive_factory() -> NaiveMM:
    return NaiveMM(agent_id=0, cash=1_000_000, inventory=0, quote_qty=5, half_spread_ticks=2.0)


def _skew_factory() -> InventorySkewMM:
    return InventorySkewMM(
        agent_id=0, cash=1_000_000, inventory=0, quote_qty=5, half_spread_ticks=2.0, skew_k=0.05
    )


class TestSingleTrial:
    def test_pnl_decomposes_into_realized_plus_unrealized(self) -> None:
        result = run_trial(_naive_factory, seed=1, n_ticks=N_TICKS, n_noise=N_NOISE)
        assert result.terminal_pnl == pytest.approx(result.realized_pnl + result.unrealized_pnl)

    def test_fill_count_is_positive_over_a_real_run(self) -> None:
        result = run_trial(_naive_factory, seed=1, n_ticks=N_TICKS, n_noise=N_NOISE)
        assert result.fill_count > 0

    def test_had_two_sided_tick_true_for_a_healthy_run(self) -> None:
        result = run_trial(_naive_factory, seed=1, n_ticks=N_TICKS, n_noise=N_NOISE)
        assert result.had_two_sided_tick is True

    def test_same_seed_gives_identical_trial_result(self) -> None:
        a = run_trial(_naive_factory, seed=7, n_ticks=N_TICKS, n_noise=N_NOISE)
        b = run_trial(_naive_factory, seed=7, n_ticks=N_TICKS, n_noise=N_NOISE)
        assert a == b


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
            n_noise=N_NOISE,
        )
        assert summary.n_trials == 30
        assert isinstance(summary.mean_pnl, float)
        assert isinstance(summary.var_pnl, float)
        assert isinstance(summary.skew_pnl, float)
        assert summary.p5_pnl <= summary.mean_pnl <= summary.p95_pnl
        lower, upper = summary.mean_pnl_ci95
        assert lower < summary.mean_pnl < upper

    def test_two_sided_tick_fraction_at_least_99_percent(self) -> None:
        summary = run_monte_carlo(
            _naive_factory,
            strategy_name="NaiveMM",
            seeds=range(40),
            n_ticks=N_TICKS,
            n_noise=N_NOISE,
        )
        assert summary.two_sided_tick_fraction >= 0.99, (
            f"only {summary.two_sided_tick_fraction:.1%} of trials had a two-sided tick"
        )

    def test_same_seed_set_reproducible_across_runs(self) -> None:
        seeds = range(15)
        a = run_monte_carlo(
            _naive_factory, strategy_name="NaiveMM", seeds=seeds, n_ticks=N_TICKS, n_noise=N_NOISE
        )
        b = run_monte_carlo(
            _naive_factory, strategy_name="NaiveMM", seeds=seeds, n_ticks=N_TICKS, n_noise=N_NOISE
        )
        assert a.mean_pnl == b.mean_pnl
        assert a.trials == b.trials

    def test_inventory_skew_has_tighter_terminal_inventory_than_naive(self) -> None:
        """The same claim as test_inventory_skew.py, through the runner end to end."""
        seeds = range(25)
        naive = run_monte_carlo(
            _naive_factory, strategy_name="NaiveMM", seeds=seeds, n_ticks=N_TICKS, n_noise=N_NOISE
        )
        skew = run_monte_carlo(
            _skew_factory,
            strategy_name="InventorySkewMM",
            seeds=seeds,
            n_ticks=N_TICKS,
            n_noise=N_NOISE,
        )

        naive_mean_abs_inv = sum(abs(t.terminal_inventory) for t in naive.trials) / naive.n_trials
        skew_mean_abs_inv = sum(abs(t.terminal_inventory) for t in skew.trials) / skew.n_trials
        assert skew_mean_abs_inv < naive_mean_abs_inv
