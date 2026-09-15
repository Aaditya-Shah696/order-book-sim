"""Market-impact sweep tests.

Levels, seeds and ticks here are far below what scripts/run_market_impact.py
uses. These test the machinery, not the effect size.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from lob_simulator.agents.quoting import AvellanedaStoikovMM, InventorySkewMM
from lob_simulator.market import UNINFORMED, MarketSpec
from lob_simulator.research.market_impact import analyze_market_impact, run_impact_trial

N_TICKS = 400
K_HAT, SIGMA_HAT, GAMMA = 0.48, 3.56, 1e-5
"""Price-unit sigma (ticks per sqrt(tick)); see tests/test_avellaneda_stoikov.py."""


def _spec(n_noise: int) -> MarketSpec:
    return replace(UNINFORMED, n_noise=n_noise)


def _skew_factory() -> InventorySkewMM:
    return InventorySkewMM(
        agent_id=0, cash=1_000_000, inventory=0, quote_qty=5, half_spread_ticks=2.0, skew_k=0.05
    )


def _as_factory() -> AvellanedaStoikovMM:
    return AvellanedaStoikovMM(
        agent_id=0,
        cash=1_000_000,
        inventory=0,
        quote_qty=5,
        gamma=GAMMA,
        sigma=SIGMA_HAT,
        k=K_HAT,
        horizon_t=float(N_TICKS),
    )


class TestRunImpactTrial:
    def test_mm_share_is_a_fraction_between_zero_and_one(self) -> None:
        trial = run_impact_trial(_skew_factory, spec=_spec(10), seed=1, n_ticks=N_TICKS)
        assert 0.0 <= trial.mm_share <= 1.0

    def test_mm_share_is_higher_with_fewer_noise_agents(self) -> None:
        """The mechanism the sweep uses to vary the MM's share of order flow."""
        low_noise = run_impact_trial(_skew_factory, spec=_spec(3), seed=1, n_ticks=N_TICKS)
        high_noise = run_impact_trial(_skew_factory, spec=_spec(30), seed=1, n_ticks=N_TICKS)
        assert low_noise.mm_share > high_noise.mm_share

    def test_reproducible_given_the_same_seed(self) -> None:
        a = run_impact_trial(_skew_factory, spec=_spec(8), seed=5, n_ticks=N_TICKS)
        b = run_impact_trial(_skew_factory, spec=_spec(8), seed=5, n_ticks=N_TICKS)
        assert a == b


class TestAnalyzeMarketImpact:
    def test_finding_shape_and_bounds(self) -> None:
        finding = analyze_market_impact(
            skew_factory=_skew_factory,
            as_factory=_as_factory,
            spec=UNINFORMED,
            n_noise_values=[3, 8, 20],
            seeds=range(15),
            n_ticks=N_TICKS,
            n_bootstrap=300,
        )
        assert (
            len(finding.n_noise_values)
            == len(finding.mean_mm_share)
            == len(finding.variance_ratio)
            == 3
        )
        assert all(0.0 <= s <= 1.0 for s in finding.mean_mm_share)
        assert -1.0 <= finding.correlation <= 1.0
        lower, upper = finding.correlation_ci95
        assert lower <= finding.correlation <= upper
        assert finding.is_null == (lower <= 0 <= upper)

    def test_mm_share_decreases_monotonically_with_more_noise_agents(self) -> None:
        finding = analyze_market_impact(
            skew_factory=_skew_factory,
            as_factory=_as_factory,
            spec=UNINFORMED,
            n_noise_values=[3, 8, 20, 40],
            seeds=range(10),
            n_ticks=N_TICKS,
            n_bootstrap=100,
        )
        assert list(finding.mean_mm_share) == sorted(finding.mean_mm_share, reverse=True)

    def test_ci_widens_with_fewer_bootstrap_replicates_is_at_least_well_formed(self) -> None:
        """Checks only that the CI machinery returns a non-degenerate interval.

        Measuring the direction of the real effect is the script's job.
        """
        finding = analyze_market_impact(
            skew_factory=_skew_factory,
            as_factory=_as_factory,
            spec=UNINFORMED,
            n_noise_values=[3, 10, 25],
            seeds=range(20),
            n_ticks=N_TICKS,
            n_bootstrap=500,
        )
        lower, upper = finding.correlation_ci95
        assert not np.isnan(lower)
        assert not np.isnan(upper)
        assert lower < upper
