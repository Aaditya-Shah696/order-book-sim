"""Rolling volatility estimator and lambda(delta) = A*exp(-k*delta) fit tests."""

from __future__ import annotations

import math
import statistics
from dataclasses import replace

import numpy as np
import pytest

from lob_simulator.agents.zero_intelligence import ZeroIntelligenceAgent
from lob_simulator.calibration import (
    DIFFUSIVE_HORIZON,
    DepthFillCount,
    calibrate,
    estimate_sigma,
    estimate_sigma_at_horizon,
    fit_fill_intensity,
    rolling_volatility,
)
from lob_simulator.engine import Engine
from lob_simulator.market import INFORMED, UNINFORMED, build_market
from lob_simulator.seeding import spawn_rngs

REFERENCE_PRICE = 100
N_NOISE = 15
SPEC = replace(UNINFORMED, n_noise=N_NOISE)


def _run_zi_marks(seed: int, n_ticks: int) -> list[float]:
    rngs = spawn_rngs(seed, N_NOISE)
    noise = [
        ZeroIntelligenceAgent(
            agent_id=i + 1,
            cash=1_000_000,
            inventory=1000,
            reference_price=REFERENCE_PRICE,
            rng=rngs[i],
        )
        for i in range(N_NOISE)
    ]
    engine = Engine(noise, reference_price=float(REFERENCE_PRICE))
    engine.run(n_ticks)
    return [m.mark for m in engine.mark_log]


class TestRollingVolatility:
    def test_zero_volatility_for_a_constant_price_series(self) -> None:
        prices = [100.0] * 50
        rolling = rolling_volatility(prices, window=10)
        valid = rolling.dropna()
        assert (valid == 0.0).all()

    def test_matches_hand_computed_stdev_on_a_known_series(self) -> None:
        """Changes of [100, 101, 100, 101, ...] alternate between two fixed
        values, so the stdev of that two-point series is known exactly, in
        either unit."""
        prices = [100.0, 101.0] * 20
        price_rolling = rolling_volatility(prices, window=4, units="price")
        assert price_rolling.dropna().iloc[0] == pytest.approx(statistics.stdev([1, -1, 1, -1]))

        log_rolling = rolling_volatility(prices, window=4, units="log")
        r_up = math.log(101.0) - math.log(100.0)
        r_down = math.log(100.0) - math.log(101.0)
        expected = statistics.stdev([r_up, r_down, r_up, r_down])
        assert log_rolling.dropna().iloc[0] == pytest.approx(expected)

    def test_default_units_are_price_ticks_not_log_returns(self) -> None:
        """Avellaneda-Stoikov takes sigma in price units; the default must be that."""
        prices = [100.0, 103.0, 99.0, 104.0, 100.0, 102.0] * 10
        assert estimate_sigma(prices, window=5) == estimate_sigma(prices, window=5, units="price")
        assert estimate_sigma(prices, window=5) != estimate_sigma(prices, window=5, units="log")

    def test_price_sigma_is_log_sigma_times_price_level_to_first_order(self) -> None:
        prices = [1000.0, 1003.0, 999.0, 1004.0, 1000.0, 1002.0] * 10
        price_sigma = estimate_sigma(prices, window=5, units="price")
        log_sigma = estimate_sigma(prices, window=5, units="log")
        assert price_sigma == pytest.approx(log_sigma * 1000.0, rel=0.01)

    def test_estimate_sigma_requires_enough_history(self) -> None:
        with pytest.raises(ValueError):
            estimate_sigma([100.0, 100.5], window=100)


class TestSigmaStabilityAcrossSeeds:
    def test_sigma_estimator_converges_and_is_stable_across_seeds(self) -> None:
        sigmas = [
            estimate_sigma(_run_zi_marks(seed=2000 + s, n_ticks=1000), window=80) for s in range(8)
        ]

        mean_sigma = sum(sigmas) / len(sigmas)
        variance = sum((s - mean_sigma) ** 2 for s in sigmas) / len(sigmas)
        cv = (variance**0.5) / mean_sigma

        assert mean_sigma > 0
        assert cv < 0.15, f"sigma varies too much across seeds: CV={cv:.3f}, values={sigmas}"


class TestSigmaAtHorizon:
    """``std(m[t+h] - m[t]) / sqrt(h)``: flat in ``h`` for a random walk, decaying for bounce."""

    def test_random_walk_gives_the_same_sigma_at_every_horizon(self) -> None:
        rng = np.random.default_rng(0)
        steps = rng.choice([-1.0, 1.0], size=20_000)
        prices = (1000.0 + np.cumsum(steps)).tolist()
        by_h = [estimate_sigma_at_horizon(prices, h) for h in (1, 10, 50)]
        assert by_h[0] == pytest.approx(1.0, rel=0.02)
        for value in by_h[1:]:
            assert value == pytest.approx(by_h[0], rel=0.15)

    def test_period_two_bounce_is_exactly_zero_at_even_horizons(self) -> None:
        prices = [100.0, 101.0] * 200
        assert estimate_sigma_at_horizon(prices, 2) == 0.0
        assert estimate_sigma_at_horizon(prices, 50) == 0.0
        assert estimate_sigma_at_horizon(prices, 1) > 0.5

    def test_matches_a_hand_computed_value(self) -> None:
        prices = [0.0, 1.0, 3.0, 6.0, 10.0]  # 2-tick changes: 3, 5, 7
        expected = statistics.stdev([3.0, 5.0, 7.0]) / math.sqrt(2)
        assert estimate_sigma_at_horizon(prices, 2) == pytest.approx(expected)

    def test_rejects_bad_inputs(self) -> None:
        with pytest.raises(ValueError):
            estimate_sigma_at_horizon([1.0, 2.0, 3.0], 0)
        with pytest.raises(ValueError):
            estimate_sigma_at_horizon([1.0, 2.0, 3.0], 2)  # only one 2-tick difference

    def test_uninformed_market_sigma_is_mostly_bounce(self) -> None:
        """No fundamental: the anchor never moves, so the multi-tick sigma
        collapses relative to the one-tick one, and ``calibrate`` reports the
        collapsed value as ``sigma_diffusive``."""
        cal = calibrate(SPEC, seed=21, n_ticks=800)
        assert cal.sigma_horizon == DIFFUSIVE_HORIZON
        assert cal.sigma_diffusive < 0.5 * cal.sigma, (
            f"sigma_diffusive {cal.sigma_diffusive:.3f} vs one-tick {cal.sigma:.3f}"
        )
        engine = build_market(SPEC, seed=21).engine
        engine.run(800)
        marks = [m.mark for m in engine.mark_log]
        assert cal.sigma_diffusive == estimate_sigma_at_horizon(marks, DIFFUSIVE_HORIZON)

    def test_informed_market_diffusive_sigma_is_the_fundamentals_own(self) -> None:
        """With a jump fundamental the mark diffuses; at h=50 the bounce has
        averaged out and what is left is the fundamental's per-tick std."""
        assert INFORMED.fundamental is not None
        target = INFORMED.fundamental.per_tick_std
        engine = build_market(INFORMED, seed=21).engine
        engine.run(2000)
        marks = [m.mark for m in engine.mark_log]
        one_tick = estimate_sigma_at_horizon(marks, 1)
        diffusive = estimate_sigma_at_horizon(marks, DIFFUSIVE_HORIZON)
        assert diffusive == pytest.approx(target, rel=0.3), (
            f"h={DIFFUSIVE_HORIZON} sigma {diffusive:.3f} vs fundamental {target:.3f}"
        )
        assert one_tick > 1.3 * target, "the one-tick sigma should be inflated by bounce"


class TestFillIntensityFit:
    def test_fill_rate_property(self) -> None:
        count = DepthFillCount(depth_ticks=3, fills=40, ticks=800)
        assert count.fill_rate == pytest.approx(0.05)

    def test_exponential_intensity_fits_well_and_decays_with_depth(self) -> None:
        fit = fit_fill_intensity(
            [1, 3, 5, 8, 12], spec=SPEC, n_ticks=800, seed=42
        )

        assert not math.isnan(fit.r_squared), (
            "got a degenerate (all-zero) fit -- widen the probed depths"
        )
        assert fit.r_squared > 0.8, f"exponential intensity fit is poor: R^2={fit.r_squared:.3f}"
        assert fit.A > 0
        assert fit.k > 0, f"intensity should decay with depth, got k={fit.k:.3f}"

        rates = [d.fill_rate for d in fit.depths]
        assert rates == sorted(rates, reverse=True), (
            "fill rate should decrease monotonically with depth"
        )

    def test_degenerate_all_zero_fill_rate_reports_nan_instead_of_crashing(self) -> None:
        """Depth 500 is far outside the ZI agents' +-10 tick offset range, so the
        probe never fills."""
        fit = fit_fill_intensity(
            [500], spec=SPEC, n_ticks=200, seed=1
        )
        assert fit.depths[0].fills == 0
        assert math.isnan(fit.r_squared)
        assert math.isnan(fit.A)
        assert math.isnan(fit.k)
