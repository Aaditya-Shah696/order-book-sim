"""Rolling volatility estimator and lambda(delta) = A*exp(-k*delta) fit tests."""

from __future__ import annotations

import math
import statistics

import pytest

from lob_simulator.agents.zero_intelligence import ZeroIntelligenceAgent
from lob_simulator.calibration import (
    DepthFillCount,
    estimate_sigma,
    fit_fill_intensity,
    rolling_volatility,
)
from lob_simulator.engine import Engine
from lob_simulator.seeding import spawn_rngs

REFERENCE_PRICE = 100
N_NOISE = 15


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
        """Log-returns of [100, 101, 100, 101, ...] alternate between two fixed
        values, so the stdev of that two-point series is known exactly."""
        prices = [100.0, 101.0] * 20
        rolling = rolling_volatility(prices, window=4)
        r_up = math.log(101.0) - math.log(100.0)
        r_down = math.log(100.0) - math.log(101.0)
        expected = statistics.stdev([r_up, r_down, r_up, r_down])
        assert rolling.dropna().iloc[0] == pytest.approx(expected)

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


class TestFillIntensityFit:
    def test_fill_rate_property(self) -> None:
        count = DepthFillCount(depth_ticks=3, fills=40, ticks=800)
        assert count.fill_rate == pytest.approx(0.05)

    def test_exponential_intensity_fits_well_and_decays_with_depth(self) -> None:
        fit = fit_fill_intensity(
            [1, 3, 5, 8, 12], n_ticks=800, n_noise=N_NOISE, seed=42, reference_price=REFERENCE_PRICE
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
            [500], n_ticks=200, n_noise=N_NOISE, seed=1, reference_price=REFERENCE_PRICE
        )
        assert fit.depths[0].fills == 0
        assert math.isnan(fit.r_squared)
        assert math.isnan(fit.A)
        assert math.isnan(fit.k)
