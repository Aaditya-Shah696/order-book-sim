"""Stylized-fact measurement functions, and what they find on the simulated tape.

The volatility-clustering mismatch at the bottom is asserted, so it fails if
the simulated tape ever starts reproducing clustering it should not.
"""

from __future__ import annotations

import numpy as np
import pytest

from lob_simulator.agents.zero_intelligence import ZeroIntelligenceAgent
from lob_simulator.engine import Engine
from lob_simulator.research.stylized_facts import (
    StylizedFactReport,
    abs_return_acf,
    acf_of,
    build_report,
    compute_signed_volume,
    log_returns,
    ofi_price_change_correlation,
    order_flow_imbalance,
    pearson_kurtosis,
    return_acf,
)
from lob_simulator.seeding import spawn_rngs
from lob_simulator.types import Side

REFERENCE_PRICE = 100
N_NOISE = 20
N_TICKS = 8000


def _simulated_report(seed: int) -> StylizedFactReport:
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
    engine.run(N_TICKS)
    marks = [m.mark for m in engine.mark_log]
    signed_volume = compute_signed_volume(
        N_TICKS,
        [c.t for c in engine.trade_context],
        [t.qty for t in engine.trade_log],
        [c.aggressor_side for c in engine.trade_context],
    )
    return build_report(
        "sim", prices=marks, signed_volume=signed_volume, acf_lags=25, ofi_window=10
    )


class TestLogReturns:
    def test_matches_hand_computation(self) -> None:
        returns = log_returns([100.0, 110.0, 99.0])
        assert returns[0] == pytest.approx(np.log(110.0 / 100.0))
        assert returns[1] == pytest.approx(np.log(99.0 / 110.0))


class TestPearsonKurtosis:
    def test_standard_normal_is_near_three(self) -> None:
        rng = np.random.default_rng(1)
        sample = rng.standard_normal(200_000)
        assert pearson_kurtosis(sample) == pytest.approx(3.0, abs=0.05)

    def test_laplace_is_clearly_above_three(self) -> None:
        """The Laplace distribution has Pearson kurtosis exactly 6."""
        rng = np.random.default_rng(2)
        sample = rng.laplace(size=200_000)
        assert pearson_kurtosis(sample) == pytest.approx(6.0, abs=0.3)


class TestAcf:
    def test_white_noise_has_near_zero_acf_at_every_lag(self) -> None:
        rng = np.random.default_rng(3)
        noise = rng.standard_normal(20_000)
        values = acf_of(noise, n_lags=10)
        assert np.all(np.abs(values) < 0.03)

    def test_strong_ar1_series_has_high_lag1_acf(self) -> None:
        rng = np.random.default_rng(4)
        n = 20_000
        phi = 0.8
        series = np.empty(n)
        series[0] = 0.0
        shocks = rng.standard_normal(n)
        for i in range(1, n):
            series[i] = phi * series[i - 1] + shocks[i]
        values = return_acf(series, n_lags=5)
        assert values[0] == pytest.approx(phi, abs=0.05)

    def test_abs_return_acf_is_acf_of_absolute_values(self) -> None:
        returns = np.array([0.01, -0.02, 0.015, -0.01, 0.02, -0.015, 0.01, -0.02] * 20)
        assert np.allclose(abs_return_acf(returns, 3), acf_of(np.abs(returns), 3))


class TestOrderFlowImbalance:
    def test_perfect_predictive_relationship_gives_correlation_near_one(self) -> None:
        n_ticks = 500
        window = 5
        rng = np.random.default_rng(5)
        signed_volume = rng.integers(-10, 11, size=n_ticks).astype(float)

        log_prices = np.zeros(n_ticks + window)
        for t in range(window, n_ticks):
            window_ofi = signed_volume[t - window : t].sum()
            log_prices[t + window] = log_prices[t] + 0.001 * window_ofi
        prices = list(np.exp(log_prices))

        samples = order_flow_imbalance(prices, signed_volume, window=window)
        corr = ofi_price_change_correlation(samples)
        assert corr == pytest.approx(1.0, abs=1e-6)

    def test_no_variation_in_ofi_gives_nan_not_a_crash(self) -> None:
        samples = order_flow_imbalance([100.0] * 20, np.zeros(20), window=3)
        assert np.isnan(ofi_price_change_correlation(samples))

    def test_compute_signed_volume_nets_buys_and_sells_per_tick(self) -> None:
        signed = compute_signed_volume(
            n_ticks=3,
            trade_ticks=[0, 0, 1],
            trade_qtys=[5, 3, 7],
            trade_sides=[Side.BUY, Side.SELL, Side.SELL],
        )
        assert list(signed) == [5 - 3, -7, 0]


class TestSimulatedTapeStylizedFacts:
    """The stylized-fact checks, measured on the real simulated tape."""

    def test_kurtosis_exceeds_three(self) -> None:
        report = _simulated_report(seed=10)
        assert report.pearson_kurtosis > 3.0, report

    def test_raw_return_acf_negative_at_lag1_and_near_zero_beyond(self) -> None:
        report = _simulated_report(seed=11)
        assert report.return_acf_lag1 < 0, report
        assert abs(report.return_acf_mean_lag2_plus) < 0.05, report

    def test_abs_return_acf_positive_at_lag1(self) -> None:
        report = _simulated_report(seed=12)
        assert report.abs_return_acf_lag1 > 0, report

    def test_ofi_correlates_positively_with_subsequent_price_change(self) -> None:
        report = _simulated_report(seed=13)
        assert report.ofi_price_change_corr > 0, report


class TestVolatilityClusteringMismatch:
    """|return| ACF decays fast, not slowly: this ZI-only model has no clustering mechanism."""

    def test_abs_return_acf_decays_away_within_a_handful_of_lags(self) -> None:
        rngs = spawn_rngs(20, N_NOISE)
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
        engine.run(N_TICKS)
        returns = log_returns([m.mark for m in engine.mark_log])
        acf_values = abs_return_acf(returns, n_lags=25)

        assert acf_values[0] > 0.08, "expected a real lag-1 volatility echo to start from"
        assert abs(acf_values[9]) < 0.05, (
            f"abs-return ACF at lag 10 is {acf_values[9]:.3f} -- decaying more slowly, "
            "re-check whether this still counts as a mismatch"
        )
