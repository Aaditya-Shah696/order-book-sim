"""Parameter estimation: rolling volatility, and the lambda(delta) = A*exp(-k*delta) fit.

Sigma is measured on the simulated tape; A and k come from fitting a probe
quoter's fill rate at several depths. ``IntensityFit.r_squared`` reports how
good the fit was.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .agents.quoting import NaiveMM
from .agents.zero_intelligence import ZeroIntelligenceAgent
from .engine import Engine
from .seeding import spawn_rngs

SESSION_HORIZON_T = 2000
"""One simulated session, in ticks.

Calibration measures over exactly this length so the fitted constants and an
Avellaneda-Stoikov agent's T share a timescale.
"""


def rolling_volatility(prices: Sequence[float], window: int) -> pd.Series:
    """Rolling stdev of tick-to-tick log returns, in per-tick units."""
    log_prices = np.log(np.asarray(prices, dtype=float))
    returns = pd.Series(np.diff(log_prices))
    return returns.rolling(window).std()


def estimate_sigma(prices: Sequence[float], window: int = 100) -> float:
    """One per-tick sigma for a mark tape: the median of the rolling estimator."""
    rolling = rolling_volatility(prices, window)
    valid = rolling.dropna()
    if valid.empty:
        raise ValueError("not enough price history to estimate volatility")
    return float(valid.median())


@dataclass(frozen=True, kw_only=True)
class DepthFillCount:
    """Raw measurement at one probed depth."""

    depth_ticks: int
    fills: int
    ticks: int

    @property
    def fill_rate(self) -> float:
        """Fills per tick: the empirical intensity lambda(depth_ticks)."""
        return self.fills / self.ticks


@dataclass(frozen=True, kw_only=True)
class IntensityFit:
    """lambda(delta) = A * exp(-k * delta), fit by log-linear regression on ln(lambda)."""

    A: float
    k: float
    r_squared: float
    depths: tuple[DepthFillCount, ...]


def measure_fill_rate_at_depth(
    *,
    depth_ticks: int,
    n_ticks: int,
    n_noise: int,
    seed: int,
    reference_price: int = 100,
    probe_qty: int = 5,
) -> DepthFillCount:
    """Run the ZI market with one added ``NaiveMM`` probe held at a fixed
    ``depth_ticks`` half-spread, and count the passive fills against it."""
    rngs = spawn_rngs(seed, n_noise)
    noise = [
        ZeroIntelligenceAgent(
            agent_id=i + 1,
            cash=1_000_000,
            inventory=1000,
            reference_price=reference_price,
            rng=rngs[i],
        )
        for i in range(n_noise)
    ]
    probe = NaiveMM(
        agent_id=0,
        cash=1_000_000,
        inventory=0,
        quote_qty=probe_qty,
        half_spread_ticks=float(depth_ticks),
    )
    engine = Engine([probe, *noise], reference_price=float(reference_price))
    engine.run(n_ticks)

    fills = sum(1 for t in engine.trade_log if t.resting_agent == probe.agent_id)
    return DepthFillCount(depth_ticks=depth_ticks, fills=fills, ticks=n_ticks)


def fit_fill_intensity(
    depths: Sequence[int],
    *,
    n_ticks: int = SESSION_HORIZON_T,
    n_noise: int = 20,
    seed: int = 0,
    reference_price: int = 100,
) -> IntensityFit:
    """Measure fill rate at each depth and fit lambda(delta) = A*exp(-k*delta).

    A zero fill rate at any depth yields a NaN fit alongside the raw counts,
    rather than a crash or a silently dropped point.
    """
    counts = tuple(
        measure_fill_rate_at_depth(
            depth_ticks=d,
            n_ticks=n_ticks,
            n_noise=n_noise,
            seed=seed,
            reference_price=reference_price,
        )
        for d in depths
    )
    rates = [c.fill_rate for c in counts]
    if any(r <= 0 for r in rates):
        return IntensityFit(A=float("nan"), k=float("nan"), r_squared=float("nan"), depths=counts)

    x = np.array([c.depth_ticks for c in counts], dtype=float)
    y = np.log(np.array(rates, dtype=float))
    result = stats.linregress(x, y)
    k = -result.slope
    A = math.exp(result.intercept)
    return IntensityFit(A=A, k=k, r_squared=result.rvalue**2, depths=counts)
