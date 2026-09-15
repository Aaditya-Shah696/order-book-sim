"""Parameter estimation: rolling volatility, and the lambda(delta) = A*exp(-k*delta) fit.

Sigma is measured on the simulated tape; A and k come from fitting a probe
quoter's fill rate at several depths. ``IntensityFit.r_squared`` reports how
good the fit was.

Units. Avellaneda-Stoikov models the mid as an arithmetic Brownian motion in
price units, so the sigma it consumes is the per-tick standard deviation of
mark *differences*, in ticks. That is ``units="price"``, the default.
``units="log"`` gives the dimensionless log-return volatility instead, which
is the conventional quantity for describing a tape but is *not* what the
A-S formulas take: feeding it in silently rescales ``gamma`` by roughly the
square of the price level.

Horizon. The one-tick sigma of a simulated mark is mostly bid-ask bounce:
noise quotes arrive and cancel around an anchor, so the mark jitters by a
few ticks every tick whether or not the anchor moves. A-S's ``sigma^2 (T-t)``
is the variance the *anchor* accumulates to the horizon, so the sigma it
should be fed is the one implied by mark changes over many ticks, where the
bounce has averaged out: ``estimate_sigma_at_horizon`` with
``DIFFUSIVE_HORIZON``. The one-tick figure is kept for reporting.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from .agents.quoting import NaiveMM
from .market import SUBJECT_AGENT_ID, MarketSpec, build_market

SESSION_HORIZON_T = 2000
"""One simulated session, in ticks.

Calibration measures over exactly this length so the fitted constants and an
Avellaneda-Stoikov agent's T share a timescale.
"""

DIFFUSIVE_HORIZON = 50
"""Ticks over which mark changes are measured for the sigma fed to A-S.

Far enough past the one-tick bounce that on the informed market the estimate
lands within about 10% of the fundamental's own per-tick std (h=10 is still
inflated); short enough that a 2000-tick session holds ~40 non-overlapping
windows, so the estimate is not itself noise. On the uninformed market,
where the anchor never moves, the estimate keeps falling with the horizon
(0.7 at h=50, 0.35 at h=200): there it measures residual bounce, not
diffusion, and the README says so.
"""

SigmaUnits = Literal["price", "log"]


def rolling_volatility(
    prices: Sequence[float], window: int, *, units: SigmaUnits = "price"
) -> pd.Series:
    """Rolling stdev of tick-to-tick changes, in per-tick units.

    ``units="price"``: changes are price differences (ticks).
    ``units="log"``: changes are log returns (dimensionless).
    """
    values = np.asarray(prices, dtype=float)
    if units == "log":
        values = np.log(values)
    returns = pd.Series(np.diff(values))
    return returns.rolling(window).std()


def estimate_sigma(
    prices: Sequence[float], window: int = 100, *, units: SigmaUnits = "price"
) -> float:
    """One per-tick sigma for a mark tape: the median of the rolling estimator."""
    rolling = rolling_volatility(prices, window, units=units)
    valid = rolling.dropna()
    if valid.empty:
        raise ValueError("not enough price history to estimate volatility")
    return float(valid.median())


def estimate_sigma_at_horizon(prices: Sequence[float], horizon: int) -> float:
    """Per-sqrt(tick) sigma implied by mark changes over ``horizon`` ticks, in price units.

    ``std(m[t+h] - m[t], ddof=1) / sqrt(h)`` over every overlapping window of
    the tape. For a mark that diffuses this equals the one-tick sigma at every
    ``h``; for one that bounces around a fixed anchor it decays toward zero as
    ``h`` grows. The ratio to the one-tick figure therefore says how much of
    the one-tick sigma is bounce rather than diffusion.

    Unlike ``estimate_sigma`` this is a whole-tape estimate, not a median of
    rolling windows, so at ``horizon=1`` the two differ slightly.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1 tick, got {horizon}")
    values = np.asarray(prices, dtype=float)
    if len(values) <= horizon + 1:
        raise ValueError(
            f"need more than {horizon + 1} prices for a {horizon}-tick horizon, got {len(values)}"
        )
    diffs = values[horizon:] - values[:-horizon]
    return float(np.std(diffs, ddof=1) / math.sqrt(horizon))


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
    spec: MarketSpec,
    seed: int,
    probe_qty: int = 5,
) -> DepthFillCount:
    """Run ``spec`` with one added ``NaiveMM`` probe held at a fixed
    ``depth_ticks`` half-spread, and count the passive fills against it."""
    probe = NaiveMM(
        agent_id=SUBJECT_AGENT_ID,
        cash=1_000_000,
        inventory=0,
        quote_qty=probe_qty,
        half_spread_ticks=float(depth_ticks),
    )
    engine = build_market(spec, seed=seed, subject=probe).engine
    engine.run(n_ticks)

    fills = sum(1 for t in engine.trade_log if t.resting_agent == probe.agent_id)
    return DepthFillCount(depth_ticks=depth_ticks, fills=fills, ticks=n_ticks)


def fit_fill_intensity(
    depths: Sequence[int],
    *,
    spec: MarketSpec,
    n_ticks: int = SESSION_HORIZON_T,
    seed: int = 0,
) -> IntensityFit:
    """Measure fill rate at each depth and fit lambda(delta) = A*exp(-k*delta).

    A zero fill rate at any depth yields a NaN fit alongside the raw counts,
    rather than a crash or a silently dropped point.
    """
    counts = tuple(
        measure_fill_rate_at_depth(depth_ticks=d, n_ticks=n_ticks, spec=spec, seed=seed)
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


CALIBRATION_DEPTHS = (1, 2, 3, 4, 5, 6, 8, 10, 12)
"""Probe depths for the intensity fit.

Stops at 12: at 15 ticks the expected fill count over a session is in the
single digits, so a zero count -- and with it a NaN fit -- is a matter of
which seed was drawn, and the point's log-scale noise would dominate the
regression anyway.
"""


@dataclass(frozen=True, kw_only=True)
class Calibration:
    """The constants an Avellaneda-Stoikov agent needs, measured on one market."""

    sigma: float
    """One-tick stdev of mark differences, in ticks (``units="price"``). Mostly
    bounce; kept for reporting. Not what A-S should be fed."""
    sigma_diffusive: float
    """Sigma implied by mark changes over ``sigma_horizon`` ticks, in ticks
    per sqrt(tick): the diffusive part of ``sigma``, and the value an
    Avellaneda-Stoikov agent's ``sigma`` should be."""
    sigma_horizon: int
    """The horizon ``sigma_diffusive`` was measured at (``DIFFUSIVE_HORIZON``)."""
    sigma_log: float
    """The same tape's one-tick log-return volatility, for reporting only."""
    A: float
    k: float
    r_squared: float
    depths: tuple[DepthFillCount, ...]
    """The raw fill counts behind ``A`` and ``k``."""


def calibrate(
    spec: MarketSpec,
    *,
    seed: int,
    n_ticks: int = SESSION_HORIZON_T,
    diffusive_horizon: int = DIFFUSIVE_HORIZON,
) -> Calibration:
    """Fit sigma (one-tick and diffusive), A and k on ``spec`` with no strategy
    present, from ``seed`` alone."""
    fit = fit_fill_intensity(CALIBRATION_DEPTHS, spec=spec, n_ticks=n_ticks, seed=seed)
    engine = build_market(spec, seed=seed).engine
    engine.run(n_ticks)
    marks = [m.mark for m in engine.mark_log]
    return Calibration(
        sigma=estimate_sigma(marks, window=100, units="price"),
        sigma_diffusive=estimate_sigma_at_horizon(marks, diffusive_horizon),
        sigma_horizon=diffusive_horizon,
        sigma_log=estimate_sigma(marks, window=100, units="log"),
        A=fit.A,
        k=fit.k,
        r_squared=fit.r_squared,
        depths=fit.depths,
    )
