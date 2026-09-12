"""Stylized-fact measurements: return kurtosis, autocorrelation, order-flow imbalance.

Every function takes plain arrays rather than an Engine, so the same code
measures a simulated tape and a real LOBSTER tape.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats as scipy_stats
from statsmodels.tsa.stattools import acf as sm_acf

from ..types import Side


def log_returns(prices: npt.ArrayLike) -> np.ndarray:
    log_prices = np.log(np.asarray(prices, dtype=float))
    return np.diff(log_prices)


def pearson_kurtosis(returns: np.ndarray) -> float:
    """Pearson kurtosis: 3 for a normal distribution, >3 for fat tails."""
    return float(scipy_stats.kurtosis(returns, fisher=False))


def acf_of(series: np.ndarray, n_lags: int) -> np.ndarray:
    """Autocorrelation at lags 1..n_lags (lag 0, always 1 by definition, is dropped)."""
    return np.asarray(sm_acf(series, nlags=n_lags, fft=True))[1:]


def return_acf(returns: np.ndarray, n_lags: int) -> np.ndarray:
    return acf_of(returns, n_lags)


def abs_return_acf(returns: np.ndarray, n_lags: int) -> np.ndarray:
    """ACF of |returns|: the volatility-clustering stylized fact."""
    return acf_of(np.abs(returns), n_lags)


@dataclass(frozen=True, kw_only=True)
class OfiSample:
    """One (order-flow imbalance, subsequent price change) pair."""

    ofi: float
    price_change: float


def compute_signed_volume(
    n_ticks: int,
    trade_ticks: Sequence[int],
    trade_qtys: Sequence[int],
    trade_sides: Sequence[Side],
) -> np.ndarray:
    """Per-tick net signed volume: +qty per buy-aggressor fill, -qty per sell-aggressor fill.

    A signed-trade-volume proxy for depth-based OFI (Cont, Kukanov & Stoikov),
    accumulated over a trailing window by ``order_flow_imbalance``.
    """
    signed = np.zeros(n_ticks)
    for tick, qty, side in zip(trade_ticks, trade_qtys, trade_sides, strict=True):
        signed[tick] += qty if side is Side.BUY else -qty
    return signed


def order_flow_imbalance(
    marks: npt.ArrayLike, signed_volume: np.ndarray, *, window: int
) -> list[OfiSample]:
    """Pair signed volume over [t-window, t) with the log price change over [t, t+window].

    One sample per ``t`` with a full window on both sides.
    """
    log_marks = np.log(np.asarray(marks, dtype=float))
    n_ticks = len(log_marks)
    samples = []
    for t in range(window, n_ticks - window):
        ofi = float(signed_volume[t - window : t].sum())
        price_change = float(log_marks[t + window] - log_marks[t])
        samples.append(OfiSample(ofi=ofi, price_change=price_change))
    return samples


def ofi_price_change_correlation(samples: Sequence[OfiSample]) -> float:
    """Pearson correlation between OFI and the subsequent price change.

    Positive if buy pressure precedes price rises.
    """
    if len(samples) < 2:
        return float("nan")
    ofis = np.array([s.ofi for s in samples])
    changes = np.array([s.price_change for s in samples])
    if np.std(ofis) == 0 or np.std(changes) == 0:
        return float("nan")
    return float(np.corrcoef(ofis, changes)[0, 1])


@dataclass(frozen=True, kw_only=True)
class StylizedFactReport:
    """Every measured statistic for one tape, for printing as a row of a table."""

    label: str
    n_observations: int
    pearson_kurtosis: float
    return_acf_lag1: float
    return_acf_mean_lag2_plus: float
    abs_return_acf_lag1: float
    abs_return_acf_lag20: float
    ofi_price_change_corr: float


def build_report(
    label: str,
    *,
    prices: npt.ArrayLike,
    signed_volume: np.ndarray | None = None,
    acf_lags: int = 20,
    ofi_window: int = 10,
) -> StylizedFactReport:
    returns = log_returns(prices)
    r_acf = return_acf(returns, acf_lags)
    abs_acf = abs_return_acf(returns, acf_lags)

    ofi_corr = float("nan")
    if signed_volume is not None:
        samples = order_flow_imbalance(prices, signed_volume, window=ofi_window)
        ofi_corr = ofi_price_change_correlation(samples)

    return StylizedFactReport(
        label=label,
        n_observations=len(returns),
        pearson_kurtosis=pearson_kurtosis(returns),
        return_acf_lag1=float(r_acf[0]),
        return_acf_mean_lag2_plus=float(np.mean(r_acf[1:])),
        abs_return_acf_lag1=float(abs_acf[0]),
        abs_return_acf_lag20=float(abs_acf[-1]),
        ofi_price_change_corr=ofi_corr,
    )
