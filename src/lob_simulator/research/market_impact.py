"""Does Avellaneda-Stoikov's PnL-variance advantage erode as its own market share grows?

A-S is derived under an exogenous mid-price; here the mid is endogenous. The
MM's share of total order flow is varied by varying the ZI noise-agent count,
and A-S's PnL variance relative to InventorySkewMM's is correlated against it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from ..agents.quoting import QuotingAgent
from ..agents.zero_intelligence import ZeroIntelligenceAgent
from ..engine import Engine
from ..seeding import spawn_rngs


@dataclass(frozen=True, kw_only=True)
class ImpactTrial:
    """One trial's outcome. ``mm_share`` is the subject MM's share of all trades
    in the run, counting it as either aggressor or resting side."""

    n_noise: int
    seed: int
    mm_share: float
    terminal_pnl: float


def run_impact_trial(
    mm_factory: Callable[[], QuotingAgent],
    *,
    n_noise: int,
    seed: int,
    n_ticks: int,
    reference_price: int = 100,
) -> ImpactTrial:
    """One subject MM against ``n_noise`` ZI agents, for one seed.

    Terminal PnL uses the same last-two-sided-tick rule as
    ``monte_carlo.run_trial``; this variant also records ``mm_share``.
    """
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
    mm = mm_factory()
    engine = Engine([mm, *noise], reference_price=float(reference_price))

    last_two_sided_mark: float | None = None
    for _ in range(n_ticks):
        engine.step()
        if engine.book.best_bid is not None and engine.book.best_ask is not None:
            last_two_sided_mark = engine.mark
    mark_for_pnl = last_two_sided_mark if last_two_sided_mark is not None else engine.mark

    total_trades = len(engine.trade_log)
    mm_trades = sum(
        1
        for t in engine.trade_log
        if t.aggressor_agent == mm.agent_id or t.resting_agent == mm.agent_id
    )
    mm_share = mm_trades / total_trades if total_trades > 0 else float("nan")

    return ImpactTrial(
        n_noise=n_noise, seed=seed, mm_share=mm_share, terminal_pnl=mm.pnl(mark_for_pnl)
    )


def _variance_ratio(
    skew_trials: Sequence[ImpactTrial], as_trials: Sequence[ImpactTrial], indices: np.ndarray
) -> float:
    """as_variance / skew_variance over the trials ``indices`` selects.

    Above 1 means A-S has the wider PnL distribution.
    """
    skew_pnls = np.array([skew_trials[i].terminal_pnl for i in indices])
    as_pnls = np.array([as_trials[i].terminal_pnl for i in indices])
    skew_var = np.var(skew_pnls, ddof=1)
    as_var = np.var(as_pnls, ddof=1)
    return float(as_var / skew_var) if skew_var > 0 else float("nan")


@dataclass(frozen=True, kw_only=True)
class MarketImpactFinding:
    """The sweep's effect size and confidence interval.

    ``variance_ratio`` is A-S PnL variance over InventorySkew PnL variance at
    each ``n_noise`` level; ``correlation`` is the Pearson correlation between
    ``mean_mm_share`` and ``variance_ratio`` across levels. ``is_null`` is True
    iff the CI includes 0.
    """

    n_noise_values: tuple[int, ...]
    mean_mm_share: tuple[float, ...]
    variance_ratio: tuple[float, ...]
    correlation: float
    correlation_ci95: tuple[float, float]
    is_null: bool


def analyze_market_impact(
    *,
    skew_factory: Callable[[], QuotingAgent],
    as_factory: Callable[[], QuotingAgent],
    n_noise_values: Sequence[int],
    seeds: Sequence[int],
    n_ticks: int,
    reference_price: int = 100,
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 0,
) -> MarketImpactFinding:
    """Sweep the MM's share of order flow and correlate it with the PnL-variance ratio.

    Each bootstrap replicate draws one set of seed indices and applies it at
    every ``n_noise`` level, so the CI accounts for seed sampling across the
    whole sweep rather than per point.
    """
    skew_by_level = {
        n: [
            run_impact_trial(
                skew_factory, n_noise=n, seed=s, n_ticks=n_ticks, reference_price=reference_price
            )
            for s in seeds
        ]
        for n in n_noise_values
    }
    as_by_level = {
        n: [
            run_impact_trial(
                as_factory, n_noise=n, seed=s, n_ticks=n_ticks, reference_price=reference_price
            )
            for s in seeds
        ]
        for n in n_noise_values
    }

    all_indices = np.arange(len(seeds))
    mean_shares = [
        float(
            np.mean([t.mm_share for t in skew_by_level[n]] + [t.mm_share for t in as_by_level[n]])
        )
        for n in n_noise_values
    ]
    point_ratios = [
        _variance_ratio(skew_by_level[n], as_by_level[n], all_indices) for n in n_noise_values
    ]
    point_corr = float(np.corrcoef(mean_shares, point_ratios)[0, 1])

    rng = np.random.default_rng(bootstrap_seed)
    n_seeds = len(seeds)
    boot_corrs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_seeds, size=n_seeds)
        ratios = [_variance_ratio(skew_by_level[n], as_by_level[n], idx) for n in n_noise_values]
        if np.std(ratios) == 0 or any(np.isnan(r) for r in ratios):
            continue
        boot_corrs.append(np.corrcoef(mean_shares, ratios)[0, 1])
    boot_corrs_arr = np.array(boot_corrs)
    lower, upper = (
        float(np.percentile(boot_corrs_arr, 2.5)),
        float(np.percentile(boot_corrs_arr, 97.5)),
    )

    return MarketImpactFinding(
        n_noise_values=tuple(n_noise_values),
        mean_mm_share=tuple(mean_shares),
        variance_ratio=tuple(point_ratios),
        correlation=point_corr,
        correlation_ci95=(lower, upper),
        is_null=bool(lower <= 0 <= upper),
    )
