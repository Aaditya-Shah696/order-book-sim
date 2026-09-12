"""Monte Carlo batch runner across seeds."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats

from ..agents.quoting import QuotingAgent
from ..agents.zero_intelligence import ZeroIntelligenceAgent
from ..engine import Engine
from ..seeding import spawn_rngs


@dataclass(frozen=True, kw_only=True)
class TrialResult:
    seed: int
    terminal_pnl: float
    terminal_inventory: int
    fill_count: int
    realized_pnl: float
    unrealized_pnl: float
    had_two_sided_tick: bool


def run_trial(
    mm_factory: Callable[[], QuotingAgent],
    *,
    seed: int,
    n_ticks: int,
    n_noise: int,
    reference_price: int = 100,
) -> TrialResult:
    """One strategy instance against fresh ZI noise flow, for one seed.

    Terminal PnL is measured at the last tick where the book was two-sided, so
    the mark is a mid rather than a stale last-trade price. If it was never
    two-sided, the final mark is used instead of discarding the trial.
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

    had_two_sided_tick = last_two_sided_mark is not None
    mark_for_pnl = last_two_sided_mark if last_two_sided_mark is not None else engine.mark

    fill_count = sum(
        1
        for t in engine.trade_log
        if t.aggressor_agent == mm.agent_id or t.resting_agent == mm.agent_id
    )

    return TrialResult(
        seed=seed,
        terminal_pnl=mm.pnl(mark_for_pnl),
        terminal_inventory=mm.inventory,
        fill_count=fill_count,
        realized_pnl=float(mm.realized_pnl()),
        unrealized_pnl=mm.unrealized_pnl(mark_for_pnl),
        had_two_sided_tick=had_two_sided_tick,
    )


def bootstrap_mean_ci(
    values: np.ndarray, *, n_bootstrap: int = 2000, seed: int = 0, confidence: float = 0.95
) -> tuple[float, float]:
    """Percentile-bootstrap confidence interval for the mean of ``values``."""
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.array(
        [rng.choice(values, size=n, replace=True).mean() for _ in range(n_bootstrap)]
    )
    alpha = (1 - confidence) / 2
    lower = float(np.percentile(boot_means, 100 * alpha))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha)))
    return lower, upper


@dataclass(frozen=True, kw_only=True)
class MonteCarloSummary:
    strategy_name: str
    n_trials: int
    mean_pnl: float
    var_pnl: float
    skew_pnl: float
    p5_pnl: float
    p95_pnl: float
    mean_pnl_ci95: tuple[float, float]
    two_sided_tick_fraction: float
    trials: tuple[TrialResult, ...]


def run_monte_carlo(
    mm_factory: Callable[[], QuotingAgent],
    *,
    strategy_name: str,
    seeds: Sequence[int],
    n_ticks: int,
    n_noise: int,
    reference_price: int = 100,
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 0,
) -> MonteCarloSummary:
    """Run one strategy across many seeds; summarize terminal PnL with a bootstrapped CI."""
    trials = tuple(
        run_trial(
            mm_factory,
            seed=s,
            n_ticks=n_ticks,
            n_noise=n_noise,
            reference_price=reference_price,
        )
        for s in seeds
    )
    pnls = np.array([t.terminal_pnl for t in trials])
    two_sided_fraction = sum(t.had_two_sided_tick for t in trials) / len(trials)

    return MonteCarloSummary(
        strategy_name=strategy_name,
        n_trials=len(trials),
        mean_pnl=float(np.mean(pnls)),
        var_pnl=float(np.var(pnls, ddof=1)),
        skew_pnl=float(scipy_stats.skew(pnls)),
        p5_pnl=float(np.percentile(pnls, 5)),
        p95_pnl=float(np.percentile(pnls, 95)),
        mean_pnl_ci95=bootstrap_mean_ci(pnls, n_bootstrap=n_bootstrap, seed=bootstrap_seed),
        two_sided_tick_fraction=two_sided_fraction,
        trials=trials,
    )
