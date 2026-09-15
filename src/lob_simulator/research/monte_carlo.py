"""Monte Carlo batch runner across seeds."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats

from ..agents.quoting import QuotingAgent
from ..engine import Engine
from ..market import Market, MarketSpec, build_market
from ..types import Side

MARKOUT_HORIZON = 50


@dataclass(frozen=True, kw_only=True)
class TrialResult:
    seed: int
    terminal_pnl: float
    terminal_inventory: int
    fill_count: int
    """Trades the strategy was party to, on either side."""
    passive_units: int
    """Units filled on the strategy's own resting quotes."""
    aggressor_units: int
    """Units the strategy filled by crossing the spread. A quoter whose
    reservation shift pushes one side through the mid becomes a taker; this
    counts how much."""
    realized_pnl: float
    unrealized_pnl: float
    had_two_sided_tick: bool
    mean_markout: float
    """Mean per-unit markout of the strategy's passive fills, in ticks: the
    signed gap between the mark ``MARKOUT_HORIZON`` ticks after each fill and
    the fill price. Negative means fills were adversely selected."""
    informed_unit_share: float | None
    """Share of passive units whose aggressor was the informed trader.
    ``None`` in the uninformed market, which has no informed trader (not NaN:
    NaN would mean "had one, but no fills against it", and would also break
    trial equality)."""
    markout_vs_informed: float | None
    """``mean_markout`` restricted to passive fills against the informed
    trader. ``None`` in the uninformed market; NaN if there were none."""
    markout_vs_noise: float | None
    """``mean_markout`` restricted to passive fills against everyone else.
    ``None`` in the uninformed market, where it would equal ``mean_markout``."""


@dataclass(frozen=True, kw_only=True)
class MarkoutSplit:
    """An agent's passive fills split by who took the other side.

    Markouts are volume-weighted means, in ticks per unit, of the signed gap
    between the mark ``horizon`` ticks after each fill and the fill price;
    ``nan`` where there were no such fills. ``counterparty_*`` covers fills
    whose aggressor was in the given set; ``other_*`` covers the rest.
    """

    horizon: int
    total_units: int
    total_markout: float
    counterparty_units: int
    counterparty_markout: float
    other_units: int
    other_markout: float

    @property
    def counterparty_unit_share(self) -> float:
        """Fraction of passive units taken by the counterparty set; ``nan`` if none."""
        return self.counterparty_units / self.total_units if self.total_units else float("nan")


def _weighted_mean(total: float, qty: int) -> float:
    return total / qty if qty else float("nan")


def fill_markout_by_counterparty(
    engine: Engine,
    agent_id: int,
    *,
    counterparties: Iterable[int],
    horizon: int = MARKOUT_HORIZON,
) -> MarkoutSplit:
    """Split ``agent_id``'s passive-fill markout by whether the aggressor was in ``counterparties``.

    A fill where the agent bought at ``p`` and the mark ``horizon`` ticks later
    is ``m`` earns ``m - p`` per unit; a sale earns ``p - m``. The horizon is
    clipped at the end of the tape.
    """
    if horizon < 0:
        raise ValueError(f"horizon must be non-negative, got {horizon}")
    targets = frozenset(counterparties)
    marks = [m.mark for m in engine.mark_log]
    last = len(marks) - 1
    cp_total = 0.0
    cp_qty = 0
    other_total = 0.0
    other_qty = 0
    if marks:
        for trade, ctx in zip(engine.trade_log, engine.trade_context, strict=True):
            if trade.resting_agent != agent_id:
                continue
            later = marks[min(ctx.t + horizon, last)]
            agent_bought = ctx.aggressor_side is Side.SELL
            per_unit = (later - trade.price) if agent_bought else (trade.price - later)
            if trade.aggressor_agent in targets:
                cp_total += per_unit * trade.qty
                cp_qty += trade.qty
            else:
                other_total += per_unit * trade.qty
                other_qty += trade.qty
    return MarkoutSplit(
        horizon=horizon,
        total_units=cp_qty + other_qty,
        total_markout=_weighted_mean(cp_total + other_total, cp_qty + other_qty),
        counterparty_units=cp_qty,
        counterparty_markout=_weighted_mean(cp_total, cp_qty),
        other_units=other_qty,
        other_markout=_weighted_mean(other_total, other_qty),
    )


def mean_fill_markout(engine: Engine, agent_id: int, *, horizon: int = MARKOUT_HORIZON) -> float:
    """Volume-weighted mean markout of ``agent_id``'s passive fills, in ticks per unit.

    ``nan`` if the agent had no passive fills. The counterparty-agnostic view
    of ``fill_markout_by_counterparty``.
    """
    return fill_markout_by_counterparty(
        engine, agent_id, counterparties=(), horizon=horizon
    ).total_markout


@dataclass(frozen=True, kw_only=True)
class Session:
    """One finished run of a subject strategy, plus the mark its PnL is measured at."""

    engine: Engine
    market: Market
    mark_for_pnl: float
    had_two_sided_tick: bool


def run_session(mm: QuotingAgent, *, spec: MarketSpec, seed: int, n_ticks: int) -> Session:
    """Run ``mm`` in a fresh ``spec`` market and pick the terminal mark.

    Terminal PnL is measured at the last tick where the book was two-sided, so
    the mark is a mid rather than a stale last-trade price. If it was never
    two-sided, the final mark is used instead of discarding the trial.
    """
    market = build_market(spec, seed=seed, subject=mm)
    engine = market.engine
    last_two_sided_mark: float | None = None
    for _ in range(n_ticks):
        engine.step()
        if engine.book.best_bid is not None and engine.book.best_ask is not None:
            last_two_sided_mark = engine.mark
    return Session(
        engine=engine,
        market=market,
        mark_for_pnl=last_two_sided_mark if last_two_sided_mark is not None else engine.mark,
        had_two_sided_tick=last_two_sided_mark is not None,
    )


def run_trial(
    mm_factory: Callable[[], QuotingAgent],
    *,
    spec: MarketSpec,
    seed: int,
    n_ticks: int,
) -> TrialResult:
    """One strategy instance against a fresh ``spec`` market, for one seed."""
    mm = mm_factory()
    session = run_session(mm, spec=spec, seed=seed, n_ticks=n_ticks)
    return trial_from_session(mm, session, seed=seed)


def trial_from_session(mm: QuotingAgent, session: Session, *, seed: int) -> TrialResult:
    """The per-trial accounting for a finished ``session`` run with ``mm`` as subject.

    Split out of ``run_trial`` so research code that also needs the engine
    (for other agents' PnL, say) can run the session itself and still report
    the same fields.
    """
    engine = session.engine
    mark_for_pnl = session.mark_for_pnl

    fill_count = 0
    passive_units = 0
    aggressor_units = 0
    for trade in engine.trade_log:
        if trade.resting_agent == mm.agent_id:
            fill_count += 1
            passive_units += trade.qty
        elif trade.aggressor_agent == mm.agent_id:
            fill_count += 1
            aggressor_units += trade.qty

    informed = session.market.informed_agent_id
    split = fill_markout_by_counterparty(
        engine, mm.agent_id, counterparties=() if informed is None else (informed,)
    )
    return TrialResult(
        seed=seed,
        terminal_pnl=mm.pnl(mark_for_pnl),
        terminal_inventory=mm.inventory,
        fill_count=fill_count,
        passive_units=passive_units,
        aggressor_units=aggressor_units,
        realized_pnl=float(mm.realized_pnl()),
        unrealized_pnl=mm.unrealized_pnl(mark_for_pnl),
        had_two_sided_tick=session.had_two_sided_tick,
        mean_markout=split.total_markout,
        informed_unit_share=None if informed is None else split.counterparty_unit_share,
        markout_vs_informed=None if informed is None else split.counterparty_markout,
        markout_vs_noise=None if informed is None else split.other_markout,
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
    mean_abs_terminal_inventory: float
    std_terminal_inventory: float
    mean_fill_count: float
    mean_passive_units: float
    mean_aggressor_units: float
    mean_realized_pnl: float
    mean_unrealized_pnl: float
    mean_markout: float
    informed_unit_share: float
    """Mean over trials of the per-trial share; NaN in the uninformed market,
    where every trial's share is ``None``."""
    markout_vs_informed: float
    markout_vs_noise: float
    trials: tuple[TrialResult, ...]

    @property
    def std_pnl(self) -> float:
        return math.sqrt(self.var_pnl)

    @property
    def mean_over_std_pnl(self) -> float:
        """Per-session mean PnL divided by its standard deviation."""
        return self.mean_pnl / self.std_pnl if self.std_pnl > 0 else float("nan")


def _nanmean(values: Iterable[float | None]) -> float:
    """Mean over the finite entries, skipping ``None``; ``nan`` (without a
    warning) if there are none."""
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(finite.mean()) if finite.size else float("nan")


def summarize(strategy_name: str, trials: Sequence[TrialResult], *, n_bootstrap: int = 2000,
              bootstrap_seed: int = 0) -> MonteCarloSummary:
    pnls = np.array([t.terminal_pnl for t in trials])
    inventories = np.array([t.terminal_inventory for t in trials], dtype=float)
    two_sided_fraction = sum(t.had_two_sided_tick for t in trials) / len(trials)
    return MonteCarloSummary(
        strategy_name=strategy_name,
        n_trials=len(trials),
        mean_pnl=float(np.mean(pnls)),
        var_pnl=float(np.var(pnls, ddof=1)) if len(pnls) > 1 else 0.0,
        skew_pnl=float(scipy_stats.skew(pnls)),
        p5_pnl=float(np.percentile(pnls, 5)),
        p95_pnl=float(np.percentile(pnls, 95)),
        mean_pnl_ci95=bootstrap_mean_ci(pnls, n_bootstrap=n_bootstrap, seed=bootstrap_seed),
        two_sided_tick_fraction=two_sided_fraction,
        mean_abs_terminal_inventory=float(np.mean(np.abs(inventories))),
        std_terminal_inventory=(
            float(np.std(inventories, ddof=1)) if len(inventories) > 1 else 0.0
        ),
        mean_fill_count=float(np.mean([t.fill_count for t in trials])),
        mean_passive_units=float(np.mean([t.passive_units for t in trials])),
        mean_aggressor_units=float(np.mean([t.aggressor_units for t in trials])),
        mean_realized_pnl=float(np.mean([t.realized_pnl for t in trials])),
        mean_unrealized_pnl=float(np.mean([t.unrealized_pnl for t in trials])),
        mean_markout=_nanmean(t.mean_markout for t in trials),
        informed_unit_share=_nanmean(t.informed_unit_share for t in trials),
        markout_vs_informed=_nanmean(t.markout_vs_informed for t in trials),
        markout_vs_noise=_nanmean(t.markout_vs_noise for t in trials),
        trials=tuple(trials),
    )


def run_monte_carlo(
    mm_factory: Callable[[], QuotingAgent],
    *,
    strategy_name: str,
    spec: MarketSpec,
    seeds: Sequence[int],
    n_ticks: int,
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 0,
) -> MonteCarloSummary:
    """Run one strategy across many seeds; summarize terminal PnL with a bootstrapped CI."""
    trials = [run_trial(mm_factory, spec=spec, seed=s, n_ticks=n_ticks) for s in seeds]
    return summarize(
        strategy_name, trials, n_bootstrap=n_bootstrap, bootstrap_seed=bootstrap_seed
    )
