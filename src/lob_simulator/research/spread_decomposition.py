"""Splitting A-S's gap to the constant-skew heuristic into its spread and its skew schedule.

Avellaneda-Stoikov differs from ``InventorySkewMM`` in two places, not one:
its reservation skew decays linearly to zero at the horizon, and its spread
carries the same ``gamma * sigma^2 * (T - t)`` term on top of the
``(2/gamma) ln(1 + gamma/k)`` floor. After outward tick snapping the floor
alone (half 1.83 on the informed market) gives NaiveMM's quotes, but the
variance term does not: at the chosen gamma the half-spread is 2.14 at the
open and only drops below 2.0 around tick 900, so for the first half of the
session A-S quotes a tick wider on each side at every integer mark.

Two A-S variants isolate the two differences on matched seeds:

- ``AvellanedaStoikovFixedSpreadMM``: A-S's reservation skew, with the
  half-spread pinned to a caller-supplied constant (NaiveMM's 2.0 in the
  study). Whatever gap remains to the heuristic is the skew schedule.
- ``AvellanedaStoikovFlatSpreadMM``: A-S's reservation skew, with the
  variance term dropped from the spread and the floor kept. The spread
  the paper's formula gives as ``t -> T``, held for the whole session.

``decompose`` runs the heuristic, published A-S and both variants on the
same seeds and reports each arm's summary plus paired differences against
the heuristic, since matched seeds make the paired test the right one.
It also records the informed trader's and the noise agents' aggregate PnL
per session, so the reader can see who pays the market maker.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats

from ..agent import Snapshot
from ..agents.quoting import AvellanedaStoikovMM, InventorySkewMM, QuotingAgent
from ..market import SUBJECT_AGENT_ID, MarketSpec
from .monte_carlo import bootstrap_mean_ci, run_session, trial_from_session


class AvellanedaStoikovFixedSpreadMM(AvellanedaStoikovMM):
    """A-S reservation price with a constant half-spread supplied by the caller."""

    def __init__(
        self,
        *,
        agent_id: int,
        cash: int,
        inventory: int,
        quote_qty: int,
        gamma: float,
        sigma: float,
        k: float,
        horizon_t: float,
        half_spread_ticks: float,
    ) -> None:
        if half_spread_ticks <= 0:
            raise ValueError(f"half_spread_ticks must be positive, got {half_spread_ticks}")
        super().__init__(
            agent_id=agent_id,
            cash=cash,
            inventory=inventory,
            quote_qty=quote_qty,
            gamma=gamma,
            sigma=sigma,
            k=k,
            horizon_t=horizon_t,
        )
        self.half_spread_ticks = half_spread_ticks

    def quote(self, snapshot: Snapshot) -> tuple[float, float]:
        reservation = snapshot.mark - self.inventory * self.skew_per_unit(snapshot.t)
        return reservation, self.half_spread_ticks


class AvellanedaStoikovFlatSpreadMM(AvellanedaStoikovMM):
    """A-S reservation price with the variance term dropped from the spread.

    Spread is ``(2/gamma) ln(1 + gamma/k)`` at every ``t``: the value the
    full formula reaches at the horizon.
    """

    def floor_half_spread(self) -> float:
        return (2.0 / self.gamma) * math.log(1.0 + self.gamma / self.k) / 2.0

    def quote(self, snapshot: Snapshot) -> tuple[float, float]:
        reservation = snapshot.mark - self.inventory * self.skew_per_unit(snapshot.t)
        return reservation, self.floor_half_spread()


@dataclass(frozen=True, kw_only=True)
class ArmResult:
    """One strategy's per-seed outcomes on the decomposition seeds."""

    label: str
    seeds: tuple[int, ...]
    terminal_pnl: np.ndarray
    terminal_inventory: np.ndarray
    passive_units: np.ndarray
    aggressor_units: np.ndarray
    mean_markout: np.ndarray
    """Per-seed mean per-unit markout of the subject's passive fills."""
    informed_pnl: np.ndarray
    """The informed trader's terminal PnL per seed; NaN where there is none."""
    noise_pnl: np.ndarray
    """The noise agents' aggregate terminal PnL per seed."""

    @property
    def mean_pnl(self) -> float:
        return float(np.mean(self.terminal_pnl))

    @property
    def std_pnl(self) -> float:
        return float(np.std(self.terminal_pnl, ddof=1)) if len(self.seeds) > 1 else 0.0

    @property
    def mean_over_std_pnl(self) -> float:
        return self.mean_pnl / self.std_pnl if self.std_pnl > 0 else float("nan")


@dataclass(frozen=True, kw_only=True)
class PairedDifference:
    """``arm - reference`` in terminal PnL, seed by seed."""

    label: str
    reference: str
    mean_diff: float
    std_diff: float
    ci95: tuple[float, float]
    """Percentile bootstrap of the mean paired difference."""
    t_stat: float
    p_value: float
    """Two-sided paired t-test that the mean difference is zero."""


@dataclass(frozen=True, kw_only=True)
class Decomposition:
    market: str
    gamma: float
    sigma: float
    k: float
    skew_k: float
    baseline_half_spread_ticks: float
    n_seeds: int
    n_ticks: int
    arms: tuple[ArmResult, ...]
    paired: tuple[PairedDifference, ...]
    """Every arm but the reference, against the reference (the heuristic)."""


def run_arm(
    mm_factory: Callable[[], QuotingAgent],
    *,
    label: str,
    spec: MarketSpec,
    seeds: Sequence[int],
    n_ticks: int,
) -> ArmResult:
    """Run ``mm_factory`` once per seed and keep the subject's trial fields plus
    the other side's PnL."""
    n = len(seeds)
    pnl = np.zeros(n)
    inventory = np.zeros(n, dtype=int)
    passive = np.zeros(n, dtype=int)
    aggressor = np.zeros(n, dtype=int)
    markout = np.zeros(n)
    informed_pnl = np.full(n, float("nan"))
    noise_pnl = np.zeros(n)
    for i, seed in enumerate(seeds):
        mm = mm_factory()
        session = run_session(mm, spec=spec, seed=seed, n_ticks=n_ticks)
        trial = trial_from_session(mm, session, seed=seed)
        pnl[i] = trial.terminal_pnl
        inventory[i] = trial.terminal_inventory
        passive[i] = trial.passive_units
        aggressor[i] = trial.aggressor_units
        markout[i] = trial.mean_markout
        informed_id = session.market.informed_agent_id
        mark = session.mark_for_pnl
        noise_total = 0.0
        for agent in session.engine.agents:
            if agent.agent_id == mm.agent_id:
                continue
            if agent.agent_id == informed_id:
                informed_pnl[i] = agent.pnl(mark)
            else:
                noise_total += agent.pnl(mark)
        noise_pnl[i] = noise_total
    return ArmResult(
        label=label,
        seeds=tuple(seeds),
        terminal_pnl=pnl,
        terminal_inventory=inventory,
        passive_units=passive,
        aggressor_units=aggressor,
        mean_markout=markout,
        informed_pnl=informed_pnl,
        noise_pnl=noise_pnl,
    )


def paired_difference(
    arm: ArmResult, reference: ArmResult, *, n_bootstrap: int = 2000, bootstrap_seed: int = 0
) -> PairedDifference:
    if arm.seeds != reference.seeds:
        raise ValueError(
            f"{arm.label!r} and {reference.label!r} were not run on the same seeds"
        )
    diff = arm.terminal_pnl - reference.terminal_pnl
    if len(diff) < 2:
        raise ValueError("need at least two seeds for a paired difference")
    t_stat, p_value = scipy_stats.ttest_1samp(diff, 0.0)
    return PairedDifference(
        label=arm.label,
        reference=reference.label,
        mean_diff=float(np.mean(diff)),
        std_diff=float(np.std(diff, ddof=1)),
        ci95=bootstrap_mean_ci(diff, n_bootstrap=n_bootstrap, seed=bootstrap_seed),
        t_stat=float(t_stat),
        p_value=float(p_value),
    )


SKEW_LABEL = "InventorySkewMM"
AS_LABEL = "AvellanedaStoikovMM"
AS_FIXED_SPREAD_LABEL = "AvellanedaStoikovMM, half-spread fixed"
AS_FLAT_SPREAD_LABEL = "AvellanedaStoikovMM, variance term dropped from spread"


def decompose(
    *,
    spec: MarketSpec,
    seeds: Sequence[int],
    n_ticks: int,
    gamma: float,
    sigma: float,
    k: float,
    skew_k: float,
    quote_qty: int = 5,
    baseline_half_spread_ticks: float = 2.0,
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 0,
) -> Decomposition:
    """Heuristic, published A-S and the two A-S variants on the same seeds,
    with every A-S arm paired against the heuristic."""
    if not seeds:
        raise ValueError("seeds must be non-empty")
    cash, inventory, horizon_t = 1_000_000, 0, float(n_ticks)

    def skew() -> QuotingAgent:
        return InventorySkewMM(
            agent_id=SUBJECT_AGENT_ID,
            cash=cash,
            inventory=inventory,
            quote_qty=quote_qty,
            half_spread_ticks=baseline_half_spread_ticks,
            skew_k=skew_k,
        )

    def as_full() -> QuotingAgent:
        return AvellanedaStoikovMM(
            agent_id=SUBJECT_AGENT_ID,
            cash=cash,
            inventory=inventory,
            quote_qty=quote_qty,
            gamma=gamma,
            sigma=sigma,
            k=k,
            horizon_t=horizon_t,
        )

    def as_fixed_spread() -> QuotingAgent:
        return AvellanedaStoikovFixedSpreadMM(
            agent_id=SUBJECT_AGENT_ID,
            cash=cash,
            inventory=inventory,
            quote_qty=quote_qty,
            gamma=gamma,
            sigma=sigma,
            k=k,
            horizon_t=horizon_t,
            half_spread_ticks=baseline_half_spread_ticks,
        )

    def as_flat_spread() -> QuotingAgent:
        return AvellanedaStoikovFlatSpreadMM(
            agent_id=SUBJECT_AGENT_ID,
            cash=cash,
            inventory=inventory,
            quote_qty=quote_qty,
            gamma=gamma,
            sigma=sigma,
            k=k,
            horizon_t=horizon_t,
        )

    factories: dict[str, Callable[[], QuotingAgent]] = {
        SKEW_LABEL: skew,
        AS_LABEL: as_full,
        AS_FIXED_SPREAD_LABEL: as_fixed_spread,
        AS_FLAT_SPREAD_LABEL: as_flat_spread,
    }
    arms = tuple(
        run_arm(factory, label=label, spec=spec, seeds=seeds, n_ticks=n_ticks)
        for label, factory in factories.items()
    )
    reference = arms[0]
    paired = tuple(
        paired_difference(arm, reference, n_bootstrap=n_bootstrap, bootstrap_seed=bootstrap_seed)
        for arm in arms[1:]
    )
    return Decomposition(
        market=spec.name,
        gamma=gamma,
        sigma=sigma,
        k=k,
        skew_k=skew_k,
        baseline_half_spread_ticks=baseline_half_spread_ticks,
        n_seeds=len(seeds),
        n_ticks=n_ticks,
        arms=arms,
        paired=paired,
    )


def _finite(x: float) -> float | None:
    return x if math.isfinite(x) else None


def _arm_to_dict(arm: ArmResult) -> dict[str, object]:
    return {
        "label": arm.label,
        "mean_pnl": arm.mean_pnl,
        "std_pnl": arm.std_pnl,
        "mean_over_std_pnl": _finite(arm.mean_over_std_pnl),
        "mean_abs_terminal_inventory": float(np.mean(np.abs(arm.terminal_inventory))),
        "mean_passive_units": float(np.mean(arm.passive_units)),
        "mean_aggressor_units": float(np.mean(arm.aggressor_units)),
        "mean_markout_50": _finite(float(np.nanmean(arm.mean_markout))),
        "mean_informed_pnl": _finite(float(np.nanmean(arm.informed_pnl)))
        if np.isfinite(arm.informed_pnl).any()
        else None,
        "mean_noise_pnl": float(np.mean(arm.noise_pnl)),
        "per_seed": {
            "seed": list(arm.seeds),
            "terminal_pnl": [float(v) for v in arm.terminal_pnl],
            "terminal_inventory": [int(v) for v in arm.terminal_inventory],
            "passive_units": [int(v) for v in arm.passive_units],
            "aggressor_units": [int(v) for v in arm.aggressor_units],
        },
    }


def decomposition_to_dict(d: Decomposition) -> dict[str, object]:
    """JSON-ready form, as ``scripts/run_spread_decomposition.py`` writes it.

    NaN becomes ``null`` so the file stays standard JSON.
    """
    return {
        "market": d.market,
        "gamma": d.gamma,
        "sigma": d.sigma,
        "sigma_is": "diffusive (calibration.sigma_diffusive)",
        "k": d.k,
        "skew_k": d.skew_k,
        "baseline_half_spread_ticks": d.baseline_half_spread_ticks,
        "n_seeds": d.n_seeds,
        "n_ticks": d.n_ticks,
        "seeds": list(d.arms[0].seeds),
        "arms": [_arm_to_dict(a) for a in d.arms],
        "paired_vs_reference": [
            {
                "label": p.label,
                "reference": p.reference,
                "mean_diff": p.mean_diff,
                "std_diff": p.std_diff,
                "ci95": list(p.ci95),
                "t_stat": p.t_stat,
                "p_value": p.p_value,
            }
            for p in d.paired
        ],
    }
