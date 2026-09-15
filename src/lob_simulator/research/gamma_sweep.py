"""Choosing Avellaneda-Stoikov's gamma and InventorySkewMM's skew_k by one rule, on one market.

``gamma`` is the one A-S input that is not measured, and ``skew_k`` is the
heuristic's only parameter. Both are swept here over the same seeds, disjoint
from any comparison they are later used in, recording mean PnL, PnL
dispersion and inventory both mid-session and at the horizon, and both are
picked by ``SELECTION_RULE``. Tuning one strategy and hand-picking the other
would make any comparison between them a property of the tuning, not of the
strategies; sweeping both by the same rule on the same seeds is what makes
the comparison fair. ``NaiveMM`` runs on the same seeds as the one fixed
reference.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..agents.quoting import AvellanedaStoikovMM, InventorySkewMM, NaiveMM
from ..market import SUBJECT_AGENT_ID, MarketSpec
from .inventory_path import InventoryPaths, run_inventory_paths

SELECTION_RULE = "max mean(terminal PnL) / std(terminal PnL)"

DEFAULT_SKEW_KS: tuple[float, ...] = (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)
"""Constant reservation shifts, in ticks per unit of inventory, to sweep for
``InventorySkewMM``. Spans from "barely leaning" (0.02) to "one fill of 5
units moves the quotes 2.5 ticks" (0.5), past which the agent's own quotes
cross the mid after a handful of fills and mean markout goes negative."""


@dataclass(frozen=True, kw_only=True)
class SweepPoint:
    """One strategy's outcome on the sweep seeds.

    Exactly one of ``gamma`` and ``skew_k`` is set for a swept point (A-S and
    InventorySkewMM respectively); both are ``None`` for the fixed baseline.
    """

    label: str
    gamma: float | None
    skew_k: float | None = None
    mean_pnl: float
    std_pnl: float
    mean_abs_inventory_midsession: float
    """Mean |inventory| over the middle half of the session."""
    mean_abs_inventory_terminal: float
    mean_markout: float

    @property
    def mean_over_std_pnl(self) -> float:
        return self.mean_pnl / self.std_pnl if self.std_pnl > 0 else float("nan")


def _point(
    label: str, paths: InventoryPaths, *, gamma: float | None = None, skew_k: float | None = None
) -> SweepPoint:
    n = paths.n_ticks
    return SweepPoint(
        label=label,
        gamma=gamma,
        skew_k=skew_k,
        mean_pnl=float(np.mean(paths.terminal_pnl)),
        std_pnl=float(np.std(paths.terminal_pnl, ddof=1)) if len(paths.seeds) > 1 else 0.0,
        mean_abs_inventory_midsession=paths.mean_abs_inventory_over(n // 4, 3 * n // 4),
        mean_abs_inventory_terminal=paths.mean_abs_inventory_at(n - 1),
        mean_markout=float(np.nanmean(paths.mean_markout)),
    )


@dataclass(frozen=True, kw_only=True)
class GammaSweep:
    """Both sweeps on one market, with the value each picked by ``selection_rule``."""

    market: str
    sigma: float
    """The sigma the A-S points were built with, in ticks per sqrt(tick). Since
    the 2026-09-14 fixes this is the diffusive (multi-tick) estimate, not the
    one-tick one; ``results/calibration.json`` carries both."""
    k: float
    n_seeds: int
    n_ticks: int
    points: tuple[SweepPoint, ...]
    """A-S, one per gamma."""
    skew_points: tuple[SweepPoint, ...]
    """InventorySkewMM, one per skew_k."""
    baselines: tuple[SweepPoint, ...]
    """The fixed reference(s): NaiveMM."""
    chosen_gamma: float
    chosen_skew_k: float
    selection_rule: str = SELECTION_RULE


def _choose(points: Sequence[SweepPoint]) -> SweepPoint:
    """The point ``SELECTION_RULE`` picks: argmax of mean/std, NaNs ignored."""
    scores = [p.mean_over_std_pnl for p in points]
    return points[int(np.nanargmax(scores))]


def sweep_gamma(
    *,
    spec: MarketSpec,
    gammas: Sequence[float],
    seeds: Sequence[int],
    n_ticks: int,
    sigma: float,
    k: float,
    skew_ks: Sequence[float] = DEFAULT_SKEW_KS,
    quote_qty: int = 5,
    baseline_half_spread_ticks: float = 2.0,
) -> GammaSweep:
    if not gammas:
        raise ValueError("gammas must be non-empty")
    if any(g <= 0 for g in gammas):
        raise ValueError(f"every gamma must be positive, got {list(gammas)}")
    if not skew_ks:
        raise ValueError("skew_ks must be non-empty")
    if any(s < 0 for s in skew_ks):
        raise ValueError(f"every skew_k must be non-negative, got {list(skew_ks)}")

    def as_factory(gamma: float) -> Callable[[], AvellanedaStoikovMM]:
        def make() -> AvellanedaStoikovMM:
            return AvellanedaStoikovMM(
                agent_id=SUBJECT_AGENT_ID,
                cash=1_000_000,
                inventory=0,
                quote_qty=quote_qty,
                gamma=gamma,
                sigma=sigma,
                k=k,
                horizon_t=float(n_ticks),
            )

        return make

    def skew_factory(skew_k: float) -> Callable[[], InventorySkewMM]:
        def make() -> InventorySkewMM:
            return InventorySkewMM(
                agent_id=SUBJECT_AGENT_ID,
                cash=1_000_000,
                inventory=0,
                quote_qty=quote_qty,
                half_spread_ticks=baseline_half_spread_ticks,
                skew_k=skew_k,
            )

        return make

    points = []
    for gamma in gammas:
        label = f"AvellanedaStoikovMM(gamma={gamma:g})"
        paths = run_inventory_paths(
            as_factory(gamma), strategy_name=label, spec=spec, seeds=seeds, n_ticks=n_ticks
        )
        points.append(_point(label, paths, gamma=gamma))

    skew_points = []
    for skew_k in skew_ks:
        label = f"InventorySkewMM(skew_k={skew_k:g})"
        paths = run_inventory_paths(
            skew_factory(skew_k), strategy_name=label, spec=spec, seeds=seeds, n_ticks=n_ticks
        )
        skew_points.append(_point(label, paths, skew_k=skew_k))

    baselines = [
        _point(
            "NaiveMM",
            run_inventory_paths(
                lambda: NaiveMM(
                    agent_id=SUBJECT_AGENT_ID,
                    cash=1_000_000,
                    inventory=0,
                    quote_qty=quote_qty,
                    half_spread_ticks=baseline_half_spread_ticks,
                ),
                strategy_name="NaiveMM",
                spec=spec,
                seeds=seeds,
                n_ticks=n_ticks,
            ),
        ),
    ]

    chosen_gamma = _choose(points).gamma
    chosen_skew_k = _choose(skew_points).skew_k
    assert chosen_gamma is not None
    assert chosen_skew_k is not None
    return GammaSweep(
        market=spec.name,
        sigma=sigma,
        k=k,
        n_seeds=len(seeds),
        n_ticks=n_ticks,
        points=tuple(points),
        skew_points=tuple(skew_points),
        baselines=tuple(baselines),
        chosen_gamma=chosen_gamma,
        chosen_skew_k=chosen_skew_k,
    )


def _point_to_dict(p: SweepPoint, *, sigma: float, n_ticks: int) -> dict[str, object]:
    d: dict[str, object] = {
        "label": p.label,
        "gamma": p.gamma,
        "skew_k": p.skew_k,
        "mean_pnl": p.mean_pnl,
        "std_pnl": p.std_pnl,
        "mean_over_std_pnl": p.mean_over_std_pnl,
        "mean_abs_inventory_midsession": p.mean_abs_inventory_midsession,
        "mean_abs_inventory_terminal": p.mean_abs_inventory_terminal,
        "mean_markout": p.mean_markout,
    }
    if p.gamma is not None:
        # Derived, for reading the two sweeps in one unit: A-S's reservation
        # shift per unit of inventory at the open, gamma * sigma^2 * T, which
        # decays linearly to zero at T. Its time average is half this.
        d["skew_per_unit_open"] = p.gamma * sigma**2 * n_ticks
    return d


def sweep_to_dict(sweep: GammaSweep) -> dict[str, object]:
    """JSON-ready form of a sweep, as ``scripts/run_gamma_sweep.py`` writes it."""

    def points(ps: Sequence[SweepPoint]) -> list[dict[str, object]]:
        return [_point_to_dict(p, sigma=sweep.sigma, n_ticks=sweep.n_ticks) for p in ps]

    return {
        "market": sweep.market,
        "sigma": sweep.sigma,
        "k": sweep.k,
        "n_seeds": sweep.n_seeds,
        "n_ticks": sweep.n_ticks,
        "selection_rule": sweep.selection_rule,
        "chosen_gamma": sweep.chosen_gamma,
        "chosen_skew_k": sweep.chosen_skew_k,
        "points": points(sweep.points),
        "skew_points": points(sweep.skew_points),
        "baselines": points(sweep.baselines),
    }


def _num(d: dict[str, object], key: str) -> float:
    value = d[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{key!r} must be a number, got {type(value).__name__}")
    return float(value)


def _optional_num(d: dict[str, object], key: str) -> float | None:
    return None if d.get(key) is None else _num(d, key)


def _point_from_dict(d: dict[str, object]) -> SweepPoint:
    return SweepPoint(
        label=str(d["label"]),
        gamma=_optional_num(d, "gamma"),
        skew_k=_optional_num(d, "skew_k"),
        mean_pnl=_num(d, "mean_pnl"),
        std_pnl=_num(d, "std_pnl"),
        mean_abs_inventory_midsession=_num(d, "mean_abs_inventory_midsession"),
        mean_abs_inventory_terminal=_num(d, "mean_abs_inventory_terminal"),
        mean_markout=_num(d, "mean_markout"),
    )


def _point_list(d: dict[str, object], key: str) -> tuple[SweepPoint, ...]:
    value = d[key]
    if not isinstance(value, list):
        raise TypeError(f"{key!r} must be a list")
    return tuple(_point_from_dict(p) for p in value)


def sweep_from_dict(d: dict[str, object]) -> GammaSweep:
    """Inverse of ``sweep_to_dict``."""
    return GammaSweep(
        market=str(d["market"]),
        sigma=_num(d, "sigma"),
        k=_num(d, "k"),
        n_seeds=int(_num(d, "n_seeds")),
        n_ticks=int(_num(d, "n_ticks")),
        points=_point_list(d, "points"),
        skew_points=_point_list(d, "skew_points"),
        baselines=_point_list(d, "baselines"),
        chosen_gamma=_num(d, "chosen_gamma"),
        chosen_skew_k=_num(d, "chosen_skew_k"),
        selection_rule=str(d["selection_rule"]),
    )


def _load_chosen(path: Path, key: str) -> float:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run scripts/run_gamma_sweep.py for this market first"
        )
    with open(path) as f:
        data = json.load(f)
    if key not in data:
        raise KeyError(
            f"{path} has no {key!r}: it predates the skew_k sweep -- "
            "re-run scripts/run_gamma_sweep.py for this market"
        )
    return float(data[key])


def load_chosen_gamma(path: Path) -> float:
    """Read ``chosen_gamma`` back from a sweep JSON; fails loudly if it is not there."""
    return _load_chosen(path, "chosen_gamma")


def load_chosen_skew_k(path: Path) -> float:
    """Read ``chosen_skew_k`` back from a sweep JSON; fails loudly if it is not there."""
    return _load_chosen(path, "chosen_skew_k")
