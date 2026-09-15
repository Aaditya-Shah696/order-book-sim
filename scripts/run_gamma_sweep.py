"""Sweep Avellaneda-Stoikov's gamma and InventorySkewMM's skew_k on one market, same rule.

Calibrates sigma and k on CALIBRATION_SEED, then runs A-S at each gamma and
InventorySkewMM at each skew_k over SWEEP_SEEDS -- disjoint from the Monte
Carlo's trial seeds (0..499) and from the calibration seed -- with NaiveMM on
the same seeds for reference. Both parameters are picked by the same rule on
the same seeds so the later comparison between the two strategies is not a
property of which one was tuned. Writes results/gamma_sweep_<market>.json,
which scripts/run_monte_carlo.py, scripts/run_inventory_path.py and
scripts/run_market_impact.py read both chosen values from.

Usage: uv run python scripts/run_gamma_sweep.py --market {uninformed,informed} [--plot-only]

``--plot-only`` redraws the figure from the saved JSON without re-running.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import NullLocator

from lob_simulator.calibration import SESSION_HORIZON_T, calibrate
from lob_simulator.market import SPECS_BY_NAME
from lob_simulator.research.gamma_sweep import (
    DEFAULT_SKEW_KS,
    GammaSweep,
    SweepPoint,
    sweep_from_dict,
    sweep_gamma,
    sweep_to_dict,
)

matplotlib.use("Agg")

CALIBRATION_SEED = 999_999
SWEEP_SEEDS = range(10_000, 10_060)
GAMMAS = [1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 2e-5, 3e-5, 5e-5, 7e-5, 1e-4, 1.5e-4, 2e-4, 3e-4, 1e-3]
"""Denser between 1e-5 and 3e-4 than a plain 1-3-10 ladder: with the diffusive
sigma the useful reservation shifts (0.1 to 1 tick per unit at the open) fall
in that decade and a factor-3 grid would make the chosen gamma a grid artefact."""
SKEW_KS = list(DEFAULT_SKEW_KS)

OUT_DIR = Path(__file__).parent.parent / "results"

NAIVE_COLOR = "tab:red"


def _row(
    axes: Sequence[Axes],
    points: Sequence[SweepPoint],
    xs: Sequence[float],
    *,
    xlabel: str,
    chosen: float,
    name: str,
    baselines: Sequence[SweepPoint],
    explicit_ticks: bool = False,
) -> None:
    axes[0].semilogx(xs, [p.mean_pnl for p in points], marker="o", label=name)
    axes[1].semilogx(xs, [p.std_pnl for p in points], marker="o", label=name)
    axes[2].semilogx(
        xs, [p.mean_abs_inventory_midsession for p in points], marker="o",
        label=f"{name}, mid-session",
    )
    axes[2].semilogx(
        xs, [p.mean_abs_inventory_terminal for p in points], marker="s", label=f"{name}, terminal"
    )
    for b in baselines:
        axes[0].axhline(b.mean_pnl, color=NAIVE_COLOR, linestyle="--", linewidth=0.9, label=b.label)
        axes[1].axhline(b.std_pnl, color=NAIVE_COLOR, linestyle="--", linewidth=0.9, label=b.label)
        axes[2].axhline(
            b.mean_abs_inventory_terminal, color=NAIVE_COLOR, linestyle="--", linewidth=0.9,
            label=f"{b.label}, T",
        )
    for ax in axes:
        ax.axvline(chosen, color="grey", linewidth=0.8, alpha=0.7)
        ax.set_xlabel(xlabel)
        ax.legend(fontsize=8)
        if explicit_ticks:
            # A one-decade log axis prints overlapping "2x10^-2" minor labels;
            # label the swept values themselves instead.
            ax.set_xticks(list(xs))
            ax.set_xticklabels([f"{x:g}" for x in xs])
            ax.xaxis.set_minor_locator(NullLocator())
    axes[0].set_ylabel("mean terminal PnL")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("std terminal PnL (log)")
    axes[2].set_ylabel("mean |inventory|")


def plot(sweep: GammaSweep, out_png: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    _row(
        axes[0],
        sweep.points,
        [p.gamma for p in sweep.points if p.gamma is not None],
        xlabel="gamma",
        chosen=sweep.chosen_gamma,
        name="A-S",
        baselines=sweep.baselines,
    )
    _row(
        axes[1],
        sweep.skew_points,
        [p.skew_k for p in sweep.skew_points if p.skew_k is not None],
        xlabel="skew_k (ticks per unit of inventory)",
        chosen=sweep.chosen_skew_k,
        name="InventorySkewMM",
        baselines=sweep.baselines,
        explicit_ticks=True,
    )
    fig.suptitle(
        f"{sweep.market} market, N={sweep.n_seeds} seeds/point, rule: {sweep.selection_rule}\n"
        f"chosen gamma={sweep.chosen_gamma:g} (top), chosen skew_k={sweep.chosen_skew_k:g} (bottom)"
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"saved {out_png}")


def _print_table(title: str, points: Sequence[SweepPoint]) -> None:
    print(f"\n{title}")
    print(
        f"{'strategy':34s} {'mean_pnl':>9s} {'std_pnl':>8s} {'mean/std':>8s} "
        f"{'|inv| mid':>9s} {'|inv| T':>8s} {'markout':>8s}"
    )
    for p in points:
        print(
            f"{p.label:34s} {p.mean_pnl:9.0f} {p.std_pnl:8.0f} {p.mean_over_std_pnl:8.2f} "
            f"{p.mean_abs_inventory_midsession:9.1f} {p.mean_abs_inventory_terminal:8.1f} "
            f"{p.mean_markout:+8.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", choices=sorted(SPECS_BY_NAME), required=True)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    spec = SPECS_BY_NAME[args.market]
    out_json = OUT_DIR / f"gamma_sweep_{spec.name}.json"
    out_png = OUT_DIR / f"gamma_sweep_{spec.name}.png"

    if args.plot_only:
        plot(sweep_from_dict(json.loads(out_json.read_text())), out_png)
        return

    cal = calibrate(spec, seed=CALIBRATION_SEED)
    print(
        f"[{spec.name}] calibration: sigma_diffusive={cal.sigma_diffusive:.4f} ticks/tick^0.5 "
        f"at h={cal.sigma_horizon} (one-tick sigma {cal.sigma:.4f}, log-return "
        f"{cal.sigma_log:.5f}), A={cal.A:.3f}, k={cal.k:.4f}, R^2={cal.r_squared:.3f}; "
        f"A-S is fed sigma_diffusive"
    )

    t0 = time.perf_counter()
    sweep = sweep_gamma(
        spec=spec,
        gammas=GAMMAS,
        seeds=SWEEP_SEEDS,
        n_ticks=SESSION_HORIZON_T,
        sigma=cal.sigma_diffusive,
        k=cal.k,
        skew_ks=SKEW_KS,
    )
    elapsed = time.perf_counter() - t0

    _print_table("A-S by gamma (NaiveMM first, for reference)", (*sweep.baselines, *sweep.points))
    _print_table("InventorySkewMM by skew_k", sweep.skew_points)
    print(
        f"\nchosen gamma = {sweep.chosen_gamma:g}, chosen skew_k = {sweep.chosen_skew_k:g}  "
        f"by rule: {sweep.selection_rule}  ({elapsed:.0f}s)"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(sweep_to_dict(sweep), f, indent=2)
    print(f"saved {out_json}")
    plot(sweep, out_png)


if __name__ == "__main__":
    main()
