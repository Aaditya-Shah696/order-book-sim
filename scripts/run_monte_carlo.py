"""Monte Carlo comparison: NaiveMM vs InventorySkewMM vs AvellanedaStoikovMM.

500 seeds per strategy, matched across strategies, on one market. Calibrates
sigma and k fresh on a seed disjoint from the trial seeds; gamma and skew_k
both come from results/gamma_sweep_<market>.json (run
scripts/run_gamma_sweep.py first), so both tuned strategies were tuned the
same way on the same held-out seeds.

Usage: uv run python scripts/run_monte_carlo.py --market {uninformed,informed}
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from lob_simulator.agents.quoting import (
    AvellanedaStoikovMM,
    InventorySkewMM,
    NaiveMM,
    QuotingAgent,
)
from lob_simulator.calibration import SESSION_HORIZON_T, calibrate
from lob_simulator.market import SPECS_BY_NAME, SUBJECT_AGENT_ID
from lob_simulator.research.gamma_sweep import load_chosen_gamma, load_chosen_skew_k
from lob_simulator.research.monte_carlo import MonteCarloSummary, run_monte_carlo

matplotlib.use("Agg")

N_TRIALS = 500
CALIBRATION_SEED = 999_999

OUT_DIR = Path(__file__).parent.parent / "results"


def _json_num(x: float) -> float | None:
    """``None`` for NaN/inf so the JSON stays standard (``json.dump`` would write ``NaN``)."""
    return x if math.isfinite(x) else None


def summary_to_dict(s: MonteCarloSummary) -> dict[str, object]:
    return {
        "strategy_name": s.strategy_name,
        "n_trials": s.n_trials,
        "mean_pnl": s.mean_pnl,
        "var_pnl": s.var_pnl,
        "std_pnl": s.std_pnl,
        "mean_over_std_pnl": s.mean_over_std_pnl,
        "skew_pnl": s.skew_pnl,
        "p5_pnl": s.p5_pnl,
        "p95_pnl": s.p95_pnl,
        "mean_pnl_ci95": list(s.mean_pnl_ci95),
        "two_sided_tick_fraction": s.two_sided_tick_fraction,
        "mean_abs_terminal_inventory": s.mean_abs_terminal_inventory,
        "std_terminal_inventory": s.std_terminal_inventory,
        "mean_fill_count": s.mean_fill_count,
        "mean_passive_units": s.mean_passive_units,
        "mean_aggressor_units": s.mean_aggressor_units,
        "mean_realized_pnl": s.mean_realized_pnl,
        "mean_unrealized_pnl": s.mean_unrealized_pnl,
        "mean_markout_50": _json_num(s.mean_markout),
        "informed_unit_share": _json_num(s.informed_unit_share),
        "markout_50_vs_informed": _json_num(s.markout_vs_informed),
        "markout_50_vs_noise": _json_num(s.markout_vs_noise),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", choices=sorted(SPECS_BY_NAME), required=True)
    args = parser.parse_args()
    spec = SPECS_BY_NAME[args.market]

    sweep_json = OUT_DIR / f"gamma_sweep_{spec.name}.json"
    gamma = load_chosen_gamma(sweep_json)
    skew_k = load_chosen_skew_k(sweep_json)
    cal = calibrate(spec, seed=CALIBRATION_SEED)
    print(
        f"[{spec.name}] calibration: sigma_diffusive={cal.sigma_diffusive:.4f} ticks/tick^0.5 "
        f"at h={cal.sigma_horizon} (one-tick sigma {cal.sigma:.4f}, log-return "
        f"{cal.sigma_log:.5f}), A={cal.A:.3f}, k={cal.k:.4f}, R^2={cal.r_squared:.3f}; "
        f"gamma={gamma:g}, skew_k={skew_k:g} from sweep; A-S is fed sigma_diffusive"
    )
    seeds = range(N_TRIALS)

    strategies: dict[str, Callable[[], QuotingAgent]] = {
        "NaiveMM": lambda: NaiveMM(
            agent_id=SUBJECT_AGENT_ID,
            cash=1_000_000,
            inventory=0,
            quote_qty=5,
            half_spread_ticks=2.0,
        ),
        "InventorySkewMM": lambda: InventorySkewMM(
            agent_id=SUBJECT_AGENT_ID,
            cash=1_000_000,
            inventory=0,
            quote_qty=5,
            half_spread_ticks=2.0,
            skew_k=skew_k,
        ),
        "AvellanedaStoikovMM": lambda: AvellanedaStoikovMM(
            agent_id=SUBJECT_AGENT_ID,
            cash=1_000_000,
            inventory=0,
            quote_qty=5,
            gamma=gamma,
            sigma=cal.sigma_diffusive,
            k=cal.k,
            horizon_t=float(SESSION_HORIZON_T),
        ),
    }

    summaries: dict[str, MonteCarloSummary] = {}
    for name, factory in strategies.items():
        t0 = time.perf_counter()
        summary = run_monte_carlo(
            factory, strategy_name=name, spec=spec, seeds=seeds, n_ticks=SESSION_HORIZON_T
        )
        elapsed = time.perf_counter() - t0
        summaries[name] = summary
        print(
            f"{name:20s} n={summary.n_trials} "
            f"mean_pnl={summary.mean_pnl:9.1f} "
            f"ci95=({summary.mean_pnl_ci95[0]:8.1f}, {summary.mean_pnl_ci95[1]:8.1f}) "
            f"std_pnl={summary.std_pnl:8.1f} mean/std={summary.mean_over_std_pnl:5.2f} "
            f"|inv|={summary.mean_abs_terminal_inventory:6.1f} "
            f"markout50={summary.mean_markout:+.2f} "
            f"two_sided_frac={summary.two_sided_tick_fraction:.3f}  ({elapsed:.0f}s)"
        )
        print(
            f"{'':20s} passive_units={summary.mean_passive_units:7.1f} "
            f"aggressor_units={summary.mean_aggressor_units:6.1f} "
            f"informed_share={summary.informed_unit_share:6.1%} "
            f"markout50 vs informed={summary.markout_vs_informed:+.2f} "
            f"vs noise={summary.markout_vs_noise:+.2f}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = {
        "market": spec.name,
        "n_trials": N_TRIALS,
        "n_ticks": SESSION_HORIZON_T,
        "n_noise": spec.n_noise,
        "reference_price": spec.reference_price,
        "calibration": {
            "sigma": cal.sigma,
            "sigma_diffusive": cal.sigma_diffusive,
            "sigma_horizon": cal.sigma_horizon,
            "sigma_fed_to_as": "sigma_diffusive",
            "sigma_log": cal.sigma_log,
            "A": cal.A,
            "k": cal.k,
            "r_squared": cal.r_squared,
            "gamma": gamma,
            "skew_k": skew_k,
        },
        "strategies": {name: summary_to_dict(s) for name, s in summaries.items()},
    }
    json_path = OUT_DIR / f"monte_carlo_summary_{spec.name}.json"
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2, allow_nan=False)
    print(f"\nsaved {json_path}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = list(summaries.keys())
    pnl_data = [[t.terminal_pnl for t in s.trials] for s in summaries.values()]
    axes[0].boxplot(pnl_data, tick_labels=labels, showmeans=True)
    axes[0].set_ylabel("terminal PnL")
    axes[0].set_title(f"Terminal PnL, {spec.name} market (N={N_TRIALS} seeds)")
    axes[0].tick_params(axis="x", rotation=20)

    inv_data = [[abs(t.terminal_inventory) for t in s.trials] for s in summaries.values()]
    axes[1].boxplot(inv_data, tick_labels=labels, showmeans=True)
    axes[1].set_ylabel("|terminal inventory|")
    axes[1].set_title(f"Terminal |inventory|, {spec.name} market (N={N_TRIALS} seeds)")
    axes[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    png_path = OUT_DIR / f"monte_carlo_comparison_{spec.name}.png"
    fig.savefig(png_path, dpi=130)
    print(f"saved {png_path}")


if __name__ == "__main__":
    main()
