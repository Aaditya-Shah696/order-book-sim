"""Mean |inventory| through the session for InventorySkewMM and AvellanedaStoikovMM.

A-S's inventory skew is gamma * sigma^2 * (T - t): it is zero at the horizon
by construction, so a terminal-inventory comparison judges A-S at the one
tick where it is built to stop caring. This script shows the whole path, on
both markets, with gamma and skew_k from each market's sweep and A-S fed the
diffusive sigma.

Usage: uv run python scripts/run_inventory_path.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from lob_simulator.agents.quoting import AvellanedaStoikovMM, InventorySkewMM
from lob_simulator.calibration import SESSION_HORIZON_T, calibrate
from lob_simulator.market import INFORMED, SUBJECT_AGENT_ID, UNINFORMED, MarketSpec
from lob_simulator.research.gamma_sweep import load_chosen_gamma, load_chosen_skew_k
from lob_simulator.research.inventory_path import InventoryPaths, run_inventory_paths

matplotlib.use("Agg")

N_SEEDS = 60
CALIBRATION_SEED = 999_999
CHECKPOINTS = [250, 500, 1000, 1500, 1800, 1900, 1999]

OUT_DIR = Path(__file__).parent.parent / "results"


def run_market(
    spec: MarketSpec,
) -> tuple[InventoryPaths, InventoryPaths, AvellanedaStoikovMM, float]:
    sweep_json = OUT_DIR / f"gamma_sweep_{spec.name}.json"
    gamma = load_chosen_gamma(sweep_json)
    skew_k = load_chosen_skew_k(sweep_json)
    cal = calibrate(spec, seed=CALIBRATION_SEED)

    def make_as() -> AvellanedaStoikovMM:
        return AvellanedaStoikovMM(
            agent_id=SUBJECT_AGENT_ID,
            cash=1_000_000,
            inventory=0,
            quote_qty=5,
            gamma=gamma,
            sigma=cal.sigma_diffusive,
            k=cal.k,
            horizon_t=float(SESSION_HORIZON_T),
        )

    skew = run_inventory_paths(
        lambda: InventorySkewMM(
            agent_id=SUBJECT_AGENT_ID,
            cash=1_000_000,
            inventory=0,
            quote_qty=5,
            half_spread_ticks=2.0,
            skew_k=skew_k,
        ),
        strategy_name="InventorySkewMM",
        spec=spec,
        seeds=range(N_SEEDS),
        n_ticks=SESSION_HORIZON_T,
    )
    as_paths = run_inventory_paths(
        make_as,
        strategy_name="AvellanedaStoikovMM",
        spec=spec,
        seeds=range(N_SEEDS),
        n_ticks=SESSION_HORIZON_T,
    )
    return skew, as_paths, make_as(), skew_k


def main() -> None:
    t0 = time.perf_counter()
    out: dict[str, object] = {"n_seeds": N_SEEDS, "n_ticks": SESSION_HORIZON_T, "markets": {}}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)

    for ax, spec in zip(axes, (UNINFORMED, INFORMED), strict=True):
        skew, as_paths, as_agent, skew_k = run_market(spec)
        skew_curve = skew.mean_abs_inventory()
        as_curve = as_paths.mean_abs_inventory()
        skew_per_unit = [as_agent.skew_per_unit(t) for t in CHECKPOINTS]

        print(
            f"=== {spec.name} market (gamma={as_agent.gamma:g}, "
            f"sigma_diffusive={as_agent.sigma:.3f}, skew_k={skew_k:g}) ==="
        )
        print("t:                    " + " ".join(f"{c:7d}" for c in CHECKPOINTS))
        print("Skew  mean|inv|:      " + " ".join(f"{skew_curve[c]:7.1f}" for c in CHECKPOINTS))
        print("A-S   mean|inv|:      " + " ".join(f"{as_curve[c]:7.1f}" for c in CHECKPOINTS))
        print("A-S skew/unit (ticks):" + " ".join(f"{s:7.3f}" for s in skew_per_unit))
        print("Skew  skew/unit:      " + " ".join(f"{skew_k:7.3f}" for _ in CHECKPOINTS))

        market_out: dict[str, object] = {
            "gamma": as_agent.gamma,
            "sigma": as_agent.sigma,
            "sigma_is": "diffusive (calibration.sigma_diffusive)",
            "k": as_agent.k,
            "skew_k": skew_k,
            "checkpoints": CHECKPOINTS,
            "skew_mean_abs_inventory": [float(skew_curve[c]) for c in CHECKPOINTS],
            "as_mean_abs_inventory": [float(as_curve[c]) for c in CHECKPOINTS],
            "as_skew_per_unit": skew_per_unit,
            "skew_mean_abs_inventory_midsession": skew.mean_abs_inventory_over(500, 1500),
            "as_mean_abs_inventory_midsession": as_paths.mean_abs_inventory_over(500, 1500),
        }
        markets = out["markets"]
        assert isinstance(markets, dict)
        markets[spec.name] = market_out

        ticks = np.arange(SESSION_HORIZON_T)
        ax.plot(ticks, skew_curve, label=f"InventorySkewMM (skew {skew_k:g}/unit, constant)")
        ax.plot(
            ticks, as_curve,
            label=f"AvellanedaStoikovMM (skew gamma*sigma^2*(T-t), gamma={as_agent.gamma:g})",
        )
        ax2 = ax.twinx()
        ax2.plot(
            ticks, [as_agent.skew_per_unit(t) for t in ticks],
            color="grey", linestyle=":", linewidth=1.0, label="A-S skew per unit (right axis)",
        )
        ax2.axhline(skew_k, color="grey", linestyle="--", linewidth=0.8)
        ax2.set_ylabel("reservation shift per unit inventory (ticks)")
        ax.set_xlabel("tick")
        ax.set_ylabel(f"mean |inventory| over {N_SEEDS} seeds")
        ax.set_title(f"{spec.name} market")
        handles, labels = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(handles + h2, labels + l2, fontsize=7, loc="upper left")

    fig.suptitle("Inventory through the session: A-S's skew vanishes at T by construction")
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "inventory_path.png", dpi=130)
    with open(OUT_DIR / "inventory_path.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {OUT_DIR / 'inventory_path.png'} and inventory_path.json "
          f"({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
