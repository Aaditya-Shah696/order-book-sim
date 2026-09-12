"""Monte Carlo comparison: NaiveMM vs InventorySkewMM vs AvellanedaStoikovMM.

500 seeds per strategy, matched across strategies. Calibrates sigma, A and k
fresh on a seed disjoint from the trial seeds, so the script stands alone.

Usage: uv run python scripts/run_monte_carlo.py
"""

from __future__ import annotations

import json
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
from lob_simulator.agents.zero_intelligence import ZeroIntelligenceAgent
from lob_simulator.calibration import SESSION_HORIZON_T, estimate_sigma, fit_fill_intensity
from lob_simulator.engine import Engine
from lob_simulator.research.monte_carlo import MonteCarloSummary, run_monte_carlo
from lob_simulator.seeding import spawn_rngs

matplotlib.use("Agg")

REFERENCE_PRICE = 100
N_NOISE = 20
N_TRIALS = 500
CALIBRATION_SEED = 999_999
GAMMA = 0.2

OUT_DIR = Path(__file__).parent.parent / "results"


def calibrate() -> tuple[float, float]:
    """Measure sigma and k on a fresh ZI market, using CALIBRATION_SEED only."""
    fit = fit_fill_intensity(
        [1, 2, 3, 4, 5, 6, 8, 10, 12, 15],
        n_ticks=SESSION_HORIZON_T,
        n_noise=N_NOISE,
        seed=CALIBRATION_SEED,
        reference_price=REFERENCE_PRICE,
    )
    rngs = spawn_rngs(CALIBRATION_SEED, N_NOISE)
    noise = [
        ZeroIntelligenceAgent(
            agent_id=i + 1,
            cash=1_000_000,
            inventory=1000,
            reference_price=REFERENCE_PRICE,
            rng=rngs[i],
        )
        for i in range(N_NOISE)
    ]
    engine = Engine(noise, reference_price=float(REFERENCE_PRICE))
    engine.run(SESSION_HORIZON_T)
    sigma = estimate_sigma([m.mark for m in engine.mark_log], window=100)

    print(f"calibration: A={fit.A:.4f} k={fit.k:.4f} R^2={fit.r_squared:.4f} sigma={sigma:.5f}")
    return sigma, fit.k


def summary_to_dict(s: MonteCarloSummary) -> dict[str, object]:
    return {
        "strategy_name": s.strategy_name,
        "n_trials": s.n_trials,
        "mean_pnl": s.mean_pnl,
        "var_pnl": s.var_pnl,
        "skew_pnl": s.skew_pnl,
        "p5_pnl": s.p5_pnl,
        "p95_pnl": s.p95_pnl,
        "mean_pnl_ci95": list(s.mean_pnl_ci95),
        "two_sided_tick_fraction": s.two_sided_tick_fraction,
        "mean_abs_terminal_inventory": sum(abs(t.terminal_inventory) for t in s.trials)
        / s.n_trials,
        "std_terminal_inventory": (
            sum(
                (t.terminal_inventory - sum(x.terminal_inventory for x in s.trials) / s.n_trials)
                ** 2
                for t in s.trials
            )
            / s.n_trials
        )
        ** 0.5,
        "mean_fill_count": sum(t.fill_count for t in s.trials) / s.n_trials,
        "mean_realized_pnl": sum(t.realized_pnl for t in s.trials) / s.n_trials,
        "mean_unrealized_pnl": sum(t.unrealized_pnl for t in s.trials) / s.n_trials,
    }


def main() -> None:
    sigma, k = calibrate()
    seeds = range(N_TRIALS)

    strategies: dict[str, Callable[[], QuotingAgent]] = {
        "NaiveMM": lambda: NaiveMM(
            agent_id=0, cash=1_000_000, inventory=0, quote_qty=5, half_spread_ticks=2.0
        ),
        "InventorySkewMM": lambda: InventorySkewMM(
            agent_id=0, cash=1_000_000, inventory=0, quote_qty=5, half_spread_ticks=2.0, skew_k=0.05
        ),
        "AvellanedaStoikovMM": lambda: AvellanedaStoikovMM(
            agent_id=0,
            cash=1_000_000,
            inventory=0,
            quote_qty=5,
            gamma=GAMMA,
            sigma=sigma,
            k=k,
            horizon_t=float(SESSION_HORIZON_T),
        ),
    }

    summaries: dict[str, MonteCarloSummary] = {}
    for name, factory in strategies.items():
        t0 = time.perf_counter()
        summary = run_monte_carlo(
            factory,
            strategy_name=name,
            seeds=seeds,
            n_ticks=SESSION_HORIZON_T,
            n_noise=N_NOISE,
            reference_price=REFERENCE_PRICE,
        )
        elapsed = time.perf_counter() - t0
        summaries[name] = summary
        print(
            f"{name:20s} n={summary.n_trials} "
            f"mean_pnl={summary.mean_pnl:9.1f} "
            f"ci95=({summary.mean_pnl_ci95[0]:8.1f}, {summary.mean_pnl_ci95[1]:8.1f}) "
            f"std_pnl={summary.var_pnl**0.5:8.1f} skew={summary.skew_pnl:6.2f} "
            f"p5={summary.p5_pnl:9.1f} p95={summary.p95_pnl:9.1f} "
            f"two_sided_frac={summary.two_sided_tick_fraction:.3f}  ({elapsed:.0f}s)"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = {
        "n_trials": N_TRIALS,
        "n_ticks": SESSION_HORIZON_T,
        "n_noise": N_NOISE,
        "calibration": {"sigma": sigma, "k": k, "gamma": GAMMA},
        "strategies": {name: summary_to_dict(s) for name, s in summaries.items()},
    }
    with open(OUT_DIR / "monte_carlo_summary.json", "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\nsaved {OUT_DIR / 'monte_carlo_summary.json'}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = list(summaries.keys())
    pnl_data = [[t.terminal_pnl for t in s.trials] for s in summaries.values()]
    axes[0].boxplot(pnl_data, tick_labels=labels, showmeans=True)
    axes[0].set_ylabel("terminal PnL")
    axes[0].set_title(f"Terminal PnL distribution (N={N_TRIALS} seeds)")
    axes[0].tick_params(axis="x", rotation=20)

    inv_data = [[abs(t.terminal_inventory) for t in s.trials] for s in summaries.values()]
    axes[1].boxplot(inv_data, tick_labels=labels, showmeans=True)
    axes[1].set_ylabel("|terminal inventory|")
    axes[1].set_title(f"Terminal |inventory| distribution (N={N_TRIALS} seeds)")
    axes[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "monte_carlo_comparison.png", dpi=130)
    print(f"saved {OUT_DIR / 'monte_carlo_comparison.png'}")


if __name__ == "__main__":
    main()
