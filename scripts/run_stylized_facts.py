"""Stylized-fact comparison table: simulated ZI tape vs real LOBSTER data.

Usage: uv run python scripts/run_stylized_facts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from lob_simulator.agents.zero_intelligence import ZeroIntelligenceAgent
from lob_simulator.engine import Engine
from lob_simulator.research.lobster import load_lobster_sample
from lob_simulator.research.stylized_facts import (
    StylizedFactReport,
    abs_return_acf,
    build_report,
    compute_signed_volume,
    log_returns,
    return_acf,
)
from lob_simulator.seeding import spawn_rngs

matplotlib.use("Agg")

REFERENCE_PRICE = 100
N_NOISE = 20
N_TICKS = 20_000
SEED = 7
ACF_LAGS = 25
OFI_WINDOW = 10

LOBSTER_DIR = Path(__file__).parent.parent / "data" / "lobster"
OUT_DIR = Path(__file__).parent.parent / "results"


def simulate() -> tuple[np.ndarray, np.ndarray]:
    rngs = spawn_rngs(SEED, N_NOISE)
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
    engine.run(N_TICKS)
    marks = np.array([m.mark for m in engine.mark_log])
    signed_volume = compute_signed_volume(
        N_TICKS,
        [c.t for c in engine.trade_context],
        [t.qty for t in engine.trade_log],
        [c.aggressor_side for c in engine.trade_context],
    )
    return marks, signed_volume


def print_table(sim: StylizedFactReport, real: StylizedFactReport) -> None:
    rows = [
        ("n_observations", f"{sim.n_observations}", f"{real.n_observations}"),
        (
            "Pearson kurtosis (>3 = fat tails)",
            f"{sim.pearson_kurtosis:.3f}",
            f"{real.pearson_kurtosis:.3f}",
        ),
        (
            "return ACF, lag 1 (bid-ask bounce)",
            f"{sim.return_acf_lag1:.3f}",
            f"{real.return_acf_lag1:.3f}",
        ),
        (
            "return ACF, mean lag 2+ (~0 expected)",
            f"{sim.return_acf_mean_lag2_plus:.4f}",
            f"{real.return_acf_mean_lag2_plus:.4f}",
        ),
        (
            "|return| ACF, lag 1",
            f"{sim.abs_return_acf_lag1:.3f}",
            f"{real.abs_return_acf_lag1:.3f}",
        ),
        (
            "|return| ACF, lag 20 (clustering persists?)",
            f"{sim.abs_return_acf_lag20:.3f}",
            f"{real.abs_return_acf_lag20:.3f}",
        ),
        (
            "OFI vs. subsequent price change, corr",
            f"{sim.ofi_price_change_corr:.3f}",
            f"{real.ofi_price_change_corr:.3f}",
        ),
    ]
    label_w = max(len(r[0]) for r in rows)
    print(f"{'stat':<{label_w}}  {'simulated':>12}  {'LOBSTER AAPL':>12}")
    print("-" * (label_w + 30))
    for label, sim_v, real_v in rows:
        print(f"{label:<{label_w}}  {sim_v:>12}  {real_v:>12}")


def main() -> None:
    if not (LOBSTER_DIR / "message.csv").exists():
        print(
            f"No LOBSTER data at {LOBSTER_DIR}. Run scripts/fetch_lobster_sample.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("simulating ZI market...")
    sim_marks, sim_signed_volume = simulate()
    sim_report = build_report(
        "simulated ZI market",
        prices=sim_marks,
        signed_volume=sim_signed_volume,
        acf_lags=ACF_LAGS,
        ofi_window=OFI_WINDOW,
    )

    print("loading LOBSTER AAPL 2012-06-21...")
    real_mids, real_signed_volume = load_lobster_sample(LOBSTER_DIR)
    real_report = build_report(
        "LOBSTER AAPL 2012-06-21",
        prices=real_mids,
        signed_volume=real_signed_volume,
        acf_lags=ACF_LAGS,
        ofi_window=OFI_WINDOW,
    )

    print()
    print_table(sim_report, real_report)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "stylized_facts.json", "w") as f:
        json.dump(
            {"simulated": sim_report.__dict__, "lobster": real_report.__dict__},
            f,
            indent=2,
        )
    print(f"\nsaved {OUT_DIR / 'stylized_facts.json'}")

    sim_returns = log_returns(sim_marks)
    real_returns = log_returns(real_mids)
    lags = np.arange(1, ACF_LAGS + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(
        lags, return_acf(sim_returns, ACF_LAGS), marker="o", markersize=3, label="simulated"
    )
    axes[0].plot(
        lags, return_acf(real_returns, ACF_LAGS), marker="o", markersize=3, label="LOBSTER AAPL"
    )
    axes[0].axhline(0, color="grey", linewidth=0.6)
    axes[0].set_title("Raw return ACF")
    axes[0].set_xlabel("lag")
    axes[0].legend()

    axes[1].plot(
        lags, abs_return_acf(sim_returns, ACF_LAGS), marker="o", markersize=3, label="simulated"
    )
    axes[1].plot(
        lags, abs_return_acf(real_returns, ACF_LAGS), marker="o", markersize=3, label="LOBSTER AAPL"
    )
    axes[1].axhline(0, color="grey", linewidth=0.6)
    axes[1].set_title("|return| ACF (volatility clustering)")
    axes[1].set_xlabel("lag")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUT_DIR / "stylized_facts_acf.png", dpi=130)
    print(f"saved {OUT_DIR / 'stylized_facts_acf.png'}")


if __name__ == "__main__":
    main()
