"""Correlate the MM's share of order flow with A-S's PnL variance relative to inventory-skew.

Takes about 40 minutes: 15 noise levels x 2 strategies x 120 seeds x 2000 ticks.
Runs on the informed market; sigma (diffusive) and k are calibrated at the
spec's default n_noise and held fixed across the sweep, and gamma and skew_k
come from results/gamma_sweep_informed.json.

Usage: uv run python scripts/run_market_impact.py
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
from lob_simulator.market import INFORMED, SUBJECT_AGENT_ID
from lob_simulator.research.gamma_sweep import load_chosen_gamma, load_chosen_skew_k
from lob_simulator.research.market_impact import analyze_market_impact

matplotlib.use("Agg")

SPEC = INFORMED
N_NOISE_VALUES = [2, 3, 4, 5, 6, 8, 10, 13, 17, 22, 28, 35, 45, 60, 80]
N_SEEDS = 120
N_BOOTSTRAP = 3000
CALIBRATION_SEED = 999_999

OUT_DIR = Path(__file__).parent.parent / "results"


def main() -> None:
    sweep_json = OUT_DIR / f"gamma_sweep_{SPEC.name}.json"
    gamma = load_chosen_gamma(sweep_json)
    skew_k = load_chosen_skew_k(sweep_json)
    cal = calibrate(SPEC, seed=CALIBRATION_SEED)
    print(
        f"[{SPEC.name}] sigma_diffusive={cal.sigma_diffusive:.4f} (one-tick {cal.sigma:.4f}) "
        f"k={cal.k:.4f} gamma={gamma:g} skew_k={skew_k:g} "
        f"(calibrated at n_noise={SPEC.n_noise}, held fixed across the sweep)"
    )

    t0 = time.perf_counter()
    finding = analyze_market_impact(
        skew_factory=lambda: InventorySkewMM(
            agent_id=SUBJECT_AGENT_ID,
            cash=1_000_000,
            inventory=0,
            quote_qty=5,
            half_spread_ticks=2.0,
            skew_k=skew_k,
        ),
        as_factory=lambda: AvellanedaStoikovMM(
            agent_id=SUBJECT_AGENT_ID,
            cash=1_000_000,
            inventory=0,
            quote_qty=5,
            gamma=gamma,
            sigma=cal.sigma_diffusive,
            k=cal.k,
            horizon_t=float(SESSION_HORIZON_T),
        ),
        spec=SPEC,
        n_noise_values=N_NOISE_VALUES,
        seeds=range(N_SEEDS),
        n_ticks=SESSION_HORIZON_T,
        n_bootstrap=N_BOOTSTRAP,
    )
    elapsed = time.perf_counter() - t0

    print(f"n_noise values:  {finding.n_noise_values}")
    print(f"mean MM share:   {[round(x, 3) for x in finding.mean_mm_share]}")
    print(f"variance ratio:  {[round(x, 3) for x in finding.variance_ratio]}  (A-S var / skew var)")
    print(
        f"correlation(mm_share, variance_ratio) = {finding.correlation:.3f}, "
        f"95% CI = ({finding.correlation_ci95[0]:.3f}, {finding.correlation_ci95[1]:.3f})"
    )
    print("NULL RESULT (CI includes 0)" if finding.is_null else "EFFECT CONFIRMED (CI excludes 0)")
    print(f"({elapsed:.0f}s, N={N_SEEDS} seeds/level)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "market_impact.json", "w") as f:
        json.dump(
            {
                "market": SPEC.name,
                "n_seeds_per_level": N_SEEDS,
                "calibration": {
                    "sigma": cal.sigma_diffusive,
                    "sigma_is": "diffusive (calibration.sigma_diffusive)",
                    "sigma_one_tick": cal.sigma,
                    "sigma_horizon": cal.sigma_horizon,
                    "k": cal.k,
                    "gamma": gamma,
                    "skew_k": skew_k,
                },
                "n_noise_values": finding.n_noise_values,
                "mean_mm_share": finding.mean_mm_share,
                "variance_ratio": finding.variance_ratio,
                "correlation": finding.correlation,
                "correlation_ci95": finding.correlation_ci95,
                "is_null": finding.is_null,
            },
            f,
            indent=2,
        )
    print(f"saved {OUT_DIR / 'market_impact.json'}")

    shares = np.array(finding.mean_mm_share)
    ratios = np.array(finding.variance_ratio)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(shares, ratios, s=45, zorder=3, label=f"N={N_SEEDS} seeds/level (15 levels)")
    slope, intercept = np.polyfit(shares, ratios, 1)
    x_line = np.linspace(shares.min(), shares.max(), 100)
    ax.plot(
        x_line,
        slope * x_line + intercept,
        color="tab:blue",
        linewidth=1.2,
        alpha=0.6,
        label="linear fit",
    )
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, label="equal variance")
    ax.set_xlabel("mean MM share of total order flow")
    ax.set_ylabel("PnL variance ratio (A-S / InventorySkew)")
    title_suffix = "null result" if finding.is_null else "effect confirmed"
    ax.set_title(
        f"{SPEC.name} market: corr={finding.correlation:.2f}, "
        f"95% CI=({finding.correlation_ci95[0]:.2f}, {finding.correlation_ci95[1]:.2f}) "
        f"-- {title_suffix}"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "market_impact.png", dpi=130)
    print(f"saved {OUT_DIR / 'market_impact.png'}")


if __name__ == "__main__":
    main()
