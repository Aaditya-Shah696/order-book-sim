"""Correlate the MM's share of order flow with A-S's PnL variance relative to inventory-skew.

Takes about 40 minutes: 15 noise levels x 2 strategies x 120 seeds x 2000 ticks.

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
from lob_simulator.calibration import SESSION_HORIZON_T
from lob_simulator.research.market_impact import analyze_market_impact

matplotlib.use("Agg")

REFERENCE_PRICE = 100
N_NOISE_VALUES = [2, 3, 4, 5, 6, 8, 10, 13, 17, 22, 28, 35, 45, 60, 80]
N_SEEDS = 120
N_BOOTSTRAP = 3000
GAMMA = 0.2
K_HAT = 0.4986
SIGMA_HAT = 0.03554

OUT_DIR = Path(__file__).parent.parent / "results"


def main() -> None:
    t0 = time.perf_counter()
    finding = analyze_market_impact(
        skew_factory=lambda: InventorySkewMM(
            agent_id=0, cash=1_000_000, inventory=0, quote_qty=5, half_spread_ticks=2.0, skew_k=0.05
        ),
        as_factory=lambda: AvellanedaStoikovMM(
            agent_id=0,
            cash=1_000_000,
            inventory=0,
            quote_qty=5,
            gamma=GAMMA,
            sigma=SIGMA_HAT,
            k=K_HAT,
            horizon_t=float(SESSION_HORIZON_T),
        ),
        n_noise_values=N_NOISE_VALUES,
        seeds=range(N_SEEDS),
        n_ticks=SESSION_HORIZON_T,
        reference_price=REFERENCE_PRICE,
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
                "n_seeds_per_level": N_SEEDS,
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
        f"corr={finding.correlation:.2f}, 95% CI=({finding.correlation_ci95[0]:.2f}, "
        f"{finding.correlation_ci95[1]:.2f}) -- {title_suffix}"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "market_impact.png", dpi=130)
    print(f"saved {OUT_DIR / 'market_impact.png'}")


if __name__ == "__main__":
    main()
