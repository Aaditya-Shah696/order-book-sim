"""Calibrate both markets and write results/calibration.json.

Per market: the constants every other script uses (fit on CALIBRATION_SEED,
the seed they all calibrate on), the one-tick sigma's stability across ten
other seeds, and sigma implied by mark changes at several horizons, which is
what separates bid-ask bounce from diffusion and is why A-S is fed the
DIFFUSIVE_HORIZON figure rather than the one-tick one.

Usage: uv run python scripts/run_calibration.py
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from lob_simulator.calibration import (
    DIFFUSIVE_HORIZON,
    SESSION_HORIZON_T,
    calibrate,
    estimate_sigma,
    estimate_sigma_at_horizon,
)
from lob_simulator.market import INFORMED, UNINFORMED, MarketSpec, build_market

CALIBRATION_SEED = 999_999
"""The seed run_gamma_sweep.py, run_monte_carlo.py, run_inventory_path.py and
run_market_impact.py all calibrate on; the numbers written here are the ones
they used."""
STABILITY_SEEDS = range(1000, 1010)
HORIZONS = (1, 10, DIFFUSIVE_HORIZON, 200)

OUT_DIR = Path(__file__).parent.parent / "results"


def report(spec: MarketSpec) -> dict[str, object]:
    print(f"=== {spec.name} market (n_noise={spec.n_noise}, reference={spec.reference_price}) ===")
    t0 = time.perf_counter()
    cal = calibrate(spec, seed=CALIBRATION_SEED)
    t1 = time.perf_counter()
    print(f"fit on seed {CALIBRATION_SEED}: lambda(delta) = A * exp(-k * delta)")
    for d in cal.depths:
        print(f"  depth={d.depth_ticks:3d}  fills={d.fills:5d}  rate={d.fill_rate:.5f}")
    print(f"  A={cal.A:.5f}  k={cal.k:.5f}  R^2={cal.r_squared:.4f}  ({t1 - t0:.1f}s)")
    print(
        f"  sigma (1 tick, rolling median) = {cal.sigma:.4f} ticks/sqrt(tick)   "
        f"sigma_diffusive (h={cal.sigma_horizon}) = {cal.sigma_diffusive:.4f}   "
        f"log-return sigma = {cal.sigma_log:.6f}"
    )

    print(f"\nsigma across seeds {STABILITY_SEEDS.start}..{STABILITY_SEEDS.stop - 1}")
    print("  seed   sigma(1)  " + "  ".join(f"sigma@h={h:<4d}" for h in HORIZONS))
    sigmas: list[float] = []
    sigma_logs: list[float] = []
    by_horizon: dict[int, list[float]] = {h: [] for h in HORIZONS}
    for seed in STABILITY_SEEDS:
        engine = build_market(spec, seed=seed).engine
        engine.run(SESSION_HORIZON_T)
        marks = [m.mark for m in engine.mark_log]
        sigmas.append(estimate_sigma(marks, window=100, units="price"))
        sigma_logs.append(estimate_sigma(marks, window=100, units="log"))
        for h in HORIZONS:
            by_horizon[h].append(estimate_sigma_at_horizon(marks, h))
        print(
            f"  {seed}  {sigmas[-1]:8.4f}  "
            + "  ".join(f"{by_horizon[h][-1]:12.4f}" for h in HORIZONS)
        )

    mean_sigma = statistics.mean(sigmas)
    std_sigma = statistics.pstdev(sigmas)
    cv = std_sigma / mean_sigma
    medians = {h: statistics.median(by_horizon[h]) for h in HORIZONS}
    fundamental_std = None if spec.fundamental is None else spec.fundamental.per_tick_std
    print(
        f"  sigma(1): mean={mean_sigma:.4f} std={std_sigma:.4f} CV={cv:.3f};  "
        f"median sigma@h: "
        + ", ".join(f"h={h}: {medians[h]:.3f}" for h in HORIZONS)
        + (
            f";  fundamental's own per-tick std = {fundamental_std:.3f}"
            if fundamental_std is not None
            else ";  no fundamental (anchor never moves)"
        )
    )
    print()

    return {
        "n_noise": spec.n_noise,
        "reference_price": spec.reference_price,
        "calibration_seed": CALIBRATION_SEED,
        "sigma": cal.sigma,
        "sigma_diffusive": cal.sigma_diffusive,
        "sigma_horizon": cal.sigma_horizon,
        "sigma_log": cal.sigma_log,
        "A": cal.A,
        "k": cal.k,
        "r_squared": cal.r_squared,
        "depths": [
            {"depth_ticks": d.depth_ticks, "fills": d.fills, "ticks": d.ticks, "rate": d.fill_rate}
            for d in cal.depths
        ],
        "sigma_stability": {
            "seeds": list(STABILITY_SEEDS),
            "sigma": sigmas,
            "sigma_log": sigma_logs,
            "mean": mean_sigma,
            "std": std_sigma,
            "cv": cv,
        },
        "sigma_at_horizon": {
            "horizons": list(HORIZONS),
            "median": [medians[h] for h in HORIZONS],
            "per_seed": {str(h): by_horizon[h] for h in HORIZONS},
        },
        "fundamental_per_tick_std": fundamental_std,
    }


def main() -> None:
    print(f"session horizon T = {SESSION_HORIZON_T} ticks\n")
    out: dict[str, object] = {
        "session_horizon_t": SESSION_HORIZON_T,
        "calibration_seed": CALIBRATION_SEED,
        "diffusive_horizon": DIFFUSIVE_HORIZON,
        "markets": {spec.name: report(spec) for spec in (UNINFORMED, INFORMED)},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "calibration.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, allow_nan=False)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
