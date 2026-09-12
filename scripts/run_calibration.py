"""Fit lambda(delta) = A*exp(-k*delta) and check sigma stability across seeds.

Usage: uv run python scripts/run_calibration.py
"""

from __future__ import annotations

import time

from lob_simulator.agents.zero_intelligence import ZeroIntelligenceAgent
from lob_simulator.calibration import SESSION_HORIZON_T, estimate_sigma, fit_fill_intensity
from lob_simulator.engine import Engine
from lob_simulator.seeding import spawn_rngs

REFERENCE_PRICE = 100
N_NOISE = 20


def main() -> None:
    print(f"session horizon T = {SESSION_HORIZON_T} ticks\n")

    print("fitting lambda(delta) = A * exp(-k * delta)")
    depths = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]
    t0 = time.perf_counter()
    fit = fit_fill_intensity(depths, n_ticks=SESSION_HORIZON_T, n_noise=N_NOISE, seed=42)
    t1 = time.perf_counter()
    for d in fit.depths:
        print(f"  depth={d.depth_ticks:3d}  fills={d.fills:5d}  rate={d.fill_rate:.5f}")
    print(f"  A={fit.A:.5f}  k={fit.k:.5f}  R^2={fit.r_squared:.4f}  ({t1 - t0:.1f}s)")

    print("\nsigma stability across seeds")
    sigmas = []
    for seed in range(10):
        rngs = spawn_rngs(1000 + seed, N_NOISE)
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
        marks = [m.mark for m in engine.mark_log]
        sigma = estimate_sigma(marks, window=100)
        sigmas.append(sigma)
        print(f"  seed={1000 + seed}  sigma={sigma:.6f}")

    mean_sigma = sum(sigmas) / len(sigmas)
    std_sigma = (sum((s - mean_sigma) ** 2 for s in sigmas) / len(sigmas)) ** 0.5
    cv = std_sigma / mean_sigma
    print(f"  mean={mean_sigma:.6f}  std={std_sigma:.6f}  coefficient of variation={cv:.3f}")


if __name__ == "__main__":
    main()
