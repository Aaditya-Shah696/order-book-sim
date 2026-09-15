"""Split A-S's gap to the tuned constant-skew heuristic into spread and skew schedule.

Four arms on the same seeds: InventorySkewMM at its swept skew_k, A-S as
published (swept gamma, diffusive sigma), A-S with its half-spread pinned to
NaiveMM's 2.0, and A-S with the gamma*sigma^2*(T-t) term dropped from the
spread. Every A-S arm is paired against the heuristic seed by seed. Seeds are
disjoint from the calibration seed, the sweep's and the Monte Carlo's.

Usage: uv run python scripts/run_spread_decomposition.py --market {uninformed,informed}
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from lob_simulator.calibration import SESSION_HORIZON_T, calibrate
from lob_simulator.market import SPECS_BY_NAME
from lob_simulator.research.gamma_sweep import load_chosen_gamma, load_chosen_skew_k
from lob_simulator.research.spread_decomposition import decompose, decomposition_to_dict

CALIBRATION_SEED = 999_999
SEEDS = range(20_000, 20_200)
"""200 seeds, disjoint from the Monte Carlo's 0..499, the calibration
stability seeds 1000..1009, the sweep's 10000..10059 and the calibration seed."""

OUT_DIR = Path(__file__).parent.parent / "results"


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
        f"[{spec.name}] gamma={gamma:g} skew_k={skew_k:g} "
        f"sigma_diffusive={cal.sigma_diffusive:.4f} k={cal.k:.4f}; "
        f"{len(SEEDS)} seeds {SEEDS.start}..{SEEDS.stop - 1}"
    )

    t0 = time.perf_counter()
    result = decompose(
        spec=spec,
        seeds=SEEDS,
        n_ticks=SESSION_HORIZON_T,
        gamma=gamma,
        sigma=cal.sigma_diffusive,
        k=cal.k,
        skew_k=skew_k,
    )
    elapsed = time.perf_counter() - t0

    out = decomposition_to_dict(result)
    print(
        f"\n{'arm':58s} {'mean':>7s} {'std':>6s} {'m/s':>5s} {'|inv|T':>6s} "
        f"{'passive':>7s} {'aggr':>6s} {'markout':>8s}"
    )
    arms = out["arms"]
    assert isinstance(arms, list)
    for a in arms:
        print(
            f"{a['label']:58s} {a['mean_pnl']:7.0f} {a['std_pnl']:6.0f} "
            f"{a['mean_over_std_pnl']:5.1f} {a['mean_abs_terminal_inventory']:6.1f} "
            f"{a['mean_passive_units']:7.0f} {a['mean_aggressor_units']:6.1f} "
            f"{a['mean_markout_50']:+8.2f}"
        )
    print("\npaired terminal-PnL differences vs InventorySkewMM:")
    paired = out["paired_vs_reference"]
    assert isinstance(paired, list)
    for p in paired:
        lo, hi = p["ci95"]
        print(
            f"  {p['label']:58s} diff={p['mean_diff']:7.0f} "
            f"ci95=({lo:7.0f}, {hi:7.0f}) sd={p['std_diff']:6.0f} p={p['p_value']:.1e}"
        )
    first = arms[0]
    print(
        f"\nwho pays (InventorySkewMM arm, per session): subject {first['mean_pnl']:+.0f}, "
        f"informed {first['mean_informed_pnl']}, noise aggregate {first['mean_noise_pnl']:+.0f}"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"spread_decomposition_{spec.name}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, allow_nan=False)
    print(f"\nsaved {path}  ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
