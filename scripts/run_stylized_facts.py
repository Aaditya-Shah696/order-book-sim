"""Stylized-fact comparison table: simulated tapes vs real LOBSTER data.

Four columns: the uninformed ZI market, the informed market, LOBSTER in raw
event time, and LOBSTER resampled so that one bucket holds as many events as
one simulated tick applies. The resampled column is the like-for-like one.

Usage: uv run python scripts/run_stylized_facts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from lob_simulator.market import INFORMED, UNINFORMED, MarketSpec, build_market
from lob_simulator.research.lobster import load_lobster_sample, resample_by_events
from lob_simulator.research.stylized_facts import (
    StylizedFactReport,
    abs_return_acf,
    build_report,
    compute_signed_volume,
    log_returns,
    return_acf,
)

matplotlib.use("Agg")

N_TICKS = 20_000
SEED = 7
ACF_LAGS = 25
OFI_WINDOW = 10

LOBSTER_DIR = Path(__file__).parent.parent / "data" / "lobster"
OUT_DIR = Path(__file__).parent.parent / "results"


def simulate(spec: MarketSpec) -> tuple[np.ndarray, np.ndarray, float]:
    """Marks, per-tick signed volume, and the mean number of events applied per tick."""
    engine = build_market(spec, seed=SEED).engine
    engine.run(N_TICKS)
    marks = np.array([m.mark for m in engine.mark_log])
    signed_volume = compute_signed_volume(
        N_TICKS,
        [c.t for c in engine.trade_context],
        [t.qty for t in engine.trade_log],
        [c.aggressor_side for c in engine.trade_context],
    )
    return marks, signed_volume, engine.intent_count / N_TICKS


def report_to_dict(r: StylizedFactReport) -> dict[str, object]:
    return dict(r.__dict__)


def print_table(reports: list[StylizedFactReport]) -> None:
    def row(label: str, fmt: str, attr: str) -> tuple[str, ...]:
        return (label, *(format(getattr(r, attr), fmt) for r in reports))

    rows = [
        row("n_observations", "d", "n_observations"),
        row("Pearson kurtosis (>3 = fat tails)", ".3f", "pearson_kurtosis"),
        row("return ACF, lag 1 (bid-ask bounce)", ".3f", "return_acf_lag1"),
        row("return ACF, mean lag 2+ (~0 expected)", ".4f", "return_acf_mean_lag2_plus"),
        row("|return| ACF, lag 1", ".3f", "abs_return_acf_lag1"),
        row("|return| ACF, lag 20 (clustering persists?)", ".3f", "abs_return_acf_lag20"),
        row("OFI vs. subsequent price change, corr", ".3f", "ofi_price_change_corr"),
    ]
    label_w = max(len(r[0]) for r in rows)
    col_w = max(14, *(len(r.label) for r in reports))
    print(f"{'stat':<{label_w}}  " + "  ".join(f"{r.label:>{col_w}}" for r in reports))
    print("-" * (label_w + 2 + (col_w + 2) * len(reports)))
    for label, *values in rows:
        print(f"{label:<{label_w}}  " + "  ".join(f"{v:>{col_w}}" for v in values))


def main() -> None:
    if not (LOBSTER_DIR / "message.csv").exists():
        print(
            f"No LOBSTER data at {LOBSTER_DIR}. Run scripts/fetch_lobster_sample.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("simulating uninformed and informed markets...")
    sim_marks, sim_sv, events_per_tick = simulate(UNINFORMED)
    inf_marks, inf_sv, inf_events_per_tick = simulate(INFORMED)
    bucket = max(1, round(events_per_tick))
    print(
        f"events per tick: uninformed {events_per_tick:.2f}, informed {inf_events_per_tick:.2f}; "
        f"LOBSTER bucket = {bucket} events"
    )

    print("loading LOBSTER AAPL 2012-06-21...")
    real_mids, real_sv = load_lobster_sample(LOBSTER_DIR)
    real_mids_rs, real_sv_rs = resample_by_events(real_mids, real_sv, events_per_bucket=bucket)

    reports = [
        build_report("sim uninformed", prices=sim_marks, signed_volume=sim_sv,
                     acf_lags=ACF_LAGS, ofi_window=OFI_WINDOW),
        build_report("sim informed", prices=inf_marks, signed_volume=inf_sv,
                     acf_lags=ACF_LAGS, ofi_window=OFI_WINDOW),
        build_report("LOBSTER event", prices=real_mids, signed_volume=real_sv,
                     acf_lags=ACF_LAGS, ofi_window=OFI_WINDOW),
        build_report(f"LOBSTER /{bucket}", prices=real_mids_rs, signed_volume=real_sv_rs,
                     acf_lags=ACF_LAGS, ofi_window=OFI_WINDOW),
    ]

    print()
    print_table(reports)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "stylized_facts.json", "w") as f:
        json.dump(
            {
                "events_per_tick_uninformed": events_per_tick,
                "events_per_tick_informed": inf_events_per_tick,
                "lobster_events_per_bucket": bucket,
                "simulated_uninformed": report_to_dict(reports[0]),
                "simulated_informed": report_to_dict(reports[1]),
                "lobster_event_time": report_to_dict(reports[2]),
                "lobster_resampled": report_to_dict(reports[3]),
            },
            f,
            indent=2,
        )
    print(f"\nsaved {OUT_DIR / 'stylized_facts.json'}")

    series = [
        ("sim uninformed", log_returns(sim_marks)),
        ("sim informed", log_returns(inf_marks)),
        ("LOBSTER, event time", log_returns(real_mids)),
        (f"LOBSTER, {bucket} events/bucket", log_returns(real_mids_rs)),
    ]
    lags = np.arange(1, ACF_LAGS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for label, returns in series:
        axes[0].plot(lags, return_acf(returns, ACF_LAGS), marker="o", markersize=3, label=label)
        axes[1].plot(
            lags, abs_return_acf(returns, ACF_LAGS), marker="o", markersize=3, label=label
        )
    for ax, title in zip(axes, ("Raw return ACF", "|return| ACF (volatility clustering)"),
                         strict=True):
        ax.axhline(0, color="grey", linewidth=0.6)
        ax.set_title(title)
        ax.set_xlabel("lag")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "stylized_facts_acf.png", dpi=130)
    print(f"saved {OUT_DIR / 'stylized_facts_acf.png'}")


if __name__ == "__main__":
    main()
