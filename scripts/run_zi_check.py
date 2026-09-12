"""Run the zero-intelligence market and check that it is not degenerate.

Usage: uv run python scripts/run_zi_check.py
"""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from lob_simulator.agents.zero_intelligence import ZeroIntelligenceAgent
from lob_simulator.engine import Engine

matplotlib.use("Agg")

N_AGENTS = 30
N_TICKS = 10_000
REFERENCE_PRICE = 100
SEED = 7

OUT_DIR = Path(__file__).parent.parent / "results"


def main() -> None:
    rng_master = random.Random(SEED)
    agents = [
        ZeroIntelligenceAgent(
            agent_id=i,
            cash=1_000_000,
            inventory=1000,
            reference_price=REFERENCE_PRICE,
            rng=random.Random(rng_master.randrange(2**32)),
        )
        for i in range(N_AGENTS)
    ]
    engine = Engine(agents, reference_price=float(REFERENCE_PRICE))

    marks: list[float] = []
    spreads: list[int] = []
    two_sided_count = 0
    best_bids: list[int | None] = []
    best_asks: list[int | None] = []

    for _ in range(N_TICKS):
        engine.step()
        bid, ask = engine.book.best_bid, engine.book.best_ask
        best_bids.append(bid)
        best_asks.append(ask)
        marks.append(engine.mark)
        if bid is not None and ask is not None:
            two_sided_count += 1
            spreads.append(ask - bid)

    n_trades = len(engine.trade_log)
    two_sided_frac = two_sided_count / N_TICKS
    trades_per_1k = n_trades / (N_TICKS / 1000)

    print(f"ticks: {N_TICKS}, agents: {N_AGENTS}, reference_price: {REFERENCE_PRICE}")
    print(f"two-sided fraction: {two_sided_frac:.4f}")
    print(f"trades: {n_trades} ({trades_per_1k:.1f} per 1k ticks)")
    if spreads:
        print(
            f"spread: min={min(spreads)} max={max(spreads)} mean={sum(spreads) / len(spreads):.2f}"
        )
        half = len(spreads) // 2
        print(
            f"spread first-half mean={sum(spreads[:half]) / half:.2f}  "
            f"second-half mean={sum(spreads[half:]) / (len(spreads) - half):.2f}"
        )
    print(f"mark: min={min(marks):.2f} max={max(marks):.2f} last={marks[-1]:.2f}")

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    axes[0].plot(marks, linewidth=0.6, color="black", label="mark")
    axes[0].axhline(
        REFERENCE_PRICE, color="tab:red", linestyle="--", linewidth=0.8, label="reference_price"
    )
    axes[0].set_ylabel("mark price")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title(
        f"ZI market emergence check (seed={SEED}, {N_AGENTS} agents, {N_TICKS} ticks)"
    )

    bid_series = [b if b is not None else float("nan") for b in best_bids]
    ask_series = [a if a is not None else float("nan") for a in best_asks]
    axes[1].plot(bid_series, linewidth=0.5, color="tab:green", label="best_bid")
    axes[1].plot(ask_series, linewidth=0.5, color="tab:red", label="best_ask")
    axes[1].set_ylabel("price")
    axes[1].legend(loc="upper right", fontsize=8)

    spread_series = [
        (a - b) if (a is not None and b is not None) else float("nan")
        for a, b in zip(best_asks, best_bids, strict=True)
    ]
    axes[2].plot(spread_series, linewidth=0.5, color="tab:blue")
    axes[2].set_ylabel("spread (ticks)")
    axes[2].set_xlabel("tick")

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "zi_market_emergence.png"
    fig.savefig(out_path, dpi=130)
    print(f"saved plot to {out_path}")


if __name__ == "__main__":
    main()
