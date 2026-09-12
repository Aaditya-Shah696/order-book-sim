"""Zero-intelligence agent and market emergence tests.

These are the checks that the simulated market is not degenerate.
scripts/run_zi_check.py runs the same ones over 10k ticks and plots them.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from lob_simulator.agents.zero_intelligence import ZeroIntelligenceAgent
from lob_simulator.engine import Engine

N_AGENTS = 20
N_TICKS = 3000
REFERENCE_PRICE = 100
MAX_OFFSET_TICKS = 10


@dataclass
class _ZiRun:
    two_sided_flags: list[bool]
    spreads: list[int]
    marks: list[float]
    depths: list[int]
    trade_count: int


def _run_zi_market(seed: int, n_ticks: int = N_TICKS) -> _ZiRun:
    rng_master = random.Random(seed)
    agents = [
        ZeroIntelligenceAgent(
            agent_id=i,
            cash=1_000_000,
            inventory=1000,
            reference_price=REFERENCE_PRICE,
            rng=random.Random(rng_master.randrange(2**32)),
            max_offset_ticks=MAX_OFFSET_TICKS,
        )
        for i in range(N_AGENTS)
    ]
    engine = Engine(agents, reference_price=float(REFERENCE_PRICE))

    two_sided_flags: list[bool] = []
    spreads: list[int] = []
    marks: list[float] = []
    depths: list[int] = []
    for _ in range(n_ticks):
        engine.step()
        bid, ask = engine.book.best_bid, engine.book.best_ask
        two_sided_flags.append(bid is not None and ask is not None)
        if bid is not None and ask is not None:
            spreads.append(ask - bid)
        marks.append(engine.mark)
        depths.append(engine.book.live_order_count)

    return _ZiRun(
        two_sided_flags=two_sided_flags,
        spreads=spreads,
        marks=marks,
        depths=depths,
        trade_count=len(engine.trade_log),
    )


class TestMarketEmergence:
    def test_book_is_two_sided_almost_always(self) -> None:
        run = _run_zi_market(seed=1)
        fraction = sum(run.two_sided_flags) / len(run.two_sided_flags)
        assert fraction > 0.95, f"book two-sided only {fraction:.1%} of ticks"

    def test_spread_does_not_diverge(self) -> None:
        run = _run_zi_market(seed=2)
        assert run.spreads, "book was never two-sided -- nothing to measure"
        half = len(run.spreads) // 2
        first_half_mean = sum(run.spreads[:half]) / half
        second_half_mean = sum(run.spreads[half:]) / (len(run.spreads) - half)
        assert second_half_mean < first_half_mean * 3, (
            f"spread grew from {first_half_mean:.2f} to {second_half_mean:.2f} -- looks divergent"
        )

    def test_price_never_hits_zero_or_runs_away(self) -> None:
        """Every quoted price is reference_price + offset with |offset| bounded by
        max_offset_ticks, so the mark cannot leave that band."""
        run = _run_zi_market(seed=3)
        lower = REFERENCE_PRICE - MAX_OFFSET_TICKS
        upper = REFERENCE_PRICE + MAX_OFFSET_TICKS
        assert min(run.marks) >= lower
        assert max(run.marks) <= upper
        assert min(run.marks) > 0

    def test_trade_count_per_1k_ticks_is_non_trivial(self) -> None:
        run = _run_zi_market(seed=4)
        trades_per_1k = run.trade_count / (N_TICKS / 1000)
        assert trades_per_1k > 10, f"only {trades_per_1k:.1f} trades per 1k ticks"

    def test_book_depth_reaches_a_stable_equilibrium_not_unbounded_growth(self) -> None:
        """The last quarter's mean live-order count must not be growing away from
        the first quarter's."""
        run = _run_zi_market(seed=5, n_ticks=6000)
        quarter = len(run.depths) // 4
        early_mean = sum(run.depths[:quarter]) / quarter
        late_mean = sum(run.depths[-quarter:]) / quarter
        assert late_mean < early_mean * 1.5, (
            f"book depth grew from ~{early_mean:.0f} to ~{late_mean:.0f} live orders -- "
            "looks like unbounded accumulation, not equilibrium"
        )
