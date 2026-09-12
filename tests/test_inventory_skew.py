"""InventorySkewMM tests."""

from __future__ import annotations

from typing import cast

from lob_simulator.agent import Snapshot
from lob_simulator.agents.quoting import InventorySkewMM, NaiveMM, QuotingAgent
from lob_simulator.agents.zero_intelligence import ZeroIntelligenceAgent
from lob_simulator.engine import Engine
from lob_simulator.seeding import spawn_rngs

FLAT_SNAPSHOT = Snapshot(t=0, best_bid=None, best_ask=None, mark=100.0, my_orders=())


def _make_skew_mm(
    agent_id: int = 0, half_spread_ticks: float = 1.0, skew_k: float = 0.05
) -> InventorySkewMM:
    return InventorySkewMM(
        agent_id=agent_id,
        cash=1_000_000,
        inventory=0,
        quote_qty=5,
        half_spread_ticks=half_spread_ticks,
        skew_k=skew_k,
    )


def _make_noise_agents(n: int, seed: int) -> list[ZeroIntelligenceAgent]:
    rngs = spawn_rngs(seed, n)
    return [
        ZeroIntelligenceAgent(
            agent_id=i + 1, cash=1_000_000, inventory=1000, reference_price=100, rng=rngs[i]
        )
        for i in range(n)
    ]


def _terminal_abs_inventory(
    mm: QuotingAgent, seed: int, n_ticks: int = 1500, n_noise: int = 15
) -> int:
    engine = Engine([mm, *_make_noise_agents(n_noise, seed)], reference_price=100.0)
    engine.run(n_ticks)
    return abs(mm.inventory)


class TestQuoteShift:
    def test_positive_inventory_shifts_both_quotes_down(self) -> None:
        mm = _make_skew_mm()
        center_flat, half_spread = mm.quote(FLAT_SNAPSHOT)

        mm.inventory = 50
        center_long, half_spread_long = mm.quote(FLAT_SNAPSHOT)

        assert half_spread_long == half_spread, "only the center should move"
        assert center_long < center_flat
        bid_flat, ask_flat = round(center_flat - half_spread), round(center_flat + half_spread)
        bid_long, ask_long = round(center_long - half_spread), round(center_long + half_spread)
        assert bid_long < bid_flat
        assert ask_long < ask_flat

    def test_negative_inventory_shifts_both_quotes_up(self) -> None:
        mm = _make_skew_mm()
        center_flat, _ = mm.quote(FLAT_SNAPSHOT)

        mm.inventory = -50
        center_short, _ = mm.quote(FLAT_SNAPSHOT)

        assert center_short > center_flat

    def test_zero_inventory_matches_naive_mm_exactly(self) -> None:
        skew = _make_skew_mm(skew_k=0.3)
        naive = NaiveMM(agent_id=1, cash=1_000_000, inventory=0, quote_qty=5, half_spread_ticks=1.0)
        assert skew.quote(FLAT_SNAPSHOT) == naive.quote(FLAT_SNAPSHOT)

    def test_shift_is_exactly_proportional_to_inventory(self) -> None:
        mm = _make_skew_mm(skew_k=0.05)
        center_flat, _ = mm.quote(FLAT_SNAPSHOT)
        mm.inventory = 40
        center_40, _ = mm.quote(FLAT_SNAPSHOT)
        mm.inventory = 80
        center_80, _ = mm.quote(FLAT_SNAPSHOT)

        assert (center_flat - center_40) == (center_40 - center_80), "skew is linear in q"


class TestSharesBaseWithNaive:
    """`quote` must be the only code difference from NaiveMM."""

    def test_shares_act_with_naive_mm(self) -> None:
        assert InventorySkewMM.act is QuotingAgent.act
        assert NaiveMM.act is QuotingAgent.act

    def test_quote_is_the_only_overridden_method(self) -> None:
        assert InventorySkewMM.quote is not QuotingAgent.quote
        assert cast(object, InventorySkewMM.quote) is not cast(object, NaiveMM.quote)

        def own_public_names(cls: type) -> set[str]:
            """Names the class defines directly, not what an instance ends up with."""
            return {name for name in vars(cls) if not name.startswith("_")}

        assert own_public_names(NaiveMM) == own_public_names(InventorySkewMM) == {"quote"}


class TestTerminalInventoryTighterThanNaive:
    def test_terminal_abs_inventory_distribution_is_tighter_than_naive_across_matched_seeds(
        self,
    ) -> None:
        seeds = range(200, 210)

        naive_terms = [
            _terminal_abs_inventory(
                NaiveMM(
                    agent_id=0, cash=1_000_000, inventory=0, quote_qty=5, half_spread_ticks=1.0
                ),
                seed,
            )
            for seed in seeds
        ]
        skew_terms = [_terminal_abs_inventory(_make_skew_mm(), seed) for seed in seeds]

        naive_mean = sum(naive_terms) / len(naive_terms)
        skew_mean = sum(skew_terms) / len(skew_terms)
        assert skew_mean < naive_mean, (
            f"skew mean |inv|={skew_mean:.1f} not tighter than naive mean={naive_mean:.1f}"
        )
        assert max(skew_terms) < max(naive_terms), (
            "skew must be tighter as a worst case, not only on average"
        )
