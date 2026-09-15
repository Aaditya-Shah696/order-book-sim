"""QuotingAgent base and NaiveMM tests."""

from __future__ import annotations

import random

import pytest

from lob_simulator.agent import Snapshot
from lob_simulator.agents.quoting import NaiveMM, QuotingAgent
from lob_simulator.agents.zero_intelligence import ZeroIntelligenceAgent
from lob_simulator.engine import Engine
from lob_simulator.seeding import spawn_rngs
from lob_simulator.types import Place, Side


def _make_naive_mm(
    agent_id: int = 0, half_spread_ticks: float = 2.0, quote_qty: int = 5
) -> NaiveMM:
    return NaiveMM(
        agent_id=agent_id,
        cash=1_000_000,
        inventory=0,
        quote_qty=quote_qty,
        half_spread_ticks=half_spread_ticks,
    )


def _make_noise_agents(n: int, seed: int) -> list[ZeroIntelligenceAgent]:
    rngs = spawn_rngs(seed, n)
    return [
        ZeroIntelligenceAgent(
            agent_id=i + 1, cash=1_000_000, inventory=1000, reference_price=100, rng=rngs[i]
        )
        for i in range(n)
    ]


class TestNaiveMMQuoting:
    def test_quotes_are_symmetric_around_mark(self) -> None:
        mm = _make_naive_mm(half_spread_ticks=3.0)
        noise = ZeroIntelligenceAgent(
            agent_id=1, cash=1_000_000, inventory=1000, reference_price=100, rng=random.Random(1)
        )
        engine = Engine([mm, noise], reference_price=100.0)
        engine.step()

        live = engine.book.orders_for_agent(mm.agent_id)
        assert len(live) == 2
        bid = next(o for o in live if o.side is Side.BUY)
        ask = next(o for o in live if o.side is Side.SELL)
        assert bid.price is not None and ask.price is not None
        center = (bid.price + ask.price) / 2
        assert ask.price - center == pytest.approx(center - bid.price)
        assert ask.price - bid.price == pytest.approx(2 * 3.0)

    def test_live_order_count_never_exceeds_two(self) -> None:
        mm = _make_naive_mm()
        engine = Engine([mm, *_make_noise_agents(10, seed=1)], reference_price=100.0)
        for _ in range(1000):
            engine.step()
            live = engine.book.orders_for_agent(mm.agent_id)
            assert len(live) <= 2, f"MM has {len(live)} live orders at t={engine.t}"

    def test_prior_orders_cancelled_and_exactly_two_new_ones_placed_each_tick(self) -> None:
        mm = _make_naive_mm()
        noise = ZeroIntelligenceAgent(
            agent_id=1, cash=1_000_000, inventory=1000, reference_price=100, rng=random.Random(2)
        )
        engine = Engine([mm, noise], reference_price=100.0)

        engine.step()
        first_tick_ids = {o.id for o in engine.book.orders_for_agent(mm.agent_id)}
        assert len(first_tick_ids) == 2

        engine.step()
        second_tick_ids = {o.id for o in engine.book.orders_for_agent(mm.agent_id)}
        assert len(second_tick_ids) == 2
        assert first_tick_ids.isdisjoint(second_tick_ids)

    def test_rejects_nonpositive_half_spread(self) -> None:
        with pytest.raises(ValueError):
            _make_naive_mm(half_spread_ticks=0.0)
        with pytest.raises(ValueError):
            _make_naive_mm(half_spread_ticks=-1.0)


class TestNaiveMMInventoryDrift:
    def test_inventory_drifts_without_a_restoring_force(self) -> None:
        """Drift is measured as the inventory range traveled relative to one fill.

        Tight oscillation around flat keeps that ratio small; an unbounded walk
        makes it large.
        """
        mm = _make_naive_mm(half_spread_ticks=1.0, quote_qty=5)
        engine = Engine([mm, *_make_noise_agents(20, seed=3)], reference_price=100.0)

        inventories: list[int] = []
        for _ in range(5000):
            engine.step()
            inventories.append(mm.inventory)

        traveled_range = max(inventories) - min(inventories)
        assert traveled_range > 10 * mm.quote_qty, (
            f"NaiveMM inventory only ranged over {traveled_range} "
            f"(vs. quote_qty={mm.quote_qty}) -- doesn't look like unbounded drift"
        )


class TestQuoteSnapping:
    """Quotes snap outward to the tick grid: floor the bid, ceil the ask."""

    def test_half_integer_mark_keeps_both_sides_at_least_half_spread_away(self) -> None:
        bid, ask = QuotingAgent.snap(100.5, 2.0)
        assert (bid, ask) == (98, 103)
        assert 100.5 - bid >= 2.0 and ask - 100.5 >= 2.0

    def test_integer_mark_and_integer_half_spread_are_exact(self) -> None:
        assert QuotingAgent.snap(100.0, 2.0) == (98, 102)

    def test_fractional_half_spread_never_rounds_inward(self) -> None:
        for center in (100.0, 100.5, 99.25):
            for half in (1.66, 1.91, 2.4):
                bid, ask = QuotingAgent.snap(center, half)
                assert center - bid >= half
                assert ask - center >= half

    def test_no_parity_asymmetry_at_exact_halves(self) -> None:
        """round() would have made these (98, 102) and (100, 102): 2.5 one side, 1.5 the other."""
        assert QuotingAgent.snap(100.5, 2.0) == (98, 103)
        assert QuotingAgent.snap(99.5, 2.0) == (97, 102)

    def test_bid_floors_at_one_tick_and_ask_stays_above_it(self) -> None:
        bid, ask = QuotingAgent.snap(1.5, 3.0)
        assert bid == 1
        assert ask == 5

    def test_rejects_nonpositive_half_spread(self) -> None:
        with pytest.raises(ValueError):
            QuotingAgent.snap(100.0, 0.0)

    def test_naive_mm_quotes_on_a_half_integer_mark_are_symmetric(self) -> None:
        mm = _make_naive_mm(half_spread_ticks=2.0)
        snapshot = Snapshot(t=0, best_bid=100, best_ask=101, mark=100.5, my_orders=())
        places = [i for i in mm.act(snapshot) if isinstance(i, Place)]
        bid = next(p for p in places if p.side is Side.BUY)
        ask = next(p for p in places if p.side is Side.SELL)
        assert (bid.price, ask.price) == (98, 103)
