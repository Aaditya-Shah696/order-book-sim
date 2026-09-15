"""Agent base, Snapshot, and Engine loop tests."""

from __future__ import annotations

import random
from dataclasses import FrozenInstanceError

import pytest

from helpers import RandomTraderAgent, ScriptedAgent
from lob_simulator.agent import Agent, OrderView, Snapshot
from lob_simulator.engine import Engine
from lob_simulator.types import Cancel, OrderIntent, Place, Side


class TestAgentPnl:
    def test_pnl_zero_at_construction(self) -> None:
        agent = ScriptedAgent(agent_id=1, cash=1000, inventory=10, script=[])
        assert agent.pnl(mark=50.0) == 0.0

    def test_frozen_initial_baseline_survives_state_changes(self) -> None:
        agent = ScriptedAgent(agent_id=1, cash=1000, inventory=10, script=[])
        agent.cash = 500
        agent.inventory = 20
        assert agent.initial_cash == 1000
        assert agent.initial_inventory == 10
        assert agent.pnl(mark=50.0) == (500 - 1000) + (20 - 10) * 50.0

    def test_pnl_decomposes_into_realized_plus_unrealized(self) -> None:
        agent = ScriptedAgent(agent_id=1, cash=1000, inventory=10, script=[])
        agent.cash = 800
        agent.inventory = 15
        mark = 42.0
        assert agent.pnl(mark) == agent.realized_pnl() + agent.unrealized_pnl(mark)

    def test_initial_cash_is_read_only(self) -> None:
        agent = ScriptedAgent(agent_id=1, cash=1000, inventory=10, script=[])
        with pytest.raises(AttributeError):
            agent.initial_cash = 2000  # type: ignore[misc]


class TestSnapshot:
    def test_my_orders_equals_exactly_the_agents_book_dict_entries(self) -> None:
        maker = ScriptedAgent(
            agent_id=1,
            cash=10_000,
            inventory=0,
            script=[
                [
                    Place(agent_id=1, side=Side.BUY, price=99, qty=3),
                    Place(agent_id=1, side=Side.SELL, price=101, qty=2),
                ]
            ],
        )
        other = ScriptedAgent(agent_id=2, cash=10_000, inventory=0, script=[[]])
        engine = Engine([maker, other], reference_price=100.0)

        engine.step()
        snapshot = engine.snapshot_for(maker)
        book_orders = engine.book.orders_for_agent(1)

        assert len(snapshot.my_orders) == len(book_orders) == 2
        for view, order in zip(snapshot.my_orders, book_orders, strict=True):
            assert view.id == order.id
            assert view.side == order.side
            assert view.price == order.price
            assert view.remaining == order.remaining

    def test_my_orders_is_empty_for_an_agent_with_no_live_orders(self) -> None:
        maker = ScriptedAgent(agent_id=1, cash=10_000, inventory=0, script=[[]])
        engine = Engine([maker], reference_price=100.0)
        engine.step()
        assert engine.snapshot_for(maker).my_orders == ()

    def test_projection_is_immutable(self) -> None:
        view = OrderView(id=1, side=Side.BUY, price=100, qty=5, remaining=5)
        with pytest.raises(FrozenInstanceError):
            view.remaining = 3  # type: ignore[misc]

        maker = ScriptedAgent(agent_id=1, cash=10_000, inventory=0, script=[[]])
        engine = Engine([maker], reference_price=100.0)
        snapshot = engine.snapshot_for(maker)
        with pytest.raises(TypeError):
            snapshot.my_orders[0:0] = ()  # type: ignore[index]


class TestMarkLadder:
    def test_mark_is_reference_price_before_anything_exists(self) -> None:
        agent = ScriptedAgent(agent_id=1, cash=1000, inventory=0, script=[[]])
        engine = Engine([agent], reference_price=123.5)
        assert engine.mark == 123.5

    def test_mark_is_mid_once_two_sided(self) -> None:
        agent = ScriptedAgent(
            agent_id=1,
            cash=10_000,
            inventory=0,
            script=[[Place(agent_id=1, side=Side.BUY, price=98, qty=1)]],
        )
        other = ScriptedAgent(
            agent_id=2,
            cash=10_000,
            inventory=0,
            script=[[Place(agent_id=2, side=Side.SELL, price=102, qty=1)]],
        )
        engine = Engine([agent, other], reference_price=100.0)
        engine.step()
        assert engine.mark == (98 + 102) / 2

    def test_mark_falls_back_to_last_trade_when_one_sided(self) -> None:
        seller = ScriptedAgent(
            agent_id=1,
            cash=10_000,
            inventory=10,
            script=[[Place(agent_id=1, side=Side.SELL, price=100, qty=1)], []],
        )
        buyer = ScriptedAgent(
            agent_id=2,
            cash=10_000,
            inventory=0,
            script=[[], [Place(agent_id=2, side=Side.BUY, price=100, qty=1)]],
        )
        engine = Engine([seller, buyer], reference_price=50.0)
        engine.step()
        engine.step()
        assert engine.book.best_bid is None
        assert engine.book.best_ask is None
        assert engine.mark == 100.0, "mark should fall back to the last trade, not reference_price"


class TestSettlementAndConservation:
    def test_full_fill_moves_cash_and_inventory_symmetrically(self) -> None:
        seller = ScriptedAgent(
            agent_id=1,
            cash=10_000,
            inventory=5,
            script=[[Place(agent_id=1, side=Side.SELL, price=100, qty=5)], []],
        )
        buyer = ScriptedAgent(
            agent_id=2,
            cash=10_000,
            inventory=0,
            script=[[], [Place(agent_id=2, side=Side.BUY, price=100, qty=5)]],
        )
        engine = Engine([seller, buyer], reference_price=100.0)
        engine.run(2)

        assert seller.cash == 10_000 + 500
        assert seller.inventory == 0
        assert buyer.cash == 10_000 - 500
        assert buyer.inventory == 5
        assert seller.cash + buyer.cash == 20_000
        assert seller.inventory + buyer.inventory == 5

    def test_trade_context_stays_parallel_to_trade_log(self) -> None:
        seller = ScriptedAgent(
            agent_id=1,
            cash=10_000,
            inventory=5,
            script=[[Place(agent_id=1, side=Side.SELL, price=100, qty=5)], []],
        )
        buyer = ScriptedAgent(
            agent_id=2,
            cash=10_000,
            inventory=0,
            script=[[], [Place(agent_id=2, side=Side.BUY, price=100, qty=5)]],
        )
        engine = Engine([seller, buyer], reference_price=100.0)
        engine.run(2)

        assert len(engine.trade_context) == len(engine.trade_log) == 1
        assert engine.trade_context[0].t == 1
        assert engine.trade_context[0].aggressor_side is Side.BUY

    def test_conservation_holds_across_a_long_randomized_run(self) -> None:
        n_agents = 6
        initial_inventory_each = 100
        agents = [
            RandomTraderAgent(
                agent_id=i,
                cash=100_000,
                inventory=initial_inventory_each,
                rng=random.Random(1000 + i),
                price_range=(90, 110),
            )
            for i in range(n_agents)
        ]
        engine = Engine(agents, reference_price=100.0)
        engine.run(500)

        assert engine.trade_log, "test is vacuous if nothing ever traded"
        total_inventory = sum(a.inventory for a in agents)
        total_cash = sum(a.cash for a in agents)
        assert total_inventory == n_agents * initial_inventory_each
        assert total_cash == sum(a.initial_cash for a in agents)


class TestIntentCount:
    def test_counts_every_place_and_cancel_applied(self) -> None:
        agent = ScriptedAgent(
            agent_id=1,
            cash=10_000,
            inventory=0,
            script=[
                [
                    Place(agent_id=1, side=Side.BUY, price=99, qty=1),
                    Place(agent_id=1, side=Side.SELL, price=101, qty=1),
                ],
                [Cancel(agent_id=1, order_id=1)],
                [],
            ],
        )
        engine = Engine([agent], reference_price=100.0)
        assert engine.intent_count == 0
        engine.step()
        assert engine.intent_count == 2
        engine.step()
        assert engine.intent_count == 3
        engine.step()
        assert engine.intent_count == 3


class TestFixedAgentOrder:
    def test_agents_act_in_construction_order_every_tick(self) -> None:
        call_order: list[int] = []

        class OrderRecordingAgent(Agent):
            def act(self, snapshot: Snapshot) -> list[OrderIntent]:
                call_order.append(self.agent_id)
                return []

        agents = [OrderRecordingAgent(agent_id=i, cash=0, inventory=0) for i in (3, 1, 2)]
        engine = Engine(agents, reference_price=100.0)
        engine.run(3)

        assert call_order == [3, 1, 2, 3, 1, 2, 3, 1, 2]


class TestEngineGuards:
    def test_duplicate_agent_ids_rejected(self) -> None:
        a1 = ScriptedAgent(agent_id=1, cash=100, inventory=0, script=[[]])
        a2 = ScriptedAgent(agent_id=1, cash=100, inventory=0, script=[[]])
        with pytest.raises(ValueError):
            Engine([a1, a2], reference_price=100.0)
