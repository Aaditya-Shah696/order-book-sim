"""InformedTrader tests: it picks off quotes priced on the wrong side of next tick's value."""

from __future__ import annotations

import random

import pytest

from helpers import ScriptedAgent
from lob_simulator.agent import OrderView, Snapshot
from lob_simulator.agents.informed import InformedTrader
from lob_simulator.engine import Engine
from lob_simulator.fundamental import FundamentalValue
from lob_simulator.types import Cancel, Place, Side


def _flat_fundamental(value: int = 1000) -> FundamentalValue:
    return FundamentalValue(initial=value, rng=random.Random(0), p_jump=0.0, max_jump_ticks=1)


def _stepping_fundamental(values: list[int]) -> FundamentalValue:
    """A fundamental whose path is exactly ``values`` (then constant)."""
    f = FundamentalValue(initial=values[0], rng=random.Random(0), p_jump=0.0, max_jump_ticks=1)
    f._path = list(values)  # noqa: SLF001 -- pin the path for a deterministic test
    return f


def _trader(fundamental: FundamentalValue, threshold: int = 1, qty: int = 5) -> InformedTrader:
    return InformedTrader(
        agent_id=9,
        cash=1_000_000,
        inventory=1000,
        fundamental=fundamental,
        threshold_ticks=threshold,
        qty=qty,
    )


def _snapshot(t: int, mark: float) -> Snapshot:
    return Snapshot(t=t, best_bid=None, best_ask=None, mark=mark, my_orders=())


class TestRule:
    def test_buys_at_next_value_when_mark_is_below_it(self) -> None:
        trader = _trader(_stepping_fundamental([1000, 1003]))
        intents = trader.act(_snapshot(t=0, mark=1000.0))
        assert intents == [Place(agent_id=9, side=Side.BUY, price=1003, qty=5)]

    def test_sells_at_next_value_when_mark_is_above_it(self) -> None:
        trader = _trader(_stepping_fundamental([1000, 997]))
        intents = trader.act(_snapshot(t=0, mark=1000.0))
        assert intents == [Place(agent_id=9, side=Side.SELL, price=997, qty=5)]

    def test_does_nothing_when_mark_is_within_threshold(self) -> None:
        trader = _trader(_flat_fundamental(1000), threshold=2)
        assert trader.act(_snapshot(t=0, mark=1001.0)) == []
        assert trader.act(_snapshot(t=0, mark=999.5)) == []

    def test_lookahead_zero_trades_on_the_current_value(self) -> None:
        trader = InformedTrader(
            agent_id=9,
            cash=0,
            inventory=0,
            fundamental=_stepping_fundamental([1000, 1010]),
            threshold_ticks=1,
            qty=1,
            lookahead=0,
        )
        assert trader.act(_snapshot(t=0, mark=1000.0)) == []

    def test_cancels_every_resting_order_first(self) -> None:
        trader = _trader(_flat_fundamental(1000))
        snapshot = Snapshot(
            t=0,
            best_bid=None,
            best_ask=None,
            mark=1000.0,
            my_orders=(OrderView(id=4, side=Side.BUY, price=1000, qty=5, remaining=5),),
        )
        assert trader.act(snapshot) == [Cancel(agent_id=9, order_id=4)]


class TestPicksOffStaleQuotes:
    def test_takes_an_ask_priced_below_next_value(self) -> None:
        """A quoter rests 3 at 1002; value is about to be 1004; the informed
        trader lifts them, leaving the quoter short at 1002 against a
        fundamental of 1004, and rests its remaining 2 at fair value."""
        quoter = ScriptedAgent(
            agent_id=1,
            cash=0,
            inventory=0,
            script=[[Place(agent_id=1, side=Side.SELL, price=1002, qty=3)], []],
        )
        trader = _trader(_stepping_fundamental([1000, 1004, 1004]))
        engine = Engine([quoter, trader], reference_price=1000.0)
        engine.step()  # quoter rests; trader sees mark 1000 <= 1004 - 1, buys 5 at 1004
        assert len(engine.trade_log) == 1
        trade = engine.trade_log[0]
        assert (trade.price, trade.qty) == (1002, 3)
        assert trade.resting_agent == 1
        assert trade.aggressor_agent == 9
        assert quoter.inventory == -3
        assert engine.book.best_bid == 1004
        assert engine.book.orders_for_agent(9)[0].remaining == 2

    def test_leaves_a_fairly_priced_ask_alone(self) -> None:
        quoter = ScriptedAgent(
            agent_id=1,
            cash=0,
            inventory=0,
            script=[[Place(agent_id=1, side=Side.SELL, price=1005, qty=5)], []],
        )
        trader = _trader(_stepping_fundamental([1000, 1004, 1004]))
        engine = Engine([quoter, trader], reference_price=1000.0)
        engine.step()
        assert engine.trade_log == []


class TestGuards:
    def test_rejects_bad_parameters(self) -> None:
        f = _flat_fundamental()
        with pytest.raises(ValueError):
            _trader(f, threshold=0)
        with pytest.raises(ValueError):
            _trader(f, qty=0)
        with pytest.raises(ValueError):
            InformedTrader(
                agent_id=9, cash=0, inventory=0, fundamental=f, threshold_ticks=1, qty=1,
                lookahead=-1,
            )
