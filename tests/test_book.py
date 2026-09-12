"""OrderBook resting and matching tests, limit orders only.

Cancellation, self-trade prevention and market orders have their own files.
"""

from __future__ import annotations

import pytest

from lob_simulator.book import OrderBook
from lob_simulator.types import Side


class TestFullFill:
    def test_aggressor_and_resting_both_fully_filled_at_equal_qty(self) -> None:
        book = OrderBook()
        resting = book.new_order(agent_id=1, side=Side.SELL, price=100, qty=10)
        book.submit(resting)

        aggressor = book.new_order(agent_id=2, side=Side.BUY, price=100, qty=10)
        trades = book.submit(aggressor)

        assert len(trades) == 1
        trade = trades[0]
        assert trade.qty == 10
        assert trade.price == 100
        assert trade.aggressor_id == aggressor.id
        assert trade.resting_id == resting.id
        assert book.live_order_count == 0
        assert book.best_bid is None
        assert book.best_ask is None


class TestPartialFill:
    def test_partial_fill_leaves_correct_remainder_resting(self) -> None:
        book = OrderBook()
        resting = book.new_order(agent_id=1, side=Side.SELL, price=100, qty=10)
        book.submit(resting)

        aggressor = book.new_order(agent_id=2, side=Side.BUY, price=100, qty=4)
        trades = book.submit(aggressor)

        assert len(trades) == 1
        assert trades[0].qty == 4
        assert resting.remaining == 6
        assert book.live_order_count == 1
        assert book.best_ask == 100
        assert book.best_bid is None

    def test_aggressor_partial_when_resting_is_smaller(self) -> None:
        book = OrderBook()
        resting = book.new_order(agent_id=1, side=Side.SELL, price=100, qty=4)
        book.submit(resting)

        aggressor = book.new_order(agent_id=2, side=Side.BUY, price=100, qty=10)
        trades = book.submit(aggressor)

        assert len(trades) == 1
        assert trades[0].qty == 4
        assert aggressor.remaining == 6
        assert book.best_bid == 100
        assert book.best_ask is None


class TestPriceTimePriority:
    def test_two_orders_at_same_price_fill_lower_id_first(self) -> None:
        book = OrderBook()
        first = book.new_order(agent_id=1, side=Side.SELL, price=100, qty=5)
        book.submit(first)
        second = book.new_order(agent_id=2, side=Side.SELL, price=100, qty=5)
        book.submit(second)
        assert first.id < second.id

        aggressor = book.new_order(agent_id=3, side=Side.BUY, price=100, qty=5)
        trades = book.submit(aggressor)

        assert len(trades) == 1
        assert trades[0].resting_id == first.id
        assert first.remaining == 0
        assert second.remaining == 5

    def test_better_price_fills_before_better_time(self) -> None:
        book = OrderBook()
        worse_but_earlier = book.new_order(agent_id=1, side=Side.SELL, price=101, qty=5)
        book.submit(worse_but_earlier)
        better_but_later = book.new_order(agent_id=2, side=Side.SELL, price=100, qty=5)
        book.submit(better_but_later)

        aggressor = book.new_order(agent_id=3, side=Side.BUY, price=101, qty=5)
        trades = book.submit(aggressor)

        assert trades[0].resting_id == better_but_later.id


class TestNonCrossing:
    def test_non_crossing_prices_produce_zero_trades(self) -> None:
        book = OrderBook()
        book.submit(book.new_order(agent_id=1, side=Side.SELL, price=101, qty=5))
        trades = book.submit(book.new_order(agent_id=2, side=Side.BUY, price=100, qty=5))

        assert trades == []
        assert book.best_bid == 100
        assert book.best_ask == 101

    def test_sweeps_multiple_price_levels_one_trade_per_resting_order(self) -> None:
        book = OrderBook()
        book.submit(book.new_order(agent_id=1, side=Side.SELL, price=100, qty=3))
        book.submit(book.new_order(agent_id=2, side=Side.SELL, price=101, qty=3))
        book.submit(book.new_order(agent_id=3, side=Side.SELL, price=102, qty=3))

        trades = book.submit(book.new_order(agent_id=4, side=Side.BUY, price=102, qty=9))

        assert len(trades) == 3
        assert [t.price for t in trades] == [100, 101, 102]
        assert [t.qty for t in trades] == [3, 3, 3]
        assert book.live_order_count == 0


class TestGuards:
    def test_submitting_a_live_id_again_raises(self) -> None:
        book = OrderBook()
        order = book.new_order(agent_id=1, side=Side.SELL, price=100, qty=5)
        book.submit(order)
        with pytest.raises(ValueError):
            book.submit(order)
