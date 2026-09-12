"""Cancel and lazy deletion tests."""

from __future__ import annotations

from lob_simulator.book import OrderBook
from lob_simulator.types import Side


class TestCancel:
    def test_cancel_live_order_removes_from_dict_and_drops_depth(self) -> None:
        book = OrderBook()
        order = book.new_order(agent_id=1, side=Side.BUY, price=100, qty=5)
        book.submit(order)
        assert book.best_bid == 100
        assert order.id in book

        result = book.cancel(order.id)

        assert result is True
        assert order.id not in book
        assert book.best_bid is None

    def test_cancel_unknown_id_is_a_noop_not_an_error(self) -> None:
        book = OrderBook()
        result = book.cancel(999)
        assert result is False

    def test_cancel_then_crossing_submit_does_not_fill_against_cancelled_order(self) -> None:
        book = OrderBook()
        resting = book.new_order(agent_id=1, side=Side.SELL, price=100, qty=5)
        book.submit(resting)
        book.cancel(resting.id)

        trades = book.submit(book.new_order(agent_id=2, side=Side.BUY, price=100, qty=5))

        assert trades == []
        assert book.best_ask is None

    def test_cancel_partially_filled_order_removes_the_remainder(self) -> None:
        book = OrderBook()
        resting = book.new_order(agent_id=1, side=Side.SELL, price=100, qty=10)
        book.submit(resting)
        book.submit(book.new_order(agent_id=2, side=Side.BUY, price=100, qty=4))
        assert resting.remaining == 6

        result = book.cancel(resting.id)

        assert result is True
        assert book.best_ask is None

    def test_lazy_deletion_heap_size_may_exceed_dict_size(self) -> None:
        book = OrderBook()
        orders = [book.new_order(agent_id=1, side=Side.BUY, price=100, qty=1) for _ in range(5)]
        for o in orders:
            book.submit(o)
        for o in orders:
            book.cancel(o.id)

        assert book.live_order_count == 0
        assert book.bid_heap_size == 5
        assert book.bid_heap_size > book.live_order_count

    def test_corpses_do_not_leak_liquidity(self) -> None:
        """A live order behind a corpse at a worse price must still fill."""
        book = OrderBook()
        cancelled = book.new_order(agent_id=1, side=Side.SELL, price=100, qty=5)
        book.submit(cancelled)
        book.cancel(cancelled.id)
        behind = book.new_order(agent_id=2, side=Side.SELL, price=101, qty=5)
        book.submit(behind)

        trades = book.submit(book.new_order(agent_id=3, side=Side.BUY, price=101, qty=5))

        assert len(trades) == 1
        assert trades[0].resting_id == behind.id
