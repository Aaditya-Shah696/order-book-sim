"""Market order and unified IOC invariant tests."""

from __future__ import annotations

from lob_simulator.book import OrderBook
from lob_simulator.types import Side


class TestMarketOrders:
    def test_market_buy_fills_available_depth_at_resting_prices(self) -> None:
        book = OrderBook()
        book.submit(book.new_order(agent_id=1, side=Side.SELL, price=100, qty=3))
        book.submit(book.new_order(agent_id=1, side=Side.SELL, price=101, qty=3))

        trades = book.submit(book.new_order(agent_id=2, side=Side.BUY, price=None, qty=6))

        assert len(trades) == 2
        assert [t.price for t in trades] == [100, 101]
        assert [t.qty for t in trades] == [3, 3]
        assert book.live_order_count == 0

    def test_market_order_with_insufficient_depth_fills_what_exists_and_remainder_vanishes(
        self,
    ) -> None:
        book = OrderBook()
        book.submit(book.new_order(agent_id=1, side=Side.SELL, price=100, qty=3))

        market_order = book.new_order(agent_id=2, side=Side.BUY, price=None, qty=10)
        trades = book.submit(market_order)

        assert len(trades) == 1
        assert trades[0].qty == 3
        assert market_order.remaining == 7
        assert market_order.id not in book
        assert book.live_order_count == 0
        assert book.best_bid is None
        assert book.best_ask is None

    def test_market_order_against_empty_book_produces_zero_trades_and_never_rests(self) -> None:
        book = OrderBook()
        market_order = book.new_order(agent_id=1, side=Side.BUY, price=None, qty=5)

        trades = book.submit(market_order)

        assert trades == []
        assert market_order.id not in book
        assert book.best_bid is None

    def test_market_order_ignores_self_trade_quarantine_and_still_takes_third_party_depth(
        self,
    ) -> None:
        book = OrderBook()
        own_order = book.new_order(agent_id=1, side=Side.SELL, price=100, qty=5)
        book.submit(own_order)
        third_party = book.new_order(agent_id=2, side=Side.SELL, price=101, qty=5)
        book.submit(third_party)

        trades = book.submit(book.new_order(agent_id=1, side=Side.BUY, price=None, qty=5))

        assert len(trades) == 1
        assert trades[0].resting_id == third_party.id
        assert own_order.id in book
        assert own_order.remaining == 5


class TestSelfTradeAdjacentIOC:
    """A limit remainder crossing a quarantined same-owner order is cancelled, never rested."""

    def test_self_crossing_limit_remainder_is_cancelled_not_rested(self) -> None:
        book = OrderBook()
        resting = book.new_order(agent_id=1, side=Side.SELL, price=100, qty=5)
        book.submit(resting)

        buy_order = book.new_order(agent_id=1, side=Side.BUY, price=101, qty=5)
        trades = book.submit(buy_order)

        assert trades == []
        assert buy_order.id not in book
        assert resting.id in book
        assert resting.remaining == 5
        assert book.best_bid is None, (
            "the remainder must not rest: it would lock the book against the "
            "same-owner order it skipped"
        )
        assert book.best_ask == 100

    def test_self_crossing_remainder_after_partial_third_party_fill_is_still_cancelled(
        self,
    ) -> None:
        book = OrderBook()
        third_party = book.new_order(agent_id=2, side=Side.SELL, price=100, qty=2)
        book.submit(third_party)
        own_order = book.new_order(agent_id=1, side=Side.SELL, price=100, qty=5)
        book.submit(own_order)

        aggressor = book.new_order(agent_id=1, side=Side.BUY, price=100, qty=6)
        trades = book.submit(aggressor)

        assert len(trades) == 1
        assert trades[0].qty == 2
        assert trades[0].resting_id == third_party.id
        assert aggressor.remaining == 4
        assert aggressor.id not in book
        assert own_order.id in book
        assert own_order.remaining == 5
        assert book.best_bid is None
        assert book.best_ask == 100

    def test_ordinary_non_self_trade_limit_remainder_still_rests_normally(self) -> None:
        book = OrderBook()
        trades = book.submit(book.new_order(agent_id=1, side=Side.BUY, price=100, qty=5))

        assert trades == []
        assert book.best_bid == 100
