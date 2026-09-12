"""Self-trade prevention tests (quarantine + restore).

Only the quarantine/restore mechanics. What happens to the aggressor's own
remainder is the unified IOC rule, tested in test_market_orders.py.
"""

from __future__ import annotations

from lob_simulator.book import OrderBook
from lob_simulator.types import Side


class TestSelfTradePrevention:
    def test_crossing_own_resting_order_produces_zero_trades_and_restores_it(self) -> None:
        book = OrderBook()
        resting = book.new_order(agent_id=1, side=Side.SELL, price=100, qty=5)
        book.submit(resting)

        trades = book.submit(book.new_order(agent_id=1, side=Side.BUY, price=101, qty=5))

        assert trades == []
        assert resting.id in book
        assert resting.remaining == 5
        assert book.best_ask == 100

    def test_third_party_ahead_of_quarantined_self_order_still_fills_first(self) -> None:
        book = OrderBook()
        third_party = book.new_order(agent_id=2, side=Side.SELL, price=100, qty=5)
        book.submit(third_party)
        own_order = book.new_order(agent_id=1, side=Side.SELL, price=100, qty=5)
        book.submit(own_order)
        assert third_party.id < own_order.id

        trades = book.submit(book.new_order(agent_id=1, side=Side.BUY, price=100, qty=5))

        assert len(trades) == 1
        assert trades[0].resting_id == third_party.id
        assert trades[0].aggressor_agent == 1
        assert trades[0].resting_agent == 2
        assert third_party.remaining == 0
        assert third_party.id not in book
        assert own_order.id in book
        assert own_order.remaining == 5

    def test_quarantine_skips_past_self_order_to_reach_third_party_behind_it(self) -> None:
        book = OrderBook()
        own_order = book.new_order(agent_id=1, side=Side.SELL, price=100, qty=5)
        book.submit(own_order)
        third_party = book.new_order(agent_id=2, side=Side.SELL, price=100, qty=5)
        book.submit(third_party)
        assert own_order.id < third_party.id

        trades = book.submit(book.new_order(agent_id=1, side=Side.BUY, price=100, qty=5))

        assert len(trades) == 1
        assert trades[0].resting_id == third_party.id
        assert own_order.id in book
        assert own_order.remaining == 5
        assert third_party.id not in book

    def test_multiple_consecutive_self_orders_are_all_quarantined_and_restored(self) -> None:
        book = OrderBook()
        own_1 = book.new_order(agent_id=1, side=Side.SELL, price=100, qty=3)
        book.submit(own_1)
        own_2 = book.new_order(agent_id=1, side=Side.SELL, price=100, qty=3)
        book.submit(own_2)
        third_party = book.new_order(agent_id=2, side=Side.SELL, price=100, qty=3)
        book.submit(third_party)

        trades = book.submit(book.new_order(agent_id=1, side=Side.BUY, price=100, qty=3))

        assert len(trades) == 1
        assert trades[0].resting_id == third_party.id
        assert own_1.remaining == 3
        assert own_2.remaining == 3
        assert own_1.id in book
        assert own_2.id in book

    def test_non_crossing_self_order_is_left_alone_not_quarantined(self) -> None:
        book = OrderBook()
        resting = book.new_order(agent_id=1, side=Side.SELL, price=105, qty=5)
        book.submit(resting)

        trades = book.submit(book.new_order(agent_id=1, side=Side.BUY, price=100, qty=5))

        assert trades == []
        assert book.best_bid == 100
        assert book.best_ask == 105
