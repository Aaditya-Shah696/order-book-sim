"""OrderBook: the matching engine."""

from __future__ import annotations

import heapq
import itertools

from .types import Order, Side, Trade

_HeapEntry = tuple[int, int]


class OrderBook:
    """Single-instrument limit order book with price-time priority.

    Bids are a min-heap of ``(-price, id)``, asks of ``(price, id)``; the id
    breaks price ties in arrival order. ``self._orders`` is the sole source of
    truth for liveness. Cancelled and filled orders leave their heap entry
    behind; those stale entries are skipped lazily by ``_top_live``.
    """

    def __init__(self) -> None:
        self._bids: list[_HeapEntry] = []
        self._asks: list[_HeapEntry] = []
        self._orders: dict[int, Order] = {}
        self._orders_by_agent: dict[int, dict[int, Order]] = {}
        self._next_order_id = itertools.count(1)
        self._next_trade_seq = itertools.count(1)

    def _register(self, order: Order) -> None:
        self._orders[order.id] = order
        self._orders_by_agent.setdefault(order.agent_id, {})[order.id] = order

    def _deregister(self, order_id: int) -> Order | None:
        order = self._orders.pop(order_id, None)
        if order is not None:
            agent_orders = self._orders_by_agent[order.agent_id]
            del agent_orders[order_id]
            if not agent_orders:
                del self._orders_by_agent[order.agent_id]
        return order

    def new_order(self, *, agent_id: int, side: Side, price: int | None, qty: int) -> Order:
        """Mint an Order carrying this book instance's next id."""
        return Order(
            agent_id=agent_id,
            side=side,
            price=price,
            qty=qty,
            remaining=qty,
            id=next(self._next_order_id),
        )

    def submit(self, order: Order) -> list[Trade]:
        """Match ``order`` while it crosses, then rest or cancel the remainder.

        Self-trade prevention pulls same-owner resting orders out of the match
        path and restores them afterwards, so the aggressor matches past them
        without consuming them.

        The remainder is cancelled rather than rested when the aggressor is a
        market order, or when it is a limit order that skipped a same-owner
        order during this call -- resting it would lock or cross the book
        against that order.
        """
        if order.id in self._orders:
            raise ValueError(f"order id {order.id} is already live in this book")

        self._register(order)
        book_side, opposite = (
            (self._bids, self._asks) if order.side is Side.BUY else (self._asks, self._bids)
        )

        trades: list[Trade] = []
        quarantined: list[_HeapEntry] = []
        try:
            while order.remaining > 0:
                resting = self._top_live(opposite)
                if resting is None or not self._crosses(order, resting):
                    break
                if resting.agent_id == order.agent_id:
                    quarantined.append(heapq.heappop(opposite))
                    continue
                trades.append(self._fill(order, resting, opposite))
        finally:
            for entry in quarantined:
                heapq.heappush(opposite, entry)

        rests = order.remaining > 0 and order.price is not None and not quarantined
        if rests:
            heapq.heappush(book_side, self._key(order))
        else:
            self._deregister(order.id)

        return trades

    def _fill(self, aggressor: Order, resting: Order, resting_heap: list[_HeapEntry]) -> Trade:
        assert resting.price is not None
        qty = min(aggressor.remaining, resting.remaining)
        price = resting.price
        trade = Trade(
            price=price,
            qty=qty,
            aggressor_id=aggressor.id,
            resting_id=resting.id,
            aggressor_agent=aggressor.agent_id,
            resting_agent=resting.agent_id,
            seq=next(self._next_trade_seq),
        )
        aggressor.remaining -= qty
        resting.remaining -= qty
        if resting.remaining == 0:
            heapq.heappop(resting_heap)
            self._deregister(resting.id)
        return trade

    @staticmethod
    def _crosses(aggressor: Order, resting: Order) -> bool:
        assert resting.price is not None
        if aggressor.price is None:
            return True
        if aggressor.side is Side.BUY:
            return aggressor.price >= resting.price
        return aggressor.price <= resting.price

    @staticmethod
    def _key(order: Order) -> _HeapEntry:
        assert order.price is not None
        return (-order.price, order.id) if order.side is Side.BUY else (order.price, order.id)

    def _top_live(self, heap: list[_HeapEntry]) -> Order | None:
        """Pop stale entries off the top of ``heap`` and return the live Order now on top.

        Never pops the entry it returns.
        """
        while heap:
            _, oid = heap[0]
            resting = self._orders.get(oid)
            if resting is not None:
                return resting
            heapq.heappop(heap)
        return None

    def cancel(self, order_id: int) -> bool:
        """Remove a live order in O(1). Returns whether anything was live to remove."""
        return self._deregister(order_id) is not None

    @property
    def best_bid(self) -> int | None:
        order = self._top_live(self._bids)
        return order.price if order is not None else None

    @property
    def best_ask(self) -> int | None:
        order = self._top_live(self._asks)
        return order.price if order is not None else None

    @property
    def live_order_count(self) -> int:
        return len(self._orders)

    @property
    def bid_heap_size(self) -> int:
        return len(self._bids)

    @property
    def ask_heap_size(self) -> int:
        return len(self._asks)

    def __contains__(self, order_id: int) -> bool:
        return order_id in self._orders

    def orders_for_agent(self, agent_id: int) -> list[Order]:
        """Live orders owned by ``agent_id``, in id order.

        O(k log k) in that agent's own live order count, not in book size.
        """
        agent_orders = self._orders_by_agent.get(agent_id)
        if not agent_orders:
            return []
        return sorted(agent_orders.values(), key=lambda o: o.id)
