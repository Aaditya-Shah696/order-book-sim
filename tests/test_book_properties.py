"""Property-based test: the book is never crossed or locked.

The example count is the strength of this test; 1000 is the floor it is
meant to run at.
"""

from __future__ import annotations

from typing import NamedTuple

from hypothesis import given, settings
from hypothesis import strategies as st

from lob_simulator.book import OrderBook
from lob_simulator.types import Side


class _PlaceCmd(NamedTuple):
    agent_id: int
    side: Side
    price: int | None
    qty: int


class _CancelCmd(NamedTuple):
    order_id: int


_prices = st.one_of(st.integers(min_value=90, max_value=110), st.none())
_sides = st.sampled_from([Side.BUY, Side.SELL])
_qtys = st.integers(min_value=1, max_value=20)
_agents = st.integers(min_value=1, max_value=4)

_place_cmd = st.builds(_PlaceCmd, agent_id=_agents, side=_sides, price=_prices, qty=_qtys)
_cancel_cmd = st.builds(_CancelCmd, order_id=st.integers(min_value=0, max_value=60))

_command = st.one_of(_place_cmd, _cancel_cmd)


@given(st.lists(_command, min_size=1, max_size=150))
@settings(max_examples=1000, deadline=None)
def test_book_never_crossed_or_locked(commands: list[_PlaceCmd | _CancelCmd]) -> None:
    """best_bid < best_ask whenever both exist, after every command in the sequence.

    Covers limit orders, market orders (price=None), and cancels interleaved.
    """
    book = OrderBook()
    for cmd in commands:
        if isinstance(cmd, _PlaceCmd):
            order = book.new_order(
                agent_id=cmd.agent_id, side=cmd.side, price=cmd.price, qty=cmd.qty
            )
            book.submit(order)
        else:
            book.cancel(cmd.order_id)

        bid, ask = book.best_bid, book.best_ask
        if bid is not None and ask is not None:
            assert bid < ask, f"book crossed or locked: bid={bid} ask={ask}"
