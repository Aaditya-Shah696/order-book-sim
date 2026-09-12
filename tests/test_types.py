"""Order, Trade, and OrderIntent value type tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from helpers import make_order, make_trade
from lob_simulator.types import Cancel, Order, Place, Side


class TestOrder:
    def test_remaining_is_explicit_not_auto_derived(self) -> None:
        o = make_order(qty=10, remaining=7)
        assert o.qty == 10
        assert o.remaining == 7

    def test_market_order_price_is_none(self) -> None:
        assert make_order(price=None).is_market is True

    def test_limit_order_is_not_market(self) -> None:
        assert make_order(price=100).is_market is False

    def test_rejects_nonpositive_qty(self) -> None:
        with pytest.raises(ValueError):
            make_order(qty=0, remaining=0)
        with pytest.raises(ValueError):
            make_order(qty=-5, remaining=0)

    def test_rejects_remaining_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            make_order(qty=10, remaining=11)
        with pytest.raises(ValueError):
            make_order(qty=10, remaining=-1)

    def test_rejects_bool_price(self) -> None:
        """bool is an int subclass, so this is the one case mypy cannot catch."""
        with pytest.raises(TypeError):
            Order(agent_id=1, side=Side.BUY, price=True, qty=1, remaining=1)

    def test_two_orders_in_sequence_have_strictly_increasing_ids(self) -> None:
        o1 = make_order()
        o2 = make_order()
        assert o2.id > o1.id

    def test_explicit_id_is_respected(self) -> None:
        o = Order(agent_id=1, side=Side.BUY, price=100, qty=1, remaining=1, id=999)
        assert o.id == 999

    def test_no_liveness_field_exists(self) -> None:
        """Liveness is book-dict membership only."""
        o = make_order()
        assert not hasattr(o, "is_live")
        assert not hasattr(o, "cancelled")
        assert not hasattr(o, "status")


class TestTrade:
    def test_is_frozen(self) -> None:
        t = make_trade()
        with pytest.raises(FrozenInstanceError):
            t.price = 200  # type: ignore[misc]

    def test_fields_roundtrip(self) -> None:
        t = make_trade(
            price=101,
            qty=3,
            aggressor_id=5,
            resting_id=6,
            aggressor_agent=50,
            resting_agent=60,
        )
        assert t.price == 101
        assert t.qty == 3
        assert t.aggressor_id == 5
        assert t.resting_id == 6
        assert t.aggressor_agent == 50
        assert t.resting_agent == 60

    def test_two_trades_in_sequence_have_strictly_increasing_seq(self) -> None:
        t1 = make_trade()
        t2 = make_trade()
        assert t2.seq > t1.seq


class TestOrderIntent:
    def test_place_is_frozen(self) -> None:
        p = Place(agent_id=1, side=Side.BUY, price=100, qty=1)
        with pytest.raises(FrozenInstanceError):
            p.qty = 2  # type: ignore[misc]

    def test_cancel_is_frozen(self) -> None:
        c = Cancel(agent_id=1, order_id=42)
        with pytest.raises(FrozenInstanceError):
            c.order_id = 43  # type: ignore[misc]

    def test_kind_discriminant(self) -> None:
        assert Place(agent_id=1, side=Side.BUY, price=100, qty=1).kind == "place"
        assert Cancel(agent_id=1, order_id=1).kind == "cancel"

    def test_place_carries_no_order_id(self) -> None:
        assert not hasattr(Place(agent_id=1, side=Side.BUY, price=100, qty=1), "id")
