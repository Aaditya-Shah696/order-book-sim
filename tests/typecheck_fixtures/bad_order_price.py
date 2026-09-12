"""Violates Order.price's `int | None` annotation with a float. Never executed."""

from lob_simulator.types import Order, Side

Order(agent_id=1, side=Side.BUY, price=1.5, qty=10, remaining=10)
