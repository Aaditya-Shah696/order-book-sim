"""Zero-intelligence noise trader."""

from __future__ import annotations

import random

from ..agent import Agent, Snapshot
from ..fundamental import FundamentalValue
from ..types import Cancel, OrderIntent, Place, Side


class ZeroIntelligenceAgent(Agent):
    """Noise trader in the Gode & Sunder sense: side, price offset and size are
    drawn independently at random, with no strategy or information behind them.

    Prices are anchored to a fixed point rather than the live mid, so every
    quote stays within ``max_offset_ticks`` of it and the mark cannot
    random-walk away on its own. The anchor is ``reference_price`` unless a
    ``fundamental`` is supplied, in which case it is that process's value at
    the current tick, so the noise flow re-centres on the fundamental as soon
    as it moves. Each live order is cancelled independently with probability
    ``p_cancel_per_order`` per tick, which holds the live order count at a
    stable equilibrium.
    """

    def __init__(
        self,
        *,
        agent_id: int,
        cash: int,
        inventory: int,
        reference_price: int,
        rng: random.Random,
        fundamental: FundamentalValue | None = None,
        max_offset_ticks: int = 10,
        max_qty: int = 5,
        p_active: float = 0.5,
        p_cancel_per_order: float = 0.1,
    ) -> None:
        super().__init__(agent_id=agent_id, cash=cash, inventory=inventory)
        self._reference_price = reference_price
        self._rng = rng
        self._fundamental = fundamental
        self._max_offset_ticks = max_offset_ticks
        self._max_qty = max_qty
        self._p_active = p_active
        self._p_cancel_per_order = p_cancel_per_order

    def anchor_at(self, t: int) -> int:
        """The price this agent's quotes are centred on at tick ``t``."""
        if self._fundamental is None:
            return self._reference_price
        return self._fundamental.value_at(t)

    def act(self, snapshot: Snapshot) -> list[OrderIntent]:
        intents: list[OrderIntent] = [
            Cancel(agent_id=self.agent_id, order_id=o.id)
            for o in snapshot.my_orders
            if self._rng.random() < self._p_cancel_per_order
        ]

        if self._rng.random() < self._p_active:
            side = self._rng.choice([Side.BUY, Side.SELL])
            offset = self._rng.randint(-self._max_offset_ticks, self._max_offset_ticks)
            price = max(1, self.anchor_at(snapshot.t) + offset)
            qty = self._rng.randint(1, self._max_qty)
            intents.append(Place(agent_id=self.agent_id, side=side, price=price, qty=qty))

        return intents
