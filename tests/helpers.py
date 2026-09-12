"""Shared construction helpers for the test suite."""

from __future__ import annotations

import random

from lob_simulator.agent import Agent, Snapshot
from lob_simulator.types import Cancel, Order, OrderIntent, Place, Side, Trade


def make_order(
    *,
    agent_id: int = 1,
    side: Side = Side.BUY,
    price: int | None = 100,
    qty: int = 10,
    remaining: int | None = None,
) -> Order:
    return Order(
        agent_id=agent_id,
        side=side,
        price=price,
        qty=qty,
        remaining=qty if remaining is None else remaining,
    )


def make_trade(
    *,
    price: int = 100,
    qty: int = 5,
    aggressor_id: int = 1,
    resting_id: int = 2,
    aggressor_agent: int = 10,
    resting_agent: int = 20,
) -> Trade:
    return Trade(
        price=price,
        qty=qty,
        aggressor_id=aggressor_id,
        resting_id=resting_id,
        aggressor_agent=aggressor_agent,
        resting_agent=resting_agent,
    )


class ScriptedAgent(Agent):
    """Returns a pre-programmed list of intents, one entry per tick.

    Ticks past the end of the script return no intents.
    """

    def __init__(
        self,
        *,
        agent_id: int,
        cash: int,
        inventory: int,
        script: list[list[OrderIntent]],
    ) -> None:
        super().__init__(agent_id=agent_id, cash=cash, inventory=inventory)
        self._script = script
        self._tick = 0

    def act(self, snapshot: Snapshot) -> list[OrderIntent]:
        intents = self._script[self._tick] if self._tick < len(self._script) else []
        self._tick += 1
        return intents


class RandomTraderAgent(Agent):
    """Places small random limit orders and occasionally cancels one of its own.

    Stresses Engine-level invariants without depending on ZeroIntelligenceAgent.
    """

    def __init__(
        self,
        *,
        agent_id: int,
        cash: int,
        inventory: int,
        rng: random.Random,
        price_range: tuple[int, int],
    ) -> None:
        super().__init__(agent_id=agent_id, cash=cash, inventory=inventory)
        self._rng = rng
        self._price_range = price_range

    def act(self, snapshot: Snapshot) -> list[OrderIntent]:
        intents: list[OrderIntent] = []
        if snapshot.my_orders and self._rng.random() < 0.3:
            target = self._rng.choice(snapshot.my_orders)
            intents.append(Cancel(agent_id=self.agent_id, order_id=target.id))
        if self._rng.random() < 0.7:
            side = self._rng.choice([Side.BUY, Side.SELL])
            price = self._rng.randint(*self._price_range)
            qty = self._rng.randint(1, 5)
            intents.append(Place(agent_id=self.agent_id, side=side, price=price, qty=qty))
        return intents
