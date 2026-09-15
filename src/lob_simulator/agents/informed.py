"""Informed trader: knows where the fundamental is going and trades against stale quotes."""

from __future__ import annotations

from ..agent import Agent, Snapshot
from ..fundamental import FundamentalValue
from ..types import Cancel, OrderIntent, Place, Side


class InformedTrader(Agent):
    """Trades toward next tick's fundamental value whenever the mark lags it.

    Each tick it cancels whatever it has resting and reads the fundamental
    ``lookahead`` ticks ahead, ``v``. If ``mark <= v - threshold_ticks`` it
    places a buy limit at ``v`` (a sell limit at ``v`` if
    ``mark >= v + threshold_ticks``). A limit at ``v`` takes every quote
    priced on the wrong side of where value is about to be and rests the
    remainder at that value, so a market maker whose quotes were set off the
    current mid is filled at a price that is about to be wrong. That is the
    adverse selection the uninformed market lacks: the classic Glosten-Milgrom
    picking-off of stale quotes by a trader with a one-tick information edge.
    """

    def __init__(
        self,
        *,
        agent_id: int,
        cash: int,
        inventory: int,
        fundamental: FundamentalValue,
        threshold_ticks: int,
        qty: int,
        lookahead: int = 1,
    ) -> None:
        if threshold_ticks <= 0:
            raise ValueError(f"threshold_ticks must be positive, got {threshold_ticks}")
        if qty <= 0:
            raise ValueError(f"qty must be positive, got {qty}")
        if lookahead < 0:
            raise ValueError(f"lookahead must be non-negative, got {lookahead}")
        super().__init__(agent_id=agent_id, cash=cash, inventory=inventory)
        self._fundamental = fundamental
        self._threshold_ticks = threshold_ticks
        self._qty = qty
        self._lookahead = lookahead

    def target_at(self, t: int) -> int:
        """The value this trader is trading toward at tick ``t``."""
        return self._fundamental.value_at(t + self._lookahead)

    def act(self, snapshot: Snapshot) -> list[OrderIntent]:
        intents: list[OrderIntent] = [
            Cancel(agent_id=self.agent_id, order_id=o.id) for o in snapshot.my_orders
        ]
        value = self.target_at(snapshot.t)
        if snapshot.mark <= value - self._threshold_ticks:
            intents.append(Place(agent_id=self.agent_id, side=Side.BUY, price=value, qty=self._qty))
        elif snapshot.mark >= value + self._threshold_ticks:
            intents.append(
                Place(agent_id=self.agent_id, side=Side.SELL, price=value, qty=self._qty)
            )
        return intents
