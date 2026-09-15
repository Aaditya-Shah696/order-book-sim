"""QuotingAgent base, NaiveMM, InventorySkewMM, and AvellanedaStoikovMM."""

from __future__ import annotations

import math
from abc import abstractmethod

from ..agent import Agent, Snapshot
from ..types import Cancel, OrderIntent, Place, Side


class QuotingAgent(Agent):
    """Base for every market-making strategy: cancel all live orders, then place a fresh pair.

    Subclasses implement only ``quote()``. Cancels are returned ahead of the
    new placements and the engine applies intents in order, so the old pair is
    always gone before the new pair goes in.

    Quotes are snapped outward to the tick grid: the bid is the floor of
    ``center - half_spread`` and the ask the ceiling of ``center + half_spread``.
    Rounding to nearest would let a half-integer mark pull one side a tick
    closer than the other, and Python's round-half-to-even would pick which
    side by parity.
    """

    def __init__(self, *, agent_id: int, cash: int, inventory: int, quote_qty: int) -> None:
        if quote_qty <= 0:
            raise ValueError(f"quote_qty must be positive, got {quote_qty}")
        super().__init__(agent_id=agent_id, cash=cash, inventory=inventory)
        self.quote_qty = quote_qty

    @staticmethod
    def snap(center: float, half_spread: float) -> tuple[int, int]:
        """``(bid, ask)`` in integer ticks, never tighter than ``half_spread`` either side.

        The bid is floored at 1 tick; the ask is at least one tick above it.
        """
        if half_spread <= 0:
            raise ValueError(f"half_spread must be positive, got {half_spread}")
        bid = max(1, math.floor(center - half_spread))
        ask = max(bid + 1, math.ceil(center + half_spread))
        return bid, ask

    def act(self, snapshot: Snapshot) -> list[OrderIntent]:
        intents: list[OrderIntent] = [
            Cancel(agent_id=self.agent_id, order_id=o.id) for o in snapshot.my_orders
        ]
        center, half_spread = self.quote(snapshot)
        bid_price, ask_price = self.snap(center, half_spread)
        intents.append(
            Place(agent_id=self.agent_id, side=Side.BUY, price=bid_price, qty=self.quote_qty)
        )
        intents.append(
            Place(agent_id=self.agent_id, side=Side.SELL, price=ask_price, qty=self.quote_qty)
        )
        return intents

    @abstractmethod
    def quote(self, snapshot: Snapshot) -> tuple[float, float]:
        """Return ``(center, half_spread)``, both in integer-tick price units, for this tick."""
        raise NotImplementedError


class NaiveMM(QuotingAgent):
    """Constant half-spread around the current mark, ignoring inventory. The control."""

    def __init__(
        self,
        *,
        agent_id: int,
        cash: int,
        inventory: int,
        quote_qty: int,
        half_spread_ticks: float,
    ) -> None:
        if half_spread_ticks <= 0:
            raise ValueError(f"half_spread_ticks must be positive, got {half_spread_ticks}")
        super().__init__(agent_id=agent_id, cash=cash, inventory=inventory, quote_qty=quote_qty)
        self.half_spread_ticks = half_spread_ticks

    def quote(self, snapshot: Snapshot) -> tuple[float, float]:
        return snapshot.mark, self.half_spread_ticks


class InventorySkewMM(QuotingAgent):
    """NaiveMM plus an inventory-proportional skew on the quote center.

    Positive inventory shifts the center down, making the ask easier to hit and
    the bid harder, pushing the agent back toward flat.
    """

    def __init__(
        self,
        *,
        agent_id: int,
        cash: int,
        inventory: int,
        quote_qty: int,
        half_spread_ticks: float,
        skew_k: float,
    ) -> None:
        if half_spread_ticks <= 0:
            raise ValueError(f"half_spread_ticks must be positive, got {half_spread_ticks}")
        if skew_k < 0:
            raise ValueError(f"skew_k must be non-negative, got {skew_k}")
        super().__init__(agent_id=agent_id, cash=cash, inventory=inventory, quote_qty=quote_qty)
        self.half_spread_ticks = half_spread_ticks
        self.skew_k = skew_k

    def quote(self, snapshot: Snapshot) -> tuple[float, float]:
        center = snapshot.mark - self.skew_k * self.inventory
        return center, self.half_spread_ticks


class AvellanedaStoikovMM(QuotingAgent):
    """Avellaneda & Stoikov (2008) optimal market-making quotes.

    reservation price:  r     = mid - q * gamma * sigma^2 * (T - t)
    optimal spread:     delta = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)

    Quotes at ``center = r``, ``half_spread = delta / 2``.

    Units: the model's mid is an arithmetic Brownian motion ``dS = sigma dW``
    in *price* units, so ``sigma`` must be the per-tick standard deviation of
    mark *differences* in ticks (``calibration.estimate_sigma`` with
    ``units="price"``), not a log-return volatility. ``k`` is the fill
    intensity's decay per tick of depth and ``horizon_t`` is in ticks. The
    inventory skew ``gamma * sigma^2 * (T - t)`` is therefore in ticks per
    unit of inventory and vanishes at ``T`` by construction. ``gamma``,
    ``sigma``, ``k`` and ``horizon_t`` are supplied by the caller; this class
    does no calibration of its own.
    """

    def __init__(
        self,
        *,
        agent_id: int,
        cash: int,
        inventory: int,
        quote_qty: int,
        gamma: float,
        sigma: float,
        k: float,
        horizon_t: float,
    ) -> None:
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if horizon_t <= 0:
            raise ValueError(f"horizon_t must be positive, got {horizon_t}")
        super().__init__(agent_id=agent_id, cash=cash, inventory=inventory, quote_qty=quote_qty)
        self.gamma = gamma
        self.sigma = sigma
        self.k = k
        self.horizon_t = horizon_t

    def skew_per_unit(self, t: int) -> float:
        """``gamma * sigma^2 * (T - t)``: ticks of reservation shift per unit of inventory."""
        tau = max(0.0, self.horizon_t - t)
        return self.gamma * self.sigma**2 * tau

    def quote(self, snapshot: Snapshot) -> tuple[float, float]:
        variance_to_go = self.skew_per_unit(snapshot.t)
        reservation = snapshot.mark - self.inventory * variance_to_go
        spread = variance_to_go + (2.0 / self.gamma) * math.log(1.0 + self.gamma / self.k)
        return reservation, spread / 2.0
