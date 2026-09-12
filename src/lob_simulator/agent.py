"""Agent base class and the Snapshot it acts on."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .types import OrderIntent, Side


@dataclass(frozen=True, kw_only=True)
class OrderView:
    """Read-only projection of one of the acting agent's own live orders."""

    id: int
    side: Side
    price: int | None
    qty: int
    remaining: int


@dataclass(frozen=True, kw_only=True)
class Snapshot:
    """Everything an agent's ``act()`` sees for one tick. Built fresh per agent per tick."""

    t: int
    best_bid: int | None
    best_ask: int | None
    mark: float
    my_orders: tuple[OrderView, ...]


class Agent(ABC):
    """Base class holding cash and inventory; behavior varies via ``act()``."""

    def __init__(self, *, agent_id: int, cash: int, inventory: int) -> None:
        self.agent_id = agent_id
        self.cash = cash
        self.inventory = inventory
        self._initial_cash = cash
        self._initial_inventory = inventory

    @property
    def initial_cash(self) -> int:
        return self._initial_cash

    @property
    def initial_inventory(self) -> int:
        return self._initial_inventory

    def pnl(self, mark: float) -> float:
        """``(cash - initial_cash) + (inventory - initial_inventory) * mark``.

        The position and the t=0 baseline are valued at the same ``mark``.
        """
        return self.realized_pnl() + self.unrealized_pnl(mark)

    def realized_pnl(self) -> int:
        """The cash-flow component: money that has actually changed hands."""
        return self.cash - self._initial_cash

    def unrealized_pnl(self, mark: float) -> float:
        """The mark-to-market component: net new inventory valued at ``mark``."""
        return (self.inventory - self._initial_inventory) * mark

    @abstractmethod
    def act(self, snapshot: Snapshot) -> list[OrderIntent]:
        """Return this tick's intents. Never touches the book directly."""
        raise NotImplementedError
