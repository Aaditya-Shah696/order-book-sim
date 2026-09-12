"""Core value types: Order, Trade, OrderIntent."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Literal


class Side(Enum):
    BUY = "buy"
    SELL = "sell"


_default_order_ids = itertools.count(1)
_default_trade_seqs = itertools.count(1)


def _next_default_order_id() -> int:
    return next(_default_order_ids)


def _next_default_trade_seq() -> int:
    return next(_default_trade_seqs)


@dataclass(kw_only=True)
class Order:
    """A resting or in-flight order. ``remaining`` decrements as it fills.

    ``price`` is in integer ticks; ``None`` means a market order. There is no
    liveness field: an id is live iff it is a key in an ``OrderBook``'s dict.
    """

    agent_id: int
    side: Side
    price: int | None
    qty: int
    remaining: int
    id: int = field(default_factory=_next_default_order_id)

    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError(f"qty must be positive, got {self.qty}")
        if not (0 <= self.remaining <= self.qty):
            raise ValueError(f"remaining ({self.remaining}) must be in [0, qty={self.qty}]")
        if self.price is not None and (
            not isinstance(self.price, int) or isinstance(self.price, bool)
        ):
            raise TypeError(
                f"price must be int ticks or None (market), got {type(self.price).__name__}"
            )

    @property
    def is_market(self) -> bool:
        return self.price is None


@dataclass(frozen=True, kw_only=True)
class Trade:
    """One fill. An aggressor consuming k resting orders produces k Trades.

    Priced at the resting order's price.
    """

    price: int
    qty: int
    aggressor_id: int
    resting_id: int
    aggressor_agent: int
    resting_agent: int
    seq: int = field(default_factory=_next_default_trade_seq)


@dataclass(frozen=True, kw_only=True)
class Place:
    """Request to submit a new order. The book mints the id on acceptance."""

    kind: ClassVar[Literal["place"]] = "place"
    agent_id: int
    side: Side
    price: int | None
    qty: int


@dataclass(frozen=True, kw_only=True)
class Cancel:
    """Request to cancel a live order. A no-op if that id is not live."""

    kind: ClassVar[Literal["cancel"]] = "cancel"
    agent_id: int
    order_id: int


OrderIntent = Place | Cancel
