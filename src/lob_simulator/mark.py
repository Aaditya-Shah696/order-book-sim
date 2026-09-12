"""The mark ladder: mid, then last trade, then reference price."""

from __future__ import annotations


def resolve_mark(
    *,
    best_bid: int | None,
    best_ask: int | None,
    last_trade_price: int | None,
    reference_price: float,
) -> float:
    """Best available mark. ``reference_price`` is the floor, so t=0 has a mark."""
    if best_bid is not None and best_ask is not None:
        return (best_bid + best_ask) / 2
    if last_trade_price is not None:
        return float(last_trade_price)
    return reference_price
