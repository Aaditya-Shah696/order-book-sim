"""Load a LOBSTER sample into the arrays ``stylized_facts`` measures.

message.csv columns (no header): Time, EventType, OrderID, Size, Price,
Direction. EventType 1=new limit order, 2=partial cancel, 3=total delete,
4=visible execution, 5=hidden execution, 6=cross trade, 7=trading halt.
Prices are integers in units of $0.0001.

orderbook.csv columns (no header, 1 level): AskPrice1, AskSize1, BidPrice1,
BidSize1, same units. Row k of message.csv produced row k of orderbook.csv.

``Direction`` describes the resting order that was hit, so the aggressor's
sign is its negation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_EXECUTION_EVENT_TYPES = {4, 5}

_SENTINEL_ABS_PRICE = 1_000_000_000
"""Threshold separating LOBSTER's empty-level sentinel from a real price.

The sentinel is +-9999999900 ($999,999.99); this cut is $100,000/share in the
same units, well above any real equity price and well below the sentinel.
"""


def load_lobster_sample(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (mid_prices, signed_execution_volume), one entry per aligned message/orderbook row.

    mid_prices are in dollars. signed_execution_volume is 0 for non-execution
    rows, +size for a buy-initiated execution, -size for a sell-initiated one.
    Rows where the book was in a sentinel (empty-level) state are dropped from
    both arrays at the same positions.
    """
    message = pd.read_csv(
        data_dir / "message.csv",
        header=None,
        names=["time", "event_type", "order_id", "size", "price", "direction"],
    )
    orderbook = pd.read_csv(
        data_dir / "orderbook.csv",
        header=None,
        names=["ask_price_1", "ask_size_1", "bid_price_1", "bid_size_1"],
    )
    if len(message) != len(orderbook):
        raise ValueError(
            f"message.csv ({len(message)} rows) and orderbook.csv ({len(orderbook)} rows) "
            "are not row-aligned -- LOBSTER guarantees 1:1, so the files are corrupt"
        )

    valid = (orderbook["ask_price_1"].abs() < _SENTINEL_ABS_PRICE) & (
        orderbook["bid_price_1"].abs() < _SENTINEL_ABS_PRICE
    )
    mid_prices = ((orderbook["ask_price_1"] + orderbook["bid_price_1"]) / 2.0 / 10_000.0).where(
        valid
    )

    is_execution = message["event_type"].isin(_EXECUTION_EVENT_TYPES)
    aggressor_sign = -message["direction"]
    signed_volume = (message["size"] * aggressor_sign).where(is_execution, other=0)

    keep = valid.to_numpy()
    return mid_prices.to_numpy()[keep], signed_volume.to_numpy(dtype=float)[keep]


def resample_by_events(
    mid_prices: np.ndarray, signed_volume: np.ndarray, *, events_per_bucket: int
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse an event-time tape into buckets of ``events_per_bucket`` consecutive events.

    Each bucket contributes the mid at its last event and the sum of its
    signed volume, which is what one simulator tick records: the mark after
    the tick's intents have been applied, and the tick's net signed volume.
    A trailing partial bucket is dropped. Comparing an event-time tape with a
    tick-time one without this step compares different clocks: about half of
    LOBSTER's per-event mid changes are exactly zero, which inflates kurtosis
    and every autocorrelation in a way that has nothing to do with the market.
    """
    if events_per_bucket <= 0:
        raise ValueError(f"events_per_bucket must be positive, got {events_per_bucket}")
    if len(mid_prices) != len(signed_volume):
        raise ValueError("mid_prices and signed_volume must be the same length")
    n_buckets = len(mid_prices) // events_per_bucket
    if n_buckets == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    n_used = n_buckets * events_per_bucket
    mids = np.asarray(mid_prices[:n_used], dtype=float).reshape(n_buckets, events_per_bucket)
    volumes = np.asarray(signed_volume[:n_used], dtype=float).reshape(
        n_buckets, events_per_bucket
    )
    return mids[:, -1].copy(), volumes.sum(axis=1)
