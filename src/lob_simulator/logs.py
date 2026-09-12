"""Per-tick log entry types populated by Engine.step()."""

from __future__ import annotations

from dataclasses import dataclass

from .types import Side


@dataclass(frozen=True, kw_only=True)
class TradeContext:
    """Per-trade metadata for analysis. ``trade_context[i]`` describes ``trade_log[i]``."""

    t: int
    aggressor_side: Side


@dataclass(frozen=True, kw_only=True)
class MarkLogEntry:
    """One row per tick."""

    t: int
    mark: float
    best_bid: int | None
    best_ask: int | None


@dataclass(frozen=True, kw_only=True)
class AgentStateLogEntry:
    """One row per (tick, agent). ``pnl`` uses that tick's mark for every agent."""

    t: int
    agent_id: int
    cash: int
    inventory: int
    pnl: float
