"""Engine: the tick loop tying Agent, Snapshot, and OrderBook together."""

from __future__ import annotations

from collections.abc import Sequence

from .agent import Agent, OrderView, Snapshot
from .book import OrderBook
from .logs import AgentStateLogEntry, MarkLogEntry, TradeContext
from .mark import resolve_mark
from .types import Cancel, OrderIntent, Side, Trade


class Engine:
    """Runs agents against a single OrderBook, tick by tick, in construction order.

    Each agent gets a freshly built Snapshot and returns intents, which are
    applied and settled immediately -- a later agent in the same tick sees the
    state an earlier one left behind.
    """

    def __init__(
        self,
        agents: Sequence[Agent],
        *,
        reference_price: float,
        book: OrderBook | None = None,
    ) -> None:
        ids = [a.agent_id for a in agents]
        if len(ids) != len(set(ids)):
            raise ValueError(f"agent_id values must be unique, got {ids}")

        self.agents: list[Agent] = list(agents)
        self._agents_by_id: dict[int, Agent] = {a.agent_id: a for a in agents}
        self.book = book if book is not None else OrderBook()
        self.reference_price = reference_price
        self.t: int = 0
        self.trade_log: list[Trade] = []
        self.trade_context: list[TradeContext] = []
        self.mark_log: list[MarkLogEntry] = []
        self.agent_state_log: list[AgentStateLogEntry] = []
        self.intent_count: int = 0
        """Every Place and Cancel applied so far: the tape's event count, for
        matching a real event-time tape's sampling rate to this tick clock."""
        self._last_trade_price: int | None = None

    @property
    def mark(self) -> float:
        return resolve_mark(
            best_bid=self.book.best_bid,
            best_ask=self.book.best_ask,
            last_trade_price=self._last_trade_price,
            reference_price=self.reference_price,
        )

    def snapshot_for(self, agent: Agent) -> Snapshot:
        my_orders = tuple(
            OrderView(id=o.id, side=o.side, price=o.price, qty=o.qty, remaining=o.remaining)
            for o in self.book.orders_for_agent(agent.agent_id)
        )
        return Snapshot(
            t=self.t,
            best_bid=self.book.best_bid,
            best_ask=self.book.best_ask,
            mark=self.mark,
            my_orders=my_orders,
        )

    def step(self) -> None:
        for agent in self.agents:
            snapshot = self.snapshot_for(agent)
            intents = agent.act(snapshot)
            self.intent_count += len(intents)
            for intent in intents:
                self._apply(agent, intent)

        final_mark = self.mark
        self.mark_log.append(
            MarkLogEntry(
                t=self.t, mark=final_mark, best_bid=self.book.best_bid, best_ask=self.book.best_ask
            )
        )
        for agent in self.agents:
            self.agent_state_log.append(
                AgentStateLogEntry(
                    t=self.t,
                    agent_id=agent.agent_id,
                    cash=agent.cash,
                    inventory=agent.inventory,
                    pnl=agent.pnl(final_mark),
                )
            )

        self.t += 1

    def run(self, n_ticks: int) -> None:
        for _ in range(n_ticks):
            self.step()

    def _apply(self, agent: Agent, intent: OrderIntent) -> None:
        if isinstance(intent, Cancel):
            self.book.cancel(intent.order_id)
            return

        order = self.book.new_order(
            agent_id=agent.agent_id, side=intent.side, price=intent.price, qty=intent.qty
        )
        trades = self.book.submit(order)
        for trade in trades:
            self._settle(trade, aggressor_side=order.side)
            self._last_trade_price = trade.price
        self.trade_log.extend(trades)
        self.trade_context.extend(TradeContext(t=self.t, aggressor_side=order.side) for _ in trades)

    def _settle(self, trade: Trade, *, aggressor_side: Side) -> None:
        """Move cash and inventory between the two parties to ``trade``.

        The resting side is the opposite of the aggressor's, which identifies
        buyer and seller without re-reading the book.
        """
        aggressor = self._agents_by_id[trade.aggressor_agent]
        resting = self._agents_by_id[trade.resting_agent]
        buyer, seller = (aggressor, resting) if aggressor_side is Side.BUY else (resting, aggressor)
        notional = trade.price * trade.qty
        buyer.cash -= notional
        buyer.inventory += trade.qty
        seller.cash += notional
        seller.inventory -= trade.qty
