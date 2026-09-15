"""MarketSpec and build_market(): the one place a background market is assembled.

Every script and research module drops a subject strategy into the same
market by calling ``build_market``; the spec is the whole description of
what that market is. Two specs are used throughout:

- ``UNINFORMED``: zero-intelligence noise only, anchored to a fixed reference
  price. Nobody knows anything, so no fill is ever adversely selected.
- ``INFORMED``: the same noise, but anchored to a jump random-walk
  fundamental, plus one informed trader who knows that fundamental and hits
  any quote priced on the wrong side of it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .agent import Agent
from .agents.informed import InformedTrader
from .agents.zero_intelligence import ZeroIntelligenceAgent
from .engine import Engine
from .fundamental import FundamentalValue
from .seeding import spawn_rngs

SUBJECT_AGENT_ID = 0
"""The id a subject strategy must carry: noise agents are numbered from 1."""

NOISE_CASH = 1_000_000
NOISE_INVENTORY = 1000


@dataclass(frozen=True, kw_only=True)
class FundamentalSpec:
    """A fundamental-value process and the informed trader who follows it."""

    p_jump: float = 0.5
    max_jump_ticks: int = 4
    informed_threshold_ticks: int = 1
    informed_qty: int = 5
    informed_lookahead: int = 1

    @property
    def per_tick_std(self) -> float:
        """Std of one tick's change in the fundamental, in ticks.

        The jump is ``+-J`` with ``J`` uniform on ``1..max_jump_ticks`` and a
        symmetric sign, so the change has mean zero and variance
        ``p_jump * E[J^2]``. This is the diffusive sigma the informed market's
        mark should show once bid-ask bounce has averaged out, and the
        yardstick ``calibration.estimate_sigma_at_horizon`` is checked against.
        """
        mean_sq = sum(j * j for j in range(1, self.max_jump_ticks + 1)) / self.max_jump_ticks
        return math.sqrt(self.p_jump * mean_sq)


@dataclass(frozen=True, kw_only=True)
class MarketSpec:
    """Everything that defines the background market a strategy is dropped into."""

    n_noise: int = 20
    reference_price: int = 100
    fundamental: FundamentalSpec | None = None

    @property
    def name(self) -> str:
        return "uninformed" if self.fundamental is None else "informed"

    @property
    def n_agents(self) -> int:
        """Background agents only, excluding any subject."""
        return self.n_noise + (0 if self.fundamental is None else 1)


UNINFORMED = MarketSpec()
INFORMED = MarketSpec(reference_price=1000, fundamental=FundamentalSpec())
"""The informed market starts at 1000 ticks rather than 100 so a fundamental
that wanders by tens of ticks over a session stays far from the price floor."""

SPECS_BY_NAME: dict[str, MarketSpec] = {UNINFORMED.name: UNINFORMED, INFORMED.name: INFORMED}


@dataclass(frozen=True, kw_only=True)
class Market:
    engine: Engine
    subject: Agent | None
    fundamental: FundamentalValue | None
    informed_agent_id: int | None
    """The informed trader's id, or ``None`` in the uninformed market. The one
    place that id is derived; research code that splits fills by counterparty
    reads it from here rather than recomputing ``n_noise + 1``."""

    def fundamental_at(self, t: int) -> int:
        """Fair value at tick ``t``: the fundamental if there is one, else the reference."""
        if self.fundamental is None:
            return int(self.engine.reference_price)
        return self.fundamental.value_at(t)


def build_market(spec: MarketSpec, *, seed: int, subject: Agent | None = None) -> Market:
    """Assemble ``spec`` around an optional ``subject`` strategy.

    The subject, if any, acts first each tick and must carry
    ``SUBJECT_AGENT_ID``. Noise agents get ids ``1..n_noise``; the informed
    trader, if any, comes after them. One master ``seed`` derives every random
    stream, so the same ``(spec, seed, subject)`` is the same market.
    """
    if subject is not None and subject.agent_id != SUBJECT_AGENT_ID:
        raise ValueError(
            f"subject agent_id must be {SUBJECT_AGENT_ID}, got {subject.agent_id}"
        )

    rngs = spawn_rngs(seed, spec.n_agents)

    fundamental: FundamentalValue | None = None
    if spec.fundamental is not None:
        fundamental = FundamentalValue(
            initial=spec.reference_price,
            rng=rngs[spec.n_noise],
            p_jump=spec.fundamental.p_jump,
            max_jump_ticks=spec.fundamental.max_jump_ticks,
        )

    agents: list[Agent] = [] if subject is None else [subject]
    agents.extend(
        ZeroIntelligenceAgent(
            agent_id=i + 1,
            cash=NOISE_CASH,
            inventory=NOISE_INVENTORY,
            reference_price=spec.reference_price,
            rng=rngs[i],
            fundamental=fundamental,
        )
        for i in range(spec.n_noise)
    )
    informed_agent_id: int | None = None
    if spec.fundamental is not None and fundamental is not None:
        informed_agent_id = spec.n_noise + 1
        agents.append(
            InformedTrader(
                agent_id=informed_agent_id,
                cash=NOISE_CASH,
                inventory=NOISE_INVENTORY,
                fundamental=fundamental,
                threshold_ticks=spec.fundamental.informed_threshold_ticks,
                qty=spec.fundamental.informed_qty,
                lookahead=spec.fundamental.informed_lookahead,
            )
        )

    engine = Engine(agents, reference_price=float(spec.reference_price))
    return Market(
        engine=engine,
        subject=subject,
        fundamental=fundamental,
        informed_agent_id=informed_agent_id,
    )
