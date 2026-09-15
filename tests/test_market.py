"""build_market tests: the uninformed and informed markets, and what differs between them."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from lob_simulator.agents.informed import InformedTrader
from lob_simulator.agents.quoting import NaiveMM
from lob_simulator.agents.zero_intelligence import ZeroIntelligenceAgent
from lob_simulator.market import (
    INFORMED,
    SPECS_BY_NAME,
    SUBJECT_AGENT_ID,
    UNINFORMED,
    FundamentalSpec,
    MarketSpec,
    build_market,
)
from lob_simulator.research.monte_carlo import mean_fill_markout, run_session


def _naive() -> NaiveMM:
    return NaiveMM(
        agent_id=SUBJECT_AGENT_ID, cash=1_000_000, inventory=0, quote_qty=5, half_spread_ticks=2.0
    )


class TestSpecs:
    def test_names_and_registry(self) -> None:
        assert UNINFORMED.name == "uninformed"
        assert INFORMED.name == "informed"
        assert SPECS_BY_NAME == {"uninformed": UNINFORMED, "informed": INFORMED}

    def test_n_agents_counts_the_informed_trader(self) -> None:
        assert UNINFORMED.n_agents == UNINFORMED.n_noise
        assert INFORMED.n_agents == INFORMED.n_noise + 1

    def test_replace_n_noise_keeps_everything_else(self) -> None:
        spec = replace(INFORMED, n_noise=3)
        assert spec.n_noise == 3
        assert spec.fundamental == INFORMED.fundamental
        assert spec.reference_price == INFORMED.reference_price


class TestBuild:
    def test_uninformed_has_only_noise_agents_after_the_subject(self) -> None:
        market = build_market(UNINFORMED, seed=1, subject=_naive())
        agents = market.engine.agents
        assert isinstance(agents[0], NaiveMM)
        assert all(isinstance(a, ZeroIntelligenceAgent) for a in agents[1:])
        assert len(agents) == UNINFORMED.n_noise + 1
        assert market.fundamental is None
        assert market.fundamental_at(500) == UNINFORMED.reference_price

    def test_informed_appends_one_informed_trader_sharing_the_fundamental(self) -> None:
        market = build_market(INFORMED, seed=1, subject=_naive())
        agents = market.engine.agents
        assert isinstance(agents[-1], InformedTrader)
        assert len(agents) == INFORMED.n_noise + 2
        assert market.fundamental is not None
        assert market.fundamental_at(0) == INFORMED.reference_price
        noise = agents[1]
        assert isinstance(noise, ZeroIntelligenceAgent)
        for t in (0, 100, 999):
            assert noise.anchor_at(t) == market.fundamental.value_at(t)

    def test_no_subject_numbers_noise_from_one(self) -> None:
        engine = build_market(UNINFORMED, seed=1).engine
        assert [a.agent_id for a in engine.agents] == list(range(1, UNINFORMED.n_noise + 1))

    def test_subject_must_carry_the_subject_id(self) -> None:
        bad = NaiveMM(agent_id=3, cash=0, inventory=0, quote_qty=1, half_spread_ticks=1.0)
        with pytest.raises(ValueError):
            build_market(UNINFORMED, seed=0, subject=bad)

    def test_same_seed_same_market_byte_for_byte(self) -> None:
        def run(spec: MarketSpec) -> list[tuple[int, int]]:
            engine = build_market(spec, seed=11, subject=_naive()).engine
            engine.run(300)
            return [(t.price, t.qty) for t in engine.trade_log]

        assert run(INFORMED) == run(INFORMED)
        assert run(UNINFORMED) == run(UNINFORMED)
        assert run(INFORMED) != run(UNINFORMED)

    def test_uninformed_noise_streams_do_not_depend_on_the_fundamental_stream(self) -> None:
        """Adding the fundamental takes a new stream; it does not reshuffle the noise ones."""
        a = build_market(UNINFORMED, seed=4).engine.agents[0]
        b = build_market(INFORMED, seed=4).engine.agents[0]
        assert isinstance(a, ZeroIntelligenceAgent) and isinstance(b, ZeroIntelligenceAgent)
        assert a._rng.random() == b._rng.random()  # noqa: SLF001


class TestWhatTheInformedMarketChanges:
    """The properties Result 3b rests on, measured end to end."""

    def test_mid_tracks_the_fundamental(self) -> None:
        market = build_market(INFORMED, seed=2)
        engine = market.engine
        engine.run(1500)
        marks = np.array([m.mark for m in engine.mark_log])
        fundamental = np.array([market.fundamental_at(t) for t in range(1500)])
        assert np.corrcoef(marks, fundamental)[0, 1] > 0.95
        assert np.mean(np.abs(marks - fundamental)) < 5

    def test_price_diffuses_in_the_informed_market_but_not_the_uninformed_one(self) -> None:
        def end_spread(spec: MarketSpec) -> float:
            ends = []
            for seed in range(30):
                engine = build_market(spec, seed=seed).engine
                engine.run(1000)
                ends.append(engine.mark - spec.reference_price)
            return float(np.std(ends))

        assert end_spread(INFORMED) > 4 * end_spread(UNINFORMED)

    def test_fills_carry_adverse_selection_only_in_the_informed_market(self) -> None:
        """Markout erodes from fill time to 50 ticks later against the informed
        trader; against noise alone it does not."""

        def erosion(spec: MarketSpec) -> float:
            now, later = [], []
            for seed in range(6):
                mm = _naive()
                session = run_session(mm, spec=spec, seed=seed, n_ticks=1500)
                now.append(mean_fill_markout(session.engine, mm.agent_id, horizon=0))
                later.append(mean_fill_markout(session.engine, mm.agent_id, horizon=50))
            return float(np.mean(now) - np.mean(later))

        assert erosion(INFORMED) > 0.1
        assert abs(erosion(UNINFORMED)) < 0.1

    def test_informed_trader_profits_on_average(self) -> None:
        pnls = []
        for seed in range(10):
            market = build_market(INFORMED, seed=seed)
            market.engine.run(1000)
            trader = market.engine.agents[-1]
            pnls.append(trader.pnl(market.engine.mark))
        assert np.mean(pnls) > 0

    def test_custom_fundamental_spec_is_honoured(self) -> None:
        still = MarketSpec(
            reference_price=500, fundamental=FundamentalSpec(p_jump=0.0, max_jump_ticks=1)
        )
        market = build_market(still, seed=0)
        assert market.fundamental is not None
        assert market.fundamental.path(200) == [500] * 200
