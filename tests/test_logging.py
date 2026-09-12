"""Logging, seeding, and export tests."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import pandas as pd
import pytest

from helpers import ScriptedAgent
from lob_simulator.agents.zero_intelligence import ZeroIntelligenceAgent
from lob_simulator.engine import Engine
from lob_simulator.export import agent_states_to_frame, export_run, marks_to_frame, trades_to_frame
from lob_simulator.seeding import spawn_rngs
from lob_simulator.types import Place, Side


def _build_engine(seed: int, n_agents: int = 10) -> Engine:
    rngs = spawn_rngs(seed, n_agents)
    agents = [
        ZeroIntelligenceAgent(
            agent_id=i, cash=1_000_000, inventory=1000, reference_price=100, rng=rngs[i]
        )
        for i in range(n_agents)
    ]
    return Engine(agents, reference_price=100.0)


class TestSeeding:
    def test_same_seed_produces_identical_rng_streams(self) -> None:
        draws_a = [r.random() for r in spawn_rngs(42, 5)]
        draws_b = [r.random() for r in spawn_rngs(42, 5)]
        assert draws_a == draws_b

    def test_different_seeds_produce_different_streams(self) -> None:
        draws_a = [r.random() for r in spawn_rngs(42, 5)]
        draws_b = [r.random() for r in spawn_rngs(43, 5)]
        assert draws_a != draws_b

    def test_same_seed_produces_byte_identical_simulation_runs(self) -> None:
        engine_a = _build_engine(seed=100)
        engine_a.run(500)
        engine_b = _build_engine(seed=100)
        engine_b.run(500)

        trades_a = [(t.price, t.qty, t.aggressor_id, t.resting_id) for t in engine_a.trade_log]
        trades_b = [(t.price, t.qty, t.aggressor_id, t.resting_id) for t in engine_b.trade_log]
        assert trades_a == trades_b
        assert [m.mark for m in engine_a.mark_log] == [m.mark for m in engine_b.mark_log]

    def test_different_seeds_diverge(self) -> None:
        engine_a = _build_engine(seed=100)
        engine_a.run(500)
        engine_b = _build_engine(seed=200)
        engine_b.run(500)

        marks_a = [m.mark for m in engine_a.mark_log]
        marks_b = [m.mark for m in engine_b.mark_log]
        assert marks_a != marks_b


class TestLogs:
    def test_mark_log_has_one_entry_per_tick(self) -> None:
        engine = _build_engine(seed=1)
        engine.run(200)
        assert len(engine.mark_log) == 200
        assert [m.t for m in engine.mark_log] == list(range(200))

    def test_agent_state_log_has_one_entry_per_agent_per_tick(self) -> None:
        n_agents = 10
        engine = _build_engine(seed=1, n_agents=n_agents)
        engine.run(50)
        assert len(engine.agent_state_log) == 50 * n_agents

    def test_agent_state_log_pnl_matches_hand_computed_value_at_a_later_tick(self) -> None:
        seller = ScriptedAgent(
            agent_id=1,
            cash=10_000,
            inventory=5,
            script=[[Place(agent_id=1, side=Side.SELL, price=105, qty=5)], [], []],
        )
        buyer = ScriptedAgent(
            agent_id=2,
            cash=10_000,
            inventory=0,
            script=[[], [Place(agent_id=2, side=Side.BUY, price=105, qty=5)], []],
        )
        quoter_a = ScriptedAgent(
            agent_id=3,
            cash=10_000,
            inventory=50,
            script=[[], [], [Place(agent_id=3, side=Side.BUY, price=88, qty=1)]],
        )
        quoter_b = ScriptedAgent(
            agent_id=4,
            cash=10_000,
            inventory=50,
            script=[[], [], [Place(agent_id=4, side=Side.SELL, price=92, qty=1)]],
        )
        engine = Engine([seller, buyer, quoter_a, quoter_b], reference_price=100.0)
        engine.run(3)

        tick2_mark = next(m.mark for m in engine.mark_log if m.t == 2)
        assert tick2_mark == pytest.approx((88 + 92) / 2)

        expected = (buyer.cash - buyer.initial_cash) + (
            buyer.inventory - buyer.initial_inventory
        ) * tick2_mark
        buyer_entry = next(s for s in engine.agent_state_log if s.t == 2 and s.agent_id == 2)
        assert buyer_entry.pnl == pytest.approx(expected)
        assert buyer_entry.pnl != pytest.approx(0.0), "the mark has to actually move pnl here"


class TestExport:
    def test_frames_have_expected_shape_and_columns(self) -> None:
        engine = _build_engine(seed=1, n_agents=8)
        engine.run(300)

        trades = trades_to_frame(engine)
        marks = marks_to_frame(engine)
        states = agent_states_to_frame(engine)

        assert len(trades) == len(engine.trade_log)
        assert list(trades.columns) == [
            "t",
            "seq",
            "price",
            "qty",
            "aggressor_id",
            "resting_id",
            "aggressor_agent",
            "resting_agent",
            "aggressor_side",
        ]
        assert len(marks) == 300
        assert list(marks.columns) == ["t", "mark", "best_bid", "best_ask"]
        assert len(states) == 300 * 8
        assert list(states.columns) == ["t", "agent_id", "cash", "inventory", "pnl"]

    def test_export_round_trips_to_identical_dataframes(self, tmp_path: Path) -> None:
        engine = _build_engine(seed=1, n_agents=8)
        engine.run(300)

        paths = export_run(engine, tmp_path)
        for name, frame_fn in (
            ("trades", trades_to_frame),
            ("marks", marks_to_frame),
            ("agent_states", agent_states_to_frame),
        ):
            original = frame_fn(engine)
            roundtripped = pd.read_parquet(paths[name])
            pd.testing.assert_frame_equal(original, roundtripped)

    def test_same_seed_exports_byte_identical_parquet(self, tmp_path: Path) -> None:
        engine_a = _build_engine(seed=77, n_agents=6)
        engine_a.run(400)
        engine_b = _build_engine(seed=77, n_agents=6)
        engine_b.run(400)

        paths_a = export_run(engine_a, tmp_path / "run_a")
        paths_b = export_run(engine_b, tmp_path / "run_b")

        for name in ("trades", "marks", "agent_states"):
            hash_a = hashlib.sha256(paths_a[name].read_bytes()).hexdigest()
            hash_b = hashlib.sha256(paths_b[name].read_bytes()).hexdigest()
            assert hash_a == hash_b, f"{name}.parquet differs between two runs of the same seed"

    def test_different_seed_exports_different_parquet_hash(self, tmp_path: Path) -> None:
        engine_a = _build_engine(seed=77, n_agents=6)
        engine_a.run(400)
        engine_b = _build_engine(seed=78, n_agents=6)
        engine_b.run(400)

        paths_a = export_run(engine_a, tmp_path / "run_a")
        paths_b = export_run(engine_b, tmp_path / "run_b")

        hash_a = hashlib.sha256(paths_a["marks"].read_bytes()).hexdigest()
        hash_b = hashlib.sha256(paths_b["marks"].read_bytes()).hexdigest()
        assert hash_a != hash_b

    def test_export_of_a_10k_tick_run_completes_quickly(self, tmp_path: Path) -> None:
        engine = _build_engine(seed=9, n_agents=20)
        engine.run(10_000)

        start = time.perf_counter()
        export_run(engine, tmp_path)
        elapsed = time.perf_counter() - start

        assert elapsed < 30.0, f"export took {elapsed:.1f}s -- unexpectedly slow"
