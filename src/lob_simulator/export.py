"""Export simulation logs to pandas DataFrames and Parquet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .engine import Engine


def trades_to_frame(engine: Engine) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "t": ctx.t,
                "seq": t.seq,
                "price": t.price,
                "qty": t.qty,
                "aggressor_id": t.aggressor_id,
                "resting_id": t.resting_id,
                "aggressor_agent": t.aggressor_agent,
                "resting_agent": t.resting_agent,
                "aggressor_side": ctx.aggressor_side.value,
            }
            for t, ctx in zip(engine.trade_log, engine.trade_context, strict=True)
        ]
    )


def marks_to_frame(engine: Engine) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"t": m.t, "mark": m.mark, "best_bid": m.best_bid, "best_ask": m.best_ask}
            for m in engine.mark_log
        ]
    )


def agent_states_to_frame(engine: Engine) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "t": s.t,
                "agent_id": s.agent_id,
                "cash": s.cash,
                "inventory": s.inventory,
                "pnl": s.pnl,
            }
            for s in engine.agent_state_log
        ]
    )


def export_run(engine: Engine, out_dir: Path) -> dict[str, Path]:
    """Write trades/marks/agent_states to ``<out_dir>/{trades,marks,agent_states}.parquet``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "trades": out_dir / "trades.parquet",
        "marks": out_dir / "marks.parquet",
        "agent_states": out_dir / "agent_states.parquet",
    }
    trades_to_frame(engine).to_parquet(paths["trades"], index=False)
    marks_to_frame(engine).to_parquet(paths["marks"], index=False)
    agent_states_to_frame(engine).to_parquet(paths["agent_states"], index=False)
    return paths
