"""Spread-decomposition tests: the two A-S variants and the paired runner."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from lob_simulator.agent import Snapshot
from lob_simulator.agents.quoting import AvellanedaStoikovMM, NaiveMM
from lob_simulator.market import INFORMED, SUBJECT_AGENT_ID, UNINFORMED
from lob_simulator.research.spread_decomposition import (
    AS_FIXED_SPREAD_LABEL,
    AS_FLAT_SPREAD_LABEL,
    AS_LABEL,
    SKEW_LABEL,
    AvellanedaStoikovFixedSpreadMM,
    AvellanedaStoikovFlatSpreadMM,
    decompose,
    decomposition_to_dict,
    paired_difference,
    run_arm,
)

T = 2000
GAMMA, SIGMA, K = 7e-5, 2.118, 0.547


def _snap(t: int, mark: float) -> Snapshot:
    return Snapshot(t=t, best_bid=None, best_ask=None, mark=mark, my_orders=())


def _full() -> AvellanedaStoikovMM:
    return AvellanedaStoikovMM(
        agent_id=SUBJECT_AGENT_ID, cash=1_000_000, inventory=0, quote_qty=5,
        gamma=GAMMA, sigma=SIGMA, k=K, horizon_t=float(T),
    )


def _fixed(half_spread_ticks: float = 2.0) -> AvellanedaStoikovFixedSpreadMM:
    return AvellanedaStoikovFixedSpreadMM(
        agent_id=SUBJECT_AGENT_ID, cash=1_000_000, inventory=0, quote_qty=5,
        gamma=GAMMA, sigma=SIGMA, k=K, horizon_t=float(T), half_spread_ticks=half_spread_ticks,
    )


def _flat() -> AvellanedaStoikovFlatSpreadMM:
    return AvellanedaStoikovFlatSpreadMM(
        agent_id=SUBJECT_AGENT_ID, cash=1_000_000, inventory=0, quote_qty=5,
        gamma=GAMMA, sigma=SIGMA, k=K, horizon_t=float(T),
    )


def _naive() -> NaiveMM:
    return NaiveMM(
        agent_id=SUBJECT_AGENT_ID, cash=1_000_000, inventory=0, quote_qty=5, half_spread_ticks=2.0
    )


class TestFixedSpreadVariant:
    def test_half_spread_is_the_constant_at_every_t(self) -> None:
        mm = _fixed()
        for t in (0, T // 2, T):
            _, half = mm.quote(_snap(t, 1000.0))
            assert half == 2.0

    def test_reservation_matches_full_as(self) -> None:
        fixed, full = _fixed(), _full()
        fixed.inventory = full.inventory = 12
        for t in (0, 700, T):
            assert fixed.quote(_snap(t, 1000.0))[0] == full.quote(_snap(t, 1000.0))[0]

    def test_quotes_equal_naive_when_flat(self) -> None:
        fixed, naive = _fixed(), _naive()
        for mark in (1000.0, 1000.5):
            s = _snap(0, mark)
            assert fixed.snap(*fixed.quote(s)) == naive.snap(*naive.quote(s))

    def test_rejects_nonpositive_half_spread(self) -> None:
        with pytest.raises(ValueError):
            _fixed(half_spread_ticks=0.0)


class TestFlatSpreadVariant:
    def test_half_spread_is_the_horizon_floor_at_every_t(self) -> None:
        flat, full = _flat(), _full()
        _, floor = full.quote(_snap(T, 1000.0))
        for t in (0, T // 2, T):
            _, half = flat.quote(_snap(t, 1000.0))
            assert half == pytest.approx(floor)
        assert floor == pytest.approx((2.0 / GAMMA) * math.log(1.0 + GAMMA / K) / 2.0)

    def test_full_as_is_wider_than_flat_before_the_horizon(self) -> None:
        flat, full = _flat(), _full()
        assert full.quote(_snap(0, 1000.0))[1] > flat.quote(_snap(0, 1000.0))[1]
        assert full.quote(_snap(T, 1000.0))[1] == pytest.approx(flat.quote(_snap(T, 1000.0))[1])


class TestRunner:
    SEEDS = (5, 6, 7)
    N_TICKS = 150

    def test_arm_reports_the_other_sides_pnl_and_it_sums_to_zero(self) -> None:
        arm = run_arm(_full, label="as", spec=INFORMED, seeds=self.SEEDS, n_ticks=self.N_TICKS)
        assert arm.seeds == self.SEEDS
        assert np.isfinite(arm.informed_pnl).all()
        total = arm.terminal_pnl + arm.informed_pnl + arm.noise_pnl
        assert np.allclose(total, 0.0)

    def test_arm_has_no_informed_pnl_on_the_uninformed_market(self) -> None:
        arm = run_arm(
            _full, label="as", spec=UNINFORMED, seeds=self.SEEDS, n_ticks=self.N_TICKS
        )
        assert np.isnan(arm.informed_pnl).all()
        assert np.allclose(arm.terminal_pnl + arm.noise_pnl, 0.0)

    def test_paired_difference_requires_matched_seeds(self) -> None:
        a = run_arm(_full, label="a", spec=INFORMED, seeds=(1, 2), n_ticks=50)
        b = run_arm(_full, label="b", spec=INFORMED, seeds=(1, 3), n_ticks=50)
        with pytest.raises(ValueError):
            paired_difference(a, b)

    def test_paired_difference_of_an_arm_with_itself_is_zero(self) -> None:
        a = run_arm(_full, label="a", spec=INFORMED, seeds=self.SEEDS, n_ticks=self.N_TICKS)
        d = paired_difference(a, a, n_bootstrap=50)
        assert d.mean_diff == 0.0
        assert d.std_diff == 0.0
        assert d.ci95 == (0.0, 0.0)

    def test_decompose_runs_four_arms_and_three_pairs_and_serialises(self) -> None:
        result = decompose(
            spec=INFORMED,
            seeds=self.SEEDS,
            n_ticks=self.N_TICKS,
            gamma=GAMMA,
            sigma=SIGMA,
            k=K,
            skew_k=0.2,
            n_bootstrap=50,
        )
        assert [a.label for a in result.arms] == [
            SKEW_LABEL, AS_LABEL, AS_FIXED_SPREAD_LABEL, AS_FLAT_SPREAD_LABEL
        ]
        assert [p.label for p in result.paired] == [
            AS_LABEL, AS_FIXED_SPREAD_LABEL, AS_FLAT_SPREAD_LABEL
        ]
        assert all(p.reference == SKEW_LABEL for p in result.paired)
        d = decomposition_to_dict(result)
        text = json.dumps(d, allow_nan=False)
        back = json.loads(text)
        assert back["n_seeds"] == len(self.SEEDS)
        assert back["seeds"] == list(self.SEEDS)
        assert len(back["arms"]) == 4
        assert len(back["arms"][0]["per_seed"]["terminal_pnl"]) == len(self.SEEDS)

    def test_decompose_rejects_empty_seeds(self) -> None:
        with pytest.raises(ValueError):
            decompose(
                spec=INFORMED, seeds=(), n_ticks=10, gamma=GAMMA, sigma=SIGMA, k=K, skew_k=0.2
            )
