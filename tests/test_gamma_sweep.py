"""Gamma and skew_k sweep tests: the machinery and the selection rule, on a tiny grid."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from lob_simulator.market import INFORMED, UNINFORMED
from lob_simulator.research.gamma_sweep import (
    DEFAULT_SKEW_KS,
    SELECTION_RULE,
    GammaSweep,
    load_chosen_gamma,
    load_chosen_skew_k,
    sweep_from_dict,
    sweep_gamma,
    sweep_to_dict,
)

SPEC = replace(INFORMED, n_noise=10)
SKEW_KS = (0.05, 0.2)


def _sweep(
    gammas: Sequence[float] = (1e-6, 1e-4),
    seeds: range = range(4),
    skew_ks: Sequence[float] = SKEW_KS,
) -> GammaSweep:
    return sweep_gamma(
        spec=SPEC, gammas=gammas, seeds=seeds, n_ticks=300, sigma=3.2, k=0.6, skew_ks=skew_ks
    )


class TestSweep:
    def test_one_point_per_gamma_one_per_skew_k_plus_naive_baseline(self) -> None:
        sweep = _sweep()
        assert [p.gamma for p in sweep.points] == [1e-6, 1e-4]
        assert all(p.skew_k is None for p in sweep.points)
        assert [p.skew_k for p in sweep.skew_points] == list(SKEW_KS)
        assert all(p.gamma is None for p in sweep.skew_points)
        assert [b.label for b in sweep.baselines] == ["NaiveMM"]
        assert all(b.gamma is None and b.skew_k is None for b in sweep.baselines)
        assert sweep.market == "informed"
        assert sweep.n_seeds == 4
        assert sweep.n_ticks == 300
        assert sweep.selection_rule == SELECTION_RULE

    def test_labels_carry_the_swept_value(self) -> None:
        sweep = _sweep()
        assert [p.label for p in sweep.points] == [
            "AvellanedaStoikovMM(gamma=1e-06)",
            "AvellanedaStoikovMM(gamma=0.0001)",
        ]
        assert [p.label for p in sweep.skew_points] == [
            "InventorySkewMM(skew_k=0.05)",
            "InventorySkewMM(skew_k=0.2)",
        ]

    def test_chosen_gamma_maximises_mean_over_std(self) -> None:
        sweep = _sweep()
        best = max(sweep.points, key=lambda p: p.mean_over_std_pnl)
        assert sweep.chosen_gamma == best.gamma

    def test_chosen_skew_k_maximises_mean_over_std(self) -> None:
        sweep = _sweep()
        best = max(sweep.skew_points, key=lambda p: p.mean_over_std_pnl)
        assert sweep.chosen_skew_k == best.skew_k

    def test_default_skew_grid_is_used_when_none_is_given(self) -> None:
        sweep = sweep_gamma(
            spec=replace(UNINFORMED, n_noise=6), gammas=[1e-5], seeds=range(2), n_ticks=100,
            sigma=3.5, k=0.5,
        )
        assert tuple(p.skew_k for p in sweep.skew_points) == DEFAULT_SKEW_KS

    def test_reproducible(self) -> None:
        assert sweep_to_dict(_sweep()) == sweep_to_dict(_sweep())

    def test_market_name_follows_spec(self) -> None:
        sweep = sweep_gamma(
            spec=replace(UNINFORMED, n_noise=8), gammas=[1e-5], seeds=range(2), n_ticks=200,
            sigma=3.5, k=0.5, skew_ks=[0.1],
        )
        assert sweep.market == "uninformed"

    def test_rejects_empty_or_nonpositive_gammas(self) -> None:
        with pytest.raises(ValueError):
            _sweep(gammas=[])
        with pytest.raises(ValueError):
            _sweep(gammas=[0.0])

    def test_rejects_empty_or_negative_skew_ks(self) -> None:
        with pytest.raises(ValueError):
            _sweep(skew_ks=[])
        with pytest.raises(ValueError):
            _sweep(skew_ks=[-0.1])


class TestSerialization:
    def test_round_trips_through_json_and_load_chosen_values(self, tmp_path: Path) -> None:
        sweep = _sweep()
        path = tmp_path / "gamma_sweep_informed.json"
        with open(path, "w") as f:
            json.dump(sweep_to_dict(sweep), f)
        assert load_chosen_gamma(path) == sweep.chosen_gamma
        assert load_chosen_skew_k(path) == sweep.chosen_skew_k

    def test_sweep_from_dict_inverts_sweep_to_dict(self) -> None:
        sweep = _sweep()
        assert sweep_from_dict(json.loads(json.dumps(sweep_to_dict(sweep)))) == sweep

    def test_as_points_carry_skew_per_unit_at_the_open_and_skew_points_do_not(self) -> None:
        sweep = _sweep()
        d = sweep_to_dict(sweep)
        points = d["points"]
        skew_points = d["skew_points"]
        assert isinstance(points, list) and isinstance(skew_points, list)
        assert points[0]["skew_per_unit_open"] == pytest.approx(1e-6 * 3.2**2 * 300)
        assert "skew_per_unit_open" not in skew_points[0]

    def test_missing_file_fails_loudly(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="run_gamma_sweep"):
            load_chosen_gamma(tmp_path / "nope.json")
        with pytest.raises(FileNotFoundError, match="run_gamma_sweep"):
            load_chosen_skew_k(tmp_path / "nope.json")

    def test_pre_fix_json_without_chosen_skew_k_fails_loudly(self, tmp_path: Path) -> None:
        """A sweep JSON written before skew_k was swept must not silently yield a default."""
        path = tmp_path / "gamma_sweep_informed.json"
        path.write_text(json.dumps({"chosen_gamma": 3e-5}))
        assert load_chosen_gamma(path) == 3e-5
        with pytest.raises(KeyError, match="run_gamma_sweep"):
            load_chosen_skew_k(path)
