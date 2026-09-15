"""FundamentalValue tests."""

from __future__ import annotations

import random

import numpy as np
import pytest

from lob_simulator.fundamental import FundamentalValue
from lob_simulator.market import INFORMED, FundamentalSpec


def _make(seed: int = 0, p_jump: float = 0.5, max_jump_ticks: int = 4) -> FundamentalValue:
    return FundamentalValue(
        initial=1000, rng=random.Random(seed), p_jump=p_jump, max_jump_ticks=max_jump_ticks
    )


class TestPath:
    def test_starts_at_initial(self) -> None:
        assert _make().value_at(0) == 1000

    def test_same_seed_same_path(self) -> None:
        assert _make(3).path(500) == _make(3).path(500)

    def test_different_seeds_diverge(self) -> None:
        assert _make(3).path(500) != _make(4).path(500)

    def test_query_order_does_not_change_the_path(self) -> None:
        """Whoever asks first, and however far ahead, the answer is the same."""
        forward = _make(5)
        forward_values = [forward.value_at(t) for t in range(300)]
        backward = _make(5)
        backward_values = [backward.value_at(t) for t in reversed(range(300))][::-1]
        assert forward_values == backward_values

    def test_path_is_a_prefix_of_a_longer_path(self) -> None:
        short = _make(6).path(100)
        long = _make(6).path(400)
        assert long[:100] == short

    def test_steps_are_bounded_by_max_jump(self) -> None:
        path = np.array(_make(7, max_jump_ticks=4).path(5000))
        assert np.abs(np.diff(path)).max() <= 4

    def test_jump_frequency_matches_p_jump(self) -> None:
        path = np.array(_make(8, p_jump=0.3).path(20_000))
        moved = np.mean(np.diff(path) != 0)
        assert moved == pytest.approx(0.3, abs=0.02)

    def test_zero_p_jump_is_constant(self) -> None:
        assert set(_make(9, p_jump=0.0).path(1000)) == {1000}

    def test_variance_grows_with_horizon(self) -> None:
        """A random walk, not the stationary noise the uninformed market has."""
        ends_short = [_make(s).value_at(50) for s in range(200)]
        ends_long = [_make(s).value_at(1000) for s in range(200)]
        assert np.var(ends_long) > 5 * np.var(ends_short)

    def test_never_below_one(self) -> None:
        f = FundamentalValue(initial=2, rng=random.Random(0), p_jump=1.0, max_jump_ticks=5)
        assert min(f.path(2000)) >= 1


class TestPerTickStd:
    def test_default_spec_is_about_1_94(self) -> None:
        """sqrt(0.5 * mean(1, 4, 9, 16)) = sqrt(3.75)."""
        assert FundamentalSpec().per_tick_std == pytest.approx(3.75**0.5)
        assert INFORMED.fundamental is not None
        assert INFORMED.fundamental.per_tick_std == pytest.approx(1.936, abs=0.001)

    @pytest.mark.parametrize(("p_jump", "max_jump"), [(0.5, 4), (0.3, 1), (1.0, 6)])
    def test_matches_the_simulated_paths_std(self, p_jump: float, max_jump: int) -> None:
        spec = FundamentalSpec(p_jump=p_jump, max_jump_ticks=max_jump)
        path = np.array(_make(11, p_jump=p_jump, max_jump_ticks=max_jump).path(40_000))
        assert np.std(np.diff(path)) == pytest.approx(spec.per_tick_std, rel=0.03)


class TestGuards:
    def test_rejects_bad_parameters(self) -> None:
        rng = random.Random(0)
        with pytest.raises(ValueError):
            FundamentalValue(initial=0, rng=rng, p_jump=0.5, max_jump_ticks=1)
        with pytest.raises(ValueError):
            FundamentalValue(initial=100, rng=rng, p_jump=1.5, max_jump_ticks=1)
        with pytest.raises(ValueError):
            FundamentalValue(initial=100, rng=rng, p_jump=0.5, max_jump_ticks=0)
        with pytest.raises(ValueError):
            _make().value_at(-1)

    def test_empty_path_for_zero_ticks(self) -> None:
        assert _make().path(0) == []
