"""AvellanedaStoikovMM tests.

K_HAT and SIGMA_HAT come from scripts/run_calibration.py, averaged over
several seeds. GAMMA was chosen by sweeping against those constants.
"""

from __future__ import annotations

import statistics

import pytest

from lob_simulator.agent import Snapshot
from lob_simulator.agents.quoting import AvellanedaStoikovMM, NaiveMM
from lob_simulator.agents.zero_intelligence import ZeroIntelligenceAgent
from lob_simulator.calibration import SESSION_HORIZON_T, estimate_sigma, fit_fill_intensity
from lob_simulator.engine import Engine
from lob_simulator.seeding import spawn_rngs

REFERENCE_PRICE = 100
N_NOISE = 20
T = SESSION_HORIZON_T

GAMMA = 0.2
K_HAT = 0.4986
SIGMA_HAT = 0.03554

FLAT_SNAPSHOT = Snapshot(t=0, best_bid=None, best_ask=None, mark=100.0, my_orders=())


def _make_as_mm(agent_id: int = 0, gamma: float = GAMMA) -> AvellanedaStoikovMM:
    return AvellanedaStoikovMM(
        agent_id=agent_id,
        cash=1_000_000,
        inventory=0,
        quote_qty=5,
        gamma=gamma,
        sigma=SIGMA_HAT,
        k=K_HAT,
        horizon_t=float(T),
    )


def _make_noise_agents(n: int, seed: int) -> list[ZeroIntelligenceAgent]:
    rngs = spawn_rngs(seed, n)
    return [
        ZeroIntelligenceAgent(
            agent_id=i + 1,
            cash=1_000_000,
            inventory=1000,
            reference_price=REFERENCE_PRICE,
            rng=rngs[i],
        )
        for i in range(n)
    ]


class TestCalibrationConstantsAreReal:
    """Re-derive the constants above and confirm they land in the same ballpark."""

    def test_recalibrating_reproduces_the_hardcoded_constants(self) -> None:
        fit = fit_fill_intensity(
            [1, 2, 3, 4, 5, 6, 8, 10, 12, 15], n_ticks=T, n_noise=N_NOISE, seed=11
        )
        assert fit.r_squared > 0.9
        assert fit.k == pytest.approx(K_HAT, rel=0.3)

        engine = Engine(
            _make_noise_agents(N_NOISE, seed=11), reference_price=float(REFERENCE_PRICE)
        )
        engine.run(T)
        sigma = estimate_sigma([m.mark for m in engine.mark_log], window=100)
        assert sigma == pytest.approx(SIGMA_HAT, rel=0.3)


class TestReservationPriceConvergesToMid:
    def test_at_t_equals_T_reservation_equals_mid_regardless_of_inventory(self) -> None:
        mm = _make_as_mm()
        snapshot_at_horizon = Snapshot(t=T, best_bid=None, best_ask=None, mark=105.0, my_orders=())
        for inv in (-80, -1, 0, 1, 80):
            mm.inventory = inv
            center, _ = mm.quote(snapshot_at_horizon)
            assert center == pytest.approx(105.0)

    def test_reservation_moves_monotonically_toward_mid_as_t_increases(self) -> None:
        mm = _make_as_mm()
        mm.inventory = 40
        distances = []
        for t in (0, T // 4, T // 2, (3 * T) // 4, T):
            snap = Snapshot(t=t, best_bid=None, best_ask=None, mark=100.0, my_orders=())
            center, _ = mm.quote(snap)
            distances.append(abs(center - 100.0))
        assert distances == sorted(distances, reverse=True)
        assert distances[-1] == pytest.approx(0.0)

    def test_spread_narrows_as_t_increases(self) -> None:
        mm = _make_as_mm()
        spreads = []
        for t in (0, T // 2, T):
            snap = Snapshot(t=t, best_bid=None, best_ask=None, mark=100.0, my_orders=())
            _, half_spread = mm.quote(snap)
            spreads.append(half_spread)
        assert spreads == sorted(spreads, reverse=True)


class TestInventorySkew:
    def test_positive_inventory_shifts_reservation_below_mid(self) -> None:
        mm = _make_as_mm()
        mm.inventory = 50
        center, _ = mm.quote(FLAT_SNAPSHOT)
        assert center < FLAT_SNAPSHOT.mark

    def test_negative_inventory_shifts_reservation_above_mid(self) -> None:
        mm = _make_as_mm()
        mm.inventory = -50
        center, _ = mm.quote(FLAT_SNAPSHOT)
        assert center > FLAT_SNAPSHOT.mark

    def test_zero_inventory_reservation_equals_mid(self) -> None:
        mm = _make_as_mm()
        mm.inventory = 0
        center, _ = mm.quote(FLAT_SNAPSHOT)
        assert center == pytest.approx(FLAT_SNAPSHOT.mark)


class TestConstructorGuards:
    def test_rejects_nonpositive_gamma(self) -> None:
        with pytest.raises(ValueError):
            _make_as_mm(gamma=0.0)
        with pytest.raises(ValueError):
            _make_as_mm(gamma=-1.0)


class TestQualitativeReproduction:
    """Tighter terminal inventory and lower PnL variance than NaiveMM, at some cost in mean PnL."""

    def test_tighter_inventory_lower_pnl_variance_some_pnl_cost_vs_naive(self) -> None:
        seeds = range(400, 425)

        def run(mm: NaiveMM | AvellanedaStoikovMM, seed: int) -> tuple[int, float]:
            engine = Engine(
                [mm, *_make_noise_agents(N_NOISE, seed)], reference_price=float(REFERENCE_PRICE)
            )
            engine.run(T)
            return mm.inventory, mm.pnl(engine.mark)

        naive_inv, naive_pnl, as_inv, as_pnl = [], [], [], []
        for seed in seeds:
            inv, pnl = run(
                NaiveMM(
                    agent_id=0, cash=1_000_000, inventory=0, quote_qty=5, half_spread_ticks=2.0
                ),
                seed,
            )
            naive_inv.append(inv)
            naive_pnl.append(pnl)
            inv, pnl = run(_make_as_mm(), seed)
            as_inv.append(inv)
            as_pnl.append(pnl)

        naive_mean_abs_inv = statistics.mean(abs(x) for x in naive_inv)
        as_mean_abs_inv = statistics.mean(abs(x) for x in as_inv)
        naive_pnl_std = statistics.pstdev(naive_pnl)
        as_pnl_std = statistics.pstdev(as_pnl)
        naive_mean_pnl = statistics.mean(naive_pnl)
        as_mean_pnl = statistics.mean(as_pnl)

        assert as_mean_abs_inv < naive_mean_abs_inv, (
            f"A-S mean|inv|={as_mean_abs_inv:.1f} not tighter than Naive={naive_mean_abs_inv:.1f}"
        )
        assert as_pnl_std < naive_pnl_std, (
            f"A-S PnL std={as_pnl_std:.1f} not lower than Naive={naive_pnl_std:.1f}"
        )
        assert as_mean_pnl < naive_mean_pnl, (
            f"A-S mean PnL={as_mean_pnl:.1f} should cost something vs Naive={naive_mean_pnl:.1f}"
        )
