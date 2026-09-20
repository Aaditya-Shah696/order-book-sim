"""AvellanedaStoikovMM tests.

The *_HAT constants are what ``calibration.calibrate`` measures on each
market on seed 999,999 (scripts/run_calibration.py, results/calibration.json).
SIGMA_HAT is the one-tick figure, mostly bid-ask bounce and kept for
reporting; SIGMA_DIFFUSIVE_HAT is the 50-tick figure A-S is actually fed,
in ticks per sqrt(tick). GAMMA is the value scripts/run_gamma_sweep.py chose
for each market by its stated rule with that sigma.
"""

from __future__ import annotations

import statistics

import pytest

from lob_simulator.agent import Snapshot
from lob_simulator.agents.quoting import AvellanedaStoikovMM, NaiveMM
from lob_simulator.calibration import SESSION_HORIZON_T, calibrate, estimate_sigma
from lob_simulator.market import INFORMED, SUBJECT_AGENT_ID, UNINFORMED, MarketSpec, build_market
from lob_simulator.research.inventory_path import run_inventory_paths

T = SESSION_HORIZON_T

UNINFORMED_K_HAT, UNINFORMED_SIGMA_HAT, UNINFORMED_SIGMA_DIFFUSIVE_HAT = 0.465, 3.556, 0.703
UNINFORMED_GAMMA = 5e-5
INFORMED_K_HAT, INFORMED_SIGMA_HAT, INFORMED_SIGMA_DIFFUSIVE_HAT = 0.547, 3.250, 2.118
INFORMED_GAMMA = 7e-5

FLAT_SNAPSHOT = Snapshot(t=0, best_bid=None, best_ask=None, mark=100.0, my_orders=())


def _make_as_mm(
    gamma: float = INFORMED_GAMMA,
    sigma: float = INFORMED_SIGMA_DIFFUSIVE_HAT,
    k: float = INFORMED_K_HAT,
) -> AvellanedaStoikovMM:
    return AvellanedaStoikovMM(
        agent_id=SUBJECT_AGENT_ID,
        cash=1_000_000,
        inventory=0,
        quote_qty=5,
        gamma=gamma,
        sigma=sigma,
        k=k,
        horizon_t=float(T),
    )


def _make_naive() -> NaiveMM:
    return NaiveMM(
        agent_id=SUBJECT_AGENT_ID, cash=1_000_000, inventory=0, quote_qty=5, half_spread_ticks=2.0
    )


class TestCalibrationConstantsAreReal:
    """Re-derive the constants above on a different seed and land in the same ballpark."""

    @pytest.mark.parametrize(
        ("spec", "k_hat", "sigma_hat", "sigma_diffusive_hat"),
        [
            (UNINFORMED, UNINFORMED_K_HAT, UNINFORMED_SIGMA_HAT, UNINFORMED_SIGMA_DIFFUSIVE_HAT),
            (INFORMED, INFORMED_K_HAT, INFORMED_SIGMA_HAT, INFORMED_SIGMA_DIFFUSIVE_HAT),
        ],
        ids=["uninformed", "informed"],
    )
    def test_recalibrating_reproduces_the_hardcoded_constants(
        self, spec: MarketSpec, k_hat: float, sigma_hat: float, sigma_diffusive_hat: float
    ) -> None:
        cal = calibrate(spec, seed=11)
        assert cal.r_squared > 0.9
        assert cal.k == pytest.approx(k_hat, rel=0.3)
        assert cal.sigma == pytest.approx(sigma_hat, rel=0.3)
        assert cal.sigma_diffusive == pytest.approx(sigma_diffusive_hat, rel=0.3)

    @pytest.mark.parametrize("spec", [UNINFORMED, INFORMED], ids=["uninformed", "informed"])
    def test_sigma_is_in_price_units_not_log_units(self, spec: MarketSpec) -> None:
        """Price-unit sigma is the log-return sigma times the price level, to
        first order. Feeding the log figure to A-S would rescale gamma by the
        square of that level: 10^4 at a price of 100, 10^6 at 1000."""
        engine = build_market(spec, seed=11).engine
        engine.run(T)
        marks = [m.mark for m in engine.mark_log]
        price_sigma = estimate_sigma(marks, units="price")
        log_sigma = estimate_sigma(marks, units="log")
        level = statistics.median(marks)
        assert level > 50
        assert price_sigma == pytest.approx(log_sigma * level, rel=0.05)

    def test_diffusive_sigma_is_well_below_the_one_tick_sigma_on_both_markets(self) -> None:
        """The one-tick figure is mostly bounce: on the uninformed market
        nothing diffuses at all, on the informed market only the fundamental
        does, and it moves less than the mark jitters."""
        assert UNINFORMED_SIGMA_DIFFUSIVE_HAT < 0.25 * UNINFORMED_SIGMA_HAT
        assert INFORMED_SIGMA_DIFFUSIVE_HAT < 0.7 * INFORMED_SIGMA_HAT
        assert INFORMED.fundamental is not None
        assert INFORMED_SIGMA_DIFFUSIVE_HAT == pytest.approx(
            INFORMED.fundamental.per_tick_std, rel=0.15
        )


class TestReservationPriceConvergesToMid:
    def test_at_t_equals_T_reservation_equals_mid_regardless_of_inventory(self) -> None:
        mm = _make_as_mm()
        mm.inventory = 500
        snapshot = Snapshot(t=T, best_bid=None, best_ask=None, mark=100.0, my_orders=())
        center, _ = mm.quote(snapshot)
        assert center == pytest.approx(100.0)
        assert mm.skew_per_unit(T) == 0.0

    def test_reservation_moves_monotonically_toward_mid_as_t_increases(self) -> None:
        mm = _make_as_mm()
        mm.inventory = 50
        gaps = []
        for t in (0, T // 4, T // 2, 3 * T // 4, T):
            snapshot = Snapshot(t=t, best_bid=None, best_ask=None, mark=100.0, my_orders=())
            center, _ = mm.quote(snapshot)
            gaps.append(abs(100.0 - center))
        assert gaps == sorted(gaps, reverse=True)
        assert gaps[0] > 0

    def test_spread_narrows_as_t_increases(self) -> None:
        mm = _make_as_mm()
        spreads = []
        for t in (0, T // 2, T):
            snapshot = Snapshot(t=t, best_bid=None, best_ask=None, mark=100.0, my_orders=())
            _, half = mm.quote(snapshot)
            spreads.append(half)
        assert spreads[0] > spreads[1] > spreads[2] > 0

    def test_spread_floor_is_the_gamma_to_zero_limit(self) -> None:
        """At t = T the spread is (2/gamma) ln(1 + gamma/k), which tends to 2/k."""
        _, half_at_T = _make_as_mm().quote(
            Snapshot(t=T, best_bid=None, best_ask=None, mark=100.0, my_orders=())
        )
        assert 2 * half_at_T == pytest.approx(2.0 / INFORMED_K_HAT, rel=0.01)

    def test_spread_floor_alone_snaps_to_naive_quotes(self) -> None:
        """2/k = 3.66 ticks, half 1.83: snapped outward that is the same bid
        and ask NaiveMM's 2.0 half-spread gives on every mark. This is the
        spread at t = T, where the variance term has decayed to zero."""
        mm = _make_as_mm()
        naive = _make_naive()
        for mark in (100.0, 100.5, 1000.0, 1000.5):
            snapshot = Snapshot(t=T, best_bid=None, best_ask=None, mark=mark, my_orders=())
            assert mm.snap(*mm.quote(snapshot)) == naive.snap(*naive.quote(snapshot))

    def test_variance_term_widens_the_quote_by_a_tick_for_the_first_half_of_the_session(
        self,
    ) -> None:
        """The spread also carries gamma*sigma^2*(T-t). At the chosen gamma
        the half-spread is 2.14 at the open and crosses below 2.0 only
        around tick 900, so until then A-S quotes a tick wider on each side
        than NaiveMM at every integer mark (half-integer marks snap the
        same either way). An earlier version of this file asserted the
        spread term was inert; it is only inert in the t -> T limit."""
        mm = _make_as_mm()
        naive = _make_naive()
        flat_snapshot = Snapshot(t=T, best_bid=None, best_ask=None, mark=0.0, my_orders=())
        floor_half = mm.quote(flat_snapshot)[1]
        crossover = T - (2 * 2.0 - 2 * floor_half) / (mm.gamma * mm.sigma**2)
        assert 850 < crossover < 1000

        _, half_at_open = mm.quote(FLAT_SNAPSHOT)
        assert 2.1 < half_at_open < 2.2

        for t, mark, wider in ((0, 1000.0, True), (500, 1000.0, True), (1500, 1000.0, False)):
            snapshot = Snapshot(t=t, best_bid=None, best_ask=None, mark=mark, my_orders=())
            as_bid, as_ask = mm.snap(*mm.quote(snapshot))
            nv_bid, nv_ask = naive.snap(*naive.quote(snapshot))
            if wider:
                assert (as_bid, as_ask) == (nv_bid - 1, nv_ask + 1)
            else:
                assert (as_bid, as_ask) == (nv_bid, nv_ask)
        for t in (0, 500, 1500):
            snapshot = Snapshot(t=t, best_bid=None, best_ask=None, mark=1000.5, my_orders=())
            assert mm.snap(*mm.quote(snapshot)) == naive.snap(*naive.quote(snapshot))


class TestInventorySkew:
    def test_positive_inventory_shifts_reservation_below_mid(self) -> None:
        mm = _make_as_mm()
        mm.inventory = 10
        center, _ = mm.quote(FLAT_SNAPSHOT)
        assert center < 100.0

    def test_negative_inventory_shifts_reservation_above_mid(self) -> None:
        mm = _make_as_mm()
        mm.inventory = -10
        center, _ = mm.quote(FLAT_SNAPSHOT)
        assert center > 100.0

    def test_zero_inventory_reservation_equals_mid(self) -> None:
        center, _ = _make_as_mm().quote(FLAT_SNAPSHOT)
        assert center == pytest.approx(100.0)

    def test_skew_per_unit_is_gamma_sigma_squared_tau_in_ticks(self) -> None:
        mm = _make_as_mm()
        assert mm.skew_per_unit(0) == pytest.approx(
            INFORMED_GAMMA * INFORMED_SIGMA_DIFFUSIVE_HAT**2 * T
        )
        mm.inventory = 7
        center, _ = mm.quote(FLAT_SNAPSHOT)
        assert 100.0 - center == pytest.approx(7 * mm.skew_per_unit(0))

    def test_chosen_gamma_gives_a_sub_tick_skew_per_unit_at_the_open(self) -> None:
        """Past about a tick per unit the agent's own quotes cross the mid
        after one 5-unit fill and it dumps inventory below fair value; the
        sweep's chosen gamma sits well inside that on both markets."""
        informed = _make_as_mm().skew_per_unit(0)
        uninformed = _make_as_mm(
            gamma=UNINFORMED_GAMMA, sigma=UNINFORMED_SIGMA_DIFFUSIVE_HAT, k=UNINFORMED_K_HAT
        ).skew_per_unit(0)
        assert 0.3 < informed < 1.0
        assert 0.0 < uninformed < 0.1


class TestConstructorGuards:
    def test_rejects_nonpositive_gamma(self) -> None:
        with pytest.raises(ValueError):
            _make_as_mm(gamma=0.0)
        with pytest.raises(ValueError):
            _make_as_mm(gamma=-1.0)


class TestQualitativeReproduction:
    """Avellaneda-Stoikov's own result, on the market it is built for.

    On the informed market the mid diffuses, so inventory carries real risk.
    There A-S at the swept gamma holds inventory far tighter than NaiveMM and
    has a far narrower PnL distribution. How it compares with the
    constant-skew heuristic is a *result*, not an invariant: the sweep in
    results/gamma_sweep_informed.json tunes both by the same rule on the same
    seeds and README Results 3 and 4 report what came out. Nothing about that
    ordering is asserted here.
    """

    SEEDS = range(400, 430)

    def test_tighter_inventory_and_lower_pnl_variance_than_naive(self) -> None:
        naive = run_inventory_paths(
            _make_naive, strategy_name="Naive", spec=INFORMED, seeds=self.SEEDS, n_ticks=T
        )
        as_paths = run_inventory_paths(
            _make_as_mm, strategy_name="A-S", spec=INFORMED, seeds=self.SEEDS, n_ticks=T
        )
        assert as_paths.mean_abs_inventory_over(T // 4, 3 * T // 4) < 0.25 * (
            naive.mean_abs_inventory_over(T // 4, 3 * T // 4)
        )
        assert as_paths.mean_abs_inventory_at(T - 1) < naive.mean_abs_inventory_at(T - 1)
        assert statistics.pstdev(as_paths.terminal_pnl) < 0.25 * statistics.pstdev(
            naive.terminal_pnl
        )
