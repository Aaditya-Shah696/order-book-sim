"""LOBSTER loader tests, against the real downloaded sample.

Skipped, not failed, when data/lobster/*.csv is absent: that data is fetched
on demand and never committed, so a fresh clone still passes cleanly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lob_simulator.research.lobster import load_lobster_sample
from lob_simulator.research.stylized_facts import build_report

DATA_DIR = Path(__file__).parent.parent / "data" / "lobster"
HAS_DATA = (DATA_DIR / "message.csv").exists() and (DATA_DIR / "orderbook.csv").exists()

pytestmark = pytest.mark.skipif(
    not HAS_DATA, reason="data/lobster/*.csv not present -- run scripts/fetch_lobster_sample.py"
)


class TestLoadLobsterSample:
    def test_returns_equal_length_arrays(self) -> None:
        mids, signed_volume = load_lobster_sample(DATA_DIR)
        assert len(mids) == len(signed_volume)
        assert len(mids) > 0

    def test_mid_prices_are_plausible_equity_prices(self) -> None:
        """AAPL traded $577-588 on 2012-06-21, before the 2014 7-for-1 split."""
        mids, _ = load_lobster_sample(DATA_DIR)
        assert np.all(mids > 100)
        assert np.all(mids < 10_000)
        assert 570 < mids.mean() < 600

    def test_no_sentinel_prices_survive(self) -> None:
        mids, _ = load_lobster_sample(DATA_DIR)
        assert np.all(np.abs(mids) < 100_000)

    def test_signed_volume_has_both_directions_and_is_mostly_zero(self) -> None:
        _, signed_volume = load_lobster_sample(DATA_DIR)
        assert (signed_volume > 0).any()
        assert (signed_volume < 0).any()
        assert (signed_volume == 0).sum() > (signed_volume != 0).sum()

    def test_mismatched_file_lengths_raise(self, tmp_path: Path) -> None:
        (tmp_path / "message.csv").write_text("1.0,4,1,10,100000,1\n2.0,4,2,10,100000,1\n")
        (tmp_path / "orderbook.csv").write_text("100100,10,99900,10\n")
        with pytest.raises(ValueError):
            load_lobster_sample(tmp_path)


class TestLobsterStylizedFactReport:
    def test_report_builds_without_error(self) -> None:
        mids, signed_volume = load_lobster_sample(DATA_DIR)
        report = build_report(
            "LOBSTER AAPL 2012-06-21",
            prices=mids,
            signed_volume=signed_volume,
            acf_lags=25,
            ofi_window=10,
        )
        assert report.n_observations == len(mids) - 1

    def test_real_market_shows_fat_tails_and_negative_lag1_bounce(self) -> None:
        mids, signed_volume = load_lobster_sample(DATA_DIR)
        report = build_report(
            "LOBSTER", prices=mids, signed_volume=signed_volume, acf_lags=25, ofi_window=10
        )
        assert report.pearson_kurtosis > 3.0
        assert report.return_acf_lag1 < 0

    def test_real_market_volatility_clustering_persists_further_than_simulated(self) -> None:
        mids, signed_volume = load_lobster_sample(DATA_DIR)
        report = build_report(
            "LOBSTER", prices=mids, signed_volume=signed_volume, acf_lags=25, ofi_window=10
        )
        assert report.abs_return_acf_lag20 > 0.05, (
            "expected persistent volatility clustering in real data, got lag-20 "
            f"ACF={report.abs_return_acf_lag20:.3f}"
        )
