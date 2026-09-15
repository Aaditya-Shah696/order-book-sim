"""LOBSTER loader tests, against the real downloaded sample.

Skipped, not failed, when data/lobster/*.csv is absent: that data is fetched
on demand and never committed, so a fresh clone still passes cleanly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lob_simulator.research.lobster import load_lobster_sample, resample_by_events
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


class TestResampleByEvents:
    def test_bucket_takes_last_mid_and_sums_signed_volume(self) -> None:
        mids = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        volume = np.array([1.0, -1.0, 2.0, 0.0, 0.0, 5.0, 9.0])
        out_mids, out_volume = resample_by_events(mids, volume, events_per_bucket=3)
        assert list(out_mids) == [3.0, 6.0]  # trailing partial bucket dropped
        assert list(out_volume) == [2.0, 5.0]

    def test_bucket_of_one_is_the_identity(self) -> None:
        mids, volume = load_lobster_sample(DATA_DIR)
        out_mids, out_volume = resample_by_events(mids, volume, events_per_bucket=1)
        assert np.array_equal(out_mids, mids)
        assert np.array_equal(out_volume, volume)

    def test_guards(self) -> None:
        with pytest.raises(ValueError):
            resample_by_events(np.zeros(4), np.zeros(4), events_per_bucket=0)
        with pytest.raises(ValueError):
            resample_by_events(np.zeros(4), np.zeros(3), events_per_bucket=2)
        m, v = resample_by_events(np.zeros(2), np.zeros(2), events_per_bucket=5)
        assert len(m) == 0 and len(v) == 0

    def test_event_time_zero_returns_mostly_vanish_once_resampled(self) -> None:
        """About half of per-event mid changes are exactly zero; per-bucket ones are not."""
        mids, _ = load_lobster_sample(DATA_DIR)
        event_zero = np.mean(np.diff(mids) == 0)
        bucket_mids, _ = resample_by_events(mids, np.zeros_like(mids), events_per_bucket=12)
        bucket_zero = np.mean(np.diff(bucket_mids) == 0)
        assert event_zero > 0.3
        assert bucket_zero < event_zero / 2

    def test_resampled_real_tape_still_shows_fat_tails_and_persistent_clustering(self) -> None:
        """The two conclusions that survive aligning the clocks."""
        mids, volume = load_lobster_sample(DATA_DIR)
        r_mids, r_volume = resample_by_events(mids, volume, events_per_bucket=12)
        report = build_report(
            "LOBSTER/12", prices=r_mids, signed_volume=r_volume, acf_lags=25, ofi_window=10
        )
        assert report.pearson_kurtosis > 3.5
        assert report.abs_return_acf_lag20 > 0.03

    def test_bid_ask_bounce_is_an_event_time_artifact(self) -> None:
        """Negative lag-1 return autocorrelation in event time does not survive resampling."""
        mids, volume = load_lobster_sample(DATA_DIR)
        event = build_report("event", prices=mids, signed_volume=volume, acf_lags=5, ofi_window=10)
        r_mids, r_volume = resample_by_events(mids, volume, events_per_bucket=12)
        bucket = build_report(
            "bucket", prices=r_mids, signed_volume=r_volume, acf_lags=5, ofi_window=10
        )
        assert event.return_acf_lag1 < -0.1
        assert bucket.return_acf_lag1 > -0.05
