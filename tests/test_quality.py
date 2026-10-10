import pandas as pd
import pytest

from gridadaptive.data.quality import (
    QualityReport,
    check_positive_prices,
    check_price_plausibility,
    classify_gaps,
    detect_range_outliers,
    quality_report,
)


def _bars(rows, freq="5min", start="2024-01-01"):
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
        },
        index=idx,
    )


def _gap(prev, nxt):
    return {
        "prev_timestamp": prev,
        "next_timestamp": nxt,
        "actual_interval": nxt - prev,
        "expected_interval": pd.Timedelta("5min"),
    }


def test_classify_gaps_identifies_weekend_pattern():
    prev = pd.Timestamp("2024-01-05T21:00:00Z")  # Friday
    nxt = pd.Timestamp("2024-01-07T22:00:00Z")  # Sunday, ~49h later
    classified = classify_gaps([_gap(prev, nxt)])
    assert classified[0]["classification"] == "weekend_holiday"


def test_classify_gaps_flags_anomalous_midweek_gap():
    prev = pd.Timestamp("2024-01-03T10:00:00Z")  # Wednesday
    nxt = prev + pd.Timedelta(hours=3)
    classified = classify_gaps([_gap(prev, nxt)])
    assert classified[0]["classification"] == "anomalous"


def test_classify_gaps_treats_small_gap_as_minor():
    prev = pd.Timestamp("2024-01-03T10:00:00Z")
    nxt = prev + pd.Timedelta(minutes=10)
    classified = classify_gaps([_gap(prev, nxt)])
    assert classified[0]["classification"] == "minor"


def test_check_price_plausibility_flags_out_of_band_bar():
    df = _bars([(1.10, 1.11, 1.09, 1.10), (11.0, 11.1, 10.9, 11.0)])
    flagged = check_price_plausibility(df, low_bound=0.95, high_bound=1.60)
    assert flagged == [df.index[1]]


def test_check_positive_prices_flags_non_positive():
    df = _bars([(1.10, 1.11, 1.09, 1.10), (0.0, 1.1, -0.1, 1.0)])
    assert check_positive_prices(df) == [df.index[1]]


def test_detect_range_outliers_flags_spike():
    rows = [(1.10, 1.101, 1.099, 1.10)] * 5 + [(1.10, 1.30, 0.90, 1.10)]
    df = _bars(rows)
    flagged = detect_range_outliers(df, factor=5.0)
    assert flagged == [df.index[-1]]


def test_detect_range_outliers_handles_zero_median_gracefully():
    df = _bars([(1.10, 1.10, 1.10, 1.10)] * 3)
    assert detect_range_outliers(df) == []


def test_quality_report_bundles_everything_and_flags_nothing_for_clean_data():
    df = _bars([(1.10, 1.11, 1.09, 1.10)] * 3)
    report = quality_report(df, timeframe="5min", bar_convention="close", price_bounds=(0.95, 1.60))
    assert isinstance(report, QualityReport)
    assert report.has_material_anomalies is False


def test_quality_report_flags_material_anomaly():
    rows = [(1.10, 1.11, 1.09, 1.10)] * 3 + [(11.0, 11.1, 10.9, 11.0)]
    df = _bars(rows)
    report = quality_report(df, timeframe="5min", bar_convention="close", price_bounds=(0.95, 1.60))
    assert report.has_material_anomalies is True
    assert len(report.price_plausibility_violations) == 1
