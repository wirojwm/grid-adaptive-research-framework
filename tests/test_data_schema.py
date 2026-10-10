import pandas as pd
import pytest

from gridadaptive.data.schema import OHLCVValidationError, validate_ohlcv


def _valid_df():
    idx = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": [1.10, 1.101, 1.102],
            "high": [1.105, 1.106, 1.107],
            "low": [1.095, 1.096, 1.097],
            "close": [1.101, 1.102, 1.103],
            "volume": [100, 110, 120],
        },
        index=idx,
    )


def test_valid_ohlcv_passes_with_no_gaps():
    report = validate_ohlcv(_valid_df(), timeframe="1h", bar_convention="close")
    assert report.gaps == []


def test_missing_column_rejected():
    df = _valid_df().drop(columns=["close"])
    with pytest.raises(OHLCVValidationError):
        validate_ohlcv(df, timeframe="1h", bar_convention="close")


def test_tz_naive_index_rejected():
    df = _valid_df()
    df.index = df.index.tz_localize(None)
    with pytest.raises(OHLCVValidationError):
        validate_ohlcv(df, timeframe="1h", bar_convention="close")


def test_non_monotonic_index_rejected():
    df = _valid_df().iloc[[1, 0, 2]]
    with pytest.raises(OHLCVValidationError):
        validate_ohlcv(df, timeframe="1h", bar_convention="close")


def test_duplicate_timestamp_rejected():
    df = _valid_df()
    df.index = pd.DatetimeIndex([df.index[0], df.index[0], df.index[2]])
    with pytest.raises(OHLCVValidationError):
        validate_ohlcv(df, timeframe="1h", bar_convention="close")


def test_high_below_low_rejected():
    df = _valid_df()
    df.iloc[0, df.columns.get_loc("high")] = 1.0
    with pytest.raises(OHLCVValidationError):
        validate_ohlcv(df, timeframe="1h", bar_convention="close")


def test_negative_volume_rejected():
    df = _valid_df()
    df.iloc[0, df.columns.get_loc("volume")] = -5
    with pytest.raises(OHLCVValidationError):
        validate_ohlcv(df, timeframe="1h", bar_convention="close")


def test_invalid_bar_convention_rejected():
    with pytest.raises(OHLCVValidationError):
        validate_ohlcv(_valid_df(), timeframe="1h", bar_convention="midpoint")


def test_gap_is_flagged_not_raised():
    idx = pd.DatetimeIndex(
        ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z", "2024-01-01T03:00:00Z"]
    )
    df = pd.DataFrame(
        {
            "open": [1.1, 1.1, 1.1],
            "high": [1.11, 1.11, 1.11],
            "low": [1.09, 1.09, 1.09],
            "close": [1.1, 1.1, 1.1],
        },
        index=idx,
    )
    report = validate_ohlcv(df, timeframe="1h", bar_convention="close")
    assert len(report.gaps) == 1
    gap = report.gaps[0]
    assert gap["expected_interval"] == pd.Timedelta("1h")
    assert gap["actual_interval"] == pd.Timedelta("2h")
