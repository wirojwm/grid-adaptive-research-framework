import pandas as pd
import pytest

from gridadaptive.data.ingest import (
    bar_grid_alignment_check,
    convert_to_close_labeled,
    deduplicate_bars,
    localize_to_utc,
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


def test_convert_close_labeled_is_a_noop():
    df = _bars([(1.1, 1.11, 1.09, 1.1)] * 3)
    out = convert_to_close_labeled(df, source_bar_convention="close", timeframe="5min")
    pd.testing.assert_index_equal(out.index, df.index)


def test_convert_open_labeled_shifts_forward_by_one_bar():
    df = _bars([(1.1, 1.11, 1.09, 1.1)] * 3)
    out = convert_to_close_labeled(df, source_bar_convention="open", timeframe="5min")
    expected = df.index + pd.Timedelta("5min")
    pd.testing.assert_index_equal(out.index, expected)
    assert (out["close"].values == df["close"].values).all()


def test_convert_invalid_convention_raises():
    df = _bars([(1.1, 1.11, 1.09, 1.1)])
    with pytest.raises(ValueError):
        convert_to_close_labeled(df, source_bar_convention="midpoint", timeframe="5min")


def test_bar_grid_alignment_check_passes_clean_grid():
    df = _bars([(1.1, 1.11, 1.09, 1.1)] * 5, freq="5min")
    assert bar_grid_alignment_check(df, timeframe="5min") == []


def test_bar_grid_alignment_check_flags_misaligned_timestamps():
    df = _bars([(1.1, 1.11, 1.09, 1.1)] * 3, freq="5min")
    shifted = df.index.to_series()
    shifted.iloc[1] = shifted.iloc[1] + pd.Timedelta(seconds=37)
    df.index = pd.DatetimeIndex(shifted.values, tz="UTC")

    misaligned = bar_grid_alignment_check(df, timeframe="5min")
    assert len(misaligned) == 1
    assert misaligned[0] == df.index[1]


def test_deduplicate_drops_exact_duplicate_rows():
    idx = pd.DatetimeIndex(["2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z", "2024-01-01T00:05:00Z"])
    df = pd.DataFrame(
        {"open": [1.1, 1.1, 1.2], "high": [1.11, 1.11, 1.21], "low": [1.09, 1.09, 1.19], "close": [1.1, 1.1, 1.2]},
        index=idx,
    )
    cleaned, report = deduplicate_bars(df)
    assert len(cleaned) == 2
    assert report.duplicate_timestamps_dropped == 1
    assert cleaned.index.is_unique


def test_deduplicate_raises_on_conflicting_duplicate_values():
    idx = pd.DatetimeIndex(["2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z"])
    df = pd.DataFrame({"open": [1.1, 1.2], "high": [1.11, 1.21], "low": [1.09, 1.19], "close": [1.1, 1.2]}, index=idx)
    with pytest.raises(ValueError):
        deduplicate_bars(df)


def test_deduplicate_is_noop_when_no_duplicates():
    df = _bars([(1.1, 1.11, 1.09, 1.1)] * 3, freq="5min")
    cleaned, report = deduplicate_bars(df)
    assert len(cleaned) == 3
    assert report.duplicate_timestamps_dropped == 0


def _naive_bars(rows, freq="1min", start="2024-01-03 10:00:00"):
    idx = pd.date_range(start, periods=len(rows), freq=freq)
    return pd.DataFrame(
        {
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
        },
        index=idx,
    )


def test_localize_to_utc_shifts_fixed_offset_naive_timestamps():
    # source local = EST year-round, UTC-5 (fixed, no DST) -> offset = -5h
    df = _naive_bars([(1.1, 1.11, 1.09, 1.1)], start="2024-01-03 10:00:00")
    out = localize_to_utc(df, source_utc_offset=pd.Timedelta(hours=-5))
    assert out.index.tz is not None
    assert out.index[0] == pd.Timestamp("2024-01-03 15:00:00", tz="UTC")


def test_localize_to_utc_raises_if_already_tz_aware():
    df = _bars([(1.1, 1.11, 1.09, 1.1)], freq="1min")
    with pytest.raises(ValueError):
        localize_to_utc(df, source_utc_offset=pd.Timedelta(hours=-5))


def test_localize_to_utc_accepts_dst_observing_iana_zone_winter():
    # EET (typical broker server time convention): UTC+2 in winter
    df = _naive_bars([(1.1, 1.11, 1.09, 1.1)], start="2024-01-10 12:00:00")
    out = localize_to_utc(df, source_utc_offset="EET")
    assert out.index[0] == pd.Timestamp("2024-01-10 10:00:00", tz="UTC")


def test_localize_to_utc_accepts_dst_observing_iana_zone_summer():
    # same zone in summer: UTC+3 (EEST) - proves the DST shift is actually applied,
    # not just a fixed +2h offset silently reused year-round
    df = _naive_bars([(1.1, 1.11, 1.09, 1.1)], start="2024-07-10 12:00:00")
    out = localize_to_utc(df, source_utc_offset="EET")
    assert out.index[0] == pd.Timestamp("2024-07-10 09:00:00", tz="UTC")
