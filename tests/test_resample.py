import pandas as pd
import pytest

from gridadaptive.data.ingest import convert_to_close_labeled, deduplicate_bars, localize_to_utc
from gridadaptive.data.resample import ResampleReport, resample_ohlcv
from gridadaptive.data.schema import OHLCVValidationError


def _m1_bars(rows, start, volumes=None):
    """Build a close-labeled, tz-aware UTC M1 fixture. `start` is the label of the
    first bar (each row already represents one whole minute)."""
    idx = pd.date_range(start, periods=len(rows), freq="1min", tz="UTC")
    data = {
        "open": [r[0] for r in rows],
        "high": [r[1] for r in rows],
        "low": [r[2] for r in rows],
        "close": [r[3] for r in rows],
    }
    if volumes is not None:
        data["volume"] = volumes
    return pd.DataFrame(data, index=idx)


def _five_min_window_rows():
    # open, high, low, close per constituent M1 bar, 00:01..00:05 -> M5 bar labeled 00:05
    return [
        (1.1000, 1.1005, 1.0998, 1.1002),
        (1.1002, 1.1010, 1.1001, 1.1008),
        (1.1008, 1.1009, 1.0990, 1.0995),  # lowest low of the window
        (1.0995, 1.1020, 1.0993, 1.1015),  # highest high of the window
        (1.1015, 1.1016, 1.1010, 1.1012),
    ]


def test_normal_five_bar_aggregation():
    df = _m1_bars(_five_min_window_rows(), start="2024-01-03 00:01")
    out, report = resample_ohlcv(df, source_timeframe="1min", target_timeframe="5min")

    assert len(out) == 1
    bar = out.iloc[0]
    assert out.index[0] == pd.Timestamp("2024-01-03 00:05", tz="UTC")
    assert bar["open"] == pytest.approx(1.1000)  # first M1 open
    assert bar["close"] == pytest.approx(1.1012)  # last M1 close
    assert bar["constituent_count"] == 5
    assert report.bars_produced == 1
    assert report.incomplete_windows == []


def test_high_low_taken_across_all_constituents_not_just_endpoints():
    df = _m1_bars(_five_min_window_rows(), start="2024-01-03 00:01")
    out, _ = resample_ohlcv(df, source_timeframe="1min", target_timeframe="5min")
    bar = out.iloc[0]
    assert bar["high"] == pytest.approx(1.1020)  # from the 4th bar, not first/last
    assert bar["low"] == pytest.approx(1.0990)  # from the 3rd bar, not first/last


def test_missing_constituent_bar_mid_window_flagged_incomplete_not_fabricated():
    rows = _five_min_window_rows()
    idx = pd.date_range("2024-01-03 00:01", periods=5, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
        },
        index=idx,
    )
    df = df.drop(index=idx[2])  # drop the 00:03 bar - a real gap, not filled

    out, report = resample_ohlcv(df, source_timeframe="1min", target_timeframe="5min")

    assert len(out) == 1  # still produced from the 4 bars that exist
    bar = out.iloc[0]
    assert bar["constituent_count"] == 4
    assert report.incomplete_windows == [pd.Timestamp("2024-01-03 00:05", tz="UTC")]
    # high/low reflect only the bars actually present (dropped bar had the window low)
    assert bar["low"] == pytest.approx(1.0993)


def test_duplicate_timestamp_in_input_raises_not_silently_resolved():
    df = _m1_bars(_five_min_window_rows(), start="2024-01-03 00:01")
    dup = pd.concat([df, df.iloc[[0]]]).sort_index()
    with pytest.raises(OHLCVValidationError):
        resample_ohlcv(dup, source_timeframe="1min", target_timeframe="5min")


def test_conflicting_duplicate_raised_by_dedup_step_before_resampling():
    idx = pd.DatetimeIndex(
        ["2024-01-03T00:01:00Z", "2024-01-03T00:01:00Z"]
    )
    df = pd.DataFrame(
        {"open": [1.10, 1.11], "high": [1.101, 1.111], "low": [1.099, 1.109], "close": [1.100, 1.110]},
        index=idx,
    )
    with pytest.raises(ValueError):
        deduplicate_bars(df)


def test_incomplete_trailing_window_still_produced_and_flagged():
    rows = _five_min_window_rows()[:3]  # only 3 of the expected 5, at the end of the series
    df = _m1_bars(rows, start="2024-01-03 00:01")
    out, report = resample_ohlcv(df, source_timeframe="1min", target_timeframe="5min")

    assert len(out) == 1
    assert out.iloc[0]["constituent_count"] == 3
    assert out.index[0] == pd.Timestamp("2024-01-03 00:03", tz="UTC")  # last available constituent
    assert report.incomplete_windows == [pd.Timestamp("2024-01-03 00:03", tz="UTC")]


def test_boundary_bar_exactly_on_grid_belongs_to_the_earlier_window():
    rows = _five_min_window_rows() + [(1.1012, 1.1013, 1.1011, 1.1012)]  # 6th bar starts next window
    df = _m1_bars(rows, start="2024-01-03 00:01")

    out, report = resample_ohlcv(df, source_timeframe="1min", target_timeframe="5min")

    assert len(out) == 2
    first_window = out.loc[pd.Timestamp("2024-01-03 00:05", tz="UTC")]
    assert first_window["constituent_count"] == 5  # the 00:05 bar stays in the first window
    second_window = out.loc[pd.Timestamp("2024-01-03 00:06", tz="UTC")]
    assert second_window["constituent_count"] == 1
    assert report.incomplete_windows == [pd.Timestamp("2024-01-03 00:06", tz="UTC")]


def test_target_must_be_a_multiple_of_source_timeframe():
    df = _m1_bars(_five_min_window_rows(), start="2024-01-03 00:01")
    with pytest.raises(ValueError):
        resample_ohlcv(df, source_timeframe="1min", target_timeframe="150s")  # 2.5min, not a whole multiple


def test_volume_summed_across_constituents_when_present():
    df = _m1_bars(_five_min_window_rows(), start="2024-01-03 00:01", volumes=[10, 20, 5, 30, 15])
    out, _ = resample_ohlcv(df, source_timeframe="1min", target_timeframe="5min")
    assert out.iloc[0]["volume"] == 80


def test_volume_column_absent_when_not_supplied_by_source():
    df = _m1_bars(_five_min_window_rows(), start="2024-01-03 00:01")
    out, _ = resample_ohlcv(df, source_timeframe="1min", target_timeframe="5min")
    assert "volume" not in out.columns


def test_timezone_conversion_before_resample_full_pipeline():
    # naive EST (UTC-5, fixed, no DST) source timestamps, close-labeled
    idx = pd.date_range("2024-01-03 05:01:00", periods=5, freq="1min")  # local EST clock
    rows = _five_min_window_rows()
    raw = pd.DataFrame(
        {
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
        },
        index=idx,
    )
    utc = localize_to_utc(raw, source_utc_offset=pd.Timedelta(hours=-5))
    out, _ = resample_ohlcv(utc, source_timeframe="1min", target_timeframe="5min")
    # 05:05 local EST -> 10:05 UTC
    assert out.index[0] == pd.Timestamp("2024-01-03 10:05:00", tz="UTC")


def test_open_label_conversion_before_resample_full_pipeline():
    rows = _five_min_window_rows()
    # source labels each bar by its OPEN time: 00:00..00:04 instead of 00:01..00:05
    open_labeled = _m1_bars(rows, start="2024-01-03 00:00")
    close_labeled = convert_to_close_labeled(open_labeled, source_bar_convention="open", timeframe="1min")

    out, _ = resample_ohlcv(close_labeled, source_timeframe="1min", target_timeframe="5min")
    assert out.index[0] == pd.Timestamp("2024-01-03 00:05", tz="UTC")
    assert out.iloc[0]["constituent_count"] == 5


def test_close_label_conversion_is_noop_before_resample():
    rows = _five_min_window_rows()
    already_close_labeled = _m1_bars(rows, start="2024-01-03 00:01")
    converted = convert_to_close_labeled(already_close_labeled, source_bar_convention="close", timeframe="1min")

    out, _ = resample_ohlcv(converted, source_timeframe="1min", target_timeframe="5min")
    assert out.index[0] == pd.Timestamp("2024-01-03 00:05", tz="UTC")


def test_anti_lookahead_bar_labeled_at_last_not_first_constituent():
    df = _m1_bars(_five_min_window_rows(), start="2024-01-03 00:01")
    out, _ = resample_ohlcv(df, source_timeframe="1min", target_timeframe="5min")
    # the M5 bar's own label must equal its LAST constituent's timestamp (00:05),
    # never its first (00:01) - the earlier label would mean the bar claims to be
    # known before its own high/low/close were actually determined.
    assert out.index[0] == df.index.max()
    assert out.index[0] != df.index.min()


def test_anti_lookahead_no_bar_produced_beyond_available_data():
    # only 2 of 5 minutes into a window exist (e.g. ingestion stopped mid-window) -
    # the resampler must not project/complete a bar past what's actually known.
    rows = _five_min_window_rows()[:2]
    df = _m1_bars(rows, start="2024-01-03 00:01")
    out, report = resample_ohlcv(df, source_timeframe="1min", target_timeframe="5min")
    assert len(out) == 1
    assert out.index[0] == pd.Timestamp("2024-01-03 00:02", tz="UTC")  # last known bar, not 00:05
    assert isinstance(report, ResampleReport)
