"""Real-data quality checks.

Every check here flags for manual review — none of them silently repair, drop, or
alter market data. `quality_report` is the single entry point for the QA pass a
sourced dataset should go through before being accepted for a baseline run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from gridadaptive.data.schema import validate_ohlcv

_DEFAULT_WEEKDAY_ANOMALY_THRESHOLD = pd.Timedelta(minutes=30)
_WEEKEND_GAP_MIN_DURATION = pd.Timedelta(hours=40)
_WEEKEND_GAP_MAX_DURATION = pd.Timedelta(hours=76)
_FRIDAY = 4


def classify_gaps(
    gaps: list[dict], weekday_anomaly_threshold: pd.Timedelta = _DEFAULT_WEEKDAY_ANOMALY_THRESHOLD
) -> list[dict]:
    """Add a `"classification"` key to each gap dict from `GapReport.gaps`:
    `"weekend_holiday"` (starts Friday evening, 40-76h duration — the ordinary FX
    weekly close), `"anomalous"` (exceeds `weekday_anomaly_threshold`, not weekend
    shaped — a candidate feed outage or data error), or `"minor"` (small, most
    likely a single missed bar).
    """
    classified = []
    for gap in gaps:
        prev_ts = gap["prev_timestamp"]
        duration = gap["actual_interval"]
        if (
            prev_ts.weekday() == _FRIDAY
            and prev_ts.hour >= 20
            and _WEEKEND_GAP_MIN_DURATION <= duration <= _WEEKEND_GAP_MAX_DURATION
        ):
            classification = "weekend_holiday"
        elif duration > weekday_anomaly_threshold:
            classification = "anomalous"
        else:
            classification = "minor"
        classified.append({**gap, "classification": classification})
    return classified


def check_price_plausibility(df: pd.DataFrame, low_bound: float, high_bound: float) -> list[pd.Timestamp]:
    """Timestamps where any OHLC value falls outside a broad, deliberately wide
    plausibility band — catches gross errors (decimal mis-scaling, corrupted rows),
    not real volatility."""
    out_of_band = (
        (df[["open", "high", "low", "close"]] < low_bound) | (df[["open", "high", "low", "close"]] > high_bound)
    ).any(axis=1)
    return list(df.index[out_of_band])


def check_positive_prices(df: pd.DataFrame) -> list[pd.Timestamp]:
    non_positive = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    return list(df.index[non_positive])


def detect_range_outliers(df: pd.DataFrame, factor: float = 10.0) -> list[pd.Timestamp]:
    """Timestamps whose (high - low) range is a large multiple of the dataset's
    median range. Flags for manual review only — real volatility events (Brexit,
    COVID, etc.) are expected to appear here alongside genuine bad ticks; this
    check cannot and does not distinguish between them."""
    ranges = df["high"] - df["low"]
    median_range = ranges.median()
    if median_range <= 0:
        return []
    outliers = ranges > factor * median_range
    return list(df.index[outliers])


@dataclass
class QualityReport:
    gaps: list[dict]
    price_plausibility_violations: list[pd.Timestamp] = field(default_factory=list)
    non_positive_price_violations: list[pd.Timestamp] = field(default_factory=list)
    range_outliers: list[pd.Timestamp] = field(default_factory=list)

    @property
    def has_material_anomalies(self) -> bool:
        anomalous_gaps = any(g["classification"] == "anomalous" for g in self.gaps)
        return (
            anomalous_gaps
            or bool(self.price_plausibility_violations)
            or bool(self.non_positive_price_violations)
            or bool(self.range_outliers)
        )


def quality_report(
    df: pd.DataFrame,
    timeframe: str,
    bar_convention: str,
    price_bounds: tuple[float, float],
    range_outlier_factor: float = 10.0,
    weekday_anomaly_threshold: pd.Timedelta = _DEFAULT_WEEKDAY_ANOMALY_THRESHOLD,
) -> QualityReport:
    gap_report = validate_ohlcv(df, timeframe=timeframe, bar_convention=bar_convention)
    low, high = price_bounds
    return QualityReport(
        gaps=classify_gaps(gap_report.gaps, weekday_anomaly_threshold=weekday_anomaly_threshold),
        price_plausibility_violations=check_price_plausibility(df, low, high),
        non_positive_price_violations=check_positive_prices(df),
        range_outliers=detect_range_outliers(df, factor=range_outlier_factor),
    )
