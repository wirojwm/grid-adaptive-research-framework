"""OHLCV data contract validation.

This module validates the structural correctness of a single already-loaded
dataset. Point-in-time integrity across the strategy/execution boundary is a
separate concern, enforced by the execution simulator (a decision may only be
evaluated against bars already present in the dataset it was made from).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close")
BAR_CONVENTIONS = ("open", "close")


class OHLCVValidationError(ValueError):
    """Raised when a dataset violates the OHLCV data contract."""


@dataclass
class GapReport:
    expected_interval: pd.Timedelta
    gaps: list[dict] = field(default_factory=list)


def validate_ohlcv(df: pd.DataFrame, timeframe: str, bar_convention: str) -> GapReport:
    """Validate structural correctness and return a (non-raising) gap report.

    Gaps are flagged, not treated as errors: a gap can be a legitimate market
    closure (e.g. a weekend) and must remain visible to the researcher rather
    than being silently dropped or filled.
    """
    if bar_convention not in BAR_CONVENTIONS:
        raise OHLCVValidationError(
            f"bar_convention must be one of {BAR_CONVENTIONS}, got {bar_convention!r}"
        )

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise OHLCVValidationError(f"missing required columns: {missing}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise OHLCVValidationError("index must be a DatetimeIndex")

    if df.index.tz is None:
        raise OHLCVValidationError(
            "index must be tz-aware (canonical timezone is UTC, "
            "by contract); refusing to silently assume a timezone for naive timestamps"
        )

    if not df.index.is_monotonic_increasing:
        raise OHLCVValidationError("timestamps must be sorted ascending")

    if df.index.has_duplicates:
        dupes = df.index[df.index.duplicated()].unique().tolist()
        raise OHLCVValidationError(f"duplicate timestamps found: {dupes}")

    if (df["high"] < df["low"]).any():
        raise OHLCVValidationError("found bar(s) where high < low")

    if (df["high"] < df[["open", "close"]].max(axis=1)).any():
        raise OHLCVValidationError("found bar(s) where high < max(open, close)")

    if (df["low"] > df[["open", "close"]].min(axis=1)).any():
        raise OHLCVValidationError("found bar(s) where low > min(open, close)")

    if "volume" in df.columns and (df["volume"] < 0).any():
        raise OHLCVValidationError("found bar(s) with negative volume")

    expected_interval = pd.Timedelta(timeframe)
    diffs = df.index.to_series().diff().dropna()
    gaps = []
    for ts, delta in diffs.items():
        if delta > expected_interval:
            gaps.append(
                {
                    "prev_timestamp": ts - delta,
                    "next_timestamp": ts,
                    "actual_interval": delta,
                    "expected_interval": expected_interval,
                }
            )

    return GapReport(expected_interval=expected_interval, gaps=gaps)
