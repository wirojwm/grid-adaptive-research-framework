"""Data-ingestion utilities: timestamp-convention conversion and deduplication.

Never silently repairs questionable data — duplicate rows with disagreeing OHLC
values raise rather than picking one, and callers must state the source's bar
convention explicitly (no default) before any timestamp shifting happens.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

_BAR_CONVENTIONS = ("open", "close")


def convert_to_close_labeled(df: pd.DataFrame, source_bar_convention: str, timeframe: str) -> pd.DataFrame:
    """Return a copy of `df` re-indexed to the project's canonical close-labeled
    convention. If the source already labels bars by close, this is a pure copy —
    no shifting happens implicitly either way.
    """
    if source_bar_convention not in _BAR_CONVENTIONS:
        raise ValueError(f"source_bar_convention must be one of {_BAR_CONVENTIONS}, got {source_bar_convention!r}")

    out = df.copy()
    if source_bar_convention == "open":
        out.index = out.index + pd.Timedelta(timeframe)
    return out


def localize_to_utc(df: pd.DataFrame, source_utc_offset: pd.Timedelta | str) -> pd.DataFrame:
    """Convert a naive-timestamp-indexed df (source-local wall-clock time) to
    tz-aware UTC.

    `source_utc_offset` is either:
    - a `pd.Timedelta` — the source's *fixed* offset from UTC (e.g.
      `pd.Timedelta(hours=-5)` for a vendor documented as fixed EST year-round, no
      DST adjustment); or
    - a `str` IANA/tzdata zone name — for a source whose offset shifts with DST
      (e.g. `"EET"` for an MT4/5-style broker server time documented as "GMT+2,
      GMT+3 during daylight savings time"). The zone's own historical DST
      transition rules are applied via the system tzdata, not hand-rolled here.

    The source's actual convention (which of the two, and the exact
    offset/zone) must be verified per vendor,
    never assumed.
    """
    if df.index.tz is not None:
        raise ValueError("index is already tz-aware; localize_to_utc expects naive source-local timestamps")
    out = df.copy()
    if isinstance(source_utc_offset, str):
        out.index = out.index.tz_localize(source_utc_offset).tz_convert("UTC")
    else:
        out.index = (out.index - source_utc_offset).tz_localize("UTC")
    return out


def bar_grid_alignment_check(df: pd.DataFrame, timeframe: str) -> list[pd.Timestamp]:
    """Return timestamps that don't land on the expected periodic grid (e.g., for
    M5, not on a 5-minute clock boundary). Anchored at the Unix epoch, which is
    itself aligned to every standard sub-daily timeframe, so any nonzero remainder
    indicates either a bad timestamp-convention shift or corrupted data — this is a
    heuristic sanity check, not proof that the labeled convention (open vs close)
    is semantically correct; only vendor documentation/spot-checking can confirm that.
    """
    interval = pd.Timedelta(timeframe)
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    offsets = (df.index - epoch) % interval
    misaligned_mask = offsets != pd.Timedelta(0)
    return list(df.index[misaligned_mask])


@dataclass
class DedupReport:
    duplicate_timestamps_dropped: int
    conflicting_timestamps: list[pd.Timestamp] = field(default_factory=list)


def deduplicate_bars(df: pd.DataFrame) -> tuple[pd.DataFrame, DedupReport]:
    """Drop exact-duplicate-value rows sharing a timestamp (keep first).

    Raises if any duplicate-timestamp group has *disagreeing* OHLC values —
    that is questionable data, not a case for silent repair
    "Do not silently repair questionable market data").
    """
    if not df.index.has_duplicates:
        return df.copy(), DedupReport(duplicate_timestamps_dropped=0)

    dup_subset = df[df.index.duplicated(keep=False)]
    conflicting = [ts for ts, group in dup_subset.groupby(level=0) if not (group.nunique() <= 1).all()]
    if conflicting:
        raise ValueError(
            "duplicate timestamps with differing OHLC values found (not silently "
            f"repaired, flagged for manual review): {conflicting}"
        )

    dropped = int(df.index.duplicated(keep="first").sum())
    cleaned = df[~df.index.duplicated(keep="first")]
    return cleaned, DedupReport(duplicate_timestamps_dropped=dropped)
