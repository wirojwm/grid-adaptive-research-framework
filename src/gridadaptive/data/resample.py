"""Canonical, source-agnostic OHLCV bar resampling — aggregating finer bars into
coarser ones (e.g. M1 -> M5).

This module has no knowledge of any vendor's file format, timezone, or bar-labeling
convention. Callers must first parse and normalize source data into the project's
canonical form — tz-aware UTC, close-labeled, deduplicated — using
`gridadaptive.data.ingest` (`localize_to_utc`, `convert_to_close_labeled`,
`deduplicate_bars`) before calling `resample_ohlcv`. Keeping vendor-specific parsing
and resampling separate means the same resampler works for any source that supplies
finer-than-target bars, not just HistData's M1 files.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from gridadaptive.data.schema import validate_ohlcv


@dataclass
class ResampleReport:
    source_timeframe: str
    target_timeframe: str
    expected_constituents: int
    bars_produced: int
    incomplete_windows: list[pd.Timestamp] = field(default_factory=list)


def resample_ohlcv(
    df: pd.DataFrame, source_timeframe: str, target_timeframe: str
) -> tuple[pd.DataFrame, ResampleReport]:
    """Aggregate close-labeled source bars into close-labeled target bars.

    Precondition, enforced by raising rather than silent repair: `df` must already
    be in the project's canonical close-labeled, UTC, deduplicated, structurally
    valid form — i.e. it must pass `validate_ohlcv(df, timeframe=source_timeframe,
    bar_convention="close")`. Call `gridadaptive.data.ingest.localize_to_utc`,
    `.convert_to_close_labeled`, and `.deduplicate_bars` first if the source data
    isn't already in this form.

    Each target bar is built strictly from source bars whose own close-label
    timestamp falls in the half-open-on-the-left interval `(window_start,
    window_end]` of the fixed target-timeframe grid, and is itself labeled with the
    timestamp of its own last (latest) actual constituent — *not* the nominal grid
    boundary. For a complete window these coincide (the grid boundary is exactly
    the last constituent's own aligned timestamp), but for an incomplete window
    (see below) they do not, and using the grid boundary would fabricate a label
    later than any data actually observed — exactly the lookahead bug this project
    guards against. This guarantees a target bar's label never precedes, and never
    claims to be later than, the information it summarizes. Open/high/low/close are
    therefore guaranteed valid by construction (each constituent's own high/low
    already bounds its open/close, so the aggregate does too) — no redundant
    re-validation of the output.

    A target bar is produced only if at least one constituent source bar exists in
    its window — no fabricated or forward-filled bars for windows with zero
    constituents (a real gap, e.g. a weekend, simply produces no target bar, exactly
    as it would for any other source). A target bar built from fewer than the
    expected number of constituents (a genuine gap in the source data, or a trailing
    window where ingestion stopped mid-window) is still produced from whatever
    constituents exist — not silently dropped, not padded to look complete — but is
    flagged via the `constituent_count` output column and
    `ResampleReport.incomplete_windows`, mirroring the existing tick_count
    diagnostic-column pattern.

    `volume`, if present in the source, is summed across constituents (the only
    arithmetically valid aggregate for an additive per-bar quantity, whether it
    represents genuine traded volume or a tick-count proxy). Whether the resulting
    number carries genuine trading-volume meaning for a given source is a semantic
    question this module does not resolve — see the source's pilot verification.
    """
    validate_ohlcv(df, timeframe=source_timeframe, bar_convention="close")

    source_interval = pd.Timedelta(source_timeframe)
    target_interval = pd.Timedelta(target_timeframe)
    if target_interval <= source_interval:
        raise ValueError(
            f"target_timeframe ({target_timeframe}) must be strictly coarser than "
            f"source_timeframe ({source_timeframe})"
        )
    ratio = target_interval / source_interval
    if ratio != int(ratio):
        raise ValueError(
            f"target_timeframe ({target_timeframe}) must be an exact multiple of "
            f"source_timeframe ({source_timeframe}), got ratio {ratio}"
        )
    expected_constituents = int(ratio)

    has_volume = "volume" in df.columns
    grouped = df.groupby(pd.Grouper(freq=target_timeframe, label="right", closed="right"))

    window_ends: list[pd.Timestamp] = []
    records: list[dict] = []
    incomplete_windows: list[pd.Timestamp] = []

    for _grid_boundary, group in grouped:
        if len(group) == 0:
            continue  # no fabricated bar for an empty window (e.g. weekend gap)

        label = group.index.max()  # last actual constituent, never the nominal grid edge
        record = {
            "open": group["open"].iloc[0],
            "high": group["high"].max(),
            "low": group["low"].min(),
            "close": group["close"].iloc[-1],
            "constituent_count": len(group),
        }
        if has_volume:
            record["volume"] = group["volume"].sum()

        window_ends.append(label)
        records.append(record)
        if len(group) < expected_constituents:
            incomplete_windows.append(label)

    out = pd.DataFrame(records, index=pd.DatetimeIndex(window_ends))
    out.index.name = "timestamp"

    report = ResampleReport(
        source_timeframe=source_timeframe,
        target_timeframe=target_timeframe,
        expected_constituents=expected_constituents,
        bars_produced=len(out),
        incomplete_windows=incomplete_windows,
    )
    return out, report
