"""ATR feature — Wilder's smoothed True Range.

Pure function of OHLC data only: no order, portfolio, or strategy-config knowledge. Every value at index `i` depends only on bars at
or before `i` — never on a later bar.

First-bar convention (fixed convention): when no previous
bar's close exists, `TR = high - low` (the close-based terms are simply undefined,
not fabricated). `DataFrame.max(axis=1)` already skips `NaN` entries by default,
which is exactly this convention applied to the first row's undefined
`prev_close`-based terms — no special-cased branch is needed.

`window - 1` prior true-range observations must exist before ATR is valid at a
given bar (Wilder's seed = the plain mean of the first `window` observations); bars before that are reported as `NaN`
("warm-up" invalid sentinel).
"""

from __future__ import annotations

import pandas as pd


def wilder_atr(bars: pd.DataFrame, window: int = 288) -> pd.Series:
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    prev_close = bars["close"].astype(float).shift(1)

    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    atr = pd.Series(index=bars.index, dtype=float)
    if len(bars) < window:
        return atr

    atr.iloc[window - 1] = true_range.iloc[:window].mean()
    for i in range(window, len(bars)):
        atr.iloc[i] = (atr.iloc[i - 1] * (window - 1) + true_range.iloc[i]) / window

    return atr
