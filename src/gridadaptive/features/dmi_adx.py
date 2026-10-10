"""DMI/ADX feature — Wilder's Directional Movement Index / Average Directional Index.

Pure function of OHLC data only — no order/portfolio/strategy knowledge,
reusing `features/atr.py`'s exact True Range definition and Wilder-recursion form.

Edge cases:
- First bar: no previous bar exists, so `+DM = -DM = 0`.
- `Smoothed_TR == 0`: `+DI := 0`, `-DI := 0`, therefore `DX := 0`
  (extending the `DX := 0 if denominator == 0` rule).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DmiAdxResult:
    plus_di: pd.Series
    minus_di: pd.Series
    dx: pd.Series
    adx: pd.Series


def wilder_dmi_adx(bars: pd.DataFrame, window: int = 288) -> DmiAdxResult:
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = bars["close"].astype(float).shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    # First bar: no prior bar, so no measurable directional movement (decision 3).
    plus_dm.iloc[0] = 0.0
    minus_dm.iloc[0] = 0.0

    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    index = bars.index
    plus_di = pd.Series(index=index, dtype=float)
    minus_di = pd.Series(index=index, dtype=float)
    dx = pd.Series(index=index, dtype=float)
    adx = pd.Series(index=index, dtype=float)

    if len(bars) < window:
        return DmiAdxResult(plus_di=plus_di, minus_di=minus_di, dx=dx, adx=adx)

    def _di_dx(s_tr: float, s_plus: float, s_minus: float) -> tuple[float, float, float]:
        if s_tr == 0:
            return 0.0, 0.0, 0.0
        p_di = 100 * s_plus / s_tr
        m_di = 100 * s_minus / s_tr
        denom = p_di + m_di
        dx_val = 0.0 if denom == 0 else 100 * abs(p_di - m_di) / denom
        return p_di, m_di, dx_val

    s_tr = true_range.iloc[:window].mean()
    s_plus = plus_dm.iloc[:window].mean()
    s_minus = minus_dm.iloc[:window].mean()

    p_di, m_di, dx_val = _di_dx(s_tr, s_plus, s_minus)
    plus_di.iloc[window - 1] = p_di
    minus_di.iloc[window - 1] = m_di
    dx.iloc[window - 1] = dx_val

    for i in range(window, len(bars)):
        s_tr = (s_tr * (window - 1) + true_range.iloc[i]) / window
        s_plus = (s_plus * (window - 1) + plus_dm.iloc[i]) / window
        s_minus = (s_minus * (window - 1) + minus_dm.iloc[i]) / window
        p_di, m_di, dx_val = _di_dx(s_tr, s_plus, s_minus)
        plus_di.iloc[i] = p_di
        minus_di.iloc[i] = m_di
        dx.iloc[i] = dx_val

    adx_start = 2 * window - 2
    if len(bars) > adx_start:
        s_adx = dx.iloc[window - 1 : adx_start + 1].mean()
        adx.iloc[adx_start] = s_adx
        for i in range(adx_start + 1, len(bars)):
            s_adx = (s_adx * (window - 1) + dx.iloc[i]) / window
            adx.iloc[i] = s_adx

    return DmiAdxResult(plus_di=plus_di, minus_di=minus_di, dx=dx, adx=adx)
