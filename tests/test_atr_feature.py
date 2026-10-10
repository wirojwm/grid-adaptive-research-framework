"""Synthetic tests for the pure Wilder ATR feature function.

These scenarios are the specification of the function. Do not weaken or reinterpret
them to make an implementation pass.

Convention adopted for the very first bar of any input (no prior bar's close is
available): TR_1 = high_1 - low_1 (the close-based terms are simply undefined, not
substituted with a fabricated value). This is the standard convention used by
virtually every reference ATR implementation and is left implicit by the usual
formula; it is stated here as the resolution assumed by this test suite. It does not affect any assertion below, since every fixture bar is
constructed with high-low already equal to the intended true range.
"""

import pandas as pd
import pytest

from gridadaptive.features.atr import wilder_atr

REFERENCE_TR = 0.0020
SPIKE_TR = 0.0100
CLOSE_LEVEL = 1.10000


def _flat_tr_row(tr, close_level=CLOSE_LEVEL):
    half = tr / 2
    return (close_level, close_level + half, close_level - half, close_level)


def _rows_to_bars(rows):
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
        },
        index=idx,
    )


def _a1_fixture_rows():
    return [_flat_tr_row(REFERENCE_TR) for _ in range(288)] + [_flat_tr_row(SPIKE_TR)]


# A1. Wilder ATR hand-computed: warm-up mean + recursive step ----------------------


def test_wilder_atr_warmup_mean_and_recursive_step_hand_computed():
    bars = _rows_to_bars(_a1_fixture_rows())
    atr = wilder_atr(bars)

    # Bars 0-286 (287 bars): fewer than 288 TR observations exist yet -> invalid.
    assert atr.iloc[:287].isna().all()

    # Bar 287 (the 288th bar, 0-indexed): first valid ATR = plain mean of 288
    # identical TR=0.0020 observations.
    assert atr.iloc[287] == pytest.approx(0.0020)

    # Bar 288 (the 289th bar, the TR=0.0100 spike): Wilder recursive update, not a
    # fresh mean.
    expected_step = (0.0020 * 287 + 0.0100) / 288
    assert expected_step == pytest.approx(0.0020277778, abs=1e-9)  # hand-computable check
    assert atr.iloc[288] == pytest.approx(expected_step, abs=1e-9)


# A2. No-lookahead (feature level) --------------------------------------------------


def test_wilder_atr_no_lookahead():
    base_rows = _a1_fixture_rows()
    extended_rows = base_rows + [_flat_tr_row(0.5000), _flat_tr_row(0.5000)]

    prefix_bars = _rows_to_bars(base_rows)
    full_bars = _rows_to_bars(extended_rows)

    prefix_atr = wilder_atr(prefix_bars)
    full_atr = wilder_atr(full_bars)

    ts_288 = prefix_bars.index[288]
    assert full_atr.loc[ts_288] == prefix_atr.loc[ts_288]


# A3. Determinism --------------------------------------------------------------------


def test_wilder_atr_deterministic():
    bars = _rows_to_bars(_a1_fixture_rows())
    atr_1 = wilder_atr(bars)
    atr_2 = wilder_atr(bars)
    assert atr_1.equals(atr_2)
