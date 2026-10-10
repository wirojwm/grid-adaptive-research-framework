"""Synthetic tests for the pure Wilder DMI/ADX feature function.

These scenarios are the specification of the function. Do not weaken or reinterpret
them to make an implementation pass.

Expected values below are computed independently in each test via the same
explicit Wilder-recursion arithmetic (not by calling the function under test),
matching the convention in tests/test_atr_feature.py. Hand-rounded worked-example
values appear only as sanity comments, not as the source of truth.
"""

import pandas as pd
import pytest

from gridadaptive.features.dmi_adx import wilder_dmi_adx


def _bars(rows):
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


# A1-fixture rows, reused by A1, A2, A8 (prefix) -------------------------------------

_A1_ROWS = [
    (1.1000, 1.1010, 1.0990, 1.1000),
    (1.1000, 1.1030, 1.1000, 1.1020),
    (1.1020, 1.1050, 1.1015, 1.1040),
    (1.1040, 1.1045, 1.1005, 1.1010),
    (1.1010, 1.1020, 1.0995, 1.1005),
]


def _wilder_seed_and_steps(tr, plus_dm, minus_dm, window):
    """Reference Wilder recursion, independent of the function under test."""
    s_tr = sum(tr[:window]) / window
    s_plus = sum(plus_dm[:window]) / window
    s_minus = sum(minus_dm[:window]) / window
    seeds = [(s_tr, s_plus, s_minus)]
    for i in range(window, len(tr)):
        s_tr = (s_tr * (window - 1) + tr[i]) / window
        s_plus = (s_plus * (window - 1) + plus_dm[i]) / window
        s_minus = (s_minus * (window - 1) + minus_dm[i]) / window
        seeds.append((s_tr, s_plus, s_minus))
    return seeds


def _di_dx(s_tr, s_plus, s_minus):
    plus_di = 100 * s_plus / s_tr
    minus_di = 100 * s_minus / s_tr
    denom = plus_di + minus_di
    dx = 0.0 if denom == 0 else 100 * abs(plus_di - minus_di) / denom
    return plus_di, minus_di, dx


# A1. Hand-computed formula chain (window=3) -----------------------------------------


def test_a1_hand_computed_formula_chain_window3():
    bars = _bars(_A1_ROWS)
    result = wilder_dmi_adx(bars, window=3)

    tr = [0.0020, 0.0030, 0.0035, 0.0040, 0.0025]
    plus_dm = [0.0, 0.0020, 0.0020, 0.0, 0.0]
    minus_dm = [0.0, 0.0, 0.0, 0.0010, 0.0010]

    seeds = _wilder_seed_and_steps(tr, plus_dm, minus_dm, window=3)
    plus_di_2, minus_di_2, dx_2 = _di_dx(*seeds[0])
    plus_di_3, minus_di_3, dx_3 = _di_dx(*seeds[1])
    plus_di_4, minus_di_4, dx_4 = _di_dx(*seeds[2])
    adx_4 = (dx_2 + dx_3 + dx_4) / 3

    # Hand-computable sanity check
    assert dx_2 == pytest.approx(100.0)
    assert plus_di_2 == pytest.approx(47.0588, abs=1e-3)
    assert dx_3 == pytest.approx(45.4545, abs=1e-3)
    assert adx_4 == pytest.approx(49.5602, abs=1e-3)

    assert result.plus_di.iloc[:2].isna().all()
    assert result.minus_di.iloc[:2].isna().all()
    assert result.dx.iloc[:2].isna().all()
    assert result.adx.iloc[:4].isna().all()

    assert result.plus_di.iloc[2] == pytest.approx(plus_di_2)
    assert result.minus_di.iloc[2] == pytest.approx(minus_di_2)
    assert result.dx.iloc[2] == pytest.approx(dx_2)

    assert result.plus_di.iloc[3] == pytest.approx(plus_di_3)
    assert result.minus_di.iloc[3] == pytest.approx(minus_di_3)
    assert result.dx.iloc[3] == pytest.approx(dx_3)

    assert result.plus_di.iloc[4] == pytest.approx(plus_di_4)
    assert result.minus_di.iloc[4] == pytest.approx(minus_di_4)
    assert result.dx.iloc[4] == pytest.approx(dx_4)

    assert result.adx.iloc[4] == pytest.approx(adx_4)


# A2. First-bar +DM/-DM convention (decision 3: both 0, no prior bar) ---------------


def test_a2_first_bar_dm_convention_is_zero():
    # Same 3-bar prefix as A1's seed window. If bar 0 wrongly contributed a
    # nonzero +DM (e.g. reusing TR's high-low convention instead of 0), the
    # seeded +DI at index 2 would be ~70.588 instead of ~47.0588 — a sharp,
    # unambiguous divergence.
    bars = _bars(_A1_ROWS[:3])
    result = wilder_dmi_adx(bars, window=3)

    tr = [0.0020, 0.0030, 0.0035]
    plus_dm_correct = [0.0, 0.0020, 0.0020]  # decision 3: bar 0 contributes 0
    minus_dm_correct = [0.0, 0.0, 0.0]

    s_tr = sum(tr) / 3
    s_plus_correct = sum(plus_dm_correct) / 3
    plus_di_correct = 100 * s_plus_correct / s_tr

    plus_dm_buggy = [0.0020, 0.0020, 0.0020]  # bar 0 wrongly = high0-low0
    s_plus_buggy = sum(plus_dm_buggy) / 3
    plus_di_buggy = 100 * s_plus_buggy / s_tr

    assert plus_di_correct == pytest.approx(47.0588, abs=1e-3)
    assert plus_di_buggy == pytest.approx(70.588, abs=1e-3)
    assert result.plus_di.iloc[2] == pytest.approx(plus_di_correct)
    assert result.plus_di.iloc[2] != pytest.approx(plus_di_buggy)
    assert result.minus_di.iloc[2] == pytest.approx(0.0)


# A3. Exact N=288 warm-up boundary; A4 shares the same fixture ----------------------


def _clean_uptrend_row(i, step=0.0005, base_high=1.1010, range_width=0.0020):
    high = base_high + i * step
    low = high - range_width
    mid = high - range_width / 2  # matches close = midpoint, including i=0
    return (mid, high, low, mid)


_WARMUP_ROWS = [_clean_uptrend_row(i) for i in range(575)]  # 2*288 - 1 bars


def test_a3_exact_n288_warmup_boundary():
    bars = _bars(_WARMUP_ROWS)
    result = wilder_dmi_adx(bars, window=288)

    assert result.plus_di.iloc[:287].isna().all()
    assert not pd.isna(result.plus_di.iloc[287])
    assert result.minus_di.iloc[:287].isna().all()
    assert not pd.isna(result.minus_di.iloc[287])
    assert result.dx.iloc[:287].isna().all()
    assert not pd.isna(result.dx.iloc[287])

    assert result.adx.iloc[:574].isna().all()
    assert not pd.isna(result.adx.iloc[574])


# A4. First valid ADX timestamp, exposed as a timestamp not just an index -----------


def test_a4_first_valid_adx_timestamp():
    bars = _bars(_WARMUP_ROWS)
    result = wilder_dmi_adx(bars, window=288)

    assert result.adx.dropna().index[0] == bars.index[574]


# A5. Zero-DX-denominator edge case (+DI=-DI=0) convention ------------------


def test_a5_zero_dx_denominator_reports_dx_zero_not_nan():
    # "Inside bars": each bar's high/low strictly inside the previous bar's
    # range -> up_move/down_move both <= 0 for i>=1 -> +DM=-DM=0 throughout,
    # while TR>0 (nonzero range). window=3.
    rows = [
        (1.1000, 1.1050, 1.0950, 1.1000),
        (1.1000, 1.1030, 1.0970, 1.1000),
        (1.1000, 1.1020, 1.0980, 1.1000),
        (1.1000, 1.1010, 1.0990, 1.1000),
    ]
    bars = _bars(rows)
    result = wilder_dmi_adx(bars, window=3)

    assert result.plus_di.iloc[2:].eq(0.0).all()
    assert result.minus_di.iloc[2:].eq(0.0).all()
    assert result.dx.iloc[2:].eq(0.0).all()
    assert not result.dx.iloc[2:].isna().any()


# A6. No NaN/inf propagation over a longer, non-degenerate run ----------------------


def _zigzag_row(i, cycle=40, step=0.0004, range_width=0.0020, base=1.1000):
    # Fully deterministic triangular wave: rises for cycle/2 bars, falls for
    # cycle/2 bars, repeating. Varied, well-defined directional movement with
    # no zero-range or zero-DM degeneracy (distinct from A5/A7's edge cases).
    phase = i % cycle
    tri = phase if phase < cycle // 2 else cycle - phase
    level = base + tri * step
    high = level + range_width / 2
    low = level - range_width / 2
    return (level, high, low, level)


_ZIGZAG_ROWS = [_zigzag_row(i) for i in range(600)]


def test_a6_no_nan_inf_over_longer_run():
    import numpy as np

    bars = _bars(_ZIGZAG_ROWS)
    result = wilder_dmi_adx(bars, window=288)

    assert np.isfinite(result.plus_di.iloc[574:].astype(float)).all()
    assert np.isfinite(result.minus_di.iloc[574:].astype(float)).all()
    assert np.isfinite(result.dx.iloc[574:].astype(float)).all()
    assert np.isfinite(result.adx.iloc[574:].astype(float)).all()


# A7. Smoothed_TR = 0 edge case (decision 4: DI=DX=0, no NaN/inf) -------------------


def test_a7_zero_smoothed_tr_reports_zero_not_nan_or_inf():
    rows = [(1.1000, 1.1000, 1.1000, 1.1000) for _ in range(5)]
    bars = _bars(rows)
    result = wilder_dmi_adx(bars, window=3)

    assert not result.plus_di.iloc[2:].isna().any()
    assert not result.minus_di.iloc[2:].isna().any()
    assert not result.dx.iloc[2:].isna().any()
    assert result.plus_di.iloc[2:].eq(0.0).all()
    assert result.minus_di.iloc[2:].eq(0.0).all()
    assert result.dx.iloc[2:].eq(0.0).all()


# A8. No-lookahead (feature level) ---------------------------------------------------


def test_a8_no_lookahead():
    extended_rows = _A1_ROWS + [
        (1.1005, 1.2000, 1.0500, 1.1500),
        (1.1500, 1.3000, 1.0000, 1.2000),
    ]

    prefix_bars = _bars(_A1_ROWS)
    full_bars = _bars(extended_rows)

    prefix_result = wilder_dmi_adx(prefix_bars, window=3)
    full_result = wilder_dmi_adx(full_bars, window=3)

    ts = prefix_bars.index[4]
    assert full_result.plus_di.loc[ts] == prefix_result.plus_di.loc[ts]
    assert full_result.minus_di.loc[ts] == prefix_result.minus_di.loc[ts]
    assert full_result.dx.loc[ts] == prefix_result.dx.loc[ts]
    assert full_result.adx.loc[ts] == prefix_result.adx.loc[ts]


# A9. Determinism ---------------------------------------------------------------------


def test_a9_deterministic():
    bars = _bars(_A1_ROWS)
    result_1 = wilder_dmi_adx(bars, window=3)
    result_2 = wilder_dmi_adx(bars, window=3)

    assert result_1.plus_di.equals(result_2.plus_di)
    assert result_1.minus_di.equals(result_2.minus_di)
    assert result_1.dx.equals(result_2.dx)
    assert result_1.adx.equals(result_2.adx)
