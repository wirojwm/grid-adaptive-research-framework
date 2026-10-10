"""Unit tests for grid_metrics, using hand-built FixedGridResult fixtures —
independent of FixedGridEngine, so a bug in one can't mask a bug in the other."""

import pandas as pd
import pytest

from gridadaptive.metrics.grid_metrics import (
    compute_grid_metrics,
    equity_curve,
    mark_to_market_curve,
)
from gridadaptive.strategy.fixed_grid import FixedGridResult, GridCycle, GridLevel, OpenPosition

STARTING_CAPITAL = 10_000.0


def _bars(closes):
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="5min", tz="UTC")
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes}, index=idx
    )


def _cycle(
    level_id="buy_1",
    side="buy",
    open_time=None,
    open_price=1.1000,
    close_time=None,
    close_price=1.1010,
    size=1000.0,
    spread_cost=0.0,
    commission_cost=0.0,
    slippage_cost=0.0,
    gross_pnl=1.0,
):
    all_in = spread_cost + commission_cost + slippage_cost
    return GridCycle(
        level_id=level_id,
        side=side,
        open_time=open_time,
        open_price=open_price,
        close_time=close_time,
        close_price=close_price,
        size=size,
        spread_cost=spread_cost,
        commission_cost=commission_cost,
        slippage_cost=slippage_cost,
        gross_pnl=gross_pnl,
        all_in_cost=all_in,
        realized_pnl=gross_pnl - all_in,
    )


def _open_position(
    level_id="buy_1",
    side="buy",
    open_time=None,
    open_price=1.1000,
    size=1000.0,
    spread_cost=0.0,
    commission_cost=0.0,
    slippage_cost=0.0,
):
    return OpenPosition(
        level_id=level_id,
        side=side,
        open_time=open_time,
        open_price=open_price,
        size=size,
        spread_cost=spread_cost,
        commission_cost=commission_cost,
        slippage_cost=slippage_cost,
    )


def _levels():
    return [GridLevel(level_id="buy_1", side="buy", price=1.0990)]


# --- empty result ------------------------------------------------------------------


def test_empty_result_has_zero_metrics_and_no_division_errors():
    bars = _bars([1.1000, 1.1000, 1.1000])
    result = FixedGridResult(anchor_price=1.1000, levels=_levels(), cycles=[], open_positions=[])
    metrics = compute_grid_metrics(result, bars, STARTING_CAPITAL)

    assert metrics.grid_cycles == 0
    assert metrics.layers_filled == 0
    assert metrics.max_simultaneous_layers == 0
    assert metrics.realized_pnl == 0.0
    assert metrics.unrealized_pnl_at_end == 0.0
    assert metrics.max_adverse_inventory == 0.0
    assert metrics.recovery_time_bars == 0
    assert metrics.capital_utilization == 0.0
    assert metrics.turnover == 0.0
    assert metrics.total_transaction_cost == 0.0
    assert metrics.transaction_cost_drag_vs_capital == 0.0
    assert metrics.transaction_cost_drag_vs_gross_pnl is None
    assert metrics.position_buildup_during_trend == 0
    assert metrics.recenter_frequency == 0
    assert metrics.shutdown_frequency == 0
    assert metrics.risk_guard_activations == 0


# --- single closed cycle -------------------------------------------------------------


def test_single_closed_cycle_basic_metrics():
    bars = _bars([1.1000, 1.1005, 1.1010, 1.1010])
    idx = bars.index
    cycle = _cycle(open_time=idx[0], open_price=1.1000, close_time=idx[2], close_price=1.1010, gross_pnl=1.0)
    result = FixedGridResult(anchor_price=1.1000, levels=_levels(), cycles=[cycle], open_positions=[])

    metrics = compute_grid_metrics(result, bars, STARTING_CAPITAL)

    assert metrics.grid_cycles == 1
    assert metrics.layers_filled == 1
    assert metrics.max_simultaneous_layers == 1
    assert metrics.realized_pnl == pytest.approx(1.0)
    assert metrics.unrealized_pnl_at_end == pytest.approx(0.0)

    expected_turnover_notional = 1.1000 * 1000 + 1.1010 * 1000
    assert metrics.turnover == pytest.approx(expected_turnover_notional / STARTING_CAPITAL)
    expected_peak_notional = 1.1000 * 1000
    assert metrics.capital_utilization == pytest.approx(expected_peak_notional / STARTING_CAPITAL)


def test_mark_to_market_curve_reflects_open_window_only():
    bars = _bars([1.1000, 1.1005, 1.1010, 1.1010])
    idx = bars.index
    # position open [t0, t2), closes exactly at t2 -> not counted as open at t2
    cycle = _cycle(open_time=idx[0], open_price=1.1000, close_time=idx[2], close_price=1.1010, gross_pnl=1.0)
    result = FixedGridResult(anchor_price=1.1000, levels=_levels(), cycles=[cycle], open_positions=[])

    mtm = mark_to_market_curve(result, bars)
    assert mtm.iloc[0] == pytest.approx(0.0)  # (1.1000 - 1.1000) * 1000
    assert mtm.iloc[1] == pytest.approx(0.5)  # (1.1005 - 1.1000) * 1000
    assert mtm.iloc[2] == pytest.approx(0.0)  # already closed, excluded by t < close_time
    assert mtm.iloc[3] == pytest.approx(0.0)


# --- adverse inventory / recovery ----------------------------------------------------


def test_max_adverse_inventory_and_recovery_time_when_it_recovers():
    bars = _bars([1.1000, 1.0990, 1.0980, 1.1000])
    idx = bars.index
    position = _open_position(open_time=idx[0], open_price=1.1000)
    result = FixedGridResult(anchor_price=1.1000, levels=_levels(), cycles=[], open_positions=[position])

    metrics = compute_grid_metrics(result, bars, STARTING_CAPITAL)
    assert metrics.max_adverse_inventory == pytest.approx(2.0)  # worst point: (1.0980-1.1000)*1000 = -2.0
    assert metrics.recovery_time_bars == 1  # equity returns to its prior peak one bar after the trough


def test_recovery_time_is_none_when_it_never_recovers_in_window():
    bars = _bars([1.1000, 1.0990, 1.0980])
    idx = bars.index
    position = _open_position(open_time=idx[0], open_price=1.1000)
    result = FixedGridResult(anchor_price=1.1000, levels=_levels(), cycles=[], open_positions=[position])

    metrics = compute_grid_metrics(result, bars, STARTING_CAPITAL)
    assert metrics.max_adverse_inventory == pytest.approx(2.0)
    assert metrics.recovery_time_bars is None


def test_no_adverse_inventory_reports_zero_and_trivial_recovery():
    bars = _bars([1.1000, 1.1005, 1.1010])
    idx = bars.index
    position = _open_position(open_time=idx[0], open_price=1.1000)
    result = FixedGridResult(anchor_price=1.1000, levels=_levels(), cycles=[], open_positions=[position])

    metrics = compute_grid_metrics(result, bars, STARTING_CAPITAL)
    assert metrics.max_adverse_inventory == 0.0
    assert metrics.recovery_time_bars == 0


# --- position buildup during trend ---------------------------------------------------


def test_position_buildup_counts_consecutive_same_side_entries_with_no_close():
    bars = _bars([1.1000, 1.0995, 1.0990, 1.0985])
    idx = bars.index
    positions = [
        _open_position(level_id="buy_1", side="buy", open_time=idx[0], open_price=1.1000),
        _open_position(level_id="buy_2", side="buy", open_time=idx[1], open_price=1.0995),
        _open_position(level_id="buy_3", side="buy", open_time=idx[2], open_price=1.0990),
    ]
    result = FixedGridResult(anchor_price=1.1000, levels=_levels(), cycles=[], open_positions=positions)
    metrics = compute_grid_metrics(result, bars, STARTING_CAPITAL)
    assert metrics.position_buildup_during_trend == 3


def test_position_buildup_resets_on_an_intervening_close():
    bars = _bars([1.1000, 1.0995, 1.0990, 1.0985])
    idx = bars.index
    cycle = _cycle(level_id="buy_1", open_time=idx[0], close_time=idx[1], gross_pnl=1.0)
    later_positions = [
        _open_position(level_id="buy_2", side="buy", open_time=idx[2], open_price=1.0990),
        _open_position(level_id="buy_3", side="buy", open_time=idx[3], open_price=1.0985),
    ]
    result = FixedGridResult(
        anchor_price=1.1000, levels=_levels(), cycles=[cycle], open_positions=later_positions
    )
    metrics = compute_grid_metrics(result, bars, STARTING_CAPITAL)
    assert metrics.position_buildup_during_trend == 2  # close at t1 breaks the streak


# --- transaction cost drag ------------------------------------------------------------


def test_transaction_cost_drag_with_nonzero_costs():
    bars = _bars([1.1000, 1.1005, 1.1010, 1.1010])
    idx = bars.index
    cycle = _cycle(
        open_time=idx[0],
        open_price=1.1000,
        close_time=idx[2],
        close_price=1.1010,
        gross_pnl=1.0,
        spread_cost=0.2,
        commission_cost=0.02,
    )
    result = FixedGridResult(anchor_price=1.1000, levels=_levels(), cycles=[cycle], open_positions=[])
    metrics = compute_grid_metrics(result, bars, STARTING_CAPITAL)

    assert metrics.total_transaction_cost == pytest.approx(0.22)
    assert metrics.transaction_cost_drag_vs_capital == pytest.approx(0.22 / STARTING_CAPITAL)
    assert metrics.transaction_cost_drag_vs_gross_pnl == pytest.approx(0.22 / 1.0)
    assert metrics.realized_pnl == pytest.approx(0.78)


# --- structural sanity checks (Fixed Grid has none of these components) -----------------


def test_recenter_shutdown_and_risk_guard_counts_are_always_zero():
    bars = _bars([1.1000])
    result = FixedGridResult(anchor_price=1.1000, levels=_levels(), cycles=[], open_positions=[])
    metrics = compute_grid_metrics(result, bars, STARTING_CAPITAL)
    assert metrics.recenter_frequency == 0
    assert metrics.shutdown_frequency == 0
    assert metrics.risk_guard_activations == 0


def test_equity_curve_combines_realized_and_unrealized():
    bars = _bars([1.1000, 1.1005, 1.1010, 1.1010])
    idx = bars.index
    cycle = _cycle(open_time=idx[0], open_price=1.1000, close_time=idx[2], close_price=1.1010, gross_pnl=1.0)
    result = FixedGridResult(anchor_price=1.1000, levels=_levels(), cycles=[cycle], open_positions=[])

    equity = equity_curve(result, bars, STARTING_CAPITAL)
    assert equity.iloc[0] == pytest.approx(STARTING_CAPITAL)
    assert equity.iloc[1] == pytest.approx(STARTING_CAPITAL + 0.5)  # unrealized only
    assert equity.iloc[2] == pytest.approx(STARTING_CAPITAL + 1.0)  # realized, now closed
    assert equity.iloc[3] == pytest.approx(STARTING_CAPITAL + 1.0)


# --- input validation ------------------------------------------------------------------


def test_non_positive_starting_capital_raises():
    bars = _bars([1.1000])
    result = FixedGridResult(anchor_price=1.1000, levels=_levels(), cycles=[], open_positions=[])
    with pytest.raises(ValueError):
        compute_grid_metrics(result, bars, 0.0)
    with pytest.raises(ValueError):
        compute_grid_metrics(result, bars, -100.0)
