import pandas as pd
import pytest

from gridadaptive.execution.costs import CostModel
from gridadaptive.execution.simulator import ExecutionSimulator
from gridadaptive.strategy.reference import ReferenceTrade, buy_and_hold_reference, no_trade_equity_curve

STARTING_CAPITAL = 10_000.0


def _bars(opens):
    idx = pd.date_range("2024-01-01", periods=len(opens), freq="5min", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": [o + 0.0005 for o in opens], "low": [o - 0.0005 for o in opens], "close": opens},
        index=idx,
    )


def test_no_trade_equity_curve_is_flat_at_starting_capital():
    bars = _bars([1.1000, 1.1005, 1.1010])
    curve = no_trade_equity_curve(bars, STARTING_CAPITAL)
    assert (curve == STARTING_CAPITAL).all()
    assert curve.index.equals(bars.index)


def test_buy_and_hold_entry_and_exit_timing():
    bars = _bars([1.1000, 1.1010, 1.1020, 1.1030])
    sim = ExecutionSimulator(bars, CostModel(spread=0.0, commission_per_unit=0.0))
    trade = buy_and_hold_reference(bars, sim, symbol="EURUSD", size=1000)

    assert trade.open_time == bars.index[1]  # decision at bar[0] -> fills at bar[1] open
    assert trade.close_time == bars.index[3]  # decision at bar[-2] -> fills at bar[-1] open
    assert trade.open_price == pytest.approx(1.1010)
    assert trade.close_price == pytest.approx(1.1030)


def test_buy_and_hold_zero_cost_pnl():
    bars = _bars([1.1000, 1.1010, 1.1020, 1.1030])
    sim = ExecutionSimulator(bars, CostModel(spread=0.0, commission_per_unit=0.0))
    trade = buy_and_hold_reference(bars, sim, symbol="EURUSD", size=1000)

    assert trade.gross_pnl == pytest.approx(2.0)  # (1.1030 - 1.1010) * 1000
    assert trade.realized_pnl == pytest.approx(2.0)


def test_buy_and_hold_costed_pnl_subtracts_only_commission_beyond_fill_price():
    bars = _bars([1.1000, 1.1010, 1.1020, 1.1030])
    cost_model = CostModel(spread=0.0002, commission_per_unit=0.00002, slippage_base=0.00003)
    sim = ExecutionSimulator(bars, cost_model)
    trade = buy_and_hold_reference(bars, sim, symbol="EURUSD", size=1000)

    assert isinstance(trade, ReferenceTrade)
    assert trade.spread_cost == pytest.approx(0.2)
    assert trade.commission_cost == pytest.approx(0.04)
    assert trade.slippage_cost == pytest.approx(0.06)
    # spread & slippage are already embedded in the fill prices (and hence gross_pnl);
    # only commission sits outside the price and must be subtracted separately
    assert trade.gross_pnl == pytest.approx(1.74)
    assert trade.realized_pnl == pytest.approx(1.70)
    assert trade.realized_pnl == pytest.approx(trade.gross_pnl - trade.commission_cost)


def test_buy_and_hold_too_few_bars_raises():
    bars = _bars([1.1000, 1.1010])
    sim = ExecutionSimulator(bars, CostModel(spread=0.0, commission_per_unit=0.0))
    with pytest.raises(ValueError):
        buy_and_hold_reference(bars, sim, symbol="EURUSD", size=1000)
