import math

import pandas as pd
import pytest

from gridadaptive.metrics.standard_metrics import StandardMetrics, compute_standard_metrics
from gridadaptive.strategy.fixed_grid import GridCycle


def _trade(open_time, close_time, realized_pnl, side="buy", size=1000.0):
    return GridCycle(
        level_id="x",
        side=side,
        open_time=open_time,
        open_price=1.1000,
        close_time=close_time,
        close_price=1.1010,
        size=size,
        spread_cost=0.0,
        commission_cost=0.0,
        slippage_cost=0.0,
        gross_pnl=realized_pnl,
        all_in_cost=0.0,
        realized_pnl=realized_pnl,
    )


def _equity(values, freq="5min", start="2024-01-01"):
    idx = pd.date_range(start, periods=len(values), freq=freq, tz="UTC")
    return pd.Series(values, index=idx, name="equity")


STARTING_CAPITAL = 10_000.0


def test_total_return_and_net_pnl():
    equity = _equity([10_000, 10_050, 10_100])
    metrics = compute_standard_metrics([], equity, STARTING_CAPITAL)
    assert metrics.net_pnl == pytest.approx(100.0)
    assert metrics.total_return == pytest.approx(0.01)


def test_max_drawdown_recovers():
    equity = _equity([10_000, 9_900, 9_800, 10_000])
    metrics = compute_standard_metrics([], equity, STARTING_CAPITAL)
    assert metrics.max_drawdown == pytest.approx(200.0)
    assert metrics.max_drawdown_pct == pytest.approx(0.02)
    assert metrics.drawdown_duration_bars == 1


def test_max_drawdown_never_recovers_in_window():
    equity = _equity([10_000, 9_900, 9_800])
    metrics = compute_standard_metrics([], equity, STARTING_CAPITAL)
    assert metrics.max_drawdown == pytest.approx(200.0)
    assert metrics.drawdown_duration_bars is None


def test_no_drawdown_when_equity_never_declines():
    equity = _equity([10_000, 10_010, 10_020])
    metrics = compute_standard_metrics([], equity, STARTING_CAPITAL)
    assert metrics.max_drawdown == 0.0
    assert metrics.max_drawdown_pct == 0.0
    assert metrics.drawdown_duration_bars == 0


def test_cagr_matches_the_documented_formula():
    idx = pd.DatetimeIndex(["2020-01-01T00:00:00Z", "2021-01-01T00:00:00Z"])
    equity = pd.Series([10_000.0, 11_000.0], index=idx)
    metrics = compute_standard_metrics([], equity, 10_000.0)

    elapsed_days = (idx[-1] - idx[0]).total_seconds() / 86400
    years = elapsed_days / 365.25
    expected_cagr = (11_000.0 / 10_000.0) ** (1 / years) - 1
    assert metrics.cagr == pytest.approx(expected_cagr)


def test_cagr_is_none_when_final_equity_is_non_positive():
    equity = _equity([10_000, -500])
    metrics = compute_standard_metrics([], equity, STARTING_CAPITAL)
    assert metrics.cagr is None


def test_sharpe_matches_mean_over_std_annualized():
    idx = pd.date_range("2024-01-01", periods=6, freq="1D", tz="UTC")
    equity = pd.Series([10_000, 10_100, 10_050, 10_200, 10_150, 10_300], index=idx, dtype=float)
    metrics = compute_standard_metrics([], equity, STARTING_CAPITAL)

    daily_returns = equity.pct_change().dropna()
    expected_sharpe = daily_returns.mean() / daily_returns.std() * math.sqrt(252)
    assert metrics.sharpe_ratio == pytest.approx(expected_sharpe)


def test_sharpe_is_none_for_zero_variance_returns():
    idx = pd.date_range("2024-01-01", periods=4, freq="1D", tz="UTC")
    flat = pd.Series([10_000, 10_000, 10_000, 10_000], index=idx, dtype=float)  # zero return every day
    metrics = compute_standard_metrics([], flat, STARTING_CAPITAL)
    assert metrics.sharpe_ratio is None


def test_sortino_is_none_when_no_downside_observed():
    idx = pd.date_range("2024-01-01", periods=5, freq="1D", tz="UTC")
    equity = pd.Series([10_000, 10_100, 10_200, 10_300, 10_400], index=idx, dtype=float)
    metrics = compute_standard_metrics([], equity, STARTING_CAPITAL)
    assert metrics.sortino_ratio is None


def test_trade_statistics_win_rate_profit_factor_expectancy():
    idx = pd.date_range("2024-01-01", periods=4, freq="1D", tz="UTC")
    trades = [
        _trade(idx[0], idx[1], 100.0),
        _trade(idx[1], idx[2], 50.0),
        _trade(idx[2], idx[3], -30.0),
    ]
    equity = _equity([10_000, 10_100, 10_150, 10_120], freq="1D")
    metrics = compute_standard_metrics(trades, equity, STARTING_CAPITAL)

    assert metrics.trade_count == 3
    assert metrics.win_rate == pytest.approx(2 / 3)
    assert metrics.profit_factor == pytest.approx(150.0 / 30.0)
    assert metrics.expectancy == pytest.approx(40.0)
    assert metrics.average_trade == pytest.approx(40.0)


def test_trade_statistics_are_none_when_there_are_no_trades():
    equity = _equity([10_000, 10_000])
    metrics = compute_standard_metrics([], equity, STARTING_CAPITAL)
    assert metrics.trade_count == 0
    assert metrics.win_rate is None
    assert metrics.profit_factor is None
    assert metrics.expectancy is None
    assert metrics.average_trade is None


def test_profit_factor_is_none_when_there_are_no_losing_trades():
    idx = pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")
    trades = [_trade(idx[0], idx[1], 100.0), _trade(idx[1], idx[2], 50.0)]
    equity = _equity([10_000, 10_100, 10_150], freq="1D")
    metrics = compute_standard_metrics(trades, equity, STARTING_CAPITAL)
    assert metrics.profit_factor is None


def test_exposure_time_fraction_counts_only_the_open_window():
    idx = pd.date_range("2024-01-01", periods=5, freq="1D", tz="UTC")
    trades = [_trade(idx[1], idx[3], 10.0)]  # open for bars[1] and [2], closed by bar[3]
    equity = _equity([10_000] * 5, freq="1D")
    metrics = compute_standard_metrics(trades, equity, STARTING_CAPITAL)
    assert metrics.exposure_time_fraction == pytest.approx(2 / 5)


def test_non_positive_starting_capital_raises():
    equity = _equity([10_000, 10_000])
    with pytest.raises(ValueError):
        compute_standard_metrics([], equity, 0.0)


def test_empty_equity_series_raises():
    empty = pd.Series([], dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    with pytest.raises(ValueError):
        compute_standard_metrics([], empty, STARTING_CAPITAL)
