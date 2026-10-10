"""Standard performance metrics.

Works on any equity curve + trade list, not just Fixed Grid — the same function
computes metrics for the Fixed Grid baseline, the buy-and-hold reference (one
synthetic trade), and the no-trade reference (zero trades, flat equity), via
`TradeLike` structural typing: anything exposing `open_time`, `close_time`, `size`,
`side`, `realized_pnl` works, so `gridadaptive.strategy.fixed_grid.GridCycle`
instances can be passed directly with no adapter.

## Interpretation caveats specific to an FX grid strategy — read before reporting these

- **Sharpe/Sortino** are computed on *daily-resampled* equity returns, not raw M5
  bar returns — bar-level returns for a strategy that is flat most of the time would
  be a heavily zero-inflated series whose annualized Sharpe is close to meaningless.
  Daily resampling is the more standard choice, but the underlying assumption behind
  Sharpe/Sortino — a continuously-invested strategy — still doesn't quite fit a grid
  that is only actually exposed during a fraction of the window. Read these
  alongside `exposure_time_fraction`, not instead of it.
- **CAGR** annualizes the return on the *entire starting capital pool* over the
  elapsed calendar span, even though that capital is mostly idle for a grid
  strategy (only a fraction of it is ever deployed at once, see `capital_utilization`). Treat it as "return on the capital you set aside," not
  "return on capital actually at risk."
- **win_rate / profit_factor / expectancy / average_trade** are computed over
  *closed* grid cycles only. A position still open when the window ends is excluded
  here entirely — it shows up in the grid-specific metrics
  (`unrealized_pnl_at_end`, `layers_filled` vs `grid_cycles`,
  `gridadaptive.metrics.grid_metrics`) instead. A run that ends mid-trend can look
  better on these trade statistics than it should, precisely because its worst
  open positions aren't "trades" yet — always read the grid-specific metrics
  alongside these, never as a substitute for them.
- **max_drawdown** is measured on the mark-to-market equity curve (so it already
  reflects unrealized inventory risk, not just realized PnL), but it can only see
  drawdown at the bar-close granularity of the input data — true intra-bar
  drawdown could be worse. This is a generic bar-based-backtest limitation, not
  specific to grid strategies, but worth stating once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

import pandas as pd

_TRADING_DAYS_PER_YEAR = 252
_DAYS_PER_YEAR = 365.25


class TradeLike(Protocol):
    open_time: pd.Timestamp
    close_time: pd.Timestamp
    size: float
    side: str
    realized_pnl: float


@dataclass(frozen=True)
class StandardMetrics:
    total_return: float
    net_pnl: float
    max_drawdown: float
    max_drawdown_pct: float
    drawdown_duration_bars: int | None
    cagr: float | None
    sharpe_ratio: float | None
    sortino_ratio: float | None
    trade_count: int
    win_rate: float | None
    profit_factor: float | None
    expectancy: float | None
    average_trade: float | None
    exposure_time_fraction: float


def _drawdown_stats(equity: pd.Series) -> tuple[float, float, int | None]:
    running_peak = equity.cummax()
    drawdown = running_peak - equity
    max_dd = float(drawdown.max())
    if max_dd <= 0:
        return 0.0, 0.0, 0

    trough_pos = int(drawdown.values.argmax())
    peak_at_trough = float(running_peak.iloc[trough_pos])
    max_dd_pct = max_dd / peak_at_trough if peak_at_trough > 0 else 0.0

    duration = None
    for i in range(trough_pos + 1, len(equity)):
        if equity.iloc[i] >= peak_at_trough:
            duration = i - trough_pos
            break
    return max_dd, max_dd_pct, duration


def _cagr(equity: pd.Series, starting_capital: float) -> float | None:
    if len(equity) < 2:
        return None
    final_equity = float(equity.iloc[-1])
    if final_equity <= 0 or starting_capital <= 0:
        return None
    elapsed_days = (equity.index[-1] - equity.index[0]).total_seconds() / 86400
    if elapsed_days <= 0:
        return None
    years = elapsed_days / _DAYS_PER_YEAR
    growth = final_equity / starting_capital
    return growth ** (1 / years) - 1


def _daily_returns(equity: pd.Series) -> pd.Series:
    daily_equity = equity.resample("1D").last().dropna()
    return daily_equity.pct_change().dropna()


def _annualized_sharpe(daily_returns: pd.Series) -> float | None:
    if len(daily_returns) < 2:
        return None
    std = daily_returns.std()
    if not std or pd.isna(std):
        return None
    return float(daily_returns.mean() / std * math.sqrt(_TRADING_DAYS_PER_YEAR))


def _annualized_sortino(daily_returns: pd.Series) -> float | None:
    if len(daily_returns) < 2:
        return None
    downside = daily_returns[daily_returns < 0]
    if len(downside) == 0:
        return None
    downside_std = downside.std()
    if not downside_std or pd.isna(downside_std):
        return None
    return float(daily_returns.mean() / downside_std * math.sqrt(_TRADING_DAYS_PER_YEAR))


def _exposure_time_fraction(trades: Sequence[TradeLike], index: pd.DatetimeIndex) -> float:
    if len(index) == 0:
        return 0.0
    open_mask = pd.Series(False, index=index)
    for t in trades:
        open_mask |= (index >= t.open_time) & (index < t.close_time)
    return float(open_mask.mean())


def compute_standard_metrics(
    trades: Sequence[TradeLike], equity: pd.Series, starting_capital: float
) -> StandardMetrics:
    if starting_capital <= 0:
        raise ValueError("starting_capital must be positive")
    if len(equity) == 0:
        raise ValueError("equity series must not be empty")

    final_equity = float(equity.iloc[-1])
    net_pnl = final_equity - starting_capital
    total_return = net_pnl / starting_capital

    max_dd, max_dd_pct, dd_duration = _drawdown_stats(equity)
    cagr = _cagr(equity, starting_capital)
    daily_returns = _daily_returns(equity)
    sharpe = _annualized_sharpe(daily_returns)
    sortino = _annualized_sortino(daily_returns)

    pnls = [t.realized_pnl for t in trades]
    trade_count = len(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    win_rate = (len(wins) / trade_count) if trade_count else None
    gross_win = sum(wins)
    gross_loss = -sum(losses)
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else None
    average_trade = (sum(pnls) / trade_count) if trade_count else None
    expectancy = average_trade  # numerically identical; kept as a distinct named field for reporting

    return StandardMetrics(
        total_return=total_return,
        net_pnl=net_pnl,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        drawdown_duration_bars=dd_duration,
        cagr=cagr,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        trade_count=trade_count,
        win_rate=win_rate,
        profit_factor=profit_factor,
        expectancy=expectancy,
        average_trade=average_trade,
        exposure_time_fraction=_exposure_time_fraction(trades, equity.index),
    )
