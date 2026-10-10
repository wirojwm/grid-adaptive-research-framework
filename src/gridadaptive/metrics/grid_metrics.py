"""Grid-specific metrics for the Fixed Grid baseline.

`FixedGridEngine` never steps bar-by-bar (it runs each level as an independent event
chain — see gridadaptive.strategy.fixed_grid), so it has no natural continuous
equity/inventory curve. Metrics that only need discrete open/close events (grid
cycles, layers filled, max simultaneous layers, turnover, capital utilization,
position buildup) are computed directly from those events. Metrics that need a full
price path (max adverse inventory, recovery time) are computed here as a second pass
over the `bars` DataFrame, revaluing every open position at every bar's close. This
is O(bars x positions) — fine for the synthetic scenarios this module is tested
against; worth revisiting for performance if a real multi-month dataset
makes it a bottleneck.

Sign convention: `max_adverse_inventory` is reported as a non-negative magnitude (0
if unrealized PnL never went negative).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from gridadaptive.strategy.fixed_grid import FixedGridResult


@dataclass(frozen=True)
class GridMetrics:
    grid_cycles: int
    layers_filled: int
    max_simultaneous_layers: int
    realized_pnl: float
    unrealized_pnl_at_end: float
    max_adverse_inventory: float
    recovery_time_bars: int | None
    capital_utilization: float
    margin_utilization_approx: float
    position_buildup_during_trend: int
    turnover: float
    total_transaction_cost: float
    transaction_cost_drag_vs_capital: float
    transaction_cost_drag_vs_gross_pnl: float | None
    recenter_frequency: int = 0
    shutdown_frequency: int = 0
    risk_guard_activations: int = 0


def _position_events(result: FixedGridResult) -> list[tuple[pd.Timestamp, pd.Timestamp | None, str, float, float]]:
    """(open_time, close_time_or_None, side, open_price, size) for every position."""
    events = []
    for c in result.cycles:
        events.append((c.open_time, c.close_time, c.side, c.open_price, c.size))
    for p in result.open_positions:
        events.append((p.open_time, None, p.side, p.open_price, p.size))
    return events


def _sweep_max(deltas: dict[pd.Timestamp, float]) -> float:
    running = 0.0
    peak = 0.0
    for t in sorted(deltas):
        running += deltas[t]
        peak = max(peak, running)
    return peak


def _max_simultaneous_layers(events) -> int:
    deltas: dict[pd.Timestamp, float] = {}
    for open_time, close_time, *_ in events:
        deltas[open_time] = deltas.get(open_time, 0.0) + 1
        if close_time is not None:
            deltas[close_time] = deltas.get(close_time, 0.0) - 1
    return int(_sweep_max(deltas))


def _peak_notional_exposure(events) -> float:
    deltas: dict[pd.Timestamp, float] = {}
    for open_time, close_time, _side, open_price, size in events:
        notional = open_price * size
        deltas[open_time] = deltas.get(open_time, 0.0) + notional
        if close_time is not None:
            deltas[close_time] = deltas.get(close_time, 0.0) - notional
    return _sweep_max(deltas)


def mark_to_market_curve(result: FixedGridResult, bars: pd.DataFrame) -> pd.Series:
    """Aggregate unrealized PnL of open positions at every bar's own close.

    A position is "open at bar t" iff open_time <= t < close_time (or still open,
    close_time is None). Uses only each bar's own close — no lookahead.
    """
    events = _position_events(result)
    values = []
    for t in bars.index:
        close_px = float(bars.loc[t, "close"])
        total = 0.0
        for open_time, close_time, side, open_price, size in events:
            if open_time <= t and (close_time is None or t < close_time):
                if side == "buy":
                    total += (close_px - open_price) * size
                else:
                    total += (open_price - close_px) * size
        values.append(total)
    return pd.Series(values, index=bars.index, name="unrealized_pnl")


def realized_pnl_curve(result: FixedGridResult, bars: pd.DataFrame) -> pd.Series:
    """Cumulative realized PnL, stepping up at each cycle's close_time."""
    increments: dict[pd.Timestamp, float] = {}
    for c in result.cycles:
        increments[c.close_time] = increments.get(c.close_time, 0.0) + c.realized_pnl
    cum = 0.0
    values = []
    for t in bars.index:
        cum += increments.get(t, 0.0)
        values.append(cum)
    return pd.Series(values, index=bars.index, name="realized_pnl_cum")


def equity_curve(result: FixedGridResult, bars: pd.DataFrame, starting_capital: float) -> pd.Series:
    return starting_capital + realized_pnl_curve(result, bars) + mark_to_market_curve(result, bars)


def _max_adverse_inventory_and_recovery(
    mtm: pd.Series, equity: pd.Series
) -> tuple[float, int | None]:
    if len(mtm) == 0:
        return 0.0, 0

    trough_pos = int(mtm.values.argmin())
    max_adverse = -min(0.0, float(mtm.iloc[trough_pos]))
    if max_adverse == 0.0:
        return 0.0, 0

    prior_peak = float(equity.iloc[: trough_pos + 1].max())
    for i in range(trough_pos + 1, len(equity)):
        if equity.iloc[i] >= prior_peak:
            return max_adverse, i - trough_pos
    return max_adverse, None


def _position_buildup_during_trend(result: FixedGridResult) -> int:
    events: list[tuple[pd.Timestamp, str, str]] = []
    for c in result.cycles:
        events.append((c.open_time, "entry", c.side))
        events.append((c.close_time, "close", c.side))
    for p in result.open_positions:
        events.append((p.open_time, "entry", p.side))
    events.sort(key=lambda e: e[0])

    max_streak = 0
    current_streak = 0
    current_side: str | None = None
    for _t, kind, side in events:
        if kind == "close":
            current_streak = 0
            current_side = None
            continue
        if side == current_side:
            current_streak += 1
        else:
            current_streak = 1
            current_side = side
        max_streak = max(max_streak, current_streak)
    return max_streak


def _turnover_notional(result: FixedGridResult) -> float:
    total = 0.0
    for c in result.cycles:
        total += c.open_price * c.size + c.close_price * c.size
    for p in result.open_positions:
        total += p.open_price * p.size
    return total


def _total_transaction_cost(result: FixedGridResult) -> float:
    total = sum(c.all_in_cost for c in result.cycles)
    total += sum(p.spread_cost + p.commission_cost + p.slippage_cost for p in result.open_positions)
    return total


def _gross_pnl(result: FixedGridResult) -> float:
    return sum(c.gross_pnl for c in result.cycles)


def compute_grid_metrics(
    result: FixedGridResult, bars: pd.DataFrame, starting_capital: float
) -> GridMetrics:
    if starting_capital <= 0:
        raise ValueError("starting_capital must be positive")

    events = _position_events(result)
    mtm = mark_to_market_curve(result, bars)
    equity = equity_curve(result, bars, starting_capital)
    max_adverse, recovery_bars = _max_adverse_inventory_and_recovery(mtm, equity)

    unrealized_at_end = float(mtm.iloc[-1]) if len(mtm) else 0.0
    realized_total = sum(c.realized_pnl for c in result.cycles)
    gross_pnl = _gross_pnl(result)
    total_cost = _total_transaction_cost(result)
    turnover_notional = _turnover_notional(result)
    peak_notional = _peak_notional_exposure(events)

    return GridMetrics(
        grid_cycles=len(result.cycles),
        layers_filled=len(result.cycles) + len(result.open_positions),
        max_simultaneous_layers=_max_simultaneous_layers(events),
        realized_pnl=realized_total,
        unrealized_pnl_at_end=unrealized_at_end,
        max_adverse_inventory=max_adverse,
        recovery_time_bars=recovery_bars,
        capital_utilization=peak_notional / starting_capital,
        margin_utilization_approx=peak_notional / starting_capital,
        position_buildup_during_trend=_position_buildup_during_trend(result),
        turnover=turnover_notional / starting_capital,
        total_transaction_cost=total_cost,
        transaction_cost_drag_vs_capital=total_cost / starting_capital,
        transaction_cost_drag_vs_gross_pnl=(total_cost / gross_pnl) if gross_pnl != 0 else None,
    )
