"""No-Trade and Buy-and-Hold reference strategies.

Buy-and-Hold is a behavioral/reference benchmark here, not a target the Fixed Grid
strategy is assumed to have to beat directly.

`ReferenceTrade` deliberately does **not** reuse `gridadaptive.strategy.fixed_grid.GridCycle`
even though the fields look similar: `GridCycle.gross_pnl` is the undiscounted grid
spacing (`ΔP * size`), cost-free by construction, so `realized_pnl = gross_pnl -
all_in_cost` holds exactly for a grid cycle. A buy-and-hold trade has no such fixed
distance — its `gross_pnl` is the raw fill-price difference, which the
execution simulator already computes *net of* spread and slippage (both are baked
into `fill_price` for a market order). Reusing `GridCycle`'s formula here would
silently double-count spread and slippage, the same bug class corrected in the grid
take-profit logic — so this type has its own,
deliberately different, cost formula: `realized_pnl = gross_pnl - commission_cost`
only.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from gridadaptive.core.intents import TradingIntent
from gridadaptive.execution.simulator import ExecutionSimulator


@dataclass(frozen=True)
class ReferenceTrade:
    label: str
    side: str
    open_time: pd.Timestamp
    open_price: float
    close_time: pd.Timestamp
    close_price: float
    size: float
    spread_cost: float
    commission_cost: float
    slippage_cost: float
    gross_pnl: float  # fill-price-based; spread & slippage already embedded
    realized_pnl: float  # gross_pnl - commission_cost (spread/slippage NOT subtracted again)


def no_trade_equity_curve(bars: pd.DataFrame, starting_capital: float) -> pd.Series:
    """The no-trade reference: capital never moves."""
    return pd.Series(starting_capital, index=bars.index, name="equity")


def buy_and_hold_reference(
    bars: pd.DataFrame, simulator: ExecutionSimulator, symbol: str, size: float
) -> ReferenceTrade:
    """Buy at the window's first eligible bar, hold, sell at the window's last bar —
    one round trip through the same execution simulator every other strategy uses,
    so it is costed identically."""
    if len(bars) < 3:
        raise ValueError(
            "buy_and_hold_reference needs at least 3 bars (entry decision, entry fill, exit fill)"
        )

    entry_intent = TradingIntent(
        intent_id="buy_and_hold-entry",
        symbol=symbol,
        side="buy",
        order_type="market",
        decision_time=bars.index[0],
        requested_size=size,
        layer_id="buy_and_hold",
        reason="reference_buy_and_hold",
    )
    (entry_outcome,) = simulator.simulate([entry_intent])
    if entry_outcome.status != "filled":
        raise RuntimeError(f"buy_and_hold entry did not fill: {entry_outcome.status} ({entry_outcome.reason})")

    exit_intent = TradingIntent(
        intent_id="buy_and_hold-exit",
        symbol=symbol,
        side="sell",
        order_type="market",
        decision_time=bars.index[-2],
        requested_size=size,
        layer_id="buy_and_hold",
        reason="reference_buy_and_hold",
    )
    (exit_outcome,) = simulator.simulate([exit_intent])
    if exit_outcome.status != "filled":
        raise RuntimeError(f"buy_and_hold exit did not fill: {exit_outcome.status} ({exit_outcome.reason})")

    gross_pnl = (exit_outcome.fill_price - entry_outcome.fill_price) * size
    spread_cost = entry_outcome.spread_cost + exit_outcome.spread_cost
    commission_cost = entry_outcome.commission_cost + exit_outcome.commission_cost
    slippage_cost = entry_outcome.slippage_cost + exit_outcome.slippage_cost

    return ReferenceTrade(
        label="buy_and_hold",
        side="buy",
        open_time=entry_outcome.fill_time,
        open_price=entry_outcome.fill_price,
        close_time=exit_outcome.fill_time,
        close_price=exit_outcome.fill_price,
        size=size,
        spread_cost=spread_cost,
        commission_cost=commission_cost,
        slippage_cost=slippage_cost,
        gross_pnl=gross_pnl,
        realized_pnl=gross_pnl - commission_cost,
    )
