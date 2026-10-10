# Fixed Grid mechanics

The Fixed Grid is the control strategy. Every adaptive component is compared against it.

## Structure

- An **anchor price** (by default the first bar's close).
- **N BUY levels** below the anchor and **N SELL levels** above it, spaced `ΔP` apart: **2N structural levels** in total.
- **One position per level.** BUY and SELL level chains are independent of each other.

## Level life cycle

1. **Entry:** a resting limit order at the level price.
2. **Take-profit:** once filled, an exit is placed one grid step away, at the adjacent fixed grid line (`level + ΔP` for BUY, `level − ΔP` for SELL). It is priced off the level, not the actual fill price, so entry and exit spread costs do not partially cancel.
3. **Exit:** the take-profit fills when touched.
4. **Re-arm:** after the exit, the level's entry order is placed again, starting from the exit fill time.

## Sizing and costs

- Fixed position size per level.
- For a closed cycle, `realized_pnl = ΔP × size − (spread_cost + commission_cost + slippage_cost)`. Gross PnL is the undiscounted grid distance, so costs are subtracted exactly once.

## Accounting

- **Realized PnL** from closed cycles.
- **Mark-to-market PnL** of open positions, revalued at every bar's close.
- **Equity** = starting capital + realized PnL + mark-to-market PnL.
- **Drawdown, exposure, turnover, capital utilization, transaction-cost drag.**
- **Max adverse inventory:** the worst open-position loss, as a non-negative magnitude.

## Closed-cycle profitability alone is not sufficient

A grid can realize many small profits while building large open inventory in a trend. A run that ends mid-trend can look good on closed-trade statistics precisely because its worst positions are not yet "trades". Always read realized PnL together with unrealized PnL, equity, drawdown, adverse inventory, exposure, terminal open inventory and transaction costs.

Deeper grids are not automatically better. More levels add more independent chains, and more potential inventory.
