"""Execution simulator.

Design decisions made explicit here:

- No same-bar execution for *any* order type. `decision_time` marks the bar
  whose close produced the intent; fills are only searched among bars with a
  timestamp strictly greater than `decision_time`. A description of a limit order filling "intrabar at bar t" refers to touch
  detection within whichever *future* bar is being scanned, not the bar that
  produced the decision — that reading would let an order use information
  (that same bar's high/low) that wasn't yet known when the order was placed.
  This ambiguity is resolved here, conservatively, against the strategy.
- Market intents fill at the open of the first eligible (future) bar.
- Limit intents rest until the first eligible bar whose high/low touches the
  limit price, and fill at the limit price itself (not a more favorable
  intrabar price), plus spread and commission. No slippage is modeled on
  limit fills — the price is assumed guaranteed once touched.
  Partial fills, order-book depth, and stop-vs-limit same-bar tie-breaking
  (relevant once risk-guard stops exist) are out of scope until
  those components exist.
- `decision_time` must correspond to a bar timestamp already present in the
  dataset. An intent whose `decision_time` is unknown to the dataset is
  rejected rather than guessed at — this is a lookahead/misalignment guard,
  not a normal "unfilled" outcome, and must not be silently treated as one.
"""

from __future__ import annotations

import pandas as pd

from gridadaptive.core.fills import IntentOutcome
from gridadaptive.core.intents import TradingIntent
from gridadaptive.execution.costs import CostModel


class ExecutionSimulator:
    def __init__(self, bars: pd.DataFrame, cost_model: CostModel):
        if not bars.index.is_monotonic_increasing:
            raise ValueError("bars must be sorted ascending by timestamp")
        self.bars = bars
        self.cost_model = cost_model

    def simulate(self, intents: list[TradingIntent]) -> list[IntentOutcome]:
        return [self._simulate_one(intent) for intent in intents]

    def _simulate_one(self, intent: TradingIntent) -> IntentOutcome:
        if intent.decision_time not in self.bars.index:
            return IntentOutcome(
                intent_id=intent.intent_id,
                symbol=intent.symbol,
                side=intent.side,
                status="rejected",
                reason=(
                    "decision_time is not a known bar timestamp in this dataset; "
                    "refusing to fill against data the backtest has not reached"
                ),
            )

        future_bars = self.bars.loc[self.bars.index > intent.decision_time]
        if future_bars.empty:
            return IntentOutcome(
                intent_id=intent.intent_id,
                symbol=intent.symbol,
                side=intent.side,
                status="unfilled",
                reason="no future bar available after decision_time",
            )

        if intent.order_type == "market":
            return self._fill_market(intent, future_bars)
        return self._fill_limit(intent, future_bars)

    def _fill_market(self, intent: TradingIntent, future_bars: pd.DataFrame) -> IntentOutcome:
        fill_ts = future_bars.index[0]
        bar = future_bars.iloc[0]
        raw_price = float(bar["open"])
        bar_range = float(bar["high"] - bar["low"])
        return self._apply_costs(intent, fill_ts, raw_price, bar_range, apply_slippage=True)

    def _fill_limit(self, intent: TradingIntent, future_bars: pd.DataFrame) -> IntentOutcome:
        for ts, bar in future_bars.iterrows():
            touched = (intent.side == "buy" and bar["low"] <= intent.limit_price) or (
                intent.side == "sell" and bar["high"] >= intent.limit_price
            )
            if touched:
                bar_range = float(bar["high"] - bar["low"])
                return self._apply_costs(
                    intent, ts, intent.limit_price, bar_range, apply_slippage=False
                )
        return IntentOutcome(
            intent_id=intent.intent_id,
            symbol=intent.symbol,
            side=intent.side,
            status="unfilled",
            reason="limit price never touched in remaining data",
        )

    def _apply_costs(
        self,
        intent: TradingIntent,
        fill_ts: pd.Timestamp,
        raw_price: float,
        bar_range: float,
        apply_slippage: bool,
    ) -> IntentOutcome:
        spread_per_unit = self.cost_model.spread_cost_per_unit()
        slip_per_unit = (
            self.cost_model.slippage_cost_per_unit(bar_range) if apply_slippage else 0.0
        )
        direction = 1 if intent.side == "buy" else -1
        fill_price = raw_price + direction * (spread_per_unit + slip_per_unit)

        return IntentOutcome(
            intent_id=intent.intent_id,
            symbol=intent.symbol,
            side=intent.side,
            status="filled",
            fill_time=fill_ts,
            fill_price=fill_price,
            filled_size=intent.requested_size,
            spread_cost=spread_per_unit * intent.requested_size,
            slippage_cost=slip_per_unit * intent.requested_size,
            commission_cost=self.cost_model.commission_per_unit * intent.requested_size,
        )
