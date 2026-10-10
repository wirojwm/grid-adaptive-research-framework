"""Symmetric Fixed Grid baseline strategy .

This is the control (baseline) strategy: a two-sided
mean-reversion grid with a fixed anchor, fixed spacing, and a fixed number of levels
per side, set once and never recalculated. No ATR/ADXVMA/regime input, no dynamic
sizing, no recenter, no reinvestment — those are separate, independently-tested components.

Architecture note: each grid level is run as an independent chain of
`ExecutionSimulator.simulate()` calls (entry -> take-profit -> re-arm -> ...), each
new intent's `decision_time` set from the previous outcome's `fill_time`. This needs
no changes to the execution simulator, because `simulate()` already scans forward from
an arbitrary `decision_time` to first touch and fills limit orders at an exact price.

Take-profit pricing: the exit for a filled level is priced at the *adjacent fixed
grid line* — `level.price + ΔP` for a buy level, `level.price - ΔP` for a sell level
— not at the entry's actual (spread-adjusted) fill price. Anchoring the exit to the
entry's fill price instead is also a defensible design and was an early draft's wording,
but it makes the entry and exit spread costs partially
cancel in a way that breaks the simple relationship
`realized_pnl = ΔP * size - (spread_cost + commission_cost + slippage_cost)`. Pricing
the exit off the level's own fixed price keeps that relationship exact, which matters
for a component whose entire purpose is to be simple and auditable.

Cost accounting: `IntentOutcome.fill_price` already has spread (and, for market
orders, slippage) baked in by the execution simulator. `GridCycle.gross_pnl` here is
computed from the *undiscounted* grid distance (`ΔP * size`), not from fill-price
differences, specifically so that `all_in_cost` (spread + commission + slippage) can
be subtracted from it exactly once. Do not compute gross_pnl from fill prices and
then also subtract spread_cost — that double-counts spread.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from gridadaptive.core.fills import IntentOutcome
from gridadaptive.core.intents import TradingIntent
from gridadaptive.execution.simulator import ExecutionSimulator

_ENTRY_SIDES = ("buy", "sell")

# Optional entry-decision hook:
# given a level and its natural entry decision_time (t0 or the previous cycle's
# exit fill_time), returns the decision_time to actually submit the entry with,
# or None if no entry should ever be submitted for this arm attempt. When absent
# (the default), behavior is byte-identical to the unhooked engine — this is
# what makes a guard's disabled/inert path reproduce Fixed Grid exactly.
EntryDecisionHook = Callable[["GridLevel", pd.Timestamp], "pd.Timestamp | None"]


@dataclass(frozen=True)
class FixedGridConfig:
    symbol: str
    grid_spacing: float
    levels_per_side: int
    size_per_level: float
    take_profit_distance: float | None = None
    max_cycles_per_level: int = 100_000

    def __post_init__(self) -> None:
        if self.grid_spacing <= 0:
            raise ValueError("grid_spacing must be positive")
        if not isinstance(self.levels_per_side, int) or self.levels_per_side <= 0:
            raise ValueError("levels_per_side must be a positive integer")
        if self.size_per_level <= 0:
            raise ValueError("size_per_level must be positive")
        if self.take_profit_distance is not None and self.take_profit_distance <= 0:
            raise ValueError("take_profit_distance must be positive when set")
        if self.max_cycles_per_level <= 0:
            raise ValueError("max_cycles_per_level must be positive")

    @property
    def resolved_take_profit_distance(self) -> float:
        return self.take_profit_distance if self.take_profit_distance is not None else self.grid_spacing


@dataclass(frozen=True)
class GridLevel:
    level_id: str
    side: str  # "buy" or "sell" — the level's entry side
    price: float


@dataclass(frozen=True)
class GridCycle:
    level_id: str
    side: str
    open_time: pd.Timestamp
    open_price: float
    close_time: pd.Timestamp
    close_price: float
    size: float
    spread_cost: float  # informational: already reflected in fill prices, not a second subtrahend
    commission_cost: float
    slippage_cost: float
    gross_pnl: float  # ΔP * size, undiscounted grid-spacing profit
    all_in_cost: float  # spread_cost + commission_cost + slippage_cost
    realized_pnl: float  # gross_pnl - all_in_cost


@dataclass(frozen=True)
class OpenPosition:
    level_id: str
    side: str
    open_time: pd.Timestamp
    open_price: float
    size: float
    spread_cost: float
    commission_cost: float
    slippage_cost: float


@dataclass(frozen=True)
class FixedGridResult:
    anchor_price: float
    levels: list[GridLevel]
    cycles: list[GridCycle]
    open_positions: list[OpenPosition]


def default_anchor_price(bars: pd.DataFrame) -> float:
    """Default anchor: the first bar's close."""
    return float(bars.iloc[0]["close"])


def build_levels(anchor_price: float, config: FixedGridConfig) -> list[GridLevel]:
    levels: list[GridLevel] = []
    for k in range(1, config.levels_per_side + 1):
        levels.append(GridLevel(level_id=f"buy_{k}", side="buy", price=anchor_price - k * config.grid_spacing))
    for k in range(1, config.levels_per_side + 1):
        levels.append(GridLevel(level_id=f"sell_{k}", side="sell", price=anchor_price + k * config.grid_spacing))
    return levels


def _leg_cost(outcome: IntentOutcome) -> tuple[float, float, float]:
    return outcome.spread_cost, outcome.commission_cost, outcome.slippage_cost


class FixedGridEngine:
    def __init__(
        self,
        config: FixedGridConfig,
        simulator: ExecutionSimulator,
        anchor_price: float,
        entry_decision_hook: EntryDecisionHook | None = None,
    ):
        self.config = config
        self.simulator = simulator
        self.anchor_price = anchor_price
        self.levels = build_levels(anchor_price, config)
        self.entry_decision_hook = entry_decision_hook

    def run(self) -> FixedGridResult:
        t0 = self.simulator.bars.index[0]
        cycles: list[GridCycle] = []
        open_positions: list[OpenPosition] = []
        for level in self.levels:
            level_cycles, open_position = self._run_level(level, t0)
            cycles.extend(level_cycles)
            if open_position is not None:
                open_positions.append(open_position)
        return FixedGridResult(
            anchor_price=self.anchor_price,
            levels=self.levels,
            cycles=cycles,
            open_positions=open_positions,
        )

    def _run_level(
        self, level: GridLevel, t0: pd.Timestamp
    ) -> tuple[list[GridCycle], OpenPosition | None]:
        cycles: list[GridCycle] = []
        decision_time = t0
        exit_side = "sell" if level.side == "buy" else "buy"
        tp_distance = self.config.resolved_take_profit_distance
        take_profit_price = level.price + tp_distance if level.side == "buy" else level.price - tp_distance

        for cycle_index in range(self.config.max_cycles_per_level):
            if self.entry_decision_hook is not None:
                resolved_decision_time = self.entry_decision_hook(level, decision_time)
                if resolved_decision_time is None:
                    return cycles, None
                decision_time = resolved_decision_time

            entry_intent = TradingIntent(
                intent_id=f"{level.level_id}-entry-{cycle_index}",
                symbol=self.config.symbol,
                side=level.side,
                order_type="limit",
                decision_time=decision_time,
                requested_size=self.config.size_per_level,
                layer_id=level.level_id,
                limit_price=level.price,
                reason="grid_entry",
            )
            (entry_outcome,) = self.simulator.simulate([entry_intent])
            if entry_outcome.status == "rejected":
                raise RuntimeError(
                    f"unexpected rejection for level {level.level_id} entry: {entry_outcome.reason}"
                )
            if entry_outcome.status == "unfilled":
                return cycles, None

            exit_intent = TradingIntent(
                intent_id=f"{level.level_id}-exit-{cycle_index}",
                symbol=self.config.symbol,
                side=exit_side,
                order_type="limit",
                decision_time=entry_outcome.fill_time,
                requested_size=self.config.size_per_level,
                layer_id=level.level_id,
                limit_price=take_profit_price,
                reason="grid_take_profit",
            )
            (exit_outcome,) = self.simulator.simulate([exit_intent])
            if exit_outcome.status == "rejected":
                raise RuntimeError(
                    f"unexpected rejection for level {level.level_id} exit: {exit_outcome.reason}"
                )
            if exit_outcome.status == "unfilled":
                spread, commission, slippage = _leg_cost(entry_outcome)
                open_position = OpenPosition(
                    level_id=level.level_id,
                    side=level.side,
                    open_time=entry_outcome.fill_time,
                    open_price=entry_outcome.fill_price,
                    size=self.config.size_per_level,
                    spread_cost=spread,
                    commission_cost=commission,
                    slippage_cost=slippage,
                )
                return cycles, open_position

            cycles.append(
                self._build_cycle(level, entry_outcome, exit_outcome, self.config.size_per_level, tp_distance)
            )
            decision_time = exit_outcome.fill_time

        raise RuntimeError(
            f"level {level.level_id} exceeded max_cycles_per_level="
            f"{self.config.max_cycles_per_level}; check for a zero-distance or misconfigured loop"
        )

    @staticmethod
    def _build_cycle(
        level: GridLevel,
        entry_outcome: IntentOutcome,
        exit_outcome: IntentOutcome,
        size: float,
        tp_distance: float,
    ) -> GridCycle:
        entry_spread, entry_commission, entry_slippage = _leg_cost(entry_outcome)
        exit_spread, exit_commission, exit_slippage = _leg_cost(exit_outcome)
        spread_cost = entry_spread + exit_spread
        commission_cost = entry_commission + exit_commission
        slippage_cost = entry_slippage + exit_slippage
        all_in_cost = spread_cost + commission_cost + slippage_cost
        gross_pnl = tp_distance * size

        return GridCycle(
            level_id=level.level_id,
            side=level.side,
            open_time=entry_outcome.fill_time,
            open_price=entry_outcome.fill_price,
            close_time=exit_outcome.fill_time,
            close_price=exit_outcome.fill_price,
            size=size,
            spread_cost=spread_cost,
            commission_cost=commission_cost,
            slippage_cost=slippage_cost,
            gross_pnl=gross_pnl,
            all_in_cost=all_in_cost,
            realized_pnl=gross_pnl - all_in_cost,
        )
