"""Outcomes of executing a TradingIntent against simulated market data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

Status = Literal["filled", "unfilled", "rejected"]


@dataclass(frozen=True)
class IntentOutcome:
    intent_id: str
    symbol: str
    side: str
    status: Status
    fill_time: pd.Timestamp | None = None
    fill_price: float | None = None
    filled_size: float | None = None
    spread_cost: float = 0.0
    slippage_cost: float = 0.0
    commission_cost: float = 0.0
    reason: str = ""
