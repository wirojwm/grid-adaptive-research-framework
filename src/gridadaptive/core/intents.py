"""TradingIntent — the only artifact the strategy layer is allowed to produce.

An intent carries no broker or execution knowledge (architecture rule):
it names what should happen and why, not how it will be filled. Fill
mechanics live entirely in gridadaptive.execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

Side = Literal["buy", "sell"]
OrderType = Literal["market", "limit"]

_VALID_SIDES = ("buy", "sell")
_VALID_ORDER_TYPES = ("market", "limit")


@dataclass(frozen=True)
class TradingIntent:
    intent_id: str
    symbol: str
    side: Side
    order_type: OrderType
    decision_time: pd.Timestamp
    requested_size: float
    layer_id: str | None = None
    limit_price: float | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.side not in _VALID_SIDES:
            raise ValueError(f"invalid side: {self.side!r}")
        if self.order_type not in _VALID_ORDER_TYPES:
            raise ValueError(f"invalid order_type: {self.order_type!r}")
        if self.requested_size <= 0:
            raise ValueError("requested_size must be positive")
        if self.order_type == "limit" and self.limit_price is None:
            raise ValueError("limit intents require limit_price")
        if self.decision_time.tzinfo is None:
            raise ValueError("decision_time must be tz-aware (canonical timezone UTC)")
