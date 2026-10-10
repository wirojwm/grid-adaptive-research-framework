"""Cost models for the execution simulator.

Zero-cost defaults are intentionally not provided: every cost component must
be an explicit, config-visible choice (spread and commission are required
constructor arguments). Zero is allowed, but only if written down.
"""

from __future__ import annotations

from dataclasses import dataclass

_NON_NEGATIVE_FIELDS = ("spread", "commission_per_unit", "slippage_base", "slippage_vol_coeff")


@dataclass(frozen=True)
class CostModel:
    spread: float  # full bid-ask spread, in price units; half applied per fill side
    commission_per_unit: float  # currency per unit of requested_size, charged per fill
    slippage_base: float = 0.0  # fixed adverse slippage, in price units
    slippage_vol_coeff: float = 0.0  # multiplied by the fill bar's (high - low) range

    def __post_init__(self) -> None:
        for name in _NON_NEGATIVE_FIELDS:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    def spread_cost_per_unit(self) -> float:
        return self.spread / 2.0

    def slippage_cost_per_unit(self, bar_range: float) -> float:
        return self.slippage_base + self.slippage_vol_coeff * bar_range
