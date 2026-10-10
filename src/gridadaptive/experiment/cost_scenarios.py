"""Pre-specified cost-sensitivity scenarios.

These measure how sensitive a result is to the still-placeholder, still-unvalidated
execution-cost assumptions — a robustness diagnostic on *execution assumptions*, not
a parameter search. Grid strategy parameters
are never changed in response to these results
(core principle). Every scenario here is pre-specified
before any run — none were chosen after seeing a result, and none will be added or
removed post hoc.
"""

from __future__ import annotations

from gridadaptive.execution.costs import CostModel

# The baseline placeholder.
BASELINE = CostModel(
    spread=0.00012, commission_per_unit=0.000035, slippage_base=0.00002, slippage_vol_coeff=0.0
)

# Best case: isolates strategy signal from all execution frictions.
ZERO_COST = CostModel(spread=0.0, commission_per_unit=0.0, slippage_base=0.0, slippage_vol_coeff=0.0)

# Adverse case: spread doubled, holding commission/slippage fixed — a simple stress
# test pending a variable-spread/spread-stress model.
DOUBLE_SPREAD = CostModel(
    spread=BASELINE.spread * 2,
    commission_per_unit=BASELINE.commission_per_unit,
    slippage_base=BASELINE.slippage_base,
    slippage_vol_coeff=BASELINE.slippage_vol_coeff,
)

COST_SENSITIVITY_SCENARIOS: dict[str, CostModel] = {
    "baseline": BASELINE,
    "zero_cost": ZERO_COST,
    "double_spread": DOUBLE_SPREAD,
}
