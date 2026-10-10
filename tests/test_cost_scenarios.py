from gridadaptive.execution.costs import CostModel
from gridadaptive.experiment.cost_scenarios import COST_SENSITIVITY_SCENARIOS


def test_baseline_matches_the_documented_placeholder_values():
    baseline = COST_SENSITIVITY_SCENARIOS["baseline"]
    assert baseline.spread == 0.00012
    assert baseline.commission_per_unit == 0.000035
    assert baseline.slippage_base == 0.00002
    assert baseline.slippage_vol_coeff == 0.0


def test_zero_cost_scenario_is_fully_zeroed():
    zero = COST_SENSITIVITY_SCENARIOS["zero_cost"]
    assert zero.spread == 0.0
    assert zero.commission_per_unit == 0.0
    assert zero.slippage_base == 0.0


def test_double_spread_scenario_only_changes_spread():
    baseline = COST_SENSITIVITY_SCENARIOS["baseline"]
    double = COST_SENSITIVITY_SCENARIOS["double_spread"]
    assert double.spread == baseline.spread * 2
    assert double.commission_per_unit == baseline.commission_per_unit
    assert double.slippage_base == baseline.slippage_base


def test_exactly_three_prespecified_scenarios():
    assert len(COST_SENSITIVITY_SCENARIOS) == 3
    assert all(isinstance(cm, CostModel) for cm in COST_SENSITIVITY_SCENARIOS.values())
