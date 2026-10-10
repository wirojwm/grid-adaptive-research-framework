# Validation workflow

1. Define the claimed edge and why it might exist.
2. Define failure modes and what would falsify the claim.
3. Establish the data contract (see `data/schema.py`, `data/quality.py`).
4. Define strategy mechanics.
5. Write RED tests: hand-computable synthetic scenarios that pin the behavior.
6. Implement the minimum mechanics; get GREEN.
7. Freeze parameters and criteria **before** economic observation.
8. Run development-partition research.
9. Evaluate economics, drawdown, exposure, tail risk, turnover and transaction costs.
10. Run robustness and sensitivity checks (e.g. `experiment/cost_scenarios.py`).
11. Use untouched evidence only once, after preregistration.
12. Assign a verdict. Do not rescue a failed result through result-aware tuning.

## Grid-specific checks

Report these alongside any generic backtest metric: realized vs. unrealized PnL, max adverse inventory, recovery time, max simultaneous layers, peak notional exposure, position build-up during trends, turnover, and transaction-cost drag. A smooth equity curve can hide large open inventory.

## Reproducibility

Each experiment record should capture: experiment ID, strategy version, git commit, dataset identity and hash, symbol, timeframe, date range, full parameters, execution and cost assumptions, starting capital, random seed, and output metrics (see `experiment/record.py`, `experiment/repro.py`, `data/provenance.py`).
