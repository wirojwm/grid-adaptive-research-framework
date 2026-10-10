"""Run the Fixed Grid control on a seeded synthetic mean-reverting series and print metrics.

The data is synthetic. The output demonstrates the API and the metrics; it is not
evidence about any market.

    python examples/synthetic_fixed_grid.py
"""

import numpy as np
import pandas as pd

from gridadaptive.execution.costs import CostModel
from gridadaptive.execution.simulator import ExecutionSimulator
from gridadaptive.metrics.grid_metrics import compute_grid_metrics
from gridadaptive.strategy.fixed_grid import FixedGridConfig, FixedGridEngine, default_anchor_price


def synthetic_bars(n: int = 2000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = np.empty(n)
    x = 0.0
    for i in range(n):  # Ornstein-Uhlenbeck-style mean reversion around 1.1000
        x += -0.05 * x + rng.normal(0.0, 0.0004)
        close[i] = 1.1000 + x
    open_ = np.concatenate([[1.1000], close[:-1]])
    wiggle = np.abs(rng.normal(0.0, 0.0001, n))
    index = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + wiggle,
            "low": np.minimum(open_, close) - wiggle,
            "close": close,
        },
        index=index,
    )


def main() -> None:
    bars = synthetic_bars()
    config = FixedGridConfig(symbol="SYNTH", grid_spacing=0.0010, levels_per_side=3, size_per_level=1000)
    sim = ExecutionSimulator(bars, CostModel(spread=0.00012, commission_per_unit=0.000035))
    result = FixedGridEngine(config, sim, anchor_price=default_anchor_price(bars)).run()
    metrics = compute_grid_metrics(result, bars, starting_capital=10_000.0)
    print(f"closed cycles:              {metrics.grid_cycles}")
    print(f"open positions at end:      {len(result.open_positions)}")
    print(f"realized PnL:               {metrics.realized_pnl:.4f}")
    print(f"unrealized PnL at end:      {metrics.unrealized_pnl_at_end:.4f}")
    print(f"max adverse inventory:      {metrics.max_adverse_inventory:.4f}")
    print(f"total transaction cost:     {metrics.total_transaction_cost:.4f}")


if __name__ == "__main__":
    main()
