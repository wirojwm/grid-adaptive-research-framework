# Grid Adaptive
## Adaptive Grid Strategy Research & Validation Framework

[![CI](https://github.com/wirojwm/grid-adaptive-research-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/wirojwm/grid-adaptive-research-framework/actions/workflows/ci.yml)

This repository is a quantitative research framework for designing, testing, falsifying, and validating fixed and adaptive grid trading hypotheses.

Its purpose is not to maximize historical backtest profit.

Its purpose is to determine whether an apparent trading edge is real, robust, economically meaningful, and sufficiently controlled to justify further development.

> **THIS IS A RESEARCH AND EDGE-VALIDATION FRAMEWORK.**
>
> It is **not** a production-ready trading bot, **not** a validated profitable strategy, **not** proof that grid trading is profitable, and **not** proof that grid trading has no edge.

The framework grew out of extensive internal research. This repository contains the reusable code, tests and methodology only. It contains no market data and publishes no empirical results about any market.

## Evidence at a glance

**Implemented and tested framework components:** Fixed Grid control mechanics, execution-cost simulation, accounting and metrics, data validation and provenance utilities, ATR and DMI/ADX features, and experiment/reproducibility utilities. Automated tests use synthetic data.

**Public validation checkpoint:** [v0.1.0](https://github.com/wirojwm/grid-adaptive-research-framework/releases/tag/v0.1.0) at commit `45c47ffc991849f40e0b8c36e05caf2a1357b069` — [150 public/synthetic tests passed](https://github.com/wirojwm/grid-adaptive-research-framework/actions/runs/38039317857) in GitHub Actions. This is evidence of tested framework behavior, not real-market economic edge or strategy profitability.

**Research status:** Fixed Grid is a mechanically validated research control; its economic edge remains unproven. No real-market empirical verdict is published here. Adaptive grid enhancements, regime/risk guards, recentering, dynamic sizing, and future ML filters remain research candidates, not validated improvements.

**Explore the evidence:**

- [Research methodology](docs/research_methodology.md) and [validation workflow](docs/validation_workflow.md)
- [Architecture](docs/architecture.md) and [Fixed Grid mechanics](docs/fixed_grid_mechanics.md)
- [Verdict definitions and evidence limitations](docs/research_verdicts.md)
- [Synthetic example](examples/synthetic_fixed_grid.py) and [automated tests](tests/)
- [Public v0.1.0 release](https://github.com/wirojwm/grid-adaptive-research-framework/releases/tag/v0.1.0) and [CI run](https://github.com/wirojwm/grid-adaptive-research-framework/actions/runs/38039317857)

## Research philosophy

```
Research → Hypothesis → Experiment → Evidence → Verdict → Improvement → Implementation
```

```
Hypothesis → Test Specification → RED Tests → Implementation → GREEN Tests
           → Backtest → Robustness Test → Verdict
```

Verdicts: **SUPPORTED**, **PARTIALLY SUPPORTED**, **NOT SUPPORTED**, **INCONCLUSIVE**. Failed hypotheses are useful evidence and are not retuned until a backtest looks profitable. See [docs/research_methodology.md](docs/research_methodology.md) and [docs/research_verdicts.md](docs/research_verdicts.md).

## Architecture

```
Data → Features → Regime Detection → Strategy → Risk → Order Planning
     → Execution → Portfolio Control → Monitoring
```

Strategy logic produces intents only. A separate execution simulator is the only layer that models fills, spread, commission and slippage. Risk controls sit outside the strategy and have priority over it. See [docs/architecture.md](docs/architecture.md).

Code is in `src/gridadaptive/` (`core`, `data`, `features`, `strategy`, `execution`, `metrics`, `experiment`), with tests in `tests/` and an example in `examples/`.

## Fixed Grid mechanics

The Fixed Grid is the control strategy:

- an **anchor price**;
- **N** BUY levels below the anchor and **N** SELL levels above it (**2N** structural levels);
- **one open position per level**; BUY and SELL chains are independent;
- per level: entry → take-profit → exit → re-arm;
- fixed size per level, explicit transaction costs, mark-to-market of open inventory, equity and drawdown accounting.

**Closed-cycle profitability alone is not sufficient.** A grid can realize many small profits while accumulating large open inventory. Evaluate realized PnL, unrealized PnL, equity, drawdown, adverse inventory, exposure, terminal open inventory and costs together. Deeper grids are not automatically better. See [docs/fixed_grid_mechanics.md](docs/fixed_grid_mechanics.md).

## Adaptive modules are hypotheses

ATR or volatility-adaptive spacing, regime filters, dynamic depth, exposure guards, recenter/reset, cascading grids, dynamic position sizing, deterministic risk controls and future ML permission/filter layers are **research candidates, not proven improvements.**

**Portability and improvement are hypotheses to test, not assumptions.**

## Anti-overfitting rules

- No lookahead; no future-data leakage; no same-bar execution.
- No random time-series split as evidence of generalization.
- No repeated threshold hunting; no test-set tuning.
- Transaction costs required; spread, slippage and execution assumptions explicit.
- One new component at a time, against a control that always stays runnable.
- Sealed validation or holdout data, touched only after the methodology is frozen.
- Failed hypotheses are not rescued by result-aware tuning.

## Adding a new market

Moving to BTCUSDT, XAUUSD, another FX pair, 1-second data, trade-level data, or another exchange or broker requires separate validation of the data schema, timestamps, price normalization, fees, spreads, slippage, fill semantics, exchange filters, sizing and accounting. No specific market, broker or exchange data feed is validated by this repository. See [docs/adding_a_new_market.md](docs/adding_a_new_market.md).

## Running tests

Requires Python 3.12+.

```
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest
python examples/synthetic_fixed_grid.py
```

All tests use synthetic data. No market data is needed or included.

## Project status / maturity

Early-stage research tooling. The mechanics (Fixed Grid, execution simulation, accounting, metrics, selected features) are covered by synthetic tests. There is no live trading, no broker connectivity and no claim of profitability. **Live trading is out of scope.**

## License

MIT, for the code and documentation in this repository. See [LICENSE](LICENSE). It does not cover any third-party market data or broker material, none of which is included.
