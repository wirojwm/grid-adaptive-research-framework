# Architecture

```
Data → Features → Regime Detection → Strategy → Risk → Order Planning
     → Execution → Portfolio Control → Monitoring
```

Package layout (`src/gridadaptive/`):

| Package | Responsibility |
|---|---|
| `core` | `TradingIntent` (what a strategy asks for) and fill outcome types |
| `data` | OHLCV schema validation, CSV loading, quality checks, resampling, UTC localization, dataset provenance |
| `features` | Pure functions of OHLC data (e.g. Wilder ATR, DMI/ADX) |
| `strategy` | Fixed Grid engine and reference strategies (no-trade, buy-and-hold) |
| `execution` | `ExecutionSimulator` and `CostModel`: the only layer that models fills, spread, commission and slippage |
| `metrics` | Standard performance metrics and grid-specific metrics |
| `experiment` | Experiment records (append-only ledger), reproducibility metadata, cost-sensitivity scenarios |

The Fixed Grid engine exposes an optional entry-decision hook, which is the extension point for adding risk guards without changing the control's behavior. With no hook, the engine is the unmodified control.

## Boundaries

- Strategy logic produces `TradingIntent`s only. It does not know about broker mechanics, fills, or spread.
- Execution simulation independently models fills and costs. It is the only layer allowed to model broker-like behavior.
- Risk guards are evaluated independently of strategy logic and may veto or shrink intents. Every activation should be logged with its cause.
- Features are pure functions of past and present bars. A value at index `i` must never depend on a bar after `i`.
- Broker or exchange adapters must not silently alter strategy semantics.

## Execution contract

- No same-bar execution. An intent decided at a bar's close can only fill on a strictly later bar.
- Limit orders fill at the limit price once touched, plus spread and commission.
- An intent whose `decision_time` is not in the dataset is rejected, not guessed at.
- Zero-cost defaults are not provided: spread and commission are required, explicit choices.
