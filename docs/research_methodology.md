# Research methodology

```
Research → Hypothesis → Experiment → Evidence → Verdict → Improvement → Implementation
```

```
Hypothesis → Test Specification → RED Tests → Implementation → GREEN Tests
           → Backtest → Robustness Test → Verdict
```

## Principles

- The goal is to determine whether an apparent edge is **real, robust, economically meaningful, and sufficiently controlled** — not to maximize historical backtest profit.
- Validated edge, weak or inconclusive evidence, and no validated edge are all acceptable conclusions.
- Never change experiments, parameters, metrics or validation rules to make a result look better. If a change would improve a result and cannot be justified independently of that effect, do not make it.
- Always keep a simple control runnable and reported alongside any variant.
- Change **one component at a time** relative to the control, so any change in outcome can be attributed.
- Write the hypothesis, the metric, the pass/fail criteria and the verdict mapping **before** looking at results.

## Verdicts

`SUPPORTED`, `PARTIALLY SUPPORTED`, `NOT SUPPORTED`, `INCONCLUSIVE` (see `research_verdicts.md`). A failed hypothesis is useful evidence and is not retuned until a profitable backtest appears.

## Adaptive components are candidates

ATR or volatility-adaptive spacing, regime filters, dynamic depth, exposure guards, recenter/reset, cascading grids, dynamic position sizing, deterministic risk controls and future ML permission/filter layers are **research candidates, not proven improvements.**

**Portability and improvement are hypotheses to test, not assumptions.**

ML, if added, is a supporting layer and never the trading authority:

```
Research → Shadow → Advisory → Permission / Filter → Controlled Parameter Adjustment
```

ML must never bypass deterministic risk controls.

## Anti-overfitting rules

- No lookahead; no future-data leakage.
- No random time-series split used as evidence of generalization.
- No repeated threshold hunting. No test-set tuning.
- Transaction costs are required; spread, slippage and execution assumptions are explicit.
- A sharp isolated parameter optimum is suspicious. Prefer stable parameter regions.
- Keep a sealed validation or holdout partition, touched only after the methodology is frozen.
- Record every experiment (ledger entries are append-only; a correction is a new record that references the one it supersedes).
- New hypotheses require an independent, ex-ante rationale.
