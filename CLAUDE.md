# CLAUDE.md — contributor / coding-agent rules

This is a research and edge-validation framework, not a trading bot.

- Preserve research semantics. Do not change fill, cost, or accounting behavior
  as a side effect of another change.
- Do not weaken or delete tests to make an implementation pass. Follow
  Spec → RED tests → Implementation → GREEN tests.
- No lookahead, no same-bar execution, no omitted costs.
- Never tune parameters or relax criteria to make a result look better.
- No secrets, broker credentials, or `.env` contents in the repository.
- No live trading or real order placement code.
- No market data in the repository; use synthetic fixtures.
- Make no unsupported claims about profitability or edge.
