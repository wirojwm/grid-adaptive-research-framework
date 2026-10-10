# Adding a new market

Portability to another market is a **hypothesis to test, not an assumption.** Nothing in this repository has been validated for any particular market, broker or exchange.

Moving to another instrument or data source (for example BTCUSDT, XAUUSD, another FX pair, 1-second bars, trade-level data, or another exchange or broker) requires separate validation of:

- **Data schema** and timestamp convention (open- vs. close-labeled bars, timezone, DST).
- **Tick and price normalization** (precision, tick size, bid vs. ask vs. last).
- **Fees,** spreads and slippage models appropriate to the venue.
- **Fill semantics** (what counts as a touch; limit-order priority; partial fills).
- **Exchange or broker filters** (minimum size, lot step, price filters).
- **Position sizing** and margin rules.
- **Accounting** (quote vs. base currency, financing or funding costs, contract specifications).
- **Mechanical-equivalence tests** showing the engine still behaves as specified on the new data.

Only after those pass can an economic question about the new market be asked, and it then needs its own preregistration and untouched evidence.
