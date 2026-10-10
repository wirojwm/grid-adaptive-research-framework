import pandas as pd
import pytest

from gridadaptive.core.intents import TradingIntent
from gridadaptive.execution.costs import CostModel
from gridadaptive.execution.simulator import ExecutionSimulator


def _bars() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": [1.1000, 1.1010, 1.0950, 1.1005, 1.1020],
            "high": [1.1015, 1.1020, 1.0960, 1.1030, 1.1035],
            "low": [1.0995, 1.1000, 1.0940, 1.1000, 1.1010],
            "close": [1.1010, 1.0950, 1.1005, 1.1020, 1.1025],
        },
        index=idx,
    )


def _sim(cost_model: CostModel | None = None) -> ExecutionSimulator:
    cm = cost_model or CostModel(spread=0.0002, commission_per_unit=0.00001)
    return ExecutionSimulator(_bars(), cm)


def test_market_intent_fills_at_next_bar_open_not_decision_bar():
    bars = _bars()
    sim = _sim()
    intent = TradingIntent(
        intent_id="i1",
        symbol="EURUSD",
        side="buy",
        order_type="market",
        decision_time=bars.index[0],
        requested_size=1000,
    )
    [outcome] = sim.simulate([intent])
    assert outcome.status == "filled"
    assert outcome.fill_time == bars.index[1]
    assert outcome.fill_price > bars.iloc[1]["open"]  # buy pays spread+slippage above raw


def test_market_intent_with_no_future_bar_is_unfilled():
    bars = _bars()
    sim = _sim()
    intent = TradingIntent(
        intent_id="i2",
        symbol="EURUSD",
        side="buy",
        order_type="market",
        decision_time=bars.index[-1],
        requested_size=1000,
    )
    [outcome] = sim.simulate([intent])
    assert outcome.status == "unfilled"


def test_unknown_decision_time_is_rejected_not_guessed():
    sim = _sim()
    bogus_time = pd.Timestamp("2024-01-01T00:30:00", tz="UTC")
    intent = TradingIntent(
        intent_id="i3",
        symbol="EURUSD",
        side="buy",
        order_type="market",
        decision_time=bogus_time,
        requested_size=1000,
    )
    [outcome] = sim.simulate([intent])
    assert outcome.status == "rejected"


def test_limit_buy_fills_at_limit_price_not_a_more_favorable_price():
    bars = _bars()
    sim = _sim()
    intent = TradingIntent(
        intent_id="i4",
        symbol="EURUSD",
        side="buy",
        order_type="limit",
        decision_time=bars.index[0],
        requested_size=1000,
        limit_price=1.0955,
    )
    # bar index 2 has low=1.0940 <= 1.0955 -> first touch
    [outcome] = sim.simulate([intent])
    assert outcome.status == "filled"
    assert outcome.fill_time == bars.index[2]
    spread_half = 0.0002 / 2
    assert outcome.fill_price == pytest.approx(1.0955 + spread_half)
    assert outcome.slippage_cost == 0.0  # no slippage modeled on limit fills


def test_limit_never_touched_is_unfilled():
    bars = _bars()
    sim = _sim()
    intent = TradingIntent(
        intent_id="i5",
        symbol="EURUSD",
        side="buy",
        order_type="limit",
        decision_time=bars.index[0],
        requested_size=1000,
        limit_price=0.5,
    )
    [outcome] = sim.simulate([intent])
    assert outcome.status == "unfilled"


def test_costs_are_always_adverse_to_the_trader():
    bars = _bars()
    sim = _sim()
    raw_open = bars.iloc[1]["open"]
    buy = TradingIntent(
        intent_id="i6",
        symbol="EURUSD",
        side="buy",
        order_type="market",
        decision_time=bars.index[0],
        requested_size=1000,
    )
    sell = TradingIntent(
        intent_id="i7",
        symbol="EURUSD",
        side="sell",
        order_type="market",
        decision_time=bars.index[0],
        requested_size=1000,
    )
    [buy_outcome] = sim.simulate([buy])
    [sell_outcome] = sim.simulate([sell])
    assert buy_outcome.fill_price > raw_open
    assert sell_outcome.fill_price < raw_open
    assert buy_outcome.spread_cost > 0
    assert buy_outcome.commission_cost > 0


def test_zero_cost_model_must_be_explicit_not_a_hidden_default():
    bars = _bars()
    cm = CostModel(spread=0.0, commission_per_unit=0.0)
    sim = _sim(cm)
    intent = TradingIntent(
        intent_id="i8",
        symbol="EURUSD",
        side="buy",
        order_type="market",
        decision_time=bars.index[0],
        requested_size=1000,
    )
    [outcome] = sim.simulate([intent])
    assert outcome.fill_price == pytest.approx(bars.iloc[1]["open"])


def test_negative_cost_parameters_rejected():
    with pytest.raises(ValueError):
        CostModel(spread=-0.0001, commission_per_unit=0.0)
