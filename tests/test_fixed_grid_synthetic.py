"""Synthetic scenario tests for the symmetric Fixed Grid baseline.

Bar geometry and cost values are hand-computable on purpose. `levels_per_side` here
is deliberately small (1-2) for test simplicity; it is not a research
parameter choice — synthetic unit tests
are free to use whatever grid size makes the arithmetic clean.
"""

import pandas as pd
import pytest

from gridadaptive.execution.costs import CostModel
from gridadaptive.execution.simulator import ExecutionSimulator
from gridadaptive.strategy.fixed_grid import FixedGridConfig, FixedGridEngine, default_anchor_price

ANCHOR = 1.1000
SPACING = 0.0010


def _bars(ohlc_rows):
    idx = pd.date_range("2024-01-01", periods=len(ohlc_rows), freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open": [r[0] for r in ohlc_rows],
            "high": [r[1] for r in ohlc_rows],
            "low": [r[2] for r in ohlc_rows],
            "close": [r[3] for r in ohlc_rows],
        },
        index=idx,
    )


def _zero_cost():
    return CostModel(spread=0.0, commission_per_unit=0.0)


def _config(**overrides):
    defaults = dict(
        symbol="EURUSD",
        grid_spacing=SPACING,
        levels_per_side=2,
        size_per_level=1000,
    )
    defaults.update(overrides)
    return FixedGridConfig(**defaults)


def _run(bars, config, cost_model=None, anchor=ANCHOR):
    sim = ExecutionSimulator(bars, cost_model or _zero_cost())
    engine = FixedGridEngine(config, sim, anchor_price=anchor)
    return engine.run()


# 1. Flat market -----------------------------------------------------------------


def test_flat_market_produces_no_fills():
    bars = _bars(
        [
            (1.1000, 1.1000, 1.1000, 1.1000),
            (1.1000, 1.1000, 1.1000, 1.1000),
            (1.1000, 1.1000, 1.1000, 1.1000),
        ]
    )
    result = _run(bars, _config())
    assert result.cycles == []
    assert result.open_positions == []


# 2/3. Single dip-and-recover / rally-and-revert (parametrized by side) ----------

_DIP_RECOVER_BARS = {
    "buy": _bars(
        [
            (1.1000, 1.1000, 1.1000, 1.1000),  # t0 anchor
            (1.1000, 1.1000, 1.0985, 1.0990),  # dip touches buy_1 (1.0990)
            (1.0990, 1.1005, 1.0985, 1.1000),  # rises to touch TP (1.1000)
        ]
    ),
    "sell": _bars(
        [
            (1.1000, 1.1000, 1.1000, 1.1000),  # t0 anchor
            (1.1000, 1.1015, 1.1000, 1.1010),  # rally touches sell_1 (1.1010)
            (1.1010, 1.1015, 1.0995, 1.1000),  # falls to touch TP (1.1000)
        ]
    ),
}


@pytest.mark.parametrize("side", ["buy", "sell"])
def test_single_dip_or_rally_produces_one_grid_cycle(side):
    bars = _DIP_RECOVER_BARS[side]
    config = _config(levels_per_side=1)
    result = _run(bars, config)

    assert len(result.cycles) == 1
    assert result.open_positions == []
    cycle = result.cycles[0]
    assert cycle.side == side
    assert cycle.level_id == f"{side}_1"
    if side == "buy":
        assert cycle.open_price == pytest.approx(1.0990)
        assert cycle.close_price == pytest.approx(1.1000)
    else:
        assert cycle.open_price == pytest.approx(1.1010)
        assert cycle.close_price == pytest.approx(1.1000)
    assert cycle.gross_pnl == pytest.approx(1.0)  # ΔP(0.0010) * size(1000)
    assert cycle.realized_pnl == pytest.approx(1.0)  # zero-cost model


# 4/5. Repeated oscillation -> level re-arms (parametrized by side) --------------

_OSCILLATION_BARS = {
    "buy": _bars(
        [
            (1.1000, 1.1000, 1.1000, 1.1000),
            (1.1000, 1.1000, 1.0985, 1.0990),
            (1.0990, 1.1005, 1.0985, 1.1000),
            (1.1000, 1.1000, 1.0985, 1.0990),
            (1.0990, 1.1005, 1.0985, 1.1000),
        ]
    ),
    "sell": _bars(
        [
            (1.1000, 1.1000, 1.1000, 1.1000),
            (1.1000, 1.1015, 1.1000, 1.1010),
            (1.1010, 1.1015, 1.0995, 1.1000),
            (1.1000, 1.1015, 1.1000, 1.1010),
            (1.1010, 1.1015, 1.0995, 1.1000),
        ]
    ),
}


@pytest.mark.parametrize("side", ["buy", "sell"])
def test_repeated_oscillation_rearms_the_level(side):
    bars = _OSCILLATION_BARS[side]
    config = _config(levels_per_side=1)
    result = _run(bars, config)

    assert len(result.cycles) == 2
    assert result.open_positions == []
    assert all(c.level_id == f"{side}_1" for c in result.cycles)
    assert sum(c.realized_pnl for c in result.cycles) == pytest.approx(2.0)
    # cycles are chronologically ordered and non-overlapping
    assert result.cycles[0].close_time < result.cycles[1].open_time


# 6/7. Monotonic trend, no reversion -> layers stack, no cycles ------------------

_TREND_BARS = {
    "buy": _bars(
        [
            (1.1000, 1.1000, 1.1000, 1.1000),  # anchor
            (1.1000, 1.1000, 1.0985, 1.0990),  # touches buy_1 (1.0990)
            (1.0990, 1.0990, 1.0975, 1.0980),  # touches buy_2 (1.0980); no TP1 touch
            (1.0980, 1.0980, 1.0970, 1.0975),  # stays low
            (1.0975, 1.0980, 1.0970, 1.0975),  # stays low
        ]
    ),
    "sell": _bars(
        [
            (1.1000, 1.1000, 1.1000, 1.1000),  # anchor
            (1.1000, 1.1015, 1.1000, 1.1010),  # touches sell_1 (1.1010)
            (1.1010, 1.1025, 1.1010, 1.1020),  # touches sell_2 (1.1020); no TP1 touch
            (1.1020, 1.1030, 1.1020, 1.1025),  # stays high
            (1.1025, 1.1030, 1.1020, 1.1025),  # stays high
        ]
    ),
}


@pytest.mark.parametrize("side", ["buy", "sell"])
def test_monotonic_trend_stacks_layers_without_closing(side):
    bars = _TREND_BARS[side]
    config = _config(levels_per_side=2)
    result = _run(bars, config)

    assert result.cycles == []
    assert len(result.open_positions) == 2
    assert {p.level_id for p in result.open_positions} == {f"{side}_1", f"{side}_2"}
    other_side = "sell" if side == "buy" else "buy"
    other_side_levels = [lvl for lvl in result.levels if lvl.side == other_side]
    assert len(other_side_levels) == 2  # levels exist (symmetric) but never fill


# 8. Price never reaches the grid -------------------------------------------------


def test_price_confined_between_innermost_levels_produces_no_fills():
    bars = _bars(
        [
            (1.1000, 1.1000, 1.1000, 1.1000),
            (1.1000, 1.1005, 1.0995, 1.1002),
            (1.1002, 1.1004, 1.0996, 1.0998),
            (1.0998, 1.1003, 1.0997, 1.1001),
        ]
    )
    result = _run(bars, _config(levels_per_side=1))
    assert result.cycles == []
    assert result.open_positions == []


# 9. No-lookahead regression: exit can never use its own entry bar's touch -------

_LOOKAHEAD_BARS = {
    "buy": _bars(
        [
            (1.1000, 1.1000, 1.1000, 1.1000),  # anchor
            (1.1000, 1.1005, 1.0985, 1.0995),  # entry touch AND would-be TP touch, same bar
            (1.0995, 1.0995, 1.0990, 1.0992),  # TP never reached again
            (1.0992, 1.0995, 1.0988, 1.0990),
        ]
    ),
    "sell": _bars(
        [
            (1.1000, 1.1000, 1.1000, 1.1000),  # anchor
            (1.1000, 1.1015, 1.0995, 1.1005),  # entry touch AND would-be TP touch, same bar
            (1.1005, 1.1008, 1.1002, 1.1004),  # TP never reached again
            (1.1004, 1.1008, 1.1001, 1.1003),
        ]
    ),
}


@pytest.mark.parametrize("side", ["buy", "sell"])
def test_exit_never_fills_on_its_own_entry_bar(side):
    """A buggy engine that scanned for the exit fill starting at t0 (instead of the
    entry's actual fill_time) would find bar[1]'s own high/low touching the take-
    profit price and report an impossible same-bar entry+exit. The correct engine
    must leave the position open instead."""
    bars = _LOOKAHEAD_BARS[side]
    config = _config(levels_per_side=1)
    result = _run(bars, config)

    assert result.cycles == []
    assert len(result.open_positions) == 1
    position = result.open_positions[0]
    assert position.level_id == f"{side}_1"
    assert position.open_time == bars.index[1]


# 10. Cost non-omission: costs must flow through the aggregated cycle PnL --------


@pytest.mark.parametrize("side", ["buy", "sell"])
def test_costs_reduce_realized_pnl_by_exactly_the_expected_amount(side):
    bars = _DIP_RECOVER_BARS[side]
    config = _config(levels_per_side=1)

    zero_cost_result = _run(bars, config, _zero_cost())
    cost_model = CostModel(spread=0.0002, commission_per_unit=0.00001)
    costed_result = _run(bars, config, cost_model)

    assert len(zero_cost_result.cycles) == 1
    assert len(costed_result.cycles) == 1

    zero_cost_pnl = zero_cost_result.cycles[0].realized_pnl
    costed_pnl = costed_result.cycles[0].realized_pnl

    size = config.size_per_level
    expected_spread_cost = cost_model.spread * size  # both legs, half each
    expected_commission_cost = cost_model.commission_per_unit * size * 2  # both legs
    expected_total_cost = expected_spread_cost + expected_commission_cost

    assert zero_cost_pnl - costed_pnl == pytest.approx(expected_total_cost)
    assert costed_result.cycles[0].all_in_cost == pytest.approx(expected_total_cost)
    assert costed_result.cycles[0].slippage_cost == pytest.approx(0.0)  # inert for limit fills


# 11. Config validation ------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"grid_spacing": 0.0},
        {"grid_spacing": -0.001},
        {"levels_per_side": 0},
        {"levels_per_side": -1},
        {"levels_per_side": 2.5},
        {"size_per_level": 0.0},
        {"size_per_level": -100},
        {"take_profit_distance": -0.001},
        {"max_cycles_per_level": 0},
    ],
)
def test_invalid_config_raises_before_any_simulation(overrides):
    with pytest.raises(ValueError):
        _config(**overrides)


def test_default_anchor_price_is_first_bar_close():
    bars = _bars([(1.1000, 1.1010, 1.0990, 1.1005), (1.1005, 1.1010, 1.1000, 1.1008)])
    assert default_anchor_price(bars) == pytest.approx(1.1005)


# 12. Both sides active simultaneously in one run ---------------------------------


def test_buy_and_sell_sides_operate_independently_in_the_same_run():
    bars = _bars(
        [
            (1.1000, 1.1000, 1.1000, 1.1000),  # anchor
            (1.1000, 1.1000, 1.0985, 1.0990),  # buy_1 touched
            (1.0990, 1.1005, 1.0985, 1.1000),  # buy_1 TP touched -> cycle closes
            (1.1000, 1.1015, 1.1000, 1.1010),  # sell_1 touched
            (1.1010, 1.1015, 1.0995, 1.1000),  # sell_1 TP touched -> cycle closes
        ]
    )
    result = _run(bars, _config(levels_per_side=1))

    assert len(result.cycles) == 2
    sides = {c.side for c in result.cycles}
    assert sides == {"buy", "sell"}
    assert result.open_positions == []
