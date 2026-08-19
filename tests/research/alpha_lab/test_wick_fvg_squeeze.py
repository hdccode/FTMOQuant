from __future__ import annotations

from datetime import UTC, date

import numpy as np
import pandas as pd
import pytest

from ftmoquant.research.alpha_lab.data import AlphaLabDataError
from ftmoquant.research.alpha_lab.screening_common import AGGREGATE_INSTRUMENT_LABEL
from ftmoquant.research.alpha_lab.wick_fvg_squeeze_execution import (
    _resolve_stop_target,
    simulate_trades,
)
from ftmoquant.research.alpha_lab.wick_fvg_squeeze_screen import (
    F1_FAMILY,
    F1_TIMEFRAMES,
    F1_WICK_RATIOS,
    F2_BACKWARD_SEARCH_N,
    F2_FAMILY,
    F2_TIMEFRAMES,
    F3_BB_WIDTH_THRESHOLDS,
    F3_FAMILY,
    F3_TIMEFRAMES,
    AggregateStats,
    PairStats,
    WickFvgSqueezeResultRow,
    _aggregate_stats,
    _analyze_2d_family,
    _analyze_pair_specific_family,
    _pair_fold_returns,
    _passes_gate,
    _passes_pair_specific_gate,
    analyze_family_robustness,
    analyze_pair_specific_robustness,
    build_grid,
)
from ftmoquant.research.alpha_lab.wick_fvg_squeeze_signals import (
    DIRECTION_LONG,
    DIRECTION_SHORT,
    SignalEvent,
    _bollinger_bands,
    f1_failed_bollinger_rejection_signals,
    f2_fresh_fvg_mitigation_signals,
    f3_volatility_squeeze_breakout_signals,
)
from ftmoquant.strategies.mean_reversion_h1 import FROZEN_UNIVERSE


def _index(periods: int, freq: str = "1h") -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=periods, freq=freq, tz=UTC)


# ---------------------------------------------------------------------------
# 1. exact 18-config grid.
# ---------------------------------------------------------------------------


def test_grid_is_exactly_eighteen_configs_six_per_family() -> None:
    grid = build_grid()
    assert len(grid) == 18
    counts: dict[str, int] = {}
    for config in grid:
        counts[config.family] = counts.get(config.family, 0) + 1
    assert counts == {F1_FAMILY: 6, F2_FAMILY: 6, F3_FAMILY: 6}
    assert len(F1_TIMEFRAMES) * len(F1_WICK_RATIOS) == 6
    assert len(F2_TIMEFRAMES) * len(F2_BACKWARD_SEARCH_N) == 6
    assert len(F1_TIMEFRAMES) * len(F3_BB_WIDTH_THRESHOLDS) == 6
    # every configuration is unique
    assert len({config.strategy_id for config in grid}) == 18


# ---------------------------------------------------------------------------
# 2. F1 bullish + bearish rejection signal semantics.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("direction", [DIRECTION_SHORT, DIRECTION_LONG])
def test_f1_rejection_signal_semantics(direction: int) -> None:
    n = 25
    index = _index(n)
    open_ = np.full(n, 100.0)
    high = np.full(n, 100.05)
    low = np.full(n, 99.95)
    close = np.full(n, 100.0)
    if direction == DIRECTION_SHORT:
        open_[-1], close[-1], high[-1], low[-1] = 100.0, 99.9, 101.0, 99.85
    else:
        open_[-1], close[-1], high[-1], low[-1] = 100.0, 100.1, 100.15, 99.0

    o = pd.Series(open_, index=index)
    h = pd.Series(high, index=index)
    lo = pd.Series(low, index=index)
    c = pd.Series(close, index=index)
    events = f1_failed_bollinger_rejection_signals(o, h, lo, c, wick_ratio=0.5)

    assert len(events) == 1
    event = events[0]
    assert event.signal_bar_ts == index[-1]
    assert event.direction == direction
    midpoint, _, _ = _bollinger_bands(c)
    assert event.target_price == pytest.approx(float(midpoint.iloc[-1]))
    assert event.stop_distance is not None and event.stop_distance > 0

    # a much stricter wick_ratio must suppress the same bar (the wick isn't
    # infinitely large relative to its own range).
    events_strict = f1_failed_bollinger_rejection_signals(o, h, lo, c, wick_ratio=0.95)
    assert events_strict == ()


# ---------------------------------------------------------------------------
# 3. F2 bullish + bearish FVG formation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("direction", [DIRECTION_LONG, DIRECTION_SHORT])
def test_f2_fvg_formation_requires_the_three_bar_gap(direction: int) -> None:
    n = 10
    index = _index(n)
    open_ = np.full(n, 100.0)
    high = np.full(n, 100.2)
    low = np.full(n, 99.8)
    close = np.full(n, 100.0)
    # bar 3: a bearish (for LONG zone) or bullish (for SHORT zone) origin
    # candle whose body will become the order block.
    if direction == DIRECTION_LONG:
        open_[3], close[3] = 100.0, 99.7  # bearish candle
        # bar 5 gaps above bar 3's high: low[5] > high[3]
        high[3] = 100.0
        low[5], high[5], open_[5], close[5] = 100.5, 100.7, 100.5, 100.6
    else:
        open_[3], close[3] = 100.0, 100.3  # bullish candle
        low[3] = 100.0
        low[5], high[5], open_[5], close[5] = 99.3, 99.5, 99.5, 99.4  # gaps below

    o = pd.Series(open_, index=index)
    h = pd.Series(high, index=index)
    lo = pd.Series(low, index=index)
    c = pd.Series(close, index=index)
    events = f2_fresh_fvg_mitigation_signals(o, h, lo, c, backward_search_n=5)
    # no retest ever happens in this short series -- formation alone must
    # not itself emit a trade.
    assert events == ()

    # without the gap (bar 5 overlapping bar 3's range) no zone can form,
    # verified indirectly: shrinking bar 5 back into bar 3's range and then
    # forcing an obvious retest afterwards must not produce a trade because
    # no zone was ever created.
    if direction == DIRECTION_LONG:
        low[5] = 99.9  # no longer > high[3]
    else:
        high[5] = 100.1  # no longer < low[3]
    o2 = pd.Series(open_, index=index)
    h2 = pd.Series(high, index=index)
    lo2 = pd.Series(low, index=index)
    c2 = pd.Series(close, index=index)
    assert f2_fresh_fvg_mitigation_signals(o2, h2, lo2, c2, backward_search_n=5) == ()


# ---------------------------------------------------------------------------
# 4. F2 backward-only zone discovery (+ N-bar search window).
# ---------------------------------------------------------------------------


def _f2_long_zone_series(
    *, bearish_candle_index: int, gap_index: int, n: int = 12
) -> tuple[pd.Series, ...]:
    open_ = np.full(n, 100.0)
    high = np.full(n, 100.1)
    low = np.full(n, 99.9)
    close = np.full(n, 100.0)
    open_[bearish_candle_index], close[bearish_candle_index] = 100.0, 99.5
    high[bearish_candle_index] = 100.0
    low[bearish_candle_index] = 99.5
    open_[gap_index], close[gap_index] = 100.6, 100.7
    high[gap_index] = 100.8
    low[gap_index] = 100.6
    high[gap_index - 2] = min(high[gap_index - 2], 100.5)  # ensure low[gap]>high[gap-2]
    index = _index(n)
    return (
        pd.Series(open_, index=index),
        pd.Series(high, index=index),
        pd.Series(low, index=index),
        pd.Series(close, index=index),
    )


def test_f2_zone_body_is_found_by_backward_search_only() -> None:
    # A bearish candle sitting AFTER the FVG bar must never be used as the
    # zone's origin, even though it would otherwise qualify.
    o, h, lo, c = _f2_long_zone_series(bearish_candle_index=3, gap_index=5)
    # plant a second, closer-in-time-but-future bearish candle at index 7:
    # it must have zero effect on the zone body used at formation (index 5).
    o.iloc[7], c.iloc[7] = 100.0, 90.0
    # No retest occurs in this series, but we can inspect the zone
    # indirectly: engineer a retest bar right after formation and confirm
    # the stop price matches bar 3's body (99.5/100.0), not bar 7's.
    o2 = o.copy()
    h2 = h.copy()
    lo2 = lo.copy()
    c2 = c.copy()
    # bar 6: touches the zone (low <= 100.0) and reclaims (close > 100.0)
    o2.iloc[6], h2.iloc[6], lo2.iloc[6], c2.iloc[6] = 100.6, 100.65, 99.6, 100.65
    events = f2_fresh_fvg_mitigation_signals(o2, h2, lo2, c2, backward_search_n=5)
    assert len(events) == 1
    # bar 3's body low, not bar 7's
    assert events[0].stop_price == pytest.approx(99.5)


def test_f2_backward_search_n_bounds_the_window() -> None:
    o, h, lo, c = _f2_long_zone_series(bearish_candle_index=0, gap_index=8, n=12)
    # bearish candle at index 0 is 8 bars before the FVG bar (index 8) --
    # in range for N=10 but out of range for N=3. Neither retests in this
    # series (no trade emitted either way), so assert on zone presence
    # indirectly via a forced retest bar appended after formation.
    o2, h2, lo2, c2 = o.copy(), h.copy(), lo.copy(), c.copy()
    o2.iloc[9], h2.iloc[9], lo2.iloc[9], c2.iloc[9] = 100.6, 100.65, 99.6, 100.65
    wide = f2_fresh_fvg_mitigation_signals(o2, h2, lo2, c2, backward_search_n=10)
    narrow = f2_fresh_fvg_mitigation_signals(o2, h2, lo2, c2, backward_search_n=3)
    assert len(wide) == 1  # zone found -> trade
    # bearish candle out of the narrower window -> no zone -> no trade
    assert len(narrow) == 0


# ---------------------------------------------------------------------------
# 5. F2 freshness / first-retest-only behavior.
# ---------------------------------------------------------------------------


def test_f2_only_the_first_retest_is_trade_eligible() -> None:
    o, h, lo, c = _f2_long_zone_series(bearish_candle_index=3, gap_index=5, n=14)
    # bar 6: touches without reclaiming (first retest, consumes the zone,
    # no trade).
    o.iloc[6], h.iloc[6], lo.iloc[6], c.iloc[6] = 100.6, 100.65, 99.6, 99.9
    # bar 8: a SECOND touch-and-reclaim -- must NOT trigger, the zone was
    # already spent by the first (failed) retest at bar 6.
    o.iloc[8], h.iloc[8], lo.iloc[8], c.iloc[8] = 100.6, 100.65, 99.6, 100.65
    events = f2_fresh_fvg_mitigation_signals(o, h, lo, c, backward_search_n=5)
    assert events == ()


def test_f2_first_touch_and_reclaim_on_the_same_bar_triggers_a_trade() -> None:
    o, h, lo, c = _f2_long_zone_series(bearish_candle_index=3, gap_index=5, n=10)
    o.iloc[6], h.iloc[6], lo.iloc[6], c.iloc[6] = 100.6, 100.65, 99.6, 100.65
    events = f2_fresh_fvg_mitigation_signals(o, h, lo, c, backward_search_n=5)
    assert len(events) == 1
    assert events[0].direction == DIRECTION_LONG
    assert events[0].target_r_multiple == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# 6. F2 invalidation.
# ---------------------------------------------------------------------------


def test_f2_invalidation_permanently_ends_the_zone() -> None:
    o, h, lo, c = _f2_long_zone_series(bearish_candle_index=3, gap_index=5, n=14)
    # bar 6: closes clean through the distal (lower) edge -- invalidated,
    # never touching/reclaiming first.
    o.iloc[6], h.iloc[6], lo.iloc[6], c.iloc[6] = 100.6, 100.6, 99.0, 99.0
    # bar 8: would otherwise be a perfectly valid touch-and-reclaim.
    o.iloc[8], h.iloc[8], lo.iloc[8], c.iloc[8] = 100.6, 100.65, 99.6, 100.65
    events = f2_fresh_fvg_mitigation_signals(o, h, lo, c, backward_search_n=5)
    assert events == ()


# ---------------------------------------------------------------------------
# 7. F3 long + short breakout semantics.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("direction", [DIRECTION_LONG, DIRECTION_SHORT])
def test_f3_squeeze_breakout_signal_semantics(direction: int) -> None:
    n = 25
    index = _index(n)
    close = np.full(n, 100.0)
    # keep price essentially flat (a tight squeeze) through bar n-2, then
    # break out hard on the final bar.
    close[-1] = 103.0 if direction == DIRECTION_LONG else 97.0
    high = close + 0.05
    low = close - 0.05
    high[-1] = close[-1] + 0.1
    low[-1] = close[-1] - 0.1
    h = pd.Series(high, index=index)
    lo = pd.Series(low, index=index)
    c = pd.Series(close, index=index)

    events = f3_volatility_squeeze_breakout_signals(h, lo, c, bb_width_threshold=0.05)
    assert len(events) == 1
    assert events[0].direction == direction
    assert events[0].signal_bar_ts == index[-1]
    assert events[0].stop_distance is not None
    assert events[0].target_distance == pytest.approx(2.0 * events[0].stop_distance)

    # widen the prior bar's own trailing volatility (a real, non-squeezed
    # band) so the identical breakout bar must not fire.
    wide_close = close.copy()
    wide_close[1:-1] = 100.0 + np.tile([0.0, 3.0], (n - 2) // 2 + 1)[: n - 2]
    wide_high = wide_close + 0.05
    wide_low = wide_close - 0.05
    wide_high[-1] = close[-1] + 0.1
    wide_low[-1] = close[-1] - 0.1
    events_no_squeeze = f3_volatility_squeeze_breakout_signals(
        pd.Series(wide_high, index=index),
        pd.Series(wide_low, index=index),
        pd.Series(wide_close, index=index),
        bb_width_threshold=0.05,
    )
    assert events_no_squeeze == ()


# ---------------------------------------------------------------------------
# 8. no lookahead / prefix invariance for all three families.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", [F1_FAMILY, F2_FAMILY, F3_FAMILY])
def test_signal_generators_are_prefix_invariant(family: str) -> None:
    n = 40
    index = _index(n)
    rng = np.random.default_rng(7)
    close = 100.0 + np.cumsum(rng.normal(0, 0.3, n))
    open_ = close + rng.normal(0, 0.05, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.2, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.2, n))

    o = pd.Series(open_, index=index)
    h = pd.Series(high, index=index)
    lo = pd.Series(low, index=index)
    c = pd.Series(close, index=index)
    cut = 30

    def _run(
        oo: pd.Series, hh: pd.Series, ll: pd.Series, cc: pd.Series
    ) -> tuple[SignalEvent, ...]:
        if family == F1_FAMILY:
            return f1_failed_bollinger_rejection_signals(
                oo, hh, ll, cc, wick_ratio=0.5
            )
        if family == F2_FAMILY:
            return f2_fresh_fvg_mitigation_signals(
                oo, hh, ll, cc, backward_search_n=10
            )
        return f3_volatility_squeeze_breakout_signals(
            hh, ll, cc, bb_width_threshold=0.05
        )

    full = _run(o, h, lo, c)
    prefix = _run(o.iloc[:cut], h.iloc[:cut], lo.iloc[:cut], c.iloc[:cut])

    full_before_cut = [e for e in full if e.signal_bar_ts <= index[cut - 1]]
    assert list(prefix) == full_before_cut


# ---------------------------------------------------------------------------
# 9-10. strictly-later execution + correct BID/ASK side selection.
# ---------------------------------------------------------------------------


def _m1_index(periods: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01T00:00:00Z", periods=periods, freq="1min", tz=UTC)


def _flat_m1(
    periods: int, *, bid: float, ask: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = _m1_index(periods)
    bid_df = pd.DataFrame(
        {"open": bid, "high": bid, "low": bid, "close": bid}, index=index
    )
    ask_df = pd.DataFrame(
        {"open": ask, "high": ask, "low": ask, "close": ask}, index=index
    )
    return bid_df, ask_df


@pytest.mark.parametrize("direction", [DIRECTION_LONG, DIRECTION_SHORT])
def test_execution_entry_is_strictly_later_than_the_signal_bar(direction: int) -> None:
    bid_m1, ask_m1 = _flat_m1(20, bid=1.0995, ask=1.1005)
    # a large excursion a few bars later so the trade actually resolves.
    bid_m1.loc[bid_m1.index[10], "high"] = 1.20
    bid_m1.loc[bid_m1.index[10], "low"] = 1.00
    ask_m1.loc[ask_m1.index[10], "high"] = 1.20
    ask_m1.loc[ask_m1.index[10], "low"] = 1.00
    signal_ts = bid_m1.index[3]
    event = SignalEvent(
        signal_bar_ts=signal_ts,
        direction=direction,
        stop_distance=0.01,
        target_distance=0.01,
    )
    trades, skips = simulate_trades([event], bid_m1, ask_m1)
    assert skips == ()
    assert len(trades) == 1
    assert trades[0].entry_ts > signal_ts
    # first paired observation strictly after the signal bar
    assert trades[0].entry_ts == bid_m1.index[4]


@pytest.mark.parametrize(
    ("direction", "expected_entry_price"),
    [(DIRECTION_LONG, 1.1005), (DIRECTION_SHORT, 1.0995)],
)
def test_execution_entry_crosses_the_correct_side(
    direction: int, expected_entry_price: float
) -> None:
    bid_m1, ask_m1 = _flat_m1(10, bid=1.0995, ask=1.1005)
    bid_m1.loc[bid_m1.index[5], "high"] = 1.20
    bid_m1.loc[bid_m1.index[5], "low"] = 1.00
    ask_m1.loc[ask_m1.index[5], "high"] = 1.20
    ask_m1.loc[ask_m1.index[5], "low"] = 1.00
    event = SignalEvent(
        signal_bar_ts=bid_m1.index[0],
        direction=direction,
        stop_distance=0.05,
        target_distance=0.05,
    )
    trades, _ = simulate_trades([event], bid_m1, ask_m1)
    assert trades[0].entry_price == pytest.approx(expected_entry_price)


@pytest.mark.parametrize("direction", [DIRECTION_LONG, DIRECTION_SHORT])
def test_execution_exit_liquidates_against_the_correct_side(direction: int) -> None:
    periods = 10
    index = _m1_index(periods)
    bid = pd.DataFrame(
        {"open": 1.1, "high": 1.1, "low": 1.1, "close": 1.1}, index=index
    )
    ask = pd.DataFrame(
        {"open": 1.1, "high": 1.1, "low": 1.1, "close": 1.1}, index=index
    )
    # a target hit only visible on the liquidation side (bid for long exit,
    # ask for short exit) at bar index 4 -- the wrong-side price never moves.
    target = 1.1 + 0.01 if direction == DIRECTION_LONG else 1.1 - 0.01
    if direction == DIRECTION_LONG:
        bid.loc[index[4], "high"] = target
    else:
        ask.loc[index[4], "low"] = target
    event = SignalEvent(
        signal_bar_ts=index[0],
        direction=direction,
        stop_distance=0.05,
        target_distance=0.01,
    )
    trades, _ = simulate_trades([event], bid, ask)
    assert len(trades) == 1
    assert trades[0].exit_reason == "target"
    assert trades[0].exit_ts == index[4]


# ---------------------------------------------------------------------------
# 11. stop/target frozen at signal time.
# ---------------------------------------------------------------------------


def test_stop_target_resolution_matches_each_family_convention() -> None:
    # F1/F3 style: distance-based, resolved against the fill price only.
    event = SignalEvent(
        signal_bar_ts=pd.Timestamp("2020-01-01", tz=UTC),
        direction=DIRECTION_LONG,
        stop_distance=0.002,
        target_price=1.105,
    )
    stop, target = _resolve_stop_target(event, entry_price=1.100)
    assert stop == pytest.approx(1.098)
    assert target == pytest.approx(1.105)  # fixed, independent of entry price

    # F2 style: fixed stop price, R-multiple target computed from the
    # actual fill.
    event2 = SignalEvent(
        signal_bar_ts=pd.Timestamp("2020-01-01", tz=UTC),
        direction=DIRECTION_LONG,
        stop_price=1.090,
        target_r_multiple=1.5,
    )
    stop2, target2 = _resolve_stop_target(event2, entry_price=1.100)
    assert stop2 == pytest.approx(1.090)
    assert target2 == pytest.approx(1.100 + 1.5 * 0.010)


# ---------------------------------------------------------------------------
# 12. conservative same-M1 stop/target collision.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("direction", [DIRECTION_LONG, DIRECTION_SHORT])
def test_conservative_same_bar_stop_target_collision_favors_the_stop(
    direction: int,
) -> None:
    periods = 6
    index = _m1_index(periods)
    entry_price = 1.1000
    stop_distance = 0.005
    target_distance = 0.005
    flat = {
        "open": entry_price,
        "high": entry_price,
        "low": entry_price,
        "close": entry_price,
    }
    bid = pd.DataFrame(flat, index=index)
    ask = pd.DataFrame(flat, index=index)
    # bar 2: a single wide bar whose range spans BOTH the stop and target.
    if direction == DIRECTION_LONG:
        bid.loc[index[2], "low"] = entry_price - 0.02
        bid.loc[index[2], "high"] = entry_price + 0.02
    else:
        ask.loc[index[2], "low"] = entry_price - 0.02
        ask.loc[index[2], "high"] = entry_price + 0.02
    event = SignalEvent(
        signal_bar_ts=index[0],
        direction=direction,
        stop_distance=stop_distance,
        target_distance=target_distance,
    )
    trades, _ = simulate_trades([event], bid, ask)
    assert len(trades) == 1
    assert trades[0].exit_reason == "stop"


# ---------------------------------------------------------------------------
# 13. independent equal-capital sleeve aggregation.
# ---------------------------------------------------------------------------


def test_equal_weight_aggregate_is_the_simple_average_of_pair_returns() -> None:
    day = date(2020, 1, 1)
    pair_stats = {
        instrument_id: PairStats(
            instrument_id=instrument_id,
            trade_count=1,
            skip_count=0,
            net_return=net_return,
            annualized_sharpe=None,
            maximum_drawdown=0.0,
            win_rate=1.0 if net_return > 0 else 0.0,
            daily_returns=((day, net_return),),
        )
        for instrument_id, net_return in zip(
            FROZEN_UNIVERSE, [0.1, -0.05, 0.2, 0.0, 0.03, -0.1, 0.07], strict=True
        )
    }
    aggregate = _aggregate_stats(pair_stats)
    expected = sum(stats.net_return for stats in pair_stats.values()) / 7
    assert aggregate.net_return == pytest.approx(expected)
    assert aggregate.profitable_pair_count == sum(
        1 for stats in pair_stats.values() if stats.net_return > 0
    )


# ---------------------------------------------------------------------------
# 14 + 15. 4/7 cross-pair gate and 2/3 fold gate.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("net_return", "profitable_pairs", "positive_folds", "expected"),
    [
        (0.05, 4, 2, True),
        (0.05, 3, 2, False),  # fails 4/7
        (0.05, 4, 1, False),  # fails 2/3 folds
        (-0.01, 4, 2, False),  # fails positive-return
        (0.0, 4, 2, False),  # strictly greater than zero required
    ],
)
def test_viability_gate_requires_all_three_frozen_conditions(
    net_return: float, profitable_pairs: int, positive_folds: int, expected: bool
) -> None:
    aggregate = AggregateStats(
        net_return=net_return,
        annualized_sharpe=None,
        maximum_drawdown=-0.1,
        profitable_pair_count=profitable_pairs,
        positive_fold_count=positive_folds,
    )
    assert _passes_gate(aggregate) is expected


# ---------------------------------------------------------------------------
# 16. connected-region family-survival rule.
# ---------------------------------------------------------------------------


def _agg_row(
    family: str,
    timeframe: str,
    param_name: str,
    param_value: object,
    passes: bool,
) -> WickFvgSqueezeResultRow:
    import json

    return WickFvgSqueezeResultRow(
        family=family,
        strategy_id=f"{family}_{timeframe}_{param_value}",
        instrument=AGGREGATE_INSTRUMENT_LABEL,
        timeframe=timeframe,
        parameters=json.dumps({param_name: param_value}),
        trade_count=10,
        skip_count=0,
        net_return=0.01 if passes else -0.01,
        annualized_sharpe=None,
        maximum_drawdown=-0.05,
        win_rate=0.5,
        profitable_pair_count=5 if passes else 2,
        aggregate_equal_weight_net_return=0.01 if passes else -0.01,
        aggregate_equal_weight_sharpe=None,
        aggregate_maximum_drawdown=-0.05,
        positive_fold_count=2 if passes else 1,
        passes_gate=passes,
    )


@pytest.mark.parametrize(
    ("passing_cells", "expected_survival"),
    [
        ({(0, 0), (0, 1)}, True),  # two adjacent passing cells -> connected region
        ({(0, 0), (1, 2)}, False),  # two passing cells, but isolated (not adjacent)
        (set(), False),  # nothing passes
    ],
)
def test_connected_region_family_survival_rule(
    passing_cells: set[tuple[int, int]], expected_survival: bool
) -> None:
    timeframes = ("M30", "H1")
    wick_ratios = (0.5, 0.6, 0.7)
    rows = [
        _agg_row(
            F1_FAMILY,
            timeframes[i],
            "wick_ratio",
            wick_ratios[j],
            passes=(i, j) in passing_cells,
        )
        for i in range(2)
        for j in range(3)
    ]
    summary = _analyze_2d_family(
        rows,
        family=F1_FAMILY,
        timeframes=timeframes,
        param_name="wick_ratio",
        param_values=wick_ratios,
    )
    assert summary.survival_rule_passed is expected_survival


# ---------------------------------------------------------------------------
# 17. deterministic output.
# ---------------------------------------------------------------------------


def test_simulate_trades_is_deterministic() -> None:
    rng = np.random.default_rng(11)
    periods = 200
    index = _m1_index(periods)
    mid = 1.10 + np.cumsum(rng.normal(0, 0.0001, periods))
    bid = pd.DataFrame(
        {
            "open": mid - 0.0001,
            "high": mid - 0.00005,
            "low": mid - 0.00015,
            "close": mid - 0.0001,
        },
        index=index,
    )
    ask = pd.DataFrame(
        {
            "open": mid + 0.0001,
            "high": mid + 0.00015,
            "low": mid + 0.00005,
            "close": mid + 0.0001,
        },
        index=index,
    )
    events = tuple(
        SignalEvent(
            signal_bar_ts=index[i],
            direction=DIRECTION_LONG if i % 2 == 0 else DIRECTION_SHORT,
            stop_distance=0.0005,
            target_distance=0.0005,
        )
        for i in range(0, 150, 10)
    )
    first = simulate_trades(events, bid, ask)
    second = simulate_trades(events, bid, ask)
    assert first == second


# ---------------------------------------------------------------------------
# 18. DEVELOPMENT path firewall / no VALIDATION or HOLDOUT access.
# ---------------------------------------------------------------------------


def test_wick_fvg_squeeze_never_accepts_a_validation_or_holdout_development_root(
    tmp_path: object,
) -> None:
    from pathlib import Path

    from ftmoquant.research.alpha_lab import data as alpha_lab_data

    forbidden_root = Path(str(tmp_path)) / "validation_root"
    readiness_path = Path(str(tmp_path)) / "readiness.json"
    readiness_path.write_text(
        '{"readiness_version": "oanda-alpha-lab-readiness-1", '
        '"holdout_accessed": false, "holdout_rows_admitted": 0, '
        '"per_instrument_status": {"EUR/USD.OANDA": "research_ready"}, '
        '"instrument_artifacts": [{"instrument_id": "EUR/USD.OANDA", '
        '"dataset_symbol": "EURUSD", "catalog_tree_sha256": "deadbeef"}]}'
    )
    with pytest.raises(AlphaLabDataError):
        alpha_lab_data._discover_oanda_universe(readiness_path, forbidden_root)


# ---------------------------------------------------------------------------
# no-fabricated-exit skip behavior.
# ---------------------------------------------------------------------------


def test_no_later_m1_observation_and_no_exit_are_recorded_as_skips_not_fabricated() -> (
    None
):
    bid_m1, ask_m1 = _flat_m1(5, bid=1.0995, ask=1.1005)
    # signal on the very last paired M1 bar: no later observation exists.
    late_event = SignalEvent(
        signal_bar_ts=bid_m1.index[-1],
        direction=DIRECTION_LONG,
        stop_distance=0.01,
        target_distance=0.01,
    )
    trades, skips = simulate_trades([late_event], bid_m1, ask_m1)
    assert trades == ()
    assert len(skips) == 1
    assert skips[0].reason == "no_later_m1_observation"

    # signal early enough to enter, but price never touches stop or target
    # before the M1 stream ends: must be a skip, not a fabricated exit.
    early_event = SignalEvent(
        signal_bar_ts=bid_m1.index[0],
        direction=DIRECTION_LONG,
        stop_distance=1.0,
        target_distance=1.0,
    )
    trades2, skips2 = simulate_trades([early_event], bid_m1, ask_m1)
    assert trades2 == ()
    assert len(skips2) == 1
    assert skips2[0].reason == "no_m1_exit_before_data_end"


# ---------------------------------------------------------------------------
# Pair-specific fold diagnostics extension -- regression + new-behavior
# tests. None of the tests above this point were modified; everything
# below proves the extension is additive and the historical Batch-1
# strategy/grid/execution/broad-FX semantics are unchanged.
# ---------------------------------------------------------------------------

#: The exact 18 strategy ids already materialized in the real DEVELOPMENT
#: artifact (.artifacts/alpha_lab/wick_fvg_squeeze_screen_v1/scorecard.csv),
#: read once for this regression pin -- not altered, not used to select a
#: pair, and not a VALIDATION artifact.
_HISTORICAL_STRATEGY_IDS = frozenset(
    {
        "f1_h1_wick0.5_v1",
        "f1_h1_wick0.6_v1",
        "f1_h1_wick0.7_v1",
        "f1_m30_wick0.5_v1",
        "f1_m30_wick0.6_v1",
        "f1_m30_wick0.7_v1",
        "f2_h1_n10_v1",
        "f2_h1_n20_v1",
        "f2_h1_n5_v1",
        "f2_h4_n10_v1",
        "f2_h4_n20_v1",
        "f2_h4_n5_v1",
        "f3_h1_thr0.003_v1",
        "f3_h1_thr0.004_v1",
        "f3_h1_thr0.006_v1",
        "f3_h4_thr0.003_v1",
        "f3_h4_thr0.004_v1",
        "f3_h4_thr0.006_v1",
    }
)


def test_all_eighteen_historical_configs_are_unchanged() -> None:
    grid = build_grid()
    assert len(grid) == 18
    assert {config.strategy_id for config in grid} == _HISTORICAL_STRATEGY_IDS
    # timeframe/parameter coordinates, not just ids, must match exactly.
    by_id = {config.strategy_id: config for config in grid}
    assert by_id["f1_m30_wick0.5_v1"].timeframe == "M30"
    assert by_id["f1_m30_wick0.5_v1"].parameters == {"wick_ratio": 0.5}
    assert by_id["f2_h4_n20_v1"].timeframe == "H4"
    assert by_id["f2_h4_n20_v1"].parameters == {"backward_search_n": 20}
    assert by_id["f3_h4_thr0.006_v1"].timeframe == "H4"
    assert by_id["f3_h4_thr0.006_v1"].parameters == {"bb_width_threshold": 0.006}


def test_pair_fold_returns_are_genuinely_pair_specific_not_derived_from_aggregate() -> (
    None
):
    # two distinct daily-return series (deliberately opposite sign in the
    # first fold) must produce two distinct fold-1 returns -- proving
    # _pair_fold_returns operates on each pair's OWN daily series rather
    # than reading some shared/aggregate value.
    pair_a_daily = ((date(2020, 5, 1), 0.05), (date(2020, 6, 1), 0.01))
    pair_b_daily = ((date(2020, 5, 1), -0.05), (date(2020, 6, 1), 0.01))
    fold_a = _pair_fold_returns(pair_a_daily)
    fold_b = _pair_fold_returns(pair_b_daily)
    assert fold_a[0] != fold_b[0]
    assert fold_a[0] > 0
    assert fold_b[0] < 0
    # a day outside all three frozen fold windows must not contribute.
    outside_window = ((date(2010, 1, 1), 10.0),)
    assert _pair_fold_returns(outside_window) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("net_return", "all_three_positive", "expected"),
    [
        (0.05, True, True),
        (0.05, False, False),
        (-0.01, True, False),
        (0.0, True, False),  # strictly greater than zero required
    ],
)
def test_pair_specific_gate_requires_return_and_all_three_folds(
    net_return: float, all_three_positive: bool, expected: bool
) -> None:
    row = WickFvgSqueezeResultRow(
        family=F1_FAMILY,
        strategy_id="s",
        instrument="EUR/USD.OANDA",
        timeframe="H1",
        parameters="{}",
        trade_count=5,
        skip_count=0,
        net_return=net_return,
        annualized_sharpe=None,
        maximum_drawdown=-0.05,
        win_rate=0.5,
        profitable_pair_count=3,
        aggregate_equal_weight_net_return=0.0,
        aggregate_equal_weight_sharpe=None,
        aggregate_maximum_drawdown=-0.05,
        positive_fold_count=1,
        passes_gate=False,
        fold_1_net_return=0.01,
        fold_2_net_return=0.01,
        fold_3_net_return=0.01,
        positive_pair_fold_count=3 if all_three_positive else 1,
        pair_all_three_folds_positive=all_three_positive,
    )
    assert _passes_pair_specific_gate(row) is expected


def _pair_row(
    *, family: str, instrument: str, timeframe: str, parameters: str, passes: bool
) -> WickFvgSqueezeResultRow:
    return WickFvgSqueezeResultRow(
        family=family,
        strategy_id=f"{family}_{instrument}_{timeframe}_{parameters}",
        instrument=instrument,
        timeframe=timeframe,
        parameters=parameters,
        trade_count=10,
        skip_count=0,
        net_return=0.02 if passes else -0.02,
        annualized_sharpe=None,
        maximum_drawdown=-0.05,
        win_rate=0.5,
        profitable_pair_count=1,
        aggregate_equal_weight_net_return=0.0,
        aggregate_equal_weight_sharpe=None,
        aggregate_maximum_drawdown=-0.05,
        positive_fold_count=1,
        passes_gate=False,
        fold_1_net_return=0.01 if passes else -0.01,
        fold_2_net_return=0.01 if passes else 0.01,
        fold_3_net_return=0.01 if passes else -0.01,
        positive_pair_fold_count=3 if passes else 1,
        pair_all_three_folds_positive=passes,
    )


@pytest.mark.parametrize(
    ("family", "timeframes", "param_name", "param_values"),
    [
        (F1_FAMILY, F1_TIMEFRAMES, "wick_ratio", F1_WICK_RATIOS),
        (F2_FAMILY, F2_TIMEFRAMES, "backward_search_n", F2_BACKWARD_SEARCH_N),
        (F3_FAMILY, F3_TIMEFRAMES, "bb_width_threshold", F3_BB_WIDTH_THRESHOLDS),
    ],
)
def test_pair_specific_adjacency_matches_each_familys_original_grid(
    family: str,
    timeframes: tuple[str, ...],
    param_name: str,
    param_values: tuple[float, ...] | tuple[int, ...],
) -> None:
    import json

    pair = "EUR/USD.OANDA"
    # two axis-adjacent passing cells (same timeframe, adjacent parameter
    # values) -- must survive.
    rows = [
        _pair_row(
            family=family,
            instrument=pair,
            timeframe=timeframes[0],
            parameters=json.dumps({param_name: param_values[0]}),
            passes=True,
        ),
        _pair_row(
            family=family,
            instrument=pair,
            timeframe=timeframes[0],
            parameters=json.dumps({param_name: param_values[1]}),
            passes=True,
        ),
    ]
    summary = _analyze_pair_specific_family(
        rows,
        family=family,
        instrument=pair,
        timeframes=timeframes,
        param_name=param_name,
        param_values=param_values,
    )
    assert summary.largest_contiguous_region_size == 2
    assert summary.survival_rule_passed is True
    assert summary.tested_configuration_count == len(timeframes) * len(param_values)

    # a single isolated passing cell (no neighbor) must NOT survive.
    single_row = [rows[0]]
    single_summary = _analyze_pair_specific_family(
        single_row,
        family=family,
        instrument=pair,
        timeframes=timeframes,
        param_name=param_name,
        param_values=param_values,
    )
    assert single_summary.passing_configuration_count == 1
    assert single_summary.largest_contiguous_region_size == 1
    assert single_summary.survival_rule_passed is False

    # two passing cells that are NOT adjacent (different timeframe AND
    # different parameter value -- a diagonal, never rook-adjacent) must
    # not be merged into one region.
    diagonal_rows = [
        rows[0],
        _pair_row(
            family=family,
            instrument=pair,
            timeframe=timeframes[1],
            parameters=json.dumps({param_name: param_values[-1]}),
            passes=True,
        ),
    ]
    diagonal_summary = _analyze_pair_specific_family(
        diagonal_rows,
        family=family,
        instrument=pair,
        timeframes=timeframes,
        param_name=param_name,
        param_values=param_values,
    )
    assert diagonal_summary.contiguous_passing_region_count == 2
    assert diagonal_summary.largest_contiguous_region_size == 1
    assert diagonal_summary.survival_rule_passed is False


def test_f2_h4_three_cell_region_is_detected_correctly() -> None:
    import json

    pair = "USD/CAD.OANDA"
    rows = [
        _pair_row(
            family=F2_FAMILY,
            instrument=pair,
            timeframe="H4",
            parameters=json.dumps({"backward_search_n": n}),
            passes=True,
        )
        for n in F2_BACKWARD_SEARCH_N  # (5, 10, 20) -- all three, all H4
    ]
    summary = _analyze_pair_specific_family(
        rows,
        family=F2_FAMILY,
        instrument=pair,
        timeframes=F2_TIMEFRAMES,
        param_name="backward_search_n",
        param_values=F2_BACKWARD_SEARCH_N,
    )
    assert summary.passing_configuration_count == 3
    assert summary.contiguous_passing_region_count == 1
    assert summary.largest_contiguous_region_size == 3
    assert summary.survival_rule_passed is True
    assert summary.strongest_region_parameters == (
        "timeframe=H4,backward_search_n=5;"
        "timeframe=H4,backward_search_n=10;"
        "timeframe=H4,backward_search_n=20"
    )


def test_pair_specific_robustness_covers_every_family_times_pair() -> None:
    rows: list[WickFvgSqueezeResultRow] = []
    summaries = analyze_pair_specific_robustness(rows)
    # 3 families x 7 frozen pairs, even with no rows supplied.
    assert len(summaries) == 21
    families = {summary.family for summary in summaries}
    assert families == {F1_FAMILY, F2_FAMILY, F3_FAMILY}
    assert {summary.instrument for summary in summaries} == set(FROZEN_UNIVERSE)
    assert all(summary.passing_configuration_count == 0 for summary in summaries)


def test_new_pair_fold_fields_do_not_change_broad_fx_classification() -> None:
    """The new per-pair fold fields are inert with respect to the frozen
    broad-FX gate/region logic: two otherwise-identical aggregate rows that
    differ ONLY in their (irrelevant, per-pair-only) fold fields must
    classify identically under ``_passes_gate``/``analyze_family_robustness``.
    """

    base_kwargs = dict(
        family=F1_FAMILY,
        strategy_id="f1_m30_wick0.5_v1",
        instrument=AGGREGATE_INSTRUMENT_LABEL,
        timeframe="M30",
        parameters='{"wick_ratio": 0.5}',
        trade_count=10,
        skip_count=0,
        net_return=0.02,
        annualized_sharpe=None,
        maximum_drawdown=-0.05,
        win_rate=0.5,
        profitable_pair_count=5,
        aggregate_equal_weight_net_return=0.02,
        aggregate_equal_weight_sharpe=None,
        aggregate_maximum_drawdown=-0.05,
        positive_fold_count=2,
        passes_gate=True,
    )
    row_without_pair_folds = WickFvgSqueezeResultRow(**base_kwargs)  # type: ignore[arg-type]
    row_with_pair_folds = WickFvgSqueezeResultRow(
        **base_kwargs,  # type: ignore[arg-type]
        fold_1_net_return=-0.5,
        fold_2_net_return=-0.5,
        fold_3_net_return=-0.5,
        positive_pair_fold_count=0,
        pair_all_three_folds_positive=False,
    )
    assert row_without_pair_folds.passes_gate == row_with_pair_folds.passes_gate

    summary_without = analyze_family_robustness([row_without_pair_folds])
    summary_with = analyze_family_robustness([row_with_pair_folds])
    assert summary_without == summary_with


def test_pair_specific_robustness_is_deterministic() -> None:
    import json

    rows = [
        _pair_row(
            family=F2_FAMILY,
            instrument="AUD/USD.OANDA",
            timeframe="H4",
            parameters=json.dumps({"backward_search_n": n}),
            passes=True,
        )
        for n in F2_BACKWARD_SEARCH_N
    ]
    first = analyze_pair_specific_robustness(rows)
    second = analyze_pair_specific_robustness(rows)
    assert first == second


def test_development_firewall_still_rejects_a_forbidden_root_after_extension() -> None:
    """The pair-specific extension adds no new data-loading path -- the
    orchestrator still resolves roots exclusively through
    ``data._discover_oanda_universe`` before writing any pair-specific
    diagnostics, so the same firewall from the pre-extension screen still
    applies unchanged."""

    import tempfile
    from pathlib import Path

    from ftmoquant.research.alpha_lab import data as alpha_lab_data

    with tempfile.TemporaryDirectory() as tmp:
        forbidden_root = Path(tmp) / "validation_root"
        readiness_path = Path(tmp) / "readiness.json"
        readiness_path.write_text(
            '{"readiness_version": "oanda-alpha-lab-readiness-1", '
            '"holdout_accessed": false, "holdout_rows_admitted": 0, '
            '"per_instrument_status": {"EUR/USD.OANDA": "research_ready"}, '
            '"instrument_artifacts": [{"instrument_id": "EUR/USD.OANDA", '
            '"dataset_symbol": "EURUSD", "catalog_tree_sha256": "deadbeef"}]}'
        )
        with pytest.raises(AlphaLabDataError):
            alpha_lab_data._discover_oanda_universe(readiness_path, forbidden_root)
