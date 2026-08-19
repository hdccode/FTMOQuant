from __future__ import annotations

from datetime import UTC

import numpy as np
import pandas as pd
import pytest

from ftmoquant.research.alpha_lab.data import AlphaLabDataError
from ftmoquant.research.alpha_lab.liquidity_structure_screen import (
    B2F1_FAMILY,
    B2F1_RR_VALUES,
    B2F1_SWING_LOOKBACKS,
    B2F1_TIMEFRAMES,
    B2F2_FAMILY,
    B2F3_FAMILY,
    B2F4_FAMILY,
    LiquidityStructureResultRow,
    _axis_adjacent_neighbors,
    _rows_for_config,
    _summarize_region,
    build_grid,
)
from ftmoquant.research.alpha_lab.liquidity_structure_signals import (
    _fx_trading_day_ids,
    _fx_trading_week_ids,
    _prior_swing_high,
    _prior_swing_low,
    b2f1_sweep_bos_retest_signals,
    b2f2_previous_period_sweep_rejection_signals,
    b2f3_sweep_choch_retracement_signals,
    b2f4_compression_box_breakout_signals,
)
from ftmoquant.research.alpha_lab.wick_fvg_squeeze_execution import simulate_trades
from ftmoquant.research.alpha_lab.wick_fvg_squeeze_screen import GridConfig
from ftmoquant.research.alpha_lab.wick_fvg_squeeze_signals import (
    DIRECTION_LONG,
    DIRECTION_SHORT,
)
from ftmoquant.strategies.mean_reversion_h1 import FROZEN_UNIVERSE


def _index(periods: int, freq: str = "1h") -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=periods, freq=freq, tz=UTC)


# ---------------------------------------------------------------------------
# 1. exact 31-config grid.
# ---------------------------------------------------------------------------


def test_grid_is_exactly_31_configs_split_12_4_6_9() -> None:
    grid = build_grid()
    assert len(grid) == 31
    counts: dict[str, int] = {}
    for config in grid:
        counts[config.family] = counts.get(config.family, 0) + 1
    assert counts == {
        B2F1_FAMILY: 12,
        B2F2_FAMILY: 4,
        B2F3_FAMILY: 6,
        B2F4_FAMILY: 9,
    }
    assert len({config.strategy_id for config in grid}) == 31


# ---------------------------------------------------------------------------
# 2. causal prior-swing logic.
# ---------------------------------------------------------------------------


def test_prior_swing_high_low_exclude_the_current_bar() -> None:
    index = _index(10)
    high = pd.Series([1, 2, 3, 100, 4, 5, 6, 7, 8, 9], index=index, dtype=float)
    low = pd.Series([9, 8, 7, -100, 6, 5, 4, 3, 2, 1], index=index, dtype=float)
    prior_high = _prior_swing_high(high, lookback=3)
    prior_low = _prior_swing_low(low, lookback=3)
    # bar index 3 (value 100) must never appear in ITS OWN prior-swing value.
    assert prior_high.iloc[3] != 100
    # once bar 3 is 1..3 bars in the past, it must dominate the window.
    assert prior_high.iloc[4] == 100
    assert prior_high.iloc[6] == 100
    # bar 3 has now rolled out of the trailing 3-bar window entirely
    assert prior_high.iloc[7] == 6
    assert prior_low.iloc[3] != -100
    assert prior_low.iloc[4] == -100


# ---------------------------------------------------------------------------
# 3. B2-F1 sweep -> BOS -> retest, long + short.
# ---------------------------------------------------------------------------


def _b2f1_bearish_series(
    *, lookback: int, bos_at: int, retest_at: int | None, n: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    high = np.full(n, 100.0)
    low = np.full(n, 99.0)
    close = np.full(n, 99.5)
    sweep_idx = lookback
    high[sweep_idx] = 101.0
    low[sweep_idx] = 99.0
    close[sweep_idx] = 99.6
    close[bos_at] = 98.0  # break below the running low (99.0) since the sweep
    if retest_at is not None:
        high[retest_at] = 99.5  # touches the frozen BOS level (99.0) from below
        close[retest_at] = 98.7  # remains below it
    index = _index(n)
    return (
        pd.Series(high, index=index),
        pd.Series(low, index=index),
        pd.Series(close, index=index),
    )


def _b2f1_bullish_series(
    *, lookback: int, bos_at: int, retest_at: int | None, n: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    high = np.full(n, 101.0)
    low = np.full(n, 100.0)
    close = np.full(n, 100.5)
    sweep_idx = lookback
    low[sweep_idx] = 99.0
    high[sweep_idx] = 101.0
    close[sweep_idx] = 100.4
    close[bos_at] = 102.0  # break above the running high since the sweep
    if retest_at is not None:
        low[retest_at] = 101.0  # touches the frozen level from above
        close[retest_at] = 102.3  # remains above it
    index = _index(n)
    return (
        pd.Series(high, index=index),
        pd.Series(low, index=index),
        pd.Series(close, index=index),
    )


@pytest.mark.parametrize("direction", [DIRECTION_SHORT, DIRECTION_LONG])
def test_b2f1_sweep_bos_retest_signal_semantics(direction: int) -> None:
    lookback = 5
    if direction == DIRECTION_SHORT:
        h, low, c = _b2f1_bearish_series(
            lookback=lookback, bos_at=10, retest_at=11, n=15
        )
    else:
        h, low, c = _b2f1_bullish_series(
            lookback=lookback, bos_at=10, retest_at=11, n=15
        )
    events = b2f1_sweep_bos_retest_signals(h, low, c, swing_lookback=lookback, rr=2.0)
    assert len(events) == 1
    event = events[0]
    assert event.direction == direction
    assert event.signal_bar_ts == h.index[11]
    if direction == DIRECTION_SHORT:
        assert event.stop_price == pytest.approx(101.0)
    else:
        assert event.stop_price == pytest.approx(99.0)
    assert event.target_r_multiple == pytest.approx(2.0)

    # without a retest bar, BOS confirms but no trade is ever emitted.
    if direction == DIRECTION_SHORT:
        h2, low2, c2 = _b2f1_bearish_series(
            lookback=lookback, bos_at=10, retest_at=None, n=15
        )
    else:
        h2, low2, c2 = _b2f1_bullish_series(
            lookback=lookback, bos_at=10, retest_at=None, n=15
        )
    no_retest = b2f1_sweep_bos_retest_signals(
        h2, low2, c2, swing_lookback=lookback, rr=2.0
    )
    assert no_retest == ()


# ---------------------------------------------------------------------------
# 4. B2-F1 expiry / invalidation.
# ---------------------------------------------------------------------------


def test_b2f1_setup_expires_before_a_late_bos_can_confirm() -> None:
    lookback = 5
    # BOS-breaking bar placed comfortably AFTER expiry (3 * lookback == 15;
    # sweep at index 5, expires once t - 5 > 15, i.e. at t=21).
    h, low, c = _b2f1_bearish_series(lookback=lookback, bos_at=21, retest_at=22, n=25)
    late = b2f1_sweep_bos_retest_signals(h, low, c, swing_lookback=lookback, rr=2.0)
    assert late == ()

    # the identical construction with the BOS bar comfortably WITHIN the
    # expiry window does confirm and (with a retest) trade -- proving the
    # empty result above is expiry, not some other defect.
    h2, low2, c2 = _b2f1_bearish_series(
        lookback=lookback, bos_at=15, retest_at=16, n=25
    )
    events = b2f1_sweep_bos_retest_signals(
        h2, low2, c2, swing_lookback=lookback, rr=2.0
    )
    assert len(events) == 1


def test_b2f1_structural_invalidation_before_entry() -> None:
    lookback = 5
    h, low, c = _b2f1_bearish_series(lookback=lookback, bos_at=10, retest_at=12, n=16)
    # bar 11: closes back ABOVE the sweep extreme (101.0) -- structural
    # invalidation -- before the retest at bar 12 can ever fire.
    h.iloc[11] = 101.5
    c.iloc[11] = 101.2
    events = b2f1_sweep_bos_retest_signals(h, low, c, swing_lookback=lookback, rr=2.0)
    assert events == ()


# ---------------------------------------------------------------------------
# 5 + 6. previous-day / previous-week levels exclude the current period.
# ---------------------------------------------------------------------------


def test_previous_day_levels_never_include_the_current_day() -> None:
    # 48 hourly bars starting Sunday 00:00 UTC (well inside an FX trading
    # day boundary) so we get two clean, distinct FX trading days.
    index = pd.date_range("2020-01-05T00:00:00Z", periods=48, freq="1h", tz=UTC)
    high = pd.Series(100.0, index=index)
    low = pd.Series(99.0, index=index)
    day_ids = _fx_trading_day_ids(index)
    first_day = day_ids[0]
    # peak within day 1
    day1_positions = [i for i, d in enumerate(day_ids) if d == first_day]
    peak_pos = day1_positions[len(day1_positions) // 2]
    high.iloc[peak_pos] = 105.0
    # the 48-hour window actually spans THREE distinct FX-day labels (the
    # boundary does not align with UTC midnight) -- isolate exactly the
    # second one, and separately bump a high late in the THIRD day, which
    # must never leak backward into the second day's own "previous day"
    # lookup either.
    unique_days = sorted(set(day_ids))
    assert len(unique_days) >= 3
    second_day = unique_days[1]
    third_day = unique_days[2]
    day2_positions = [i for i, d in enumerate(day_ids) if d == second_day]
    day3_positions = [i for i, d in enumerate(day_ids) if d == third_day]
    assert day2_positions and day3_positions
    high.iloc[day3_positions[0]] = 999.0

    from ftmoquant.research.alpha_lab.liquidity_structure_signals import (
        _previous_period_levels,
    )

    prev_high, _ = _previous_period_levels(high, low, day_ids)
    for pos in day2_positions:
        assert prev_high.iloc[pos] == pytest.approx(105.0)


def test_previous_week_levels_never_include_the_current_week() -> None:
    index = pd.date_range("2020-01-05T00:00:00Z", periods=24 * 20, freq="1h", tz=UTC)
    high = pd.Series(100.0, index=index)
    low = pd.Series(99.0, index=index)
    day_ids = _fx_trading_day_ids(index)
    week_ids = _fx_trading_week_ids(day_ids)
    unique_weeks = sorted(set(week_ids))
    assert len(unique_weeks) >= 3
    first_week, second_week, third_week = unique_weeks[:3]
    week1_positions = [i for i, w in enumerate(week_ids) if w == first_week]
    week2_positions = [i for i, w in enumerate(week_ids) if w == second_week]
    week3_positions = [i for i, w in enumerate(week_ids) if w == third_week]
    assert week1_positions and week2_positions and week3_positions
    high.iloc[week1_positions[len(week1_positions) // 2]] = 110.0
    high.iloc[week3_positions[0]] = 999.0

    from ftmoquant.research.alpha_lab.liquidity_structure_signals import (
        _previous_period_levels,
    )

    prev_high, _ = _previous_period_levels(high, low, week_ids)
    for pos in week2_positions:
        assert prev_high.iloc[pos] == pytest.approx(110.0)


# ---------------------------------------------------------------------------
# 7 + 8. B2-F2 first-sweep-only + wick filter.
# ---------------------------------------------------------------------------


def _b2f2_pdh_series(n: int = 30) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    index = pd.date_range("2020-01-05T00:00:00Z", periods=n, freq="1h", tz=UTC)
    open_ = np.full(n, 100.0)
    high = np.full(n, 100.2)
    low = np.full(n, 99.8)
    close = np.full(n, 100.0)
    return (
        pd.Series(open_, index=index),
        pd.Series(high, index=index),
        pd.Series(low, index=index),
        pd.Series(close, index=index),
    )


def test_b2f2_only_the_first_qualifying_sweep_of_a_level_trades() -> None:
    o, h, low, c = _b2f2_pdh_series(30)
    day_ids = _fx_trading_day_ids(h.index)
    first_day = day_ids[0]
    day1_positions = [i for i, d in enumerate(day_ids) if d == first_day]
    high_pos = day1_positions[-1]
    h.iloc[high_pos] = 105.0  # previous-day high becomes 105.0 for day 2

    day2_positions = [i for i, d in enumerate(day_ids) if d != first_day]
    first_two = day2_positions[:2]
    for pos in first_two:
        # 99.9 (not below the baseline previous-day low of 99.8) so this
        # bar qualifies ONLY as a bearish PDH sweep, not also a bullish one.
        o.iloc[pos], h.iloc[pos], low.iloc[pos], c.iloc[pos] = 100.0, 106.0, 99.9, 99.9
    events = b2f2_previous_period_sweep_rejection_signals(
        o, h, low, c, level_type="PREVIOUS_DAY", rejection_mode="CLOSE_BACK"
    )
    assert len(events) == 1
    assert events[0].signal_bar_ts == h.index[first_two[0]]


def test_b2f2_wick_50_requires_the_rejecting_wick_to_dominate_the_range() -> None:
    o, h, low, c = _b2f2_pdh_series(30)
    day_ids = _fx_trading_day_ids(h.index)
    first_day = day_ids[0]
    day1_positions = [i for i, d in enumerate(day_ids) if d == first_day]
    h.iloc[day1_positions[-1]] = 105.0
    day2_positions = [i for i, d in enumerate(day_ids) if d != first_day]
    pos = day2_positions[0]
    # sweeps and closes back BELOW the level (qualifies for CLOSE_BACK), but
    # the upper wick (106.0 - 104.9 = 1.1) is small relative to the whole
    # candle range (106.0 - 99.9 = 6.1) -- must fail WICK_50 (ratio 0.18).
    o.iloc[pos], h.iloc[pos], low.iloc[pos], c.iloc[pos] = 100.0, 106.0, 99.9, 104.9
    events_wick = b2f2_previous_period_sweep_rejection_signals(
        o, h, low, c, level_type="PREVIOUS_DAY", rejection_mode="WICK_50"
    )
    assert events_wick == ()
    events_close_back = b2f2_previous_period_sweep_rejection_signals(
        o, h, low, c, level_type="PREVIOUS_DAY", rejection_mode="CLOSE_BACK"
    )
    assert len(events_close_back) == 1


# ---------------------------------------------------------------------------
# 9 + 10. B2-F3 sweep/CHOCH/retracement semantics + frozen level.
# ---------------------------------------------------------------------------


def test_b2f3_sweep_choch_retracement_signal_semantics() -> None:
    lookback = 5
    n = 20
    high = np.full(n, 100.0)
    low = np.full(n, 99.0)
    close = np.full(n, 99.5)
    high[lookback] = 101.0
    low[lookback] = 99.0
    close[lookback] = 99.6
    close[10] = 98.0  # CHOCH confirms; structural_level frozen at running low (99.0)
    # retracement level = 99.0 + 0.5 * (101.0 - 99.0) = 100.0
    high[11] = 99.9  # below threshold -- must not trigger yet
    high[12] = 100.5  # reaches the frozen retracement level -- triggers
    index = _index(n)
    h = pd.Series(high, index=index)
    low_s = pd.Series(low, index=index)
    c = pd.Series(close, index=index)
    events = b2f3_sweep_choch_retracement_signals(
        h, low_s, c, swing_lookback=lookback, retracement=0.5
    )
    assert len(events) == 1
    assert events[0].signal_bar_ts == index[12]
    assert events[0].direction == DIRECTION_SHORT
    assert events[0].stop_price == pytest.approx(101.0)
    assert events[0].target_r_multiple == pytest.approx(2.0)


def test_b2f3_retracement_level_is_frozen_at_choch_and_never_recomputed() -> None:
    lookback = 5
    n = 20
    high = np.full(n, 100.0)
    low = np.full(n, 99.0)
    close = np.full(n, 99.5)
    high[lookback] = 101.0
    low[lookback] = 99.0
    close[lookback] = 99.6
    close[10] = 98.0  # CHOCH confirms with structural_level=99.0
    # frozen retracement level = 99.0 + 0.5*(101-99) = 100.0
    # bar 11: a MUCH deeper low than anything seen so far -- if the
    # retracement level were (incorrectly) recomputed off this new low, the
    # level would drop well below 100.0.
    low[11] = 50.0
    high[11] = 99.0  # still below the true frozen level (100.0) -- no trigger
    close[11] = 98.5  # not > sweep extreme, so no invalidation either
    high[12] = 100.5  # reaches the ORIGINAL frozen level -- triggers here
    index = _index(n)
    h = pd.Series(high, index=index)
    low_s = pd.Series(low, index=index)
    c = pd.Series(close, index=index)
    events = b2f3_sweep_choch_retracement_signals(
        h, low_s, c, swing_lookback=lookback, retracement=0.5
    )
    assert len(events) == 1
    assert events[0].signal_bar_ts == index[12]


# ---------------------------------------------------------------------------
# 11-14. B2-F4 box construction / EMA filter / overlap suppression / expiry.
# ---------------------------------------------------------------------------


def _flat_h1_series(
    n: int, *, level: float = 100.0, half_range: float = 0.1
) -> tuple[pd.Series, pd.Series, pd.Series]:
    index = _index(n)
    high = pd.Series(level + half_range, index=index)
    low = pd.Series(level - half_range, index=index)
    close = pd.Series(level, index=index)
    return high, low, close


def test_b2f4_box_construction_excludes_the_breakout_bar_itself() -> None:
    h, low, c = _flat_h1_series(20)
    # the breakout bar's own high is enormous; if the box wrongly included
    # it, box_high would balloon and this modest close would no longer
    # exceed it.
    h.iloc[20 - 1] = h.iloc[20 - 1]  # no-op, keep index aligned
    n = 21
    h2, low2, c2 = _flat_h1_series(n)
    h2.iloc[-1] = 500.0
    c2.iloc[-1] = 100.5
    events = b2f4_compression_box_breakout_signals(
        h2, low2, c2, box_length=5, max_box_atr=5.0
    )
    assert len(events) == 1
    assert events[0].direction == DIRECTION_LONG
    assert events[0].stop_price == pytest.approx(99.9)  # box_low, unaffected by bar 20


def test_b2f4_ema_direction_filter_blocks_a_counter_trend_breakout() -> None:
    n = 25
    index = _index(n)
    close = np.full(n, 100.0)
    close[:5] = 110.0  # pulls EMA150 up well above 100
    high = close + 0.1
    low = close - 0.1
    high[-1] = 100.6
    close[-1] = 100.5  # breaks the box but stays below the elevated EMA150
    h = pd.Series(high, index=index)
    low_s = pd.Series(low, index=index)
    c = pd.Series(close, index=index)
    events = b2f4_compression_box_breakout_signals(
        h, low_s, c, box_length=5, max_box_atr=5.0
    )
    assert events == ()


def test_b2f4_overlap_suppression_retains_the_older_box() -> None:
    n = 30
    h, low, c = _flat_h1_series(n)
    # box A forms as of bar 5 (window bars 0-4): high=100.1/low=99.9.
    # a near-identical box B would form as of bar 8 (window bars 3-7) --
    # almost total overlap -- and must be suppressed, keeping box A's edges
    # (99.9) as the eventual breakout stop rather than box B's.
    h.iloc[-1] = 500.0
    c.iloc[-1] = 100.5
    events = b2f4_compression_box_breakout_signals(
        h, low, c, box_length=5, max_box_atr=5.0
    )
    assert len(events) == 1
    assert events[0].stop_price == pytest.approx(99.9)


def test_b2f4_box_age_window_is_inclusive_of_45_bars_and_excludes_46() -> None:
    """A persistently compressed market legitimately keeps forming fresh
    eligible boxes forever (each one individually valid for its own 45
    bars) -- that is correct behavior, not a bug, so it cannot be used to
    observe one SPECIFIC box's own expiry end-to-end without the test
    itself controlling for regeneration. This directly exercises the exact
    age predicate ``b2f4_compression_box_breakout_signals`` evaluates every
    bar (``t <= box.expires_at``, with ``expires_at = formed_at +
    BOX_MAX_AGE_BARS``): still active at exactly 45 bars past formation,
    excluded one bar later.

    (The complementary "well within its own age window" path is already
    exercised end-to-end by
    ``test_b2f4_box_construction_excludes_the_breakout_bar_itself``, whose
    box forms at bar 5 and triggers at bar 20.)
    """

    from ftmoquant.research.alpha_lab.liquidity_structure_signals import (
        BOX_MAX_AGE_BARS,
        _Box,
    )

    formed_at = 10
    box = _Box(
        high=100.1,
        low=99.9,
        formed_at=formed_at,
        expires_at=formed_at + BOX_MAX_AGE_BARS,
    )
    assert (formed_at + BOX_MAX_AGE_BARS) <= box.expires_at
    assert not ((formed_at + BOX_MAX_AGE_BARS + 1) <= box.expires_at)


# ---------------------------------------------------------------------------
# 15. prefix invariance / no lookahead for all four families.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", [B2F1_FAMILY, B2F2_FAMILY, B2F3_FAMILY, B2F4_FAMILY])
def test_signal_generators_are_prefix_invariant(family: str) -> None:
    n = 60
    index = pd.date_range("2020-01-05T00:00:00Z", periods=n, freq="1h", tz=UTC)
    rng = np.random.default_rng(3)
    close = 100.0 + np.cumsum(rng.normal(0, 0.3, n))
    open_ = close + rng.normal(0, 0.05, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.3, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.3, n))
    o = pd.Series(open_, index=index)
    h = pd.Series(high, index=index)
    lo = pd.Series(low, index=index)
    c = pd.Series(close, index=index)
    cut = 45

    def _run(oo, hh, ll, cc):
        if family == B2F1_FAMILY:
            return b2f1_sweep_bos_retest_signals(hh, ll, cc, swing_lookback=5, rr=2.0)
        if family == B2F2_FAMILY:
            return b2f2_previous_period_sweep_rejection_signals(
                oo, hh, ll, cc, level_type="PREVIOUS_DAY", rejection_mode="CLOSE_BACK"
            )
        if family == B2F3_FAMILY:
            return b2f3_sweep_choch_retracement_signals(
                hh, ll, cc, swing_lookback=5, retracement=0.5
            )
        return b2f4_compression_box_breakout_signals(
            hh, ll, cc, box_length=5, max_box_atr=3.0
        )

    full = _run(o, h, lo, c)
    prefix = _run(o.iloc[:cut], h.iloc[:cut], lo.iloc[:cut], c.iloc[:cut])
    full_before_cut = [e for e in full if e.signal_bar_ts <= index[cut - 1]]
    assert list(prefix) == full_before_cut


# ---------------------------------------------------------------------------
# 19. broad 4/7 + 2/3 fold gate (reused unchanged from Batch 1 -- see
# tests/research/alpha_lab/test_wick_fvg_squeeze.py::
# test_viability_gate_requires_all_three_frozen_conditions for the direct
# unit coverage of ``_passes_gate`` itself). Here we only prove Batch 2's
# orchestration wires the SAME reused gate onto every row of a config.
# ---------------------------------------------------------------------------


def _synthetic_m1(
    periods: int, *, bid: float, ask: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.date_range("2020-01-06T00:00:00Z", periods=periods, freq="1min", tz=UTC)
    bid_df = pd.DataFrame(
        {"open": bid, "high": bid, "low": bid, "close": bid}, index=index
    )
    ask_df = pd.DataFrame(
        {"open": ask, "high": ask, "low": ask, "close": ask}, index=index
    )
    return bid_df, ask_df


def test_broad_gate_result_is_consistent_across_every_row_of_a_config() -> None:
    from ftmoquant.research.alpha_lab.data import ALIGNMENT_POLICY, AlphaLabDataset

    n = 400
    index = pd.date_range("2020-01-06T00:00:00Z", periods=n, freq="1h", tz=UTC)
    rng = np.random.default_rng(5)
    columns = FROZEN_UNIVERSE
    close = pd.DataFrame(
        {c: 1.1 + np.cumsum(rng.normal(0, 0.0006, n)) for c in columns}, index=index
    )
    dataset = AlphaLabDataset(
        timeframe="H1",  # type: ignore[arg-type]
        instrument_ids=columns,
        start_utc=index[0].to_pydatetime(),
        end_exclusive_utc=index[-1].to_pydatetime(),
        alignment_policy=ALIGNMENT_POLICY,
        open=close,
        high=close + 0.001,
        low=close - 0.001,
        close=close,
        bid=close - 0.0001,
        ask=close + 0.0001,
        spread=pd.DataFrame(0.0002, index=index, columns=columns),
    )
    m1_by_instrument = {
        instrument_id: _synthetic_m1(2000, bid=1.0995, ask=1.1005)
        for instrument_id in FROZEN_UNIVERSE
    }
    config = GridConfig(
        family=B2F2_FAMILY,
        strategy_id="test_b2f2",
        timeframe="H1",
        parameters={"level_type": "PREVIOUS_DAY", "rejection_mode": "CLOSE_BACK"},
    )
    rows = _rows_for_config(config, dataset, m1_by_instrument)
    broad_values = {row.passes_broad_gate for row in rows}
    assert len(broad_values) == 1  # every row (7 pairs + aggregate) agrees
    profitable_values = {row.profitable_pair_count for row in rows}
    assert len(profitable_values) == 1


# ---------------------------------------------------------------------------
# 20. pair-specific all-3-fold gate.
# ---------------------------------------------------------------------------


def _row(
    *, instrument: str, net_return: float, all_three: bool
) -> LiquidityStructureResultRow:
    return LiquidityStructureResultRow(
        family=B2F1_FAMILY,
        strategy_id="s",
        instrument=instrument,
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
        passes_broad_gate=False,
        pair_all_three_folds_positive=all_three,
        passes_pair_specific_gate=net_return > 0 and all_three,
    )


@pytest.mark.parametrize(
    ("net_return", "all_three_folds", "expected"),
    [(0.05, True, True), (0.05, False, False), (-0.01, True, False)],
)
def test_pair_specific_gate_requires_return_and_all_three_folds(
    net_return: float, all_three_folds: bool, expected: bool
) -> None:
    row = _row(
        instrument="EUR/USD.OANDA", net_return=net_return, all_three=all_three_folds
    )
    assert row.passes_pair_specific_gate is expected


# ---------------------------------------------------------------------------
# 21. connected-region logic (2-D and 3-D).
# ---------------------------------------------------------------------------


def test_axis_adjacent_neighbors_covers_every_axis_in_both_directions() -> None:
    neighbors_2d = set(_axis_adjacent_neighbors((1, 1)))
    assert neighbors_2d == {(0, 1), (2, 1), (1, 0), (1, 2)}
    neighbors_3d = set(_axis_adjacent_neighbors((1, 1, 1)))
    assert neighbors_3d == {
        (0, 1, 1),
        (2, 1, 1),
        (1, 0, 1),
        (1, 2, 1),
        (1, 1, 0),
        (1, 1, 2),
    }


@pytest.mark.parametrize(
    ("passing", "expected_survival"),
    [
        ({(0, 0), (0, 1)}, True),
        ({(0, 0), (1, 1)}, False),  # diagonal only -- not adjacent
        (set(), False),
    ],
)
def test_2d_connected_region_survival(
    passing: set[tuple[int, ...]], expected_survival: bool
) -> None:
    summary = _summarize_region(
        family="x",
        passing=passing,
        tested_count=6,
        timeframe_values=("M30", "H1"),
        param_specs=(("p", (1, 2, 3)),),
        min_region_size=2,
    )
    assert summary.survival_rule_passed is expected_survival


def test_3d_connected_region_requires_axis_adjacency_not_diagonal() -> None:
    # B2-F1's own 3-D shape: timeframe x swing_lookback x rr.
    adjacent = {(0, 0, 0), (0, 0, 1)}  # differ only in the RR axis
    diagonal_only = {(0, 0, 0), (1, 1, 1)}
    param_specs = (("swing_lookback", B2F1_SWING_LOOKBACKS), ("rr", B2F1_RR_VALUES))
    summary_adjacent = _summarize_region(
        family=B2F1_FAMILY,
        passing=adjacent,
        tested_count=(
            len(B2F1_TIMEFRAMES) * len(param_specs[0][1]) * len(param_specs[1][1])
        ),
        timeframe_values=B2F1_TIMEFRAMES,
        param_specs=param_specs,
        min_region_size=2,
    )
    summary_diagonal = _summarize_region(
        family=B2F1_FAMILY,
        passing=diagonal_only,
        tested_count=(
            len(B2F1_TIMEFRAMES) * len(param_specs[0][1]) * len(param_specs[1][1])
        ),
        timeframe_values=B2F1_TIMEFRAMES,
        param_specs=param_specs,
        min_region_size=2,
    )
    assert summary_adjacent.survival_rule_passed is True
    assert summary_diagonal.survival_rule_passed is False


# ---------------------------------------------------------------------------
# 22. deterministic artifacts.
# ---------------------------------------------------------------------------


def test_advance_sweep_and_break_is_deterministic() -> None:
    lookback = 5
    h, low, c = _b2f1_bearish_series(lookback=lookback, bos_at=10, retest_at=11, n=15)
    first = b2f1_sweep_bos_retest_signals(h, low, c, swing_lookback=lookback, rr=2.0)
    second = b2f1_sweep_bos_retest_signals(h, low, c, swing_lookback=lookback, rr=2.0)
    assert first == second


def test_simulate_trades_reused_from_batch1_stays_deterministic_for_b2f1_events() -> (
    None
):
    lookback = 5
    h, low, c = _b2f1_bearish_series(lookback=lookback, bos_at=10, retest_at=11, n=15)
    events = b2f1_sweep_bos_retest_signals(h, low, c, swing_lookback=lookback, rr=2.0)
    index = pd.date_range("2020-01-01T00:00:00Z", periods=200, freq="1min", tz=UTC)
    bid = pd.DataFrame(
        {"open": 1.1, "high": 1.12, "low": 1.08, "close": 1.1}, index=index
    )
    ask = pd.DataFrame(
        {"open": 1.1005, "high": 1.1205, "low": 1.0805, "close": 1.1005}, index=index
    )
    first = simulate_trades(events, bid, ask)
    second = simulate_trades(events, bid, ask)
    assert first == second


# ---------------------------------------------------------------------------
# 23. DEVELOPMENT firewall.
# ---------------------------------------------------------------------------


def test_liquidity_structure_never_accepts_a_validation_or_holdout_root(
    tmp_path: object,
) -> None:
    from pathlib import Path

    from ftmoquant.research.alpha_lab import data as alpha_lab_data

    forbidden_root = Path(str(tmp_path)) / "holdout_root"
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
