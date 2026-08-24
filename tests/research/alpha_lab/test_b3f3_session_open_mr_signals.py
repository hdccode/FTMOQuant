from __future__ import annotations

import datetime as dt
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from ftmoquant.research.alpha_lab.b3f3_session_open_mr_signals import (
    ATR_WINDOW,
    B3F3Config,
    B3F3SignalError,
    build_b3f3_grid,
    compute_lagged_atr,
    compute_opening_anchors,
    derive_m5_mid_ohlc,
    generate_b3f3_decisions,
    is_in_entry_window,
    is_session_open_bar,
    london_local_date,
    time_exit_boundary_utc,
)


def _m1(prices: list[float], *, start: str) -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(prices), freq="min", tz="UTC")
    return pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices}, index=idx
    )


def _m5_bar(open_: float, high: float, low: float, close: float) -> dict[str, float]:
    return {"open": open_, "high": high, "low": low, "close": close}


def _m5_frame(rows: list[dict[str, float]], *, start: str) -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(rows), freq="5min", tz="UTC")
    return pd.DataFrame(rows, index=idx)


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------


def test_grid_has_exactly_12_configs() -> None:
    grid = build_b3f3_grid()
    assert len(grid) == 12
    assert len({c.config_id for c in grid}) == 12


def test_config_rejects_non_frozen_values() -> None:
    with pytest.raises(B3F3SignalError):
        B3F3Config(Decimal("3.0"), Decimal("0.75"), "OPEN_ANCHOR")
    with pytest.raises(B3F3SignalError):
        B3F3Config(Decimal("1.0"), Decimal("0.50"), "OPEN_ANCHOR")
    with pytest.raises(B3F3SignalError):
        B3F3Config(Decimal("1.0"), Decimal("0.75"), "FULL_REVERSION")


# ---------------------------------------------------------------------------
# Session windows / DST
# ---------------------------------------------------------------------------


def test_session_open_bar_is_the_0800_0805_close_label() -> None:
    winter_date = "2024-01-15"  # GMT (UTC+0)
    assert is_session_open_bar(pd.Timestamp(f"{winter_date}T08:05:00Z"))
    assert not is_session_open_bar(pd.Timestamp(f"{winter_date}T08:00:00Z"))
    assert not is_session_open_bar(pd.Timestamp(f"{winter_date}T08:10:00Z"))


def test_entry_window_is_dst_safe_across_bst_and_gmt() -> None:
    # 08:05 London local in BST (UTC+1, July) is 07:05 UTC.
    summer_ts = pd.Timestamp("2024-07-15T07:05:00Z")
    assert is_in_entry_window(summer_ts)
    # 08:05 London local in GMT (UTC+0, January) is 08:05 UTC.
    winter_ts = pd.Timestamp("2024-01-15T08:05:00Z")
    assert is_in_entry_window(winter_ts)
    # 07:00 UTC in January is 07:00 London local -- before the window opens.
    assert not is_in_entry_window(pd.Timestamp("2024-01-15T07:00:00Z"))


def test_entry_window_inclusive_at_both_endpoints() -> None:
    assert is_in_entry_window(pd.Timestamp("2024-01-15T08:05:00Z"))
    assert is_in_entry_window(pd.Timestamp("2024-01-15T09:00:00Z"))
    assert not is_in_entry_window(pd.Timestamp("2024-01-15T09:01:00Z"))
    assert not is_in_entry_window(pd.Timestamp("2024-01-15T08:04:00Z"))


def test_time_exit_boundary_is_dst_safe() -> None:
    winter_boundary = time_exit_boundary_utc(dt.date(2024, 1, 15))
    summer_boundary = time_exit_boundary_utc(dt.date(2024, 7, 15))
    assert winter_boundary.hour == 11  # GMT: 11:00 local == 11:00 UTC
    assert summer_boundary.hour == 10  # BST: 11:00 local == 10:00 UTC


def test_london_local_date_matches_civil_calendar() -> None:
    assert london_local_date(pd.Timestamp("2024-01-15T23:30:00Z")) == dt.date(
        2024, 1, 15
    )
    assert london_local_date(pd.Timestamp("2024-07-15T23:30:00Z")) == dt.date(
        2024, 7, 16
    )


# ---------------------------------------------------------------------------
# M5 derivation (causality)
# ---------------------------------------------------------------------------


def test_m5_derivation_uses_only_completed_windows() -> None:
    n = 15  # bars at :01..:15 -> exactly 3 complete 5-bar buckets
    prices = [1.10 + 0.0001 * i for i in range(n)]
    bid = _m1(prices, start="2024-01-01T00:01:00Z")
    ask = _m1([p + 0.0002 for p in prices], start="2024-01-01T00:01:00Z")
    m5 = derive_m5_mid_ohlc(bid, ask)
    assert len(m5) == 3


def test_m5_derivation_drops_incomplete_trailing_window() -> None:
    n = 8  # 1 complete window (bars 1-5) + 3 leftover bars
    prices = [1.10] * n
    bid = _m1(prices, start="2024-01-01T00:01:00Z")
    ask = _m1([p + 0.0002 for p in prices], start="2024-01-01T00:01:00Z")
    m5 = derive_m5_mid_ohlc(bid, ask)
    assert len(m5) == 1


def test_m5_derivation_mid_price_matches_established_convention() -> None:
    n = 5
    bid_prices = [1.1000] * n
    ask_prices = [1.1002] * n
    bid = _m1(bid_prices, start="2024-01-01T00:01:00Z")
    ask = _m1(ask_prices, start="2024-01-01T00:01:00Z")
    m5 = derive_m5_mid_ohlc(bid, ask)
    assert m5["close"].iloc[0] == pytest.approx(1.1001)


def test_m5_future_append_cannot_alter_historical_bars() -> None:
    n = 15
    rng = np.random.default_rng(0)
    prices = 1.10 + np.cumsum(rng.normal(scale=0.0001, size=n))
    bid = _m1(list(prices), start="2024-01-01T00:01:00Z")
    ask = _m1(list(prices + 0.0002), start="2024-01-01T00:01:00Z")
    short_m5 = derive_m5_mid_ohlc(bid.iloc[:10], ask.iloc[:10])

    extra = 5
    more_prices = np.concatenate(
        [prices, prices[-1] + np.cumsum(rng.normal(scale=0.0001, size=extra))]
    )
    full_bid = _m1(list(more_prices), start="2024-01-01T00:01:00Z")
    full_ask = _m1(list(more_prices + 0.0002), start="2024-01-01T00:01:00Z")
    full_m5 = derive_m5_mid_ohlc(full_bid, full_ask)

    pd.testing.assert_frame_equal(short_m5, full_m5.iloc[: len(short_m5)])


def test_m5_rejects_mismatched_index() -> None:
    bid = _m1([1.1] * 5, start="2024-01-01T00:01:00Z")
    ask = _m1([1.1] * 5, start="2024-01-01T00:02:00Z")
    with pytest.raises(B3F3SignalError):
        derive_m5_mid_ohlc(bid, ask)


# ---------------------------------------------------------------------------
# Opening anchor
# ---------------------------------------------------------------------------


def test_opening_anchor_is_close_of_the_0800_0805_bar() -> None:
    m5 = _m5_frame(
        [_m5_bar(1.1000, 1.1005, 1.0998, 1.1003)], start="2024-01-01T08:05:00Z"
    )
    anchors = compute_opening_anchors(m5)
    assert anchors[dt.date(2024, 1, 1)] == pytest.approx(1.1003)


def test_missing_opening_bar_produces_no_anchor_for_that_date() -> None:
    m5 = _m5_frame(
        [_m5_bar(1.1000, 1.1005, 1.0998, 1.1003)], start="2024-01-01T09:00:00Z"
    )
    anchors = compute_opening_anchors(m5)
    assert dt.date(2024, 1, 1) not in anchors


def test_opening_anchor_frozen_before_entry_window() -> None:
    """The anchor computed from ONLY the opening bar must be identical
    regardless of what happens later in the entry window."""

    open_bar = [_m5_bar(1.1000, 1.1005, 1.0998, 1.1003)]
    m5_a = _m5_frame(
        open_bar + [_m5_bar(1.10, 1.10, 1.10, 1.10)] * 4, start="2024-01-01T08:05:00Z"
    )
    m5_b = _m5_frame(
        open_bar + [_m5_bar(2.0, 5.0, 0.5, 2.0)] * 4, start="2024-01-01T08:05:00Z"
    )
    anchors_a = compute_opening_anchors(m5_a)
    anchors_b = compute_opening_anchors(m5_b)
    assert anchors_a[dt.date(2024, 1, 1)] == anchors_b[dt.date(2024, 1, 1)]


# ---------------------------------------------------------------------------
# Lagged ATR
# ---------------------------------------------------------------------------


def test_lagged_atr_excludes_current_bar() -> None:
    n = ATR_WINDOW + 3
    rows = [_m5_bar(1.10, 1.1010, 1.0990, 1.10) for _ in range(n)]
    m5 = _m5_frame(rows, start="2024-01-01T00:00:00Z")
    lagged = compute_lagged_atr(m5)
    # First ATR_WINDOW values must be NaN (not enough history) and the
    # value at each bar t must be unaffected by bar t's own high/low.
    assert lagged.iloc[:ATR_WINDOW].isna().all()
    m5_spiked = m5.copy()
    spike_pos = ATR_WINDOW + 1
    m5_spiked.iloc[spike_pos, m5_spiked.columns.get_loc("high")] = (
        5.0  # huge spike on the decision bar itself
    )
    lagged_spiked = compute_lagged_atr(m5_spiked)
    assert lagged.iloc[spike_pos] == pytest.approx(lagged_spiked.iloc[spike_pos])


def test_lagged_atr_reflects_prior_bars_true_range() -> None:
    n = ATR_WINDOW + 2
    rows = [_m5_bar(1.10, 1.1010, 1.0990, 1.10) for _ in range(n)]
    m5 = _m5_frame(rows, start="2024-01-01T00:00:00Z")
    baseline = compute_lagged_atr(m5)

    m5_wide = m5.copy()
    wide_pos = ATR_WINDOW - 1  # inside the lookback window for decision bar ATR_WINDOW
    m5_wide.iloc[wide_pos, m5_wide.columns.get_loc("high")] = 1.20
    widened = compute_lagged_atr(m5_wide)
    assert widened.iloc[ATR_WINDOW] > baseline.iloc[ATR_WINDOW]


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------


def _decision_frame(
    bars: list[dict[str, float]],
) -> tuple[pd.DataFrame, dict[dt.date, float], pd.Series]:
    m5 = _m5_frame(bars, start="2024-01-01T08:05:00Z")
    opening_anchors = {dt.date(2024, 1, 1): 1.1000}
    lagged_atr = pd.Series([0.0010] * len(m5), index=m5.index)
    return m5, opening_anchors, lagged_atr


def test_positive_displacement_triggers_short() -> None:
    # displacement = (1.1035 - 1.1000) / 0.0010 = 3.5 >= threshold 2.0
    m5, anchors, atr = _decision_frame([_m5_bar(1.10, 1.1040, 1.0995, 1.1035)])
    config = B3F3Config(Decimal("2.0"), Decimal("0.75"), "OPEN_ANCHOR")
    decisions = generate_b3f3_decisions(
        m5, anchors, atr, instrument_id="EUR/USD.OANDA", config=config
    )
    assert len(decisions) == 1
    assert decisions[0].direction == -1


def test_negative_displacement_triggers_long() -> None:
    # displacement = (1.0965 - 1.1000) / 0.0010 = -3.5 <= -threshold
    m5, anchors, atr = _decision_frame([_m5_bar(1.10, 1.1005, 1.0960, 1.0965)])
    config = B3F3Config(Decimal("2.0"), Decimal("0.75"), "OPEN_ANCHOR")
    decisions = generate_b3f3_decisions(
        m5, anchors, atr, instrument_id="EUR/USD.OANDA", config=config
    )
    assert len(decisions) == 1
    assert decisions[0].direction == 1


def test_displacement_below_threshold_produces_no_signal() -> None:
    # displacement = (1.1015 - 1.1000) / 0.0010 = 1.5 < threshold 2.0
    m5, anchors, atr = _decision_frame([_m5_bar(1.10, 1.1020, 1.0995, 1.1015)])
    config = B3F3Config(Decimal("2.0"), Decimal("0.75"), "OPEN_ANCHOR")
    decisions = generate_b3f3_decisions(
        m5, anchors, atr, instrument_id="EUR/USD.OANDA", config=config
    )
    assert decisions == ()


def test_one_entry_per_day_regardless_of_direction() -> None:
    m5, anchors, atr = _decision_frame(
        [
            _m5_bar(1.10, 1.1040, 1.0995, 1.1035),  # SHORT signal
            _m5_bar(1.10, 1.1005, 1.0960, 1.0965),  # LONG signal, same day
        ]
    )
    config = B3F3Config(Decimal("2.0"), Decimal("0.75"), "OPEN_ANCHOR")
    decisions = generate_b3f3_decisions(
        m5, anchors, atr, instrument_id="EUR/USD.OANDA", config=config
    )
    assert len(decisions) == 1
    assert decisions[0].direction == -1


def test_entry_outside_window_is_rejected() -> None:
    m5 = _m5_frame(
        [_m5_bar(1.10, 1.1040, 1.0995, 1.1035)], start="2024-01-01T09:05:00Z"
    )  # 09:05 is outside the 08:05-09:00 entry window
    anchors = {dt.date(2024, 1, 1): 1.1000}
    atr = pd.Series([0.0010] * len(m5), index=m5.index)
    config = B3F3Config(Decimal("2.0"), Decimal("0.75"), "OPEN_ANCHOR")
    decisions = generate_b3f3_decisions(
        m5, anchors, atr, instrument_id="EUR/USD.OANDA", config=config
    )
    assert decisions == ()


def test_missing_anchor_for_date_produces_no_signal() -> None:
    m5 = _m5_frame(
        [_m5_bar(1.10, 1.1040, 1.0995, 1.1035)], start="2024-01-01T08:05:00Z"
    )
    atr = pd.Series([0.0010] * len(m5), index=m5.index)
    config = B3F3Config(Decimal("2.0"), Decimal("0.75"), "OPEN_ANCHOR")
    decisions = generate_b3f3_decisions(
        m5, {}, atr, instrument_id="EUR/USD.OANDA", config=config
    )
    assert decisions == ()


def test_nan_or_nonpositive_atr_produces_no_signal() -> None:
    m5 = _m5_frame(
        [_m5_bar(1.10, 1.1040, 1.0995, 1.1035)], start="2024-01-01T08:05:00Z"
    )
    anchors = {dt.date(2024, 1, 1): 1.1000}
    atr_nan = pd.Series([np.nan], index=m5.index)
    config = B3F3Config(Decimal("2.0"), Decimal("0.75"), "OPEN_ANCHOR")
    assert (
        generate_b3f3_decisions(
            m5, anchors, atr_nan, instrument_id="EUR/USD.OANDA", config=config
        )
        == ()
    )
    atr_zero = pd.Series([0.0], index=m5.index)
    assert (
        generate_b3f3_decisions(
            m5, anchors, atr_zero, instrument_id="EUR/USD.OANDA", config=config
        )
        == ()
    )


def test_mismatched_atr_index_rejected() -> None:
    m5 = _m5_frame(
        [_m5_bar(1.10, 1.1040, 1.0995, 1.1035)], start="2024-01-01T08:05:00Z"
    )
    anchors = {dt.date(2024, 1, 1): 1.1000}
    atr = pd.Series([0.0010], index=pd.date_range("2024-01-01T09:05:00Z", periods=1))
    config = B3F3Config(Decimal("2.0"), Decimal("0.75"), "OPEN_ANCHOR")
    with pytest.raises(B3F3SignalError):
        generate_b3f3_decisions(
            m5, anchors, atr, instrument_id="EUR/USD.OANDA", config=config
        )


# ---------------------------------------------------------------------------
# Stop / target computation
# ---------------------------------------------------------------------------


def test_short_stop_is_entry_plus_stop_atr_times_atr() -> None:
    m5, anchors, atr = _decision_frame([_m5_bar(1.10, 1.1040, 1.0995, 1.1035)])
    config = B3F3Config(Decimal("2.0"), Decimal("1.0"), "OPEN_ANCHOR")
    decisions = generate_b3f3_decisions(
        m5, anchors, atr, instrument_id="EUR/USD.OANDA", config=config
    )
    # entry = 1.1035, stop = entry + 1.0 * 0.0010
    assert decisions[0].stop_price == pytest.approx(1.1045)


def test_long_stop_is_entry_minus_stop_atr_times_atr() -> None:
    m5, anchors, atr = _decision_frame([_m5_bar(1.10, 1.1005, 1.0960, 1.0965)])
    config = B3F3Config(Decimal("2.0"), Decimal("1.0"), "OPEN_ANCHOR")
    decisions = generate_b3f3_decisions(
        m5, anchors, atr, instrument_id="EUR/USD.OANDA", config=config
    )
    # entry = 1.0965, stop = entry - 1.0 * 0.0010
    assert decisions[0].stop_price == pytest.approx(1.0955)


def test_open_anchor_target_equals_opening_price() -> None:
    m5, anchors, atr = _decision_frame([_m5_bar(1.10, 1.1040, 1.0995, 1.1035)])
    config = B3F3Config(Decimal("2.0"), Decimal("0.75"), "OPEN_ANCHOR")
    decisions = generate_b3f3_decisions(
        m5, anchors, atr, instrument_id="EUR/USD.OANDA", config=config
    )
    assert decisions[0].target_price == pytest.approx(1.1000)


def test_half_reversion_target_short() -> None:
    m5, anchors, atr = _decision_frame([_m5_bar(1.10, 1.1040, 1.0995, 1.1035)])
    config = B3F3Config(Decimal("2.0"), Decimal("0.75"), "HALF_REVERSION")
    decisions = generate_b3f3_decisions(
        m5, anchors, atr, instrument_id="EUR/USD.OANDA", config=config
    )
    # short target = entry - 0.5*(entry - anchor) = 1.1035 - 0.5*0.0035
    assert decisions[0].target_price == pytest.approx(1.10175)


def test_half_reversion_target_long() -> None:
    m5, anchors, atr = _decision_frame([_m5_bar(1.10, 1.1005, 1.0960, 1.0965)])
    config = B3F3Config(Decimal("2.0"), Decimal("0.75"), "HALF_REVERSION")
    decisions = generate_b3f3_decisions(
        m5, anchors, atr, instrument_id="EUR/USD.OANDA", config=config
    )
    # long target = entry + 0.5*(anchor - entry) = 1.0965 + 0.5*0.0035
    assert decisions[0].target_price == pytest.approx(1.09825)


def test_stop_and_target_fixed_at_signal_time_not_recomputed() -> None:
    m5, anchors, atr = _decision_frame([_m5_bar(1.10, 1.1040, 1.0995, 1.1035)])
    config = B3F3Config(Decimal("2.0"), Decimal("0.75"), "OPEN_ANCHOR")
    decisions = generate_b3f3_decisions(
        m5, anchors, atr, instrument_id="EUR/USD.OANDA", config=config
    )
    intent = decisions[0]
    assert intent.stop_price == pytest.approx(1.1035 + 0.75 * 0.0010)
    assert intent.target_price == pytest.approx(1.1000)
