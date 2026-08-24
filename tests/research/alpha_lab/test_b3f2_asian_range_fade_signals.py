from __future__ import annotations

import datetime as dt
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from ftmoquant.research.alpha_lab.b3f2_asian_range_fade_signals import (
    AsianRange,
    B3F2Config,
    B3F2SignalError,
    build_b3f2_grid,
    compute_asian_ranges,
    derive_m15_mid_ohlc,
    generate_b3f2_decisions,
    is_in_asian_range_window,
    is_in_entry_window,
    london_local_date,
    time_exit_boundary_utc,
)


def _m1(prices: list[float], *, start: str) -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(prices), freq="min", tz="UTC")
    return pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices}, index=idx
    )


def _m15_bar(open_: float, high: float, low: float, close: float) -> dict[str, float]:
    return {"open": open_, "high": high, "low": low, "close": close}


def _m15_frame(rows: list[dict[str, float]], *, start: str) -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(rows), freq="15min", tz="UTC")
    return pd.DataFrame(rows, index=idx)


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------


def test_grid_has_exactly_12_configs() -> None:
    grid = build_b3f2_grid()
    assert len(grid) == 12
    assert len({c.config_id for c in grid}) == 12


def test_config_rejects_non_frozen_values() -> None:
    with pytest.raises(B3F2SignalError):
        B3F2Config(Decimal("0.15"), Decimal("0.05"), "MID")
    with pytest.raises(B3F2SignalError):
        B3F2Config(Decimal("0.05"), Decimal("0.15"), "MID")
    with pytest.raises(B3F2SignalError):
        B3F2Config(Decimal("0.05"), Decimal("0.05"), "SHALLOW")


# ---------------------------------------------------------------------------
# Session windows / DST
# ---------------------------------------------------------------------------


def test_asian_range_window_boundaries() -> None:
    winter_date = "2024-01-15"  # GMT (UTC+0)
    assert is_in_asian_range_window(pd.Timestamp(f"{winter_date}T00:00:00Z"))
    assert is_in_asian_range_window(pd.Timestamp(f"{winter_date}T05:59:00Z"))
    assert not is_in_asian_range_window(pd.Timestamp(f"{winter_date}T06:00:00Z"))


def test_entry_window_is_dst_safe_across_bst_and_gmt() -> None:
    # 07:00 London local in BST (UTC+1, e.g. July) is 06:00 UTC.
    summer_ts = pd.Timestamp("2024-07-15T06:00:00Z")
    assert is_in_entry_window(summer_ts)
    # 07:00 London local in GMT (UTC+0, e.g. January) is 07:00 UTC.
    winter_ts = pd.Timestamp("2024-01-15T07:00:00Z")
    assert is_in_entry_window(winter_ts)
    # The SAME UTC instant (07:00 UTC) is NOT in the entry window in July
    # (that's 08:00 BST local, past the 10:00 window's start but let's
    # pick an instant clearly outside): 06:00 UTC in January is 06:00
    # London local -- before the entry window opens.
    assert not is_in_entry_window(pd.Timestamp("2024-01-15T06:00:00Z"))


def test_time_exit_boundary_is_dst_safe() -> None:
    winter_boundary = time_exit_boundary_utc(dt.date(2024, 1, 15))
    summer_boundary = time_exit_boundary_utc(dt.date(2024, 7, 15))
    assert winter_boundary.hour == 16  # GMT: 16:00 local == 16:00 UTC
    assert summer_boundary.hour == 15  # BST: 16:00 local == 15:00 UTC


def test_london_local_date_matches_civil_calendar() -> None:
    # 23:30 UTC on Jan 15 is 23:30 GMT -- still Jan 15 in London.
    assert london_local_date(pd.Timestamp("2024-01-15T23:30:00Z")) == dt.date(
        2024, 1, 15
    )
    # 23:30 UTC on Jul 15 is 00:30 BST the next day in London.
    assert london_local_date(pd.Timestamp("2024-07-15T23:30:00Z")) == dt.date(
        2024, 7, 16
    )


# ---------------------------------------------------------------------------
# M15 derivation (causality)
# ---------------------------------------------------------------------------


def test_m15_derivation_uses_only_completed_windows() -> None:
    n = 45  # bars at :01..:45 -> exactly 3 complete 15-bar buckets (:15,:30,:45)
    prices = [1.10 + 0.0001 * i for i in range(n)]
    bid = _m1(prices, start="2024-01-01T00:01:00Z")
    ask = _m1([p + 0.0002 for p in prices], start="2024-01-01T00:01:00Z")
    m15 = derive_m15_mid_ohlc(bid, ask)
    assert len(m15) == 3


def test_m15_derivation_drops_incomplete_trailing_window() -> None:
    n = 20  # 1 complete window (bars 1-15) + 5 leftover bars
    prices = [1.10] * n
    bid = _m1(prices, start="2024-01-01T00:01:00Z")
    ask = _m1([p + 0.0002 for p in prices], start="2024-01-01T00:01:00Z")
    m15 = derive_m15_mid_ohlc(bid, ask)
    assert len(m15) == 1


def test_m15_derivation_mid_price_matches_established_convention() -> None:
    n = 15
    bid_prices = [1.1000] * n
    ask_prices = [1.1002] * n
    bid = _m1(bid_prices, start="2024-01-01T00:01:00Z")
    ask = _m1(ask_prices, start="2024-01-01T00:01:00Z")
    m15 = derive_m15_mid_ohlc(bid, ask)
    assert m15["close"].iloc[0] == pytest.approx(1.1001)


def test_m15_future_append_cannot_alter_historical_bars() -> None:
    n = 45
    rng = np.random.default_rng(0)
    prices = 1.10 + np.cumsum(rng.normal(scale=0.0001, size=n))
    bid = _m1(list(prices), start="2024-01-01T00:01:00Z")
    ask = _m1(list(prices + 0.0002), start="2024-01-01T00:01:00Z")
    short_m15 = derive_m15_mid_ohlc(bid.iloc[:30], ask.iloc[:30])

    extra = 15
    more_prices = np.concatenate(
        [prices, prices[-1] + np.cumsum(rng.normal(scale=0.0001, size=extra))]
    )
    full_bid = _m1(list(more_prices), start="2024-01-01T00:01:00Z")
    full_ask = _m1(list(more_prices + 0.0002), start="2024-01-01T00:01:00Z")
    full_m15 = derive_m15_mid_ohlc(full_bid, full_ask)

    pd.testing.assert_frame_equal(short_m15, full_m15.iloc[: len(short_m15)])


def test_m15_rejects_mismatched_index() -> None:
    bid = _m1([1.1] * 15, start="2024-01-01T00:01:00Z")
    ask = _m1([1.1] * 15, start="2024-01-01T00:02:00Z")
    with pytest.raises(B3F2SignalError):
        derive_m15_mid_ohlc(bid, ask)


# ---------------------------------------------------------------------------
# Asian range construction
# ---------------------------------------------------------------------------


def test_asian_range_uses_only_completed_asian_window_bars() -> None:
    # 24 bars spanning 00:00-06:00 (Asian window, 24 bars of 15min = 6h)
    # plus bars after 06:00 that must NOT affect the range.
    asian_rows = [_m15_bar(1.10, 1.101, 1.099, 1.10) for _ in range(24)]
    asian_rows[5] = _m15_bar(1.10, 1.150, 1.099, 1.10)  # a spike inside the window
    post_rows = [_m15_bar(1.10, 9.999, 0.001, 1.10) for _ in range(4)]  # outside window
    m15 = _m15_frame(asian_rows + post_rows, start="2024-01-01T00:00:00Z")
    ranges = compute_asian_ranges(m15)
    r = ranges[dt.date(2024, 1, 1)]
    assert r.asian_high == pytest.approx(1.150)
    assert r.asian_low == pytest.approx(1.099)  # NOT the post-window spike (0.001)


def test_asian_range_frozen_before_entry_window() -> None:
    """A range computed from ONLY the Asian-window bars must be identical
    regardless of what happens later in the entry window -- range
    construction must never look at entry-window bars."""

    asian_rows = [_m15_bar(1.10, 1.101, 1.099, 1.10) for _ in range(24)]
    m15_a = _m15_frame(
        asian_rows + [_m15_bar(1.10, 1.10, 1.10, 1.10)] * 4,
        start="2024-01-01T00:00:00Z",
    )
    m15_b = _m15_frame(
        asian_rows + [_m15_bar(2.0, 5.0, 0.5, 2.0)] * 4, start="2024-01-01T00:00:00Z"
    )
    ranges_a = compute_asian_ranges(m15_a)
    ranges_b = compute_asian_ranges(m15_b)
    assert ranges_a[dt.date(2024, 1, 1)] == ranges_b[dt.date(2024, 1, 1)]


def test_zero_width_range_produces_no_entry_for_that_date() -> None:
    asian_rows = [_m15_bar(1.10, 1.10, 1.10, 1.10) for _ in range(24)]
    m15 = _m15_frame(asian_rows, start="2024-01-01T00:00:00Z")
    ranges = compute_asian_ranges(m15)
    assert dt.date(2024, 1, 1) not in ranges


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------


def _range_bar_frame(
    bars: list[dict[str, float]],
) -> tuple[pd.DataFrame, dict[dt.date, AsianRange]]:
    m15 = _m15_frame(bars, start="2024-01-01T07:00:00Z")
    ranges = {
        dt.date(2024, 1, 1): AsianRange(
            dt.date(2024, 1, 1), asian_high=1.1010, asian_low=1.0990
        )
    }
    return m15, ranges


def test_short_probe_and_reentry_triggers_short() -> None:
    m15, ranges = _range_bar_frame([_m15_bar(1.1000, 1.1020, 1.0999, 1.1005)])
    config = B3F2Config(Decimal("0.05"), Decimal("0.05"), "MID")
    decisions = generate_b3f2_decisions(
        m15, ranges, instrument_id="EUR/USD.OANDA", config=config
    )
    assert len(decisions) == 1
    assert decisions[0].direction == -1


def test_long_probe_and_reentry_triggers_long() -> None:
    m15, ranges = _range_bar_frame([_m15_bar(1.1000, 1.1001, 1.0980, 1.0995)])
    config = B3F2Config(Decimal("0.05"), Decimal("0.05"), "MID")
    decisions = generate_b3f2_decisions(
        m15, ranges, instrument_id="EUR/USD.OANDA", config=config
    )
    assert len(decisions) == 1
    assert decisions[0].direction == 1


def test_excursion_below_threshold_is_rejected() -> None:
    # Excursion = (1.1011 - 1.1010) / 0.0020 = 0.05 exactly; require > 0.10.
    m15, ranges = _range_bar_frame([_m15_bar(1.1000, 1.1011, 1.0999, 1.1005)])
    config = B3F2Config(Decimal("0.10"), Decimal("0.05"), "MID")
    decisions = generate_b3f2_decisions(
        m15, ranges, instrument_id="EUR/USD.OANDA", config=config
    )
    assert decisions == ()


def test_dual_direction_bar_is_skipped() -> None:
    # high beyond asian_high AND low beyond asian_low on the SAME bar.
    m15, ranges = _range_bar_frame([_m15_bar(1.1000, 1.1050, 1.0950, 1.1000)])
    config = B3F2Config(Decimal("0.05"), Decimal("0.05"), "MID")
    decisions = generate_b3f2_decisions(
        m15, ranges, instrument_id="EUR/USD.OANDA", config=config
    )
    assert decisions == ()


def test_one_attempt_per_direction_per_day() -> None:
    m15, ranges = _range_bar_frame(
        [
            _m15_bar(1.1000, 1.1020, 1.0999, 1.1005),  # short #1
            _m15_bar(
                1.1000, 1.1025, 1.0999, 1.1005
            ),  # short #2 -- same day, must be skipped
        ]
    )
    config = B3F2Config(Decimal("0.05"), Decimal("0.05"), "MID")
    decisions = generate_b3f2_decisions(
        m15, ranges, instrument_id="EUR/USD.OANDA", config=config
    )
    assert len(decisions) == 1


def test_entry_outside_window_is_rejected() -> None:
    m15 = _m15_frame(
        [_m15_bar(1.1000, 1.1020, 1.0999, 1.1005)], start="2024-01-01T11:00:00Z"
    )  # 11:00 is outside the 07:00-10:00 entry window
    ranges = {
        dt.date(2024, 1, 1): AsianRange(
            dt.date(2024, 1, 1), asian_high=1.1010, asian_low=1.0990
        )
    }
    config = B3F2Config(Decimal("0.05"), Decimal("0.05"), "MID")
    decisions = generate_b3f2_decisions(
        m15, ranges, instrument_id="EUR/USD.OANDA", config=config
    )
    assert decisions == ()


def test_missing_asian_range_for_date_produces_no_signal() -> None:
    m15 = _m15_frame(
        [_m15_bar(1.1000, 1.1020, 1.0999, 1.1005)], start="2024-01-01T07:00:00Z"
    )
    config = B3F2Config(Decimal("0.05"), Decimal("0.05"), "MID")
    decisions = generate_b3f2_decisions(
        m15, {}, instrument_id="EUR/USD.OANDA", config=config
    )
    assert decisions == ()


# ---------------------------------------------------------------------------
# Stop / target computation
# ---------------------------------------------------------------------------


def test_stop_buffer_calculation_exact() -> None:
    m15, ranges = _range_bar_frame([_m15_bar(1.1000, 1.1020, 1.0999, 1.1005)])
    config = B3F2Config(Decimal("0.05"), Decimal("0.10"), "MID")
    decisions = generate_b3f2_decisions(
        m15, ranges, instrument_id="EUR/USD.OANDA", config=config
    )
    # short: stop = probe_high + stop_buffer * range_width = 1.1020 + 0.10*0.0020
    assert decisions[0].stop_price == pytest.approx(1.1022)


def test_midpoint_target() -> None:
    m15, ranges = _range_bar_frame([_m15_bar(1.1000, 1.1020, 1.0999, 1.1005)])
    config = B3F2Config(Decimal("0.05"), Decimal("0.05"), "MID")
    decisions = generate_b3f2_decisions(
        m15, ranges, instrument_id="EUR/USD.OANDA", config=config
    )
    assert decisions[0].target_price == pytest.approx(1.1000)  # (1.1010+1.0990)/2


def test_deep_reversion_target() -> None:
    m15, ranges = _range_bar_frame([_m15_bar(1.1000, 1.1020, 1.0999, 1.1005)])
    config = B3F2Config(Decimal("0.05"), Decimal("0.05"), "DEEP_REVERSION")
    decisions = generate_b3f2_decisions(
        m15, ranges, instrument_id="EUR/USD.OANDA", config=config
    )
    # short target = asian_low + 0.25*range_width = 1.0990 + 0.25*0.0020 = 1.0995
    assert decisions[0].target_price == pytest.approx(1.0995)
