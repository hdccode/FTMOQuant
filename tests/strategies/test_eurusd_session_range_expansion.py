from __future__ import annotations

from datetime import UTC, datetime, time
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from ftmoquant.strategies.eurusd_session_range_expansion import (
    EurusdSessionRangeExpansionParameters,
    EurusdSessionRangeExpansionState,
    EurusdSessionRangeExpansionValidationError,
)
from ftmoquant.strategies.ts_momentum import RawDirectionalTarget

_MIN = 60_000_000_000
_LONDON = ZoneInfo("Europe/London")


def _params(
    breakout: time = time(12, 0), exit_: time = time(16, 0)
) -> EurusdSessionRangeExpansionParameters:
    return EurusdSessionRangeExpansionParameters(
        breakout_window_end=breakout, scheduled_exit=exit_
    )


def _ns(local_dt: datetime) -> int:
    return int(local_dt.astimezone(UTC).timestamp() * 1_000_000_000)


def _build_range(
    state: EurusdSessionRangeExpansionState,
    day: datetime,
    price: Decimal,
    *,
    count: int = 480,
) -> int:
    """Feed exactly ``count`` consecutive range-window minute closes.

    Each close's *information* time (bar close) is what carries the local
    time-of-day used for range membership; its *event* time (bar open) is
    one minute earlier, matching real Nautilus 1-minute bar semantics.
    """
    info = _ns(day)
    for _ in range(count):
        state.on_minute_close(info - _MIN, info, price)
        info += _MIN
    return info


def test_parameters_reject_invalid_ordering() -> None:
    with pytest.raises(EurusdSessionRangeExpansionValidationError):
        EurusdSessionRangeExpansionParameters(time(7, 0), time(16, 0))
    with pytest.raises(EurusdSessionRangeExpansionValidationError):
        EurusdSessionRangeExpansionParameters(time(12, 0), time(11, 0))
    with pytest.raises(EurusdSessionRangeExpansionValidationError):
        EurusdSessionRangeExpansionParameters(time(12, 0), time(12, 0))


def test_exact_range_construction_requires_480_consecutive_closes() -> None:
    state = EurusdSessionRangeExpansionState(_params())
    day = datetime(2020, 6, 15, 0, 0, tzinfo=_LONDON)
    info = _build_range(state, day, Decimal("1.10000"))
    assert state._range is not None
    assert state._range.count == 480
    assert state._range.range_is_valid is True
    # First eligible breakout-window frame is exactly local 08:00.
    local = datetime.fromtimestamp(info / 1e9, tz=UTC).astimezone(_LONDON)
    assert local.time() == time(8, 0)


def test_incomplete_range_from_a_gap_produces_no_trade() -> None:
    state = EurusdSessionRangeExpansionState(_params())
    day = datetime(2020, 6, 15, 0, 0, tzinfo=_LONDON)
    info = _ns(day)
    for i in range(479):
        state.on_minute_close(info - _MIN, info, Decimal("1.10000"))
        info += _MIN
        if i == 200:  # skip one minute -> gap
            info += _MIN
    assert state._range is not None
    assert state._range.complete is False
    # Breakout window: a huge move must not trigger without a valid range.
    signal = state.on_minute_close(info - _MIN, info, Decimal("1.30000"))
    assert signal is None


def test_invalid_price_during_range_window_invalidates_that_day() -> None:
    state = EurusdSessionRangeExpansionState(_params())
    day = datetime(2020, 6, 15, 0, 0, tzinfo=_LONDON)
    info = _ns(day)
    for i in range(480):
        price = Decimal("-1") if i == 100 else Decimal("1.10000")
        state.on_minute_close(info - _MIN, info, price)
        info += _MIN
    assert state._range is not None
    assert state._range.range_is_valid is False


def test_first_close_above_high_is_long_below_low_is_short() -> None:
    up = EurusdSessionRangeExpansionState(_params())
    day = datetime(2020, 6, 15, 0, 0, tzinfo=_LONDON)
    info = _build_range(up, day, Decimal("1.10000"))
    signal = up.on_minute_close(info - _MIN, info, Decimal("1.10500"))
    assert signal is not None and signal.target is RawDirectionalTarget.LONG

    down = EurusdSessionRangeExpansionState(_params())
    info = _build_range(down, day, Decimal("1.10000"))
    signal = down.on_minute_close(info - _MIN, info, Decimal("1.09500"))
    assert signal is not None and signal.target is RawDirectionalTarget.SHORT


def test_only_the_first_breakout_of_the_day_is_taken() -> None:
    state = EurusdSessionRangeExpansionState(_params(exit_=time(18, 0)))
    day = datetime(2020, 6, 15, 0, 0, tzinfo=_LONDON)
    info = _build_range(state, day, Decimal("1.10000"))
    first = state.on_minute_close(info - _MIN, info, Decimal("1.10500"))
    assert first is not None
    info += _MIN
    # A second, even bigger breakout the same day must be ignored.
    second = state.on_minute_close(info - _MIN, info, Decimal("1.20000"))
    assert second is None


def test_one_entry_per_london_day_even_across_a_flat_position() -> None:
    state = EurusdSessionRangeExpansionState(_params(breakout=time(11, 0)))
    day = datetime(2020, 6, 15, 0, 0, tzinfo=_LONDON)
    info = _build_range(state, day, Decimal("1.10000"))
    signal = state.on_minute_close(info - _MIN, info, Decimal("1.10500"))
    assert signal is not None
    # Even after the breakout window closes with no second entry, the day's
    # range stays consumed.
    assert state._range is not None
    assert state._range.entry_consumed is True


def test_scheduled_exit_flattens_at_first_eligible_frame_at_or_after_exit_time() -> (
    None
):
    state = EurusdSessionRangeExpansionState(_params(exit_=time(16, 0)))
    day = datetime(2020, 6, 15, 0, 0, tzinfo=_LONDON)
    info = _build_range(state, day, Decimal("1.10000"))
    state.on_minute_close(info - _MIN, info, Decimal("1.10500"))
    entry_signal_info = info
    info += _MIN
    state.on_minute_close(info - _MIN, info, Decimal("1.10500"))
    entry = state.on_execution_frame(info - _MIN, info)
    assert entry is not None
    assert entry.execution_information_ns > entry_signal_info
    after_entry = state.latest_target
    assert after_entry is RawDirectionalTarget.LONG

    exit_info = _ns(day.replace(hour=16, minute=0))
    exit_signal = state.on_minute_close(exit_info - _MIN, exit_info, Decimal("1.10500"))
    assert exit_signal is not None and exit_signal.target is RawDirectionalTarget.FLAT
    info2 = exit_info + _MIN
    state.on_minute_close(info2 - _MIN, info2, Decimal("1.10500"))
    exit_exec = state.on_execution_frame(info2 - _MIN, info2)
    assert exit_exec is not None
    assert exit_exec.execution_information_ns > exit_signal.signal_information_ns
    after_exit = state.latest_target
    assert after_exit is RawDirectionalTarget.FLAT
    assert state.has_open_or_pending_position is False


def test_execution_is_strictly_later_than_signal_never_same_frame() -> None:
    state = EurusdSessionRangeExpansionState(_params())
    day = datetime(2020, 6, 15, 0, 0, tzinfo=_LONDON)
    info = _build_range(state, day, Decimal("1.10000"))
    state.on_minute_close(info - _MIN, info, Decimal("1.10500"))
    # Same-frame execution attempt must not release the pending target.
    assert state.on_execution_frame(info - _MIN, info) is None


def test_warmup_never_emits_a_signal_but_still_builds_the_range() -> None:
    state = EurusdSessionRangeExpansionState(_params())
    day = datetime(2020, 6, 15, 0, 0, tzinfo=_LONDON)
    info = _ns(day)
    for _ in range(480):
        state.warmup_minute_close(info - _MIN, info, Decimal("1.10000"))
        info += _MIN
    assert state._range is not None
    assert state._range.range_is_valid is True
    # A live breakout after warmup still fires using the warmed range.
    signal = state.on_minute_close(info - _MIN, info, Decimal("1.10500"))
    assert signal is not None and signal.target is RawDirectionalTarget.LONG


def test_dst_spring_forward_day_produces_an_incomplete_range() -> None:
    """UK clocks spring forward at 01:00 on 2020-03-29; local [00:00,08:00)
    that day spans only 7 real hours (420 minutes), not 8 (480), so the
    range must be marked incomplete and no trade should ever fire."""
    state = EurusdSessionRangeExpansionState(_params())
    day = datetime(2020, 3, 29, 0, 0, tzinfo=_LONDON)
    info = _ns(day)
    for _ in range(480):
        local = datetime.fromtimestamp(info / 1e9, tz=UTC).astimezone(_LONDON)
        if local.time() >= time(8, 0):
            break
        state.on_minute_close(info - _MIN, info, Decimal("1.10000"))
        info += _MIN
    range_state = state._range
    assert range_state is not None
    assert range_state.session_date == day.date()
    assert range_state.count == 420
    assert range_state.range_is_valid is False


def test_dst_fall_back_day_also_fails_the_exact_count_check() -> None:
    """UK clocks fall back at 02:00 on 2020-10-25; local [00:00,08:00) that
    day spans 9 real hours (540 minutes, with a repeated local hour), so an
    exact-480-consecutive-minute range must not validate. Feed enough real
    minutes (600) to walk all the way past local 08:00 that day."""
    state = EurusdSessionRangeExpansionState(_params())
    day = datetime(2020, 10, 25, 0, 0, tzinfo=_LONDON)
    info = _ns(day)
    for _ in range(600):
        local = datetime.fromtimestamp(info / 1e9, tz=UTC).astimezone(_LONDON)
        if local.time() >= time(8, 0):
            break
        state.on_minute_close(info - _MIN, info, Decimal("1.10000"))
        info += _MIN
    range_state = state._range
    assert range_state is not None
    assert range_state.session_date == day.date()
    assert range_state.count == 540
    assert range_state.range_is_valid is False


def test_reset_clears_all_state_at_a_fold_boundary() -> None:
    state = EurusdSessionRangeExpansionState(_params())
    day = datetime(2020, 6, 15, 0, 0, tzinfo=_LONDON)
    info = _build_range(state, day, Decimal("1.10000"))
    state.on_minute_close(info - _MIN, info, Decimal("1.10500"))
    assert state.has_open_or_pending_position is True

    state.reset()

    assert state.latest_target is RawDirectionalTarget.FLAT
    assert state.has_open_or_pending_position is False
    assert state._range is None


def test_information_times_must_be_strictly_ordered() -> None:
    state = EurusdSessionRangeExpansionState(_params())
    info = _ns(datetime(2020, 6, 15, 0, 0, tzinfo=_LONDON))
    state.on_minute_close(info - _MIN, info, Decimal("1.10000"))
    with pytest.raises(EurusdSessionRangeExpansionValidationError):
        state.on_minute_close(info - _MIN, info, Decimal("1.10001"))
