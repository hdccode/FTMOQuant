from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from ftmoquant.research.leo_gbpusd_spec import load_leo_gbpusd_spec
from ftmoquant.strategies.leo_gbpusd import (
    LeoCompleted15mBar,
    LeoEntry,
    LeoExit,
    LeoExitReason,
    LeoGbpUsdStateMachine,
    LeoNamedSession,
    LeoSignal,
)
from ftmoquant.strategies.trend_pullback import Direction, PriceBar

LONDON = ZoneInfo("Europe/London")
DAY = date(2020, 7, 6)


@pytest.mark.parametrize(
    ("direction", "signal_values", "entry_close", "stop", "target"),
    [
        (
            Direction.SHORT,
            {"high": "1.3010", "low": "1.2980", "close": "1.2990"},
            "1.2980",
            "1.3010",
            "1.2890",
        ),
        (
            Direction.LONG,
            {"high": "1.2920", "low": "1.2890", "close": "1.2910"},
            "1.2920",
            "1.2890",
            "1.3010",
        ),
    ],
)
def test_upper_and_lower_rejections_bind_actual_entry_to_exact_3r(
    direction: Direction,
    signal_values: dict[str, str],
    entry_close: str,
    stop: str,
    target: str,
) -> None:
    machine = _with_complete_asia()

    signal = _only(machine.on_bar(_bar(DAY, time(9, 15), **signal_values)), LeoSignal)
    assert signal.direction is direction
    assert machine.active_entry is None

    entry = _only(machine.on_bar(_bar(DAY, time(9, 30), close=entry_close)), LeoEntry)
    assert entry.direction is direction
    assert entry.entry_information_time_utc > signal.signal_information_time_utc
    assert entry.stop_price == Decimal(stop)
    assert entry.target_price == Decimal(target)


def test_close_that_does_not_return_strictly_inside_is_not_a_signal() -> None:
    machine = _with_complete_asia()

    assert machine.on_bar(_bar(DAY, time(9), high="1.3010", close="1.3000")) == ()
    assert machine.on_bar(_bar(DAY, time(9, 15), low="1.2890", close="1.2900")) == ()


def test_invalid_or_zero_stop_distance_skips_the_pending_trade() -> None:
    machine = _with_complete_asia()
    signal = _only(
        machine.on_bar(_bar(DAY, time(9, 15), low="1.2890", close="1.2910")),
        LeoSignal,
    )

    assert signal.direction is Direction.LONG
    assert machine.on_bar(_bar(DAY, time(9, 30), close="1.2890")) == ()
    assert machine.active_entry is None
    assert machine.pending_signal is None


def test_entry_uses_the_actual_g0_7_bid_or_ask_price_for_3r() -> None:
    short_machine = _with_complete_asia()
    short_machine.on_bar(_bar(DAY, time(9), high="1.3010", close="1.2990"))
    short = _only(
        short_machine.on_bar(
            _bar(DAY, time(9, 15), close="1.2980", bid_adjust="-0.0002")
        ),
        LeoEntry,
    )
    assert short.entry_price == Decimal("1.2978")
    assert short.target_price == Decimal("1.2882")

    long_machine = _with_complete_asia()
    long_machine.on_bar(_bar(DAY, time(9), low="1.2890", close="1.2910"))
    long = _only(
        long_machine.on_bar(
            _bar(DAY, time(9, 15), close="1.2920", ask_adjust="0.0002")
        ),
        LeoEntry,
    )
    assert long.entry_price == Decimal("1.2922")
    assert long.target_price == Decimal("1.3018")


def test_london_and_new_york_references_are_completed_without_lookahead() -> None:
    machine = _with_complete_asia()
    assert machine.asia_reference == (Decimal("1.3000"), Decimal("1.2900"))
    _feed_london_reference(machine, high="1.3500", low="1.3200", close="1.3300")
    assert machine.london_reference == (Decimal("1.3500"), Decimal("1.3200"))

    # The 13:00 bar is outside the frozen reference window and must not alter it.
    assert machine.on_bar(_bar(DAY, time(13), high="1.4000", close="1.3900")) == ()
    assert machine.london_reference == (Decimal("1.3500"), Decimal("1.3200"))
    assert machine.on_bar(_bar(DAY, time(13, 45), high="1.3600", close="1.3400")) == ()
    signal = _only(
        machine.on_bar(_bar(DAY, time(14), high="1.3600", close="1.3400")),
        LeoSignal,
    )
    assert signal.named_session is LeoNamedSession.NEW_YORK
    assert signal.reference_high == Decimal("1.3500")
    assert machine.on_bar(_bar(DAY, time(16), high="1.3600", close="1.3400")) == ()


@pytest.mark.parametrize(
    ("day", "expected_utc_hour"),
    [(date(2020, 1, 6), 9), (date(2020, 7, 6), 8)],
)
def test_london_dst_transitions_keep_local_windows_stable(
    day: date, expected_utc_hour: int
) -> None:
    machine = _with_complete_asia(day)
    signal = _only(
        machine.on_bar(_bar(day, time(9, 15), high="1.3010", close="1.2990")),
        LeoSignal,
    )

    assert signal.signal_end_time_utc.hour == expected_utc_hour
    assert signal.signal_end_time_utc.astimezone(LONDON).time() == time(9, 15)


def test_exact_entry_boundaries_and_one_entry_per_named_session_per_day() -> None:
    machine = _with_complete_asia()
    assert machine.on_bar(_bar(DAY, time(8), high="1.3010", close="1.2990")) == ()
    signal = _only(
        machine.on_bar(_bar(DAY, time(9), high="1.3010", close="1.2990")), LeoSignal
    )
    assert signal.named_session is LeoNamedSession.LONDON
    entry = _only(machine.on_bar(_bar(DAY, time(9, 15), close="1.2980")), LeoEntry)
    assert entry.named_session is LeoNamedSession.LONDON
    stop = _only(
        machine.on_bar(_bar(DAY, time(9, 30), high="1.3020", close="1.3010")),
        LeoExit,
    )
    assert stop.reason is LeoExitReason.STOP_LOSS
    assert machine.on_bar(_bar(DAY, time(10), high="1.3010", close="1.2990")) == ()
    assert machine.on_bar(_bar(DAY, time(11), high="1.3010", close="1.2990")) == ()


def test_active_position_blocks_an_overlapping_setup() -> None:
    machine = _with_complete_asia()
    machine.on_bar(_bar(DAY, time(9), low="1.2890", close="1.2910"))
    machine.on_bar(_bar(DAY, time(9, 15), close="1.2920"))

    assert machine.active_entry is not None
    assert machine.on_bar(_bar(DAY, time(9, 30), high="1.3005", close="1.2990")) == ()


def test_position_is_force_closed_at_named_session_end_when_no_barrier_hit() -> None:
    machine = _with_complete_asia()
    machine.on_bar(_bar(DAY, time(9), high="1.3010", close="1.2990"))
    machine.on_bar(_bar(DAY, time(9, 15), close="1.2980"))

    exit_action = _only(machine.on_bar(_bar(DAY, time(11), close="1.2985")), LeoExit)
    assert exit_action.reason is LeoExitReason.SESSION_WINDOW_END
    assert exit_action.exit_price == Decimal("1.2985")
    assert machine.active_entry is None


@pytest.mark.parametrize(
    ("high", "low", "expected"),
    [
        ("1.3020", "1.2970", LeoExitReason.STOP_LOSS),
        ("1.2980", "1.2880", LeoExitReason.TAKE_PROFIT),
        ("1.3020", "1.2880", LeoExitReason.STOP_LOSS),
    ],
)
def test_stop_target_ordering_is_stop_first_when_both_are_touched(
    high: str, low: str, expected: LeoExitReason
) -> None:
    machine = _entered_short()

    exit_action = _only(
        machine.on_bar(_bar(DAY, time(9, 30), high=high, low=low, close="1.2980")),
        LeoExit,
    )
    assert exit_action.reason is expected


def _with_complete_asia(day: date = DAY) -> LeoGbpUsdStateMachine:
    machine = LeoGbpUsdStateMachine(load_leo_gbpusd_spec())
    start = datetime.combine(day, time(0), tzinfo=LONDON)
    for offset in range(32):
        end = (start + offset * timedelta(minutes=15)).time()
        assert (
            machine.on_bar(_bar(day, end, high="1.3000", low="1.2900", close="1.2950"))
            == ()
        )
    return machine


def _feed_london_reference(
    machine: LeoGbpUsdStateMachine, *, high: str, low: str, close: str
) -> None:
    start = datetime.combine(DAY, time(8), tzinfo=LONDON)
    for offset in range(20):
        end = (start + offset * timedelta(minutes=15)).time()
        assert machine.on_bar(_bar(DAY, end, high=high, low=low, close=close)) == ()


def _entered_short() -> LeoGbpUsdStateMachine:
    machine = _with_complete_asia()
    machine.on_bar(_bar(DAY, time(9), high="1.3010", close="1.2990"))
    entry = _only(machine.on_bar(_bar(DAY, time(9, 15), close="1.2980")), LeoEntry)
    assert entry.stop_price == Decimal("1.3010")
    assert entry.target_price == Decimal("1.2890")
    return machine


def _bar(
    day: date,
    end: time,
    *,
    open: str | None = None,
    high: str | None = None,
    low: str | None = None,
    close: str = "1.2950",
    bid_adjust: str = "0",
    ask_adjust: str = "0",
) -> LeoCompleted15mBar:
    end_utc = datetime.combine(day, end, tzinfo=LONDON).astimezone(UTC)
    close_decimal = Decimal(close)
    open_decimal = close_decimal if open is None else Decimal(open)
    high_decimal = max(open_decimal, close_decimal) if high is None else Decimal(high)
    low_decimal = min(open_decimal, close_decimal) if low is None else Decimal(low)
    values = (open_decimal, high_decimal, low_decimal, close_decimal)
    bid = PriceBar(*(value + Decimal(bid_adjust) for value in values))
    ask = PriceBar(*(value + Decimal(ask_adjust) for value in values))
    return LeoCompleted15mBar(
        instrument_id="GBP/USD.DUKASCOPY",
        start_time_utc=end_utc - timedelta(minutes=15),
        end_time_utc=end_utc,
        available_at_utc=end_utc,
        bid=bid,
        ask=ask,
    )


def _only(actions: tuple[object, ...], expected_type: type[object]) -> object:
    assert len(actions) == 1
    assert isinstance(actions[0], expected_type)
    return actions[0]
