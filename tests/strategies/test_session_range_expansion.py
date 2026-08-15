from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from typing import Any, cast
from zoneinfo import ZoneInfo

import pytest

from ftmoquant.research.session_range_expansion_spec import (
    load_session_range_expansion_spec,
)
from ftmoquant.research.stage_g import (
    DEVELOPMENT_END_EXCLUSIVE,
    DEVELOPMENT_START,
    FROZEN_INSTRUMENT_IDS,
    DevelopmentResearchContext,
    InstrumentBarObservation,
    ResearchPartition,
    StageGValidationError,
    SynchronizedClockFrame,
    frozen_development_folds,
)
from ftmoquant.strategies.session_range_expansion import (
    SessionRangeExpansionDevelopmentFold,
    SessionRangeExpansionStateMachine,
)
from ftmoquant.strategies.ts_momentum import RawDirectionalTarget

EURUSD, GBPUSD = FROZEN_INSTRUMENT_IDS
LONDON = ZoneInfo("Europe/London")


@pytest.mark.parametrize(
    ("day", "expected_utc_hour"),
    [(date(2020, 1, 6), 8), (date(2020, 7, 6), 7)],
)
def test_london_range_boundaries_are_dst_aware(
    day: date, expected_utc_hour: int
) -> None:
    machine = _machine()
    _feed_complete_range(machine, day, Decimal("1.10000"))
    breakout = _frame_at_local(day, time(8), Decimal("1.20000"))

    signals = machine.on_frame(breakout)

    assert len(signals) == 2
    assert all(item.target is RawDirectionalTarget.LONG for item in signals)
    assert signals[0].signal_information_time_utc.hour == expected_utc_hour
    assert signals[0].signal_information_time_utc.astimezone(LONDON).time() == time(8)


@pytest.mark.parametrize(
    ("range_price", "breakout_price", "target"),
    [
        ("1.10000", "1.20000", RawDirectionalTarget.LONG),
        ("1.10000", "1.00000", RawDirectionalTarget.SHORT),
    ],
)
def test_first_breakout_maps_to_raw_direction(
    range_price: str, breakout_price: str, target: RawDirectionalTarget
) -> None:
    machine = _machine()
    day = date(2020, 7, 6)
    _feed_complete_range(machine, day, Decimal(range_price))

    signals = machine.on_frame(_frame_at_local(day, time(8), Decimal(breakout_price)))

    assert [item.target for item in signals] == [target, target]


def test_no_breakout_remains_flat_and_one_entry_is_consumed() -> None:
    machine = _machine()
    day = date(2020, 7, 6)
    _feed_complete_range(machine, day, Decimal("1.10000"))

    assert machine.on_frame(_frame_at_local(day, time(8), Decimal("1.10000"))) == ()
    assert machine.on_frame(_frame_at_local(day, time(9), Decimal("1.20000")))
    assert machine.on_frame(_frame_at_local(day, time(9, 1), Decimal("1.00000"))) == ()


def test_target_executes_only_strictly_later_and_exits_after_1600() -> None:
    machine = _machine()
    day = date(2020, 7, 6)
    _feed_complete_range(machine, day, Decimal("1.10000"))
    breakout = _frame_at_local(day, time(8), Decimal("1.20000"))
    signals = machine.on_frame(breakout)

    assert machine.on_execution_frame(breakout) == ()
    entry_frame = _frame_at_local(day, time(8, 1), Decimal("1.20000"))
    entry = machine.on_execution_frame(entry_frame)
    assert [item.target for item in entry] == [RawDirectionalTarget.LONG] * 2
    assert machine.active_targets == (
        (EURUSD, RawDirectionalTarget.LONG),
        (GBPUSD, RawDirectionalTarget.LONG),
    )

    exit_frame = _frame_at_local(day, time(16), Decimal("1.20000"))
    exit_signals = machine.on_frame(exit_frame)
    assert [item.target for item in exit_signals] == [RawDirectionalTarget.FLAT] * 2
    assert machine.on_execution_frame(exit_frame) == ()
    exits = machine.on_execution_frame(
        _frame_at_local(day, time(16, 1), Decimal("1.20000"))
    )
    assert [item.target for item in exits] == [RawDirectionalTarget.FLAT] * 2
    assert machine.active_targets == ()
    assert all(
        item.execution_information_time_utc
        > signals[0].signal_information_time_utc
        for item in entry
    )


def test_incomplete_range_blocks_only_the_affected_instrument() -> None:
    machine = _machine()
    day = date(2020, 7, 6)
    start = datetime.combine(day, time(0), tzinfo=LONDON)
    for index in range(8 * 60):
        missing = GBPUSD if index == 100 else None
        machine.on_frame(
            _frame(start + timedelta(minutes=index), Decimal("1.10000"), missing)
        )

    signals = machine.on_frame(_frame_at_local(day, time(8), Decimal("1.20000")))

    assert [(item.instrument_id, item.target) for item in signals] == [
        (EURUSD, RawDirectionalTarget.LONG)
    ]


def test_invalid_range_minute_blocks_only_the_affected_instrument() -> None:
    machine = _machine()
    day = date(2020, 7, 6)
    start = datetime.combine(day, time(0), tzinfo=LONDON)
    for index in range(8 * 60):
        frame = _frame(start + timedelta(minutes=index), Decimal("1.10000"))
        if index == 100:
            frame = _invalid_frame(frame, GBPUSD)
        machine.on_frame(frame)

    signals = machine.on_frame(_frame_at_local(day, time(8), Decimal("1.20000")))

    assert [(item.instrument_id, item.target) for item in signals] == [
        (EURUSD, RawDirectionalTarget.LONG)
    ]


def test_deterministic_independent_outputs_and_fold_reset() -> None:
    day = date(2020, 7, 6)

    def run() -> tuple[object, ...]:
        machine = _machine()
        _feed_complete_range(machine, day, Decimal("1.10000"))
        return (
            machine.on_frame(_frame_at_local(day, time(8), Decimal("1.20000"))),
            machine.on_execution_frame(
                _frame_at_local(day, time(8, 1), Decimal("1.20000"))
            ),
        )

    assert run() == run()
    context = DevelopmentResearchContext(
        universe=cast(Any, None),
        artifacts=(),
        start_utc=DEVELOPMENT_START,
        end_exclusive_utc=DEVELOPMENT_END_EXCLUSIVE,
    )
    first = SessionRangeExpansionDevelopmentFold(
        context,
        frozen_development_folds().folds[0],
        load_session_range_expansion_spec(),
    )
    start = datetime.combine(day, time(0), tzinfo=LONDON)
    for index in range(8 * 60):
        first.on_frame(_frame(start + timedelta(minutes=index), Decimal("1.10000")))
    first.on_frame(_frame_at_local(day, time(8), Decimal("1.20000")))
    first.on_execution_frame(_frame_at_local(day, time(8, 1), Decimal("1.20000")))
    assert first.state.active_targets
    second = SessionRangeExpansionDevelopmentFold(
        context,
        frozen_development_folds().folds[1],
        load_session_range_expansion_spec(),
    )
    assert second.state.active_targets == ()
    with pytest.raises(StageGValidationError, match="validation is LOCKED"):
        first.on_frame(
            _frame_at_local(day, time(8), Decimal("1.20000")),
            partition=ResearchPartition.VALIDATION,
        )


def _machine() -> SessionRangeExpansionStateMachine:
    return SessionRangeExpansionStateMachine(load_session_range_expansion_spec())


def _feed_complete_range(
    machine: SessionRangeExpansionStateMachine, day: date, price: Decimal
) -> None:
    start = datetime.combine(day, time(0), tzinfo=LONDON)
    for index in range(8 * 60):
        assert machine.on_frame(_frame(start + timedelta(minutes=index), price)) == ()


def _frame_at_local(
    day: date, local_time: time, price: Decimal
) -> SynchronizedClockFrame:
    return _frame(datetime.combine(day, local_time, tzinfo=LONDON), price)


def _frame(
    information_london: datetime, price: Decimal, missing: str | None = None
) -> SynchronizedClockFrame:
    information = information_london.astimezone(UTC)
    event = information - timedelta(minutes=1)
    observations = tuple(
        None
        if instrument_id == missing
        else InstrumentBarObservation(
            instrument_id=instrument_id,
            timestamp_utc=event,
            available_at_utc=information,
            bid=price - Decimal("0.00010"),
            ask=price + Decimal("0.00010"),
        )
        for instrument_id in FROZEN_INSTRUMENT_IDS
    )
    return SynchronizedClockFrame(
        timestamp_utc=event,
        available_at_utc=information,
        observations=observations,
        tradable=missing is None,
    )


def _invalid_frame(
    frame: SynchronizedClockFrame, instrument_id: str
) -> SynchronizedClockFrame:
    observations = tuple(
        InstrumentBarObservation(
            instrument_id=item.instrument_id,
            timestamp_utc=item.timestamp_utc,
            available_at_utc=item.available_at_utc,
            bid=Decimal("NaN") if item.instrument_id == instrument_id else item.bid,
            ask=item.ask,
        )
        if item is not None
        else None
        for item in frame.observations
    )
    return SynchronizedClockFrame(
        timestamp_utc=frame.timestamp_utc,
        available_at_utc=frame.available_at_utc,
        observations=observations,
        tradable=frame.tradable,
    )
