from __future__ import annotations

import math
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from typing import Any, cast
from zoneinfo import ZoneInfo

import pytest
from nautilus_trader.indicators import RateOfChange

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
from ftmoquant.research.ts_momentum_spec import load_ts_momentum_spec
from ftmoquant.strategies.ts_momentum import (
    DailyMidpointClose,
    RawDirectionalTarget,
    TsMomentumDevelopmentFold,
    TsMomentumStateMachine,
    derive_daily_midpoint_closes,
)

EURUSD, GBPUSD = FROZEN_INSTRUMENT_IDS
NY = ZoneInfo("America/New_York")


def test_native_roc_period_253_is_exact_252_observation_log_change() -> None:
    indicator = RateOfChange(253, use_log=True)
    prices = [1.0, 2.0, *([1.25] * 250), 1.5]

    for price in prices[:-1]:
        indicator.update_raw(price)
    assert indicator.initialized is False

    indicator.update_raw(prices[-1])
    oracle = math.log(prices[-1] / prices[0])

    assert indicator.initialized is True
    assert indicator.value.hex() == oracle.hex()


def test_exact_252_prior_eligible_observation_lag() -> None:
    machine = _machine()
    dates = _weekdays(date(2019, 3, 11), 253)
    prices = [Decimal("100"), Decimal("200"), *([Decimal("125")] * 250)]
    prices.append(Decimal("150"))

    outputs = [
        machine.on_daily_close(_close(EURUSD, day, price))
        for day, price in zip(dates, prices, strict=True)
    ]

    assert all(output is None for output in outputs[:252])
    assert outputs[252] is not None
    assert outputs[252].target is RawDirectionalTarget.LONG


@pytest.mark.parametrize(
    ("first", "current", "expected"),
    [
        ("1.00000", "1.00001", RawDirectionalTarget.LONG),
        ("1.00000", "0.99999", RawDirectionalTarget.SHORT),
        ("1.00000", "1.00000", RawDirectionalTarget.FLAT),
    ],
)
def test_long_short_flat_mapping(
    first: str, current: str, expected: RawDirectionalTarget
) -> None:
    machine = _machine()
    dates = _weekdays(date(2019, 3, 11), 253)
    prices = [Decimal(first), *([Decimal("1.00000")] * 251), Decimal(current)]

    outputs = [
        machine.on_daily_close(_close(EURUSD, day, price))
        for day, price in zip(dates, prices, strict=True)
    ]

    assert outputs[-1] is not None
    assert outputs[-1].target is expected
    assert machine.active_targets == ()


def test_invalid_and_missing_observations_do_not_enter_eligible_history() -> None:
    machine = _machine()
    dates = _weekdays(date(2019, 3, 11), 255)
    for day in dates[:251]:
        assert machine.on_daily_close(_close(EURUSD, day, Decimal("1.00"))) is None

    assert (
        machine.on_daily_close(_close(EURUSD, dates[251], Decimal("NaN")))
        is None
    )
    # dates[252] is deliberately missing: no call means no history observation.
    assert (
        machine.on_daily_close(_close(EURUSD, dates[253], Decimal("1.00")))
        is None
    )
    signal = machine.on_daily_close(_close(EURUSD, dates[254], Decimal("1.01")))

    assert signal is not None
    assert signal.target is RawDirectionalTarget.LONG


def test_no_lookahead_and_deterministic_output() -> None:
    dates = _weekdays(date(2019, 3, 11), 254)
    prefix = [Decimal("1.00"), *([Decimal("1.01")] * 251), Decimal("1.02")]

    def run(future: Decimal) -> tuple[object, ...]:
        machine = _machine()
        prices = [*prefix, future]
        return tuple(
            machine.on_daily_close(_close(EURUSD, day, price))
            for day, price in zip(dates, prices, strict=True)
        )

    rising = run(Decimal("9.00"))
    falling = run(Decimal("0.10"))

    assert rising[:253] == falling[:253]
    assert rising == run(Decimal("9.00"))


def test_target_executes_only_on_first_strictly_later_stage_g_point() -> None:
    machine = _machine()
    dates = _weekdays(date(2019, 3, 11), 253)
    for day in dates[:-1]:
        machine.on_daily_close(_close(EURUSD, day, Decimal("1.00")))
    signal = machine.on_daily_close(_close(EURUSD, dates[-1], Decimal("1.01")))
    assert signal is not None

    same_time = _frame(signal.signal_information_time_utc, Decimal("1.01"))
    later = _frame(
        signal.signal_information_time_utc + timedelta(minutes=1), Decimal("1.01")
    )

    assert machine.on_execution_frame(same_time) == ()
    executable = machine.on_execution_frame(later)
    assert len(executable) == 1
    assert executable[0].target is RawDirectionalTarget.LONG
    assert executable[0].execution_information_time_utc > (
        executable[0].signal_information_time_utc
    )
    assert machine.active_targets == ((EURUSD, RawDirectionalTarget.LONG),)
    assert machine.on_execution_frame(later) == ()


def test_latest_target_is_held_until_a_changed_daily_signal() -> None:
    machine = _machine()
    dates = _weekdays(date(2019, 3, 11), 254)
    for day in dates[:-2]:
        machine.on_daily_close(_close(EURUSD, day, Decimal("1.00")))
    first = machine.on_daily_close(_close(EURUSD, dates[-2], Decimal("1.01")))
    assert first is not None
    machine.on_execution_frame(
        _frame(
            first.signal_information_time_utc + timedelta(minutes=1),
            Decimal("1.01"),
        )
    )

    unchanged = machine.on_daily_close(_close(EURUSD, dates[-1], Decimal("1.02")))

    assert unchanged is None
    assert machine.active_targets == ((EURUSD, RawDirectionalTarget.LONG),)


def test_instrument_history_and_targets_are_independent() -> None:
    machine = _machine()
    dates = _weekdays(date(2019, 3, 11), 253)
    eur_signal = None
    for index, day in enumerate(dates):
        eur_signal = machine.on_daily_close(
            _close(EURUSD, day, Decimal("1.10") + Decimal(index) / Decimal("10000"))
        )
        if index % 7 == 0:
            assert machine.on_daily_close(_close(GBPUSD, day, None)) is None

    assert eur_signal is not None
    assert eur_signal.instrument_id == EURUSD
    assert eur_signal.target is RawDirectionalTarget.LONG


def test_daily_midpoints_are_actual_independent_session_closes_across_dst() -> None:
    winter_info = datetime(2020, 1, 6, 17, tzinfo=NY).astimezone(UTC)
    summer_info = datetime(2020, 7, 6, 17, tzinfo=NY).astimezone(UTC)
    winter = _gap_frame(winter_info, EURUSD, Decimal("1.2000"))
    non_close = _gap_frame(summer_info - timedelta(minutes=1), EURUSD, Decimal("9"))
    summer = _gap_frame(summer_info, GBPUSD, Decimal("1.3000"))

    closes = derive_daily_midpoint_closes((winter, non_close, summer))

    assert [(item.instrument_id, item.close) for item in closes] == [
        (EURUSD, Decimal("1.2000")),
        (GBPUSD, Decimal("1.3000")),
    ]
    assert winter_info.hour == 22
    assert summer_info.hour == 21


def test_fold_warmup_isolated_and_validation_holdout_refused() -> None:
    fold = frozen_development_folds().folds[0]
    context = DevelopmentResearchContext(
        universe=cast(Any, None),
        artifacts=(),
        start_utc=DEVELOPMENT_START,
        end_exclusive_utc=DEVELOPMENT_END_EXCLUSIVE,
    )
    candidate = TsMomentumDevelopmentFold(context, fold, load_ts_momentum_spec())
    training_dates = [
        item
        for item in _weekdays(date(2019, 3, 11), 400)
        if _information_time(item) < fold.compare_start_utc
    ][-252:]
    for day in training_dates:
        assert (
            candidate.on_daily_close(_close(EURUSD, day, Decimal("1.00"))) is None
        )
    comparison_day = next(
        item
        for item in _weekdays(fold.compare_start_utc.date(), 10)
        if _information_time(item) >= fold.compare_start_utc
    )
    signal = candidate.on_daily_close(
        _close(EURUSD, comparison_day, Decimal("1.01"))
    )
    assert signal is not None

    with pytest.raises(StageGValidationError, match="validation is LOCKED"):
        candidate.on_daily_close(
            _close(GBPUSD, comparison_day, Decimal("1.20")),
            partition=ResearchPartition.VALIDATION,
        )
    with pytest.raises(StageGValidationError, match="holdout is LOCKED"):
        candidate.on_execution_frame(
            _frame(signal.signal_information_time_utc, Decimal("1.01")),
            partition=ResearchPartition.HOLDOUT,
        )
    with pytest.raises(StageGValidationError, match="locked validation"):
        candidate.on_daily_close(
            _close(EURUSD, date(2023, 4, 12), Decimal("1.01"))
        )
    with pytest.raises(StageGValidationError, match="locked holdout"):
        candidate.on_daily_close(
            _close(EURUSD, date(2024, 8, 21), Decimal("1.01"))
        )


def _machine() -> TsMomentumStateMachine:
    return TsMomentumStateMachine(load_ts_momentum_spec())


def _weekdays(start: date, count: int) -> list[date]:
    result: list[date] = []
    current = start
    while len(result) < count:
        if current.weekday() < 5:
            result.append(current)
        current += timedelta(days=1)
    return result


def _information_time(day: date) -> datetime:
    return datetime.combine(day, time(17), tzinfo=NY).astimezone(UTC)


def _close(
    instrument_id: str, day: date, value: Decimal | None
) -> DailyMidpointClose:
    information = _information_time(day)
    return DailyMidpointClose(
        instrument_id=instrument_id,
        session_date=day,
        event_time_utc=information - timedelta(minutes=1),
        information_time_utc=information,
        close=value,
    )


def _frame(information: datetime, price: Decimal) -> SynchronizedClockFrame:
    event = information - timedelta(minutes=1)
    observations = tuple(
        InstrumentBarObservation(
            instrument_id=instrument_id,
            timestamp_utc=event,
            available_at_utc=information,
            bid=price - Decimal("0.0001"),
            ask=price + Decimal("0.0001"),
        )
        for instrument_id in FROZEN_INSTRUMENT_IDS
    )
    return SynchronizedClockFrame(event, information, observations, True)


def _gap_frame(
    information: datetime, instrument_id: str, price: Decimal
) -> SynchronizedClockFrame:
    event = information - timedelta(minutes=1)
    observed = InstrumentBarObservation(
        instrument_id=instrument_id,
        timestamp_utc=event,
        available_at_utc=information,
        bid=price - Decimal("0.0001"),
        ask=price + Decimal("0.0001"),
    )
    observations = tuple(
        observed if item == instrument_id else None for item in FROZEN_INSTRUMENT_IDS
    )
    return SynchronizedClockFrame(event, information, observations, False)
