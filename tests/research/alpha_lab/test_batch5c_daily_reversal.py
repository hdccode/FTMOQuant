from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import pandas as pd  # type: ignore[import-untyped]

from ftmoquant.research.alpha_lab.batch5_daily import CompletedFxDay, ny_fx_boundary
from ftmoquant.research.alpha_lab.batch5_execution import Batch5TradeResult
from ftmoquant.research.alpha_lab.batch5c_daily_reversal_execution import execute_events
from ftmoquant.research.alpha_lab.batch5c_daily_reversal_signals import (
    B5CEvent,
    build_next_day_intents,
    generate_events,
)


def day(
    local_date: date, return_value: Decimal, instrument: str = "EUR/JPY.OANDA"
) -> CompletedFxDay:
    return CompletedFxDay(
        instrument,
        local_date,
        ny_fx_boundary(local_date - timedelta(days=1)),
        ny_fx_boundary(local_date),
        Decimal(100),
        Decimal(100) * (Decimal(1) + return_value),
    )


def prior_days(start: date) -> list[CompletedFxDay]:
    return [
        day(
            start + timedelta(days=index),
            Decimal("0.01") if index % 2 else Decimal("-0.01"),
        )
        for index in range(30)
    ]


def test_sample_sd_strict_threshold_and_reversal_direction() -> None:
    start = date(2021, 1, 1)
    prior = prior_days(start)
    probe = generate_events([*prior, day(start + timedelta(days=30), Decimal("0.03"))])
    assert len(probe) == 1 and probe[0].direction == "SELL"
    zero_prior = [day(start + timedelta(days=index), Decimal(0)) for index in range(30)]
    equal = generate_events([*zero_prior, day(start + timedelta(days=30), Decimal(0))])
    negative = generate_events(
        [*prior, day(start + timedelta(days=30), Decimal("-0.03"))]
    )
    assert equal == ()
    assert len(negative) == 1 and negative[0].direction == "BUY"


def test_next_complete_day_holding_and_overlapping_event_ignored() -> None:
    start = date(2021, 1, 1)
    rows = prior_days(start)
    rows.extend(
        [
            day(start + timedelta(days=30), Decimal("0.03")),
            day(start + timedelta(days=31), Decimal("-0.03")),
            day(start + timedelta(days=32), Decimal("0")),
        ]
    )
    events = generate_events(rows)
    intents = build_next_day_intents(events, {"EUR/JPY.OANDA": rows})
    assert len(events) >= 2
    assert len(intents) == 1
    assert intents[0].exit_decision_timestamp == rows[31].end_utc
    assert intents[0].instrument_id == "EUR/JPY.OANDA"


def test_native_eurjpy_one_day_execution_converts_jpy_through_usdjpy() -> None:
    start_date = date(2021, 1, 2)
    days = [
        day(start_date, Decimal("0.03")),
        day(start_date + timedelta(days=1), Decimal(0)),
    ]
    event = B5CEvent(
        "B5C_daily_fx_overreaction_reversal",
        "B5C_FROZEN_DAILY_OVERREACTION_REVERSAL",
        "B5C_EURJPY",
        "EUR/JPY.OANDA",
        days[0].end_utc,
        "SELL",
        Decimal("0.03"),
        Decimal(0),
        Decimal("0.01"),
        0,
    )
    timestamps = pd.DatetimeIndex(
        [
            days[0].end_utc + timedelta(minutes=1),
            days[1].end_utc + timedelta(minutes=1),
        ]
    )

    def frame(values: list[float]) -> pd.DataFrame:
        return pd.DataFrame(
            {name: values for name in ("open", "high", "low", "close")},
            index=timestamps,
        )

    results = execute_events(
        [event],
        days_by_instrument={"EUR/JPY.OANDA": days},
        native_frames={"EUR/JPY.OANDA": (frame([130.0, 129.0]), frame([130.1, 129.1]))},
        usdjpy_conversion_frames=(
            frame([110.0, 111.0]),
            frame([110.1, 111.1]),
        ),
    )
    assert len(results) == 1 and isinstance(results[0], Batch5TradeResult)
    assert results[0].actual_entry_timestamp > event.signal_timestamp
    assert results[0].actual_exit_timestamp > days[1].end_utc
    assert results[0].pnl_usd > 0
