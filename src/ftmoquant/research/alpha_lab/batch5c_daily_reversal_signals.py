"""Pure prior-30-day extreme-move reversal events for frozen Batch 5C."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal

from ftmoquant.research.alpha_lab.batch5_daily import CompletedFxDay
from ftmoquant.research.alpha_lab.batch5_execution import TradeIntent
from ftmoquant.research.alpha_lab.batch5_preregistration import (
    FAMILY_B5C,
    verify_preregistration,
)

LOOKBACK = 30
THRESHOLD = Decimal(2)
UNIVERSE = (
    "EUR/USD.OANDA",
    "USD/JPY.OANDA",
    "USD/CAD.OANDA",
    "AUD/USD.OANDA",
    "EUR/JPY.OANDA",
)
Direction = Literal["BUY", "SELL"]


class Batch5CSignalError(ValueError):
    """Raised when B5C inputs or frozen metadata drift."""


@dataclass(frozen=True, slots=True)
class B5CEvent:
    family: str
    strategy_id: str
    sleeve_id: str
    instrument_id: str
    signal_timestamp: datetime
    direction: Direction
    event_return: Decimal
    prior_mean_30: Decimal
    prior_sample_sd_30: Decimal
    event_day_index: int


def _mean(values: Sequence[Decimal]) -> Decimal:
    return sum(values, Decimal(0)) / len(values)


def _sample_sd(values: Sequence[Decimal], mean: Decimal) -> Decimal:
    if len(values) < 2:
        raise Batch5CSignalError("sample SD needs at least two values")
    variance = sum(((value - mean) ** 2 for value in values), Decimal(0)) / (
        len(values) - 1
    )
    return variance.sqrt()


def generate_events(days: Sequence[CompletedFxDay]) -> tuple[B5CEvent, ...]:
    document = verify_preregistration()
    frozen = document["families"][FAMILY_B5C]["literature_anchored_rule"]
    if tuple(frozen["universe"]) != UNIVERSE:
        raise Batch5CSignalError("frozen B5C universe drift")
    ordered = sorted(days, key=lambda row: row.end_utc)
    if not ordered:
        return ()
    instrument = ordered[0].instrument_id
    if instrument not in UNIVERSE or any(
        row.instrument_id != instrument for row in ordered
    ):
        raise Batch5CSignalError("events require exactly one native frozen instrument")
    if len({row.end_utc for row in ordered}) != len(ordered):
        raise Batch5CSignalError("duplicate completed-day boundary")
    returns = [row.open_to_close_return for row in ordered]
    events: list[B5CEvent] = []
    for index in range(LOOKBACK, len(ordered)):
        prior = returns[index - LOOKBACK : index]
        mean = _mean(prior)
        sd = _sample_sd(prior, mean)
        current = returns[index]
        upper = mean + THRESHOLD * sd
        lower = mean - THRESHOLD * sd
        direction: Direction | None = (
            "SELL" if current > upper else "BUY" if current < lower else None
        )
        if direction is not None:
            events.append(
                B5CEvent(
                    FAMILY_B5C,
                    "B5C_FROZEN_DAILY_OVERREACTION_REVERSAL",
                    f"B5C_{instrument.split('.')[0].replace('/', '')}",
                    instrument,
                    ordered[index].end_utc,
                    direction,
                    current,
                    mean,
                    sd,
                    index,
                )
            )
    return tuple(events)


def build_next_day_intents(
    events: Sequence[B5CEvent],
    days_by_instrument: Mapping[str, Sequence[CompletedFxDay]],
) -> tuple[TradeIntent, ...]:
    """Schedule one full next valid FX day and ignore overlapping events."""

    active_until: dict[str, datetime] = {}
    intents: list[TradeIntent] = []
    for event in sorted(
        events, key=lambda row: (row.signal_timestamp, row.instrument_id)
    ):
        days = sorted(
            days_by_instrument[event.instrument_id], key=lambda row: row.end_utc
        )
        if event.event_day_index + 1 >= len(days):
            continue
        exit_boundary = days[event.event_day_index + 1].end_utc
        if event.signal_timestamp <= active_until.get(
            event.instrument_id,
            datetime.min.replace(tzinfo=event.signal_timestamp.tzinfo),
        ):
            continue
        intents.append(
            TradeIntent(
                event.family,
                event.strategy_id,
                event.sleeve_id,
                event.instrument_id,
                event.signal_timestamp,
                event.signal_timestamp,
                exit_boundary,
                event.direction,
                metadata={
                    "event_return": str(event.event_return),
                    "prior_mean_30": str(event.prior_mean_30),
                    "prior_sample_sd_30": str(event.prior_sample_sd_30),
                },
            )
        )
        active_until[event.instrument_id] = exit_boundary
    return tuple(intents)
