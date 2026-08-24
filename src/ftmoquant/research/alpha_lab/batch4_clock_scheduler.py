"""Pure civil-clock scheduler for the frozen Batch 4 hypotheses.

The clock is the signal.  This module verifies and derives every schedule
from the write-once preregistration; it never reads market or result data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from ftmoquant.research.alpha_lab.batch4_preregistration import (
    PREREGISTRATION_PATH,
    Batch4PreregistrationError,
    verify_preregistration,
)

EXPECTED_PREREGISTRATION_SEMANTIC_SHA256 = (
    "4a019140ab798cdccc14ba3c6a0817dfca10e4d626de58b95d1c5c0d7c01dd98"
)
Direction = Literal["BUY", "SELL"]


class Batch4ClockError(ValueError):
    """Raised when frozen identity or civil-clock invariants are violated."""


@dataclass(frozen=True, slots=True)
class FrozenClockSpec:
    """One immutable pair-sleeve and clock hypothesis."""

    hypothesis_id: str
    family: str
    instrument_id: str
    direction: Direction
    timezone: str
    local_start_time: time
    local_end_time: time

    def __post_init__(self) -> None:
        if self.local_end_time <= self.local_start_time:
            raise Batch4ClockError(
                "frozen Batch 4 windows must be same-day and end after start"
            )
        try:
            ZoneInfo(self.timezone)
        except Exception as error:
            raise Batch4ClockError(f"unknown timezone: {self.timezone}") from error


@dataclass(frozen=True, slots=True)
class ScheduledOccurrence:
    """One price-independent occurrence with exact UTC decision instants."""

    hypothesis_id: str
    family: str
    instrument_id: str
    direction: Direction
    timezone: str
    local_date: date
    scheduled_entry_utc: datetime
    scheduled_exit_utc: datetime

    def __post_init__(self) -> None:
        for value in (self.scheduled_entry_utc, self.scheduled_exit_utc):
            if value.tzinfo is None or value.utcoffset() != timedelta(0):
                raise Batch4ClockError("scheduled timestamps must be aware UTC")
        if self.scheduled_exit_utc <= self.scheduled_entry_utc:
            raise Batch4ClockError("scheduled exit must be after entry")


def _clock(value: object) -> time:
    if not isinstance(value, str):
        raise Batch4ClockError("frozen clock must be an ISO local-time string")
    parsed = time.fromisoformat(value)
    if parsed.tzinfo is not None:
        raise Batch4ClockError("frozen local clocks must not embed an offset")
    return parsed


def _direction(value: object) -> Direction:
    if value == "BUY":
        return "BUY"
    if value == "SELL":
        return "SELL"
    raise Batch4ClockError(f"invalid frozen direction: {value!r}")


def _verified_document(path: Path) -> dict[str, object]:
    try:
        document = verify_preregistration(path)
    except (OSError, ValueError, Batch4PreregistrationError) as error:
        raise Batch4ClockError("Batch 4 preregistration verification failed") from error
    actual = document.get("preregistration_semantic_sha256")
    if actual != EXPECTED_PREREGISTRATION_SEMANTIC_SHA256:
        raise Batch4ClockError(
            "Batch 4 preregistration identity mismatch: "
            f"expected {EXPECTED_PREREGISTRATION_SEMANTIC_SHA256}, got {actual!r}"
        )
    return document


def load_frozen_clock_specs(
    path: Path = PREREGISTRATION_PATH,
) -> tuple[FrozenClockSpec, ...]:
    """Verify the artifact and derive all 91 executable frozen clocks."""

    document = _verified_document(path)
    families = document.get("families")
    if not isinstance(families, dict):
        raise Batch4ClockError("preregistration families object is missing")
    specs: list[FrozenClockSpec] = []

    local_family = "B4F1A_local_hours_flow_seasonality"
    local = families.get(local_family)
    if not isinstance(local, dict) or not isinstance(local.get("sleeves"), list):
        raise Batch4ClockError("frozen B4F1A sleeves are missing")
    for row in local["sleeves"]:
        if not isinstance(row, dict):
            raise Batch4ClockError("invalid B4F1A sleeve")
        specs.append(
            FrozenClockSpec(
                hypothesis_id=str(row["sleeve_id"]),
                family=local_family,
                instrument_id=str(row["pair"]),
                direction=_direction(row["entry_side"]),
                timezone=str(row["timezone"]),
                local_start_time=_clock(row["start_local"]),
                local_end_time=_clock(row["end_local"]),
            )
        )

    for family in (
        "B4F1B_london_fix_flow_reversal",
        "B4F1C_tokyo_fix_flow_reversal",
    ):
        frozen_family = families.get(family)
        if not isinstance(frozen_family, dict) or not isinstance(
            frozen_family.get("configurations"), list
        ):
            raise Batch4ClockError(f"frozen {family} configurations are missing")
        for row in frozen_family["configurations"]:
            if not isinstance(row, dict):
                raise Batch4ClockError(f"invalid {family} configuration")
            sides = row.get("pair_entry_sides")
            eligible = row.get("eligible_pairs")
            if not isinstance(sides, dict) or not isinstance(eligible, list):
                raise Batch4ClockError(f"invalid {family} pair mapping")
            for instrument_id in eligible:
                instrument = str(instrument_id)
                specs.append(
                    FrozenClockSpec(
                        hypothesis_id=(
                            f"{family}:{row['configuration_id']}:{instrument}"
                        ),
                        family=family,
                        instrument_id=instrument,
                        direction=_direction(sides.get(instrument)),
                        timezone=str(row["timezone"]),
                        local_start_time=_clock(row["start_local"]),
                        local_end_time=_clock(row["end_local"]),
                    )
                )

    grid = document.get("grid_accounting")
    if not isinstance(grid, dict):
        raise Batch4ClockError("preregistration grid accounting is missing")
    expected_count = grid.get("total_executable_sleeve_configuration_hypotheses")
    if expected_count != 91 or len(specs) != expected_count:
        raise Batch4ClockError(
            f"frozen grid mismatch: artifact={expected_count!r}, derived={len(specs)}"
        )
    if len({spec.hypothesis_id for spec in specs}) != len(specs):
        raise Batch4ClockError("duplicate frozen hypothesis_id")
    return tuple(specs)


def schedule_occurrence(spec: FrozenClockSpec, local_date: date) -> ScheduledOccurrence:
    """Create one occurrence without consulting prices or market calendars."""

    zone = ZoneInfo(spec.timezone)
    local_entry = datetime.combine(local_date, spec.local_start_time, tzinfo=zone)
    local_exit = datetime.combine(local_date, spec.local_end_time, tzinfo=zone)
    entry_utc = local_entry.astimezone(UTC)
    exit_utc = local_exit.astimezone(UTC)
    if entry_utc.astimezone(zone).date() != local_date:
        raise Batch4ClockError("entry local-date round trip failed")
    if exit_utc.astimezone(zone).date() != local_date:
        raise Batch4ClockError("exit local-date round trip failed")
    return ScheduledOccurrence(
        hypothesis_id=spec.hypothesis_id,
        family=spec.family,
        instrument_id=spec.instrument_id,
        direction=spec.direction,
        timezone=spec.timezone,
        local_date=local_date,
        scheduled_entry_utc=entry_utc,
        scheduled_exit_utc=exit_utc,
    )


def generate_occurrences(
    spec: FrozenClockSpec, start_local_date: date, end_local_date_exclusive: date
) -> tuple[ScheduledOccurrence, ...]:
    """Generate exactly one occurrence per civil date in ``[start, end)``.

    Weekends are intentionally included: missing market observations are an
    executor concern and the scheduler never fabricates or consults bars.
    """

    if end_local_date_exclusive < start_local_date:
        raise Batch4ClockError("end_local_date_exclusive must not precede start")
    count = (end_local_date_exclusive - start_local_date).days
    return tuple(
        schedule_occurrence(spec, start_local_date + timedelta(days=offset))
        for offset in range(count)
    )
