"""Native paired-M1 execution for frozen Batch 4 clock occurrences.

Only caller-supplied DataFrames are accepted.  There is no loader, CLI,
or research-partition access in this infrastructure module.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal

import pandas as pd  # type: ignore[import-untyped]

from ftmoquant.research.alpha_lab.batch4_clock_scheduler import (
    ScheduledOccurrence,
    load_frozen_clock_specs,
    schedule_occurrence,
)
from ftmoquant.research.alpha_lab.cost_stress import widen_bid_ask_frame

REFERENCE_NOTIONAL_USD = Decimal("100000")
EXIT_REASON_SCHEDULED_TIME = "scheduled_time_exit"
ALLOWED_STRESS_MULTIPLIERS = (Decimal("1.0"), Decimal("1.5"), Decimal("2.0"))
SkipReason = Literal[
    "no_entry_observation",
    "entry_not_before_scheduled_exit",
    "entry_outside_local_date",
    "no_exit_observation",
    "exit_outside_local_date",
]


class Batch4ExecutionError(ValueError):
    """Raised on malformed input or a frozen execution-contract violation."""


@dataclass(frozen=True, slots=True)
class ScheduledTradeResult:
    hypothesis_id: str
    family: str
    instrument_id: str
    direction: Literal["BUY", "SELL"]
    local_date: str
    scheduled_entry_utc: datetime
    scheduled_exit_utc: datetime
    actual_entry_utc: datetime
    actual_exit_utc: datetime
    entry_price: Decimal
    exit_price: Decimal
    quantity: Decimal
    pnl_account_currency: Decimal
    account_currency: str
    reference_notional_usd: Decimal
    return_on_reference_notional: Decimal
    holding_seconds: int
    exit_reason: str = EXIT_REASON_SCHEDULED_TIME

    def __post_init__(self) -> None:
        if self.actual_entry_utc <= self.scheduled_entry_utc:
            raise Batch4ExecutionError("entry fill must be strictly later")
        if self.actual_exit_utc <= self.scheduled_exit_utc:
            raise Batch4ExecutionError("exit fill must be strictly later")
        if self.actual_exit_utc <= self.actual_entry_utc:
            raise Batch4ExecutionError("exit fill must follow entry fill")
        if self.exit_reason != EXIT_REASON_SCHEDULED_TIME:
            raise Batch4ExecutionError("Batch 4.1 supports scheduled time exits only")


@dataclass(frozen=True, slots=True)
class ScheduledSkipRecord:
    hypothesis_id: str
    instrument_id: str
    local_date: str
    reason: SkipReason
    relevant_scheduled_utc: datetime


def _instrument_currencies(instrument_id: str) -> tuple[str, str]:
    try:
        base, quote_venue = instrument_id.split("/", maxsplit=1)
        quote, venue = quote_venue.split(".", maxsplit=1)
    except ValueError as error:
        raise Batch4ExecutionError(f"invalid instrument_id: {instrument_id}") from error
    if venue != "OANDA" or len(base) != 3 or len(quote) != 3:
        raise Batch4ExecutionError(f"invalid frozen OANDA instrument: {instrument_id}")
    if "USD" not in {base, quote}:
        raise Batch4ExecutionError("Batch 4 requires a direct USD pair")
    return base, quote


def _usd_gross_to_quantity(
    gross_usd: Decimal, entry_price: Decimal, base: str, quote: str
) -> Decimal:
    """Batch-3 dimensional sizing semantics, isolated from loader imports."""

    if not gross_usd.is_finite() or gross_usd <= 0:
        raise Batch4ExecutionError("reference notional must be positive and finite")
    if not entry_price.is_finite() or entry_price <= 0:
        raise Batch4ExecutionError("entry price must be positive and finite")
    if base == "USD":
        return gross_usd
    if quote == "USD":
        return gross_usd / entry_price
    raise Batch4ExecutionError("cannot size a non-USD-cross instrument")


def _quote_pnl_to_usd(
    quote_pnl: Decimal, entry_price: Decimal, base: str, quote: str
) -> Decimal:
    """Existing single-conversion-at-entry-price account-currency semantics."""

    if quote == "USD":
        return quote_pnl
    if base == "USD":
        return quote_pnl / entry_price
    raise Batch4ExecutionError("cannot convert P&L for a non-USD-cross instrument")


def _validate_frames(bid_m1: pd.DataFrame, ask_m1: pd.DataFrame) -> pd.DatetimeIndex:
    if not bid_m1.index.equals(ask_m1.index):
        raise Batch4ExecutionError("BID and ASK must share an identical paired index")
    if not isinstance(bid_m1.index, pd.DatetimeIndex):
        raise Batch4ExecutionError("paired index must be a DatetimeIndex")
    if bid_m1.index.tz is None or str(bid_m1.index.tz) not in {"UTC", "UTC+00:00"}:
        raise Batch4ExecutionError("paired M1 index must be timezone-aware UTC")
    if not bid_m1.index.is_monotonic_increasing or not bid_m1.index.is_unique:
        raise Batch4ExecutionError("paired M1 index must be sorted and unique")
    for frame, label in ((bid_m1, "bid_m1"), (ask_m1, "ask_m1")):
        if "close" not in frame.columns:
            raise Batch4ExecutionError(f"{label} has no close column")
    if (ask_m1["close"] < bid_m1["close"]).any():
        raise Batch4ExecutionError("crossed paired market")
    return bid_m1.index


def _same_local_date(timestamp: pd.Timestamp, occurrence: ScheduledOccurrence) -> bool:
    return bool(
        timestamp.tz_convert(occurrence.timezone).date() == occurrence.local_date
    )


def execute_scheduled_occurrences(
    occurrences: Sequence[ScheduledOccurrence],
    *,
    bid_m1: pd.DataFrame,
    ask_m1: pd.DataFrame,
    reference_notional_usd: Decimal = REFERENCE_NOTIONAL_USD,
    cost_stress_multiplier: Decimal = Decimal("1.0"),
) -> tuple[tuple[ScheduledTradeResult, ...], tuple[ScheduledSkipRecord, ...]]:
    """Execute independent frozen clocks against genuine paired M1 closes.

    Every fill uses ``bisect_right`` (strictly later).  Overlapping distinct
    hypotheses are intentionally independent; there is no global busy flag.
    """

    if cost_stress_multiplier not in ALLOWED_STRESS_MULTIPLIERS:
        raise Batch4ExecutionError(
            f"cost stress must be one of {ALLOWED_STRESS_MULTIPLIERS}"
        )
    frozen_specs = {spec.hypothesis_id: spec for spec in load_frozen_clock_specs()}
    if cost_stress_multiplier != Decimal("1.0"):
        bid_m1, ask_m1 = widen_bid_ask_frame(
            bid_m1, ask_m1, float(cost_stress_multiplier)
        )
    index = _validate_frames(bid_m1, ask_m1)
    paired_ns = index.as_unit("ns").asi8.tolist()
    trades: list[ScheduledTradeResult] = []
    skips: list[ScheduledSkipRecord] = []
    seen: set[tuple[str, str]] = set()

    ordered = sorted(
        occurrences,
        key=lambda row: (
            row.scheduled_entry_utc,
            row.hypothesis_id,
            row.instrument_id,
        ),
    )
    for occurrence in ordered:
        frozen_spec = frozen_specs.get(occurrence.hypothesis_id)
        if frozen_spec is None:
            raise Batch4ExecutionError(
                f"occurrence has no frozen hypothesis: {occurrence.hypothesis_id}"
            )
        if occurrence != schedule_occurrence(frozen_spec, occurrence.local_date):
            raise Batch4ExecutionError(
                "occurrence clock, direction, family, instrument, or timezone "
                "does not match its frozen hypothesis"
            )
        occurrence_key = (occurrence.hypothesis_id, occurrence.local_date.isoformat())
        if occurrence_key in seen:
            raise Batch4ExecutionError("duplicate hypothesis/local-date occurrence")
        seen.add(occurrence_key)
        base, quote = _instrument_currencies(occurrence.instrument_id)
        entry_decision_ns = pd.Timestamp(occurrence.scheduled_entry_utc).value
        exit_decision_ns = pd.Timestamp(occurrence.scheduled_exit_utc).value
        entry_pos = bisect.bisect_right(paired_ns, entry_decision_ns)
        if entry_pos >= len(index):
            skips.append(
                ScheduledSkipRecord(
                    occurrence.hypothesis_id,
                    occurrence.instrument_id,
                    occurrence.local_date.isoformat(),
                    "no_entry_observation",
                    occurrence.scheduled_entry_utc,
                )
            )
            continue
        entry_ts = index[entry_pos]
        if int(entry_ts.value) >= exit_decision_ns:
            skips.append(
                ScheduledSkipRecord(
                    occurrence.hypothesis_id,
                    occurrence.instrument_id,
                    occurrence.local_date.isoformat(),
                    "entry_not_before_scheduled_exit",
                    occurrence.scheduled_entry_utc,
                )
            )
            continue
        if not _same_local_date(entry_ts, occurrence):
            skips.append(
                ScheduledSkipRecord(
                    occurrence.hypothesis_id,
                    occurrence.instrument_id,
                    occurrence.local_date.isoformat(),
                    "entry_outside_local_date",
                    occurrence.scheduled_entry_utc,
                )
            )
            continue

        exit_pos = bisect.bisect_right(paired_ns, exit_decision_ns)
        if exit_pos >= len(index):
            skips.append(
                ScheduledSkipRecord(
                    occurrence.hypothesis_id,
                    occurrence.instrument_id,
                    occurrence.local_date.isoformat(),
                    "no_exit_observation",
                    occurrence.scheduled_exit_utc,
                )
            )
            continue
        exit_ts = index[exit_pos]
        if not _same_local_date(exit_ts, occurrence):
            skips.append(
                ScheduledSkipRecord(
                    occurrence.hypothesis_id,
                    occurrence.instrument_id,
                    occurrence.local_date.isoformat(),
                    "exit_outside_local_date",
                    occurrence.scheduled_exit_utc,
                )
            )
            continue

        is_buy = occurrence.direction == "BUY"
        entry_value = (
            ask_m1["close"].iloc[entry_pos]
            if is_buy
            else bid_m1["close"].iloc[entry_pos]
        )
        exit_value = (
            bid_m1["close"].iloc[exit_pos] if is_buy else ask_m1["close"].iloc[exit_pos]
        )
        entry_price = Decimal(str(entry_value))
        exit_price = Decimal(str(exit_value))
        quantity = _usd_gross_to_quantity(
            reference_notional_usd, entry_price, base, quote
        )
        direction_sign = Decimal(1) if is_buy else Decimal(-1)
        pnl_quote = direction_sign * quantity * (exit_price - entry_price)
        pnl_usd = _quote_pnl_to_usd(pnl_quote, entry_price, base, quote)
        actual_entry = entry_ts.to_pydatetime().astimezone(UTC)
        actual_exit = exit_ts.to_pydatetime().astimezone(UTC)
        holding_seconds = int((actual_exit - actual_entry).total_seconds())
        trades.append(
            ScheduledTradeResult(
                hypothesis_id=occurrence.hypothesis_id,
                family=occurrence.family,
                instrument_id=occurrence.instrument_id,
                direction=occurrence.direction,
                local_date=occurrence.local_date.isoformat(),
                scheduled_entry_utc=occurrence.scheduled_entry_utc,
                scheduled_exit_utc=occurrence.scheduled_exit_utc,
                actual_entry_utc=actual_entry,
                actual_exit_utc=actual_exit,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=quantity,
                pnl_account_currency=pnl_usd,
                account_currency="USD",
                reference_notional_usd=reference_notional_usd,
                return_on_reference_notional=pnl_usd / reference_notional_usd,
                holding_seconds=holding_seconds,
            )
        )
    return tuple(trades), tuple(skips)
