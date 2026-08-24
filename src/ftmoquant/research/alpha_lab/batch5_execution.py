"""Shared loader-free execution types and primitives for frozen Batch 5."""

from __future__ import annotations

import calendar
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal, cast

import pandas as pd  # type: ignore[import-untyped]

from ftmoquant.research.alpha_lab.batch4_execution import _validate_frames
from ftmoquant.research.alpha_lab.cost_stress import widen_bid_ask_frame

REFERENCE_NOTIONAL_USD = Decimal("100000")
ALLOWED_STRESS_MULTIPLIERS = (Decimal("1.0"), Decimal("1.5"), Decimal("2.0"))
Direction = Literal["BUY", "SELL"]


class Batch5ExecutionError(ValueError):
    """Raised when a synthetic or future loader supplies invalid execution data."""


@dataclass(frozen=True, slots=True)
class TradeIntent:
    family: str
    strategy_id: str
    sleeve_id: str
    instrument_id: str
    signal_timestamp: datetime
    entry_decision_timestamp: datetime
    exit_decision_timestamp: datetime
    direction: Direction
    cohort_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        timestamps = (
            self.signal_timestamp,
            self.entry_decision_timestamp,
            self.exit_decision_timestamp,
        )
        if any(value.tzinfo is None for value in timestamps):
            raise Batch5ExecutionError("intent timestamps must be timezone-aware")
        if self.signal_timestamp > self.entry_decision_timestamp:
            raise Batch5ExecutionError("signal cannot follow its entry decision")
        if self.exit_decision_timestamp <= self.entry_decision_timestamp:
            raise Batch5ExecutionError("exit decision must follow entry decision")


@dataclass(frozen=True, slots=True)
class Batch5TradeResult:
    family: str
    strategy_id: str
    sleeve_id: str
    instrument_id: str
    signal_timestamp: datetime
    actual_entry_timestamp: datetime
    actual_exit_timestamp: datetime
    direction: Direction
    quantity: Decimal
    entry_price: Decimal
    exit_price: Decimal
    pnl_usd: Decimal
    return_on_reference_notional: Decimal
    holding_seconds: int
    cohort_id: str | None
    skip_reason: None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            self.actual_entry_timestamp.tzinfo is None
            or self.actual_exit_timestamp.tzinfo is None
        ):
            raise Batch5ExecutionError("result timestamps must be timezone-aware")
        if self.actual_exit_timestamp <= self.actual_entry_timestamp:
            raise Batch5ExecutionError("result exit must follow entry")
        if self.quantity <= 0 or self.entry_price <= 0 or self.exit_price <= 0:
            raise Batch5ExecutionError("result quantity and prices must be positive")
        expected_holding = int(
            (self.actual_exit_timestamp - self.actual_entry_timestamp).total_seconds()
        )
        if self.holding_seconds != expected_holding:
            raise Batch5ExecutionError("holding_seconds does not match timestamps")
        if self.return_on_reference_notional != self.pnl_usd / REFERENCE_NOTIONAL_USD:
            raise Batch5ExecutionError("result return does not match frozen notional")


@dataclass(frozen=True, slots=True)
class Batch5SkipRecord:
    family: str
    strategy_id: str
    sleeve_id: str
    instrument_id: str
    signal_timestamp: datetime
    cohort_id: str | None
    skip_reason: str


def add_calendar_months(timestamp: datetime, months: int) -> datetime:
    """Add whole calendar months, clamping end-of-month deterministically."""

    if timestamp.tzinfo is None or months <= 0:
        raise Batch5ExecutionError(
            "calendar-month arithmetic needs aware time and months > 0"
        )
    index = timestamp.year * 12 + timestamp.month - 1 + months
    year, month_index = divmod(index, 12)
    month = month_index + 1
    day = min(timestamp.day, calendar.monthrange(year, month)[1])
    return timestamp.replace(year=year, month=month, day=day)


def _currencies(instrument_id: str) -> tuple[str, str]:
    try:
        base, remainder = instrument_id.split("/", maxsplit=1)
        quote, venue = remainder.split(".", maxsplit=1)
    except ValueError as error:
        raise Batch5ExecutionError(f"invalid instrument_id: {instrument_id}") from error
    if venue != "OANDA" or len(base) != 3 or len(quote) != 3:
        raise Batch5ExecutionError(f"invalid OANDA instrument: {instrument_id}")
    return base, quote


def _position(index: pd.DatetimeIndex, timestamp: datetime) -> int | None:
    if timestamp.tzinfo is None:
        raise Batch5ExecutionError("decision timestamps must be timezone-aware")
    position = int(index.searchsorted(pd.Timestamp(timestamp), side="right"))
    return position if position < len(index) else None


def first_strictly_later_timestamp(
    bid_m1: pd.DataFrame, ask_m1: pd.DataFrame, decision: datetime
) -> datetime | None:
    """Expose the shared paired-index ``bisect_right`` timestamp primitive."""

    index = _validate_frames(bid_m1, ask_m1)
    position = _position(index, decision)
    if position is None:
        return None
    return cast(datetime, index[position].to_pydatetime()).astimezone(UTC)


def _conversion_quote(
    conversion_bid: pd.DataFrame,
    conversion_ask: pd.DataFrame,
    timestamp: datetime,
) -> tuple[Decimal, Decimal]:
    index = _validate_frames(conversion_bid, conversion_ask)
    position = int(index.searchsorted(pd.Timestamp(timestamp), side="right")) - 1
    if position < 0:
        raise Batch5ExecutionError("no causal USD conversion quote at or before fill")
    bid = Decimal(str(conversion_bid["close"].iloc[position]))
    ask = Decimal(str(conversion_ask["close"].iloc[position]))
    if bid <= 0 or ask < bid:
        raise Batch5ExecutionError("invalid causal USD conversion quote")
    return bid, ask


def _quantity(
    direction: Direction,
    entry: Decimal,
    base: str,
    quote: str,
    conversion: tuple[Decimal, Decimal] | None,
    reference: Decimal,
) -> Decimal:
    if base == "USD":
        return reference
    if quote == "USD":
        return reference / entry
    if conversion is None:
        raise Batch5ExecutionError("non-USD cross requires causal USD conversion")
    conversion_bid, conversion_ask = conversion
    rate = conversion_bid if direction == "BUY" else conversion_ask
    return reference * rate / entry


def _pnl_usd(
    pnl_quote: Decimal,
    base: str,
    quote: str,
    conversion_price: Decimal,
    conversion: tuple[Decimal, Decimal] | None,
) -> Decimal:
    if quote == "USD":
        return pnl_quote
    if base == "USD":
        return pnl_quote / conversion_price
    if conversion is None:
        raise Batch5ExecutionError("non-USD P&L requires causal USD conversion")
    conversion_bid, conversion_ask = conversion
    return pnl_quote / (conversion_ask if pnl_quote >= 0 else conversion_bid)


def execute_intent(
    intent: TradeIntent,
    *,
    bid_m1: pd.DataFrame,
    ask_m1: pd.DataFrame,
    conversion_bid_m1: pd.DataFrame | None = None,
    conversion_ask_m1: pd.DataFrame | None = None,
    reference_notional_usd: Decimal = REFERENCE_NOTIONAL_USD,
    cost_stress_multiplier: Decimal = Decimal("1.0"),
) -> Batch5TradeResult | Batch5SkipRecord:
    """Execute one intent using first-strictly-later paired native closes."""

    if cost_stress_multiplier not in ALLOWED_STRESS_MULTIPLIERS:
        raise Batch5ExecutionError("cost stress is frozen to native, 1.5x, or 2.0x")
    if reference_notional_usd != REFERENCE_NOTIONAL_USD:
        raise Batch5ExecutionError("Batch 5 reference notional is frozen at USD 100000")
    if cost_stress_multiplier != Decimal("1.0"):
        bid_m1, ask_m1 = widen_bid_ask_frame(
            bid_m1, ask_m1, float(cost_stress_multiplier)
        )
        if conversion_bid_m1 is not None and conversion_ask_m1 is not None:
            conversion_bid_m1, conversion_ask_m1 = widen_bid_ask_frame(
                conversion_bid_m1,
                conversion_ask_m1,
                float(cost_stress_multiplier),
            )
    index = _validate_frames(bid_m1, ask_m1)
    entry_position = _position(index, intent.entry_decision_timestamp)
    exit_position = _position(index, intent.exit_decision_timestamp)
    if entry_position is None:
        return _skip(intent, "no_strictly_later_entry")
    if exit_position is None:
        return _skip(intent, "no_strictly_later_exit")
    if exit_position <= entry_position:
        return _skip(intent, "exit_not_after_entry")
    entry_timestamp = index[entry_position].to_pydatetime().astimezone(UTC)
    exit_timestamp = index[exit_position].to_pydatetime().astimezone(UTC)
    is_buy = intent.direction == "BUY"
    entry = Decimal(
        str(
            ask_m1["close"].iloc[entry_position]
            if is_buy
            else bid_m1["close"].iloc[entry_position]
        )
    )
    exit_price = Decimal(
        str(
            bid_m1["close"].iloc[exit_position]
            if is_buy
            else ask_m1["close"].iloc[exit_position]
        )
    )
    base, quote = _currencies(intent.instrument_id)
    needs_conversion = "USD" not in {base, quote}
    if needs_conversion and (conversion_bid_m1 is None or conversion_ask_m1 is None):
        raise Batch5ExecutionError("native cross execution needs USD conversion frames")
    entry_conversion = (
        _conversion_quote(conversion_bid_m1, conversion_ask_m1, entry_timestamp)
        if needs_conversion
        and conversion_bid_m1 is not None
        and conversion_ask_m1 is not None
        else None
    )
    exit_conversion = (
        _conversion_quote(conversion_bid_m1, conversion_ask_m1, exit_timestamp)
        if needs_conversion
        and conversion_bid_m1 is not None
        and conversion_ask_m1 is not None
        else None
    )
    quantity = _quantity(
        intent.direction,
        entry,
        base,
        quote,
        entry_conversion,
        reference_notional_usd,
    )
    sign = Decimal(1) if is_buy else Decimal(-1)
    pnl_quote = sign * quantity * (exit_price - entry)
    # Preserve the existing audited Batch-4 convention for direct USD-base
    # pairs: convert quote P&L once at the entry execution price. Native
    # non-USD crosses use the separately timestamped causal conversion quote.
    pnl_usd = _pnl_usd(pnl_quote, base, quote, entry, exit_conversion)
    return Batch5TradeResult(
        family=intent.family,
        strategy_id=intent.strategy_id,
        sleeve_id=intent.sleeve_id,
        instrument_id=intent.instrument_id,
        signal_timestamp=intent.signal_timestamp,
        actual_entry_timestamp=entry_timestamp,
        actual_exit_timestamp=exit_timestamp,
        direction=intent.direction,
        quantity=quantity,
        entry_price=entry,
        exit_price=exit_price,
        pnl_usd=pnl_usd,
        return_on_reference_notional=pnl_usd / reference_notional_usd,
        holding_seconds=int((exit_timestamp - entry_timestamp).total_seconds()),
        cohort_id=intent.cohort_id,
        metadata=dict(intent.metadata),
    )


def _skip(intent: TradeIntent, reason: str) -> Batch5SkipRecord:
    return Batch5SkipRecord(
        intent.family,
        intent.strategy_id,
        intent.sleeve_id,
        intent.instrument_id,
        intent.signal_timestamp,
        intent.cohort_id,
        reason,
    )
