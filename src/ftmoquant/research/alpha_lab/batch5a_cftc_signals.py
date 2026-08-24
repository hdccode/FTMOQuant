"""Pure causal monthly signal construction for frozen Batch 5A."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Literal

from ftmoquant.research.alpha_lab.batch5_cftc_availability import verify_amendment
from ftmoquant.research.alpha_lab.batch5_preregistration import (
    FAMILY_B5A,
    verify_preregistration,
)

Direction = Literal["BUY", "SELL"]


class Batch5ACftcSignalError(ValueError):
    """Raised on methodology drift or malformed CFTC observations."""


@dataclass(frozen=True, order=True, slots=True)
class YearMonth:
    year: int
    month: int

    def __post_init__(self) -> None:
        if not 1 <= self.month <= 12:
            raise Batch5ACftcSignalError("month must be 1..12")

    def add(self, months: int) -> YearMonth:
        index = self.year * 12 + self.month - 1 + months
        year, month = divmod(index, 12)
        return YearMonth(year, month + 1)


@dataclass(frozen=True, slots=True)
class CftcDealerObservation:
    currency: str
    report_date: date
    dealer_long: Decimal
    dealer_short: Decimal
    open_interest: Decimal
    availability_timestamp: datetime | None
    availability_status: str
    source_vintage: str

    def __post_init__(self) -> None:
        if (
            self.availability_timestamp is not None
            and self.availability_timestamp.tzinfo is None
        ):
            raise Batch5ACftcSignalError("availability timestamp must be aware")
        if self.open_interest <= 0:
            raise Batch5ACftcSignalError("open interest must be positive")
        if (
            self.availability_status == "UNRESOLVED"
            and self.availability_timestamp is not None
        ):
            raise Batch5ACftcSignalError(
                "UNRESOLVED observations cannot have availability"
            )

    @property
    def month(self) -> YearMonth:
        return YearMonth(self.report_date.year, self.report_date.month)

    @property
    def dealer_net(self) -> Decimal:
        return self.dealer_long - self.dealer_short


@dataclass(frozen=True, slots=True)
class B5ASignal:
    family: str
    strategy_id: str
    sleeve_id: str
    instrument_id: str
    currency: str
    formation_month: YearMonth
    formation_timestamp: datetime
    direction: Direction
    scaled_dealer_net: Decimal
    dealer_position_change: Decimal
    cftc_report_date: date
    cftc_availability_timestamp: datetime


def _frozen_sleeves() -> dict[str, dict[str, str]]:
    verify_amendment()
    document = verify_preregistration()
    rows = document["families"][FAMILY_B5A]["sleeves"]
    return {str(row["currency_k"]): row for row in rows}


def _mean(values: Sequence[Decimal]) -> Decimal:
    return sum(values, Decimal(0)) / len(values)


def signal_for_month(
    observations: Sequence[CftcDealerObservation],
    *,
    currency: str,
    formation_month: YearMonth,
    formation_timestamp: datetime,
) -> B5ASignal | None:
    """Form one strict-sign signal using only rows public by formation time."""

    if formation_timestamp.tzinfo is None:
        raise Batch5ACftcSignalError("formation timestamp must be aware")
    following = formation_month.add(1)
    earliest = datetime(following.year, following.month, 1, tzinfo=UTC)
    if formation_timestamp < earliest:
        raise Batch5ACftcSignalError("monthly formation cannot precede month end")
    sleeves = _frozen_sleeves()
    sleeve = sleeves.get(currency)
    if sleeve is None:
        raise Batch5ACftcSignalError("currency is outside the frozen seven sleeves")
    visible = [
        row
        for row in observations
        if row.currency == currency
        and row.availability_timestamp is not None
        and row.availability_timestamp <= formation_timestamp
        and row.month <= formation_month
    ]
    by_month: dict[YearMonth, list[CftcDealerObservation]] = defaultdict(list)
    for row in visible:
        by_month[row.month].append(row)
    required = [formation_month.add(offset) for offset in range(-12, 1)]
    if any(not by_month[month] for month in required):
        return None

    def scaled(month: YearMonth) -> Decimal:
        monthly_net = _mean([row.dealer_net for row in by_month[month]])
        oi_months = [month.add(offset) for offset in range(-11, 1)]
        if any(not by_month[item] for item in oi_months):
            raise Batch5ACftcSignalError("required causal OI month disappeared")
        monthly_oi = [
            _mean([row.open_interest for row in by_month[item]]) for item in oi_months
        ]
        return Decimal(100) * monthly_net / _mean(monthly_oi)

    current = scaled(formation_month)
    previous = scaled(formation_month.add(-1))
    change = current - previous
    if change == 0:
        return None
    direction_key = (
        "positive_delta_position_side" if change > 0 else "negative_delta_position_side"
    )
    current_rows = by_month[formation_month]
    latest = max(current_rows, key=lambda row: row.report_date)
    assert latest.availability_timestamp is not None
    return B5ASignal(
        family=FAMILY_B5A,
        strategy_id="B5A_FROZEN_CFTC_DEALER_DEMAND_SHOCK",
        sleeve_id=str(sleeve["sleeve_id"]),
        instrument_id=str(sleeve["spot_instrument"]),
        currency=currency,
        formation_month=formation_month,
        formation_timestamp=formation_timestamp,
        direction=str(sleeve[direction_key]),  # type: ignore[arg-type]
        scaled_dealer_net=current,
        dealer_position_change=change,
        cftc_report_date=latest.report_date,
        cftc_availability_timestamp=latest.availability_timestamp,
    )
