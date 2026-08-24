from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pandas as pd  # type: ignore[import-untyped]

from ftmoquant.research.alpha_lab.batch5_execution import (
    Batch5TradeResult,
    add_calendar_months,
)
from ftmoquant.research.alpha_lab.batch5a_cftc_execution import execute_cohorts
from ftmoquant.research.alpha_lab.batch5a_cftc_signals import (
    CftcDealerObservation,
    YearMonth,
    signal_for_month,
)


def observations(
    currency: str,
    *,
    current_net: Decimal,
    current_available: datetime | None,
    end_month: YearMonth = YearMonth(2021, 1),
) -> list[CftcDealerObservation]:
    rows = []
    first = end_month.add(-12)
    for offset in range(13):
        month = first.add(offset)
        available = datetime(month.year, month.month, 20, tzinfo=UTC)
        net = Decimal(1) if offset < 12 else current_net
        rows.append(
            CftcDealerObservation(
                currency,
                date(month.year, month.month, 10),
                Decimal(10) + net,
                Decimal(10),
                Decimal(100),
                current_available if offset == 12 else available,
                "UNRESOLVED"
                if offset == 12 and current_available is None
                else "VERIFIED_OFFICIAL",
                "fixture",
            )
        )
    return rows


def test_positive_negative_zero_and_futures_spot_sign_mapping() -> None:
    formation = datetime(2021, 2, 1, tzinfo=UTC)
    positive = signal_for_month(
        observations(
            "EUR",
            current_net=Decimal(2),
            current_available=formation - timedelta(days=1),
        ),
        currency="EUR",
        formation_month=YearMonth(2021, 1),
        formation_timestamp=formation,
    )
    negative = signal_for_month(
        observations(
            "EUR",
            current_net=Decimal(0),
            current_available=formation - timedelta(days=1),
        ),
        currency="EUR",
        formation_month=YearMonth(2021, 1),
        formation_timestamp=formation,
    )
    jpy = signal_for_month(
        observations(
            "JPY",
            current_net=Decimal(2),
            current_available=formation - timedelta(days=1),
        ),
        currency="JPY",
        formation_month=YearMonth(2021, 1),
        formation_timestamp=formation,
    )
    zero = signal_for_month(
        observations(
            "EUR",
            current_net=Decimal(1),
            current_available=formation - timedelta(days=1),
        ),
        currency="EUR",
        formation_month=YearMonth(2021, 1),
        formation_timestamp=formation,
    )
    assert positive is not None and positive.direction == "BUY"
    assert negative is not None and negative.direction == "SELL"
    assert jpy is not None and jpy.direction == "SELL"
    assert zero is None


def test_unreleased_unresolved_and_delayed_rows_are_invisible() -> None:
    formation = datetime(2021, 2, 1, tzinfo=UTC)
    unresolved = observations("EUR", current_net=Decimal(2), current_available=None)
    delayed = observations(
        "EUR", current_net=Decimal(2), current_available=formation + timedelta(days=7)
    )
    assert (
        signal_for_month(
            unresolved,
            currency="EUR",
            formation_month=YearMonth(2021, 1),
            formation_timestamp=formation,
        )
        is None
    )
    assert (
        signal_for_month(
            delayed,
            currency="EUR",
            formation_month=YearMonth(2021, 1),
            formation_timestamp=formation,
        )
        is None
    )


def test_three_month_cohorts_overlap_and_expire_from_actual_entry() -> None:
    signals = []
    for offset in range(3):
        month = YearMonth(2021, 1).add(offset)
        next_month = month.add(1)
        timestamp = datetime(next_month.year, next_month.month, 1, 20, tzinfo=UTC)
        signal = signal_for_month(
            observations(
                "EUR",
                current_net=Decimal(2),
                current_available=timestamp - timedelta(days=1),
                end_month=month,
            ),
            currency="EUR",
            formation_month=month,
            formation_timestamp=timestamp,
        )
        assert signal is not None
        signals.append(signal)
    timestamps = pd.date_range("2021-01-31 20:00", "2021-07-03", freq="12h", tz="UTC")
    bid = pd.DataFrame(
        {name: 1.1 for name in ("open", "high", "low", "close")}, index=timestamps
    )
    ask = pd.DataFrame(
        {name: 1.1001 for name in ("open", "high", "low", "close")}, index=timestamps
    )
    results = execute_cohorts(signals, native_frames={"EUR/USD.OANDA": (bid, ask)})
    trades = [row for row in results if isinstance(row, Batch5TradeResult)]
    assert len(trades) == 3
    assert len({row.cohort_id for row in trades}) == 3
    assert trades[1].actual_entry_timestamp < trades[0].actual_exit_timestamp
    assert trades[2].actual_entry_timestamp < trades[1].actual_exit_timestamp
    for trade in trades:
        assert (
            trade.metadata["cohort_expiry"]
            == add_calendar_months(trade.actual_entry_timestamp, 3).isoformat()
        )
