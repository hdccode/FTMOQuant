from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pandas as pd
import pytest

from ftmoquant.data.dukascopy import SourceBar
from ftmoquant.data.instruments import AUDCAD_OANDA_SPEC, to_nautilus_bars
from ftmoquant.research.alpha_lab.batch5_daily import (
    NEW_YORK,
    build_completed_fx_days,
    build_completed_fx_days_with_diagnostics,
    build_provider_aware_fx_days,
    build_provider_aware_fx_days_with_diagnostics,
    ny_fx_boundary,
)


def _frames(
    timestamps: list[datetime], closes: list[float]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.DatetimeIndex(timestamps)
    bid = pd.DataFrame(
        {name: closes for name in ("open", "high", "low", "close")}, index=index
    )
    ask = bid + 0.0002
    return bid, ask


def _old_start_label_builder_count(
    bid: pd.DataFrame, ask: pd.DataFrame
) -> int:
    midpoint = (bid["close"] + ask["close"]) / 2
    boundaries = {
        timestamp.to_pydatetime().astimezone(NEW_YORK).date(): midpoint.loc[timestamp]
        for timestamp in bid.index
        if timestamp.to_pydatetime().astimezone(NEW_YORK).hour == 17
        and timestamp.minute == 0
    }
    return sum(day - timedelta(days=1) in boundaries for day in boundaries)


def test_canonical_oanda_m1_timestamp_is_start_and_init_is_completion() -> None:
    start = datetime(2021, 1, 5, 21, 59, tzinfo=UTC)
    source = SourceBar(
        timestamp=start,
        open=Decimal("1.0"),
        high=Decimal("1.1"),
        low=Decimal("0.9"),
        close=Decimal("1.05"),
        volume=Decimal(1),
    )
    bar = to_nautilus_bars((source,), "BID", AUDCAD_OANDA_SPEC)[0]
    assert bar.ts_event == int(start.timestamp() * 1_000_000_000)
    assert bar.ts_init == int(
        (start + timedelta(minutes=1)).timestamp() * 1_000_000_000
    )


def test_oanda_start_time_boundary_uses_t_minus_one_and_never_t() -> None:
    opening = ny_fx_boundary(date(2021, 1, 4))
    closing = ny_fx_boundary(date(2021, 1, 5))
    bid, ask = _frames(
        [opening - timedelta(minutes=1), closing - timedelta(minutes=1), closing],
        [1.0, 1.2, 9.0],
    )
    days = build_completed_fx_days("AUD/CAD.OANDA", bid, ask)
    assert len(days) == 1
    assert days[0].end_utc == closing
    assert days[0].close_mid == Decimal("1.2001")
    assert days[0].close_mid != Decimal("9.0001")


def test_pre_correction_differential_rejects_while_corrected_accepts() -> None:
    opening = ny_fx_boundary(date(2021, 1, 4))
    closing = ny_fx_boundary(date(2021, 1, 5))
    bid, ask = _frames(
        [opening - timedelta(minutes=1), closing - timedelta(minutes=1)], [1.0, 1.1]
    )
    assert _old_start_label_builder_count(bid, ask) == 0
    assert len(build_completed_fx_days("AUD/CAD.OANDA", bid, ask)) == 1


def test_interior_gap_does_not_invalidate_completed_boundary_day() -> None:
    opening = ny_fx_boundary(date(2021, 2, 1))
    closing = ny_fx_boundary(date(2021, 2, 2))
    bid, ask = _frames(
        [
            opening - timedelta(minutes=1),
            opening + timedelta(hours=7),
            closing - timedelta(minutes=1),
        ],
        [1.0, 1.05, 1.1],
    )
    days, diagnostics = build_completed_fx_days_with_diagnostics(
        "EUR/USD.OANDA", bid, ask
    )
    assert len(days) == 1
    assert diagnostics.accepted_completed_day_count == 1


def test_missing_boundary_and_weekend_fail_closed_without_fabrication() -> None:
    friday = ny_fx_boundary(date(2021, 1, 8))
    monday = ny_fx_boundary(date(2021, 1, 11))
    bid, ask = _frames(
        [friday - timedelta(minutes=1), monday - timedelta(hours=1)], [1.0, 1.1]
    )
    days, diagnostics = build_completed_fx_days_with_diagnostics(
        "EUR/USD.OANDA", bid, ask
    )
    assert days == ()
    assert diagnostics.no_observation_count >= 1
    assert diagnostics.rejected_boundary_close_count >= 1


def test_dst_boundaries_remain_local_1700_and_can_span_23_hours() -> None:
    opening = ny_fx_boundary(date(2021, 3, 13))
    closing = ny_fx_boundary(date(2021, 3, 14))
    bid, ask = _frames(
        [opening - timedelta(minutes=1), closing - timedelta(minutes=1)], [1.0, 1.1]
    )
    day = build_completed_fx_days("EUR/USD.OANDA", bid, ask)[0]
    assert day.start_utc == datetime(2021, 3, 13, 22, tzinfo=UTC)
    assert day.end_utc == datetime(2021, 3, 14, 21, tzinfo=UTC)
    assert day.end_utc - day.start_utc == timedelta(hours=23)


def test_provider_aware_close_never_uses_post_boundary_observation() -> None:
    opening = ny_fx_boundary(date(2022, 1, 3))
    closing = ny_fx_boundary(date(2022, 1, 4))
    bid, ask = _frames(
        [
            opening - timedelta(minutes=2),
            opening + timedelta(minutes=3),
            closing - timedelta(minutes=2),
            closing + timedelta(minutes=3),
        ],
        [0.8, 1.0, 1.2, 9.0],
    )
    day = build_provider_aware_fx_days("AUD/CAD.OANDA", bid, ask)[0]
    assert day.open_mid == Decimal("1.0001")
    assert day.close_mid == Decimal("1.2001")
    assert day.close_mid != Decimal("9.0001")


def test_provider_aware_2022_rollover_gap_yields_one_valid_fx_day() -> None:
    opening = ny_fx_boundary(date(2022, 2, 1))
    closing = ny_fx_boundary(date(2022, 2, 2))
    bid, ask = _frames(
        [
            opening - timedelta(minutes=2),
            opening + timedelta(minutes=3),
            closing - timedelta(minutes=2),
            closing + timedelta(minutes=3),
        ],
        [0.8, 1.0, 1.1, 1.2],
    )
    days, diagnostics = build_provider_aware_fx_days_with_diagnostics(
        "EUR/JPY.OANDA", bid, ask
    )
    day = next(row for row in days if row.local_close_date == date(2022, 2, 2))
    assert day.open_mid == Decimal("1.0001")
    assert day.close_mid == Decimal("1.1001")
    assert diagnostics.provider_rollover_gap_day_count >= 1


def test_provider_aware_exact_boundary_candle_is_causal() -> None:
    opening = ny_fx_boundary(date(2021, 2, 1))
    closing = ny_fx_boundary(date(2021, 2, 2))
    bid, ask = _frames(
        [opening, closing - timedelta(minutes=1), closing],
        [1.0, 1.1, 8.0],
    )
    day = build_provider_aware_fx_days("EUR/USD.OANDA", bid, ask)[0]
    assert day.open_mid == Decimal("1.0001")
    assert day.close_mid == Decimal("1.1001")


def test_provider_aware_builder_refuses_unpaired_data_instead_of_filling() -> None:
    opening = ny_fx_boundary(date(2022, 2, 1))
    closing = ny_fx_boundary(date(2022, 2, 2))
    bid, ask = _frames(
        [opening, closing - timedelta(minutes=2), closing],
        [1.0, 1.1, 1.2],
    )
    ask = ask.drop(index=closing - timedelta(minutes=2))
    with pytest.raises(ValueError, match="identical paired index"):
        build_provider_aware_fx_days("EUR/USD.OANDA", bid, ask)


def test_provider_aware_weekend_and_full_day_closure_are_not_bridged() -> None:
    friday = ny_fx_boundary(date(2022, 1, 7))
    sunday = ny_fx_boundary(date(2022, 1, 9))
    monday = ny_fx_boundary(date(2022, 1, 10))
    tuesday = ny_fx_boundary(date(2022, 1, 11))
    bid, ask = _frames(
        [
            friday - timedelta(minutes=2),
            sunday + timedelta(minutes=3),
            monday - timedelta(minutes=2),
            tuesday + timedelta(minutes=3),
        ],
        [1.0, 1.1, 1.2, 1.3],
    )
    days = build_provider_aware_fx_days("USD/CAD.OANDA", bid, ask)
    assert [row.local_close_date for row in days] == [date(2022, 1, 10)]
    assert all(row.local_close_date.weekday() < 5 for row in days)


def test_provider_aware_dst_boundaries_stay_at_new_york_1700() -> None:
    opening = ny_fx_boundary(date(2021, 3, 14))
    closing = ny_fx_boundary(date(2021, 3, 15))
    bid, ask = _frames(
        [
            opening - timedelta(minutes=2),
            opening + timedelta(minutes=3),
            closing - timedelta(minutes=2),
            closing + timedelta(minutes=3),
        ],
        [0.8, 1.0, 1.1, 1.2],
    )
    day = build_provider_aware_fx_days("EUR/USD.OANDA", bid, ask)[0]
    assert day.start_utc == datetime(2021, 3, 14, 21, tzinfo=UTC)
    assert day.end_utc == datetime(2021, 3, 15, 21, tzinfo=UTC)
    assert day.start_utc.astimezone(NEW_YORK).hour == 17
    assert day.end_utc.astimezone(NEW_YORK).hour == 17
