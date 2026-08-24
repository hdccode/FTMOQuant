"""DST-safe completed New York FX-day value types for Batch 5."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pandas as pd  # type: ignore[import-untyped]

from ftmoquant.research.alpha_lab.batch4_execution import _validate_frames

NEW_YORK = ZoneInfo("America/New_York")
ONE_MINUTE = timedelta(minutes=1)


class Batch5DailyError(ValueError):
    """Raised when a purported completed FX day violates frozen boundaries."""


def ny_fx_boundary(local_date: date) -> datetime:
    """Return that civil date's 17:00 New York boundary in UTC."""

    return datetime.combine(local_date, time(17), NEW_YORK).astimezone(UTC)


@dataclass(frozen=True, slots=True)
class CompletedFxDay:
    instrument_id: str
    local_close_date: date
    start_utc: datetime
    end_utc: datetime
    open_mid: Decimal
    close_mid: Decimal

    def __post_init__(self) -> None:
        expected_end = ny_fx_boundary(self.local_close_date)
        expected_start = ny_fx_boundary(self.local_close_date - timedelta(days=1))
        if self.start_utc != expected_start or self.end_utc != expected_end:
            raise Batch5DailyError(
                "FX day must span consecutive 17:00 New York boundaries"
            )
        if self.open_mid <= 0 or self.close_mid <= 0:
            raise Batch5DailyError("daily midpoint prices must be positive")

    @property
    def open_to_close_return(self) -> Decimal:
        return self.close_mid / self.open_mid - Decimal(1)


@dataclass(frozen=True, slots=True)
class FxDayBuildDiagnostics:
    """Report-only structural funnel for native NY-boundary construction."""

    candidate_fx_day_count: int
    days_with_observations: int
    accepted_completed_day_count: int
    rejected_boundary_open_count: int
    rejected_boundary_close_count: int
    no_observation_count: int
    first_accepted_day: date | None
    last_accepted_day: date | None
    provider_rollover_gap_day_count: int = 0
    missing_market_data_day_count: int = 0


def _first_candidate_close(timestamp: datetime) -> date:
    local = timestamp.astimezone(NEW_YORK)
    boundary = datetime.combine(local.date(), time(17), NEW_YORK)
    return local.date() if local <= boundary else local.date() + timedelta(days=1)


def _last_candidate_close(timestamp: datetime) -> date:
    local = timestamp.astimezone(NEW_YORK)
    boundary = datetime.combine(local.date(), time(17), NEW_YORK)
    return local.date() if local >= boundary else local.date() - timedelta(days=1)


def _observation_fx_close_date(completion: datetime) -> date:
    local = completion.astimezone(NEW_YORK)
    boundary = datetime.combine(local.date(), time(17), NEW_YORK)
    return local.date() if local <= boundary else local.date() + timedelta(days=1)


def build_completed_fx_days_with_diagnostics(
    instrument_id: str,
    bid_m1: pd.DataFrame,
    ask_m1: pd.DataFrame,
) -> tuple[tuple[CompletedFxDay, ...], FxDayBuildDiagnostics]:
    """Build days from native M1 candles completing at each NY boundary.

    Canonical OANDA ``ts_event`` is the candle start.  A boundary observation
    at ``T`` is therefore the close of the paired candle ``[T-1min, T)``.
    Missing interior candles do not invalidate the day; missing boundary
    candles fail closed without filling or interpolation.
    """

    index = _validate_frames(bid_m1, ask_m1)
    midpoint = (bid_m1["close"] + ask_m1["close"]) / 2
    boundary_prices: dict[date, Decimal] = {}
    observed_days: set[date] = set()
    for timestamp in index:
        start = timestamp.to_pydatetime()
        completion = start + ONE_MINUTE
        observed_days.add(_observation_fx_close_date(completion))
        local_completion = completion.astimezone(NEW_YORK)
        if (
            local_completion.hour == 17
            and local_completion.minute == 0
            and local_completion.second == 0
        ):
            boundary_prices[local_completion.date()] = Decimal(
                str(midpoint.loc[timestamp])
            )

    first_completion = index[0].to_pydatetime() + ONE_MINUTE
    last_completion = index[-1].to_pydatetime() + ONE_MINUTE
    first_candidate = _first_candidate_close(first_completion)
    last_candidate = _last_candidate_close(last_completion)
    candidate_dates: list[date] = []
    current = first_candidate
    while current <= last_candidate:
        candidate_dates.append(current)
        current += timedelta(days=1)

    days: list[CompletedFxDay] = []
    rejected_open = 0
    rejected_close = 0
    for local_close_date in candidate_dates:
        prior = local_close_date - timedelta(days=1)
        has_open = prior in boundary_prices
        has_close = local_close_date in boundary_prices
        rejected_open += not has_open
        rejected_close += not has_close
        if not (has_open and has_close):
            continue
        days.append(
            CompletedFxDay(
                instrument_id=instrument_id,
                local_close_date=local_close_date,
                start_utc=ny_fx_boundary(prior),
                end_utc=ny_fx_boundary(local_close_date),
                open_mid=boundary_prices[prior],
                close_mid=boundary_prices[local_close_date],
            )
        )
    accepted_dates = [row.local_close_date for row in days]
    diagnostics = FxDayBuildDiagnostics(
        candidate_fx_day_count=len(candidate_dates),
        days_with_observations=sum(day in observed_days for day in candidate_dates),
        accepted_completed_day_count=len(days),
        rejected_boundary_open_count=rejected_open,
        rejected_boundary_close_count=rejected_close,
        no_observation_count=sum(day not in observed_days for day in candidate_dates),
        first_accepted_day=accepted_dates[0] if accepted_dates else None,
        last_accepted_day=accepted_dates[-1] if accepted_dates else None,
    )
    return tuple(days), diagnostics


def build_completed_fx_days(
    instrument_id: str,
    bid_m1: pd.DataFrame,
    ask_m1: pd.DataFrame,
) -> tuple[CompletedFxDay, ...]:
    """Materialize paired-M1 17:00 New York boundary-to-boundary days.

    A day exists only when native paired M1 candles complete at both civil
    boundaries.  Interior gaps do not invalidate an otherwise usable day.
    There is no resampling, forward filling, or weekend interpolation.
    """

    days, _ = build_completed_fx_days_with_diagnostics(
        instrument_id, bid_m1, ask_m1
    )
    return days


def _belongs_to_open_boundary(timestamp: datetime, boundary: datetime) -> bool:
    """Require an opening observation on the boundary's New York civil date."""

    local_timestamp = timestamp.astimezone(NEW_YORK)
    local_boundary = boundary.astimezone(NEW_YORK)
    return (
        local_timestamp.date() == local_boundary.date()
        and local_timestamp >= local_boundary
    )


def _belongs_to_close_boundary(completion: datetime, boundary: datetime) -> bool:
    """Require a completed close on the boundary's New York civil date."""

    local_completion = completion.astimezone(NEW_YORK)
    local_boundary = boundary.astimezone(NEW_YORK)
    return (
        local_completion.date() == local_boundary.date()
        and local_completion <= local_boundary
    )


def build_provider_aware_fx_days_with_diagnostics(
    instrument_id: str,
    bid_m1: pd.DataFrame,
    ask_m1: pd.DataFrame,
) -> tuple[tuple[CompletedFxDay, ...], FxDayBuildDiagnostics]:
    """Build provider-aware weekday FX days without crossing civil closures.

    For ``[T0, T1]``, the open is the first genuine paired M1 open whose start
    is at or after ``T0`` and the close is the last genuine paired M1 close
    whose completion is at or before ``T1``.  Both observations must remain
    inside the intended interval and on the New York civil date belonging to
    their respective boundary.  Only Monday-through-Friday close dates are
    candidates, so no Saturday or Sunday FX day can be invented.
    """

    index = _validate_frames(bid_m1, ask_m1)
    open_midpoint = (bid_m1["open"] + ask_m1["open"]) / 2
    close_midpoint = (bid_m1["close"] + ask_m1["close"]) / 2
    first_start = index[0].to_pydatetime()
    last_completion = index[-1].to_pydatetime() + ONE_MINUTE
    first_candidate = _first_candidate_close(first_start)
    last_candidate = last_completion.astimezone(NEW_YORK).date()
    candidate_dates: list[date] = []
    current = first_candidate
    while current <= last_candidate:
        if current.weekday() < 5:
            candidate_dates.append(current)
        current += timedelta(days=1)

    days: list[CompletedFxDay] = []
    days_with_observations = 0
    rejected_open = 0
    rejected_close = 0
    rollover_gap_days = 0
    missing_market_days = 0
    for local_close_date in candidate_dates:
        start = ny_fx_boundary(local_close_date - timedelta(days=1))
        end = ny_fx_boundary(local_close_date)
        left = index.searchsorted(pd.Timestamp(start), side="left")
        right = index.searchsorted(pd.Timestamp(end), side="left")
        if left >= right:
            rejected_open += 1
            rejected_close += 1
            missing_market_days += 1
            continue
        days_with_observations += 1
        open_timestamp = index[left].to_pydatetime()
        close_timestamp = index[right - 1].to_pydatetime()
        close_completion = close_timestamp + ONE_MINUTE
        has_open = (
            first_start <= start
            and _belongs_to_open_boundary(open_timestamp, start)
        )
        has_close = (
            index[-1].to_pydatetime() >= end
            and _belongs_to_close_boundary(close_completion, end)
        )
        rejected_open += not has_open
        rejected_close += not has_close
        if not (has_open and has_close):
            missing_market_days += 1
            continue
        if open_timestamp > start or close_completion < end:
            rollover_gap_days += 1
        days.append(
            CompletedFxDay(
                instrument_id=instrument_id,
                local_close_date=local_close_date,
                start_utc=start,
                end_utc=end,
                open_mid=Decimal(str(open_midpoint.iloc[left])),
                close_mid=Decimal(str(close_midpoint.iloc[right - 1])),
            )
        )

    accepted_dates = [row.local_close_date for row in days]
    diagnostics = FxDayBuildDiagnostics(
        candidate_fx_day_count=len(candidate_dates),
        days_with_observations=days_with_observations,
        accepted_completed_day_count=len(days),
        rejected_boundary_open_count=rejected_open,
        rejected_boundary_close_count=rejected_close,
        no_observation_count=len(candidate_dates) - days_with_observations,
        first_accepted_day=accepted_dates[0] if accepted_dates else None,
        last_accepted_day=accepted_dates[-1] if accepted_dates else None,
        provider_rollover_gap_day_count=rollover_gap_days,
        missing_market_data_day_count=missing_market_days,
    )
    return tuple(days), diagnostics


def build_provider_aware_fx_days(
    instrument_id: str,
    bid_m1: pd.DataFrame,
    ask_m1: pd.DataFrame,
) -> tuple[CompletedFxDay, ...]:
    """Materialize the frozen provider-aware B5B/B5C weekday FX days."""

    days, _ = build_provider_aware_fx_days_with_diagnostics(
        instrument_id, bid_m1, ask_m1
    )
    return days
