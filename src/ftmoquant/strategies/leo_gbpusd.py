"""Synthetic-fixture state machine for the frozen ``leo_gbpusd_v1`` contract.

Signals use completed 15-minute midpoint OHLC pairs, as in the existing G1
signal convention. Native G0.7 execution remains responsible for production
fills and costs; this pure component uses the executable bid/ask close supplied
by its fixture to bind the frozen sweep stop and 3R target.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from enum import StrEnum
from zoneinfo import ZoneInfo

from ftmoquant.research.leo_gbpusd_spec import (
    LEO_GBPUSD_CONFIG_SHA256,
    LeoGbpUsdSpec,
    leo_gbpusd_config_sha256,
)
from ftmoquant.strategies.trend_pullback import Direction, PriceBar

_INSTRUMENT_ID = "GBP/USD.DUKASCOPY"
_LONDON = ZoneInfo("Europe/London")
_BAR_INTERVAL = timedelta(minutes=15)
_ASIA_START = time(0, 0)
_ASIA_END = time(8, 0)
_LONDON_REFERENCE_START = time(8, 0)
_LONDON_REFERENCE_END = time(13, 0)
_LONDON_ENTRY_START = time(9, 0)
_LONDON_ENTRY_END = time(11, 0)
_NEW_YORK_ENTRY_START = time(14, 0)
_NEW_YORK_ENTRY_END = time(16, 0)


class LeoGbpUsdValidationError(ValueError):
    """Raised when a synthetic completed-bar stream violates frozen semantics."""


class LeoExitReason(StrEnum):
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    SESSION_WINDOW_END = "session_window_end"


class LeoNamedSession(StrEnum):
    LONDON = "london"
    NEW_YORK = "new_york"


@dataclass(frozen=True, slots=True)
class LeoCompleted15mBar:
    """One causally available completed GBP/USD 15-minute bid/ask bar pair."""

    instrument_id: str
    start_time_utc: datetime
    end_time_utc: datetime
    available_at_utc: datetime
    bid: PriceBar
    ask: PriceBar

    def validate(self) -> None:
        if self.instrument_id != _INSTRUMENT_ID:
            raise LeoGbpUsdValidationError("bar instrument must be frozen GBP/USD")
        start = _utc(self.start_time_utc, "start_time_utc")
        end = _utc(self.end_time_utc, "end_time_utc")
        available = _utc(self.available_at_utc, "available_at_utc")
        if end - start != _BAR_INTERVAL:
            raise LeoGbpUsdValidationError("bar must be a completed 15m interval")
        if available < end:
            raise LeoGbpUsdValidationError("bar cannot be available before completion")
        if end.minute % 15 or end.second or end.microsecond:
            raise LeoGbpUsdValidationError("bar end must align to a 15m boundary")
        if any(
            ask < bid
            for bid, ask in zip(
                (self.bid.open, self.bid.high, self.bid.low, self.bid.close),
                (self.ask.open, self.ask.high, self.ask.low, self.ask.close),
                strict=True,
            )
        ):
            raise LeoGbpUsdValidationError("ASK OHLC cannot be below BID OHLC")

    @property
    def midpoint(self) -> PriceBar:
        two = Decimal(2)
        return PriceBar(
            open=(self.bid.open + self.ask.open) / two,
            high=(self.bid.high + self.ask.high) / two,
            low=(self.bid.low + self.ask.low) / two,
            close=(self.bid.close + self.ask.close) / two,
        )


@dataclass(frozen=True, slots=True)
class LeoSignal:
    direction: Direction
    named_session: LeoNamedSession
    session_date: date
    signal_end_time_utc: datetime
    signal_information_time_utc: datetime
    stop_price: Decimal
    reference_high: Decimal
    reference_low: Decimal


@dataclass(frozen=True, slots=True)
class LeoEntry:
    direction: Direction
    named_session: LeoNamedSession
    session_date: date
    signal_information_time_utc: datetime
    entry_information_time_utc: datetime
    entry_price: Decimal
    stop_price: Decimal
    target_price: Decimal


@dataclass(frozen=True, slots=True)
class LeoExit:
    direction: Direction
    reason: LeoExitReason
    exit_information_time_utc: datetime
    exit_price: Decimal
    stop_price: Decimal
    target_price: Decimal


LeoAction = LeoSignal | LeoEntry | LeoExit


@dataclass(slots=True)
class _ReferenceRange:
    expected_end_utc: datetime
    expected_count: int
    high: Decimal | None = None
    low: Decimal | None = None
    count: int = 0
    complete: bool = True

    def add(self, bar: LeoCompleted15mBar) -> None:
        if bar.end_time_utc != self.expected_end_utc:
            self.complete = False
        self.expected_end_utc += _BAR_INTERVAL
        midpoint = bar.midpoint
        self.high = (
            midpoint.high if self.high is None else max(self.high, midpoint.high)
        )
        self.low = midpoint.low if self.low is None else min(self.low, midpoint.low)
        self.count += 1

    @property
    def valid(self) -> bool:
        return (
            self.complete
            and self.count == self.expected_count
            and self.high is not None
            and self.low is not None
        )


@dataclass(slots=True)
class _DayState:
    asia: _ReferenceRange
    london: _ReferenceRange


@dataclass(frozen=True, slots=True)
class _PendingEntry:
    signal: LeoSignal
    entry_window_end_utc: datetime


@dataclass(frozen=True, slots=True)
class _Position:
    entry: LeoEntry
    entry_window_end_utc: datetime


class LeoGbpUsdStateMachine:
    """One-position, strict-later 15-minute realization of the frozen v1 spec."""

    def __init__(self, spec: LeoGbpUsdSpec) -> None:
        if leo_gbpusd_config_sha256(spec) != LEO_GBPUSD_CONFIG_SHA256:
            raise LeoGbpUsdValidationError("strategy config hash is frozen")
        self.spec = spec
        self._day: date | None = None
        self._state: _DayState | None = None
        self._pending: _PendingEntry | None = None
        self._position: _Position | None = None
        self._entered_sessions: set[tuple[date, LeoNamedSession]] = set()
        self._invalid_stop_distance_skip_count = 0
        self._last_end_utc: datetime | None = None
        self._last_available_utc: datetime | None = None

    @property
    def pending_signal(self) -> LeoSignal | None:
        return None if self._pending is None else self._pending.signal

    @property
    def active_entry(self) -> LeoEntry | None:
        return None if self._position is None else self._position.entry

    @property
    def invalid_stop_distance_skip_count(self) -> int:
        """Number of strict-later fills rejected by the frozen stop rule."""
        return self._invalid_stop_distance_skip_count

    @property
    def asia_reference(self) -> tuple[Decimal, Decimal] | None:
        return _reference(self._state, "asia")

    @property
    def london_reference(self) -> tuple[Decimal, Decimal] | None:
        return _reference(self._state, "london")

    def on_bar(self, bar: LeoCompleted15mBar) -> tuple[LeoAction, ...]:
        """Consume one completed bar; no data source, backtest, or P/L is read."""
        bar.validate()
        self._validate_order(bar)
        local_end = bar.end_time_utc.astimezone(_LONDON)
        self._reset_for_day(local_end.date())
        actions: list[LeoAction] = []

        entered = self._try_enter(bar)
        if entered is not None:
            actions.append(entered)
        if self._position is not None and entered is None:
            exit_action = self._evaluate_exit(bar)
            if exit_action is not None:
                actions.append(exit_action)

        self._accumulate_reference(bar, local_end)
        signal = self._detect_signal(bar, local_end)
        if signal is not None:
            self._pending = _PendingEntry(signal, _entry_window_end(signal))
            actions.append(signal)
        return tuple(actions)

    def _try_enter(self, bar: LeoCompleted15mBar) -> LeoEntry | None:
        pending = self._pending
        if pending is None:
            return None
        if bar.available_at_utc <= pending.signal.signal_information_time_utc:
            return None
        self._pending = None
        if bar.available_at_utc >= pending.entry_window_end_utc:
            return None
        signal = pending.signal
        entry_price = (
            bar.ask.close if signal.direction is Direction.LONG else bar.bid.close
        )
        distance = (
            entry_price - signal.stop_price
            if signal.direction is Direction.LONG
            else signal.stop_price - entry_price
        )
        if not distance.is_finite() or distance <= 0:
            self._invalid_stop_distance_skip_count += 1
            return None
        target = (
            entry_price + Decimal(3) * distance
            if signal.direction is Direction.LONG
            else entry_price - Decimal(3) * distance
        )
        if target <= 0:
            self._invalid_stop_distance_skip_count += 1
            return None
        entry = LeoEntry(
            direction=signal.direction,
            named_session=signal.named_session,
            session_date=signal.session_date,
            signal_information_time_utc=signal.signal_information_time_utc,
            entry_information_time_utc=bar.available_at_utc,
            entry_price=entry_price,
            stop_price=signal.stop_price,
            target_price=target,
        )
        self._position = _Position(entry, pending.entry_window_end_utc)
        self._entered_sessions.add((signal.session_date, signal.named_session))
        return entry

    def _evaluate_exit(self, bar: LeoCompleted15mBar) -> LeoExit | None:
        assert self._position is not None
        position = self._position
        entry = position.entry
        liquidation = bar.bid if entry.direction is Direction.LONG else bar.ask
        stop_hit = (
            liquidation.low <= entry.stop_price
            if entry.direction is Direction.LONG
            else liquidation.high >= entry.stop_price
        )
        target_hit = (
            liquidation.high >= entry.target_price
            if entry.direction is Direction.LONG
            else liquidation.low <= entry.target_price
        )
        if stop_hit:
            return self._exit(
                entry, LeoExitReason.STOP_LOSS, bar.available_at_utc, entry.stop_price
            )
        if target_hit:
            return self._exit(
                entry,
                LeoExitReason.TAKE_PROFIT,
                bar.available_at_utc,
                entry.target_price,
            )
        if bar.end_time_utc >= position.entry_window_end_utc:
            return self._exit(
                entry,
                LeoExitReason.SESSION_WINDOW_END,
                bar.available_at_utc,
                liquidation.close,
            )
        return None

    def _exit(
        self,
        entry: LeoEntry,
        reason: LeoExitReason,
        information_time_utc: datetime,
        price: Decimal,
    ) -> LeoExit:
        self._position = None
        return LeoExit(
            direction=entry.direction,
            reason=reason,
            exit_information_time_utc=information_time_utc,
            exit_price=price,
            stop_price=entry.stop_price,
            target_price=entry.target_price,
        )

    def _accumulate_reference(
        self, bar: LeoCompleted15mBar, local_end: datetime
    ) -> None:
        assert self._state is not None
        clock = local_end.timetz().replace(tzinfo=None)
        if _ASIA_START <= clock < _ASIA_END:
            self._state.asia.add(bar)
        elif _LONDON_REFERENCE_START <= clock < _LONDON_REFERENCE_END:
            self._state.london.add(bar)

    def _detect_signal(
        self, bar: LeoCompleted15mBar, local_end: datetime
    ) -> LeoSignal | None:
        if self._position is not None or self._pending is not None:
            return None
        named_session = _named_entry_session(local_end.timetz().replace(tzinfo=None))
        if (
            named_session is None
            or (local_end.date(), named_session) in self._entered_sessions
        ):
            return None
        reference = _reference(
            self._state, "asia" if named_session is LeoNamedSession.LONDON else "london"
        )
        if reference is None:
            return None
        high, low = reference
        midpoint = bar.midpoint
        if midpoint.high > high and midpoint.close < high:
            direction, stop = Direction.SHORT, midpoint.high
        elif midpoint.low < low and midpoint.close > low:
            direction, stop = Direction.LONG, midpoint.low
        else:
            return None
        return LeoSignal(
            direction=direction,
            named_session=named_session,
            session_date=local_end.date(),
            signal_end_time_utc=bar.end_time_utc,
            signal_information_time_utc=bar.available_at_utc,
            stop_price=stop,
            reference_high=high,
            reference_low=low,
        )

    def _reset_for_day(self, current_day: date) -> None:
        if self._day == current_day:
            return
        if self._position is not None:
            raise LeoGbpUsdValidationError("position crossed a London trading day")
        self._day = current_day
        self._pending = None
        self._state = _DayState(
            asia=_new_range(current_day, _ASIA_START, _ASIA_END),
            london=_new_range(
                current_day, _LONDON_REFERENCE_START, _LONDON_REFERENCE_END
            ),
        )

    def _validate_order(self, bar: LeoCompleted15mBar) -> None:
        if self._last_end_utc is not None and bar.end_time_utc <= self._last_end_utc:
            raise LeoGbpUsdValidationError("completed bars must be strictly ordered")
        if (
            self._last_available_utc is not None
            and bar.available_at_utc <= self._last_available_utc
        ):
            raise LeoGbpUsdValidationError("bar availability must be strictly ordered")
        self._last_end_utc = bar.end_time_utc
        self._last_available_utc = bar.available_at_utc


def _new_range(session_day: date, start: time, end: time) -> _ReferenceRange:
    start_utc = datetime.combine(session_day, start, tzinfo=_LONDON).astimezone(UTC)
    count = int(
        (datetime.combine(session_day, end) - datetime.combine(session_day, start))
        / _BAR_INTERVAL
    )
    return _ReferenceRange(start_utc, count)


def _reference(state: _DayState | None, name: str) -> tuple[Decimal, Decimal] | None:
    if state is None:
        return None
    value = state.asia if name == "asia" else state.london
    if not value.valid:
        return None
    assert value.high is not None and value.low is not None
    return value.high, value.low


def _named_entry_session(clock: time) -> LeoNamedSession | None:
    if _LONDON_ENTRY_START <= clock < _LONDON_ENTRY_END:
        return LeoNamedSession.LONDON
    if _NEW_YORK_ENTRY_START <= clock < _NEW_YORK_ENTRY_END:
        return LeoNamedSession.NEW_YORK
    return None


def _entry_window_end(signal: LeoSignal) -> datetime:
    end = (
        _LONDON_ENTRY_END
        if signal.named_session is LeoNamedSession.LONDON
        else _NEW_YORK_ENTRY_END
    )
    return datetime.combine(signal.session_date, end, tzinfo=_LONDON).astimezone(UTC)


def _utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise LeoGbpUsdValidationError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)
