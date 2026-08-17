"""Frozen raw-target state machine for ``eurusd_session_range_expansion_v1``.

EUR/USD-only causal London-session-range-breakout signal state, generalized
over ``(breakout_window_end, scheduled_exit)`` while keeping the overnight
range window fixed at ``00:00 <= Europe/London time < 08:00`` (the frozen
economic hypothesis). This module intentionally has no return calculation,
costs, sizing, evaluator, or portfolio logic; it consumes causal completed
one-minute midpoint closes and emits raw directional targets only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from decimal import Decimal
from zoneinfo import ZoneInfo

from ftmoquant.strategies.ts_momentum import RawDirectionalTarget

_LONDON = ZoneInfo("Europe/London")
_RANGE_START = time(0, 0)
_RANGE_END = time(8, 0)
_ONE_MINUTE_NS = 60_000_000_000
_RANGE_CLOSE_COUNT = 8 * 60


class EurusdSessionRangeExpansionValidationError(ValueError):
    """Raised when causal candidate inputs violate the frozen contract."""


@dataclass(frozen=True, slots=True)
class EurusdSessionRangeExpansionParameters:
    breakout_window_end: time
    scheduled_exit: time

    def __post_init__(self) -> None:
        if not (_RANGE_END < self.breakout_window_end <= time(23, 59)):
            raise EurusdSessionRangeExpansionValidationError(
                "breakout_window_end must be strictly after the range window"
            )
        if self.scheduled_exit <= self.breakout_window_end:
            raise EurusdSessionRangeExpansionValidationError(
                "scheduled_exit must be strictly after breakout_window_end"
            )


@dataclass(frozen=True, slots=True)
class DirectionalTargetSignal:
    target: RawDirectionalTarget
    signal_event_ns: int
    signal_information_ns: int


@dataclass(frozen=True, slots=True)
class ExecutableDirectionalTarget:
    target: RawDirectionalTarget
    signal_information_ns: int
    execution_event_ns: int
    execution_information_ns: int


@dataclass(slots=True)
class _RangeState:
    session_date: date
    expected_information_ns: int
    high: Decimal | None = None
    low: Decimal | None = None
    count: int = 0
    complete: bool = True
    entry_consumed: bool = False

    def add(self, information_ns: int, midpoint: Decimal) -> None:
        if information_ns != self.expected_information_ns:
            self.complete = False
        self.expected_information_ns = information_ns + _ONE_MINUTE_NS
        self.high = midpoint if self.high is None else max(self.high, midpoint)
        self.low = midpoint if self.low is None else min(self.low, midpoint)
        self.count += 1

    @property
    def range_is_valid(self) -> bool:
        return (
            self.complete
            and self.count == _RANGE_CLOSE_COUNT
            and self.high is not None
            and self.low is not None
        )


class EurusdSessionRangeExpansionState:
    """Single-instrument (EUR/USD) London-range-breakout signal state."""

    def __init__(self, parameters: EurusdSessionRangeExpansionParameters) -> None:
        self.parameters = parameters
        self._range: _RangeState | None = None
        self._pending: DirectionalTargetSignal | None = None
        self._active: RawDirectionalTarget | None = None
        self._last_information_ns: int | None = None

    @property
    def latest_target(self) -> RawDirectionalTarget:
        return self._active if self._active is not None else RawDirectionalTarget.FLAT

    @property
    def has_open_or_pending_position(self) -> bool:
        return self._active is not None or self._pending is not None

    def reset(self) -> None:
        """Discard all state at a fold boundary so no position can cross it."""
        self._range = None
        self._pending = None
        self._active = None
        self._last_information_ns = None

    def warmup_minute_close(
        self, event_ns: int, information_ns: int, midpoint: Decimal
    ) -> None:
        """Build range history without ever scheduling an entry or exit."""
        self._update(event_ns, information_ns, midpoint, emit=False)

    def on_minute_close(
        self, event_ns: int, information_ns: int, midpoint: Decimal
    ) -> DirectionalTargetSignal | None:
        """Update the day's range and emit at most one entry/exit per call."""
        return self._update(event_ns, information_ns, midpoint, emit=True)

    def _update(
        self, event_ns: int, information_ns: int, midpoint: Decimal, *, emit: bool
    ) -> DirectionalTargetSignal | None:
        if (
            self._last_information_ns is not None
            and information_ns <= self._last_information_ns
        ):
            raise EurusdSessionRangeExpansionValidationError(
                "minute closes must be strictly information ordered"
            )
        self._last_information_ns = information_ns
        if not midpoint.is_finite() or midpoint <= 0:
            self._invalidate(_local_date(information_ns))
            return None
        local = _local_time(information_ns)
        session_date = _local_date(information_ns)
        if _RANGE_START <= local < _RANGE_END:
            self._add_range_close(session_date, event_ns, information_ns, midpoint)
            return None
        if not emit:
            return None
        if self._pending is not None or self._active is not None:
            if local >= self.parameters.scheduled_exit:
                return self._schedule_exit(event_ns, information_ns)
            return None
        state = self._range
        if (
            _RANGE_END <= local < self.parameters.breakout_window_end
            and state is not None
            and state.session_date == session_date
            and state.range_is_valid
            and not state.entry_consumed
        ):
            assert state.high is not None
            assert state.low is not None
            target = (
                RawDirectionalTarget.LONG
                if midpoint > state.high
                else RawDirectionalTarget.SHORT
                if midpoint < state.low
                else None
            )
            if target is not None:
                state.entry_consumed = True
                signal = DirectionalTargetSignal(target, event_ns, information_ns)
                self._pending = signal
                return signal
        return None

    def on_execution_frame(
        self, event_ns: int, information_ns: int
    ) -> ExecutableDirectionalTarget | None:
        """Release a pending target only strictly after its signal time."""

        if (
            self._pending is None
            or information_ns <= self._pending.signal_information_ns
        ):
            return None
        pending = self._pending
        self._pending = None
        self._active = (
            None if pending.target is RawDirectionalTarget.FLAT else (pending.target)
        )
        return ExecutableDirectionalTarget(
            pending.target, pending.signal_information_ns, event_ns, information_ns
        )

    def _add_range_close(
        self, session_date: date, event_ns: int, information_ns: int, midpoint: Decimal
    ) -> None:
        state = self._range
        if state is None or state.session_date != session_date:
            expected = _first_range_information_ns(session_date)
            state = _RangeState(session_date, expected)
            self._range = state
        state.add(information_ns, midpoint)

    def _invalidate(self, session_date: date) -> None:
        state = self._range
        if state is not None and state.session_date == session_date:
            state.complete = False

    def _schedule_exit(
        self, event_ns: int, information_ns: int
    ) -> DirectionalTargetSignal | None:
        if (
            self._pending is not None
            and self._pending.target is RawDirectionalTarget.FLAT
        ):
            return None
        if self._active is None and self._pending is None:
            return None
        signal = DirectionalTargetSignal(
            RawDirectionalTarget.FLAT, event_ns, information_ns
        )
        self._pending = signal
        return signal


def _first_range_information_ns(session_date: date) -> int:
    combined = datetime.combine(session_date, _RANGE_START, tzinfo=_LONDON)
    return int(combined.astimezone(UTC).timestamp() * 1_000_000_000)


def _local_time(information_ns: int) -> time:
    return _datetime(information_ns).astimezone(_LONDON).time().replace(microsecond=0)


def _local_date(information_ns: int) -> date:
    return _datetime(information_ns).astimezone(_LONDON).date()


def _datetime(information_ns: int) -> datetime:
    return datetime.fromtimestamp(information_ns / 1_000_000_000, tz=UTC)
