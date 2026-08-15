"""Frozen ``session_range_expansion_v1`` raw-target state machine.

It consumes only causal Stage G midpoint observations and emits raw directional
targets. Execution, sizing, costs, P/L, and portfolio limits remain outside
this candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from ftmoquant.research.session_range_expansion_spec import (
    SESSION_RANGE_EXPANSION_CONFIG_SHA256,
    SessionRangeExpansionSpec,
    session_range_expansion_config_sha256,
)
from ftmoquant.research.stage_g import (
    DEVELOPMENT_END_EXCLUSIVE,
    DEVELOPMENT_START,
    FROZEN_INSTRUMENT_IDS,
    DevelopmentFold,
    DevelopmentResearchContext,
    ResearchPartition,
    StageGValidationError,
    SynchronizedClockFrame,
    frozen_development_folds,
)
from ftmoquant.strategies.ts_momentum import (
    DirectionalTargetSignal,
    ExecutableDirectionalTarget,
    RawDirectionalTarget,
)

_LONDON = ZoneInfo("Europe/London")
_RANGE_START = time(0, 0)
_RANGE_END = time(8, 0)
_BREAKOUT_END = time(12, 0)
_EXIT_TIME = time(16, 0)
_ONE_MINUTE = timedelta(minutes=1)
_ONE_MICROSECOND = timedelta(microseconds=1)
_RANGE_CLOSE_COUNT = 8 * 60


class SessionRangeExpansionValidationError(ValueError):
    """Raised when session-range inputs violate the frozen causal contract."""


@dataclass(slots=True)
class _RangeState:
    session_date: date
    expected_information_time_utc: datetime
    high: Decimal | None = None
    low: Decimal | None = None
    count: int = 0
    complete: bool = True
    entry_consumed: bool = False

    def add(self, information_time_utc: datetime, midpoint: Decimal) -> None:
        if information_time_utc != self.expected_information_time_utc:
            self.complete = False
        self.expected_information_time_utc = information_time_utc + _ONE_MINUTE
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


class SessionRangeExpansionStateMachine:
    """Independent London-session range and strict-later target scheduling."""

    def __init__(self, spec: SessionRangeExpansionSpec) -> None:
        if (
            session_range_expansion_config_sha256(spec)
            != SESSION_RANGE_EXPANSION_CONFIG_SHA256
        ):
            raise SessionRangeExpansionValidationError("strategy config hash is frozen")
        self.spec = spec
        self._ranges: dict[str, _RangeState] = {}
        self._pending: dict[str, DirectionalTargetSignal] = {}
        self._active: dict[str, RawDirectionalTarget] = {}
        self._active_entry_date: dict[str, date] = {}
        self._last_information: dict[str, datetime] = {}

    @property
    def active_targets(self) -> tuple[tuple[str, RawDirectionalTarget], ...]:
        return tuple(
            (instrument_id, self._active[instrument_id])
            for instrument_id in FROZEN_INSTRUMENT_IDS
            if instrument_id in self._active
        )

    def on_frame(
        self, frame: SynchronizedClockFrame
    ) -> tuple[DirectionalTargetSignal, ...]:
        """Update independent ranges and emit at most one entry/day/instrument."""

        self._validate_frame(frame)
        signals: list[DirectionalTargetSignal] = []
        for instrument_id, observation in zip(
            FROZEN_INSTRUMENT_IDS, frame.observations, strict=True
        ):
            if observation is None:
                continue
            information = _utc(observation.available_at_utc, "observation availability")
            previous = self._last_information.get(instrument_id)
            if previous is not None and information <= previous:
                raise SessionRangeExpansionValidationError(
                    "instrument observations must be strictly ordered"
                )
            self._last_information[instrument_id] = information
            local = information.astimezone(_LONDON)
            if (
                not observation.bid.is_finite()
                or not observation.ask.is_finite()
                or observation.bid <= 0
                or observation.ask < observation.bid
            ):
                self._invalidate(instrument_id, local.date())
                continue
            midpoint = (observation.bid + observation.ask) / Decimal(2)
            if not midpoint.is_finite() or midpoint <= 0:
                self._invalidate(instrument_id, local.date())
                continue
            if _RANGE_START <= local.time().replace(tzinfo=None) < _RANGE_END:
                self._add_range_close(instrument_id, local, information, midpoint)
                continue
            state = self._ranges.get(instrument_id)
            if (
                _RANGE_END <= local.time().replace(tzinfo=None) < _BREAKOUT_END
                and state is not None
                and state.session_date == local.date()
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
                    signals.append(
                        self._schedule(
                            instrument_id,
                            target,
                            observation.timestamp_utc,
                            information,
                        )
                    )
            if local.time().replace(tzinfo=None) >= _EXIT_TIME:
                exit_signal = self._schedule_exit_if_needed(
                    instrument_id, observation.timestamp_utc, information
                )
                if exit_signal is not None:
                    signals.append(exit_signal)
        return tuple(signals)

    def on_execution_frame(
        self, frame: SynchronizedClockFrame
    ) -> tuple[ExecutableDirectionalTarget, ...]:
        """Release targets only on a complete Stage G point strictly after signal."""

        self._validate_frame(frame, require_valid_prices=True)
        information = _utc(frame.available_at_utc, "execution information")
        event = _utc(frame.timestamp_utc, "execution event")
        if not frame.tradable:
            return ()
        if any(item is None for item in frame.observations):
            raise SessionRangeExpansionValidationError(
                "tradable frame cannot contain a missing observation"
            )
        executable: list[ExecutableDirectionalTarget] = []
        for instrument_id in FROZEN_INSTRUMENT_IDS:
            pending = self._pending.get(instrument_id)
            if pending is None or information <= pending.signal_information_time_utc:
                continue
            target = ExecutableDirectionalTarget(
                instrument_id=instrument_id,
                target=pending.target,
                signal_information_time_utc=pending.signal_information_time_utc,
                execution_event_time_utc=event,
                execution_information_time_utc=information,
            )
            executable.append(target)
            if target.target is RawDirectionalTarget.FLAT:
                self._active.pop(instrument_id, None)
                self._active_entry_date.pop(instrument_id, None)
            else:
                self._active[instrument_id] = target.target
                self._active_entry_date[instrument_id] = (
                    target.signal_information_time_utc.astimezone(_LONDON).date()
                )
            del self._pending[instrument_id]
        return tuple(executable)

    def _add_range_close(
        self,
        instrument_id: str,
        local_information: datetime,
        information_time_utc: datetime,
        midpoint: Decimal,
    ) -> None:
        session_date = local_information.date()
        state = self._ranges.get(instrument_id)
        if state is None or state.session_date != session_date:
            expected = datetime.combine(
                session_date, _RANGE_START, tzinfo=_LONDON
            ).astimezone(UTC)
            state = _RangeState(session_date, expected)
            self._ranges[instrument_id] = state
        state.add(information_time_utc, midpoint)

    def _invalidate(self, instrument_id: str, session_date: date) -> None:
        state = self._ranges.get(instrument_id)
        if state is not None and state.session_date == session_date:
            state.complete = False

    def _schedule(
        self,
        instrument_id: str,
        target: RawDirectionalTarget,
        event_time_utc: datetime,
        information_time_utc: datetime,
    ) -> DirectionalTargetSignal:
        signal = DirectionalTargetSignal(
            instrument_id=instrument_id,
            target=target,
            signal_event_time_utc=_utc(event_time_utc, "signal event"),
            signal_information_time_utc=_utc(
                information_time_utc, "signal information"
            ),
        )
        self._pending[instrument_id] = signal
        return signal

    def _schedule_exit_if_needed(
        self,
        instrument_id: str,
        event_time_utc: datetime,
        information_time_utc: datetime,
    ) -> DirectionalTargetSignal | None:
        pending = self._pending.get(instrument_id)
        if instrument_id not in self._active and (
            pending is None or pending.target is RawDirectionalTarget.FLAT
        ):
            return None
        if pending is not None and pending.target is RawDirectionalTarget.FLAT:
            return None
        return self._schedule(
            instrument_id,
            RawDirectionalTarget.FLAT,
            event_time_utc,
            information_time_utc,
        )

    @staticmethod
    def _validate_frame(
        frame: SynchronizedClockFrame, *, require_valid_prices: bool = False
    ) -> None:
        event = _utc(frame.timestamp_utc, "frame event")
        information = _utc(frame.available_at_utc, "frame availability")
        if len(frame.observations) != len(FROZEN_INSTRUMENT_IDS):
            raise SessionRangeExpansionValidationError("clock frame shape is frozen")
        for instrument_id, observation in zip(
            FROZEN_INSTRUMENT_IDS, frame.observations, strict=True
        ):
            if observation is None:
                continue
            if require_valid_prices:
                observation.validate()
            if observation.instrument_id != instrument_id:
                raise SessionRangeExpansionValidationError(
                    "clock instrument order drifted"
                )
            if (
                observation.timestamp_utc != event
                or observation.available_at_utc < event
                or observation.available_at_utc > information
            ):
                raise SessionRangeExpansionValidationError("observation is not causal")


class SessionRangeExpansionDevelopmentFold:
    """Fresh-state DEVELOPMENT fold adapter with no target carry across folds."""

    def __init__(
        self,
        context: DevelopmentResearchContext,
        fold: DevelopmentFold,
        spec: SessionRangeExpansionSpec,
    ) -> None:
        if context.start_utc != DEVELOPMENT_START or (
            context.end_exclusive_utc != DEVELOPMENT_END_EXCLUSIVE
        ):
            raise SessionRangeExpansionValidationError("development context drifted")
        if fold not in frozen_development_folds().folds:
            raise SessionRangeExpansionValidationError("development fold is not frozen")
        self.context = context
        self.fold = fold
        self.state = SessionRangeExpansionStateMachine(spec)

    def on_frame(
        self,
        frame: SynchronizedClockFrame,
        *,
        partition: ResearchPartition = ResearchPartition.DEVELOPMENT,
    ) -> tuple[DirectionalTargetSignal, ...]:
        self._admit(frame.available_at_utc, partition)
        if frame.available_at_utc < self.fold.compare_start_utc:
            return ()
        return self.state.on_frame(frame)

    def on_execution_frame(
        self,
        frame: SynchronizedClockFrame,
        *,
        partition: ResearchPartition = ResearchPartition.DEVELOPMENT,
    ) -> tuple[ExecutableDirectionalTarget, ...]:
        self._admit(frame.available_at_utc, partition)
        if frame.available_at_utc < self.fold.compare_start_utc:
            return ()
        return self.state.on_execution_frame(frame)

    def _admit(self, information_time: datetime, partition: ResearchPartition) -> None:
        timestamp = _utc(information_time, "fold information time")
        self.context.require_range(
            timestamp, timestamp + _ONE_MICROSECOND, partition=partition
        )
        if not (
            self.fold.train_start_utc
            <= timestamp
            < self.fold.compare_end_exclusive_utc
        ):
            raise StageGValidationError(
                "observation is outside the frozen fold horizon"
            )


def _utc(value: datetime, description: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise SessionRangeExpansionValidationError(
            f"{description} must be timezone-aware UTC"
        )
    return value.astimezone(UTC)
