"""Frozen ``ts_momentum_v1`` raw-target state machine.

There is no backtester, sizing, order, cost, P/L, return, or selection code in
this module. Stage G and Nautilus retain those responsibilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from enum import IntEnum
from zoneinfo import ZoneInfo

from nautilus_trader.indicators import RateOfChange

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
from ftmoquant.research.ts_momentum_spec import (
    TS_MOMENTUM_CONFIG_SHA256,
    TsMomentumSpec,
    ts_momentum_config_sha256,
)

_SESSION_ZONE = ZoneInfo("America/New_York")
_SESSION_CLOSE = time(17, 0)
_ONE_MINUTE = timedelta(minutes=1)
_ONE_MICROSECOND = timedelta(microseconds=1)


class TsMomentumValidationError(ValueError):
    """Raised when candidate inputs violate the frozen causal contract."""


class RawDirectionalTarget(IntEnum):
    SHORT = -1
    FLAT = 0
    LONG = 1


@dataclass(frozen=True, slots=True)
class DailyMidpointClose:
    """One observed, completed provider-session midpoint close."""

    instrument_id: str
    session_date: date
    event_time_utc: datetime
    information_time_utc: datetime
    close: Decimal | None

    def validate_identity_and_time(self) -> None:
        if self.instrument_id not in FROZEN_INSTRUMENT_IDS:
            raise TsMomentumValidationError("daily close instrument is not frozen")
        event = _utc(self.event_time_utc, "event_time_utc")
        information = _utc(self.information_time_utc, "information_time_utc")
        local_information = information.astimezone(_SESSION_ZONE)
        if information - event != _ONE_MINUTE:
            raise TsMomentumValidationError(
                "daily close must be an observed completed one-minute pair"
            )
        if (
            local_information.time().replace(tzinfo=None) != _SESSION_CLOSE
            or local_information.weekday() > 4
            or local_information.date() != self.session_date
        ):
            raise TsMomentumValidationError(
                "daily close information time is not the frozen FX session close"
            )

    @property
    def price_is_eligible(self) -> bool:
        return (
            self.close is not None
            and self.close.is_finite()
            and self.close > Decimal(0)
        )


@dataclass(frozen=True, slots=True)
class DirectionalTargetSignal:
    instrument_id: str
    target: RawDirectionalTarget
    signal_event_time_utc: datetime
    signal_information_time_utc: datetime


@dataclass(frozen=True, slots=True)
class ExecutableDirectionalTarget:
    instrument_id: str
    target: RawDirectionalTarget
    signal_information_time_utc: datetime
    execution_event_time_utc: datetime
    execution_information_time_utc: datetime


def derive_daily_midpoint_closes(
    frames: tuple[SynchronizedClockFrame, ...],
) -> tuple[DailyMidpointClose, ...]:
    """Select actual session-ending closes independently, without fill or fallback."""

    result: list[DailyMidpointClose] = []
    last_frame_time: datetime | None = None
    last_session_by_instrument: dict[str, date] = {}
    for frame in frames:
        timestamp = _utc(frame.timestamp_utc, "frame timestamp")
        _utc(frame.available_at_utc, "frame availability")
        if last_frame_time is not None and timestamp <= last_frame_time:
            raise TsMomentumValidationError("clock frames must be strictly ordered")
        if len(frame.observations) != len(FROZEN_INSTRUMENT_IDS):
            raise TsMomentumValidationError(
                "clock frame instrument shape is not frozen"
            )
        for instrument_id, observation in zip(
            FROZEN_INSTRUMENT_IDS, frame.observations, strict=True
        ):
            if observation is None:
                continue
            observation.validate()
            if observation.instrument_id != instrument_id:
                raise TsMomentumValidationError("clock frame instrument order drifted")
            if observation.timestamp_utc != timestamp or (
                observation.available_at_utc > frame.available_at_utc
            ):
                raise TsMomentumValidationError(
                    "observation is not causally aligned to its clock frame"
                )
            information = _utc(observation.available_at_utc, "observation availability")
            local = information.astimezone(_SESSION_ZONE)
            if (
                information - observation.timestamp_utc != _ONE_MINUTE
                or local.time().replace(tzinfo=None) != _SESSION_CLOSE
                or local.weekday() > 4
            ):
                continue
            session_date = local.date()
            if last_session_by_instrument.get(instrument_id) == session_date:
                raise TsMomentumValidationError("duplicate daily close for instrument")
            close = DailyMidpointClose(
                instrument_id=instrument_id,
                session_date=session_date,
                event_time_utc=observation.timestamp_utc,
                information_time_utc=information,
                close=(observation.bid + observation.ask) / Decimal(2),
            )
            close.validate_identity_and_time()
            result.append(close)
            last_session_by_instrument[instrument_id] = session_date
        last_frame_time = timestamp
    return tuple(result)


class TsMomentumStateMachine:
    """Independent native ROC state and strictly-later target scheduling."""

    def __init__(self, spec: TsMomentumSpec) -> None:
        if ts_momentum_config_sha256(spec) != TS_MOMENTUM_CONFIG_SHA256:
            raise TsMomentumValidationError("strategy config hash is not frozen")
        self.spec = spec
        self._indicators = {
            instrument_id: RateOfChange(spec.native_period, use_log=True)
            for instrument_id in FROZEN_INSTRUMENT_IDS
        }
        self._last_session_dates: dict[str, date] = {}
        self._latest_signal_targets: dict[str, RawDirectionalTarget] = {}
        self._active_targets: dict[str, RawDirectionalTarget] = {}
        self._pending: dict[str, DirectionalTargetSignal] = {}

    @property
    def active_targets(self) -> tuple[tuple[str, RawDirectionalTarget], ...]:
        """Targets currently held at the execution boundary, in universe order."""

        return tuple(
            (instrument_id, self._active_targets[instrument_id])
            for instrument_id in FROZEN_INSTRUMENT_IDS
            if instrument_id in self._active_targets
        )

    def warmup_daily_close(self, close: DailyMidpointClose) -> None:
        """Update eligible history without creating a signal or pending target."""

        self._update_daily_close(close, emit=False)

    def on_daily_close(
        self, close: DailyMidpointClose
    ) -> DirectionalTargetSignal | None:
        """Map native log ROC sign to a changed raw target only."""

        return self._update_daily_close(close, emit=True)

    def on_execution_frame(
        self, frame: SynchronizedClockFrame
    ) -> tuple[ExecutableDirectionalTarget, ...]:
        """Release pending targets only at the first strictly-later eligible frame."""

        event = _utc(frame.timestamp_utc, "execution event")
        information = _utc(frame.available_at_utc, "execution information")
        if len(frame.observations) != len(FROZEN_INSTRUMENT_IDS):
            raise TsMomentumValidationError("execution frame shape is not frozen")
        if not frame.tradable:
            return ()
        if any(observation is None for observation in frame.observations):
            raise TsMomentumValidationError("tradable frame cannot contain a gap")
        for instrument_id, observation in zip(
            FROZEN_INSTRUMENT_IDS, frame.observations, strict=True
        ):
            assert observation is not None
            observation.validate()
            if observation.instrument_id != instrument_id:
                raise TsMomentumValidationError("execution frame order drifted")
            if observation.timestamp_utc != event or (
                observation.available_at_utc > information
            ):
                raise TsMomentumValidationError(
                    "execution observation is not causally aligned"
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
            self._active_targets[instrument_id] = pending.target
            del self._pending[instrument_id]
        return tuple(executable)

    def _update_daily_close(
        self, close: DailyMidpointClose, *, emit: bool
    ) -> DirectionalTargetSignal | None:
        close.validate_identity_and_time()
        previous_date = self._last_session_dates.get(close.instrument_id)
        if previous_date is not None and close.session_date <= previous_date:
            raise TsMomentumValidationError(
                "daily observations must be strictly ordered per instrument"
            )
        self._last_session_dates[close.instrument_id] = close.session_date
        if not close.price_is_eligible:
            return None
        assert close.close is not None
        indicator = self._indicators[close.instrument_id]
        indicator.update_raw(float(close.close))
        if not indicator.initialized or not emit:
            return None
        target = (
            RawDirectionalTarget.LONG
            if indicator.value > 0.0
            else RawDirectionalTarget.SHORT
            if indicator.value < 0.0
            else RawDirectionalTarget.FLAT
        )
        if self._latest_signal_targets.get(close.instrument_id) is target:
            return None
        signal = DirectionalTargetSignal(
            instrument_id=close.instrument_id,
            target=target,
            signal_event_time_utc=close.event_time_utc,
            signal_information_time_utc=close.information_time_utc,
        )
        self._latest_signal_targets[close.instrument_id] = target
        self._pending[close.instrument_id] = signal
        return signal


class TsMomentumDevelopmentFold:
    """Fold-isolated DEVELOPMENT adapter; it exposes no return calculation."""

    def __init__(
        self,
        context: DevelopmentResearchContext,
        fold: DevelopmentFold,
        spec: TsMomentumSpec,
    ) -> None:
        if context.start_utc != DEVELOPMENT_START or (
            context.end_exclusive_utc != DEVELOPMENT_END_EXCLUSIVE
        ):
            raise TsMomentumValidationError("development context boundaries drifted")
        if fold not in frozen_development_folds().folds:
            raise TsMomentumValidationError("development fold is not frozen")
        self.context = context
        self.fold = fold
        self.state = TsMomentumStateMachine(spec)

    def on_daily_close(
        self,
        close: DailyMidpointClose,
        *,
        partition: ResearchPartition = ResearchPartition.DEVELOPMENT,
    ) -> DirectionalTargetSignal | None:
        self._admit(close.information_time_utc, partition)
        if close.information_time_utc < self.fold.compare_start_utc:
            self.state.warmup_daily_close(close)
            return None
        return self.state.on_daily_close(close)

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
        raise TsMomentumValidationError(f"{description} must be timezone-aware UTC")
    return value.astimezone(UTC)
