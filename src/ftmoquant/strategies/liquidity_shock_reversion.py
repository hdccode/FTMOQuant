"""Frozen raw-target state machine for ``liquidity_shock_reversion_v1``.

This module intentionally has no return calculation, costs, sizing, evaluator,
or portfolio logic.  It consumes causal Stage G observations and emits targets.
"""
# ruff: noqa: E501

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from enum import IntEnum

from ftmoquant.research.liquidity_shock_reversion_spec import (
    LIQUIDITY_SHOCK_REVERSION_CONFIG_SHA256,
    LiquidityShockReversionSpec,
    liquidity_shock_reversion_config_sha256,
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

_ONE_MINUTE = timedelta(minutes=1)
_ONE_MICROSECOND = timedelta(microseconds=1)


class LiquidityShockReversionValidationError(ValueError):
    """Raised when causal candidate inputs violate the frozen contract."""


class RawDirectionalTarget(IntEnum):
    SHORT = -1
    FLAT = 0
    LONG = 1


@dataclass(frozen=True, slots=True)
class MinuteMidpointClose:
    instrument_id: str
    event_time_utc: datetime
    information_time_utc: datetime
    close: Decimal | None

    def validate(self) -> None:
        if self.instrument_id not in FROZEN_INSTRUMENT_IDS:
            raise LiquidityShockReversionValidationError("instrument is not frozen")
        event = _utc(self.event_time_utc, "event_time_utc")
        information = _utc(self.information_time_utc, "information_time_utc")
        if information - event != _ONE_MINUTE:
            raise LiquidityShockReversionValidationError("close must be completed 1m")

    @property
    def eligible(self) -> bool:
        return self.close is not None and self.close.is_finite() and self.close > 0


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


def derive_minute_midpoint_closes(
    frames: tuple[SynchronizedClockFrame, ...],
) -> tuple[MinuteMidpointClose, ...]:
    """Derive observed midpoints only from complete synchronized Stage G frames."""
    result: list[MinuteMidpointClose] = []
    previous: datetime | None = None
    for frame in frames:
        timestamp = _utc(frame.timestamp_utc, "frame timestamp")
        available = _utc(frame.available_at_utc, "frame availability")
        if previous is not None and timestamp <= previous:
            raise LiquidityShockReversionValidationError("frames must be ordered")
        if len(frame.observations) != len(FROZEN_INSTRUMENT_IDS):
            raise LiquidityShockReversionValidationError("frame shape drifted")
        if frame.tradable:
            for instrument_id, observation in zip(
                FROZEN_INSTRUMENT_IDS, frame.observations, strict=True
            ):
                if observation is None:
                    raise LiquidityShockReversionValidationError(
                        "tradable frame has gap"
                    )
                observation.validate()
                if (
                    observation.instrument_id != instrument_id
                    or observation.timestamp_utc != timestamp
                    or observation.available_at_utc > available
                ):
                    raise LiquidityShockReversionValidationError(
                        "frame is not causally aligned"
                    )
                result.append(
                    MinuteMidpointClose(
                        instrument_id,
                        timestamp,
                        observation.available_at_utc,
                        (observation.bid + observation.ask) / Decimal(2),
                    )
                )
        previous = timestamp
    return tuple(result)


class LiquidityShockReversionStateMachine:
    """Per-instrument, 60-return shock inversion with a fixed 15-minute hold."""

    def __init__(self, spec: LiquidityShockReversionSpec) -> None:
        if (
            liquidity_shock_reversion_config_sha256(spec)
            != LIQUIDITY_SHOCK_REVERSION_CONFIG_SHA256
        ):
            raise LiquidityShockReversionValidationError(
                "strategy config hash is not frozen"
            )
        self.spec = spec
        self._returns = {
            item: deque[Decimal](maxlen=spec.baseline_prior_eligible_returns)
            for item in FROZEN_INSTRUMENT_IDS
        }
        self._previous: dict[str, MinuteMidpointClose | None] = {
            item: None for item in FROZEN_INSTRUMENT_IDS
        }
        self._last_information: dict[str, datetime] = {}
        self._pending: dict[str, DirectionalTargetSignal] = {}
        self._active: dict[str, RawDirectionalTarget] = {}
        self._entry_information: dict[str, datetime] = {}
        self._eligible_after_entry: dict[str, int] = {}

    @property
    def active_targets(self) -> tuple[tuple[str, RawDirectionalTarget], ...]:
        return tuple(
            (item, self._active[item])
            for item in FROZEN_INSTRUMENT_IDS
            if item in self._active
        )

    def reset(self) -> None:
        """Discard all state at a fold boundary so no position can cross it."""
        self._returns = {
            item: deque(maxlen=self.spec.baseline_prior_eligible_returns)
            for item in FROZEN_INSTRUMENT_IDS
        }
        self._previous = {item: None for item in FROZEN_INSTRUMENT_IDS}
        self._last_information = {}
        self._pending = {}
        self._active = {}
        self._entry_information = {}
        self._eligible_after_entry = {}

    def warmup_minute_close(self, close: MinuteMidpointClose) -> None:
        self._update(close, emit=False)

    def on_minute_close(
        self, close: MinuteMidpointClose
    ) -> DirectionalTargetSignal | None:
        return self._update(close, emit=True)

    def on_execution_frame(
        self, frame: SynchronizedClockFrame
    ) -> tuple[ExecutableDirectionalTarget, ...]:
        event, information = _validate_execution_frame(frame)
        executable: list[ExecutableDirectionalTarget] = []
        for instrument_id in FROZEN_INSTRUMENT_IDS:
            pending = self._pending.get(instrument_id)
            if pending is None or information <= pending.signal_information_time_utc:
                continue
            target = ExecutableDirectionalTarget(
                instrument_id,
                pending.target,
                pending.signal_information_time_utc,
                event,
                information,
            )
            executable.append(target)
            del self._pending[instrument_id]
            if target.target is RawDirectionalTarget.FLAT:
                self._active.pop(instrument_id, None)
                self._entry_information.pop(instrument_id, None)
                self._eligible_after_entry.pop(instrument_id, None)
            else:
                self._active[instrument_id] = target.target
                self._entry_information[instrument_id] = information
                self._eligible_after_entry[instrument_id] = 0
        for instrument_id in FROZEN_INSTRUMENT_IDS:
            if instrument_id not in self._active or instrument_id in self._pending:
                continue
            if information <= self._entry_information[instrument_id]:
                continue
            self._eligible_after_entry[instrument_id] += 1
            if (
                self._eligible_after_entry[instrument_id]
                == self.spec.hold_eligible_completed_minutes
            ):
                self._pending[instrument_id] = DirectionalTargetSignal(
                    instrument_id, RawDirectionalTarget.FLAT, event, information
                )
        return tuple(executable)

    def _update(
        self, close: MinuteMidpointClose, *, emit: bool
    ) -> DirectionalTargetSignal | None:
        close.validate()
        last = self._last_information.get(close.instrument_id)
        if last is not None and close.information_time_utc <= last:
            raise LiquidityShockReversionValidationError(
                "closes must be ordered per instrument"
            )
        self._last_information[close.instrument_id] = close.information_time_utc
        previous = self._previous[close.instrument_id]
        if not close.eligible:
            self._previous[close.instrument_id] = None
            return None
        if (
            previous is None
            or close.event_time_utc - previous.event_time_utc != _ONE_MINUTE
        ):
            self._previous[close.instrument_id] = close
            return None
        assert close.close is not None and previous.close is not None
        try:
            current_return = close.close.ln() - previous.close.ln()
        except (InvalidOperation, ValueError) as error:
            self._previous[close.instrument_id] = None
            raise LiquidityShockReversionValidationError(
                "log return is invalid"
            ) from error
        history = self._returns[close.instrument_id]
        signal: DirectionalTargetSignal | None = None
        if (
            emit
            and close.instrument_id not in self._active
            and close.instrument_id not in self._pending
            and len(history) == self.spec.baseline_prior_eligible_returns
        ):
            baseline = _median_absolute(history)
            if (
                baseline.is_finite()
                and baseline > 0
                and abs(current_return) > Decimal(self.spec.shock_multiple) * baseline
            ):
                target = (
                    RawDirectionalTarget.SHORT
                    if current_return > 0
                    else RawDirectionalTarget.LONG
                )
                signal = DirectionalTargetSignal(
                    close.instrument_id,
                    target,
                    close.event_time_utc,
                    close.information_time_utc,
                )
                self._pending[close.instrument_id] = signal
        history.append(current_return)
        self._previous[close.instrument_id] = close
        return signal


class LiquidityShockReversionDevelopmentFold:
    """Fold-isolated adapter with no evaluator or performance-access surface."""

    def __init__(
        self,
        context: DevelopmentResearchContext,
        fold: DevelopmentFold,
        spec: LiquidityShockReversionSpec,
    ) -> None:
        if (
            context.start_utc != DEVELOPMENT_START
            or context.end_exclusive_utc != DEVELOPMENT_END_EXCLUSIVE
            or fold not in frozen_development_folds().folds
        ):
            raise LiquidityShockReversionValidationError(
                "development fold contract drifted"
            )
        self.context, self.fold, self.state = (
            context,
            fold,
            LiquidityShockReversionStateMachine(spec),
        )

    def on_minute_close(
        self,
        close: MinuteMidpointClose,
        *,
        partition: ResearchPartition = ResearchPartition.DEVELOPMENT,
    ) -> DirectionalTargetSignal | None:
        self._admit(close.information_time_utc, partition)
        if close.information_time_utc < self.fold.compare_start_utc:
            self.state.warmup_minute_close(close)
            return None
        return self.state.on_minute_close(close)

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

    def _admit(self, information: datetime, partition: ResearchPartition) -> None:
        timestamp = _utc(information, "fold information time")
        self.context.require_range(
            timestamp, timestamp + _ONE_MICROSECOND, partition=partition
        )
        if (
            not self.fold.train_start_utc
            <= timestamp
            < self.fold.compare_end_exclusive_utc
        ):
            self.state.reset()
            raise StageGValidationError(
                "observation is outside the frozen fold horizon"
            )


def _median_absolute(values: deque[Decimal]) -> Decimal:
    ordered = sorted(abs(value) for value in values)
    middle = len(ordered) // 2
    return (ordered[middle - 1] + ordered[middle]) / Decimal(2)


def _validate_execution_frame(
    frame: SynchronizedClockFrame,
) -> tuple[datetime, datetime]:
    event, information = (
        _utc(frame.timestamp_utc, "execution event"),
        _utc(frame.available_at_utc, "execution information"),
    )
    if (
        not frame.tradable
        or len(frame.observations) != len(FROZEN_INSTRUMENT_IDS)
        or any(item is None for item in frame.observations)
    ):
        raise LiquidityShockReversionValidationError(
            "execution frame must be synchronized and tradable"
        )
    return event, information


def _utc(value: datetime, description: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise LiquidityShockReversionValidationError(
            f"{description} must be timezone-aware UTC"
        )
    return value.astimezone(UTC)
