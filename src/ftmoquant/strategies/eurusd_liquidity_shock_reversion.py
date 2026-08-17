"""Frozen raw-target state machine for ``eurusd_liquidity_shock_reversion_v1``.

EUR/USD-only causal shock-detection and hold/flatten signal state, generalized
over ``(baseline_prior_returns, shock_multiple, hold_eligible_minutes)``. This
module intentionally has no return calculation, costs, sizing, evaluator, or
portfolio logic; it consumes causal completed one-minute midpoint closes and
emits raw directional targets only.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from ftmoquant.strategies.ts_momentum import RawDirectionalTarget

_ONE_MINUTE_NS = 60_000_000_000


class EurusdLiquidityShockReversionValidationError(ValueError):
    """Raised when causal candidate inputs violate the frozen contract."""


@dataclass(frozen=True, slots=True)
class EurusdLiquidityShockReversionParameters:
    baseline_prior_returns: int
    shock_multiple: Decimal
    hold_eligible_minutes: int

    def __post_init__(self) -> None:
        if isinstance(self.baseline_prior_returns, bool) or (
            self.baseline_prior_returns <= 0
        ):
            raise EurusdLiquidityShockReversionValidationError(
                "baseline_prior_returns must be a positive integer"
            )
        if isinstance(self.hold_eligible_minutes, bool) or (
            self.hold_eligible_minutes <= 0
        ):
            raise EurusdLiquidityShockReversionValidationError(
                "hold_eligible_minutes must be a positive integer"
            )
        if not self.shock_multiple.is_finite() or self.shock_multiple <= 0:
            raise EurusdLiquidityShockReversionValidationError(
                "shock_multiple must be a positive finite Decimal"
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


class EurusdLiquidityShockReversionState:
    """Single-instrument (EUR/USD) median-absolute shock/reversion state."""

    def __init__(self, parameters: EurusdLiquidityShockReversionParameters) -> None:
        self.parameters = parameters
        self._returns: deque[Decimal] = deque(
            maxlen=parameters.baseline_prior_returns
        )
        self._previous: tuple[int, Decimal] | None = None
        self._last_information_ns: int | None = None
        self._pending: DirectionalTargetSignal | None = None
        self._active: RawDirectionalTarget | None = None
        self._entry_information_ns: int | None = None
        self._eligible_after_entry = 0

    @property
    def latest_target(self) -> RawDirectionalTarget:
        return self._active if self._active is not None else RawDirectionalTarget.FLAT

    @property
    def has_open_or_pending_position(self) -> bool:
        return self._active is not None or self._pending is not None

    def reset(self) -> None:
        """Discard all state at a fold boundary so no position can cross it."""
        self._returns = deque(maxlen=self.parameters.baseline_prior_returns)
        self._previous = None
        self._last_information_ns = None
        self._pending = None
        self._active = None
        self._entry_information_ns = None
        self._eligible_after_entry = 0

    def warmup_minute_close(
        self, event_ns: int, information_ns: int, midpoint: Decimal
    ) -> None:
        self._update(event_ns, information_ns, midpoint, emit=False)

    def on_minute_close(
        self, event_ns: int, information_ns: int, midpoint: Decimal
    ) -> DirectionalTargetSignal | None:
        return self._update(event_ns, information_ns, midpoint, emit=True)

    def on_execution_frame(
        self, event_ns: int, information_ns: int
    ) -> ExecutableDirectionalTarget | None:
        """Promote a pending signal to executable and advance the hold clock."""
        if (
            self._pending is not None
            and information_ns > self._pending.signal_information_ns
        ):
            pending = self._pending
            self._pending = None
            if pending.target is RawDirectionalTarget.FLAT:
                self._active = None
                self._entry_information_ns = None
                self._eligible_after_entry = 0
            else:
                self._active = pending.target
                self._entry_information_ns = information_ns
                self._eligible_after_entry = 0
            return ExecutableDirectionalTarget(
                pending.target,
                pending.signal_information_ns,
                event_ns,
                information_ns,
            )
        if (
            self._active is not None
            and self._pending is None
            and self._entry_information_ns is not None
            and information_ns > self._entry_information_ns
        ):
            self._eligible_after_entry += 1
            if self._eligible_after_entry == self.parameters.hold_eligible_minutes:
                self._pending = DirectionalTargetSignal(
                    RawDirectionalTarget.FLAT, event_ns, information_ns
                )
        return None

    def _update(
        self, event_ns: int, information_ns: int, midpoint: Decimal, *, emit: bool
    ) -> DirectionalTargetSignal | None:
        if (
            self._last_information_ns is not None
            and information_ns <= self._last_information_ns
        ):
            raise EurusdLiquidityShockReversionValidationError(
                "minute closes must be strictly information ordered"
            )
        self._last_information_ns = information_ns
        if not midpoint.is_finite() or midpoint <= 0:
            self._previous = None
            return None
        previous = self._previous
        if previous is None or event_ns - previous[0] != _ONE_MINUTE_NS:
            self._previous = (event_ns, midpoint)
            return None
        try:
            current_return = midpoint.ln() - previous[1].ln()
        except (InvalidOperation, ValueError) as error:
            self._previous = None
            raise EurusdLiquidityShockReversionValidationError(
                "log return is invalid"
            ) from error
        history = self._returns
        signal: DirectionalTargetSignal | None = None
        if (
            emit
            and self._active is None
            and self._pending is None
            and len(history) == self.parameters.baseline_prior_returns
        ):
            baseline = _median_absolute(history)
            if (
                baseline.is_finite()
                and baseline > 0
                and abs(current_return)
                > self.parameters.shock_multiple * baseline
            ):
                target = (
                    RawDirectionalTarget.SHORT
                    if current_return > 0
                    else RawDirectionalTarget.LONG
                )
                signal = DirectionalTargetSignal(target, event_ns, information_ns)
                self._pending = signal
        history.append(current_return)
        self._previous = (event_ns, midpoint)
        return signal


def _median_absolute(values: Sequence[Decimal]) -> Decimal:
    ordered = sorted(abs(value) for value in values)
    middle = len(ordered) // 2
    return (ordered[middle - 1] + ordered[middle]) / Decimal(2)
