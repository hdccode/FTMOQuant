"""Runtime account events and state for prop-rule evaluation."""

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum

from ftmoquant.prop_rules.models import EvaluationPhase

ZERO = Decimal("0")


class AccountStatus(StrEnum):
    """Lifecycle status of an evaluation account."""

    ACTIVE = "active"
    PASSED = "passed"
    BREACHED = "breached"


class BreachReason(StrEnum):
    """Terminal rule breach reasons."""

    MAXIMUM_DAILY_LOSS = "maximum_daily_loss"
    MAXIMUM_LOSS = "maximum_loss"


class AccountEventError(ValueError):
    """Raised when an account event cannot be applied consistently."""


@dataclass(frozen=True, slots=True)
class RuntimeAccountConfig:
    """Account-specific inputs kept separate from provider rules."""

    initial_capital: Decimal
    currency: str
    active_phase: EvaluationPhase

    def __post_init__(self) -> None:
        _require_finite_decimal(self.initial_capital, "initial_capital")
        if self.initial_capital <= ZERO:
            raise ValueError("initial_capital must be greater than zero")
        if not isinstance(self.currency, str) or not self.currency.strip():
            raise ValueError("currency must be a non-empty string")
        if not isinstance(self.active_phase, EvaluationPhase):
            raise ValueError("active_phase must be an EvaluationPhase")


@dataclass(frozen=True, slots=True)
class AccountEvent:
    """An atomic account update at a single timestamp.

    Commission is a non-negative charge. Realised P/L and swap are signed.
    ``floating_pnl`` is the aggregate value after the event.
    """

    timestamp: datetime
    floating_pnl: Decimal
    opened_position_ids: tuple[str, ...] = ()
    closed_position_ids: tuple[str, ...] = ()
    realised_pnl: Decimal = ZERO
    commission: Decimal = ZERO
    swap: Decimal = ZERO

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise AccountEventError("event timestamp must be timezone-aware")
        _require_finite_decimal(self.floating_pnl, "floating_pnl", AccountEventError)
        _require_finite_decimal(self.realised_pnl, "realised_pnl", AccountEventError)
        _require_finite_decimal(self.commission, "commission", AccountEventError)
        _require_finite_decimal(self.swap, "swap", AccountEventError)
        if self.commission < ZERO:
            raise AccountEventError("commission must be a non-negative charge")
        _validate_position_ids(self.opened_position_ids, "opened_position_ids")
        _validate_position_ids(self.closed_position_ids, "closed_position_ids")
        if set(self.opened_position_ids) & set(self.closed_position_ids):
            raise AccountEventError(
                "a position cannot open and close in the same event"
            )


@dataclass(frozen=True, slots=True)
class AccountState:
    """Complete deterministic state of one evaluation account."""

    config: RuntimeAccountConfig
    balance: Decimal
    floating_pnl: Decimal
    open_position_ids: frozenset[str]
    trading_days: frozenset[date]
    current_ftmo_day: date
    daily_reset_balance: Decimal
    status: AccountStatus
    breach_reason: BreachReason | None
    last_event_at: datetime

    @property
    def equity(self) -> Decimal:
        """Current balance plus aggregate floating P/L."""

        return self.balance + self.floating_pnl


def _require_finite_decimal(
    value: object,
    field: str,
    error_type: type[ValueError] = ValueError,
) -> None:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise error_type(f"{field} must be a finite Decimal")


def _validate_position_ids(position_ids: tuple[str, ...], field: str) -> None:
    if not isinstance(position_ids, tuple):
        raise AccountEventError(f"{field} must be a tuple")
    if any(
        not isinstance(position_id, str) or not position_id
        for position_id in position_ids
    ):
        raise AccountEventError(f"{field} must contain non-empty strings")
    if len(position_ids) != len(set(position_ids)):
        raise AccountEventError(f"{field} must not contain duplicates")
