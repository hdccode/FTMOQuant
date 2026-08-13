"""Preserved G0.3 reducer used only as an independent differential oracle.

Source: FTMOQuant branch ``g0-ftmo-oracle`` at commit
``620642ac03d0abf7f019dcaae5b70a4306296d66``.  This test-only copy deliberately
does not call the production Nautilus overlay.
"""

from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from enum import StrEnum

from ftmoquant.prop_rules.models import DailyReset, EvaluationPhase, PropRuleSet

ZERO = Decimal("0")


class OracleStatus(StrEnum):
    ACTIVE = "active"
    PASSED = "passed"
    BREACHED = "breached"


class OracleBreachReason(StrEnum):
    MAXIMUM_DAILY_LOSS = "maximum_daily_loss"
    MAXIMUM_LOSS = "maximum_loss"


@dataclass(frozen=True, slots=True)
class OracleEvent:
    timestamp: datetime
    floating_pnl: Decimal
    opened_position_ids: tuple[str, ...] = ()
    closed_position_ids: tuple[str, ...] = ()
    realised_pnl: Decimal = ZERO
    commission: Decimal = ZERO
    swap: Decimal = ZERO


@dataclass(frozen=True, slots=True)
class OracleState:
    initial_capital: Decimal
    currency: str
    active_phase: EvaluationPhase
    balance: Decimal
    floating_pnl: Decimal
    open_position_ids: frozenset[str]
    trading_days: frozenset[date]
    current_ftmo_day: date
    daily_reset_balance: Decimal
    status: OracleStatus
    breach_reason: OracleBreachReason | None
    last_event_at: datetime

    @property
    def equity(self) -> Decimal:
        return self.balance + self.floating_pnl


def create_oracle(
    initial_capital: Decimal,
    currency: str,
    active_phase: EvaluationPhase,
    rules: PropRuleSet,
    started_at: datetime,
) -> OracleState:
    """Create the preserved G0.3 state for a differential fixture."""

    return OracleState(
        initial_capital=initial_capital,
        currency=currency,
        active_phase=active_phase,
        balance=initial_capital,
        floating_pnl=ZERO,
        open_position_ids=frozenset(),
        trading_days=frozenset(),
        current_ftmo_day=_ftmo_day(started_at, rules.daily_reset),
        daily_reset_balance=initial_capital,
        status=OracleStatus.ACTIVE,
        breach_reason=None,
        last_event_at=started_at,
    )


def apply_oracle_event(
    state: OracleState,
    event: OracleEvent,
    rules: PropRuleSet,
) -> OracleState:
    """Apply one atomic fixture event with the original G0.3 accounting."""

    if event.timestamp.astimezone(UTC) <= state.last_event_at.astimezone(UTC):
        raise ValueError("oracle events must be strictly ordered")

    advanced = _advance_resets(state, event.timestamp, rules)
    if advanced.status is OracleStatus.BREACHED:
        return replace(advanced, last_event_at=event.timestamp)

    open_ids = (
        set(advanced.open_position_ids) - set(event.closed_position_ids)
    ) | set(event.opened_position_ids)
    balance = advanced.balance + event.realised_pnl - event.commission + event.swap
    trading_days = advanced.trading_days
    if event.opened_position_ids:
        trading_days = trading_days | {advanced.current_ftmo_day}

    updated = replace(
        advanced,
        balance=balance,
        floating_pnl=event.floating_pnl,
        open_position_ids=frozenset(open_ids),
        trading_days=frozenset(trading_days),
        last_event_at=event.timestamp,
    )
    return _evaluate(updated, rules)


def daily_floor(state: OracleState, rules: PropRuleSet) -> Decimal:
    amount = state.initial_capital * rules.loss_limits.maximum_daily_loss
    return state.daily_reset_balance - amount


def maximum_floor(state: OracleState, rules: PropRuleSet) -> Decimal:
    amount = state.initial_capital * rules.loss_limits.maximum_loss
    return state.initial_capital - amount


def _advance_resets(
    state: OracleState,
    timestamp: datetime,
    rules: PropRuleSet,
) -> OracleState:
    target_day = _ftmo_day(timestamp, rules.daily_reset)
    advanced = state
    while advanced.current_ftmo_day < target_day:
        advanced = replace(
            advanced,
            current_ftmo_day=advanced.current_ftmo_day + timedelta(days=1),
            daily_reset_balance=advanced.balance,
        )
        advanced = _evaluate(advanced, rules)
        if advanced.status is OracleStatus.BREACHED:
            break
    return advanced


def _evaluate(state: OracleState, rules: PropRuleSet) -> OracleState:
    if state.equity < daily_floor(state, rules):
        return replace(
            state,
            status=OracleStatus.BREACHED,
            breach_reason=OracleBreachReason.MAXIMUM_DAILY_LOSS,
        )
    if state.equity < maximum_floor(state, rules):
        return replace(
            state,
            status=OracleStatus.BREACHED,
            breach_reason=OracleBreachReason.MAXIMUM_LOSS,
        )

    phase = next(item for item in rules.phases if item.phase is state.active_phase)
    target = state.initial_capital * (Decimal("1") + phase.profit_target)
    if (
        state.balance >= target
        and len(state.trading_days) >= phase.minimum_trading_days
        and not state.open_position_ids
    ):
        return replace(state, status=OracleStatus.PASSED)
    return state


def _ftmo_day(timestamp: datetime, reset: DailyReset) -> date:
    local = timestamp.astimezone(reset.timezone)
    reset_today = datetime.combine(local.date(), reset.time, tzinfo=reset.timezone)
    return local.date() if local >= reset_today else local.date() - timedelta(days=1)
