"""Deterministic FTMO account-state and rule evaluation."""

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from ftmoquant.prop_rules.models import DailyReset, PhaseRules, PropRuleSet
from ftmoquant.prop_rules.state import (
    ZERO,
    AccountEvent,
    AccountEventError,
    AccountState,
    AccountStatus,
    BreachReason,
    RuntimeAccountConfig,
)


def create_account_state(
    config: RuntimeAccountConfig,
    rules: PropRuleSet,
    started_at: datetime,
) -> AccountState:
    """Create an active account at the supplied aware timestamp."""

    _require_aware(started_at, "started_at")
    _phase_rules(rules, config)
    return AccountState(
        config=config,
        balance=config.initial_capital,
        floating_pnl=ZERO,
        open_position_ids=frozenset(),
        trading_days=frozenset(),
        current_ftmo_day=_ftmo_day(started_at, rules.daily_reset),
        daily_reset_balance=config.initial_capital,
        status=AccountStatus.ACTIVE,
        breach_reason=None,
        last_event_at=started_at,
    )


def apply_account_event(
    state: AccountState,
    event: AccountEvent,
    rules: PropRuleSet,
) -> AccountState:
    """Atomically apply one ordered event and evaluate configured rules."""

    if state.status is not AccountStatus.ACTIVE:
        raise AccountEventError(
            f"cannot apply events to a {state.status.value} account"
        )
    if _as_utc(event.timestamp) <= _as_utc(state.last_event_at):
        raise AccountEventError("event timestamp must be later than the previous event")

    _validate_position_operations(state, event)
    advanced = _advance_reset_boundaries(state, event.timestamp, rules)
    if advanced.status is AccountStatus.BREACHED:
        return replace(advanced, last_event_at=event.timestamp)

    open_positions = (
        set(advanced.open_position_ids) - set(event.closed_position_ids)
    ) | set(event.opened_position_ids)
    balance = (
        advanced.balance + event.realised_pnl - event.commission + event.swap
    )
    trading_days = advanced.trading_days
    if event.opened_position_ids:
        trading_days = trading_days | {advanced.current_ftmo_day}

    updated = replace(
        advanced,
        balance=balance,
        floating_pnl=event.floating_pnl,
        open_position_ids=frozenset(open_positions),
        trading_days=frozenset(trading_days),
        last_event_at=event.timestamp,
    )
    return _evaluate_terminal_status(updated, rules)


def daily_loss_floor(state: AccountState, rules: PropRuleSet) -> Decimal:
    """Return the active daily equity floor."""

    amount = state.config.initial_capital * rules.loss_limits.maximum_daily_loss
    return state.daily_reset_balance - amount


def maximum_loss_floor(state: AccountState, rules: PropRuleSet) -> Decimal:
    """Return the static maximum-loss equity floor."""

    amount = state.config.initial_capital * rules.loss_limits.maximum_loss
    return state.config.initial_capital - amount


def _validate_position_operations(state: AccountState, event: AccountEvent) -> None:
    opened = set(event.opened_position_ids)
    closed = set(event.closed_position_ids)
    existing = set(state.open_position_ids)

    duplicate_opens = opened & existing
    if duplicate_opens:
        raise AccountEventError(
            f"positions already open: {', '.join(sorted(duplicate_opens))}"
        )
    missing_closes = closed - existing
    if missing_closes:
        raise AccountEventError(
            f"positions not open: {', '.join(sorted(missing_closes))}"
        )
    if event.realised_pnl != ZERO and not closed:
        raise AccountEventError("non-zero realised P/L requires a position close")

    resulting_positions = (existing - closed) | opened
    if not resulting_positions and event.floating_pnl != ZERO:
        raise AccountEventError(
            "floating P/L must be zero when no positions remain open"
        )


def _advance_reset_boundaries(
    state: AccountState,
    timestamp: datetime,
    rules: PropRuleSet,
) -> AccountState:
    target_day = _ftmo_day(timestamp, rules.daily_reset)
    advanced = state
    while advanced.current_ftmo_day < target_day:
        advanced = replace(
            advanced,
            current_ftmo_day=advanced.current_ftmo_day + timedelta(days=1),
            daily_reset_balance=advanced.balance,
        )
        advanced = _evaluate_terminal_status(advanced, rules)
        if advanced.status is AccountStatus.BREACHED:
            break
    return advanced


def _evaluate_terminal_status(state: AccountState, rules: PropRuleSet) -> AccountState:
    if state.equity < daily_loss_floor(state, rules):
        return replace(
            state,
            status=AccountStatus.BREACHED,
            breach_reason=BreachReason.MAXIMUM_DAILY_LOSS,
        )
    if state.equity < maximum_loss_floor(state, rules):
        return replace(
            state,
            status=AccountStatus.BREACHED,
            breach_reason=BreachReason.MAXIMUM_LOSS,
        )

    phase = _phase_rules(rules, state.config)
    target_balance = state.config.initial_capital * (Decimal("1") + phase.profit_target)
    if (
        state.balance >= target_balance
        and len(state.trading_days) >= phase.minimum_trading_days
        and not state.open_position_ids
    ):
        return replace(state, status=AccountStatus.PASSED, breach_reason=None)
    return state


def _phase_rules(rules: PropRuleSet, config: RuntimeAccountConfig) -> PhaseRules:
    for phase in rules.phases:
        if phase.phase is config.active_phase:
            return phase
    raise ValueError(f"rules do not define phase {config.active_phase.value}")


def _ftmo_day(timestamp: datetime, reset: DailyReset) -> date:
    local = timestamp.astimezone(reset.timezone)
    reset_today = datetime.combine(local.date(), reset.time, tzinfo=reset.timezone)
    if local >= reset_today:
        return local.date()
    return local.date() - timedelta(days=1)


def _as_utc(timestamp: datetime) -> datetime:
    return timestamp.astimezone(UTC)


def _require_aware(timestamp: datetime, field: str) -> None:
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
