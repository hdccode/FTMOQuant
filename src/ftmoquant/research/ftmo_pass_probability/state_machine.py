"""Pure, deterministic FTMO 2-Step evaluation state machine.

Independent of Monte Carlo, Nautilus, and pandas/numpy. It exists so that
thousands of synthetic (bootstrap-resampled) trade paths can be replayed
cheaply, without the cost of a Nautilus ``BacktestEngine``.

This module deliberately reuses the FTMO *rule model*
(:class:`ftmoquant.prop_rules.models.PropRuleSet`) rather than re-deriving
profit targets, loss limits, or minimum trading days. The one piece of logic
duplicated from :mod:`ftmoquant.risk.ftmo_overlay` is the CE(S)T
trading-day/reset computation, because that overlay's version is private and
coupled to a live Nautilus ``Clock``. ``tests/research/ftmo_pass_probability/
test_state_machine.py::test_trading_day_matches_nautilus_ftmo_overlay`` proves
byte-for-byte parity between the two implementations across DST transitions,
so this is a parity-tested reimplementation, not an independent guess.

Breach precedence (daily loss checked before max loss, both checked before
any profit target) mirrors ``NautilusFtmoOverlay._evaluate`` exactly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from enum import StrEnum

from ftmoquant.prop_rules.models import EvaluationPhase, PhaseRules, PropRuleSet

ZERO = Decimal("0")


class FtmoPathStatus(StrEnum):
    """Terminal-aware status of one simulated phase path."""

    ACTIVE = "active"
    PASSED = "passed"
    FAILED_DAILY_LOSS = "failed_daily_loss"
    FAILED_MAX_LOSS = "failed_max_loss"
    CENSORED_NOT_PASSED = "censored_not_passed"


@dataclass(frozen=True, slots=True)
class TradeEvent:
    """One causal trade to replay through the state machine.

    ``floor_equity_delta`` is the worst mark-to-market equity offset (<= 0,
    relative to the balance at this trade's entry) reached while the trade
    was open. See ``path_extraction.py`` for how it is derived: exact for
    stop-exits (the exit *is* the worst mark, because a stop order fires on
    first touch/gap-through), a proven conservative (never-optimistic) bound
    for target-exits (bounded by the frozen stop distance, which the trade
    never reached).

    ``realized_pnl`` is the dollar P/L already scaled by the sizing policy
    under evaluation, in the account currency (USD).
    """

    entry_ns: int
    exit_ns: int
    floor_equity_delta: Decimal
    realized_pnl: Decimal

    def __post_init__(self) -> None:
        if self.exit_ns <= self.entry_ns:
            raise ValueError("trade exit must be strictly after entry")
        if not self.floor_equity_delta.is_finite() or self.floor_equity_delta > ZERO:
            raise ValueError("floor_equity_delta must be finite and <= 0")
        if not self.realized_pnl.is_finite():
            raise ValueError("realized_pnl must be finite")


@dataclass(frozen=True, slots=True)
class PhaseOutcome:
    """Immutable result of replaying one phase to a terminal or censored state."""

    status: FtmoPathStatus
    ending_balance: Decimal
    trading_days: int
    trades_replayed: int
    breach_trade_index: int | None
    passed_trade_index: int | None
    max_drawdown: Decimal


def simulate_phase(
    trades: Sequence[TradeEvent],
    *,
    rules: PropRuleSet,
    phase: EvaluationPhase,
    initial_capital: Decimal,
    horizon_ns: int,
) -> PhaseOutcome:
    """Replay a chronological trade sequence against one FTMO phase.

    ``trades`` must be in strictly increasing ``entry_ns`` order and contain
    no overlapping holds (both preconditions of the single-position frozen
    strategy this module was built for; violated inputs raise rather than
    silently reordering).

    A path that neither passes nor breaches before ``horizon_ns`` elapses
    (measured from the first trade's entry) is returned ``ACTIVE`` — the
    caller decides whether that is ``CENSORED_NOT_PASSED`` (it always is, at
    the fixed synthetic horizon this module is designed for; kept separate
    so a shorter smoke horizon can distinguish "still active" from
    "censored" in tests).
    """

    phase_rules = _phase_rules(rules, phase)
    if not initial_capital.is_finite() or initial_capital <= ZERO:
        raise ValueError("initial_capital must be a positive finite Decimal")
    if horizon_ns <= 0:
        raise ValueError("horizon_ns must be positive")
    _validate_chronology(trades)

    balance = initial_capital
    peak = initial_capital
    max_drawdown = ZERO
    trading_days: set[date] = set()
    max_loss_floor = initial_capital - initial_capital * rules.loss_limits.maximum_loss
    target = initial_capital * (Decimal("1") + phase_rules.profit_target)
    horizon_end_ns = trades[0].entry_ns + horizon_ns if trades else None
    processed = 0

    for index, trade in enumerate(trades):
        if horizon_end_ns is not None and trade.entry_ns >= horizon_end_ns:
            break
        processed = index + 1

        trading_days.add(_trading_day_from_ns(trade.entry_ns, rules))
        daily_floor = balance - initial_capital * rules.loss_limits.maximum_daily_loss

        floor_equity = balance + trade.floor_equity_delta
        max_drawdown = max(max_drawdown, _drawdown(peak, floor_equity, initial_capital))
        breach = _breach_status(floor_equity, daily_floor, max_loss_floor)
        if breach is not None:
            return PhaseOutcome(
                status=breach,
                ending_balance=balance,
                trading_days=len(trading_days),
                trades_replayed=index + 1,
                breach_trade_index=index,
                passed_trade_index=None,
                max_drawdown=max_drawdown,
            )

        balance = balance + trade.realized_pnl
        peak = max(peak, balance)
        max_drawdown = max(max_drawdown, _drawdown(peak, balance, initial_capital))
        breach = _breach_status(balance, daily_floor, max_loss_floor)
        if breach is not None:
            return PhaseOutcome(
                status=breach,
                ending_balance=balance,
                trading_days=len(trading_days),
                trades_replayed=index + 1,
                breach_trade_index=index,
                passed_trade_index=None,
                max_drawdown=max_drawdown,
            )

        if balance >= target and len(trading_days) >= phase_rules.minimum_trading_days:
            return PhaseOutcome(
                status=FtmoPathStatus.PASSED,
                ending_balance=balance,
                trading_days=len(trading_days),
                trades_replayed=index + 1,
                breach_trade_index=None,
                passed_trade_index=index,
                max_drawdown=max_drawdown,
            )

    return PhaseOutcome(
        status=FtmoPathStatus.ACTIVE,
        ending_balance=balance,
        trading_days=len(trading_days),
        trades_replayed=processed,
        breach_trade_index=None,
        passed_trade_index=None,
        max_drawdown=max_drawdown,
    )


def _breach_status(
    equity: Decimal, daily_floor: Decimal, max_loss_floor: Decimal
) -> FtmoPathStatus | None:
    if equity < daily_floor:
        return FtmoPathStatus.FAILED_DAILY_LOSS
    if equity < max_loss_floor:
        return FtmoPathStatus.FAILED_MAX_LOSS
    return None


def _drawdown(peak: Decimal, equity: Decimal, initial_capital: Decimal) -> Decimal:
    if equity >= peak:
        return ZERO
    return (peak - equity) / initial_capital


def _validate_chronology(trades: Sequence[TradeEvent]) -> None:
    previous_exit_ns: int | None = None
    previous_entry_ns: int | None = None
    for trade in trades:
        if previous_entry_ns is not None and trade.entry_ns <= previous_entry_ns:
            raise ValueError("trades must be strictly increasing by entry_ns")
        if previous_exit_ns is not None and trade.entry_ns < previous_exit_ns:
            raise ValueError("trades must not overlap (single open position)")
        previous_entry_ns = trade.entry_ns
        previous_exit_ns = trade.exit_ns


def _phase_rules(rules: PropRuleSet, phase: EvaluationPhase) -> PhaseRules:
    for phase_rules in rules.phases:
        if phase_rules.phase is phase:
            return phase_rules
    raise ValueError(f"rules do not define phase {phase.value}")


def _trading_day_from_ns(timestamp_ns: int, rules: PropRuleSet) -> date:
    """Return the CE(S)T FTMO trading day for a UTC nanosecond timestamp.

    Parity-tested against ``ftmoquant.risk.ftmo_overlay._trading_day`` (see
    module docstring); DST-aware via ``zoneinfo`` exactly as that overlay is.
    """

    timestamp = _from_ns(timestamp_ns)
    local = timestamp.astimezone(rules.daily_reset.timezone)
    reset = datetime.combine(
        local.date(), rules.daily_reset.time, tzinfo=rules.daily_reset.timezone
    )
    return local.date() if local >= reset else local.date() - timedelta(days=1)


def _from_ns(timestamp_ns: int) -> datetime:
    seconds, nanoseconds = divmod(timestamp_ns, 1_000_000_000)
    return datetime.fromtimestamp(seconds, tz=UTC) + timedelta(
        microseconds=nanoseconds // 1_000
    )
