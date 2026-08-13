"""FTMO 2-Step state overlay backed by NautilusTrader account state."""

from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Protocol, cast

from nautilus_trader.common import Cache, Clock, TimeEvent
from nautilus_trader.model import (
    Currency,
    MarginAccount,
    Money,
    OrderFilled,
    Position,
    PositionOpened,
    Venue,
)
from nautilus_trader.portfolio import Portfolio

from ftmoquant.prop_rules.models import EvaluationPhase, PhaseRules, PropRuleSet

ZERO = Decimal("0")


class FtmoStatus(StrEnum):
    """Terminal-aware status of the FTMO evaluation overlay."""

    ACTIVE = "active"
    PASSED = "passed"
    BREACHED = "breached"


class FtmoBreachReason(StrEnum):
    """FTMO loss limits enforced by this overlay."""

    MAXIMUM_DAILY_LOSS = "maximum_daily_loss"
    MAXIMUM_LOSS = "maximum_loss"


@dataclass(frozen=True, slots=True)
class FtmoRuntimeConfig:
    """Account-specific inputs that do not belong in the provider rule set."""

    initial_capital: Decimal
    currency: str
    active_phase: EvaluationPhase

    def __post_init__(self) -> None:
        if (
            not isinstance(self.initial_capital, Decimal)
            or not self.initial_capital.is_finite()
            or self.initial_capital <= ZERO
        ):
            raise ValueError("initial_capital must be a positive finite Decimal")
        if not isinstance(self.currency, str) or not self.currency.strip():
            raise ValueError("currency must be a non-empty string")
        if not isinstance(self.active_phase, EvaluationPhase):
            raise ValueError("active_phase must be an EvaluationPhase")


@dataclass(frozen=True, slots=True)
class FtmoBreachDetails:
    """Immutable evidence captured when a loss floor is first breached."""

    reason: FtmoBreachReason
    timestamp_ns: int
    equity: Decimal
    floor: Decimal


@dataclass(frozen=True, slots=True)
class FtmoOverlayState:
    """Only the state Nautilus does not own."""

    current_trading_day: date
    midnight_reset_balance: Decimal
    daily_loss_floor: Decimal
    maximum_loss_floor: Decimal
    trading_days: frozenset[date]
    status: FtmoStatus = FtmoStatus.ACTIVE
    breach: FtmoBreachDetails | None = None


@dataclass(frozen=True, slots=True)
class NativeAccountSnapshot:
    """Ephemeral read-only view of Nautilus-owned account state."""

    balance: Decimal
    equity: Decimal
    open_position_ids: frozenset[str]


class AccountSnapshotSource(Protocol):
    """Minimal boundary through which the overlay reads Nautilus state."""

    @property
    def currency(self) -> str:
        """Return the account snapshot currency code."""

    def snapshot(self) -> NativeAccountSnapshot:
        """Return current native account and portfolio values."""


class NautilusAccountSnapshotSource:
    """Read balance, equity, and positions directly from Nautilus."""

    def __init__(
        self,
        cache: Cache,
        portfolio: Portfolio,
        venue: Venue,
        currency: Currency,
    ) -> None:
        self._cache = cache
        self._portfolio = portfolio
        self._venue = venue
        self._currency = currency

    @property
    def currency(self) -> str:
        """Return the native account currency code."""

        return self._currency.code

    def snapshot(self) -> NativeAccountSnapshot:
        """Read a non-persistent snapshot without calculating another ledger."""

        account = cast(
            MarginAccount | None,
            self._cache.account_for_venue(self._venue),
        )
        if account is None:
            raise RuntimeError(f"no Nautilus account for venue {self._venue.value}")
        balance = account.balance_total(self._currency)
        if balance is None:
            raise RuntimeError(
                f"no {self._currency.code} balance for venue {self._venue.value}"
            )

        equities = cast(
            dict[Currency, Money],
            self._portfolio.equity(venue=self._venue),
        )
        equity = equities.get(self._currency)
        if equity is None:
            raise RuntimeError(
                f"no {self._currency.code} equity for venue {self._venue.value}"
            )

        positions = cast(
            list[Position],
            self._cache.positions_open(venue=self._venue),
        )
        return NativeAccountSnapshot(
            balance=balance.as_decimal(),
            equity=equity.as_decimal(),
            open_position_ids=frozenset(position.id.value for position in positions),
        )


class NautilusFtmoOverlay:
    """Apply FTMO-only rules to state read from NautilusTrader."""

    def __init__(
        self,
        runtime: FtmoRuntimeConfig,
        rules: PropRuleSet,
        clock: Clock,
        account_source: AccountSnapshotSource,
        timer_name: str = "ftmo-daily-reset",
    ) -> None:
        self.runtime = runtime
        self.rules = rules
        self._phase = _phase_rules(rules, runtime.active_phase)
        self._clock = clock
        self._account_source = account_source
        self._timer_name = timer_name
        if account_source.currency != runtime.currency:
            raise ValueError(
                "runtime currency must match the Nautilus account snapshot currency"
            )

        now = clock.utc_now()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("Nautilus clock must return timezone-aware timestamps")
        initial_daily_floor = (
            runtime.initial_capital
            - runtime.initial_capital * rules.loss_limits.maximum_daily_loss
        )
        maximum_floor = (
            runtime.initial_capital
            - runtime.initial_capital * rules.loss_limits.maximum_loss
        )
        self._state = FtmoOverlayState(
            current_trading_day=_trading_day(now, rules),
            midnight_reset_balance=runtime.initial_capital,
            daily_loss_floor=initial_daily_floor,
            maximum_loss_floor=maximum_floor,
            trading_days=frozenset(),
        )
        self._schedule_next_reset(now)

    @property
    def state(self) -> FtmoOverlayState:
        """Return the current immutable FTMO state."""

        return self._state

    @property
    def next_reset_ns(self) -> int:
        """Return the next reset scheduled on the native Nautilus clock."""

        scheduled = self._clock.next_time_ns(self._timer_name)
        if scheduled is None:
            raise RuntimeError("FTMO daily reset is not scheduled")
        return scheduled

    def on_position_opened(self, event: PositionOpened) -> None:
        """Consume a native position-open event and count its FTMO day once."""

        self.record_position_opened(event.ts_event)

    def on_order_filled(self, event: OrderFilled) -> None:
        """Re-evaluate after Nautilus has applied a native execution event."""

        self.refresh(event.ts_event)

    def record_position_opened(self, timestamp_ns: int) -> None:
        """Record a qualifying position-open timestamp from Nautilus."""

        self.record_position_opened_day(timestamp_ns)
        self.refresh(timestamp_ns)

    def record_position_opened_day(self, timestamp_ns: int) -> None:
        """Count the native opening day without reading floating equity."""

        if self._state.status is not FtmoStatus.ACTIVE:
            return
        event_time = _from_ns(timestamp_ns)
        trading_day = _trading_day(event_time, self.rules)
        self._state = replace(
            self._state,
            trading_days=self._state.trading_days | {trading_day},
        )

    def refresh(self, timestamp_ns: int | None = None) -> None:
        """Evaluate FTMO status against the latest Nautilus-owned snapshot."""

        if self._state.status is not FtmoStatus.ACTIVE:
            return
        observed_at = (
            self._clock.timestamp_ns() if timestamp_ns is None else timestamp_ns
        )
        self._evaluate(self._account_source.snapshot(), observed_at)

    def on_time_event(self, event: TimeEvent) -> None:
        """Capture the native balance at a scheduled Prague reset boundary."""

        if (
            event.name != self._timer_name
            or self._state.status is not FtmoStatus.ACTIVE
        ):
            return
        snapshot = self._account_source.snapshot()
        event_time = _from_ns(event.ts_event)
        daily_floor = (
            snapshot.balance
            - self.runtime.initial_capital
            * self.rules.loss_limits.maximum_daily_loss
        )
        self._state = replace(
            self._state,
            current_trading_day=_trading_day(event_time, self.rules),
            midnight_reset_balance=snapshot.balance,
            daily_loss_floor=daily_floor,
        )
        self._evaluate(snapshot, event.ts_event)
        if self._state.status is FtmoStatus.ACTIVE:
            self._schedule_next_reset(event_time)

    def _evaluate(self, snapshot: NativeAccountSnapshot, timestamp_ns: int) -> None:
        if snapshot.equity < self._state.daily_loss_floor:
            self._breach(
                FtmoBreachReason.MAXIMUM_DAILY_LOSS,
                snapshot.equity,
                self._state.daily_loss_floor,
                timestamp_ns,
            )
            return
        if snapshot.equity < self._state.maximum_loss_floor:
            self._breach(
                FtmoBreachReason.MAXIMUM_LOSS,
                snapshot.equity,
                self._state.maximum_loss_floor,
                timestamp_ns,
            )
            return

        target = self.runtime.initial_capital * (
            Decimal("1") + self._phase.profit_target
        )
        if (
            snapshot.balance >= target
            and len(self._state.trading_days) >= self._phase.minimum_trading_days
            and not snapshot.open_position_ids
        ):
            self._state = replace(self._state, status=FtmoStatus.PASSED)

    def _breach(
        self,
        reason: FtmoBreachReason,
        equity: Decimal,
        floor: Decimal,
        timestamp_ns: int,
    ) -> None:
        self._state = replace(
            self._state,
            status=FtmoStatus.BREACHED,
            breach=FtmoBreachDetails(
                reason=reason,
                timestamp_ns=timestamp_ns,
                equity=equity,
                floor=floor,
            ),
        )

    def _schedule_next_reset(self, after: datetime) -> None:
        next_reset = _next_reset(after, self.rules)
        self._clock.set_time_alert(
            name=self._timer_name,
            alert_time=next_reset,
            callback=self.on_time_event,
        )


def _phase_rules(rules: PropRuleSet, phase: EvaluationPhase) -> PhaseRules:
    for phase_rules in rules.phases:
        if phase_rules.phase is phase:
            return phase_rules
    raise ValueError(f"rules do not define phase {phase.value}")


def _trading_day(timestamp: datetime, rules: PropRuleSet) -> date:
    local = timestamp.astimezone(rules.daily_reset.timezone)
    reset = datetime.combine(
        local.date(),
        rules.daily_reset.time,
        tzinfo=rules.daily_reset.timezone,
    )
    return local.date() if local >= reset else local.date() - timedelta(days=1)


def _next_reset(after: datetime, rules: PropRuleSet) -> datetime:
    local = after.astimezone(rules.daily_reset.timezone)
    candidate = datetime.combine(
        local.date(),
        rules.daily_reset.time,
        tzinfo=rules.daily_reset.timezone,
    )
    if candidate <= local:
        candidate = datetime.combine(
            local.date() + timedelta(days=1),
            rules.daily_reset.time,
            tzinfo=rules.daily_reset.timezone,
        )
    return candidate.astimezone(UTC)


def _from_ns(timestamp_ns: int) -> datetime:
    seconds, nanoseconds = divmod(timestamp_ns, 1_000_000_000)
    return datetime.fromtimestamp(seconds, tz=UTC) + timedelta(
        microseconds=nanoseconds // 1_000
    )
