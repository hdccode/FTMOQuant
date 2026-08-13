from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from nautilus_trader.common import Clock, TimeEvent
from nautilus_trader.core import UUID4

from ftmoquant.prop_rules import EvaluationPhase, load_prop_rule_set
from ftmoquant.risk import (
    FtmoRuntimeConfig,
    FtmoStatus,
    NativeAccountSnapshot,
    NautilusFtmoOverlay,
)
from tests.oracle.g03 import (
    OracleEvent,
    OracleState,
    apply_oracle_event,
    create_oracle,
    daily_floor,
    maximum_floor,
)

CONFIG_PATH = Path("config/prop/ftmo_2step_swing_2026-08.yaml")
NS = 1_000_000_000


@dataclass(slots=True)
class _SnapshotSource:
    value: NativeAccountSnapshot
    currency: str

    def snapshot(self) -> NativeAccountSnapshot:
        return self.value


class _DifferentialFixture:
    def __init__(
        self,
        started_at: datetime,
        initial_capital: Decimal = Decimal("100000"),
        currency: str = "USD",
        phase: EvaluationPhase = EvaluationPhase.CHALLENGE,
    ) -> None:
        self.rules = load_prop_rule_set(CONFIG_PATH)
        self.clock = Clock.new_test()
        self.clock.set_time(_to_ns(started_at))
        self.oracle = create_oracle(
            initial_capital,
            currency,
            phase,
            self.rules,
            started_at,
        )
        self.source = _SnapshotSource(_snapshot(self.oracle), currency)
        self.overlay = NautilusFtmoOverlay(
            runtime=FtmoRuntimeConfig(initial_capital, currency, phase),
            rules=self.rules,
            clock=self.clock,
            account_source=self.source,
        )
        self.assert_equivalent()

    def apply(self, event: OracleEvent) -> None:
        self.clock.set_time(_to_ns(event.timestamp))
        self.oracle = apply_oracle_event(self.oracle, event, self.rules)
        self.source.value = _snapshot(self.oracle)
        if event.opened_position_ids:
            for _position_id in event.opened_position_ids:
                self.overlay.record_position_opened(_to_ns(event.timestamp))
        else:
            self.overlay.refresh(_to_ns(event.timestamp))
        self.assert_equivalent()

    def reset(self, timestamp: datetime) -> None:
        timestamp_ns = _to_ns(timestamp)
        self.clock.set_time(timestamp_ns)
        self.clock.cancel_timer("ftmo-daily-reset")
        self.source.value = _snapshot(self.oracle)
        self.overlay.on_time_event(
            TimeEvent("ftmo-daily-reset", UUID4(), timestamp_ns, timestamp_ns)
        )
        self.oracle = apply_oracle_event(
            self.oracle,
            OracleEvent(timestamp=timestamp, floating_pnl=self.oracle.floating_pnl),
            self.rules,
        )
        self.source.value = _snapshot(self.oracle)
        self.assert_equivalent()

    def assert_equivalent(self) -> None:
        actual = self.overlay.state
        expected = self.oracle
        assert actual.current_trading_day == expected.current_ftmo_day
        assert actual.midnight_reset_balance == expected.daily_reset_balance
        assert actual.daily_loss_floor == daily_floor(expected, self.rules)
        assert actual.maximum_loss_floor == maximum_floor(expected, self.rules)
        assert actual.trading_days == expected.trading_days
        assert actual.status.value == expected.status.value
        actual_reason = actual.breach.reason.value if actual.breach else None
        assert actual_reason == expected.breach_reason


def test_floating_loss_crossing_midnight_matches_g03_oracle() -> None:
    fixture = _DifferentialFixture(_utc(2026, 1, 4, 21))
    fixture.apply(
        OracleEvent(
            _utc(2026, 1, 4, 21, 30),
            Decimal("0"),
            opened_position_ids=("P-0",),
        )
    )
    fixture.apply(
        OracleEvent(
            _utc(2026, 1, 4, 22),
            Decimal("0"),
            closed_position_ids=("P-0",),
            realised_pnl=Decimal("4000"),
        )
    )
    fixture.apply(
        OracleEvent(
            _utc(2026, 1, 4, 22, 30),
            Decimal("-6000"),
            opened_position_ids=("P-1",),
        )
    )

    assert fixture.overlay.state.status is FtmoStatus.ACTIVE
    fixture.reset(_utc(2026, 1, 4, 23))

    assert fixture.overlay.state.status is FtmoStatus.BREACHED


def test_profitable_day_raises_next_daily_floor_matches_g03_oracle() -> None:
    fixture = _DifferentialFixture(_utc(2026, 1, 4, 20))
    fixture.apply(
        OracleEvent(
            _utc(2026, 1, 4, 21),
            Decimal("-4000"),
            opened_position_ids=("P-1",),
        )
    )
    fixture.apply(
        OracleEvent(
            _utc(2026, 1, 4, 22),
            Decimal("0"),
            closed_position_ids=("P-1",),
            realised_pnl=Decimal("2000"),
        )
    )

    fixture.reset(_utc(2026, 1, 4, 23))

    assert fixture.overlay.state.midnight_reset_balance == Decimal("102000")
    assert fixture.overlay.state.daily_loss_floor == Decimal("97000")


def test_commission_driven_breach_matches_g03_oracle() -> None:
    fixture = _DifferentialFixture(_utc(2026, 1, 5, 8))
    fixture.apply(
        OracleEvent(
            _utc(2026, 1, 5, 9),
            Decimal("-4999"),
            opened_position_ids=("P-1",),
        )
    )
    fixture.apply(
        OracleEvent(
            _utc(2026, 1, 5, 10),
            Decimal("-4999"),
            commission=Decimal("2"),
        )
    )

    assert fixture.overlay.state.status is FtmoStatus.BREACHED
    breach = fixture.overlay.state.breach
    fixture.source.value = NativeAccountSnapshot(
        balance=Decimal("100000"),
        equity=Decimal("100000"),
        open_position_ids=frozenset(),
    )
    fixture.overlay.refresh(_to_ns(_utc(2026, 1, 5, 11)))
    assert fixture.overlay.state.breach is breach


def test_multiple_simultaneous_positions_match_g03_oracle() -> None:
    fixture = _DifferentialFixture(_utc(2026, 1, 5, 8))
    fixture.apply(
        OracleEvent(
            _utc(2026, 1, 5, 9),
            Decimal("25"),
            opened_position_ids=("P-1", "P-2"),
        )
    )
    fixture.apply(
        OracleEvent(
            _utc(2026, 1, 5, 10),
            Decimal("10"),
            closed_position_ids=("P-1",),
            realised_pnl=Decimal("15"),
        )
    )

    assert len(fixture.overlay.state.trading_days) == 1
    assert fixture.oracle.open_position_ids == frozenset({"P-2"})


def test_target_reached_with_open_position_does_not_pass() -> None:
    fixture = _fixture_with_four_days_and_open_position()
    fixture.apply(
        OracleEvent(
            _utc(2026, 1, 8, 10),
            Decimal("0"),
            closed_position_ids=("P-4",),
            realised_pnl=Decimal("10000"),
            opened_position_ids=("P-5",),
        )
    )

    assert fixture.overlay.state.status is FtmoStatus.ACTIVE

    fixture.apply(
        OracleEvent(
            _utc(2026, 1, 8, 11),
            Decimal("0"),
            closed_position_ids=("P-5",),
        )
    )
    assert fixture.overlay.state.status is FtmoStatus.PASSED


def test_exact_loss_floor_equality_survives() -> None:
    fixture = _DifferentialFixture(_utc(2026, 1, 5, 8))
    fixture.apply(
        OracleEvent(
            _utc(2026, 1, 5, 9),
            Decimal("-5000"),
            opened_position_ids=("P-1",),
        )
    )

    assert fixture.source.value.equity == fixture.overlay.state.daily_loss_floor
    assert fixture.overlay.state.status is FtmoStatus.ACTIVE


def test_four_distinct_trading_days_are_counted_once_each() -> None:
    fixture = _fixture_with_four_days_and_open_position()

    assert len(fixture.overlay.state.trading_days) == 4
    assert fixture.overlay.state.status is FtmoStatus.ACTIVE


def test_verification_uses_its_configured_profit_target_without_transition() -> None:
    fixture = _fixture_with_four_days_and_open_position(
        phase=EvaluationPhase.VERIFICATION
    )
    fixture.apply(
        OracleEvent(
            _utc(2026, 1, 8, 10),
            Decimal("0"),
            closed_position_ids=("P-4",),
            realised_pnl=Decimal("5000"),
        )
    )

    assert fixture.overlay.runtime.active_phase is EvaluationPhase.VERIFICATION
    assert fixture.overlay.state.status is FtmoStatus.PASSED


def test_position_opened_before_midnight_closed_after_counts_open_day() -> None:
    fixture = _DifferentialFixture(_utc(2026, 1, 4, 22, 30))
    fixture.apply(
        OracleEvent(
            _utc(2026, 1, 4, 22, 45),
            Decimal("20"),
            opened_position_ids=("P-1",),
        )
    )
    fixture.reset(_utc(2026, 1, 4, 23))
    fixture.apply(
        OracleEvent(
            _utc(2026, 1, 4, 23, 15),
            Decimal("0"),
            closed_position_ids=("P-1",),
            realised_pnl=Decimal("20"),
        )
    )

    assert len(fixture.overlay.state.trading_days) == 1


def test_account_size_and_currency_independence_match_g03_oracle() -> None:
    small = _DifferentialFixture(
        _utc(2026, 1, 5, 8), Decimal("10000"), "EUR"
    )
    large = _DifferentialFixture(
        _utc(2026, 1, 5, 8), Decimal("200000"), "USD"
    )

    assert small.overlay.state.daily_loss_floor == Decimal("9500")
    assert large.overlay.state.daily_loss_floor == Decimal("190000")
    assert small.overlay.runtime.currency == "EUR"
    assert large.overlay.runtime.currency == "USD"


def test_prague_spring_dst_reset_is_23_hours() -> None:
    fixture = _DifferentialFixture(_utc(2026, 3, 28, 22, 30))

    assert fixture.overlay.next_reset_ns == _to_ns(_utc(2026, 3, 28, 23))
    fixture.reset(_utc(2026, 3, 28, 23))
    assert fixture.overlay.next_reset_ns == _to_ns(_utc(2026, 3, 29, 22))


def test_prague_autumn_dst_reset_is_25_hours() -> None:
    fixture = _DifferentialFixture(_utc(2026, 10, 24, 21, 30))

    assert fixture.overlay.next_reset_ns == _to_ns(_utc(2026, 10, 24, 22))
    fixture.reset(_utc(2026, 10, 24, 22))
    assert fixture.overlay.next_reset_ns == _to_ns(_utc(2026, 10, 25, 23))


def test_midnight_reset_without_trade_event_matches_g03_oracle() -> None:
    fixture = _DifferentialFixture(_utc(2026, 1, 4, 22, 30))

    fixture.reset(_utc(2026, 1, 4, 23))

    assert fixture.overlay.state.current_trading_day.isoformat() == "2026-01-05"
    assert fixture.overlay.state.midnight_reset_balance == Decimal("100000")


def _fixture_with_four_days_and_open_position(
    phase: EvaluationPhase = EvaluationPhase.CHALLENGE,
) -> _DifferentialFixture:
    fixture = _DifferentialFixture(_utc(2026, 1, 5, 8), phase=phase)
    for day in range(5, 9):
        if day > 5:
            fixture.reset(_utc(2026, 1, day - 1, 23))
        fixture.apply(
            OracleEvent(
                _utc(2026, 1, day, 9),
                Decimal("0"),
                opened_position_ids=(f"P-{day - 4}",),
            )
        )
        if day < 8:
            fixture.apply(
                OracleEvent(
                    _utc(2026, 1, day, 10),
                    Decimal("0"),
                    closed_position_ids=(f"P-{day - 4}",),
                )
            )
    return fixture


def _snapshot(state: OracleState) -> NativeAccountSnapshot:
    return NativeAccountSnapshot(
        balance=state.balance,
        equity=state.equity,
        open_position_ids=state.open_position_ids,
    )


def _utc(
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int = 0,
) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def _to_ns(timestamp: datetime) -> int:
    return int(timestamp.timestamp() * NS)
