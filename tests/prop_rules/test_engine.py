from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ftmoquant.prop_rules import (
    AccountEvent,
    AccountEventError,
    AccountState,
    AccountStatus,
    BreachReason,
    EvaluationPhase,
    RuntimeAccountConfig,
    apply_account_event,
    create_account_state,
    daily_loss_floor,
    load_prop_rule_set,
    maximum_loss_floor,
)

RULES = load_prop_rule_set(Path("config/prop/ftmo_2step_swing_2026-08.yaml"))
PRAGUE = ZoneInfo("Europe/Prague")
STARTED_AT = datetime(2026, 8, 12, 9, tzinfo=PRAGUE)
ZERO = Decimal("0")


def test_floating_loss_becomes_breach_at_midnight() -> None:
    state = _new_state()
    state = _apply(state, _at(12, 10), opened=("winner",), floating=Decimal("4000"))
    state = _apply(
        state,
        _at(12, 11),
        closed=("winner",),
        realised=Decimal("4000"),
    )
    state = _apply(state, _at(12, 12), opened=("loser",), floating=Decimal("-8500"))

    assert state.equity == Decimal("95500")
    assert state.status is AccountStatus.ACTIVE

    state = _apply(state, _at(13, 0, 1), floating=Decimal("-8500"))

    assert state.status is AccountStatus.BREACHED
    assert state.breach_reason is BreachReason.MAXIMUM_DAILY_LOSS
    assert state.daily_reset_balance == Decimal("104000")
    assert daily_loss_floor(state, RULES) == Decimal("99000.00")


def test_profitable_day_raises_next_days_daily_floor() -> None:
    state = _new_state()
    state = _apply(state, _at(12, 10), opened=("winner",), floating=Decimal("4000"))
    state = _apply(
        state,
        _at(12, 11),
        closed=("winner",),
        realised=Decimal("4000"),
    )

    state = _apply(state, _at(13, 0, 1))

    assert state.daily_reset_balance == Decimal("104000")
    assert daily_loss_floor(state, RULES) == Decimal("99000.00")
    assert state.status is AccountStatus.ACTIVE


@pytest.mark.parametrize(
    ("commission", "swap"),
    [
        (Decimal("5000.01"), ZERO),
        (ZERO, Decimal("-5000.01")),
    ],
)
def test_commission_or_swap_can_cause_breach(
    commission: Decimal,
    swap: Decimal,
) -> None:
    state = _new_state()
    state = _apply(state, _at(12, 10), opened=("position",))
    state = _apply(
        state,
        _at(12, 11),
        commission=commission,
        swap=swap,
    )

    assert state.status is AccountStatus.BREACHED
    assert state.breach_reason is BreachReason.MAXIMUM_DAILY_LOSS
    assert state.balance == Decimal("94999.99")


def test_simultaneous_positions_are_accounted_independently() -> None:
    state = _new_state()
    state = _apply(
        state,
        _at(12, 10),
        opened=("one", "two"),
        floating=Decimal("300"),
    )
    state = _apply(
        state,
        _at(12, 11),
        closed=("one",),
        realised=Decimal("100"),
        floating=Decimal("-50"),
    )

    assert state.open_position_ids == frozenset({"two"})
    assert state.balance == Decimal("100100")
    assert state.equity == Decimal("100050")

    state = _apply(
        state,
        _at(12, 12),
        closed=("two",),
        realised=Decimal("-50"),
        commission=Decimal("10"),
        swap=Decimal("-5"),
    )

    assert state.open_position_ids == frozenset()
    assert state.balance == Decimal("100035")
    assert state.equity == state.balance


def test_target_reached_with_open_position_does_not_pass() -> None:
    state = _new_state()
    state = _complete_flat_days(state, count=3)
    state = _apply(
        state,
        _at(15, 10),
        opened=("target", "runner"),
        floating=Decimal("10000"),
    )
    state = _apply(
        state,
        _at(15, 11),
        closed=("target",),
        realised=Decimal("10000"),
        floating=Decimal("25"),
    )

    assert len(state.trading_days) == 4
    assert state.balance == Decimal("110000")
    assert state.status is AccountStatus.ACTIVE

    state = _apply(state, _at(15, 12), closed=("runner",))

    assert state.status is AccountStatus.PASSED


def test_exact_loss_boundary_equality_is_valid() -> None:
    state = _new_state()
    state = _apply(
        state,
        _at(12, 10),
        opened=("position",),
        commission=Decimal("5000"),
    )

    assert state.equity == daily_loss_floor(state, RULES)
    assert state.status is AccountStatus.ACTIVE


def test_four_distinct_trading_days_satisfy_minimum() -> None:
    state = _new_state()
    state = _complete_flat_days(state, count=3)
    state = _apply(state, _at(15, 10), opened=("day-4",))
    state = _apply(
        state,
        _at(15, 11),
        closed=("day-4",),
        realised=Decimal("10000"),
    )

    assert state.trading_days == {
        date(2026, 8, 12),
        date(2026, 8, 13),
        date(2026, 8, 14),
        date(2026, 8, 15),
    }
    assert state.status is AccountStatus.PASSED


def test_active_phase_selects_target_without_transitioning() -> None:
    challenge = _complete_flat_days(_new_state(), count=3)
    challenge = _apply(challenge, _at(15, 10), opened=("day-4",))
    challenge = _apply(
        challenge,
        _at(15, 11),
        closed=("day-4",),
        realised=Decimal("5000"),
    )

    verification = _complete_flat_days(
        _new_state(active_phase=EvaluationPhase.VERIFICATION), count=3
    )
    verification = _apply(verification, _at(15, 10), opened=("day-4",))
    verification = _apply(
        verification,
        _at(15, 11),
        closed=("day-4",),
        realised=Decimal("5000"),
    )

    assert challenge.status is AccountStatus.ACTIVE
    assert challenge.config.active_phase is EvaluationPhase.CHALLENGE
    assert verification.status is AccountStatus.PASSED
    assert verification.config.active_phase is EvaluationPhase.VERIFICATION


def test_open_before_midnight_close_after_counts_opening_day_only() -> None:
    state = _new_state(started_at=_at(12, 22))
    state = _apply(state, _at(12, 23, 59), opened=("overnight",))
    state = _apply(state, _at(13, 0, 1), closed=("overnight",))

    assert state.current_ftmo_day == date(2026, 8, 13)
    assert state.trading_days == {date(2026, 8, 12)}


def test_account_size_and_currency_are_runtime_independent() -> None:
    usd = _new_state(initial_capital=Decimal("100000"), currency="USD")
    eur = _new_state(initial_capital=Decimal("50000"), currency="EUR")

    assert usd.config.currency == "USD"
    assert eur.config.currency == "EUR"
    assert daily_loss_floor(usd, RULES) == Decimal("95000.00")
    assert daily_loss_floor(eur, RULES) == Decimal("47500.00")
    assert maximum_loss_floor(usd, RULES) == Decimal("90000.00")
    assert maximum_loss_floor(eur, RULES) == Decimal("45000.00")


@pytest.mark.parametrize(
    ("started_at", "event_times", "expected_days"),
    [
        (
            datetime(2026, 3, 28, 22, 30, tzinfo=UTC),
            (
                datetime(2026, 3, 28, 23, 30, tzinfo=UTC),
                datetime(2026, 3, 29, 22, 30, tzinfo=UTC),
            ),
            (date(2026, 3, 29), date(2026, 3, 30)),
        ),
        (
            datetime(2026, 10, 24, 21, 30, tzinfo=UTC),
            (
                datetime(2026, 10, 24, 22, 30, tzinfo=UTC),
                datetime(2026, 10, 25, 23, 30, tzinfo=UTC),
            ),
            (date(2026, 10, 25), date(2026, 10, 26)),
        ),
    ],
)
def test_prague_resets_follow_dst_boundaries(
    started_at: datetime,
    event_times: tuple[datetime, datetime],
    expected_days: tuple[date, date],
) -> None:
    state = _new_state(started_at=started_at)

    for event_time, expected_day in zip(event_times, expected_days, strict=True):
        state = _apply(state, event_time)
        assert state.current_ftmo_day == expected_day


def test_rejects_invalid_event_ordering_and_position_operations() -> None:
    state = _new_state()
    with pytest.raises(AccountEventError, match="later than"):
        _apply(state, STARTED_AT)
    with pytest.raises(AccountEventError, match="not open"):
        _apply(state, _at(12, 10), closed=("missing",))
    with pytest.raises(AccountEventError, match="requires a position close"):
        _apply(state, _at(12, 10), realised=Decimal("1"))
    with pytest.raises(AccountEventError, match="no positions remain"):
        _apply(state, _at(12, 10), floating=Decimal("1"))

    state = _apply(state, _at(12, 10), opened=("position",))
    with pytest.raises(AccountEventError, match="already open"):
        _apply(state, _at(12, 11), opened=("position",))
    with pytest.raises(AccountEventError, match="later than"):
        _apply(state, _at(12, 9))


def test_rejects_naive_timestamps_and_events_after_breach() -> None:
    with pytest.raises(AccountEventError, match="timezone-aware"):
        AccountEvent(timestamp=datetime(2026, 8, 12, 10), floating_pnl=ZERO)

    state = _new_state()
    state = _apply(
        state,
        _at(12, 10),
        opened=("position",),
        commission=Decimal("5000.01"),
    )
    with pytest.raises(AccountEventError, match="breached account"):
        _apply(state, _at(12, 11), floating=ZERO)


@given(
    initial_capital=st.decimals(
        min_value="100",
        max_value="1000000",
        places=2,
        allow_nan=False,
        allow_infinity=False,
    )
)
def test_loss_floors_scale_and_breach_strictly_below(initial_capital: Decimal) -> None:
    state = _new_state(initial_capital=initial_capital)
    daily_amount = initial_capital * Decimal("0.05")

    assert daily_loss_floor(state, RULES) == initial_capital - daily_amount
    assert maximum_loss_floor(state, RULES) == initial_capital * Decimal("0.90")

    state = _apply(
        state,
        _at(12, 10),
        opened=("position",),
        floating=-daily_amount,
    )
    assert state.status is AccountStatus.ACTIVE

    state = _apply(
        state,
        _at(12, 11),
        floating=-daily_amount - Decimal("0.0001"),
    )
    assert state.status is AccountStatus.BREACHED


@given(
    realised=st.decimals(min_value="-1000", max_value="1000", places=2),
    commission=st.decimals(min_value="0", max_value="100", places=2),
    swap=st.decimals(min_value="-100", max_value="100", places=2),
)
def test_balance_components_are_applied_exactly_once(
    realised: Decimal,
    commission: Decimal,
    swap: Decimal,
) -> None:
    state = _new_state()
    state = _apply(state, _at(12, 10), opened=("position",))
    state = _apply(
        state,
        _at(12, 11),
        closed=("position",),
        realised=realised,
        commission=commission,
        swap=swap,
    )

    expected = Decimal("100000") + realised - commission + swap
    assert state.balance == expected
    assert state.equity == expected


def _new_state(
    *,
    initial_capital: Decimal = Decimal("100000"),
    currency: str = "USD",
    active_phase: EvaluationPhase = EvaluationPhase.CHALLENGE,
    started_at: datetime = STARTED_AT,
) -> AccountState:
    config = RuntimeAccountConfig(
        initial_capital=initial_capital,
        currency=currency,
        active_phase=active_phase,
    )
    return create_account_state(config, RULES, started_at)


def _apply(
    state: AccountState,
    timestamp: datetime,
    *,
    opened: tuple[str, ...] = (),
    closed: tuple[str, ...] = (),
    realised: Decimal = ZERO,
    commission: Decimal = ZERO,
    swap: Decimal = ZERO,
    floating: Decimal = ZERO,
) -> AccountState:
    event = AccountEvent(
        timestamp=timestamp,
        floating_pnl=floating,
        opened_position_ids=opened,
        closed_position_ids=closed,
        realised_pnl=realised,
        commission=commission,
        swap=swap,
    )
    return apply_account_event(state, event, RULES)


def _at(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 8, day, hour, minute, tzinfo=PRAGUE)


def _complete_flat_days(state: AccountState, count: int) -> AccountState:
    for offset in range(count):
        day = 12 + offset
        position_id = f"day-{offset + 1}"
        state = _apply(state, _at(day, 10), opened=(position_id,))
        state = _apply(state, _at(day, 11), closed=(position_id,))
    return state
