"""Tests proving the pure FTMO state machine's rule semantics.

``RULE_CONFIG`` is the same real, official-source-backed
``config/prop/ftmo_2step_swing_2026-08.yaml`` used everywhere else in this
repo (e.g. ``tests/backtest/test_execution_harness.py``) -- these tests do
not invent a second, synthetic rule set.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from ftmoquant.prop_rules import EvaluationPhase, load_prop_rule_set
from ftmoquant.research.ftmo_pass_probability.state_machine import (
    FtmoPathStatus,
    TradeEvent,
    _trading_day_from_ns,
    simulate_phase,
)
from ftmoquant.risk import ftmo_overlay as _overlay_module

RULE_CONFIG = Path("config/prop/ftmo_2step_swing_2026-08.yaml").resolve()
INITIAL_CAPITAL = Decimal("100000")
ONE_YEAR_NS = 365 * 24 * 60 * 60 * 1_000_000_000
_DAY_NS = 24 * 60 * 60 * 1_000_000_000


@pytest.fixture(scope="module")
def rules():
    return load_prop_rule_set(RULE_CONFIG)


def _ns(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000_000_000)


def _trade(entry: datetime, holding_ns: int, floor_delta: str, pnl: str) -> TradeEvent:
    entry_ns = _ns(entry)
    return TradeEvent(
        entry_ns=entry_ns,
        exit_ns=entry_ns + holding_ns,
        floor_equity_delta=Decimal(floor_delta),
        realized_pnl=Decimal(pnl),
    )


@pytest.mark.parametrize(
    "phase,target_fraction",
    [(EvaluationPhase.CHALLENGE, "0.10"), (EvaluationPhase.VERIFICATION, "0.05")],
)
def test_profit_targets_match_phase(rules, phase, target_fraction) -> None:
    start = datetime(2024, 1, 2, tzinfo=UTC)
    target = INITIAL_CAPITAL * (Decimal("1") + Decimal(target_fraction))
    trades = [
        _trade(start, _DAY_NS, "-100", "500"),
        _trade(
            start + _week(1),
            _DAY_NS,
            "-100",
            str(target - INITIAL_CAPITAL - Decimal("500")),
        ),
        _trade(start + _week(2), _DAY_NS, "-100", "1"),
        _trade(start + _week(3), _DAY_NS, "-100", "1"),
    ]
    outcome = simulate_phase(
        trades,
        rules=rules,
        phase=phase,
        initial_capital=INITIAL_CAPITAL,
        horizon_ns=ONE_YEAR_NS,
    )
    assert outcome.status is FtmoPathStatus.PASSED
    assert outcome.ending_balance >= target


def _week(n: int):
    from datetime import timedelta

    return timedelta(weeks=n)


def test_daily_loss_is_5pct_of_initial_capital_static(rules) -> None:
    start = datetime(2024, 1, 2, tzinfo=UTC)
    # midnight balance == initial capital (first trade of the phase); a
    # -5.01% intratrade floor must breach, -4.99% must not.
    breaching = [_trade(start, 60_000_000_000, "-5010", "-100")]
    outcome = simulate_phase(
        breaching,
        rules=rules,
        phase=EvaluationPhase.CHALLENGE,
        initial_capital=INITIAL_CAPITAL,
        horizon_ns=ONE_YEAR_NS,
    )
    assert outcome.status is FtmoPathStatus.FAILED_DAILY_LOSS

    safe = [_trade(start, 60_000_000_000, "-4990", "-100")]
    outcome = simulate_phase(
        safe,
        rules=rules,
        phase=EvaluationPhase.CHALLENGE,
        initial_capital=INITIAL_CAPITAL,
        horizon_ns=ONE_YEAR_NS,
    )
    assert outcome.status is FtmoPathStatus.ACTIVE


def test_maximum_loss_is_10pct_of_initial_capital_static(rules) -> None:
    start = datetime(2024, 1, 2, tzinfo=UTC)
    # A gradual decline where every individual day stays within its own 5%
    # daily allowance (so DAILY_LOSS never fires) can still cross the
    # *static* 10%-of-initial-capital floor once the running balance has
    # dropped enough that the (balance-relative) daily floor sits below the
    # (initial-capital-relative) static max-loss floor -- exactly the case
    # this test isolates, proving the two floors are independently enforced.
    trades = [
        _trade(start, _DAY_NS, "-100", "-4500"),  # 100000 -> 95500
        _trade(start + _week(1), _DAY_NS, "-100", "-1600"),  # 95500 -> 93900
        _trade(start + _week(2), 60_000_000_000, "-3950", "-3950"),
    ]
    outcome = simulate_phase(
        trades,
        rules=rules,
        phase=EvaluationPhase.CHALLENGE,
        initial_capital=INITIAL_CAPITAL,
        horizon_ns=ONE_YEAR_NS,
    )
    assert outcome.status is FtmoPathStatus.FAILED_MAX_LOSS
    assert outcome.breach_trade_index == 2


def test_minimum_trading_days_blocks_early_pass(rules) -> None:
    start = datetime(2024, 1, 2, tzinfo=UTC)
    target_pnl = INITIAL_CAPITAL * Decimal("0.10")
    trades = [_trade(start, 60_000_000_000, "-1", str(target_pnl))]
    outcome = simulate_phase(
        trades,
        rules=rules,
        phase=EvaluationPhase.CHALLENGE,
        initial_capital=INITIAL_CAPITAL,
        horizon_ns=ONE_YEAR_NS,
    )
    assert outcome.status is FtmoPathStatus.ACTIVE
    assert outcome.trading_days == 1


def test_unlimited_period_never_produces_a_time_expiry_failure(rules) -> None:
    start = datetime(2024, 1, 2, tzinfo=UTC)
    trades = [_trade(start, 60_000_000_000, "-1", "1")]
    outcome = simulate_phase(
        trades,
        rules=rules,
        phase=EvaluationPhase.CHALLENGE,
        initial_capital=INITIAL_CAPITAL,
        horizon_ns=ONE_YEAR_NS,
    )
    assert outcome.status in (FtmoPathStatus.ACTIVE, FtmoPathStatus.PASSED)
    assert outcome.status not in (
        FtmoPathStatus.FAILED_DAILY_LOSS,
        FtmoPathStatus.FAILED_MAX_LOSS,
    )


def test_floating_pl_is_checked_intraday_even_if_the_trade_recovers(rules) -> None:
    start = datetime(2024, 1, 2, tzinfo=UTC)
    # ends the trade net-positive, but dips below the daily floor mid-hold.
    trades = [_trade(start, 60_000_000_000, "-5100", "+50")]
    outcome = simulate_phase(
        trades,
        rules=rules,
        phase=EvaluationPhase.CHALLENGE,
        initial_capital=INITIAL_CAPITAL,
        horizon_ns=ONE_YEAR_NS,
    )
    assert outcome.status is FtmoPathStatus.FAILED_DAILY_LOSS


def test_no_end_of_trade_shortcut_hides_an_intraday_breach(rules) -> None:
    start = datetime(2024, 1, 2, tzinfo=UTC)
    trades = [_trade(start, 60_000_000_000, "-5100", "+9000")]
    outcome = simulate_phase(
        trades,
        rules=rules,
        phase=EvaluationPhase.CHALLENGE,
        initial_capital=INITIAL_CAPITAL,
        horizon_ns=ONE_YEAR_NS,
    )
    assert outcome.status is FtmoPathStatus.FAILED_DAILY_LOSS
    assert outcome.ending_balance == INITIAL_CAPITAL


def test_daily_loss_precedes_max_loss_when_both_would_be_crossed(rules) -> None:
    start = datetime(2024, 1, 2, tzinfo=UTC)
    trades = [_trade(start, 60_000_000_000, "-11000", "-11000")]
    outcome = simulate_phase(
        trades,
        rules=rules,
        phase=EvaluationPhase.CHALLENGE,
        initial_capital=INITIAL_CAPITAL,
        horizon_ns=ONE_YEAR_NS,
    )
    assert outcome.status is FtmoPathStatus.FAILED_DAILY_LOSS


def test_pass_requires_positions_closed_no_open_position_mid_trade(rules) -> None:
    start = datetime(2024, 1, 2, tzinfo=UTC)
    target_pnl = INITIAL_CAPITAL * Decimal("0.10") + 100
    trades = [
        _trade(start, _DAY_NS, "-1", "1"),
        _trade(start + _week(1), _DAY_NS, "-1", "1"),
        _trade(start + _week(2), _DAY_NS, "-1", "1"),
        _trade(start + _week(3), 60_000_000_000, "-1", str(target_pnl)),
    ]
    outcome = simulate_phase(
        trades,
        rules=rules,
        phase=EvaluationPhase.CHALLENGE,
        initial_capital=INITIAL_CAPITAL,
        horizon_ns=ONE_YEAR_NS,
    )
    # only resolved once the final trade *closes* (positions_open == 0)
    assert outcome.status is FtmoPathStatus.PASSED
    assert outcome.passed_trade_index == 3


def test_overlapping_trades_are_rejected(rules) -> None:
    start = datetime(2024, 1, 2, tzinfo=UTC)
    first = _trade(start, _DAY_NS * 2, "-1", "1")
    second = _trade(start + _week(0) + _hours(6), _DAY_NS, "-1", "1")
    with pytest.raises(ValueError, match="overlap"):
        simulate_phase(
            [first, second],
            rules=rules,
            phase=EvaluationPhase.CHALLENGE,
            initial_capital=INITIAL_CAPITAL,
            horizon_ns=ONE_YEAR_NS,
        )


def _hours(n: int):
    from datetime import timedelta

    return timedelta(hours=n)


def test_out_of_order_trades_are_rejected(rules) -> None:
    start = datetime(2024, 1, 2, tzinfo=UTC)
    first = _trade(start + _week(1), _DAY_NS, "-1", "1")
    second = _trade(start, _DAY_NS, "-1", "1")
    with pytest.raises(ValueError, match="increasing"):
        simulate_phase(
            [first, second],
            rules=rules,
            phase=EvaluationPhase.CHALLENGE,
            initial_capital=INITIAL_CAPITAL,
            horizon_ns=ONE_YEAR_NS,
        )


def test_overnight_multi_day_hold_counts_only_its_entry_trading_day(rules) -> None:
    start = datetime(2024, 1, 2, tzinfo=UTC)  # Prague local ~01:00, still Jan 2 day
    trades = [_trade(start, _DAY_NS * 5, "-1", "100")]
    outcome = simulate_phase(
        trades,
        rules=rules,
        phase=EvaluationPhase.CHALLENGE,
        initial_capital=INITIAL_CAPITAL,
        horizon_ns=ONE_YEAR_NS,
    )
    assert outcome.trading_days == 1


def test_horizon_produces_active_not_a_breach(rules) -> None:
    start = datetime(2024, 1, 2, tzinfo=UTC)
    trades = [
        _trade(start, _DAY_NS, "-1", "1"),
        _trade(start + _week(200), _DAY_NS, "-1", "1"),
    ]
    outcome = simulate_phase(
        trades,
        rules=rules,
        phase=EvaluationPhase.CHALLENGE,
        initial_capital=INITIAL_CAPITAL,
        horizon_ns=ONE_YEAR_NS,
    )
    assert outcome.status is FtmoPathStatus.ACTIVE
    assert outcome.trades_replayed == 1


@pytest.mark.parametrize(
    "moment",
    [
        datetime(2024, 3, 30, 22, 30, tzinfo=UTC),  # around Prague spring-forward
        datetime(2024, 10, 26, 22, 30, tzinfo=UTC),  # around Prague fall-back
        datetime(2024, 6, 15, 12, 0, tzinfo=UTC),
        datetime(2024, 1, 2, 23, 5, tzinfo=UTC),
    ],
)
def test_trading_day_matches_nautilus_ftmo_overlay(rules, moment) -> None:
    ours = _trading_day_from_ns(_ns(moment), rules)
    theirs = _overlay_module._trading_day(moment.astimezone(ZoneInfo("UTC")), rules)
    assert ours == theirs
