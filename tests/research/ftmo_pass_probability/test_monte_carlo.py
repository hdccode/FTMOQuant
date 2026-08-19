from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from ftmoquant.prop_rules import load_prop_rule_set
from ftmoquant.research.ftmo_pass_probability.monte_carlo import (
    build_synthetic_trade_placements,
    precompute_trade_timing,
    simulate_two_phase_path,
    size_synthetic_path,
)
from ftmoquant.research.ftmo_pass_probability.path_extraction import TradeRecord
from ftmoquant.research.ftmo_pass_probability.sizing import SIZING_GRID
from ftmoquant.research.ftmo_pass_probability.state_machine import FtmoPathStatus

RULE_CONFIG = Path("config/prop/ftmo_2step_swing_2026-08.yaml").resolve()
ONE_YEAR_NS = 365 * 24 * 60 * 60 * 1_000_000_000
_HOUR_NS = 3_600_000_000_000


def _tiny_fixture(count: int = 30) -> tuple[TradeRecord, ...]:
    trades = []
    entry = 1_000_000_000_000
    for i in range(count):
        is_win = i % 3 != 0
        net_r = Decimal("2.0") if is_win else Decimal("-1.0")
        risk_budget = Decimal("300")
        trades.append(
            TradeRecord(
                trade_index=i,
                entry_ns=entry,
                exit_ns=entry + _HOUR_NS,
                exit_reason="target" if is_win else "stop",
                net_r=net_r,
                original_realized_pnl=net_r * risk_budget,
                original_risk_budget=risk_budget,
                usd_risk_per_unit=Decimal("0.003"),
            )
        )
        entry += 2 * _HOUR_NS + i * 60_000_000_000
    return tuple(trades)


def test_precompute_trade_timing_reuses_each_trades_own_gap_and_holding() -> None:
    trades = _tiny_fixture()
    timing = precompute_trade_timing(trades)
    assert timing[0].gap_before_ns == 0
    for trade, timed in zip(trades, timing):
        assert timed.holding_ns == trade.exit_ns - trade.entry_ns
    for previous, current, timed in zip(trades, trades[1:], timing[1:]):
        assert timed.gap_before_ns == current.entry_ns - previous.exit_ns


def test_build_synthetic_trade_placements_truncates_at_horizon() -> None:
    trades = _tiny_fixture()
    timing = precompute_trade_timing(trades)
    placements = build_synthetic_trade_placements(
        trades, timing, list(range(len(trades))), horizon_ns=5 * _HOUR_NS
    )
    assert placements
    assert all(entry_ns < 5 * _HOUR_NS for _, entry_ns, _ in placements)


def test_size_synthetic_path_is_causal_and_stops_when_balance_is_exhausted() -> None:
    trades = _tiny_fixture()
    timing = precompute_trade_timing(trades)
    policy = next(p for p in SIZING_GRID if p.policy_id == "fixed_fractional_1_00pct")
    placements = build_synthetic_trade_placements(
        trades, timing, list(range(len(trades))), horizon_ns=ONE_YEAR_NS
    )
    events = size_synthetic_path(policy, placements, Decimal("100000"))
    assert len(events) <= len(placements)
    running_balance = Decimal("100000")
    for event in events:
        assert running_balance > 0
        running_balance += event.realized_pnl


def test_two_phase_simulation_end_to_end_tiny_fixture() -> None:
    rules = load_prop_rule_set(RULE_CONFIG)
    trades = _tiny_fixture(count=40)
    timing = precompute_trade_timing(trades)
    policy = next(p for p in SIZING_GRID if p.policy_id == "fixed_notional_2_0x")
    outcome = simulate_two_phase_path(
        trades,
        timing,
        method="stationary",
        block_size=3,
        policy=policy,
        rules=rules,
        initial_capital=Decimal("100000"),
        challenge_horizon_ns=ONE_YEAR_NS,
        verification_horizon_ns=ONE_YEAR_NS,
        seed=123,
    )
    assert outcome.challenge.status in (
        FtmoPathStatus.PASSED,
        FtmoPathStatus.FAILED_DAILY_LOSS,
        FtmoPathStatus.FAILED_MAX_LOSS,
        FtmoPathStatus.CENSORED_NOT_PASSED,
    )
    if outcome.challenge.status is not FtmoPathStatus.PASSED:
        assert outcome.verification is None
        assert outcome.passed_both is False
    else:
        assert outcome.verification is not None


def test_verification_is_only_attempted_after_challenge_passes() -> None:
    rules = load_prop_rule_set(RULE_CONFIG)
    trades = _tiny_fixture(count=10)  # too few/short to ever pass Challenge
    timing = precompute_trade_timing(trades)
    policy = next(p for p in SIZING_GRID if p.policy_id == "fixed_fractional_0_25pct")
    outcome = simulate_two_phase_path(
        trades,
        timing,
        method="stationary",
        block_size=2,
        policy=policy,
        rules=rules,
        initial_capital=Decimal("100000"),
        challenge_horizon_ns=_HOUR_NS,  # tiny horizon, guarantees censoring
        verification_horizon_ns=ONE_YEAR_NS,
        seed=5,
    )
    assert outcome.challenge.status is FtmoPathStatus.CENSORED_NOT_PASSED
    assert outcome.verification is None
    assert outcome.passed_both is False


def test_a_censored_challenge_never_counts_as_passed_both() -> None:
    rules = load_prop_rule_set(RULE_CONFIG)
    trades = _tiny_fixture(count=5)
    timing = precompute_trade_timing(trades)
    policy = SIZING_GRID[0]
    for seed in range(5):
        outcome = simulate_two_phase_path(
            trades,
            timing,
            method="circular",
            block_size=2,
            policy=policy,
            rules=rules,
            initial_capital=Decimal("100000"),
            challenge_horizon_ns=1,  # smaller than any trade's own gap
            verification_horizon_ns=ONE_YEAR_NS,
            seed=seed,
        )
        assert outcome.challenge.status is FtmoPathStatus.CENSORED_NOT_PASSED
        assert outcome.passed_both is False
