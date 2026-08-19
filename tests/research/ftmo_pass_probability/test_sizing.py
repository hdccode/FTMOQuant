from __future__ import annotations

from decimal import Decimal

import pytest

from ftmoquant.data.instruments import USDCAD_OANDA_SPEC
from ftmoquant.research.ftmo_pass_probability.path_extraction import TradeRecord
from ftmoquant.research.ftmo_pass_probability.sizing import (
    NOTIONAL_REFINEMENT_GRID,
    SIZING_GRID,
    SizingFamily,
    apply_sizing,
)

_STOP_TRADE = TradeRecord(
    trade_index=0,
    entry_ns=1_000_000_000,
    exit_ns=2_000_000_000,
    exit_reason="stop",
    net_r=Decimal("-1.02"),
    original_realized_pnl=Decimal("-340"),
    original_risk_budget=Decimal("333.33"),
    usd_risk_per_unit=Decimal("0.00333333"),
)
_TARGET_TRADE = TradeRecord(
    trade_index=1,
    entry_ns=3_000_000_000,
    exit_ns=4_000_000_000,
    exit_reason="target",
    net_r=Decimal("2.0"),
    original_realized_pnl=Decimal("667"),
    original_risk_budget=Decimal("333.5"),
    usd_risk_per_unit=Decimal("0.0033350"),
)


def test_exactly_eight_frozen_candidates() -> None:
    assert len(SIZING_GRID) == 8
    fractional = [
        p for p in SIZING_GRID if p.family is SizingFamily.FIXED_FRACTIONAL_RISK
    ]
    notional = [
        p for p in SIZING_GRID if p.family is SizingFamily.FIXED_NOTIONAL_MULTIPLIER
    ]
    assert {p.risk_fraction for p in fractional} == {
        Decimal("0.0025"),
        Decimal("0.0050"),
        Decimal("0.0075"),
        Decimal("0.0100"),
    }
    assert {p.notional_multiplier for p in notional} == {
        Decimal("0.5"),
        Decimal("1.0"),
        Decimal("1.5"),
        Decimal("2.0"),
    }


@pytest.mark.parametrize(
    "policy", [p for p in SIZING_GRID if p.family is SizingFamily.FIXED_FRACTIONAL_RISK]
)
def test_fixed_fractional_risk_budget_is_causal_and_proportional(policy) -> None:
    low_balance = Decimal("50000")
    high_balance = Decimal("150000")
    low = apply_sizing(policy, _TARGET_TRADE, low_balance)
    high = apply_sizing(policy, _TARGET_TRADE, high_balance)
    # risk budget scales with balance at entry, not with the eventual R.
    ratio = high.risk_budget / low.risk_budget
    assert abs(ratio - (high_balance / low_balance)) < Decimal("0.001")
    expected_low_budget = policy.risk_fraction * low_balance
    assert abs(low.risk_budget - expected_low_budget) < Decimal("1")


@pytest.mark.parametrize(
    "policy",
    [p for p in SIZING_GRID if p.family is SizingFamily.FIXED_NOTIONAL_MULTIPLIER],
)
def test_fixed_notional_multiplier_scales_original_pnl_linearly(policy) -> None:
    sized = apply_sizing(policy, _TARGET_TRADE, Decimal("100000"))
    expected = policy.notional_multiplier * _TARGET_TRADE.original_realized_pnl
    assert abs(sized.realized_pnl - expected) < Decimal("0.01")


def test_stop_exit_floor_equals_the_realized_close_exactly() -> None:
    policy = next(p for p in SIZING_GRID if p.policy_id == "fixed_notional_1_0x")
    sized = apply_sizing(policy, _STOP_TRADE, Decimal("100000"))
    assert sized.floor_equity_delta == min(sized.realized_pnl, Decimal("0"))


def test_target_exit_floor_is_the_conservative_negative_risk_budget() -> None:
    policy = next(p for p in SIZING_GRID if p.policy_id == "fixed_notional_1_0x")
    sized = apply_sizing(policy, _TARGET_TRADE, Decimal("100000"))
    assert sized.floor_equity_delta == -sized.risk_budget
    # the conservative bound is always worse (more negative) than the
    # trade's own eventual positive outcome.
    assert sized.floor_equity_delta < sized.realized_pnl


def test_sizing_output_is_a_pure_function_of_policy_trade_and_balance() -> None:
    """No martingale/loss-recovery/streak dependence: identical inputs must
    produce an identical result regardless of any simulated trading history."""

    policy = next(p for p in SIZING_GRID if p.policy_id == "fixed_fractional_1_00pct")
    first = apply_sizing(policy, _TARGET_TRADE, Decimal("83456.12"))
    second = apply_sizing(policy, _TARGET_TRADE, Decimal("83456.12"))
    assert first == second


def test_quantity_respects_the_instrument_minimum_size_increment() -> None:
    policy = next(p for p in SIZING_GRID if p.policy_id == "fixed_fractional_0_25pct")
    sized = apply_sizing(policy, _TARGET_TRADE, Decimal("100000"))
    increment = Decimal(USDCAD_OANDA_SPEC.size_increment)
    assert sized.quantity % increment == 0


def test_zero_or_negative_balance_is_rejected() -> None:
    policy = SIZING_GRID[0]
    with pytest.raises(ValueError):
        apply_sizing(policy, _TARGET_TRADE, Decimal("0"))


def test_notional_refinement_grid_has_exactly_nine_fixed_notional_candidates() -> None:
    assert len(NOTIONAL_REFINEMENT_GRID) == 9
    assert all(
        policy.family is SizingFamily.FIXED_NOTIONAL_MULTIPLIER
        for policy in NOTIONAL_REFINEMENT_GRID
    )
    assert {policy.notional_multiplier for policy in NOTIONAL_REFINEMENT_GRID} == {
        Decimal("1.50"),
        Decimal("1.65"),
        Decimal("1.75"),
        Decimal("1.85"),
        Decimal("1.95"),
        Decimal("2.05"),
        Decimal("2.15"),
        Decimal("2.25"),
        Decimal("2.50"),
    }


def test_notional_refinement_grid_does_not_alter_the_frozen_sizing_grid() -> None:
    assert len(SIZING_GRID) == 8
    ids = {policy.policy_id for policy in SIZING_GRID}
    assert ids == {
        "fixed_fractional_0_25pct",
        "fixed_fractional_0_50pct",
        "fixed_fractional_0_75pct",
        "fixed_fractional_1_00pct",
        "fixed_notional_0_5x",
        "fixed_notional_1_0x",
        "fixed_notional_1_5x",
        "fixed_notional_2_0x",
    }
    assert ids.isdisjoint({p.policy_id for p in NOTIONAL_REFINEMENT_GRID})


@pytest.mark.parametrize("policy", list(NOTIONAL_REFINEMENT_GRID))
def test_refinement_policy_sizing_scales_original_pnl_linearly(policy) -> None:
    sized = apply_sizing(policy, _TARGET_TRADE, Decimal("100000"))
    expected = policy.notional_multiplier * _TARGET_TRADE.original_realized_pnl
    assert abs(sized.realized_pnl - expected) < Decimal("0.01")
