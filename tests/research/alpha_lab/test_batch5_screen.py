from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from ftmoquant.research.alpha_lab.batch5_execution import Batch5TradeResult
from ftmoquant.research.alpha_lab.batch5_preregistration import (
    FAMILY_B5A,
    FAMILY_B5B,
    FAMILY_B5C,
)
from ftmoquant.research.alpha_lab.batch5_screen import (
    FrequencyStats,
    SleeveScreenInput,
    evaluate_family,
    evaluate_sleeve,
    load_frozen_policy,
)

FOLDS = tuple(datetime(year, 1, 1, tzinfo=UTC) for year in range(2020, 2025))


def trade(family: str, sleeve: str, year: int, pnl: str = "100") -> Batch5TradeResult:
    timestamp = datetime(year, 6, 1, tzinfo=UTC)
    entry = timestamp - timedelta(hours=1)
    pnl_value = Decimal(pnl)
    return Batch5TradeResult(
        family=family,
        strategy_id="frozen",
        sleeve_id=sleeve,
        instrument_id="EUR/USD.OANDA",
        signal_timestamp=entry,
        actual_entry_timestamp=entry,
        actual_exit_timestamp=timestamp,
        direction="BUY",
        quantity=Decimal(1),
        entry_price=Decimal(1),
        exit_price=Decimal(1),
        pnl_usd=pnl_value,
        return_on_reference_notional=pnl_value / Decimal("100000"),
        holding_seconds=3600,
        cohort_id=None,
    )


def frequency(family: str) -> FrequencyStats:
    if family == FAMILY_B5A:
        return FrequencyStats(36, 12)
    if family == FAMILY_B5B:
        return FrequencyStats(
            daily_holding_observation_count=500,
            position_sign_change_count=20,
            rollover_supported=True,
        )
    return FrequencyStats(event_count=15)


def sleeve_input(family: str, sleeve: str) -> SleeveScreenInput:
    rows = tuple(trade(family, sleeve, year) for year in range(2020, 2024))
    return SleeveScreenInput(family, sleeve, rows, rows, rows, FOLDS, frequency(family))


def test_exact_common_thresholds_and_all_sleeve_gates() -> None:
    policy = load_frozen_policy()
    assert policy.profit_factor_gt == Decimal("1.1")
    assert policy.positive_folds_gte == 3
    assert policy.aggregate_max_drawdown_lte == Decimal("0.15")
    row = evaluate_sleeve(sleeve_input(FAMILY_B5B, "B5B_AUDCAD"))
    assert row.positive_fold_count == 4
    assert row.max_year_positive_profit_share == Decimal("0.25")
    assert row.profitable_year_fraction == Decimal(1)
    assert row.all_sleeve_gates_passed


def test_frequency_floor_and_rollover_fail_closed() -> None:
    base = sleeve_input(FAMILY_B5B, "B5B_AUDCAD")
    failed = SleeveScreenInput(
        base.family,
        base.sleeve_id,
        base.native_trades,
        base.stressed_1_5x_trades,
        base.stressed_2_0x_trades,
        base.fold_boundaries,
        FrequencyStats(
            daily_holding_observation_count=499,
            position_sign_change_count=20,
            rollover_supported=False,
        ),
    )
    assert not evaluate_sleeve(failed).frequency_floor_passed


def test_profit_factor_at_exactly_1_10_fails_strict_gate() -> None:
    base = sleeve_input(FAMILY_B5B, "B5B_AUDCAD")
    rows = (
        trade(FAMILY_B5B, "B5B_AUDCAD", 2020, "55"),
        trade(FAMILY_B5B, "B5B_AUDCAD", 2021, "55"),
        trade(FAMILY_B5B, "B5B_AUDCAD", 2022, "-50"),
        trade(FAMILY_B5B, "B5B_AUDCAD", 2023, "-50"),
    )
    inputs = SleeveScreenInput(
        base.family,
        base.sleeve_id,
        rows,
        base.stressed_1_5x_trades,
        base.stressed_2_0x_trades,
        base.fold_boundaries,
        base.frequency,
    )
    result = evaluate_sleeve(inputs)
    assert result.profit_factor == Decimal("1.1")
    assert not result.all_sleeve_gates_passed


def test_exact_breadth_and_no_pair_cherry_picking() -> None:
    b5a_ids = ("JPY", "CHF", "EUR", "GBP", "CAD", "AUD", "NZD")
    b5a = [sleeve_input(FAMILY_B5A, f"B5A_{currency}") for currency in b5a_ids]
    b5a_result = evaluate_family(b5a)
    assert b5a_result.sleeve_count == 7
    assert b5a_result.full_gate_sleeve_count == 7
    assert b5a_result.validation_eligible

    b5c_ids = ("EURUSD", "USDJPY", "USDCAD", "AUDUSD", "EURJPY")
    b5c = [sleeve_input(FAMILY_B5C, f"B5C_{name}") for name in b5c_ids]
    b5c_result = evaluate_family(b5c)
    assert b5c_result.family_event_count == 75
    assert b5c_result.validation_eligible


def test_equal_weight_family_drawdown_is_a_hard_gate() -> None:
    base = sleeve_input(FAMILY_B5B, "B5B_AUDCAD")
    losses = (
        trade(FAMILY_B5B, "B5B_AUDCAD", 2020, "20000"),
        trade(FAMILY_B5B, "B5B_AUDCAD", 2021, "-16000"),
        trade(FAMILY_B5B, "B5B_AUDCAD", 2022, "100"),
        trade(FAMILY_B5B, "B5B_AUDCAD", 2023, "100"),
    )
    inputs = SleeveScreenInput(
        base.family,
        base.sleeve_id,
        losses,
        base.stressed_1_5x_trades,
        base.stressed_2_0x_trades,
        base.fold_boundaries,
        base.frequency,
    )
    result = evaluate_family([inputs])
    assert result.equal_weight_maximum_drawdown == Decimal("0.16")
    assert not result.aggregate_drawdown_passed
    assert not result.validation_eligible
