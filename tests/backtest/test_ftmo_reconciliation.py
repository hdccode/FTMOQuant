import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from ftmoquant.backtest.execution_harness import (
    InterestRateInput,
    ProbeOrderKind,
    ProbePlan,
    ProbeSide,
)
from ftmoquant.prop_rules import EvaluationPhase
from ftmoquant.risk import FtmoObservationTrigger
from tests.backtest.test_execution_harness import (
    _catalog_root,
    _flat_minutes,
    _run,
)


def test_floating_equity_equal_floor_then_next_completed_pair_breaches(
    tmp_path: Path,
) -> None:
    prices = (
        Decimal("1.10000"),
        Decimal("1.10000"),
        Decimal("1.05000"),
        Decimal("1.04999"),
        Decimal("1.04999"),
    )
    root = _catalog_root(
        tmp_path / "data",
        _flat_minutes(len(prices)),
        base_prices=prices,
        bar_range=Decimal(0),
    )

    result = _run(root, tmp_path / "run", plan=_long_hold())

    paired = _paired_observations(result)
    assert paired[2].equity == "95000.00"
    assert result.ftmo_evaluation["terminal_status"] == "breached"
    assert result.ftmo_evaluation["breach"] == {
        "reason": "maximum_daily_loss",
        "equity": "94999.00",
        "floor": "95000.00",
        "timestamp_ns": paired[3].timestamp_ns,
    }
    assert paired[3].timestamp_ns == paired[2].timestamp_ns + 60_000_000_000


def test_native_fee_can_trigger_breach_in_fill_callback(tmp_path: Path) -> None:
    root = _catalog_root(tmp_path / "data", _flat_minutes(4), flat_ohlc=True)

    result = _run(
        root,
        tmp_path / "run",
        plan=_long_hold(quantity=Decimal("1")),
        fixed_fee=Decimal("5000.01"),
    )

    fill_observation = next(
        item
        for item in result.ftmo_observations
        if item.trigger is FtmoObservationTrigger.ORDER_FILL
    )
    assert fill_observation.balance == "94999.99"
    assert fill_observation.equity == "94999.99"
    assert result.ftmo_evaluation["breach"] == {
        "reason": "maximum_daily_loss",
        "equity": "94999.99",
        "floor": "95000.00",
        "timestamp_ns": fill_observation.timestamp_ns,
    }


def test_native_rollover_breach_is_seen_post_settlement_without_later_trade(
    tmp_path: Path,
) -> None:
    start = datetime(2024, 1, 4, 21, 58, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=index) for index in range(4)]
    root = _catalog_root(tmp_path / "data", timestamps, flat_ohlc=True)

    result = _run(
        root,
        tmp_path / "run",
        plan=_long_hold(),
        rollover=True,
        rollover_records=(
            InterestRateInput("EA19", "2024-01", Decimal(0)),
            InterestRateInput("USA", "2024-01", Decimal("2000")),
        ),
    )

    breach = result.ftmo_evaluation["breach"]
    rollover_observation = next(
        item
        for item in _paired_observations(result)
        if item.timestamp_ns
        == int(datetime(2024, 1, 4, 22, tzinfo=UTC).timestamp() * 1_000_000_000)
        + 1
    )
    assert result.result_summary["fill_count"] == 1
    assert Decimal(rollover_observation.balance) < Decimal("95000")
    assert breach["timestamp_ns"] == rollover_observation.timestamp_ns
    assert breach["equity"] == rollover_observation.equity


def test_prague_midnight_uses_native_balance_and_does_not_count_hold_day(
    tmp_path: Path,
) -> None:
    start = datetime(2024, 1, 4, 22, 57, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=index) for index in range(5)]
    root = _catalog_root(tmp_path / "data", timestamps, flat_ohlc=True)

    result = _run(
        root,
        tmp_path / "run",
        plan=_long_hold(quantity=Decimal("1")),
        fixed_fee=Decimal("100"),
    )

    evaluation = result.ftmo_evaluation
    assert evaluation["current_ftmo_trading_day"] == "2024-01-05"
    assert evaluation["midnight_reset_balance"] == "99900.00"
    assert evaluation["daily_loss_floor"] == "94900.00"
    assert evaluation["counted_trading_days"] == ["2024-01-04"]
    assert evaluation["terminal_status"] == "active"
    assert evaluation["final_native_snapshot"]["open_position_ids"]


def test_overlay_native_state_reconciles_with_reports_and_result(
    tmp_path: Path,
) -> None:
    root = _catalog_root(tmp_path / "data", _flat_minutes(5), flat_ohlc=True)

    result = _run(
        root,
        tmp_path / "run",
        plan=_long_hold(quantity=Decimal("1")),
        fixed_fee=Decimal("2"),
        ftmo_phase=EvaluationPhase.VERIFICATION,
    )

    final_snapshot = result.ftmo_evaluation["final_native_snapshot"]
    assert result.ftmo_evaluation["active_phase"] == "verification"
    assert final_snapshot["balance"] == result.result_summary["ending_balance"]
    assert final_snapshot["equity"] == result.result_summary["ending_equity"]
    assert final_snapshot["open_position_ids"] == result.result_summary[
        "open_position_ids"
    ]

    account = pd.read_csv(result.report_paths["account"])
    assert Decimal(str(account.iloc[-1]["total"])) == Decimal(
        final_snapshot["balance"]
    )
    positions = pd.read_csv(result.report_paths["positions"])
    assert positions.iloc[-1]["position_id"] in final_snapshot["open_position_ids"]
    assert all(item.open_position_ids for item in result.ftmo_observations)

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["ftmo_evaluation"] == result.ftmo_evaluation


def test_identical_runs_reproduce_ftmo_state_and_breach_evidence(
    tmp_path: Path,
) -> None:
    prices = (
        Decimal("1.10000"),
        Decimal("1.10000"),
        Decimal("1.04999"),
        Decimal("1.04999"),
    )
    root = _catalog_root(
        tmp_path / "data",
        _flat_minutes(len(prices)),
        base_prices=prices,
        bar_range=Decimal(0),
    )

    first = _run(root, tmp_path / "first", plan=_long_hold())
    second = _run(root, tmp_path / "second", plan=_long_hold())

    assert first.run_config_id == second.run_config_id
    assert first.ftmo_evaluation == second.ftmo_evaluation
    assert first.ftmo_observations == second.ftmo_observations


def _long_hold(quantity: Decimal = Decimal("100000")) -> ProbePlan:
    return ProbePlan(
        order_kind=ProbeOrderKind.MARKET,
        side=ProbeSide.BUY,
        quantity=quantity,
        entry_bar_index=0,
    )


def _paired_observations(result):
    return [
        item
        for item in result.ftmo_observations
        if item.trigger is FtmoObservationTrigger.PAIRED_BAR_POST_SETTLEMENT
    ]
