from __future__ import annotations

import dataclasses
import json
from datetime import UTC, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pytest

from ftmoquant.research.alpha_lab.batch4_clock_scheduler import (
    FrozenClockSpec,
    load_frozen_clock_specs,
)
from ftmoquant.research.alpha_lab.batch4_development_orchestrator import (
    DEVELOPMENT_FOLD_BOUNDARIES,
)
from ftmoquant.research.alpha_lab.batch4_execution import ScheduledTradeResult
from ftmoquant.research.alpha_lab.batch4_screen import (
    FAMILY_LOCAL,
    FAMILY_LONDON,
    FAMILY_TOKYO,
    Batch4ScorecardRow,
    Batch4ScreenError,
    apply_parameter_neighborhood,
    build_diagnostics_summary,
    build_family_summary,
    compute_family_robustness,
    evaluate_hypothesis,
    load_frozen_screen_policy,
    select_representative,
    write_batch4_artifacts,
)


def _spec(spec_id: str = "B4F1A_GBP") -> FrozenClockSpec:
    return next(
        spec for spec in load_frozen_clock_specs() if spec.hypothesis_id == spec_id
    )


def _trades(
    spec: FrozenClockSpec,
    *,
    count: int = 260,
    pnl_scale: Decimal = Decimal("1"),
) -> tuple[ScheduledTradeResult, ...]:
    span = DEVELOPMENT_FOLD_BOUNDARIES[-1] - DEVELOPMENT_FOLD_BOUNDARIES[0]
    gap = span / count
    rows = []
    for index in range(count):
        scheduled_entry = (
            DEVELOPMENT_FOLD_BOUNDARIES[0] + gap * index + timedelta(hours=2)
        )
        scheduled_exit = scheduled_entry + timedelta(hours=1)
        actual_entry = scheduled_entry + timedelta(minutes=1)
        actual_exit = scheduled_exit + timedelta(minutes=1)
        pnl = (Decimal("2") if index % 4 != 0 else Decimal("-1")) * pnl_scale
        rows.append(
            ScheduledTradeResult(
                hypothesis_id=spec.hypothesis_id,
                family=spec.family,
                instrument_id=spec.instrument_id,
                direction=spec.direction,
                local_date=scheduled_entry.astimezone(UTC).date().isoformat(),
                scheduled_entry_utc=scheduled_entry,
                scheduled_exit_utc=scheduled_exit,
                actual_entry_utc=actual_entry,
                actual_exit_utc=actual_exit,
                entry_price=Decimal("1.2"),
                exit_price=Decimal("1.21"),
                quantity=Decimal("100000"),
                pnl_account_currency=pnl,
                account_currency="USD",
                reference_notional_usd=Decimal("100000"),
                return_on_reference_notional=pnl / Decimal("100000"),
                holding_seconds=3600,
            )
        )
    return tuple(rows)


def _healthy_row(spec: FrozenClockSpec | None = None) -> Batch4ScorecardRow:
    selected = spec or _spec()
    trades = _trades(selected)
    return evaluate_hypothesis(
        spec=selected,
        native_trades=trades,
        native_skip_count=3,
        stressed_1_5x_trades=_trades(selected, pnl_scale=Decimal("0.8")),
        stressed_2_0x_trades=_trades(selected, pnl_scale=Decimal("0.6")),
        fold_boundaries=DEVELOPMENT_FOLD_BOUNDARIES,
        policy=load_frozen_screen_policy(),
    )


def _row_for_spec(
    spec: FrozenClockSpec,
    template: Batch4ScorecardRow,
    *,
    base_passed: bool,
    full_passed: bool | None = None,
    core_passed: bool | None = None,
) -> Batch4ScorecardRow:
    if spec.family == FAMILY_LOCAL:
        config, phase, duration = "LOCAL", None, None
    else:
        config = spec.hypothesis_id.split(":")[1]
        phase, duration_text = config.split("_")
        duration = int(duration_text.removesuffix("m"))
    core = base_passed if core_passed is None else core_passed
    full = base_passed if full_passed is None else full_passed
    return dataclasses.replace(
        template,
        hypothesis_id=spec.hypothesis_id,
        family=spec.family,
        instrument_id=spec.instrument_id,
        direction=spec.direction,
        timezone=spec.timezone,
        local_start=spec.local_start_time.isoformat(timespec="minutes"),
        local_end=spec.local_end_time.isoformat(timespec="minutes"),
        configuration_id=config,
        phase=phase,
        duration_minutes=duration,
        expectancy_usd=Decimal("2") if core else Decimal("-2"),
        profit_factor=Decimal("1.5") if core else Decimal("0.5"),
        stressed_1_5x_expectancy=Decimal("1") if core else Decimal("-1"),
        base_hard_gates_passed=base_passed,
        parameter_neighborhood_applicable=spec.family != FAMILY_LOCAL,
        parameter_neighborhood_passed=full,
        hard_gates_passed=full,
    )


def test_policy_is_loaded_exactly_from_frozen_preregistration() -> None:
    policy = load_frozen_screen_policy()
    assert policy.min_trade_count == 250
    assert policy.profit_factor_gt == Decimal("1.1")
    assert policy.min_positive_folds == 3
    assert policy.max_quarter_share == Decimal("0.4")
    assert policy.stress_multipliers == (Decimal("1.5"), Decimal("2.0"))
    assert policy.min_connected_region_size == 2
    assert policy.breadth_min_core_sleeves == 3
    assert policy.breadth_min_full_gate_sleeves == 2


def test_healthy_hypothesis_passes_base_gates_and_exact_scorecard_metrics() -> None:
    row = _healthy_row()
    assert row.trade_count == 260
    assert row.skip_count == 3
    assert row.expectancy_usd > 0
    assert row.profit_factor > Decimal("1.1")
    assert row.positive_fold_count == 4
    assert row.best_5pct_removed_expectancy > 0
    assert row.quarter_max_share is not None
    assert row.quarter_max_share <= Decimal("0.4")
    assert row.stressed_1_5x_expectancy > 0
    assert row.stressed_2_0x_expectancy > 0
    assert row.base_hard_gates_passed
    assert row.hard_gates_passed


def _replace_pnls(
    trades: tuple[ScheduledTradeResult, ...], pnls: list[Decimal]
) -> tuple[ScheduledTradeResult, ...]:
    assert len(trades) == len(pnls)
    return tuple(
        dataclasses.replace(
            trade,
            pnl_account_currency=pnl,
            return_on_reference_notional=pnl / Decimal("100000"),
        )
        for trade, pnl in zip(trades, pnls, strict=True)
    )


def test_249_trades_fails_exact_opportunity_gate() -> None:
    spec = _spec()
    trades = _trades(spec, count=249)
    row = evaluate_hypothesis(
        spec=spec,
        native_trades=trades,
        native_skip_count=0,
        stressed_1_5x_trades=trades,
        stressed_2_0x_trades=trades,
        fold_boundaries=DEVELOPMENT_FOLD_BOUNDARIES,
        policy=load_frozen_screen_policy(),
    )
    assert row.trade_count == 249
    assert not row.base_hard_gates_passed


def test_profit_factor_equal_to_1_10_fails_strict_gate() -> None:
    spec = _spec()
    source = _trades(spec, count=250)
    pnls = [
        Decimal("0.88") if index % 2 == 0 else Decimal("-0.8") for index in range(250)
    ]
    trades = _replace_pnls(source, pnls)
    row = evaluate_hypothesis(
        spec=spec,
        native_trades=trades,
        native_skip_count=0,
        stressed_1_5x_trades=trades,
        stressed_2_0x_trades=trades,
        fold_boundaries=DEVELOPMENT_FOLD_BOUNDARIES,
        policy=load_frozen_screen_policy(),
    )
    assert row.expectancy_usd > 0
    assert row.profit_factor == Decimal("1.1")
    assert not row.base_hard_gates_passed


def test_only_two_positive_folds_fails_temporal_gate() -> None:
    spec = _spec()
    source = _trades(spec)
    pnls = [
        Decimal("10")
        if trade.actual_exit_utc < DEVELOPMENT_FOLD_BOUNDARIES[2]
        else Decimal("-1")
        for trade in source
    ]
    trades = _replace_pnls(source, pnls)
    row = evaluate_hypothesis(
        spec=spec,
        native_trades=trades,
        native_skip_count=0,
        stressed_1_5x_trades=trades,
        stressed_2_0x_trades=trades,
        fold_boundaries=DEVELOPMENT_FOLD_BOUNDARIES,
        policy=load_frozen_screen_policy(),
    )
    assert row.expectancy_usd > 0
    assert row.profit_factor > Decimal("1.1")
    assert row.positive_fold_count == 2
    assert not row.base_hard_gates_passed


def test_exceptional_winner_removal_fails_when_four_trades_carry_edge() -> None:
    spec = _spec()
    source = _trades(spec)
    pnls = [Decimal("75") if index % 65 == 0 else Decimal("-1") for index in range(260)]
    trades = _replace_pnls(source, pnls)
    row = evaluate_hypothesis(
        spec=spec,
        native_trades=trades,
        native_skip_count=0,
        stressed_1_5x_trades=trades,
        stressed_2_0x_trades=trades,
        fold_boundaries=DEVELOPMENT_FOLD_BOUNDARIES,
        policy=load_frozen_screen_policy(),
    )
    assert row.expectancy_usd > 0
    assert row.profit_factor > Decimal("1.1")
    assert row.positive_fold_count == 4
    assert row.best_5pct_removed_expectancy < 0
    assert not row.base_hard_gates_passed


def test_stress_failure_recomputed_from_trades_fails_base_gate() -> None:
    spec = _spec()
    row = evaluate_hypothesis(
        spec=spec,
        native_trades=_trades(spec),
        native_skip_count=0,
        stressed_1_5x_trades=_trades(spec),
        stressed_2_0x_trades=_trades(spec, pnl_scale=Decimal("-1")),
        fold_boundaries=DEVELOPMENT_FOLD_BOUNDARIES,
        policy=load_frozen_screen_policy(),
    )
    assert row.stressed_2_0x_expectancy < 0
    assert not row.base_hard_gates_passed


def test_report_only_diagnostics_cannot_change_hard_gate_field() -> None:
    row = _healthy_row()
    corrupted = dataclasses.replace(
        row,
        rolling_50_median_expectancy=Decimal("-999"),
        monthly_max_share=Decimal("999"),
        pnl_skewness=-999.0,
        spread_cost_share_of_gross_edge=Decimal("999"),
    )
    assert corrupted.hard_gates_passed == row.hard_gates_passed


def test_exact_fix_adjacency_and_no_pre_post_or_instrument_connection() -> None:
    specs = load_frozen_clock_specs()
    template = _healthy_row()
    passing_base = {
        f"{FAMILY_LONDON}:PRE_15m:EUR/USD.OANDA",
        f"{FAMILY_LONDON}:PRE_30m:EUR/USD.OANDA",
        f"{FAMILY_LONDON}:POST_15m:EUR/USD.OANDA",
        f"{FAMILY_LONDON}:POST_60m:EUR/USD.OANDA",
        f"{FAMILY_LONDON}:PRE_60m:GBP/USD.OANDA",
    }
    rows = tuple(
        _row_for_spec(
            spec,
            template,
            base_passed=(
                spec.family == FAMILY_LOCAL or spec.hypothesis_id in passing_base
            ),
        )
        for spec in specs
    )
    gated = {
        row.hypothesis_id: row
        for row in apply_parameter_neighborhood(rows, load_frozen_screen_policy())
    }
    assert gated[f"{FAMILY_LONDON}:PRE_15m:EUR/USD.OANDA"].hard_gates_passed
    assert gated[f"{FAMILY_LONDON}:PRE_30m:EUR/USD.OANDA"].hard_gates_passed
    assert not gated[f"{FAMILY_LONDON}:POST_15m:EUR/USD.OANDA"].hard_gates_passed
    assert not gated[f"{FAMILY_LONDON}:POST_60m:EUR/USD.OANDA"].hard_gates_passed
    assert not gated[f"{FAMILY_LONDON}:PRE_60m:GBP/USD.OANDA"].hard_gates_passed


def test_local_family_has_no_artificial_contiguity_gate() -> None:
    specs = load_frozen_clock_specs()
    template = _healthy_row()
    rows = tuple(_row_for_spec(spec, template, base_passed=True) for spec in specs)
    gated = apply_parameter_neighborhood(rows, load_frozen_screen_policy())
    local = [row for row in gated if row.family == FAMILY_LOCAL]
    assert len(local) == 7
    assert all(not row.parameter_neighborhood_applicable for row in local)
    assert all(row.parameter_neighborhood_passed for row in local)
    assert all(row.hard_gates_passed for row in local)


def test_family_breadth_requires_three_core_and_two_full_without_pair_selection() -> (
    None
):
    specs = load_frozen_clock_specs()
    template = _healthy_row()
    local_specs = [spec for spec in specs if spec.family == FAMILY_LOCAL]
    core_ids = {spec.hypothesis_id for spec in local_specs[:3]}
    full_ids = {spec.hypothesis_id for spec in local_specs[:2]}
    rows = tuple(
        _row_for_spec(
            spec,
            template,
            base_passed=spec.hypothesis_id in full_ids,
            full_passed=spec.hypothesis_id in full_ids,
            core_passed=spec.hypothesis_id in core_ids,
        )
        for spec in specs
    )
    robustness = compute_family_robustness(rows, load_frozen_screen_policy())
    assert len(robustness) == 13
    local = next(unit for unit in robustness if unit["family"] == FAMILY_LOCAL)
    assert local["eligible_sleeve_count"] == 7
    assert local["breadth_core_passing_sleeve_count"] == 3
    assert local["full_gate_passing_sleeve_count"] == 2
    assert local["family_breadth_passed"] is True


def test_breadth_failure_at_either_exact_threshold() -> None:
    policy = load_frozen_screen_policy()
    specs = load_frozen_clock_specs()
    template = _healthy_row()
    rows = tuple(_row_for_spec(spec, template, base_passed=False) for spec in specs)
    robustness = compute_family_robustness(rows, policy)
    assert not any(unit["family_breadth_passed"] for unit in robustness)
    assert select_representative(robustness, policy)["selected_representative"] is None


def test_representative_selection_uses_exact_lexicographic_tie_break() -> None:
    policy = load_frozen_screen_policy()
    common = {
        "configuration_id": "LOCAL",
        "eligible_sleeve_count": 7,
        "breadth_core_passing_sleeve_count": 3,
        "full_gate_passing_sleeve_count": 2,
        "family_breadth_passed": True,
        "median_sleeve_expectancy": Decimal("2"),
        "median_sleeve_profit_factor": Decimal("1.5"),
        "median_sleeve_quarter_concentration": Decimal("0.2"),
        "aggregate_trade_count": 1000,
    }
    units = [
        {**common, "family": FAMILY_TOKYO, "strategy_id": "z"},
        {**common, "family": FAMILY_LOCAL, "strategy_id": "a"},
    ]
    selected = select_representative(units, policy)
    assert selected["selected_representative"] == "a"
    assert selected["number_permitted_for_future_validation"] == 1
    assert selected["validation_accessed"] is False
    assert selected["rescue_permitted"] is False


def test_summary_and_diagnostics_are_complete_and_report_only() -> None:
    specs = load_frozen_clock_specs()
    template = _healthy_row()
    rows = tuple(_row_for_spec(spec, template, base_passed=False) for spec in specs)
    robustness = compute_family_robustness(rows, load_frozen_screen_policy())
    summary = build_family_summary(rows, robustness)
    assert [row["tested_hypothesis_count"] for row in summary] == [7, 42, 42]
    diagnostics = build_diagnostics_summary(specs, {}, rows)
    assert diagnostics["status"] == "report_only_never_gate_rank_filter_or_rescue"
    assert len(diagnostics["per_hypothesis"]) == 91


def test_artifact_writer_produces_exact_files_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    specs = load_frozen_clock_specs()
    template = _healthy_row()
    rows = tuple(_row_for_spec(spec, template, base_passed=False) for spec in specs)
    robustness = compute_family_robustness(rows, load_frozen_screen_policy())
    output = tmp_path / "batch4"
    write_batch4_artifacts(
        scorecard=rows,
        family_summary=build_family_summary(rows, robustness),
        family_robustness=robustness,
        selection_summary=select_representative(
            robustness, load_frozen_screen_policy()
        ),
        diagnostics_summary={"status": "report_only"},
        metadata={"validation_accessed": False, "holdout_accessed": False},
        output_dir=output,
    )
    assert {path.name for path in output.iterdir()} == {
        "scorecard.csv",
        "family_summary.csv",
        "family_robustness.csv",
        "selection_summary.json",
        "diagnostics_summary.json",
        "metadata.json",
        "artifact_hashes.json",
    }
    assert len(pd.read_csv(output / "scorecard.csv")) == 91
    hashes = json.loads((output / "artifact_hashes.json").read_text())
    assert set(hashes) == {
        "scorecard.csv",
        "family_summary.csv",
        "family_robustness.csv",
        "selection_summary.json",
        "diagnostics_summary.json",
        "metadata.json",
    }
    with pytest.raises(Batch4ScreenError, match="refusing to overwrite"):
        write_batch4_artifacts(
            scorecard=rows,
            family_summary=(),
            family_robustness=(),
            selection_summary={},
            diagnostics_summary={},
            metadata={},
            output_dir=output,
        )
