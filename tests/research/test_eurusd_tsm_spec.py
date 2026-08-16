from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest

from ftmoquant.research.eurusd_tsm_spec import (
    EURUSD_TSM_SEMANTIC_SHA256,
    load_eurusd_tsm_spec,
    semantic_sha256_for_document,
)
from ftmoquant.research.g1.guards import (
    DevelopmentAccessPolicy,
    ResearchAccessError,
    ResearchPartition,
)
from ftmoquant.research.g1.parameter_space import (
    canonical_parameter_json,
    deterministic_trial_id,
)
from ftmoquant.research.g1.selector import select_candidate
from ftmoquant.research.g1.trials import (
    Attribution,
    FoldMetrics,
    TrialRegistry,
    TrialStatus,
    make_trial_record,
)
from ftmoquant.research.stage_g import (
    DEVELOPMENT_END_EXCLUSIVE,
    DEVELOPMENT_START,
    frozen_development_folds,
)
from ftmoquant.strategies.eurusd_tsm import EurusdTsmFamily


def test_preregistered_conditional_grid_is_exact_unique_and_deterministic() -> None:
    spec = load_eurusd_tsm_spec()
    family = EurusdTsmFamily(spec)
    configurations = family.enumerate_parameters()

    assert spec.semantic_sha256 == EURUSD_TSM_SEMANTIC_SHA256
    assert spec.expected_unique_trial_count == len(configurations) == 90
    assert sum(item["timeframe"] == "H1" for item in configurations) == 45
    assert sum(item["timeframe"] == "H4" for item in configurations) == 45
    assert {
        item["lookback_bars"] for item in configurations if item["timeframe"] == "H1"
    } == {24, 72, 120, 240, 480}
    assert {
        item["lookback_bars"] for item in configurations if item["timeframe"] == "H4"
    } == {6, 18, 30, 60, 120}
    canonical = tuple(canonical_parameter_json(item) for item in configurations)
    trial_ids = tuple(
        deterministic_trial_id(spec.family_id, spec.version, item)
        for item in configurations
    )
    assert len(set(canonical)) == len(set(trial_ids)) == 90
    assert trial_ids == tuple(
        deterministic_trial_id(spec.family_id, spec.version, item)
        for item in family.enumerate_parameters()
    )


def test_semantic_sha_covers_grid_and_selector_order() -> None:
    document = load_eurusd_tsm_spec().canonical_document
    assert semantic_sha256_for_document(document) == EURUSD_TSM_SEMANTIC_SHA256
    assert semantic_sha256_for_document(deepcopy(document)) == (
        EURUSD_TSM_SEMANTIC_SHA256
    )

    changed_grid = deepcopy(document)
    changed_grid["parameter_grid"]["timeframes"][0]["lookback_bars"][0] = 23
    changed_selector = deepcopy(document)
    changed_selector["selector"]["ranking_order"][0:2] = reversed(
        changed_selector["selector"]["ranking_order"][0:2]
    )

    assert semantic_sha256_for_document(changed_grid) != EURUSD_TSM_SEMANTIC_SHA256
    assert semantic_sha256_for_document(changed_selector) != (
        EURUSD_TSM_SEMANTIC_SHA256
    )


def test_pre_data_normalization_amendment_is_frozen_without_alpha_drift() -> None:
    document = load_eurusd_tsm_spec().canonical_document
    risk = document["risk_normalization"]
    amendment = document["preregistration_amendments"][0]

    assert EURUSD_TSM_SEMANTIC_SHA256 != (
        "6f2cdf187e40e9bc9c065c7e7befa16f89d1fe380c8626ab853474658466f116"
    )
    assert risk["type"] == "causal_pre_execution_underlying_vol_target"
    assert risk["observation_rule"] == (
        "daily_return_endpoint_information_time_strictly_before_decision"
    )
    assert amendment["observed_development_trial_results"] == 0
    assert amendment["observed_development_fold_results"] == 0
    assert amendment["historical_development_strategy_returns_accessed"] is False


def test_pre_data_evaluator_and_sample_count_amendment_is_frozen() -> None:
    document = load_eurusd_tsm_spec().canonical_document
    sample = document["sample_count"]
    evaluator = document["production_evaluator"]
    amendment = document["preregistration_amendments"][1]

    assert sample["fold_metrics_trade_count"] == "executed_raw_alpha_transitions"
    assert sample["risk_only_rebalances_count"] is False
    assert sample["reversal_count"] == "one_transition"
    assert sample["unexecuted_target_change_count"] is False
    assert sample["turnover_includes_all_scaled_executed_quantity_changes"] is True
    assert sample["costs_and_pnl_include_all_scaled_rebalances"] is True
    assert evaluator["module"] == "ftmoquant.research.eurusd_tsm_development"
    assert amendment["superseded_semantic_sha256"] == (
        "0a5411f003dba15d88b8f2ad7368ad9d25cc16a5474108879eb012658c670a98"
    )
    assert amendment["observed_development_trial_results"] == 0
    assert amendment["observed_development_fold_results"] == 0
    assert amendment["validation_accessed"] is False
    assert amendment["final_holdout_accessed"] is False


def test_existing_three_development_folds_are_frozen_exactly() -> None:
    folds = frozen_development_folds()
    declared = load_eurusd_tsm_spec().canonical_document["development_folds"]

    assert declared["version"] == folds.version
    assert declared["semantic_sha256"] == folds.semantic_sha256
    assert [item["fold_id"] for item in declared["folds"]] == [
        "dev_fold_1",
        "dev_fold_2",
        "dev_fold_3",
    ]
    assert declared["folds"][0]["evaluate_start_utc"] == ("2020-04-11T00:00:00Z")
    assert declared["folds"][-1]["evaluate_end_exclusive_utc"] == (
        "2023-04-11T00:00:00Z"
    )


def test_tsm_search_boundary_cannot_admit_validation_or_final_holdout() -> None:
    policy = DevelopmentAccessPolicy(DEVELOPMENT_START, DEVELOPMENT_END_EXCLUSIVE)

    with pytest.raises(ResearchAccessError, match="validation is sealed"):
        policy.require_interval(
            DEVELOPMENT_END_EXCLUSIVE,
            DEVELOPMENT_END_EXCLUSIVE + timedelta(days=1),
            partition=ResearchPartition.VALIDATION,
        )
    with pytest.raises(ResearchAccessError, match="final_holdout is sealed"):
        policy.require_interval(
            datetime(2024, 8, 21, tzinfo=UTC),
            datetime(2024, 8, 22, tzinfo=UTC),
            partition=ResearchPartition.FINAL_HOLDOUT,
        )


def _eligible_record(
    family: EurusdTsmFamily,
    parameters: dict[str, str | int | float | bool | None],
    expectancy: float,
    drawdown: float,
    fold_count: int = 3,
):
    folds = tuple(
        FoldMetrics(
            fold_id=f"dev_fold_{index}",
            trade_count=50,
            net_expectancy=expectancy if index < 3 else -expectancy / 4,
            sharpe=0.5,
            drawdown=drawdown,
            turnover=1.0,
            cost_stressed_expectancy=expectancy / 2,
            yearly_attribution=(Attribution(str(2019 + index), expectancy / 3),),
        )
        for index in range(1, fold_count + 1)
    )
    return make_trial_record(
        family_id=family.metadata.family_id,
        family_version=family.metadata.version,
        parameters=parameters,
        status=TrialStatus.COMPLETE,
        folds=folds,
    )


def test_preregistered_selector_is_deterministic_without_drawdown_gate() -> None:
    spec = load_eurusd_tsm_spec()
    family = EurusdTsmFamily(spec)
    first_parameters = {
        "timeframe": "H1",
        "lookback_bars": 24,
        "deadband": 0.0,
        "refresh_interval_bars": 1,
    }
    second_parameters = {**first_parameters, "refresh_interval_bars": 4}
    incomplete_parameters = {**first_parameters, "refresh_interval_bars": 24}
    registry = TrialRegistry()
    first_record = _eligible_record(family, first_parameters, 0.10, 9.0)
    second_record = _eligible_record(family, second_parameters, 0.05, 0.1)
    incomplete_record = _eligible_record(
        family, incomplete_parameters, 1.0, 0.0, fold_count=2
    )
    registry.add(first_record)
    registry.add(second_record)

    first = select_candidate(
        family=family, registry=registry, policy=spec.selection_policy
    )
    second = select_candidate(
        family=family, registry=registry, policy=spec.selection_policy
    )

    assert first == second
    assert first.selected_trial_id == second_record.trial_id
    first_status = next(
        item for item in first.statuses if item.trial_id == first_record.trial_id
    )
    assert first_status.hard_gate_eligible is True

    incomplete_registry = TrialRegistry()
    incomplete_registry.add(incomplete_record)
    incomplete_decision = select_candidate(
        family=family, registry=incomplete_registry, policy=spec.selection_policy
    )
    incomplete_status = incomplete_decision.statuses[0]
    assert incomplete_decision.selected_trial_id is None
    assert incomplete_status.hard_gate_eligible is False
    assert incomplete_status.reasons == (
        "required DEVELOPMENT fold count is incomplete",
    )
    assert spec.selection_policy.max_drawdown is None
    assert spec.selection_policy.min_trade_count == 100
    assert spec.selection_policy.min_positive_folds == 2
    assert spec.selection_policy.required_fold_count == 3
