from __future__ import annotations

from collections.abc import Mapping

import pytest

from ftmoquant.research.g1.family import FamilyMetadata, StrategyFamily
from ftmoquant.research.g1.guards import WalkForwardWindow
from ftmoquant.research.g1.parameter_space import (
    FloatParameter,
    ParameterMapping,
    ParameterSpace,
    ParameterValue,
)
from ftmoquant.research.g1.search import (
    InvalidTrialError,
    SearchConfig,
    SearchMode,
    run_search,
)
from ftmoquant.research.g1.trials import FoldMetrics, TrialRegistryError, TrialStatus
from tests.research.g1.helpers import IntegerFamily, context


def _metrics(fold_id: str, expectancy: float) -> FoldMetrics:
    return FoldMetrics(
        fold_id=fold_id,
        trade_count=10,
        net_expectancy=expectancy,
        net_daily_mean=expectancy / 10,
        sharpe=expectancy,
        drawdown=0.1,
        turnover=1.0,
        cost_stressed_expectancy=expectancy - 0.01,
        execution_perturbed_expectancy=expectancy - 0.005,
    )


def test_exact_search_records_losing_invalid_and_failed_cells() -> None:
    family = IntegerFamily(invalidate_two=True)

    def evaluator(
        _: StrategyFamily[object, object],
        parameters: Mapping[str, ParameterValue],
        window: WalkForwardWindow,
    ) -> FoldMetrics:
        value = int(parameters["lookback"] or 0)
        if value == 3:
            raise InvalidTrialError("synthetic invalid evaluator cell")
        if value == 4:
            raise RuntimeError("synthetic engine failure")
        return _metrics(window.fold_id, -0.2)

    result = run_search(
        family=family, context=context(), config=SearchConfig(), evaluator=evaluator
    )

    assert result.exact_trial_count == 4
    assert result.registry.total_unique_configurations == 4
    assert result.registry.valid_configurations == 1
    assert result.registry.invalid_configurations == 2
    assert result.registry.failed_configurations == 1
    statuses = tuple(record.status for record in result.registry.records)
    assert statuses == (
        TrialStatus.COMPLETE,
        TrialStatus.INVALID,
        TrialStatus.INVALID,
        TrialStatus.FAILED,
    )
    losing = result.registry.records[0]
    assert all(fold.net_expectancy < 0.0 for fold in losing.folds)
    assert all(
        record.failure_reason for record in result.registry.records if not record.folds
    )
    assert result.registry.records[2].failure_fold_id == "fold_1"
    assert result.registry.records[3].failure_fold_id == "fold_1"
    with pytest.raises(TrialRegistryError, match="duplicate"):
        result.registry.add(result.registry.records[0])


class ContinuousFamily(StrategyFamily[object, object]):
    @property
    def metadata(self) -> FamilyMetadata:
        return FamilyMetadata(
            "synthetic_continuous", "1.0.0", "Synthetic Optuna test.", ("1h",)
        )

    @property
    def parameter_space(self) -> ParameterSpace:
        return ParameterSpace((FloatParameter("threshold", 0.0, 1.0),))

    def build_signals(self, data: object, parameters: ParameterMapping) -> object:
        return data


def test_seeded_optuna_wrapper_is_deterministic_unique_and_exact_count() -> None:
    family = ContinuousFamily()

    def evaluator(
        _: StrategyFamily[object, object],
        parameters: Mapping[str, ParameterValue],
        window: WalkForwardWindow,
    ) -> FoldMetrics:
        value = float(parameters["threshold"] or 0.0)
        return _metrics(window.fold_id, value - 0.2)

    config = SearchConfig(mode=SearchMode.SEEDED_OPTUNA, seed=1947, optuna_trials=12)
    first = run_search(
        family=family, context=context(), config=config, evaluator=evaluator
    )
    second = run_search(
        family=family, context=context(), config=config, evaluator=evaluator
    )

    assert first.registry.records == second.registry.records
    assert first.config.semantic_sha256 == second.config.semantic_sha256
    assert first.exact_trial_count == second.exact_trial_count == 12
    assert first.registry.total_unique_configurations == 12
    assert all(
        record.status is TrialStatus.COMPLETE for record in first.registry.records
    )
