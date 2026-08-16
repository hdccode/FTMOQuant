from __future__ import annotations

from collections.abc import Mapping

from ftmoquant.research.g1.artifacts import (
    deterministic_artifact_bytes,
    write_deterministic_artifact,
)
from ftmoquant.research.g1.family import StrategyFamily
from ftmoquant.research.g1.guards import WalkForwardWindow
from ftmoquant.research.g1.parameter_space import ParameterValue
from ftmoquant.research.g1.search import SearchConfig, run_search
from ftmoquant.research.g1.selector import SelectionPolicy, select_candidate
from ftmoquant.research.g1.trials import Attribution, FoldMetrics
from tests.research.g1.helpers import IntegerFamily, context

EXPECTANCIES = {1: 0.10, 2: 0.08, 3: -0.10, 4: 1.00, 5: -0.10}


def _evaluator(
    _: StrategyFamily[object, object],
    parameters: Mapping[str, ParameterValue],
    window: WalkForwardWindow,
) -> FoldMetrics:
    value = EXPECTANCIES[int(parameters["lookback"] or 0)]
    return FoldMetrics(
        fold_id=window.fold_id,
        trade_count=40,
        net_expectancy=value,
        net_daily_mean=value / 20,
        sharpe=value * 2,
        drawdown=0.10,
        turnover=2.0,
        cost_stressed_expectancy=value - 0.02,
        yearly_attribution=(
            Attribution("2020", value * 0.55),
            Attribution("2021", value * 0.45),
        ),
        regime_attribution=(Attribution("synthetic", value),),
        execution_perturbed_expectancy=value - 0.01,
    )


def test_plateau_first_mechanical_selection_prefers_broad_region() -> None:
    family = IntegerFamily(1, 5)
    result = run_search(
        family=family, context=context(), config=SearchConfig(), evaluator=_evaluator
    )
    policy = SelectionPolicy(
        min_trade_count=50,
        min_positive_folds=2,
        max_drawdown=0.20,
        max_year_concentration=0.70,
        max_execution_sensitivity=0.05,
        min_evaluated_neighbours=1,
    )

    decision = select_candidate(family=family, registry=result.registry, policy=policy)

    chosen = result.registry.get(decision.selected_trial_id or "")
    assert chosen.parameter_mapping == {"lookback": 1}
    isolated = next(
        status
        for status in decision.statuses
        if result.registry.get(status.trial_id).parameter_mapping["lookback"] == 4
    )
    assert isolated.plateau is not None
    assert isolated.plateau.acceptable_fraction == 0.0
    assert decision.configurations_surviving_eligibility_gates == 3
    assert decision.configurations_considered_by_final_selector == 3


def test_tie_break_and_artifact_bytes_are_deterministic(tmp_path) -> None:  # type: ignore[no-untyped-def]
    class NoNeighboursFamily(IntegerFamily):
        def neighbours(self, parameters):  # type: ignore[no-untyped-def]
            return ()

    family = NoNeighboursFamily(1, 2)

    def equal_evaluator(
        _: StrategyFamily[object, object],
        parameters: Mapping[str, ParameterValue],
        window: WalkForwardWindow,
    ) -> FoldMetrics:
        del parameters
        return FoldMetrics(window.fold_id, 30, 0.1, 0.5, 0.1, 1.0, 0.08)

    result = run_search(
        family=family,
        context=context(),
        config=SearchConfig(seed=7),
        evaluator=equal_evaluator,
    )
    policy = SelectionPolicy(40, 2, 0.2, None, None)
    first = select_candidate(family=family, registry=result.registry, policy=policy)
    second = select_candidate(family=family, registry=result.registry, policy=policy)

    assert first == second
    assert first.selected_trial_id == min(
        record.trial_id for record in result.registry.records
    )
    content = deterministic_artifact_bytes(result, first)
    assert content == deterministic_artifact_bytes(result, second)
    assert b'"result":"positive"' in content
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    assert write_deterministic_artifact(left, result, first) == (
        write_deterministic_artifact(right, result, second)
    )
    assert left.read_bytes() == right.read_bytes() == content
