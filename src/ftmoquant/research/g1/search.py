"""Exact-grid and deterministic seeded Optuna search behind one family contract."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from ftmoquant.research.g1.family import StrategyFamily
from ftmoquant.research.g1.guards import DevelopmentSearchContext, WalkForwardWindow
from ftmoquant.research.g1.parameter_space import (
    ParameterValue,
    canonical_parameter_json,
    suggest_parameters,
)
from ftmoquant.research.g1.trials import (
    FoldMetrics,
    TrialRecord,
    TrialRegistry,
    TrialStatus,
    make_trial_record,
    summarize_trial,
)


class SearchError(RuntimeError):
    """Raised when frozen search semantics cannot be honored."""


class InvalidTrialError(ValueError):
    """Evaluator signal that a configuration is invalid, not an engine failure."""


class SearchMode(StrEnum):
    EXACT_GRID = "exact_grid"
    SEEDED_OPTUNA = "seeded_optuna"


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """Frozen bounded-search settings; Optuna is intentionally not the default."""

    mode: SearchMode = SearchMode.EXACT_GRID
    seed: int = 0
    optuna_trials: int | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise SearchError("search seed must be a non-negative integer")
        if self.mode is SearchMode.EXACT_GRID and self.optuna_trials is not None:
            raise SearchError("exact grid does not accept an Optuna trial limit")
        if self.mode is SearchMode.SEEDED_OPTUNA and (
            self.optuna_trials is None
            or isinstance(self.optuna_trials, bool)
            or self.optuna_trials <= 0
        ):
            raise SearchError("seeded Optuna requires a positive frozen trial count")

    @property
    def semantic_sha256(self) -> str:
        payload = {
            "mode": self.mode.value,
            "seed": self.seed,
            "optuna_trials": self.optuna_trials,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


TrialEvaluator = Callable[
    [StrategyFamily[Any, Any], Mapping[str, ParameterValue], WalkForwardWindow],
    FoldMetrics,
]


@dataclass(frozen=True, slots=True)
class SearchResult:
    family_id: str
    family_version: str
    config: SearchConfig
    registry: TrialRegistry

    @property
    def exact_trial_count(self) -> int:
        return self.registry.total_attempted_configurations


def run_search(
    *,
    family: StrategyFamily[Any, Any],
    context: DevelopmentSearchContext,
    config: SearchConfig,
    evaluator: TrialEvaluator,
) -> SearchResult:
    """Run only synthetic/caller-supplied DEVELOPMENT evaluation callbacks."""

    registry = TrialRegistry()
    if config.mode is SearchMode.EXACT_GRID:
        for parameters in family.enumerate_parameters():
            registry.add(
                _evaluate_configuration(family, context, evaluator, parameters)
            )
    else:
        _run_optuna(family, context, config, evaluator, registry)
    return SearchResult(
        family_id=family.metadata.family_id,
        family_version=family.metadata.version,
        config=config,
        registry=registry,
    )


def _run_optuna(
    family: StrategyFamily[Any, Any],
    context: DevelopmentSearchContext,
    config: SearchConfig,
    evaluator: TrialEvaluator,
    registry: TrialRegistry,
) -> None:
    sampler = TPESampler(seed=config.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    assert config.optuna_trials is not None
    for _ in range(config.optuna_trials):
        trial = study.ask()
        parameters = suggest_parameters(trial, family.parameter_space)
        canonical = canonical_parameter_json(parameters)
        if any(record.canonical_parameters == canonical for record in registry.records):
            study.tell(trial, state=TrialState.FAIL)
            raise SearchError(
                "Optuna suggested a duplicate configuration; use a larger/continuous "
                "space or exact finite-grid enumeration"
            )
        record = _evaluate_configuration(family, context, evaluator, parameters)
        registry.add(record)
        if record.status is TrialStatus.COMPLETE:
            study.tell(trial, summarize_trial(record).pooled_expectancy)
        else:
            study.tell(trial, state=TrialState.FAIL)


def _evaluate_configuration(
    family: StrategyFamily[Any, Any],
    context: DevelopmentSearchContext,
    evaluator: TrialEvaluator,
    parameters: Mapping[str, ParameterValue],
) -> TrialRecord:
    try:
        family.validate_parameters(parameters)
    except (ValueError, TypeError) as error:
        return make_trial_record(
            family_id=family.metadata.family_id,
            family_version=family.metadata.version,
            parameters=parameters,
            status=TrialStatus.INVALID,
            failure_reason=_failure_reason(error),
        )
    folds: list[FoldMetrics] = []
    try:
        for window in context.windows:
            window.validate(context.access_policy)
            metrics = evaluator(family, parameters, window)
            if metrics.fold_id != window.fold_id:
                raise SearchError("evaluator fold_id does not match requested fold")
            folds.append(metrics)
    except InvalidTrialError as error:
        return make_trial_record(
            family_id=family.metadata.family_id,
            family_version=family.metadata.version,
            parameters=parameters,
            status=TrialStatus.INVALID,
            failure_reason=_failure_reason(error),
            failure_fold_id=window.fold_id,
        )
    except Exception as error:
        return make_trial_record(
            family_id=family.metadata.family_id,
            family_version=family.metadata.version,
            parameters=parameters,
            status=TrialStatus.FAILED,
            failure_reason=_failure_reason(error),
            failure_fold_id=window.fold_id,
        )
    return make_trial_record(
        family_id=family.metadata.family_id,
        family_version=family.metadata.version,
        parameters=parameters,
        status=TrialStatus.COMPLETE,
        folds=folds,
    )


def _failure_reason(error: BaseException) -> str:
    message = str(error).strip()
    return type(error).__name__ if not message else f"{type(error).__name__}: {message}"
