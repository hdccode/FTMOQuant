"""Complete deterministic registry for winners, losers, invalids, and failures."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum

from ftmoquant.research.g1.normalization import G1_ANNUAL_VOLATILITY_TARGET
from ftmoquant.research.g1.parameter_space import (
    ParameterMapping,
    ParameterValue,
    canonical_parameter_json,
    deterministic_trial_id,
)


class TrialRegistryError(ValueError):
    """Raised when trial evidence is incomplete, ambiguous, or duplicated."""


class TrialStatus(StrEnum):
    COMPLETE = "complete"
    INVALID = "invalid"
    FAILED = "failed"


class FoldResult(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    ZERO = "zero"


@dataclass(frozen=True, slots=True)
class Attribution:
    label: str
    net_expectancy: float

    def __post_init__(self) -> None:
        if not self.label.strip() or not math.isfinite(self.net_expectancy):
            raise TrialRegistryError("attribution must have a label and finite value")


@dataclass(frozen=True, slots=True)
class FoldMetrics:
    """Metrics one authoritative evaluator reports for one DEVELOPMENT fold."""

    fold_id: str
    trade_count: int
    net_expectancy: float
    sharpe: float
    drawdown: float
    turnover: float
    cost_stressed_expectancy: float
    net_daily_mean: float | None = None
    yearly_attribution: tuple[Attribution, ...] = ()
    regime_attribution: tuple[Attribution, ...] = ()
    execution_perturbed_expectancy: float | None = None
    annualized_volatility_target: float = G1_ANNUAL_VOLATILITY_TARGET
    result: FoldResult = field(init=False)

    def __post_init__(self) -> None:
        if not self.fold_id.strip():
            raise TrialRegistryError("fold_id must not be empty")
        if isinstance(self.trade_count, bool) or self.trade_count < 0:
            raise TrialRegistryError("trade_count must be a non-negative integer")
        finite = (
            self.net_expectancy,
            self.sharpe,
            self.drawdown,
            self.turnover,
            self.cost_stressed_expectancy,
        )
        if not all(math.isfinite(value) for value in finite):
            raise TrialRegistryError("fold metrics must be finite")
        if self.drawdown < 0.0 or self.turnover < 0.0:
            raise TrialRegistryError("drawdown and turnover must be non-negative")
        for optional in (self.net_daily_mean, self.execution_perturbed_expectancy):
            if optional is not None and not math.isfinite(optional):
                raise TrialRegistryError("optional fold metrics must be finite")
        if self.annualized_volatility_target != G1_ANNUAL_VOLATILITY_TARGET:
            raise TrialRegistryError(
                "G1 fold metrics must use 1% annualized volatility normalization"
            )
        _unique_attribution(self.yearly_attribution, "year")
        _unique_attribution(self.regime_attribution, "regime")
        result = (
            FoldResult.POSITIVE
            if self.net_expectancy > 0.0
            else FoldResult.NEGATIVE
            if self.net_expectancy < 0.0
            else FoldResult.ZERO
        )
        object.__setattr__(self, "result", result)

    @property
    def positive(self) -> bool:
        return self.net_expectancy > 0.0


@dataclass(frozen=True, slots=True)
class TrialRecord:
    family_id: str
    family_version: str
    parameters: tuple[tuple[str, ParameterValue], ...]
    canonical_parameters: str
    trial_id: str
    status: TrialStatus
    folds: tuple[FoldMetrics, ...] = ()
    failure_reason: str | None = None
    failure_fold_id: str | None = None

    def __post_init__(self) -> None:
        mapping = dict(self.parameters)
        if len(mapping) != len(self.parameters):
            raise TrialRegistryError("trial parameter names must be unique")
        expected_json = canonical_parameter_json(mapping)
        expected_id = deterministic_trial_id(
            self.family_id, self.family_version, mapping
        )
        if self.canonical_parameters != expected_json or self.trial_id != expected_id:
            raise TrialRegistryError(
                "trial canonical identity does not match parameters"
            )
        if self.status is TrialStatus.COMPLETE:
            if (
                not self.folds
                or self.failure_reason is not None
                or self.failure_fold_id is not None
            ):
                raise TrialRegistryError("complete trials require folds and no failure")
            ids = tuple(fold.fold_id for fold in self.folds)
            if len(set(ids)) != len(ids):
                raise TrialRegistryError("trial fold IDs must be unique")
        elif not self.failure_reason or self.folds:
            raise TrialRegistryError(
                "invalid/failed trials require a reason and no partial fold evidence"
            )
        if self.failure_fold_id is not None and not self.failure_fold_id.strip():
            raise TrialRegistryError("failure_fold_id must be non-empty when present")

    @property
    def parameter_mapping(self) -> dict[str, ParameterValue]:
        return dict(self.parameters)


@dataclass(frozen=True, slots=True)
class TrialSummary:
    trial_id: str
    trade_count: int
    pooled_expectancy: float
    cost_stressed_expectancy: float
    positive_fold_count: int
    fold_count: int
    worst_fold_expectancy: float
    max_drawdown: float
    mean_turnover: float
    max_execution_sensitivity: float | None
    maximum_year_concentration: float | None


class TrialRegistry:
    """Insertion-ordered registry with fail-closed duplicate prevention."""

    def __init__(self) -> None:
        self._records: list[TrialRecord] = []
        self._trial_ids: set[str] = set()
        self._canonical_configurations: set[tuple[str, str, str]] = set()

    def add(self, record: TrialRecord) -> None:
        configuration = (
            record.family_id,
            record.family_version,
            record.canonical_parameters,
        )
        if record.trial_id in self._trial_ids or configuration in (
            self._canonical_configurations
        ):
            raise TrialRegistryError("duplicate trial configuration is forbidden")
        self._records.append(record)
        self._trial_ids.add(record.trial_id)
        self._canonical_configurations.add(configuration)

    @property
    def records(self) -> tuple[TrialRecord, ...]:
        return tuple(self._records)

    @property
    def total_attempted_configurations(self) -> int:
        return len(self._records)

    @property
    def total_unique_configurations(self) -> int:
        return len(self._canonical_configurations)

    @property
    def valid_configurations(self) -> int:
        return sum(record.status is TrialStatus.COMPLETE for record in self._records)

    @property
    def invalid_configurations(self) -> int:
        return sum(record.status is TrialStatus.INVALID for record in self._records)

    @property
    def failed_configurations(self) -> int:
        return sum(record.status is TrialStatus.FAILED for record in self._records)

    def get(self, trial_id: str) -> TrialRecord:
        for record in self._records:
            if record.trial_id == trial_id:
                return record
        raise TrialRegistryError(f"unknown trial_id: {trial_id}")


def make_trial_record(
    *,
    family_id: str,
    family_version: str,
    parameters: ParameterMapping,
    status: TrialStatus,
    folds: Iterable[FoldMetrics] = (),
    failure_reason: str | None = None,
    failure_fold_id: str | None = None,
) -> TrialRecord:
    canonical = canonical_parameter_json(parameters)
    return TrialRecord(
        family_id=family_id,
        family_version=family_version,
        parameters=tuple(sorted(parameters.items())),
        canonical_parameters=canonical,
        trial_id=deterministic_trial_id(family_id, family_version, parameters),
        status=status,
        folds=tuple(folds),
        failure_reason=failure_reason,
        failure_fold_id=failure_fold_id,
    )


def summarize_trial(record: TrialRecord) -> TrialSummary:
    if record.status is not TrialStatus.COMPLETE:
        raise TrialRegistryError("only complete trials have metric summaries")
    total_trades = sum(fold.trade_count for fold in record.folds)
    weights = tuple(fold.trade_count for fold in record.folds)
    pooled = _weighted_mean(
        tuple(fold.net_expectancy for fold in record.folds), weights
    )
    stressed = _weighted_mean(
        tuple(fold.cost_stressed_expectancy for fold in record.folds), weights
    )
    sensitivities = tuple(
        abs(fold.net_expectancy - fold.execution_perturbed_expectancy)
        for fold in record.folds
        if fold.execution_perturbed_expectancy is not None
    )
    year_values: dict[str, float] = defaultdict(float)
    for fold in record.folds:
        for item in fold.yearly_attribution:
            year_values[item.label] += item.net_expectancy
    concentration: float | None = None
    denominator = sum(abs(value) for value in year_values.values())
    if year_values and denominator > 0.0:
        concentration = max(abs(value) for value in year_values.values()) / denominator
    return TrialSummary(
        trial_id=record.trial_id,
        trade_count=total_trades,
        pooled_expectancy=pooled,
        cost_stressed_expectancy=stressed,
        positive_fold_count=sum(fold.positive for fold in record.folds),
        fold_count=len(record.folds),
        worst_fold_expectancy=min(fold.net_expectancy for fold in record.folds),
        max_drawdown=max(fold.drawdown for fold in record.folds),
        mean_turnover=sum(fold.turnover for fold in record.folds) / len(record.folds),
        max_execution_sensitivity=max(sensitivities) if sensitivities else None,
        maximum_year_concentration=concentration,
    )


def _weighted_mean(values: tuple[float, ...], weights: tuple[int, ...]) -> float:
    total = sum(weights)
    if total == 0:
        return sum(values) / len(values)
    weighted = sum(
        value * weight for value, weight in zip(values, weights, strict=True)
    )
    return weighted / total


def _unique_attribution(values: tuple[Attribution, ...], label: str) -> None:
    labels = tuple(value.label for value in values)
    if len(set(labels)) != len(labels):
        raise TrialRegistryError(f"{label} attribution labels must be unique")
