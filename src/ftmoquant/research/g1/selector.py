"""Mechanical gates, robustness filters, and lexicographic candidate selection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ftmoquant.research.g1.family import StrategyFamily
from ftmoquant.research.g1.plateau import PlateauAssessment, assess_plateau
from ftmoquant.research.g1.trials import (
    TrialRecord,
    TrialRegistry,
    TrialStatus,
    TrialSummary,
    summarize_trial,
)


class SelectionError(ValueError):
    """Raised when a frozen mechanical selection policy is invalid."""


@dataclass(frozen=True, slots=True)
class SelectionPolicy:
    min_trade_count: int
    min_positive_folds: int
    max_drawdown: float | None
    max_year_concentration: float | None
    max_execution_sensitivity: float | None
    min_evaluated_neighbours: int = 0
    min_acceptable_neighbour_fraction: float = 0.0
    required_fold_count: int | None = None

    def __post_init__(self) -> None:
        if self.min_trade_count < 0 or self.min_positive_folds <= 0:
            raise SelectionError("trade/fold minima are invalid")
        if self.max_drawdown is not None and (
            not math.isfinite(self.max_drawdown) or self.max_drawdown < 0.0
        ):
            raise SelectionError("max_drawdown must be finite and non-negative")
        for name, value in (
            ("max_year_concentration", self.max_year_concentration),
            ("max_execution_sensitivity", self.max_execution_sensitivity),
        ):
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise SelectionError(f"{name} must be finite and non-negative")
        if self.min_evaluated_neighbours < 0 or not (
            0.0 <= self.min_acceptable_neighbour_fraction <= 1.0
        ):
            raise SelectionError("plateau requirements are invalid")
        if self.required_fold_count is not None and self.required_fold_count <= 0:
            raise SelectionError("required_fold_count must be positive when set")


@dataclass(frozen=True, slots=True)
class TrialSelectionStatus:
    trial_id: str
    hard_gate_eligible: bool
    robustness_eligible: bool
    reasons: tuple[str, ...]
    plateau: PlateauAssessment | None


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    selected_trial_id: str | None
    statuses: tuple[TrialSelectionStatus, ...]
    configurations_surviving_eligibility_gates: int
    configurations_considered_by_final_selector: int
    ranking_protocol: tuple[str, ...]


RANKING_PROTOCOL = (
    "descending positive DEVELOPMENT fold fraction",
    "descending acceptable evaluated-neighbour fraction",
    "ascending maximum absolute yearly concentration (missing last)",
    "ascending execution perturbation sensitivity (missing last)",
    "descending worst-fold net expectancy",
    "descending cost-stressed pooled expectancy",
    "descending pooled net expectancy",
    "ascending maximum drawdown",
    "ascending deterministic trial_id tie-breaker",
)


def select_candidate(
    *,
    family: StrategyFamily[Any, Any],
    registry: TrialRegistry,
    policy: SelectionPolicy,
) -> SelectionDecision:
    """Select without a weighted composite score or validation information."""

    hard_survivors: dict[str, TrialSummary] = {}
    reasons_by_trial: dict[str, list[str]] = {}
    for record in registry.records:
        reasons: list[str] = []
        if record.status is not TrialStatus.COMPLETE:
            reasons.append(f"status={record.status.value}")
        else:
            summary = summarize_trial(record)
            if summary.pooled_expectancy <= 0.0:
                reasons.append("nonpositive pooled expectancy")
            if summary.cost_stressed_expectancy <= 0.0:
                reasons.append("nonpositive cost-stressed expectancy")
            if summary.trade_count < policy.min_trade_count:
                reasons.append("inadequate trade count")
            if summary.positive_fold_count < policy.min_positive_folds:
                reasons.append("insufficient positive folds")
            if (
                policy.required_fold_count is not None
                and summary.fold_count != policy.required_fold_count
            ):
                reasons.append("required DEVELOPMENT fold count is incomplete")
            if (
                policy.max_drawdown is not None
                and summary.max_drawdown > policy.max_drawdown
            ):
                reasons.append("drawdown exceeds bound")
            if not reasons:
                hard_survivors[record.trial_id] = summary
        reasons_by_trial[record.trial_id] = reasons

    acceptable_ids = frozenset(hard_survivors)
    plateaus: dict[str, PlateauAssessment] = {}
    final_survivors: list[tuple[TrialRecord, TrialSummary, PlateauAssessment]] = []
    for record in registry.records:
        current_summary = hard_survivors.get(record.trial_id)
        if current_summary is None:
            continue
        plateau = assess_plateau(family, record, registry, acceptable_ids)
        plateaus[record.trial_id] = plateau
        reasons = reasons_by_trial[record.trial_id]
        if (
            policy.max_year_concentration is not None
            and current_summary.maximum_year_concentration is not None
            and current_summary.maximum_year_concentration
            > policy.max_year_concentration
        ):
            reasons.append("year/regime concentration exceeds bound")
        if (
            policy.max_execution_sensitivity is not None
            and current_summary.max_execution_sensitivity is not None
            and current_summary.max_execution_sensitivity
            > policy.max_execution_sensitivity
        ):
            reasons.append("execution perturbation sensitivity exceeds bound")
        if plateau.evaluated_neighbours < policy.min_evaluated_neighbours:
            reasons.append("too few evaluated parameter neighbours")
        fraction = plateau.acceptable_fraction
        if fraction is not None and fraction < policy.min_acceptable_neighbour_fraction:
            reasons.append("parameter plateau is too narrow")
        if policy.min_acceptable_neighbour_fraction > 0.0 and fraction is None:
            reasons.append("parameter plateau is unavailable")
        if not reasons:
            final_survivors.append((record, current_summary, plateau))

    final_survivors.sort(key=_ranking_key)
    selected = final_survivors[0][0].trial_id if final_survivors else None
    statuses = tuple(
        TrialSelectionStatus(
            trial_id=record.trial_id,
            hard_gate_eligible=record.trial_id in hard_survivors,
            robustness_eligible=any(
                survivor[0].trial_id == record.trial_id for survivor in final_survivors
            ),
            reasons=tuple(reasons_by_trial[record.trial_id]),
            plateau=plateaus.get(record.trial_id),
        )
        for record in registry.records
    )
    return SelectionDecision(
        selected_trial_id=selected,
        statuses=statuses,
        configurations_surviving_eligibility_gates=len(hard_survivors),
        configurations_considered_by_final_selector=len(final_survivors),
        ranking_protocol=RANKING_PROTOCOL,
    )


def _ranking_key(
    item: tuple[TrialRecord, TrialSummary, PlateauAssessment],
) -> tuple[float | str, ...]:
    record, summary, plateau = item
    positive_fraction = summary.positive_fold_count / summary.fold_count
    plateau_fraction = plateau.acceptable_fraction or 0.0
    concentration = (
        summary.maximum_year_concentration
        if summary.maximum_year_concentration is not None
        else math.inf
    )
    sensitivity = (
        summary.max_execution_sensitivity
        if summary.max_execution_sensitivity is not None
        else math.inf
    )
    return (
        -positive_fraction,
        -plateau_fraction,
        concentration,
        sensitivity,
        -summary.worst_fold_expectancy,
        -summary.cost_stressed_expectancy,
        -summary.pooled_expectancy,
        summary.max_drawdown,
        record.trial_id,
    )
