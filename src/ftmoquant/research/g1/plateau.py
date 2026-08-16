"""Family-owned adjacency with generic evaluated-neighbour plateau accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ftmoquant.research.g1.family import StrategyFamily
from ftmoquant.research.g1.parameter_space import canonical_parameter_json
from ftmoquant.research.g1.trials import TrialRecord, TrialRegistry, TrialStatus


class PlateauError(ValueError):
    """Raised when a family's neighbour semantics are invalid or ambiguous."""


@dataclass(frozen=True, slots=True)
class PlateauAssessment:
    trial_id: str
    declared_neighbours: int
    evaluated_neighbours: int
    acceptable_neighbours: int
    acceptable_fraction: float | None
    missing_neighbours: int


def assess_plateau(
    family: StrategyFamily[Any, Any],
    trial: TrialRecord,
    registry: TrialRegistry,
    acceptable_trial_ids: frozenset[str],
) -> PlateauAssessment:
    """Match family-declared neighbours to complete unique registry cells."""

    if trial.status is not TrialStatus.COMPLETE:
        raise PlateauError("plateau assessment requires a complete trial")
    neighbours = family.neighbours(trial.parameter_mapping)
    canonical_neighbours: list[str] = []
    for parameters in neighbours:
        family.validate_parameters(parameters)
        canonical = canonical_parameter_json(parameters)
        if canonical == trial.canonical_parameters:
            raise PlateauError("a trial cannot be its own neighbour")
        canonical_neighbours.append(canonical)
    if len(set(canonical_neighbours)) != len(canonical_neighbours):
        raise PlateauError("family neighbour declarations must be unique")
    records_by_parameters = {
        record.canonical_parameters: record
        for record in registry.records
        if record.family_id == trial.family_id
        and record.family_version == trial.family_version
        and record.status is TrialStatus.COMPLETE
    }
    evaluated = tuple(
        records_by_parameters[canonical]
        for canonical in canonical_neighbours
        if canonical in records_by_parameters
    )
    acceptable = sum(record.trial_id in acceptable_trial_ids for record in evaluated)
    fraction = acceptable / len(evaluated) if evaluated else None
    return PlateauAssessment(
        trial_id=trial.trial_id,
        declared_neighbours=len(canonical_neighbours),
        evaluated_neighbours=len(evaluated),
        acceptable_neighbours=acceptable,
        acceptable_fraction=fraction,
        missing_neighbours=len(canonical_neighbours) - len(evaluated),
    )
