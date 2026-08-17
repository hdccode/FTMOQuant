from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from datetime import time
from pathlib import Path
from typing import Any

import pytest

from ftmoquant.research.eurusd_session_range_expansion_development import (
    EurusdSessionRangeExpansionFamily,
)
from ftmoquant.research.eurusd_session_range_expansion_spec import (
    EURUSD_SESSION_RANGE_EXPANSION_SEMANTIC_SHA256,
    EurusdSessionRangeExpansionSpecError,
    load_eurusd_session_range_expansion_spec,
    semantic_sha256_for_document,
)
from ftmoquant.research.g1.parameter_space import (
    canonical_parameter_json,
    deterministic_trial_id,
)
from ftmoquant.research.stage_g import frozen_development_folds


def test_frozen_grid_is_exact_9_cells_unique_and_deterministic() -> None:
    spec = load_eurusd_session_range_expansion_spec()
    family = EurusdSessionRangeExpansionFamily(spec)
    configurations = family.enumerate_parameters()

    assert spec.semantic_sha256 == EURUSD_SESSION_RANGE_EXPANSION_SEMANTIC_SHA256
    assert spec.expected_unique_trial_count == len(configurations) == 9
    assert spec.breakout_window_end_grid == ("11:00", "12:00", "13:00")
    assert spec.scheduled_exit_grid == ("15:00", "16:00", "17:00")
    assert spec.breakout_window_end_times == (time(11), time(12), time(13))
    assert spec.scheduled_exit_times == (time(15), time(16), time(17))
    canonical = tuple(canonical_parameter_json(item) for item in configurations)
    trial_ids = tuple(
        deterministic_trial_id(spec.family_id, spec.version, item)
        for item in configurations
    )
    assert len(set(canonical)) == len(set(trial_ids)) == 9
    assert trial_ids == tuple(
        deterministic_trial_id(spec.family_id, spec.version, item)
        for item in family.enumerate_parameters()
    )


def test_baseline_cell_matches_the_retired_session_range_expansion_v1() -> None:
    spec = load_eurusd_session_range_expansion_spec()
    assert "12:00" in spec.breakout_window_end_grid
    assert "16:00" in spec.scheduled_exit_grid


def test_semantic_sha_covers_grid_and_selector_order() -> None:
    document = load_eurusd_session_range_expansion_spec().canonical_document
    assert semantic_sha256_for_document(document) == (
        EURUSD_SESSION_RANGE_EXPANSION_SEMANTIC_SHA256
    )
    assert semantic_sha256_for_document(deepcopy(document)) == (
        EURUSD_SESSION_RANGE_EXPANSION_SEMANTIC_SHA256
    )

    changed_grid = deepcopy(document)
    changed_grid["parameter_grid"]["breakout_window_end"][0] = "10:30"
    changed_selector = deepcopy(document)
    changed_selector["selector"]["ranking_order"][0:2] = reversed(
        changed_selector["selector"]["ranking_order"][0:2]
    )

    assert semantic_sha256_for_document(changed_grid) != (
        EURUSD_SESSION_RANGE_EXPANSION_SEMANTIC_SHA256
    )
    assert semantic_sha256_for_document(changed_selector) != (
        EURUSD_SESSION_RANGE_EXPANSION_SEMANTIC_SHA256
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda doc: doc["parameter_grid"].__setitem__("range_start", "01:00"),
        lambda doc: doc["signal"].__setitem__("maximum_entries_per_london_day", 2),
        lambda doc: doc["risk_normalization"].__setitem__(
            "sizing_refresh_rule", "every_bar"
        ),
        lambda doc: doc["sample_count"].__setitem__(
            "count_entry_and_exit_separately", True
        ),
        lambda doc: doc["eligibility"].__setitem__("minimum_positive_folds", 1),
        lambda doc: doc["sealed_partitions"].__setitem__("final_holdout", "open"),
    ],
)
def test_mutating_frozen_fields_breaks_the_semantic_hash(
    mutation: Callable[[dict[str, Any]], None],
) -> None:
    document = deepcopy(load_eurusd_session_range_expansion_spec().canonical_document)
    mutation(document)
    assert semantic_sha256_for_document(document) != (
        EURUSD_SESSION_RANGE_EXPANSION_SEMANTIC_SHA256
    )


def test_loader_rejects_a_document_whose_recorded_hash_was_edited(
    tmp_path: Path,
) -> None:
    import yaml

    from ftmoquant.research.eurusd_session_range_expansion_spec import (
        EURUSD_SESSION_RANGE_EXPANSION_SPEC_PATH,
    )

    document = yaml.safe_load(
        EURUSD_SESSION_RANGE_EXPANSION_SPEC_PATH.read_text(encoding="utf-8")
    )
    document["parameter_grid"]["scheduled_exit"] = ["15:00", "16:00", "18:00"]
    tampered = tmp_path / "tampered.yaml"
    tampered.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(EurusdSessionRangeExpansionSpecError):
        load_eurusd_session_range_expansion_spec(tampered)


def test_existing_three_development_folds_are_frozen_exactly() -> None:
    folds = frozen_development_folds()
    declared = load_eurusd_session_range_expansion_spec().canonical_document[
        "development_folds"
    ]

    assert declared["version"] == folds.version
    assert declared["semantic_sha256"] == folds.semantic_sha256
    assert [item["fold_id"] for item in declared["folds"]] == [
        "dev_fold_1",
        "dev_fold_2",
        "dev_fold_3",
    ]
    assert declared["folds"][0]["evaluate_start_utc"] == "2020-04-11T00:00:00Z"
    assert declared["folds"][-1]["evaluate_end_exclusive_utc"] == (
        "2023-04-11T00:00:00Z"
    )


def test_neighbours_move_exactly_one_dimension_one_adjacent_step() -> None:
    family = EurusdSessionRangeExpansionFamily()
    corner = {"breakout_window_end": "11:00", "scheduled_exit": "15:00"}
    interior = {"breakout_window_end": "12:00", "scheduled_exit": "16:00"}

    corner_neighbours = family.neighbours(corner)
    interior_neighbours = family.neighbours(interior)

    assert len(corner_neighbours) == 2
    assert len(interior_neighbours) == 4
    for neighbour in (*corner_neighbours, *interior_neighbours):
        base = corner if neighbour in corner_neighbours else interior
        differing = {key for key in base if neighbour[key] != base[key]}
        assert len(differing) == 1
    assert corner not in corner_neighbours
    assert interior not in interior_neighbours
