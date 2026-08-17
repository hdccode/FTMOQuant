from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from ftmoquant.research.eurusd_liquidity_shock_reversion_development import (
    EurusdLiquidityShockReversionFamily,
)
from ftmoquant.research.eurusd_liquidity_shock_reversion_spec import (
    EURUSD_LIQUIDITY_SHOCK_REVERSION_SEMANTIC_SHA256,
    EurusdLiquidityShockReversionSpecError,
    load_eurusd_liquidity_shock_reversion_spec,
    semantic_sha256_for_document,
)
from ftmoquant.research.g1.parameter_space import (
    canonical_parameter_json,
    deterministic_trial_id,
)
from ftmoquant.research.stage_g import frozen_development_folds


def test_frozen_grid_is_exact_36_cells_unique_and_deterministic() -> None:
    spec = load_eurusd_liquidity_shock_reversion_spec()
    family = EurusdLiquidityShockReversionFamily(spec)
    configurations = family.enumerate_parameters()

    assert spec.semantic_sha256 == EURUSD_LIQUIDITY_SHOCK_REVERSION_SEMANTIC_SHA256
    assert spec.expected_unique_trial_count == len(configurations) == 36
    assert spec.baseline_prior_returns_grid == (30, 60, 120)
    assert spec.shock_multiple_grid == (3.0, 4.0, 5.0, 6.0)
    assert spec.hold_eligible_minutes_grid == (5, 15, 30)
    canonical = tuple(canonical_parameter_json(item) for item in configurations)
    trial_ids = tuple(
        deterministic_trial_id(spec.family_id, spec.version, item)
        for item in configurations
    )
    assert len(set(canonical)) == len(set(trial_ids)) == 36
    assert trial_ids == tuple(
        deterministic_trial_id(spec.family_id, spec.version, item)
        for item in family.enumerate_parameters()
    )


def test_semantic_sha_covers_grid_and_selector_order() -> None:
    document = load_eurusd_liquidity_shock_reversion_spec().canonical_document
    assert semantic_sha256_for_document(document) == (
        EURUSD_LIQUIDITY_SHOCK_REVERSION_SEMANTIC_SHA256
    )
    assert semantic_sha256_for_document(deepcopy(document)) == (
        EURUSD_LIQUIDITY_SHOCK_REVERSION_SEMANTIC_SHA256
    )

    changed_grid = deepcopy(document)
    changed_grid["parameter_grid"]["shock_multiple"][0] = 3.5
    changed_selector = deepcopy(document)
    changed_selector["selector"]["ranking_order"][0:2] = reversed(
        changed_selector["selector"]["ranking_order"][0:2]
    )

    assert semantic_sha256_for_document(changed_grid) != (
        EURUSD_LIQUIDITY_SHOCK_REVERSION_SEMANTIC_SHA256
    )
    assert semantic_sha256_for_document(changed_selector) != (
        EURUSD_LIQUIDITY_SHOCK_REVERSION_SEMANTIC_SHA256
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda doc: doc["parameter_grid"].__setitem__(
            "baseline_prior_returns", [30, 60, 90]
        ),
        lambda doc: doc["signal"].__setitem__("positive_shock_target", 1),
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
    document = deepcopy(load_eurusd_liquidity_shock_reversion_spec().canonical_document)
    mutation(document)
    assert semantic_sha256_for_document(document) != (
        EURUSD_LIQUIDITY_SHOCK_REVERSION_SEMANTIC_SHA256
    )


def test_loader_rejects_a_document_whose_recorded_hash_was_edited(
    tmp_path: Path,
) -> None:
    import yaml

    from ftmoquant.research.eurusd_liquidity_shock_reversion_spec import (
        EURUSD_LIQUIDITY_SHOCK_REVERSION_SPEC_PATH,
    )

    document = yaml.safe_load(
        EURUSD_LIQUIDITY_SHOCK_REVERSION_SPEC_PATH.read_text(encoding="utf-8")
    )
    document["parameter_grid"]["hold_eligible_minutes"] = [5, 15, 45]
    tampered = tmp_path / "tampered.yaml"
    tampered.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(EurusdLiquidityShockReversionSpecError):
        load_eurusd_liquidity_shock_reversion_spec(tampered)


def test_existing_three_development_folds_are_frozen_exactly() -> None:
    folds = frozen_development_folds()
    declared = load_eurusd_liquidity_shock_reversion_spec().canonical_document[
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
    family = EurusdLiquidityShockReversionFamily()
    corner = {
        "baseline_prior_returns": 30,
        "shock_multiple": 3.0,
        "hold_eligible_minutes": 5,
    }
    interior = {
        "baseline_prior_returns": 60,
        "shock_multiple": 4.0,
        "hold_eligible_minutes": 15,
    }

    corner_neighbours = family.neighbours(corner)
    interior_neighbours = family.neighbours(interior)

    assert len(corner_neighbours) == 3
    assert len(interior_neighbours) == 6
    for neighbour in (*corner_neighbours, *interior_neighbours):
        base = corner if neighbour in corner_neighbours else interior
        differing = {key for key in base if neighbour[key] != base[key]}
        assert len(differing) == 1
    assert corner not in corner_neighbours
    assert interior not in interior_neighbours
