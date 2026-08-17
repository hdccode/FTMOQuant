from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

from ftmoquant.research.eurusd_policy_rate_carry_proxy_spec import (
    EURUSD_POLICY_RATE_CARRY_PROXY_SEMANTIC_SHA256,
    EURUSD_POLICY_RATE_CARRY_PROXY_SPEC_PATH,
    EurusdPolicyRateCarryProxySpecError,
    load_eurusd_policy_rate_carry_proxy_spec,
    semantic_sha256_for_document,
)
from ftmoquant.research.stage_g import frozen_development_folds


def test_spec_loads_and_matches_the_frozen_semantic_sha() -> None:
    spec = load_eurusd_policy_rate_carry_proxy_spec()
    assert spec.family_id == "eurusd_policy_rate_carry_proxy_v1"
    assert spec.version == "1.0.0"
    assert spec.semantic_sha256 == EURUSD_POLICY_RATE_CARRY_PROXY_SEMANTIC_SHA256
    assert "cross-sectional FX carry factor" in spec.economic_hypothesis


def test_semantic_sha_is_deterministic_and_order_independent() -> None:
    document = load_eurusd_policy_rate_carry_proxy_spec().canonical_document
    assert semantic_sha256_for_document(document) == (
        EURUSD_POLICY_RATE_CARRY_PROXY_SEMANTIC_SHA256
    )
    assert semantic_sha256_for_document(deepcopy(document)) == (
        EURUSD_POLICY_RATE_CARRY_PROXY_SEMANTIC_SHA256
    )


def test_baseline_only_declaration_has_no_search_grid() -> None:
    document = load_eurusd_policy_rate_carry_proxy_spec().canonical_document
    assert document["parameter_family"]["mode"] == "baseline_only"
    assert document["parameter_family"]["parameter_count"] == 0
    assert document["search"]["mode"] == "baseline_only"
    assert document["search"]["select_candidate_invoked"] is False
    assert document["search"]["assess_plateau_invoked"] is False


def test_folds_match_the_frozen_development_fold_plan() -> None:
    folds = frozen_development_folds()
    declared = load_eurusd_policy_rate_carry_proxy_spec().canonical_document[
        "development_folds"
    ]
    assert declared["version"] == folds.version
    assert declared["semantic_sha256"] == folds.semantic_sha256
    assert [item["fold_id"] for item in declared["folds"]] == [
        "dev_fold_1",
        "dev_fold_2",
        "dev_fold_3",
    ]


def test_sealed_partitions_lock_validation_and_holdout() -> None:
    document = load_eurusd_policy_rate_carry_proxy_spec().canonical_document
    seals = document["sealed_partitions"]
    assert seals["final_holdout"] == "locked"
    assert seals["strategy_returns_accessed"] is False
    assert "2023-04-11T00:00:00Z" in seals["validation"]
    assert "2024-08-21T00:00:00Z" in seals["validation"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda doc: doc["proxy_carry"].__setitem__(
            "calibration_status", "broker_calibrated"
        ),
        lambda doc: doc["sample_count"].__setitem__("is_not_a_trade_count", False),
        lambda doc: doc["eligibility"].__setitem__(
            "minimum_pooled_eligible_observations", 1
        ),
        lambda doc: doc["search"].__setitem__("select_candidate_invoked", True),
        lambda doc: doc["sealed_partitions"].__setitem__("final_holdout", "open"),
        lambda doc: doc["execution_and_costs"].__setitem__(
            "cost_stress_multiplier", 1.0
        ),
        lambda doc: doc["lineage"].__setitem__("prior_parameters_reused", True),
    ],
)
def test_mutating_frozen_fields_breaks_the_semantic_hash(
    mutation: Callable[[dict[str, Any]], None],
) -> None:
    document = deepcopy(load_eurusd_policy_rate_carry_proxy_spec().canonical_document)
    mutation(document)
    assert semantic_sha256_for_document(document) != (
        EURUSD_POLICY_RATE_CARRY_PROXY_SEMANTIC_SHA256
    )


def test_loader_rejects_a_document_whose_recorded_hash_was_edited(
    tmp_path: Path,
) -> None:
    document = yaml.safe_load(
        EURUSD_POLICY_RATE_CARRY_PROXY_SPEC_PATH.read_text(encoding="utf-8")
    )
    document["eligibility"]["minimum_positive_folds"] = 1
    tampered = tmp_path / "tampered.yaml"
    tampered.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(EurusdPolicyRateCarryProxySpecError):
        load_eurusd_policy_rate_carry_proxy_spec(tampered)


def test_loader_rejects_a_missing_block(tmp_path: Path) -> None:
    document = yaml.safe_load(
        EURUSD_POLICY_RATE_CARRY_PROXY_SPEC_PATH.read_text(encoding="utf-8")
    )
    del document["limitations"]
    tampered = tmp_path / "tampered.yaml"
    tampered.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(EurusdPolicyRateCarryProxySpecError):
        load_eurusd_policy_rate_carry_proxy_spec(tampered)
