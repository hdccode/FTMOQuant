from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ftmoquant.research.alpha_lab.batch3_preregistration import (
    DEVELOPMENT_FOLD_BOUNDARIES,
    DEVELOPMENT_SUBPERIOD_MIDPOINT,
    EXPLICITLY_OUT_OF_SCOPE,
    IN_SCOPE_FAMILIES,
    MID_PIPELINE_EXCLUDED_FROM_BATCH3,
    PREREGISTRATION_VERSION,
    RETIRED_FAMILIES,
    STAGE_ID,
    Batch3PreregistrationError,
    _canonical_sha256,
    build_preregistration,
    verify_preregistration,
    write_preregistration,
)
from ftmoquant.research.stage_g import (
    DEVELOPMENT_END_EXCLUSIVE,
    DEVELOPMENT_START,
    HOLDOUT_START,
    VALIDATION_START,
)

_FIXED_NOW = datetime(2026, 8, 19, 12, 0, 0, tzinfo=UTC)


def test_family_scope_is_exactly_b3f1_b3f2_b3f3() -> None:
    assert set(IN_SCOPE_FAMILIES) == {
        "B3F1_spread_mean_reversion",
        "B3F2_asian_range_fade",
        "B3F3_session_open_microstructure_mean_reversion",
    }


def test_all_in_scope_families_declare_native_bid_ask_execution_from_the_outset() -> (
    None
):
    for family_id, spec in IN_SCOPE_FAMILIES.items():
        assert spec["execution_semantics"] == "native_bid_ask_from_outset", family_id


def test_all_in_scope_families_declare_a_causality_constraint() -> None:
    for family_id, spec in IN_SCOPE_FAMILIES.items():
        assert "causality_constraint" in spec, family_id
        assert len(spec["causality_constraint"]) > 20, family_id


def test_all_in_scope_families_declare_structurally_two_sided_true() -> None:
    for family_id, spec in IN_SCOPE_FAMILIES.items():
        assert spec["structurally_two_sided"] is True, family_id


def test_cross_sectional_basket_is_explicitly_out_of_scope() -> None:
    assert "cross_sectional_carry_momentum_basket" in EXPLICITLY_OUT_OF_SCOPE


def test_no_retired_family_id_collides_with_an_in_scope_family_id() -> None:
    retired_ids = {row["strategy_id"] for row in RETIRED_FAMILIES}
    assert retired_ids.isdisjoint(set(IN_SCOPE_FAMILIES))


def test_retired_family_list_has_no_duplicate_ids() -> None:
    ids = [row["strategy_id"] for row in RETIRED_FAMILIES]
    assert len(ids) == len(set(ids))


def test_retired_family_list_includes_the_batch2_lesson_candidate() -> None:
    retired_ids = {row["strategy_id"] for row in RETIRED_FAMILIES}
    assert "B2F1_sweep_bos_retest" in retired_ids


def test_mean_reversion_h1_is_excluded_as_mid_pipeline_not_retired() -> None:
    mid_pipeline_ids = {row["strategy_id"] for row in MID_PIPELINE_EXCLUDED_FROM_BATCH3}
    retired_ids = {row["strategy_id"] for row in RETIRED_FAMILIES}
    assert "mean_reversion_h1_v1" in mid_pipeline_ids
    assert "mean_reversion_h1_v1" not in retired_ids
    assert "mean_reversion_h1_v1" not in IN_SCOPE_FAMILIES


def test_development_fold_boundaries_span_the_frozen_development_interval() -> None:
    assert DEVELOPMENT_FOLD_BOUNDARIES[0] == DEVELOPMENT_START
    assert DEVELOPMENT_FOLD_BOUNDARIES[-1] == DEVELOPMENT_END_EXCLUSIVE
    assert len(DEVELOPMENT_FOLD_BOUNDARIES) == 5


def test_development_fold_boundaries_are_four_equal_373_day_folds() -> None:
    from datetime import timedelta

    for start, end in zip(
        DEVELOPMENT_FOLD_BOUNDARIES[:-1], DEVELOPMENT_FOLD_BOUNDARIES[1:], strict=True
    ):
        assert end - start == timedelta(days=373)


def test_subperiod_midpoint_coincides_with_second_fold_boundary() -> None:
    assert DEVELOPMENT_SUBPERIOD_MIDPOINT == DEVELOPMENT_FOLD_BOUNDARIES[2]


def test_subperiod_midpoint_is_strictly_between_development_start_and_end() -> None:
    assert (
        DEVELOPMENT_START < DEVELOPMENT_SUBPERIOD_MIDPOINT < DEVELOPMENT_END_EXCLUSIVE
    )


def test_build_preregistration_partition_matches_stage_g_unchanged() -> None:
    document = build_preregistration(created_at_utc=_FIXED_NOW)
    partition = document["validation_partition"]
    assert partition["start_utc"] == VALIDATION_START.isoformat().replace("+00:00", "Z")
    assert partition["end_exclusive_utc"] == HOLDOUT_START.isoformat().replace(
        "+00:00", "Z"
    )


def test_build_preregistration_lifecycle_flags_are_all_false() -> None:
    document = build_preregistration(created_at_utc=_FIXED_NOW)
    lifecycle = document["lifecycle"]
    assert lifecycle == {
        "development_accessed": False,
        "validation_accessed": False,
        "holdout_accessed": False,
    }


def test_build_preregistration_is_deterministic_given_the_same_timestamp() -> None:
    first = build_preregistration(created_at_utc=_FIXED_NOW)
    second = build_preregistration(created_at_utc=_FIXED_NOW)
    assert first == second


def test_semantic_hash_excludes_only_itself_and_changes_with_content() -> None:
    document = build_preregistration(created_at_utc=_FIXED_NOW)
    recomputed = _canonical_sha256(document)
    assert document["preregistration_semantic_sha256"] == recomputed

    tampered = dict(document)
    tampered["development_gates"] = dict(tampered["development_gates"])
    tampered["development_gates"]["economic"] = {"expectancy_usd_per_trade_gt": -1}
    assert _canonical_sha256(tampered) != recomputed


def test_development_gates_freeze_exact_thresholds_from_the_task_spec() -> None:
    document = build_preregistration(created_at_utc=_FIXED_NOW)
    gates = document["development_gates"]
    assert gates["economic"]["expectancy_usd_per_trade_gt"] == 0
    assert gates["economic"]["profit_factor_gt"] == 1.15
    assert gates["rolling_stability"]["window_sizes"] == [30, 50]
    assert (
        gates["rolling_stability"]["fraction_of_eligible_windows_positive_gte"] == 0.70
    )
    assert gates["profit_concentration"]["max_single_month_share"] == 0.15
    assert gates["profit_concentration"]["max_single_quarter_share"] == 0.30
    assert gates["exceptional_winner_dependency"]["remaining_profit_factor_gt"] == 1.05
    assert gates["temporal_stability"]["fold_count"] == 4
    assert gates["temporal_stability"]["rule"] == (
        "positive net return in at least 3 of the 4 folds"
    )
    assert (
        gates["tail_and_concentration"][
            "largest_single_winning_trade_share_of_total_positive_profit_lte"
        ]
        == 0.20
    )
    assert gates["opportunity_density"]["rule"] == "completed_trade_count >= 80"
    assert (
        gates["parameter_neighborhood_robustness"]["default_min_connected_region_size"]
        == 2
    )


def test_transaction_cost_gate_declares_the_required_new_stress_mechanism() -> None:
    document = build_preregistration(created_at_utc=_FIXED_NOW)
    cost_gate = document["development_gates"]["transaction_cost_sensitivity"]
    assert "does not exist in the repository yet" in cost_gate["implementation_status"]
    assert "1.5x" in cost_gate["rule_1_5x"]
    assert "2.0x" in cost_gate["rule_2_0x"]
    assert "B3F2 and B3F3 only" in cost_gate["rule_2_0x"]


def test_ftmo_development_gates_freeze_pass_both_and_drawdown_thresholds() -> None:
    document = build_preregistration(created_at_utc=_FIXED_NOW)
    ftmo_gates = document["ftmo_development_gates"]
    assert ftmo_gates["full_development_pass_both"]["rule"] == "pass_both >= 0.70"
    assert "0.60" in ftmo_gates["cross_subperiod_stability"]["rule"]
    assert (
        "3,000" in ftmo_gates["drawdown"]["rule"]
        or "3000" in ftmo_gates["drawdown"]["rule"]
    )


def test_validation_policy_precommits_three_gates_and_forbids_retuning() -> None:
    document = build_preregistration(created_at_utc=_FIXED_NOW)
    policy = document["validation_policy"]
    gates = policy["precommitted_gates"]
    assert gates["validation_passed"] == "A AND B AND C"
    assert "0.70" in gates["C_ftmo_pass_both"]
    assert "retuning" in policy["on_failure"]


def test_firewall_states_no_batch2_validation_informed_exceptions() -> None:
    document = build_preregistration(created_at_utc=_FIXED_NOW)
    firewall_rules = " ".join(document["firewall"]["rules"])
    assert "B2F1" in firewall_rules
    assert "final holdout remains completely sealed" in firewall_rules


def test_write_preregistration_refuses_to_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "batch3_methodology_preregistration_v1.json"
    write_preregistration(path=path, created_at_utc=_FIXED_NOW)
    with pytest.raises(Batch3PreregistrationError):
        write_preregistration(path=path, created_at_utc=_FIXED_NOW)


def test_verify_preregistration_accepts_a_freshly_written_document(
    tmp_path: Path,
) -> None:
    path = tmp_path / "batch3_methodology_preregistration_v1.json"
    write_preregistration(path=path, created_at_utc=_FIXED_NOW)
    document = verify_preregistration(path=path)
    assert document["preregistration_version"] == PREREGISTRATION_VERSION
    assert document["stage"] == STAGE_ID


def test_verify_preregistration_rejects_a_tampered_self_hash(tmp_path: Path) -> None:
    path = tmp_path / "batch3_methodology_preregistration_v1.json"
    write_preregistration(path=path, created_at_utc=_FIXED_NOW)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["development_gates"]["economic"]["profit_factor_gt"] = 999
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(Batch3PreregistrationError, match="semantic_sha256"):
        verify_preregistration(path=path)


def test_verify_preregistration_rejects_a_declared_true_lifecycle_flag(
    tmp_path: Path,
) -> None:
    path = tmp_path / "batch3_methodology_preregistration_v1.json"
    document = build_preregistration(created_at_utc=_FIXED_NOW)
    document["lifecycle"]["validation_accessed"] = True
    from ftmoquant.research.alpha_lab.batch3_preregistration import _canonical_sha256

    document["preregistration_semantic_sha256"] = _canonical_sha256(document)
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(Batch3PreregistrationError, match="lifecycle"):
        verify_preregistration(path=path)


def test_verify_preregistration_rejects_a_shrunk_family_scope(tmp_path: Path) -> None:
    path = tmp_path / "batch3_methodology_preregistration_v1.json"
    document = build_preregistration(created_at_utc=_FIXED_NOW)
    del document["family_scope"]["in_scope"][
        "B3F3_session_open_microstructure_mean_reversion"
    ]
    from ftmoquant.research.alpha_lab.batch3_preregistration import _canonical_sha256

    document["preregistration_semantic_sha256"] = _canonical_sha256(document)
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(Batch3PreregistrationError, match="in_scope"):
        verify_preregistration(path=path)


def test_frozen_preregistration_artifact_on_disk_is_self_consistent() -> None:
    path = Path("config/research/batch3_methodology_preregistration_v1.json")
    if not path.exists():
        pytest.skip("preregistration artifact not yet generated in this checkout")
    document = verify_preregistration(path=path)
    assert document["family_scope"]["in_scope"].keys() == IN_SCOPE_FAMILIES.keys()
