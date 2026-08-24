from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ftmoquant.research.alpha_lab.batch3_preregistration import (
    IN_SCOPE_FAMILIES,
    Batch3PreregistrationError,
)
from ftmoquant.research.alpha_lab.batch3_preregistration import (
    PREREGISTRATION_VERSION as V1_PREREGISTRATION_VERSION,
)
from ftmoquant.research.alpha_lab.batch3_preregistration_v2 import (
    DEVELOPMENT_DIAGNOSTICS,
    DEVELOPMENT_GATES,
    FTMO_DEVELOPMENT_GATES,
    PREREGISTRATION_VERSION,
    SUPERSEDES_SEMANTIC_SHA256,
    VALIDATION_POLICY,
    _canonical_sha256,
    build_preregistration_v2,
    verify_preregistration_v2,
    write_preregistration_v2,
)

_FIXED_NOW = datetime(2026, 8, 19, 13, 0, 0, tzinfo=UTC)
_V1_HASH = "82e00f5e5b3a7cf4269bd61163bf8ea4058a31e6414b1fed0077c820a417a68b"


def test_v2_version_string_is_distinct_from_v1() -> None:
    assert PREREGISTRATION_VERSION == "batch3-methodology-preregistration-v2"
    assert PREREGISTRATION_VERSION != V1_PREREGISTRATION_VERSION


def test_family_scope_is_unchanged_from_v1() -> None:
    document = build_preregistration_v2(created_at_utc=_FIXED_NOW)
    assert document["family_scope"]["unchanged_from_v1"] is True
    assert set(document["family_scope"]["in_scope"]) == set(IN_SCOPE_FAMILIES)


def test_profit_factor_hard_threshold_is_exactly_1_10() -> None:
    assert DEVELOPMENT_GATES["economic"]["profit_factor_gt"] == 1.10


def test_quarter_concentration_hard_threshold_is_exactly_0_40() -> None:
    assert DEVELOPMENT_GATES["profit_concentration"]["max_single_quarter_share"] == 0.40


def test_profit_concentration_has_no_monthly_hard_threshold() -> None:
    concentration = DEVELOPMENT_GATES["profit_concentration"]
    assert "max_single_month_share" not in concentration
    assert "period_boundaries" in concentration
    assert concentration["period_boundaries"] == "calendar quarter in UTC"


def test_best_5_percent_gate_requires_only_remaining_expectancy() -> None:
    gate = DEVELOPMENT_GATES["exceptional_winner_dependency"]
    assert gate["remaining_expectancy_usd_per_trade_gt"] == 0
    assert "remaining_profit_factor_gt" not in gate
    assert "profitable" in gate["rule"].lower()


def test_opportunity_density_b3f1_is_50_others_are_80() -> None:
    density = DEVELOPMENT_GATES["opportunity_density"]
    assert density["default_min_trades"] == 80
    assert density["family_overrides"] == {"B3F1_spread_mean_reversion": 50}
    assert "B3F2_asian_range_fade" not in density["family_overrides"]
    assert (
        "B3F3_session_open_microstructure_mean_reversion"
        not in density["family_overrides"]
    )


def test_transaction_cost_family_requirements_match_spec() -> None:
    requirements = DEVELOPMENT_GATES["transaction_cost_sensitivity"][
        "family_requirements"
    ]
    assert requirements["B3F1_spread_mean_reversion"]["must_survive_multipliers"] == [
        1.5
    ]
    assert requirements["B3F2_asian_range_fade"]["must_survive_multipliers"] == [
        1.5,
        2.0,
    ]
    assert requirements["B3F3_session_open_microstructure_mean_reversion"][
        "must_survive_multipliers"
    ] == [1.5, 2.0]


def test_parameter_neighborhood_min_region_is_frozen_at_2() -> None:
    assert (
        DEVELOPMENT_GATES["parameter_neighborhood_robustness"][
            "min_connected_region_size"
        ]
        == 2
    )


def test_temporal_stability_gate_is_unchanged_from_v1() -> None:
    gate = DEVELOPMENT_GATES["temporal_stability"]
    assert gate["fold_count"] == 4
    assert gate["rule"] == "positive net return in at least 3 of the 4 folds"


def test_rolling_expectancy_is_diagnostic_only() -> None:
    rolling = DEVELOPMENT_DIAGNOSTICS["rolling_expectancy"]
    assert (
        DEVELOPMENT_DIAGNOSTICS["status"]
        == "report_only_not_a_gate_at_any_stage_before_b3_4"
    )
    assert "median_eligible_window_expectancy" in rolling["reported_fields"]
    assert "fraction_of_eligible_windows_positive" in rolling["reported_fields"]
    assert "removed_as_hard_gate_from_v1" in rolling


def test_monthly_concentration_is_diagnostic_only() -> None:
    monthly = DEVELOPMENT_DIAGNOSTICS["monthly_concentration"]
    assert monthly["reported_field"] == "max_single_month_share"
    assert "removed_as_hard_gate_from_v1" in monthly


def test_tail_statistics_are_diagnostic_only() -> None:
    tail = DEVELOPMENT_DIAGNOSTICS["tail_statistics"]
    assert set(tail["reported_fields"]) == {
        "largest_winning_trade_share_of_total_positive_profit",
        "pnl_skewness",
        "pnl_kurtosis",
    }
    assert "removed_as_hard_gate_from_v1" in tail


def test_directional_breakdown_is_diagnostic_and_forbids_post_hoc_filtering() -> None:
    directional = DEVELOPMENT_DIAGNOSTICS["directional_breakdown"]
    assert "profit_factor" in directional["reported_fields_per_direction"]
    assert "removed_as_hard_gate_from_v1" in directional
    assert "no_direction_specific_filtering_rule" in directional


def test_drawdown_diagnostic_removes_the_3000_hard_gate() -> None:
    drawdown = DEVELOPMENT_DIAGNOSTICS["drawdown"]
    assert "3,000" in drawdown["removed_as_hard_gate_from_v1"]


def test_b3_4_full_development_pass_both_remains_a_hard_gate() -> None:
    gate = FTMO_DEVELOPMENT_GATES["full_development_pass_both"]
    assert gate["status"] == "hard_gate"
    assert gate["rule"] == "pass_both >= 0.70"


def test_split_half_ratio_is_diagnostic_warning_not_a_gate() -> None:
    stability = FTMO_DEVELOPMENT_GATES["cross_subperiod_stability"]
    assert stability["status"] == "diagnostic_warning_not_an_advancement_gate"
    assert stability["labeling_rule"]["ratio_gte_0_60"] == "stable_diagnostic"
    assert stability["labeling_rule"]["ratio_lt_0_60"] == "instability_warning"
    assert "removed_as_hard_gate_from_v1" in stability


def test_validation_gates_are_exactly_return_and_sharpe() -> None:
    gates = VALIDATION_POLICY["precommitted_gates"]
    assert set(gates) == {
        "A_native_spread_positive_return",
        "B_native_spread_positive_sharpe",
        "validation_passed",
    }
    assert gates["validation_passed"] == "A AND B"


def test_validation_pass_both_is_diagnostic_only() -> None:
    assert VALIDATION_POLICY["validation_ftmo_pass_both"]["status"] == (
        "report_only_diagnostic"
    )
    assert "removed_as_hard_gate_from_v1" in VALIDATION_POLICY


def test_v2_supersedes_exactly_the_recorded_v1_semantic_sha() -> None:
    assert SUPERSEDES_SEMANTIC_SHA256 == _V1_HASH
    document = build_preregistration_v2(created_at_utc=_FIXED_NOW)
    assert document["supersedes_semantic_sha256"] == _V1_HASH


def test_v1_artifact_on_disk_is_untouched_and_still_matches_its_own_hash() -> None:
    v1_path = Path("config/research/batch3_methodology_preregistration_v1.json")
    if not v1_path.exists():
        pytest.skip("v1 artifact not present in this checkout")
    v1_document = json.loads(v1_path.read_text(encoding="utf-8"))
    assert v1_document["preregistration_semantic_sha256"] == _V1_HASH
    assert v1_document["preregistration_version"] == V1_PREREGISTRATION_VERSION


def test_build_preregistration_v2_is_deterministic() -> None:
    first = build_preregistration_v2(created_at_utc=_FIXED_NOW)
    second = build_preregistration_v2(created_at_utc=_FIXED_NOW)
    assert first == second


def test_semantic_hash_changes_if_a_hard_gate_value_changes() -> None:
    document = build_preregistration_v2(created_at_utc=_FIXED_NOW)
    recomputed = _canonical_sha256(document)
    assert document["preregistration_semantic_sha256"] == recomputed

    tampered = dict(document)
    tampered["development_gates"] = dict(tampered["development_gates"])
    tampered["development_gates"]["economic"] = {"profit_factor_gt": 1.0}
    assert _canonical_sha256(tampered) != recomputed


def test_write_preregistration_v2_refuses_to_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "batch3_methodology_preregistration_v2.json"
    write_preregistration_v2(path=path, created_at_utc=_FIXED_NOW)
    with pytest.raises(Batch3PreregistrationError):
        write_preregistration_v2(path=path, created_at_utc=_FIXED_NOW)


def test_verify_preregistration_v2_accepts_a_freshly_written_document(
    tmp_path: Path,
) -> None:
    path = tmp_path / "batch3_methodology_preregistration_v2.json"
    write_preregistration_v2(path=path, created_at_utc=_FIXED_NOW)
    document = verify_preregistration_v2(path=path)
    assert document["preregistration_version"] == PREREGISTRATION_VERSION


def test_verify_preregistration_v2_rejects_a_tampered_self_hash(
    tmp_path: Path,
) -> None:
    path = tmp_path / "batch3_methodology_preregistration_v2.json"
    write_preregistration_v2(path=path, created_at_utc=_FIXED_NOW)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["development_gates"]["economic"]["profit_factor_gt"] = 999
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(Batch3PreregistrationError, match="semantic_sha256"):
        verify_preregistration_v2(path=path)


def test_verify_preregistration_v2_rejects_a_wrong_supersedes_hash(
    tmp_path: Path,
) -> None:
    path = tmp_path / "batch3_methodology_preregistration_v2.json"
    write_preregistration_v2(path=path, created_at_utc=_FIXED_NOW)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["supersedes_semantic_sha256"] = "0" * 64
    document["preregistration_semantic_sha256"] = _canonical_sha256(document)
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(Batch3PreregistrationError, match="supersedes"):
        verify_preregistration_v2(path=path)


def test_frozen_v2_artifact_on_disk_is_self_consistent() -> None:
    path = Path("config/research/batch3_methodology_preregistration_v2.json")
    if not path.exists():
        pytest.skip("v2 artifact not yet generated in this checkout")
    document = verify_preregistration_v2(path=path)
    assert document["supersedes_semantic_sha256"] == _V1_HASH


def test_no_new_data_access_helper_was_introduced() -> None:
    """The v2 module must not import any DEVELOPMENT/VALIDATION/HOLDOUT
    data-loading machinery (ParquetDataCatalog, alpha_lab.data loaders,
    etc.) -- only stage_g's frozen constants and small config-file hashing,
    exactly like v1."""

    import ftmoquant.research.alpha_lab.batch3_preregistration_v2 as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    forbidden_tokens = [
        "ParquetDataCatalog",
        "load_alpha_lab_dataset",
        "assemble_aligned_dataset",
        "open_development_context",
    ]
    for token in forbidden_tokens:
        assert token not in source, f"unexpected data-access symbol: {token}"
