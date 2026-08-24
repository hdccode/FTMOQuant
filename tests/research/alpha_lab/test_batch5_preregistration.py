from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from ftmoquant.research.alpha_lab.batch5_preregistration import (
    B5A_SLEEVES,
    B5C_INSTRUMENTS,
    COMMON_GATES,
    EXPECTED_PREREGISTRATION_SEMANTIC_SHA256,
    FAMILY_B5A,
    FAMILY_B5B,
    FAMILY_B5C,
    PREREGISTRATION_PATH,
    PRIMARY_FAMILIES,
    Batch5PreregistrationError,
    _canonical_sha256,
    build_preregistration,
    verify_preregistration,
    write_preregistration,
)


@pytest.fixture
def document() -> dict[str, Any]:
    return build_preregistration()


def test_exact_three_family_scope_and_hypothesis_count(
    document: dict[str, Any],
) -> None:
    assert document["family_scope"] == {
        "primary_exact": list(PRIMARY_FAMILIES),
        "exact_family_count": 3,
        "extra_or_deferred_families": [],
    }
    accounting = document["hypothesis_accounting"]
    assert accounting == {
        "family_configurations": 3,
        "B5A_currency_sleeves": 7,
        "B5B_direct_instrument_sleeves": 1,
        "B5C_instrument_sleeves": 5,
        "total_executable_sleeve_hypotheses": 13,
        "robustness_variant_count": 0,
    }


def test_b5a_exact_report_transform_sign_timing_and_horizon(
    document: dict[str, Any],
) -> None:
    family = document["families"][FAMILY_B5A]
    replication = family["source_faithful_replication"]
    assert replication["report"] == "CFTC Traders in Financial Futures, Futures Only"
    assert replication["trader_category"] == "Dealer/Intermediary long and short"
    assert replication["legacy_commercial_substitution_permitted"] is False
    assert "j=0..11" in replication["scaled_position_formula"]
    assert replication["signal"].startswith("delta_f_star")
    assert replication["threshold"].startswith("strict sign only")
    assert (
        replication["holding_period"] == "three calendar months from actual entry fill"
    )
    assert replication["rebalance_frequency"] == "monthly independent cohorts"
    assert replication["permitted_robustness_variants"] == []

    causal = family["causal_publication_contract"]
    assert causal["nominal_release"].startswith("Friday 15:30 America/New_York")
    assert causal["report_date_is_never_availability_date"] is True
    assert causal["entry"] == "first strictly-later paired OANDA M1 observation"
    assert "actual CFTC publication timestamp" in causal["signal_available_at"]
    assert "first archived vintage" in causal["revisions"]


def test_b5a_all_source_supported_sleeves_and_spot_signs(
    document: dict[str, Any],
) -> None:
    family = document["families"][FAMILY_B5A]
    sleeves = {row["currency_k"]: row for row in family["sleeves"]}
    assert set(sleeves) == set(B5A_SLEEVES)
    assert sleeves["EUR"]["positive_delta_position_side"] == "BUY"
    assert sleeves["GBP"]["positive_delta_position_side"] == "BUY"
    assert sleeves["AUD"]["positive_delta_position_side"] == "BUY"
    assert sleeves["NZD"]["positive_delta_position_side"] == "BUY"
    assert sleeves["JPY"]["positive_delta_position_side"] == "SELL"
    assert sleeves["CHF"]["positive_delta_position_side"] == "SELL"
    assert sleeves["CAD"]["positive_delta_position_side"] == "SELL"
    for row in sleeves.values():
        assert (
            row["negative_delta_position_side"] != row["positive_delta_position_side"]
        )


def test_b5b_is_direct_chan_spec_and_missing_native_requirement(
    document: dict[str, Any],
) -> None:
    family = document["families"][FAMILY_B5B]
    rule = family["source_faithful_replication"]
    assert rule["instrument"] == "AUD/CAD.OANDA"
    assert rule["direct_instrument_required"] is True
    assert rule["synthetic_two_leg_substitution_permitted"] is False
    assert rule["lookback_completed_fx_days"] == 20
    assert rule["entry_threshold"].startswith("strictly nonzero")
    assert rule["maximum_holding_period"] is None
    assert rule["overlapping_position_policy"] == "one position only; never scale in"
    assert rule["parameter_variants"] == []

    audit = family["repository_availability_audit"]
    assert audit["present_in_OANDA_ALPHA_LAB_SPECS"] is False
    assert audit["present_in_oanda_fx_alpha_lab_v1_config"] is False
    assert audit["silent_synthesis_forbidden"] is True
    assert "native paired-M1 OANDA" in audit["required_before_future_development"]


def test_b5c_exact_universe_causal_event_and_no_filters(
    document: dict[str, Any],
) -> None:
    family = document["families"][FAMILY_B5C]
    rule = family["literature_anchored_rule"]
    assert rule["universe"] == list(B5C_INSTRUMENTS)
    assert rule["estimation_window"].startswith("30 immediately preceding")
    assert "2 * trailing_sample_std_30" in rule["positive_event"]
    assert rule["direction"] == "positive event SELL pair; negative event BUY pair"
    assert rule["holding_period"] == "one complete subsequent New York FX day"
    assert rule["entry"].startswith("first strictly-later paired OANDA M1")
    assert rule["parameter_variants"] == []
    assert set(family["forbidden_filters"]) == {
        "weekday",
        "session",
        "volatility_regime",
        "trend",
        "news",
        "spread_quantile",
    }
    conventions = family["implementation_conventions_not_literature_claims"]
    assert "does not uniquely report n" in conventions["thirty_day_window"]
    assert "Batch 4" in conventions["full_next_day_exit"]
    assert family["repository_availability_audit"]["missing_native"] == [
        "EUR/JPY.OANDA"
    ]


def test_common_and_frequency_specific_gates_are_frozen(
    document: dict[str, Any],
) -> None:
    assert COMMON_GATES["native_profit_factor"]["profit_factor_gt"] == 1.10
    folds = COMMON_GATES["chronological_stability"]
    assert folds["fold_count"] == 4
    assert folds["positive_net_return_folds_gte"] == 3
    assert COMMON_GATES["largest_winners"]["remaining_net_expectancy_gt"] == 0
    assert (
        COMMON_GATES["period_concentration"][
            "max_single_year_share_of_strictly_positive_profit_lte"
        ]
        == 0.40
    )
    assert COMMON_GATES["costs"]["stress_multipliers"] == [1.5, 2.0]
    assert (
        COMMON_GATES["drawdown"]["equal_weight_family_aggregate_maximum_drawdown_lte"]
        == 0.15
    )

    families = document["families"]
    assert families[FAMILY_B5A]["screening"] == {
        "independent_unit": "non-overlapping three-month cohort per sleeve",
        "minimum_monthly_formation_dates_per_sleeve": 36,
        "minimum_nonoverlapping_three_month_units_per_sleeve": 12,
        "minimum_distinct_calendar_years": 3,
    }
    assert (
        families[FAMILY_B5B]["screening"]["minimum_daily_holding_observations"] == 500
    )
    assert families[FAMILY_B5B]["screening"]["minimum_position_sign_changes"] == 20
    assert families[FAMILY_B5C]["screening"]["minimum_events_per_sleeve"] == 15
    assert families[FAMILY_B5C]["screening"]["minimum_events_family_total"] == 60


def test_breadth_and_deterministic_representatives(document: dict[str, Any]) -> None:
    breadth = document["breadth_rules"]
    assert breadth[FAMILY_B5A]["sleeves_positive_native_and_1_5x_expectancy_gte"] == 5
    assert breadth[FAMILY_B5A]["sleeves_passing_all_sleeve_gates_gte"] == 4
    assert breadth[FAMILY_B5B]["single_sleeve_must_pass_all_gates"] is True
    assert breadth[FAMILY_B5C]["sleeves_positive_native_and_1_5x_expectancy_gte"] == 3
    assert breadth[FAMILY_B5C]["sleeves_passing_all_sleeve_gates_gte"] == 2

    promotion = document["development_to_validation"]
    assert promotion["maximum_representatives_per_family"] == 1
    assert promotion["validation_in_this_task"] is False
    assert set(promotion["representative_selection"]) == {
        FAMILY_B5A,
        FAMILY_B5B,
        FAMILY_B5C,
        "tie_break_if_schema_error_creates_duplicate",
    }
    assert promotion["failure_action"].startswith("retire representative")
    assert len(promotion["rescue_forbidden"]) == 7


def test_relationships_and_diversification_are_explicit(
    document: dict[str, Any],
) -> None:
    relations = document["relationship_to_prior_research"]
    assert "not prices alone" in relations[FAMILY_B5A]["distinct"]
    assert "not U2" in relations[FAMILY_B5B]["distinct"]
    assert "not TSM" in relations[FAMILY_B5C]["distinct"]
    assert relations["empirical_correlations_used"] is False


def test_reuse_audit_names_existing_infrastructure(document: dict[str, Any]) -> None:
    audit = document["reuse_audit"]
    assert "DevelopmentResearchContext.require_range" in audit["partition_firewalls"]
    assert "oanda_alpha_lab_development" in audit["canonical_OANDA"]
    assert "bisect_right" in audit["native_bid_ask_and_first_later"]
    assert "widen_bid_ask_frame" in audit["transaction_cost_stress"]
    assert audit["parallel_implementation_created"] is False


def test_sources_are_primary_or_high_quality_and_family_complete(
    document: dict[str, Any],
) -> None:
    sources = document["source_provenance"]
    assert {source["family"] for source in sources} == set(PRIMARY_FAMILIES)
    assert any("Breaking Parity" in source["title"] for source in sources)
    assert any(source.get("doi") == "10.1108/JES-11-2019-0503" for source in sources)
    assert any(source.get("doi") == "10.1002/fut.20226" for source in sources)
    assert not any("blog" in source["type"].lower() for source in sources)


def test_firewall_and_lifecycle_are_entirely_closed(
    document: dict[str, Any],
) -> None:
    firewall = document["data_firewall"]
    assert firewall["development_prices_returns_pnl_or_performance_accessed"] is False
    assert firewall["validation_accessed"] is False
    assert firewall["final_holdout_accessed"] is False
    assert firewall["backtest_run"] is False
    assert firewall["signal_or_execution_implemented"] is False
    assert all(value is False for value in document["lifecycle"].values())


def test_checked_in_artifact_has_pinned_semantic_identity() -> None:
    document = verify_preregistration(PREREGISTRATION_PATH)
    assert document == build_preregistration()
    assert document["preregistration_semantic_sha256"] == _canonical_sha256(document)
    assert (
        document["preregistration_semantic_sha256"]
        == EXPECTED_PREREGISTRATION_SEMANTIC_SHA256
    )


def test_write_once_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "batch5.json"
    write_preregistration(path=path)
    with pytest.raises(Batch5PreregistrationError, match="refusing to overwrite"):
        write_preregistration(path=path)


def test_mutation_and_mutation_plus_rehash_are_rejected(tmp_path: Path) -> None:
    document = build_preregistration()
    document["stage"] = "MUTATED"
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(Batch5PreregistrationError, match="semantic_sha256"):
        verify_preregistration(path)

    document["preregistration_semantic_sha256"] = _canonical_sha256(document)
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(Batch5PreregistrationError, match="identity mismatch"):
        verify_preregistration(path)


def test_module_has_no_data_result_partition_or_strategy_imports() -> None:
    source_path = Path("src/ftmoquant/research/alpha_lab/batch5_preregistration.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not any(
        fragment in imported
        for imported in imports
        for fragment in (
            "ftmoquant.data",
            "stage_g",
            "validation",
            "holdout",
            "strategies",
            "execution",
            "pandas",
            "numpy",
        )
    )
