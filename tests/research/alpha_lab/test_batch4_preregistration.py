from __future__ import annotations

import ast
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from ftmoquant.research.alpha_lab.batch4_preregistration import (
    DEVELOPMENT_DIAGNOSTICS,
    DEVELOPMENT_GATES,
    FIX_DURATIONS_MINUTES,
    PRIMARY_FAMILIES,
    UNIVERSE,
    Batch4PreregistrationError,
    _canonical_sha256,
    build_preregistration,
    local_window_utc,
    pair_side_for_currency_move,
    verify_preregistration,
    write_preregistration,
)

_NOW = datetime(2026, 8, 24, 12, tzinfo=UTC)


@pytest.fixture
def document() -> dict[str, object]:
    return build_preregistration(created_at_utc=_NOW)


def test_exact_primary_and_deferred_family_scope(document: dict[str, object]) -> None:
    scope = document["family_scope"]
    assert isinstance(scope, dict)
    assert scope["primary_exact"] == list(PRIMARY_FAMILIES)
    assert scope["deferred_not_implemented"] == [
        "B4F2_cross_currency_lead_lag_propagation"
    ]
    assert scope["no_extra_families"] is True


def test_exact_universe(document: dict[str, object]) -> None:
    assert document["universe"] == list(UNIVERSE)
    assert len(UNIVERSE) == 7


def test_exact_local_windows_and_frozen_directions(
    document: dict[str, object],
) -> None:
    families = document["families"]
    assert isinstance(families, dict)
    family = families[PRIMARY_FAMILIES[0]]
    sleeves = family["sleeves"]
    observed = {
        row["currency_tested"]: (
            row["pair"],
            row["timezone"],
            row["start_local"],
            row["end_local"],
            row["directional_hypothesis"],
            row["entry_side"],
        )
        for row in sleeves
    }
    assert observed == {
        "AUD": (
            "AUD/USD.OANDA",
            "Australia/Sydney",
            "09:00",
            "17:00",
            "AUD_DEPRECIATES",
            "SELL",
        ),
        "EUR": (
            "EUR/USD.OANDA",
            "Europe/Berlin",
            "08:00",
            "16:00",
            "EUR_DEPRECIATES",
            "SELL",
        ),
        "GBP": (
            "GBP/USD.OANDA",
            "Europe/London",
            "08:00",
            "16:00",
            "GBP_DEPRECIATES",
            "SELL",
        ),
        "NZD": (
            "NZD/USD.OANDA",
            "Pacific/Auckland",
            "09:00",
            "17:00",
            "NZD_DEPRECIATES",
            "SELL",
        ),
        "CAD": (
            "USD/CAD.OANDA",
            "America/Toronto",
            "08:00",
            "16:00",
            "CAD_DEPRECIATES",
            "BUY",
        ),
        "CHF": (
            "USD/CHF.OANDA",
            "Europe/Zurich",
            "08:00",
            "16:00",
            "CHF_DEPRECIATES",
            "BUY",
        ),
        "JPY": (
            "USD/JPY.OANDA",
            "Asia/Tokyo",
            "09:00",
            "17:00",
            "JPY_DEPRECIATES",
            "BUY",
        ),
    }
    assert family["parameter_axis"] is None


def test_fix_windows_direction_and_no_arbitrary_offset(
    document: dict[str, object],
) -> None:
    families = document["families"]
    assert isinstance(families, dict)
    expected = {
        "PRE_15m": ("15:45", "16:00", "USD_APPRECIATES"),
        "PRE_30m": ("15:30", "16:00", "USD_APPRECIATES"),
        "PRE_60m": ("15:00", "16:00", "USD_APPRECIATES"),
        "POST_15m": ("16:00", "16:15", "USD_DEPRECIATES"),
        "POST_30m": ("16:00", "16:30", "USD_DEPRECIATES"),
        "POST_60m": ("16:00", "17:00", "USD_DEPRECIATES"),
    }
    london = families[PRIMARY_FAMILIES[1]]
    assert london["fix_timezone"] == "Europe/London"
    assert london["fix_local"] == "16:00"
    assert {
        row["configuration_id"]: (
            row["start_local"],
            row["end_local"],
            row["directional_hypothesis"],
        )
        for row in london["configurations"]
    } == expected

    tokyo = families[PRIMARY_FAMILIES[2]]
    assert tokyo["fix_timezone"] == "Asia/Tokyo"
    assert tokyo["fix_local"] == "09:55"
    assert tokyo["material_definition_conflict_found"] is False
    assert {
        row["configuration_id"]: (row["start_local"], row["end_local"])
        for row in tokyo["configurations"]
    } == {
        "PRE_15m": ("09:40", "09:55"),
        "PRE_30m": ("09:25", "09:55"),
        "PRE_60m": ("08:55", "09:55"),
        "POST_15m": ("09:55", "10:10"),
        "POST_30m": ("09:55", "10:25"),
        "POST_60m": ("09:55", "10:55"),
    }
    assert FIX_DURATIONS_MINUTES == (15, 30, 60)


def test_fix_pair_sides_freeze_usd_sign(document: dict[str, object]) -> None:
    families = document["families"]
    assert isinstance(families, dict)
    configs = families[PRIMARY_FAMILIES[1]]["configurations"]
    pre = configs[0]["pair_entry_sides"]
    post = configs[3]["pair_entry_sides"]
    assert pre == {
        "AUD/USD.OANDA": "SELL",
        "EUR/USD.OANDA": "SELL",
        "GBP/USD.OANDA": "SELL",
        "NZD/USD.OANDA": "SELL",
        "USD/CAD.OANDA": "BUY",
        "USD/CHF.OANDA": "BUY",
        "USD/JPY.OANDA": "BUY",
    }
    assert post == {
        pair: ("BUY" if side == "SELL" else "SELL") for pair, side in pre.items()
    }


@pytest.mark.parametrize(
    ("pair", "currency", "move", "side"),
    [
        ("EUR/USD.OANDA", "EUR", "APPRECIATES", "BUY"),
        ("EUR/USD.OANDA", "USD", "APPRECIATES", "SELL"),
        ("USD/JPY.OANDA", "JPY", "APPRECIATES", "SELL"),
        ("USD/JPY.OANDA", "USD", "APPRECIATES", "BUY"),
        ("USD/CAD.OANDA", "CAD", "DEPRECIATES", "BUY"),
    ],
)
def test_currency_pair_sign_mapping(
    pair: str, currency: str, move: str, side: str
) -> None:
    assert pair_side_for_currency_move(pair, currency, move) == side  # type: ignore[arg-type]


def test_currency_pair_sign_mapping_fails_closed() -> None:
    with pytest.raises(Batch4PreregistrationError):
        pair_side_for_currency_move("EUR/JPY.OANDA", "JPY", "APPRECIATES")
    with pytest.raises(Batch4PreregistrationError):
        pair_side_for_currency_move("EUR/USD.OANDA", "CAD", "APPRECIATES")


def test_timezone_windows_are_dst_safe_and_tokyo_is_stable() -> None:
    london_winter = local_window_utc(
        date(2026, 1, 15), "Europe/London", "08:00", "16:00"
    )
    london_summer = local_window_utc(
        date(2026, 7, 15), "Europe/London", "08:00", "16:00"
    )
    assert london_winter[0].hour == 8
    assert london_summer[0].hour == 7
    tokyo_winter = local_window_utc(date(2026, 1, 15), "Asia/Tokyo", "09:55", "10:10")
    tokyo_summer = local_window_utc(date(2026, 7, 15), "Asia/Tokyo", "09:55", "10:10")
    assert tokyo_winter[0].hour == tokyo_summer[0].hour == 0
    assert tokyo_winter[0].minute == tokyo_summer[0].minute == 55


def test_exact_grid_counts(document: dict[str, object]) -> None:
    assert document["grid_accounting"] == {
        "local_hours_sleeve_window_cells": 7,
        "london_fix_timing_direction_cells_before_pairs": 6,
        "tokyo_fix_timing_direction_cells_before_pairs": 6,
        "total_primary_cells_before_fix_pair_multiplication": 19,
        "london_executable_pair_cells": 42,
        "tokyo_executable_pair_cells": 42,
        "total_executable_sleeve_configuration_hypotheses": 91,
        "unique_primary_families": 3,
    }


def test_hard_gates_are_exact() -> None:
    assert DEVELOPMENT_GATES["A_opportunity_density"]["completed_trades_gte"] == 250
    assert DEVELOPMENT_GATES["B_native_expectancy"]["expectancy_usd_per_trade_gt"] == 0
    assert DEVELOPMENT_GATES["C_native_profit_factor"]["profit_factor_gt"] == 1.10
    assert (
        DEVELOPMENT_GATES["D_temporal_stability"]["positive_net_return_folds_gte"] == 3
    )
    assert (
        DEVELOPMENT_GATES["D_temporal_stability"][
            "chronological_development_fold_count"
        ]
        == 4
    )
    assert (
        DEVELOPMENT_GATES["E_exceptional_winner_dependence"][
            "remaining_expectancy_usd_per_trade_gt"
        ]
        == 0
    )
    assert "ceil(5%)" in DEVELOPMENT_GATES["E_exceptional_winner_dependence"]["remove"]
    assert (
        DEVELOPMENT_GATES["F_quarter_concentration"]["max_single_quarter_share_lte"]
        == 0.40
    )
    assert DEVELOPMENT_GATES["G_cost_stress"]["required_multipliers"] == [1.5, 2.0]
    assert (
        DEVELOPMENT_GATES["H_parameter_neighborhood"]["min_connected_passing_region"]
        == 2
    )
    assert "not_applicable" in DEVELOPMENT_GATES["H_parameter_neighborhood"]["B4F1A"]


def test_diagnostics_are_report_only_and_complete() -> None:
    assert (
        DEVELOPMENT_DIAGNOSTICS["status"]
        == "report_only_never_gate_rank_filter_or_rescue"
    )
    assert DEVELOPMENT_DIAGNOSTICS["rolling_expectancy_trade_windows"] == [50, 100]
    fields = DEVELOPMENT_DIAGNOSTICS["other_report_fields"]
    assert "spread_cost_as_percent_of_gross_edge" in fields
    assert "weekday_breakdown" in fields
    assert "stop_target_time_exit_fractions_where_applicable" in fields


def test_breadth_cost_validation_and_no_rescue_are_exact(
    document: dict[str, object],
) -> None:
    breadth = document["family_breadth_gate"]
    assert breadth["sleeves_meeting_breadth_metrics_gte"] == 3
    assert breadth["sleeves_passing_full_hard_gate_set_gte"] == 2
    assert breadth["breadth_metrics_all_required"] == {
        "native_expectancy_usd_per_trade_gt": 0,
        "native_profit_factor_gt": 1.0,
        "stress_1_5x_expectancy_usd_per_trade_gt": 0,
    }
    assert document["cost_stress"]["multipliers"] == [1.5, 2.0]
    policy = document["development_to_validation"]
    assert policy["number_proceeding"] == 1
    assert policy["one_shot_validation_gates_all_required"] == {
        "native_net_return_gt": 0,
        "native_annualized_sharpe_gt": 0,
        "stress_1_5x_expectancy_usd_per_trade_gt": 0,
    }
    assert policy["after_failure"] == "retire_no_rescue"
    assert all(
        document["multiple_testing_and_no_rescue"][key] is True
        for key in (
            "no_additional_windows",
            "no_sign_inversion",
            "no_pair_cherry_picking",
            "no_nearby_time_rescue",
            "no_weekday_rescue",
            "no_volatility_regime_rescue",
        )
    )


def test_build_hash_is_deterministic_and_tamper_is_rejected(tmp_path: Path) -> None:
    first = build_preregistration(created_at_utc=_NOW)
    second = build_preregistration(created_at_utc=_NOW)
    assert first == second
    assert first["preregistration_semantic_sha256"] == _canonical_sha256(first)
    path = tmp_path / "batch4.json"
    write_preregistration(path=path, created_at_utc=_NOW)
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["grid_accounting"]["total_executable_sleeve_configuration_hypotheses"] = 92
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(Batch4PreregistrationError, match="semantic_sha256"):
        verify_preregistration(path)


def test_writer_refuses_overwrite_and_frozen_artifact_verifies(tmp_path: Path) -> None:
    path = tmp_path / "batch4.json"
    write_preregistration(path=path, created_at_utc=_NOW)
    assert verify_preregistration(path)["stage"] == "B4.0"
    with pytest.raises(Batch4PreregistrationError, match="refusing to overwrite"):
        write_preregistration(path=path, created_at_utc=_NOW)


def test_module_import_graph_and_source_have_no_data_access_or_signal_code() -> None:
    import ftmoquant.research.alpha_lab.batch4_preregistration as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = {
        alias.name.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots |= {
        (node.module or "").split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert imported_roots <= {
        "__future__",
        "copy",
        "datetime",
        "hashlib",
        "json",
        "pathlib",
        "typing",
        "zoneinfo",
    }
    for token in (
        "ParquetDataCatalog",
        "load_alpha_lab_dataset",
        "open_development_context",
        "oanda_alpha_lab_development",
        "oanda_alpha_lab_validation",
        "run_backtest",
        "widen_bid_ask_frame(",
    ):
        assert token not in source


def test_artifact_lifecycle_and_deferred_family_are_nonimplemented(
    document: dict[str, object],
) -> None:
    assert document["lifecycle"] == {
        "development_accessed": False,
        "validation_accessed": False,
        "holdout_accessed": False,
        "signals_implemented": False,
        "execution_implemented": False,
    }
    deferred = document["deferred_B4F2"]
    assert deferred["status"] == "recorded_only_not_part_of_B4F1_not_implemented"
    assert "no contemporaneous leakage" in deferred["future_requirements"]


def test_frozen_artifact_on_disk_is_self_consistent() -> None:
    path = Path(
        "config/research/batch4_structural_intraday_flow_preregistration_v1.json"
    )
    assert verify_preregistration(path)["preregistration_version"].endswith("-v1")
