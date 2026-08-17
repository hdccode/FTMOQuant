"""Strict preregistration loader for ``eurusd_policy_rate_carry_proxy_v1``.

Mirrors the structural pattern of
:mod:`ftmoquant.research.eurusd_session_range_expansion_spec`: every block of
the frozen YAML document is validated with exact-key-set assertions and
exact-value assertions against hardcoded expected values, and the recorded
``semantic_sha256`` is cross-checked against both the canonical document
itself and a frozen module constant. Any drift -- in the YAML or in this
loader -- fails closed.

This family is baseline-only: exactly one configuration, no parameter grid,
no selector, no plateau/neighbour analysis. The schema therefore has no
``parameter_grid``/``neighbours``/``selector`` blocks; it has a
``parameter_family`` block declaring ``mode: baseline_only`` instead.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import yaml

from ftmoquant.data.policy_rates import ECBDFR_RAW_SHA256, EFFR_RAW_SHA256
from ftmoquant.research.stage_g import (
    DEVELOPMENT_END_EXCLUSIVE,
    DEVELOPMENT_START,
    FROZEN_UNIVERSE_ID,
    FROZEN_UNIVERSE_PLAN_SHA256,
    FROZEN_UNIVERSE_READINESS_SHA256,
    frozen_development_folds,
)

EURUSD_POLICY_RATE_CARRY_PROXY_SPEC_PATH = Path(
    "config/strategies/eurusd_policy_rate_carry_proxy_v1.yaml"
)
EURUSD_POLICY_RATE_CARRY_PROXY_SEMANTIC_SHA256 = (
    "b6b21d83e71e06c371645ea3dce33178e8168645c11d89c3f5ec63971c0023f9"
)
# Prior semantic SHA (superseded, never a real-returns-bearing version):
# "d200492d6c210e2b0968f8a9731fea00a6a2f273401dbce63011cf5d0cd14eae". Changed
# solely to add the frozen `execution_and_costs.pending_target_policy`
# (latest_causal_target_supersedes_unexecuted_older_target) resolving the
# market-closure instruction-collision bug in
# build_carry_proxy_instructions -- an implementation/scheduling fix, not a
# change to sign(ECBDFR - EFFR), the 00:00 UTC decision anchor, the
# one-business-day rate lag, 1% causal volatility sizing, proxy carry
# inputs, monthly rollover semantics, daily evidence semantics, or
# DEVELOPMENT gates. No real DEVELOPMENT/validation/final-holdout returns
# were ever observed under the prior SHA: the only real-data run under it
# raised EurusdPolicyRateCarryProxyEvaluationError inside
# build_carry_proxy_instructions, strictly before _run_native_fold, so no
# strategy P&L was produced and no output artifact exists.

_EXPECTED_TOP_KEYS = {
    "schema_version",
    "family",
    "dataset",
    "signal",
    "risk_normalization",
    "proxy_carry",
    "execution_and_costs",
    "development_folds",
    "pnl_decomposition",
    "sample_count",
    "eligibility",
    "production_evaluator",
    "metric_definitions",
    "parameter_family",
    "search",
    "sealed_partitions",
    "lineage",
    "limitations",
    "reuse",
    "semantic_sha256",
}


class EurusdPolicyRateCarryProxySpecError(ValueError):
    """Raised when the preregistration is ambiguous, incomplete, or changed."""


@dataclass(frozen=True, slots=True)
class EurusdPolicyRateCarryProxySpec:
    family_id: str
    version: str
    economic_hypothesis: str
    semantic_sha256: str
    canonical_document: dict[str, Any]


def load_eurusd_policy_rate_carry_proxy_spec(
    path: Path = EURUSD_POLICY_RATE_CARRY_PROXY_SPEC_PATH,
) -> EurusdPolicyRateCarryProxySpec:
    """Load and prove the exact pre-return baseline-only family protocol."""

    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise EurusdPolicyRateCarryProxySpecError(
            f"could not load spec: {error}"
        ) from error
    document = _mapping(value, "spec")
    _exact_keys(document, _EXPECTED_TOP_KEYS, "spec")

    semantic = semantic_sha256_for_document(document)
    if document["semantic_sha256"] != semantic:
        raise EurusdPolicyRateCarryProxySpecError(
            "semantic_sha256 does not match canonical semantics"
        )
    if semantic != EURUSD_POLICY_RATE_CARRY_PROXY_SEMANTIC_SHA256:
        raise EurusdPolicyRateCarryProxySpecError(
            "eurusd_policy_rate_carry_proxy_v1 preregistration semantic SHA drifted"
        )

    family = _mapping(document["family"], "family")
    _exact_keys(
        family, {"family_id", "version", "status", "economic_hypothesis"}, "family"
    )
    if (
        document["schema_version"] != 1
        or family["family_id"] != "eurusd_policy_rate_carry_proxy_v1"
        or family["version"] != "1.0.0"
        or family["status"] != "preregistered_not_run"
    ):
        raise EurusdPolicyRateCarryProxySpecError(
            "family identity/status is not frozen"
        )
    hypothesis = _string(family["economic_hypothesis"], "economic_hypothesis")

    _validate_dataset(document)
    _validate_signal(document)
    _validate_risk_normalization(document)
    _validate_proxy_carry(document)
    _validate_execution_and_costs(document)
    _validate_folds(document)
    _validate_pnl_decomposition(document)
    _validate_sample_count(document)
    _validate_eligibility(document)
    _validate_production_evaluator(document)
    _validate_metric_definitions(document)
    _validate_parameter_family(document)
    _validate_search(document)
    _validate_sealed_partitions(document)
    _validate_lineage(document)
    _validate_limitations(document)
    _validate_reuse(document)

    return EurusdPolicyRateCarryProxySpec(
        family_id="eurusd_policy_rate_carry_proxy_v1",
        version="1.0.0",
        economic_hypothesis=hypothesis,
        semantic_sha256=semantic,
        canonical_document=document,
    )


def semantic_sha256_for_document(document: dict[str, Any]) -> str:
    """Hash all semantics except the self-referential recorded digest."""

    payload = {
        key: value for key, value in document.items() if key != "semantic_sha256"
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_dataset(document: dict[str, Any]) -> None:
    dataset = _mapping(document["dataset"], "dataset")
    _exact_keys(
        dataset,
        {
            "instrument_id",
            "universe_id",
            "universe_plan_sha256",
            "universe_readiness_sha256",
            "partition",
            "development_start_utc",
            "development_end_exclusive_utc",
            "rate_source",
        },
        "dataset",
    )
    if (
        dataset["instrument_id"] != "EUR/USD.DUKASCOPY"
        or dataset["universe_id"] != FROZEN_UNIVERSE_ID
        or dataset["universe_plan_sha256"] != FROZEN_UNIVERSE_PLAN_SHA256
        or dataset["universe_readiness_sha256"] != FROZEN_UNIVERSE_READINESS_SHA256
        or dataset["partition"] != "development"
        or dataset["development_start_utc"] != _utc(DEVELOPMENT_START)
        or dataset["development_end_exclusive_utc"] != _utc(DEVELOPMENT_END_EXCLUSIVE)
    ):
        raise EurusdPolicyRateCarryProxySpecError(
            "dataset identity or DEVELOPMENT boundary drifted"
        )
    rate_source = _mapping(dataset["rate_source"], "dataset.rate_source")
    _exact_keys(
        rate_source,
        {
            "ecbdfr_raw_file",
            "ecbdfr_raw_sha256",
            "effr_raw_file",
            "effr_raw_sha256",
            "provider",
            "provenance_path",
            "loader",
            "information_lag",
            "development_window_differential_sign_never_flips",
        },
        "dataset.rate_source",
    )
    if (
        rate_source["ecbdfr_raw_file"] != "config/data/policy_rates/ECBDFR_raw.csv"
        or rate_source["ecbdfr_raw_sha256"] != ECBDFR_RAW_SHA256
        or rate_source["effr_raw_file"] != "config/data/policy_rates/EFFR_raw.csv"
        or rate_source["effr_raw_sha256"] != EFFR_RAW_SHA256
        or rate_source["provider"] != "FRED"
        or rate_source["provenance_path"] != "config/data/policy_rates/provenance.json"
        or rate_source["loader"]
        != "ftmoquant.data.policy_rates.load_policy_rate_history"
        or rate_source["information_lag"] != "one_business_day_forward_filled"
        or rate_source["development_window_differential_sign_never_flips"] is not True
    ):
        raise EurusdPolicyRateCarryProxySpecError("rate source provenance drifted")


def _validate_signal(document: dict[str, Any]) -> None:
    signal = _mapping(document["signal"], "signal")
    if (
        signal.get("implementation")
        != (
            "ftmoquant.strategies.eurusd_policy_rate_carry_proxy."
            "EurusdPolicyRateCarryProxyState"
        )
        or signal.get("rule") != "sign_of_causal_ecbdfr_minus_effr_differential"
        or signal.get("positive_differential_target") != 1
        or signal.get("negative_differential_target") != -1
        or signal.get("zero_differential_target") != 0
        or signal.get("parameters_or_lookback") != "none"
        or signal.get("decision_anchor_utc_time") != "00:00"
        or signal.get("decision_anchor_rationale")
        != "avoid_colliding_with_17_00_america_new_york_rollover_accrual_window"
        or signal.get("signal_source")
        != "ftmoquant.data.policy_rates.causal_differential_series"
    ):
        raise EurusdPolicyRateCarryProxySpecError("signal semantics drifted")


def _validate_risk_normalization(document: dict[str, Any]) -> None:
    risk = _mapping(document["risk_normalization"], "risk_normalization")
    if risk != {
        "type": "causal_pre_execution_underlying_vol_target",
        "target_annualized_volatility": 0.01,
        "implementation": "ftmoquant.research.g1.normalization.G1VolatilityNormalizer",
        "underlying_instrument": "EUR/USD.DUKASCOPY",
        "source": "completed_bid_ask_midpoint_daily_log_returns",
        "daily_observation_semantics": (
            "existing_17_00_America_New_York_completed_pair"
        ),
        "estimator": {
            "library_primitive": "pandas.Series.ewm.var",
            "kind": "exponentially_weighted_variance",
            "center_of_mass_trading_days": 60.0,
            "adjust": True,
            "bias": False,
            "ignore_na": False,
            "minimum_completed_returns": 20,
            "annualization_days": 252,
        },
        "observation_rule": (
            "daily_return_endpoint_information_time_strictly_before_decision"
        ),
        "warmup_behavior": "unavailable_zero_nonzero_exposure",
        "future_backfill": "forbidden",
        "pathological_volatility_behavior": "fail_closed_zero_nonzero_exposure",
        "exposure_formula": (
            "directional_signal_times_0.01_divided_by_ex_ante_annualized_volatility"
        ),
        "sizing_refresh_rule": "resized_daily_even_if_direction_unchanged",
        "base_units_per_unit_exposure": "100000",
        "target_quantity_rule": (
            "desired_exposure_times_base_units_before_native_execution"
        ),
        "ftmo_or_g4_sizing": False,
    }:
        raise EurusdPolicyRateCarryProxySpecError("risk normalization drifted")


def _validate_proxy_carry(document: dict[str, Any]) -> None:
    carry = _mapping(document["proxy_carry"], "proxy_carry")
    if (
        carry.get("native_module")
        != "nautilus_trader.backtest.FXRolloverInterestModule"
        or carry.get("wiring") != "ftmoquant.backtest.execution_harness.RolloverConfig"
        or carry.get("mode") != "fx_rollover_interest"
        or carry.get("calibration_status") != "proxy"
        or carry.get("native_resolution") != "monthly"
        or carry.get("bucketing_rule")
        != "last_business_day_of_prior_month_causal_value_per_month"
        or carry.get("record_construction")
        != (
            "ftmoquant.research.eurusd_policy_rate_carry_proxy_development."
            "build_monthly_interest_rate_inputs"
        )
        or carry.get("record_inputs")
        != "raw_ecbdfr_and_effr_rates_not_precomputed_differential"
        or carry.get("sign_convention")
        != "module_computed_internally_from_raw_records_no_second_calculation"
        or carry.get("accrual_schedule")
        != "existing_frozen_17_00_america_new_york_weekday_trigger_wed_fri_tripled"
        or carry.get("effective_lag_min_business_days") != 1
        or carry.get("effective_lag_max_calendar_days_approx") != 31
        or not _string(carry.get("lag_disclosure"), "proxy_carry.lag_disclosure")
    ):
        raise EurusdPolicyRateCarryProxySpecError("proxy carry semantics drifted")

    verified_against = _mapping(
        carry["verified_against"], "proxy_carry.verified_against"
    )
    _exact_keys(
        verified_against,
        {
            "package",
            "source_commit",
            "source_file",
            "verified_native_key_formats",
            "daily_records_supported",
        },
        "proxy_carry.verified_against",
    )
    if (
        verified_against.get("package") != "nautilus-trader==2.0.0rc2"
        or verified_against.get("source_commit")
        != "cad455e35c4489a3314e6fc9bad5c7b5c103ace8"
        or verified_against.get("source_file")
        != "crates/backtest/src/modules/fx_rollover.rs"
        or verified_against.get("verified_native_key_formats") != ["YYYY-MM", "YYYY-Qn"]
        or verified_against.get("daily_records_supported") is not False
    ):
        raise EurusdPolicyRateCarryProxySpecError(
            "proxy carry rc2 verification citation drifted"
        )


def _validate_execution_and_costs(document: dict[str, Any]) -> None:
    execution = _mapping(document["execution_and_costs"], "execution_and_costs")
    if (
        execution.get("engine") != "existing_nautilus_g0_7_bid_ask_boundary"
        or execution.get("entry_timing_rule")
        != "first_eligible_information_time_strictly_after_signal_information_time"
        or execution.get("decision_timing_rule") != "daily_00_00_utc_decision_anchor"
        or execution.get("max_open_positions") != 1
        or execution.get("stop_loss") != "none"
        or execution.get("take_profit") != "none"
        or execution.get("spread") != "observed_paired_bid_ask"
        or execution.get("cost_stress_multiplier") != 1.5
        or execution.get("cost_stress_scope")
        != "variable_execution_cost_component_only_never_the_carry_accrual"
        or execution.get("position_sizing_timing") != "before_native_order_submission"
        or execution.get("native_cost_quantity_basis")
        != "actual_executed_scaled_delta_base_units"
        or execution.get("parallel_backtester") is not False
        or execution.get("ftmo_optimization") is not False
        or execution.get("pending_target_policy")
        != "latest_causal_target_supersedes_unexecuted_older_target"
        or not _string(
            execution.get("pending_target_policy_scope"), "pending_target_policy_scope"
        )
    ):
        raise EurusdPolicyRateCarryProxySpecError(
            "execution/cost semantics drifted"
        )


def _validate_folds(document: dict[str, Any]) -> None:
    declared = _mapping(document["development_folds"], "development_folds")
    folds = frozen_development_folds()
    if (
        declared.get("version") != folds.version
        or declared.get("semantic_sha256") != folds.semantic_sha256
    ):
        raise EurusdPolicyRateCarryProxySpecError("DEVELOPMENT fold identity drifted")
    expected = [
        {
            "fold_id": fold.fold_id,
            "train_start_utc": _utc(fold.train_start_utc),
            "train_end_exclusive_utc": _utc(fold.train_end_exclusive_utc),
            "evaluate_start_utc": _utc(fold.compare_start_utc),
            "evaluate_end_exclusive_utc": _utc(fold.compare_end_exclusive_utc),
        }
        for fold in folds.folds
    ]
    if declared.get("folds") != expected:
        raise EurusdPolicyRateCarryProxySpecError("declared fold timestamps drifted")


def _validate_pnl_decomposition(document: dict[str, Any]) -> None:
    decomposition = _mapping(document["pnl_decomposition"], "pnl_decomposition")
    if (
        decomposition.get("components")
        != ["spot_pnl", "execution_cost", "proxy_carry_accrual"]
        or decomposition.get("identity")
        != (
            "total_daily_equity_change_equals_spot_pnl_plus_carry_accrual_"
            "minus_execution_cost"
        )
        or decomposition.get("daily_mark_boundary")
        != "existing_17_00_america_new_york_completed_pair"
        or decomposition.get("bracket_marks")
        != ["16:59:59_America/New_York", "17:00:05_America/New_York"]
        or decomposition.get("fold_reset")
        != "pre_fold_equity_and_cost_excluded_from_fold_metrics"
    ):
        raise EurusdPolicyRateCarryProxySpecError("P&L decomposition semantics drifted")


def _validate_sample_count(document: dict[str, Any]) -> None:
    sample = _mapping(document["sample_count"], "sample_count")
    if (
        sample.get("fold_metrics_trade_count_field_repurposed_as")
        != "eligible_daily_return_observation_count"
        or sample.get("definition")
        != (
            "one_eligible_causal_trading_day_marked_total_return_equals_"
            "one_evidential_observation"
        )
        or sample.get("is_not_a_trade_count") is not True
        or sample.get("count_orders") is not False
        or sample.get("count_fills") is not False
        or sample.get("count_sign_changes") is not False
        or sample.get("count_rebalances_only") is not False
    ):
        raise EurusdPolicyRateCarryProxySpecError("sample count semantics drifted")


def _validate_eligibility(document: dict[str, Any]) -> None:
    eligibility = _mapping(document["eligibility"], "eligibility")
    if (
        eligibility.get("required_fold_count") != 3
        or eligibility.get("minimum_pooled_eligible_observations") != 500
        or eligibility.get("minimum_positive_folds") != 2
        or eligibility.get("gates")
        != [
            "status_is_complete",
            "all_required_folds_present",
            "minimum_pooled_eligible_observations_met",
            "pooled_mean_daily_total_return_gt_zero",
            "pooled_cost_stressed_mean_daily_total_return_gt_zero",
            "strict_majority_of_fold_mean_returns_positive",
            "all_numerical_validity_checks_pass",
        ]
        or eligibility.get("maximum_drawdown_hard_gate") is not False
        or eligibility.get("plateau_hard_gate") is not False
        or eligibility.get("year_concentration_hard_gate") is not False
    ):
        raise EurusdPolicyRateCarryProxySpecError("eligibility gates drifted")


def _validate_production_evaluator(document: dict[str, Any]) -> None:
    evaluator = _mapping(document["production_evaluator"], "production_evaluator")
    if (
        evaluator.get("version")
        != "eurusd-policy-rate-carry-proxy-v1-development-evaluator-1"
        or evaluator.get("module")
        != "ftmoquant.research.eurusd_policy_rate_carry_proxy_development"
        or evaluator.get("runner")
        != (
            "ftmoquant.research.eurusd_policy_rate_carry_proxy_development."
            "run_eurusd_policy_rate_carry_proxy_development"
        )
        or evaluator.get("command")
        != (
            "uv run python -m "
            "ftmoquant.research.eurusd_policy_rate_carry_proxy_development"
        )
        or evaluator.get("fold_warmup")
        != "process_train_start_through_evaluation_without_pre_fold_pnl_in_metrics"
        or evaluator.get("end_boundary")
        != "final_available_tradable_observation_strictly_before_evaluate_end"
        or evaluator.get("read_beyond_evaluate_end") is not False
    ):
        raise EurusdPolicyRateCarryProxySpecError("production evaluator drifted")


def _validate_metric_definitions(document: dict[str, Any]) -> None:
    metrics = _mapping(document["metric_definitions"], "metric_definitions")
    if metrics != {
        "trade_count": "eligible_daily_return_observation_count",
        "net_expectancy": "pooled_mean_eligible_daily_total_return",
        "cost_stressed_expectancy": (
            "pooled_mean_eligible_daily_total_return_after_1_5x_execution_cost_stress"
        ),
        "net_daily_mean": "arithmetic_mean_of_eligible_daily_total_returns",
        "sharpe": "existing_ftmoquant_annualized_daily_sharpe_sqrt_252",
        "drawdown": "existing_ftmoquant_compounded_daily_return_maximum_drawdown",
        "yearly_attribution": "calendar_year_net_return_contribution",
        "rate_regime_attribution": "differential_sign_epoch_net_return_contribution",
        "spot_pnl_contribution": "pooled_spot_pnl_component_of_total_return",
        "carry_accrual_contribution": "pooled_carry_accrual_component_of_total_return",
        "execution_cost_contribution": (
            "pooled_execution_cost_component_of_total_return"
        ),
        "dependence_aware_confidence_interval": (
            "diagnostic_stationary_bootstrap_not_hard_gated"
        ),
    }:
        raise EurusdPolicyRateCarryProxySpecError("metric definitions drifted")


def _validate_parameter_family(document: dict[str, Any]) -> None:
    family = _mapping(document["parameter_family"], "parameter_family")
    if (
        family.get("mode") != "baseline_only"
        or family.get("parameter_count") != 0
        or not _string(family.get("design_note"), "parameter_family.design_note")
    ):
        raise EurusdPolicyRateCarryProxySpecError("parameter family drifted")


def _validate_search(document: dict[str, Any]) -> None:
    search = _mapping(document["search"], "search")
    if search != {
        "mode": "baseline_only",
        "optuna": False,
        "parameter_grid_frozen_before_returns": True,
        "select_candidate_invoked": False,
        "assess_plateau_invoked": False,
    }:
        raise EurusdPolicyRateCarryProxySpecError("search declaration drifted")


def _validate_sealed_partitions(document: dict[str, Any]) -> None:
    seals = _mapping(document["sealed_partitions"], "sealed_partitions")
    if (
        seals.get("validation")
        != "locked_until_[2023-04-11T00:00:00Z,_2024-08-21T00:00:00Z)"
        or seals.get("final_holdout") != "locked"
        or seals.get("strategy_returns_accessed") is not False
    ):
        raise EurusdPolicyRateCarryProxySpecError("sealed partitions drifted")


def _validate_lineage(document: dict[str, Any]) -> None:
    lineage = _mapping(document["lineage"], "lineage")
    if (
        lineage.get("prior_families_all_development_burned") is not True
        or lineage.get("lineage_inventory_path")
        != "docs/research/strategy_lineage_inventory.md"
        or lineage.get("new_information_source") is not True
        or lineage.get("new_information_source_description")
        != "policy_rate_differential_never_previously_used_as_alpha_signal"
        or lineage.get("prior_failed_hypotheses_reused_at")
        != "mechanism_category_level_only"
        or lineage.get("prior_parameters_reused") is not False
    ):
        raise EurusdPolicyRateCarryProxySpecError("lineage statement drifted")


def _validate_limitations(document: dict[str, Any]) -> None:
    limitations = _mapping(document["limitations"], "limitations")
    if (
        limitations.get("does_not_establish_broad_cross_sectional_fx_carry_factor")
        is not True
        or limitations.get("does_not_establish_actual_ftmo_broker_swap_replication")
        is not True
        or limitations.get("rollover_module_native_resolution") != "monthly"
    ):
        raise EurusdPolicyRateCarryProxySpecError("limitations statement drifted")


def _validate_reuse(document: dict[str, Any]) -> None:
    reuse = _mapping(document["reuse"], "reuse")
    if reuse != {
        "raw_target_enum": "ftmoquant.strategies.ts_momentum.RawDirectionalTarget",
        "volatility_normalizer": (
            "ftmoquant.research.g1.normalization.G1VolatilityNormalizer"
        ),
        "rollover_module": "nautilus_trader.backtest.FXRolloverInterestModule",
        "execution_engine": "existing_nautilus_g0_7_bid_ask_boundary",
        "fold_definitions": "ftmoquant.research.stage_g.frozen_development_folds",
        "parallel_execution_engine": False,
    }:
        raise EurusdPolicyRateCarryProxySpecError("reuse declaration drifted")


def _mapping(value: object, description: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise EurusdPolicyRateCarryProxySpecError(
            f"{description} must be a string-keyed mapping"
        )
    return cast(dict[str, Any], value)


def _string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EurusdPolicyRateCarryProxySpecError(
            f"{description} must be a non-empty string"
        )
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], description: str) -> None:
    if set(value) != expected:
        raise EurusdPolicyRateCarryProxySpecError(
            f"{description} fields are not exact"
        )


def _utc(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")
