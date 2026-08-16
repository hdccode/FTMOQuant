"""Immutable specification for the Carver trend/carry FTMO-five candidate."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from ftmoquant.research.stage_g import (
    DEVELOPMENT_END_EXCLUSIVE,
    DEVELOPMENT_START,
    frozen_development_folds,
)

CARVER_TREND_CARRY_FTMO5_SPEC_PATH = Path(
    "config/strategies/carver_trend_carry_ftmo5_v1.yaml"
)
CARVER_TREND_CARRY_FTMO5_CONFIG_SHA256 = (
    "f1831cf1cdedeedfc21610054da8e542796c8b3c0cbc26a62074bdbf1ab39365"
)


class CarverTrendCarryFtmo5SpecValidationError(ValueError):
    """Raised when the frozen Carver candidate specification drifts."""


@dataclass(frozen=True, slots=True)
class CarverTrendCarryFtmo5Spec:
    candidate_id: str
    version: str
    semantic_sha256: str
    canonical_document: dict[str, Any]
    evaluator: CarverDevelopmentEvaluatorSpec


@dataclass(frozen=True, slots=True)
class CarverDevelopmentEvaluatorSpec:
    research_capital: Decimal
    annual_volatility_target: Decimal
    business_days_per_year: int
    average_absolute_forecast: Decimal
    instrument_weights: tuple[tuple[str, Decimal], ...]
    instrument_diversification_multiplier: Decimal
    bootstrap_confidence_level: float
    bootstrap_method: str
    bootstrap_block_size: int
    bootstrap_repetitions: int
    bootstrap_seed: int
    sharpe_annualisation_days: int
    result_schema: str


def load_carver_trend_carry_ftmo5_spec(
    path: Path = CARVER_TREND_CARRY_FTMO5_SPEC_PATH,
) -> CarverTrendCarryFtmo5Spec:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise CarverTrendCarryFtmo5SpecValidationError(
            f"could not load Carver candidate spec: {error}"
        ) from error
    if not isinstance(value, dict):
        raise CarverTrendCarryFtmo5SpecValidationError("spec must be a mapping")
    _validate(value)
    return CarverTrendCarryFtmo5Spec(
        candidate_id=value["candidate_id"],
        version=value["version"],
        semantic_sha256=carver_trend_carry_ftmo5_config_sha256(value),
        canonical_document=value,
        evaluator=_evaluator_spec(value),
    )


def carver_trend_carry_ftmo5_config_sha256(
    spec: CarverTrendCarryFtmo5Spec | dict[str, Any],
) -> str:
    document = (
        spec.canonical_document if isinstance(spec, CarverTrendCarryFtmo5Spec) else spec
    )
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate(document: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "candidate_id",
        "version",
        "status",
        "provenance",
        "universe",
        "trend",
        "carry",
        "combination",
        "causality",
        "portfolio",
        "execution_and_costs",
        "research_boundary",
        "development_gate",
        "development_evaluator",
    }
    if set(document) != required:
        raise CarverTrendCarryFtmo5SpecValidationError("spec keys are not exact")
    if (
        document["schema_version"],
        document["candidate_id"],
        document["version"],
        document["status"],
    ) != (1, "carver_trend_carry_ftmo5_v1", "1.1.0", "preregistered_not_evaluated"):
        raise CarverTrendCarryFtmo5SpecValidationError("candidate identity drifted")
    provenance = _mapping(document["provenance"], "provenance")
    if (
        provenance.get("pysystemtrade_commit")
        != "b4a25e6e1e33a54a3ecfb45c0f6db5e2b60b84f8"
        or len(provenance.get("source_sha256", {})) != 10
    ):
        raise CarverTrendCarryFtmo5SpecValidationError("reference provenance drifted")
    if any(
        not isinstance(item, str) or len(item) != 64
        for item in provenance["source_sha256"].values()
    ):
        raise CarverTrendCarryFtmo5SpecValidationError(
            "source SHA-256 manifest is invalid"
        )
    universe = _mapping(document["universe"], "universe")
    if universe != {
        "futures_to_execution": {
            "EUR": "EUR/USD.DUKASCOPY",
            "GOLD": "XAU_USD.OANDA",
            "SP500": "SPX500_USD.OANDA",
            "CRUDE_W": "WTICO_USD.OANDA",
            "SOYBEAN": "SOYBN_USD.OANDA",
        },
        "futures_role": "signal_data_only",
        "execution_price_proxy_providers": {
            "EUR": "Dukascopy",
            "GOLD": "OANDA_v20_practice",
            "SP500": "OANDA_v20_practice",
            "CRUDE_W": "OANDA_v20_practice",
            "SOYBEAN": "OANDA_v20_practice",
        },
        "execution_price_proxy_provider_metadata": {
            "OANDA_v20_practice_confirmed_instruments": [
                "XAU_USD",
                "SPX500_USD",
                "WTICO_USD",
                "SOYBN_USD",
            ],
        },
        "execution_price_scale_to_ftmo": {"SOYBN_USD.OANDA": 100},
        "execution_role": "numeric_BID_ASK_execution_price_proxy_only",
        "execution_economics_role": "frozen_FTMO_G0_8_only",
        "basis_mismatch": "explicit_report_required_no_favorable_basis_assumption",
    }:
        raise CarverTrendCarryFtmo5SpecValidationError(
            "execution-price provider mapping drifted"
        )
    trend = _mapping(document["trend"], "trend")
    if (
        trend.get("rules")
        != [
            {"fast": 16, "slow": 64, "forecast_scalar": 3.75, "weight": 0.21},
            {"fast": 32, "slow": 128, "forecast_scalar": 2.65, "weight": 0.08},
            {"fast": 64, "slow": 256, "forecast_scalar": 1.87, "weight": 0.21},
        ]
        or trend.get("forecast_cap_absolute") != 20
    ):
        raise CarverTrendCarryFtmo5SpecValidationError("trend rules drifted")
    carry = _mapping(document["carry"], "carry")
    if (
        carry.get("smoothing_ewm_span_days") != 90
        or carry.get("forecast_scalar") != 30
        or carry.get("weight") != 0.50
        or carry.get("forecast_cap_absolute") != 20
    ):
        raise CarverTrendCarryFtmo5SpecValidationError("carry rules drifted")
    if _mapping(document["combination"], "combination") != {
        "forecast_diversification_multiplier": 1.31,
        "parameter_optimization": "forbidden",
    }:
        raise CarverTrendCarryFtmo5SpecValidationError("forecast combination drifted")
    _validate_development_evaluator(document)
    boundary = _mapping(document["research_boundary"], "research_boundary")
    folds = frozen_development_folds()
    if boundary != {
        "development_start_utc": _utc(DEVELOPMENT_START),
        "development_end_exclusive_utc": _utc(DEVELOPMENT_END_EXCLUSIVE),
        "development_folds_version": folds.version,
        "development_folds_sha256": folds.semantic_sha256,
        "validation": "locked",
        "final_holdout": "locked",
        "returns_accessed": False,
    }:
        raise CarverTrendCarryFtmo5SpecValidationError("research boundary drifted")
    gate = _mapping(document["development_gate"], "development_gate")
    if (
        gate.get("pooled_net_daily_mean_gt") != 0
        or gate.get("positive_fold_means_at_least") != 2
        or gate.get("fold_count") != 3
        or gate.get("median_fold_mean_gt") != 0
        or gate.get("pooled_mean_under_1_5x_cost_gt") != 0
        or gate.get("failure_rule") != "reject_retire_without_tuning"
    ):
        raise CarverTrendCarryFtmo5SpecValidationError("development gate drifted")


def _evaluator_spec(document: dict[str, Any]) -> CarverDevelopmentEvaluatorSpec:
    evaluator = _mapping(document["development_evaluator"], "development_evaluator")
    account = _mapping(evaluator["account"], "development_evaluator.account")
    sizing = _mapping(evaluator["sizing"], "development_evaluator.sizing")
    statistics = _mapping(evaluator["statistics"], "development_evaluator.statistics")
    bootstrap = _mapping(
        statistics["bootstrap"], "development_evaluator.statistics.bootstrap"
    )
    artifacts = _mapping(evaluator["artifacts"], "development_evaluator.artifacts")
    weights = _mapping(
        sizing["instrument_weights"], "development_evaluator.sizing.instrument_weights"
    )
    return CarverDevelopmentEvaluatorSpec(
        research_capital=Decimal(str(account["research_capital"])),
        annual_volatility_target=Decimal(
            str(account["annual_percentage_volatility_target"])
        )
        / Decimal(100),
        business_days_per_year=int(sizing["business_days_per_year"]),
        average_absolute_forecast=Decimal(str(sizing["average_absolute_forecast"])),
        instrument_weights=tuple(
            (instrument, Decimal(str(weight))) for instrument, weight in weights.items()
        ),
        instrument_diversification_multiplier=Decimal(
            str(sizing["instrument_diversification_multiplier"])
        ),
        bootstrap_confidence_level=float(bootstrap["confidence_level"]),
        bootstrap_method=str(bootstrap["method"]),
        bootstrap_block_size=int(bootstrap["block_size"]),
        bootstrap_repetitions=int(bootstrap["repetitions"]),
        bootstrap_seed=int(bootstrap["seed"]),
        sharpe_annualisation_days=int(statistics["sharpe_annualisation_days"]),
        result_schema=str(artifacts["result_schema"]),
    )


def _validate_development_evaluator(document: dict[str, Any]) -> None:
    evaluator = _mapping(document["development_evaluator"], "development_evaluator")
    if set(evaluator) != {
        "input_contract",
        "account",
        "sizing",
        "signal_clock",
        "execution",
        "constraints",
        "accounting",
        "statistics",
        "artifacts",
        "decision_provenance",
    }:
        raise CarverTrendCarryFtmo5SpecValidationError(
            "development evaluator keys are not exact"
        )
    parsed = _evaluator_spec(document)
    if (
        parsed.research_capital != Decimal("500000")
        or parsed.annual_volatility_target != Decimal("0.25")
        or parsed.business_days_per_year != 256
        or parsed.average_absolute_forecast != Decimal("10")
        or parsed.instrument_weights
        != tuple(
            (item, Decimal("0.2"))
            for item in ("EUR", "GOLD", "SP500", "CRUDE_W", "SOYBEAN")
        )
        or parsed.instrument_diversification_multiplier != Decimal("1.0")
        or parsed.bootstrap_confidence_level != 0.95
        or parsed.bootstrap_method != "basic"
        or parsed.bootstrap_block_size != 20
        or parsed.bootstrap_repetitions != 10_000
        or parsed.bootstrap_seed != 14_042_026
        or parsed.sharpe_annualisation_days != 252
        or parsed.result_schema
        != "ftmoquant.carver-trend-carry-ftmo5-development-results-v1"
    ):
        raise CarverTrendCarryFtmo5SpecValidationError(
            "development evaluator numeric semantics drifted"
        )
    exact = {
        "signal_clock": {
            "daily_reference_aggregation": (
                "adjusted_price_last_and_annualised_roll_mean_by_UTC_date"
            ),
            "completed_at": "next_UTC_midnight_after_reference_date",
            "multiple_rows_per_day": (
                "aggregate_before_next_midnight_no_intraday_rebalance"
            ),
            "rebalance": "once_per_instrument_per_completed_daily_signal",
        },
        "execution": {
            "candle_component": "close",
            "buy_field": "ask_close",
            "sell_field": "bid_close",
            "observation_available_at": "candle_start_plus_one_minute",
            "selection": (
                "first_genuine_session_eligible_observation_strictly_later_than_"
                "signal_completion"
            ),
            "sparse_provider_observations": (
                "consume_returned_sequence_only_no_fill_or_interpolation"
            ),
            "evaluation_view": (
                "retain_each_UTC_dates_first_session_eligible_and_last_genuine_"
                "observation_only"
            ),
            "transition": (
                "market_delta_from_current_continuous_lots_to_desired_continuous_lots"
            ),
            "soybean_scale_boundary": (
                "multiply_raw_bid_ask_by_100_immediately_before_G0_8_economics"
            ),
        },
        "constraints": {
            "ordering": (
                "compute_unconstrained_target_then_validate_session_then_validate_"
                "aggregate_swing_margin_then_execute"
            ),
            "aggregate_swing_margin_limit": "research_capital",
            "breach_action": "fail_closed_no_clipping_or_rescaling",
            "challenge_loss_limits": "not_applied_G1_core_edge",
        },
    }
    for key, expected in exact.items():
        if _mapping(evaluator[key], f"development_evaluator.{key}") != expected:
            raise CarverTrendCarryFtmo5SpecValidationError(
                f"development evaluator {key} drifted"
            )


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CarverTrendCarryFtmo5SpecValidationError(f"{label} must be a mapping")
    return value


def _utc(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")
