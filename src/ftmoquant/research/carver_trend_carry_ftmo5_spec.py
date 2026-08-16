"""Immutable specification for the Carver trend/carry FTMO-five candidate."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
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
    "489b53abff19e041afda9cc5ba210a67252bf89dea87c8b9994f08c4422e210d"
)


class CarverTrendCarryFtmo5SpecValidationError(ValueError):
    """Raised when the frozen Carver candidate specification drifts."""


@dataclass(frozen=True, slots=True)
class CarverTrendCarryFtmo5Spec:
    candidate_id: str
    version: str
    semantic_sha256: str
    canonical_document: dict[str, Any]


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
    }
    if set(document) != required:
        raise CarverTrendCarryFtmo5SpecValidationError("spec keys are not exact")
    if (
        document["schema_version"],
        document["candidate_id"],
        document["version"],
        document["status"],
    ) != (1, "carver_trend_carry_ftmo5_v1", "1.0.0", "preregistered_not_evaluated"):
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


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CarverTrendCarryFtmo5SpecValidationError(f"{label} must be a mapping")
    return value


def _utc(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")
