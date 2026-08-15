"""Immutable preregistration for the first macro-event candidate.

This module validates research semantics only.  It does not load market data,
event data, price returns, validation data, or holdout data.
"""

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

USD_MACRO_SURPRISE_MOMENTUM_SPEC_PATH = Path(
    "config/strategies/usd_macro_surprise_momentum_v1.yaml"
)
USD_MACRO_SURPRISE_MOMENTUM_CONFIG_SHA256 = (
    "ce997472bfd600d3411dd7c30a9d2df04bce353c1481a3d3eab0d5efb6d9df66"
)
_EXPECTED_ARCHIVE_SHA256 = (
    "3fb4421df0ea63cac570b7adcd16892ec50909ad7d3c441d462443245a5d84ce"
)
_FAMILIES = ("US_NFP_HEADLINE_EMPLOYMENT_CHANGE", "US_CPI_HEADLINE_M_M")


class UsdMacroSurpriseMomentumSpecValidationError(ValueError):
    """Raised when the preregistration has drifted or is incomplete."""


@dataclass(frozen=True, slots=True)
class UsdMacroSurpriseMomentumSpec:
    candidate_id: str
    version: str
    semantic_sha256: str
    canonical_document: dict[str, Any]


def load_usd_macro_surprise_momentum_spec(
    path: Path = USD_MACRO_SURPRISE_MOMENTUM_SPEC_PATH,
) -> UsdMacroSurpriseMomentumSpec:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise UsdMacroSurpriseMomentumSpecValidationError(
            f"could not load macro-event spec: {error}"
        ) from error
    if not isinstance(document, dict):
        raise UsdMacroSurpriseMomentumSpecValidationError("spec must be a mapping")
    _validate(document)
    return UsdMacroSurpriseMomentumSpec(
        candidate_id=document["candidate_id"],
        version=document["version"],
        semantic_sha256=usd_macro_surprise_momentum_config_sha256(document),
        canonical_document=document,
    )


def usd_macro_surprise_momentum_config_sha256(
    spec: UsdMacroSurpriseMomentumSpec | dict[str, Any],
) -> str:
    document = (
        spec.canonical_document
        if isinstance(spec, UsdMacroSurpriseMomentumSpec)
        else spec
    )
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate(document: dict[str, Any]) -> None:
    expected = {
        "schema_version",
        "candidate_id",
        "version",
        "status",
        "event_source",
        "research_boundary",
        "universe",
        "signal",
        "direction",
        "timing",
        "execution",
        "evaluation",
    }
    if set(document) != expected:
        raise UsdMacroSurpriseMomentumSpecValidationError("spec keys are not exact")
    if (
        document["schema_version"],
        document["candidate_id"],
        document["version"],
        document["status"],
    ) != (1, "usd_macro_surprise_momentum_v1", "1.0.0", "preregistered_not_evaluated"):
        raise UsdMacroSurpriseMomentumSpecValidationError(
            "candidate identity is frozen"
        )
    event = _mapping(document["event_source"], "event_source")
    if event != {
        "name": "Hanover / Forex Factory UTC archive",
        "zip_sha256": _EXPECTED_ARCHIVE_SHA256,
        "timestamp_semantics": "UTC release timestamp",
        "permitted_event_families": list(_FAMILIES),
        "excluded_event_families": [
            "US_CPI_HEADLINE_Y_Y",
            "US_CPI_CORE_M_M",
            "US_CPI_CORE_Y_Y",
            "central_bank_rate_decisions",
        ],
    }:
        raise UsdMacroSurpriseMomentumSpecValidationError(
            "event source or families drifted"
        )
    boundary = _mapping(document["research_boundary"], "research_boundary")
    folds = frozen_development_folds()
    expected_boundary = {
        "development_start_utc": _utc(DEVELOPMENT_START),
        "development_end_exclusive_utc": _utc(DEVELOPMENT_END_EXCLUSIVE),
        "development_folds_version": folds.version,
        "development_folds_sha256": folds.semantic_sha256,
        "validation": "locked",
        "final_holdout": "locked",
        "fx_returns_accessed": False,
    }
    if boundary != expected_boundary:
        raise UsdMacroSurpriseMomentumSpecValidationError("research boundary drifted")
    universe = _mapping(document["universe"], "universe")
    if universe != {
        "ordered_instruments": ["EUR/USD.DUKASCOPY", "GBP/USD.DUKASCOPY"],
        "inference_unit": "one_macro_release_timestamp",
        "instrument_dependence": (
            "equal_weight_event_portfolio_not_independent_observations"
        ),
    }:
        raise UsdMacroSurpriseMomentumSpecValidationError(
            "universe/inference semantics drifted"
        )
    signal = _mapping(document["signal"], "signal")
    if signal != {
        "formula": "actual_minus_forecast",
        "positive_surprise": "USD_positive",
        "negative_surprise": "USD_negative",
        "zero_surprise": "no_trade",
        "surprise_threshold": "none",
        "surprise_standardization": "none",
        "parameter_optimization": "forbidden",
    }:
        raise UsdMacroSurpriseMomentumSpecValidationError("signal semantics drifted")
    timing = _mapping(document["timing"], "timing")
    direction = _mapping(document["direction"], "direction")
    if direction != {
        "USD_positive": {"EUR/USD.DUKASCOPY": "short", "GBP/USD.DUKASCOPY": "short"},
        "USD_negative": {"EUR/USD.DUKASCOPY": "long", "GBP/USD.DUKASCOPY": "long"},
    }:
        raise UsdMacroSurpriseMomentumSpecValidationError("direction semantics drifted")
    if timing != {
        "event_timestamp": "frozen_utc_release_timestamp",
        "immediate_reaction_exclusion_minutes": 5,
        "entry": (
            "first_eligible_executable_market_observation_"
            "strictly_at_or_after_t_plus_5m"
        ),
        "exit": "first_eligible_executable_market_observation_at_or_after_t_plus_60m",
        "stop_loss": "none",
        "take_profit": "none",
        "trailing_logic": "none",
        "discretionary_filters": "forbidden",
        "overlapping_position_same_instrument": "forbidden",
    }:
        raise UsdMacroSurpriseMomentumSpecValidationError("timing semantics drifted")
    execution = _mapping(document["execution"], "execution")
    if execution != {
        "framework": "deterministic_G0_7_bid_ask_execution_and_cost_framework",
        "news_fill_model": "none",
        "cost_scenarios": ["base", "1.5x"],
    }:
        raise UsdMacroSurpriseMomentumSpecValidationError("execution semantics drifted")
    evaluation = _mapping(document["evaluation"], "evaluation")
    promotion = _mapping(evaluation.get("promotion_rule"), "promotion_rule")
    if (
        promotion
        != {
            "pooled_mean_net_event_return_gt": 0,
            "positive_mean_folds_at_least": 2,
            "fold_count": 3,
            "median_fold_result_gt": 0,
            "pooled_mean_under_1_5x_cost_gt": 0,
            "implementation_or_data_integrity_failure": "fail",
        }
        or evaluation.get("failure_rule") != "reject_retire_without_changes_to_v1"
        or evaluation.get("future_mean_reversion") != "separate_candidate_required"
        or evaluation.get("bootstrap")
        != {"level": "event", "required_to_report": True, "mandatory_for_pass": False}
    ):
        raise UsdMacroSurpriseMomentumSpecValidationError("promotion rule drifted")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise UsdMacroSurpriseMomentumSpecValidationError(f"{label} must be a mapping")
    return value


def _utc(value: object) -> str:
    if not isinstance(value, datetime):
        raise TypeError("expected datetime")
    return value.isoformat().replace("+00:00", "Z")
