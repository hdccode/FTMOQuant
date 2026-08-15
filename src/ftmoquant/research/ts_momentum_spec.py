"""Strict machine-readable preregistration for ``ts_momentum_v1``."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import yaml

from ftmoquant.research.stage_g import (
    DEVELOPMENT_END_EXCLUSIVE,
    DEVELOPMENT_START,
    FROZEN_INSTRUMENT_IDS,
    FROZEN_UNIVERSE_ID,
    FROZEN_UNIVERSE_PLAN_SHA256,
    FROZEN_UNIVERSE_READINESS_SHA256,
    frozen_development_folds,
)

TS_MOMENTUM_SPEC_PATH = Path("config/strategies/ts_momentum_v1.yaml")
TS_MOMENTUM_CONFIG_SHA256 = (
    "edcbe2e4afe631e5fde1223558122ecf4d796abd0610729313ebbb32a468ccd5"
)


class TsMomentumSpecValidationError(ValueError):
    """Raised when the frozen candidate contract changes or is ambiguous."""


@dataclass(frozen=True, slots=True)
class TsMomentumSpec:
    schema_version: int
    strategy_id: str
    version: str
    status: str
    universe_id: str
    ordered_instruments: tuple[str, ...]
    daily_session_timezone: str
    daily_session_close: str
    lookback_prior_eligible_observations: int
    native_period: int
    development_folds_sha256: str
    canonical_document: dict[str, Any]


def load_ts_momentum_spec(path: Path = TS_MOMENTUM_SPEC_PATH) -> TsMomentumSpec:
    """Load the one frozen baseline and reject additions or substitutions."""

    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise TsMomentumSpecValidationError(
            f"could not load strategy spec: {error}"
        ) from error
    document = _mapping(value, "strategy spec")
    _exact_keys(
        document,
        {
            "schema_version",
            "strategy_id",
            "version",
            "status",
            "universe",
            "data_semantics",
            "signal",
            "execution",
            "research_boundary",
            "parameter_family",
        },
        "strategy spec",
    )
    universe = _mapping(document["universe"], "universe")
    _exact_keys(
        universe,
        {
            "universe_id",
            "universe_plan_sha256",
            "universe_readiness_sha256",
            "ordered_instruments",
        },
        "universe",
    )
    data = _mapping(document["data_semantics"], "data_semantics")
    _exact_keys(
        data,
        {
            "source",
            "midpoint",
            "daily_session_timezone",
            "daily_session_close",
            "daily_eligible_weekdays",
            "daily_close_rule",
            "missing_or_invalid_rule",
            "fill_or_interpolation",
            "instruments_are_independent",
        },
        "data_semantics",
    )
    signal = _mapping(document["signal"], "signal")
    _exact_keys(
        signal,
        {
            "native_indicator",
            "native_use_log",
            "native_period",
            "lookback_prior_eligible_observations",
            "formula",
            "positive_target",
            "negative_target",
            "zero_target",
            "insufficient_history",
            "emit_only_when_target_changes",
            "target_hold_rule",
        },
        "signal",
    )
    execution = _mapping(document["execution"], "execution")
    _exact_keys(
        execution,
        {
            "output",
            "admitted_targets",
            "eligible_point",
            "timing_rule",
            "engine",
            "parallel_backtester",
            "strategy_sizing",
            "ftmo_optimization",
        },
        "execution",
    )
    boundary = _mapping(document["research_boundary"], "research_boundary")
    _exact_keys(
        boundary,
        {
            "development_start_utc",
            "development_end_exclusive_utc",
            "development_folds_version",
            "development_folds_sha256",
            "fold_warmup_rule",
            "fold_comparison_rule",
            "validation",
            "final_holdout",
            "strategy_returns_accessed",
        },
        "research_boundary",
    )
    parameters = _mapping(document["parameter_family"], "parameter_family")
    _exact_keys(parameters, {"mode", "permitted_variants"}, "parameter_family")

    expected = {
        "identity": (
            document["schema_version"],
            document["strategy_id"],
            document["version"],
            document["status"],
        ),
        "universe": (
            universe["universe_id"],
            universe["universe_plan_sha256"],
            universe["universe_readiness_sha256"],
            tuple(_strings(universe["ordered_instruments"], "ordered_instruments")),
        ),
        "data": (
            data["source"],
            data["midpoint"],
            data["daily_session_timezone"],
            data["daily_session_close"],
            tuple(_strings(data["daily_eligible_weekdays"], "daily weekdays")),
            data["daily_close_rule"],
            data["missing_or_invalid_rule"],
            data["fill_or_interpolation"],
            data["instruments_are_independent"],
        ),
        "signal": (
            signal["native_indicator"],
            signal["native_use_log"],
            signal["native_period"],
            signal["lookback_prior_eligible_observations"],
            signal["formula"],
            signal["positive_target"],
            signal["negative_target"],
            signal["zero_target"],
            signal["insufficient_history"],
            signal["emit_only_when_target_changes"],
            signal["target_hold_rule"],
        ),
        "execution": (
            execution["output"],
            tuple(execution["admitted_targets"]),
            execution["eligible_point"],
            execution["timing_rule"],
            execution["engine"],
            execution["parallel_backtester"],
            execution["strategy_sizing"],
            execution["ftmo_optimization"],
        ),
        "boundary": (
            boundary["development_start_utc"],
            boundary["development_end_exclusive_utc"],
            boundary["development_folds_version"],
            boundary["development_folds_sha256"],
            boundary["fold_warmup_rule"],
            boundary["fold_comparison_rule"],
            boundary["validation"],
            boundary["final_holdout"],
            boundary["strategy_returns_accessed"],
        ),
        "parameters": (parameters["mode"], tuple(parameters["permitted_variants"])),
    }
    folds = frozen_development_folds()
    required = {
        "identity": (1, "ts_momentum_v1", "1.0.0", "implemented_not_evaluated"),
        "universe": (
            FROZEN_UNIVERSE_ID,
            FROZEN_UNIVERSE_PLAN_SHA256,
            FROZEN_UNIVERSE_READINESS_SHA256,
            FROZEN_INSTRUMENT_IDS,
        ),
        "data": (
            "stage_g_synchronized_observed_1m_bid_ask_closes",
            "(bid_close_plus_ask_close)_divided_by_2",
            "America/New_York",
            "17:00",
            ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday"),
            "observed_1m_pair_with_information_time_at_session_close",
            "no_signal_and_not_an_eligible_history_observation",
            False,
            True,
        ),
        "signal": (
            "nautilus_trader.indicators.RateOfChange",
            True,
            253,
            252,
            "ln(C_t / C_(t-252))",
            1,
            -1,
            0,
            "no_signal",
            True,
            "hold_latest_valid_target_until_changed",
        ),
        "execution": (
            "raw_directional_target_only",
            (-1, 0, 1),
            "synchronized_tradable_stage_g_execution_frame",
            "first_eligible_information_time_strictly_after_signal_information_time",
            "existing_nautilus_g0_7_boundary",
            False,
            False,
            False,
        ),
        "boundary": (
            _utc(DEVELOPMENT_START),
            _utc(DEVELOPMENT_END_EXCLUSIVE),
            folds.version,
            folds.semantic_sha256,
            "train_interval_updates_history_but_emits_no_executable_target",
            "targets_may_originate_only_inside_comparison_interval",
            "locked",
            "locked",
            False,
        ),
        "parameters": ("baseline_only", ()),
    }
    if expected != required:
        raise TsMomentumSpecValidationError(
            "strategy spec does not match frozen Phase 1"
        )
    return TsMomentumSpec(
        schema_version=1,
        strategy_id="ts_momentum_v1",
        version="1.0.0",
        status="implemented_not_evaluated",
        universe_id=FROZEN_UNIVERSE_ID,
        ordered_instruments=FROZEN_INSTRUMENT_IDS,
        daily_session_timezone="America/New_York",
        daily_session_close="17:00",
        lookback_prior_eligible_observations=252,
        native_period=253,
        development_folds_sha256=folds.semantic_sha256,
        canonical_document=document,
    )


def ts_momentum_config_sha256(spec: TsMomentumSpec) -> str:
    """Hash semantic YAML content independently of formatting and key order."""

    return hashlib.sha256(
        json.dumps(
            spec.canonical_document, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _mapping(value: object, description: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TsMomentumSpecValidationError(f"{description} must be a mapping")
    return cast(dict[str, Any], value)


def _exact_keys(value: dict[str, Any], expected: set[str], description: str) -> None:
    if set(value) != expected:
        missing = ", ".join(sorted(expected - set(value)))
        extra = ", ".join(sorted(set(value) - expected))
        raise TsMomentumSpecValidationError(
            f"{description} fields are not exact; missing={missing}; extra={extra}"
        )


def _strings(value: object, description: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TsMomentumSpecValidationError(f"{description} must be a string list")
    return tuple(value)


def _utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise TsMomentumSpecValidationError("development boundary is invalid")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
