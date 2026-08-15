"""Strict machine-readable preregistration for ``liquidity_shock_reversion_v1``."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
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

LIQUIDITY_SHOCK_REVERSION_SPEC_PATH = Path(
    "config/strategies/liquidity_shock_reversion_v1.yaml"
)
LIQUIDITY_SHOCK_REVERSION_CONFIG_SHA256 = (
    "55a2a1814a507ebf53f5713d9672cf5d4fda19790d9c808efc0a9773bed27acd"
)


class LiquidityShockReversionSpecValidationError(ValueError):
    """Raised when the immutable Phase 1 baseline is changed or ambiguous."""


@dataclass(frozen=True, slots=True)
class LiquidityShockReversionSpec:
    schema_version: int
    strategy_id: str
    version: str
    status: str
    universe_id: str
    ordered_instruments: tuple[str, ...]
    baseline_prior_eligible_returns: int
    shock_multiple: int
    hold_eligible_completed_minutes: int
    development_folds_sha256: str
    canonical_document: dict[str, Any]


def load_liquidity_shock_reversion_spec(
    path: Path = LIQUIDITY_SHOCK_REVERSION_SPEC_PATH,
) -> LiquidityShockReversionSpec:
    """Load exactly one no-tuning Phase 1 candidate contract."""
    try:
        document = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "spec")
    except (OSError, yaml.YAMLError) as error:
        raise LiquidityShockReversionSpecValidationError(
            f"could not load strategy spec: {error}"
        ) from error
    _exact(
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
        "spec",
    )
    universe = _mapping(document["universe"], "universe")
    data = _mapping(document["data_semantics"], "data_semantics")
    signal = _mapping(document["signal"], "signal")
    execution = _mapping(document["execution"], "execution")
    boundary = _mapping(document["research_boundary"], "research_boundary")
    parameters = _mapping(document["parameter_family"], "parameter_family")
    _exact(
        universe,
        {
            "universe_id",
            "universe_plan_sha256",
            "universe_readiness_sha256",
            "ordered_instruments",
        },
        "universe",
    )
    _exact(
        data,
        {
            "source",
            "midpoint",
            "return_formula",
            "eligible_return",
            "missing_or_invalid_rule",
            "fill_or_interpolation",
            "instruments_are_independent",
        },
        "data_semantics",
    )
    _exact(
        signal,
        {
            "baseline_scale",
            "baseline_prior_eligible_returns",
            "shock_rule",
            "positive_shock_target",
            "negative_shock_target",
            "zero_or_invalid_scale",
            "insufficient_history",
            "signals_while_position_or_pending",
        },
        "signal",
    )
    _exact(
        execution,
        {
            "output",
            "admitted_targets",
            "eligible_point",
            "entry_timing_rule",
            "hold_eligible_completed_minutes_after_entry",
            "exit_timing_rule",
            "max_open_positions_per_instrument",
            "engine",
            "parallel_backtester",
            "strategy_sizing",
            "ftmo_optimization",
        },
        "execution",
    )
    _exact(
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
    _exact(parameters, {"mode", "permitted_variants"}, "parameter_family")
    folds = frozen_development_folds()
    required = {
        "identity": (
            1,
            "liquidity_shock_reversion_v1",
            "1.0.0",
            "implemented_not_evaluated",
        ),
        "universe": (
            FROZEN_UNIVERSE_ID,
            FROZEN_UNIVERSE_PLAN_SHA256,
            FROZEN_UNIVERSE_READINESS_SHA256,
            FROZEN_INSTRUMENT_IDS,
        ),
        "data": (
            "stage_g_synchronized_completed_1m_bid_ask_closes",
            "(bid_close_plus_ask_close)_divided_by_2",
            "ln(C_t_divided_by_C_t_minus_1)",
            "consecutive_observed_valid_completed_1m_midpoint_closes",
            "no_signal_and_not_an_eligible_return",
            False,
            True,
        ),
        "signal": (
            "median_absolute_log_return",
            60,
            "absolute_r_t_strictly_greater_than_5_times_baseline_scale",
            -1,
            1,
            "no_signal",
            "no_signal",
            "ignore",
        ),
        "execution": (
            "raw_directional_target_only",
            (-1, 0, 1),
            "synchronized_tradable_stage_g_execution_frame",
            "first_eligible_information_time_strictly_after_signal_information_time",
            15,
            "flat_at_first_eligible_information_time_strictly_after_fifteenth_eligible_completed_minute_after_entry",
            1,
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
            "targets_may_originate_only_inside_comparison_interval_and_state_resets_at_fold_boundary",
            "locked",
            "locked",
            False,
        ),
        "parameters": ("baseline_only", ()),
    }
    actual = {
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
        "data": tuple(
            data[key]
            for key in (
                "source",
                "midpoint",
                "return_formula",
                "eligible_return",
                "missing_or_invalid_rule",
                "fill_or_interpolation",
                "instruments_are_independent",
            )
        ),
        "signal": tuple(
            signal[key]
            for key in (
                "baseline_scale",
                "baseline_prior_eligible_returns",
                "shock_rule",
                "positive_shock_target",
                "negative_shock_target",
                "zero_or_invalid_scale",
                "insufficient_history",
                "signals_while_position_or_pending",
            )
        ),
        "execution": (
            execution["output"],
            tuple(execution["admitted_targets"]),
            *tuple(
                execution[key]
                for key in (
                    "eligible_point",
                    "entry_timing_rule",
                    "hold_eligible_completed_minutes_after_entry",
                    "exit_timing_rule",
                    "max_open_positions_per_instrument",
                    "engine",
                    "parallel_backtester",
                    "strategy_sizing",
                    "ftmo_optimization",
                )
            ),
        ),
        "boundary": tuple(
            boundary[key]
            for key in (
                "development_start_utc",
                "development_end_exclusive_utc",
                "development_folds_version",
                "development_folds_sha256",
                "fold_warmup_rule",
                "fold_comparison_rule",
                "validation",
                "final_holdout",
                "strategy_returns_accessed",
            )
        ),
        "parameters": (parameters["mode"], tuple(parameters["permitted_variants"])),
    }
    if actual != required:
        raise LiquidityShockReversionSpecValidationError(
            "strategy spec does not match frozen Phase 1"
        )
    return LiquidityShockReversionSpec(
        1,
        "liquidity_shock_reversion_v1",
        "1.0.0",
        "implemented_not_evaluated",
        FROZEN_UNIVERSE_ID,
        FROZEN_INSTRUMENT_IDS,
        60,
        5,
        15,
        folds.semantic_sha256,
        document,
    )


def liquidity_shock_reversion_config_sha256(spec: LiquidityShockReversionSpec) -> str:
    return hashlib.sha256(
        json.dumps(
            spec.canonical_document, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _mapping(value: object, description: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise LiquidityShockReversionSpecValidationError(
            f"{description} must be a mapping"
        )
    return cast(dict[str, Any], value)


def _exact(value: dict[str, Any], expected: set[str], description: str) -> None:
    if set(value) != expected:
        raise LiquidityShockReversionSpecValidationError(
            f"{description} fields are not exact"
        )


def _strings(value: object, description: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise LiquidityShockReversionSpecValidationError(
            f"{description} must be a string list"
        )
    return tuple(value)


def _utc(value: object) -> str:
    return str(cast(Any, value).isoformat().replace("+00:00", "Z"))
