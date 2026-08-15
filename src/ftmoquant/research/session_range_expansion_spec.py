"""Strict machine-readable preregistration for ``session_range_expansion_v1``."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
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

SESSION_RANGE_EXPANSION_SPEC_PATH = Path(
    "config/strategies/session_range_expansion_v1.yaml"
)
SESSION_RANGE_EXPANSION_CONFIG_SHA256 = (
    "5303094db45b3d9164d1854787c39d9ce0e69974875689dc51742e4495b9e472"
)


class SessionRangeExpansionSpecValidationError(ValueError):
    """Raised when the frozen session-range candidate contract drifts."""


@dataclass(frozen=True, slots=True)
class SessionRangeExpansionSpec:
    schema_version: int
    strategy_id: str
    version: str
    status: str
    ordered_instruments: tuple[str, ...]
    session_timezone: str
    range_start: str
    range_end: str
    breakout_end: str
    exit_time: str
    development_folds_sha256: str
    canonical_document: dict[str, Any]


def load_session_range_expansion_spec(
    path: Path = SESSION_RANGE_EXPANSION_SPEC_PATH,
) -> SessionRangeExpansionSpec:
    """Load exactly one no-parameter frozen session-range baseline."""

    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise SessionRangeExpansionSpecValidationError(
            f"could not load strategy spec: {error}"
        ) from error
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise SessionRangeExpansionSpecValidationError(
            "strategy spec must be a mapping"
        )
    document = cast(dict[str, Any], value)
    if document != _frozen_document():
        raise SessionRangeExpansionSpecValidationError(
            "strategy spec does not match frozen Phase 1"
        )
    return SessionRangeExpansionSpec(
        schema_version=1,
        strategy_id="session_range_expansion_v1",
        version="1.0.0",
        status="implemented_not_evaluated",
        ordered_instruments=FROZEN_INSTRUMENT_IDS,
        session_timezone="Europe/London",
        range_start="00:00",
        range_end="08:00",
        breakout_end="12:00",
        exit_time="16:00",
        development_folds_sha256=frozen_development_folds().semantic_sha256,
        canonical_document=document,
    )


def session_range_expansion_config_sha256(spec: SessionRangeExpansionSpec) -> str:
    """Hash semantic YAML content independently of formatting and key order."""

    return hashlib.sha256(
        json.dumps(
            spec.canonical_document, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _frozen_document() -> dict[str, Any]:
    folds = frozen_development_folds()
    return {
        "schema_version": 1,
        "strategy_id": "session_range_expansion_v1",
        "version": "1.0.0",
        "status": "implemented_not_evaluated",
        "universe": {
            "universe_id": FROZEN_UNIVERSE_ID,
            "universe_plan_sha256": FROZEN_UNIVERSE_PLAN_SHA256,
            "universe_readiness_sha256": FROZEN_UNIVERSE_READINESS_SHA256,
            "ordered_instruments": list(FROZEN_INSTRUMENT_IDS),
        },
        "data_semantics": {
            "source": "stage_g_synchronized_observed_1m_bid_ask_closes",
            "midpoint": "(bid_close_plus_ask_close)_divided_by_2",
            "session_timezone": "Europe/London",
            "range_window": "00:00 <= London time < 08:00",
            "breakout_window": "08:00 <= London time < 12:00",
            "exit_time": "16:00 London time",
            "range_rule": "max_min_of_completed_1m_midpoints",
            "range_completeness": "exactly_480_consecutive_completed_1m_midpoints",
            "incomplete_or_invalid_range": "no_trade",
            "instruments_are_independent": True,
        },
        "signal": {
            "first_close_above_range_high_target": 1,
            "first_close_below_range_low_target": -1,
            "no_breakout_target": 0,
            "maximum_entries_per_instrument_per_london_day": 1,
            "exit_target": 0,
            "entry_hold_rule": "hold_until_scheduled_exit",
        },
        "execution": {
            "output": "raw_directional_target_only",
            "admitted_targets": [-1, 0, 1],
            "eligible_point": "synchronized_tradable_stage_g_execution_frame",
            "timing_rule": (
                "first_eligible_information_time_strictly_after_signal_information_time"
            ),
            "engine": "existing_nautilus_g0_7_boundary",
            "parallel_backtester": False,
            "strategy_sizing": False,
            "ftmo_optimization": False,
        },
        "research_boundary": {
            "development_start_utc": _utc(DEVELOPMENT_START),
            "development_end_exclusive_utc": _utc(DEVELOPMENT_END_EXCLUSIVE),
            "development_folds_version": folds.version,
            "development_folds_sha256": folds.semantic_sha256,
            "fold_reset_rule": "fresh_state_and_flat_position_at_each_fold_boundary",
            "validation": "locked",
            "final_holdout": "locked",
            "strategy_returns_accessed": False,
        },
        "parameter_family": {"mode": "baseline_only", "permitted_variants": []},
    }


def _utc(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")
