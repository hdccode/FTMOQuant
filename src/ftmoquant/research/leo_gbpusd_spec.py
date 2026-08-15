"""Immutable preregistration contract for ``leo_gbpusd_v1`` only.

This module validates semantic configuration.  It deliberately has no signal,
order, position, or backtest implementation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

LEO_GBPUSD_SPEC_PATH = Path("config/strategies/leo_gbpusd_v1.yaml")
LEO_GBPUSD_CONFIG_SHA256 = (
    "5c89c825db340c49f10c18fdc3982b03f23ade718a5889e4e2ef662081abbf4c"
)


class LeoGbpUsdSpecValidationError(ValueError):
    """Raised when the frozen Leo mechanical approximation drifts."""


@dataclass(frozen=True, slots=True)
class LeoGbpUsdSpec:
    strategy_id: str
    version: str
    status: str
    instrument_id: str
    bar_timeframe: str
    session_timezone: str
    canonical_document: dict[str, Any]


def load_leo_gbpusd_spec(path: Path = LEO_GBPUSD_SPEC_PATH) -> LeoGbpUsdSpec:
    """Load the one frozen, unevaluated Leo GBPUSD v1 contract."""
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise LeoGbpUsdSpecValidationError(
            f"could not load strategy spec: {error}"
        ) from error
    if not isinstance(loaded, dict) or any(not isinstance(key, str) for key in loaded):
        raise LeoGbpUsdSpecValidationError("strategy spec must be a mapping")
    document = cast(dict[str, Any], loaded)
    if document != _frozen_document():
        raise LeoGbpUsdSpecValidationError("strategy spec does not match frozen v1")
    return LeoGbpUsdSpec(
        strategy_id="leo_gbpusd_v1",
        version="1.0.0",
        status="specified_not_run",
        instrument_id="GBP/USD.DUKASCOPY",
        bar_timeframe="15m",
        session_timezone="Europe/London",
        canonical_document=document,
    )


def leo_gbpusd_config_sha256(spec: LeoGbpUsdSpec) -> str:
    """Hash semantic YAML content independently of formatting and key order."""
    return hashlib.sha256(
        json.dumps(
            spec.canonical_document, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _frozen_document() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "strategy_id": "leo_gbpusd_v1",
        "version": "1.0.0",
        "status": "specified_not_run",
        "provenance": {
            "source_type": "externally_sourced_hypothesis",
            "subject": "Leo FTMO trader public methodology",
            "interpretation": "mechanical_approximation_only",
            "note_path": "docs/research/leo_gbpusd_v1.md",
        },
        "instrument": {"instrument_id": "GBP/USD.DUKASCOPY", "asset_class": "FX"},
        "data_semantics": {
            "bar_timeframe": "15m",
            "bar_completion": "completed_bar_only",
            "price_input": "canonical_bid_ask_ohlc",
            "session_timezone": "Europe/London",
            "dst_policy": "IANA_Europe_London_zoneinfo",
            "incomplete_or_invalid_bar": "no_signal",
        },
        "sessions": {
            "asia_reference": {
                "window": "00:00 <= London bar end < 08:00",
                "reference": "completed_session_high_low",
            },
            "london_reference": {
                "window": "08:00 <= London bar end < 13:00",
                "reference": "completed_session_high_low",
            },
            "london_entry": {
                "window": "09:00 <= London completed signal bar end < 11:00",
                "reference": "asia_reference",
            },
            "new_york_entry": {
                "window": "14:00 <= London completed signal bar end < 16:00",
                "reference": "london_reference",
            },
        },
        "signal": {
            "upper_rejection": (
                "completed_bar_high_strictly_gt_reference_high_and_"
                "close_strictly_lt_reference_high"
            ),
            "lower_rejection": (
                "completed_bar_low_strictly_lt_reference_low_and_"
                "close_strictly_gt_reference_low"
            ),
            "upper_rejection_direction": "short",
            "lower_rejection_direction": "long",
            "excluded_conditions": [
                "wick_threshold",
                "trend_filter",
                "volatility_filter",
                "fibonacci",
                "double_top_requirement",
                "discretionary_confirmation",
                "alternate_session_windows",
            ],
        },
        "execution": {
            "entry_timing": (
                "first_synchronized_tradable_g0_7_frame_strictly_after_"
                "completed_signal_bar"
            ),
            "entry_order_type": "market",
            "engine": "existing_nautilus_g0_7_boundary",
            "costs": "existing_g0_7_cost_profile",
            "same_bar_stop_target_policy": "lower_timeframe_path_else_stop_first",
        },
        "exits": {
            "stop": "sweep_bar_extreme",
            "profit_target": (
                "exactly_3_times_initial_stop_distance_from_executed_entry"
            ),
            "unresolved_position": "close_at_named_entry_window_end",
        },
        "positions": {
            "max_open_positions": 1,
            "entries_while_position_open": "discard",
            "maximum_entries_per_named_session_per_london_day": 1,
            "overlapping_setup_policy": "no_new_entry_while_position_open",
        },
        "research_boundary": {
            "validation": "locked",
            "final_holdout": "locked",
            "strategy_returns_accessed": False,
        },
        "parameter_family": {"mode": "baseline_only", "permitted_variants": []},
    }
