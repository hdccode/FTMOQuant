"""Strict preregistration loader for the generic ``eurusd_tsm_v1`` family."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import yaml

from ftmoquant.research.g1.selector import RANKING_PROTOCOL, SelectionPolicy
from ftmoquant.research.stage_g import (
    DEVELOPMENT_END_EXCLUSIVE,
    DEVELOPMENT_START,
    FROZEN_UNIVERSE_ID,
    FROZEN_UNIVERSE_PLAN_SHA256,
    FROZEN_UNIVERSE_READINESS_SHA256,
    frozen_development_folds,
)

EURUSD_TSM_SPEC_PATH = Path("config/strategies/eurusd_tsm_v1.yaml")
EURUSD_TSM_SEMANTIC_SHA256 = (
    "0a5411f003dba15d88b8f2ad7368ad9d25cc16a5474108879eb012658c670a98"
)


class EurusdTsmSpecError(ValueError):
    """Raised when the preregistration is ambiguous, incomplete, or changed."""


@dataclass(frozen=True, slots=True)
class TsmTimeframeGrid:
    timeframe: str
    bid_bar_type: str
    ask_bar_type: str
    lookback_bars: tuple[int, ...]
    refresh_interval_bars: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EurusdTsmSpec:
    family_id: str
    version: str
    economic_hypothesis: str
    timeframe_grids: tuple[TsmTimeframeGrid, ...]
    deadbands: tuple[float, ...]
    expected_unique_trial_count: int
    semantic_sha256: str
    canonical_document: dict[str, Any]

    @property
    def selection_policy(self) -> SelectionPolicy:
        raw = _mapping(
            _mapping(self.canonical_document["selector"], "selector")["policy"],
            "selector.policy",
        )
        return SelectionPolicy(
            min_trade_count=_integer(raw["min_trade_count"], "min_trade_count"),
            min_positive_folds=_integer(
                raw["min_positive_folds"], "min_positive_folds"
            ),
            max_drawdown=_optional_float(raw["max_drawdown"], "max_drawdown"),
            max_year_concentration=_optional_float(
                raw["max_year_concentration"], "max_year_concentration"
            ),
            max_execution_sensitivity=_optional_float(
                raw["max_execution_sensitivity"], "max_execution_sensitivity"
            ),
            min_evaluated_neighbours=_integer(
                raw["min_evaluated_neighbours"], "min_evaluated_neighbours"
            ),
            min_acceptable_neighbour_fraction=_float(
                raw["min_acceptable_neighbour_fraction"],
                "min_acceptable_neighbour_fraction",
            ),
            required_fold_count=_integer(
                raw["required_fold_count"], "required_fold_count"
            ),
        )


def load_eurusd_tsm_spec(path: Path = EURUSD_TSM_SPEC_PATH) -> EurusdTsmSpec:
    """Load and prove the exact pre-return 90-cell family protocol."""

    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise EurusdTsmSpecError(f"could not load EUR/USD TSM spec: {error}") from error
    document = _mapping(value, "spec")
    _exact_keys(
        document,
        {
            "schema_version",
            "family",
            "dataset",
            "session",
            "parameter_grid",
            "signal",
            "risk_normalization",
            "execution_and_costs",
            "development_folds",
            "eligibility",
            "neighbours",
            "selector",
            "search",
            "sealed_partitions",
            "reuse",
            "preregistration_amendments",
            "semantic_sha256",
        },
        "spec",
    )
    semantic = semantic_sha256_for_document(document)
    if document["semantic_sha256"] != semantic:
        raise EurusdTsmSpecError("semantic_sha256 does not match canonical semantics")
    if semantic != EURUSD_TSM_SEMANTIC_SHA256:
        raise EurusdTsmSpecError("EUR/USD TSM preregistration semantic SHA drifted")

    family = _mapping(document["family"], "family")
    _exact_keys(
        family,
        {"family_id", "version", "status", "economic_hypothesis"},
        "family",
    )
    if (
        document["schema_version"] != 1
        or family["family_id"] != "eurusd_tsm_v1"
        or family["version"] != "1.0.0"
        or family["status"] != "preregistered_not_run"
    ):
        raise EurusdTsmSpecError("family identity/status is not frozen")
    hypothesis = _string(family["economic_hypothesis"], "economic_hypothesis")

    dataset = _mapping(document["dataset"], "dataset")
    if (
        dataset.get("instrument_id") != "EUR/USD.DUKASCOPY"
        or dataset.get("universe_id") != FROZEN_UNIVERSE_ID
        or dataset.get("universe_plan_sha256") != FROZEN_UNIVERSE_PLAN_SHA256
        or dataset.get("universe_readiness_sha256") != FROZEN_UNIVERSE_READINESS_SHA256
        or dataset.get("partition") != "development"
        or dataset.get("development_start_utc") != _utc(DEVELOPMENT_START)
        or dataset.get("development_end_exclusive_utc")
        != _utc(DEVELOPMENT_END_EXCLUSIVE)
    ):
        raise EurusdTsmSpecError("dataset identity or DEVELOPMENT boundary drifted")
    session = _mapping(document["session"], "session")
    if session != {"session_id": "all", "optimized": False}:
        raise EurusdTsmSpecError("session must be frozen to All")

    parameter_grid = _mapping(document["parameter_grid"], "parameter_grid")
    _exact_keys(
        parameter_grid,
        {
            "timeframes",
            "deadband_volatility_units",
            "conditional_combination_rule",
            "expected_unique_trial_count",
        },
        "parameter_grid",
    )
    timeframe_grids = _timeframe_grids(parameter_grid["timeframes"])
    if tuple(item.timeframe for item in timeframe_grids) != ("H1", "H4"):
        raise EurusdTsmSpecError("timeframes must be exactly H1 then H4")
    if timeframe_grids[0].lookback_bars != (24, 72, 120, 240, 480) or (
        timeframe_grids[0].refresh_interval_bars != (1, 4, 24)
    ):
        raise EurusdTsmSpecError("H1 parameter grid drifted")
    if timeframe_grids[1].lookback_bars != (6, 18, 30, 60, 120) or (
        timeframe_grids[1].refresh_interval_bars != (1, 3, 6)
    ):
        raise EurusdTsmSpecError("H4 parameter grid drifted")
    deadbands = tuple(
        _float(item, "deadband")
        for item in _list(parameter_grid["deadband_volatility_units"], "deadbands")
    )
    if deadbands != (0.0, 0.25, 0.5):
        raise EurusdTsmSpecError("deadband grid drifted")
    expected_count = _integer(
        parameter_grid["expected_unique_trial_count"], "expected trial count"
    )
    actual_count = sum(
        len(item.lookback_bars) * len(item.refresh_interval_bars) * len(deadbands)
        for item in timeframe_grids
    )
    if expected_count != 90 or actual_count != expected_count:
        raise EurusdTsmSpecError("conditional grid must contain exactly 90 cells")

    _validate_signal(document)
    _validate_risk_execution(document)
    _validate_folds(document)
    _validate_selection_and_seals(document)
    return EurusdTsmSpec(
        family_id="eurusd_tsm_v1",
        version="1.0.0",
        economic_hypothesis=hypothesis,
        timeframe_grids=timeframe_grids,
        deadbands=deadbands,
        expected_unique_trial_count=expected_count,
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


def _validate_signal(document: dict[str, Any]) -> None:
    signal = _mapping(document["signal"], "signal")
    volatility = _mapping(signal.get("volatility"), "signal.volatility")
    targets = _mapping(signal.get("targets"), "signal.targets")
    refresh = _mapping(signal.get("refresh"), "signal.refresh")
    exits = _mapping(signal.get("exits"), "signal.exits")
    if (
        signal.get("price") != "synchronized_completed_bid_ask_midpoint_close"
        or signal.get("trailing_return") != "ln(C_t / C_t_minus_lookback_bars)"
        or volatility.get("observation_window") != "max(20, lookback_bars)"
        or volatility.get("degrees_of_freedom") != 1
        or volatility.get("backfill") is not False
        or volatility.get("floor_or_tiny_value_manufacture") is not False
        or targets.get("raw_magnitudes") != [-1, 0, 1]
        or targets.get("signal_strength_parameter") is not False
        or refresh.get("emit_only_when_target_changes") is not True
        or exits
        != {
            "stop_loss": "none",
            "take_profit": "none",
            "reverse_or_flatten": "subsequent_causal_changed_target_only",
        }
    ):
        raise EurusdTsmSpecError("signal/volatility semantics drifted")


def _validate_risk_execution(document: dict[str, Any]) -> None:
    risk = _mapping(document["risk_normalization"], "risk_normalization")
    execution = _mapping(document["execution_and_costs"], "execution_and_costs")
    if (
        risk
        != {
            "type": "causal_pre_execution_underlying_vol_target",
            "target_annualized_volatility": 0.01,
            "implementation": (
                "ftmoquant.research.g1.normalization.G1VolatilityNormalizer"
            ),
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
            "pathological_volatility_behavior": (
                "fail_closed_zero_nonzero_exposure"
            ),
            "exposure_formula": (
                "directional_signal_times_0.01_divided_by_ex_ante_annualized_volatility"
            ),
            "sizing_refresh_rule": (
                "every_configured_refresh_including_unchanged_direction"
            ),
            "base_units_per_unit_exposure": "100000",
            "target_quantity_rule": (
                "desired_exposure_times_base_units_before_native_execution"
            ),
            "ftmo_or_g4_sizing": False,
        }
        or execution.get("engine") != "existing_nautilus_g0_7_bid_ask_boundary"
        or execution.get("spread") != "observed_paired_bid_ask"
        or execution.get("cost_stress_multiplier") != 1.5
        or execution.get("cost_stress_formula")
        != "net_return_minus_one_half_realized_base_cost"
        or execution.get("position_sizing_timing")
        != "before_native_order_submission"
        or execution.get("native_cost_quantity_basis")
        != "actual_executed_scaled_delta_base_units"
        or execution.get("turnover_basis")
        != "actual_executed_scaled_target_changes"
        or execution.get("adverse_execution_perturbation")
        != "unused_no_generic_implementation"
    ):
        raise EurusdTsmSpecError("risk/execution/cost semantics drifted")


def _validate_folds(document: dict[str, Any]) -> None:
    declared = _mapping(document["development_folds"], "development_folds")
    folds = frozen_development_folds()
    if (
        declared.get("version") != folds.version
        or declared.get("semantic_sha256") != folds.semantic_sha256
    ):
        raise EurusdTsmSpecError("DEVELOPMENT fold identity drifted")
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
        raise EurusdTsmSpecError("declared fold timestamps drifted")


def _validate_selection_and_seals(document: dict[str, Any]) -> None:
    eligibility = _mapping(document["eligibility"], "eligibility")
    selector = _mapping(document["selector"], "selector")
    policy = _mapping(selector.get("policy"), "selector.policy")
    search = _mapping(document["search"], "search")
    seals = _mapping(document["sealed_partitions"], "sealed_partitions")
    amendments = _list(
        document["preregistration_amendments"], "preregistration_amendments"
    )
    if (
        eligibility.get("required_fold_count") != 3
        or eligibility.get("minimum_positive_folds") != 2
        or eligibility.get("minimum_pooled_executed_transitions") != 100
        or eligibility.get("maximum_drawdown_hard_gate") is not False
        or eligibility.get("plateau_hard_gate") is not False
        or eligibility.get("year_concentration_hard_gate") is not False
        or tuple(_strings(selector.get("ranking_order"), "selector ranking"))
        != RANKING_PROTOCOL
        or policy
        != {
            "min_trade_count": 100,
            "min_positive_folds": 2,
            "max_drawdown": None,
            "max_year_concentration": None,
            "max_execution_sensitivity": None,
            "min_evaluated_neighbours": 0,
            "min_acceptable_neighbour_fraction": 0.0,
            "required_fold_count": 3,
        }
        or selector.get("weighted_score") is not False
        or search.get("mode") != "exact_grid"
        or search.get("optuna") is not False
        or search.get("parameter_grid_frozen_before_returns") is not True
        or seals.get("strategy_returns_accessed") is not False
        or not str(seals.get("validation", "")).startswith("locked")
        or seals.get("final_holdout") != "locked"
        or amendments
        != [
            {
                "amendment_id": "causal_g1_normalization_pre_data_2026_08_16",
                "superseded_semantic_sha256": (
                    "6f2cdf187e40e9bc9c065c7e7befa16f89d1fe380c8626ab853474658466f116"
                ),
                "reason_superseded": (
                    "generic_normalizer_used_whole_realized_return_series"
                ),
                "amendment_date": "2026-08-16",
                "repository_commit_before_amendment": (
                    "ca53c43d0c3486427cdc533892cd359581325752"
                ),
                "observed_development_trial_results": 0,
                "observed_development_fold_results": 0,
                "historical_development_strategy_returns_accessed": False,
                "validation_accessed": False,
                "final_holdout_accessed": False,
                "scope": "risk_normalization_only",
            }
        ]
    ):
        raise EurusdTsmSpecError("eligibility, selector, search, or seals drifted")


def _timeframe_grids(value: object) -> tuple[TsmTimeframeGrid, ...]:
    result: list[TsmTimeframeGrid] = []
    for index, raw in enumerate(_list(value, "timeframes")):
        item = _mapping(raw, f"timeframes[{index}]")
        _exact_keys(
            item,
            {
                "timeframe",
                "bid_bar_type",
                "ask_bar_type",
                "lookback_bars",
                "refresh_interval_bars",
            },
            f"timeframes[{index}]",
        )
        result.append(
            TsmTimeframeGrid(
                timeframe=_string(item["timeframe"], "timeframe"),
                bid_bar_type=_string(item["bid_bar_type"], "bid_bar_type"),
                ask_bar_type=_string(item["ask_bar_type"], "ask_bar_type"),
                lookback_bars=tuple(
                    _integer(value, "lookback")
                    for value in _list(item["lookback_bars"], "lookbacks")
                ),
                refresh_interval_bars=tuple(
                    _integer(value, "refresh")
                    for value in _list(
                        item["refresh_interval_bars"], "refresh intervals"
                    )
                ),
            )
        )
    return tuple(result)


def _mapping(value: object, description: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise EurusdTsmSpecError(f"{description} must be a string-keyed mapping")
    return cast(dict[str, Any], value)


def _list(value: object, description: str) -> list[Any]:
    if not isinstance(value, list):
        raise EurusdTsmSpecError(f"{description} must be a list")
    return value


def _strings(value: object, description: str) -> tuple[str, ...]:
    return tuple(_string(item, description) for item in _list(value, description))


def _string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EurusdTsmSpecError(f"{description} must be a non-empty string")
    return value


def _integer(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EurusdTsmSpecError(f"{description} must be an integer")
    return value


def _float(value: object, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EurusdTsmSpecError(f"{description} must be numeric")
    return float(value)


def _optional_float(value: object, description: str) -> float | None:
    return None if value is None else _float(value, description)


def _exact_keys(value: dict[str, Any], expected: set[str], description: str) -> None:
    if set(value) != expected:
        raise EurusdTsmSpecError(f"{description} fields are not exact")


def _utc(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")
