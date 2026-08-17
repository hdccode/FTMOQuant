"""Strict preregistration loader for ``eurusd_session_range_expansion_v1``."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, time
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

EURUSD_SESSION_RANGE_EXPANSION_SPEC_PATH = Path(
    "config/strategies/eurusd_session_range_expansion_v1.yaml"
)
EURUSD_SESSION_RANGE_EXPANSION_SEMANTIC_SHA256 = (
    "3fc1fd836fdfe6999a0ff370e7285752768078e70dca9547723a5772f8a16585"
)


class EurusdSessionRangeExpansionSpecError(ValueError):
    """Raised when the preregistration is ambiguous, incomplete, or changed."""


@dataclass(frozen=True, slots=True)
class EurusdSessionRangeExpansionSpec:
    family_id: str
    version: str
    economic_hypothesis: str
    breakout_window_end_grid: tuple[str, ...]
    scheduled_exit_grid: tuple[str, ...]
    expected_unique_trial_count: int
    semantic_sha256: str
    canonical_document: dict[str, Any]

    @property
    def breakout_window_end_times(self) -> tuple[time, ...]:
        return tuple(_parse_hhmm(value) for value in self.breakout_window_end_grid)

    @property
    def scheduled_exit_times(self) -> tuple[time, ...]:
        return tuple(_parse_hhmm(value) for value in self.scheduled_exit_grid)

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


def load_eurusd_session_range_expansion_spec(
    path: Path = EURUSD_SESSION_RANGE_EXPANSION_SPEC_PATH,
) -> EurusdSessionRangeExpansionSpec:
    """Load and prove the exact pre-return 9-cell family protocol."""

    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise EurusdSessionRangeExpansionSpecError(
            f"could not load spec: {error}"
        ) from error
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
            "sample_count",
            "production_evaluator",
            "metric_definitions",
            "neighbours",
            "selector",
            "search",
            "sealed_partitions",
            "reuse",
            "semantic_sha256",
        },
        "spec",
    )
    semantic = semantic_sha256_for_document(document)
    if document["semantic_sha256"] != semantic:
        raise EurusdSessionRangeExpansionSpecError(
            "semantic_sha256 does not match canonical semantics"
        )
    if semantic != EURUSD_SESSION_RANGE_EXPANSION_SEMANTIC_SHA256:
        raise EurusdSessionRangeExpansionSpecError(
            "eurusd_session_range_expansion_v1 preregistration semantic SHA drifted"
        )

    family = _mapping(document["family"], "family")
    _exact_keys(
        family, {"family_id", "version", "status", "economic_hypothesis"}, "family"
    )
    if (
        document["schema_version"] != 1
        or family["family_id"] != "eurusd_session_range_expansion_v1"
        or family["version"] != "1.0.0"
        or family["status"] != "preregistered_not_run"
    ):
        raise EurusdSessionRangeExpansionSpecError(
            "family identity/status is not frozen"
        )
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
        raise EurusdSessionRangeExpansionSpecError(
            "dataset identity or DEVELOPMENT boundary drifted"
        )
    session = _mapping(document["session"], "session")
    if session != {"session_id": "all", "optimized": False}:
        raise EurusdSessionRangeExpansionSpecError("session must be frozen to All")

    parameter_grid = _mapping(document["parameter_grid"], "parameter_grid")
    _exact_keys(
        parameter_grid,
        {
            "session_timezone",
            "range_start",
            "range_end",
            "breakout_window_end",
            "scheduled_exit",
            "combination_rule",
            "expected_unique_trial_count",
            "design_note",
        },
        "parameter_grid",
    )
    if (
        parameter_grid["session_timezone"] != "Europe/London"
        or parameter_grid["range_start"] != "00:00"
        or parameter_grid["range_end"] != "08:00"
    ):
        raise EurusdSessionRangeExpansionSpecError("range window is not frozen")
    breakout_grid = tuple(
        _string(item, "breakout_window_end")
        for item in _list(parameter_grid["breakout_window_end"], "breakout grid")
    )
    exit_grid = tuple(
        _string(item, "scheduled_exit")
        for item in _list(parameter_grid["scheduled_exit"], "exit grid")
    )
    if breakout_grid != ("11:00", "12:00", "13:00"):
        raise EurusdSessionRangeExpansionSpecError("breakout_window_end grid drifted")
    if exit_grid != ("15:00", "16:00", "17:00"):
        raise EurusdSessionRangeExpansionSpecError("scheduled_exit grid drifted")
    if parameter_grid["combination_rule"] != "full_unconditional_cartesian_product":
        raise EurusdSessionRangeExpansionSpecError("combination rule drifted")
    expected_count = _integer(
        parameter_grid["expected_unique_trial_count"], "expected trial count"
    )
    actual_count = len(breakout_grid) * len(exit_grid)
    if expected_count != 9 or actual_count != expected_count:
        raise EurusdSessionRangeExpansionSpecError(
            "exact grid must contain exactly 9 cells"
        )

    _validate_signal(document)
    _validate_risk_execution(document)
    _validate_folds(document)
    _validate_evaluator_sample_metrics(document)
    _validate_selection_and_seals(document)
    return EurusdSessionRangeExpansionSpec(
        family_id="eurusd_session_range_expansion_v1",
        version="1.0.0",
        economic_hypothesis=hypothesis,
        breakout_window_end_grid=breakout_grid,
        scheduled_exit_grid=exit_grid,
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
    if (
        signal.get("price") != "synchronized_completed_1m_bid_ask_midpoint_close"
        or signal.get("range_rule")
        != "max_min_of_completed_1m_midpoints_in_range_window"
        or signal.get("range_completeness")
        != "exactly_480_consecutive_completed_1m_midpoints"
        or signal.get("incomplete_or_invalid_range") != "no_trade"
        or signal.get("first_close_above_range_high_target") != 1
        or signal.get("first_close_below_range_low_target") != -1
        or signal.get("no_breakout_target") != 0
        or signal.get("maximum_entries_per_london_day") != 1
        or signal.get("exit_target") != 0
        or signal.get("entry_hold_rule") != "hold_until_scheduled_exit"
        or signal.get("signals_while_position_or_pending") != "ignore"
    ):
        raise EurusdSessionRangeExpansionSpecError("signal semantics drifted")


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
            "pathological_volatility_behavior": ("fail_closed_zero_nonzero_exposure"),
            "exposure_formula": (
                "directional_signal_times_0.01_divided_by_ex_ante_annualized_volatility"
            ),
            "sizing_refresh_rule": "sized_once_at_entry_no_in_trade_resizing",
            "base_units_per_unit_exposure": "100000",
            "target_quantity_rule": (
                "desired_exposure_times_base_units_before_native_execution"
            ),
            "ftmo_or_g4_sizing": False,
        }
        or execution.get("engine") != "existing_nautilus_g0_7_bid_ask_boundary"
        or execution.get("entry_timing_rule")
        != "first_eligible_information_time_strictly_after_signal_information_time"
        or execution.get("exit_timing_rule")
        != "first_eligible_information_time_strictly_after_exit_signal_information_time"
        or execution.get("max_open_positions") != 1
        or execution.get("overlapping_breakouts_while_positioned_or_pending")
        != "ignored"
        or execution.get("stop_loss") != "none"
        or execution.get("take_profit") != "none"
        or execution.get("spread") != "observed_paired_bid_ask"
        or execution.get("cost_stress_multiplier") != 1.5
        or execution.get("cost_stress_formula")
        != "net_return_minus_one_half_realized_base_cost"
        or execution.get("position_sizing_timing") != "before_native_order_submission"
        or execution.get("native_cost_quantity_basis")
        != "actual_executed_scaled_delta_base_units"
        or execution.get("turnover_basis") != "actual_executed_scaled_target_changes"
        or execution.get("adverse_execution_perturbation")
        != "unused_no_generic_implementation"
        or execution.get("parallel_backtester") is not False
        or execution.get("ftmo_optimization") is not False
    ):
        raise EurusdSessionRangeExpansionSpecError(
            "risk/execution/cost semantics drifted"
        )


def _validate_folds(document: dict[str, Any]) -> None:
    declared = _mapping(document["development_folds"], "development_folds")
    folds = frozen_development_folds()
    if (
        declared.get("version") != folds.version
        or declared.get("semantic_sha256") != folds.semantic_sha256
    ):
        raise EurusdSessionRangeExpansionSpecError("DEVELOPMENT fold identity drifted")
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
        raise EurusdSessionRangeExpansionSpecError("declared fold timestamps drifted")


def _validate_selection_and_seals(document: dict[str, Any]) -> None:
    eligibility = _mapping(document["eligibility"], "eligibility")
    selector = _mapping(document["selector"], "selector")
    policy = _mapping(selector.get("policy"), "selector.policy")
    search = _mapping(document["search"], "search")
    seals = _mapping(document["sealed_partitions"], "sealed_partitions")
    neighbours = _mapping(document["neighbours"], "neighbours")
    if (
        eligibility.get("required_fold_count") != 3
        or eligibility.get("minimum_positive_folds") != 2
        or eligibility.get("minimum_pooled_executed_transitions") != 100
        or eligibility.get("maximum_drawdown_hard_gate") is not False
        or eligibility.get("plateau_hard_gate") is not False
        or eligibility.get("year_concentration_hard_gate") is not False
        or neighbours.get("exactly_one_dimension_moves") is not True
        or neighbours.get("adjacent_steps_only") is not True
        or neighbours.get("ordered_dimensions")
        != ["breakout_window_end", "scheduled_exit"]
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
    ):
        raise EurusdSessionRangeExpansionSpecError(
            "eligibility, neighbours, selector, search, or seals drifted"
        )


def _validate_evaluator_sample_metrics(document: dict[str, Any]) -> None:
    sample = _mapping(document["sample_count"], "sample_count")
    evaluator = _mapping(document["production_evaluator"], "production_evaluator")
    metrics = _mapping(document["metric_definitions"], "metric_definitions")
    if (
        sample.get("fold_metrics_trade_count")
        != "completed_executed_session_breakout_round_trips"
        or sample.get("definition")
        != "one_entry_plus_its_eventual_scheduled_exit_equals_one_evidential_trade"
        or sample.get("count_entry_and_exit_separately") is not False
        or sample.get("count_ignored_second_breakouts") is not False
        or sample.get("count_unexecuted_signals") is not False
        or sample.get("count_orders") is not False
        or sample.get("count_fills") is not False
        or sample.get("count_bars") is not False
        or sample.get("count_risk_only_rebalances") is not False
        or sample.get("turnover_includes_all_scaled_executed_quantity_changes")
        is not True
        or sample.get("costs_and_pnl_include_all_scaled_rebalances") is not True
        or evaluator.get("version")
        != "eurusd-session-range-expansion-v1-development-evaluator-1"
        or evaluator.get("module")
        != "ftmoquant.research.eurusd_session_range_expansion_development"
        or evaluator.get("generic_search_engine")
        != "ftmoquant.research.g1.search.run_search"
        or evaluator.get("end_boundary")
        != "final_available_tradable_observation_strictly_before_evaluate_end"
        or evaluator.get("read_beyond_evaluate_end") is not False
        or metrics.get("trade_count")
        != "completed_executed_session_breakout_round_trips"
        or metrics.get("execution_perturbed_expectancy") is not None
    ):
        raise EurusdSessionRangeExpansionSpecError(
            "sample count, production evaluator, or metric definitions drifted"
        )


def _parse_hhmm(value: str) -> time:
    hour, _, minute = value.partition(":")
    return time(int(hour), int(minute))


def _mapping(value: object, description: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise EurusdSessionRangeExpansionSpecError(
            f"{description} must be a string-keyed mapping"
        )
    return cast(dict[str, Any], value)


def _list(value: object, description: str) -> list[Any]:
    if not isinstance(value, list):
        raise EurusdSessionRangeExpansionSpecError(f"{description} must be a list")
    return value


def _strings(value: object, description: str) -> tuple[str, ...]:
    return tuple(_string(item, description) for item in _list(value, description))


def _string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EurusdSessionRangeExpansionSpecError(
            f"{description} must be a non-empty string"
        )
    return value


def _integer(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EurusdSessionRangeExpansionSpecError(f"{description} must be an integer")
    return value


def _float(value: object, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EurusdSessionRangeExpansionSpecError(f"{description} must be numeric")
    return float(value)


def _optional_float(value: object, description: str) -> float | None:
    return None if value is None else _float(value, description)


def _exact_keys(value: dict[str, Any], expected: set[str], description: str) -> None:
    if set(value) != expected:
        raise EurusdSessionRangeExpansionSpecError(
            f"{description} fields are not exact"
        )


def _utc(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")
