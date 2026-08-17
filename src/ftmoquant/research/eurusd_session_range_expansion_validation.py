"""One-shot, candidate-only validation adapter for
``eurusd_session_range_expansion_v1``.

Importing this module never opens validation data.  The CLI admits exactly the
frozen DEVELOPMENT winner and writes to one fixed output directory.  It never
calls :func:`ftmoquant.research.g1.search.run_search` or
:func:`ftmoquant.research.g1.selector.select_candidate` -- there is no search
or selection surface here, only one authorized evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

from nautilus_trader.model import Bar, CurrencyPair
from nautilus_trader.persistence import ParquetDataCatalog

from ftmoquant.backtest.execution_harness import (
    _instrument_for_profile,
    canonical_execution_profile,
)
from ftmoquant.data.universe_readiness import SPLIT_VIEW_FILENAME, _sha256_tree
from ftmoquant.research.eurusd_session_range_expansion_development import (
    BASE_RESEARCH_UNITS,
    EurusdSessionRangeExpansionFamily,
    _build_scaled_instructions,
    _development_root,
    _run_native_fold,
)
from ftmoquant.research.eurusd_session_range_expansion_spec import (
    EURUSD_SESSION_RANGE_EXPANSION_SEMANTIC_SHA256,
)
from ftmoquant.research.eurusd_tsm_development import (
    NativeFoldResult,
    _canonical_bytes,
    _daily_midpoints_from_minute_bars,
    _fold_metrics,
    _minute_execution_frames,
    _ns,
    _validate_bar_stream,
)
from ftmoquant.research.g1.guards import WalkForwardWindow
from ftmoquant.research.stage_g import (
    DEVELOPMENT_START,
    FROZEN_INSTRUMENT_IDS,
    FROZEN_UNIVERSE_PLAN_SHA256,
    FROZEN_UNIVERSE_READINESS_SHA256,
    HOLDOUT_START,
    VALIDATION_START,
    open_development_context,
)
from ftmoquant.research.ts_momentum_development import (
    DevelopmentEvaluationConfig,
    _validate_catalog_trees,
    load_development_evaluation_config,
)

PROTOCOL_PATH = Path(
    "config/validation/eurusd_session_range_expansion_v1_one_shot.json"
)
PROTOCOL_SEMANTIC_SHA256 = (
    "689f08bc55bf2a81bf9f04066030ea51870b030eac751645bab3742ff9cb1f07"
)
SELECTED_TRIAL_ID = "cce6c53bd97b188ae3bf735f3ce52b9bfbdc3dc2ff7aaa456a40758a21b4bc69"
SELECTED_CANDIDATE_SHA256 = (
    "7b8d42b30f04dc5694550a28825fa92f1eb548fdf1faa808b1e5add9efdacf0f"
)
TRIAL_REGISTRY_SHA256 = (
    "9b31ee9630c40c0858d7df56a4ab5b44f01c664faa24f8874e4593e48d2f01de"
)
DEVELOPMENT_RESULT_SHA256 = (
    "29624afabebb7f68583871805831f01d25e0fddb1a3d00bd45bddb19ee4aeb27"
)
VALIDATION_OUTPUT_DIR = Path(
    ".artifacts/g1_4h/eurusd_session_range_expansion_v1/validation_run"
)
VALIDATION_RESULT_SCHEMA = (
    "ftmoquant.eurusd-session-range-expansion-v1-one-shot-validation-result"
)
_EURUSD = "EUR/USD.DUKASCOPY"
_SELECTED_PARAMETERS: dict[str, str] = {
    "breakout_window_end": "11:00",
    "scheduled_exit": "17:00",
}


class EurusdSessionRangeExpansionValidationError(ValueError):
    """Raised before or during the one authorized validation evaluation."""


@dataclass(frozen=True, slots=True)
class ValidationProtocol:
    semantic_sha256: str
    selected_trial_id: str
    selected_parameters: Mapping[str, str]
    selected_candidate_sha256: str
    trial_registry_sha256: str
    development_result_sha256: str
    start_utc: datetime
    end_exclusive_utc: datetime
    duration_days: int
    minimum_duration_days: int
    minimum_trades: int
    validation_split_sha256: Mapping[str, str]
    canonical_document: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ValidationMetrics:
    fold_id: str
    completed_executed_trades: int
    normalized_turnover: float
    net_return: float
    stressed_net_return: float
    net_expectancy: float
    stressed_expectancy: float
    sharpe: float
    maximum_drawdown: float
    net_daily_mean: float
    yearly_attribution: tuple[dict[str, str | float], ...]
    realized_variable_cost_usd: float
    realized_variable_cost_return: float


@dataclass(frozen=True, slots=True)
class ValidationPreparedData:
    instrument: CurrencyPair
    validation_minute_bars: tuple[Bar, ...]
    execution_frames: tuple[tuple[int, int, Decimal], ...]
    daily_midpoints: tuple[Any, ...]
    validation_manifest_sha256: str
    validation_catalog_tree_sha256: str


def load_validation_protocol(path: Path = PROTOCOL_PATH) -> ValidationProtocol:
    document = _json_object(path, "validation protocol")
    declared = document.get("semantic_sha256")
    computed = _semantic_sha(document)
    if declared != computed or declared != PROTOCOL_SEMANTIC_SHA256:
        raise EurusdSessionRangeExpansionValidationError(
            "validation protocol semantic SHA mismatch"
        )
    family = _mapping(document.get("family"), "family")
    candidate = _mapping(document.get("selected_candidate"), "selected_candidate")
    evidence = _mapping(document.get("development_evidence"), "development_evidence")
    partition = _mapping(document.get("validation_partition"), "validation_partition")
    acceptance = _mapping(document.get("acceptance"), "acceptance")
    one_shot = _mapping(document.get("one_shot"), "one_shot")
    holdout = _mapping(document.get("final_holdout"), "final_holdout")
    warmup = _mapping(
        document.get("cross_partition_session_warmup"), "cross_partition_session_warmup"
    )
    dst = _mapping(document.get("dst_semantics"), "dst_semantics")
    risk = _mapping(document.get("risk_sizing"), "risk_sizing")
    sample = _mapping(document.get("sample_count"), "sample_count")
    parameters = _mapping(candidate.get("parameters"), "selected parameters")
    validation_hashes = _mapping(
        partition.get("validation_split_semantic_sha256"),
        "validation split hashes",
    )
    start = _utc(partition.get("start_utc"), "validation start")
    end = _utc(partition.get("end_exclusive_utc"), "validation end")
    expected_gates = [
        "deterministic_evaluation_complete_and_numerically_valid",
        "completed_executed_session_breakout_round_trips_gte_50",
        "net_validation_return_gt_zero",
        "cost_stressed_validation_return_gt_zero",
        "net_expectancy_per_trade_gt_zero",
        "cost_stressed_expectancy_per_trade_gt_zero",
    ]
    if (
        document.get("schema")
        != "ftmoquant.eurusd-session-range-expansion-v1-one-shot-validation-protocol"
        or document.get("status") != "frozen_pre_validation_unobserved"
        or document.get("validation_returns_accessed") is not False
        or family
        != {
            "family_id": "eurusd_session_range_expansion_v1",
            "family_version": "1.0.0",
            "family_semantic_sha256": EURUSD_SESSION_RANGE_EXPANSION_SEMANTIC_SHA256,
        }
        or candidate.get("trial_id") != SELECTED_TRIAL_ID
        or parameters != _SELECTED_PARAMETERS
        or evidence.get("selected_candidate_sha256") != SELECTED_CANDIDATE_SHA256
        or evidence.get("trial_registry_sha256") != TRIAL_REGISTRY_SHA256
        or evidence.get("development_result_sha256") != DEVELOPMENT_RESULT_SHA256
        or one_shot.get("permitted_trial_ids") != [SELECTED_TRIAL_ID]
        or one_shot.get("maximum_configurations") != 1
        or one_shot.get("validation_search_or_selection") != "forbidden"
        or one_shot.get("alternate_candidate_after_rejection") != "forbidden"
        or one_shot.get("fixed_output_directory") != str(VALIDATION_OUTPUT_DIR)
        or holdout.get("locked") is not True
        or holdout.get("accessed") is not False
        or start != VALIDATION_START
        or end != HOLDOUT_START
        or partition.get("partition") != "validation"
        or partition.get("duration_days") != 498
        or partition.get("minimum_duration_days_for_frozen_threshold") != 365
        or (end - start).days != partition.get("duration_days")
        or acceptance.get("minimum_completed_executed_trades") != 50
        or acceptance.get("gates") != expected_gates
        or acceptance.get("all_required") is not True
        or warmup.get("session_or_range_state_reset_at_validation_start") is not False
        or warmup.get("development_execution_or_pnl_counted_in_validation") is not False
        or warmup.get("only_validation_interval_signals_may_generate_validation_trades")
        is not True
        or warmup.get(
            "first_validation_day_may_use_a_range_partly_formed_from_development_"
            "observations"
        )
        is not True
        or dst.get("short_or_long_dst_transition_days")
        != "remain_invalid_no_trade_not_normalized_or_fixed_for_validation"
        or risk.get("in_trade_resizing") != "none"
        or risk.get("exit_target_exposure") != 0
        or sample.get("definition") != "completed_executed_session_breakout_round_trips"
        or sample.get("one_entry_plus_scheduled_exit_equals_one_trade") is not True
    ):
        raise EurusdSessionRangeExpansionValidationError(
            "validation protocol semantics drifted"
        )
    return ValidationProtocol(
        semantic_sha256=cast(str, declared),
        selected_trial_id=SELECTED_TRIAL_ID,
        selected_parameters=cast(dict[str, str], parameters),
        selected_candidate_sha256=SELECTED_CANDIDATE_SHA256,
        trial_registry_sha256=TRIAL_REGISTRY_SHA256,
        development_result_sha256=DEVELOPMENT_RESULT_SHA256,
        start_utc=start,
        end_exclusive_utc=end,
        duration_days=cast(int, partition["duration_days"]),
        minimum_duration_days=cast(
            int, partition["minimum_duration_days_for_frozen_threshold"]
        ),
        minimum_trades=cast(int, acceptance["minimum_completed_executed_trades"]),
        validation_split_sha256=cast(dict[str, str], validation_hashes),
        canonical_document=document,
    )


def verify_development_evidence(
    protocol: ValidationProtocol,
    *,
    selected_candidate_path: Path,
    trial_registry_path: Path,
    development_result_path: Path,
) -> dict[str, Any]:
    for path, expected, label in (
        (
            selected_candidate_path,
            protocol.selected_candidate_sha256,
            "selected candidate",
        ),
        (trial_registry_path, protocol.trial_registry_sha256, "trial registry"),
        (
            development_result_path,
            protocol.development_result_sha256,
            "DEVELOPMENT result",
        ),
    ):
        if _sha256(path) != expected:
            raise EurusdSessionRangeExpansionValidationError(
                f"{label} artifact SHA mismatch"
            )
    candidate = _json_object(selected_candidate_path, "selected candidate")
    _verify_selected_candidate_document(candidate, protocol)
    registry = _json_object(trial_registry_path, "trial registry")
    result = _json_object(development_result_path, "DEVELOPMENT result")
    if (
        registry.get("family_id") != "eurusd_session_range_expansion_v1"
        or registry.get("selection", {}).get("selected_trial_id") != SELECTED_TRIAL_ID
        or result.get("selected_trial_id") != SELECTED_TRIAL_ID
        or result.get("family_semantic_sha256")
        != EURUSD_SESSION_RANGE_EXPANSION_SEMANTIC_SHA256
    ):
        raise EurusdSessionRangeExpansionValidationError(
            "DEVELOPMENT evidence identity mismatch"
        )
    matching = [
        item
        for item in cast(list[dict[str, Any]], registry.get("trials"))
        if item.get("trial_id") == SELECTED_TRIAL_ID
    ]
    if len(matching) != 1 or dict(matching[0].get("parameters", [])) != dict(
        _SELECTED_PARAMETERS
    ):
        raise EurusdSessionRangeExpansionValidationError(
            "selected trial registry identity mismatch"
        )
    return {"candidate": candidate, "trial": matching[0], "result": result}


def validation_passes(metrics: ValidationMetrics, protocol: ValidationProtocol) -> bool:
    numerical = (
        metrics.normalized_turnover,
        metrics.net_return,
        metrics.stressed_net_return,
        metrics.net_expectancy,
        metrics.stressed_expectancy,
        metrics.sharpe,
        metrics.maximum_drawdown,
        metrics.net_daily_mean,
        metrics.realized_variable_cost_usd,
        metrics.realized_variable_cost_return,
    )
    return (
        all(math.isfinite(item) for item in numerical)
        and metrics.completed_executed_trades >= protocol.minimum_trades
        and metrics.net_return > 0.0
        and metrics.stressed_net_return > 0.0
        and metrics.net_expectancy > 0.0
        and metrics.stressed_expectancy > 0.0
        and metrics.normalized_turnover >= 0.0
        and metrics.maximum_drawdown >= 0.0
        and metrics.realized_variable_cost_usd >= 0.0
    )


def evaluate_prepared_validation(
    prepared: ValidationPreparedData,
    protocol: ValidationProtocol,
    evaluation_config: DevelopmentEvaluationConfig,
) -> tuple[ValidationMetrics, NativeFoldResult]:
    family = EurusdSessionRangeExpansionFamily()
    parameters = dict(protocol.selected_parameters)
    family.validate_parameters(parameters)
    window = _validation_window(protocol)
    instructions = _build_scaled_instructions(
        family,
        parameters,
        prepared.execution_frames,
        prepared.daily_midpoints,
        window,
    )
    native = _run_native_fold(
        fold_id=window.fold_id,
        instrument=prepared.instrument,
        minute_bars=prepared.validation_minute_bars,
        instructions=instructions,
        config=evaluation_config,
        start_ns=_ns(protocol.start_utc),
        end_exclusive_ns=_ns(protocol.end_exclusive_utc),
    )
    fold = _fold_metrics(native, window)
    count = native.executed_raw_alpha_transitions
    net_return = float(
        (native.equity_points[-1].equity - native.equity_points[0].equity)
        / BASE_RESEARCH_UNITS
    )
    cost_return = float(native.realized_variable_cost / BASE_RESEARCH_UNITS)
    stressed_return = net_return - 0.5 * cost_return
    metrics = ValidationMetrics(
        fold_id=fold.fold_id,
        completed_executed_trades=count,
        normalized_turnover=fold.turnover,
        net_return=net_return,
        stressed_net_return=stressed_return,
        net_expectancy=net_return / count if count else 0.0,
        stressed_expectancy=stressed_return / count if count else 0.0,
        sharpe=fold.sharpe,
        maximum_drawdown=fold.drawdown,
        net_daily_mean=fold.net_daily_mean or 0.0,
        yearly_attribution=tuple(asdict(item) for item in fold.yearly_attribution),
        realized_variable_cost_usd=float(native.realized_variable_cost),
        realized_variable_cost_return=cost_return,
    )
    return metrics, native


def prepare_validation_market_data(
    *,
    protocol: ValidationProtocol,
    universe_readiness_path: Path,
    development_roots: Mapping[str, Path],
    validation_root: Path,
) -> ValidationPreparedData:
    """Open validation only after every non-data preflight has passed."""

    context = open_development_context(universe_readiness_path, development_roots)
    _validate_catalog_trees(context, development_roots)
    readiness = _json_object(universe_readiness_path, "universe readiness")
    if readiness.get("semantic_sha256") != FROZEN_UNIVERSE_READINESS_SHA256:
        raise EurusdSessionRangeExpansionValidationError(
            "universe readiness SHA mismatch"
        )
    refs = {
        item.get("instrument_id"): item.get("validation_split_sha256")
        for item in cast(list[dict[str, Any]], readiness.get("instrument_artifacts"))
    }
    if refs != dict(protocol.validation_split_sha256):
        raise EurusdSessionRangeExpansionValidationError(
            "validation split readiness identity mismatch"
        )
    root = validation_root.resolve()
    _reject_final_holdout_path(root)
    manifest_path = root / SPLIT_VIEW_FILENAME
    manifest = _json_object(manifest_path, "EUR/USD validation split")
    expected_manifest_sha = protocol.validation_split_sha256[_EURUSD]
    expected_range = {
        "start_utc": _format_utc(protocol.start_utc),
        "end_exclusive_utc": _format_utc(protocol.end_exclusive_utc),
    }
    if (
        manifest.get("semantic_sha256") != _semantic_sha(manifest)
        or manifest.get("semantic_sha256") != expected_manifest_sha
        or manifest.get("instrument_id") != _EURUSD
        or manifest.get("universe_plan_sha256") != FROZEN_UNIVERSE_PLAN_SHA256
        or manifest.get("split") != "validation"
        or manifest.get("range") != expected_range
        or manifest.get("holdout_rows") != 0
        or manifest.get("network_access_required") is not False
        or manifest.get("access_policy")
        != "encrypted_normally_unmounted_validation_runner_read_only"
    ):
        raise EurusdSessionRangeExpansionValidationError(
            "sealed validation manifest is incompatible"
        )
    declared_tree = manifest.get("catalog_tree_sha256")
    if not isinstance(declared_tree, str) or _sha256_tree(root / "catalog") != (
        declared_tree
    ):
        raise EurusdSessionRangeExpansionValidationError(
            "validation catalog tree SHA mismatch"
        )
    development_catalog = ParquetDataCatalog(
        str(development_roots[_EURUSD].resolve() / "catalog")
    )
    validation_catalog = ParquetDataCatalog(str(root / "catalog"))
    found = validation_catalog.instruments([_EURUSD])
    if len(found) != 1 or not isinstance(found[0], CurrencyPair):
        raise EurusdSessionRangeExpansionValidationError(
            "validation EUR/USD instrument is unavailable"
        )
    instrument = _instrument_for_profile(found[0], canonical_execution_profile().fee)
    development_minute = _query_minute_bars(
        development_catalog, DEVELOPMENT_START, protocol.start_utc
    )
    validation_minute = _query_minute_bars(
        validation_catalog, protocol.start_utc, protocol.end_exclusive_utc
    )
    combined_minute = tuple(
        sorted(
            (*development_minute, *validation_minute),
            key=lambda bar: (bar.ts_event, str(bar.bar_type)),
        )
    )
    execution_frames = _minute_execution_frames(combined_minute)
    daily = _daily_midpoints_from_minute_bars(combined_minute)
    if (
        not execution_frames
        or execution_frames[0][1] >= _ns(protocol.start_utc)
        or execution_frames[-1][1] >= _ns(protocol.end_exclusive_utc)
    ):
        raise EurusdSessionRangeExpansionValidationError(
            "causal validation warm-up boundaries failed"
        )
    if _sha256_tree(root / "catalog") != declared_tree:
        raise EurusdSessionRangeExpansionValidationError(
            "validation catalog changed during read"
        )
    _validate_catalog_trees(context, development_roots)
    return ValidationPreparedData(
        instrument=instrument,
        validation_minute_bars=validation_minute,
        execution_frames=execution_frames,
        daily_midpoints=daily,
        validation_manifest_sha256=expected_manifest_sha,
        validation_catalog_tree_sha256=declared_tree,
    )


def run_eurusd_session_range_expansion_validation(
    *,
    protocol_path: Path,
    selected_candidate_path: Path,
    trial_registry_path: Path,
    development_result_path: Path,
    universe_readiness_path: Path,
    development_roots: Mapping[str, Path],
    validation_root: Path,
    evaluation_config_path: Path,
    output_dir: Path = VALIDATION_OUTPUT_DIR,
) -> dict[str, Any]:
    if output_dir != VALIDATION_OUTPUT_DIR or output_dir.exists():
        raise EurusdSessionRangeExpansionValidationError(
            "one-shot validation output is fixed and must not already exist"
        )
    _require_clean_worktree()
    protocol = load_validation_protocol(protocol_path)
    if protocol.duration_days < protocol.minimum_duration_days:
        raise EurusdSessionRangeExpansionValidationError(
            "validation period is too short for the frozen trade-count threshold"
        )
    if (
        EURUSD_SESSION_RANGE_EXPANSION_SEMANTIC_SHA256
        != protocol.canonical_document["family"]["family_semantic_sha256"]
    ):
        raise EurusdSessionRangeExpansionValidationError("family semantic SHA mismatch")
    evidence = verify_development_evidence(
        protocol,
        selected_candidate_path=selected_candidate_path,
        trial_registry_path=trial_registry_path,
        development_result_path=development_result_path,
    )
    config = load_development_evaluation_config(evaluation_config_path)
    prepared = prepare_validation_market_data(
        protocol=protocol,
        universe_readiness_path=universe_readiness_path,
        development_roots=development_roots,
        validation_root=validation_root,
    )
    metrics, _ = evaluate_prepared_validation(prepared, protocol, config)
    passed = validation_passes(metrics, protocol)
    development_diagnostics = _development_diagnostics(evidence["trial"])
    diagnostics = {
        "validation_to_development_expectancy": _ratio(
            metrics.net_expectancy,
            development_diagnostics["pooled_expectancy"],
        ),
        "validation_to_development_stressed_expectancy": _ratio(
            metrics.stressed_expectancy,
            development_diagnostics["stressed_expectancy"],
        ),
        "validation_sharpe_vs_development_mean_fold_sharpe": {
            "validation": metrics.sharpe,
            "development": development_diagnostics["mean_fold_sharpe"],
        },
        "validation_drawdown_vs_development_max_drawdown": {
            "validation": metrics.maximum_drawdown,
            "development": development_diagnostics["max_drawdown"],
        },
        "gating": False,
    }
    payload = {
        "schema": VALIDATION_RESULT_SCHEMA,
        "schema_version": 1,
        "protocol_semantic_sha256": protocol.semantic_sha256,
        "family_semantic_sha256": EURUSD_SESSION_RANGE_EXPANSION_SEMANTIC_SHA256,
        "selected_trial_id": SELECTED_TRIAL_ID,
        "parameters": dict(protocol.selected_parameters),
        "validation_partition": {
            "start_utc": _format_utc(protocol.start_utc),
            "end_exclusive_utc": _format_utc(protocol.end_exclusive_utc),
            "duration_days": protocol.duration_days,
            "manifest_sha256": prepared.validation_manifest_sha256,
            "catalog_tree_sha256": prepared.validation_catalog_tree_sha256,
        },
        "metrics": asdict(metrics),
        "acceptance": {
            "outcome": "VALIDATION_PASSED" if passed else "VALIDATION_REJECTED",
            "all_gates_passed": passed,
            "minimum_trades": protocol.minimum_trades,
        },
        "diagnostics_vs_development": diagnostics,
        "development_warmup_pnl_carried": False,
        "final_holdout_accessed": False,
    }
    _write_validation_artifacts(output_dir, payload)
    return payload


def _query_minute_bars(
    catalog: ParquetDataCatalog, start: datetime, end: datetime
) -> tuple[Bar, ...]:
    start_ns = _ns(start)
    end_ns = _ns(end)
    bid_type = f"{_EURUSD}-1-MINUTE-BID-EXTERNAL"
    ask_type = f"{_EURUSD}-1-MINUTE-ASK-EXTERNAL"
    bid = tuple(catalog.query_bars([bid_type], start=start_ns, end=end_ns - 1))
    ask = tuple(catalog.query_bars([ask_type], start=start_ns, end=end_ns - 1))
    _validate_bar_stream(bid, bid_type, start_ns, end_ns)
    _validate_bar_stream(ask, ask_type, start_ns, end_ns)
    return tuple(
        sorted((*bid, *ask), key=lambda bar: (bar.ts_event, str(bar.bar_type)))
    )


def _validation_window(protocol: ValidationProtocol) -> WalkForwardWindow:
    return WalkForwardWindow(
        "validation_one_shot",
        DEVELOPMENT_START,
        protocol.start_utc,
        protocol.start_utc,
        protocol.end_exclusive_utc,
    )


def _verify_selected_candidate_document(
    candidate: Mapping[str, Any], protocol: ValidationProtocol
) -> None:
    if (
        candidate.get("family_id") != "eurusd_session_range_expansion_v1"
        or candidate.get("family_version") != "1.0.0"
        or candidate.get("family_semantic_sha256")
        != EURUSD_SESSION_RANGE_EXPANSION_SEMANTIC_SHA256
        or candidate.get("selected_trial_id") != protocol.selected_trial_id
        or candidate.get("parameters") != dict(protocol.selected_parameters)
    ):
        raise EurusdSessionRangeExpansionValidationError(
            "selected candidate identity mismatch"
        )


def _development_diagnostics(trial: Mapping[str, Any]) -> dict[str, float]:
    folds = cast(list[dict[str, Any]], trial["folds"])
    count = sum(cast(int, fold["trade_count"]) for fold in folds)
    return {
        "pooled_expectancy": sum(
            cast(float, fold["net_expectancy"]) * cast(int, fold["trade_count"])
            for fold in folds
        )
        / count,
        "stressed_expectancy": sum(
            cast(float, fold["cost_stressed_expectancy"])
            * cast(int, fold["trade_count"])
            for fold in folds
        )
        / count,
        "mean_fold_sharpe": sum(cast(float, fold["sharpe"]) for fold in folds)
        / len(folds),
        "max_drawdown": max(cast(float, fold["drawdown"]) for fold in folds),
    }


def _write_validation_artifacts(output_dir: Path, payload: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    normalized = json.loads(_canonical_bytes(payload))
    semantic = hashlib.sha256(_canonical_bytes(normalized)).hexdigest()
    result_bytes = _canonical_bytes({**normalized, "semantic_sha256": semantic}) + b"\n"
    result_path = output_dir / "validation_result.json"
    result_path.write_bytes(result_bytes)
    result_sha = hashlib.sha256(result_bytes).hexdigest()
    (output_dir / "artifact_hashes.json").write_bytes(
        _canonical_bytes({result_path.name: result_sha}) + b"\n"
    )
    (output_dir / "runtime_provenance.json").write_bytes(
        _canonical_bytes(
            {
                "git_commit": _git_commit(),
                "python_module": (
                    "ftmoquant.research.eurusd_session_range_expansion_validation"
                ),
            }
        )
        + b"\n"
    )


def _semantic_sha(document: Mapping[str, Any]) -> str:
    payload = {
        key: value for key, value in document.items() if key != "semantic_sha256"
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EurusdSessionRangeExpansionValidationError(
            f"could not load {label}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise EurusdSessionRangeExpansionValidationError(
            f"{label} must be a JSON object"
        )
    return cast(dict[str, Any], value)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise EurusdSessionRangeExpansionValidationError(
            f"{label} must be a string-keyed object"
        )
    return cast(dict[str, Any], value)


def _utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise EurusdSessionRangeExpansionValidationError(f"{label} must be UTC text")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise EurusdSessionRangeExpansionValidationError(
            f"{label} is invalid"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise EurusdSessionRangeExpansionValidationError(f"{label} must be UTC")
    return parsed


def _format_utc(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0.0 else numerator / denominator


def _reject_final_holdout_path(path: Path) -> None:
    if any("holdout" in part.casefold() for part in path.parts):
        raise EurusdSessionRangeExpansionValidationError(
            "final-holdout path is forbidden"
        )


def _require_clean_worktree() -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise EurusdSessionRangeExpansionValidationError(
            "one-shot validation requires a clean worktree"
        )


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validation_root(value: str) -> Path:
    path = Path(value)
    try:
        _reject_final_holdout_path(path)
    except EurusdSessionRangeExpansionValidationError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the frozen one-shot eurusd_session_range_expansion_v1 "
            "validation candidate"
        )
    )
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--selected-candidate", type=Path, required=True)
    parser.add_argument("--trial-registry", type=Path, required=True)
    parser.add_argument("--development-result", type=Path, required=True)
    parser.add_argument("--universe-readiness", type=Path, required=True)
    parser.add_argument(
        "--development-root", type=_development_root, action="append", required=True
    )
    parser.add_argument("--validation-root", type=_validation_root, required=True)
    parser.add_argument("--evaluation-config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    pairs = cast(list[tuple[str, Path]], args.development_root)
    roots = dict(pairs)
    if len(roots) != len(pairs) or set(roots) != set(FROZEN_INSTRUMENT_IDS):
        raise EurusdSessionRangeExpansionValidationError(
            "development roots must contain the exact frozen universe"
        )
    result = run_eurusd_session_range_expansion_validation(
        protocol_path=cast(Path, args.protocol),
        selected_candidate_path=cast(Path, args.selected_candidate),
        trial_registry_path=cast(Path, args.trial_registry),
        development_result_path=cast(Path, args.development_result),
        universe_readiness_path=cast(Path, args.universe_readiness),
        development_roots=roots,
        validation_root=cast(Path, args.validation_root),
        evaluation_config_path=cast(Path, args.evaluation_config),
    )
    print(_canonical_bytes(result).decode())


if __name__ == "__main__":
    main()
