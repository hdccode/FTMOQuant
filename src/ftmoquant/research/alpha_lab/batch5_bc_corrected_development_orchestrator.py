"""Corrected DEVELOPMENT-only orchestration for frozen Batch 5B and Batch 5C.

This runner applies only the three versioned implementation corrections.  It
never executes or scores B5A and has no strategy or parameter override surface.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ftmoquant.research.alpha_lab.batch5_bc_correction_protocol import (
    EXPECTED_CORRECTION_PROTOCOL_SEMANTIC_SHA256,
    ORIGINAL_ARTIFACT_HASH_MANIFEST_SHA256,
    verify_correction_protocol,
)
from ftmoquant.research.alpha_lab.batch5_daily import (
    CompletedFxDay,
    FxDayBuildDiagnostics,
    build_completed_fx_days_with_diagnostics,
)
from ftmoquant.research.alpha_lab.batch5_development_orchestrator import (
    COST_MULTIPLIERS,
    DEVELOPMENT_FOLD_BOUNDARIES,
    Batch5DevelopmentOrchestratorError,
    CostFrameCache,
    _b5b_frequency,
    _read_json,
    _require_stress_timing_identity,
    _semantic_sha,
    _trades,
    _verify_cross_readiness,
    reject_forbidden_root,
    reserve_output_directory,
)
from ftmoquant.research.alpha_lab.batch5_development_scorecard import (
    DevelopmentSleeveInput,
    DevelopmentSleeveScorecard,
    build_diagnostics_summary,
    evaluate_development_sleeve,
)
from ftmoquant.research.alpha_lab.batch5_execution import (
    Batch5SkipRecord,
    Batch5TradeResult,
)
from ftmoquant.research.alpha_lab.batch5_preregistration import (
    B5C_INSTRUMENTS,
    EXPECTED_PREREGISTRATION_SEMANTIC_SHA256,
    FAMILY_B5B,
    FAMILY_B5C,
    verify_preregistration,
)
from ftmoquant.research.alpha_lab.batch5_screen import (
    FrequencyStats,
    evaluate_family,
    load_frozen_policy,
)
from ftmoquant.research.alpha_lab.batch5b_direct_mr_execution import (
    build_position_intents,
    execute_positions,
)
from ftmoquant.research.alpha_lab.batch5b_direct_mr_signals import generate_signals
from ftmoquant.research.alpha_lab.batch5c_daily_reversal_execution import (
    execute_events,
)
from ftmoquant.research.alpha_lab.batch5c_daily_reversal_signals import (
    build_next_day_intents,
    generate_events,
)
from ftmoquant.research.alpha_lab.cost_stress import widen_bid_ask_frame
from ftmoquant.research.alpha_lab.data import _discover_oanda_universe
from ftmoquant.research.stage_g import DEVELOPMENT_END_EXCLUSIVE, DEVELOPMENT_START

ARTIFACT_ROOT = Path(".artifacts/alpha_lab/batch5_bc_corrected_development_v1")
ORIGINAL_ARTIFACT_ROOT = Path(
    ".artifacts/alpha_lab/batch5_three_fx_alpha_families_v1"
)
BC_FAMILIES = (FAMILY_B5B, FAMILY_B5C)
EXPECTED_SLEEVES = {
    "B5B_AUDCAD",
    *(f"B5C_{item.split('.')[0].replace('/', '')}" for item in B5C_INSTRUMENTS),
}
DailyBuilder = Callable[
    [str, pd.DataFrame, pd.DataFrame],
    tuple[tuple[CompletedFxDay, ...], FxDayBuildDiagnostics],
]


class Batch5BCCorrectedOrchestratorError(Batch5DevelopmentOrchestratorError):
    """Raised before unsafe access or on corrected-run identity drift."""


class _SignalTimestamped(Protocol):
    @property
    def signal_timestamp(self) -> datetime: ...


def partition_development_signals[SignalT: _SignalTimestamped](
    rows: Sequence[SignalT],
) -> tuple[SignalT, ...]:
    """Admit signal formation warm-up, but never pre-DEVELOPMENT execution."""

    if any(row.signal_timestamp.tzinfo is None for row in rows):
        raise Batch5BCCorrectedOrchestratorError(
            "executable signal timestamps must be timezone-aware"
        )
    return tuple(
        row
        for row in rows
        if DEVELOPMENT_START <= row.signal_timestamp < DEVELOPMENT_END_EXCLUSIVE
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_original_artifact(root: Path = ORIGINAL_ARTIFACT_ROOT) -> None:
    reject_forbidden_root(root, label="original Batch 5 artifact")
    manifest = root / "artifact_hashes.json"
    if (
        not manifest.is_file()
        or _sha256_file(manifest) != ORIGINAL_ARTIFACT_HASH_MANIFEST_SHA256
    ):
        raise Batch5BCCorrectedOrchestratorError(
            "original Batch 5 artifact hash-manifest identity drift"
        )
    hashes = _read_json(manifest, label="original Batch 5 artifact hashes")
    for filename, expected in hashes.items():
        path = root / filename
        if not path.is_file() or _sha256_file(path) != expected:
            raise Batch5BCCorrectedOrchestratorError(
                f"original Batch 5 artifact member drift: {filename}"
            )


def verify_corrected_preflight(
    *,
    development_root: Path,
    universe_readiness: Path,
    batch5_cross_root: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Verify frozen/corrected identities and DEVELOPMENT-only OANDA roots."""

    for label, path in (
        ("--development-root", development_root),
        ("--universe-readiness", universe_readiness),
        ("--batch5-cross-root", batch5_cross_root),
    ):
        reject_forbidden_root(path, label=label)
    methodology = verify_preregistration()
    load_frozen_policy()
    correction = verify_correction_protocol()
    _verify_original_artifact()
    cross_readiness, _ = _verify_cross_readiness(batch5_cross_root)
    canonical = _read_json(universe_readiness, label="canonical OANDA readiness")
    if (
        canonical.get("semantic_sha256") != _semantic_sha(canonical)
        or canonical.get("readiness_version") != "oanda-alpha-lab-readiness-1"
        or canonical.get("lineage_id") != "oanda_fx_alpha_lab_v1"
        or canonical.get("development_start_utc") != "2019-03-11T00:00:00Z"
        or canonical.get("development_end_exclusive_utc")
        != "2023-04-11T00:00:00Z"
        or canonical.get("research_ready") is not True
        or canonical.get("holdout_accessed") is not False
        or canonical.get("holdout_rows_admitted") != 0
    ):
        raise Batch5BCCorrectedOrchestratorError(
            "canonical OANDA readiness identity drift"
        )
    instruments, _ = _discover_oanda_universe(universe_readiness, development_root)
    expected = (
        "AUD/USD.OANDA",
        "EUR/USD.OANDA",
        "GBP/USD.OANDA",
        "NZD/USD.OANDA",
        "USD/CAD.OANDA",
        "USD/CHF.OANDA",
        "USD/JPY.OANDA",
    )
    if instruments != expected:
        raise Batch5BCCorrectedOrchestratorError("canonical OANDA universe drift")
    return methodology, correction, {"cross": cross_readiness, "canonical": canonical}


def _daily_diagnostics(value: FxDayBuildDiagnostics) -> dict[str, Any]:
    return asdict(value)


def run_corrected_b5b(
    cache: CostFrameCache,
    *,
    daily_builder: DailyBuilder | None = None,
) -> tuple[DevelopmentSleeveInput, dict[str, Any]]:
    daily_builder = daily_builder or build_completed_fx_days_with_diagnostics
    native_bid, native_ask = cache.frames("AUD/CAD.OANDA", Decimal("1.0"))
    days, daily = daily_builder("AUD/CAD.OANDA", native_bid, native_ask)
    all_signals = generate_signals(days)
    signals = partition_development_signals(all_signals)
    intents = build_position_intents(signals)
    passes: list[tuple[Batch5TradeResult, ...]] = []
    native_results: Sequence[Batch5TradeResult | Batch5SkipRecord] = ()
    for multiplier in COST_MULTIPLIERS:
        bid, ask = cache.frames("AUD/CAD.OANDA", multiplier)
        conversion_bid, conversion_ask = cache.frames("USD/CAD.OANDA", multiplier)
        results = execute_positions(
            signals,
            bid_m1=bid,
            ask_m1=ask,
            usdcad_bid_m1=conversion_bid,
            usdcad_ask_m1=conversion_ask,
        )
        if multiplier == Decimal("1.0"):
            native_results = results
        passes.append(_trades(results))
    _require_stress_timing_identity(passes)
    holding_days, changes, years = _b5b_frequency(signals)
    sleeve = DevelopmentSleeveInput(
        FAMILY_B5B,
        "B5B_FROZEN_DIRECT_AUDCAD_MR",
        "B5B_AUDCAD",
        "AUD/CAD.OANDA",
        passes[0],
        passes[1],
        passes[2],
        DEVELOPMENT_FOLD_BOUNDARIES,
        FrequencyStats(
            daily_holding_observation_count=holding_days,
            position_sign_change_count=changes,
            rollover_supported=False,
            active_year_count=years,
        ),
        holding_days,
        years,
    )
    diagnostics = {
        **_daily_diagnostics(daily),
        "days_with_20_close_statistic": max(0, len(days) - 19),
        "all_formed_signal_count": len(all_signals),
        "warmup_signal_count": len(all_signals) - len(signals),
        "executable_signal_count": len(signals),
        "non_flat_signal_count": sum(row.direction != "FLAT" for row in signals),
        "sign_change_count": changes,
        "intended_trade_count": len(intents),
        "execution_skips": dict(
            sorted(
                Counter(
                    row.skip_reason
                    for row in native_results
                    if isinstance(row, Batch5SkipRecord)
                ).items()
            )
        ),
        "completed_trade_count": len(passes[0]),
        "rollover_supported": False,
        "rollover_effect": "frequency_gate_only; price_trade_generation_not_suppressed",
    }
    return sleeve, diagnostics


def run_corrected_b5c(
    cache: CostFrameCache,
    *,
    daily_builder: DailyBuilder | None = None,
) -> tuple[tuple[DevelopmentSleeveInput, ...], dict[str, dict[str, Any]]]:
    daily_builder = daily_builder or build_completed_fx_days_with_diagnostics
    days_by_instrument = {}
    daily_by_instrument = {}
    events_by_instrument = {}
    for instrument in B5C_INSTRUMENTS:
        bid, ask = cache.frames(instrument, Decimal("1.0"))
        days, daily = daily_builder(instrument, bid, ask)
        days_by_instrument[instrument] = days
        daily_by_instrument[instrument] = daily
        all_events = generate_events(days)
        events_by_instrument[instrument] = (
            all_events,
            partition_development_signals(all_events),
        )

    sleeves: list[DevelopmentSleeveInput] = []
    diagnostics: dict[str, dict[str, Any]] = {}
    for instrument in B5C_INSTRUMENTS:
        days = days_by_instrument[instrument]
        all_events, events = events_by_instrument[instrument]
        intents = build_next_day_intents(events, days_by_instrument)
        unavailable = sum(event.event_day_index + 1 >= len(days) for event in events)
        passes: list[tuple[Batch5TradeResult, ...]] = []
        native_results: Sequence[Batch5TradeResult | Batch5SkipRecord] = ()
        for multiplier in COST_MULTIPLIERS:
            frames = cache.frames(instrument, multiplier)
            conversion = cache.frames("USD/JPY.OANDA", multiplier)
            results = execute_events(
                events,
                days_by_instrument=days_by_instrument,
                native_frames={instrument: frames},
                usdjpy_conversion_frames=conversion,
            )
            if multiplier == Decimal("1.0"):
                native_results = results
            passes.append(_trades(results))
        _require_stress_timing_identity(passes)
        sleeve_id = f"B5C_{instrument.split('.')[0].replace('/', '')}"
        years = len({event.signal_timestamp.year for event in events})
        sleeves.append(
            DevelopmentSleeveInput(
                FAMILY_B5C,
                "B5C_FROZEN_DAILY_OVERREACTION_REVERSAL",
                sleeve_id,
                instrument,
                passes[0],
                passes[1],
                passes[2],
                DEVELOPMENT_FOLD_BOUNDARIES,
                FrequencyStats(event_count=len(events), active_year_count=years),
                len(events),
                years,
            )
        )
        diagnostics[instrument] = {
            **_daily_diagnostics(daily_by_instrument[instrument]),
            "days_with_30_day_statistic": max(0, len(days) - 30),
            "all_formed_event_count": len(all_events),
            "warmup_event_count": len(all_events) - len(events),
            "executable_event_count": len(events),
            "raw_below_minus_2_event_count": sum(
                event.direction == "BUY" for event in events
            ),
            "raw_above_plus_2_event_count": sum(
                event.direction == "SELL" for event in events
            ),
            "events_removed_by_active_position_rule": (
                len(events) - unavailable - len(intents)
            ),
            "events_lost_next_complete_day_unavailable": unavailable,
            "execution_skips": dict(
                sorted(
                    Counter(
                        row.skip_reason
                        for row in native_results
                        if isinstance(row, Batch5SkipRecord)
                    ).items()
                )
            ),
            "completed_event_trades": len(passes[0]),
        }
    return tuple(sleeves), diagnostics


def build_bc_family_summary(
    inputs: Sequence[DevelopmentSleeveInput],
) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    for family in BC_FAMILIES:
        family_inputs = tuple(item for item in inputs if item.family == family)
        family_result = evaluate_family(
            tuple(item.screen_input() for item in family_inputs)
        )
        scorecards = tuple(evaluate_development_sleeve(item) for item in family_inputs)
        family_frequency = all(row.frequency_gate for row in scorecards)
        if family == FAMILY_B5C:
            family_frequency = (
                family_frequency and family_result.family_event_count >= 60
            )
        eligible = (
            family_frequency
            and family_result.aggregate_drawdown_passed
            and family_result.breadth_passed
        )
        if eligible != family_result.validation_eligible:
            raise Batch5BCCorrectedOrchestratorError("family eligibility drift")
        result.append(
            {
                "family": family,
                "strategy_id": scorecards[0].strategy_id,
                "tested_sleeve_count": family_result.sleeve_count,
                "core_positive_count": family_result.positive_native_and_1_5x_count,
                "full_gate_passing_count": family_result.full_gate_sleeve_count,
                "family_event_count": family_result.family_event_count,
                "equal_weight_maximum_drawdown": (
                    family_result.equal_weight_maximum_drawdown
                ),
                "family_frequency_passed": family_frequency,
                "family_drawdown_passed": family_result.aggregate_drawdown_passed,
                "family_breadth_passed": family_result.breadth_passed,
                "eligible_for_future_validation": eligible,
            }
        )
    return tuple(result)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        if value.is_infinite():
            return "Infinity" if value > 0 else "-Infinity"
        if value.is_nan():
            return "NaN"
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if hasattr(value, "isoformat") and not isinstance(value, str):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise Batch5BCCorrectedOrchestratorError(
            f"refusing empty artifact: {path.name}"
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(_jsonable(row) for row in rows)


def write_corrected_artifacts(
    *,
    scorecards: Sequence[DevelopmentSleeveScorecard],
    family_summary: Sequence[Mapping[str, Any]],
    correction_summary: Mapping[str, Any],
    diagnostics_summary: Mapping[str, Any],
    metadata: Mapping[str, Any],
    output_dir: Path,
) -> None:
    if output_dir.exists():
        raise Batch5BCCorrectedOrchestratorError(
            f"{output_dir} already exists; refusing to overwrite"
        )
    if len(scorecards) != 6 or len(family_summary) != 2:
        raise Batch5BCCorrectedOrchestratorError(
            "corrected artifacts require exactly six sleeves and two families"
        )
    output_dir.mkdir(parents=True)
    ordered = sorted(scorecards, key=lambda row: (row.family, row.sleeve_id))
    _write_csv(output_dir / "sleeve_scorecard.csv", [asdict(row) for row in ordered])
    _write_csv(output_dir / "family_summary.csv", list(family_summary))
    for filename, payload in (
        ("correction_summary.json", correction_summary),
        ("diagnostics_summary.json", diagnostics_summary),
        ("metadata.json", metadata),
    ):
        (output_dir / filename).write_text(
            json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    hashes = {
        path.name: _sha256_file(path)
        for path in sorted(output_dir.iterdir())
        if path.name != "artifact_hashes.json"
    }
    (output_dir / "artifact_hashes.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_corrected_development(
    *,
    development_root: Path,
    universe_readiness: Path,
    batch5_cross_root: Path,
    output_dir: Path,
    daily_builder: DailyBuilder = build_completed_fx_days_with_diagnostics,
    stage: str = "B5_BC_corrected_DEVELOPMENT_v1",
    methodology_amendment: Mapping[str, Any] | None = None,
) -> None:
    """Run only corrected B5B/B5C after refusing an existing output."""

    reject_forbidden_root(output_dir, label="--output")
    reserve_output_directory(output_dir)
    methodology, correction, readiness = verify_corrected_preflight(
        development_root=development_root,
        universe_readiness=universe_readiness,
        batch5_cross_root=batch5_cross_root,
    )
    cache = CostFrameCache(
        development_root=development_root,
        batch5_cross_root=batch5_cross_root,
    )
    b5b, b5b_diagnostics = run_corrected_b5b(cache, daily_builder=daily_builder)
    b5c, b5c_diagnostics = run_corrected_b5c(cache, daily_builder=daily_builder)
    inputs = (b5b, *b5c)
    if {item.sleeve_id for item in inputs} != EXPECTED_SLEEVES:
        raise Batch5BCCorrectedOrchestratorError("corrected sleeve identity drift")
    scorecards = tuple(
        evaluate_development_sleeve(item)
        for item in sorted(inputs, key=lambda row: (row.family, row.sleeve_id))
    )
    families = build_bc_family_summary(inputs)
    performance_diagnostics = build_diagnostics_summary(scorecards)
    diagnostics = {
        **performance_diagnostics,
        "daily_funnel": {"B5B_AUDCAD": b5b_diagnostics, **b5c_diagnostics},
    }
    correction_summary = {
        "correction_protocol_semantic_sha256": (
            EXPECTED_CORRECTION_PROTOCOL_SEMANTIC_SHA256
        ),
        "original_development_artifact_hash_manifest_sha256": (
            ORIGINAL_ARTIFACT_HASH_MANIFEST_SHA256
        ),
        "included_families": list(BC_FAMILIES),
        "excluded_family": "B5A_cftc_dealer_demand_shock_fx",
        "corrections": correction["corrections"],
        "original_gates_and_breadth_unchanged": True,
        "parameter_tuning_or_rescue_used": False,
        "family_eligibility": {
            row["family"]: row["eligible_for_future_validation"] for row in families
        },
    }
    if methodology_amendment is not None:
        correction_summary["methodology_amendment_semantic_sha256"] = (
            methodology_amendment["amendment_semantic_sha256"]
        )
        correction_summary["methodology_amendment"] = methodology_amendment
        correction_summary["one_shot_provider_aware_development_run"] = True
    metadata = {
        "stage": stage,
        "preregistration_semantic_sha256": (
            EXPECTED_PREREGISTRATION_SEMANTIC_SHA256
        ),
        "correction_protocol_semantic_sha256": (
            EXPECTED_CORRECTION_PROTOCOL_SEMANTIC_SHA256
        ),
        "original_development_artifact_hash_manifest_sha256": (
            ORIGINAL_ARTIFACT_HASH_MANIFEST_SHA256
        ),
        "development_start_utc": DEVELOPMENT_START,
        "development_end_exclusive_utc": DEVELOPMENT_END_EXCLUSIVE,
        "development_fold_boundaries_utc": DEVELOPMENT_FOLD_BOUNDARIES,
        "family_ids": list(BC_FAMILIES),
        "sleeve_ids": sorted(EXPECTED_SLEEVES),
        "cost_stress_multipliers": list(COST_MULTIPLIERS),
        "gate_definitions": methodology["common_development_gates"],
        "frequency_gate_definitions": {
            family: methodology["families"][family]["screening"]
            for family in BC_FAMILIES
        },
        "breadth_rules": {
            family: methodology["breadth_rules"][family] for family in BC_FAMILIES
        },
        "native_m1_load_counts": dict(sorted(cache.native_load_count.items())),
        "stress_widen_counts": {
            f"{instrument}@{multiplier}x": count
            for (instrument, multiplier), count in sorted(cache.widen_count.items())
        },
        "canonical_oanda_readiness_semantic_sha256": readiness["canonical"].get(
            "semantic_sha256"
        ),
        "batch5_cross_readiness_semantic_sha256": readiness["cross"].get(
            "semantic_sha256"
        ),
        "b5a_executed": False,
        "b5b_rollover_supported": False,
        "b5b_rollover_policy": (
            "frequency_gate_preserved; price-only trades generated and scored"
        ),
        "development_accessed": True,
        "validation_accessed": False,
        "holdout_accessed": False,
        "parameter_search_run": False,
        "fallback_or_rescue_used": False,
        "git_commit": _git_commit(),
        "dependency_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "nautilus_trader": importlib.metadata.version("nautilus_trader"),
        },
    }
    if methodology_amendment is not None:
        metadata["methodology_amendment_semantic_sha256"] = (
            methodology_amendment["amendment_semantic_sha256"]
        )
        metadata["provider_aware_fx_day_methodology"] = True
        metadata["maximum_permitted_runs_under_amendment"] = 1
    write_corrected_artifacts(
        scorecards=scorecards,
        family_summary=families,
        correction_summary=correction_summary,
        diagnostics_summary=diagnostics,
        metadata=metadata,
        output_dir=output_dir,
    )


def benchmark_corrected_runtime(row_count: int = 200_000) -> dict[str, float | int]:
    """Small-fixture benchmark and conservative six-stream runtime estimate."""

    if row_count < 10_000:
        raise Batch5BCCorrectedOrchestratorError(
            "runtime benchmark requires at least 10000 rows"
        )
    index = pd.date_range(DEVELOPMENT_START, periods=row_count, freq="min", tz="UTC")
    base = np.linspace(1.0, 1.01, row_count)
    bid = pd.DataFrame(
        {name: base for name in ("open", "high", "low", "close")}, index=index
    )
    ask = bid + 0.0002
    started = time.perf_counter()
    widen_bid_ask_frame(bid, ask, 1.5)
    widen_bid_ask_frame(bid, ask, 2.0)
    widening = time.perf_counter() - started
    started = time.perf_counter()
    build_completed_fx_days_with_diagnostics("EUR/USD.OANDA", bid, ask)
    daily = time.perf_counter() - started
    development_days = (DEVELOPMENT_END_EXCLUSIVE - DEVELOPMENT_START).days
    expected_rows = int(development_days * 1440 * 5 / 7)
    scale = expected_rows / row_count
    compute = (widening + daily) * scale * 6
    assumed_load = 6 * 30.0
    assumed_preflight_and_reporting = 180.0
    total = compute + assumed_load + assumed_preflight_and_reporting
    return {
        "synthetic_row_count": row_count,
        "unique_real_m1_streams": 6,
        "expected_rows_per_instrument": expected_rows,
        "two_widening_passes_seconds": widening,
        "corrected_daily_build_seconds": daily,
        "estimated_compute_seconds": compute,
        "assumed_load_seconds": assumed_load,
        "assumed_preflight_and_reporting_seconds": assumed_preflight_and_reporting,
        "estimated_total_seconds": total,
        "estimated_total_minutes": total / 60,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run corrected frozen Batch 5B/B5C DEVELOPMENT exactly once."
    )
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--universe-readiness", type=Path, required=True)
    parser.add_argument("--batch5-cross-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_corrected_development(
        development_root=cast(Path, args.development_root),
        universe_readiness=cast(Path, args.universe_readiness),
        batch5_cross_root=cast(Path, args.batch5_cross_root),
        output_dir=cast(Path, args.output),
    )


if __name__ == "__main__":
    main()
