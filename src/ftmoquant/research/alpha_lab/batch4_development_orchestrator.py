"""Batch 4.2 real-DEVELOPMENT-only orchestration and thin CLI.

The real screen is never run on import.  All clocks, directions, gates,
breadth, and selection rules are verified before any catalog access.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ftmoquant.data.instruments import (
    OANDA_ALPHA_LAB_SPECS,
    InstrumentSpec,
    oanda_symbol,
)
from ftmoquant.research.alpha_lab.batch4_clock_scheduler import (
    EXPECTED_PREREGISTRATION_SEMANTIC_SHA256,
    FrozenClockSpec,
    ScheduledOccurrence,
    generate_occurrences,
    load_frozen_clock_specs,
)
from ftmoquant.research.alpha_lab.batch4_execution import (
    ScheduledSkipRecord,
    ScheduledTradeResult,
    execute_scheduled_occurrences,
)
from ftmoquant.research.alpha_lab.batch4_preregistration import (
    PREREGISTRATION_PATH,
    verify_preregistration,
)
from ftmoquant.research.alpha_lab.batch4_screen import (
    FAMILIES,
    Batch4ScorecardRow,
    FrozenScreenPolicy,
    apply_parameter_neighborhood,
    build_diagnostics_summary,
    build_family_summary,
    compute_family_robustness,
    evaluate_hypothesis,
    load_frozen_screen_policy,
    select_representative,
    write_batch4_artifacts,
)
from ftmoquant.research.alpha_lab.cost_stress import widen_bid_ask_frame
from ftmoquant.research.alpha_lab.data import AlphaLabDataset, load_alpha_lab_dataset
from ftmoquant.research.alpha_lab.wick_fvg_squeeze_execution import load_m1_bidask
from ftmoquant.research.stage_g import DEVELOPMENT_END_EXCLUSIVE, DEVELOPMENT_START

FROZEN_PREREGISTRATION_SHA256 = EXPECTED_PREREGISTRATION_SEMANTIC_SHA256
ARTIFACT_ROOT = Path(".artifacts/alpha_lab/batch4_structural_intraday_flow_v1")
EXPECTED_DEVELOPMENT_ROOT = Path(
    "/Users/Shared/FTMOQuant-data/oanda_fx_alpha_lab_v1/canonical"
)
EXPECTED_READINESS = Path(
    "/Users/Shared/FTMOQuant-data/oanda_fx_alpha_lab_v1/readiness/"
    "ftmoquant_oanda_alpha_lab_readiness.json"
)
_FORBIDDEN_ROOT_TOKENS = (
    "validation",
    "holdout",
    "final_holdout",
    "final-test",
    "final_test",
)
_DEVELOPMENT_DAYS = (DEVELOPMENT_END_EXCLUSIVE - DEVELOPMENT_START).days
_FOLD_STEP_DAYS = _DEVELOPMENT_DAYS // 4
DEVELOPMENT_FOLD_BOUNDARIES = tuple(
    DEVELOPMENT_START + timedelta(days=index * _FOLD_STEP_DAYS) for index in range(5)
)
if DEVELOPMENT_FOLD_BOUNDARIES[-1] != DEVELOPMENT_END_EXCLUSIVE:
    raise RuntimeError("DEVELOPMENT interval no longer divides into four frozen folds")


class Batch4OrchestratorError(ValueError):
    """Raised before any unsafe or methodology-drifted orchestration."""


def reject_forbidden_root(path: Path, *, label: str) -> None:
    lowered = str(path).lower()
    for token in _FORBIDDEN_ROOT_TOKENS:
        if token in lowered:
            raise Batch4OrchestratorError(
                f"{label} path contains forbidden token {token!r}: {path}"
            )


def reserve_output_directory(output_dir: Path) -> None:
    if output_dir.exists():
        raise Batch4OrchestratorError(
            f"{output_dir} already exists; refusing to overwrite"
        )


def verify_frozen_methodology(
    preregistration_path: Path = PREREGISTRATION_PATH,
) -> tuple[tuple[FrozenClockSpec, ...], FrozenScreenPolicy, dict[str, Any]]:
    """Fail closed on every frozen identity before price loading."""

    document = verify_preregistration(preregistration_path)
    if document["preregistration_semantic_sha256"] != FROZEN_PREREGISTRATION_SHA256:
        raise Batch4OrchestratorError("Batch 4 semantic SHA mismatch")
    specs = load_frozen_clock_specs(preregistration_path)
    policy = load_frozen_screen_policy(preregistration_path)
    if len(specs) != 91:
        raise Batch4OrchestratorError("Batch 4 must contain exactly 91 hypotheses")
    family_counts = {
        family: sum(spec.family == family for spec in specs) for family in FAMILIES
    }
    if tuple(family_counts.values()) != (7, 42, 42):
        raise Batch4OrchestratorError(
            f"frozen family membership drift: {family_counts}"
        )
    universe = {spec.instrument_id for spec in specs}
    expected_universe = {spec.instrument_id for spec in OANDA_ALPHA_LAB_SPECS}
    if universe != expected_universe or len(universe) != 7:
        raise Batch4OrchestratorError("frozen instrument universe drift")
    return specs, policy, document


def _development_occurrences(spec: FrozenClockSpec) -> tuple[ScheduledOccurrence, ...]:
    zone = __import__("zoneinfo").ZoneInfo(spec.timezone)
    first_local = DEVELOPMENT_START.astimezone(zone).date() - timedelta(days=1)
    last_exclusive = DEVELOPMENT_END_EXCLUSIVE.astimezone(zone).date() + timedelta(
        days=1
    )
    generated = generate_occurrences(spec, first_local, last_exclusive)
    return tuple(
        occurrence
        for occurrence in generated
        if occurrence.scheduled_entry_utc >= DEVELOPMENT_START
        and occurrence.scheduled_exit_utc < DEVELOPMENT_END_EXCLUSIVE
    )


def _instrument_m1_root(spec: InstrumentSpec, development_root: Path) -> Path:
    return development_root / oanda_symbol(spec.dataset_symbol)


def _validate_m1_partition(
    bid_m1: pd.DataFrame, ask_m1: pd.DataFrame, instrument_id: str
) -> None:
    if bid_m1.empty or ask_m1.empty or not bid_m1.index.equals(ask_m1.index):
        raise Batch4OrchestratorError(f"invalid paired M1 data: {instrument_id}")
    if bid_m1.index[0].to_pydatetime() < DEVELOPMENT_START:
        raise Batch4OrchestratorError("M1 data begins before DEVELOPMENT")
    if bid_m1.index[-1].to_pydatetime() >= DEVELOPMENT_END_EXCLUSIVE:
        raise Batch4OrchestratorError("M1 data extends beyond DEVELOPMENT")


def _group_trades(
    trades: Sequence[ScheduledTradeResult],
) -> dict[str, tuple[ScheduledTradeResult, ...]]:
    grouped: dict[str, list[ScheduledTradeResult]] = defaultdict(list)
    for trade in trades:
        grouped[trade.hypothesis_id].append(trade)
    return {key: tuple(value) for key, value in grouped.items()}


def _group_skips(
    skips: Sequence[ScheduledSkipRecord],
) -> dict[str, tuple[ScheduledSkipRecord, ...]]:
    grouped: dict[str, list[ScheduledSkipRecord]] = defaultdict(list)
    for skip in skips:
        grouped[skip.hypothesis_id].append(skip)
    return {key: tuple(value) for key, value in grouped.items()}


def run_instrument_job(
    *,
    instrument_spec: InstrumentSpec,
    frozen_specs: Sequence[FrozenClockSpec],
    development_root: Path,
    policy: FrozenScreenPolicy,
) -> tuple[
    tuple[Batch4ScorecardRow, ...],
    dict[str, tuple[ScheduledTradeResult, ...]],
]:
    """Load once, widen twice, and execute three shared occurrence passes."""

    instrument_specs = tuple(
        spec
        for spec in frozen_specs
        if spec.instrument_id == instrument_spec.instrument_id
    )
    if len(instrument_specs) != 13:
        raise Batch4OrchestratorError(
            f"{instrument_spec.instrument_id} must own exactly 13 hypotheses"
        )
    occurrences = tuple(
        occurrence
        for spec in instrument_specs
        for occurrence in _development_occurrences(spec)
    )
    bid_m1, ask_m1 = load_m1_bidask(
        instrument_id=instrument_spec.instrument_id,
        root=_instrument_m1_root(instrument_spec, development_root),
        start_utc=DEVELOPMENT_START,
        end_exclusive_utc=DEVELOPMENT_END_EXCLUSIVE,
    )
    _validate_m1_partition(bid_m1, ask_m1, instrument_spec.instrument_id)
    stressed_1_5x = widen_bid_ask_frame(bid_m1, ask_m1, 1.5)
    stressed_2_0x = widen_bid_ask_frame(bid_m1, ask_m1, 2.0)
    pass_frames = (
        (bid_m1, ask_m1),
        stressed_1_5x,
        stressed_2_0x,
    )
    executions = tuple(
        execute_scheduled_occurrences(
            occurrences,
            bid_m1=bid,
            ask_m1=ask,
            cost_stress_multiplier=Decimal("1.0"),
        )
        for bid, ask in pass_frames
    )
    native_trades = _group_trades(executions[0][0])
    native_skips = _group_skips(executions[0][1])
    stress_1_trades = _group_trades(executions[1][0])
    stress_2_trades = _group_trades(executions[2][0])
    rows = tuple(
        evaluate_hypothesis(
            spec=spec,
            native_trades=native_trades.get(spec.hypothesis_id, ()),
            native_skip_count=len(native_skips.get(spec.hypothesis_id, ())),
            stressed_1_5x_trades=stress_1_trades.get(spec.hypothesis_id, ()),
            stressed_2_0x_trades=stress_2_trades.get(spec.hypothesis_id, ()),
            fold_boundaries=DEVELOPMENT_FOLD_BOUNDARIES,
            policy=policy,
        )
        for spec in instrument_specs
    )
    return rows, native_trades


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[4],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def build_metadata(
    *,
    specs: Sequence[FrozenClockSpec],
    methodology: Mapping[str, Any],
    readiness_document: Mapping[str, Any],
    readiness_path: Path,
) -> dict[str, Any]:
    identities = [spec.hypothesis_id for spec in specs]
    identity_sha = hashlib.sha256(
        json.dumps(identities, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "stage": "B4.2_DEVELOPMENT_screen",
        "preregistration_semantic_sha256": FROZEN_PREREGISTRATION_SHA256,
        "hypothesis_count": 91,
        "hypothesis_ids": identities,
        "hypothesis_identity_sha256": identity_sha,
        "family_counts": {
            family: sum(spec.family == family for spec in specs) for family in FAMILIES
        },
        "development_start_utc": _iso(DEVELOPMENT_START),
        "development_end_exclusive_utc": _iso(DEVELOPMENT_END_EXCLUSIVE),
        "development_fold_boundaries_utc": [
            _iso(value) for value in DEVELOPMENT_FOLD_BOUNDARIES
        ],
        "readiness_file_sha256": _sha256_file(readiness_path),
        "readiness_semantic_sha256": readiness_document.get("semantic_sha256"),
        "readiness_lineage_id": readiness_document.get("lineage_id"),
        "cost_stress_multipliers": ["1.0", "1.5", "2.0"],
        "cache_design": (
            "one native paired-M1 load plus one 1.5x and one 2.0x frame build "
            "per instrument; all 13 instrument hypotheses share each pass"
        ),
        "gate_definitions": methodology["development_gates"],
        "breadth_rule": methodology["family_breadth_gate"],
        "selection_rule": methodology["development_to_validation"],
        "diagnostics_status": "report_only_never_gate_rank_filter_or_rescue",
        "metric_conventions": {
            "net_return": "sum of fixed-100k per-trade USD P&L divided by 100000",
            "annualized_sharpe": (
                "UTC calendar-daily fixed-reference returns including zero-return "
                "days, sample standard deviation, sqrt(365) annualization"
            ),
            "maximum_drawdown": (
                "maximum peak-to-trough decline of additive fixed-reference returns"
            ),
            "spread_cost_share_of_gross_edge": (
                "native-minus-2x net P&L divided by native net P&L plus that "
                "difference; exact because 2x adds one native spread cost and "
                "timestamps are invariant"
            ),
        },
        "git_commit": _git_commit(),
        "dependency_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "nautilus_trader": importlib.metadata.version("nautilus_trader"),
        },
        "development_accessed": True,
        "validation_accessed": False,
        "holdout_accessed": False,
        "monte_carlo_run": False,
    }


def run_batch4_development_screen(
    *, development_root: Path, universe_readiness: Path, output_dir: Path
) -> None:
    """Run the real screen when explicitly invoked by the user, never here."""

    reject_forbidden_root(development_root, label="--development-root")
    reject_forbidden_root(universe_readiness, label="--universe-readiness")
    reject_forbidden_root(output_dir, label="--output")
    reserve_output_directory(output_dir)
    specs, policy, methodology = verify_frozen_methodology()

    readiness_document = json.loads(universe_readiness.read_text(encoding="utf-8"))
    readiness_dataset: AlphaLabDataset = load_alpha_lab_dataset(
        readiness_path=universe_readiness,
        development_root_dir=development_root,
        timeframe="H1",
        source="oanda",
    )
    frozen_instruments = {spec.instrument_id for spec in specs}
    if set(readiness_dataset.instrument_ids) != frozen_instruments:
        raise Batch4OrchestratorError(
            "readiness universe does not match frozen Batch 4"
        )

    scorecard: list[Batch4ScorecardRow] = []
    native_trades: dict[str, tuple[ScheduledTradeResult, ...]] = {}
    spec_by_id = {spec.instrument_id: spec for spec in OANDA_ALPHA_LAB_SPECS}
    for instrument_id in sorted(frozen_instruments):
        rows, trades = run_instrument_job(
            instrument_spec=spec_by_id[instrument_id],
            frozen_specs=specs,
            development_root=development_root,
            policy=policy,
        )
        scorecard.extend(rows)
        native_trades.update(trades)

    ordered_base = tuple(sorted(scorecard, key=lambda row: row.hypothesis_id))
    gated = apply_parameter_neighborhood(ordered_base, policy)
    robustness = compute_family_robustness(gated, policy)
    family_summary = build_family_summary(gated, robustness)
    selection = select_representative(robustness, policy)
    diagnostics = build_diagnostics_summary(specs, native_trades, gated)
    metadata = build_metadata(
        specs=specs,
        methodology=methodology,
        readiness_document=readiness_document,
        readiness_path=universe_readiness,
    )
    write_batch4_artifacts(
        scorecard=gated,
        family_summary=family_summary,
        family_robustness=robustness,
        selection_summary=selection,
        diagnostics_summary=diagnostics,
        metadata=metadata,
        output_dir=output_dir,
    )


def benchmark_synthetic_runtime(row_count: int = 200_000) -> dict[str, float | int]:
    """Synthetic-only compute benchmark and conservative seven-pair estimate."""

    if row_count < 10_000:
        raise Batch4OrchestratorError("benchmark row_count must be at least 10,000")
    specs, _, _ = verify_frozen_methodology()
    instrument_specs = tuple(
        spec for spec in specs if spec.instrument_id == "EUR/USD.OANDA"
    )
    occurrences = tuple(
        occurrence
        for spec in instrument_specs
        for occurrence in _development_occurrences(spec)
    )
    index = pd.date_range(DEVELOPMENT_START, periods=row_count, freq="min", tz="UTC")
    base = np.linspace(1.05, 1.06, row_count)
    bid = pd.DataFrame(
        {"open": base, "high": base, "low": base, "close": base}, index=index
    )
    ask = pd.DataFrame(
        {
            "open": base + 0.0002,
            "high": base + 0.0002,
            "low": base + 0.0002,
            "close": base + 0.0002,
        },
        index=index,
    )
    start = time.perf_counter()
    stress_1 = widen_bid_ask_frame(bid, ask, 1.5)
    stress_2 = widen_bid_ask_frame(bid, ask, 2.0)
    widening_seconds = time.perf_counter() - start
    start = time.perf_counter()
    for frames in ((bid, ask), stress_1, stress_2):
        execute_scheduled_occurrences(
            occurrences,
            bid_m1=frames[0],
            ask_m1=frames[1],
            cost_stress_multiplier=Decimal("1.0"),
        )
    execution_seconds = time.perf_counter() - start
    expected_rows = round(_DEVELOPMENT_DAYS * 1440 * 5 / 7)
    scale = expected_rows / row_count
    compute_seconds_all_pairs = (widening_seconds + execution_seconds) * scale * 7
    assumed_load_seconds_all_pairs = 7 * 30.0
    assumed_readiness_preflight_seconds = 60.0
    assumed_scoring_and_artifact_seconds = 60.0
    estimated_total_seconds = (
        compute_seconds_all_pairs
        + assumed_load_seconds_all_pairs
        + assumed_readiness_preflight_seconds
        + assumed_scoring_and_artifact_seconds
    )
    return {
        "synthetic_row_count": row_count,
        "expected_m1_rows_per_instrument": expected_rows,
        "two_widening_passes_seconds": widening_seconds,
        "three_execution_passes_seconds": execution_seconds,
        "linear_scale_factor": scale,
        "assumed_m1_load_seconds_all_pairs": assumed_load_seconds_all_pairs,
        "assumed_readiness_preflight_seconds": assumed_readiness_preflight_seconds,
        "assumed_scoring_and_artifact_seconds": assumed_scoring_and_artifact_seconds,
        "estimated_total_seconds_all_91_hypotheses": estimated_total_seconds,
        "estimated_total_minutes_all_91_hypotheses": estimated_total_seconds / 60,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen Batch 4 structural-flow DEVELOPMENT screen."
    )
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--universe-readiness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_batch4_development_screen(
        development_root=args.development_root,
        universe_readiness=args.universe_readiness,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
