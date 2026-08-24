"""B3F2 real-DEVELOPMENT orchestrator + CLI.

Wires the already-frozen, already-tested signal
(:mod:`ftmoquant.research.alpha_lab.b3f2_asian_range_fade_signals`),
execution
(:mod:`ftmoquant.research.alpha_lab.b3f2_asian_range_fade_execution`), and
gate (:mod:`ftmoquant.research.alpha_lab.b3f2_asian_range_fade_screen`)
modules to real OANDA M1 BID/ASK data. Contains NO new signal, sizing,
execution, or gate logic -- only real-data loading, the frozen 7-instrument
x 12-config loop, and artifact writing.

Per instrument (section 18): load M1 BID/ASK once; derive the M15
mid-price signal frame and Asian ranges once; widen to 1.5x/2.0x once;
reuse all of that across the 12 configs sharing that instrument. This
mirrors the B3.2c redundant-call-elimination lesson learned for B3F1 --
applied here from the start rather than discovered after a slow first run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import scipy  # type: ignore[import-untyped]
import statsmodels  # type: ignore[import-untyped]

from ftmoquant.data.instruments import InstrumentSpec, oanda_symbol
from ftmoquant.research.alpha_lab.b3f1_spread_execution import GROSS_NOTIONAL_USD
from ftmoquant.research.alpha_lab.b3f1_spread_signals import PAIR_UNIVERSE
from ftmoquant.research.alpha_lab.b3f2_asian_range_fade_execution import (
    simulate_b3f2_intents,
)
from ftmoquant.research.alpha_lab.b3f2_asian_range_fade_screen import (
    FAMILY_ID,
    MAX_QUARTER_SHARE,
    MIN_CONNECTED_REGION_SIZE,
    MIN_TRADE_COUNT,
    PROFIT_FACTOR_GT,
    STRESS_MULTIPLIERS,
    B3F2ScorecardRow,
    evaluate_b3f2_config,
    largest_connected_passing_region,
    write_b3f2_artifacts,
)
from ftmoquant.research.alpha_lab.b3f2_asian_range_fade_signals import (
    ASIAN_RANGE_END_LOCAL,
    ASIAN_RANGE_START_LOCAL,
    ENTRY_WINDOW_END_LOCAL,
    ENTRY_WINDOW_START_LOCAL,
    EXCURSION_MIN_GRID,
    LONDON_TIMEZONE,
    STOP_BUFFER_GRID,
    TARGET_MODE_GRID,
    TIME_EXIT_LOCAL,
    build_b3f2_grid,
    compute_asian_ranges,
    derive_m15_mid_ohlc,
    generate_b3f2_decisions,
)
from ftmoquant.research.alpha_lab.batch3_preregistration import (
    DEVELOPMENT_FOLD_BOUNDARIES,
)
from ftmoquant.research.alpha_lab.batch3_preregistration_v2 import (
    PREREGISTRATION_VERSION as V2_PREREGISTRATION_VERSION,
)
from ftmoquant.research.alpha_lab.cost_stress import widen_bid_ask_frame
from ftmoquant.research.alpha_lab.data import load_alpha_lab_dataset
from ftmoquant.research.alpha_lab.liquidity_structure_screen import (
    _axis_adjacent_neighbors,
)
from ftmoquant.research.alpha_lab.screening_stage2 import _connected_components
from ftmoquant.research.alpha_lab.wick_fvg_squeeze_execution import load_m1_bidask
from ftmoquant.research.stage_g import DEVELOPMENT_END_EXCLUSIVE, DEVELOPMENT_START

FROZEN_PREREGISTRATION_SHA256 = (
    "e5cd74527004585cfd24bea55549d84e9cb66b05ffc6498360dac5007b651f7c"
)
_FORBIDDEN_ROOT_TOKENS = ("validation", "holdout", "final_holdout", "final_test")
_FOLD_BOUNDARIES: tuple[datetime, ...] = DEVELOPMENT_FOLD_BOUNDARIES


class B3F2OrchestratorError(ValueError):
    """Raised on any violation of the frozen B3F2 orchestration contract."""


def reject_forbidden_root(path: Path, *, label: str) -> None:
    lowered = str(path).lower()
    for token in _FORBIDDEN_ROOT_TOKENS:
        if token in lowered:
            raise B3F2OrchestratorError(
                f"{label} path {path} contains the forbidden token {token!r} "
                "-- refusing before any catalog access"
            )


def reserve_output_directory(output_dir: Path) -> None:
    if output_dir.exists():
        raise B3F2OrchestratorError(
            f"{output_dir} already exists; refusing to overwrite"
        )


def _instrument_m1_root(spec: InstrumentSpec, development_root_dir: Path) -> Path:
    return development_root_dir / oanda_symbol(spec.dataset_symbol)


def run_instrument_job(
    *, spec: InstrumentSpec, development_root_dir: Path
) -> tuple[B3F2ScorecardRow, ...]:
    """Run all 12 frozen configs for ONE instrument: derive the M15 signal
    frame and Asian ranges once, widen M1 to 1.5x/2.0x once, reuse across
    every config. Picklable/top-level so it can run in a worker process if
    ever parallelized."""

    bid_m1, ask_m1 = load_m1_bidask(
        instrument_id=spec.instrument_id,
        root=_instrument_m1_root(spec, development_root_dir),
        start_utc=DEVELOPMENT_START,
        end_exclusive_utc=DEVELOPMENT_END_EXCLUSIVE,
    )
    m15_mid = derive_m15_mid_ohlc(bid_m1, ask_m1)
    asian_ranges = compute_asian_ranges(m15_mid)

    stressed_1_5x = widen_bid_ask_frame(bid_m1, ask_m1, float(STRESS_MULTIPLIERS[0]))
    stressed_2_0x = widen_bid_ask_frame(bid_m1, ask_m1, float(STRESS_MULTIPLIERS[1]))

    rows: list[B3F2ScorecardRow] = []
    for config in build_b3f2_grid():
        decisions = generate_b3f2_decisions(
            m15_mid, asian_ranges, instrument_id=spec.instrument_id, config=config
        )
        native_trades, _ = simulate_b3f2_intents(
            decisions,
            instrument_spec=spec,
            bid_m1=bid_m1,
            ask_m1=ask_m1,
            gross_notional_usd=GROSS_NOTIONAL_USD,
            cost_stress_multiplier=Decimal("1"),
        )
        stressed_1_5x_trades, _ = simulate_b3f2_intents(
            decisions,
            instrument_spec=spec,
            bid_m1=stressed_1_5x[0],
            ask_m1=stressed_1_5x[1],
            gross_notional_usd=GROSS_NOTIONAL_USD,
            cost_stress_multiplier=Decimal("1"),
        )
        stressed_2_0x_trades, _ = simulate_b3f2_intents(
            decisions,
            instrument_spec=spec,
            bid_m1=stressed_2_0x[0],
            ask_m1=stressed_2_0x[1],
            gross_notional_usd=GROSS_NOTIONAL_USD,
            cost_stress_multiplier=Decimal("1"),
        )
        rows.append(
            evaluate_b3f2_config(
                instrument_id=spec.instrument_id,
                config=config,
                native_trades=native_trades,
                stressed_1_5x_trades=stressed_1_5x_trades,
                stressed_2_0x_trades=stressed_2_0x_trades,
                fold_boundaries=_FOLD_BOUNDARIES,
            )
        )
    return tuple(rows)


# ---------------------------------------------------------------------------
# Instrument robustness (independent per instrument; no cross-instrument
# ranking).
# ---------------------------------------------------------------------------


def compute_instrument_robustness(
    instrument_id: str, rows: tuple[B3F2ScorecardRow, ...]
) -> dict[str, Any]:
    passing_rows = [row for row in rows if row.hard_gates_passed]
    largest_region = largest_connected_passing_region(rows)
    from ftmoquant.research.alpha_lab.b3f2_asian_range_fade_screen import (
        _cell_for_config,
    )

    passing_cells: set[tuple[int, ...]] = {
        _cell_for_config(row.config) for row in passing_rows
    }
    connected_region_count = 0
    strongest: dict[str, Any] | None = None
    if passing_cells:
        components = _connected_components(passing_cells, _axis_adjacent_neighbors)
        connected_region_count = len(components)
        largest_component = max(components, key=len)
        cells_by_key: dict[tuple[int, ...], Any] = {
            _cell_for_config(row.config): row for row in passing_rows
        }
        best_cell = min(
            largest_component,
            key=lambda cell: (
                sum(
                    sum(abs(a - b) for a, b in zip(cell, other, strict=True))
                    for other in largest_component
                ),
                cell,
            ),
        )
        best_row = cells_by_key[best_cell]
        strongest = {
            "excursion_min": str(best_row.config.excursion_min),
            "stop_buffer": str(best_row.config.stop_buffer),
            "target_mode": best_row.config.target_mode,
        }

    return {
        "instrument_id": instrument_id,
        "tested_configuration_count": len(rows),
        "passing_configuration_count": len(passing_rows),
        "connected_region_count": connected_region_count,
        "largest_connected_region_size": largest_region,
        "survival_rule_passed": largest_region >= int(MIN_CONNECTED_REGION_SIZE),
        "strongest_region_excursion_min": (strongest or {}).get("excursion_min"),
        "strongest_region_stop_buffer": (strongest or {}).get("stop_buffer"),
        "strongest_region_target_mode": (strongest or {}).get("target_mode"),
    }


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[4],
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, OSError, FileNotFoundError):
        return "unknown"


def build_metadata(
    *,
    readiness_document: dict[str, Any],
    tested_instrument_count: int,
    tested_config_count: int,
) -> dict[str, Any]:
    return {
        "preregistration_version": V2_PREREGISTRATION_VERSION,
        "preregistration_semantic_sha256": FROZEN_PREREGISTRATION_SHA256,
        "family_id": FAMILY_ID,
        "development_start_utc": DEVELOPMENT_START.astimezone(UTC)
        .isoformat()
        .replace("+00:00", "Z"),
        "development_end_exclusive_utc": DEVELOPMENT_END_EXCLUSIVE.astimezone(UTC)
        .isoformat()
        .replace("+00:00", "Z"),
        "readiness_semantic_sha256": readiness_document.get("semantic_sha256"),
        "lineage_id": readiness_document.get("lineage_id"),
        "alpha_lab_config_sha256": readiness_document.get("alpha_lab_config_sha256"),
        "ordered_instrument_universe": [spec.instrument_id for spec in PAIR_UNIVERSE],
        "session_definitions": {
            "timezone": LONDON_TIMEZONE,
            "asian_range_local": [
                str(ASIAN_RANGE_START_LOCAL),
                str(ASIAN_RANGE_END_LOCAL),
            ],
            "entry_window_local": [
                str(ENTRY_WINDOW_START_LOCAL),
                str(ENTRY_WINDOW_END_LOCAL),
            ],
            "time_exit_local": str(TIME_EXIT_LOCAL),
            "conflict_with_existing_convention": (
                "ftmoquant.research.g1.sessions.ASIAN_SESSION/LONDON_SESSION use "
                "Asia/Tokyo 09:00-18:00 and Europe/London 08:00-17:00 local hours "
                "respectively -- materially different windows from the "
                "London-local 00:00-06:00/07:00-10:00 frozen here. Reported, "
                "not silently resolved -- see b3f2_asian_range_fade_signals "
                "module docstring."
            ),
        },
        "grid": {
            "excursion_min_grid": [str(v) for v in EXCURSION_MIN_GRID],
            "stop_buffer_grid": [str(v) for v in STOP_BUFFER_GRID],
            "target_mode_grid": list(TARGET_MODE_GRID),
            "configs_per_instrument": len(build_b3f2_grid()),
        },
        "signal_statement": (
            "M15 mid-price causal signal: Asian range frozen from completed "
            "00:00-06:00 London-local M15 bars; entry only within 07:00-10:00 "
            "London-local window; excursion-beyond-range + completed-bar "
            "re-entry required; at most one long and one short attempt per "
            "London-local date"
        ),
        "execution_semantics": {
            "primary": "native BID/ASK, M1, first strictly later paired observation",
            "cost_stress_multipliers": [str(m) for m in STRESS_MULTIPLIERS],
            "time_exit_local": str(TIME_EXIT_LOCAL) + " " + LONDON_TIMEZONE,
        },
        "hard_gates": {
            "min_trade_count": MIN_TRADE_COUNT,
            "profit_factor_gt": str(PROFIT_FACTOR_GT),
            "max_quarter_share": str(MAX_QUARTER_SHARE),
            "min_connected_region_size": MIN_CONNECTED_REGION_SIZE,
        },
        "tested_instrument_count": tested_instrument_count,
        "tested_config_count": tested_config_count,
        "library_versions": {
            "statsmodels": statsmodels.__version__,
            "scipy": scipy.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "python": sys.version.split()[0],
        },
        "git_commit": _git_commit(),
        "validation_accessed": False,
        "holdout_accessed": False,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_b3f2_development_screen(
    *, development_root: Path, universe_readiness: Path, output_dir: Path
) -> None:
    """The full real-DEVELOPMENT B3F2 screen. Write-once check happens
    FIRST, before any data loading."""

    reject_forbidden_root(development_root, label="--development-root")
    reject_forbidden_root(universe_readiness, label="--universe-readiness")
    reserve_output_directory(output_dir)

    readiness_document = json.loads(universe_readiness.read_text(encoding="utf-8"))
    # Confirm the readiness document is genuinely reachable/valid before
    # the expensive per-instrument loop (mirrors load_alpha_lab_dataset's
    # own validation, done here only to fail fast/consistently).
    load_alpha_lab_dataset(
        readiness_path=universe_readiness,
        development_root_dir=development_root,
        timeframe="H1",
        source="oanda",
    )

    scorecard: list[B3F2ScorecardRow] = []
    instrument_robustness: list[dict[str, Any]] = []
    for spec in PAIR_UNIVERSE:
        rows = run_instrument_job(spec=spec, development_root_dir=development_root)
        scorecard.extend(rows)
        instrument_robustness.append(
            compute_instrument_robustness(spec.instrument_id, rows)
        )

    metadata = build_metadata(
        readiness_document=readiness_document,
        tested_instrument_count=len(PAIR_UNIVERSE),
        tested_config_count=len(scorecard),
    )

    write_b3f2_artifacts(
        scorecard=scorecard,
        pair_robustness=instrument_robustness,
        metadata=metadata,
        output_dir=output_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen 7-instrument x 12-config B3F2 (Asian-range "
            "fade) DEVELOPMENT screen against real OANDA M1 BID/ASK "
            "execution."
        )
    )
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--universe-readiness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_b3f2_development_screen(
        development_root=args.development_root,
        universe_readiness=args.universe_readiness,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
