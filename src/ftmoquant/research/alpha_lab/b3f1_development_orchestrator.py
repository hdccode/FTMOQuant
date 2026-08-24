"""B3.2b: real-DEVELOPMENT orchestrator + CLI for the frozen B3F1
(cross-instrument spread mean reversion) screen.

Wires the already-frozen, already-tested signal
(:mod:`ftmoquant.research.alpha_lab.b3f1_spread_signals`), execution
(:mod:`ftmoquant.research.alpha_lab.b3f1_spread_execution`), and gate
(:mod:`ftmoquant.research.alpha_lab.b3f1_spread_screen`) modules to the
real OANDA alpha-lab DEVELOPMENT loader
(:mod:`ftmoquant.research.alpha_lab.data`). This module contains NO new
signal, sizing, execution, or gate logic -- it only loads real data,
iterates the frozen 21-pair x 18-config grid, and writes artifacts.

Loop structure (task section 3), per pair:

    A. load the shared H1 log-mid-close series once (done ONCE for all 21
       pairs, in the caller, via the existing
       :func:`~ftmoquant.research.alpha_lab.data.load_alpha_lab_dataset`)
    B. for each formation window, compute the formation series exactly
       once (:func:`~ftmoquant.research.alpha_lab.b3f1_spread_signals.
       compute_formation_series`)
    C. reuse that exact formation series across all 6 (z_entry, z_stop)
       combinations sharing the window
    D. generate the causal decision sequence once per config
       (:func:`~ftmoquant.research.alpha_lab.b3f1_spread_signals.
       generate_b3f1_decisions`) -- identical for native and stressed
       execution, since it depends only on the formation series, never on
       execution/cost-stress parameters
    E. execute that SAME decision sequence twice: once native, once at
       1.5x cost stress (:func:`~ftmoquant.research.alpha_lab.
       b3f1_spread_execution.simulate_b3f1_intents`) -- ADF/OLS/decisions
       are never recomputed for the stressed run

Memory strategy (task section 4): the H1 series (8,597 rows x 7
instruments of floats) is loaded once, shared by every pair -- trivial
memory cost. M1 BID/ASK (the large series) is loaded PER PAIR, fresh, only
for that pair's own two instruments, and is not retained once that pair's
18 configs are done. When ``--workers`` > 1, each worker process loads its
own pairs' M1 data directly from disk (never receives a pickled M1
DataFrame from the parent) -- so parallelism never multiplies M1 memory
beyond ``workers`` pairs' worth at a time.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import scipy  # type: ignore[import-untyped]
import statsmodels  # type: ignore[import-untyped]

from ftmoquant.data.instruments import InstrumentSpec, oanda_symbol
from ftmoquant.research.alpha_lab.b3f1_spread_execution import (
    GROSS_NOTIONAL_USD,
    simulate_b3f1_intents,
)
from ftmoquant.research.alpha_lab.b3f1_spread_screen import (
    FAMILY_ID,
    MAX_QUARTER_SHARE,
    MIN_CONNECTED_REGION_SIZE,
    MIN_TRADE_COUNT,
    PROFIT_FACTOR_GT,
    STRESS_MULTIPLIERS,
    B3F1ScorecardRow,
    evaluate_b3f1_config,
    largest_connected_passing_region,
    write_b3f1_artifacts,
)
from ftmoquant.research.alpha_lab.b3f1_spread_signals import (
    FORMATION_WINDOWS,
    PAIR_UNIVERSE,
    Z_ENTRY_GRID,
    Z_STOP_GRID,
    B3F1Config,
    OrderedPair,
    compute_formation_series,
    enumerate_candidate_pairs,
    generate_b3f1_decisions,
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


class B3F1OrchestratorError(ValueError):
    """Raised on any violation of the frozen B3.2b orchestration contract."""


def reject_forbidden_root(path: Path, *, label: str) -> None:
    """Fail closed BEFORE any catalog access if a supplied root/readiness
    path looks like it could point at VALIDATION or (final) HOLDOUT data.
    Deliberately checked on the raw path string, before any file I/O."""

    lowered = str(path).lower()
    for token in _FORBIDDEN_ROOT_TOKENS:
        if token in lowered:
            raise B3F1OrchestratorError(
                f"{label} path {path} contains the forbidden token {token!r} "
                "-- refusing before any catalog access"
            )


def reserve_output_directory(output_dir: Path) -> None:
    """Refuse to proceed if ``output_dir`` already exists. Called FIRST,
    before any data loading or computation, so a long run is never
    discarded only at final write time."""

    if output_dir.exists():
        raise B3F1OrchestratorError(
            f"{output_dir} already exists; refusing to overwrite"
        )


# ---------------------------------------------------------------------------
# Per-pair job (the unit of parallelism)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _PairJob:
    pair: OrderedPair
    log_y: pd.Series
    log_x: pd.Series
    development_root_dir: Path


def _leg_m1_root(spec: InstrumentSpec, development_root_dir: Path) -> Path:
    return development_root_dir / oanda_symbol(spec.dataset_symbol)


#: Per-process, instrument-level cache of ALREADY-stressed M1 frames, keyed
#: by ``(instrument_id, str(multiplier))``. Populated lazily; read-only
#: after insertion (never mutated in place). Each of the 7 frozen
#: instruments appears in 6 of the 21 pairs, and the widened result for a
#: given instrument is a pure function of that instrument's own DEVELOPMENT
#: M1 data and the multiplier -- never pair-dependent -- so caching it here
#: is safe: no pair-specific state ever leaks between cache hits. In
#: multi-worker (``--workers`` > 1) runs each worker process holds its own
#: independent copy of this dict (no cross-process sharing/IPC, so no race
#: condition is possible); a worker that happens to process several pairs
#: sharing an instrument still benefits from within-process cache hits.
_stress_cache: dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame]] = {}


def _get_stressed_m1(
    instrument_id: str, bid_m1: pd.DataFrame, ask_m1: pd.DataFrame, multiplier: Decimal
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the ``multiplier``-stressed BID/ASK M1 frames for
    ``instrument_id``, computing them via
    :func:`~ftmoquant.research.alpha_lab.cost_stress.widen_bid_ask_frame`
    only on the first request for this ``(instrument_id, multiplier)``
    pair and reusing the cached, immutable result afterward -- eliminates
    the redundant per-config re-widening
    :mod:`~ftmoquant.research.alpha_lab.b3f1_development_orchestrator`'s
    B3.2b runtime audit found (756 widen calls -> at most 7 per process)."""

    key = (instrument_id, str(multiplier))
    cached = _stress_cache.get(key)
    if cached is not None:
        return cached
    stressed = widen_bid_ask_frame(bid_m1, ask_m1, float(multiplier))
    _stress_cache[key] = stressed
    return stressed


def run_pair_job(job: _PairJob) -> tuple[B3F1ScorecardRow, ...]:
    """Run all 18 frozen configs for ONE pair: compute each formation
    window's series exactly once, reuse it across the 6 (z_entry, z_stop)
    configs sharing that window, and execute the SAME decision sequence
    both natively and at 1.5x cost stress -- widened ONCE per instrument
    (see :data:`_stress_cache`), not once per config. Picklable/top-level
    so it can run in a worker process.

    Passing the already-stressed frames back into
    :func:`~ftmoquant.research.alpha_lab.b3f1_spread_execution.
    simulate_b3f1_intents` with ``cost_stress_multiplier=Decimal("1")`` (a
    no-op inside that function) is semantically IDENTICAL to letting it
    widen the frames itself with ``cost_stress_multiplier=1.5`` -- same
    function, same inputs, same output -- so no execution semantics change,
    only where the (now much cheaper, but still redundant-if-repeated)
    widening happens.
    """

    y_bid, y_ask = load_m1_bidask(
        instrument_id=job.pair.y.instrument_id,
        root=_leg_m1_root(job.pair.y, job.development_root_dir),
        start_utc=DEVELOPMENT_START,
        end_exclusive_utc=DEVELOPMENT_END_EXCLUSIVE,
    )
    x_bid, x_ask = load_m1_bidask(
        instrument_id=job.pair.x.instrument_id,
        root=_leg_m1_root(job.pair.x, job.development_root_dir),
        start_utc=DEVELOPMENT_START,
        end_exclusive_utc=DEVELOPMENT_END_EXCLUSIVE,
    )
    stress_multiplier = STRESS_MULTIPLIERS[0]
    stressed_y_bid, stressed_y_ask = _get_stressed_m1(
        job.pair.y.instrument_id, y_bid, y_ask, stress_multiplier
    )
    stressed_x_bid, stressed_x_ask = _get_stressed_m1(
        job.pair.x.instrument_id, x_bid, x_ask, stress_multiplier
    )

    rows: list[B3F1ScorecardRow] = []
    for window in FORMATION_WINDOWS:
        formation = compute_formation_series(job.log_y, job.log_x, window)
        for z_entry in Z_ENTRY_GRID:
            for z_stop in Z_STOP_GRID:
                decisions = generate_b3f1_decisions(
                    formation,
                    job.log_y,
                    job.log_x,
                    sleeve_id=job.pair.sleeve_id,
                    z_entry=z_entry,
                    z_stop=z_stop,
                )
                native_episodes, _ = simulate_b3f1_intents(
                    decisions,
                    y_spec=job.pair.y,
                    x_spec=job.pair.x,
                    y_bid_m1=y_bid,
                    y_ask_m1=y_ask,
                    x_bid_m1=x_bid,
                    x_ask_m1=x_ask,
                    gross_notional_usd=GROSS_NOTIONAL_USD,
                    cost_stress_multiplier=Decimal("1"),
                )
                stressed_episodes, _ = simulate_b3f1_intents(
                    decisions,
                    y_spec=job.pair.y,
                    x_spec=job.pair.x,
                    y_bid_m1=stressed_y_bid,
                    y_ask_m1=stressed_y_ask,
                    x_bid_m1=stressed_x_bid,
                    x_ask_m1=stressed_x_ask,
                    gross_notional_usd=GROSS_NOTIONAL_USD,
                    cost_stress_multiplier=Decimal("1"),
                )
                config = B3F1Config(window, z_entry, z_stop)
                rows.append(
                    evaluate_b3f1_config(
                        sleeve_id=job.pair.sleeve_id,
                        config=config,
                        native_episodes=native_episodes,
                        stressed_1_5x_episodes=stressed_episodes,
                        fold_boundaries=_FOLD_BOUNDARIES,
                    )
                )
    return tuple(rows)


# ---------------------------------------------------------------------------
# Pair robustness (independent per pair; pair identity never an adjacency
# axis; no cross-pair ranking).
# ---------------------------------------------------------------------------


def _cell_for_row(row: B3F1ScorecardRow) -> tuple[int, int, int]:
    return (
        FORMATION_WINDOWS.index(row.config.formation_window),
        Z_ENTRY_GRID.index(row.config.z_entry),
        Z_STOP_GRID.index(row.config.z_stop),
    )


def _strongest_region_parameters(
    rows: Sequence[B3F1ScorecardRow],
) -> dict[str, Any] | None:
    """Deterministic, non-performance-based representative cell within the
    LARGEST connected passing region: the cell minimizing the sum of
    Manhattan grid-index distances to every other cell in that region
    (ties broken by the smaller grid-index tuple) -- the identical
    algorithm already frozen for broad-FX VALIDATION representative
    selection in
    ``config/validation/oanda_alpha_lab_stage2_survivors_v1_preregistration.json``,
    reused here rather than re-derived. Purely geometric: never looks at
    expectancy, PF, or any other performance field."""

    passing = [row for row in rows if row.hard_gates_passed]
    if not passing:
        return None
    cells: dict[tuple[int, ...], B3F1ScorecardRow] = {
        _cell_for_row(row): row for row in passing
    }
    components = _connected_components(set(cells), _axis_adjacent_neighbors)
    largest = max(components, key=len)
    best_cell = min(
        largest,
        key=lambda cell: (
            sum(
                sum(abs(a - b) for a, b in zip(cell, other, strict=True))
                for other in largest
            ),
            cell,
        ),
    )
    row = cells[best_cell]
    return {
        "formation_window": row.config.formation_window,
        "z_entry": str(row.config.z_entry),
        "z_stop": str(row.config.z_stop),
    }


def compute_pair_robustness(
    sleeve_id: str, rows: Sequence[B3F1ScorecardRow]
) -> dict[str, Any]:
    """One pair's row of ``pair_robustness.csv`` -- computed entirely from
    THAT pair's own 18 rows, independent of every other pair. No global
    pair ranking."""

    passing_rows = [row for row in rows if row.hard_gates_passed]
    largest_region = largest_connected_passing_region(
        rows,
        formation_windows=FORMATION_WINDOWS,
        z_entries=Z_ENTRY_GRID,
        z_stops=Z_STOP_GRID,
    )
    passing_cells: set[tuple[int, ...]] = {_cell_for_row(row) for row in passing_rows}
    connected_region_count = 0
    if passing_cells:
        connected_region_count = len(
            _connected_components(passing_cells, _axis_adjacent_neighbors)
        )

    strongest = _strongest_region_parameters(rows)
    return {
        "sleeve_id": sleeve_id,
        "tested_configuration_count": len(rows),
        "passing_configuration_count": len(passing_rows),
        "connected_region_count": connected_region_count,
        "largest_connected_region_size": largest_region,
        "survival_rule_passed": largest_region >= MIN_CONNECTED_REGION_SIZE,
        "strongest_region_formation_window": (strongest or {}).get("formation_window"),
        "strongest_region_z_entry": (strongest or {}).get("z_entry"),
        "strongest_region_z_stop": (strongest or {}).get("z_stop"),
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
    h1_row_count: int,
    tested_pair_count: int,
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
        "pair_ids": [pair.sleeve_id for pair in enumerate_candidate_pairs()],
        "grid": {
            "formation_windows": list(FORMATION_WINDOWS),
            "z_entry_grid": [str(v) for v in Z_ENTRY_GRID],
            "z_stop_grid": [str(v) for v in Z_STOP_GRID],
            "configs_per_pair": len(FORMATION_WINDOWS)
            * len(Z_ENTRY_GRID)
            * len(Z_STOP_GRID),
        },
        "formation_statement": (
            "H1 causal formation: OLS log(Y) = alpha + beta*log(X) fit on "
            "[t-window, t) only; current spread uses bar t's own current "
            "log-mid-close combined with the lagged alpha/beta; z-score "
            "normalized by the same lagged window's residual mean/std"
        ),
        "ols_semantics": (
            "vectorized closed-form rolling OLS, proven numerically "
            "identical to statsmodels.api.OLS"
        ),
        "adf_settings": {
            "regression": "c",
            "autolag": "AIC",
            "p_value_threshold": 0.05,
        },
        "execution_semantics": {
            "primary": "native BID/ASK, M1, first strictly later paired observation",
            "cost_stress_multipliers": [str(m) for m in STRESS_MULTIPLIERS],
        },
        "hard_gates": {
            "min_trade_count": MIN_TRADE_COUNT,
            "profit_factor_gt": str(PROFIT_FACTOR_GT),
            "max_quarter_share": str(MAX_QUARTER_SHARE),
            "min_connected_region_size": MIN_CONNECTED_REGION_SIZE,
        },
        "diagnostics": [
            "rolling_30_median_expectancy",
            "rolling_30_fraction_positive",
            "rolling_50_median_expectancy",
            "rolling_50_fraction_positive",
            "monthly_max_share",
            "largest_trade_share",
            "pnl_skewness",
            "pnl_kurtosis",
            "rich_trade_count",
            "rich_expectancy",
            "rich_profit_factor",
            "cheap_trade_count",
            "cheap_expectancy",
            "cheap_profit_factor",
        ],
        "h1_aligned_row_count": h1_row_count,
        "tested_pair_count": tested_pair_count,
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


def run_b3f1_development_screen(
    *,
    development_root: Path,
    universe_readiness: Path,
    output_dir: Path,
    workers: int = 1,
) -> None:
    """The full real-DEVELOPMENT B3F1 screen. Write-once check happens
    FIRST, before any data loading (section 5)."""

    reject_forbidden_root(development_root, label="--development-root")
    reject_forbidden_root(universe_readiness, label="--universe-readiness")
    reserve_output_directory(output_dir)

    # Cache lifetime is exactly one orchestrator run, never longer: a
    # module-level cache that outlived this call could otherwise return a
    # stale stressed frame for an instrument_id if this function were ever
    # invoked twice in the same process against different data (as
    # multiple tests in one pytest session do).
    _stress_cache.clear()

    readiness_document = json.loads(universe_readiness.read_text(encoding="utf-8"))

    dataset = load_alpha_lab_dataset(
        readiness_path=universe_readiness,
        development_root_dir=development_root,
        timeframe="H1",
        source="oanda",
    )
    log_close = np.log(dataset.close)

    pairs = enumerate_candidate_pairs()
    jobs = [
        _PairJob(
            pair=pair,
            log_y=log_close[pair.y.instrument_id],
            log_x=log_close[pair.x.instrument_id],
            development_root_dir=development_root,
        )
        for pair in pairs
    ]

    if workers <= 1:
        results = [run_pair_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # .map preserves submission order regardless of completion
            # order -- deterministic output ordering.
            results = list(executor.map(run_pair_job, jobs))

    scorecard: list[B3F1ScorecardRow] = []
    pair_robustness: list[dict[str, Any]] = []
    for pair, rows in zip(pairs, results, strict=True):
        scorecard.extend(rows)
        pair_robustness.append(compute_pair_robustness(pair.sleeve_id, rows))

    metadata = build_metadata(
        readiness_document=readiness_document,
        h1_row_count=len(dataset.close),
        tested_pair_count=len(pairs),
        tested_config_count=len(scorecard),
    )

    write_b3f1_artifacts(
        scorecard=scorecard,
        pair_robustness=pair_robustness,
        metadata=metadata,
        output_dir=output_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen 21-pair x 18-config B3F1 (cross-instrument "
            "spread mean reversion) DEVELOPMENT screen against real OANDA "
            "M1 BID/ASK execution."
        )
    )
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--universe-readiness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of pair-level worker processes (deterministic output ordering).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_b3f1_development_screen(
        development_root=args.development_root,
        universe_readiness=args.universe_readiness,
        output_dir=args.output,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
