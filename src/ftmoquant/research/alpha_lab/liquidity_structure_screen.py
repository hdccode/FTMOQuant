"""Batch 2: four funded-account-oriented structural DEVELOPMENT screens
(B2-F1 liquidity-sweep -> BOS -> retest, B2-F2 previous-day/week
liquidity-sweep rejection, B2-F3 liquidity-sweep -> CHOCH -> retracement,
B2-F4 Orion-style compression-box breakout) across the seven-pair OANDA
alpha-lab universe, executed against real canonical M1 BID/ASK data.

Reuses, rather than re-derives, everything Batch 1
(:mod:`ftmoquant.research.alpha_lab.wick_fvg_squeeze_screen`) already
built:

- :func:`~ftmoquant.research.alpha_lab.wick_fvg_squeeze_execution.simulate_trades`
  / ``load_m1_bidask`` -- the M1 BID/ASK trade-lifecycle engine, unchanged.
  Every Batch-2 stop is a fixed price and every target an R-multiple, the
  exact convention Batch 1's F2 family already exercised.
- ``wick_fvg_squeeze_screen``'s ``PairStats``/``AggregateStats``/
  ``_pair_stats``/``_aggregate_stats``/``_annualized_sharpe``/
  ``_max_drawdown_from_returns``/``_positive_fold_count``/``_passes_gate``/
  ``GridConfig`` -- the per-pair/equal-weight-aggregate statistics pipeline
  and the broad-FX viability gate are IDENTICAL rules to Batch 1's, reused
  by direct import rather than copied.
- :data:`ftmoquant.research.alpha_lab.screening_common.FOLD_WINDOWS` /
  ``AGGREGATE_INSTRUMENT_LABEL``.
- :func:`ftmoquant.research.alpha_lab.screening_stage2._connected_components`
  and its
  :class:`~ftmoquant.research.alpha_lab.screening_stage2.FamilyRobustnessSummary`,
  generalized here from Batch 1's 2-D rook adjacency to N-D axis adjacency
  (B2-F1's grid is 3-D: timeframe x swing_lookback x RR) via
  :func:`_axis_adjacent_neighbors`.
- :func:`ftmoquant.research.alpha_lab.data.load_alpha_lab_dataset` /
  ``data._discover_oanda_universe`` for the causal decision input and its
  DEVELOPMENT-only firewall.
- :data:`ftmoquant.strategies.mean_reversion_h1.FROZEN_UNIVERSE`.

Batch 2 is new only in: the four signal families themselves
(:mod:`ftmoquant.research.alpha_lab.liquidity_structure_signals`), the
31-configuration grid, and a SECOND, pair-specific survival track alongside
the broad-FX one Batch 1 already had.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from ftmoquant.research.alpha_lab import data as alpha_lab_data
from ftmoquant.research.alpha_lab.data import (
    AlphaLabDataset,
    Timeframe,
    load_alpha_lab_dataset,
)
from ftmoquant.research.alpha_lab.liquidity_structure_signals import (
    b2f1_sweep_bos_retest_signals,
    b2f2_previous_period_sweep_rejection_signals,
    b2f3_sweep_choch_retracement_signals,
    b2f4_compression_box_breakout_signals,
)
from ftmoquant.research.alpha_lab.screening_common import (
    AGGREGATE_INSTRUMENT_LABEL,
    FOLD_WINDOWS,
)
from ftmoquant.research.alpha_lab.screening_stage2 import (
    FamilyRobustnessSummary,
    _connected_components,
)
from ftmoquant.research.alpha_lab.wick_fvg_squeeze_execution import (
    load_m1_bidask,
    simulate_trades,
)
from ftmoquant.research.alpha_lab.wick_fvg_squeeze_screen import (
    MIN_CONNECTED_REGION_SIZE,
    GridConfig,
    PairStats,
    _aggregate_stats,
    _pair_stats,
    _passes_gate,
    _positive_fold_count,
)
from ftmoquant.research.alpha_lab.wick_fvg_squeeze_signals import SignalEvent
from ftmoquant.research.stage_g import DEVELOPMENT_END_EXCLUSIVE, DEVELOPMENT_START
from ftmoquant.strategies.mean_reversion_h1 import FROZEN_UNIVERSE


class LiquidityStructureScreenError(ValueError):
    """Raised on any grid/universe/data-shape violation of the frozen
    experiment. Never silently substitutes a default."""


# ---------------------------------------------------------------------------
# The frozen 31-configuration grid (12 + 4 + 6 + 9).
# ---------------------------------------------------------------------------

B2F1_FAMILY = "B2F1_sweep_bos_retest"
B2F2_FAMILY = "B2F2_previous_period_sweep_rejection"
B2F3_FAMILY = "B2F3_sweep_choch_retracement"
B2F4_FAMILY = "B2F4_compression_box_breakout"

B2F1_TIMEFRAMES: tuple[Timeframe, ...] = ("M30", "H1")
B2F1_SWING_LOOKBACKS: tuple[int, ...] = (10, 20, 40)
B2F1_RR_VALUES: tuple[float, ...] = (1.5, 2.0)

B2F2_LEVEL_TYPES: tuple[str, ...] = ("PREVIOUS_DAY", "PREVIOUS_WEEK")
B2F2_REJECTION_MODES: tuple[str, ...] = ("CLOSE_BACK", "WICK_50")

B2F3_SWING_LOOKBACKS: tuple[int, ...] = (10, 20, 40)
B2F3_RETRACEMENTS: tuple[float, ...] = (0.50, 0.618)

B2F4_BOX_LENGTHS: tuple[int, ...] = (10, 15, 20)
B2F4_MAX_BOX_ATRS: tuple[float, ...] = (2.0, 2.5, 3.0)

#: Frozen BEFORE this screen was ever run against real data.
MIN_NET_RETURN = 0.0
MIN_PROFITABLE_PAIRS = 4
MIN_POSITIVE_FOLDS = 2


def build_grid() -> list[GridConfig]:
    configs: list[GridConfig] = []
    for timeframe in B2F1_TIMEFRAMES:
        for swing_lookback in B2F1_SWING_LOOKBACKS:
            for rr in B2F1_RR_VALUES:
                configs.append(
                    GridConfig(
                        family=B2F1_FAMILY,
                        strategy_id=(
                            f"b2f1_{timeframe.lower()}_sw{swing_lookback}_rr{rr}_v1"
                        ),
                        timeframe=timeframe,
                        parameters={"swing_lookback": swing_lookback, "rr": rr},
                    )
                )
    for level_type in B2F2_LEVEL_TYPES:
        for rejection_mode in B2F2_REJECTION_MODES:
            configs.append(
                GridConfig(
                    family=B2F2_FAMILY,
                    strategy_id=f"b2f2_h1_{level_type.lower()}_{rejection_mode.lower()}_v1",
                    timeframe="H1",
                    parameters={
                        "level_type": level_type,
                        "rejection_mode": rejection_mode,
                    },
                )
            )
    for swing_lookback in B2F3_SWING_LOOKBACKS:
        for retracement in B2F3_RETRACEMENTS:
            configs.append(
                GridConfig(
                    family=B2F3_FAMILY,
                    strategy_id=f"b2f3_m30_sw{swing_lookback}_r{retracement}_v1",
                    timeframe="M30",
                    parameters={
                        "swing_lookback": swing_lookback,
                        "retracement": retracement,
                    },
                )
            )
    for box_length in B2F4_BOX_LENGTHS:
        for max_box_atr in B2F4_MAX_BOX_ATRS:
            configs.append(
                GridConfig(
                    family=B2F4_FAMILY,
                    strategy_id=f"b2f4_h1_box{box_length}_atr{max_box_atr}_v1",
                    timeframe="H1",
                    parameters={
                        "box_length": box_length,
                        "max_box_atr": max_box_atr,
                    },
                )
            )
    if len(configs) != 31:
        raise LiquidityStructureScreenError(
            f"frozen grid must have exactly 31 configurations, built {len(configs)}"
        )
    return configs


def _events_for_config(
    config: GridConfig, dataset: AlphaLabDataset, instrument_id: str
) -> tuple[SignalEvent, ...]:
    o = dataset.open[instrument_id]
    h = dataset.high[instrument_id]
    low = dataset.low[instrument_id]
    c = dataset.close[instrument_id]
    if config.family == B2F1_FAMILY:
        return b2f1_sweep_bos_retest_signals(
            h,
            low,
            c,
            swing_lookback=int(config.parameters["swing_lookback"]),
            rr=float(config.parameters["rr"]),
        )
    if config.family == B2F2_FAMILY:
        return b2f2_previous_period_sweep_rejection_signals(
            o,
            h,
            low,
            c,
            level_type=str(config.parameters["level_type"]),
            rejection_mode=str(config.parameters["rejection_mode"]),
        )
    if config.family == B2F3_FAMILY:
        return b2f3_sweep_choch_retracement_signals(
            h,
            low,
            c,
            swing_lookback=int(config.parameters["swing_lookback"]),
            retracement=float(config.parameters["retracement"]),
        )
    if config.family == B2F4_FAMILY:
        return b2f4_compression_box_breakout_signals(
            h,
            low,
            c,
            box_length=int(config.parameters["box_length"]),
            max_box_atr=float(config.parameters["max_box_atr"]),
        )
    raise LiquidityStructureScreenError(f"unknown family: {config.family}")


# ---------------------------------------------------------------------------
# Standardized result row + orchestration.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LiquidityStructureResultRow:
    family: str
    strategy_id: str
    instrument: str
    timeframe: str
    parameters: str
    trade_count: int
    skip_count: int
    net_return: float
    annualized_sharpe: float | None
    maximum_drawdown: float
    win_rate: float
    profitable_pair_count: int
    aggregate_equal_weight_net_return: float
    aggregate_equal_weight_sharpe: float | None
    aggregate_maximum_drawdown: float
    positive_fold_count: int
    passes_broad_gate: bool
    pair_all_three_folds_positive: bool
    passes_pair_specific_gate: bool


@dataclass(frozen=True, slots=True)
class LiquidityStructureScreenResult:
    rows: tuple[LiquidityStructureResultRow, ...]
    family_robustness: tuple[FamilyRobustnessSummary, ...]
    pair_specific_robustness: tuple[FamilyRobustnessSummary, ...]
    metadata: dict[str, object] = field(default_factory=dict)


def _rows_for_config(
    config: GridConfig,
    dataset: AlphaLabDataset,
    m1_by_instrument: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> list[LiquidityStructureResultRow]:
    pair_stats: dict[str, PairStats] = {}
    for instrument_id in FROZEN_UNIVERSE:
        events = _events_for_config(config, dataset, instrument_id)
        bid_m1, ask_m1 = m1_by_instrument[instrument_id]
        trades, skips = simulate_trades(events, bid_m1, ask_m1)
        pair_stats[instrument_id] = _pair_stats(instrument_id, trades, skips)

    aggregate = _aggregate_stats(pair_stats)
    broad_pass = _passes_gate(aggregate)
    params_json = json.dumps(config.parameters, sort_keys=True)

    def _row(instrument: str, stats: PairStats | None) -> LiquidityStructureResultRow:
        pair_all_three = (
            bool(stats) and _positive_fold_count(stats.daily_returns) == 3
            if stats
            else False
        )
        pair_specific_pass = bool(stats and stats.net_return > 0 and pair_all_three)
        return LiquidityStructureResultRow(
            family=config.family,
            strategy_id=config.strategy_id,
            instrument=instrument,
            timeframe=config.timeframe,
            parameters=params_json,
            trade_count=stats.trade_count if stats else 0,
            skip_count=stats.skip_count if stats else 0,
            net_return=stats.net_return if stats else aggregate.net_return,
            annualized_sharpe=(
                stats.annualized_sharpe if stats else aggregate.annualized_sharpe
            ),
            maximum_drawdown=(
                stats.maximum_drawdown if stats else aggregate.maximum_drawdown
            ),
            win_rate=stats.win_rate if stats else 0.0,
            profitable_pair_count=aggregate.profitable_pair_count,
            aggregate_equal_weight_net_return=aggregate.net_return,
            aggregate_equal_weight_sharpe=aggregate.annualized_sharpe,
            aggregate_maximum_drawdown=aggregate.maximum_drawdown,
            positive_fold_count=aggregate.positive_fold_count,
            passes_broad_gate=broad_pass,
            pair_all_three_folds_positive=pair_all_three,
            passes_pair_specific_gate=pair_specific_pass,
        )

    rows = [
        _row(instrument_id, pair_stats[instrument_id])
        for instrument_id in FROZEN_UNIVERSE
    ]
    rows.append(_row(AGGREGATE_INSTRUMENT_LABEL, None))
    return rows


# ---------------------------------------------------------------------------
# N-D axis-adjacent connected-region robustness (generalizes Batch 1's 2-D
# ``_analyze_2d_family``; B2-F1's grid is 3-D).
# ---------------------------------------------------------------------------


def _axis_adjacent_neighbors(cell: tuple[int, ...]) -> list[tuple[int, ...]]:
    neighbors: list[tuple[int, ...]] = []
    for axis in range(len(cell)):
        for delta in (-1, 1):
            neighbor = list(cell)
            neighbor[axis] += delta
            neighbors.append(tuple(neighbor))
    return neighbors


#: One (parameter name, ordered grid values) pair per non-timeframe axis.
_ParamSpec = tuple[str, tuple[object, ...]]


def _cell_for_row(
    row: LiquidityStructureResultRow,
    *,
    timeframe_values: tuple[str, ...] | None,
    param_specs: tuple[_ParamSpec, ...],
) -> tuple[int, ...]:
    cell: list[int] = []
    if timeframe_values is not None:
        cell.append(timeframe_values.index(row.timeframe))
    params = json.loads(row.parameters)
    for name, values in param_specs:
        cell.append(values.index(params[name]))
    return tuple(cell)


def _describe_cell(
    cell: tuple[int, ...],
    *,
    timeframe_values: tuple[str, ...] | None,
    param_specs: tuple[_ParamSpec, ...],
) -> str:
    parts: list[str] = []
    index = 0
    if timeframe_values is not None:
        parts.append(f"timeframe={timeframe_values[cell[index]]}")
        index += 1
    for name, values in param_specs:
        parts.append(f"{name}={values[cell[index]]}")
        index += 1
    return ",".join(parts)


def _summarize_region(
    *,
    family: str,
    passing: set[tuple[int, ...]],
    tested_count: int,
    timeframe_values: tuple[str, ...] | None,
    param_specs: tuple[_ParamSpec, ...],
    min_region_size: int,
) -> FamilyRobustnessSummary:
    components = _connected_components(passing, _axis_adjacent_neighbors)
    largest = max(components, key=len, default=[])
    strongest = (
        ";".join(
            _describe_cell(
                cell, timeframe_values=timeframe_values, param_specs=param_specs
            )
            for cell in sorted(largest)
        )
        if largest
        else "none"
    )
    return FamilyRobustnessSummary(
        family=family,
        tested_configuration_count=tested_count,
        passing_configuration_count=len(passing),
        contiguous_passing_region_count=len(components),
        largest_contiguous_region_size=len(largest),
        survival_rule_passed=len(largest) >= min_region_size,
        strongest_region_parameters=strongest,
    )


def _grid_axes(family: str) -> tuple[tuple[str, ...] | None, tuple[_ParamSpec, ...]]:
    if family == B2F1_FAMILY:
        return B2F1_TIMEFRAMES, (
            ("swing_lookback", B2F1_SWING_LOOKBACKS),
            ("rr", B2F1_RR_VALUES),
        )
    if family == B2F2_FAMILY:
        return None, (
            ("level_type", B2F2_LEVEL_TYPES),
            ("rejection_mode", B2F2_REJECTION_MODES),
        )
    if family == B2F3_FAMILY:
        return None, (
            ("swing_lookback", B2F3_SWING_LOOKBACKS),
            ("retracement", B2F3_RETRACEMENTS),
        )
    if family == B2F4_FAMILY:
        return None, (
            ("box_length", B2F4_BOX_LENGTHS),
            ("max_box_atr", B2F4_MAX_BOX_ATRS),
        )
    raise LiquidityStructureScreenError(f"unknown family: {family}")


def _tested_count(
    timeframe_values: tuple[str, ...] | None, param_specs: tuple[_ParamSpec, ...]
) -> int:
    total = len(timeframe_values) if timeframe_values is not None else 1
    for _, values in param_specs:
        total *= len(values)
    return total


def analyze_broad_family_robustness(
    rows: Sequence[LiquidityStructureResultRow],
) -> tuple[FamilyRobustnessSummary, ...]:
    summaries: list[FamilyRobustnessSummary] = []
    for family in (B2F1_FAMILY, B2F2_FAMILY, B2F3_FAMILY, B2F4_FAMILY):
        timeframe_values, param_specs = _grid_axes(family)
        row_by_cell: dict[tuple[int, ...], LiquidityStructureResultRow] = {}
        for row in rows:
            if row.family != family or row.instrument != AGGREGATE_INSTRUMENT_LABEL:
                continue
            row_by_cell[
                _cell_for_row(
                    row, timeframe_values=timeframe_values, param_specs=param_specs
                )
            ] = row
        passing = {cell for cell, row in row_by_cell.items() if row.passes_broad_gate}
        summaries.append(
            _summarize_region(
                family=family,
                passing=passing,
                tested_count=_tested_count(timeframe_values, param_specs),
                timeframe_values=timeframe_values,
                param_specs=param_specs,
                min_region_size=MIN_CONNECTED_REGION_SIZE,
            )
        )
    return tuple(summaries)


def analyze_pair_specific_robustness(
    rows: Sequence[LiquidityStructureResultRow],
) -> tuple[FamilyRobustnessSummary, ...]:
    """One summary per family x pair (28 total): the SAME connected-region
    machinery as the broad-FX track, but gated per pair by
    ``passes_pair_specific_gate`` instead of the aggregate row's
    ``passes_broad_gate`` -- no cross-pair breadth requirement."""

    summaries: list[FamilyRobustnessSummary] = []
    for family in (B2F1_FAMILY, B2F2_FAMILY, B2F3_FAMILY, B2F4_FAMILY):
        timeframe_values, param_specs = _grid_axes(family)
        for pair in FROZEN_UNIVERSE:
            row_by_cell: dict[tuple[int, ...], LiquidityStructureResultRow] = {}
            for row in rows:
                if row.family != family or row.instrument != pair:
                    continue
                row_by_cell[
                    _cell_for_row(
                        row, timeframe_values=timeframe_values, param_specs=param_specs
                    )
                ] = row
            passing = {
                cell
                for cell, row in row_by_cell.items()
                if row.passes_pair_specific_gate
            }
            summary = _summarize_region(
                family=f"{family}::{pair}",
                passing=passing,
                tested_count=_tested_count(timeframe_values, param_specs),
                timeframe_values=timeframe_values,
                param_specs=param_specs,
                min_region_size=MIN_CONNECTED_REGION_SIZE,
            )
            summaries.append(summary)
    return tuple(summaries)


def run_liquidity_structure_screen(
    *, readiness_path: Path, development_root_dir: Path
) -> LiquidityStructureScreenResult:
    """Run all 31 frozen configurations across the seven frozen pairs.

    DEVELOPMENT-only by construction, identically to Batch 1: instrument
    discovery and every dataset/M1 load go through
    ``data._discover_oanda_universe`` / ``load_alpha_lab_dataset``
    (source="oanda"), which reject any root path containing
    "validation"/"holdout" and verify the OANDA readiness manifest before
    any bar is read.
    """

    grid = build_grid()

    instrument_ids, development_roots = alpha_lab_data._discover_oanda_universe(
        readiness_path, development_root_dir
    )
    if set(instrument_ids) != set(FROZEN_UNIVERSE):
        raise LiquidityStructureScreenError(
            "DEVELOPMENT universe must be exactly the seven frozen pairs"
        )

    needed_timeframes: tuple[Timeframe, ...] = ("M30", "H1")
    datasets_by_timeframe: dict[Timeframe, AlphaLabDataset] = {
        timeframe: load_alpha_lab_dataset(
            readiness_path=readiness_path,
            development_root_dir=development_root_dir,
            timeframe=timeframe,
            source="oanda",
        )
        for timeframe in needed_timeframes
    }

    m1_by_instrument: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {
        instrument_id: load_m1_bidask(
            instrument_id=instrument_id,
            root=development_roots[instrument_id],
            start_utc=DEVELOPMENT_START,
            end_exclusive_utc=DEVELOPMENT_END_EXCLUSIVE,
        )
        for instrument_id in FROZEN_UNIVERSE
    }

    rows: list[LiquidityStructureResultRow] = []
    for config in grid:
        dataset = datasets_by_timeframe[config.timeframe]
        rows.extend(_rows_for_config(config, dataset, m1_by_instrument))

    broad_summaries = analyze_broad_family_robustness(rows)
    pair_specific_summaries = analyze_pair_specific_robustness(rows)
    metadata = {
        "configuration_count": len(grid),
        "families": [B2F1_FAMILY, B2F2_FAMILY, B2F3_FAMILY, B2F4_FAMILY],
        "universe": list(FROZEN_UNIVERSE),
        "fold_windows": [
            {"label": label, "start_utc": start, "end_exclusive_utc": end}
            for label, start, end in FOLD_WINDOWS
        ],
        "min_net_return": MIN_NET_RETURN,
        "min_profitable_pairs": MIN_PROFITABLE_PAIRS,
        "min_positive_folds": MIN_POSITIVE_FOLDS,
        "min_connected_region_size": MIN_CONNECTED_REGION_SIZE,
        "pair_specific_gate": "net_return > 0 AND all 3 DEVELOPMENT folds positive",
    }
    return LiquidityStructureScreenResult(
        rows=tuple(rows),
        family_robustness=broad_summaries,
        pair_specific_robustness=pair_specific_summaries,
        metadata=metadata,
    )


def write_results(result: LiquidityStructureScreenResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    fieldnames = list(asdict(result.rows[0]).keys())
    with (output_dir / "scorecard.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in result.rows:
            record = asdict(row)
            for key, value in record.items():
                if value is None:
                    record[key] = ""
            writer.writerow(record)

    summary_fieldnames = list(asdict(result.family_robustness[0]).keys())
    with (output_dir / "family_robustness.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=summary_fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        for summary in result.family_robustness:
            writer.writerow(asdict(summary))

    with (output_dir / "pair_specific_robustness.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=summary_fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        for summary in result.pair_specific_robustness:
            writer.writerow(asdict(summary))

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(result.metadata, stream, sort_keys=True, indent=2)
        stream.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen 31-configuration Batch-2 structural DEVELOPMENT "
            "screen (real OANDA M1 BID/ASK execution) across the "
            "seven-pair alpha-lab universe."
        )
    )
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--universe-readiness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_liquidity_structure_screen(
        readiness_path=args.universe_readiness,
        development_root_dir=args.development_root,
    )
    write_results(result, args.output)
    print(json.dumps(result.metadata, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
