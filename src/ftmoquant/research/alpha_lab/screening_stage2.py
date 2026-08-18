"""Stage-2 DEVELOPMENT robustness screen for the three families that showed
promising regions in ``screening_batch_v1``. Tests whether each region
survives modest parameter perturbation -- not a search for a higher Sharpe.

Reuses :func:`~ftmoquant.research.alpha_lab.screening_batch.run_screening_batch`
and its underlying signal/execution/extraction code verbatim (Stage-2
configurations use the same family labels as Stage 1, so the existing
per-family dispatch and :func:`~ftmoquant.research.alpha_lab.
screening_common.run_family_grid` runner apply unchanged). The only new
logic here is the Stage-2 grid and the frozen family-survival gate.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from ftmoquant.research.alpha_lab.data import (
    AlphaLabDataset,
    DataSource,
    Timeframe,
    load_alpha_lab_dataset,
)
from ftmoquant.research.alpha_lab.screening_batch import (
    GridConfig,
    ScreeningBatchResult,
    run_screening_batch,
    write_results,
)
from ftmoquant.research.alpha_lab.screening_common import (
    AGGREGATE_INSTRUMENT_LABEL,
    ScreeningResultRow,
)

DONCHIAN_H4_LOOKBACKS: tuple[int, ...] = (35, 40, 45, 50, 55, 60, 70)
TSM_H4_LOOKBACKS: tuple[int, ...] = (25, 30, 40, 50, 60, 75, 90)
MEAN_REVERSION_LOOKBACKS: tuple[int, ...] = (30, 40, 50)
MEAN_REVERSION_Z_ENTRIES: tuple[float, ...] = (1.5, 2.0, 2.5)

DONCHIAN_FAMILY = "B_donchian_breakout"
TSM_FAMILY = "A_time_series_momentum"
MEAN_REVERSION_FAMILY = "C_short_horizon_mean_reversion"

#: Frozen BEFORE Stage-2 results were observed (task spec section 5). A
#: configuration passes the gate iff all three hold on its aggregate row.
MIN_STRESSED_RETURN = 0.0
MIN_PROFITABLE_PAIRS = 4
MIN_POSITIVE_FOLDS = 2

#: Minimum contiguous/connected passing-region size required for a family to
#: survive Stage 2. An isolated strong cell is never sufficient.
DONCHIAN_MIN_REGION_SIZE = 2
TSM_MIN_REGION_SIZE = 2
MEAN_REVERSION_MIN_REGION_SIZE = 3


def build_stage2_grid() -> list[GridConfig]:
    configs: list[GridConfig] = []
    for lookback in DONCHIAN_H4_LOOKBACKS:
        configs.append(
            GridConfig(
                family=DONCHIAN_FAMILY,
                strategy_id=f"stage2_donchian_h4_lookback{lookback}_v1",
                timeframe="H4",
                parameters={"lookback": lookback},
            )
        )
    for lookback in TSM_H4_LOOKBACKS:
        configs.append(
            GridConfig(
                family=TSM_FAMILY,
                strategy_id=f"stage2_tsm_h4_lookback{lookback}_v1",
                timeframe="H4",
                parameters={"lookback": lookback},
            )
        )
    for lookback in MEAN_REVERSION_LOOKBACKS:
        for z_entry in MEAN_REVERSION_Z_ENTRIES:
            configs.append(
                GridConfig(
                    family=MEAN_REVERSION_FAMILY,
                    strategy_id=f"stage2_meanrev_h1_lookback{lookback}_z{z_entry}_v1",
                    timeframe="H1",
                    parameters={"lookback": lookback, "z_entry": z_entry},
                )
            )
    return configs


@dataclass(frozen=True, slots=True)
class FamilyRobustnessSummary:
    family: str
    tested_configuration_count: int
    passing_configuration_count: int
    contiguous_passing_region_count: int
    largest_contiguous_region_size: int
    survival_rule_passed: bool
    strongest_region_parameters: str


def _passes_gate(row: ScreeningResultRow) -> bool:
    return (
        row.stressed_cost_return > MIN_STRESSED_RETURN
        and (row.profitable_pair_count or 0) >= MIN_PROFITABLE_PAIRS
        and (row.positive_fold_count or 0) >= MIN_POSITIVE_FOLDS
    )


def _connected_components(
    nodes: set[tuple[int, ...]],
    neighbors_of: Callable[[tuple[int, ...]], list[tuple[int, ...]]],
) -> list[list[tuple[int, ...]]]:
    seen: set[tuple[int, ...]] = set()
    components: list[list[tuple[int, ...]]] = []
    for start in nodes:
        if start in seen:
            continue
        stack = [start]
        component: list[tuple[int, ...]] = []
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            component.append(node)
            stack.extend(n for n in neighbors_of(node) if n in nodes and n not in seen)
        components.append(component)
    return components


def _aggregate_rows_by_family(
    rows: list[ScreeningResultRow], family: str
) -> list[ScreeningResultRow]:
    return [
        row
        for row in rows
        if row.family == family and row.instrument == AGGREGATE_INSTRUMENT_LABEL
    ]


def _analyze_1d_family(
    rows: list[ScreeningResultRow],
    *,
    family: str,
    param_name: str,
    sorted_values: tuple[int, ...],
    min_region_size: int,
) -> FamilyRobustnessSummary:
    row_by_value = {
        json.loads(row.parameters)[param_name]: row
        for row in _aggregate_rows_by_family(rows, family)
    }
    passing: set[tuple[int, ...]] = {
        (i,)
        for i, value in enumerate(sorted_values)
        if _passes_gate(row_by_value[value])
    }
    components = _connected_components(
        passing, lambda node: [(node[0] - 1,), (node[0] + 1,)]
    )
    return _summarize(
        family=family,
        tested_count=len(sorted_values),
        passing_count=len(passing),
        components=components,
        min_region_size=min_region_size,
        describe=lambda component: f"{param_name}="
        + ",".join(str(sorted_values[i[0]]) for i in sorted(component)),
    )


def _analyze_mean_reversion_family(
    rows: list[ScreeningResultRow],
) -> FamilyRobustnessSummary:
    row_by_cell: dict[tuple[int, ...], ScreeningResultRow] = {}
    for row in _aggregate_rows_by_family(rows, MEAN_REVERSION_FAMILY):
        params = json.loads(row.parameters)
        cell = (
            MEAN_REVERSION_LOOKBACKS.index(params["lookback"]),
            MEAN_REVERSION_Z_ENTRIES.index(params["z_entry"]),
        )
        row_by_cell[cell] = row
    passing: set[tuple[int, ...]] = {
        cell for cell, row in row_by_cell.items() if _passes_gate(row)
    }

    def neighbors(cell: tuple[int, ...]) -> list[tuple[int, ...]]:
        i, j = cell
        return [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]

    components = _connected_components(passing, neighbors)
    return _summarize(
        family=MEAN_REVERSION_FAMILY,
        tested_count=len(MEAN_REVERSION_LOOKBACKS) * len(MEAN_REVERSION_Z_ENTRIES),
        passing_count=len(passing),
        components=components,
        min_region_size=MEAN_REVERSION_MIN_REGION_SIZE,
        describe=lambda component: ";".join(
            f"lookback={MEAN_REVERSION_LOOKBACKS[i]},z_entry={MEAN_REVERSION_Z_ENTRIES[j]}"
            for i, j in sorted(component)
        ),
    )


def _summarize(
    *,
    family: str,
    tested_count: int,
    passing_count: int,
    components: list[list[tuple[int, ...]]],
    min_region_size: int,
    describe: Callable[[list[tuple[int, ...]]], str],
) -> FamilyRobustnessSummary:
    largest = max(components, key=len, default=[])
    return FamilyRobustnessSummary(
        family=family,
        tested_configuration_count=tested_count,
        passing_configuration_count=passing_count,
        contiguous_passing_region_count=len(components),
        largest_contiguous_region_size=len(largest),
        survival_rule_passed=len(largest) >= min_region_size,
        strongest_region_parameters=describe(largest) if largest else "none",
    )


def analyze_family_robustness(
    rows: list[ScreeningResultRow],
) -> list[FamilyRobustnessSummary]:
    return [
        _analyze_1d_family(
            rows,
            family=DONCHIAN_FAMILY,
            param_name="lookback",
            sorted_values=DONCHIAN_H4_LOOKBACKS,
            min_region_size=DONCHIAN_MIN_REGION_SIZE,
        ),
        _analyze_1d_family(
            rows,
            family=TSM_FAMILY,
            param_name="lookback",
            sorted_values=TSM_H4_LOOKBACKS,
            min_region_size=TSM_MIN_REGION_SIZE,
        ),
        _analyze_mean_reversion_family(rows),
    ]


def write_stage2_results(
    result: ScreeningBatchResult,
    summaries: list[FamilyRobustnessSummary],
    output_dir: Path,
) -> None:
    write_results(result, output_dir)
    fieldnames = list(asdict(summaries[0]).keys())
    with (output_dir / "family_robustness.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for summary in summaries:
            writer.writerow(asdict(summary))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Stage-2 DEVELOPMENT robustness screen (23 exact "
            "configurations across Donchian H4, TSM H4, and mean-reversion "
            "H1) and the frozen family-survival gate."
        )
    )
    parser.add_argument("--source", choices=("dukascopy", "oanda"), default="oanda")
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--universe-readiness", type=Path, required=True)
    parser.add_argument(
        "--universe-plan",
        type=Path,
        default=None,
        help="Dukascopy universe plan; required only for --source dukascopy",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    source: DataSource = args.source

    grid = build_stage2_grid()
    needed_timeframes: list[Timeframe] = sorted({config.timeframe for config in grid})
    datasets_by_timeframe: dict[Timeframe, AlphaLabDataset] = {
        timeframe: load_alpha_lab_dataset(
            readiness_path=args.universe_readiness,
            plan_path=None if source == "oanda" else args.universe_plan,
            development_root_dir=args.development_root,
            timeframe=timeframe,
            source=source,
        )
        for timeframe in needed_timeframes
    }

    result = run_screening_batch(datasets_by_timeframe=datasets_by_timeframe, grid=grid)
    summaries = analyze_family_robustness(list(result.rows))
    write_stage2_results(result, summaries, args.output)
    payload = [asdict(summary) for summary in summaries]
    print(json.dumps(payload, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
