"""First real DEVELOPMENT screening batch across the five implemented
families (Family F / ProBot is BLOCKED_RULE_RECONSTRUCTION -- see
:mod:`ftmoquant.research.alpha_lab.families`). Exact grid, no Optuna:
canonical DEVELOPMENT data -> aligned matrices (once per timeframe) ->
per-family signal function -> :func:`~ftmoquant.research.alpha_lab.
screening_common.run_family_grid` -> one standardized scorecard.

Exploratory screening only: every configuration's result is preserved, none
is discarded, and no single winner is selected here.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ftmoquant.research.alpha_lab.data import (
    AlphaLabDataset,
    DataSource,
    Timeframe,
    load_alpha_lab_dataset,
)
from ftmoquant.research.alpha_lab.families import (
    PROBOT_STATUS,
    PROBOT_STATUS_REASON,
    donchian_breakout_signals,
    mean_reversion_signals,
    momentum_signals,
    session_continuation_signals,
    volatility_breakout_signals,
)
from ftmoquant.research.alpha_lab.screening_common import (
    FOLD_WINDOWS,
    ScreeningResultRow,
    assign_diagnostic_ranks,
    run_family_grid,
)


@dataclass(frozen=True, slots=True)
class GridConfig:
    family: str
    strategy_id: str
    timeframe: Timeframe
    parameters: dict[str, int | float | str]


#: Exact grids, taken directly from the task's own bounded examples --
#: breadth of hypotheses, not fine tuning. 49 configurations total.
MOMENTUM_H1_LOOKBACKS: tuple[int, ...] = (24, 72, 120, 240, 480)
MOMENTUM_H4_LOOKBACKS: tuple[int, ...] = (6, 18, 30, 60, 120)
DONCHIAN_LOOKBACKS: tuple[int, ...] = (20, 50, 100)
DONCHIAN_TIMEFRAMES: tuple[Timeframe, ...] = ("H1", "H4")
MEAN_REVERSION_LOOKBACKS: tuple[int, ...] = (10, 20, 40)
MEAN_REVERSION_Z_ENTRIES: tuple[float, ...] = (1.0, 1.5, 2.0)
MEAN_REVERSION_TIMEFRAMES: tuple[Timeframe, ...] = ("M30", "H1")
VOL_BREAKOUT_LOOKBACKS: tuple[int, ...] = (20, 50)
VOL_BREAKOUT_THRESHOLDS: tuple[float, ...] = (1.0, 1.5, 2.0)
VOL_BREAKOUT_TIMEFRAMES: tuple[Timeframe, ...] = ("M30", "H1")
SESSIONS: tuple[str, ...] = ("london", "new_york", "london_new_york_overlap")


def build_grid() -> list[GridConfig]:
    configs: list[GridConfig] = []

    for lookback in MOMENTUM_H1_LOOKBACKS:
        configs.append(
            GridConfig(
                family="A_time_series_momentum",
                strategy_id=f"tsm_h1_lookback{lookback}_v1",
                timeframe="H1",
                parameters={"lookback": lookback},
            )
        )
    for lookback in MOMENTUM_H4_LOOKBACKS:
        configs.append(
            GridConfig(
                family="A_time_series_momentum",
                strategy_id=f"tsm_h4_lookback{lookback}_v1",
                timeframe="H4",
                parameters={"lookback": lookback},
            )
        )

    for timeframe in DONCHIAN_TIMEFRAMES:
        for lookback in DONCHIAN_LOOKBACKS:
            configs.append(
                GridConfig(
                    family="B_donchian_breakout",
                    strategy_id=f"donchian_{timeframe.lower()}_lookback{lookback}_v1",
                    timeframe=timeframe,
                    parameters={"lookback": lookback},
                )
            )

    for timeframe in MEAN_REVERSION_TIMEFRAMES:
        for lookback in MEAN_REVERSION_LOOKBACKS:
            for z_entry in MEAN_REVERSION_Z_ENTRIES:
                configs.append(
                    GridConfig(
                        family="C_short_horizon_mean_reversion",
                        strategy_id=(
                            f"meanrev_{timeframe.lower()}_lookback{lookback}"
                            f"_z{z_entry}_v1"
                        ),
                        timeframe=timeframe,
                        parameters={"lookback": lookback, "z_entry": z_entry},
                    )
                )

    for timeframe in VOL_BREAKOUT_TIMEFRAMES:
        for lookback in VOL_BREAKOUT_LOOKBACKS:
            for threshold in VOL_BREAKOUT_THRESHOLDS:
                configs.append(
                    GridConfig(
                        family="D_volatility_breakout",
                        strategy_id=(
                            f"volbreak_{timeframe.lower()}_lookback{lookback}"
                            f"_thr{threshold}_v1"
                        ),
                        timeframe=timeframe,
                        parameters={"lookback": lookback, "threshold": threshold},
                    )
                )

    for session in SESSIONS:
        configs.append(
            GridConfig(
                family="E_session_effect_baseline",
                strategy_id=f"session_continuation_{session}_m30_v1",
                timeframe="M30",
                parameters={"session": session},
            )
        )

    return configs


def _signals_for_config(
    config: GridConfig, dataset: AlphaLabDataset
) -> tuple[Any, Any, Any, Any]:
    if config.family == "A_time_series_momentum":
        lookback = int(config.parameters["lookback"])
        return momentum_signals(dataset.close, lookback)
    if config.family == "B_donchian_breakout":
        lookback = int(config.parameters["lookback"])
        return donchian_breakout_signals(
            dataset.high, dataset.low, dataset.close, lookback
        )
    if config.family == "C_short_horizon_mean_reversion":
        lookback = int(config.parameters["lookback"])
        z_entry = float(config.parameters["z_entry"])
        return mean_reversion_signals(dataset.close, lookback, z_entry)
    if config.family == "D_volatility_breakout":
        lookback = int(config.parameters["lookback"])
        threshold = float(config.parameters["threshold"])
        return volatility_breakout_signals(
            dataset.high, dataset.low, dataset.close, lookback, threshold
        )
    if config.family == "E_session_effect_baseline":
        session = str(config.parameters["session"])
        return session_continuation_signals(dataset.close, session)
    raise ValueError(f"unknown family: {config.family}")


@dataclass(frozen=True, slots=True)
class ScreeningBatchResult:
    rows: tuple[ScreeningResultRow, ...]
    metadata: dict[str, object] = field(default_factory=dict)


def run_screening_batch(
    *,
    datasets_by_timeframe: dict[Timeframe, AlphaLabDataset],
    grid: list[GridConfig] | None = None,
) -> ScreeningBatchResult:
    configs = grid if grid is not None else build_grid()
    rows: list[ScreeningResultRow] = []
    for config in configs:
        dataset = datasets_by_timeframe[config.timeframe]
        entries, exits, short_entries, short_exits = _signals_for_config(
            config, dataset
        )
        rows.extend(
            run_family_grid(
                dataset=dataset,
                family=config.family,
                strategy_id=config.strategy_id,
                parameters=config.parameters,
                entries=entries,
                exits=exits,
                short_entries=short_entries,
                short_exits=short_exits,
            )
        )
    rows = assign_diagnostic_ranks(rows)

    metadata: dict[str, object] = {
        "configuration_count": len(configs),
        "families_implemented": sorted({config.family for config in configs}),
        "probot_status": PROBOT_STATUS,
        "probot_status_reason": PROBOT_STATUS_REASON,
        "fold_windows": [
            {"label": label, "start_utc": start, "end_exclusive_utc": end}
            for label, start, end in FOLD_WINDOWS
        ],
    }
    return ScreeningBatchResult(rows=tuple(rows), metadata=metadata)


def write_results(result: ScreeningBatchResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    fieldnames = list(asdict(result.rows[0]).keys())
    scorecard_path = output_dir / "scorecard.csv"
    with scorecard_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in result.rows:
            record = asdict(row)
            for key, value in record.items():
                if value is None:
                    record[key] = ""
            writer.writerow(record)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(result.metadata, stream, sort_keys=True, indent=2)
        stream.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the first real vectorbt DEVELOPMENT screening batch (five "
            "families, exact grid) across the seven-pair OANDA alpha-lab "
            "universe."
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

    grid = build_grid()
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
    write_results(result, args.output)
    print(json.dumps(result.metadata, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
