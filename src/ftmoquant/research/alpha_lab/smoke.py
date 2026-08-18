"""One trivial fixed-rule VectorBT smoke screen across DEVELOPMENT FX pairs.

Proves mechanics only: canonical DEVELOPMENT data -> aligned pandas matrices
-> ``vectorbt.Portfolio`` -> a standardized result row per
``strategy_config x instrument`` (plus one equal-weight aggregate row).

No optimization, no strategy selection, no research conclusions.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import vectorbt as vbt  # type: ignore[import-untyped]

from ftmoquant.research.alpha_lab.data import (
    AlphaLabDataset,
    DataSource,
    Timeframe,
    load_alpha_lab_dataset,
)

STRATEGY_ID = "smoke_sma20_sma50_crossover_v1"
FAST_WINDOW = 20
SLOW_WINDOW = 50
AGGREGATE_INSTRUMENT_LABEL = "AGGREGATE_EQUAL_WEIGHT"

_TIMEFRAME_FREQ: dict[str, str] = {"M30": "30min", "H1": "1h", "H4": "4h"}

#: This is a screening-stage cost approximation, not a final execution
#: cost model. When aligned BID/ASK DEVELOPMENT data is available (it is,
#: for every instrument this smoke test runs on) the per-instrument
#: proportional fee is the mean relative half-spread computed over the
#: *entire* DEVELOPMENT window: mean((ask.close - bid.close) / mid.close) / 2,
#: applied by vectorbt as a proportional fee on both entry and exit. It is a
#: fixed, whole-sample estimate, not a bar-by-bar causal quantity: it is not
#: safe to use inside a per-bar decision rule, only as an offline screening
#: cost assumption.
SCREENING_COST_LABEL = "fixed_mean_half_spread_estimated_from_full_development"


@dataclass(frozen=True, slots=True)
class ScreeningResultRow:
    strategy_id: str
    instrument: str
    timeframe: str
    parameters: str
    start: str
    end: str
    observation_count: int
    trade_count: int
    cumulative_return: float
    annualized_return: float
    annualized_sharpe: float
    maximum_drawdown: float
    win_rate: float
    turnover_or_trade_activity: float
    screening_cost_assumption: str


@dataclass(frozen=True, slots=True)
class SmokeRunResult:
    rows: tuple[ScreeningResultRow, ...]
    metadata: dict[str, object] = field(default_factory=dict)


def _screening_cost_per_instrument(dataset: AlphaLabDataset) -> pd.Series:
    relative_spread = dataset.spread / dataset.close
    return (relative_spread.mean() / 2.0).astype(float)


def _sma_crossover_signals(
    close: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fast = vbt.MA.run(close, FAST_WINDOW, short_name="fast")
    slow = vbt.MA.run(close, SLOW_WINDOW, short_name="slow")
    entries = fast.ma_crossed_above(slow)
    exits = fast.ma_crossed_below(slow)
    entries.columns = entries.columns.droplevel(
        list(range(entries.columns.nlevels - 1))
    )
    exits.columns = exits.columns.droplevel(list(range(exits.columns.nlevels - 1)))
    return entries, exits


def _build_aggregate_portfolio(
    close: pd.DataFrame,
    entries: pd.DataFrame,
    exits: pd.DataFrame,
    *,
    per_column_fees: object,
    freq: str,
) -> object:
    """One equal-weight aggregate: every instrument gets its own equal
    ``init_cash`` sleeve and trades independently (``cash_sharing=False``);
    ``group_by=True`` only combines those sleeves' equity/stats into a
    single aggregate curve, which is their sum, not a shared-cash-pool
    portfolio competing for fills.
    """

    return vbt.Portfolio.from_signals(
        close,
        entries,
        exits,
        fees=per_column_fees,
        freq=freq,
        group_by=True,
        cash_sharing=False,
    )


def _row_from_portfolio(
    *,
    portfolio: object,
    instrument: str,
    timeframe: Timeframe,
    observation_count: int,
    start: str,
    end: str,
    cost_label: str,
) -> ScreeningResultRow:
    total_return = float(portfolio.total_return())  # type: ignore[attr-defined]
    annualized_return = float(portfolio.annualized_return())  # type: ignore[attr-defined]
    sharpe = float(portfolio.sharpe_ratio())  # type: ignore[attr-defined]
    max_dd = float(portfolio.max_drawdown())  # type: ignore[attr-defined]
    trade_count = int(portfolio.trades.count())  # type: ignore[attr-defined]
    win_rate = float(portfolio.trades.win_rate()) if trade_count else 0.0  # type: ignore[attr-defined]
    turnover = trade_count / observation_count if observation_count else 0.0
    return ScreeningResultRow(
        strategy_id=STRATEGY_ID,
        instrument=instrument,
        timeframe=timeframe,
        parameters=json.dumps(
            {"fast_window": FAST_WINDOW, "slow_window": SLOW_WINDOW}, sort_keys=True
        ),
        start=start,
        end=end,
        observation_count=observation_count,
        trade_count=trade_count,
        cumulative_return=total_return,
        annualized_return=annualized_return,
        annualized_sharpe=sharpe,
        maximum_drawdown=max_dd,
        win_rate=win_rate,
        turnover_or_trade_activity=turnover,
        screening_cost_assumption=cost_label,
    )


def run_sma_crossover_smoke(dataset: AlphaLabDataset) -> SmokeRunResult:
    """Run the fixed SMA20/SMA50 crossover independently per instrument, plus
    one equal-weight aggregate portfolio, using ``vectorbt.Portfolio`` directly."""

    if dataset.close.empty:
        raise ValueError("dataset has no aligned observations")

    freq = _TIMEFRAME_FREQ[dataset.timeframe]
    cost = _screening_cost_per_instrument(dataset)
    entries, exits = _sma_crossover_signals(dataset.close)
    observation_count = len(dataset.close.index)
    start = dataset.close.index[0].isoformat()
    end = dataset.close.index[-1].isoformat()

    rows: list[ScreeningResultRow] = []
    for instrument in dataset.instrument_ids:
        fee = float(cost[instrument])
        portfolio = vbt.Portfolio.from_signals(
            dataset.close[instrument],
            entries[instrument],
            exits[instrument],
            fees=fee,
            freq=freq,
        )
        rows.append(
            _row_from_portfolio(
                portfolio=portfolio,
                instrument=instrument,
                timeframe=dataset.timeframe,
                observation_count=observation_count,
                start=start,
                end=end,
                cost_label=f"{SCREENING_COST_LABEL}={fee:.8f}",
            )
        )

    if len(dataset.instrument_ids) > 1:
        aggregate_fee = float(cost.mean())
        per_column_fees = cost.reindex(dataset.close.columns).to_numpy().reshape(1, -1)
        aggregate_portfolio = _build_aggregate_portfolio(
            dataset.close,
            entries,
            exits,
            per_column_fees=per_column_fees,
            freq=freq,
        )
        rows.append(
            _row_from_portfolio(
                portfolio=aggregate_portfolio,
                instrument=AGGREGATE_INSTRUMENT_LABEL,
                timeframe=dataset.timeframe,
                observation_count=observation_count,
                start=start,
                end=end,
                cost_label=f"{SCREENING_COST_LABEL}_mean={aggregate_fee:.8f}",
            )
        )

    metadata = {
        "strategy_id": STRATEGY_ID,
        "timeframe": dataset.timeframe,
        "instrument_ids": list(dataset.instrument_ids),
        "alignment_policy": dataset.alignment_policy,
        "development_start_utc": dataset.start_utc.isoformat(),
        "development_end_exclusive_utc": dataset.end_exclusive_utc.isoformat(),
        "observation_count": observation_count,
        "screening_cost_model": SCREENING_COST_LABEL,
    }
    return SmokeRunResult(rows=tuple(rows), metadata=metadata)


def write_results(result: SmokeRunResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    fieldnames = list(asdict(result.rows[0]).keys())
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in result.rows:
            writer.writerow(asdict(row))
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(result.metadata, stream, sort_keys=True, indent=2)
        stream.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed SMA20/SMA50 vectorbt smoke screen across every "
            "discovered DEVELOPMENT FX pair, from either the rigorous "
            "Dukascopy lineage (default) or the prospective OANDA "
            "alpha-lab lineage."
        )
    )
    parser.add_argument(
        "--source", choices=("dukascopy", "oanda"), default="dukascopy"
    )
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--universe-readiness", type=Path, required=True)
    parser.add_argument(
        "--universe-plan",
        type=Path,
        default=Path("config/data/g1_4_fx_usd_liquid_v1.yaml"),
        help="Dukascopy universe plan; irrelevant and unused for --source oanda",
    )
    parser.add_argument("--timeframe", choices=("M30", "H1", "H4"), default="H1")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    source: DataSource = args.source
    dataset = load_alpha_lab_dataset(
        readiness_path=args.universe_readiness,
        plan_path=None if source == "oanda" else args.universe_plan,
        development_root_dir=args.development_root,
        timeframe=args.timeframe,
        source=source,
    )
    result = run_sma_crossover_smoke(dataset)
    write_results(result, args.output)
    print(json.dumps(result.metadata, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
