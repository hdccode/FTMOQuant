"""Shared result schema and per-configuration runner for the DEVELOPMENT
screening batch (:mod:`ftmoquant.research.alpha_lab.families`).

Not a strategy/optimizer framework: one dataclass, one cost helper, and one
function that calls ``vectorbt.Portfolio.from_signals`` directly per
instrument (plus one equal-weight aggregate). Each family module supplies
only its own small signal-generation function; everything below is reused
across families purely to keep the standardized scorecard schema consistent.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import vectorbt as vbt  # type: ignore[import-untyped]

from ftmoquant.research.alpha_lab.data import AlphaLabDataset, Timeframe

AGGREGATE_INSTRUMENT_LABEL = "AGGREGATE_EQUAL_WEIGHT"

#: Same screening-stage cost approximation as the smoke test: the
#: per-instrument proportional fee is the mean relative half-spread over the
#: *entire* DEVELOPMENT window. Fixed, whole-sample, offline only.
SCREENING_COST_LABEL = "fixed_mean_half_spread_estimated_from_full_development"
STRESSED_COST_MULTIPLIER = 1.5

_TIMEFRAME_FREQ: dict[str, str] = {"M30": "30min", "H1": "1h", "H4": "4h"}

#: Calendar-day annualization factor per bar (``365 days / bar length``),
#: matching vectorbt's own convention. Used only for the fold-diagnostic
#: Sharpe below (a slice of an already-computed return series) -- never for
#: the primary per-portfolio Sharpe, which vectorbt computes itself.
_ANNUALIZATION_FACTOR: dict[str, float] = {
    "M30": 365.0 * 24.0 * 2.0,
    "H1": 365.0 * 24.0,
    "H4": 365.0 * 6.0,
}

#: Diagnostic-only temporal folds within the frozen DEVELOPMENT window.
#: Not a walk-forward or optimization process: a fixed post-hoc slice of the
#: return series already produced by the single DEVELOPMENT-window backtest.
FOLD_WINDOWS: tuple[tuple[str, str, str], ...] = (
    ("fold_1_2020_2021", "2020-04-11", "2021-04-11"),
    ("fold_2_2021_2022", "2021-04-11", "2022-04-11"),
    ("fold_3_2022_2023", "2022-04-11", "2023-04-11"),
)


@dataclass(frozen=True, slots=True)
class ScreeningResultRow:
    family: str
    strategy_id: str
    instrument: str
    timeframe: str
    parameters: str
    observation_count: int
    trade_count: int
    cumulative_return: float
    annualized_return: float
    annualized_sharpe: float
    maximum_drawdown: float
    win_rate: float
    turnover_or_trade_activity: float
    base_cost_return: float
    stressed_cost_return: float
    # Cross-pair robustness, one value per *configuration* (repeated on
    # every row -- instrument and aggregate -- of that configuration).
    profitable_pair_count: int | None
    positive_sharpe_pair_count: int | None
    median_pair_sharpe: float | None
    worst_pair_sharpe: float | None
    aggregate_equal_weight_return: float | None
    aggregate_equal_weight_sharpe: float | None
    # Temporal fold diagnostics, also one value per configuration.
    positive_fold_count: int | None
    worst_fold_return: float | None
    median_fold_sharpe: float | None
    # Diagnostic ranking (section 5); filled in once all configurations of a
    # batch have been run, never during a single configuration's own run.
    diagnostic_rank: int | None


def screening_cost_per_instrument(dataset: AlphaLabDataset) -> pd.Series:
    relative_spread = dataset.spread / dataset.close
    return (relative_spread.mean() / 2.0).astype(float)


def _extract_stats(portfolio: Any, observation_count: int) -> dict[str, float]:
    trade_count = int(portfolio.trades.count())
    win_rate = float(portfolio.trades.win_rate()) if trade_count else 0.0
    return {
        "trade_count": float(trade_count),
        "cumulative_return": float(portfolio.total_return()),
        "annualized_return": float(portfolio.annualized_return()),
        "annualized_sharpe": float(portfolio.sharpe_ratio()),
        "maximum_drawdown": float(portfolio.max_drawdown()),
        "win_rate": win_rate,
        "turnover_or_trade_activity": trade_count / observation_count
        if observation_count
        else 0.0,
    }


def _build_portfolio(
    close: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    short_entries: pd.Series,
    short_exits: pd.Series,
    *,
    fees: float,
    freq: str,
) -> Any:
    return vbt.Portfolio.from_signals(
        close,
        entries,
        exits,
        short_entries=short_entries,
        short_exits=short_exits,
        fees=fees,
        freq=freq,
    )


def _build_aggregate_portfolio(
    close: pd.DataFrame,
    entries: pd.DataFrame,
    exits: pd.DataFrame,
    short_entries: pd.DataFrame,
    short_exits: pd.DataFrame,
    *,
    per_column_fees: Any,
    freq: str,
) -> Any:
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
        short_entries=short_entries,
        short_exits=short_exits,
        fees=per_column_fees,
        freq=freq,
        group_by=True,
        cash_sharing=False,
    )


def _fold_diagnostics(
    returns: pd.Series, timeframe: Timeframe
) -> tuple[int, float, float]:
    ann_factor = _ANNUALIZATION_FACTOR[timeframe]
    fold_returns: list[float] = []
    fold_sharpes: list[float] = []
    for _, start, end in FOLD_WINDOWS:
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC")
        window = returns.loc[(returns.index >= start_ts) & (returns.index < end_ts)]
        if window.empty:
            fold_returns.append(0.0)
            fold_sharpes.append(0.0)
            continue
        cumulative = float((1.0 + window).prod() - 1.0)
        std = float(window.std(ddof=1))
        mean = float(window.mean())
        sharpe = (mean / std) * float(np.sqrt(ann_factor)) if std > 0 else 0.0
        fold_returns.append(cumulative)
        fold_sharpes.append(sharpe)
    positive_fold_count = sum(1 for r in fold_returns if r > 0)
    worst_fold_return = min(fold_returns)
    median_fold_sharpe = float(np.median(fold_sharpes))
    return positive_fold_count, worst_fold_return, median_fold_sharpe


def _pair_robustness(
    per_instrument_stats: dict[str, dict[str, float]],
) -> tuple[int, int, float, float]:
    returns = [stats["cumulative_return"] for stats in per_instrument_stats.values()]
    sharpes = [stats["annualized_sharpe"] for stats in per_instrument_stats.values()]
    profitable = sum(1 for r in returns if r > 0)
    positive_sharpe = sum(1 for s in sharpes if s > 0)
    return profitable, positive_sharpe, float(np.median(sharpes)), float(min(sharpes))


def run_family_grid(
    *,
    dataset: AlphaLabDataset,
    family: str,
    strategy_id: str,
    parameters: Mapping[str, object],
    entries: pd.DataFrame,
    exits: pd.DataFrame,
    short_entries: pd.DataFrame,
    short_exits: pd.DataFrame,
) -> list[ScreeningResultRow]:
    """Run one fully-specified configuration -- independently per instrument,
    plus one equal-weight aggregate -- and return its standardized rows.

    ``entries``/``exits``/``short_entries``/``short_exits`` must already be
    boolean ``DataFrame``s aligned to ``dataset.close`` (same index and
    columns); each family module builds these from its own signal function.
    """

    if dataset.close.empty:
        raise ValueError("dataset has no aligned observations")

    freq = _TIMEFRAME_FREQ[dataset.timeframe]
    cost = screening_cost_per_instrument(dataset)
    observation_count = len(dataset.close.index)
    params_json = json.dumps(parameters, sort_keys=True)

    per_instrument_stats: dict[str, dict[str, float]] = {}
    for instrument in dataset.instrument_ids:
        fee = float(cost[instrument])
        base_pf = _build_portfolio(
            dataset.close[instrument],
            entries[instrument],
            exits[instrument],
            short_entries[instrument],
            short_exits[instrument],
            fees=fee,
            freq=freq,
        )
        stressed_pf = _build_portfolio(
            dataset.close[instrument],
            entries[instrument],
            exits[instrument],
            short_entries[instrument],
            short_exits[instrument],
            fees=fee * STRESSED_COST_MULTIPLIER,
            freq=freq,
        )
        stats = _extract_stats(base_pf, observation_count)
        stats["stressed_cost_return"] = float(stressed_pf.total_return())
        per_instrument_stats[instrument] = stats

    multi_instrument = len(dataset.instrument_ids) > 1
    profitable_pair_count: int | None = None
    positive_sharpe_pair_count: int | None = None
    median_pair_sharpe: float | None = None
    worst_pair_sharpe: float | None = None
    aggregate_stats: dict[str, float] | None = None
    positive_fold_count: int | None = None
    worst_fold_return: float | None = None
    median_fold_sharpe: float | None = None

    if multi_instrument:
        (
            profitable_pair_count,
            positive_sharpe_pair_count,
            median_pair_sharpe,
            worst_pair_sharpe,
        ) = _pair_robustness(per_instrument_stats)

        per_column_fees_base = (
            cost.reindex(dataset.close.columns).to_numpy().reshape(1, -1)
        )
        base_agg_pf = _build_aggregate_portfolio(
            dataset.close,
            entries,
            exits,
            short_entries,
            short_exits,
            per_column_fees=per_column_fees_base,
            freq=freq,
        )
        stressed_agg_pf = _build_aggregate_portfolio(
            dataset.close,
            entries,
            exits,
            short_entries,
            short_exits,
            per_column_fees=per_column_fees_base * STRESSED_COST_MULTIPLIER,
            freq=freq,
        )
        aggregate_stats = _extract_stats(base_agg_pf, observation_count)
        aggregate_stats["stressed_cost_return"] = float(stressed_agg_pf.total_return())
        positive_fold_count, worst_fold_return, median_fold_sharpe = _fold_diagnostics(
            base_agg_pf.returns(), dataset.timeframe
        )

    aggregate_equal_weight_return = (
        aggregate_stats["cumulative_return"] if aggregate_stats else None
    )
    aggregate_equal_weight_sharpe = (
        aggregate_stats["annualized_sharpe"] if aggregate_stats else None
    )

    def _row(instrument: str, stats: dict[str, float]) -> ScreeningResultRow:
        return ScreeningResultRow(
            family=family,
            strategy_id=strategy_id,
            instrument=instrument,
            timeframe=dataset.timeframe,
            parameters=params_json,
            observation_count=observation_count,
            trade_count=int(stats["trade_count"]),
            cumulative_return=stats["cumulative_return"],
            annualized_return=stats["annualized_return"],
            annualized_sharpe=stats["annualized_sharpe"],
            maximum_drawdown=stats["maximum_drawdown"],
            win_rate=stats["win_rate"],
            turnover_or_trade_activity=stats["turnover_or_trade_activity"],
            base_cost_return=stats["cumulative_return"],
            stressed_cost_return=stats["stressed_cost_return"],
            profitable_pair_count=profitable_pair_count,
            positive_sharpe_pair_count=positive_sharpe_pair_count,
            median_pair_sharpe=median_pair_sharpe,
            worst_pair_sharpe=worst_pair_sharpe,
            aggregate_equal_weight_return=aggregate_equal_weight_return,
            aggregate_equal_weight_sharpe=aggregate_equal_weight_sharpe,
            positive_fold_count=positive_fold_count,
            worst_fold_return=worst_fold_return,
            median_fold_sharpe=median_fold_sharpe,
            diagnostic_rank=None,
        )

    rows = [
        _row(instrument, per_instrument_stats[instrument])
        for instrument in dataset.instrument_ids
    ]
    if aggregate_stats is not None:
        rows.append(_row(AGGREGATE_INSTRUMENT_LABEL, aggregate_stats))
    return rows


def assign_diagnostic_ranks(
    rows: list[ScreeningResultRow],
) -> list[ScreeningResultRow]:
    """Rank configurations diagnostically (section 5 of the screening spec)
    by, in order: profitable pair count, median pair Sharpe, worst pair
    Sharpe, stressed aggregate return, aggregate Sharpe. Every row of a
    configuration (instrument rows and its aggregate row alike) receives the
    same ``diagnostic_rank``. This is a read-only sort of already-computed
    results -- no configuration is discarded or re-run.
    """

    aggregate_rows = [
        row for row in rows if row.instrument == AGGREGATE_INSTRUMENT_LABEL
    ]
    neg_inf = float("-inf")
    key_by_strategy: dict[str, tuple[float, float, float, float, float]] = {}
    for row in aggregate_rows:
        median_sharpe = row.median_pair_sharpe
        median_sharpe = median_sharpe if median_sharpe is not None else neg_inf
        worst_sharpe = row.worst_pair_sharpe
        worst_sharpe = worst_sharpe if worst_sharpe is not None else neg_inf
        key_by_strategy[row.strategy_id] = (
            -(row.profitable_pair_count or 0),
            -median_sharpe,
            -worst_sharpe,
            -row.stressed_cost_return,
            -row.annualized_sharpe,
        )
    ordered_strategy_ids = sorted(key_by_strategy, key=lambda sid: key_by_strategy[sid])
    rank_by_strategy = {
        sid: rank for rank, sid in enumerate(ordered_strategy_ids, start=1)
    }
    return [
        replace(row, diagnostic_rank=rank_by_strategy.get(row.strategy_id))
        for row in rows
    ]
