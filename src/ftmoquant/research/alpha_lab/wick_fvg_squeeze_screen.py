"""Frozen 18-configuration DEVELOPMENT screen: three independently-specified
strategy families (F1 failed-Bollinger-rejection, F2 fresh-FVG/order-block
mitigation, F3 volatility-squeeze breakout), each a 2x3 grid, across the
seven-pair OANDA alpha-lab universe, executed against real canonical M1
BID/ASK data (see
:mod:`ftmoquant.research.alpha_lab.wick_fvg_squeeze_execution`) rather than
the existing vectorbt same-bar-close screening approximation.

Reuses, rather than re-derives:

- :func:`ftmoquant.research.alpha_lab.data.load_alpha_lab_dataset` /
  ``data._discover_oanda_universe`` for the causal H1/H4/M30 decision input
  and its own DEVELOPMENT-only firewall (rejects validation/holdout paths,
  verifies the OANDA readiness manifest, pins the catalog tree hash).
- :data:`ftmoquant.research.alpha_lab.screening_common.FOLD_WINDOWS` for
  the three fixed DEVELOPMENT fold windows.
- :func:`ftmoquant.research.alpha_lab.screening_stage2._connected_components`
  and its
  :class:`~ftmoquant.research.alpha_lab.screening_stage2.FamilyRobustnessSummary`
  for the rook-adjacent connected-region family-survival rule.
- :data:`ftmoquant.strategies.mean_reversion_h1.FROZEN_UNIVERSE` for the
  exact seven frozen pairs.

Exploratory screening only: every one of the 18 configurations' results is
preserved, none is discarded, and no single winner is selected. The
per-configuration viability gate and the family connected-region rule are
both frozen *before* this module is ever run against real data -- neither
may be changed after seeing results.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from ftmoquant.research.alpha_lab import data as alpha_lab_data
from ftmoquant.research.alpha_lab.data import (
    AlphaLabDataset,
    Timeframe,
    load_alpha_lab_dataset,
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
    SkipRecord,
    Trade,
    load_m1_bidask,
    simulate_trades,
)
from ftmoquant.research.alpha_lab.wick_fvg_squeeze_signals import (
    SignalEvent,
    f1_failed_bollinger_rejection_signals,
    f2_fresh_fvg_mitigation_signals,
    f3_volatility_squeeze_breakout_signals,
)
from ftmoquant.research.stage_g import DEVELOPMENT_END_EXCLUSIVE, DEVELOPMENT_START
from ftmoquant.strategies.mean_reversion_h1 import FROZEN_UNIVERSE


class WickFvgSqueezeScreenError(ValueError):
    """Raised on any grid/universe/data-shape violation of the frozen
    experiment. Never silently substitutes a default."""


# ---------------------------------------------------------------------------
# The frozen 18-configuration grid (exactly 6 F1 + 6 F2 + 6 F3).
# ---------------------------------------------------------------------------

F1_FAMILY = "F1_failed_bollinger_rejection"
F2_FAMILY = "F2_fresh_fvg_mitigation"
F3_FAMILY = "F3_volatility_squeeze_breakout"

F1_TIMEFRAMES: tuple[Timeframe, ...] = ("M30", "H1")
F1_WICK_RATIOS: tuple[float, ...] = (0.50, 0.60, 0.70)
F2_TIMEFRAMES: tuple[Timeframe, ...] = ("H1", "H4")
F2_BACKWARD_SEARCH_N: tuple[int, ...] = (5, 10, 20)
F3_TIMEFRAMES: tuple[Timeframe, ...] = ("H1", "H4")
F3_BB_WIDTH_THRESHOLDS: tuple[float, ...] = (0.003, 0.004, 0.006)

#: Frozen BEFORE this screen was ever run against real data.
MIN_NET_RETURN = 0.0
MIN_PROFITABLE_PAIRS = 4
MIN_POSITIVE_FOLDS = 2
MIN_CONNECTED_REGION_SIZE = 2


@dataclass(frozen=True, slots=True)
class GridConfig:
    family: str
    strategy_id: str
    timeframe: Timeframe
    parameters: dict[str, int | float | str]


def build_grid() -> list[GridConfig]:
    configs: list[GridConfig] = []
    for timeframe in F1_TIMEFRAMES:
        for wick_ratio in F1_WICK_RATIOS:
            configs.append(
                GridConfig(
                    family=F1_FAMILY,
                    strategy_id=f"f1_{timeframe.lower()}_wick{wick_ratio}_v1",
                    timeframe=timeframe,
                    parameters={"wick_ratio": wick_ratio},
                )
            )
    for timeframe in F2_TIMEFRAMES:
        for backward_search_n in F2_BACKWARD_SEARCH_N:
            configs.append(
                GridConfig(
                    family=F2_FAMILY,
                    strategy_id=f"f2_{timeframe.lower()}_n{backward_search_n}_v1",
                    timeframe=timeframe,
                    parameters={"backward_search_n": backward_search_n},
                )
            )
    for timeframe in F3_TIMEFRAMES:
        for threshold in F3_BB_WIDTH_THRESHOLDS:
            configs.append(
                GridConfig(
                    family=F3_FAMILY,
                    strategy_id=f"f3_{timeframe.lower()}_thr{threshold}_v1",
                    timeframe=timeframe,
                    parameters={"bb_width_threshold": threshold},
                )
            )
    if len(configs) != 18:
        raise WickFvgSqueezeScreenError(
            f"frozen grid must have exactly 18 configurations, built {len(configs)}"
        )
    return configs


def _events_for_config(
    config: GridConfig, dataset: AlphaLabDataset, instrument_id: str
) -> tuple[SignalEvent, ...]:
    o = dataset.open[instrument_id]
    h = dataset.high[instrument_id]
    low = dataset.low[instrument_id]
    c = dataset.close[instrument_id]
    if config.family == F1_FAMILY:
        return f1_failed_bollinger_rejection_signals(
            o, h, low, c, wick_ratio=float(config.parameters["wick_ratio"])
        )
    if config.family == F2_FAMILY:
        return f2_fresh_fvg_mitigation_signals(
            o, h, low, c, backward_search_n=int(config.parameters["backward_search_n"])
        )
    if config.family == F3_FAMILY:
        return f3_volatility_squeeze_breakout_signals(
            h, low, c, bb_width_threshold=float(config.parameters["bb_width_threshold"])
        )
    raise WickFvgSqueezeScreenError(f"unknown family: {config.family}")


# ---------------------------------------------------------------------------
# Per-pair / aggregate statistics -- reimplemented locally with plain floats
# (rather than forcing reuse of mean_reversion_h1_development's Decimal,
# production-account-shaped PairPerformance/AggregatePerformance) because
# this screen has no notion of "realized variable cost" or a Nautilus
# account: costs are already embedded in the BID/ASK crossing itself. The
# *pattern* -- one EquityPoint-like curve per independent equal-capital
# sleeve, reduced to daily returns, equal-weight-averaged across pairs -- is
# the same identity documented in
# mean_reversion_h1_development._aggregate_daily_returns; this module keeps
# the per-day date alongside each return (that function intentionally
# discards it) because the frozen DEVELOPMENT fold windows need it.
# ---------------------------------------------------------------------------


def _annualized_sharpe(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    std = variance**0.5
    if std == 0.0:
        return None
    return float((mean / std) * (365.0**0.5))


@dataclass(frozen=True, slots=True)
class PairStats:
    instrument_id: str
    trade_count: int
    skip_count: int
    net_return: float
    annualized_sharpe: float | None
    maximum_drawdown: float
    win_rate: float
    daily_returns: tuple[tuple[date, float], ...]


def _pair_stats(
    instrument_id: str, trades: Sequence[Trade], skips: Sequence[SkipRecord]
) -> PairStats:
    if not trades:
        return PairStats(instrument_id, 0, len(skips), 0.0, None, 0.0, 0.0, ())

    ordered = sorted(trades, key=lambda trade: trade.exit_ts)
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    wins = 0
    points: list[tuple[pd.Timestamp, float]] = [(ordered[0].entry_ts, equity)]
    for trade in ordered:
        equity *= 1.0 + trade.return_frac
        if trade.return_frac > 0:
            wins += 1
        peak = max(peak, equity)
        max_dd = min(max_dd, equity / peak - 1.0)
        points.append((trade.exit_ts, equity))

    series = pd.Series(
        [value for _, value in points], index=pd.DatetimeIndex([ts for ts, _ in points])
    )
    daily = series.resample("1D").last().ffill()
    daily_returns_series = daily.pct_change().dropna()
    daily_returns = tuple(
        (ts.date(), float(value)) for ts, value in daily_returns_series.items()
    )
    return PairStats(
        instrument_id=instrument_id,
        trade_count=len(trades),
        skip_count=len(skips),
        net_return=equity - 1.0,
        annualized_sharpe=_annualized_sharpe([v for _, v in daily_returns]),
        maximum_drawdown=max_dd,
        win_rate=wins / len(trades),
        daily_returns=daily_returns,
    )


@dataclass(frozen=True, slots=True)
class AggregateStats:
    net_return: float
    annualized_sharpe: float | None
    maximum_drawdown: float
    profitable_pair_count: int
    positive_fold_count: int


def _aggregate_daily_returns(
    pair_stats: Mapping[str, PairStats],
) -> tuple[tuple[date, float], ...]:
    by_day: dict[date, dict[str, float]] = {}
    for instrument_id, stats in pair_stats.items():
        for day, value in stats.daily_returns:
            by_day.setdefault(day, {})[instrument_id] = value
    pair_ids = tuple(pair_stats)
    pair_count = len(pair_ids)
    return tuple(
        (
            day,
            sum(by_day[day].get(instrument_id, 0.0) for instrument_id in pair_ids)
            / pair_count,
        )
        for day in sorted(by_day)
    )


def _max_drawdown_from_returns(values: Sequence[float]) -> float:
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for value in values:
        equity *= 1.0 + value
        peak = max(peak, equity)
        max_dd = min(max_dd, equity / peak - 1.0)
    return max_dd


def _positive_fold_count(daily_returns: Sequence[tuple[date, float]]) -> int:
    count = 0
    for _, start, end in FOLD_WINDOWS:
        start_date = pd.Timestamp(start).date()
        end_date = pd.Timestamp(end).date()
        cumulative = 1.0
        for day, value in daily_returns:
            if start_date <= day < end_date:
                cumulative *= 1.0 + value
        if cumulative - 1.0 > 0:
            count += 1
    return count


def _pair_fold_returns(
    daily_returns: Sequence[tuple[date, float]],
) -> tuple[float, float, float]:
    """Per-fold cumulative net return over ``FOLD_WINDOWS``, computed from
    ONE pair's own realized daily-equity return series (never derived from
    the aggregate). Deliberately mirrors ``_positive_fold_count``'s own
    windowing/compounding arithmetic exactly (same ``FOLD_WINDOWS``, same
    half-open date filtering, same compounding order) rather than reusing
    it internally, so the frozen aggregate/broad-FX gate computation path
    is left completely untouched by this addition -- this function is
    purely additive.
    """

    fold_returns: list[float] = []
    for _, start, end in FOLD_WINDOWS:
        start_date = pd.Timestamp(start).date()
        end_date = pd.Timestamp(end).date()
        cumulative = 1.0
        for day, value in daily_returns:
            if start_date <= day < end_date:
                cumulative *= 1.0 + value
        fold_returns.append(cumulative - 1.0)
    first, second, third = fold_returns
    return first, second, third


def _aggregate_stats(pair_stats: Mapping[str, PairStats]) -> AggregateStats:
    if set(pair_stats) != set(FROZEN_UNIVERSE):
        raise WickFvgSqueezeScreenError(
            "aggregate statistics require exactly the seven frozen pairs"
        )
    net_return = sum(stats.net_return for stats in pair_stats.values()) / len(
        pair_stats
    )
    profitable_pair_count = sum(
        1 for stats in pair_stats.values() if stats.net_return > 0
    )
    aggregate_daily = _aggregate_daily_returns(pair_stats)
    values = [value for _, value in aggregate_daily]
    return AggregateStats(
        net_return=net_return,
        annualized_sharpe=_annualized_sharpe(values),
        maximum_drawdown=_max_drawdown_from_returns(values),
        profitable_pair_count=profitable_pair_count,
        positive_fold_count=_positive_fold_count(aggregate_daily),
    )


def _passes_gate(aggregate: AggregateStats) -> bool:
    return (
        aggregate.net_return > MIN_NET_RETURN
        and aggregate.profitable_pair_count >= MIN_PROFITABLE_PAIRS
        and aggregate.positive_fold_count >= MIN_POSITIVE_FOLDS
    )


# ---------------------------------------------------------------------------
# Standardized result row + orchestration.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WickFvgSqueezeResultRow:
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
    passes_gate: bool
    # Pair-specific fold diagnostics (added for the predeclared pair-specific
    # survival rule; see module docstring). Computed from THIS row's own
    # instrument's realized daily-equity return series -- never derived from
    # the aggregate. ``None`` on the AGGREGATE_EQUAL_WEIGHT row, where
    # "pair" has no meaning.
    fold_1_net_return: float | None = None
    fold_2_net_return: float | None = None
    fold_3_net_return: float | None = None
    positive_pair_fold_count: int | None = None
    pair_all_three_folds_positive: bool | None = None


@dataclass(frozen=True, slots=True)
class WickFvgSqueezeScreenResult:
    rows: tuple[WickFvgSqueezeResultRow, ...]
    family_summaries: tuple[FamilyRobustnessSummary, ...]
    pair_specific_summaries: tuple[PairSpecificRobustnessSummary, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)


def _rows_for_config(
    config: GridConfig,
    dataset: AlphaLabDataset,
    m1_by_instrument: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> list[WickFvgSqueezeResultRow]:
    pair_stats: dict[str, PairStats] = {}
    for instrument_id in FROZEN_UNIVERSE:
        events = _events_for_config(config, dataset, instrument_id)
        bid_m1, ask_m1 = m1_by_instrument[instrument_id]
        trades, skips = simulate_trades(events, bid_m1, ask_m1)
        pair_stats[instrument_id] = _pair_stats(instrument_id, trades, skips)

    aggregate = _aggregate_stats(pair_stats)
    gate_result = _passes_gate(aggregate)
    params_json = json.dumps(config.parameters, sort_keys=True)

    def _row(instrument: str, stats: PairStats | None) -> WickFvgSqueezeResultRow:
        pair_folds = (
            _pair_fold_returns(stats.daily_returns) if stats is not None else None
        )
        pair_positive_folds = (
            sum(1 for value in pair_folds if value > 0)
            if pair_folds is not None
            else None
        )
        return WickFvgSqueezeResultRow(
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
            passes_gate=gate_result,
            fold_1_net_return=pair_folds[0] if pair_folds is not None else None,
            fold_2_net_return=pair_folds[1] if pair_folds is not None else None,
            fold_3_net_return=pair_folds[2] if pair_folds is not None else None,
            positive_pair_fold_count=pair_positive_folds,
            pair_all_three_folds_positive=(
                pair_positive_folds == 3 if pair_positive_folds is not None else None
            ),
        )

    rows = [
        _row(instrument_id, pair_stats[instrument_id])
        for instrument_id in FROZEN_UNIVERSE
    ]
    rows.append(_row(AGGREGATE_INSTRUMENT_LABEL, None))
    return rows


def _analyze_2d_family(
    rows: Sequence[WickFvgSqueezeResultRow],
    *,
    family: str,
    timeframes: tuple[str, ...],
    param_name: str,
    param_values: tuple[float, ...] | tuple[int, ...],
) -> FamilyRobustnessSummary:
    row_by_cell: dict[tuple[int, int], WickFvgSqueezeResultRow] = {}
    for row in rows:
        if row.family != family or row.instrument != AGGREGATE_INSTRUMENT_LABEL:
            continue
        params = json.loads(row.parameters)
        cell = (timeframes.index(row.timeframe), param_values.index(params[param_name]))
        row_by_cell[cell] = row

    passing: set[tuple[int, ...]] = {
        cell for cell, row in row_by_cell.items() if row.passes_gate
    }

    def neighbors(cell: tuple[int, ...]) -> list[tuple[int, ...]]:
        i, j = cell
        return [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]

    components = _connected_components(passing, neighbors)
    largest = max(components, key=len, default=[])

    def describe(component: list[tuple[int, ...]]) -> str:
        return ";".join(
            f"timeframe={timeframes[i]},{param_name}={param_values[j]}"
            for i, j in sorted(component)
        )

    return FamilyRobustnessSummary(
        family=family,
        tested_configuration_count=len(timeframes) * len(param_values),
        passing_configuration_count=len(passing),
        contiguous_passing_region_count=len(components),
        largest_contiguous_region_size=len(largest),
        survival_rule_passed=len(largest) >= MIN_CONNECTED_REGION_SIZE,
        strongest_region_parameters=describe(largest) if largest else "none",
    )


def analyze_family_robustness(
    rows: Sequence[WickFvgSqueezeResultRow],
) -> tuple[FamilyRobustnessSummary, ...]:
    return (
        _analyze_2d_family(
            rows,
            family=F1_FAMILY,
            timeframes=F1_TIMEFRAMES,
            param_name="wick_ratio",
            param_values=F1_WICK_RATIOS,
        ),
        _analyze_2d_family(
            rows,
            family=F2_FAMILY,
            timeframes=F2_TIMEFRAMES,
            param_name="backward_search_n",
            param_values=F2_BACKWARD_SEARCH_N,
        ),
        _analyze_2d_family(
            rows,
            family=F3_FAMILY,
            timeframes=F3_TIMEFRAMES,
            param_name="bb_width_threshold",
            param_values=F3_BB_WIDTH_THRESHOLDS,
        ),
    )


# ---------------------------------------------------------------------------
# Pair-specific robustness (the predeclared pair-specific survival rule --
# see module docstring). A pair/config passes iff its own native-spread
# net_return > 0 AND that SAME pair is profitable in all 3 frozen
# DEVELOPMENT folds; a family survives on one pair iff that pair has >=2
# rook-adjacent passing configurations forming one connected region. Uses
# the EXACT same per-family adjacency (timeframe x the family's own single
# parameter) as the historical broad-FX analysis above -- never a diagonal
# connection, never a parameter value the original frozen grid did not
# test. Purely additive: does not read or alter ``passes_gate``/
# ``analyze_family_robustness``'s broad-FX computation.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PairSpecificRobustnessSummary:
    family: str
    instrument: str
    tested_configuration_count: int
    passing_configuration_count: int
    contiguous_passing_region_count: int
    largest_contiguous_region_size: int
    survival_rule_passed: bool
    strongest_region_parameters: str


def _passes_pair_specific_gate(row: WickFvgSqueezeResultRow) -> bool:
    return bool(
        row.net_return > MIN_NET_RETURN and row.pair_all_three_folds_positive
    )


def _rook_neighbors(cell: tuple[int, ...]) -> list[tuple[int, ...]]:
    i, j = cell
    return [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]


def _analyze_pair_specific_family(
    rows: Sequence[WickFvgSqueezeResultRow],
    *,
    family: str,
    instrument: str,
    timeframes: tuple[str, ...],
    param_name: str,
    param_values: tuple[float, ...] | tuple[int, ...],
) -> PairSpecificRobustnessSummary:
    row_by_cell: dict[tuple[int, int], WickFvgSqueezeResultRow] = {}
    for row in rows:
        if row.family != family or row.instrument != instrument:
            continue
        params = json.loads(row.parameters)
        cell = (timeframes.index(row.timeframe), param_values.index(params[param_name]))
        row_by_cell[cell] = row

    passing: set[tuple[int, ...]] = {
        cell for cell, row in row_by_cell.items() if _passes_pair_specific_gate(row)
    }

    components = _connected_components(passing, _rook_neighbors)
    largest = max(components, key=len, default=[])

    def describe(component: list[tuple[int, ...]]) -> str:
        return ";".join(
            f"timeframe={timeframes[i]},{param_name}={param_values[j]}"
            for i, j in sorted(component)
        )

    return PairSpecificRobustnessSummary(
        family=family,
        instrument=instrument,
        tested_configuration_count=len(timeframes) * len(param_values),
        passing_configuration_count=len(passing),
        contiguous_passing_region_count=len(components),
        largest_contiguous_region_size=len(largest),
        survival_rule_passed=len(largest) >= MIN_CONNECTED_REGION_SIZE,
        strongest_region_parameters=describe(largest) if largest else "none",
    )


#: (family, timeframes, parameter name, parameter values) -- the exact same
#: three (family, adjacency) specifications ``analyze_family_robustness``
#: already uses, reused verbatim rather than re-declared.
_FAMILY_GRID_SPECS: tuple[
    tuple[str, tuple[str, ...], str, tuple[float, ...] | tuple[int, ...]], ...
] = (
    (F1_FAMILY, F1_TIMEFRAMES, "wick_ratio", F1_WICK_RATIOS),
    (F2_FAMILY, F2_TIMEFRAMES, "backward_search_n", F2_BACKWARD_SEARCH_N),
    (F3_FAMILY, F3_TIMEFRAMES, "bb_width_threshold", F3_BB_WIDTH_THRESHOLDS),
)


def analyze_pair_specific_robustness(
    rows: Sequence[WickFvgSqueezeResultRow],
) -> tuple[PairSpecificRobustnessSummary, ...]:
    """One summary per family x pair (3 families x 7 pairs = 21 rows)."""

    summaries: list[PairSpecificRobustnessSummary] = []
    for family, timeframes, param_name, param_values in _FAMILY_GRID_SPECS:
        for instrument in FROZEN_UNIVERSE:
            summaries.append(
                _analyze_pair_specific_family(
                    rows,
                    family=family,
                    instrument=instrument,
                    timeframes=timeframes,
                    param_name=param_name,
                    param_values=param_values,
                )
            )
    return tuple(summaries)


def run_wick_fvg_squeeze_screen(
    *, readiness_path: Path, development_root_dir: Path
) -> WickFvgSqueezeScreenResult:
    """Run all 18 frozen configurations across the seven frozen pairs.

    DEVELOPMENT-only by construction: instrument discovery and every
    dataset load below go through
    :func:`ftmoquant.research.alpha_lab.data.load_alpha_lab_dataset` /
    ``data._discover_oanda_universe`` (source="oanda"), which reject any
    root path containing "validation"/"holdout" and verify the OANDA
    readiness manifest's own ``holdout_accessed is False`` /
    ``holdout_rows_admitted == 0`` fields and catalog-tree hash pin before
    any bar is read. The M1 execution stream is loaded from the identical,
    already-firewalled per-instrument roots returned by that same
    discovery call -- never from an independently supplied path.
    """

    grid = build_grid()

    instrument_ids, development_roots = alpha_lab_data._discover_oanda_universe(
        readiness_path, development_root_dir
    )
    if set(instrument_ids) != set(FROZEN_UNIVERSE):
        raise WickFvgSqueezeScreenError(
            "DEVELOPMENT universe must be exactly the seven frozen pairs"
        )

    needed_timeframes: tuple[Timeframe, ...] = ("M30", "H1", "H4")
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

    rows: list[WickFvgSqueezeResultRow] = []
    for config in grid:
        dataset = datasets_by_timeframe[config.timeframe]
        rows.extend(_rows_for_config(config, dataset, m1_by_instrument))

    summaries = analyze_family_robustness(rows)
    pair_specific_summaries = analyze_pair_specific_robustness(rows)
    metadata = {
        "configuration_count": len(grid),
        "families": [F1_FAMILY, F2_FAMILY, F3_FAMILY],
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
    return WickFvgSqueezeScreenResult(
        rows=tuple(rows),
        family_summaries=summaries,
        pair_specific_summaries=pair_specific_summaries,
        metadata=metadata,
    )


def write_results(result: WickFvgSqueezeScreenResult, output_dir: Path) -> None:
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

    summary_fieldnames = list(asdict(result.family_summaries[0]).keys())
    with (output_dir / "family_robustness.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=summary_fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        for summary in result.family_summaries:
            writer.writerow(asdict(summary))

    pair_specific_fieldnames = list(asdict(result.pair_specific_summaries[0]).keys())
    with (output_dir / "pair_specific_robustness.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=pair_specific_fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        for pair_summary in result.pair_specific_summaries:
            writer.writerow(asdict(pair_summary))

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(result.metadata, stream, sort_keys=True, indent=2)
        stream.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen 18-configuration F1/F2/F3 DEVELOPMENT screen "
            "(real OANDA M1 BID/ASK execution) across the seven-pair "
            "alpha-lab universe."
        )
    )
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--universe-readiness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_wick_fvg_squeeze_screen(
        readiness_path=args.universe_readiness,
        development_root_dir=args.development_root,
    )
    write_results(result, args.output)
    print(json.dumps(result.metadata, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
