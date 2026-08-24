"""B3F2 DEVELOPMENT screen: wires the causal signal
(:mod:`ftmoquant.research.alpha_lab.b3f2_asian_range_fade_signals`) and
execution (:mod:`ftmoquant.research.alpha_lab.b3f2_asian_range_fade_execution`)
layers into the frozen v2 hard gates and diagnostics, per instrument sleeve
and per statistical config.

Reuses, rather than re-derives:

- Thresholds read PROGRAMMATICALLY from
  :data:`ftmoquant.research.alpha_lab.batch3_preregistration_v2.DEVELOPMENT_GATES`
  -- identical convention to
  :mod:`ftmoquant.research.alpha_lab.b3f1_spread_screen`.
- :func:`ftmoquant.research.alpha_lab.screening_stage2._connected_components`
  / :func:`ftmoquant.research.alpha_lab.liquidity_structure_screen.
  _axis_adjacent_neighbors` for the 3-D (excursion_min x stop_buffer x
  target_mode) parameter-neighborhood robustness check.
- The gate/diagnostic ARITHMETIC (expectancy, profit factor, best-5%
  removal, quarter concentration, fold counting, rolling windows,
  skew/kurtosis) is the identical logic already proven in
  ``b3f1_spread_screen.py`` -- reimplemented here (not imported) only
  because B3F1's row/gate dataclasses are shaped around its 2-leg
  RelativeValueEpisode inputs, whereas B3F2 is single-instrument; the
  underlying formulas are unchanged.

Per the B3.2 precedent: B3.4's FTMO Monte Carlo/pass_both selection is NOT
implemented here -- that is explicitly a later stage for alpha survivors.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from ftmoquant.research.alpha_lab.b3f2_asian_range_fade_execution import B3F2Trade
from ftmoquant.research.alpha_lab.b3f2_asian_range_fade_signals import (
    EXCURSION_MIN_GRID,
    EXIT_REASON_STOP,
    EXIT_REASON_TARGET,
    EXIT_REASON_TIME,
    STOP_BUFFER_GRID,
    TARGET_MODE_GRID,
    B3F2Config,
)
from ftmoquant.research.alpha_lab.batch3_preregistration_v2 import DEVELOPMENT_GATES
from ftmoquant.research.alpha_lab.liquidity_structure_screen import (
    _axis_adjacent_neighbors,
)
from ftmoquant.research.alpha_lab.screening_stage2 import _connected_components

FAMILY_ID = "B3F2_asian_range_fade"
MIN_TRADE_COUNT = DEVELOPMENT_GATES["opportunity_density"]["default_min_trades"]
assert FAMILY_ID not in DEVELOPMENT_GATES["opportunity_density"]["family_overrides"]
EXPECTANCY_GT = Decimal(
    str(DEVELOPMENT_GATES["economic"]["expectancy_usd_per_trade_gt"])
)
PROFIT_FACTOR_GT = Decimal(str(DEVELOPMENT_GATES["economic"]["profit_factor_gt"]))
MAX_QUARTER_SHARE = Decimal(
    str(DEVELOPMENT_GATES["profit_concentration"]["max_single_quarter_share"])
)
BEST_5PCT_REMAINING_EXPECTANCY_GT = Decimal(
    str(
        DEVELOPMENT_GATES["exceptional_winner_dependency"][
            "remaining_expectancy_usd_per_trade_gt"
        ]
    )
)
MIN_POSITIVE_FOLDS = 3
MIN_CONNECTED_REGION_SIZE = DEVELOPMENT_GATES["parameter_neighborhood_robustness"][
    "min_connected_region_size"
]
STRESS_MULTIPLIERS = tuple(
    Decimal(str(value))
    for value in DEVELOPMENT_GATES["transaction_cost_sensitivity"][
        "family_requirements"
    ][FAMILY_ID]["must_survive_multipliers"]
)
assert STRESS_MULTIPLIERS == (Decimal("1.5"), Decimal("2.0"))

ARTIFACT_ROOT = Path(".artifacts/alpha_lab/batch3_b3f2_asian_range_fade_v1")


class B3F2ScreenError(ValueError):
    """Raised on any violation of the frozen B3F2 screen contract."""


@dataclass(frozen=True, slots=True)
class _TradeRow:
    exit_ts: datetime
    pnl: Decimal
    direction: int
    exit_reason: str
    holding_seconds: float


def _rows_from_trades(trades: Sequence[B3F2Trade]) -> tuple[_TradeRow, ...]:
    rows = [
        _TradeRow(
            exit_ts=datetime.fromtimestamp(trade.exit_ns / 1_000_000_000, tz=UTC),
            pnl=trade.realized_pnl_usd,
            direction=trade.direction,
            exit_reason=trade.exit_reason,
            holding_seconds=(trade.exit_ns - trade.entry_ns) / 1_000_000_000,
        )
        for trade in trades
    ]
    return tuple(sorted(rows, key=lambda row: row.exit_ts))


def expectancy_and_profit_factor(pnls: Sequence[Decimal]) -> tuple[Decimal, Decimal]:
    if not pnls:
        raise B3F2ScreenError("expectancy/profit-factor requires at least one trade")
    expectancy = sum(pnls, Decimal(0)) / len(pnls)
    gross_profit = sum((p for p in pnls if p > 0), Decimal(0))
    gross_loss = sum((-p for p in pnls if p < 0), Decimal(0))
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = Decimal("Infinity")
    else:
        profit_factor = Decimal(0)
    return expectancy, profit_factor


def best_5pct_removed_expectancy(rows: Sequence[_TradeRow]) -> Decimal:
    if not rows:
        raise B3F2ScreenError("best-5%% removal requires at least one trade")
    profitable = [row for row in rows if row.pnl > 0]
    remove_count = math.ceil(Decimal("0.05") * len(profitable)) if profitable else 0
    ranked = sorted(profitable, key=lambda row: (-row.pnl, row.exit_ts))
    removed_ids = {id(row) for row in ranked[:remove_count]}
    remaining = [row for row in rows if id(row) not in removed_ids]
    if not remaining:
        return Decimal(0)
    return sum((row.pnl for row in remaining), Decimal(0)) / len(remaining)


def _period_share(rows: Sequence[_TradeRow], *, quarter: bool) -> Decimal | None:
    positive = [row for row in rows if row.pnl > 0]
    total_positive = sum((row.pnl for row in positive), Decimal(0))
    if total_positive <= 0:
        return None
    buckets: dict[tuple[int, int], Decimal] = {}
    for row in positive:
        period = ((row.exit_ts.month - 1) // 3 + 1) if quarter else row.exit_ts.month
        key = (row.exit_ts.year, period)
        buckets[key] = buckets.get(key, Decimal(0)) + row.pnl
    return max(buckets.values()) / total_positive


def quarter_max_share(rows: Sequence[_TradeRow]) -> Decimal | None:
    return _period_share(rows, quarter=True)


def monthly_max_share(rows: Sequence[_TradeRow]) -> Decimal | None:
    return _period_share(rows, quarter=False)


def fold_positive_count(
    rows: Sequence[_TradeRow], fold_boundaries: Sequence[datetime]
) -> int:
    if len(fold_boundaries) < 2:
        raise B3F2ScreenError("fold_boundaries must contain at least 2 edges")
    positive = 0
    for start, end in zip(fold_boundaries[:-1], fold_boundaries[1:], strict=False):
        total = sum((row.pnl for row in rows if start <= row.exit_ts < end), Decimal(0))
        if total > 0:
            positive += 1
    return positive


def largest_trade_share(rows: Sequence[_TradeRow]) -> Decimal | None:
    positive = [row.pnl for row in rows if row.pnl > 0]
    total_positive = sum(positive, Decimal(0))
    if total_positive <= 0:
        return None
    return max(positive) / total_positive


def rolling_diagnostics(
    rows: Sequence[_TradeRow], window: int
) -> tuple[Decimal | None, Decimal | None]:
    if len(rows) < window:
        return None, None
    pnls = [row.pnl for row in rows]
    window_means = [
        sum(pnls[index - window + 1 : index + 1], Decimal(0)) / window
        for index in range(window - 1, len(pnls))
    ]
    median = statistics.median(window_means)
    fraction_positive = Decimal(sum(1 for value in window_means if value > 0)) / len(
        window_means
    )
    return median, fraction_positive


def _skew_kurtosis(pnls: Sequence[Decimal]) -> tuple[float | None, float | None]:
    if len(pnls) < 3:
        return None, None
    from scipy import stats as scipy_stats  # type: ignore[import-untyped]

    values = [float(p) for p in pnls]
    return float(scipy_stats.skew(values)), float(scipy_stats.kurtosis(values))


def _direction_stats(
    rows: Sequence[_TradeRow], direction: int
) -> tuple[int, Decimal | None, Decimal | None]:
    subset = [row.pnl for row in rows if row.direction == direction]
    if not subset:
        return 0, None, None
    expectancy, profit_factor = expectancy_and_profit_factor(subset)
    return len(subset), expectancy, profit_factor


def _exit_reason_fraction(rows: Sequence[_TradeRow], reason: str) -> Decimal | None:
    if not rows:
        return None
    count = sum(1 for row in rows if row.exit_reason == reason)
    return Decimal(count) / len(rows)


def _holding_time_stats(rows: Sequence[_TradeRow]) -> tuple[float | None, float | None]:
    if not rows:
        return None, None
    holdings = [row.holding_seconds for row in rows]
    return statistics.mean(holdings), statistics.median(holdings)


@dataclass(frozen=True, slots=True)
class B3F2ScorecardRow:
    instrument_id: str
    config: B3F2Config

    native_trade_count: int
    native_expectancy: Decimal
    native_profit_factor: Decimal
    fold_positive_count: int
    best_5pct_removed_expectancy: Decimal
    quarter_max_share: Decimal | None
    stressed_1_5x_trade_count: int
    stressed_1_5x_expectancy: Decimal
    stressed_1_5x_profit_factor: Decimal
    stressed_2_0x_trade_count: int
    stressed_2_0x_expectancy: Decimal
    stressed_2_0x_profit_factor: Decimal
    hard_gates_passed: bool

    # --- diagnostics (report-only; never gate) ---
    rolling_30_median_expectancy: Decimal | None
    rolling_30_fraction_positive: Decimal | None
    rolling_50_median_expectancy: Decimal | None
    rolling_50_fraction_positive: Decimal | None
    monthly_max_share: Decimal | None
    largest_trade_share: Decimal | None
    pnl_skewness: float | None
    pnl_kurtosis: float | None
    long_trade_count: int
    long_expectancy: Decimal | None
    long_profit_factor: Decimal | None
    short_trade_count: int
    short_expectancy: Decimal | None
    short_profit_factor: Decimal | None
    stop_fraction: Decimal | None
    target_fraction: Decimal | None
    time_exit_fraction: Decimal | None
    mean_holding_seconds: float | None
    median_holding_seconds: float | None


def evaluate_b3f2_config(
    *,
    instrument_id: str,
    config: B3F2Config,
    native_trades: Sequence[B3F2Trade],
    stressed_1_5x_trades: Sequence[B3F2Trade],
    stressed_2_0x_trades: Sequence[B3F2Trade],
    fold_boundaries: Sequence[datetime],
) -> B3F2ScorecardRow:
    """Compute one full scorecard row from already-executed native and
    stressed trade lists for one (instrument, config) cell. Pure function:
    no data access, no randomness."""

    native_rows = _rows_from_trades(native_trades)
    trade_count = len(native_rows)

    if trade_count == 0:
        native_expectancy = Decimal(0)
        native_pf = Decimal(0)
        best5 = Decimal(0)
        folds = 0
        quarter_share: Decimal | None = None
    else:
        native_expectancy, native_pf = expectancy_and_profit_factor(
            [row.pnl for row in native_rows]
        )
        best5 = best_5pct_removed_expectancy(native_rows)
        folds = fold_positive_count(native_rows, fold_boundaries)
        quarter_share = quarter_max_share(native_rows)

    stressed_1_5x_rows = _rows_from_trades(stressed_1_5x_trades)
    stressed_2_0x_rows = _rows_from_trades(stressed_2_0x_trades)
    if stressed_1_5x_rows:
        stressed_1_5x_expectancy, stressed_1_5x_pf = expectancy_and_profit_factor(
            [row.pnl for row in stressed_1_5x_rows]
        )
    else:
        stressed_1_5x_expectancy, stressed_1_5x_pf = Decimal(0), Decimal(0)
    if stressed_2_0x_rows:
        stressed_2_0x_expectancy, stressed_2_0x_pf = expectancy_and_profit_factor(
            [row.pnl for row in stressed_2_0x_rows]
        )
    else:
        stressed_2_0x_expectancy, stressed_2_0x_pf = Decimal(0), Decimal(0)

    hard_gates_passed = (
        trade_count >= MIN_TRADE_COUNT
        and native_expectancy > EXPECTANCY_GT
        and native_pf > PROFIT_FACTOR_GT
        and folds >= MIN_POSITIVE_FOLDS
        and best5 > BEST_5PCT_REMAINING_EXPECTANCY_GT
        and quarter_share is not None
        and quarter_share <= MAX_QUARTER_SHARE
        and len(stressed_1_5x_rows) > 0
        and stressed_1_5x_expectancy > EXPECTANCY_GT
        and len(stressed_2_0x_rows) > 0
        and stressed_2_0x_expectancy > EXPECTANCY_GT
    )

    rolling_30_median, rolling_30_frac = rolling_diagnostics(native_rows, 30)
    rolling_50_median, rolling_50_frac = rolling_diagnostics(native_rows, 50)
    skew, kurtosis = _skew_kurtosis([row.pnl for row in native_rows])
    long_count, long_expectancy, long_pf = _direction_stats(native_rows, 1)
    short_count, short_expectancy, short_pf = _direction_stats(native_rows, -1)
    mean_holding, median_holding = _holding_time_stats(native_rows)

    return B3F2ScorecardRow(
        instrument_id=instrument_id,
        config=config,
        native_trade_count=trade_count,
        native_expectancy=native_expectancy,
        native_profit_factor=native_pf,
        fold_positive_count=folds,
        best_5pct_removed_expectancy=best5,
        quarter_max_share=quarter_share,
        stressed_1_5x_trade_count=len(stressed_1_5x_rows),
        stressed_1_5x_expectancy=stressed_1_5x_expectancy,
        stressed_1_5x_profit_factor=stressed_1_5x_pf,
        stressed_2_0x_trade_count=len(stressed_2_0x_rows),
        stressed_2_0x_expectancy=stressed_2_0x_expectancy,
        stressed_2_0x_profit_factor=stressed_2_0x_pf,
        hard_gates_passed=hard_gates_passed,
        rolling_30_median_expectancy=rolling_30_median,
        rolling_30_fraction_positive=rolling_30_frac,
        rolling_50_median_expectancy=rolling_50_median,
        rolling_50_fraction_positive=rolling_50_frac,
        monthly_max_share=monthly_max_share(native_rows),
        largest_trade_share=largest_trade_share(native_rows),
        pnl_skewness=skew,
        pnl_kurtosis=kurtosis,
        long_trade_count=long_count,
        long_expectancy=long_expectancy,
        long_profit_factor=long_pf,
        short_trade_count=short_count,
        short_expectancy=short_expectancy,
        short_profit_factor=short_pf,
        stop_fraction=_exit_reason_fraction(native_rows, EXIT_REASON_STOP),
        target_fraction=_exit_reason_fraction(native_rows, EXIT_REASON_TARGET),
        time_exit_fraction=_exit_reason_fraction(native_rows, EXIT_REASON_TIME),
        mean_holding_seconds=mean_holding,
        median_holding_seconds=median_holding,
    )


# ---------------------------------------------------------------------------
# Parameter-neighborhood robustness (3-D: excursion_min x stop_buffer x
# target_mode; instrument identity never an adjacency dimension)
# ---------------------------------------------------------------------------


def _cell_for_config(config: B3F2Config) -> tuple[int, int, int]:
    return (
        EXCURSION_MIN_GRID.index(config.excursion_min),
        STOP_BUFFER_GRID.index(config.stop_buffer),
        TARGET_MODE_GRID.index(config.target_mode),
    )


def largest_connected_passing_region(rows: Sequence[B3F2ScorecardRow]) -> int:
    instrument_ids = {row.instrument_id for row in rows}
    if len(instrument_ids) > 1:
        raise B3F2ScreenError(
            "largest_connected_passing_region must be called with a single "
            "instrument's rows -- instrument identity is not an adjacency dimension"
        )
    passing: set[tuple[int, ...]] = {
        _cell_for_config(row.config) for row in rows if row.hard_gates_passed
    }
    if not passing:
        return 0
    components = _connected_components(passing, _axis_adjacent_neighbors)
    return max(len(component) for component in components)


def sleeve_passes_robustness(rows: Sequence[B3F2ScorecardRow]) -> bool:
    return largest_connected_passing_region(rows) >= int(MIN_CONNECTED_REGION_SIZE)


# ---------------------------------------------------------------------------
# Artifacts (write-once)
# ---------------------------------------------------------------------------

_SCORECARD_FIELDS = [
    field.name for field in B3F2ScorecardRow.__dataclass_fields__.values()
]


def _scorecard_row_to_dict(row: B3F2ScorecardRow) -> dict[str, Any]:
    payload = asdict(row)
    config = payload.pop("config")
    payload["excursion_min"] = str(config["excursion_min"])
    payload["stop_buffer"] = str(config["stop_buffer"])
    payload["target_mode"] = config["target_mode"]
    for key, value in list(payload.items()):
        if isinstance(value, Decimal):
            payload[key] = str(value)
    return payload


def write_b3f2_artifacts(
    *,
    scorecard: Sequence[B3F2ScorecardRow],
    pair_robustness: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
    output_dir: Path,
) -> None:
    if output_dir.exists():
        raise B3F2ScreenError(f"{output_dir} already exists; refusing to overwrite")
    output_dir.mkdir(parents=True)

    scorecard_path = output_dir / "scorecard.csv"
    with scorecard_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["excursion_min", "stop_buffer", "target_mode"] + [
            name for name in _SCORECARD_FIELDS if name != "config"
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in scorecard:
            writer.writerow(_scorecard_row_to_dict(row))

    robustness_path = output_dir / "pair_robustness.csv"
    with robustness_path.open("w", newline="", encoding="utf-8") as handle:
        if pair_robustness:
            writer = csv.DictWriter(handle, fieldnames=list(pair_robustness[0].keys()))
            writer.writeheader()
            writer.writerows(pair_robustness)
        else:
            handle.write("")

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
