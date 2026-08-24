"""B3F1 DEVELOPMENT screen: wires the causal signal
(:mod:`ftmoquant.research.alpha_lab.b3f1_spread_signals`) and execution
(:mod:`ftmoquant.research.alpha_lab.b3f1_spread_execution`) layers into the
frozen v2 hard gates and diagnostics, per sleeve (instrument-pair) and per
statistical config.

Thresholds are read PROGRAMMATICALLY from the frozen v2 preregistration
(:data:`ftmoquant.research.alpha_lab.batch3_preregistration_v2.
DEVELOPMENT_GATES`) rather than re-hardcoded, so this module can never
silently drift from ``config/research/batch3_methodology_preregistration_v2.json``
(semantic hash
``e5cd74527004585cfd24bea55549d84e9cb66b05ffc6498360dac5007b651f7c``).

Reuses, rather than re-derives, the existing N-D axis-adjacency /
connected-component parameter-robustness check
(:func:`ftmoquant.research.alpha_lab.screening_stage2._connected_components`,
:func:`ftmoquant.research.alpha_lab.liquidity_structure_screen.
_axis_adjacent_neighbors`) -- B3F1's grid is 3-D (formation_window x
z_entry x z_stop), the same shape B2F1 already used this machinery for.

Per the B3.2 task brief: B3.4's FTMO Monte Carlo/pass_both selection is
NOT implemented here -- that is explicitly a later stage for alpha
survivors. Every diagnostic field this module DOES compute is read-only
and never gates B3.2/B3.3 advancement (``development_diagnostics`` in the
v2 preregistration).
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

from ftmoquant.research.alpha_lab.b3f1_spread_signals import B3F1Config
from ftmoquant.research.alpha_lab.batch3_preregistration_v2 import DEVELOPMENT_GATES
from ftmoquant.research.alpha_lab.liquidity_structure_screen import (
    _axis_adjacent_neighbors,
)
from ftmoquant.research.alpha_lab.relative_value_adapter import RelativeValueEpisode
from ftmoquant.research.alpha_lab.screening_stage2 import _connected_components

FAMILY_ID = "B3F1_spread_mean_reversion"
MIN_TRADE_COUNT = DEVELOPMENT_GATES["opportunity_density"]["family_overrides"][
    FAMILY_ID
]
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
FOLD_COUNT = DEVELOPMENT_GATES["temporal_stability"]["fold_count"]
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
assert STRESS_MULTIPLIERS == (Decimal("1.5"),)

ARTIFACT_ROOT = Path(".artifacts/alpha_lab/batch3_b3f1_spread_mr_v1")

_ROLLING_WINDOWS = (30, 50)


class B3F1ScreenError(ValueError):
    """Raised on any violation of the frozen B3F1 screen contract."""


@dataclass(frozen=True, slots=True)
class _TradeRow:
    exit_ts: datetime
    pnl: Decimal
    side: str  # "rich" or "cheap"


def _side_for_episode(episode: RelativeValueEpisode) -> str:
    """Derived purely mechanically from leg_a (Y)'s own direction -- never
    a second source of truth: short Y (-1) is RICH, long Y (+1) is CHEAP,
    matching ``b3f1_spread_signals._direction_for_side`` exactly."""

    return "rich" if episode.leg_a.direction == -1 else "cheap"


def _rows_from_episodes(
    episodes: Sequence[RelativeValueEpisode],
) -> tuple[_TradeRow, ...]:
    rows = []
    for episode in episodes:
        exit_ts = datetime.fromtimestamp(episode.exit_ns / 1_000_000_000, tz=UTC)
        rows.append(
            _TradeRow(
                exit_ts=exit_ts,
                pnl=episode.realized_pnl(),
                side=_side_for_episode(episode),
            )
        )
    return tuple(sorted(rows, key=lambda row: row.exit_ts))


def expectancy_and_profit_factor(pnls: Sequence[Decimal]) -> tuple[Decimal, Decimal]:
    """Mean P&L and gross-profit / gross-loss ratio. ``profit_factor`` is
    ``Decimal("Infinity")`` when there are no losing trades but at least
    one winner (unambiguously passes any finite threshold), and ``0`` when
    there is no profit at all."""

    if not pnls:
        raise B3F1ScreenError("expectancy/profit-factor requires at least one trade")
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
    """Remove the top ``ceil(5%)`` PROFITABLE trades by realized P&L (ties
    at the boundary broken by earlier exit timestamp removed first) and
    return the remaining trades' expectancy. Matches
    ``development_gates.exceptional_winner_dependency`` exactly."""

    if not rows:
        raise B3F1ScreenError("best-5%% removal requires at least one trade")
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
    """Max single-calendar-quarter share of total positive P&L. ``None``
    means the fail-closed case (total positive profit <= 0)."""

    return _period_share(rows, quarter=True)


def monthly_max_share(rows: Sequence[_TradeRow]) -> Decimal | None:
    """Diagnostic-only monthly counterpart of :func:`quarter_max_share`."""

    return _period_share(rows, quarter=False)


def fold_positive_count(
    rows: Sequence[_TradeRow], fold_boundaries: Sequence[datetime]
) -> int:
    """Count of the ``len(fold_boundaries) - 1`` chronological folds with a
    positive net P&L sum. ``fold_boundaries`` is caller-supplied so tests
    can use tiny synthetic windows; real runs pass the frozen
    ``DEVELOPMENT_FOLD_BOUNDARIES``."""

    if len(fold_boundaries) < 2:
        raise B3F1ScreenError("fold_boundaries must contain at least 2 edges")
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
    """Trailing rolling-window expectancy diagnostics. Windows requiring
    fewer than ``window`` completed trades do not exist and are never
    padded or zero-filled -- identical convention to
    ``ftmoquant.research.ftmo_pass_probability.alpha_diagnostic.
    _rolling_expectancy``."""

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
    rows: Sequence[_TradeRow], side: str
) -> tuple[int, Decimal | None, Decimal | None]:
    subset = [row.pnl for row in rows if row.side == side]
    if not subset:
        return 0, None, None
    expectancy, profit_factor = expectancy_and_profit_factor(subset)
    return len(subset), expectancy, profit_factor


@dataclass(frozen=True, slots=True)
class B3F1ScorecardRow:
    """One (sleeve, config) cell of the broad B3F1 screen. Fields above
    ``# --- diagnostics ---`` participate in ``hard_gates_passed``; fields
    below never do (development_diagnostics in the v2 preregistration)."""

    sleeve_id: str
    config: B3F1Config

    native_trade_count: int
    native_expectancy: Decimal
    native_profit_factor: Decimal
    fold_positive_count: int
    best_5pct_removed_expectancy: Decimal
    quarter_max_share: Decimal | None
    stressed_1_5x_trade_count: int
    stressed_1_5x_expectancy: Decimal
    stressed_1_5x_profit_factor: Decimal
    hard_gates_passed: bool

    # --- diagnostics (report-only; never gate B3.2/B3.3) ---
    rolling_30_median_expectancy: Decimal | None
    rolling_30_fraction_positive: Decimal | None
    rolling_50_median_expectancy: Decimal | None
    rolling_50_fraction_positive: Decimal | None
    monthly_max_share: Decimal | None
    largest_trade_share: Decimal | None
    pnl_skewness: float | None
    pnl_kurtosis: float | None
    rich_trade_count: int
    rich_expectancy: Decimal | None
    rich_profit_factor: Decimal | None
    cheap_trade_count: int
    cheap_expectancy: Decimal | None
    cheap_profit_factor: Decimal | None


def evaluate_b3f1_config(
    *,
    sleeve_id: str,
    config: B3F1Config,
    native_episodes: Sequence[RelativeValueEpisode],
    stressed_1_5x_episodes: Sequence[RelativeValueEpisode],
    fold_boundaries: Sequence[datetime],
) -> B3F1ScorecardRow:
    """Compute one full scorecard row -- every frozen hard gate plus every
    report-only diagnostic -- from already-executed native and 1.5x-
    stressed episode lists for one (sleeve, config) cell. Pure function:
    no data access, no randomness."""

    native_rows = _rows_from_episodes(native_episodes)
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

    stressed_rows = _rows_from_episodes(stressed_1_5x_episodes)
    if stressed_rows:
        stressed_expectancy, stressed_pf = expectancy_and_profit_factor(
            [row.pnl for row in stressed_rows]
        )
    else:
        stressed_expectancy, stressed_pf = Decimal(0), Decimal(0)

    hard_gates_passed = (
        trade_count >= MIN_TRADE_COUNT
        and native_expectancy > EXPECTANCY_GT
        and native_pf > PROFIT_FACTOR_GT
        and folds >= MIN_POSITIVE_FOLDS
        and best5 > BEST_5PCT_REMAINING_EXPECTANCY_GT
        and quarter_share is not None
        and quarter_share <= MAX_QUARTER_SHARE
        and len(stressed_rows) > 0
        and stressed_expectancy > EXPECTANCY_GT
    )

    rolling_30_median, rolling_30_frac = rolling_diagnostics(native_rows, 30)
    rolling_50_median, rolling_50_frac = rolling_diagnostics(native_rows, 50)
    skew, kurtosis = _skew_kurtosis([row.pnl for row in native_rows])
    rich_count, rich_expectancy, rich_pf = _direction_stats(native_rows, "rich")
    cheap_count, cheap_expectancy, cheap_pf = _direction_stats(native_rows, "cheap")

    return B3F1ScorecardRow(
        sleeve_id=sleeve_id,
        config=config,
        native_trade_count=trade_count,
        native_expectancy=native_expectancy,
        native_profit_factor=native_pf,
        fold_positive_count=folds,
        best_5pct_removed_expectancy=best5,
        quarter_max_share=quarter_share,
        stressed_1_5x_trade_count=len(stressed_rows),
        stressed_1_5x_expectancy=stressed_expectancy,
        stressed_1_5x_profit_factor=stressed_pf,
        hard_gates_passed=hard_gates_passed,
        rolling_30_median_expectancy=rolling_30_median,
        rolling_30_fraction_positive=rolling_30_frac,
        rolling_50_median_expectancy=rolling_50_median,
        rolling_50_fraction_positive=rolling_50_frac,
        monthly_max_share=monthly_max_share(native_rows),
        largest_trade_share=largest_trade_share(native_rows),
        pnl_skewness=skew,
        pnl_kurtosis=kurtosis,
        rich_trade_count=rich_count,
        rich_expectancy=rich_expectancy,
        rich_profit_factor=rich_pf,
        cheap_trade_count=cheap_count,
        cheap_expectancy=cheap_expectancy,
        cheap_profit_factor=cheap_pf,
    )


# ---------------------------------------------------------------------------
# Parameter-neighborhood robustness (reuses the existing N-D adjacency
# check, restricted to the SAME sleeve's 18 cells -- pair identity is never
# an adjacency dimension, per section 18).
# ---------------------------------------------------------------------------


def _cell_for_config(
    config: B3F1Config,
    *,
    formation_windows: Sequence[int],
    z_entries: Sequence[Decimal],
    z_stops: Sequence[Decimal],
) -> tuple[int, int, int]:
    return (
        formation_windows.index(config.formation_window),
        z_entries.index(config.z_entry),
        z_stops.index(config.z_stop),
    )


def largest_connected_passing_region(
    rows: Sequence[B3F1ScorecardRow],
    *,
    formation_windows: Sequence[int],
    z_entries: Sequence[Decimal],
    z_stops: Sequence[Decimal],
) -> int:
    """Size of the largest axis-adjacency-connected region of
    ``hard_gates_passed`` cells within ONE sleeve's 3-D grid, reusing
    :func:`~ftmoquant.research.alpha_lab.screening_stage2._connected_components`
    and
    :func:`~ftmoquant.research.alpha_lab.liquidity_structure_screen._axis_adjacent_neighbors`
    unchanged. Returns 0 if no cell passes."""

    sleeve_ids = {row.sleeve_id for row in rows}
    if len(sleeve_ids) > 1:
        raise B3F1ScreenError(
            "largest_connected_passing_region must be called with a single "
            "sleeve's rows -- pair identity is not an adjacency dimension"
        )
    passing: set[tuple[int, ...]] = {
        _cell_for_config(
            row.config,
            formation_windows=formation_windows,
            z_entries=z_entries,
            z_stops=z_stops,
        )
        for row in rows
        if row.hard_gates_passed
    }
    if not passing:
        return 0
    components = _connected_components(passing, _axis_adjacent_neighbors)
    return max(len(component) for component in components)


def sleeve_passes_robustness(
    rows: Sequence[B3F1ScorecardRow],
    *,
    formation_windows: Sequence[int],
    z_entries: Sequence[Decimal],
    z_stops: Sequence[Decimal],
) -> bool:
    return (
        largest_connected_passing_region(
            rows,
            formation_windows=formation_windows,
            z_entries=z_entries,
            z_stops=z_stops,
        )
        >= int(MIN_CONNECTED_REGION_SIZE)
    )


# ---------------------------------------------------------------------------
# Artifacts (write-once)
# ---------------------------------------------------------------------------

_SCORECARD_FIELDS = [
    field.name for field in B3F1ScorecardRow.__dataclass_fields__.values()
]


def _scorecard_row_to_dict(row: B3F1ScorecardRow) -> dict[str, Any]:
    payload = asdict(row)
    config = payload.pop("config")
    payload["formation_window"] = config["formation_window"]
    payload["z_entry"] = str(config["z_entry"])
    payload["z_stop"] = str(config["z_stop"])
    for key, value in list(payload.items()):
        if isinstance(value, Decimal):
            payload[key] = str(value)
    return payload


def write_b3f1_artifacts(
    *,
    scorecard: Sequence[B3F1ScorecardRow],
    pair_robustness: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
    output_dir: Path,
) -> None:
    """Write ``scorecard.csv``, ``pair_robustness.csv``, ``metadata.json``.
    Refuses to overwrite an existing output directory -- matching the
    existing repo-wide "write-once" convention for frozen screen
    artifacts."""

    if output_dir.exists():
        raise B3F1ScreenError(f"{output_dir} already exists; refusing to overwrite")
    output_dir.mkdir(parents=True)

    scorecard_path = output_dir / "scorecard.csv"
    with scorecard_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["formation_window", "z_entry", "z_stop"] + [
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
