"""Pure scorecards, gates, breadth, diagnostics, and selection for Batch 4."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ftmoquant.research.alpha_lab.batch4_clock_scheduler import (
    EXPECTED_PREREGISTRATION_SEMANTIC_SHA256,
    FrozenClockSpec,
)
from ftmoquant.research.alpha_lab.batch4_execution import ScheduledTradeResult
from ftmoquant.research.alpha_lab.batch4_preregistration import (
    PREREGISTRATION_PATH,
    verify_preregistration,
)

FAMILY_LOCAL = "B4F1A_local_hours_flow_seasonality"
FAMILY_LONDON = "B4F1B_london_fix_flow_reversal"
FAMILY_TOKYO = "B4F1C_tokyo_fix_flow_reversal"
FAMILIES = (FAMILY_LOCAL, FAMILY_LONDON, FAMILY_TOKYO)
DURATION_AXIS = (15, 30, 60)
STRESS_MULTIPLIERS = (Decimal("1.0"), Decimal("1.5"), Decimal("2.0"))


class Batch4ScreenError(ValueError):
    """Raised on methodology drift or invalid screening inputs."""


@dataclass(frozen=True, slots=True)
class FrozenScreenPolicy:
    min_trade_count: int
    expectancy_gt: Decimal
    profit_factor_gt: Decimal
    fold_count: int
    min_positive_folds: int
    best_5pct_expectancy_gt: Decimal
    max_quarter_share: Decimal
    stress_multipliers: tuple[Decimal, Decimal]
    min_connected_region_size: int
    breadth_min_core_sleeves: int
    breadth_native_pf_gt: Decimal
    breadth_min_full_gate_sleeves: int
    representative_ranking: tuple[str, ...]


def load_frozen_screen_policy(path: Path = PREREGISTRATION_PATH) -> FrozenScreenPolicy:
    """Verify and parse the exact gates/breadth/ranking before price access."""

    document = verify_preregistration(path)
    if (
        document.get("preregistration_semantic_sha256")
        != EXPECTED_PREREGISTRATION_SEMANTIC_SHA256
    ):
        raise Batch4ScreenError("Batch 4 preregistration identity mismatch")
    gates = document["development_gates"]
    breadth = document["family_breadth_gate"]
    selection = document["development_to_validation"]
    policy = FrozenScreenPolicy(
        min_trade_count=int(gates["A_opportunity_density"]["completed_trades_gte"]),
        expectancy_gt=Decimal(
            str(gates["B_native_expectancy"]["expectancy_usd_per_trade_gt"])
        ),
        profit_factor_gt=Decimal(
            str(gates["C_native_profit_factor"]["profit_factor_gt"])
        ),
        fold_count=int(
            gates["D_temporal_stability"]["chronological_development_fold_count"]
        ),
        min_positive_folds=int(
            gates["D_temporal_stability"]["positive_net_return_folds_gte"]
        ),
        best_5pct_expectancy_gt=Decimal(
            str(
                gates["E_exceptional_winner_dependence"][
                    "remaining_expectancy_usd_per_trade_gt"
                ]
            )
        ),
        max_quarter_share=Decimal(
            str(gates["F_quarter_concentration"]["max_single_quarter_share_lte"])
        ),
        stress_multipliers=tuple(
            Decimal(str(value))
            for value in gates["G_cost_stress"]["required_multipliers"]
        ),  # type: ignore[arg-type]
        min_connected_region_size=int(
            gates["H_parameter_neighborhood"]["min_connected_passing_region"]
        ),
        breadth_min_core_sleeves=int(breadth["sleeves_meeting_breadth_metrics_gte"]),
        breadth_native_pf_gt=Decimal(
            str(breadth["breadth_metrics_all_required"]["native_profit_factor_gt"])
        ),
        breadth_min_full_gate_sleeves=int(
            breadth["sleeves_passing_full_hard_gate_set_gte"]
        ),
        representative_ranking=tuple(selection["representative_ranking"]),
    )
    expected_ranking = (
        "highest median sleeve expectancy",
        "highest median sleeve profit factor",
        "lowest median sleeve max-quarter positive-profit concentration",
        "highest aggregate completed trade count",
        "lexicographically smallest strategy_id",
    )
    if policy != FrozenScreenPolicy(
        250,
        Decimal(0),
        Decimal("1.1"),
        4,
        3,
        Decimal(0),
        Decimal("0.4"),
        (Decimal("1.5"), Decimal("2.0")),
        2,
        3,
        Decimal("1.0"),
        2,
        expected_ranking,
    ):
        raise Batch4ScreenError("frozen Batch 4 gate, breadth, or ranking drift")
    return policy


@dataclass(frozen=True, slots=True)
class _TradeRow:
    exit_utc: datetime
    local_date: date
    pnl: Decimal
    holding_seconds: int
    scheduled_holding_seconds: int
    utc_offset_seconds: int


def _trade_rows(
    trades: Sequence[ScheduledTradeResult], spec: FrozenClockSpec
) -> tuple[_TradeRow, ...]:
    zone = ZoneInfo(spec.timezone)
    rows = []
    for trade in trades:
        if trade.hypothesis_id != spec.hypothesis_id:
            raise Batch4ScreenError("trade belongs to a different hypothesis")
        local_entry = trade.scheduled_entry_utc.astimezone(zone)
        offset = local_entry.utcoffset()
        if offset is None:
            raise Batch4ScreenError("scheduled entry has no UTC offset")
        rows.append(
            _TradeRow(
                exit_utc=trade.actual_exit_utc,
                local_date=date.fromisoformat(trade.local_date),
                pnl=trade.pnl_account_currency,
                holding_seconds=trade.holding_seconds,
                scheduled_holding_seconds=int(
                    (
                        trade.scheduled_exit_utc - trade.scheduled_entry_utc
                    ).total_seconds()
                ),
                utc_offset_seconds=int(offset.total_seconds()),
            )
        )
    return tuple(sorted(rows, key=lambda row: row.exit_utc))


def expectancy_and_profit_factor(
    pnls: Sequence[Decimal],
) -> tuple[Decimal, Decimal]:
    if not pnls:
        return Decimal(0), Decimal(0)
    expectancy = sum(pnls, Decimal(0)) / len(pnls)
    gross_profit = sum((value for value in pnls if value > 0), Decimal(0))
    gross_loss = sum((-value for value in pnls if value < 0), Decimal(0))
    if gross_loss > 0:
        return expectancy, gross_profit / gross_loss
    return expectancy, Decimal("Infinity") if gross_profit > 0 else Decimal(0)


def best_5pct_removed_expectancy(rows: Sequence[_TradeRow]) -> Decimal:
    profitable = [row for row in rows if row.pnl > 0]
    remove_count = math.ceil(0.05 * len(profitable)) if profitable else 0
    ranked = sorted(profitable, key=lambda row: (-row.pnl, row.exit_utc))
    removed = {id(row) for row in ranked[:remove_count]}
    remaining = [row.pnl for row in rows if id(row) not in removed]
    return sum(remaining, Decimal(0)) / len(remaining) if remaining else Decimal(0)


def _positive_period_share(
    rows: Sequence[_TradeRow], *, quarter: bool
) -> Decimal | None:
    positive = [row for row in rows if row.pnl > 0]
    total = sum((row.pnl for row in positive), Decimal(0))
    if total <= 0:
        return None
    buckets: dict[tuple[int, int], Decimal] = defaultdict(Decimal)
    for row in positive:
        period = (row.exit_utc.month - 1) // 3 + 1 if quarter else row.exit_utc.month
        buckets[(row.exit_utc.year, period)] += row.pnl
    return max(buckets.values()) / total


def _rolling(
    rows: Sequence[_TradeRow], window: int
) -> tuple[Decimal | None, Decimal | None]:
    if len(rows) < window:
        return None, None
    pnls = [row.pnl for row in rows]
    values = [
        sum(pnls[index : index + window], Decimal(0)) / window
        for index in range(len(pnls) - window + 1)
    ]
    return (
        statistics.median(values),
        Decimal(sum(value > 0 for value in values)) / len(values),
    )


def _skew_kurtosis(pnls: Sequence[Decimal]) -> tuple[float | None, float | None]:
    if len(pnls) < 3:
        return None, None
    values = [float(value) for value in pnls]
    mean = statistics.mean(values)
    m2 = statistics.mean((value - mean) ** 2 for value in values)
    if m2 == 0:
        return None, None
    m3 = statistics.mean((value - mean) ** 3 for value in values)
    skew = m3 / (m2**1.5)
    if len(values) < 4:
        return skew, None
    m4 = statistics.mean((value - mean) ** 4 for value in values)
    return skew, m4 / (m2**2) - 3.0


def _daily_returns(
    rows: Sequence[_TradeRow], start: datetime, end_exclusive: datetime
) -> list[float]:
    by_day: dict[date, Decimal] = defaultdict(Decimal)
    for row in rows:
        by_day[row.exit_utc.date()] += row.pnl / Decimal("100000")
    days = (end_exclusive.date() - start.date()).days
    return [
        float(by_day.get(start.date() + timedelta(days=offset), Decimal(0)))
        for offset in range(days)
    ]


def _annualized_sharpe(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    variance = statistics.variance(values)
    if variance <= 0 or not math.isfinite(variance):
        return 0.0
    return math.sqrt(365.0) * statistics.mean(values) / math.sqrt(variance)


def _maximum_drawdown(rows: Sequence[_TradeRow]) -> Decimal:
    cumulative = Decimal(0)
    peak = Decimal(0)
    max_drawdown = Decimal(0)
    for row in rows:
        cumulative += row.pnl / Decimal("100000")
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)
    return max_drawdown


def _configuration(spec: FrozenClockSpec) -> tuple[str, str | None, int | None]:
    if spec.family == FAMILY_LOCAL:
        return "LOCAL", None, None
    parts = spec.hypothesis_id.split(":")
    if len(parts) != 3:
        raise Batch4ScreenError(f"invalid fix hypothesis id: {spec.hypothesis_id}")
    config = parts[1]
    phase, duration_text = config.split("_", maxsplit=1)
    duration = int(duration_text.removesuffix("m"))
    if phase not in {"PRE", "POST"} or duration not in DURATION_AXIS:
        raise Batch4ScreenError(f"invalid frozen fix configuration: {config}")
    return config, phase, duration


@dataclass(frozen=True, slots=True)
class Batch4ScorecardRow:
    hypothesis_id: str
    family: str
    instrument_id: str
    direction: str
    timezone: str
    local_start: str
    local_end: str
    configuration_id: str
    phase: str | None
    duration_minutes: int | None
    trade_count: int
    skip_count: int
    expectancy_usd: Decimal
    profit_factor: Decimal
    net_return: Decimal
    annualized_sharpe: float
    maximum_drawdown: Decimal
    win_rate: Decimal
    fold_1_net_return: Decimal
    fold_2_net_return: Decimal
    fold_3_net_return: Decimal
    fold_4_net_return: Decimal
    positive_fold_count: int
    best_5pct_removed_expectancy: Decimal
    quarter_max_share: Decimal | None
    stressed_1_5x_trade_count: int
    stressed_1_5x_expectancy: Decimal
    stressed_1_5x_profit_factor: Decimal
    stressed_2_0x_trade_count: int
    stressed_2_0x_expectancy: Decimal
    stressed_2_0x_profit_factor: Decimal
    base_hard_gates_passed: bool
    parameter_neighborhood_applicable: bool
    parameter_neighborhood_passed: bool
    hard_gates_passed: bool
    rolling_50_median_expectancy: Decimal | None
    rolling_50_fraction_positive: Decimal | None
    rolling_100_median_expectancy: Decimal | None
    rolling_100_fraction_positive: Decimal | None
    monthly_max_share: Decimal | None
    largest_trade_share: Decimal | None
    pnl_skewness: float | None
    pnl_kurtosis: float | None
    mean_holding_seconds: float | None
    median_holding_seconds: float | None
    trades_per_year: float
    spread_cost_share_of_gross_edge: Decimal | None


def evaluate_hypothesis(
    *,
    spec: FrozenClockSpec,
    native_trades: Sequence[ScheduledTradeResult],
    native_skip_count: int,
    stressed_1_5x_trades: Sequence[ScheduledTradeResult],
    stressed_2_0x_trades: Sequence[ScheduledTradeResult],
    fold_boundaries: Sequence[datetime],
    policy: FrozenScreenPolicy,
) -> Batch4ScorecardRow:
    """Compute base hard gates and report-only metrics for one hypothesis."""

    if len(fold_boundaries) != policy.fold_count + 1:
        raise Batch4ScreenError("fold boundary count does not match frozen policy")
    rows = _trade_rows(native_trades, spec)
    stress_1 = _trade_rows(stressed_1_5x_trades, spec)
    stress_2 = _trade_rows(stressed_2_0x_trades, spec)
    pnls = [row.pnl for row in rows]
    expectancy, pf = expectancy_and_profit_factor(pnls)
    stress_1_expectancy, stress_1_pf = expectancy_and_profit_factor(
        [row.pnl for row in stress_1]
    )
    stress_2_expectancy, stress_2_pf = expectancy_and_profit_factor(
        [row.pnl for row in stress_2]
    )
    fold_returns = tuple(
        sum(
            (
                row.pnl / Decimal("100000")
                for row in rows
                if start <= row.exit_utc < end
            ),
            Decimal(0),
        )
        for start, end in zip(fold_boundaries[:-1], fold_boundaries[1:], strict=True)
    )
    positive_folds = sum(value > 0 for value in fold_returns)
    best5 = best_5pct_removed_expectancy(rows)
    quarter = _positive_period_share(rows, quarter=True)
    base_passed = (
        len(rows) >= policy.min_trade_count
        and expectancy > policy.expectancy_gt
        and pf > policy.profit_factor_gt
        and positive_folds >= policy.min_positive_folds
        and best5 > policy.best_5pct_expectancy_gt
        and quarter is not None
        and quarter <= policy.max_quarter_share
        and bool(stress_1)
        and stress_1_expectancy > policy.expectancy_gt
        and bool(stress_2)
        and stress_2_expectancy > policy.expectancy_gt
    )
    rolling_50 = _rolling(rows, 50)
    rolling_100 = _rolling(rows, 100)
    skew, kurtosis = _skew_kurtosis(pnls)
    holdings = [row.holding_seconds for row in rows]
    positive = [value for value in pnls if value > 0]
    total_positive = sum(positive, Decimal(0))
    largest_share = max(positive) / total_positive if total_positive > 0 else None
    native_total = sum(pnls, Decimal(0))
    stress_2_total = sum((row.pnl for row in stress_2), Decimal(0))
    native_spread_cost = native_total - stress_2_total
    gross_edge = native_total + native_spread_cost
    spread_share = native_spread_cost / gross_edge if gross_edge > 0 else None
    configuration, phase, duration = _configuration(spec)
    elapsed_years = (fold_boundaries[-1] - fold_boundaries[0]).total_seconds() / (
        365.2425 * 86_400
    )
    daily = _daily_returns(rows, fold_boundaries[0], fold_boundaries[-1])
    return Batch4ScorecardRow(
        hypothesis_id=spec.hypothesis_id,
        family=spec.family,
        instrument_id=spec.instrument_id,
        direction=spec.direction,
        timezone=spec.timezone,
        local_start=spec.local_start_time.isoformat(timespec="minutes"),
        local_end=spec.local_end_time.isoformat(timespec="minutes"),
        configuration_id=configuration,
        phase=phase,
        duration_minutes=duration,
        trade_count=len(rows),
        skip_count=native_skip_count,
        expectancy_usd=expectancy,
        profit_factor=pf,
        net_return=sum(pnls, Decimal(0)) / Decimal("100000"),
        annualized_sharpe=_annualized_sharpe(daily),
        maximum_drawdown=_maximum_drawdown(rows),
        win_rate=(
            Decimal(sum(value > 0 for value in pnls)) / len(pnls)
            if pnls
            else Decimal(0)
        ),
        fold_1_net_return=fold_returns[0],
        fold_2_net_return=fold_returns[1],
        fold_3_net_return=fold_returns[2],
        fold_4_net_return=fold_returns[3],
        positive_fold_count=positive_folds,
        best_5pct_removed_expectancy=best5,
        quarter_max_share=quarter,
        stressed_1_5x_trade_count=len(stress_1),
        stressed_1_5x_expectancy=stress_1_expectancy,
        stressed_1_5x_profit_factor=stress_1_pf,
        stressed_2_0x_trade_count=len(stress_2),
        stressed_2_0x_expectancy=stress_2_expectancy,
        stressed_2_0x_profit_factor=stress_2_pf,
        base_hard_gates_passed=base_passed,
        parameter_neighborhood_applicable=spec.family != FAMILY_LOCAL,
        parameter_neighborhood_passed=spec.family == FAMILY_LOCAL,
        hard_gates_passed=base_passed if spec.family == FAMILY_LOCAL else False,
        rolling_50_median_expectancy=rolling_50[0],
        rolling_50_fraction_positive=rolling_50[1],
        rolling_100_median_expectancy=rolling_100[0],
        rolling_100_fraction_positive=rolling_100[1],
        monthly_max_share=_positive_period_share(rows, quarter=False),
        largest_trade_share=largest_share,
        pnl_skewness=skew,
        pnl_kurtosis=kurtosis,
        mean_holding_seconds=statistics.mean(holdings) if holdings else None,
        median_holding_seconds=statistics.median(holdings) if holdings else None,
        trades_per_year=len(rows) / elapsed_years,
        spread_cost_share_of_gross_edge=spread_share,
    )


def apply_parameter_neighborhood(
    rows: Sequence[Batch4ScorecardRow], policy: FrozenScreenPolicy
) -> tuple[Batch4ScorecardRow, ...]:
    """Apply 15-30-60 adjacency within instrument and PRE/POST only."""

    updated: dict[str, Batch4ScorecardRow] = {}
    groups: dict[tuple[str, str, str], list[Batch4ScorecardRow]] = defaultdict(list)
    for row in rows:
        if row.family == FAMILY_LOCAL:
            updated[row.hypothesis_id] = replace(
                row,
                parameter_neighborhood_applicable=False,
                parameter_neighborhood_passed=True,
                hard_gates_passed=row.base_hard_gates_passed,
            )
        else:
            if row.phase is None:
                raise Batch4ScreenError("fix row has no PRE/POST phase")
            groups[(row.family, row.instrument_id, row.phase)].append(row)
    for group in groups.values():
        if {row.duration_minutes for row in group} != set(DURATION_AXIS):
            raise Batch4ScreenError("fix adjacency group must contain 15/30/60")
        passing = {row.duration_minutes for row in group if row.base_hard_gates_passed}
        members: set[int] = set()
        for left, right in zip(DURATION_AXIS[:-1], DURATION_AXIS[1:], strict=True):
            if left in passing and right in passing:
                members.update((left, right))
        for row in group:
            neighborhood = row.duration_minutes in members
            updated[row.hypothesis_id] = replace(
                row,
                parameter_neighborhood_applicable=True,
                parameter_neighborhood_passed=neighborhood,
                hard_gates_passed=row.base_hard_gates_passed and neighborhood,
            )
    result = tuple(updated[row.hypothesis_id] for row in rows)
    if len(result) != 91:
        raise Batch4ScreenError(f"scorecard must contain 91 rows, got {len(result)}")
    return result


def _breadth_unit_id(row: Batch4ScorecardRow) -> str:
    return "LOCAL" if row.family == FAMILY_LOCAL else row.configuration_id


def compute_family_robustness(
    rows: Sequence[Batch4ScorecardRow], policy: FrozenScreenPolicy
) -> tuple[dict[str, Any], ...]:
    groups: dict[tuple[str, str], list[Batch4ScorecardRow]] = defaultdict(list)
    for row in rows:
        groups[(row.family, _breadth_unit_id(row))].append(row)
    expected_units = {(FAMILY_LOCAL, "LOCAL")}
    expected_units.update(
        (family, f"{phase}_{duration}m")
        for family in (FAMILY_LONDON, FAMILY_TOKYO)
        for phase in ("PRE", "POST")
        for duration in DURATION_AXIS
    )
    if set(groups) != expected_units:
        raise Batch4ScreenError("breadth units drifted from frozen 13-unit structure")
    results = []
    for family, unit in sorted(groups):
        unit_rows = groups[(family, unit)]
        if len(unit_rows) != 7:
            raise Batch4ScreenError("every breadth unit must contain all seven sleeves")
        core = [
            row
            for row in unit_rows
            if row.expectancy_usd > 0
            and row.profit_factor > policy.breadth_native_pf_gt
            and row.stressed_1_5x_expectancy > 0
        ]
        full = [row for row in unit_rows if row.hard_gates_passed]
        results.append(
            {
                "family": family,
                "configuration_id": unit,
                "eligible_sleeve_count": 7,
                "breadth_core_passing_sleeve_count": len(core),
                "full_gate_passing_sleeve_count": len(full),
                "family_breadth_passed": (
                    len(core) >= policy.breadth_min_core_sleeves
                    and len(full) >= policy.breadth_min_full_gate_sleeves
                ),
                "median_sleeve_expectancy": statistics.median(
                    row.expectancy_usd for row in unit_rows
                ),
                "median_sleeve_profit_factor": statistics.median(
                    row.profit_factor for row in unit_rows
                ),
                "median_sleeve_quarter_concentration": statistics.median(
                    row.quarter_max_share
                    if row.quarter_max_share is not None
                    else Decimal("Infinity")
                    for row in unit_rows
                ),
                "aggregate_trade_count": sum(row.trade_count for row in unit_rows),
                "strategy_id": f"{family}:{unit}",
            }
        )
    return tuple(results)


def build_family_summary(
    rows: Sequence[Batch4ScorecardRow], robustness: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], ...]:
    summary = []
    for family in FAMILIES:
        family_rows = [row for row in rows if row.family == family]
        units = [unit for unit in robustness if unit["family"] == family]
        passing_units = [unit for unit in units if unit["family_breadth_passed"]]
        strongest = _rank_units(passing_units)[0] if passing_units else None
        summary.append(
            {
                "family": family,
                "tested_hypothesis_count": len(family_rows),
                "positive_native_expectancy_count": sum(
                    row.expectancy_usd > 0 for row in family_rows
                ),
                "profit_factor_gt_1_count": sum(
                    row.profit_factor > 1 for row in family_rows
                ),
                "stressed_1_5x_positive_count": sum(
                    row.stressed_1_5x_expectancy > 0 for row in family_rows
                ),
                "full_hard_gate_passing_count": sum(
                    row.hard_gates_passed for row in family_rows
                ),
                "breadth_core_passing_sleeve_count": (
                    max(
                        int(unit["breadth_core_passing_sleeve_count"]) for unit in units
                    )
                    if units
                    else 0
                ),
                "full_gate_passing_sleeve_count": (
                    max(int(unit["full_gate_passing_sleeve_count"]) for unit in units)
                    if units
                    else 0
                ),
                "family_breadth_passed": bool(passing_units),
                "strongest_passing_region_or_configuration": (
                    strongest["configuration_id"] if strongest else None
                ),
            }
        )
    return tuple(summary)


def _finite_rank_decimal(value: object, *, high_is_good: bool) -> float:
    decimal = Decimal(str(value))
    if decimal.is_infinite():
        return -math.inf if high_is_good else math.inf
    number = float(decimal)
    return -number if high_is_good else number


def _rank_units(units: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        units,
        key=lambda unit: (
            _finite_rank_decimal(unit["median_sleeve_expectancy"], high_is_good=True),
            _finite_rank_decimal(
                unit["median_sleeve_profit_factor"], high_is_good=True
            ),
            _finite_rank_decimal(
                unit["median_sleeve_quarter_concentration"], high_is_good=False
            ),
            -int(unit["aggregate_trade_count"]),
            str(unit["strategy_id"]),
        ),
    )


def select_representative(
    robustness: Sequence[Mapping[str, Any]], policy: FrozenScreenPolicy
) -> dict[str, Any]:
    if policy.representative_ranking != (
        "highest median sleeve expectancy",
        "highest median sleeve profit factor",
        "lowest median sleeve max-quarter positive-profit concentration",
        "highest aggregate completed trade count",
        "lexicographically smallest strategy_id",
    ):
        raise Batch4ScreenError("representative ranking drift")
    family_representatives = []
    for family in FAMILIES:
        passing = [
            unit
            for unit in robustness
            if unit["family"] == family and unit["family_breadth_passed"]
        ]
        if passing:
            family_representatives.append(_rank_units(passing)[0])
    selected = (
        _rank_units(family_representatives)[0] if family_representatives else None
    )
    return {
        "selected_representative": (
            str(selected["strategy_id"]) if selected is not None else None
        ),
        "selected_family": str(selected["family"]) if selected is not None else None,
        "selected_configuration_id": (
            str(selected["configuration_id"]) if selected is not None else None
        ),
        "eligible_family_representative_count": len(family_representatives),
        "ranking_rule": list(policy.representative_ranking),
        "number_permitted_for_future_validation": 1,
        "validation_accessed": False,
        "rescue_permitted": False,
    }


def build_diagnostics_summary(
    specs: Sequence[FrozenClockSpec],
    native_trades_by_hypothesis: Mapping[str, Sequence[ScheduledTradeResult]],
    rows: Sequence[Batch4ScorecardRow],
) -> dict[str, Any]:
    score_by_id = {row.hypothesis_id: row for row in rows}
    per_hypothesis: dict[str, Any] = {}
    for spec in specs:
        trades = native_trades_by_hypothesis.get(spec.hypothesis_id, ())
        trade_rows = _trade_rows(trades, spec)
        weekday_rows: dict[str, list[_TradeRow]] = defaultdict(list)
        year_rows: dict[str, list[_TradeRow]] = defaultdict(list)
        offsets: dict[str, int] = defaultdict(int)
        durations: dict[str, int] = defaultdict(int)
        for row in trade_rows:
            weekday_rows[row.local_date.strftime("%A")].append(row)
            year_rows[str(row.local_date.year)].append(row)
            offsets[str(row.utc_offset_seconds)] += 1
            durations[str(row.scheduled_holding_seconds)] += 1
        score = score_by_id[spec.hypothesis_id]

        def breakdown(groups: Mapping[str, Sequence[_TradeRow]]) -> dict[str, Any]:
            result = {}
            for label, group in sorted(groups.items()):
                pnls = [row.pnl for row in group]
                expectancy, profit_factor = expectancy_and_profit_factor(pnls)
                result[label] = {
                    "trade_count": len(group),
                    "expectancy_usd": expectancy,
                    "profit_factor": profit_factor,
                    "net_return": sum(pnls, Decimal(0)) / Decimal("100000"),
                }
            return result

        per_hypothesis[spec.hypothesis_id] = {
            "status": "report_only_not_used_by_any_gate_or_ranking",
            "local_weekday_breakdown": breakdown(weekday_rows),
            "year_breakdown": breakdown(year_rows),
            "fold_net_returns": [
                score.fold_1_net_return,
                score.fold_2_net_return,
                score.fold_3_net_return,
                score.fold_4_net_return,
            ],
            "utc_offset_seconds_counts": dict(sorted(offsets.items())),
            "scheduled_utc_holding_seconds_counts": dict(sorted(durations.items())),
            "phase": score.phase,
            "duration_minutes": score.duration_minutes,
            "fix_date_count": len({row.local_date for row in trade_rows}),
            "native_to_1_5x_expectancy_degradation": (
                score.expectancy_usd - score.stressed_1_5x_expectancy
            ),
            "native_to_2_0x_expectancy_degradation": (
                score.expectancy_usd - score.stressed_2_0x_expectancy
            ),
        }
    return {
        "status": "report_only_never_gate_rank_filter_or_rescue",
        "per_hypothesis": per_hypothesis,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(_jsonable(row) for row in rows)


def write_batch4_artifacts(
    *,
    scorecard: Sequence[Batch4ScorecardRow],
    family_summary: Sequence[Mapping[str, Any]],
    family_robustness: Sequence[Mapping[str, Any]],
    selection_summary: Mapping[str, Any],
    diagnostics_summary: Mapping[str, Any],
    metadata: Mapping[str, Any],
    output_dir: Path,
) -> None:
    """Write all seven deterministic artifacts, refusing any overwrite."""

    if output_dir.exists():
        raise Batch4ScreenError(f"{output_dir} already exists; refusing to overwrite")
    if len(scorecard) != 91:
        raise Batch4ScreenError(f"scorecard must contain 91 rows, got {len(scorecard)}")
    output_dir.mkdir(parents=True)
    ordered_scorecard = sorted(scorecard, key=lambda row: row.hypothesis_id)
    _write_csv(output_dir / "scorecard.csv", [asdict(row) for row in ordered_scorecard])
    _write_csv(output_dir / "family_summary.csv", list(family_summary))
    _write_csv(output_dir / "family_robustness.csv", list(family_robustness))
    json_payloads = {
        "selection_summary.json": selection_summary,
        "diagnostics_summary.json": diagnostics_summary,
        "metadata.json": metadata,
    }
    for filename, payload in json_payloads.items():
        (output_dir / filename).write_text(
            json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output_dir.iterdir())
        if path.name != "artifact_hashes.json"
    }
    (output_dir / "artifact_hashes.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
