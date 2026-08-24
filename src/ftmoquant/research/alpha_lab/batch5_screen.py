"""Pure frozen Batch-5 scorecards, gates, aggregate drawdown, and breadth."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

from ftmoquant.research.alpha_lab.batch4_screen import (
    _annualized_sharpe,
    expectancy_and_profit_factor,
)
from ftmoquant.research.alpha_lab.batch5_execution import Batch5TradeResult
from ftmoquant.research.alpha_lab.batch5_preregistration import (
    FAMILY_B5A,
    FAMILY_B5B,
    FAMILY_B5C,
    verify_preregistration,
)

REFERENCE_NOTIONAL = Decimal("100000")


class Batch5ScreenError(ValueError):
    """Raised on frozen policy drift or invalid pure screening inputs."""


@dataclass(frozen=True, slots=True)
class FrozenBatch5ScreenPolicy:
    profit_factor_gt: Decimal
    fold_count: int
    positive_folds_gte: int
    best_5pct_expectancy_gt: Decimal
    max_year_share: Decimal
    profitable_year_fraction_gte: Decimal
    stress_multipliers: tuple[Decimal, Decimal]
    aggregate_max_drawdown_lte: Decimal


def load_frozen_policy() -> FrozenBatch5ScreenPolicy:
    document = verify_preregistration()
    gates = document["common_development_gates"]
    policy = FrozenBatch5ScreenPolicy(
        Decimal(str(gates["native_profit_factor"]["profit_factor_gt"])),
        int(gates["chronological_stability"]["fold_count"]),
        int(gates["chronological_stability"]["positive_net_return_folds_gte"]),
        Decimal(str(gates["largest_winners"]["remaining_net_expectancy_gt"])),
        Decimal(
            str(
                gates["period_concentration"][
                    "max_single_year_share_of_strictly_positive_profit_lte"
                ]
            )
        ),
        Decimal(
            str(gates["period_concentration"]["profitable_calendar_year_fraction_gte"])
        ),
        tuple(Decimal(str(value)) for value in gates["costs"]["stress_multipliers"]),  # type: ignore[arg-type]
        Decimal(
            str(gates["drawdown"]["equal_weight_family_aggregate_maximum_drawdown_lte"])
        ),
    )
    expected = FrozenBatch5ScreenPolicy(
        Decimal("1.1"),
        4,
        3,
        Decimal(0),
        Decimal("0.4"),
        Decimal("0.6"),
        (Decimal("1.5"), Decimal("2.0")),
        Decimal("0.15"),
    )
    if policy != expected:
        raise Batch5ScreenError("frozen Batch 5 common gate drift")
    return policy


@dataclass(frozen=True, slots=True)
class FrequencyStats:
    monthly_formation_count: int = 0
    nonoverlapping_three_month_units: int = 0
    daily_holding_observation_count: int = 0
    position_sign_change_count: int = 0
    event_count: int = 0
    rollover_supported: bool = False
    active_year_count: int = 0


@dataclass(frozen=True, slots=True)
class SleeveScreenInput:
    family: str
    sleeve_id: str
    native_trades: Sequence[Batch5TradeResult]
    stressed_1_5x_trades: Sequence[Batch5TradeResult]
    stressed_2_0x_trades: Sequence[Batch5TradeResult]
    fold_boundaries: Sequence[datetime]
    frequency: FrequencyStats


@dataclass(frozen=True, slots=True)
class SleeveScorecard:
    family: str
    sleeve_id: str
    trade_count: int
    expectancy: Decimal
    net_return: Decimal
    profit_factor: Decimal
    positive_fold_count: int
    fold_net_returns: tuple[Decimal, ...]
    best_5pct_removed_expectancy: Decimal
    max_year_positive_profit_share: Decimal | None
    profitable_year_fraction: Decimal
    stressed_1_5x_expectancy: Decimal
    stressed_2_0x_expectancy: Decimal
    annualized_sharpe: float
    maximum_drawdown: Decimal
    frequency_floor_passed: bool
    all_sleeve_gates_passed: bool


def _validate_trades(
    rows: Sequence[Batch5TradeResult], family: str, sleeve_id: str
) -> tuple[Batch5TradeResult, ...]:
    if any(row.family != family or row.sleeve_id != sleeve_id for row in rows):
        raise Batch5ScreenError("trade belongs to another family or sleeve")
    return tuple(sorted(rows, key=lambda row: row.actual_exit_timestamp))


def _best_5pct(rows: Sequence[Batch5TradeResult]) -> Decimal:
    profitable = [row for row in rows if row.pnl_usd > 0]
    count = math.ceil(Decimal("0.05") * len(profitable)) if profitable else 0
    removed = {
        id(row)
        for row in sorted(
            profitable,
            key=lambda row: (-row.pnl_usd, row.actual_exit_timestamp),
        )[:count]
    }
    remaining = [row.pnl_usd for row in rows if id(row) not in removed]
    return sum(remaining, Decimal(0)) / len(remaining) if remaining else Decimal(0)


def _year_metrics(
    rows: Sequence[Batch5TradeResult],
) -> tuple[Decimal | None, Decimal, int]:
    positive_by_year: dict[int, Decimal] = defaultdict(Decimal)
    net_by_year: dict[int, Decimal] = defaultdict(Decimal)
    for row in rows:
        year = row.actual_exit_timestamp.year
        net_by_year[year] += row.pnl_usd
        if row.pnl_usd > 0:
            positive_by_year[year] += row.pnl_usd
    total_positive = sum(positive_by_year.values(), Decimal(0))
    share = (
        max(positive_by_year.values()) / total_positive
        if total_positive > 0 and positive_by_year
        else None
    )
    fraction = (
        Decimal(sum(value > 0 for value in net_by_year.values())) / len(net_by_year)
        if net_by_year
        else Decimal(0)
    )
    return share, fraction, len(net_by_year)


def _drawdown(rows: Sequence[Batch5TradeResult]) -> Decimal:
    equity = Decimal(0)
    peak = Decimal(0)
    drawdown = Decimal(0)
    for row in rows:
        equity += row.return_on_reference_notional
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return drawdown


def _sharpe(rows: Sequence[Batch5TradeResult]) -> float:
    by_day: dict[date, Decimal] = defaultdict(Decimal)
    for row in rows:
        by_day[row.actual_exit_timestamp.date()] += row.return_on_reference_notional
    values = [
        float(value) for _, value in sorted(by_day.items(), key=lambda item: item[0])
    ]
    return _annualized_sharpe(values)


def _frequency_passed(
    family: str, frequency: FrequencyStats, distinct_years: int
) -> bool:
    active_years = frequency.active_year_count or distinct_years
    if family == FAMILY_B5A:
        return (
            frequency.monthly_formation_count >= 36
            and frequency.nonoverlapping_three_month_units >= 12
            and active_years >= 3
        )
    if family == FAMILY_B5B:
        return (
            frequency.daily_holding_observation_count >= 500
            and frequency.position_sign_change_count >= 20
            and active_years >= 3
            and frequency.rollover_supported
        )
    if family == FAMILY_B5C:
        return frequency.event_count >= 15 and active_years >= 3
    raise Batch5ScreenError("unknown Batch 5 family")


def evaluate_sleeve(inputs: SleeveScreenInput) -> SleeveScorecard:
    policy = load_frozen_policy()
    if len(inputs.fold_boundaries) != policy.fold_count + 1:
        raise Batch5ScreenError("exactly four frozen chronological folds required")
    if any(
        left.tzinfo is None or right <= left
        for left, right in zip(
            inputs.fold_boundaries[:-1], inputs.fold_boundaries[1:], strict=True
        )
    ):
        raise Batch5ScreenError("fold boundaries must be aware and increasing")
    native = _validate_trades(inputs.native_trades, inputs.family, inputs.sleeve_id)
    stress_1 = _validate_trades(
        inputs.stressed_1_5x_trades, inputs.family, inputs.sleeve_id
    )
    stress_2 = _validate_trades(
        inputs.stressed_2_0x_trades, inputs.family, inputs.sleeve_id
    )
    expectancy, profit_factor = expectancy_and_profit_factor(
        [row.pnl_usd for row in native]
    )
    stress_1_expectancy, _ = expectancy_and_profit_factor(
        [row.pnl_usd for row in stress_1]
    )
    stress_2_expectancy, _ = expectancy_and_profit_factor(
        [row.pnl_usd for row in stress_2]
    )
    fold_returns = tuple(
        sum(
            (
                row.return_on_reference_notional
                for row in native
                if start <= row.actual_exit_timestamp < end
            ),
            Decimal(0),
        )
        for start, end in zip(
            inputs.fold_boundaries[:-1], inputs.fold_boundaries[1:], strict=True
        )
    )
    positive_folds = sum(value > 0 for value in fold_returns)
    best5 = _best_5pct(native)
    year_share, profitable_fraction, distinct_years = _year_metrics(native)
    frequency_passed = _frequency_passed(
        inputs.family, inputs.frequency, distinct_years
    )
    net_return = sum((row.return_on_reference_notional for row in native), Decimal(0))
    passed = (
        bool(native)
        and expectancy > 0
        and net_return > 0
        and profit_factor > policy.profit_factor_gt
        and positive_folds >= policy.positive_folds_gte
        and best5 > policy.best_5pct_expectancy_gt
        and year_share is not None
        and year_share <= policy.max_year_share
        and profitable_fraction >= policy.profitable_year_fraction_gte
        and bool(stress_1)
        and stress_1_expectancy > 0
        and bool(stress_2)
        and stress_2_expectancy > 0
        and frequency_passed
    )
    return SleeveScorecard(
        inputs.family,
        inputs.sleeve_id,
        len(native),
        expectancy,
        net_return,
        profit_factor,
        positive_folds,
        fold_returns,
        best5,
        year_share,
        profitable_fraction,
        stress_1_expectancy,
        stress_2_expectancy,
        _sharpe(native),
        _drawdown(native),
        frequency_passed,
        passed,
    )


@dataclass(frozen=True, slots=True)
class FamilyScorecard:
    family: str
    sleeve_count: int
    positive_native_and_1_5x_count: int
    full_gate_sleeve_count: int
    family_event_count: int
    equal_weight_maximum_drawdown: Decimal
    aggregate_drawdown_passed: bool
    breadth_passed: bool
    validation_eligible: bool


def _expected_sleeves() -> dict[str, frozenset[str]]:
    document = verify_preregistration()
    return {
        FAMILY_B5A: frozenset(
            row["sleeve_id"] for row in document["families"][FAMILY_B5A]["sleeves"]
        ),
        FAMILY_B5B: frozenset({"B5B_AUDCAD"}),
        FAMILY_B5C: frozenset(
            f"B5C_{instrument.split('.')[0].replace('/', '')}"
            for instrument in document["families"][FAMILY_B5C][
                "literature_anchored_rule"
            ]["universe"]
        ),
    }


def _equal_weight_drawdown(
    inputs: Sequence[SleeveScreenInput], sleeve_count: int
) -> Decimal:
    by_exit: dict[datetime, Decimal] = defaultdict(Decimal)
    for item in inputs:
        for row in item.native_trades:
            by_exit[row.actual_exit_timestamp] += row.pnl_usd / (
                REFERENCE_NOTIONAL * sleeve_count
            )
    equity = Decimal(0)
    peak = Decimal(0)
    maximum = Decimal(0)
    for timestamp in sorted(by_exit):
        equity += by_exit[timestamp]
        peak = max(peak, equity)
        maximum = max(maximum, peak - equity)
    return maximum


def evaluate_family(inputs: Sequence[SleeveScreenInput]) -> FamilyScorecard:
    if not inputs:
        raise Batch5ScreenError("family evaluation requires sleeves")
    family = inputs[0].family
    if any(item.family != family for item in inputs):
        raise Batch5ScreenError("cannot mix families")
    expected = _expected_sleeves().get(family)
    if expected is None or {item.sleeve_id for item in inputs} != expected:
        raise Batch5ScreenError("all and only frozen family sleeves must be retained")
    scorecards = [evaluate_sleeve(item) for item in inputs]
    core = sum(
        row.expectancy > 0 and row.stressed_1_5x_expectancy > 0 for row in scorecards
    )
    full = sum(row.all_sleeve_gates_passed for row in scorecards)
    event_count = sum(item.frequency.event_count for item in inputs)
    drawdown = _equal_weight_drawdown(inputs, len(expected))
    drawdown_passed = drawdown <= load_frozen_policy().aggregate_max_drawdown_lte
    if family == FAMILY_B5A:
        breadth = core >= 5 and full >= 4
    elif family == FAMILY_B5B:
        breadth = full == 1
    else:
        breadth = core >= 3 and full >= 2 and event_count >= 60
    return FamilyScorecard(
        family,
        len(expected),
        core,
        full,
        event_count,
        drawdown,
        drawdown_passed,
        breadth,
        breadth and drawdown_passed,
    )
