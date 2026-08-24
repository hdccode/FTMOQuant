"""Pure DEVELOPMENT reporting and write-once artifacts for frozen Batch 5."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from ftmoquant.research.alpha_lab.batch4_screen import (
    _skew_kurtosis,
    expectancy_and_profit_factor,
)
from ftmoquant.research.alpha_lab.batch5_execution import Batch5TradeResult
from ftmoquant.research.alpha_lab.batch5_preregistration import (
    FAMILY_B5A,
    FAMILY_B5C,
    PRIMARY_FAMILIES,
)
from ftmoquant.research.alpha_lab.batch5_screen import (
    FrequencyStats,
    SleeveScreenInput,
    evaluate_family,
    evaluate_sleeve,
    load_frozen_policy,
)


class Batch5DevelopmentScorecardError(ValueError):
    """Raised when a DEVELOPMENT report would drift from the frozen screen."""


@dataclass(frozen=True, slots=True)
class DevelopmentSleeveInput:
    family: str
    strategy_id: str
    sleeve_id: str
    instrument_id: str
    native_trades: Sequence[Batch5TradeResult]
    stressed_1_5x_trades: Sequence[Batch5TradeResult]
    stressed_2_0x_trades: Sequence[Batch5TradeResult]
    fold_boundaries: Sequence[datetime]
    frequency: FrequencyStats
    independent_unit_count: int
    active_year_count: int

    def screen_input(self) -> SleeveScreenInput:
        return SleeveScreenInput(
            self.family,
            self.sleeve_id,
            self.native_trades,
            self.stressed_1_5x_trades,
            self.stressed_2_0x_trades,
            self.fold_boundaries,
            replace(self.frequency, active_year_count=self.active_year_count),
        )


@dataclass(frozen=True, slots=True)
class DevelopmentSleeveScorecard:
    family: str
    strategy_id: str
    sleeve_id: str
    instrument_id: str
    independent_unit_count: int
    trade_count: int
    expectancy_usd: Decimal
    net_return: Decimal
    profit_factor: Decimal
    annualized_sharpe: float
    maximum_drawdown: Decimal
    win_rate: Decimal
    fold_1_return: Decimal
    fold_2_return: Decimal
    fold_3_return: Decimal
    fold_4_return: Decimal
    positive_fold_count: int
    profitable_year_fraction: Decimal
    best_5pct_removed_expectancy: Decimal
    max_calendar_year_positive_profit_share: Decimal | None
    stressed_1_5x_trade_count: int
    stressed_1_5x_expectancy: Decimal
    stressed_1_5x_profit_factor: Decimal
    stressed_2_0x_trade_count: int
    stressed_2_0x_expectancy: Decimal
    stressed_2_0x_profit_factor: Decimal
    monthly_formation_count: int
    nonoverlapping_three_month_units: int
    daily_holding_observation_count: int
    position_sign_change_count: int
    event_count: int
    active_year_count: int
    rollover_supported: bool
    native_expectancy_gate: bool
    native_net_return_gate: bool
    native_profit_factor_gate: bool
    chronological_folds_gate: bool
    best_5pct_removed_gate: bool
    calendar_year_concentration_gate: bool
    profitable_year_fraction_gate: bool
    stressed_1_5x_expectancy_gate: bool
    stressed_2_0x_expectancy_gate: bool
    frequency_gate: bool
    sleeve_hard_gates_passed: bool
    rolling_window_units: int
    rolling_expectancy_median: Decimal | None
    rolling_expectancy_positive_fraction: Decimal | None
    largest_trade_or_unit_share: Decimal | None
    pnl_skewness: float | None
    pnl_kurtosis: float | None
    mean_holding_seconds: float | None
    median_holding_seconds: float | None
    native_to_1_5x_expectancy_degradation: Decimal
    native_to_2_0x_expectancy_degradation: Decimal


def _rolling_expectancy(
    rows: Sequence[Batch5TradeResult], window: int
) -> tuple[Decimal | None, Decimal | None]:
    if len(rows) < window:
        return None, None
    values = [row.pnl_usd for row in rows]
    rolling = [
        sum(values[index - window : index], Decimal(0)) / window
        for index in range(window, len(values) + 1)
    ]
    return (
        Decimal(str(statistics.median(rolling))),
        Decimal(sum(value > 0 for value in rolling)) / len(rolling),
    )


def _largest_positive_share(
    rows: Sequence[Batch5TradeResult],
) -> Decimal | None:
    positive = [row.pnl_usd for row in rows if row.pnl_usd > 0]
    total = sum(positive, Decimal(0))
    return max(positive) / total if total > 0 else None


def evaluate_development_sleeve(
    inputs: DevelopmentSleeveInput,
) -> DevelopmentSleeveScorecard:
    """Expand the frozen pure gate result into the sealed CSV schema."""

    if inputs.family not in PRIMARY_FAMILIES:
        raise Batch5DevelopmentScorecardError("unknown Batch 5 family")
    if inputs.independent_unit_count < 0 or inputs.active_year_count < 0:
        raise Batch5DevelopmentScorecardError("frequency counts cannot be negative")
    native = tuple(
        sorted(inputs.native_trades, key=lambda row: row.actual_exit_timestamp)
    )
    stress_1 = tuple(
        sorted(inputs.stressed_1_5x_trades, key=lambda row: row.actual_exit_timestamp)
    )
    stress_2 = tuple(
        sorted(inputs.stressed_2_0x_trades, key=lambda row: row.actual_exit_timestamp)
    )
    for rows in (native, stress_1, stress_2):
        if any(
            row.family != inputs.family
            or row.sleeve_id != inputs.sleeve_id
            or row.strategy_id != inputs.strategy_id
            or row.instrument_id != inputs.instrument_id
            for row in rows
        ):
            raise Batch5DevelopmentScorecardError("trade identity drift")
    base = evaluate_sleeve(inputs.screen_input())
    policy = load_frozen_policy()
    stress_1_expectancy, stress_1_pf = expectancy_and_profit_factor(
        [row.pnl_usd for row in stress_1]
    )
    stress_2_expectancy, stress_2_pf = expectancy_and_profit_factor(
        [row.pnl_usd for row in stress_2]
    )
    win_rate = (
        Decimal(sum(row.pnl_usd > 0 for row in native)) / len(native)
        if native
        else Decimal(0)
    )
    rolling_window = 12 if inputs.family == FAMILY_B5A else 50
    rolling_median, rolling_positive = _rolling_expectancy(native, rolling_window)
    skew, kurtosis = _skew_kurtosis([row.pnl_usd for row in native])
    holdings = [row.holding_seconds for row in native]
    year_share = base.max_year_positive_profit_share
    gates = (
        bool(native) and base.expectancy > 0,
        base.net_return > 0,
        base.profit_factor > policy.profit_factor_gt,
        base.positive_fold_count >= policy.positive_folds_gte,
        base.best_5pct_removed_expectancy > policy.best_5pct_expectancy_gt,
        year_share is not None and year_share <= policy.max_year_share,
        base.profitable_year_fraction >= policy.profitable_year_fraction_gte,
        bool(stress_1) and stress_1_expectancy > 0,
        bool(stress_2) and stress_2_expectancy > 0,
        base.frequency_floor_passed,
    )
    if all(gates) != base.all_sleeve_gates_passed:
        raise Batch5DevelopmentScorecardError("expanded gate booleans drifted")
    folds = base.fold_net_returns
    return DevelopmentSleeveScorecard(
        inputs.family,
        inputs.strategy_id,
        inputs.sleeve_id,
        inputs.instrument_id,
        inputs.independent_unit_count,
        len(native),
        base.expectancy,
        base.net_return,
        base.profit_factor,
        base.annualized_sharpe,
        base.maximum_drawdown,
        win_rate,
        folds[0],
        folds[1],
        folds[2],
        folds[3],
        base.positive_fold_count,
        base.profitable_year_fraction,
        base.best_5pct_removed_expectancy,
        year_share,
        len(stress_1),
        stress_1_expectancy,
        stress_1_pf,
        len(stress_2),
        stress_2_expectancy,
        stress_2_pf,
        inputs.frequency.monthly_formation_count,
        inputs.frequency.nonoverlapping_three_month_units,
        inputs.frequency.daily_holding_observation_count,
        inputs.frequency.position_sign_change_count,
        inputs.frequency.event_count,
        inputs.active_year_count,
        inputs.frequency.rollover_supported,
        *gates,
        base.all_sleeve_gates_passed,
        rolling_window,
        rolling_median,
        rolling_positive,
        _largest_positive_share(native),
        skew,
        kurtosis,
        float(statistics.mean(holdings)) if holdings else None,
        float(statistics.median(holdings)) if holdings else None,
        base.expectancy - stress_1_expectancy,
        base.expectancy - stress_2_expectancy,
    )


def build_family_summary(
    inputs: Sequence[DevelopmentSleeveInput],
) -> tuple[dict[str, Any], ...]:
    """Return one deterministic verdict per frozen family; never rank families."""

    result: list[dict[str, Any]] = []
    for family in PRIMARY_FAMILIES:
        family_inputs = tuple(item for item in inputs if item.family == family)
        family_result = evaluate_family(
            tuple(item.screen_input() for item in family_inputs)
        )
        scorecards = tuple(evaluate_development_sleeve(item) for item in family_inputs)
        family_frequency = all(row.frequency_gate for row in scorecards)
        if family == FAMILY_B5C:
            family_frequency = (
                family_frequency and family_result.family_event_count >= 60
            )
        eligible = (
            family_frequency
            and family_result.aggregate_drawdown_passed
            and family_result.breadth_passed
        )
        if eligible != family_result.validation_eligible:
            raise Batch5DevelopmentScorecardError("family eligibility drift")
        result.append(
            {
                "family": family,
                "strategy_id": scorecards[0].strategy_id,
                "tested_sleeve_count": family_result.sleeve_count,
                "core_positive_count": family_result.positive_native_and_1_5x_count,
                "full_gate_passing_count": family_result.full_gate_sleeve_count,
                "family_event_count": family_result.family_event_count,
                "equal_weight_maximum_drawdown": (
                    family_result.equal_weight_maximum_drawdown
                ),
                "family_frequency_passed": family_frequency,
                "family_drawdown_passed": family_result.aggregate_drawdown_passed,
                "family_breadth_passed": family_result.breadth_passed,
                "eligible_for_future_validation": eligible,
            }
        )
    return tuple(result)


def build_selection_summary(
    family_summary: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    eligible = [
        {"family": row["family"], "strategy_id": row["strategy_id"]}
        for row in family_summary
        if row["eligible_for_future_validation"] is True
    ]
    return {
        "selection_scope": "binary_per_family_no_global_ranking",
        "maximum_representatives_per_family": 1,
        "eligible_representatives": eligible,
        "eligible_family_count": len(eligible),
        "validation_accessed": False,
        "fallback_or_rescue_used": False,
    }


def build_diagnostics_summary(
    rows: Sequence[DevelopmentSleeveScorecard],
) -> dict[str, Any]:
    return {
        "status": "report_only_never_gate_rank_filter_or_rescue",
        "per_sleeve": {
            row.sleeve_id: {
                "rolling_window_units": row.rolling_window_units,
                "rolling_expectancy_median": row.rolling_expectancy_median,
                "rolling_expectancy_positive_fraction": (
                    row.rolling_expectancy_positive_fraction
                ),
                "largest_trade_or_unit_share": row.largest_trade_or_unit_share,
                "pnl_skewness": row.pnl_skewness,
                "pnl_kurtosis": row.pnl_kurtosis,
                "mean_holding_seconds": row.mean_holding_seconds,
                "median_holding_seconds": row.median_holding_seconds,
                "native_to_1_5x_expectancy_degradation": (
                    row.native_to_1_5x_expectancy_degradation
                ),
                "native_to_2_0x_expectancy_degradation": (
                    row.native_to_2_0x_expectancy_degradation
                ),
            }
            for row in sorted(rows, key=lambda item: item.sleeve_id)
        },
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        if value.is_infinite():
            return "Infinity" if value > 0 else "-Infinity"
        if value.is_nan():
            return "NaN"
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise Batch5DevelopmentScorecardError(f"refusing empty artifact: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(_jsonable(row) for row in rows)


def write_batch5_artifacts(
    *,
    sleeve_scorecard: Sequence[DevelopmentSleeveScorecard],
    family_summary: Sequence[Mapping[str, Any]],
    selection_summary: Mapping[str, Any],
    diagnostics_summary: Mapping[str, Any],
    metadata: Mapping[str, Any],
    output_dir: Path,
) -> None:
    """Write the exact six artifacts once and hash every primary artifact."""

    if output_dir.exists():
        raise Batch5DevelopmentScorecardError(
            f"{output_dir} already exists; refusing to overwrite"
        )
    if len(sleeve_scorecard) != 13:
        raise Batch5DevelopmentScorecardError(
            f"sleeve scorecard must contain 13 rows, got {len(sleeve_scorecard)}"
        )
    if len(family_summary) != 3:
        raise Batch5DevelopmentScorecardError("family summary must contain 3 rows")
    output_dir.mkdir(parents=True)
    ordered = sorted(sleeve_scorecard, key=lambda row: (row.family, row.sleeve_id))
    _write_csv(output_dir / "sleeve_scorecard.csv", [asdict(row) for row in ordered])
    _write_csv(output_dir / "family_summary.csv", list(family_summary))
    for filename, payload in (
        ("selection_summary.json", selection_summary),
        ("diagnostics_summary.json", diagnostics_summary),
        ("metadata.json", metadata),
    ):
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
