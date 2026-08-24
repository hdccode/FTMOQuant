"""Native paired-M1 three-calendar-month cohort execution for Batch 5A."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal

import pandas as pd  # type: ignore[import-untyped]

from ftmoquant.research.alpha_lab.batch5_execution import (
    Batch5SkipRecord,
    Batch5TradeResult,
    TradeIntent,
    add_calendar_months,
    execute_intent,
    first_strictly_later_timestamp,
)
from ftmoquant.research.alpha_lab.batch5a_cftc_signals import B5ASignal


def execute_cohorts(
    signals: Sequence[B5ASignal],
    *,
    native_frames: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]],
    cost_stress_multiplier: Decimal = Decimal("1.0"),
) -> tuple[Batch5TradeResult | Batch5SkipRecord, ...]:
    """Execute independent monthly cohorts; overlapping cohorts are retained."""

    results: list[Batch5TradeResult | Batch5SkipRecord] = []
    for signal in sorted(
        signals, key=lambda row: (row.formation_timestamp, row.sleeve_id)
    ):
        bid, ask = native_frames[signal.instrument_id]
        entry = first_strictly_later_timestamp(bid, ask, signal.formation_timestamp)
        month_label = (
            f"{signal.formation_month.year:04d}-{signal.formation_month.month:02d}"
        )
        cohort_id = f"{signal.sleeve_id}:{month_label}"
        if entry is None:
            results.append(
                Batch5SkipRecord(
                    signal.family,
                    signal.strategy_id,
                    signal.sleeve_id,
                    signal.instrument_id,
                    signal.formation_timestamp,
                    cohort_id,
                    "no_strictly_later_entry",
                )
            )
            continue
        expiry = add_calendar_months(entry, 3)
        intent = TradeIntent(
            family=signal.family,
            strategy_id=signal.strategy_id,
            sleeve_id=signal.sleeve_id,
            instrument_id=signal.instrument_id,
            signal_timestamp=signal.formation_timestamp,
            entry_decision_timestamp=signal.formation_timestamp,
            exit_decision_timestamp=expiry,
            direction=signal.direction,
            cohort_id=cohort_id,
            metadata={
                "cftc_report_date": signal.cftc_report_date.isoformat(),
                "cftc_availability_timestamp": (
                    signal.cftc_availability_timestamp.isoformat()
                ),
                "formation_month": month_label,
                "scaled_dealer_net": str(signal.scaled_dealer_net),
                "dealer_position_change": str(signal.dealer_position_change),
                "cohort_expiry": expiry.isoformat(),
            },
        )
        results.append(
            execute_intent(
                intent,
                bid_m1=bid,
                ask_m1=ask,
                cost_stress_multiplier=cost_stress_multiplier,
            )
        )
    return tuple(results)
