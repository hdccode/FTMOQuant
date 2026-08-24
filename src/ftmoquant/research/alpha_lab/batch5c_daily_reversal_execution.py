"""Native next-complete-FX-day execution for frozen Batch 5C."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal

import pandas as pd  # type: ignore[import-untyped]

from ftmoquant.research.alpha_lab.batch5_daily import CompletedFxDay
from ftmoquant.research.alpha_lab.batch5_execution import (
    Batch5SkipRecord,
    Batch5TradeResult,
    execute_intent,
)
from ftmoquant.research.alpha_lab.batch5c_daily_reversal_signals import (
    B5CEvent,
    build_next_day_intents,
)


def execute_events(
    events: Sequence[B5CEvent],
    *,
    days_by_instrument: Mapping[str, Sequence[CompletedFxDay]],
    native_frames: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]],
    usdjpy_conversion_frames: tuple[pd.DataFrame, pd.DataFrame],
    cost_stress_multiplier: Decimal = Decimal("1.0"),
) -> tuple[Batch5TradeResult | Batch5SkipRecord, ...]:
    results: list[Batch5TradeResult | Batch5SkipRecord] = []
    for intent in build_next_day_intents(events, days_by_instrument):
        bid, ask = native_frames[intent.instrument_id]
        conversion_bid: pd.DataFrame | None = None
        conversion_ask: pd.DataFrame | None = None
        if intent.instrument_id == "EUR/JPY.OANDA":
            conversion_bid, conversion_ask = usdjpy_conversion_frames
        results.append(
            execute_intent(
                intent,
                bid_m1=bid,
                ask_m1=ask,
                conversion_bid_m1=conversion_bid,
                conversion_ask_m1=conversion_ask,
                cost_stress_multiplier=cost_stress_multiplier,
            )
        )
    return tuple(results)
