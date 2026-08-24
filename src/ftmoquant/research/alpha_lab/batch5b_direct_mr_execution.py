"""One-position native AUD/CAD execution for frozen Batch 5B."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal

import pandas as pd  # type: ignore[import-untyped]

from ftmoquant.research.alpha_lab.batch5_execution import (
    Batch5SkipRecord,
    Batch5TradeResult,
    TradeIntent,
    execute_intent,
)
from ftmoquant.research.alpha_lab.batch5b_direct_mr_signals import B5BSignal


def build_position_intents(signals: Sequence[B5BSignal]) -> tuple[TradeIntent, ...]:
    """Open once, hold through same-sign signals, and exit/reverse on crossing."""

    active: B5BSignal | None = None
    intents: list[TradeIntent] = []
    for signal in sorted(signals, key=lambda row: row.signal_timestamp):
        timestamp = signal.signal_timestamp
        if not isinstance(timestamp, datetime):
            raise ValueError("B5B signal timestamp must be datetime")
        if active is None:
            if signal.direction != "FLAT":
                active = signal
            continue
        if signal.direction == active.direction:
            continue
        entry_timestamp = active.signal_timestamp
        assert isinstance(entry_timestamp, datetime)
        intents.append(
            TradeIntent(
                active.family,
                active.strategy_id,
                active.sleeve_id,
                active.instrument_id,
                entry_timestamp,
                entry_timestamp,
                timestamp,
                active.direction,  # type: ignore[arg-type]
                metadata={
                    "entry_deviation": str(active.deviation),
                    "exit_deviation": str(signal.deviation),
                },
            )
        )
        active = signal if signal.direction != "FLAT" else None
    return tuple(intents)


def execute_positions(
    signals: Sequence[B5BSignal],
    *,
    bid_m1: pd.DataFrame,
    ask_m1: pd.DataFrame,
    usdcad_bid_m1: pd.DataFrame,
    usdcad_ask_m1: pd.DataFrame,
    cost_stress_multiplier: Decimal = Decimal("1.0"),
) -> tuple[Batch5TradeResult | Batch5SkipRecord, ...]:
    return tuple(
        execute_intent(
            intent,
            bid_m1=bid_m1,
            ask_m1=ask_m1,
            conversion_bid_m1=usdcad_bid_m1,
            conversion_ask_m1=usdcad_ask_m1,
            cost_stress_multiplier=cost_stress_multiplier,
        )
        for intent in build_position_intents(signals)
    )
