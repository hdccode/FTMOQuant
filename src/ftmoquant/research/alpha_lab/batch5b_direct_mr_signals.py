"""Pure 20-prior-day direct AUD/CAD mean-reversion signals for Batch 5B."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal

from ftmoquant.research.alpha_lab.batch5_daily import CompletedFxDay
from ftmoquant.research.alpha_lab.batch5_preregistration import (
    FAMILY_B5B,
    verify_preregistration,
)

INSTRUMENT_ID = "AUD/CAD.OANDA"
LOOKBACK = 20
Direction = Literal["BUY", "SELL", "FLAT"]


class Batch5BSignalError(ValueError):
    """Raised when direct-cross signal inputs violate the frozen contract."""


@dataclass(frozen=True, slots=True)
class B5BSignal:
    family: str
    strategy_id: str
    sleeve_id: str
    instrument_id: str
    signal_timestamp: datetime
    direction: Direction
    close_mid: Decimal
    trailing_mean_20: Decimal
    deviation: Decimal


def generate_signals(days: Sequence[CompletedFxDay]) -> tuple[B5BSignal, ...]:
    """Use exactly 20 completed closes ending with the current close."""

    document = verify_preregistration()
    frozen = document["families"][FAMILY_B5B]["source_faithful_replication"]
    if (
        frozen["instrument"] != INSTRUMENT_ID
        or int(frozen["lookback_completed_fx_days"]) != LOOKBACK
    ):
        raise Batch5BSignalError("frozen B5B instrument/lookback drift")
    ordered = sorted(days, key=lambda row: row.end_utc)
    if len({row.end_utc for row in ordered}) != len(ordered):
        raise Batch5BSignalError("duplicate completed-day boundary")
    if any(row.instrument_id != INSTRUMENT_ID for row in ordered):
        raise Batch5BSignalError("B5B requires the native AUD/CAD instrument only")
    signals: list[B5BSignal] = []
    for index in range(LOOKBACK - 1, len(ordered)):
        current = ordered[index]
        window = ordered[index - LOOKBACK + 1 : index + 1]
        mean = sum((row.close_mid for row in window), Decimal(0)) / LOOKBACK
        deviation = current.close_mid - mean
        direction: Direction = (
            "BUY" if deviation < 0 else "SELL" if deviation > 0 else "FLAT"
        )
        signals.append(
            B5BSignal(
                FAMILY_B5B,
                "B5B_FROZEN_DIRECT_AUDCAD_MR",
                "B5B_AUDCAD",
                INSTRUMENT_ID,
                current.end_utc,
                direction,
                current.close_mid,
                mean,
                deviation,
            )
        )
    return tuple(signals)
