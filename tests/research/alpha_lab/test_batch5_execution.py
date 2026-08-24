from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pandas as pd  # type: ignore[import-untyped]
import pytest

from ftmoquant.research.alpha_lab.batch5_execution import (
    Batch5ExecutionError,
    Batch5TradeResult,
    TradeIntent,
    execute_intent,
    first_strictly_later_timestamp,
)


def frames(
    timestamps: list[datetime], bids: list[float], asks: list[float]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.DatetimeIndex(timestamps)

    def side(values: list[float]) -> pd.DataFrame:
        return pd.DataFrame(
            {name: values for name in ("open", "high", "low", "close")},
            index=index,
        )

    return side(bids), side(asks)


def intent(instrument: str, direction: str, start: datetime) -> TradeIntent:
    return TradeIntent(
        "family",
        "strategy",
        "sleeve",
        instrument,
        start,
        start,
        start + timedelta(minutes=2),
        direction,  # type: ignore[arg-type]
    )


def test_first_strictly_later_and_spread_crossing() -> None:
    start = datetime(2022, 1, 1, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=value) for value in (0, 1, 2, 3)]
    bid, ask = frames(timestamps, [1.0, 1.1, 1.2, 1.3], [1.01, 1.11, 1.21, 1.31])
    assert first_strictly_later_timestamp(bid, ask, start) == timestamps[1]
    result = execute_intent(
        intent("EUR/USD.OANDA", "BUY", start), bid_m1=bid, ask_m1=ask
    )
    assert isinstance(result, Batch5TradeResult)
    assert result.actual_entry_timestamp == timestamps[1]
    assert result.actual_exit_timestamp == timestamps[3]
    assert result.entry_price == Decimal("1.11")
    assert result.exit_price == Decimal("1.3")


def test_non_usd_cross_uses_causal_conversion_and_stress_widens_prices() -> None:
    start = datetime(2022, 1, 1, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=value) for value in (0, 1, 2, 3)]
    bid, ask = frames(timestamps, [0.90] * 4, [0.91] * 4)
    conversion_bid, conversion_ask = frames(timestamps, [1.25] * 4, [1.26] * 4)
    native = execute_intent(
        intent("AUD/CAD.OANDA", "BUY", start),
        bid_m1=bid,
        ask_m1=ask,
        conversion_bid_m1=conversion_bid,
        conversion_ask_m1=conversion_ask,
    )
    stressed = execute_intent(
        intent("AUD/CAD.OANDA", "BUY", start),
        bid_m1=bid,
        ask_m1=ask,
        conversion_bid_m1=conversion_bid,
        conversion_ask_m1=conversion_ask,
        cost_stress_multiplier=Decimal("2.0"),
    )
    assert isinstance(native, Batch5TradeResult)
    assert isinstance(stressed, Batch5TradeResult)
    assert native.quantity == Decimal("100000") * Decimal("1.25") / Decimal("0.91")
    assert stressed.entry_price > native.entry_price
    assert stressed.exit_price < native.exit_price
    assert stressed.pnl_usd < native.pnl_usd


def test_no_runtime_cost_or_notional_parameter_overrides() -> None:
    start = datetime(2022, 1, 1, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=value) for value in range(4)]
    bid, ask = frames(timestamps, [1.0] * 4, [1.01] * 4)
    with pytest.raises(Batch5ExecutionError, match="cost stress"):
        execute_intent(
            intent("EUR/USD.OANDA", "BUY", start),
            bid_m1=bid,
            ask_m1=ask,
            cost_stress_multiplier=Decimal("1.2"),
        )
    with pytest.raises(Batch5ExecutionError, match="reference notional"):
        execute_intent(
            intent("EUR/USD.OANDA", "BUY", start),
            bid_m1=bid,
            ask_m1=ask,
            reference_notional_usd=Decimal("99999"),
        )
