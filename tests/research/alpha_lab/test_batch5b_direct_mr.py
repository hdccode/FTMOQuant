from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pandas as pd  # type: ignore[import-untyped]

from ftmoquant.research.alpha_lab.batch5_daily import CompletedFxDay, ny_fx_boundary
from ftmoquant.research.alpha_lab.batch5_execution import Batch5TradeResult
from ftmoquant.research.alpha_lab.batch5b_direct_mr_execution import (
    build_position_intents,
    execute_positions,
)
from ftmoquant.research.alpha_lab.batch5b_direct_mr_signals import (
    B5BSignal,
    generate_signals,
)


def day(local_date: date, close: str) -> CompletedFxDay:
    return CompletedFxDay(
        "AUD/CAD.OANDA",
        local_date,
        ny_fx_boundary(local_date - timedelta(days=1)),
        ny_fx_boundary(local_date),
        Decimal("1"),
        Decimal(close),
    )


def test_dst_safe_fx_day_can_be_23_hours() -> None:
    row = day(date(2021, 3, 14), "1")
    assert int((row.end_utc - row.start_utc).total_seconds()) == 23 * 3600


def test_inclusive_20_mean_starts_on_observation_20_and_maps_sign() -> None:
    start = date(2021, 1, 1)
    history = [day(start + timedelta(days=index), "1") for index in range(19)]
    assert generate_signals(history) == ()
    buy = generate_signals([*history, day(start + timedelta(days=19), "0.9")])[-1]
    sell = generate_signals([*history, day(start + timedelta(days=19), "1.1")])[-1]
    flat = generate_signals([*history, day(start + timedelta(days=19), "1")])[-1]
    assert buy.trailing_mean_20 == Decimal("0.995") and buy.direction == "BUY"
    assert sell.trailing_mean_20 == Decimal("1.005") and sell.direction == "SELL"
    assert flat.direction == "FLAT"


def test_inclusive_window_is_exactly_20_with_current_once_and_no_future() -> None:
    start = date(2021, 1, 1)
    closes = [Decimal(index) for index in range(1, 22)]
    rows = [
        day(start + timedelta(days=index), str(close))
        for index, close in enumerate(closes)
    ]
    signals = generate_signals(rows)
    assert len(signals) == 2
    assert signals[0].close_mid == Decimal(20)
    assert signals[0].trailing_mean_20 == Decimal("10.5")
    assert signals[1].close_mid == Decimal(21)
    assert signals[1].trailing_mean_20 == Decimal("11.5")


def test_reversal_exits_once_and_same_sign_never_pyramids() -> None:
    start = date(2021, 1, 1)
    rows = [day(start + timedelta(days=index), "1") for index in range(19)]
    rows.extend(
        [
            day(start + timedelta(days=19), "0.9"),
            day(start + timedelta(days=20), "0.8"),
            day(start + timedelta(days=21), "1.2"),
            day(start + timedelta(days=22), "0.7"),
        ]
    )
    intents = build_position_intents(generate_signals(rows))
    assert len(intents) == 2
    assert intents[0].direction == "BUY"
    assert intents[1].direction == "SELL"


def test_native_audcad_execution_converts_cad_through_usdcad() -> None:
    start = datetime(2021, 1, 1, tzinfo=UTC)
    signals = (
        B5BSignal(
            "B5B_direct_audcad_mean_reversion",
            "B5B_FROZEN_DIRECT_AUDCAD_MR",
            "B5B_AUDCAD",
            "AUD/CAD.OANDA",
            start,
            "BUY",
            Decimal("0.9"),
            Decimal(1),
            Decimal("-0.1"),
        ),
        B5BSignal(
            "B5B_direct_audcad_mean_reversion",
            "B5B_FROZEN_DIRECT_AUDCAD_MR",
            "B5B_AUDCAD",
            "AUD/CAD.OANDA",
            start + timedelta(minutes=2),
            "SELL",
            Decimal("1.1"),
            Decimal(1),
            Decimal("0.1"),
        ),
    )
    index = pd.DatetimeIndex([start + timedelta(minutes=value) for value in range(4)])

    def frame(value: float) -> pd.DataFrame:
        return pd.DataFrame(
            {name: value for name in ("open", "high", "low", "close")},
            index=index,
        )

    results = execute_positions(
        signals,
        bid_m1=frame(0.95),
        ask_m1=frame(0.96),
        usdcad_bid_m1=frame(1.25),
        usdcad_ask_m1=frame(1.26),
    )
    assert len(results) == 1 and isinstance(results[0], Batch5TradeResult)
    assert results[0].instrument_id == "AUD/CAD.OANDA"
    assert results[0].quantity == (
        Decimal("100000") * Decimal("1.25") / Decimal("0.96")
    )
