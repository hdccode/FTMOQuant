from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pandas as pd
import pytest

from ftmoquant.data.instruments import EURUSD_OANDA_SPEC, USDJPY_OANDA_SPEC
from ftmoquant.research.alpha_lab.b3f2_asian_range_fade_execution import (
    B3F2ExecutionError,
    simulate_b3f2_intents,
)
from ftmoquant.research.alpha_lab.b3f2_asian_range_fade_signals import (
    EXIT_REASON_STOP,
    EXIT_REASON_TARGET,
    EXIT_REASON_TIME,
    B3F2Config,
    B3F2TradeIntent,
    time_exit_boundary_utc,
)


def _m1(prices: list[float], *, start: str) -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(prices), freq="min", tz="UTC")
    return pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices}, index=idx
    )


def _intent(
    *,
    direction: int,
    signal_ts: str = "2024-01-01T07:00:00Z",
    stop_price: float,
    target_price: float,
    local_date: dt.date = dt.date(2024, 1, 1),
) -> B3F2TradeIntent:
    return B3F2TradeIntent(
        instrument_id="EUR/USD.OANDA",
        config=B3F2Config(Decimal("0.05"), Decimal("0.05"), "MID"),
        local_date=local_date,
        signal_ts=pd.Timestamp(signal_ts),
        direction=direction,
        entry_reference_price=1.1005,
        stop_price=stop_price,
        target_price=target_price,
        time_exit_boundary_utc=time_exit_boundary_utc(local_date),
    )


def test_short_entry_sells_bid_exit_buys_ask() -> None:
    n = 30
    bid = _m1([1.1000] * n, start="2024-01-01T07:01:00Z")
    ask = _m1([1.1002] * n, start="2024-01-01T07:01:00Z")
    # stop == the flat ask price -> touches immediately on the next bar.
    intent = _intent(direction=-1, stop_price=1.1002, target_price=1.0000)
    trades, skips = simulate_b3f2_intents(
        [intent], instrument_spec=EURUSD_OANDA_SPEC, bid_m1=bid, ask_m1=ask
    )
    assert skips == ()
    assert trades[0].entry_price == Decimal("1.1")  # SHORT entry sells BID


def test_long_entry_buys_ask() -> None:
    n = 30
    bid = _m1([1.1000] * n, start="2024-01-01T07:01:00Z")
    ask = _m1([1.1002] * n, start="2024-01-01T07:01:00Z")
    # LONG liquidation side is BID; target == the flat bid price -> touches
    # immediately on the next bar (entry itself still crosses ASK).
    intent = _intent(direction=1, stop_price=0.0000, target_price=1.1000)
    trades, _ = simulate_b3f2_intents(
        [intent], instrument_spec=EURUSD_OANDA_SPEC, bid_m1=bid, ask_m1=ask
    )
    assert trades[0].entry_price == Decimal("1.1002")  # LONG entry buys ASK


def test_stop_exit_liquidation_side_and_price() -> None:
    n = 30
    bid_prices = [1.1000] * 5 + [1.1055] * (n - 5)
    ask_prices = [p + 0.0002 for p in bid_prices]
    bid = _m1(bid_prices, start="2024-01-01T07:01:00Z")
    ask = _m1(ask_prices, start="2024-01-01T07:01:00Z")
    intent = _intent(direction=-1, stop_price=1.1050, target_price=1.0900)
    trades, _ = simulate_b3f2_intents(
        [intent], instrument_spec=EURUSD_OANDA_SPEC, bid_m1=bid, ask_m1=ask
    )
    assert trades[0].exit_reason == EXIT_REASON_STOP
    assert trades[0].exit_price == Decimal("1.1050")


def test_target_exit() -> None:
    n = 30
    bid_prices = [1.1000] * 5 + [1.0940] * (n - 5)
    ask_prices = [p + 0.0002 for p in bid_prices]
    bid = _m1(bid_prices, start="2024-01-01T07:01:00Z")
    ask = _m1(ask_prices, start="2024-01-01T07:01:00Z")
    intent = _intent(direction=-1, stop_price=1.1050, target_price=1.0950)
    trades, _ = simulate_b3f2_intents(
        [intent], instrument_spec=EURUSD_OANDA_SPEC, bid_m1=bid, ask_m1=ask
    )
    assert trades[0].exit_reason == EXIT_REASON_TARGET
    assert trades[0].exit_price == Decimal("1.0950")


def test_stop_wins_same_bar_collision() -> None:
    n = 30
    # Both stop (>=1.1050 ask) and target (<=1.0950 ask) touched on the SAME bar.
    bid_prices = [1.1000] * 5 + [1.0900] * (n - 5)
    ask_prices = [p + 0.0200 for p in bid_prices]  # ask spikes through both levels
    bid = _m1(bid_prices, start="2024-01-01T07:01:00Z")
    ask = _m1(ask_prices, start="2024-01-01T07:01:00Z")
    intent = _intent(direction=-1, stop_price=1.1050, target_price=1.0950)
    trades, _ = simulate_b3f2_intents(
        [intent], instrument_spec=EURUSD_OANDA_SPEC, bid_m1=bid, ask_m1=ask
    )
    assert trades[0].exit_reason == EXIT_REASON_STOP


def test_time_exit_at_16_00_london_uses_first_valid_quote_at_or_after() -> None:
    # Neither stop nor target ever touched; price stays flat until the
    # 16:00 London boundary.
    start = pd.Timestamp("2024-01-01T07:01:00Z")
    end = pd.Timestamp("2024-01-01T16:05:00Z")
    n = int((end - start).total_seconds() // 60) + 1
    bid = _m1([1.1000] * n, start=str(start))
    ask = _m1([1.1002] * n, start=str(start))
    intent = _intent(direction=-1, stop_price=1.2000, target_price=1.0000)
    trades, _ = simulate_b3f2_intents(
        [intent], instrument_spec=EURUSD_OANDA_SPEC, bid_m1=bid, ask_m1=ask
    )
    assert trades[0].exit_reason == EXIT_REASON_TIME
    exit_ts = pd.Timestamp(trades[0].exit_ns, tz="UTC")
    assert exit_ts >= time_exit_boundary_utc(dt.date(2024, 1, 1))
    # No data from the NEXT trading day was used: exit stays on Jan 1.
    assert exit_ts.date() == dt.date(2024, 1, 1)


def test_no_overnight_carry_time_exit_bar_is_same_day() -> None:
    start = pd.Timestamp("2024-01-01T07:01:00Z")
    n = int((pd.Timestamp("2024-01-02T00:00:00Z") - start).total_seconds() // 60)
    bid = _m1([1.1000] * n, start=str(start))
    ask = _m1([1.1002] * n, start=str(start))
    intent = _intent(direction=1, stop_price=1.0000, target_price=1.2000)
    trades, _ = simulate_b3f2_intents(
        [intent], instrument_spec=EURUSD_OANDA_SPEC, bid_m1=bid, ask_m1=ask
    )
    exit_ts = pd.Timestamp(trades[0].exit_ns, tz="UTC")
    assert exit_ts.date() == dt.date(2024, 1, 1)


def test_strictly_later_than_decision_time() -> None:
    bid = _m1([1.1000] * 10, start="2024-01-01T07:01:00Z")
    ask = _m1([1.1002] * 10, start="2024-01-01T07:01:00Z")
    # target == the flat bid price -> touches immediately.
    intent = _intent(
        direction=1,
        signal_ts="2024-01-01T07:00:30Z",
        stop_price=0.0,
        target_price=1.1000,
    )
    trades, _ = simulate_b3f2_intents(
        [intent], instrument_spec=EURUSD_OANDA_SPEC, bid_m1=bid, ask_m1=ask
    )
    assert trades[0].entry_ns > pd.Timestamp("2024-01-01T07:00:30Z").value


def test_no_pyramiding_second_intent_while_open_is_skipped() -> None:
    # Flat for the first 20 bars (so the first trade stays open past the
    # second intent's decision time), then BID jumps to the first trade's
    # LONG target so it eventually closes.
    n = 60
    bid_prices = [1.1000] * 20 + [1.1050] * (n - 20)
    ask_prices = [p + 0.0002 for p in bid_prices]
    bid = _m1(bid_prices, start="2024-01-01T07:01:00Z")
    ask = _m1(ask_prices, start="2024-01-01T07:01:00Z")
    first = _intent(
        direction=1,
        signal_ts="2024-01-01T07:00:00Z",
        stop_price=0.0,
        target_price=1.1050,
    )
    second = _intent(
        direction=-1, signal_ts="2024-01-01T07:05:00Z", stop_price=1.2, target_price=1.0
    )
    trades, skips = simulate_b3f2_intents(
        [first, second], instrument_spec=EURUSD_OANDA_SPEC, bid_m1=bid, ask_m1=ask
    )
    assert len(trades) == 1
    assert len(skips) == 1
    assert skips[0].reason == "signal_during_open_trade"


def test_usd_account_currency_correctness_for_usd_jpy() -> None:
    n = 30
    bid_prices = [150.00] * 5 + [149.00] * (n - 5)
    ask_prices = [p + 0.02 for p in bid_prices]
    bid = _m1(bid_prices, start="2024-01-01T07:01:00Z")
    ask = _m1(ask_prices, start="2024-01-01T07:01:00Z")
    intent = _intent(direction=1, stop_price=148.0, target_price=200.0)
    # target never reached; force a stop instead so exact P&L is checkable.
    intent = _intent(direction=-1, stop_price=200.0, target_price=149.50)
    trades, _ = simulate_b3f2_intents(
        [intent], instrument_spec=USDJPY_OANDA_SPEC, bid_m1=bid, ask_m1=ask
    )
    trade = trades[0]
    # USD/JPY: base=USD -> quantity == gross notional (not divided by price).
    assert trade.quantity == Decimal("100000")
    # pnl_quote (JPY) = direction * qty * (exit - entry); convert JPY->USD
    # by dividing by the entry price (JPY per USD).
    expected_pnl_quote = (
        trade.direction * trade.quantity * (trade.exit_price - trade.entry_price)
    )
    expected_pnl_usd = expected_pnl_quote / trade.entry_price
    assert trade.realized_pnl_usd == expected_pnl_usd


def test_cost_stress_widens_execution_not_signal() -> None:
    n = 30
    bid = _m1([1.1000] * n, start="2024-01-01T07:01:00Z")
    ask = _m1([1.1002] * n, start="2024-01-01T07:01:00Z")
    # stop == the flat ask price -> touches immediately (native and
    # stressed frames both still resolve to a real trade).
    intent = _intent(direction=-1, stop_price=1.1002, target_price=0.0)
    native, _ = simulate_b3f2_intents(
        [intent],
        instrument_spec=EURUSD_OANDA_SPEC,
        bid_m1=bid,
        ask_m1=ask,
        cost_stress_multiplier=Decimal("1"),
    )
    stressed, _ = simulate_b3f2_intents(
        [intent],
        instrument_spec=EURUSD_OANDA_SPEC,
        bid_m1=bid,
        ask_m1=ask,
        cost_stress_multiplier=Decimal("1.5"),
    )
    assert native[0].entry_price != stressed[0].entry_price


def test_mismatched_index_rejected() -> None:
    bid = _m1([1.1] * 5, start="2024-01-01T07:01:00Z")
    ask = _m1([1.1] * 5, start="2024-01-01T07:02:00Z")
    intent = _intent(direction=1, stop_price=1.0, target_price=1.2)
    with pytest.raises(B3F2ExecutionError):
        simulate_b3f2_intents(
            [intent], instrument_spec=EURUSD_OANDA_SPEC, bid_m1=bid, ask_m1=ask
        )
