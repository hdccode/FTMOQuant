from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pandas as pd
import pytest

from ftmoquant.data.instruments import EURUSD_OANDA_SPEC, USDJPY_OANDA_SPEC
from ftmoquant.research.alpha_lab.b3f3_session_open_mr_execution import (
    B3F3ExecutionError,
    simulate_b3f3_intents,
)
from ftmoquant.research.alpha_lab.b3f3_session_open_mr_signals import (
    EXIT_REASON_STOP,
    EXIT_REASON_TARGET,
    EXIT_REASON_TIME,
    B3F3Config,
    B3F3TradeIntent,
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
    signal_ts: str = "2024-01-01T08:05:00Z",
    stop_price: float,
    target_price: float,
    local_date: dt.date = dt.date(2024, 1, 1),
) -> B3F3TradeIntent:
    return B3F3TradeIntent(
        instrument_id="EUR/USD.OANDA",
        config=B3F3Config(Decimal("2.0"), Decimal("0.75"), "OPEN_ANCHOR"),
        local_date=local_date,
        signal_ts=pd.Timestamp(signal_ts),
        direction=direction,
        entry_reference_price=1.1035,
        displacement_at_entry=3.5,
        stop_price=stop_price,
        target_price=target_price,
        time_exit_boundary_utc=time_exit_boundary_utc(local_date),
    )


def test_short_entry_sells_bid_exit_buys_ask() -> None:
    n = 30
    bid = _m1([1.1000] * n, start="2024-01-01T08:06:00Z")
    ask = _m1([1.1002] * n, start="2024-01-01T08:06:00Z")
    # stop == the flat ask price -> touches immediately on the next bar.
    intent = _intent(direction=-1, stop_price=1.1002, target_price=0.0)
    trades, skips = simulate_b3f3_intents(
        [intent], instrument_spec=EURUSD_OANDA_SPEC, bid_m1=bid, ask_m1=ask
    )
    assert skips == ()
    assert trades[0].entry_price == Decimal("1.1")  # SHORT entry sells BID


def test_long_entry_buys_ask() -> None:
    n = 30
    bid = _m1([1.1000] * n, start="2024-01-01T08:06:00Z")
    ask = _m1([1.1002] * n, start="2024-01-01T08:06:00Z")
    intent = _intent(direction=1, stop_price=0.0, target_price=1.1000)
    trades, _ = simulate_b3f3_intents(
        [intent], instrument_spec=EURUSD_OANDA_SPEC, bid_m1=bid, ask_m1=ask
    )
    assert trades[0].entry_price == Decimal("1.1002")  # LONG entry buys ASK


def test_stop_exit_liquidation_side_and_price() -> None:
    n = 30
    bid_prices = [1.1000] * 5 + [1.1055] * (n - 5)
    ask_prices = [p + 0.0002 for p in bid_prices]
    bid = _m1(bid_prices, start="2024-01-01T08:06:00Z")
    ask = _m1(ask_prices, start="2024-01-01T08:06:00Z")
    intent = _intent(direction=-1, stop_price=1.1050, target_price=1.0900)
    trades, _ = simulate_b3f3_intents(
        [intent], instrument_spec=EURUSD_OANDA_SPEC, bid_m1=bid, ask_m1=ask
    )
    assert trades[0].exit_reason == EXIT_REASON_STOP
    assert trades[0].exit_price == Decimal("1.1050")


def test_target_exit() -> None:
    n = 30
    bid_prices = [1.1000] * 5 + [1.0940] * (n - 5)
    ask_prices = [p + 0.0002 for p in bid_prices]
    bid = _m1(bid_prices, start="2024-01-01T08:06:00Z")
    ask = _m1(ask_prices, start="2024-01-01T08:06:00Z")
    intent = _intent(direction=-1, stop_price=1.1050, target_price=1.0950)
    trades, _ = simulate_b3f3_intents(
        [intent], instrument_spec=EURUSD_OANDA_SPEC, bid_m1=bid, ask_m1=ask
    )
    assert trades[0].exit_reason == EXIT_REASON_TARGET
    assert trades[0].exit_price == Decimal("1.0950")


def test_stop_wins_same_bar_collision() -> None:
    n = 30
    bid_prices = [1.1000] * 5 + [1.0900] * (n - 5)
    ask_prices = [p + 0.0200 for p in bid_prices]  # ask spikes through both levels
    bid = _m1(bid_prices, start="2024-01-01T08:06:00Z")
    ask = _m1(ask_prices, start="2024-01-01T08:06:00Z")
    intent = _intent(direction=-1, stop_price=1.1050, target_price=1.0950)
    trades, _ = simulate_b3f3_intents(
        [intent], instrument_spec=EURUSD_OANDA_SPEC, bid_m1=bid, ask_m1=ask
    )
    assert trades[0].exit_reason == EXIT_REASON_STOP


def test_time_exit_at_11_00_london_uses_first_valid_quote_at_or_after() -> None:
    start = pd.Timestamp("2024-01-01T08:06:00Z")
    end = pd.Timestamp("2024-01-01T11:05:00Z")
    n = int((end - start).total_seconds() // 60) + 1
    bid = _m1([1.1000] * n, start=str(start))
    ask = _m1([1.1002] * n, start=str(start))
    intent = _intent(direction=-1, stop_price=1.2000, target_price=1.0000)
    trades, _ = simulate_b3f3_intents(
        [intent], instrument_spec=EURUSD_OANDA_SPEC, bid_m1=bid, ask_m1=ask
    )
    assert trades[0].exit_reason == EXIT_REASON_TIME
    exit_ts = pd.Timestamp(trades[0].exit_ns, tz="UTC")
    assert exit_ts >= time_exit_boundary_utc(dt.date(2024, 1, 1))
    assert exit_ts.date() == dt.date(2024, 1, 1)


def test_no_overnight_carry_time_exit_bar_is_same_day() -> None:
    start = pd.Timestamp("2024-01-01T08:06:00Z")
    n = int((pd.Timestamp("2024-01-02T00:00:00Z") - start).total_seconds() // 60)
    bid = _m1([1.1000] * n, start=str(start))
    ask = _m1([1.1002] * n, start=str(start))
    intent = _intent(direction=1, stop_price=1.0000, target_price=1.2000)
    trades, _ = simulate_b3f3_intents(
        [intent], instrument_spec=EURUSD_OANDA_SPEC, bid_m1=bid, ask_m1=ask
    )
    exit_ts = pd.Timestamp(trades[0].exit_ns, tz="UTC")
    assert exit_ts.date() == dt.date(2024, 1, 1)


def test_strictly_later_than_decision_time() -> None:
    bid = _m1([1.1000] * 10, start="2024-01-01T08:06:00Z")
    ask = _m1([1.1002] * 10, start="2024-01-01T08:06:00Z")
    intent = _intent(
        direction=1,
        signal_ts="2024-01-01T08:05:30Z",
        stop_price=0.0,
        target_price=1.1000,
    )
    trades, _ = simulate_b3f3_intents(
        [intent], instrument_spec=EURUSD_OANDA_SPEC, bid_m1=bid, ask_m1=ask
    )
    assert trades[0].entry_ns > pd.Timestamp("2024-01-01T08:05:30Z").value


def test_no_pyramiding_second_intent_while_open_is_skipped() -> None:
    n = 60
    bid_prices = [1.1000] * 20 + [1.1050] * (n - 20)
    ask_prices = [p + 0.0002 for p in bid_prices]
    bid = _m1(bid_prices, start="2024-01-01T08:06:00Z")
    ask = _m1(ask_prices, start="2024-01-01T08:06:00Z")
    first = _intent(
        direction=1,
        signal_ts="2024-01-01T08:05:00Z",
        stop_price=0.0,
        target_price=1.1050,
    )
    second = _intent(
        direction=-1, signal_ts="2024-01-01T08:10:00Z", stop_price=1.2, target_price=1.0
    )
    trades, skips = simulate_b3f3_intents(
        [first, second], instrument_spec=EURUSD_OANDA_SPEC, bid_m1=bid, ask_m1=ask
    )
    assert len(trades) == 1
    assert len(skips) == 1
    assert skips[0].reason == "signal_during_open_trade"


def test_usd_account_currency_correctness_for_usd_jpy() -> None:
    n = 30
    bid_prices = [150.00] * 5 + [149.00] * (n - 5)
    ask_prices = [p + 0.02 for p in bid_prices]
    bid = _m1(bid_prices, start="2024-01-01T08:06:00Z")
    ask = _m1(ask_prices, start="2024-01-01T08:06:00Z")
    intent = _intent(direction=-1, stop_price=200.0, target_price=149.50)
    trades, _ = simulate_b3f3_intents(
        [intent], instrument_spec=USDJPY_OANDA_SPEC, bid_m1=bid, ask_m1=ask
    )
    trade = trades[0]
    # USD/JPY: base=USD -> quantity == gross notional (not divided by price).
    assert trade.quantity == Decimal("100000")
    expected_pnl_quote = (
        trade.direction * trade.quantity * (trade.exit_price - trade.entry_price)
    )
    expected_pnl_usd = expected_pnl_quote / trade.entry_price
    assert trade.realized_pnl_usd == expected_pnl_usd


def test_cost_stress_widens_execution_not_signal() -> None:
    n = 30
    bid = _m1([1.1000] * n, start="2024-01-01T08:06:00Z")
    ask = _m1([1.1002] * n, start="2024-01-01T08:06:00Z")
    intent = _intent(direction=-1, stop_price=1.1002, target_price=0.0)
    native, _ = simulate_b3f3_intents(
        [intent],
        instrument_spec=EURUSD_OANDA_SPEC,
        bid_m1=bid,
        ask_m1=ask,
        cost_stress_multiplier=Decimal("1"),
    )
    stressed, _ = simulate_b3f3_intents(
        [intent],
        instrument_spec=EURUSD_OANDA_SPEC,
        bid_m1=bid,
        ask_m1=ask,
        cost_stress_multiplier=Decimal("1.5"),
    )
    assert native[0].entry_price != stressed[0].entry_price


def test_displacement_at_entry_carried_through_to_trade() -> None:
    n = 30
    bid = _m1([1.1000] * n, start="2024-01-01T08:06:00Z")
    ask = _m1([1.1002] * n, start="2024-01-01T08:06:00Z")
    intent = _intent(direction=-1, stop_price=1.1002, target_price=0.0)
    trades, _ = simulate_b3f3_intents(
        [intent], instrument_spec=EURUSD_OANDA_SPEC, bid_m1=bid, ask_m1=ask
    )
    assert trades[0].displacement_at_entry == pytest.approx(3.5)


def test_mismatched_index_rejected() -> None:
    bid = _m1([1.1] * 5, start="2024-01-01T08:06:00Z")
    ask = _m1([1.1] * 5, start="2024-01-01T08:07:00Z")
    intent = _intent(direction=1, stop_price=1.0, target_price=1.2)
    with pytest.raises(B3F3ExecutionError):
        simulate_b3f3_intents(
            [intent], instrument_spec=EURUSD_OANDA_SPEC, bid_m1=bid, ask_m1=ask
        )


def test_no_m1_exit_before_data_end_is_skipped_not_fabricated() -> None:
    n = 5
    bid = _m1([1.1000] * n, start="2024-01-01T08:06:00Z")
    ask = _m1([1.1002] * n, start="2024-01-01T08:06:00Z")
    # neither stop nor target ever touched, and no data reaches time exit.
    intent = _intent(direction=-1, stop_price=1.2000, target_price=1.0000)
    trades, skips = simulate_b3f3_intents(
        [intent], instrument_spec=EURUSD_OANDA_SPEC, bid_m1=bid, ask_m1=ask
    )
    assert trades == ()
    assert skips[0].reason == "no_m1_exit_before_data_end"
