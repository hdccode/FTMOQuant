from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from ftmoquant.data.instruments import (
    EURUSD_OANDA_SPEC,
    USDCAD_OANDA_SPEC,
)
from ftmoquant.research.alpha_lab.b3f1_spread_execution import (
    GROSS_NOTIONAL_USD,
    B3F1ExecutionError,
    compute_leg_weights,
    simulate_b3f1_intents,
    usd_gross_to_quantity,
)
from ftmoquant.research.alpha_lab.b3f1_spread_signals import (
    EXIT_REASON_Z_MEAN_REVERSION,
    B3F1TradeIntent,
    SpreadSide,
)


def _m1(prices: list[float], *, start: str = "2024-01-01T00:01:00Z") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(prices), freq="min", tz="UTC")
    return pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices}, index=idx
    )


def _intent(
    *,
    side: SpreadSide = SpreadSide.RICH,
    entry_ts: str = "2024-01-01T00:00:00Z",
    exit_ts: str = "2024-01-01T00:10:00Z",
    sleeve_id: str = "EUR/USD.OANDA__USD/CAD.OANDA",
    frozen_beta: float = 1.2,
) -> B3F1TradeIntent:
    return B3F1TradeIntent(
        sleeve_id=sleeve_id,
        side=side,
        entry_ts=pd.Timestamp(entry_ts),
        entry_z=2.0,
        frozen_alpha=0.1,
        frozen_beta=frozen_beta,
        frozen_spread_mean=0.0,
        frozen_spread_std=0.01,
        exit_ts=pd.Timestamp(exit_ts),
        exit_reason=EXIT_REASON_Z_MEAN_REVERSION,
    )


def _y_x_frames(n: int = 30):
    y_bid = _m1([1.0790] * n)
    y_ask = _m1([1.0792] * n)
    x_bid = _m1([1.3490] * n)
    x_ask = _m1([1.3492] * n)
    return y_bid, y_ask, x_bid, x_ask


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


def test_gross_notionals_sum_to_100k() -> None:
    weight_y, weight_x = compute_leg_weights(Decimal("1.2"))
    assert weight_y + weight_x == Decimal(1)
    gross_y = GROSS_NOTIONAL_USD * weight_y
    gross_x = GROSS_NOTIONAL_USD * weight_x
    assert gross_y + gross_x == GROSS_NOTIONAL_USD


@pytest.mark.parametrize("beta", [Decimal("0.5"), Decimal("1.0"), Decimal("2.5")])
def test_hedge_magnitude_respects_beta(beta: Decimal) -> None:
    weight_y, weight_x = compute_leg_weights(beta)
    assert float(weight_x / weight_y) == pytest.approx(float(beta), rel=1e-9)


def test_weights_reject_nonpositive_beta() -> None:
    with pytest.raises(B3F1ExecutionError):
        compute_leg_weights(Decimal("-1.0"))
    with pytest.raises(B3F1ExecutionError):
        compute_leg_weights(Decimal("0"))


def test_usd_gross_to_quantity_quote_usd_divides_by_price() -> None:
    # EUR/USD: base=EUR, quote=USD -> quantity = gross / price.
    qty = usd_gross_to_quantity(
        Decimal("45000"), Decimal("1.0800"), base_currency="EUR", quote_currency="USD"
    )
    assert qty == Decimal("45000") / Decimal("1.0800")


def test_usd_gross_to_quantity_base_usd_does_not_divide_by_price() -> None:
    # USD/CAD: base=USD -> quantity == gross (dividing by price would be
    # the dimensional bug the task brief explicitly asked to audit for).
    qty = usd_gross_to_quantity(
        Decimal("55000"), Decimal("1.35"), base_currency="USD", quote_currency="CAD"
    )
    assert qty == Decimal("55000")


def test_usd_gross_to_quantity_usd_jpy_correctness() -> None:
    qty = usd_gross_to_quantity(
        Decimal("60000"), Decimal("150.00"), base_currency="USD", quote_currency="JPY"
    )
    assert qty == Decimal("60000")  # NOT 60000/150


# ---------------------------------------------------------------------------
# Execution / atomicity
# ---------------------------------------------------------------------------


def test_atomic_two_leg_entry_executes_both_legs() -> None:
    y_bid, y_ask, x_bid, x_ask = _y_x_frames()
    episodes, skips = simulate_b3f1_intents(
        [_intent()],
        y_spec=EURUSD_OANDA_SPEC,
        x_spec=USDCAD_OANDA_SPEC,
        y_bid_m1=y_bid,
        y_ask_m1=y_ask,
        x_bid_m1=x_bid,
        x_ask_m1=x_ask,
    )
    assert len(episodes) == 1
    assert skips == ()


def test_rich_side_uses_correct_bid_ask_sides() -> None:
    y_bid, y_ask, x_bid, x_ask = _y_x_frames()
    episodes, _ = simulate_b3f1_intents(
        [_intent(side=SpreadSide.RICH)],
        y_spec=EURUSD_OANDA_SPEC,
        x_spec=USDCAD_OANDA_SPEC,
        y_bid_m1=y_bid,
        y_ask_m1=y_ask,
        x_bid_m1=x_bid,
        x_ask_m1=x_ask,
    )
    episode = episodes[0]
    # Y is short (RICH): entry sells BID, exit buys ASK.
    assert episode.leg_a.direction == -1
    assert episode.leg_a.entry_price == Decimal("1.079")
    assert episode.leg_a.exit_price == Decimal("1.0792")
    # X is long (RICH): entry buys ASK, exit sells BID.
    assert episode.leg_b.direction == 1
    assert episode.leg_b.entry_price == Decimal("1.3492")
    assert episode.leg_b.exit_price == Decimal("1.349")


def test_cheap_side_uses_correct_bid_ask_sides() -> None:
    y_bid, y_ask, x_bid, x_ask = _y_x_frames()
    episodes, _ = simulate_b3f1_intents(
        [_intent(side=SpreadSide.CHEAP)],
        y_spec=EURUSD_OANDA_SPEC,
        x_spec=USDCAD_OANDA_SPEC,
        y_bid_m1=y_bid,
        y_ask_m1=y_ask,
        x_bid_m1=x_bid,
        x_ask_m1=x_ask,
    )
    episode = episodes[0]
    # Y is long (CHEAP): entry buys ASK, exit sells BID.
    assert episode.leg_a.direction == 1
    assert episode.leg_a.entry_price == Decimal("1.0792")
    assert episode.leg_a.exit_price == Decimal("1.079")
    # X is short (CHEAP): entry sells BID, exit buys ASK.
    assert episode.leg_b.direction == -1
    assert episode.leg_b.entry_price == Decimal("1.349")
    assert episode.leg_b.exit_price == Decimal("1.3492")


def test_fill_is_strictly_later_than_decision_time() -> None:
    y_bid, y_ask, x_bid, x_ask = _y_x_frames()
    episodes, _ = simulate_b3f1_intents(
        [_intent(entry_ts="2024-01-01T00:00:30Z")],  # between bar 00:01 and next
        y_spec=EURUSD_OANDA_SPEC,
        x_spec=USDCAD_OANDA_SPEC,
        y_bid_m1=y_bid,
        y_ask_m1=y_ask,
        x_bid_m1=x_bid,
        x_ask_m1=x_ask,
    )
    episode = episodes[0]
    assert episode.leg_a.entry_ns > pd.Timestamp("2024-01-01T00:00:30Z").value
    assert episode.leg_b.entry_ns > pd.Timestamp("2024-01-01T00:00:30Z").value


def test_incomplete_leg_fill_at_entry_fails_closed() -> None:
    y_bid, y_ask, x_bid, x_ask = _y_x_frames(n=30)
    # x data ends well before the intent's entry decision time -> no later
    # observation for the x leg at all.
    x_bid_short = x_bid.iloc[:1]
    x_ask_short = x_ask.iloc[:1]
    episodes, skips = simulate_b3f1_intents(
        [_intent(entry_ts="2024-01-01T00:29:00Z", exit_ts="2024-01-01T00:35:00Z")],
        y_spec=EURUSD_OANDA_SPEC,
        x_spec=USDCAD_OANDA_SPEC,
        y_bid_m1=y_bid,
        y_ask_m1=y_ask,
        x_bid_m1=x_bid_short,
        x_ask_m1=x_ask_short,
    )
    assert episodes == ()
    assert len(skips) == 1
    assert skips[0].reason == "no_later_m1_observation"


def test_logical_exit_requires_both_legs_to_close() -> None:
    y_bid, y_ask, x_bid, x_ask = _y_x_frames(n=30)
    # x data ends between entry and exit decision times -> entry succeeds
    # for both legs but the x leg has no fill for the EXIT.
    cutoff = pd.Timestamp("2024-01-01T00:05:00Z")
    x_bid_short = x_bid.loc[x_bid.index <= cutoff]
    x_ask_short = x_ask.loc[x_ask.index <= cutoff]
    episodes, skips = simulate_b3f1_intents(
        [_intent(entry_ts="2024-01-01T00:00:00Z", exit_ts="2024-01-01T00:10:00Z")],
        y_spec=EURUSD_OANDA_SPEC,
        x_spec=USDCAD_OANDA_SPEC,
        y_bid_m1=y_bid,
        y_ask_m1=y_ask,
        x_bid_m1=x_bid_short,
        x_ask_m1=x_ask_short,
    )
    assert episodes == ()
    assert len(skips) == 1
    assert skips[0].reason == "no_later_m1_observation"


def test_no_pyramiding_second_intent_while_open_is_skipped() -> None:
    y_bid, y_ask, x_bid, x_ask = _y_x_frames(n=30)
    first = _intent(entry_ts="2024-01-01T00:00:00Z", exit_ts="2024-01-01T00:10:00Z")
    second = _intent(entry_ts="2024-01-01T00:05:00Z", exit_ts="2024-01-01T00:20:00Z")
    episodes, skips = simulate_b3f1_intents(
        [first, second],
        y_spec=EURUSD_OANDA_SPEC,
        x_spec=USDCAD_OANDA_SPEC,
        y_bid_m1=y_bid,
        y_ask_m1=y_ask,
        x_bid_m1=x_bid,
        x_ask_m1=x_ask,
    )
    assert len(episodes) == 1
    assert len(skips) == 1
    assert skips[0].reason == "signal_during_open_trade"


def test_cost_stress_applied_only_to_execution_not_a_pnl_haircut() -> None:
    y_bid, y_ask, x_bid, x_ask = _y_x_frames()
    native_episodes, _ = simulate_b3f1_intents(
        [_intent()],
        y_spec=EURUSD_OANDA_SPEC,
        x_spec=USDCAD_OANDA_SPEC,
        y_bid_m1=y_bid,
        y_ask_m1=y_ask,
        x_bid_m1=x_bid,
        x_ask_m1=x_ask,
        cost_stress_multiplier=Decimal("1"),
    )
    stressed_episodes, _ = simulate_b3f1_intents(
        [_intent()],
        y_spec=EURUSD_OANDA_SPEC,
        x_spec=USDCAD_OANDA_SPEC,
        y_bid_m1=y_bid,
        y_ask_m1=y_ask,
        x_bid_m1=x_bid,
        x_ask_m1=x_ask,
        cost_stress_multiplier=Decimal("1.5"),
    )
    native = native_episodes[0]
    stressed = stressed_episodes[0]
    # Wider spread => wider entry/exit prices, never an identical price
    # with a subtracted fee.
    assert stressed.leg_a.entry_price != native.leg_a.entry_price
    assert stressed.leg_b.entry_price != native.leg_b.entry_price
    assert stressed.realized_pnl() < native.realized_pnl()


def test_stressed_frame_multiplier_of_one_reproduces_native_prices() -> None:
    y_bid, y_ask, x_bid, x_ask = _y_x_frames()
    a, _ = simulate_b3f1_intents(
        [_intent()],
        y_spec=EURUSD_OANDA_SPEC,
        x_spec=USDCAD_OANDA_SPEC,
        y_bid_m1=y_bid,
        y_ask_m1=y_ask,
        x_bid_m1=x_bid,
        x_ask_m1=x_ask,
        cost_stress_multiplier=Decimal("1"),
    )
    b, _ = simulate_b3f1_intents(
        [_intent()],
        y_spec=EURUSD_OANDA_SPEC,
        x_spec=USDCAD_OANDA_SPEC,
        y_bid_m1=y_bid,
        y_ask_m1=y_ask,
        x_bid_m1=x_bid,
        x_ask_m1=x_ask,
    )
    assert a[0].leg_a.entry_price == b[0].leg_a.entry_price
    assert a[0].leg_b.entry_price == b[0].leg_b.entry_price


def test_execution_is_deterministic() -> None:
    y_bid, y_ask, x_bid, x_ask = _y_x_frames()
    kwargs = dict(
        y_spec=EURUSD_OANDA_SPEC,
        x_spec=USDCAD_OANDA_SPEC,
        y_bid_m1=y_bid,
        y_ask_m1=y_ask,
        x_bid_m1=x_bid,
        x_ask_m1=x_ask,
    )
    first, _ = simulate_b3f1_intents([_intent()], **kwargs)
    second, _ = simulate_b3f1_intents([_intent()], **kwargs)
    assert first == second
