from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from ftmoquant.research.alpha_lab.cost_stress import (
    Bar,
    CostStressError,
    required_multiplier_for_family,
    widen_bid_ask_bar,
    widen_bid_ask_frame,
    widen_bid_ask_quotes,
)

D = Decimal


def test_1x_multiplier_reproduces_the_original_quote_exactly() -> None:
    widened = widen_bid_ask_quotes(D("1.0995"), D("1.1005"), D("1.0"))
    assert widened.bid == D("1.09950")
    assert widened.ask == D("1.10050")


@pytest.mark.parametrize("multiplier", [D("1.5"), D("2.0")])
def test_spread_scales_exactly_by_the_multiplier(multiplier: Decimal) -> None:
    bid, ask = D("1.0995"), D("1.1005")
    original_spread = ask - bid
    widened = widen_bid_ask_quotes(bid, ask, multiplier)
    assert widened.ask - widened.bid == multiplier * original_spread


@pytest.mark.parametrize("multiplier", [D("1.0"), D("1.5"), D("2.0"), D("3.25")])
def test_midpoint_is_preserved_exactly(multiplier: Decimal) -> None:
    bid, ask = D("1.0995"), D("1.1005")
    original_mid = (bid + ask) / 2
    widened = widen_bid_ask_quotes(bid, ask, multiplier)
    assert (widened.bid + widened.ask) / 2 == original_mid


def test_widening_is_symmetric_around_the_midpoint() -> None:
    bid, ask = D("1.0990"), D("1.1010")
    mid = (bid + ask) / 2
    widened = widen_bid_ask_quotes(bid, ask, D("2.0"))
    assert mid - widened.bid == widened.ask - mid


@pytest.mark.parametrize("multiplier", [D("1.0"), D("1.5"), D("2.0"), D("10.0")])
def test_zero_spread_remains_zero_at_any_multiplier(multiplier: Decimal) -> None:
    widened = widen_bid_ask_quotes(D("1.10"), D("1.10"), multiplier)
    assert widened.bid == widened.ask == D("1.10")


def test_crossed_market_is_rejected() -> None:
    with pytest.raises(CostStressError, match="crossed market"):
        widen_bid_ask_quotes(D("1.10"), D("1.09"), D("1.0"))


@pytest.mark.parametrize(
    "bid,ask",
    [
        (Decimal("NaN"), D("1.10")),
        (D("1.10"), Decimal("Infinity")),
        (Decimal("NaN"), Decimal("NaN")),
    ],
)
def test_nonfinite_quote_is_rejected(bid: Decimal, ask: Decimal) -> None:
    with pytest.raises(CostStressError):
        widen_bid_ask_quotes(bid, ask, D("1.0"))


def test_multiplier_below_1_is_rejected() -> None:
    with pytest.raises(CostStressError, match="multiplier"):
        widen_bid_ask_quotes(D("1.0995"), D("1.1005"), D("0.99"))


def test_nonfinite_multiplier_is_rejected() -> None:
    with pytest.raises(CostStressError):
        widen_bid_ask_quotes(D("1.0995"), D("1.1005"), Decimal("NaN"))


def test_transform_is_deterministic() -> None:
    first = widen_bid_ask_quotes(D("1.0995"), D("1.1005"), D("1.5"))
    second = widen_bid_ask_quotes(D("1.0995"), D("1.1005"), D("1.5"))
    assert first == second


def test_prefix_invariance_row_by_row_is_independent() -> None:
    """Widening quote k does not depend on any other quote -- proven by
    checking widening a 3-row frame gives identical row-0/row-1 results to
    widening just those two rows alone (a stand-in for "prefix invariance"
    at the scalar level, since the scalar function takes no history)."""

    rows = [(D("1.10"), D("1.101")), (D("1.11"), D("1.112")), (D("1.09"), D("1.091"))]
    full = [widen_bid_ask_quotes(bid, ask, D("1.5")) for bid, ask in rows]
    prefix = [widen_bid_ask_quotes(bid, ask, D("1.5")) for bid, ask in rows[:2]]
    assert full[:2] == prefix


# ---------------------------------------------------------------------------
# Bar-level (OHLC) transform
# ---------------------------------------------------------------------------


def _bar(open_: str, high: str, low: str, close: str) -> Bar:
    return Bar(D(open_), D(high), D(low), D(close))


def test_bar_1x_reproduces_original_bars_exactly() -> None:
    bid_bar = _bar("1.100", "1.101", "1.099", "1.0995")
    ask_bar = _bar("1.1005", "1.1015", "1.0995", "1.1005")
    widened = widen_bid_ask_bar(bid_bar, ask_bar, D("1.0"))
    assert widened.bid == bid_bar
    assert widened.ask == ask_bar


@pytest.mark.parametrize("multiplier", [D("1.5"), D("2.0")])
def test_bar_close_spread_scales_exactly(multiplier: Decimal) -> None:
    bid_bar = _bar("1.100", "1.101", "1.099", "1.0995")
    ask_bar = _bar("1.1005", "1.1015", "1.0995", "1.1005")
    original_close_spread = ask_bar.close - bid_bar.close
    widened = widen_bid_ask_bar(bid_bar, ask_bar, multiplier)
    assert widened.ask.close - widened.bid.close == multiplier * original_close_spread


def test_bar_close_midpoint_is_preserved() -> None:
    bid_bar = _bar("1.100", "1.101", "1.099", "1.0995")
    ask_bar = _bar("1.1005", "1.1015", "1.0995", "1.1005")
    original_mid = (bid_bar.close + ask_bar.close) / 2
    widened = widen_bid_ask_bar(bid_bar, ask_bar, D("2.0"))
    assert (widened.bid.close + widened.ask.close) / 2 == original_mid


def test_bar_ohlc_invariants_are_preserved_after_widening() -> None:
    bid_bar = _bar("1.100", "1.1012", "1.0988", "1.0995")
    ask_bar = _bar("1.1005", "1.1018", "1.0993", "1.1005")
    widened = widen_bid_ask_bar(bid_bar, ask_bar, D("2.0"))
    for bar in (widened.bid, widened.ask):
        assert bar.low <= bar.open <= bar.high
        assert bar.low <= bar.close <= bar.high


def test_bar_transform_never_narrows_distance_from_original_close_mid() -> None:
    bid_bar = _bar("1.100", "1.1012", "1.0988", "1.0995")
    ask_bar = _bar("1.1005", "1.1018", "1.0993", "1.1005")
    mid = (bid_bar.close + ask_bar.close) / 2
    for multiplier in (D("1.0"), D("1.5"), D("2.0")):
        widened = widen_bid_ask_bar(bid_bar, ask_bar, multiplier)
        assert mid - widened.bid.low >= mid - bid_bar.low
        assert widened.ask.high - mid >= ask_bar.high - mid


def test_bar_invalid_field_ordering_is_rejected() -> None:
    with pytest.raises(CostStressError):
        Bar(D("1.10"), D("1.09"), D("1.11"), D("1.10"))  # high < low


def test_bar_deterministic() -> None:
    bid_bar = _bar("1.100", "1.101", "1.099", "1.0995")
    ask_bar = _bar("1.1005", "1.1015", "1.0995", "1.1005")
    first = widen_bid_ask_bar(bid_bar, ask_bar, D("1.5"))
    second = widen_bid_ask_bar(bid_bar, ask_bar, D("1.5"))
    assert first == second


# ---------------------------------------------------------------------------
# Frame-level transform + conservative execution semantics on a synthetic
# fixture (proves no post-hoc P&L transformation is happening: the same
# stop/target collision logic, re-run against widened frames, changes
# outcome only via the wider entry/liquidation prices, never via a direct
# P&L adjustment).
# ---------------------------------------------------------------------------


def _frame(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(rows), freq="min", tz="UTC")
    return pd.DataFrame(
        [
            {"open": float(o), "high": float(h), "low": float(low_), "close": float(c)}
            for o, h, low_, c in rows
        ],
        index=index,
    )


def test_widen_bid_ask_frame_matches_bar_level_transform_row_by_row() -> None:
    bid = _frame(
        [
            ("1.100", "1.101", "1.099", "1.0995"),
            ("1.0995", "1.1005", "1.0990", "1.1000"),
        ]
    )
    ask = _frame(
        [
            ("1.1005", "1.1015", "1.0995", "1.1005"),
            ("1.1000", "1.1010", "1.0995", "1.1005"),
        ]
    )
    stressed_bid, stressed_ask = widen_bid_ask_frame(bid, ask, 2.0)

    for i in range(len(bid)):
        bid_bar = Bar(
            *(D(str(bid.iloc[i][c])) for c in ("open", "high", "low", "close"))
        )
        ask_bar = Bar(
            *(D(str(ask.iloc[i][c])) for c in ("open", "high", "low", "close"))
        )
        expected = widen_bid_ask_bar(bid_bar, ask_bar, D("2.0"))
        assert stressed_bid.iloc[i]["close"] == pytest.approx(float(expected.bid.close))
        assert stressed_ask.iloc[i]["close"] == pytest.approx(float(expected.ask.close))


def test_widen_bid_ask_frame_prefix_invariance() -> None:
    bid = _frame(
        [
            ("1.100", "1.101", "1.099", "1.0995"),
            ("1.0995", "1.1005", "1.0990", "1.1000"),
            ("1.1000", "1.1010", "1.0995", "1.1005"),
        ]
    )
    ask = _frame(
        [
            ("1.1005", "1.1015", "1.0995", "1.1005"),
            ("1.1000", "1.1010", "1.0995", "1.1005"),
            ("1.1005", "1.1015", "1.1000", "1.1010"),
        ]
    )
    full_bid, full_ask = widen_bid_ask_frame(bid, ask, 1.5)
    prefix_bid, prefix_ask = widen_bid_ask_frame(bid.iloc[:2], ask.iloc[:2], 1.5)
    pd.testing.assert_frame_equal(full_bid.iloc[:2], prefix_bid)
    pd.testing.assert_frame_equal(full_ask.iloc[:2], prefix_ask)


def test_widen_bid_ask_frame_rejects_mismatched_index() -> None:
    bid = _frame([("1.100", "1.101", "1.099", "1.0995")])
    ask = _frame(
        [
            ("1.1005", "1.1015", "1.0995", "1.1005"),
            ("1.1000", "1.1010", "1.0995", "1.1005"),
        ]
    )
    with pytest.raises(CostStressError):
        widen_bid_ask_frame(bid, ask, 1.5)


def test_conservative_stop_execution_widens_entry_and_liquidation_prices_not_pnl() -> (
    None
):
    """A tiny synthetic long trade: entry at ASK close, stop touched via
    BID low. Re-running the identical stop-touch rule against widened
    frames must change the realized loss only through the wider raw
    prices, never through a direct P&L haircut -- there is no fee
    parameter anywhere in this computation."""

    bid = _frame(
        [
            ("1.1000", "1.1000", "1.1000", "1.1000"),
            ("1.0950", "1.0950", "1.0940", "1.0950"),
        ]
    )
    ask = _frame(
        [
            ("1.1010", "1.1010", "1.1010", "1.1010"),
            ("1.0960", "1.0960", "1.0950", "1.0960"),
        ]
    )

    def run(bid_frame: pd.DataFrame, ask_frame: pd.DataFrame) -> float:
        entry_price = float(ask_frame.iloc[0]["close"])
        stop_price = entry_price - 0.01
        liquidation_low = float(bid_frame.iloc[1]["low"])
        stop_touched = liquidation_low <= stop_price
        exit_price = stop_price if stop_touched else float(bid_frame.iloc[1]["close"])
        return exit_price - entry_price

    base_pnl = run(bid, ask)
    stressed_bid, stressed_ask = widen_bid_ask_frame(bid, ask, 2.0)
    stressed_pnl = run(stressed_bid, stressed_ask)

    assert (
        stressed_pnl < base_pnl
    )  # wider entry + wider adverse liquidation -> worse P&L
    assert float(stressed_ask.iloc[0]["close"]) > float(ask.iloc[0]["close"])
    assert float(stressed_bid.iloc[1]["low"]) < float(bid.iloc[1]["low"])


def test_required_multiplier_for_family_matches_frozen_v2_gate() -> None:
    assert required_multiplier_for_family("B3F1") == (D("1.5"),)
    assert required_multiplier_for_family("B3F2") == (D("1.5"), D("2.0"))
    assert required_multiplier_for_family("B3F3") == (D("1.5"), D("2.0"))


def test_required_multiplier_for_family_rejects_unknown_family() -> None:
    with pytest.raises(CostStressError):
        required_multiplier_for_family("B3F4")  # type: ignore[arg-type]
