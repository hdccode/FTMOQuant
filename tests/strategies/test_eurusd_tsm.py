from __future__ import annotations

import math
from decimal import Decimal

import pytest

from ftmoquant.research.g1.normalization import G1VolatilityNormalizer
from ftmoquant.strategies.eurusd_tsm import (
    EurusdTsmFamily,
    EurusdTsmValidationError,
)
from ftmoquant.strategies.trend_pullback import CompletedPair, PriceBar, Timeframe
from ftmoquant.strategies.ts_momentum import RawDirectionalTarget

HOUR_NS = 3_600_000_000_000


def _prices(returns: list[float], start: float = 1.1) -> list[float]:
    result = [start]
    for value in returns:
        result.append(result[-1] * math.exp(value))
    return result


def _pair(
    price: float,
    index: int,
    *,
    timeframe: Timeframe = Timeframe.FOUR_HOURS,
    contiguous: bool = True,
) -> CompletedPair:
    midpoint = Decimal(str(price))
    half_spread = Decimal("0.00001")
    bid_value = midpoint - half_spread
    ask_value = midpoint + half_spread
    bid = PriceBar(bid_value, bid_value, bid_value, bid_value)
    ask = PriceBar(ask_value, ask_value, ask_value, ask_value)
    interval = HOUR_NS if timeframe is Timeframe.HOUR else 4 * HOUR_NS
    timestamp = (index + 1) * interval
    return CompletedPair(
        timeframe=timeframe,
        bid=bid,
        ask=ask,
        ts_event=timestamp,
        info_time_ns=timestamp,
        contiguous=contiguous,
    )


def _parameters(
    *, deadband: float = 0.0, refresh: int = 1
) -> dict[str, str | int | float | bool | None]:
    return {
        "timeframe": "H4",
        "lookback_bars": 6,
        "deadband": deadband,
        "refresh_interval_bars": refresh,
    }


def test_warmup_stays_unavailable_and_never_backfills_early_signals() -> None:
    family = EurusdTsmFamily()
    returns = [0.01 if index % 2 else -0.01 for index in range(14)] + [0.01] * 6
    pairs = tuple(_pair(price, index) for index, price in enumerate(_prices(returns)))

    assert family.build_signals(pairs[:-1], _parameters()) == ()
    signals = family.build_signals(pairs, _parameters())

    assert len(signals) == 1
    assert signals[0].target is RawDirectionalTarget.LONG
    assert signals[0].signal_information_ns == pairs[-1].info_time_ns


def test_long_short_flat_and_deadband_behaviour_are_exact() -> None:
    family = EurusdTsmFamily()
    alternating = [0.01 if index % 2 else -0.01 for index in range(14)]
    long_pairs = tuple(
        _pair(price, index)
        for index, price in enumerate(_prices(alternating + [0.01] * 6))
    )
    short_pairs = tuple(
        _pair(price, index)
        for index, price in enumerate(_prices(alternating + [-0.01] * 6))
    )
    flat_pairs = tuple(
        _pair(price, index)
        for index, price in enumerate(
            _prices([0.01 if index % 2 else -0.01 for index in range(20)])
        )
    )
    weak_positive = alternating + [-0.01, 0.01, -0.01, 0.01, -0.01, 0.02]
    weak_pairs = tuple(
        _pair(price, index) for index, price in enumerate(_prices(weak_positive))
    )

    assert family.build_signals(long_pairs, _parameters())[0].target is (
        RawDirectionalTarget.LONG
    )
    assert family.build_signals(short_pairs, _parameters())[0].target is (
        RawDirectionalTarget.SHORT
    )
    assert family.build_signals(flat_pairs, _parameters())[0].target is (
        RawDirectionalTarget.FLAT
    )
    assert family.build_signals(weak_pairs, _parameters(deadband=0.0))[0].target is (
        RawDirectionalTarget.LONG
    )
    assert family.build_signals(weak_pairs, _parameters(deadband=0.5))[0].target is (
        RawDirectionalTarget.FLAT
    )


def test_refresh_interval_delays_reversal_until_predetermined_bar() -> None:
    family = EurusdTsmFamily()
    state = family.create_state(_parameters(refresh=3))
    initial_returns = [0.01 if index % 2 else -0.01 for index in range(14)] + [0.01] * 6
    prices = _prices(initial_returns)
    initial_signal = None
    for index, price in enumerate(prices):
        signal = state.on_bar(_pair(price, index))
        if signal is not None:
            initial_signal = signal
    assert initial_signal is not None
    assert initial_signal.target is RawDirectionalTarget.LONG

    next_price = prices[-1]
    for offset in range(1, 3):
        next_price *= math.exp(-0.10)
        assert state.on_bar(_pair(next_price, len(prices) - 1 + offset)) is None
    next_price *= math.exp(-0.10)
    reversal = state.on_bar(_pair(next_price, len(prices) + 2))

    assert reversal is not None
    assert reversal.target is RawDirectionalTarget.SHORT


def test_unchanged_direction_remains_visible_to_risk_sizing_refreshes() -> None:
    family = EurusdTsmFamily()
    returns = [0.01 if index % 2 else -0.01 for index in range(14)] + [0.01] * 9
    pairs = tuple(_pair(price, index) for index, price in enumerate(_prices(returns)))

    changed = family.build_signals(pairs, _parameters(refresh=3))
    refreshes = family.build_target_refreshes(pairs, _parameters(refresh=3))

    assert len(changed) == 1
    assert len(refreshes) == 2
    assert refreshes[0].target_changed is True
    assert refreshes[1].target_changed is False
    assert refreshes[0].target is refreshes[1].target is RawDirectionalTarget.LONG


def test_future_bars_cannot_change_preexisting_signal_history() -> None:
    family = EurusdTsmFamily()
    returns = [0.01 if index % 2 else -0.01 for index in range(14)] + [0.01] * 6
    prices = _prices(returns)
    prefix = tuple(_pair(price, index) for index, price in enumerate(prices))
    future_price = prices[-1]
    future = []
    for offset in range(1, 8):
        future_price *= math.exp(-0.10)
        future.append(_pair(future_price, len(prices) - 1 + offset))

    prefix_signals = family.build_signals(prefix, _parameters())
    full_signals = family.build_signals((*prefix, *future), _parameters())

    assert (
        tuple(
            signal
            for signal in full_signals
            if signal.signal_information_ns <= prefix[-1].info_time_ns
        )
        == prefix_signals
    )


def test_noncontiguous_bar_resets_history_without_manufactured_volatility() -> None:
    family = EurusdTsmFamily()
    state = family.create_state(_parameters())
    returns = [0.01 if index % 2 else -0.01 for index in range(14)] + [0.01] * 6
    prices = _prices(returns)
    for index, price in enumerate(prices):
        state.on_bar(_pair(price, index))

    gap_index = len(prices)
    assert state.on_bar(_pair(prices[-1], gap_index, contiguous=False)) is None
    after_gap = prices[-1]
    for offset in range(1, 21):
        after_gap *= math.exp(-0.01 if offset % 2 else 0.01)
        assert state.on_bar(_pair(after_gap, gap_index + offset)) is None


def test_zero_volatility_fails_closed_without_a_manufactured_floor() -> None:
    family = EurusdTsmFamily()
    pairs = tuple(_pair(1.1, index) for index in range(21))

    assert family.build_signals(pairs, _parameters()) == ()


def test_grid_neighbours_move_one_adjacent_dimension_and_never_timeframe() -> None:
    family = EurusdTsmFamily()
    center = {
        "timeframe": "H1",
        "lookback_bars": 72,
        "deadband": 0.25,
        "refresh_interval_bars": 4,
    }
    neighbours = family.neighbours(center)

    assert len(neighbours) == 6
    assert all(item["timeframe"] == "H1" for item in neighbours)
    assert {item["lookback_bars"] for item in neighbours} == {24, 72, 120}
    assert {item["deadband"] for item in neighbours} == {0.0, 0.25, 0.5}
    assert {item["refresh_interval_bars"] for item in neighbours} == {1, 4, 24}
    for item in neighbours:
        changed = sum(item[key] != center[key] for key in center)
        assert changed == 1

    with pytest.raises(EurusdTsmValidationError, match="lookback"):
        family.validate_parameters({**center, "lookback_bars": 6})


def test_one_percent_normalization_is_the_family_comparison_target() -> None:
    family = EurusdTsmFamily()
    target = family.spec.canonical_document["risk_normalization"][
        "target_annualized_volatility"
    ]
    exposure = G1VolatilityNormalizer().exposure_for_volatility(1.0, 0.10)

    assert target == 0.01
    assert exposure == pytest.approx(0.10)
