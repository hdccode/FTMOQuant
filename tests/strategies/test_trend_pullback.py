from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest
from nautilus_trader.indicators import AverageTrueRange, ExponentialMovingAverage
from nautilus_trader.model import Bar, BarType, Price, Quantity

from ftmoquant.research import load_trend_pullback_spec
from ftmoquant.strategies import (
    CompletedBarPairer,
    CompletedPair,
    Direction,
    EntryOrderIntent,
    ExitOrderIntent,
    ExitReason,
    PriceBar,
    Timeframe,
    TrendPullbackStateMachine,
    TrendPullbackStrategy,
)

SPEC_PATH = "config/strategies/trend_pullback_v1.yaml"
MINUTE_NS = 60_000_000_000
HOUR_NS = 60 * MINUTE_NS
FOUR_HOURS_NS = 4 * HOUR_NS
START_NS = 1_780_000_000_000_000_000
SPREAD = Decimal("0.0002")


def test_uses_frozen_config_and_native_indicator_lifecycle() -> None:
    spec = load_trend_pullback_spec(SPEC_PATH)
    core = TrendPullbackStateMachine(spec)

    assert isinstance(core.signal_ema, ExponentialMovingAverage)
    assert isinstance(core.atr, AverageTrueRange)
    assert isinstance(core.trend_fast_ema, ExponentialMovingAverage)
    assert isinstance(core.trend_slow_ema, ExponentialMovingAverage)
    assert not core.indicators_initialized

    changed = replace(
        spec,
        signal=replace(
            spec.signal,
            pullback=replace(spec.signal.pullback, ema_period=21),
        ),
    )
    with pytest.raises(ValueError, match="frozen G1.1 hash"):
        TrendPullbackStateMachine(changed)


def test_warmup_initializes_but_cannot_trade_across_split() -> None:
    core = _core(trading_enabled=False)
    last_4h = _warm_trend(core, Direction.LONG)
    last_1h = _warm_signal(core, last_4h)
    assert core.indicators_initialized

    arm = _mid_pair(Timeframe.HOUR, Decimal("0.8000"), last_1h + HOUR_NS)
    core.on_pair(arm)
    core.enable_trading()
    trigger = _mid_pair(Timeframe.HOUR, Decimal("1.5000"), arm.ts_event + HOUR_NS)

    assert core.on_pair(trigger) == ()
    assert not core.has_entry_intent


@pytest.mark.parametrize("direction", [Direction.LONG, Direction.SHORT])
def test_symmetric_arm_trigger_and_first_strictly_later_entry(
    direction: Direction,
) -> None:
    core, trigger = _triggered_core(direction)

    same_information_time = _mid_pair(
        Timeframe.MINUTE,
        Decimal("1.2000"),
        trigger.ts_event,
        info_ns=trigger.info_time_ns,
    )
    assert core.on_pair(same_information_time) == ()

    first_later = _mid_pair(
        Timeframe.MINUTE,
        Decimal("1.2000"),
        trigger.ts_event + MINUTE_NS,
        info_ns=trigger.ts_event + MINUTE_NS + 1,
    )
    actions = core.on_pair(first_later)

    assert len(actions) == 1
    entry = actions[0]
    assert isinstance(entry, EntryOrderIntent)
    assert entry.direction is direction
    assert entry.signal_event_ns == trigger.ts_event
    assert entry.signal_info_ns == trigger.info_time_ns
    assert entry.stop_distance > 0
    assert entry.normalized_quantity * entry.stop_distance == Decimal("1.0")


def test_entry_intent_expires_after_five_minutes() -> None:
    core, trigger = _triggered_core(Direction.LONG)
    expired_pair = _mid_pair(
        Timeframe.MINUTE,
        Decimal("1.2000"),
        trigger.ts_event + 6 * MINUTE_NS,
        info_ns=trigger.ts_event + 6 * MINUTE_NS + 1,
    )

    assert core.on_pair(expired_pair) == ()
    assert not core.has_entry_intent


def test_entry_intent_is_still_eligible_at_exactly_five_minutes() -> None:
    core, trigger = _triggered_core(Direction.LONG)
    deadline = trigger.info_time_ns + 5 * MINUTE_NS
    pair = _mid_pair(
        Timeframe.MINUTE,
        Decimal("1.2000"),
        deadline - 1,
        info_ns=deadline,
    )

    actions = core.on_pair(pair)

    assert len(actions) == 1
    assert isinstance(actions[0], EntryOrderIntent)


def test_arm_expires_after_third_non_triggering_bar() -> None:
    core = _core(trading_enabled=True)
    last_4h = _warm_trend(core, Direction.LONG)
    last_1h = _warm_signal(core, last_4h)
    arm = _mid_pair(Timeframe.HOUR, Decimal("0.8000"), last_1h + HOUR_NS)
    core.on_pair(arm)
    assert core.armed_direction is Direction.LONG

    timestamp = arm.ts_event
    for value in ("0.7900", "0.7800"):
        timestamp += HOUR_NS
        core.on_pair(_mid_pair(Timeframe.HOUR, Decimal(value), timestamp))
        assert core.armed_direction is Direction.LONG
    timestamp += HOUR_NS
    core.on_pair(_mid_pair(Timeframe.HOUR, Decimal("0.7700"), timestamp))

    assert core.armed_direction is None
    assert not core.has_entry_intent


def test_gap_resets_indicator_state_and_discards_setup() -> None:
    core = _core(trading_enabled=True)
    last_4h = _warm_trend(core, Direction.LONG)
    last_1h = _warm_signal(core, last_4h)
    arm = _mid_pair(Timeframe.HOUR, Decimal("0.8000"), last_1h + HOUR_NS)
    core.on_pair(arm)
    assert core.armed_direction is Direction.LONG

    after_gap = _mid_pair(
        Timeframe.HOUR,
        Decimal("1.5000"),
        arm.ts_event + 2 * HOUR_NS,
        contiguous=False,
    )
    assert core.on_pair(after_gap) == ()

    assert not core.indicators_initialized
    assert core.armed_direction is None
    assert not core.has_entry_intent


def test_future_trend_information_is_not_read_by_an_earlier_decision() -> None:
    core = _core(trading_enabled=True)
    last_4h = _warm_trend(core, Direction.LONG)
    last_1h = _warm_signal(core, last_4h)
    arm = _mid_pair(Timeframe.HOUR, Decimal("0.8000"), last_1h + HOUR_NS)
    core.on_pair(arm)
    assert core.armed_direction is Direction.LONG

    future_event = arm.ts_event + FOUR_HOURS_NS
    future_info = future_event + 100 * HOUR_NS
    core.on_pair(
        _mid_pair(
            Timeframe.FOUR_HOURS,
            Decimal("0.1000"),
            future_event,
            info_ns=future_info,
        )
    )
    trigger = _mid_pair(
        Timeframe.HOUR,
        Decimal("1.5000"),
        arm.ts_event + HOUR_NS,
    )

    assert core.on_pair(trigger) == ()
    assert core.has_entry_intent


def test_long_same_minute_stop_and_target_resolves_stop_first() -> None:
    core, entry = _filled_core(Direction.LONG)
    stop = Decimal("1.2000") - entry.stop_distance
    target = Decimal("1.2000") + entry.stop_distance * Decimal(2)
    decision = _execution_pair(
        event_ns=entry.execution_event_ns + MINUTE_NS,
        bid=PriceBar(
            open=Decimal("1.2000"),
            high=target + Decimal("0.0001"),
            low=stop - Decimal("0.0001"),
            close=Decimal("1.2000"),
        ),
    )

    actions = core.on_pair(decision)

    assert len(actions) == 1
    exit_intent = actions[0]
    assert isinstance(exit_intent, ExitOrderIntent)
    assert exit_intent.reason is ExitReason.STOP_LOSS
    assert exit_intent.stop_price == stop
    assert exit_intent.target_price == target


def test_short_target_uses_ask_liquidation_side() -> None:
    core, entry = _filled_core(Direction.SHORT)
    target = Decimal("1.2000") - entry.stop_distance * Decimal(2)
    ask = PriceBar(
        open=Decimal("1.2002"),
        high=Decimal("1.2003"),
        low=target - Decimal("0.0001"),
        close=Decimal("1.2002"),
    )
    bid = _shift_bar(ask, -SPREAD)
    decision = CompletedPair(
        timeframe=Timeframe.MINUTE,
        bid=bid,
        ask=ask,
        ts_event=entry.execution_event_ns + MINUTE_NS,
        info_time_ns=entry.execution_info_ns + MINUTE_NS,
    )

    actions = core.on_pair(decision)

    assert len(actions) == 1
    assert isinstance(actions[0], ExitOrderIntent)
    assert actions[0].reason is ExitReason.TAKE_PROFIT


def test_time_exit_waits_for_48_later_hours_then_next_minute_pair() -> None:
    core, entry = _filled_core(Direction.LONG)
    timestamp = entry.execution_event_ns
    for _ in range(48):
        timestamp += HOUR_NS
        assert (
            core.on_pair(_mid_pair(Timeframe.HOUR, Decimal("1.2000"), timestamp)) == ()
        )

    same_time = _execution_pair(
        event_ns=timestamp,
        info_ns=timestamp + 1,
        bid=_flat_bar(Decimal("1.2000")),
    )
    assert core.on_pair(same_time) == ()

    later = _execution_pair(
        event_ns=timestamp + MINUTE_NS,
        info_ns=timestamp + MINUTE_NS + 1,
        bid=_flat_bar(Decimal("1.2000")),
    )
    actions = core.on_pair(later)

    assert len(actions) == 1
    assert isinstance(actions[0], ExitOrderIntent)
    assert actions[0].reason is ExitReason.TIME


def test_confirm_flat_requires_a_new_post_flat_arm() -> None:
    core, entry = _filled_core(Direction.LONG)
    core.confirm_flat()
    next_hour = _mid_pair(
        Timeframe.HOUR,
        Decimal("0.8000"),
        entry.execution_event_ns + HOUR_NS,
    )

    assert core.on_pair(next_hour) == ()
    assert core.armed_direction is None


def test_native_pairer_waits_for_exact_timestamp_and_uses_later_ts_init() -> None:
    spec = load_trend_pullback_spec(SPEC_PATH)
    pairer = CompletedBarPairer(spec)
    event = START_NS + HOUR_NS
    bid = _native_bar(
        spec.data_semantics.signal_bid_bar_type,
        "1.1000",
        event,
        event + 10,
    )
    ask = _native_bar(
        spec.data_semantics.signal_ask_bar_type,
        "1.1002",
        event,
        event + 20,
    )

    assert pairer.accept(ask) is None
    pair = pairer.accept(bid)

    assert pair is not None
    assert pair.timeframe is Timeframe.HOUR
    assert pair.info_time_ns == event + 20
    assert pair.midpoint.close == Decimal("1.1001")


def test_native_pairer_does_not_carry_a_missing_side_forward() -> None:
    spec = load_trend_pullback_spec(SPEC_PATH)
    pairer = CompletedBarPairer(spec)
    first = START_NS + HOUR_NS
    assert (
        pairer.accept(
            _native_bar(
                spec.data_semantics.signal_bid_bar_type,
                "1.1000",
                first,
                first,
            )
        )
        is None
    )
    assert (
        pairer.accept(
            _native_bar(
                spec.data_semantics.signal_ask_bar_type,
                "1.1002",
                first + HOUR_NS,
                first + HOUR_NS,
            )
        )
        is None
    )

    pair = pairer.accept(
        _native_bar(
            spec.data_semantics.signal_bid_bar_type,
            "1.1000",
            first + HOUR_NS,
            first + HOUR_NS,
        )
    )
    assert pair is not None
    assert pair.ts_event == first + HOUR_NS


def test_invalid_bid_ask_ordering_is_rejected() -> None:
    with pytest.raises(ValueError, match="ASK OHLC cannot be below BID"):
        CompletedPair(
            timeframe=Timeframe.MINUTE,
            bid=_flat_bar(Decimal("1.1002")),
            ask=_flat_bar(Decimal("1.1000")),
            ts_event=START_NS,
            info_time_ns=START_NS,
        )


def test_strategy_fails_closed_when_dataset_is_not_research_ready() -> None:
    strategy = TrendPullbackStrategy(research_ready=False)

    with pytest.raises(RuntimeError, match="research-ready G0.6 dataset"):
        strategy.on_start()


def _core(*, trading_enabled: bool) -> TrendPullbackStateMachine:
    return TrendPullbackStateMachine(
        load_trend_pullback_spec(SPEC_PATH), trading_enabled=trading_enabled
    )


def _triggered_core(
    direction: Direction,
) -> tuple[TrendPullbackStateMachine, CompletedPair]:
    core = _core(trading_enabled=True)
    last_4h = _warm_trend(core, direction)
    last_1h = _warm_signal(core, last_4h, direction)
    arm_value = Decimal("0.8000") if direction is Direction.LONG else Decimal("1.8000")
    trigger_value = (
        Decimal("1.5000") if direction is Direction.LONG else Decimal("0.5000")
    )
    arm = _mid_pair(Timeframe.HOUR, arm_value, last_1h + HOUR_NS)
    core.on_pair(arm)
    assert core.armed_direction is direction
    trigger = _mid_pair(Timeframe.HOUR, trigger_value, arm.ts_event + HOUR_NS)
    assert core.on_pair(trigger) == ()
    assert core.has_entry_intent
    return core, trigger


def _filled_core(
    direction: Direction,
) -> tuple[TrendPullbackStateMachine, EntryOrderIntent]:
    core, trigger = _triggered_core(direction)
    execution = _mid_pair(
        Timeframe.MINUTE,
        Decimal("1.2000"),
        trigger.ts_event + MINUTE_NS,
        info_ns=trigger.ts_event + MINUTE_NS + 1,
    )
    actions = core.on_pair(execution)
    assert len(actions) == 1 and isinstance(actions[0], EntryOrderIntent)
    entry = actions[0]
    core.confirm_entry_fill(
        direction=direction,
        fill_price=Decimal("1.2000"),
        fill_info_ns=entry.execution_info_ns,
    )
    return core, entry


def _warm_trend(core: TrendPullbackStateMachine, direction: Direction) -> int:
    timestamp = START_NS
    for index in range(200):
        timestamp += FOUR_HOURS_NS
        delta = Decimal(index) / Decimal("10000")
        close = (
            Decimal("1.0000") + delta
            if direction is Direction.LONG
            else Decimal("1.5000") - delta
        )
        core.on_pair(_mid_pair(Timeframe.FOUR_HOURS, close, timestamp))
    return timestamp


def _warm_signal(
    core: TrendPullbackStateMachine,
    after_ns: int,
    direction: Direction = Direction.LONG,
) -> int:
    timestamp = after_ns
    for index in range(20):
        timestamp += HOUR_NS
        delta = Decimal(index) / Decimal("10000")
        close = (
            Decimal("1.1000") + delta
            if direction is Direction.LONG
            else Decimal("1.1000") - delta
        )
        core.on_pair(_mid_pair(Timeframe.HOUR, close, timestamp))
    return timestamp


def _mid_pair(
    timeframe: Timeframe,
    midpoint: Decimal,
    event_ns: int,
    *,
    info_ns: int | None = None,
    contiguous: bool = True,
) -> CompletedPair:
    bid = _flat_bar(midpoint - SPREAD / 2)
    ask = _flat_bar(midpoint + SPREAD / 2)
    return CompletedPair(
        timeframe=timeframe,
        bid=bid,
        ask=ask,
        ts_event=event_ns,
        info_time_ns=event_ns + 1 if info_ns is None else info_ns,
        contiguous=contiguous,
    )


def _execution_pair(
    *,
    event_ns: int,
    bid: PriceBar,
    info_ns: int | None = None,
) -> CompletedPair:
    return CompletedPair(
        timeframe=Timeframe.MINUTE,
        bid=bid,
        ask=_shift_bar(bid, SPREAD),
        ts_event=event_ns,
        info_time_ns=event_ns + 1 if info_ns is None else info_ns,
    )


def _flat_bar(close: Decimal) -> PriceBar:
    return PriceBar(
        open=close,
        high=close + Decimal("0.0001"),
        low=close - Decimal("0.0001"),
        close=close,
    )


def _shift_bar(bar: PriceBar, shift: Decimal) -> PriceBar:
    return PriceBar(
        open=bar.open + shift,
        high=bar.high + shift,
        low=bar.low + shift,
        close=bar.close + shift,
    )


def _native_bar(bar_type: str, close: str, ts_event: int, ts_init: int) -> Bar:
    value = Decimal(close)
    return Bar(
        bar_type=BarType.from_str(bar_type),
        open=Price.from_str(str(value)),
        high=Price.from_str(str(value + Decimal("0.0001"))),
        low=Price.from_str(str(value - Decimal("0.0001"))),
        close=Price.from_str(str(value)),
        volume=Quantity.from_int(1),
        ts_event=ts_event,
        ts_init=ts_init,
    )
