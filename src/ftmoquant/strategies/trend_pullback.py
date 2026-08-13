"""Exact G1.2 implementation of the frozen ``trend_pullback_v1`` contract.

The state machine is deliberately separate from Nautilus order matching. It
owns synchronized midpoint signals and fixed-R exit decisions; the existing
G0.7 harness remains the only owner of fills, spread, costs, and latency.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from enum import StrEnum
from pathlib import Path

from nautilus_trader.indicators import (
    AverageTrueRange,
    ExponentialMovingAverage,
    MovingAverageType,
)
from nautilus_trader.model import (
    Bar,
    BarType,
    ClientOrderId,
    CurrencyPair,
    InstrumentId,
    OrderCanceled,
    OrderDenied,
    OrderExpired,
    OrderFilled,
    OrderRejected,
    OrderSide,
    PositionClosed,
    StrategyId,
)
from nautilus_trader.trading import Strategy, StrategyConfig

from ftmoquant.research import (
    TrendPullbackSpec,
    load_trend_pullback_spec,
    strategy_config_sha256,
)

FROZEN_CONFIG_SHA256 = (
    "d21100fc8412f4d258efdc90b2f1a936c0eb27cd6f88081ae082f24ae6d4cc5e"
)
DEFAULT_SPEC_PATH = Path("config/strategies/trend_pullback_v1.yaml")

_MINUTE_NS = 60_000_000_000
_HOUR_NS = 60 * _MINUTE_NS
_FOUR_HOURS_NS = 4 * _HOUR_NS


class Timeframe(StrEnum):
    """The only completed-bar streams admitted by the frozen strategy."""

    MINUTE = "1m"
    HOUR = "1h"
    FOUR_HOURS = "4h"

    @property
    def interval_ns(self) -> int:
        return {
            Timeframe.MINUTE: _MINUTE_NS,
            Timeframe.HOUR: _HOUR_NS,
            Timeframe.FOUR_HOURS: _FOUR_HOURS_NS,
        }[self]


class Direction(StrEnum):
    LONG = "long"
    SHORT = "short"


class ExitReason(StrEnum):
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TIME = "time"


@dataclass(frozen=True, slots=True)
class PriceBar:
    """One completed side-specific OHLC observation."""

    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal

    def __post_init__(self) -> None:
        values = (self.open, self.high, self.low, self.close)
        if any(not value.is_finite() or value <= 0 for value in values):
            raise ValueError("bar prices must be finite and positive")
        if self.high < max(self.open, self.close) or self.low > min(
            self.open, self.close
        ):
            raise ValueError("bar OHLC is inconsistent")


@dataclass(frozen=True, slots=True)
class CompletedPair:
    """One synchronized completed BID/ASK pair."""

    timeframe: Timeframe
    bid: PriceBar
    ask: PriceBar
    ts_event: int
    info_time_ns: int
    contiguous: bool = True

    def __post_init__(self) -> None:
        if self.ts_event < 0 or self.info_time_ns < self.ts_event:
            raise ValueError("completed-pair timestamps are invalid")
        if any(
            ask < bid
            for bid, ask in zip(
                (self.bid.open, self.bid.high, self.bid.low, self.bid.close),
                (self.ask.open, self.ask.high, self.ask.low, self.ask.close),
                strict=True,
            )
        ):
            raise ValueError("ASK OHLC cannot be below BID OHLC")

    @property
    def midpoint(self) -> PriceBar:
        two = Decimal(2)
        return PriceBar(
            open=(self.bid.open + self.ask.open) / two,
            high=(self.bid.high + self.ask.high) / two,
            low=(self.bid.low + self.ask.low) / two,
            close=(self.bid.close + self.ask.close) / two,
        )


@dataclass(frozen=True, slots=True)
class EntryOrderIntent:
    """A market entry ready for the canonical G0.7 execution boundary."""

    direction: Direction
    signal_event_ns: int
    signal_info_ns: int
    execution_event_ns: int
    execution_info_ns: int
    stop_distance: Decimal
    normalized_quantity: Decimal


@dataclass(frozen=True, slots=True)
class ExitOrderIntent:
    """A fixed stop, target, or time-exit market instruction."""

    direction: Direction
    reason: ExitReason
    decision_event_ns: int
    decision_info_ns: int
    stop_price: Decimal
    target_price: Decimal


StrategyAction = EntryOrderIntent | ExitOrderIntent


@dataclass(frozen=True, slots=True)
class _SignalPoint:
    close: Decimal
    ema: Decimal
    event_ns: int
    info_ns: int


@dataclass(frozen=True, slots=True)
class _TrendPoint:
    direction: Direction | None
    info_ns: int


@dataclass(frozen=True, slots=True)
class _ArmedSetup:
    direction: Direction
    event_ns: int
    eligible_bars_seen: int = 0


@dataclass(frozen=True, slots=True)
class _PendingSignal:
    direction: Direction
    event_ns: int
    info_ns: int
    stop_distance: Decimal


@dataclass(frozen=True, slots=True)
class _PositionState:
    direction: Direction
    fill_price: Decimal
    fill_info_ns: int
    stop_price: Decimal
    target_price: Decimal
    completed_hour_bars: int = 0
    time_ready_info_ns: int | None = None


class TrendPullbackStateMachine:
    """Pure event-ordering shell around pinned Nautilus EMA/ATR indicators."""

    def __init__(
        self,
        spec: TrendPullbackSpec,
        *,
        trading_enabled: bool = False,
    ) -> None:
        if strategy_config_sha256(spec) != FROZEN_CONFIG_SHA256:
            raise ValueError("strategy config does not match frozen G1.1 hash")
        self.spec = spec
        self.trading_enabled = trading_enabled
        self.signal_ema = ExponentialMovingAverage(spec.signal.pullback.ema_period)
        self.atr = AverageTrueRange(
            spec.signal.volatility.atr_period,
            ma_type=MovingAverageType.Exponential,
            use_previous=True,
        )
        self.trend_fast_ema = ExponentialMovingAverage(
            spec.signal.trend.fast_ema_period
        )
        self.trend_slow_ema = ExponentialMovingAverage(
            spec.signal.trend.slow_ema_period
        )
        self._last_signal: _SignalPoint | None = None
        self._trend: list[_TrendPoint] = []
        self._armed: _ArmedSetup | None = None
        self._pending: _PendingSignal | None = None
        self._awaiting_fill: EntryOrderIntent | None = None
        self._position: _PositionState | None = None
        self._exit_pending = False

    @property
    def indicators_initialized(self) -> bool:
        return bool(
            self.signal_ema.initialized
            and self.atr.initialized
            and self.trend_fast_ema.initialized
            and self.trend_slow_ema.initialized
        )

    @property
    def armed_direction(self) -> Direction | None:
        return None if self._armed is None else self._armed.direction

    @property
    def has_entry_intent(self) -> bool:
        return self._pending is not None or self._awaiting_fill is not None

    @property
    def has_position(self) -> bool:
        return self._position is not None

    def enable_trading(self) -> None:
        """End warmup without allowing a setup to cross the split boundary."""

        if self._position is not None or self.has_entry_intent:
            raise RuntimeError("cannot enable trading with active strategy state")
        self._armed = None
        self._last_signal = None
        self.trading_enabled = True

    def on_pair(self, pair: CompletedPair) -> tuple[StrategyAction, ...]:
        """Consume exactly one synchronized pair and return at most one action."""

        if pair.timeframe is Timeframe.FOUR_HOURS:
            self._on_trend_pair(pair)
            return ()
        if pair.timeframe is Timeframe.HOUR:
            return self._on_signal_pair(pair)
        return self._on_execution_pair(pair)

    def confirm_entry_fill(
        self,
        *,
        direction: Direction,
        fill_price: Decimal,
        fill_info_ns: int,
    ) -> None:
        """Bind fixed exits to the actual native fill, never a signal price."""

        intent = self._awaiting_fill
        if intent is None or intent.direction is not direction:
            raise RuntimeError("fill does not match an awaiting entry")
        if not fill_price.is_finite() or fill_price <= 0:
            raise ValueError("fill price must be finite and positive")
        distance = intent.stop_distance
        if direction is Direction.LONG:
            stop = fill_price - distance
            target = fill_price + distance * self.spec.exits.target_r_multiple
        else:
            stop = fill_price + distance
            target = fill_price - distance * self.spec.exits.target_r_multiple
        if stop <= 0 or target <= 0:
            raise ValueError("fill produces an invalid fixed exit price")
        self._position = _PositionState(
            direction=direction,
            fill_price=fill_price,
            fill_info_ns=fill_info_ns,
            stop_price=stop,
            target_price=target,
        )
        self._awaiting_fill = None

    def reject_entry(self) -> None:
        """Fail closed after a native entry rejection/cancel/expiry."""

        self._pending = None
        self._awaiting_fill = None
        self._armed = None
        self._last_signal = None

    def confirm_flat(self) -> None:
        """Clear trade state and require a new post-flat arm and trigger."""

        self._position = None
        self._exit_pending = False
        self._armed = None
        self._last_signal = None

    def invalidate(self, timeframe: Timeframe) -> None:
        """Reset every value which would otherwise be carried across a gap."""

        if timeframe is Timeframe.FOUR_HOURS:
            self.trend_fast_ema.reset()
            self.trend_slow_ema.reset()
            self._trend.clear()
            self._armed = None
        elif timeframe is Timeframe.HOUR:
            self.signal_ema.reset()
            self.atr.reset()
            self._last_signal = None
            self._armed = None
            self._pending = None
        elif self._pending is not None:
            self._pending = None

    def _on_trend_pair(self, pair: CompletedPair) -> None:
        if not pair.contiguous:
            self.invalidate(Timeframe.FOUR_HOURS)
        close = float(pair.midpoint.close)
        self.trend_fast_ema.update_raw(close)
        self.trend_slow_ema.update_raw(close)
        direction: Direction | None = None
        if self.trend_fast_ema.initialized and self.trend_slow_ema.initialized:
            if self.trend_fast_ema.value > self.trend_slow_ema.value:
                direction = Direction.LONG
            elif self.trend_fast_ema.value < self.trend_slow_ema.value:
                direction = Direction.SHORT
        self._trend.append(_TrendPoint(direction=direction, info_ns=pair.info_time_ns))
        self._trend.sort(key=lambda item: item.info_ns)
        if len(self._trend) > self.spec.signal.trend.slow_ema_period + 1:
            del self._trend[0]

    def _on_signal_pair(self, pair: CompletedPair) -> tuple[StrategyAction, ...]:
        if not pair.contiguous:
            self.invalidate(Timeframe.HOUR)
        midpoint = pair.midpoint
        self.signal_ema.update_raw(float(midpoint.close))
        self.atr.update_raw(
            float(midpoint.high), float(midpoint.low), float(midpoint.close)
        )
        current = _SignalPoint(
            close=midpoint.close,
            ema=Decimal(str(self.signal_ema.value)),
            event_ns=pair.ts_event,
            info_ns=pair.info_time_ns,
        )

        if (
            self._position is not None
            and pair.info_time_ns > self._position.fill_info_ns
        ):
            count = self._position.completed_hour_bars + 1
            ready = self._position.time_ready_info_ns
            if count == self.spec.exits.time_exit_bars:
                ready = pair.info_time_ns
            self._position = replace(
                self._position,
                completed_hour_bars=count,
                time_ready_info_ns=ready,
            )

        unavailable = (
            not self.trading_enabled
            or not self.indicators_initialized
            or self._position is not None
            or self.has_entry_intent
        )
        previous = self._last_signal
        self._last_signal = current
        if unavailable or previous is None:
            return ()

        trend = self._trend_at(pair.info_time_ns)
        if trend is None:
            self._armed = None
            return ()
        if self._armed is not None and self._armed.direction is not trend:
            self._armed = None

        was_armed = self._armed is not None
        if self._armed is not None and pair.ts_event > self._armed.event_ns:
            if self._is_trigger(self._armed.direction, previous, current):
                distance = (
                    Decimal(str(self.atr.value)) * self.spec.exits.stop_atr_multiple
                )
                if distance > 0:
                    self._pending = _PendingSignal(
                        direction=self._armed.direction,
                        event_ns=pair.ts_event,
                        info_ns=pair.info_time_ns,
                        stop_distance=distance,
                    )
                self._armed = None
                return ()
            seen = self._armed.eligible_bars_seen + 1
            self._armed = (
                None
                if seen >= self.spec.signal.pullback.max_armed_bars
                else replace(self._armed, eligible_bars_seen=seen)
            )

        if (
            not was_armed
            and self._armed is None
            and self._is_arm(trend, previous, current)
        ):
            self._armed = _ArmedSetup(direction=trend, event_ns=pair.ts_event)
        return ()

    def _on_execution_pair(self, pair: CompletedPair) -> tuple[StrategyAction, ...]:
        if not pair.contiguous:
            self.invalidate(Timeframe.MINUTE)
        pending = self._pending
        if pending is not None:
            expiry = pending.info_ns + (
                self.spec.execution.signal_expiry_minutes * _MINUTE_NS
            )
            if pair.info_time_ns > expiry:
                self._pending = None
                self._last_signal = None
            elif pair.info_time_ns > pending.info_ns:
                quantity = self.spec.risk.risk_unit / pending.stop_distance
                intent = EntryOrderIntent(
                    direction=pending.direction,
                    signal_event_ns=pending.event_ns,
                    signal_info_ns=pending.info_ns,
                    execution_event_ns=pair.ts_event,
                    execution_info_ns=pair.info_time_ns,
                    stop_distance=pending.stop_distance,
                    normalized_quantity=quantity,
                )
                self._pending = None
                self._awaiting_fill = intent
                return (intent,)

        position = self._position
        if (
            position is None
            or self._exit_pending
            or pair.info_time_ns <= position.fill_info_ns
        ):
            return ()
        liquidation = pair.bid if position.direction is Direction.LONG else pair.ask
        stop_touched = (
            liquidation.low <= position.stop_price
            if position.direction is Direction.LONG
            else liquidation.high >= position.stop_price
        )
        target_touched = (
            liquidation.high >= position.target_price
            if position.direction is Direction.LONG
            else liquidation.low <= position.target_price
        )
        reason: ExitReason | None = None
        if stop_touched:
            reason = ExitReason.STOP_LOSS
        elif target_touched:
            reason = ExitReason.TAKE_PROFIT
        elif (
            position.time_ready_info_ns is not None
            and pair.info_time_ns > position.time_ready_info_ns
        ):
            reason = ExitReason.TIME
        if reason is None:
            return ()
        self._exit_pending = True
        return (
            ExitOrderIntent(
                direction=position.direction,
                reason=reason,
                decision_event_ns=pair.ts_event,
                decision_info_ns=pair.info_time_ns,
                stop_price=position.stop_price,
                target_price=position.target_price,
            ),
        )

    def _trend_at(self, info_ns: int) -> Direction | None:
        eligible = (point for point in self._trend if point.info_ns <= info_ns)
        point = next(eligible, None)
        for candidate in eligible:
            point = candidate
        return None if point is None else point.direction

    @staticmethod
    def _is_arm(
        direction: Direction, previous: _SignalPoint, current: _SignalPoint
    ) -> bool:
        if direction is Direction.LONG:
            return previous.close > previous.ema and current.close <= current.ema
        return previous.close < previous.ema and current.close >= current.ema

    @staticmethod
    def _is_trigger(
        direction: Direction, previous: _SignalPoint, current: _SignalPoint
    ) -> bool:
        if direction is Direction.LONG:
            return previous.close <= previous.ema and current.close > current.ema
        return previous.close >= previous.ema and current.close < current.ema


@dataclass(slots=True)
class _UnpairedBar:
    bar: Bar
    side: str


class CompletedBarPairer:
    """Fail-closed adapter from native side bars to synchronized pairs."""

    def __init__(self, spec: TrendPullbackSpec) -> None:
        instrument = spec.instrument.instrument_id
        self._types = {
            spec.data_semantics.signal_bid_bar_type: (Timeframe.HOUR, "BID"),
            spec.data_semantics.signal_ask_bar_type: (Timeframe.HOUR, "ASK"),
            spec.data_semantics.trend_bid_bar_type: (Timeframe.FOUR_HOURS, "BID"),
            spec.data_semantics.trend_ask_bar_type: (Timeframe.FOUR_HOURS, "ASK"),
            f"{instrument}-1-MINUTE-BID-EXTERNAL": (Timeframe.MINUTE, "BID"),
            f"{instrument}-1-MINUTE-ASK-EXTERNAL": (Timeframe.MINUTE, "ASK"),
        }
        self._pending: dict[tuple[Timeframe, int], dict[str, Bar]] = {}
        self._last_event: dict[Timeframe, int] = {}

    @property
    def bar_types(self) -> tuple[str, ...]:
        return tuple(self._types)

    def accept(self, bar: Bar) -> CompletedPair | None:
        identity = self._types.get(str(bar.bar_type))
        if identity is None:
            return None
        timeframe, side = identity
        last = self._last_event.get(timeframe)
        if last is not None and bar.ts_event <= last:
            return None
        for key in tuple(self._pending):
            if key[0] is timeframe and key[1] < bar.ts_event:
                del self._pending[key]
        key = (timeframe, bar.ts_event)
        sides = self._pending.setdefault(key, {})
        if side in sides:
            return None
        sides[side] = bar
        if set(sides) != {"BID", "ASK"}:
            return None
        del self._pending[key]
        bid = sides["BID"]
        ask = sides["ASK"]
        contiguous = last is None or bar.ts_event - last == timeframe.interval_ns
        try:
            pair = CompletedPair(
                timeframe=timeframe,
                bid=_price_bar(bid),
                ask=_price_bar(ask),
                ts_event=bar.ts_event,
                info_time_ns=max(bid.ts_init, ask.ts_init),
                contiguous=contiguous,
            )
        except ValueError:
            return None
        self._last_event[timeframe] = bar.ts_event
        return pair


class TrendPullbackStrategy(Strategy):
    """Thin Nautilus strategy adapter for the deterministic G1.2 core."""

    def __new__(
        cls,
        *,
        spec_path: Path = DEFAULT_SPEC_PATH,
        research_ready: bool,
        trading_enabled: bool = False,
    ) -> TrendPullbackStrategy:
        del spec_path, research_ready, trading_enabled
        return super().__new__(cls)

    def __init__(
        self,
        *,
        spec_path: Path = DEFAULT_SPEC_PATH,
        research_ready: bool,
        trading_enabled: bool = False,
    ) -> None:
        super().__init__(
            StrategyConfig(
                strategy_id=StrategyId("TREND-PULLBACK-V1-001"),
                log_events=False,
                log_commands=False,
            )
        )
        spec = load_trend_pullback_spec(spec_path)
        self.state_machine = TrendPullbackStateMachine(
            spec, trading_enabled=trading_enabled
        )
        self._pairer = CompletedBarPairer(spec)
        self._research_ready = research_ready
        self._instrument_id = InstrumentId.from_str(spec.instrument.instrument_id)
        self._instrument: CurrencyPair | None = None
        self._entry_order_id: ClientOrderId | None = None
        self._entry_direction: Direction | None = None

    def on_start(self) -> None:
        if not self._research_ready:
            raise RuntimeError("G1.2 strategy requires a research-ready G0.6 dataset")
        instrument = self.cache.instrument(self._instrument_id)
        if not isinstance(instrument, CurrencyPair):
            raise RuntimeError("EUR/USD CurrencyPair is not available in cache")
        self._instrument = instrument
        for value in self._pairer.bar_types:
            self.subscribe_bars(BarType.from_str(value))

    def on_stop(self) -> None:
        for value in self._pairer.bar_types:
            self.unsubscribe_bars(BarType.from_str(value))

    def on_bar(self, bar: Bar) -> None:
        pair = self._pairer.accept(bar)
        if pair is None:
            return
        for action in self.state_machine.on_pair(pair):
            if isinstance(action, EntryOrderIntent):
                self._submit_entry(action)
            else:
                self.close_all_positions(self._instrument_id)

    def on_order_filled(self, event: OrderFilled) -> None:
        if (
            self._entry_order_id is None
            or event.client_order_id != self._entry_order_id
            or self._entry_direction is None
        ):
            return
        self.state_machine.confirm_entry_fill(
            direction=self._entry_direction,
            fill_price=event.last_px.as_decimal(),
            fill_info_ns=event.ts_event,
        )
        self._entry_order_id = None
        self._entry_direction = None

    def on_order_denied(self, event: OrderDenied) -> None:
        self._reject_matching_entry(event.client_order_id)

    def on_order_rejected(self, event: OrderRejected) -> None:
        self._reject_matching_entry(event.client_order_id)

    def on_order_canceled(self, event: OrderCanceled) -> None:
        self._reject_matching_entry(event.client_order_id)

    def on_order_expired(self, event: OrderExpired) -> None:
        self._reject_matching_entry(event.client_order_id)

    def on_position_closed(self, event: PositionClosed) -> None:
        if event.instrument_id == self._instrument_id:
            self.state_machine.confirm_flat()

    def _submit_entry(self, intent: EntryOrderIntent) -> None:
        if self._instrument is None:
            raise RuntimeError("strategy instrument is not initialized")
        quantity = self._instrument.make_qty(float(intent.normalized_quantity))
        if quantity.as_decimal() <= 0:
            raise RuntimeError("normalized one-R quantity is not executable")
        side = OrderSide.BUY if intent.direction is Direction.LONG else OrderSide.SELL
        order = self.order_factory.market(
            instrument_id=self._instrument_id,
            order_side=side,
            quantity=quantity,
            tags=(
                "trend_pullback_v1",
                f"signal_info_ns={intent.signal_info_ns}",
                f"stop_distance={intent.stop_distance}",
            ),
        )
        self._entry_order_id = order.client_order_id
        self._entry_direction = intent.direction
        self.submit_order(order)

    def _reject_matching_entry(self, client_order_id: ClientOrderId) -> None:
        if self._entry_order_id == client_order_id:
            self._entry_order_id = None
            self._entry_direction = None
            self.state_machine.reject_entry()


def _price_bar(bar: Bar) -> PriceBar:
    return PriceBar(
        open=bar.open.as_decimal(),
        high=bar.high.as_decimal(),
        low=bar.low.as_decimal(),
        close=bar.close.as_decimal(),
    )
