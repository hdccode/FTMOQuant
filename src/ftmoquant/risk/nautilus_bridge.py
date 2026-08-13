"""Reusable native-clock bridge from Nautilus strategies to the FTMO overlay."""

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import cast

from nautilus_trader.common import Cache, Clock, TimeEvent
from nautilus_trader.model import (
    Bar,
    Currency,
    InstrumentId,
    OrderFilled,
    Position,
    PositionOpened,
    Price,
)

from ftmoquant.risk.ftmo_overlay import (
    NativeAccountSnapshot,
    NautilusAccountSnapshotSource,
    NautilusFtmoOverlay,
)

FTMO_OBSERVATION_VERSION = "g0.9-2"
FTMO_VALUATION_VERSION = "g0.9-liquidation-close-1"
FTMO_OBSERVATION_DELAY_NS = 1


class FtmoObservationTrigger(StrEnum):
    """Native event which caused an observational compliance refresh."""

    ORDER_FILL = "order_fill"
    POSITION_OPENED = "position_opened"
    PAIRED_BAR_POST_SETTLEMENT = "paired_bar_post_settlement"


@dataclass(frozen=True, slots=True)
class FtmoObservation:
    """Immutable reconciliation evidence read from native Nautilus state."""

    trigger: FtmoObservationTrigger
    timestamp_ns: int
    balance: str
    equity: str
    native_portfolio_equity: str
    native_minus_ftmo_equity: str
    open_position_ids: tuple[str, ...]
    completed_mark_timestamp_ns: int | None
    bid_close: str | None
    ask_close: str | None


@dataclass(frozen=True, slots=True)
class LiquidationValuation:
    """Ephemeral native-position valuation diagnostics for one snapshot."""

    snapshot: NativeAccountSnapshot
    native_portfolio_equity: Decimal
    completed_mark_timestamp_ns: int | None
    bid_close: Price | None
    ask_close: Price | None


class PairedBarLiquidationSnapshotSource:
    """Mark EUR/USD positions at liquidation-side completed bar closes.

    Native account balance contains fees, realized P/L, and rollover. Open
    position P/L is calculated ephemerally by native ``Position.unrealized_pnl``.
    No valuation is stored or rolled forward between snapshots.
    """

    def __init__(
        self,
        *,
        cache: Cache,
        native_source: NautilusAccountSnapshotSource,
        instrument_id: InstrumentId,
        account_currency: Currency,
    ) -> None:
        if instrument_id.value != "EUR/USD.DUKASCOPY":
            raise ValueError("G0.9 liquidation valuation supports EUR/USD only")
        if account_currency.code != "USD":
            raise ValueError("G0.9 liquidation valuation requires a USD account")
        self._cache = cache
        self._native_source = native_source
        self._instrument_id = instrument_id
        self._account_currency = account_currency
        self._mark_timestamp_ns: int | None = None
        self._bid_close: Price | None = None
        self._ask_close: Price | None = None

    @property
    def currency(self) -> str:
        """Return the supported native account currency."""

        return self._native_source.currency

    def set_completed_pair(
        self,
        *,
        timestamp_ns: int,
        bid_close: Price,
        ask_close: Price,
    ) -> None:
        """Replace marks atomically only when a complete pair is available."""

        if (
            self._mark_timestamp_ns is not None
            and timestamp_ns <= self._mark_timestamp_ns
        ):
            raise RuntimeError("completed BID/ASK marks must be strictly chronological")
        if ask_close.raw < bid_close.raw:
            raise RuntimeError("completed ASK close cannot be below BID close")
        self._mark_timestamp_ns = timestamp_ns
        self._bid_close = bid_close
        self._ask_close = ask_close

    def snapshot(self) -> NativeAccountSnapshot:
        """Return authoritative FTMO equity using native position valuation."""

        return self.valuation().snapshot

    def valuation(self) -> LiquidationValuation:
        """Return authoritative and native fallback equity diagnostics."""

        native = self._native_source.snapshot()
        positions = cast(
            list[Position],
            self._cache.positions_open(instrument_id=self._instrument_id),
        )
        if not positions:
            return LiquidationValuation(
                snapshot=NativeAccountSnapshot(
                    balance=native.balance,
                    equity=native.balance,
                    open_position_ids=native.open_position_ids,
                ),
                native_portfolio_equity=native.equity,
                completed_mark_timestamp_ns=self._mark_timestamp_ns,
                bid_close=self._bid_close,
                ask_close=self._ask_close,
            )
        if self._bid_close is None or self._ask_close is None:
            raise RuntimeError(
                "open EUR/USD positions require a completed BID/ASK mark pair"
            )
        equity = native.balance
        for position in positions:
            if position.settlement_currency != self._account_currency:
                raise RuntimeError(
                    "position settlement currency must match the USD account"
                )
            mark = self._bid_close if position.is_long else self._ask_close
            unrealized = position.unrealized_pnl(mark)
            if unrealized.currency != self._account_currency:
                raise RuntimeError("native unrealized P/L currency is unsupported")
            equity += unrealized.as_decimal()
        return LiquidationValuation(
            snapshot=NativeAccountSnapshot(
                balance=native.balance,
                equity=equity,
                open_position_ids=native.open_position_ids,
            ),
            native_portfolio_equity=native.equity,
            completed_mark_timestamp_ns=self._mark_timestamp_ns,
            bid_close=self._bid_close,
            ask_close=self._ask_close,
        )


class NautilusFtmoBridge:
    """Compose a strategy's native callbacks with ``NautilusFtmoOverlay``.

    Fill and position callbacks are observed synchronously because rc2 has
    already applied their native account/portfolio state before dispatching to
    a strategy. External BID/ASK bars are observed only after both sides have
    arrived, using a one-nanosecond native-clock alert. In rc2 this callback is
    after timestamp finalization and venue modules, including FX rollover.
    """

    def __init__(
        self,
        *,
        clock: Clock,
        overlay: NautilusFtmoOverlay,
        account_source: PairedBarLiquidationSnapshotSource,
        timer_prefix: str = "ftmo-post-settlement",
    ) -> None:
        self._clock = clock
        self._overlay = overlay
        self._account_source = account_source
        self._timer_prefix = timer_prefix
        self._bar_closes: dict[int, dict[str, Price]] = {}
        self._scheduled: set[int] = set()
        self._observations: list[FtmoObservation] = []

    @property
    def overlay(self) -> NautilusFtmoOverlay:
        """Return the composed overlay."""

        return self._overlay

    @property
    def observations(self) -> tuple[FtmoObservation, ...]:
        """Return immutable native-state reconciliation observations."""

        return tuple(self._observations)

    def snapshot(self) -> NativeAccountSnapshot:
        """Return the current authoritative native account snapshot."""

        return self._account_source.snapshot()

    def valuation(self) -> LiquidationValuation:
        """Return current authoritative and native fallback diagnostics."""

        return self._account_source.valuation()

    def on_bar(self, bar: Bar) -> None:
        """Schedule one post-settlement refresh after a complete BID/ASK pair."""

        side = bar.bar_type.spec.price_type.name
        if side not in {"BID", "ASK"}:
            return
        timestamp_ns = bar.ts_init
        closes = self._bar_closes.setdefault(timestamp_ns, {})
        closes[side] = bar.close
        if set(closes) != {"BID", "ASK"} or timestamp_ns in self._scheduled:
            return
        self._account_source.set_completed_pair(
            timestamp_ns=timestamp_ns,
            bid_close=closes["BID"],
            ask_close=closes["ASK"],
        )
        self._scheduled.add(timestamp_ns)
        del self._bar_closes[timestamp_ns]
        observation_ns = timestamp_ns + FTMO_OBSERVATION_DELAY_NS
        self._clock.set_time_alert_ns(
            name=f"{self._timer_prefix}-{timestamp_ns}",
            alert_time_ns=observation_ns,
            callback=self.on_time_event,
        )

    def on_order_filled(self, event: OrderFilled) -> None:
        """Observe a native fill after account commission application."""

        self._overlay.on_order_filled(event)
        self._record(FtmoObservationTrigger.ORDER_FILL, event.ts_event)

    def on_position_opened(self, event: PositionOpened) -> None:
        """Count the native position-open day and capture native state."""

        self._overlay.on_position_opened(event)
        self._record(FtmoObservationTrigger.POSITION_OPENED, event.ts_event)

    def on_time_event(self, event: TimeEvent) -> None:
        """Observe completed paired-bar and module settlement."""

        if not event.name.startswith(f"{self._timer_prefix}-"):
            return
        source_timestamp = event.ts_event - FTMO_OBSERVATION_DELAY_NS
        self._scheduled.discard(source_timestamp)
        self._overlay.refresh(event.ts_event)
        self._record(
            FtmoObservationTrigger.PAIRED_BAR_POST_SETTLEMENT,
            event.ts_event,
        )

    def _record(self, trigger: FtmoObservationTrigger, timestamp_ns: int) -> None:
        valuation = self._account_source.valuation()
        snapshot = valuation.snapshot
        self._observations.append(
            FtmoObservation(
                trigger=trigger,
                timestamp_ns=timestamp_ns,
                balance=str(snapshot.balance),
                equity=str(snapshot.equity),
                native_portfolio_equity=str(valuation.native_portfolio_equity),
                native_minus_ftmo_equity=str(
                    valuation.native_portfolio_equity - snapshot.equity
                ),
                open_position_ids=tuple(sorted(snapshot.open_position_ids)),
                completed_mark_timestamp_ns=(
                    valuation.completed_mark_timestamp_ns
                ),
                bid_close=(
                    None if valuation.bid_close is None else str(valuation.bid_close)
                ),
                ask_close=(
                    None if valuation.ask_close is None else str(valuation.ask_close)
                ),
            )
        )
