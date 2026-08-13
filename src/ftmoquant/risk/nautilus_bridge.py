"""Reusable native-clock bridge from Nautilus strategies to the FTMO overlay."""

from dataclasses import dataclass
from enum import StrEnum

from nautilus_trader.common import Clock, TimeEvent
from nautilus_trader.model import Bar, OrderFilled, PositionOpened

from ftmoquant.risk.ftmo_overlay import (
    NativeAccountSnapshot,
    NautilusAccountSnapshotSource,
    NautilusFtmoOverlay,
)

FTMO_OBSERVATION_VERSION = "g0.9-1"
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
    open_position_ids: tuple[str, ...]


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
        account_source: NautilusAccountSnapshotSource,
        timer_prefix: str = "ftmo-post-settlement",
    ) -> None:
        self._clock = clock
        self._overlay = overlay
        self._account_source = account_source
        self._timer_prefix = timer_prefix
        self._bar_sides: dict[int, set[str]] = {}
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

    def on_bar(self, bar: Bar) -> None:
        """Schedule one post-settlement refresh after a complete BID/ASK pair."""

        side = bar.bar_type.spec.price_type.name
        if side not in {"BID", "ASK"}:
            return
        timestamp_ns = bar.ts_init
        sides = self._bar_sides.setdefault(timestamp_ns, set())
        sides.add(side)
        if sides != {"BID", "ASK"} or timestamp_ns in self._scheduled:
            return
        self._scheduled.add(timestamp_ns)
        del self._bar_sides[timestamp_ns]
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
        snapshot: NativeAccountSnapshot = self._account_source.snapshot()
        self._observations.append(
            FtmoObservation(
                trigger=trigger,
                timestamp_ns=timestamp_ns,
                balance=str(snapshot.balance),
                equity=str(snapshot.equity),
                open_position_ids=tuple(sorted(snapshot.open_position_ids)),
            )
        )
