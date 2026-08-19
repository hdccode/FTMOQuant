"""M1 BID/ASK trade-lifecycle execution engine for the frozen F1/F2/F3
DEVELOPMENT screen.

Reuses the same conventions already proven elsewhere in the repo rather
than inventing new ones:

- "first strictly later paired M1 observation" -- the same fail-closed
  ``bisect``-based semantics as
  ``ftmoquant.research.mean_reversion_h1_development._first_strictly_later_paired_ns``,
  applied here to plain integer nanoseconds rather than ``Bar`` objects.
- direction-aware BID/ASK crossing (buy -> ASK, sell -> BID; long
  liquidation -> BID, short liquidation -> ASK) and stop-before-target
  same-bar collision handling -- the same convention as
  ``ftmoquant.strategies.trend_pullback.TrendPullbackStateMachine._on_execution_pair``.
- no-fill/no-interpolation BID/ASK alignment -- the same inner-join policy
  as :func:`ftmoquant.research.alpha_lab.data.assemble_aligned_dataset`,
  applied at M1 granularity to the raw (non-aggregated) catalog bars
  instead of the pre-materialized H1/H4/M30 ones.

Not a general backtester: one instrument, one configuration, one causal
signal stream in, one deterministic trade list out. No pyramiding, no
partial fills.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
from nautilus_trader.persistence import ParquetDataCatalog

from ftmoquant.research.alpha_lab.wick_fvg_squeeze_signals import SignalEvent

EXIT_REASON_STOP = "stop"
EXIT_REASON_TARGET = "target"


class ExecutionError(ValueError):
    """Raised on missing/malformed M1 data. Never silently substitutes a
    default price or interpolates a missing observation."""


def _ns(value: datetime) -> int:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ExecutionError("timestamp must be timezone-aware UTC")
    return int(value.timestamp() * 1_000_000_000)


def load_m1_bidask(
    *,
    instrument_id: str,
    root: Path,
    start_utc: datetime,
    end_exclusive_utc: datetime,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw (non-aggregated) M1 BID and ASK OHLC frames, restricted to the
    UTC timestamps where *both* sides have a native catalog bar (inner
    join, no fill, no interpolation) -- the sole execution stream this
    module ever reads from."""

    catalog = ParquetDataCatalog(str(root / "catalog"))
    start_ns = _ns(start_utc)
    end_ns = _ns(end_exclusive_utc)
    frames: dict[str, pd.DataFrame] = {}
    for side in ("BID", "ASK"):
        bar_type = f"{instrument_id}-1-MINUTE-{side}-EXTERNAL"
        queried = catalog.query_bars([bar_type], start=start_ns, end=end_ns - 1)
        rows = [
            {
                "timestamp": pd.Timestamp(bar.ts_event, unit="ns", tz=UTC),
                "open": float(bar.open.as_decimal()),
                "high": float(bar.high.as_decimal()),
                "low": float(bar.low.as_decimal()),
                "close": float(bar.close.as_decimal()),
            }
            for bar in queried
            if start_ns <= bar.ts_event < end_ns and str(bar.bar_type) == bar_type
        ]
        if not rows:
            raise ExecutionError(f"DEVELOPMENT has no {bar_type} M1 data")
        frames[side] = pd.DataFrame(rows).set_index("timestamp").sort_index()

    shared = frames["BID"].index.intersection(frames["ASK"].index).sort_values()
    if shared.empty:
        raise ExecutionError(
            f"DEVELOPMENT M1 BID/ASK bars never overlap for {instrument_id}"
        )
    return frames["BID"].loc[shared], frames["ASK"].loc[shared]


@dataclass(frozen=True, slots=True)
class Trade:
    signal_bar_ts: pd.Timestamp
    direction: int
    entry_ts: pd.Timestamp
    entry_price: float
    exit_ts: pd.Timestamp
    exit_price: float
    exit_reason: str
    stop_price: float
    target_price: float
    return_frac: float


@dataclass(frozen=True, slots=True)
class SkipRecord:
    signal_bar_ts: pd.Timestamp
    reason: str


def _resolve_stop_target(event: SignalEvent, entry_price: float) -> tuple[float, float]:
    stop_price = (
        event.stop_price
        if event.stop_price is not None
        else entry_price - event.direction * event.stop_distance  # type: ignore[operator]
    )
    if event.target_price is not None:
        target_price = event.target_price
    elif event.target_distance is not None:
        target_price = entry_price + event.direction * event.target_distance
    else:
        assert event.target_r_multiple is not None
        risk = abs(entry_price - stop_price)
        target_price = entry_price + event.direction * event.target_r_multiple * risk
    return stop_price, target_price


def simulate_trades(
    events: Sequence[SignalEvent], bid_m1: pd.DataFrame, ask_m1: pd.DataFrame
) -> tuple[tuple[Trade, ...], tuple[SkipRecord, ...]]:
    """Walk a chronological signal stream against the paired M1 BID/ASK
    stream: enter at the first paired M1 observation strictly after each
    signal bar (skipped if none exists), then walk subsequent paired M1
    bars evaluating the executable (liquidation-side) high/low against the
    frozen stop/target until one is touched -- stop wins on a same-bar
    collision. One position at a time per instrument: a signal that arrives
    while a trade is still open is silently skipped (not queued), matching
    "no pyramiding". Never fabricates an exit: a trade still open when the
    M1 stream ends is recorded as a skip, not closed at an invented price.
    """

    if not bid_m1.index.equals(ask_m1.index):
        raise ExecutionError("bid_m1 and ask_m1 must share the identical paired index")
    paired_index = bid_m1.index
    paired_ns = [int(ts.value) for ts in paired_index]

    ordered_events = sorted(events, key=lambda event: event.signal_bar_ts)
    trades: list[Trade] = []
    skips: list[SkipRecord] = []
    busy_until_ns: int | None = None

    for event in ordered_events:
        signal_ns = int(event.signal_bar_ts.value)
        if busy_until_ns is not None and signal_ns <= busy_until_ns:
            skips.append(SkipRecord(event.signal_bar_ts, "signal_during_open_trade"))
            continue

        entry_pos = bisect.bisect_right(paired_ns, signal_ns)
        if entry_pos >= len(paired_ns):
            skips.append(SkipRecord(event.signal_bar_ts, "no_later_m1_observation"))
            continue
        entry_ts = paired_index[entry_pos]
        entry_price = (
            float(ask_m1["close"].iloc[entry_pos])
            if event.direction == 1
            else float(bid_m1["close"].iloc[entry_pos])
        )
        stop_price, target_price = _resolve_stop_target(event, entry_price)

        exit_found = False
        for i in range(entry_pos + 1, len(paired_index)):
            liquidation = bid_m1.iloc[i] if event.direction == 1 else ask_m1.iloc[i]
            stop_touched = (
                liquidation["low"] <= stop_price
                if event.direction == 1
                else liquidation["high"] >= stop_price
            )
            target_touched = (
                liquidation["high"] >= target_price
                if event.direction == 1
                else liquidation["low"] <= target_price
            )
            if not (stop_touched or target_touched):
                continue
            if stop_touched:
                exit_reason, exit_price = EXIT_REASON_STOP, stop_price
            else:
                exit_reason, exit_price = EXIT_REASON_TARGET, target_price
            exit_ts = paired_index[i]
            return_frac = event.direction * (exit_price - entry_price) / entry_price
            trades.append(
                Trade(
                    signal_bar_ts=event.signal_bar_ts,
                    direction=event.direction,
                    entry_ts=entry_ts,
                    entry_price=entry_price,
                    exit_ts=exit_ts,
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    stop_price=stop_price,
                    target_price=target_price,
                    return_frac=return_frac,
                )
            )
            busy_until_ns = int(exit_ts.value)
            exit_found = True
            break
        if not exit_found:
            skips.append(SkipRecord(event.signal_bar_ts, "no_m1_exit_before_data_end"))
            busy_until_ns = paired_ns[-1]

    return tuple(trades), tuple(skips)
