"""B3F2 (Asian-range fade) -- causal session-window definitions, M15
mid-OHLC derivation, Asian-range construction, and decision generation.

Frozen by ``config/research/batch3_methodology_preregistration_v2.json``
(semantic hash
``e5cd74527004585cfd24bea55549d84e9cb66b05ffc6498360dac5007b651f7c``).

DISCOVERED SESSION-DEFINITION CONFLICT (reported, not silently resolved):
this repo already has a canonical intraday session-window definition,
:mod:`ftmoquant.research.g1.sessions` (``ASIAN_SESSION`` =
Asia/Tokyo 09:00-18:00 local; ``LONDON_SESSION`` = Europe/London 08:00-17:00
local), used elsewhere for generic candidate-eligibility filtering. The
B3F2 task brief separately, explicitly froze a DIFFERENT window --
00:00-06:00 Europe/London local as the "Asian range" and 07:00-10:00
Europe/London local as the entry window -- reasoning that a London-local
clock (not a Tokyo-local one) keeps the window aligned with London's own
BST/GMT market-day transition. These are NOT the same window (a
Tokyo-local 09:00-18:00 window does not coincide with a London-local
00:00-06:00 window under either DST regime) and are not economically
interchangeable. Per the task brief's own instruction to report rather
than silently pick a side, this module uses the task's explicitly frozen
London-local numbers (they are given, not invented here) and this
docstring is the flag: ``g1.sessions.ASIAN_SESSION``/``LONDON_SESSION``
were deliberately NOT reused for B3F2's window definition.

M15 IS NOT a canonical derived timeframe anywhere in this repo
(``ftmoquant.research.alpha_lab.data.Timeframe`` is only
``Literal["M30", "H1", "H4"]``; ``ftmoquant.data.derived_bars`` only
derives M30/H1/H4, via a full Nautilus ``BacktestEngine`` catalog-writing
pipeline coupled to EUR/USD). Building a new catalog-writing M15 pipeline
for all 7 instruments would be a disproportionate new data-pipeline stage
for what is, for SIGNAL purposes only, a simple resample. This module
instead derives M15 mid-OHLC in-memory, directly from the same M1 BID/ASK
frames already loaded for execution, using the SAME mid-price convention
already established in
``ftmoquant.research.alpha_lab.data._load_instrument_frame``
(``(bid+ask)/2`` per OHLC field, independently per M1 bar) followed by a
standard OHLC rollup (open=first, high=max, low=min, close=last) over
each non-overlapping, clock-aligned 15-minute window -- a window is kept
only if it contains all 15 of its expected M1 observations (no fill, no
interpolation, matching the repo-wide convention), so signal formation
never quietly runs on a partially-populated bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from decimal import Decimal
from zoneinfo import ZoneInfo

import pandas as pd  # type: ignore[import-untyped]

LONDON_TIMEZONE = "Europe/London"
_LONDON_ZONE = ZoneInfo(LONDON_TIMEZONE)

#: Frozen session windows (section 5), explicitly London-local -- see the
#: module docstring's DISCOVERED SESSION-DEFINITION CONFLICT note.
ASIAN_RANGE_START_LOCAL = time(0, 0)
ASIAN_RANGE_END_LOCAL = time(6, 0)
ENTRY_WINDOW_START_LOCAL = time(7, 0)
ENTRY_WINDOW_END_LOCAL = time(10, 0)
TIME_EXIT_LOCAL = time(16, 0)

EXCURSION_MIN_GRID: tuple[Decimal, ...] = (
    Decimal("0.05"),
    Decimal("0.10"),
    Decimal("0.20"),
)
STOP_BUFFER_GRID: tuple[Decimal, ...] = (Decimal("0.05"), Decimal("0.10"))
TARGET_MODE_MID = "MID"
TARGET_MODE_DEEP_REVERSION = "DEEP_REVERSION"
TARGET_MODE_GRID: tuple[str, ...] = (TARGET_MODE_MID, TARGET_MODE_DEEP_REVERSION)

_M15_MINUTES = 15
_OHLC_COLUMNS = ("open", "high", "low", "close")


class B3F2SignalError(ValueError):
    """Raised on any violation of the frozen B3F2 session/signal contract."""


@dataclass(frozen=True, slots=True)
class B3F2Config:
    """One frozen grid cell (section 10): exactly
    ``len(EXCURSION_MIN_GRID) * len(STOP_BUFFER_GRID) * len(TARGET_MODE_GRID)
    == 12`` of these exist, identical for every pair sleeve."""

    excursion_min: Decimal
    stop_buffer: Decimal
    target_mode: str

    def __post_init__(self) -> None:
        if self.excursion_min not in EXCURSION_MIN_GRID:
            raise B3F2SignalError(f"excursion_min {self.excursion_min} not frozen")
        if self.stop_buffer not in STOP_BUFFER_GRID:
            raise B3F2SignalError(f"stop_buffer {self.stop_buffer} not frozen")
        if self.target_mode not in TARGET_MODE_GRID:
            raise B3F2SignalError(f"target_mode {self.target_mode!r} not frozen")

    @property
    def config_id(self) -> str:
        return f"exc{self.excursion_min}_sb{self.stop_buffer}_{self.target_mode}"


def build_b3f2_grid() -> tuple[B3F2Config, ...]:
    grid = tuple(
        B3F2Config(
            excursion_min=excursion_min,
            stop_buffer=stop_buffer,
            target_mode=target_mode,
        )
        for excursion_min in EXCURSION_MIN_GRID
        for stop_buffer in STOP_BUFFER_GRID
        for target_mode in TARGET_MODE_GRID
    )
    if len(grid) != 12:
        raise B3F2SignalError(f"expected exactly 12 frozen configs, got {len(grid)}")
    return grid


# ---------------------------------------------------------------------------
# M15 mid-OHLC derivation (causal, in-memory, from already-loaded M1)
# ---------------------------------------------------------------------------


def derive_m15_mid_ohlc(bid_m1: pd.DataFrame, ask_m1: pd.DataFrame) -> pd.DataFrame:
    """Derive causal M15 mid-price OHLC bars from paired M1 BID/ASK frames.

    Each M1 bar's mid OHLC is ``(bid + ask) / 2`` per field (the same
    convention already used to build mid H1/H4/M30 datasets in
    ``ftmoquant.research.alpha_lab.data._load_instrument_frame``). M15
    bars are labeled by their CLOSE time (matching this repo's bar
    convention throughout), built by assigning each M1 bar (itself
    close-labeled) to the M15 window ending at ``ts.ceil("15min")``, then
    aggregating open=first, high=max, low=min, close=last. A window is
    emitted only if it contains all 15 of its expected M1 observations --
    an incomplete window (e.g. at a data boundary) is dropped, never
    filled or interpolated.
    """

    if not bid_m1.index.equals(ask_m1.index):
        raise B3F2SignalError("bid_m1 and ask_m1 must share an identical paired index")
    if not bid_m1.index.is_monotonic_increasing:
        raise B3F2SignalError("bid_m1/ask_m1 index must be sorted ascending")

    mid = pd.DataFrame(
        {column: (bid_m1[column] + ask_m1[column]) / 2 for column in _OHLC_COLUMNS},
        index=bid_m1.index,
    )
    bucket = mid.index.ceil(f"{_M15_MINUTES}min")
    grouped = mid.groupby(bucket)
    counts = grouped.size()
    complete_buckets = counts[counts == _M15_MINUTES].index

    m15 = pd.DataFrame(
        {
            "open": grouped["open"].first(),
            "high": grouped["high"].max(),
            "low": grouped["low"].min(),
            "close": grouped["close"].last(),
        }
    )
    m15 = m15.loc[m15.index.isin(complete_buckets)].sort_index()
    return m15


# ---------------------------------------------------------------------------
# Session-window classification (DST-safe, London-local)
# ---------------------------------------------------------------------------


def london_local_date(timestamp: pd.Timestamp) -> date:
    """The London-local calendar date a (UTC) timestamp falls on."""

    result: date = timestamp.tz_convert(LONDON_TIMEZONE).date()
    return result


def _in_local_window(timestamp: pd.Timestamp, start: time, end: time) -> bool:
    local_time = timestamp.tz_convert(LONDON_TIMEZONE).timetz().replace(tzinfo=None)
    return bool(start <= local_time < end)


def is_in_asian_range_window(timestamp: pd.Timestamp) -> bool:
    return _in_local_window(timestamp, ASIAN_RANGE_START_LOCAL, ASIAN_RANGE_END_LOCAL)


def is_in_entry_window(timestamp: pd.Timestamp) -> bool:
    return _in_local_window(timestamp, ENTRY_WINDOW_START_LOCAL, ENTRY_WINDOW_END_LOCAL)


def time_exit_boundary_utc(local_trading_date: date) -> pd.Timestamp:
    """The UTC instant corresponding to 16:00 Europe/London on
    ``local_trading_date`` -- DST-safe (``ZoneInfo`` resolves the correct
    UTC offset for that specific date, whether GMT or BST)."""

    local_dt = datetime.combine(
        local_trading_date, TIME_EXIT_LOCAL, tzinfo=_LONDON_ZONE
    )
    return pd.Timestamp(local_dt).tz_convert(UTC)


# ---------------------------------------------------------------------------
# Asian range construction (frozen once the Asian window closes; never
# updated afterward)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AsianRange:
    local_date: date
    asian_high: float
    asian_low: float

    @property
    def range_width(self) -> float:
        return self.asian_high - self.asian_low


def compute_asian_ranges(m15_mid: pd.DataFrame) -> dict[date, AsianRange]:
    """One :class:`AsianRange` per London-local date that has at least one
    completed M15 bar inside the frozen Asian-range window. A date with
    zero Asian-range bars, or a degenerate (zero-width) range, is simply
    absent from the result -- callers must treat a missing date as
    "no trades that day" (section 6), never substitute a synthetic range.
    """

    if m15_mid.empty:
        return {}
    in_window = m15_mid.index.map(is_in_asian_range_window)
    windowed = m15_mid.loc[in_window]
    if windowed.empty:
        return {}
    local_dates = windowed.index.map(london_local_date)

    ranges: dict[date, AsianRange] = {}
    for local_date in sorted(set(local_dates)):
        day_bars = windowed.loc[local_dates == local_date]
        asian_high = float(day_bars["high"].max())
        asian_low = float(day_bars["low"].min())
        if asian_high - asian_low > 0:
            ranges[local_date] = AsianRange(
                local_date=local_date, asian_high=asian_high, asian_low=asian_low
            )
    return ranges


# ---------------------------------------------------------------------------
# H1-style decision walker (M15-decision cadence; section 7-11)
# ---------------------------------------------------------------------------


EXIT_REASON_STOP = "stop"
EXIT_REASON_TARGET = "target"
EXIT_REASON_TIME = "time_exit"

_LONG = 1
_SHORT = -1


@dataclass(frozen=True, slots=True)
class B3F2TradeIntent:
    """One fully-causal, M1-independent logical trade decision: an M15
    signal bar, a frozen direction/stop/target, and the UTC time-exit
    boundary for that London-local trading date. Execution (finding
    actual M1 fills) is a separate, later step."""

    instrument_id: str
    config: B3F2Config
    local_date: date
    signal_ts: pd.Timestamp
    direction: int
    entry_reference_price: float
    stop_price: float
    target_price: float
    time_exit_boundary_utc: pd.Timestamp

    def __post_init__(self) -> None:
        if self.direction not in (_LONG, _SHORT):
            raise B3F2SignalError("direction must be 1 (long) or -1 (short)")


def _excursion(direction: int, bar: pd.Series, asian_range: AsianRange) -> float:
    if direction == _SHORT:
        return float((bar["high"] - asian_range.asian_high) / asian_range.range_width)
    return float((asian_range.asian_low - bar["low"]) / asian_range.range_width)


def _stop_price(
    direction: int, bar: pd.Series, asian_range: AsianRange, stop_buffer: Decimal
) -> float:
    buffer = float(stop_buffer) * asian_range.range_width
    if direction == _SHORT:
        return float(bar["high"]) + buffer
    return float(bar["low"]) - buffer


def _target_price(direction: int, asian_range: AsianRange, target_mode: str) -> float:
    if target_mode == TARGET_MODE_MID:
        return (asian_range.asian_high + asian_range.asian_low) / 2
    quarter = 0.25 * asian_range.range_width
    if direction == _SHORT:
        return asian_range.asian_low + quarter
    return asian_range.asian_high - quarter


def generate_b3f2_decisions(
    m15_mid: pd.DataFrame,
    asian_ranges: dict[date, AsianRange],
    *,
    instrument_id: str,
    config: B3F2Config,
) -> tuple[B3F2TradeIntent, ...]:
    """Walk ``m15_mid`` bar-by-bar (chronological, M15 decision cadence
    only) and emit the causal entry decision sequence for one config.

    At most one long-entry attempt and one short-entry attempt per
    London-local date (section 11): once a direction has triggered that
    day, no second same-direction signal is taken that day, even if the
    first trade already stopped out. A bar satisfying BOTH the long and
    short excursion conditions simultaneously is skipped entirely for both
    directions (section 7) -- no directional tie-break is invented.

    "One position at a time" (no pyramiding) is deliberately NOT enforced
    here: whether an earlier same-day intent is still open at a later
    intent's entry time depends on real M1 stop/target/time-exit outcomes,
    which this purely-causal-from-M15 signal layer has no way to know.
    Every qualifying intent (subject only to the one-attempt-per-direction-
    per-day rule above) is emitted; the execution layer enforces the
    actual no-pyramiding/busy-until-exit rule from real fills, exactly
    mirroring
    :func:`ftmoquant.research.alpha_lab.b3f1_spread_execution.simulate_b3f1_intents`'s
    ``busy_until_ns`` pattern.
    """

    if m15_mid.empty:
        return ()

    intents: list[B3F2TradeIntent] = []
    used_direction_by_date: dict[date, set[int]] = {}

    for ts, bar in m15_mid.iterrows():
        if not is_in_entry_window(ts):
            continue
        local_date = london_local_date(ts)
        asian_range = asian_ranges.get(local_date)
        if asian_range is None:
            continue

        short_probe = (
            bar["high"] > asian_range.asian_high
            and bar["close"] < asian_range.asian_high
        )
        long_probe = (
            bar["low"] < asian_range.asian_low and bar["close"] > asian_range.asian_low
        )
        if short_probe and long_probe:
            continue
        if not short_probe and not long_probe:
            continue

        direction = _SHORT if short_probe else _LONG
        used = used_direction_by_date.setdefault(local_date, set())
        if direction in used:
            continue

        excursion = _excursion(direction, bar, asian_range)
        if excursion < float(config.excursion_min):
            continue

        stop_price = _stop_price(direction, bar, asian_range, config.stop_buffer)
        target_price = _target_price(direction, asian_range, config.target_mode)

        intents.append(
            B3F2TradeIntent(
                instrument_id=instrument_id,
                config=config,
                local_date=local_date,
                signal_ts=ts,
                direction=direction,
                entry_reference_price=float(bar["close"]),
                stop_price=stop_price,
                target_price=target_price,
                time_exit_boundary_utc=time_exit_boundary_utc(local_date),
            )
        )
        used.add(direction)

    return tuple(intents)
