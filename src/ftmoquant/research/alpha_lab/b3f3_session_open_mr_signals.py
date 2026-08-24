"""B3F3 (session-open microstructure mean reversion) -- causal session
clock, M5 mid-OHLC derivation, opening anchor, lagged ATR, and decision
generation.

Frozen by ``config/research/batch3_methodology_preregistration_v2.json``
(semantic hash
``e5cd74527004585cfd24bea55549d84e9cb66b05ffc6498360dac5007b651f7c``).

Reuses, rather than re-derives:

- The True Range formula already established in
  :func:`ftmoquant.research.alpha_lab.wick_fvg_squeeze_signals._true_range`
  (``max(high-low, |high-prev_close|, |low-prev_close|)``) -- imported
  directly rather than reimplemented. That module's own ``_atr`` is NOT
  reused as-is: it includes the CURRENT bar's own true range (correct for
  its callers, who only use it to size a stop/target after already
  deciding to enter on other grounds), whereas B3F3 requires an ATR that
  is strictly lagged -- available BEFORE the decision bar, since it
  directly drives the entry decision itself (section 7). This module adds
  exactly one ``.shift(1)`` on top of the reused ``_true_range``/rolling-
  mean formula to get that lagged series -- not a new ATR formula.
- The same London-local M15-style M1->M5 in-memory derivation approach
  already built for B3F2
  (:func:`ftmoquant.research.alpha_lab.b3f2_asian_range_fade_signals.
  derive_m15_mid_ohlc`), generalized to a configurable bucket size (M5
  here) rather than duplicated. M5, like M15, is not a canonical derived
  timeframe anywhere in this repo.
- ``ftmoquant.research.g1.sessions``'s Europe/London ``ZoneInfo`` handling
  pattern (DST-safe local-time comparison) -- B3F3's OWN session-open
  time (08:00 Europe/London) is a value explicitly frozen by this task's
  brief, not sourced from ``g1.sessions`` (whose ``LONDON_SESSION`` starts
  at 08:00 too, so there is no conflict here unlike B3F2's Asian-range
  window).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from decimal import Decimal
from zoneinfo import ZoneInfo

import pandas as pd  # type: ignore[import-untyped]

from ftmoquant.research.alpha_lab.wick_fvg_squeeze_signals import _true_range

LONDON_TIMEZONE = "Europe/London"
_LONDON_ZONE = ZoneInfo(LONDON_TIMEZONE)

#: Frozen session clock (section 5).
SESSION_OPEN_LOCAL = time(8, 0)
ENTRY_WINDOW_START_LOCAL = time(8, 5)
ENTRY_WINDOW_END_LOCAL = time(9, 0)
TIME_EXIT_LOCAL = time(11, 0)

ATR_WINDOW = 12
DISPLACEMENT_THRESHOLD_GRID: tuple[Decimal, ...] = (
    Decimal("1.0"),
    Decimal("1.5"),
    Decimal("2.0"),
)
STOP_ATR_GRID: tuple[Decimal, ...] = (Decimal("0.75"), Decimal("1.0"))
TARGET_MODE_OPEN_ANCHOR = "OPEN_ANCHOR"
TARGET_MODE_HALF_REVERSION = "HALF_REVERSION"
TARGET_MODE_GRID: tuple[str, ...] = (
    TARGET_MODE_OPEN_ANCHOR,
    TARGET_MODE_HALF_REVERSION,
)

_M5_MINUTES = 5
_OHLC_COLUMNS = ("open", "high", "low", "close")

_LONG = 1
_SHORT = -1

EXIT_REASON_STOP = "stop"
EXIT_REASON_TARGET = "target"
EXIT_REASON_TIME = "time_exit"


class B3F3SignalError(ValueError):
    """Raised on any violation of the frozen B3F3 session/signal contract."""


@dataclass(frozen=True, slots=True)
class B3F3Config:
    """One frozen grid cell (section 10): exactly
    ``len(DISPLACEMENT_THRESHOLD_GRID) * len(STOP_ATR_GRID) *
    len(TARGET_MODE_GRID) == 12`` of these exist, identical for every
    pair sleeve."""

    displacement_threshold: Decimal
    stop_atr: Decimal
    target_mode: str

    def __post_init__(self) -> None:
        if self.displacement_threshold not in DISPLACEMENT_THRESHOLD_GRID:
            raise B3F3SignalError(
                f"displacement_threshold {self.displacement_threshold} not frozen"
            )
        if self.stop_atr not in STOP_ATR_GRID:
            raise B3F3SignalError(f"stop_atr {self.stop_atr} not frozen")
        if self.target_mode not in TARGET_MODE_GRID:
            raise B3F3SignalError(f"target_mode {self.target_mode!r} not frozen")

    @property
    def config_id(self) -> str:
        return (
            f"disp{self.displacement_threshold}_satr{self.stop_atr}_{self.target_mode}"
        )


def build_b3f3_grid() -> tuple[B3F3Config, ...]:
    grid = tuple(
        B3F3Config(displacement_threshold=t, stop_atr=s, target_mode=m)
        for t in DISPLACEMENT_THRESHOLD_GRID
        for s in STOP_ATR_GRID
        for m in TARGET_MODE_GRID
    )
    if len(grid) != 12:
        raise B3F3SignalError(f"expected exactly 12 frozen configs, got {len(grid)}")
    return grid


# ---------------------------------------------------------------------------
# M5 mid-OHLC derivation (causal, in-memory, from already-loaded M1)
# ---------------------------------------------------------------------------


def derive_m5_mid_ohlc(bid_m1: pd.DataFrame, ask_m1: pd.DataFrame) -> pd.DataFrame:
    """Derive causal M5 mid-price OHLC bars from paired M1 BID/ASK frames.

    Identical approach to
    :func:`ftmoquant.research.alpha_lab.b3f2_asian_range_fade_signals.
    derive_m15_mid_ohlc`, generalized to a 5-minute bucket: each M1 bar's
    mid OHLC is ``(bid + ask) / 2`` per field, M5 bars are labeled by
    close time (matching this repo's bar convention), built by assigning
    each M1 bar to the M5 window ending at ``ts.ceil("5min")``, then
    aggregating open=first, high=max, low=min, close=last. A window is
    emitted only if it contains all 5 of its expected M1 observations --
    no fill, no interpolation.
    """

    if not bid_m1.index.equals(ask_m1.index):
        raise B3F3SignalError("bid_m1 and ask_m1 must share an identical paired index")
    if not bid_m1.index.is_monotonic_increasing:
        raise B3F3SignalError("bid_m1/ask_m1 index must be sorted ascending")

    mid = pd.DataFrame(
        {column: (bid_m1[column] + ask_m1[column]) / 2 for column in _OHLC_COLUMNS},
        index=bid_m1.index,
    )
    bucket = mid.index.ceil(f"{_M5_MINUTES}min")
    grouped = mid.groupby(bucket)
    counts = grouped.size()
    complete_buckets = counts[counts == _M5_MINUTES].index

    m5 = pd.DataFrame(
        {
            "open": grouped["open"].first(),
            "high": grouped["high"].max(),
            "low": grouped["low"].min(),
            "close": grouped["close"].last(),
        }
    )
    m5 = m5.loc[m5.index.isin(complete_buckets)].sort_index()
    return m5


# ---------------------------------------------------------------------------
# Session-window classification (DST-safe, London-local)
# ---------------------------------------------------------------------------


def london_local_date(timestamp: pd.Timestamp) -> date:
    result: date = timestamp.tz_convert(LONDON_TIMEZONE).date()
    return result


def _local_time(timestamp: pd.Timestamp) -> time:
    result: time = timestamp.tz_convert(LONDON_TIMEZONE).timetz().replace(tzinfo=None)
    return result


#: The close-time label of the completed opening bar (08:00-08:05).
_OPENING_BAR_CLOSE_LOCAL = time(8, 5)


def is_session_open_bar(timestamp: pd.Timestamp) -> bool:
    """True for the bar whose CLOSE label is exactly 08:05 London-local --
    i.e. the completed 08:00-08:05 opening bar."""

    return _local_time(timestamp) == _OPENING_BAR_CLOSE_LOCAL


def is_in_entry_window(timestamp: pd.Timestamp) -> bool:
    """08:05 through 09:00 London-local, inclusive of both endpoints
    (section 5). The 08:05 bar itself is harmless to include: it IS the
    opening anchor bar, so its own displacement is always exactly zero and
    can never qualify against any frozen threshold (>= 1.0)."""

    local_time = _local_time(timestamp)
    return bool(ENTRY_WINDOW_START_LOCAL <= local_time <= ENTRY_WINDOW_END_LOCAL)


def time_exit_boundary_utc(local_trading_date: date) -> pd.Timestamp:
    """The UTC instant corresponding to 11:00 Europe/London on
    ``local_trading_date`` -- DST-safe."""

    local_dt = datetime.combine(
        local_trading_date, TIME_EXIT_LOCAL, tzinfo=_LONDON_ZONE
    )
    return pd.Timestamp(local_dt).tz_convert(UTC)


# ---------------------------------------------------------------------------
# Opening anchor + lagged ATR (section 6-7)
# ---------------------------------------------------------------------------


def compute_opening_anchors(m5_mid: pd.DataFrame) -> dict[date, float]:
    """One opening-anchor price per London-local date: the CLOSE of that
    date's own completed 08:00-08:05 M5 bar. A date with no such bar is
    simply absent -- callers must treat a missing date as "no signal that
    day" (section 5), never substitute a synthetic anchor."""

    if m5_mid.empty:
        return {}
    is_open_bar = m5_mid.index.map(is_session_open_bar)
    open_bars = m5_mid.loc[is_open_bar]
    anchors: dict[date, float] = {}
    for ts, row in open_bars.iterrows():
        local_date = london_local_date(ts)
        # A London-local date has exactly one 08:00-08:05 window; if data
        # ever produced more than one (it cannot, by construction of the
        # M5 bucketing), the first-seen (chronologically earliest) wins,
        # never a later one.
        anchors.setdefault(local_date, float(row["close"]))
    return anchors


def compute_lagged_atr(m5_mid: pd.DataFrame, window: int = ATR_WINDOW) -> pd.Series:
    """ATR available AT each bar's own decision time -- i.e. computed from
    the ``window`` completed True Range values strictly BEFORE that bar,
    via the reused :func:`ftmoquant.research.alpha_lab.
    wick_fvg_squeeze_signals._true_range` formula plus a single
    ``.shift(1)`` (see module docstring for why the existing, unlagged
    ``_atr`` helper elsewhere in this repo is not reused directly here)."""

    true_range = _true_range(m5_mid["high"], m5_mid["low"], m5_mid["close"])
    return true_range.rolling(window).mean().shift(1)


# ---------------------------------------------------------------------------
# H1-style decision walker (M5-decision cadence; sections 8-10)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class B3F3TradeIntent:
    """One fully-causal, M1-independent logical trade decision: an M5
    signal bar, a frozen direction/stop/target, and the UTC time-exit
    boundary for that London-local trading date."""

    instrument_id: str
    config: B3F3Config
    local_date: date
    signal_ts: pd.Timestamp
    direction: int
    entry_reference_price: float
    displacement_at_entry: float
    stop_price: float
    target_price: float
    time_exit_boundary_utc: pd.Timestamp

    def __post_init__(self) -> None:
        if self.direction not in (_LONG, _SHORT):
            raise B3F3SignalError("direction must be 1 (long) or -1 (short)")


def _stop_price(
    direction: int, entry_price: float, atr_signal: float, stop_atr: Decimal
) -> float:
    distance = float(stop_atr) * atr_signal
    if direction == _SHORT:
        return entry_price + distance
    return entry_price - distance


def _target_price(entry_price: float, opening_price: float, target_mode: str) -> float:
    """Section 10: for a short, ``target = entry - 0.5*(entry - opening)``;
    for a long, ``target = entry + 0.5*(opening - entry)``. Both reduce to
    the identical expression ``entry - 0.5*(entry - opening)`` -- half the
    distance back toward the opening anchor -- so direction does not enter
    this formula at all."""

    if target_mode == TARGET_MODE_OPEN_ANCHOR:
        return opening_price
    half = 0.5 * (entry_price - opening_price)
    return entry_price - half


def generate_b3f3_decisions(
    m5_mid: pd.DataFrame,
    opening_anchors: dict[date, float],
    lagged_atr: pd.Series,
    *,
    instrument_id: str,
    config: B3F3Config,
) -> tuple[B3F3TradeIntent, ...]:
    """Walk ``m5_mid`` bar-by-bar (chronological, M5 decision cadence
    only) and emit the causal entry decision sequence for one config.

    At most ONE entry attempt per London-local date, in EITHER direction
    (section 8): once any qualifying entry has fired that day, no second
    entry is taken that day regardless of the first trade's eventual
    stop/target/time-exit outcome. "One position at a time" (no
    pyramiding) is enforced at the execution layer, not here -- identical
    reasoning to
    :func:`ftmoquant.research.alpha_lab.b3f2_asian_range_fade_signals.
    generate_b3f2_decisions` (this purely-causal-from-M5 layer cannot know
    real M1 exit timing).
    """

    if m5_mid.empty:
        return ()
    if not m5_mid.index.equals(lagged_atr.index):
        raise B3F3SignalError("m5_mid and lagged_atr must share an identical index")

    intents: list[B3F3TradeIntent] = []
    used_dates: set[date] = set()
    threshold = float(config.displacement_threshold)

    for ts, bar in m5_mid.iterrows():
        if not is_in_entry_window(ts):
            continue
        local_date = london_local_date(ts)
        if local_date in used_dates:
            continue
        opening_price = opening_anchors.get(local_date)
        if opening_price is None:
            continue
        atr_signal = lagged_atr.loc[ts]
        if pd.isna(atr_signal) or atr_signal <= 0:
            continue

        close = float(bar["close"])
        displacement = (close - opening_price) / float(atr_signal)

        if displacement >= threshold:
            direction = _SHORT
        elif displacement <= -threshold:
            direction = _LONG
        else:
            continue

        entry_price = close
        stop_price = _stop_price(
            direction, entry_price, float(atr_signal), config.stop_atr
        )
        target_price = _target_price(entry_price, opening_price, config.target_mode)

        intents.append(
            B3F3TradeIntent(
                instrument_id=instrument_id,
                config=config,
                local_date=local_date,
                signal_ts=ts,
                direction=direction,
                entry_reference_price=entry_price,
                displacement_at_entry=displacement,
                stop_price=stop_price,
                target_price=target_price,
                time_exit_boundary_utc=time_exit_boundary_utc(local_date),
            )
        )
        used_dates.add(local_date)

    return tuple(intents)
