"""Causal, hand-specified signal generators for the frozen three-family
DEVELOPMENT screen (F1 failed-Bollinger-rejection, F2 fresh-FVG/order-block
mitigation, F3 volatility-squeeze breakout).

Each function processes one instrument's completed-bar OHLC ``Series``
(chronological, UTC-indexed -- one column of an
:class:`~ftmoquant.research.alpha_lab.data.AlphaLabDataset`) and returns a
chronological tuple of :class:`SignalEvent`, decided only from information
available at or before the emitting bar's own close. No later bar is ever
read (Bollinger/ATR use ``rolling`` windows ending at the current bar; F3's
"compressed" check explicitly reads the *prior* bar's width via ``shift(1)``;
F2's backward zone search only ever looks at bars strictly before the
forming bar).

A ``SignalEvent`` carries its stop/target as *either* a fixed price *or* a
distance/R-multiple to be resolved against the actual M1 fill price by
:mod:`ftmoquant.research.alpha_lab.wick_fvg_squeeze_execution` -- exactly the
three conventions the frozen spec requires and no more:

- F1: stop as a distance (``1.5 * ATR`` frozen at the signal bar), target as
  a fixed price (the Bollinger midpoint frozen at the signal bar).
- F2: stop as a fixed price (the order-block's own distal edge), target as
  an R-multiple (``1.5R``) computed from the *actual* entry fill.
- F3: both stop and target as fixed distances (``2*ATR`` / ``4*ATR`` frozen
  at the signal bar).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd  # type: ignore[import-untyped]

DIRECTION_LONG = 1
DIRECTION_SHORT = -1


@dataclass(frozen=True, slots=True)
class SignalEvent:
    """One causal trade instruction, decided at ``signal_bar_ts``'s close.

    Exactly one of ``stop_price``/``stop_distance`` and exactly one of
    ``target_price``/``target_distance``/``target_r_multiple`` must be set;
    see the module docstring for which family uses which combination.
    """

    signal_bar_ts: pd.Timestamp
    direction: int
    stop_price: float | None = None
    stop_distance: float | None = None
    target_price: float | None = None
    target_distance: float | None = None
    target_r_multiple: float | None = None

    def __post_init__(self) -> None:
        if self.direction not in (DIRECTION_LONG, DIRECTION_SHORT):
            raise ValueError("direction must be +1 (long) or -1 (short)")
        if (self.stop_price is None) == (self.stop_distance is None):
            raise ValueError("exactly one of stop_price/stop_distance is required")
        target_fields = (
            self.target_price,
            self.target_distance,
            self.target_r_multiple,
        )
        if sum(field is not None for field in target_fields) != 1:
            raise ValueError(
                "exactly one of target_price/target_distance/target_r_multiple "
                "is required"
            )


def _bollinger_bands(
    close: pd.Series, window: int = 20
) -> tuple[pd.Series, pd.Series, pd.Series]:
    midpoint = close.rolling(window).mean()
    std = close.rolling(window).std(ddof=1)
    upper = midpoint + 2.0 * std
    lower = midpoint - 2.0 * std
    return midpoint, upper, lower


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Identical formula to
    :func:`ftmoquant.research.alpha_lab.families._true_range`, adapted to a
    single instrument's ``Series`` (this module always processes one pair at
    a time) instead of a multi-instrument ``DataFrame``."""

    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def _atr(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    return _true_range(high, low, close).rolling(window).mean()


def f1_failed_bollinger_rejection_signals(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    wick_ratio: float,
) -> tuple[SignalEvent, ...]:
    """F1: a completed bar that pokes through a Bollinger band on its wick
    but closes back inside it, with a wick large relative to its own range
    and a range/midpoint-distance large relative to ATR -- entered in the
    rejection direction, stop 1.5*ATR away, target the Bollinger midpoint
    (both frozen from the signal bar).

    If a single bar (a very wide bar breaching both bands) satisfies both
    the bearish and bullish conditions simultaneously, both events are
    emitted at that timestamp with a stable bearish-then-bullish order; the
    execution engine's one-position-at-a-time rule then admits whichever
    sorts first and skips the other, deterministically.
    """

    midpoint, upper, lower = _bollinger_bands(close)
    atr = _atr(high, low, close)
    candle_range = high - low
    body_high = pd.concat([open_, close], axis=1).max(axis=1)
    body_low = pd.concat([open_, close], axis=1).min(axis=1)
    upper_wick = high - body_high
    lower_wick = body_low - low

    valid_range = candle_range > 0
    bearish = (
        valid_range
        & (high > upper)
        & (close < upper)
        & ((upper_wick / candle_range) > wick_ratio)
        & (candle_range > 0.5 * atr)
        & ((close - midpoint).abs() > 0.3 * atr)
    ).fillna(False)
    bullish = (
        valid_range
        & (low < lower)
        & (close > lower)
        & ((lower_wick / candle_range) > wick_ratio)
        & (candle_range > 0.5 * atr)
        & ((midpoint - close).abs() > 0.3 * atr)
    ).fillna(False)

    events: list[SignalEvent] = []
    for ts in close.index[bearish]:
        events.append(
            SignalEvent(
                signal_bar_ts=ts,
                direction=DIRECTION_SHORT,
                stop_distance=1.5 * float(atr.loc[ts]),
                target_price=float(midpoint.loc[ts]),
            )
        )
    for ts in close.index[bullish]:
        events.append(
            SignalEvent(
                signal_bar_ts=ts,
                direction=DIRECTION_LONG,
                stop_distance=1.5 * float(atr.loc[ts]),
                target_price=float(midpoint.loc[ts]),
            )
        )
    events.sort(key=lambda event: event.signal_bar_ts)
    return tuple(events)


def f3_volatility_squeeze_breakout_signals(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    bb_width_threshold: float,
) -> tuple[SignalEvent, ...]:
    """F3: the *prior* completed bar's normalized Bollinger width was below
    ``bb_width_threshold`` (a squeeze) and the current completed bar closes
    outside the band -- entered in the breakout direction, stop 2*ATR away,
    target 4*ATR away (2R), both frozen from the signal bar."""

    midpoint, upper, lower = _bollinger_bands(close)
    bb_width = (upper - lower) / midpoint
    atr = _atr(high, low, close)
    compressed = (bb_width.shift(1) < bb_width_threshold).fillna(False)

    long_breakout = (compressed & (close > upper)).fillna(False)
    short_breakout = (compressed & (close < lower)).fillna(False)

    events: list[SignalEvent] = []
    for ts in close.index[long_breakout]:
        events.append(
            SignalEvent(
                signal_bar_ts=ts,
                direction=DIRECTION_LONG,
                stop_distance=2.0 * float(atr.loc[ts]),
                target_distance=4.0 * float(atr.loc[ts]),
            )
        )
    for ts in close.index[short_breakout]:
        events.append(
            SignalEvent(
                signal_bar_ts=ts,
                direction=DIRECTION_SHORT,
                stop_distance=2.0 * float(atr.loc[ts]),
                target_distance=4.0 * float(atr.loc[ts]),
            )
        )
    events.sort(key=lambda event: event.signal_bar_ts)
    return tuple(events)


@dataclass(slots=True)
class _Zone:
    created_index: int
    direction: int
    lower: float
    upper: float
    consumed: bool = False


def _most_recent_body(
    open_: pd.Series,
    close: pd.Series,
    *,
    at: int,
    backward_search_n: int,
    bearish: bool,
) -> tuple[float, float] | None:
    """Search strictly backward from (but not including) bar ``at`` through
    at most ``backward_search_n`` completed bars for the most recent
    bearish (``close < open``) or bullish (``close > open``) candle; return
    its body ``(lower, upper)`` edges, or ``None`` if none is found."""

    start = max(0, at - backward_search_n)
    for i in range(at - 1, start - 1, -1):
        o, c = float(open_.iloc[i]), float(close.iloc[i])
        if (bearish and c < o) or (not bearish and c > o):
            return min(o, c), max(o, c)
    return None


def f2_fresh_fvg_mitigation_signals(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    backward_search_n: int,
) -> tuple[SignalEvent, ...]:
    """F2: an objective fair-value-gap / order-block mitigation state
    machine.

    On completed bar ``t`` (index >= 2): a bullish FVG (``low[t] >
    high[t-2]``) creates a fresh *demand* zone from the body of the most
    recent bearish candle found searching strictly backward through the
    previous ``backward_search_n`` bars; a bearish FVG (``high[t] <
    low[t-2]``) creates a fresh *supply* zone from the most recent bullish
    candle's body, identically mirrored. No zone is created if no
    qualifying candle exists within the search window.

    Zone lifecycle (deterministic rule, evaluated once per zone per bar,
    starting the bar *after* its own creation bar):

    1. If the zone is still fresh and this bar's close breaches its distal
       edge, the zone is INVALIDATED and can never trade.
    2. Else if the zone is still fresh and this bar's range touches its
       proximal edge (a retest), this is defined as the zone's one and only
       eligible retest -- it is consumed here regardless of outcome. If
       this same bar's close also reclaims back beyond the proximal edge,
       a trade is triggered; if it only touches without reclaiming, the
       zone is spent with no trade. ("Only first retest is trade-eligible"
       is read literally: the first retest, successful or not, is final.)
    3. Otherwise the zone remains fresh for the next bar.

    Multiple zones can be simultaneously active. They are processed
    oldest-created-first (zones are appended in formation order, so a
    simple in-order pass already satisfies this); at most one new entry is
    admitted per bar -- once one zone triggers a trade on a bar, later
    zones evaluated on that same bar can still be invalidated/consumed but
    cannot themselves open a second position that bar.
    """

    n = len(close)
    index = close.index
    h = high.to_numpy()
    low_arr = low.to_numpy()
    c = close.to_numpy()

    zones: list[_Zone] = []
    events: list[SignalEvent] = []

    for t in range(n):
        entry_placed_this_bar = False
        for zone in zones:
            if zone.consumed or zone.created_index >= t:
                continue
            if zone.direction == DIRECTION_LONG:
                invalidated = c[t] < zone.lower
                touched = low_arr[t] <= zone.upper
                reclaimed = c[t] > zone.upper
            else:
                invalidated = c[t] > zone.upper
                touched = h[t] >= zone.lower
                reclaimed = c[t] < zone.lower

            if invalidated:
                zone.consumed = True
                continue
            if touched:
                zone.consumed = True
                if reclaimed and not entry_placed_this_bar:
                    stop_price = (
                        zone.lower if zone.direction == DIRECTION_LONG else zone.upper
                    )
                    events.append(
                        SignalEvent(
                            signal_bar_ts=index[t],
                            direction=zone.direction,
                            stop_price=float(stop_price),
                            target_r_multiple=1.5,
                        )
                    )
                    entry_placed_this_bar = True

        if t >= 2:
            if low_arr[t] > h[t - 2]:
                body = _most_recent_body(
                    open_,
                    close,
                    at=t,
                    backward_search_n=backward_search_n,
                    bearish=True,
                )
                if body is not None:
                    zones.append(
                        _Zone(
                            created_index=t,
                            direction=DIRECTION_LONG,
                            lower=body[0],
                            upper=body[1],
                        )
                    )
            if h[t] < low_arr[t - 2]:
                body = _most_recent_body(
                    open_,
                    close,
                    at=t,
                    backward_search_n=backward_search_n,
                    bearish=False,
                )
                if body is not None:
                    zones.append(
                        _Zone(
                            created_index=t,
                            direction=DIRECTION_SHORT,
                            lower=body[0],
                            upper=body[1],
                        )
                    )

    events.sort(key=lambda event: event.signal_bar_ts)
    return tuple(events)
