"""Causal, hand-specified signal generators for Batch 2's four
funded-account-oriented structural families (B2-F1 liquidity-sweep -> BOS ->
retest, B2-F2 previous-day/week liquidity-sweep rejection, B2-F3
liquidity-sweep -> CHOCH -> retracement, B2-F4 Orion-style compression-box
breakout).

Reuses :class:`~ftmoquant.research.alpha_lab.wick_fvg_squeeze_signals.SignalEvent`,
its ``DIRECTION_LONG``/``DIRECTION_SHORT`` constants, and its causal ``_atr``
helper unchanged from Batch 1 -- every Batch-2 stop is a fixed price known
at signal time and every target is an R-multiple, exactly the
``stop_price`` + ``target_r_multiple`` convention Batch 1's B2-F2-equivalent
(F2 fresh-FVG family) already established. No new execution primitive is
needed; :mod:`ftmoquant.research.alpha_lab.wick_fvg_squeeze_execution` is
reused unchanged by the Batch-2 screen module.

Each signal function processes one instrument's completed-bar OHLC
``Series`` (chronological, UTC-indexed) and returns a chronological tuple of
``SignalEvent``s decided only from information available at or before the
emitting bar's own close.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ftmoquant.research.alpha_lab.wick_fvg_squeeze_signals import (
    DIRECTION_LONG,
    DIRECTION_SHORT,
    SignalEvent,
    _atr,
)

#: B2-F2/B2-F3's fixed (non-gridded) risk/reward.
B2F2_FIXED_RR = 2.0
B2F3_FIXED_RR = 2.0
B2F4_FIXED_RR = 1.9

# ---------------------------------------------------------------------------
# Shared causal structural helpers (B2-F1, B2-F3, B2-F4 all use a "prior N
# completed bars, excluding the current one" window).
# ---------------------------------------------------------------------------


def _prior_swing_high(high: pd.Series, lookback: int) -> pd.Series:
    """``max(high[t-lookback:t])`` -- the rolling max over exactly the
    ``lookback`` bars strictly before ``t``, never including ``t`` itself."""

    return high.rolling(lookback).max().shift(1)


def _prior_swing_low(low: pd.Series, lookback: int) -> pd.Series:
    return low.rolling(lookback).min().shift(1)


# ---------------------------------------------------------------------------
# B2-F1 / B2-F3 shared sweep -> structural-break (BOS/CHOCH) state machine.
#
# Both families freeze a sweep extreme when price pokes beyond a prior swing
# level and closes back inside it, then wait for a later completed bar to
# close beyond the running low/high accumulated since the sweep bar -- B2-F1
# calls this a "BOS", B2-F3 a "CHOCH"; the underlying causal definition the
# spec gives for both is identical, so this state advance is written once
# and reused by both families' per-bar loops, which differ only in what
# happens once the break is confirmed (B2-F1: wait for a retest of the
# broken level; B2-F3: compute one frozen Fibonacci retracement level and
# wait for price to reach it).
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _StructuralSetup:
    stage: str  # "swept" | "confirmed"
    direction: int  # DIRECTION_SHORT (upside/bearish sweep) or DIRECTION_LONG
    sweep_extreme: float
    sweep_start_index: int
    running_extreme: float  # running min-low (SHORT) / max-high (LONG) since
    # the sweep bar, accumulated through the bar immediately before the
    # current one -- exactly "since the sweep event began" evaluated causally.
    structural_level: float | None = None  # frozen once "confirmed"
    extra: float | None = None  # family-specific payload (B2-F3's frozen
    # retracement level); unused by B2-F1.


def _advance_sweep_and_break(
    setup: _StructuralSetup | None,
    *,
    t: int,
    direction: int,
    h: np.ndarray,
    l: np.ndarray,  # noqa: E741 -- matches the OHLC-array convention below
    c: np.ndarray,
    prior_swing_high: np.ndarray,
    prior_swing_low: np.ndarray,
    expiry_bars: int,
) -> tuple[_StructuralSetup | None, bool]:
    """Advance one direction's sweep/structural-break state by exactly one
    completed bar. Returns ``(updated_setup, just_confirmed_this_bar)``.

    Deterministic rules (documented interpretations of the frozen spec):

    - Expiry: a setup not yet confirmed-and-entered is dropped once more
      than ``expiry_bars`` (``3 * lookback``) bars have elapsed since its
      own sweep bar.
    - Structural invalidation: at any stage, a close back beyond the
      sweep's own extreme (above the swept high for a bearish/SHORT setup,
      below the swept low for a bullish/LONG one) invalidates the setup --
      the literal rule B2-F3 section 4.4 states explicitly ("price breaks
      beyond sweep extreme before entry"), applied identically to B2-F1
      since both freeze the same kind of sweep extreme.
    - Replacement: "a new same-direction sweep replaces an unconfirmed
      older sweep only if it is more extreme" is read as: while still in
      the pre-confirmation ("swept") stage, a fresh qualifying sweep this
      bar that is strictly more extreme than the current one replaces it
      (discarding any running-extreme progress toward confirmation). A
      fresh sweep event that occurs once a setup is already "confirmed"
      (waiting for retest/retracement) is ignored -- only the pre-BOS/CHOCH
      stage is "unconfirmed".
    """

    if setup is not None and t - setup.sweep_start_index > expiry_bars:
        setup = None
    if setup is not None:
        invalidated = (
            c[t] > setup.sweep_extreme
            if direction == DIRECTION_SHORT
            else c[t] < setup.sweep_extreme
        )
        if invalidated:
            setup = None

    if direction == DIRECTION_SHORT:
        is_sweep = (
            not np.isnan(prior_swing_high[t])
            and h[t] > prior_swing_high[t]
            and c[t] < prior_swing_high[t]
        )
    else:
        is_sweep = (
            not np.isnan(prior_swing_low[t])
            and l[t] < prior_swing_low[t]
            and c[t] > prior_swing_low[t]
        )

    just_confirmed = False
    if is_sweep:
        extreme = float(h[t]) if direction == DIRECTION_SHORT else float(l[t])
        should_start_or_replace = setup is None
        if setup is not None and setup.stage == "swept":
            should_start_or_replace = (
                extreme > setup.sweep_extreme
                if direction == DIRECTION_SHORT
                else extreme < setup.sweep_extreme
            )
        if should_start_or_replace:
            seed_running = float(l[t]) if direction == DIRECTION_SHORT else float(h[t])
            setup = _StructuralSetup(
                stage="swept",
                direction=direction,
                sweep_extreme=extreme,
                sweep_start_index=t,
                running_extreme=seed_running,
            )
    elif setup is not None and setup.stage == "swept":
        candidate = setup.running_extreme
        broke = c[t] < candidate if direction == DIRECTION_SHORT else c[t] > candidate
        if broke:
            setup.stage = "confirmed"
            setup.structural_level = candidate
            just_confirmed = True
        else:
            setup.running_extreme = (
                min(setup.running_extreme, float(l[t]))
                if direction == DIRECTION_SHORT
                else max(setup.running_extreme, float(h[t]))
            )

    return setup, just_confirmed


# ---------------------------------------------------------------------------
# B2-F1: liquidity sweep -> BOS -> retest.
# ---------------------------------------------------------------------------


def b2f1_sweep_bos_retest_signals(
    high: pd.Series, low: pd.Series, close: pd.Series, *, swing_lookback: int, rr: float
) -> tuple[SignalEvent, ...]:
    n = len(close)
    index = close.index
    h = high.to_numpy()
    l = low.to_numpy()  # noqa: E741
    c = close.to_numpy()
    prior_swing_high = _prior_swing_high(high, swing_lookback).to_numpy()
    prior_swing_low = _prior_swing_low(low, swing_lookback).to_numpy()
    expiry_bars = 3 * swing_lookback

    bearish: _StructuralSetup | None = None
    bullish: _StructuralSetup | None = None
    events: list[SignalEvent] = []

    for t in range(n):
        bearish, bearish_confirmed_now = _advance_sweep_and_break(
            bearish,
            t=t,
            direction=DIRECTION_SHORT,
            h=h,
            l=l,
            c=c,
            prior_swing_high=prior_swing_high,
            prior_swing_low=prior_swing_low,
            expiry_bars=expiry_bars,
        )
        if (
            bearish is not None
            and bearish.stage == "confirmed"
            and not bearish_confirmed_now
        ):
            assert bearish.structural_level is not None
            if h[t] >= bearish.structural_level and c[t] < bearish.structural_level:
                events.append(
                    SignalEvent(
                        signal_bar_ts=index[t],
                        direction=DIRECTION_SHORT,
                        stop_price=float(bearish.sweep_extreme),
                        target_r_multiple=rr,
                    )
                )
                bearish = None

        bullish, bullish_confirmed_now = _advance_sweep_and_break(
            bullish,
            t=t,
            direction=DIRECTION_LONG,
            h=h,
            l=l,
            c=c,
            prior_swing_high=prior_swing_high,
            prior_swing_low=prior_swing_low,
            expiry_bars=expiry_bars,
        )
        if (
            bullish is not None
            and bullish.stage == "confirmed"
            and not bullish_confirmed_now
        ):
            assert bullish.structural_level is not None
            if l[t] <= bullish.structural_level and c[t] > bullish.structural_level:
                events.append(
                    SignalEvent(
                        signal_bar_ts=index[t],
                        direction=DIRECTION_LONG,
                        stop_price=float(bullish.sweep_extreme),
                        target_r_multiple=rr,
                    )
                )
                bullish = None

    events.sort(key=lambda event: event.signal_bar_ts)
    return tuple(events)


# ---------------------------------------------------------------------------
# B2-F2: previous-day / previous-week liquidity sweep rejection.
# ---------------------------------------------------------------------------

#: The repo's existing FX-trading-day rollover convention (see
#: ``eurusd_liquidity_shock_reversion_development.py`` /
#: ``eurusd_session_range_expansion_development.py``:
#: ``local.weekday() <= 4 and local.hour == 17`` in America/New_York time,
#: and ``session_coverage.py``'s documented "17:00 New York" FX day/week
#: boundary) -- reused here rather than a naive UTC-midnight day split.
_FX_ROLLOVER_ZONE = ZoneInfo("America/New_York")
_FX_ROLLOVER_HOUR = 17


def _fx_trading_day_ids(index: pd.DatetimeIndex) -> list[date]:
    """One label per completed FX trading day (the half-open window
    [D 17:00, D+1 17:00) America/New_York), identical for every bar falling
    inside that window."""

    local = index.tz_convert(_FX_ROLLOVER_ZONE)
    shifted = local - pd.Timedelta(hours=_FX_ROLLOVER_HOUR)
    return list(shifted.date)


def _fx_trading_week_ids(day_ids: Sequence[date]) -> list[date]:
    """One label per completed FX trading week: the Sunday date of the
    week containing each day id (the FX week runs Sunday 17:00 through
    Friday 17:00 America/New_York -- see ``session_coverage.py``)."""

    return [d - timedelta(days=(d.weekday() + 1) % 7) for d in day_ids]


def _previous_period_levels(
    high: pd.Series, low: pd.Series, period_ids: Sequence[date]
) -> tuple[pd.Series, pd.Series]:
    """Causal previous-period high/low, aligned to ``high``'s index: at
    each bar, the high/low of the most recent DISTINCT period id that is
    strictly earlier than the bar's own period id (never the current,
    still-forming period). Every bar belonging to the same period id
    necessarily precedes every bar of a later period id, so looking up an
    earlier period's already-fully-observed aggregate is causal by
    construction -- no naive future-aware resampling is used."""

    period_high: dict[date, float] = {}
    period_low: dict[date, float] = {}
    for pid, h, low_v in zip(period_ids, high.to_numpy(), low.to_numpy(), strict=True):
        period_high[pid] = max(period_high.get(pid, float("-inf")), float(h))
        period_low[pid] = min(period_low.get(pid, float("inf")), float(low_v))

    sorted_ids = sorted(period_high)
    predecessor = {pid: sorted_ids[i - 1] for i, pid in enumerate(sorted_ids) if i > 0}
    prev_high = [
        period_high[predecessor[pid]] if pid in predecessor else float("nan")
        for pid in period_ids
    ]
    prev_low = [
        period_low[predecessor[pid]] if pid in predecessor else float("nan")
        for pid in period_ids
    ]
    return (
        pd.Series(prev_high, index=high.index),
        pd.Series(prev_low, index=high.index),
    )


def b2f2_previous_period_sweep_rejection_signals(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    level_type: str,
    rejection_mode: str,
) -> tuple[SignalEvent, ...]:
    """B2-F2: a completed bar sweeps beyond the frozen previous-day/week
    high or low and closes back inside it (optionally requiring the
    rejecting wick to be at least half the candle's own range). Only the
    first qualifying sweep of a given frozen level is trade-eligible; a
    raw sweep that fails ``WICK_50``'s filter does not spend the level --
    the level is only marked spent once a bar actually qualifies."""

    if level_type not in ("PREVIOUS_DAY", "PREVIOUS_WEEK"):
        raise ValueError(f"unknown level_type: {level_type}")
    if rejection_mode not in ("CLOSE_BACK", "WICK_50"):
        raise ValueError(f"unknown rejection_mode: {rejection_mode}")

    day_ids = _fx_trading_day_ids(close.index)
    period_ids: list[date] = (
        day_ids if level_type == "PREVIOUS_DAY" else _fx_trading_week_ids(day_ids)
    )
    prev_high, prev_low = _previous_period_levels(high, low, period_ids)

    n = len(close)
    index = close.index
    o = open_.to_numpy()
    h = high.to_numpy()
    l = low.to_numpy()  # noqa: E741
    c = close.to_numpy()
    ph = prev_high.to_numpy()
    pl = prev_low.to_numpy()
    candle_range = h - l
    body_high = np.maximum(o, c)
    body_low = np.minimum(o, c)
    upper_wick = h - body_high
    lower_wick = body_low - l

    events: list[SignalEvent] = []
    spent_bearish: date | None = None
    spent_bullish: date | None = None
    for t in range(n):
        pid = period_ids[t]
        if not np.isnan(ph[t]) and h[t] > ph[t] and c[t] < ph[t]:
            qualifies = True
            if rejection_mode == "WICK_50":
                qualifies = candle_range[t] > 0 and (
                    upper_wick[t] / candle_range[t]
                ) >= 0.5
            if qualifies and pid != spent_bearish:
                events.append(
                    SignalEvent(
                        signal_bar_ts=index[t],
                        direction=DIRECTION_SHORT,
                        stop_price=float(h[t]),
                        target_r_multiple=B2F2_FIXED_RR,
                    )
                )
                spent_bearish = pid
        if not np.isnan(pl[t]) and l[t] < pl[t] and c[t] > pl[t]:
            qualifies = True
            if rejection_mode == "WICK_50":
                qualifies = candle_range[t] > 0 and (
                    lower_wick[t] / candle_range[t]
                ) >= 0.5
            if qualifies and pid != spent_bullish:
                events.append(
                    SignalEvent(
                        signal_bar_ts=index[t],
                        direction=DIRECTION_LONG,
                        stop_price=float(l[t]),
                        target_r_multiple=B2F2_FIXED_RR,
                    )
                )
                spent_bullish = pid

    events.sort(key=lambda event: event.signal_bar_ts)
    return tuple(events)


# ---------------------------------------------------------------------------
# B2-F3: liquidity sweep -> CHOCH -> retracement.
# ---------------------------------------------------------------------------


def b2f3_sweep_choch_retracement_signals(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    swing_lookback: int,
    retracement: float,
) -> tuple[SignalEvent, ...]:
    """B2-F3 shares B2-F1's sweep -> structural-break state machine
    unchanged (:func:`_advance_sweep_and_break`); the only difference is
    what happens once the break (here called CHOCH) confirms: instead of
    waiting for a retest of the broken level, one frozen Fibonacci
    retracement level is computed immediately from the confirmed
    displacement leg (``impulse_start`` = the sweep extreme,
    ``impulse_end`` = the CHOCH's own frozen structural extreme) and never
    recomputed; the setup then waits for a later completed bar's
    high/low to reach that level. "Another position is already active"
    (section 4.4) is enforced downstream by
    :func:`~ftmoquant.research.alpha_lab.wick_fvg_squeeze_execution.simulate_trades`'s
    existing one-position-at-a-time gating, exactly as every other family
    in this repo relies on that same shared execution-level gate rather
    than the signal generator tracking cross-family position state.
    """

    n = len(close)
    index = close.index
    h = high.to_numpy()
    l = low.to_numpy()  # noqa: E741
    c = close.to_numpy()
    prior_swing_high = _prior_swing_high(high, swing_lookback).to_numpy()
    prior_swing_low = _prior_swing_low(low, swing_lookback).to_numpy()
    expiry_bars = 3 * swing_lookback

    bearish: _StructuralSetup | None = None
    bullish: _StructuralSetup | None = None
    events: list[SignalEvent] = []

    for t in range(n):
        bearish, bearish_confirmed_now = _advance_sweep_and_break(
            bearish,
            t=t,
            direction=DIRECTION_SHORT,
            h=h,
            l=l,
            c=c,
            prior_swing_high=prior_swing_high,
            prior_swing_low=prior_swing_low,
            expiry_bars=expiry_bars,
        )
        if bearish is not None and bearish_confirmed_now:
            assert bearish.structural_level is not None
            impulse_low = bearish.structural_level
            bearish.extra = impulse_low + retracement * (
                bearish.sweep_extreme - impulse_low
            )
        elif bearish is not None and bearish.stage == "confirmed":
            assert bearish.extra is not None
            if h[t] >= bearish.extra:
                events.append(
                    SignalEvent(
                        signal_bar_ts=index[t],
                        direction=DIRECTION_SHORT,
                        stop_price=float(bearish.sweep_extreme),
                        target_r_multiple=B2F3_FIXED_RR,
                    )
                )
                bearish = None

        bullish, bullish_confirmed_now = _advance_sweep_and_break(
            bullish,
            t=t,
            direction=DIRECTION_LONG,
            h=h,
            l=l,
            c=c,
            prior_swing_high=prior_swing_high,
            prior_swing_low=prior_swing_low,
            expiry_bars=expiry_bars,
        )
        if bullish is not None and bullish_confirmed_now:
            assert bullish.structural_level is not None
            impulse_high = bullish.structural_level
            bullish.extra = impulse_high - retracement * (
                impulse_high - bullish.sweep_extreme
            )
        elif bullish is not None and bullish.stage == "confirmed":
            assert bullish.extra is not None
            if l[t] <= bullish.extra:
                events.append(
                    SignalEvent(
                        signal_bar_ts=index[t],
                        direction=DIRECTION_LONG,
                        stop_price=float(bullish.sweep_extreme),
                        target_r_multiple=B2F3_FIXED_RR,
                    )
                )
                bullish = None

    events.sort(key=lambda event: event.signal_bar_ts)
    return tuple(events)


# ---------------------------------------------------------------------------
# B2-F4: Orion-style horizontal compression-box breakout.
# ---------------------------------------------------------------------------

#: Frozen research interpretation, decided BEFORE any real run (section
#: 5.1): no prior Orion-strategy research notes exist anywhere in this
#: repository (audited via full-repo search before implementation), so no
#: source-default flatness threshold could be recovered; the spec's own
#: fallback value is used and is not gridded.
FROZEN_BOX_FLATNESS_THRESHOLD = 0.5
BOX_MAX_AGE_BARS = 45

#: Deterministic overlap-suppression rule (section 5.3): a newly detected
#: candidate box is suppressed (the older active box is retained untouched)
#: iff the price-range intersection between the two covers at least this
#: fraction of the NEW candidate's own height:
#:     overlap = max(0, min(new.high, old.high) - max(new.low, old.low))
#:     fraction = overlap / new.height
#: i.e. "how much of the box we'd otherwise add is already covered by an
#: existing active box".
BOX_OVERLAP_SUPPRESSION_FRACTION = 0.8


@dataclass(slots=True)
class _Box:
    high: float
    low: float
    formed_at: int
    expires_at: int


def _ols_slope(y: np.ndarray) -> float:
    x = np.arange(len(y), dtype=float)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = float(((x - x_mean) ** 2).sum())
    if denom == 0.0:
        return 0.0
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)


def b2f4_compression_box_breakout_signals(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    box_length: int,
    max_box_atr: float,
) -> tuple[SignalEvent, ...]:
    """B2-F4: on each completed bar, the trailing ``box_length`` bars
    (strictly excluding the current one) are tested for compression
    (height <= ``max_box_atr`` * ATR(14) as of the prior bar) and flatness
    (a causal OLS slope of close over that same window, normalized by
    ``box_length``/ATR, at or below :data:`FROZEN_BOX_FLATNESS_THRESHOLD`).
    A qualifying window becomes an active candidate box (subject to the
    overlap-suppression rule above) valid for at most
    :data:`BOX_MAX_AGE_BARS` bars; any active box is consumed (removed) the
    moment a later completed bar's close breaks beyond it in the EMA150
    trend direction."""

    n = len(close)
    index = close.index
    c = close.to_numpy()
    box_high_series = _prior_swing_high(high, box_length).to_numpy()
    box_low_series = _prior_swing_low(low, box_length).to_numpy()
    atr_prev = _atr(high, low, close).shift(1).to_numpy()
    ema150 = close.ewm(span=150, adjust=False).mean().to_numpy()

    active_boxes: list[_Box] = []
    events: list[SignalEvent] = []

    for t in range(n):
        active_boxes = [box for box in active_boxes if t <= box.expires_at]

        if box_length <= t and not np.isnan(atr_prev[t]) and atr_prev[t] > 0:
            box_high = box_high_series[t]
            box_low = box_low_series[t]
            height = box_high - box_low
            window = c[t - box_length : t]
            slope = _ols_slope(window)
            normalized_slope = abs(slope * box_length) / atr_prev[t]
            if (
                height <= max_box_atr * atr_prev[t]
                and normalized_slope <= FROZEN_BOX_FLATNESS_THRESHOLD
            ):
                candidate = _Box(
                    high=float(box_high),
                    low=float(box_low),
                    formed_at=t,
                    expires_at=t + BOX_MAX_AGE_BARS,
                )
                candidate_height = candidate.high - candidate.low
                suppressed = False
                for existing in active_boxes:
                    overlap = max(
                        0.0,
                        min(candidate.high, existing.high)
                        - max(candidate.low, existing.low),
                    )
                    fraction = (
                        overlap / candidate_height if candidate_height > 0 else 0.0
                    )
                    if fraction >= BOX_OVERLAP_SUPPRESSION_FRACTION:
                        suppressed = True
                        break
                if not suppressed:
                    active_boxes.append(candidate)

        triggered_box: _Box | None = None
        triggered_direction = 0
        for box in active_boxes:
            if c[t] > box.high and c[t] > ema150[t]:
                triggered_box = box
                triggered_direction = DIRECTION_LONG
                break
            if c[t] < box.low and c[t] < ema150[t]:
                triggered_box = box
                triggered_direction = DIRECTION_SHORT
                break
        if triggered_box is not None:
            stop_price = (
                triggered_box.low
                if triggered_direction == DIRECTION_LONG
                else triggered_box.high
            )
            events.append(
                SignalEvent(
                    signal_bar_ts=index[t],
                    direction=triggered_direction,
                    stop_price=float(stop_price),
                    target_r_multiple=B2F4_FIXED_RR,
                )
            )
            active_boxes.remove(triggered_box)

    events.sort(key=lambda event: event.signal_bar_ts)
    return tuple(events)
