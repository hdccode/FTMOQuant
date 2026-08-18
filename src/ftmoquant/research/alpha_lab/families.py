"""First real DEVELOPMENT screening batch: five mechanically simple, materially
different FX hypothesis families. Each function below is a small, stateless
signal generator over aligned :class:`AlphaLabDataset
<ftmoquant.research.alpha_lab.data.AlphaLabDataset>` matrices; nothing here
calls vectorbt directly (see
:mod:`ftmoquant.research.alpha_lab.screening_batch`, which pairs each signal
with :func:`~ftmoquant.research.alpha_lab.screening_common.run_family_grid`).

Every signal function returns ``(entries, exits, short_entries, short_exits)``
-- four boolean ``DataFrame``s aligned to the input ``close`` (same index and
columns). Where a family's positions only ever flip (never rest flat),
``exits``/``short_exits`` are returned all-``False`` and vectorbt's default
opposite-entry handling closes and reverses the position in one step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

SignalFrames = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]

#: The MQL5 ProBot supply/demand-zone strategy could not be faithfully
#: reconstructed: no saved documentation, recovered rules, or article/video
#: notes for it exist anywhere in this repository (audited via full-repo
#: search over `docs/`, `research/`, `experiments/`, and source before this
#: batch was implemented). Per the task's own stop condition, no zone,
#: multi-timeframe-agreement, or ATR(50) TP/SL logic is invented here from
#: the bulleted rule summary alone -- that would tune uncertain values from
#: the DEVELOPMENT window rather than reconstruct known rules. This family is
#: therefore intentionally not implemented in this batch.
PROBOT_STATUS = "BLOCKED_RULE_RECONSTRUCTION"
PROBOT_STATUS_REASON = (
    "No MQL5 ProBot article/video notes or recovered-rule documentation "
    "exist in this repository. Reconstructing zone construction, "
    "multi-timeframe agreement, or ATR(50) TP/SL sizing from the bulleted "
    "summary alone would mean inventing mechanics rather than replicating "
    "them, which the task explicitly forbids."
)


def _empty_bool_frame(like: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(False, index=like.index, columns=like.columns)


def momentum_signals(close: pd.DataFrame, lookback: int) -> SignalFrames:
    """Family A: time-series momentum. Long when price is above its value
    ``lookback`` bars ago, short when below; always in the market, reversing
    mechanically on a sign flip (no explicit exit signal)."""

    momentum = close - close.shift(lookback)
    entries = (momentum > 0).fillna(False)
    short_entries = (momentum < 0).fillna(False)
    flat = _empty_bool_frame(close)
    return entries, flat, short_entries, flat


def donchian_breakout_signals(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, lookback: int
) -> SignalFrames:
    """Family B: Donchian/breakout. Long on an upper-channel breakout, short
    on a lower-channel breakout (channel computed from the prior ``lookback``
    bars, never including the current bar); flat/reverse mechanically."""

    upper = high.rolling(lookback).max().shift(1)
    lower = low.rolling(lookback).min().shift(1)
    entries = (close > upper).fillna(False)
    short_entries = (close < lower).fillna(False)
    flat = _empty_bool_frame(close)
    return entries, flat, short_entries, flat


def mean_reversion_signals(
    close: pd.DataFrame, lookback: int, z_entry: float
) -> SignalFrames:
    """Family C: short-horizon mean reversion. Fade the move once the
    standardized distance from the rolling mean exceeds ``z_entry``; flat
    once price reverts back through the rolling mean."""

    mean = close.rolling(lookback).mean()
    std = close.rolling(lookback).std()
    z = (close - mean) / std
    entries = (z < -z_entry).fillna(False)
    short_entries = (z > z_entry).fillna(False)
    exits = (z >= 0).fillna(False)
    short_exits = (z <= 0).fillna(False)
    return entries, exits, short_entries, short_exits


def _true_range(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame
) -> pd.DataFrame:
    prev_close = close.shift(1)
    high_low = (high - low).to_numpy()
    high_prev = (high - prev_close).abs().to_numpy()
    low_prev = (low - prev_close).abs().to_numpy()
    combined = np.maximum(np.maximum(high_low, high_prev), low_prev)
    return pd.DataFrame(combined, index=high.index, columns=high.columns)


def volatility_breakout_signals(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    lookback: int,
    threshold: float,
) -> SignalFrames:
    """Family D: volatility breakout. Enter in the direction of the current
    bar's move when its range exceeds ``threshold`` times the trailing
    ``lookback``-bar average true range (computed from bars strictly before
    the current one); held for exactly one bar."""

    atr = _true_range(high, low, close).rolling(lookback).mean().shift(1)
    range_now = high - low
    expansion = (range_now > threshold * atr).fillna(False)
    direction_up = (close > close.shift(1)).fillna(False)
    direction_down = (close < close.shift(1)).fillna(False)
    entries = expansion & direction_up
    short_entries = expansion & direction_down
    exits = entries.shift(1).fillna(False).astype(bool)
    short_exits = short_entries.shift(1).fillna(False).astype(bool)
    return entries, exits, short_entries, short_exits


#: Fixed UTC session windows (hour-of-day, half-open [start, end)). Kept as
#: static hour bands rather than a DST-aware calendar, matching this
#: family's "keep parameterization very small" mandate.
SESSION_WINDOWS_UTC: dict[str, tuple[int, int]] = {
    "london": (7, 16),
    "new_york": (12, 21),
    "london_new_york_overlap": (12, 16),
}


def session_continuation_signals(close: pd.DataFrame, session: str) -> SignalFrames:
    """Family E: session-effect baseline. Enter at the start of today's
    session in the direction of the *previous* occurrence of that same
    session's own open-to-close return; exit at today's session close."""

    start_hour, end_hour = SESSION_WINDOWS_UTC[session]
    hours = close.index.hour
    if start_hour < end_hour:
        mask = (hours >= start_hour) & (hours < end_hour)
    else:
        mask = (hours >= start_hour) | (hours < end_hour)

    entries = _empty_bool_frame(close)
    exits = _empty_bool_frame(close)
    short_entries = _empty_bool_frame(close)
    short_exits = _empty_bool_frame(close)

    session_index = close.index[mask]
    if session_index.empty:
        return entries, exits, short_entries, short_exits
    session_dates = pd.Series(session_index.normalize(), index=session_index)

    for column in close.columns:
        series = close.loc[session_index, column]
        grouped = series.groupby(session_dates)
        session_return = grouped.last() - grouped.first()
        prior_direction = session_return.shift(1)
        first_ts = grouped.apply(lambda s: s.index[0])
        last_ts = grouped.apply(lambda s: s.index[-1])
        for date, direction in prior_direction.items():
            if pd.isna(direction) or direction == 0.0:
                continue
            entry_ts = first_ts.loc[date]
            exit_ts = last_ts.loc[date]
            if direction > 0:
                entries.loc[entry_ts, column] = True
            else:
                short_entries.loc[entry_ts, column] = True
            exits.loc[exit_ts, column] = True
            short_exits.loc[exit_ts, column] = True

    return entries, exits, short_entries, short_exits
