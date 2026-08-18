from __future__ import annotations

from datetime import UTC

import numpy as np
import pandas as pd

from ftmoquant.research.alpha_lab.families import (
    PROBOT_STATUS,
    donchian_breakout_signals,
    mean_reversion_signals,
    momentum_signals,
    session_continuation_signals,
    volatility_breakout_signals,
)


def _index(periods: int, freq: str = "1h") -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=periods, freq=freq, tz=UTC)


def test_momentum_signals_go_long_on_a_rising_series() -> None:
    index = _index(40)
    close = pd.DataFrame({"A": np.linspace(100.0, 140.0, 40)}, index=index)
    entries, exits, short_entries, short_exits = momentum_signals(close, lookback=5)
    assert bool(entries["A"].iloc[10:].all())
    assert not bool(short_entries["A"].iloc[10:].any())
    assert not exits.to_numpy().any()
    assert not short_exits.to_numpy().any()


def test_momentum_signals_go_short_on_a_falling_series() -> None:
    index = _index(40)
    close = pd.DataFrame({"A": np.linspace(140.0, 100.0, 40)}, index=index)
    _, _, short_entries, _ = momentum_signals(close, lookback=5)
    assert bool(short_entries["A"].iloc[10:].all())


def test_donchian_breakout_fires_only_above_the_prior_channel_high() -> None:
    index = _index(30)
    values = np.full(30, 100.0)
    values[20] = 200.0  # a single, unambiguous breakout bar
    close = pd.DataFrame({"A": values}, index=index)
    high = close + 0.5
    low = close - 0.5
    entries, _, short_entries, _ = donchian_breakout_signals(
        high, low, close, lookback=10
    )
    assert bool(entries["A"].iloc[20])
    assert not bool(entries["A"].iloc[10:20].any())
    assert not short_entries.to_numpy().any()


def test_donchian_breakout_fires_short_below_the_prior_channel_low() -> None:
    index = _index(30)
    values = np.full(30, 100.0)
    values[20] = 10.0
    close = pd.DataFrame({"A": values}, index=index)
    high = close + 0.5
    low = close - 0.5
    _, _, short_entries, _ = donchian_breakout_signals(high, low, close, lookback=10)
    assert bool(short_entries["A"].iloc[20])


def test_mean_reversion_fades_a_large_standardized_deviation() -> None:
    index = _index(60)
    values = np.full(60, 100.0)
    values[50] = 130.0  # far above a flat rolling mean -> should fade (short)
    close = pd.DataFrame({"A": values}, index=index)
    entries, exits, short_entries, short_exits = mean_reversion_signals(
        close, lookback=20, z_entry=1.5
    )
    assert bool(short_entries["A"].iloc[50])
    assert not bool(entries["A"].iloc[50])
    # once price is back at/below the rolling mean, the short should flatten
    assert bool(short_exits["A"].iloc[51])


def test_volatility_breakout_enters_only_on_range_expansion_with_direction() -> None:
    index = _index(40)
    values = np.full(40, 100.0)
    values[30] = 100.0
    close = pd.DataFrame({"A": values}, index=index)
    high = close + 0.1
    low = close - 0.1
    high.iloc[30, 0] = 105.0  # one large-range, upward bar
    close.iloc[30, 0] = 104.0
    entries, exits, short_entries, short_exits = volatility_breakout_signals(
        high, low, close, lookback=10, threshold=1.5
    )
    assert bool(entries["A"].iloc[30])
    assert not bool(short_entries["A"].iloc[30])
    # held for exactly one bar
    assert bool(exits["A"].iloc[31])
    assert not bool(entries["A"].iloc[32])


def test_session_continuation_enters_in_the_direction_of_the_prior_session() -> None:
    # Two consecutive days of a London session (07:00-16:00 UTC) with a
    # strong up move on day one; day two should open long at 07:00 UTC.
    index = pd.date_range("2020-01-06", periods=48, freq="1h", tz=UTC)
    close = pd.DataFrame({"A": 100.0}, index=index)
    day1 = (index.date == index[7].date()) & (index.hour >= 7) & (index.hour < 16)
    close.loc[day1, "A"] = np.linspace(100.0, 110.0, int(day1.sum()))
    entries, exits, short_entries, short_exits = session_continuation_signals(
        close, session="london"
    )
    day2_open_hour = pd.Timestamp("2020-01-07 07:00", tz=UTC)
    assert day2_open_hour in entries.index
    assert bool(entries.loc[day2_open_hour, "A"])
    assert not bool(short_entries.loc[day2_open_hour, "A"])
    day2_close_hour = pd.Timestamp("2020-01-07 15:00", tz=UTC)
    assert bool(exits.loc[day2_close_hour, "A"])


def test_probot_family_is_explicitly_blocked_not_invented() -> None:
    assert PROBOT_STATUS == "BLOCKED_RULE_RECONSTRUCTION"
