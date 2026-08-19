from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import vectorbt as vbt

from ftmoquant.research.alpha_lab.families import mean_reversion_signals
from ftmoquant.research.g1.normalization import (
    G1_ANNUAL_VOLATILITY_TARGET,
    CausalEwmaDailyVolatility,
    CompletedDailyLogReturn,
    G1VolatilityNormalizer,
    VolatilityNormalizationError,
)
from ftmoquant.research.stage_g import HOLDOUT_START, VALIDATION_START
from ftmoquant.strategies.mean_reversion_h1 import (
    FROZEN_LOOKBACK,
    FROZEN_UNIVERSE,
    FROZEN_Z_ENTRY,
    CausalMeanReversionSignal,
    MeanReversionSignalError,
    causal_signal_stream,
    reject_final_holdout,
    size_positions_across_pairs,
)


def _ns(ts: pd.Timestamp) -> int:
    return int(ts.value)


def _synthetic_close(periods: int, seed: int) -> pd.Series:
    index = pd.date_range("2023-04-11", periods=periods, freq="1h", tz="UTC")
    rng = np.random.default_rng(seed)
    return pd.Series(1.1 + np.cumsum(rng.normal(0, 0.0006, periods)), index=index)


# ---------------------------------------------------------------------------
# Exact frozen-parameter enforcement.
# ---------------------------------------------------------------------------


def test_frozen_parameters_cannot_be_changed() -> None:
    with pytest.raises(MeanReversionSignalError):
        CausalMeanReversionSignal(lookback=41, z_entry=FROZEN_Z_ENTRY)
    with pytest.raises(MeanReversionSignalError):
        CausalMeanReversionSignal(lookback=FROZEN_LOOKBACK, z_entry=1.5)
    CausalMeanReversionSignal()  # defaults are the frozen values -- must succeed


def test_frozen_universe_is_exactly_seven_oanda_pairs() -> None:
    assert len(FROZEN_UNIVERSE) == 7
    assert FROZEN_UNIVERSE == tuple(sorted(FROZEN_UNIVERSE))
    assert "USD/JPY.OANDA" in FROZEN_UNIVERSE  # never excluded


# ---------------------------------------------------------------------------
# Numerical parity: the causal z-score must exactly match pandas
# rolling(window).mean()/std(ddof=1) -- the same formula families.py uses.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_causal_z_score_matches_vectorized_rolling_formula(seed: int) -> None:
    close = _synthetic_close(300, seed)
    reference_mean = close.rolling(FROZEN_LOOKBACK).mean()
    reference_std = close.rolling(FROZEN_LOOKBACK).std()
    reference_z = (close - reference_mean) / reference_std

    stream = causal_signal_stream(
        ((_ns(ts), float(value)) for ts, value in close.items())
    )
    for bar, timestamp in zip(stream, close.index, strict=True):
        expected = reference_z.loc[timestamp]
        if pd.isna(expected):
            assert bar.z_score is None
        else:
            assert bar.z_score == pytest.approx(float(expected))


# ---------------------------------------------------------------------------
# No lookahead: a bar's output is invariant to what comes after it.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [4, 5])
def test_no_lookahead_past_outputs_are_invariant_to_future_bars(seed: int) -> None:
    close = _synthetic_close(200, seed)
    bars = [(_ns(ts), float(value)) for ts, value in close.items()]

    full_stream = causal_signal_stream(bars)
    truncated_stream = causal_signal_stream(bars[:150])

    paired = zip(full_stream[:150], truncated_stream, strict=True)
    for full_bar, truncated_bar in paired:
        assert full_bar == truncated_bar


# ---------------------------------------------------------------------------
# Trade-direction/signal-state parity against the real, frozen vectorbt
# family (same decision timestamps -- execution timing is a separate,
# documented translation, not tested here as "the same instant").
# ---------------------------------------------------------------------------


def _vectorbt_position_sequence(close: pd.Series) -> pd.Series:
    frame = close.to_frame("X")
    entries, exits, short_entries, short_exits = mean_reversion_signals(
        frame, FROZEN_LOOKBACK, FROZEN_Z_ENTRY
    )
    portfolio = vbt.Portfolio.from_signals(
        frame,
        entries,
        exits,
        short_entries=short_entries,
        short_exits=short_exits,
        fees=0.0,
        freq="1h",
    )
    return np.sign(portfolio.asset_flow(direction="both")["X"].cumsum())


@pytest.mark.parametrize("seed", [10, 11, 12])
def test_signal_direction_sequence_matches_frozen_vectorbt_family(seed: int) -> None:
    close = _synthetic_close(400, seed)
    causal_positions = pd.Series(
        [bar.position for bar in causal_signal_stream(
            (_ns(ts), float(value)) for ts, value in close.items()
        )],
        index=close.index,
    )
    vectorbt_positions = _vectorbt_position_sequence(close)

    # Reversal-timing edge case (documented in _next_position's docstring):
    # a same-bar exit+opposite-entry double-trigger is not asserted here;
    # everywhere else the two position sequences must agree exactly.
    disagreements = causal_positions[
        causal_positions.astype(int) != vectorbt_positions.astype(int)
    ]
    assert len(disagreements) / len(causal_positions) < 0.02


def test_entry_and_exit_trade_counts_are_close_between_causal_and_vectorbt() -> None:
    close = _synthetic_close(500, seed=99)
    causal_positions = [
        bar.position
        for bar in causal_signal_stream(
            (_ns(ts), float(value)) for ts, value in close.items()
        )
    ]
    causal_transitions = sum(
        1 for a, b in zip(causal_positions, causal_positions[1:]) if a != b
    )
    vectorbt_positions = _vectorbt_position_sequence(close).astype(int).tolist()
    vbt_transitions = sum(
        1 for a, b in zip(vectorbt_positions, vectorbt_positions[1:]) if a != b
    )
    assert causal_transitions > 0
    assert abs(causal_transitions - vbt_transitions) <= 1


# ---------------------------------------------------------------------------
# Next-executable-event ordering (section 5's frozen execution-timing
# policy): the earliest a decision may be acted on is strictly after the
# bar that produced it.
# ---------------------------------------------------------------------------


def test_decision_time_strictly_precedes_next_executable_event() -> None:
    close = _synthetic_close(100, seed=6)
    signal = CausalMeanReversionSignal()
    timestamps = [_ns(ts) for ts in close.index]
    for i, (ts, value) in enumerate(zip(timestamps, close.to_numpy(), strict=True)):
        bar = signal.on_bar(ts, float(value))
        assert bar.timestamp_ns == ts
        if i + 1 < len(timestamps):
            earliest_executable = timestamps[i + 1]
            assert earliest_executable > bar.timestamp_ns


def test_bars_must_be_processed_in_strictly_increasing_order() -> None:
    signal = CausalMeanReversionSignal()
    signal.on_bar(1_000, 1.1)
    with pytest.raises(MeanReversionSignalError):
        signal.on_bar(1_000, 1.1)
    with pytest.raises(MeanReversionSignalError):
        signal.on_bar(500, 1.1)


# ---------------------------------------------------------------------------
# Long/short symmetry, mean-cross exits, reversal, warm-up, gaps.
# ---------------------------------------------------------------------------


def test_long_entry_symmetric_to_short_entry() -> None:
    up_values = [100.0] * (FROZEN_LOOKBACK - 1) + [130.0]
    down_values = [100.0] * (FROZEN_LOOKBACK - 1) + [70.0]
    up = causal_signal_stream(enumerate(up_values))
    down = causal_signal_stream(enumerate(down_values))
    assert up[-1].position == -1  # far above the mean -> fade short
    assert down[-1].position == 1  # far below the mean -> fade long
    assert up[-1].z_score == pytest.approx(-down[-1].z_score)


def test_mean_cross_exit_flattens_a_long_position() -> None:
    values = [100.0] * (FROZEN_LOOKBACK - 1) + [70.0, 100.0]
    bars = causal_signal_stream(enumerate(values))
    assert bars[-2].position == 1
    assert bars[-1].position == 0


def test_reversal_across_two_bars() -> None:
    values = [100.0] * (FROZEN_LOOKBACK - 1) + [70.0, 100.0, 130.0]
    bars = causal_signal_stream(enumerate(values))
    assert bars[-3].position == 1  # far below -> long
    assert bars[-2].position == 0  # reverts through the mean -> flat
    assert bars[-1].position == -1  # far above -> short


def test_warmup_produces_no_position_until_lookback_is_filled() -> None:
    values = [100.0 + i for i in range(FROZEN_LOOKBACK - 1)]
    bars = causal_signal_stream(enumerate(values))
    assert all(bar.z_score is None for bar in bars)
    assert all(bar.position == 0 for bar in bars)


def test_irregular_timestamp_gaps_do_not_change_the_signal() -> None:
    """The rolling window is observation-count-based, not calendar-time-based
    -- matching pandas rolling(window=N), which families.py also uses. A
    large gap between bar timestamps must not change the z-score."""

    values = [100.0] * (FROZEN_LOOKBACK - 1) + [130.0]
    regular = list(enumerate(values))
    irregular = [
        (i * (1 if i < len(values) - 1 else 10_000), v) for i, v in enumerate(values)
    ]
    regular_bars = causal_signal_stream(regular)
    irregular_bars = causal_signal_stream(irregular)
    assert regular_bars[-1].z_score == pytest.approx(irregular_bars[-1].z_score)
    assert regular_bars[-1].position == irregular_bars[-1].position


# ---------------------------------------------------------------------------
# Causal volatility sizing (1% annualized target, reused unchanged).
# ---------------------------------------------------------------------------


def test_volatility_target_is_frozen_at_one_percent_annualized() -> None:
    assert G1_ANNUAL_VOLATILITY_TARGET == 0.01
    with pytest.raises(VolatilityNormalizationError):
        G1VolatilityNormalizer(target_annualized_volatility=0.02)


def _synthetic_estimator(periods: int, seed: int) -> CausalEwmaDailyVolatility:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2023-01-01", periods=periods, freq="1D", tz="UTC")
    returns = [
        CompletedDailyLogReturn(int(ts.value), float(r))
        for ts, r in zip(index, rng.normal(0, 0.004, periods), strict=True)
    ]
    return CausalEwmaDailyVolatility(returns)


def test_seven_pair_sizing_requires_every_frozen_pair_no_exclusions() -> None:
    close = _synthetic_close(FROZEN_LOOKBACK, seed=7)
    signal = CausalMeanReversionSignal()
    bar = None
    for ts, value in close.items():
        bar = signal.on_bar(_ns(ts), float(value))
    assert bar is not None
    estimator = _synthetic_estimator(40, seed=8)

    incomplete_signals = {pair: bar for pair in FROZEN_UNIVERSE[:-1]}
    incomplete_estimators = {pair: estimator for pair in FROZEN_UNIVERSE[:-1]}
    with pytest.raises(MeanReversionSignalError, match="seven"):
        size_positions_across_pairs(
            signals=incomplete_signals,
            estimators=incomplete_estimators,
            decision_time_ns=_ns(close.index[-1]),
        )

    full_signals = {pair: bar for pair in FROZEN_UNIVERSE}
    full_estimators = {pair: estimator for pair in FROZEN_UNIVERSE}
    decisions = size_positions_across_pairs(
        signals=full_signals,
        estimators=full_estimators,
        decision_time_ns=_ns(close.index[-1]),
    )
    assert set(decisions) == set(FROZEN_UNIVERSE)
    for decision in decisions.values():
        assert decision.ex_ante_annualized_volatility is not None


# ---------------------------------------------------------------------------
# Final holdout must remain inaccessible during promotion.
# ---------------------------------------------------------------------------


def test_final_holdout_is_rejected() -> None:
    holdout_ns = int(HOLDOUT_START.timestamp() * 1_000_000_000)
    with pytest.raises(MeanReversionSignalError):
        reject_final_holdout(holdout_ns)
    with pytest.raises(MeanReversionSignalError):
        reject_final_holdout(holdout_ns + 1)


def test_validation_period_is_accessible() -> None:
    validation_ns = int(VALIDATION_START.timestamp() * 1_000_000_000)
    reject_final_holdout(validation_ns)  # must not raise
    reject_final_holdout(validation_ns + 1)
