from __future__ import annotations

from decimal import Decimal

import pytest

from ftmoquant.strategies.eurusd_liquidity_shock_reversion import (
    EurusdLiquidityShockReversionParameters,
    EurusdLiquidityShockReversionState,
    EurusdLiquidityShockReversionValidationError,
)
from ftmoquant.strategies.ts_momentum import RawDirectionalTarget

_MIN = 60_000_000_000


def _params(
    baseline: int = 5, multiple: str = "3", hold: int = 2
) -> EurusdLiquidityShockReversionParameters:
    return EurusdLiquidityShockReversionParameters(
        baseline_prior_returns=baseline,
        shock_multiple=Decimal(multiple),
        hold_eligible_minutes=hold,
    )


def _warmup(
    state: EurusdLiquidityShockReversionState, start: int, prices: list[str]
) -> int:
    t = start
    for price in prices:
        t += _MIN
        state.warmup_minute_close(t, t + 1, Decimal(price))
    return t


def test_parameters_reject_non_positive_or_non_finite_values() -> None:
    with pytest.raises(EurusdLiquidityShockReversionValidationError):
        EurusdLiquidityShockReversionParameters(0, Decimal("3"), 5)
    with pytest.raises(EurusdLiquidityShockReversionValidationError):
        EurusdLiquidityShockReversionParameters(30, Decimal("0"), 5)
    with pytest.raises(EurusdLiquidityShockReversionValidationError):
        EurusdLiquidityShockReversionParameters(30, Decimal("3"), 0)
    with pytest.raises(EurusdLiquidityShockReversionValidationError):
        EurusdLiquidityShockReversionParameters(30, Decimal("Infinity"), 5)


def test_no_signal_before_baseline_window_is_full() -> None:
    state = EurusdLiquidityShockReversionState(_params(baseline=5))
    t = 0
    for _ in range(4):
        t += _MIN
        signal = state.on_minute_close(t, t + 1, Decimal("1.10000"))
        assert signal is None
    # A huge move on only the 5th close still can't trigger: baseline needs
    # 5 *prior* returns, so this is only the 4th return.
    t += _MIN
    signal = state.on_minute_close(t, t + 1, Decimal("1.20000"))
    assert signal is None


def test_positive_shock_targets_short_and_negative_shock_targets_long() -> None:
    up = EurusdLiquidityShockReversionState(_params())
    t = _warmup(
        up, 0, ["1.10001", "1.10002", "1.10003", "1.10004", "1.10005", "1.10006"]
    )
    t += _MIN
    signal = up.on_minute_close(t, t + 1, Decimal("1.11500"))
    assert signal is not None
    assert signal.target is RawDirectionalTarget.SHORT

    down = EurusdLiquidityShockReversionState(_params())
    t = _warmup(
        down, 0, ["1.10001", "1.10002", "1.10003", "1.10004", "1.10005", "1.10006"]
    )
    t += _MIN
    signal = down.on_minute_close(t, t + 1, Decimal("1.08500"))
    assert signal is not None
    assert signal.target is RawDirectionalTarget.LONG


def test_move_at_or_below_threshold_produces_no_signal() -> None:
    state = EurusdLiquidityShockReversionState(_params(multiple="3"))
    t = _warmup(
        state, 0, ["1.10001", "1.10002", "1.10003", "1.10004", "1.10005", "1.10006"]
    )
    t += _MIN
    # baseline scale is ~1e-5; a ~1e-4 move is far below a 3x threshold.
    signal = state.on_minute_close(t, t + 1, Decimal("1.10007"))
    assert signal is None


def test_signals_are_ignored_while_position_or_pending() -> None:
    state = EurusdLiquidityShockReversionState(_params(hold=5))
    t = _warmup(
        state, 0, ["1.10001", "1.10002", "1.10003", "1.10004", "1.10005", "1.10006"]
    )
    t += _MIN
    first = state.on_minute_close(t, t + 1, Decimal("1.11500"))
    assert first is not None
    # A second shock while the first is still pending must be ignored.
    t += _MIN
    second = state.on_minute_close(t, t + 1, Decimal("1.30000"))
    assert second is None
    executable = state.on_execution_frame(t, t + 1)
    assert executable is not None and executable.target is RawDirectionalTarget.SHORT
    # And ignored again once the position is open.
    t += _MIN
    third = state.on_minute_close(t, t + 1, Decimal("1.50000"))
    assert third is None


def test_entry_executes_strictly_after_signal_and_exit_after_hold_completes() -> None:
    state = EurusdLiquidityShockReversionState(_params(hold=2))
    t = _warmup(
        state, 0, ["1.10001", "1.10002", "1.10003", "1.10004", "1.10005", "1.10006"]
    )
    t += _MIN
    signal = state.on_minute_close(t, t + 1, Decimal("1.11500"))
    assert signal is not None
    # No entry on the same frame as the signal.
    assert state.on_execution_frame(t, t + 1) is None

    t += _MIN
    state.on_minute_close(t, t + 1, Decimal("1.11501"))
    entry = state.on_execution_frame(t, t + 1)
    assert entry is not None
    assert entry.target is RawDirectionalTarget.SHORT
    assert entry.execution_information_ns > signal.signal_information_ns
    after_entry = state.latest_target
    assert after_entry is RawDirectionalTarget.SHORT

    # Hold for exactly `hold` eligible minutes: no flatten on those frames.
    for _ in range(2):
        t += _MIN
        state.on_minute_close(t, t + 1, Decimal("1.11501"))
        assert state.on_execution_frame(t, t + 1) is None
        during_hold = state.latest_target
        assert during_hold is RawDirectionalTarget.SHORT

    # The next frame strictly after the hold count is reached flattens.
    t += _MIN
    state.on_minute_close(t, t + 1, Decimal("1.11501"))
    exit_signal = state.on_execution_frame(t, t + 1)
    assert exit_signal is not None
    assert exit_signal.target is RawDirectionalTarget.FLAT
    after_exit = state.latest_target
    assert after_exit is RawDirectionalTarget.FLAT
    assert state.has_open_or_pending_position is False


def test_missing_minute_gap_resets_the_return_chain_not_the_position() -> None:
    state = EurusdLiquidityShockReversionState(_params(hold=10))
    t = _warmup(
        state, 0, ["1.10001", "1.10002", "1.10003", "1.10004", "1.10005", "1.10006"]
    )
    t += _MIN
    signal = state.on_minute_close(t, t + 1, Decimal("1.11500"))
    assert signal is not None
    t += _MIN
    state.on_minute_close(t, t + 1, Decimal("1.11501"))
    entry = state.on_execution_frame(t, t + 1)
    assert entry is not None

    # A gap of 3 minutes (missing data): the return chain resets, but the
    # open position and hold clock are untouched.
    t += 3 * _MIN
    gapped = state.on_minute_close(t, t + 1, Decimal("1.20000"))
    assert gapped is None  # gap breaks the chain; no shock can be detected here
    assert state.latest_target is RawDirectionalTarget.SHORT
    executable = state.on_execution_frame(t, t + 1)
    assert executable is None  # still holding, hold=10 not reached yet


def test_reset_clears_all_state_at_a_fold_boundary() -> None:
    state = EurusdLiquidityShockReversionState(_params(hold=2))
    t = _warmup(
        state, 0, ["1.10001", "1.10002", "1.10003", "1.10004", "1.10005", "1.10006"]
    )
    t += _MIN
    signal = state.on_minute_close(t, t + 1, Decimal("1.11500"))
    assert signal is not None

    state.reset()

    assert state.latest_target is RawDirectionalTarget.FLAT
    assert state.has_open_or_pending_position is False
    # Must re-warm the baseline window from scratch after a reset.
    t2 = _warmup(state, t, ["1.10001", "1.10002", "1.10003", "1.10004"])
    t2 += _MIN
    assert state.on_minute_close(t2, t2 + 1, Decimal("1.11500")) is None


def test_information_times_must_be_strictly_ordered() -> None:
    state = EurusdLiquidityShockReversionState(_params())
    state.on_minute_close(_MIN, _MIN + 1, Decimal("1.10000"))
    with pytest.raises(EurusdLiquidityShockReversionValidationError):
        state.on_minute_close(_MIN, _MIN + 1, Decimal("1.10001"))
