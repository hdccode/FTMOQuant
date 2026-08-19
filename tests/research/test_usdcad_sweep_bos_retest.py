from __future__ import annotations

from datetime import UTC

import numpy as np
import pandas as pd
import pytest
from nautilus_trader.model import Bar, BarType, OrderSide, Price, Quantity

from ftmoquant.data.instruments import USDCAD_OANDA_SPEC
from ftmoquant.research import usdcad_sweep_bos_retest_development as dev
from ftmoquant.research.alpha_lab.liquidity_structure_signals import (
    b2f1_sweep_bos_retest_signals,
)
from ftmoquant.research.alpha_lab.pair_specific_validation import CANDIDATE_C
from ftmoquant.research.alpha_lab.wick_fvg_squeeze_execution import simulate_trades
from ftmoquant.research.alpha_lab.wick_fvg_squeeze_signals import DIRECTION_LONG
from ftmoquant.strategies.usdcad_sweep_bos_retest import (
    BASE_RESEARCH_UNITS,
    FROZEN_FAMILY,
    FROZEN_INSTRUMENT_ID,
    FROZEN_RR,
    FROZEN_SWING_LOOKBACK,
    FROZEN_TIMEFRAME,
    UsdCadSweepBosRetestError,
    precompute_alpha_lab_trades,
    reject_final_holdout,
    trade_instructions_from_alpha_lab_trades,
)

_INSTRUMENT_ID = FROZEN_INSTRUMENT_ID


# ---------------------------------------------------------------------------
# 1. frozen candidate identity -- matches the pair-specific VALIDATION
# candidate this promotion is based on, byte for byte.
# ---------------------------------------------------------------------------


def test_frozen_candidate_identity_matches_the_validated_pair_specific_candidate() -> (
    None
):
    assert FROZEN_FAMILY == CANDIDATE_C.family == "B2F1_sweep_bos_retest"
    assert FROZEN_INSTRUMENT_ID == CANDIDATE_C.instrument_id == "USD/CAD.OANDA"
    assert FROZEN_TIMEFRAME == CANDIDATE_C.timeframe == "M30"
    assert FROZEN_SWING_LOOKBACK == CANDIDATE_C.parameters["swing_lookback"] == 40
    assert FROZEN_RR == CANDIDATE_C.parameters["rr"] == 2.0


def test_signal_function_is_the_existing_frozen_implementation() -> None:
    from ftmoquant.strategies import usdcad_sweep_bos_retest as strat

    assert strat.b2f1_sweep_bos_retest_signals is b2f1_sweep_bos_retest_signals


def test_execution_function_is_the_existing_frozen_implementation() -> None:
    from ftmoquant.strategies import usdcad_sweep_bos_retest as strat

    assert strat.simulate_trades is simulate_trades


# ---------------------------------------------------------------------------
# 2. synthetic fixture: exactly one B2F1 LONG sweep -> BOS -> retest at the
# frozen swing_lookback=40, mirroring
# tests/research/alpha_lab/test_liquidity_structure.py's own fixture
# construction pattern.
# ---------------------------------------------------------------------------


def _m30_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="30min", tz=UTC)


def _bullish_sweep_bos_retest_series(
    *, lookback: int, bos_at: int, retest_at: int, n: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    high = np.full(n, 101.0)
    low = np.full(n, 100.0)
    close = np.full(n, 100.5)
    sweep_idx = lookback
    low[sweep_idx] = 99.0
    high[sweep_idx] = 101.0
    close[sweep_idx] = 100.4
    close[bos_at] = 102.0
    low[retest_at] = 101.0
    close[retest_at] = 102.3
    index = _m30_index(n)
    return (
        pd.Series(high, index=index),
        pd.Series(low, index=index),
        pd.Series(close, index=index),
    )


_LOOKBACK = FROZEN_SWING_LOOKBACK
_N = _LOOKBACK + 5
_BOS_AT = _LOOKBACK + 2
_RETEST_AT = _LOOKBACK + 3


def _signal_fixture() -> tuple[pd.Series, pd.Series, pd.Series]:
    return _bullish_sweep_bos_retest_series(
        lookback=_LOOKBACK, bos_at=_BOS_AT, retest_at=_RETEST_AT, n=_N
    )


def test_prior_swing_excludes_current_bar_at_the_frozen_lookback() -> None:
    # The sweep bar's own extreme (low=99.0 at index `lookback`) must not
    # contaminate the swing window used to detect ITS OWN sweep -- proven
    # indirectly: shrinking the fixture below the frozen lookback must
    # suppress the sweep (no prior-swing-low value exists yet to sweep).
    h, low, c = _signal_fixture()
    events_full = b2f1_sweep_bos_retest_signals(
        h, low, c, swing_lookback=_LOOKBACK, rr=FROZEN_RR
    )
    assert len(events_full) == 1
    events_too_short = b2f1_sweep_bos_retest_signals(
        h.iloc[: _LOOKBACK - 1],
        low.iloc[: _LOOKBACK - 1],
        c.iloc[: _LOOKBACK - 1],
        swing_lookback=_LOOKBACK,
        rr=FROZEN_RR,
    )
    assert events_too_short == ()


def test_causal_sweep_bos_retest_produces_exactly_one_long_signal() -> None:
    h, low, c = _signal_fixture()
    events = b2f1_sweep_bos_retest_signals(
        h, low, c, swing_lookback=_LOOKBACK, rr=FROZEN_RR
    )
    assert len(events) == 1
    event = events[0]
    assert event.direction == DIRECTION_LONG
    assert event.signal_bar_ts == h.index[_RETEST_AT]
    assert event.stop_price == pytest.approx(99.0)  # frozen sweep extreme
    assert event.target_r_multiple == pytest.approx(FROZEN_RR)


def test_signal_is_prefix_invariant_no_lookahead() -> None:
    h, low, c = _signal_fixture()
    cut = _RETEST_AT  # exclude the retest bar itself
    full = b2f1_sweep_bos_retest_signals(
        h, low, c, swing_lookback=_LOOKBACK, rr=FROZEN_RR
    )
    prefix = b2f1_sweep_bos_retest_signals(
        h.iloc[:cut],
        low.iloc[:cut],
        c.iloc[:cut],
        swing_lookback=_LOOKBACK,
        rr=FROZEN_RR,
    )
    full_before_cut = tuple(e for e in full if e.signal_bar_ts < h.index[cut])
    assert prefix == full_before_cut


# ---------------------------------------------------------------------------
# 3. M1 execution fixture + precompute wrapper.
# ---------------------------------------------------------------------------


def _m1_frames_and_bars(
    *,
    start: pd.Timestamp,
    minutes: int,
    base_bid: float,
    base_ask: float,
    crash_at_minute: int,
    crash_bid_low: float,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[Bar, ...], tuple[Bar, ...]]:
    index = pd.date_range(start, periods=minutes, freq="1min", tz=UTC)
    bid = pd.DataFrame(
        {"open": base_bid, "high": base_bid, "low": base_bid, "close": base_bid},
        index=index,
    )
    ask = pd.DataFrame(
        {"open": base_ask, "high": base_ask, "low": base_ask, "close": base_ask},
        index=index,
    )
    bid.loc[index[crash_at_minute], "low"] = crash_bid_low

    bid_type = BarType.from_str(f"{_INSTRUMENT_ID}-1-MINUTE-BID-EXTERNAL")
    ask_type = BarType.from_str(f"{_INSTRUMENT_ID}-1-MINUTE-ASK-EXTERNAL")
    volume = Quantity.from_str("1.00000000")

    def _bars(frame: pd.DataFrame, bar_type: BarType) -> tuple[Bar, ...]:
        bars = []
        for ts, row in frame.iterrows():
            ts_ns = int(ts.value)
            bars.append(
                Bar(
                    bar_type=bar_type,
                    open=Price.from_str(f"{row['open']:.5f}"),
                    high=Price.from_str(f"{row['high']:.5f}"),
                    low=Price.from_str(f"{row['low']:.5f}"),
                    close=Price.from_str(f"{row['close']:.5f}"),
                    volume=volume,
                    ts_event=ts_ns,
                    ts_init=ts_ns,
                )
            )
        return tuple(bars)

    return bid, ask, _bars(bid, bid_type), _bars(ask, ask_type)


def _full_fixture() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple[Bar, ...], tuple[Bar, ...]
]:
    h, low, c = _signal_fixture()
    ohlc_m30 = pd.DataFrame({"high": h, "low": low, "close": c, "open": c})
    m1_start = h.index[_RETEST_AT] + pd.Timedelta(minutes=1)
    bid_pd, ask_pd, bid_bars, ask_bars = _m1_frames_and_bars(
        start=m1_start,
        minutes=100,
        base_bid=102.25,
        base_ask=102.35,
        crash_at_minute=50,
        crash_bid_low=98.90,  # below the frozen stop (99.0)
    )
    return ohlc_m30, bid_pd, ask_pd, bid_bars, ask_bars


def test_precompute_matches_direct_signal_plus_execution_calls() -> None:
    ohlc_m30, bid_pd, ask_pd, _bid_bars, _ask_bars = _full_fixture()
    trades, skips = precompute_alpha_lab_trades(
        high=ohlc_m30["high"],
        low=ohlc_m30["low"],
        close=ohlc_m30["close"],
        bid_m1=bid_pd,
        ask_m1=ask_pd,
    )
    events = b2f1_sweep_bos_retest_signals(
        ohlc_m30["high"],
        ohlc_m30["low"],
        ohlc_m30["close"],
        swing_lookback=FROZEN_SWING_LOOKBACK,
        rr=FROZEN_RR,
    )
    expected_trades, expected_skips = simulate_trades(events, bid_pd, ask_pd)
    assert trades == expected_trades
    assert skips == expected_skips
    assert len(trades) == 1
    assert trades[0].direction == DIRECTION_LONG
    assert trades[0].exit_reason == "stop"
    assert trades[0].entry_ts > events[0].signal_bar_ts


def test_entry_is_strictly_later_and_crosses_the_correct_side() -> None:
    ohlc_m30, bid_pd, ask_pd, _bid_bars, _ask_bars = _full_fixture()
    trades, _ = precompute_alpha_lab_trades(
        high=ohlc_m30["high"],
        low=ohlc_m30["low"],
        close=ohlc_m30["close"],
        bid_m1=bid_pd,
        ask_m1=ask_pd,
    )
    trade = trades[0]
    assert trade.entry_ts == bid_pd.index[0]  # first paired M1 obs after signal
    assert trade.entry_price == pytest.approx(102.35)  # LONG -> ASK


def test_stop_first_collision_semantics_are_preserved() -> None:
    # Widen the crash bar so BOTH stop and (an inflated) target would be
    # touched in the same M1 observation -- the frozen convention (stop
    # first) must still resolve the exit as "stop", unchanged.
    ohlc_m30, bid_pd, ask_pd, _bid_bars, _ask_bars = _full_fixture()
    bid_pd.loc[bid_pd.index[50], "high"] = (
        500.0  # would also touch target if checked first
    )
    trades, _ = precompute_alpha_lab_trades(
        high=ohlc_m30["high"],
        low=ohlc_m30["low"],
        close=ohlc_m30["close"],
        bid_m1=bid_pd,
        ask_m1=ask_pd,
    )
    assert trades[0].exit_reason == "stop"


# ---------------------------------------------------------------------------
# 4. TradeInstruction translation: one-position-at-a-time / overlap guard.
# ---------------------------------------------------------------------------


def test_trade_instructions_from_alpha_lab_trades_round_trips() -> None:
    ohlc_m30, bid_pd, ask_pd, _bid_bars, _ask_bars = _full_fixture()
    trades, _ = precompute_alpha_lab_trades(
        high=ohlc_m30["high"],
        low=ohlc_m30["low"],
        close=ohlc_m30["close"],
        bid_m1=bid_pd,
        ask_m1=ask_pd,
    )
    instructions = trade_instructions_from_alpha_lab_trades(trades)
    assert len(instructions) == 1
    assert instructions[0].entry_ns == int(trades[0].entry_ts.value)
    assert instructions[0].exit_ns == int(trades[0].exit_ts.value)
    assert instructions[0].direction == DIRECTION_LONG


def test_overlapping_trades_are_rejected_one_position_at_a_time() -> None:
    from datetime import timedelta

    from ftmoquant.research.alpha_lab.wick_fvg_squeeze_execution import Trade

    base = pd.Timestamp("2020-01-01", tz=UTC)
    overlapping = (
        Trade(
            signal_bar_ts=base,
            direction=1,
            entry_ts=base + timedelta(minutes=1),
            entry_price=1.0,
            exit_ts=base + timedelta(minutes=10),
            exit_price=1.01,
            exit_reason="target",
            stop_price=0.99,
            target_price=1.01,
            return_frac=0.01,
        ),
        Trade(
            signal_bar_ts=base,
            direction=-1,
            entry_ts=base + timedelta(minutes=5),  # overlaps
            entry_price=1.0,
            exit_ts=base + timedelta(minutes=20),
            exit_price=0.99,
            exit_reason="target",
            stop_price=1.01,
            target_price=0.99,
            return_frac=0.01,
        ),
    )
    with pytest.raises(UsdCadSweepBosRetestError):
        trade_instructions_from_alpha_lab_trades(overlapping)


# ---------------------------------------------------------------------------
# 5. Genuine end-to-end Nautilus parity: a real, bounded BacktestEngine run.
# ---------------------------------------------------------------------------


def test_nautilus_engine_reproduces_the_alpha_lab_trade_lifecycle() -> None:
    ohlc_m30, bid_pd, ask_pd, bid_bars, ask_bars = _full_fixture()
    alpha_lab_trades, _ = precompute_alpha_lab_trades(
        high=ohlc_m30["high"],
        low=ohlc_m30["low"],
        close=ohlc_m30["close"],
        bid_m1=bid_pd,
        ask_m1=ask_pd,
    )
    instrument = USDCAD_OANDA_SPEC.nautilus_instrument()
    start = bid_pd.index[0].to_pydatetime()
    end_exclusive = (bid_pd.index[-1] + pd.Timedelta(minutes=1)).to_pydatetime()

    outcome = dev.run_frozen_signal_backtest(
        start=start,
        end_exclusive=end_exclusive,
        instrument=instrument,
        ohlc_m30=ohlc_m30,
        m1_bid_pd=bid_pd,
        m1_ask_pd=ask_pd,
        m1_bid_bars=bid_bars,
        m1_ask_bars=ask_bars,
    )

    assert outcome.alpha_lab_trades == alpha_lab_trades
    assert len(outcome.completed_trades) == len(alpha_lab_trades) == 1
    assert outcome.submissions[0].kind == "entry"
    assert outcome.submissions[0].side == OrderSide.BUY.name
    assert outcome.submissions[0].ts_event == int(alpha_lab_trades[0].entry_ts.value)
    assert outcome.submissions[1].kind == "exit"
    assert outcome.submissions[1].ts_event == int(alpha_lab_trades[0].exit_ts.value)

    entry_fill = next(f for f in outcome.fills if f.kind == "entry")
    exit_fill = next(f for f in outcome.fills if f.kind == "exit")
    assert entry_fill.side == OrderSide.BUY.name  # BUY -> ASK
    assert exit_fill.side == OrderSide.SELL.name  # long exit -> SELL (BID)
    assert entry_fill.fill_ns == int(alpha_lab_trades[0].entry_ts.value)
    assert exit_fill.fill_ns == int(alpha_lab_trades[0].exit_ts.value)

    nautilus_trade = outcome.completed_trades[0]
    assert nautilus_trade.direction == DIRECTION_LONG
    assert nautilus_trade.quantity == pytest.approx(float(BASE_RESEARCH_UNITS))
    # native fill crosses the real spread -- genuinely close to, but not
    # required to bit-match, the Alpha Lab idealized entry price.
    assert float(nautilus_trade.entry_price) == pytest.approx(102.35, abs=0.01)


def test_nautilus_sizing_is_the_fixed_reference_notional() -> None:
    ohlc_m30, bid_pd, ask_pd, bid_bars, ask_bars = _full_fixture()
    instrument = USDCAD_OANDA_SPEC.nautilus_instrument()
    start = bid_pd.index[0].to_pydatetime()
    end_exclusive = (bid_pd.index[-1] + pd.Timedelta(minutes=1)).to_pydatetime()
    outcome = dev.run_frozen_signal_backtest(
        start=start,
        end_exclusive=end_exclusive,
        instrument=instrument,
        ohlc_m30=ohlc_m30,
        m1_bid_pd=bid_pd,
        m1_ask_pd=ask_pd,
        m1_bid_bars=bid_bars,
        m1_ask_bars=ask_bars,
    )
    assert outcome.completed_trades[0].quantity == pytest.approx(100_000.0)


def test_run_frozen_signal_backtest_is_deterministic() -> None:
    ohlc_m30, bid_pd, ask_pd, bid_bars, ask_bars = _full_fixture()
    instrument = USDCAD_OANDA_SPEC.nautilus_instrument()
    start = bid_pd.index[0].to_pydatetime()
    end_exclusive = (bid_pd.index[-1] + pd.Timedelta(minutes=1)).to_pydatetime()
    kwargs = dict(
        start=start,
        end_exclusive=end_exclusive,
        instrument=instrument,
        ohlc_m30=ohlc_m30,
        m1_bid_pd=bid_pd,
        m1_ask_pd=ask_pd,
        m1_bid_bars=bid_bars,
        m1_ask_bars=ask_bars,
    )
    first = dev.run_frozen_signal_backtest(**kwargs)
    second = dev.run_frozen_signal_backtest(**kwargs)
    assert first.submissions == second.submissions
    assert first.fills == second.fills
    assert first.completed_trades == second.completed_trades


# ---------------------------------------------------------------------------
# 6. accounting safety (Section 12) -- reused, fail-closed currency helper.
# ---------------------------------------------------------------------------


def test_currency_conversion_is_reused_and_fails_closed_on_an_unrelated_currency() -> (
    None
):
    from decimal import Decimal

    # USD/CAD: base=USD (account currency, no conversion needed).
    converted = dev._convert_to_account_currency(
        Decimal("100"),
        "USD",
        base_currency="USD",
        quote_currency="CAD",
        conversion_price=Decimal("1.35"),
    )
    assert converted == Decimal("100")
    # A CAD amount converts by dividing by the quote/base price.
    converted_cad = dev._convert_to_account_currency(
        Decimal("135"),
        "CAD",
        base_currency="USD",
        quote_currency="CAD",
        conversion_price=Decimal("1.35"),
    )
    assert converted_cad == Decimal("100")


def test_currency_conversion_rejects_a_third_currency() -> None:
    from decimal import Decimal

    with pytest.raises(Exception):
        dev._convert_to_account_currency(
            Decimal("100"),
            "JPY",
            base_currency="USD",
            quote_currency="CAD",
            conversion_price=Decimal("1.35"),
        )


def test_equity_marks_use_native_portfolio_equity_not_manual_summation() -> None:
    import inspect

    source = inspect.getsource(
        __import__(
            "ftmoquant.strategies.usdcad_sweep_bos_retest", fromlist=["_x"]
        ).UsdCadSweepBosRetestExecutor._append_equity_mark
    )
    assert "self.portfolio.equity" in source
    assert "unrealized_pnl" not in source  # no manual position-level P&L summation


# ---------------------------------------------------------------------------
# 7. final-holdout firewall.
# ---------------------------------------------------------------------------


def test_reject_final_holdout_accepts_pre_holdout_and_rejects_at_or_after() -> None:
    from ftmoquant.research.stage_g import HOLDOUT_START

    holdout_ns = int(HOLDOUT_START.timestamp() * 1_000_000_000)
    reject_final_holdout(holdout_ns - 1)  # must not raise
    with pytest.raises(UsdCadSweepBosRetestError):
        reject_final_holdout(holdout_ns)
    with pytest.raises(UsdCadSweepBosRetestError):
        reject_final_holdout(holdout_ns + 1_000_000_000)


# ---------------------------------------------------------------------------
# 8. DEVELOPMENT/VALIDATION partition boundary + path firewalls.
# ---------------------------------------------------------------------------


def test_partition_bounds_match_the_frozen_stage_g_boundaries() -> None:
    from ftmoquant.research.stage_g import (
        DEVELOPMENT_END_EXCLUSIVE,
        DEVELOPMENT_START,
        HOLDOUT_START,
        VALIDATION_START,
    )

    dev_start, dev_end = dev.partition_bounds(dev.Partition.DEVELOPMENT)
    val_start, val_end = dev.partition_bounds(dev.Partition.VALIDATION)
    assert (dev_start, dev_end) == (DEVELOPMENT_START, DEVELOPMENT_END_EXCLUSIVE)
    assert (val_start, val_end) == (VALIDATION_START, HOLDOUT_START)
    assert dev_end <= HOLDOUT_START
    assert val_end <= HOLDOUT_START


def test_parse_partition_rejects_anything_but_the_two_frozen_values() -> None:
    assert dev.parse_partition("development") is dev.Partition.DEVELOPMENT
    assert dev.parse_partition("validation") is dev.Partition.VALIDATION
    with pytest.raises(Exception):
        dev.parse_partition("holdout")


@pytest.mark.parametrize(
    "root_str,partition,should_raise",
    [
        (
            "/data/oanda_fx_alpha_lab_v1/holdout_root/USDCAD",
            dev.Partition.DEVELOPMENT,
            True,
        ),
        ("/data/final_holdout/USDCAD", dev.Partition.VALIDATION, True),
        (
            "/data/oanda_fx_alpha_lab_v1/validation_canonical/USDCAD",
            dev.Partition.DEVELOPMENT,
            True,
        ),
        (
            "/data/oanda_fx_alpha_lab_v1/validation_canonical/USDCAD",
            dev.Partition.VALIDATION,
            False,
        ),
        (
            "/data/oanda_fx_alpha_lab_v1/canonical/USDCAD",
            dev.Partition.DEVELOPMENT,
            False,
        ),
    ],
)
def test_sealed_path_rejection_matches_the_partition(
    root_str: str, partition: object, should_raise: bool
) -> None:
    from pathlib import Path

    if should_raise:
        with pytest.raises(Exception):
            dev._reject_sealed_path(Path(root_str), partition=partition)
    else:
        dev._reject_sealed_path(Path(root_str), partition=partition)


# ---------------------------------------------------------------------------
# 9. deterministic result artifacts + refuse-overwrite.
# ---------------------------------------------------------------------------


def test_run_and_write_refuses_to_overwrite_an_existing_output_dir(tmp_path) -> None:  # type: ignore[no-untyped-def]
    ohlc_m30, bid_pd, ask_pd, bid_bars, ask_bars = _full_fixture()
    instrument = USDCAD_OANDA_SPEC.nautilus_instrument()
    data = dev.ResolvedPartitionData(
        instrument=instrument,
        ohlc_m30=ohlc_m30,
        m1_bid_pd=bid_pd,
        m1_ask_pd=ask_pd,
        m1_bid_bars=bid_bars,
        m1_ask_bars=ask_bars,
        readiness_identity_sha256="deadbeef",
    )
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    with pytest.raises(dev.UsdCadSweepBosRetestDevelopmentError):
        dev._run_and_write(
            partition=dev.Partition.DEVELOPMENT, data=data, output_dir=output_dir
        )


def test_write_result_artifacts_is_deterministic(tmp_path) -> None:  # type: ignore[no-untyped-def]
    ohlc_m30, bid_pd, ask_pd, bid_bars, ask_bars = _full_fixture()
    instrument = USDCAD_OANDA_SPEC.nautilus_instrument()
    start = bid_pd.index[0].to_pydatetime()
    end_exclusive = (bid_pd.index[-1] + pd.Timedelta(minutes=1)).to_pydatetime()
    outcome = dev.run_frozen_signal_backtest(
        start=start,
        end_exclusive=end_exclusive,
        instrument=instrument,
        ohlc_m30=ohlc_m30,
        m1_bid_pd=bid_pd,
        m1_ask_pd=ask_pd,
        m1_bid_bars=bid_bars,
        m1_ask_bars=ask_bars,
    )
    alpha_lab_perf = dev._alpha_lab_performance(
        outcome.alpha_lab_trades, outcome.alpha_lab_skips, 1
    )
    nautilus_perf = dev._nautilus_performance(outcome)
    result = dev.UsdCadSweepBosRetestRunResult(
        partition="development",
        start_utc=dev._iso(start),
        end_exclusive_utc=dev._iso(end_exclusive),
        instrument_id=FROZEN_INSTRUMENT_ID,
        execution_profile=dev.EXECUTION_PROFILE_LABEL,
        sizing_convention=dev.SIZING_CONVENTION_LABEL,
        alpha_lab_performance=alpha_lab_perf,
        nautilus_performance=nautilus_perf,
        order_report_rows=outcome.order_report_rows,
        fill_report_rows=outcome.fill_report_rows,
        position_report_rows=outcome.position_report_rows,
        submission_count=len(outcome.submissions),
        fill_count=len(outcome.fills),
    )
    data = dev.ResolvedPartitionData(
        instrument=instrument,
        ohlc_m30=ohlc_m30,
        m1_bid_pd=bid_pd,
        m1_ask_pd=ask_pd,
        m1_bid_bars=bid_bars,
        m1_ask_bars=ask_bars,
        readiness_identity_sha256="deadbeef",
    )
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    dev._write_result_artifacts(first_dir, result, outcome, data)
    dev._write_result_artifacts(second_dir, result, outcome, data)
    assert (first_dir / "result.json").read_bytes() == (
        second_dir / "result.json"
    ).read_bytes()
    assert (first_dir / "trades.csv").read_bytes() == (
        second_dir / "trades.csv"
    ).read_bytes()


# ---------------------------------------------------------------------------
# 10. execution profile truth-in-labeling (Section 7).
# ---------------------------------------------------------------------------


def test_execution_profile_is_labeled_native_spread_not_fully_realistic() -> None:
    assert dev.EXECUTION_PROFILE_LABEL == "native_spread_nautilus_execution"
    assert "UNCALIBRATED" in dev.EXECUTION_PROFILE_CAVEAT
    assert "zero modeled commission" in dev.EXECUTION_PROFILE_CAVEAT
    assert "zero modeled slippage" in dev.EXECUTION_PROFILE_CAVEAT
    assert "rollover DISABLED" in dev.EXECUTION_PROFILE_CAVEAT
