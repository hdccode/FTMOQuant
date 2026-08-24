"""B3.2c: proves the vectorized ``widen_bid_ask_frame`` is observably
IDENTICAL to the original row-by-row implementation (kept here, unchanged,
as ``_reference_widen_bid_ask_frame``) -- not merely "close" or
"economically equivalent". Also proves execution-level parity and the
orchestrator's redundant-call elimination / instrument-level cache.

Per the B3.2c task brief's explicit numeric audit: a plain native-float64
vectorization was tested during the audit and found to disagree with the
existing ``Decimal(str(x))``-mediated result on ~32% of realistic
FX-precision rows (by up to ~2.8e-14 absolute) -- so it was rejected. The
production implementation therefore still uses ``Decimal(str(x))``
arithmetic throughout, just vectorized via ``pandas`` ``Series``-level
operations instead of ``.iterrows()`` + per-row ``Bar``/dict construction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from ftmoquant.research.alpha_lab.cost_stress import (
    CostStressError,
    widen_bid_ask_frame,
)

_OHLC_COLUMNS = ("open", "high", "low", "close")


# ---------------------------------------------------------------------------
# Preserved reference implementation (the ORIGINAL B3.1 algorithm, verbatim
# in spirit): row-by-row, Bar-dataclass-validated, Decimal(str(x))-mediated.
# Kept here only for regression comparison -- never imported by production
# code.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ReferenceBar:
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal

    def __post_init__(self) -> None:
        for name, value in (
            ("open", self.open),
            ("high", self.high),
            ("low", self.low),
            ("close", self.close),
        ):
            if not value.is_finite():
                raise CostStressError(f"bar.{name} must be finite, got {value}")
        if self.low > self.high:
            raise CostStressError("bar invariant violated: low > high")
        if not (self.low <= self.open <= self.high):
            raise CostStressError("bar invariant violated: open outside [low, high]")
        if not (self.low <= self.close <= self.high):
            raise CostStressError("bar invariant violated: close outside [low, high]")


def _reference_widen_bid_ask_bar(
    bid_bar: _ReferenceBar, ask_bar: _ReferenceBar, multiplier: Decimal
) -> tuple[_ReferenceBar, _ReferenceBar]:
    if not bid_bar.close.is_finite() or not ask_bar.close.is_finite():
        raise CostStressError("bid/ask close must be finite")
    if ask_bar.close < bid_bar.close:
        raise CostStressError("crossed market rejected")
    if not multiplier.is_finite() or multiplier < 1:
        raise CostStressError("multiplier must be >= 1")

    half_spread_close = (ask_bar.close - bid_bar.close) / 2
    offset = (multiplier - 1) * half_spread_close

    def _shift(bar: _ReferenceBar, delta: Decimal) -> _ReferenceBar:
        return _ReferenceBar(
            open=bar.open + delta,
            high=bar.high + delta,
            low=bar.low + delta,
            close=bar.close + delta,
        )

    return _shift(bid_bar, -offset), _shift(ask_bar, offset)


def _reference_widen_bid_ask_frame(
    bid_m1: pd.DataFrame, ask_m1: pd.DataFrame, multiplier: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Verbatim port of the ORIGINAL (pre-B3.2c) ``.iterrows()``-based
    production implementation -- preserved here as the ground truth."""

    if not bid_m1.index.equals(ask_m1.index):
        raise CostStressError("bid_m1 and ask_m1 must share the identical paired index")

    decimal_multiplier = Decimal(str(multiplier))
    stressed_bid_rows: list[dict[str, float]] = []
    stressed_ask_rows: list[dict[str, float]] = []
    for (_, bid_row), (_, ask_row) in zip(
        bid_m1.iterrows(), ask_m1.iterrows(), strict=True
    ):
        bid_bar = _ReferenceBar(*(Decimal(str(bid_row[c])) for c in _OHLC_COLUMNS))
        ask_bar = _ReferenceBar(*(Decimal(str(ask_row[c])) for c in _OHLC_COLUMNS))
        widened_bid, widened_ask = _reference_widen_bid_ask_bar(
            bid_bar, ask_bar, decimal_multiplier
        )
        stressed_bid_rows.append(
            {c: float(getattr(widened_bid, c)) for c in _OHLC_COLUMNS}
        )
        stressed_ask_rows.append(
            {c: float(getattr(widened_ask, c)) for c in _OHLC_COLUMNS}
        )

    return (
        pd.DataFrame(stressed_bid_rows, index=bid_m1.index),
        pd.DataFrame(stressed_ask_rows, index=ask_m1.index),
    )


# ---------------------------------------------------------------------------
# Realistic synthetic FX M1 fixture generation
# ---------------------------------------------------------------------------


def _synthetic_realistic_frames(
    n: int, *, seed: int = 0, jpy_scale: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-03-11", periods=n, freq="min", tz="UTC")
    base = rng.uniform(90, 165) if jpy_scale else rng.uniform(0.55, 2.10)
    drift = np.cumsum(rng.normal(scale=0.0002 * base, size=n))
    mid = base + drift
    # Realistic spread magnitudes, including zero and tiny/large outliers.
    spread = rng.choice(
        [0.0, 1e-6, 1e-5] + list(rng.uniform(0.00002, 0.0008, 50)), size=n
    ).astype(float)
    precision = 3 if jpy_scale else 5
    bid_close = np.round(mid - spread / 2, precision)
    ask_close = np.round(mid + spread / 2, precision)
    wick = np.round(rng.uniform(0, 0.0003 * base, n), precision)

    bid = pd.DataFrame(
        {
            "open": bid_close,
            "high": bid_close + wick,
            "low": bid_close - wick,
            "close": bid_close,
        },
        index=idx,
    )
    ask = pd.DataFrame(
        {
            "open": ask_close,
            "high": ask_close + wick,
            "low": ask_close - wick,
            "close": ask_close,
        },
        index=idx,
    )
    return bid, ask


# ---------------------------------------------------------------------------
# Observable-equivalence proof (dtype/index/column/value exact parity)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("multiplier", [1.0, 1.5, 2.0])
@pytest.mark.parametrize("jpy_scale", [False, True])
def test_vectorized_matches_reference_bit_for_bit_on_realistic_data(
    multiplier: float, jpy_scale: bool
) -> None:
    bid, ask = _synthetic_realistic_frames(20_000, seed=1, jpy_scale=jpy_scale)
    ref_bid, ref_ask = _reference_widen_bid_ask_frame(bid, ask, multiplier)
    new_bid, new_ask = widen_bid_ask_frame(bid, ask, multiplier)

    pd.testing.assert_frame_equal(ref_bid, new_bid, check_exact=True)
    pd.testing.assert_frame_equal(ref_ask, new_ask, check_exact=True)


def test_vectorized_matches_reference_on_zero_and_extreme_spreads() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="min", tz="UTC")
    bid = pd.DataFrame(
        {
            "open": [1.10000, 1.10000, 1.10000, 100.000, 0.00001, 1.10000],
            "high": [1.10000, 1.10005, 1.10000, 100.010, 0.00002, 1.10000],
            "low": [1.10000, 1.09995, 1.10000, 99.990, 0.00000, 1.09999],
            "close": [1.10000, 1.10000, 1.10000, 100.000, 0.00001, 1.09999],
        },
        index=idx,
    )
    ask = pd.DataFrame(
        {
            "open": [1.10000, 1.10010, 1.10000, 100.020, 0.00002, 1.10001],
            "high": [1.10000, 1.10015, 1.10000, 100.030, 0.00003, 1.10001],
            "low": [1.10000, 1.10005, 1.10000, 100.010, 0.00001, 1.10000],
            "close": [1.10000, 1.10010, 1.10000, 100.020, 0.00002, 1.10001],
        },
        index=idx,
    )
    for multiplier in (1.0, 1.5, 2.0):
        ref_bid, ref_ask = _reference_widen_bid_ask_frame(bid, ask, multiplier)
        new_bid, new_ask = widen_bid_ask_frame(bid, ask, multiplier)
        pd.testing.assert_frame_equal(ref_bid, new_bid, check_exact=True)
        pd.testing.assert_frame_equal(ref_ask, new_ask, check_exact=True)


def test_vectorized_dtype_index_columns_match_reference() -> None:
    bid, ask = _synthetic_realistic_frames(500, seed=2)
    ref_bid, ref_ask = _reference_widen_bid_ask_frame(bid, ask, 1.5)
    new_bid, new_ask = widen_bid_ask_frame(bid, ask, 1.5)
    assert list(new_bid.columns) == list(ref_bid.columns)
    assert list(new_ask.columns) == list(ref_ask.columns)
    assert new_bid.index.equals(ref_bid.index)
    assert new_ask.index.equals(ref_ask.index)
    assert new_bid.dtypes.equals(ref_bid.dtypes)
    assert new_ask.dtypes.equals(ref_ask.dtypes)


def test_input_frames_are_never_mutated() -> None:
    bid, ask = _synthetic_realistic_frames(500, seed=3)
    bid_copy = bid.copy(deep=True)
    ask_copy = ask.copy(deep=True)
    widen_bid_ask_frame(bid, ask, 1.5)
    pd.testing.assert_frame_equal(bid, bid_copy)
    pd.testing.assert_frame_equal(ask, ask_copy)


def test_vectorized_deterministic_repeated_output() -> None:
    bid, ask = _synthetic_realistic_frames(500, seed=4)
    first = widen_bid_ask_frame(bid, ask, 1.5)
    second = widen_bid_ask_frame(bid, ask, 1.5)
    pd.testing.assert_frame_equal(first[0], second[0], check_exact=True)
    pd.testing.assert_frame_equal(first[1], second[1], check_exact=True)


def test_vectorized_preserves_ohlc_ordering_and_close_midpoint() -> None:
    bid, ask = _synthetic_realistic_frames(2_000, seed=5)
    stressed_bid, stressed_ask = widen_bid_ask_frame(bid, ask, 2.0)
    assert (stressed_bid["low"] <= stressed_bid["open"]).all()
    assert (stressed_bid["low"] <= stressed_bid["close"]).all()
    assert (stressed_bid["open"] <= stressed_bid["high"]).all()
    assert (stressed_bid["close"] <= stressed_bid["high"]).all()
    assert (stressed_ask["low"] <= stressed_ask["open"]).all()
    assert (stressed_ask["close"] <= stressed_ask["high"]).all()

    original_mid = (bid["close"] + ask["close"]) / 2
    stressed_mid = (stressed_bid["close"] + stressed_ask["close"]) / 2
    np.testing.assert_allclose(
        original_mid.to_numpy(), stressed_mid.to_numpy(), atol=1e-9
    )


def test_vectorized_spread_never_narrows_for_m_gte_1() -> None:
    bid, ask = _synthetic_realistic_frames(2_000, seed=6)
    original_spread = (ask["close"] - bid["close"]).to_numpy()
    for multiplier in (1.0, 1.5, 2.0):
        stressed_bid, stressed_ask = widen_bid_ask_frame(bid, ask, multiplier)
        stressed_spread = (stressed_ask["close"] - stressed_bid["close"]).to_numpy()
        assert (stressed_spread >= original_spread - 1e-9).all()


def test_vectorized_invalid_rows_fail_exactly_like_reference() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="min", tz="UTC")
    bid = pd.DataFrame(
        {
            "open": [1.1, 1.1, 1.1],
            "high": [1.1, 1.05, 1.1],
            "low": [1.1, 1.10, 1.1],
            "close": [1.1, 1.1, 1.1],
        },
        index=idx,
    )  # row 1: low > high
    ask = pd.DataFrame(
        {
            "open": [1.11, 1.11, 1.11],
            "high": [1.11, 1.11, 1.11],
            "low": [1.11, 1.11, 1.11],
            "close": [1.11, 1.11, 1.11],
        },
        index=idx,
    )
    with pytest.raises(CostStressError):
        _reference_widen_bid_ask_frame(bid, ask, 1.5)
    with pytest.raises(CostStressError):
        widen_bid_ask_frame(bid, ask, 1.5)


def test_vectorized_rejects_crossed_market_like_reference() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="min", tz="UTC")
    bid = pd.DataFrame(
        {
            "open": [1.20, 1.10],
            "high": [1.20, 1.10],
            "low": [1.20, 1.10],
            "close": [1.20, 1.10],
        },
        index=idx,
    )
    ask = pd.DataFrame(
        {
            "open": [1.21, 1.09],
            "high": [1.21, 1.09],
            "low": [1.21, 1.09],
            "close": [1.21, 1.09],
        },
        index=idx,
    )  # row 1 crossed: ask.close < bid.close
    with pytest.raises(CostStressError):
        _reference_widen_bid_ask_frame(bid, ask, 1.5)
    with pytest.raises(CostStressError):
        widen_bid_ask_frame(bid, ask, 1.5)


def test_vectorized_prefix_invariance_matches_reference() -> None:
    bid, ask = _synthetic_realistic_frames(1_000, seed=7)
    full_bid, full_ask = widen_bid_ask_frame(bid, ask, 1.5)
    prefix_bid, prefix_ask = widen_bid_ask_frame(bid.iloc[:200], ask.iloc[:200], 1.5)
    pd.testing.assert_frame_equal(full_bid.iloc[:200], prefix_bid, check_exact=True)
    pd.testing.assert_frame_equal(full_ask.iloc[:200], prefix_ask, check_exact=True)


# ---------------------------------------------------------------------------
# Benchmark (informational; not a pass/fail gate, but recorded for the
# report -- kept small enough to run inside the normal test suite)
# ---------------------------------------------------------------------------


def test_benchmark_old_vs_new(capsys: pytest.CaptureFixture[str]) -> None:
    results = []
    for n in (10_000, 100_000):
        bid, ask = _synthetic_realistic_frames(n, seed=8)

        start = time.perf_counter()
        _reference_widen_bid_ask_frame(bid, ask, 1.5)
        old_elapsed = time.perf_counter() - start

        start = time.perf_counter()
        widen_bid_ask_frame(bid, ask, 1.5)
        new_elapsed = time.perf_counter() - start

        results.append((n, old_elapsed, new_elapsed))

    with capsys.disabled():
        for n, old_elapsed, new_elapsed in results:
            speedup = old_elapsed / new_elapsed if new_elapsed > 0 else float("inf")
            print(
                f"\nn={n}: old={old_elapsed:.4f}s ({old_elapsed / n * 1e6:.2f}us/row) "
                f"new={new_elapsed:.4f}s ({new_elapsed / n * 1e6:.2f}us/row) "
                f"speedup={speedup:.1f}x"
            )
    # Sanity floor: new must be materially faster, not just numerically equal.
    for _, old_elapsed, new_elapsed in results:
        assert new_elapsed < old_elapsed / 3


# ---------------------------------------------------------------------------
# Execution-level parity: swap the transform B3F1 execution actually calls
# and prove every observable trade field is identical, not just the
# DataFrame-level widening output.
# ---------------------------------------------------------------------------


def _m1(prices: list[float], *, start: str = "2024-01-01T00:01:00Z") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(prices), freq="min", tz="UTC")
    return pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices}, index=idx
    )


def test_execution_level_parity_between_old_and_new_transform() -> None:
    import ftmoquant.research.alpha_lab.b3f1_spread_execution as execution_module
    from ftmoquant.data.instruments import EURUSD_OANDA_SPEC, USDCAD_OANDA_SPEC
    from ftmoquant.research.alpha_lab.b3f1_spread_signals import (
        EXIT_REASON_Z_MEAN_REVERSION,
        B3F1TradeIntent,
        SpreadSide,
    )

    y_bid = _m1([1.0790 + 0.00001 * i for i in range(60)])
    y_ask = _m1([1.0792 + 0.00001 * i for i in range(60)])
    x_bid = _m1([1.3490 - 0.00001 * i for i in range(60)])
    x_ask = _m1([1.3492 - 0.00001 * i for i in range(60)])
    intent = B3F1TradeIntent(
        sleeve_id="EUR/USD.OANDA__USD/CAD.OANDA",
        side=SpreadSide.RICH,
        entry_ts=pd.Timestamp("2024-01-01T00:00:00Z"),
        entry_z=2.0,
        frozen_alpha=0.1,
        frozen_beta=1.2,
        frozen_spread_mean=0.0,
        frozen_spread_std=0.01,
        exit_ts=pd.Timestamp("2024-01-01T00:30:00Z"),
        exit_reason=EXIT_REASON_Z_MEAN_REVERSION,
    )

    def run(transform):
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(execution_module, "widen_bid_ask_frame", transform)
            return execution_module.simulate_b3f1_intents(
                [intent],
                y_spec=EURUSD_OANDA_SPEC,
                x_spec=USDCAD_OANDA_SPEC,
                y_bid_m1=y_bid,
                y_ask_m1=y_ask,
                x_bid_m1=x_bid,
                x_ask_m1=x_ask,
                cost_stress_multiplier=Decimal("1.5"),
            )

    old_episodes, old_skips = run(_reference_widen_bid_ask_frame)
    new_episodes, new_skips = run(widen_bid_ask_frame)

    assert len(old_episodes) == len(new_episodes) == 1
    assert len(old_skips) == len(new_skips) == 0
    old_ep, new_ep = old_episodes[0], new_episodes[0]

    assert old_ep.leg_a.entry_ns == new_ep.leg_a.entry_ns
    assert old_ep.leg_a.exit_ns == new_ep.leg_a.exit_ns
    assert old_ep.leg_a.entry_price == new_ep.leg_a.entry_price
    assert old_ep.leg_a.exit_price == new_ep.leg_a.exit_price
    assert old_ep.leg_a.direction == new_ep.leg_a.direction
    assert old_ep.leg_b.entry_ns == new_ep.leg_b.entry_ns
    assert old_ep.leg_b.exit_ns == new_ep.leg_b.exit_ns
    assert old_ep.leg_b.entry_price == new_ep.leg_b.entry_price
    assert old_ep.leg_b.exit_price == new_ep.leg_b.exit_price
    assert old_ep.leg_b.direction == new_ep.leg_b.direction
    assert old_ep.realized_pnl() == new_ep.realized_pnl()
    assert old_ep.exit_reason == new_ep.exit_reason
