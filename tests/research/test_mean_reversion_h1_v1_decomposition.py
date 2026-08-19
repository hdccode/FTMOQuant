from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import vectorbt as vbt
from nautilus_trader.model import Bar, BarType, Price, Quantity

import ftmoquant.research.mean_reversion_h1_development as dev
import ftmoquant.research.mean_reversion_h1_v1_decomposition as decomp
from ftmoquant.data.instruments import OANDA_ALPHA_LAB_SPECS
from ftmoquant.research.alpha_lab.families import mean_reversion_signals
from ftmoquant.research.alpha_lab.screening_common import _build_portfolio
from ftmoquant.strategies.mean_reversion_h1 import FROZEN_LOOKBACK, FROZEN_Z_ENTRY

_INSTRUMENT_ID = "EUR/USD.OANDA"
_SPEC = next(s for s in OANDA_ALPHA_LAB_SPECS if s.instrument_id == _INSTRUMENT_ID)


# ---------------------------------------------------------------------------
# No VALIDATION/holdout access, ever.
# ---------------------------------------------------------------------------


def test_catalog_and_readiness_paths_never_reference_validation_or_holdout() -> None:
    for path in (decomp.DEVELOPMENT_CATALOG_ROOT, decomp.DEVELOPMENT_READINESS_PATH):
        lowered = str(path).casefold()
        assert "validation" not in lowered
        assert "holdout" not in lowered


def test_pair_bar_loading_uses_only_the_frozen_development_window() -> None:
    source = Path(decomp.__file__).read_text(encoding="utf-8")
    assert "DEVELOPMENT_START" in source
    assert "DEVELOPMENT_END_EXCLUSIVE" in source
    assert "VALIDATION_START" not in source
    assert "HOLDOUT_START" not in source


# ---------------------------------------------------------------------------
# Signal identity: the causal position sequence used by B/C/D/E and
# VectorBT's own realized target sequence (A) must agree.
# ---------------------------------------------------------------------------


def _fabricated_price_sequence() -> list[float]:
    """30 days of small daily variation (>=20 prior completed daily
    returns) then a long entry -> mean-cross exit -> short entry pattern --
    the same proven shape used throughout
    tests/research/test_mean_reversion_h1_development.py."""

    values = []
    for day in range(30):
        for hour in range(24):
            idx = day * 24 + hour
            values.append(1.10000 + 0.00010 * ((idx % 7) - 3))
    values += [1.10000] * (FROZEN_LOOKBACK - 1) + [1.07000, 1.10000, 1.13000]
    return values


def _fabricated_dataset_close(values: list[float], start: datetime) -> pd.DataFrame:
    index = pd.date_range(start, periods=len(values), freq="1h", tz="UTC")
    return pd.DataFrame({_INSTRUMENT_ID: values}, index=index)


def test_causal_position_sequence_matches_vectorbt_target_sequence() -> None:
    values = _fabricated_price_sequence()
    start = datetime(2023, 1, 2, tzinfo=UTC)
    close = _fabricated_dataset_close(values, start)

    class _FakeDataset:
        instrument_ids = (_INSTRUMENT_ID,)

    fake = _FakeDataset()
    fake.close = close  # type: ignore[attr-defined]

    causal = decomp.causal_position_sequence(fake)[_INSTRUMENT_ID]  # type: ignore[arg-type]

    entries, exits, short_entries, short_exits = mean_reversion_signals(
        close, FROZEN_LOOKBACK, FROZEN_Z_ENTRY
    )
    portfolio = vbt.Portfolio.from_signals(
        close,
        entries,
        exits,
        short_entries=short_entries,
        short_exits=short_exits,
        fees=0.0,
        freq="1h",
    )
    flow = portfolio.asset_flow(direction="both")[_INSTRUMENT_ID].cumsum()
    vectorbt_target = tuple(int(np.sign(v)) for v in flow)

    assert causal == vectorbt_target


# ---------------------------------------------------------------------------
# The one new deterministic calculator (B/C): validated directly against
# VectorBT's own all-in sizing/reversal semantics before being trusted for
# the delayed-execution variants. Zero price drift between the H1 decision
# and the (still strictly-later, per Section 2) M1 execution isolates the
# SIZING/compounding rule from any timing-induced price difference.
# ---------------------------------------------------------------------------


def _build_h1_bars(
    values: list[float], start: datetime
) -> tuple[tuple[Bar, ...], tuple[Bar, ...]]:
    bid_type = BarType.from_str(f"{_INSTRUMENT_ID}-1-HOUR-BID-INTERNAL")
    ask_type = BarType.from_str(f"{_INSTRUMENT_ID}-1-HOUR-ASK-INTERNAL")
    volume = Quantity.from_str(f"{1:.{_SPEC.size_precision}f}")
    bid_bars, ask_bars = [], []
    for i, mid in enumerate(values):
        ts = int((start + timedelta(hours=i)).timestamp() * 1_000_000_000)
        bid_price = Price.from_str(f"{mid - 0.00010:.5f}")
        ask_price = Price.from_str(f"{mid + 0.00010:.5f}")
        bid_bars.append(
            Bar(
                bar_type=bid_type,
                open=bid_price,
                high=bid_price,
                low=bid_price,
                close=bid_price,
                volume=volume,
                ts_event=ts,
                ts_init=ts,
            )
        )
        ask_bars.append(
            Bar(
                bar_type=ask_type,
                open=ask_price,
                high=ask_price,
                low=ask_price,
                close=ask_price,
                volume=volume,
                ts_event=ts,
                ts_init=ts,
            )
        )
    return tuple(bid_bars), tuple(ask_bars)


def _build_zero_drift_m1_bars(
    values: list[float], start: datetime
) -> tuple[tuple[Bar, ...], tuple[Bar, ...]]:
    """One paired M1 observation exactly one minute after each H1 bar, at
    the IDENTICAL price -- isolates the compounding/sizing rule from any
    timing-induced price movement while still being strictly later."""

    bid_type = BarType.from_str(f"{_INSTRUMENT_ID}-1-MINUTE-BID-EXTERNAL")
    ask_type = BarType.from_str(f"{_INSTRUMENT_ID}-1-MINUTE-ASK-EXTERNAL")
    volume = Quantity.from_str(f"{1:.{_SPEC.size_precision}f}")
    bid_bars, ask_bars = [], []
    for i, mid in enumerate(values):
        ts = int(
            (start + timedelta(hours=i, minutes=1)).timestamp() * 1_000_000_000
        )
        bid_price = Price.from_str(f"{mid - 0.00010:.5f}")
        ask_price = Price.from_str(f"{mid + 0.00010:.5f}")
        bid_bars.append(
            Bar(
                bar_type=bid_type,
                open=bid_price,
                high=bid_price,
                low=bid_price,
                close=bid_price,
                volume=volume,
                ts_event=ts,
                ts_init=ts,
            )
        )
        ask_bars.append(
            Bar(
                bar_type=ask_type,
                open=ask_price,
                high=ask_price,
                low=ask_price,
                close=ask_price,
                volume=volume,
                ts_event=ts,
                ts_init=ts,
            )
        )
    return tuple(bid_bars), tuple(ask_bars)


def test_all_in_compounding_matches_vectorbt_with_zero_price_drift_and_no_spread() -> (
    None
):
    """B's own sizing/reversal rule, isolated from timing/spread, must
    reproduce VectorBT's own zero-fee total return: proof that
    ``_compound_all_in_equity`` faithfully reproduces the original
    screening's all-in/accumulate=False/reverse-on-opposite-entry
    convention, not merely something plausible."""

    values = _fabricated_price_sequence()
    start = datetime(2023, 1, 2, tzinfo=UTC)
    h1_bid, h1_ask = _build_h1_bars(values, start)
    m1_bid, m1_ask = _build_zero_drift_m1_bars(values, start)

    start_ns = h1_bid[0].ts_event
    end_ns = h1_bid[-1].ts_event + 3_600_000_000_000

    transitions = decomp._resolve_direction_transitions(
        h1_bid, h1_ask, m1_bid, m1_ask, start_ns, end_ns
    )
    assert len(transitions) == 3  # long entry, mean-cross exit, short entry

    points = decomp._compound_all_in_equity(
        h1_bid,
        h1_ask,
        transitions,
        use_spread=False,
        initial_capital=decomp.DIAGNOSTIC_INITIAL_CAPITAL,
    )
    calculator_return = float(
        (points[-1].equity - points[0].equity) / decomp.DIAGNOSTIC_INITIAL_CAPITAL
    )

    close = _fabricated_dataset_close(values, start)
    entries, exits, short_entries, short_exits = mean_reversion_signals(
        close, FROZEN_LOOKBACK, FROZEN_Z_ENTRY
    )
    portfolio = _build_portfolio(
        close[_INSTRUMENT_ID],
        entries[_INSTRUMENT_ID],
        exits[_INSTRUMENT_ID],
        short_entries[_INSTRUMENT_ID],
        short_exits[_INSTRUMENT_ID],
        fees=0.0,
        freq="1h",
    )
    vectorbt_return = float(portfolio.total_return())

    assert calculator_return == pytest.approx(vectorbt_return, abs=1e-6)


def test_midpoint_variant_has_no_spread_charge() -> None:
    """B (use_spread=False) must fill every transition at the execution
    midpoint regardless of side -- structurally, no spread cost is ever
    paid. Proven by an exact match against a hand-computed mid-only
    compounding path (a second, independent computation), not merely by
    re-reading the implementation's own branch."""

    values = _fabricated_price_sequence()
    start = datetime(2023, 1, 2, tzinfo=UTC)
    h1_bid, h1_ask = _build_h1_bars(values, start)
    m1_bid, m1_ask = _build_zero_drift_m1_bars(values, start)
    start_ns = h1_bid[0].ts_event
    end_ns = h1_bid[-1].ts_event + 3_600_000_000_000

    transitions = decomp._resolve_direction_transitions(
        h1_bid, h1_ask, m1_bid, m1_ask, start_ns, end_ns
    )
    mid_points = decomp._compound_all_in_equity(
        h1_bid, h1_ask, transitions, use_spread=False, initial_capital=Decimal("100000")
    )

    # Independent hand computation: replay the same transitions using only
    # execution_mid, ignoring bid/ask entirely.
    equity = Decimal("100000")
    direction = 0
    last_price = None
    h1_by_ts = {
        bid.ts_event: (
            (Decimal(str(bid.close)) + Decimal(str(ask.close))) / 2
        )
        for bid, ask in zip(h1_bid, h1_ask, strict=True)
    }
    marks = sorted(h1_by_ts.items())
    events: list[tuple[int, Decimal, int | None]] = [
        (ts, mid, None) for ts, mid in marks
    ]
    for t in transitions:
        events.append((t.execution_ns, t.execution_mid, t.new_direction))
    events.sort(key=lambda e: (e[0], 0 if e[2] is not None else 1))
    last_ts = marks[0][0]
    last_price = marks[0][1]
    hand_final = equity
    for ts, price, new_direction in events:
        if ts < last_ts:
            continue
        if ts > last_ts and direction != 0:
            hand_final = hand_final * (price / last_price) ** direction
        last_ts, last_price = ts, price
        if new_direction is not None:
            direction = new_direction

    assert mid_points[-1].equity == hand_final


def test_spread_variant_crosses_correct_side() -> None:
    """C (use_spread=True) must fill a BUY transition at the execution
    ASK and a SELL transition at the execution BID -- proven by
    constructing a wide, asymmetric synthetic spread and checking the
    calculator's realized return differs from the midpoint variant in the
    economically correct direction (worse, since crossing a real spread
    always costs something relative to the midpoint)."""

    values = _fabricated_price_sequence()
    start = datetime(2023, 1, 2, tzinfo=UTC)
    h1_bid, h1_ask = _build_h1_bars(values, start)
    m1_bid, m1_ask = _build_zero_drift_m1_bars(values, start)
    start_ns = h1_bid[0].ts_event
    end_ns = h1_bid[-1].ts_event + 3_600_000_000_000

    transitions = decomp._resolve_direction_transitions(
        h1_bid, h1_ask, m1_bid, m1_ask, start_ns, end_ns
    )
    assert len(transitions) == 3
    for t in transitions:
        assert t.execution_bid < t.execution_mid < t.execution_ask

    mid_points = decomp._compound_all_in_equity(
        h1_bid, h1_ask, transitions, use_spread=False, initial_capital=Decimal("100000")
    )
    spread_points = decomp._compound_all_in_equity(
        h1_bid, h1_ask, transitions, use_spread=True, initial_capital=Decimal("100000")
    )

    # Every single fill crosses a real spread under C -- the spread variant
    # must never outperform the zero-cost midpoint variant.
    assert spread_points[-1].equity < mid_points[-1].equity


def test_direction_transitions_are_always_strictly_later_than_decision() -> None:
    values = _fabricated_price_sequence()
    start = datetime(2023, 1, 2, tzinfo=UTC)
    h1_bid, h1_ask = _build_h1_bars(values, start)
    m1_bid, m1_ask = _build_zero_drift_m1_bars(values, start)
    start_ns = h1_bid[0].ts_event
    end_ns = h1_bid[-1].ts_event + 3_600_000_000_000

    transitions = decomp._resolve_direction_transitions(
        h1_bid, h1_ask, m1_bid, m1_ask, start_ns, end_ns
    )
    assert len(transitions) == 3
    for t in transitions:
        assert t.execution_ns > t.decision_ns


def test_direction_transitions_rejects_duplicate_execution_frame() -> None:
    values = _fabricated_price_sequence()
    start = datetime(2023, 1, 2, tzinfo=UTC)
    h1_bid, h1_ask = _build_h1_bars(values, start)
    start_ns = h1_bid[0].ts_event
    end_ns = h1_bid[-1].ts_event + 3_600_000_000_000

    # A single M1 observation strictly after every H1 decision in the
    # fixture -- every transition resolves onto it, which must raise.
    shared_ns = h1_bid[-1].ts_event + 60_000_000_000
    bid_type = BarType.from_str(f"{_INSTRUMENT_ID}-1-MINUTE-BID-EXTERNAL")
    ask_type = BarType.from_str(f"{_INSTRUMENT_ID}-1-MINUTE-ASK-EXTERNAL")
    volume = Quantity.from_str(f"{1:.{_SPEC.size_precision}f}")
    bid_price = Price.from_str("1.09990")
    ask_price = Price.from_str("1.10010")
    m1_bid = (
        Bar(
            bar_type=bid_type,
            open=bid_price,
            high=bid_price,
            low=bid_price,
            close=bid_price,
            volume=volume,
            ts_event=shared_ns,
            ts_init=shared_ns,
        ),
    )
    m1_ask = (
        Bar(
            bar_type=ask_type,
            open=ask_price,
            high=ask_price,
            low=ask_price,
            close=ask_price,
            volume=volume,
            ts_event=shared_ns,
            ts_init=shared_ns,
        ),
    )

    with pytest.raises(decomp.DecompositionError):
        decomp._resolve_direction_transitions(
            h1_bid, h1_ask, m1_bid, m1_ask, start_ns, end_ns
        )


# ---------------------------------------------------------------------------
# D's zero-spread bar transformation: never invents a price, always the
# genuine mid of the real BID/ASK pair.
# ---------------------------------------------------------------------------


def test_zero_spread_bars_use_the_genuine_mid_on_both_sides() -> None:
    values = _fabricated_price_sequence()[:5]
    start = datetime(2023, 1, 2, tzinfo=UTC)
    bid_bars, ask_bars = _build_zero_drift_m1_bars(values, start)

    zero_bid, zero_ask = decomp._zero_spread_bars(bid_bars, ask_bars)
    assert len(zero_bid) == len(bid_bars)
    for original_bid, original_ask, new_bid, new_ask in zip(
        bid_bars, ask_bars, zero_bid, zero_ask, strict=True
    ):
        expected_mid = (
            original_bid.close.as_decimal() + original_ask.close.as_decimal()
        ) / 2
        assert new_bid.close.as_decimal() == new_ask.close.as_decimal()
        assert float(new_bid.close.as_decimal()) == pytest.approx(
            float(expected_mid), abs=1e-9
        )
        assert new_bid.ts_event == original_bid.ts_event
        assert new_ask.ts_event == original_ask.ts_event


# ---------------------------------------------------------------------------
# Classification rule (Section 5), applied mechanically.
# ---------------------------------------------------------------------------


def _fake_variant(net_return: float) -> decomp.VariantResult:
    pair_id = "EUR/USD.OANDA"
    perf = dev.PairPerformance(
        instrument_id=pair_id,
        initial_capital="100000",
        net_return=net_return,
        realized_variable_cost="0",
        cost_stress_1_5x_return=net_return,
        annualized_sharpe=None,
        daily_return_count=0,
    )
    aggregate = dev.AggregatePerformance(
        pair_count=1,
        equal_weight_net_return=net_return,
        equal_weight_cost_stress_1_5x_return=net_return,
        profitable_pair_count=1 if net_return > 0 else 0,
        annualized_sharpe=None,
        daily_return_count=0,
        cost_stress_methodology=dev.COST_STRESS_METHODOLOGY_LABEL,
    )
    return decomp.VariantResult(
        label="fake", pair_performance={pair_id: perf}, aggregate=aggregate
    )


def test_classify_same_bar_execution_dependent() -> None:
    variants = {
        "A": _fake_variant(0.05),
        "B": _fake_variant(-0.01),
        "C": _fake_variant(-0.02),
        "D": _fake_variant(-0.02),
        "E": _fake_variant(-0.02),
    }
    assert decomp.classify(variants) == decomp.SAME_BAR_EXECUTION_DEPENDENT


def test_classify_native_spread_intolerant() -> None:
    variants = {
        "A": _fake_variant(0.05),
        "B": _fake_variant(0.03),
        "C": _fake_variant(-0.01),
        "D": _fake_variant(0.02),
        "E": _fake_variant(-0.01),
    }
    assert decomp.classify(variants) == decomp.NATIVE_SPREAD_INTOLERANT


def test_classify_risk_sizing_transformation() -> None:
    variants = {
        "A": _fake_variant(0.05),
        "B": _fake_variant(0.03),
        "C": _fake_variant(0.01),
        "D": _fake_variant(-0.01),
        "E": _fake_variant(-0.02),
    }
    assert decomp.classify(variants) == decomp.RISK_SIZING_TRANSFORMATION


def test_classify_proceed_to_frozen_protocol() -> None:
    variants = {
        "A": _fake_variant(0.05),
        "B": _fake_variant(0.03),
        "C": _fake_variant(0.01),
        "D": _fake_variant(0.02),
        "E": _fake_variant(0.01),
    }
    assert decomp.classify(variants) == decomp.PROCEED_TO_FROZEN_PROTOCOL


# ---------------------------------------------------------------------------
# Common statistics pipeline (Section 3): every variant must reject a
# missing/extra pair rather than silently compute a partial aggregate.
# ---------------------------------------------------------------------------


def test_variant_result_requires_exactly_the_seven_frozen_pairs() -> None:
    from ftmoquant.strategies.mean_reversion_h1 import FROZEN_UNIVERSE

    incomplete = {
        instrument_id: (dev.EquityPoint(0, Decimal("100000")),)
        for instrument_id in FROZEN_UNIVERSE[:-1]
    }
    with pytest.raises(decomp.DecompositionError):
        decomp._variant_result("incomplete", incomplete, {})


def test_last_of_day_keeps_seed_and_one_mark_per_calendar_day() -> None:
    day_ns = 24 * 3_600_000_000_000
    points = (
        dev.EquityPoint(0, Decimal("100000")),  # unconditional seed anchor
        dev.EquityPoint(3_600_000_000_000, Decimal("100010")),  # day 1's own mark
        dev.EquityPoint(day_ns + 1, Decimal("100020")),  # day 2, first mark
        dev.EquityPoint(day_ns + 3_600_000_000_000, Decimal("100030")),  # day 2, last
    )
    reduced = decomp._last_of_day(points)
    # Seed is always kept regardless of its own day; every subsequent
    # calendar day collapses to its own LAST observation (day 2's two
    # points collapse to one, its last).
    assert reduced == (points[0], points[1], points[3])
