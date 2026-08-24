from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from nautilus_trader.model import Bar, BarType, Price, Quantity

import ftmoquant.research.b3f1_u2_execution_promotion as m
from ftmoquant.data.instruments import OANDA_ALPHA_LAB_SPECS
from ftmoquant.research.alpha_lab.b3f1_underpowered_u2_validation import X_SPEC, Y_SPEC
from ftmoquant.research.mean_reversion_h1_development import (
    _convert_to_account_currency,
)


def _spec(instrument_id: str):
    return next(s for s in OANDA_ALPHA_LAB_SPECS if s.instrument_id == instrument_id)


def _m1_bars(
    instrument_id: str, pairs: list[tuple[int, float, float]]
) -> tuple[Bar, ...]:
    """Genuine paired M1 BID/ASK bars at explicit ``(ts_ns, bid, ask)``
    points -- the sole stream ``run_frozen_u2_backtest`` ever injects."""

    spec = _spec(instrument_id)
    bid_type = BarType.from_str(f"{instrument_id}-1-MINUTE-BID-EXTERNAL")
    ask_type = BarType.from_str(f"{instrument_id}-1-MINUTE-ASK-EXTERNAL")
    volume = Quantity.from_str(f"{1:.{spec.size_precision}f}")
    bars = []
    for ts, bid, ask in pairs:
        bid_price = Price.from_str(f"{bid:.{spec.price_precision}f}")
        ask_price = Price.from_str(f"{ask:.{spec.price_precision}f}")
        bars.append(
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
        bars.append(
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
    return tuple(bars)


_START = datetime(2023, 1, 2, tzinfo=UTC)
_START_NS = int(_START.timestamp() * 1_000_000_000)
_MIN_NS = 60_000_000_000
_END_EXCLUSIVE = datetime(2023, 1, 3, tzinfo=UTC)


def _instruments():
    return {
        Y_SPEC.instrument_id: _spec(Y_SPEC.instrument_id).nautilus_instrument(),
        X_SPEC.instrument_id: _spec(X_SPEC.instrument_id).nautilus_instrument(),
    }


def _one_leg_pair_bars(
    instrument_id: str,
    entry_ns: int,
    exit_ns: int,
    entry_bid: float,
    entry_ask: float,
    exit_bid: float,
    exit_ask: float,
) -> tuple[Bar, ...]:
    return _m1_bars(
        instrument_id,
        [
            (entry_ns, entry_bid, entry_ask),
            (exit_ns, exit_bid, exit_ask),
        ],
    )


def _long_short_instruction(*, entry_ns: int, exit_ns: int) -> m._TwoLegInstruction:
    """Y (USD/CAD) LONG, X (USD/CHF) SHORT -- a genuine simultaneous
    opposite-leg spread trade, mirroring B3F1's RICH/CHEAP convention."""

    return m._TwoLegInstruction(
        logical_trade_id="test-trade-1",
        entry_ns={"Y": entry_ns, "X": entry_ns},
        exit_ns={"Y": exit_ns, "X": exit_ns},
        direction={"Y": 1, "X": -1},
        quantity={"Y": Decimal("50000"), "X": Decimal("50000")},
    )


def _run(instruction: m._TwoLegInstruction, y_bars, x_bars) -> m.EngineRunOutcome:
    return m.run_frozen_u2_backtest(
        start=_START,
        end_exclusive=_END_EXCLUSIVE,
        instruments=_instruments(),
        instructions=[instruction],
        m1_bars=[*y_bars, *x_bars],
    )


# ---------------------------------------------------------------------------
# Section 8: critical two-leg accounting tests
# ---------------------------------------------------------------------------


def test_usd_cad_conversion_divides_quote_amount_by_price() -> None:
    # USD/CAD: base=USD (account currency), quote=CAD -- a CAD amount
    # converts to USD by dividing by the USD/CAD price.
    result = _convert_to_account_currency(
        Decimal("490.00"),
        "CAD",
        base_currency="USD",
        quote_currency="CAD",
        conversion_price=Decimal("1.3502"),
    )
    assert result == Decimal("490.00") / Decimal("1.3502")


def test_usd_chf_conversion_divides_quote_amount_by_price() -> None:
    # USD/CHF: base=USD, quote=CHF -- identical rule, different pair.
    result = _convert_to_account_currency(
        Decimal("250.00"),
        "CHF",
        base_currency="USD",
        quote_currency="CHF",
        conversion_price=Decimal("0.9000"),
    )
    assert result == Decimal("250.00") / Decimal("0.9000")


def test_no_raw_cad_or_chf_amount_is_ever_summed_as_usd() -> None:
    """A logical trade's total P&L must equal the SUM of the two ALREADY-
    CONVERTED leg P&Ls -- never the sum of the raw CAD/CHF quote-currency
    amounts (which would be dimensionally meaningless)."""

    entry_ns = _START_NS + 5 * _MIN_NS
    exit_ns = _START_NS + 65 * _MIN_NS
    instruction = _long_short_instruction(entry_ns=entry_ns, exit_ns=exit_ns)

    y_bars = _one_leg_pair_bars(
        Y_SPEC.instrument_id, entry_ns, exit_ns, 1.34990, 1.35010, 1.35990, 1.36010
    )
    x_bars = _one_leg_pair_bars(
        X_SPEC.instrument_id, entry_ns, exit_ns, 0.89990, 0.90010, 0.89490, 0.89510
    )
    outcome = _run(instruction, y_bars, x_bars)

    assert len(outcome.logical_trades) == 1
    trade = outcome.logical_trades[0]

    # Y: LONG -- buys ASK (1.35010), sells BID (1.35990) on exit.
    y_pnl_quote = Decimal("50000") * (Decimal("1.35990") - Decimal("1.35010"))
    y_pnl_usd = _convert_to_account_currency(
        y_pnl_quote,
        "CAD",
        base_currency="USD",
        quote_currency="CAD",
        conversion_price=Decimal("1.35010"),
    )
    # X: SHORT -- sells BID (0.89990) entry, buys ASK (0.89510) exit.
    x_pnl_quote = (
        Decimal("-1") * Decimal("50000") * (Decimal("0.89510") - Decimal("0.89990"))
    )
    x_pnl_usd = _convert_to_account_currency(
        x_pnl_quote,
        "CHF",
        base_currency="USD",
        quote_currency="CHF",
        conversion_price=Decimal("0.89990"),
    )

    # A loose tolerance here (roughly one price-increment's worth of USD
    # notional) deliberately accommodates Nautilus's own native fill
    # mechanics landing on a slightly different last-decimal tick than the
    # bar close fed into the fixture -- that possible divergence is
    # genuine execution degradation this promotion exists to MEASURE, not
    # something to force into artificial agreement.
    assert float(trade.leg_pnl_usd["Y"]) == pytest.approx(float(y_pnl_usd), abs=1.0)
    assert float(trade.leg_pnl_usd["X"]) == pytest.approx(float(x_pnl_usd), abs=1.0)
    # The dimensionally-critical invariant: total P&L is EXACTLY the sum
    # of the two already-converted leg P&Ls, computed by the same code
    # either way -- this is not sensitive to the tick-level fill variance
    # above.
    assert Decimal(trade.total_pnl_usd) == Decimal(trade.leg_pnl_usd["Y"]) + Decimal(
        trade.leg_pnl_usd["X"]
    )
    # The raw, UNCONVERTED CAD+CHF numeric sum would be a different (and
    # dimensionally meaningless) value -- confirming conversion actually
    # happened before summation, not after.
    raw_unconverted_sum = y_pnl_quote + x_pnl_quote
    assert float(trade.total_pnl_usd) != pytest.approx(
        float(raw_unconverted_sum), abs=1e-6
    )


def test_temporary_one_leg_exposure_is_visible_before_second_leg_fills() -> None:
    """Nautilus records leg fills independently and in genuine
    chronological order -- a spread trade legs in over two DIFFERENT
    execution timestamps here, so the first leg's fill (and its
    resulting unrealized exposure on the native margin account) must be
    observable strictly before the second leg's fill."""

    y_entry_ns = _START_NS + 5 * _MIN_NS
    x_entry_ns = _START_NS + 6 * _MIN_NS  # one minute later -- genuine legging gap
    exit_ns = _START_NS + 65 * _MIN_NS
    instruction = m._TwoLegInstruction(
        logical_trade_id="test-trade-legging",
        entry_ns={"Y": y_entry_ns, "X": x_entry_ns},
        exit_ns={"Y": exit_ns, "X": exit_ns},
        direction={"Y": 1, "X": -1},
        quantity={"Y": Decimal("50000"), "X": Decimal("50000")},
    )
    y_bars = _m1_bars(
        Y_SPEC.instrument_id,
        [(y_entry_ns, 1.34990, 1.35010), (exit_ns, 1.35990, 1.36010)],
    )
    x_bars = _m1_bars(
        X_SPEC.instrument_id,
        [(x_entry_ns, 0.89990, 0.90010), (exit_ns, 0.89490, 0.89510)],
    )
    outcome = _run(instruction, y_bars, x_bars)

    entry_fills = [f for f in outcome.fills if f.action == "entry"]
    assert len(entry_fills) == 2
    y_fill = next(f for f in entry_fills if f.leg == "Y")
    x_fill = next(f for f in entry_fills if f.leg == "X")
    # The Y leg fills strictly before the X leg -- a one-legged interval
    # genuinely exists and is recorded, not silently coalesced into one
    # simultaneous event.
    assert y_fill.fill_ns < x_fill.fill_ns
    assert y_fill.fill_ns == y_entry_ns
    assert x_fill.fill_ns == x_entry_ns

    # The logical trade closes only once BOTH legs' exits are filled.
    assert len(outcome.logical_trades) == 1


def test_logical_trade_closes_only_when_both_legs_have_closed() -> None:
    entry_ns = _START_NS + 5 * _MIN_NS
    exit_ns = _START_NS + 65 * _MIN_NS
    instruction = _long_short_instruction(entry_ns=entry_ns, exit_ns=exit_ns)
    y_bars = _one_leg_pair_bars(
        Y_SPEC.instrument_id, entry_ns, exit_ns, 1.34990, 1.35010, 1.35990, 1.36010
    )
    x_bars = _one_leg_pair_bars(
        X_SPEC.instrument_id, entry_ns, exit_ns, 0.89990, 0.90010, 0.89490, 0.89510
    )
    outcome = _run(instruction, y_bars, x_bars)
    assert len(outcome.fills) == 4  # 2 legs x (entry + exit)
    assert len(outcome.logical_trades) == 1
    trade = outcome.logical_trades[0]
    assert set(trade.entry_leg_fill_ns) == {"Y", "X"}
    assert set(trade.exit_leg_fill_ns) == {"Y", "X"}


def test_deterministic_fills_and_path() -> None:
    entry_ns = _START_NS + 5 * _MIN_NS
    exit_ns = _START_NS + 65 * _MIN_NS
    instruction = _long_short_instruction(entry_ns=entry_ns, exit_ns=exit_ns)
    y_bars = _one_leg_pair_bars(
        Y_SPEC.instrument_id, entry_ns, exit_ns, 1.34990, 1.35010, 1.35990, 1.36010
    )
    x_bars = _one_leg_pair_bars(
        X_SPEC.instrument_id, entry_ns, exit_ns, 0.89990, 0.90010, 0.89490, 0.89510
    )
    first = _run(instruction, y_bars, x_bars)
    second = _run(instruction, y_bars, x_bars)
    assert first.submissions == second.submissions
    assert first.fills == second.fills
    assert first.logical_trades == second.logical_trades


# ---------------------------------------------------------------------------
# Instruction extraction / parity diagnostics (pure functions, no engine)
# ---------------------------------------------------------------------------


def test_instructions_from_episodes_extracts_both_legs_correctly() -> None:
    from ftmoquant.research.alpha_lab.relative_value_adapter import (
        LegMark,
        RelativeValueEpisode,
        RelativeValueLeg,
    )

    leg_a = RelativeValueLeg(
        instrument_id=Y_SPEC.instrument_id,
        direction=1,
        quantity=Decimal("60000"),
        base_currency="USD",
        quote_currency="CAD",
        entry_ns=1_000,
        entry_price=Decimal("1.35"),
        exit_ns=2_000,
        exit_price=Decimal("1.36"),
        marks=(LegMark(1_000, Decimal("1.35")), LegMark(2_000, Decimal("1.36"))),
    )
    leg_b = RelativeValueLeg(
        instrument_id=X_SPEC.instrument_id,
        direction=-1,
        quantity=Decimal("40000"),
        base_currency="USD",
        quote_currency="CHF",
        entry_ns=1_500,
        entry_price=Decimal("0.90"),
        exit_ns=2_500,
        exit_price=Decimal("0.89"),
        marks=(LegMark(1_500, Decimal("0.90")), LegMark(2_500, Decimal("0.89"))),
    )
    episode = RelativeValueEpisode(
        logical_trade_id="ep-1",
        leg_a=leg_a,
        leg_b=leg_b,
        exit_reason="z_mean_reversion",
    )
    instructions = m._instructions_from_episodes([episode])
    assert len(instructions) == 1
    instr = instructions[0]
    assert instr.logical_trade_id == "ep-1"
    assert instr.entry_ns == {"Y": 1_000, "X": 1_500}
    assert instr.exit_ns == {"Y": 2_000, "X": 2_500}
    assert instr.direction == {"Y": 1, "X": -1}
    assert instr.quantity == {"Y": Decimal("60000"), "X": Decimal("40000")}


def test_parity_diagnostics_reports_zero_mismatches_for_a_perfect_reproduction() -> (
    None
):
    from ftmoquant.research.alpha_lab.relative_value_adapter import (
        LegMark,
        RelativeValueEpisode,
        RelativeValueLeg,
    )

    leg_a = RelativeValueLeg(
        instrument_id=Y_SPEC.instrument_id,
        direction=1,
        quantity=Decimal("50000"),
        base_currency="USD",
        quote_currency="CAD",
        entry_ns=1_000,
        entry_price=Decimal("1.35"),
        exit_ns=2_000,
        exit_price=Decimal("1.36"),
        marks=(LegMark(1_000, Decimal("1.35")), LegMark(2_000, Decimal("1.36"))),
    )
    leg_b = RelativeValueLeg(
        instrument_id=X_SPEC.instrument_id,
        direction=-1,
        quantity=Decimal("50000"),
        base_currency="USD",
        quote_currency="CHF",
        entry_ns=1_000,
        entry_price=Decimal("0.90"),
        exit_ns=2_000,
        exit_price=Decimal("0.89"),
        marks=(LegMark(1_000, Decimal("0.90")), LegMark(2_000, Decimal("0.89"))),
    )
    episode = RelativeValueEpisode(
        logical_trade_id="ep-1",
        leg_a=leg_a,
        leg_b=leg_b,
        exit_reason="z_mean_reversion",
    )
    matching_trade = m.LogicalTradeRecord(
        logical_trade_id="ep-1",
        entry_leg_fill_ns={"Y": 1_000, "X": 1_000},
        exit_leg_fill_ns={"Y": 2_000, "X": 2_000},
        both_legs_open_ns=1_000,
        both_legs_closed_ns=2_000,
        leg_pnl_usd={"Y": "370.37", "X": "277.78"},
        total_pnl_usd=str(episode.realized_pnl()),
    )
    diagnostics = m.compute_parity_diagnostics(
        episodes=[episode],
        logical_trades=[matching_trade],
        initial_capital=Decimal("100000"),
    )
    assert diagnostics.trade_count_difference == 0
    assert diagnostics.entry_timestamp_mismatches == 0
    assert diagnostics.exit_timestamp_mismatches == 0
    assert Decimal(diagnostics.total_return_degradation_usd) == 0


def test_parity_diagnostics_detects_a_timestamp_mismatch() -> None:
    from ftmoquant.research.alpha_lab.relative_value_adapter import (
        LegMark,
        RelativeValueEpisode,
        RelativeValueLeg,
    )

    leg_a = RelativeValueLeg(
        instrument_id=Y_SPEC.instrument_id,
        direction=1,
        quantity=Decimal("50000"),
        base_currency="USD",
        quote_currency="CAD",
        entry_ns=1_000,
        entry_price=Decimal("1.35"),
        exit_ns=2_000,
        exit_price=Decimal("1.36"),
        marks=(LegMark(1_000, Decimal("1.35")), LegMark(2_000, Decimal("1.36"))),
    )
    leg_b = RelativeValueLeg(
        instrument_id=X_SPEC.instrument_id,
        direction=-1,
        quantity=Decimal("50000"),
        base_currency="USD",
        quote_currency="CHF",
        entry_ns=1_000,
        entry_price=Decimal("0.90"),
        exit_ns=2_000,
        exit_price=Decimal("0.89"),
        marks=(LegMark(1_000, Decimal("0.90")), LegMark(2_000, Decimal("0.89"))),
    )
    episode = RelativeValueEpisode(
        logical_trade_id="ep-1",
        leg_a=leg_a,
        leg_b=leg_b,
        exit_reason="z_mean_reversion",
    )
    mismatched_trade = m.LogicalTradeRecord(
        logical_trade_id="ep-1",
        entry_leg_fill_ns={"Y": 1_060_000_000_000, "X": 1_000},  # Y entry 1 minute late
        exit_leg_fill_ns={"Y": 2_000, "X": 2_000},
        both_legs_open_ns=1_060_000_000_000,
        both_legs_closed_ns=2_000,
        leg_pnl_usd={"Y": "370.37", "X": "277.78"},
        total_pnl_usd=str(episode.realized_pnl()),
    )
    diagnostics = m.compute_parity_diagnostics(
        episodes=[episode],
        logical_trades=[mismatched_trade],
        initial_capital=Decimal("100000"),
    )
    assert diagnostics.entry_timestamp_mismatches == 1
    assert diagnostics.exit_timestamp_mismatches == 0


# ---------------------------------------------------------------------------
# CLI / identity / safety
# ---------------------------------------------------------------------------


def test_cli_exposes_no_signal_pair_or_parameter_override() -> None:
    parser = m.build_parser()
    option_strings = {
        option
        for action in parser._actions  # noqa: SLF001
        for option in action.option_strings
    }
    assert option_strings == {
        "-h",
        "--help",
        "--partition",
        "--catalog-root",
        "--universe-readiness",
        "--output",
    }


def test_only_development_and_validation_partitions_are_accepted() -> None:
    assert m.parse_partition("development") is m.Partition.DEVELOPMENT
    assert m.parse_partition("validation") is m.Partition.VALIDATION
    with pytest.raises(Exception):  # noqa: B017 -- reused helper's own error type
        m.parse_partition("holdout")


def test_no_partition_bound_ever_reaches_the_final_holdout() -> None:
    from ftmoquant.research.stage_g import HOLDOUT_START

    for partition in (m.Partition.DEVELOPMENT, m.Partition.VALIDATION):
        _start, end_exclusive = m.partition_bounds(partition)
        assert end_exclusive <= HOLDOUT_START


def test_output_directory_reservation_happens_before_preregistration_read(
    tmp_path,
) -> None:
    existing_output = tmp_path / "out"
    existing_output.mkdir()
    with pytest.raises(m.B3F1U2ExecutionPromotionError, match="already exists"):
        m.run_b3f1_u2_execution_promotion(
            partition=m.Partition.DEVELOPMENT,
            catalog_root=tmp_path / "catalog",
            universe_readiness_path=tmp_path / "readiness.json",
            output_dir=existing_output,
        )


def test_frozen_candidate_identity_matches_the_validated_u2() -> None:
    assert m.SLEEVE_ID == "USD/CAD.OANDA__USD/CHF.OANDA"
    assert m.FORMATION_WINDOW == 240
    assert str(m.Z_ENTRY) == "1.5"
    assert str(m.Z_STOP) == "3.5"
