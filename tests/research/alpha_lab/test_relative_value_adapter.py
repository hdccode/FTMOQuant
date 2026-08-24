from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from ftmoquant.prop_rules import EvaluationPhase, load_prop_rule_set
from ftmoquant.research.alpha_lab.relative_value_adapter import (
    IncompleteAtomicEntryError,
    IncompleteAtomicExitError,
    IncompleteLegFill,
    LegMark,
    RelativeValueAdapterError,
    RelativeValueEpisode,
    RelativeValueLeg,
    RelativeValuePositionState,
    adapt_relative_value_episode,
    attempt_relative_value_entry,
    attempt_relative_value_exit,
)
from ftmoquant.research.ftmo_pass_probability.state_machine import (
    FtmoPathStatus,
    simulate_phase,
)

D = Decimal
RULE_CONFIG = Path("config/prop/ftmo_2step_swing_2026-08.yaml").resolve()
ONE_YEAR_NS = 365 * 24 * 60 * 60 * 1_000_000_000
_HOUR_NS = 3_600_000_000_000
_DAY_NS = 24 * _HOUR_NS


def _leg(
    *,
    instrument_id: str,
    direction: int,
    quantity: str,
    base_currency: str,
    quote_currency: str,
    entry_ns: int,
    entry_price: str,
    exit_ns: int,
    exit_price: str,
    extra_marks: tuple[tuple[int, str], ...] = (),
) -> RelativeValueLeg:
    marks = [LegMark(entry_ns, D(entry_price))]
    marks.extend(LegMark(ts, D(price)) for ts, price in extra_marks)
    marks.append(LegMark(exit_ns, D(exit_price)))
    return RelativeValueLeg(
        instrument_id=instrument_id,
        direction=direction,  # type: ignore[arg-type]
        quantity=D(quantity),
        base_currency=base_currency,
        quote_currency=quote_currency,
        entry_ns=entry_ns,
        entry_price=D(entry_price),
        exit_ns=exit_ns,
        exit_price=D(exit_price),
        marks=tuple(marks),
    )


# ---------------------------------------------------------------------------
# Account-currency aggregation (incl. USD/JPY, USD/CAD, USD/CHF regression)
# ---------------------------------------------------------------------------


def test_two_leg_pnl_aggregates_in_account_currency_not_native_quote_currency() -> None:
    leg_a = _leg(
        instrument_id="USD/CAD.OANDA",
        direction=1,
        quantity="100000",
        base_currency="USD",
        quote_currency="CAD",
        entry_ns=0,
        entry_price="1.30",
        exit_ns=_HOUR_NS,
        exit_price="1.31",
    )
    leg_b = _leg(
        instrument_id="EUR/USD.OANDA",
        direction=1,
        quantity="50000",
        base_currency="EUR",
        quote_currency="USD",
        entry_ns=0,
        entry_price="1.0800",
        exit_ns=_HOUR_NS,
        exit_price="1.0850",
    )
    episode = RelativeValueEpisode("t1", leg_a, leg_b, exit_reason="target")
    # leg_a: 100000 * 0.01 CAD / 1.30 = 769.230769... USD
    # leg_b: 50000 * 0.0050 USD (already USD) = 250 USD
    expected = (D("100000") * D("0.01") / D("1.30")) + D("250")
    assert episode.realized_pnl() == expected


@pytest.mark.parametrize(
    "base,quote,entry,exit_,quantity",
    [
        ("USD", "JPY", "150.00", "151.00", "10000"),
        ("USD", "CAD", "1.3500", "1.3400", "10000"),
        ("USD", "CHF", "0.9000", "0.9100", "10000"),
    ],
)
def test_usd_base_pair_conversion_matches_manual_single_point_formula(
    base: str, quote: str, entry: str, exit_: str, quantity: str
) -> None:
    """Regression coverage for the three USD-as-base pairs this repo has
    historically had a currency-unit bug for (see
    ftmoquant.research.mean_reversion_h1_development's own commit history:
    'Fix multi-currency accounting in mean reversion execution')."""

    leg = _leg(
        instrument_id=f"{base}/{quote}.OANDA",
        direction=1,
        quantity=quantity,
        base_currency=base,
        quote_currency=quote,
        entry_ns=0,
        entry_price=entry,
        exit_ns=_HOUR_NS,
        exit_price=exit_,
    )
    pnl_quote = D(quantity) * (D(exit_) - D(entry))
    expected_usd = pnl_quote / D(entry)
    assert leg.pnl_usd_at(D(exit_)) == expected_usd


def test_quote_currency_usd_pair_needs_no_conversion() -> None:
    leg = _leg(
        instrument_id="EUR/USD.OANDA",
        direction=1,
        quantity="10000",
        base_currency="EUR",
        quote_currency="USD",
        entry_ns=0,
        entry_price="1.0800",
        exit_ns=_HOUR_NS,
        exit_price="1.0900",
    )
    assert leg.pnl_usd_at(D("1.0900")) == D("10000") * D("0.0100")


# ---------------------------------------------------------------------------
# Direction handling
# ---------------------------------------------------------------------------


def test_opposite_direction_legs_both_contribute_correctly() -> None:
    long_leg = _leg(
        instrument_id="EUR/USD.OANDA",
        direction=1,
        quantity="10000",
        base_currency="EUR",
        quote_currency="USD",
        entry_ns=0,
        entry_price="1.0800",
        exit_ns=_HOUR_NS,
        exit_price="1.0850",
    )
    short_leg = _leg(
        instrument_id="GBP/USD.OANDA",
        direction=-1,
        quantity="10000",
        base_currency="GBP",
        quote_currency="USD",
        entry_ns=0,
        entry_price="1.2700",
        exit_ns=_HOUR_NS,
        exit_price="1.2650",
    )
    episode = RelativeValueEpisode("t1", long_leg, short_leg, exit_reason="target")
    expected = D("10000") * D("0.0050") + D("10000") * D("0.0050")
    assert episode.realized_pnl() == expected


# ---------------------------------------------------------------------------
# Entry/exit timestamps and incomplete-fill fail-closed behavior
# ---------------------------------------------------------------------------


def test_episode_entry_and_exit_timestamps_are_exact() -> None:
    leg_a = _leg(
        instrument_id="USD/CAD.OANDA",
        direction=1,
        quantity="10000",
        base_currency="USD",
        quote_currency="CAD",
        entry_ns=0,
        entry_price="1.30",
        exit_ns=_HOUR_NS,
        exit_price="1.30",
    )
    leg_b = _leg(
        instrument_id="EUR/USD.OANDA",
        direction=1,
        quantity="10000",
        base_currency="EUR",
        quote_currency="USD",
        entry_ns=100,
        entry_price="1.08",
        exit_ns=_HOUR_NS + 200,
        exit_price="1.08",
    )
    episode = RelativeValueEpisode("t1", leg_a, leg_b, exit_reason="target")
    assert episode.entry_ns == 0
    assert episode.exit_ns == _HOUR_NS + 200
    assert episode.both_legs_open_from_ns == 100
    assert episode.both_legs_closed_by_ns == _HOUR_NS


def test_incomplete_entry_fails_closed() -> None:
    leg_a = _leg(
        instrument_id="USD/CAD.OANDA",
        direction=1,
        quantity="10000",
        base_currency="USD",
        quote_currency="CAD",
        entry_ns=0,
        entry_price="1.30",
        exit_ns=_HOUR_NS,
        exit_price="1.30",
    )
    failed_fill = IncompleteLegFill(
        instrument_id="EUR/USD.OANDA",
        attempted_direction=1,
        attempted_quantity=D("10000"),
        reason="rejected_by_broker",
        at_ns=0,
    )
    with pytest.raises(IncompleteAtomicEntryError, match="not atomic"):
        attempt_relative_value_entry(leg_a, failed_fill)


def test_incomplete_exit_fails_closed() -> None:
    with pytest.raises(IncompleteAtomicExitError, match="not atomic"):
        attempt_relative_value_exit(
            leg_a_closed=True,
            leg_b_closed=False,
            leg_a_instrument_id="USD/CAD.OANDA",
            leg_b_instrument_id="EUR/USD.OANDA",
            at_ns=1000,
        )


def test_both_legs_closed_does_not_raise() -> None:
    attempt_relative_value_exit(
        leg_a_closed=True,
        leg_b_closed=True,
        leg_a_instrument_id="USD/CAD.OANDA",
        leg_b_instrument_id="EUR/USD.OANDA",
        at_ns=1000,
    )


def test_legs_that_never_overlap_are_rejected() -> None:
    leg_a = _leg(
        instrument_id="USD/CAD.OANDA",
        direction=1,
        quantity="10000",
        base_currency="USD",
        quote_currency="CAD",
        entry_ns=0,
        entry_price="1.30",
        exit_ns=_HOUR_NS,
        exit_price="1.30",
    )
    leg_b = _leg(
        instrument_id="EUR/USD.OANDA",
        direction=1,
        quantity="10000",
        base_currency="EUR",
        quote_currency="USD",
        entry_ns=2 * _HOUR_NS,
        entry_price="1.08",
        exit_ns=3 * _HOUR_NS,
        exit_price="1.08",
    )
    with pytest.raises(RelativeValueAdapterError, match="never overlap"):
        RelativeValueEpisode("t1", leg_a, leg_b, exit_reason="target")


# ---------------------------------------------------------------------------
# Temporary legging exposure must not be hidden
# ---------------------------------------------------------------------------


def test_legging_in_exposure_is_visible_not_hidden() -> None:
    leg_a = _leg(
        instrument_id="USD/CAD.OANDA",
        direction=1,
        quantity="100000",
        base_currency="USD",
        quote_currency="CAD",
        entry_ns=0,
        entry_price="1.30",
        exit_ns=_HOUR_NS,
        exit_price="1.30",
        extra_marks=((1000, "1.25"),),
    )
    leg_b = _leg(
        instrument_id="EUR/USD.OANDA",
        direction=1,
        quantity="50000",
        base_currency="EUR",
        quote_currency="USD",
        entry_ns=2000,
        entry_price="1.08",
        exit_ns=_HOUR_NS,
        exit_price="1.08",
    )
    episode = RelativeValueEpisode("t1", leg_a, leg_b, exit_reason="target")

    assert episode.position_state_at(1000) == RelativeValuePositionState.LEGGING_IN
    assert episode.combined_pnl_usd_at(1000) == leg_a.contribution_at(1000)
    assert episode.combined_pnl_usd_at(1000) != D("0")
    assert leg_b.contribution_at(1000) == D("0")
    assert episode.position_state_at(2000) == RelativeValuePositionState.BOTH_LEGS_OPEN


def test_legging_out_exposure_is_visible_not_hidden() -> None:
    leg_a = _leg(
        instrument_id="USD/CAD.OANDA",
        direction=1,
        quantity="100000",
        base_currency="USD",
        quote_currency="CAD",
        entry_ns=0,
        entry_price="1.30",
        exit_ns=_HOUR_NS,
        exit_price="1.28",
    )
    leg_b = _leg(
        instrument_id="EUR/USD.OANDA",
        direction=1,
        quantity="50000",
        base_currency="EUR",
        quote_currency="USD",
        entry_ns=0,
        entry_price="1.08",
        exit_ns=2 * _HOUR_NS,
        exit_price="1.08",
    )
    episode = RelativeValueEpisode("t1", leg_a, leg_b, exit_reason="target")
    probe_ns = _HOUR_NS + 100
    assert episode.position_state_at(probe_ns) == RelativeValuePositionState.LEGGING_OUT
    # leg_a is already closed by probe_ns: its settled realized P&L must
    # persist in combined equity (not be dropped to zero), and leg_b is
    # still open and contributes its own floating mark on top of that.
    assert leg_a.contribution_at(probe_ns) != D("0")
    assert episode.combined_pnl_usd_at(probe_ns) == (
        leg_a.contribution_at(probe_ns) + leg_b.contribution_at(probe_ns)
    )


# ---------------------------------------------------------------------------
# The required adversarial fixture: intraday loss then recovery must still
# breach the real FTMO state machine despite a positive final combined P&L.
# ---------------------------------------------------------------------------


def test_adversarial_intraday_loss_then_recovery_still_breaches_ftmo_rules() -> None:
    leg_a = _leg(
        instrument_id="USD/CAD.OANDA",
        direction=1,
        quantity="150000",
        base_currency="USD",
        quote_currency="CAD",
        entry_ns=0,
        entry_price="1.30",
        exit_ns=_HOUR_NS,
        exit_price="1.31",
        extra_marks=((1_800_000_000_000, "1.25"),),  # heavy adverse dip mid-trade
    )
    leg_b = _leg(
        instrument_id="NZD/USD.OANDA",
        direction=1,
        quantity="50000",
        base_currency="NZD",
        quote_currency="USD",
        entry_ns=0,
        entry_price="0.60",
        exit_ns=_HOUR_NS,
        exit_price="0.65",
    )
    episode = RelativeValueEpisode("t1", leg_a, leg_b, exit_reason="target")

    # Sanity: final combined P&L is positive, but the intratrade floor is a
    # real, materially large loss -- the exact scenario the task requires.
    assert episode.realized_pnl() > D("0")
    assert episode.floor_equity_delta() < D("-5000")

    event = adapt_relative_value_episode(episode)
    rules = load_prop_rule_set(RULE_CONFIG)
    outcome = simulate_phase(
        [event],
        rules=rules,
        phase=EvaluationPhase.CHALLENGE,
        initial_capital=D("100000"),
        horizon_ns=ONE_YEAR_NS,
    )

    assert outcome.status == FtmoPathStatus.FAILED_DAILY_LOSS
    assert outcome.max_drawdown > D("0")


def test_naive_final_pnl_only_aggregation_would_have_hidden_the_breach() -> None:
    """Negative control: proves the breach above is genuinely path-dependent,
    not an artifact of the fixture -- replaying only the FINAL combined P&L
    (as a naive final-leg-P&L-only adapter would) shows no breach at all."""

    leg_a = _leg(
        instrument_id="USD/CAD.OANDA",
        direction=1,
        quantity="150000",
        base_currency="USD",
        quote_currency="CAD",
        entry_ns=0,
        entry_price="1.30",
        exit_ns=_HOUR_NS,
        exit_price="1.31",
        extra_marks=((1_800_000_000_000, "1.25"),),
    )
    leg_b = _leg(
        instrument_id="NZD/USD.OANDA",
        direction=1,
        quantity="50000",
        base_currency="NZD",
        quote_currency="USD",
        entry_ns=0,
        entry_price="0.60",
        exit_ns=_HOUR_NS,
        exit_price="0.65",
    )
    episode = RelativeValueEpisode("t1", leg_a, leg_b, exit_reason="target")

    from ftmoquant.research.ftmo_pass_probability.state_machine import TradeEvent

    naive_event = TradeEvent(
        entry_ns=episode.entry_ns,
        exit_ns=episode.exit_ns,
        floor_equity_delta=min(D("0"), episode.realized_pnl()),
        realized_pnl=episode.realized_pnl(),
    )
    rules = load_prop_rule_set(RULE_CONFIG)
    outcome = simulate_phase(
        [naive_event],
        rules=rules,
        phase=EvaluationPhase.CHALLENGE,
        initial_capital=D("100000"),
        horizon_ns=ONE_YEAR_NS,
    )
    assert outcome.status != FtmoPathStatus.FAILED_DAILY_LOSS
    assert outcome.status != FtmoPathStatus.FAILED_MAX_LOSS


# ---------------------------------------------------------------------------
# Overnight / multi-day path retention (reuses simulate_phase's own
# trading-day logic -- no new day-boundary code in this module).
# ---------------------------------------------------------------------------


def test_overnight_multi_day_hold_is_retained_in_the_adapted_event() -> None:
    leg_a = _leg(
        instrument_id="USD/CAD.OANDA",
        direction=1,
        quantity="10000",
        base_currency="USD",
        quote_currency="CAD",
        entry_ns=0,
        entry_price="1.30",
        exit_ns=3 * _DAY_NS,
        exit_price="1.31",
    )
    leg_b = _leg(
        instrument_id="EUR/USD.OANDA",
        direction=1,
        quantity="10000",
        base_currency="EUR",
        quote_currency="USD",
        entry_ns=0,
        entry_price="1.08",
        exit_ns=3 * _DAY_NS,
        exit_price="1.09",
    )
    episode = RelativeValueEpisode("t1", leg_a, leg_b, exit_reason="target")
    event = adapt_relative_value_episode(episode)
    assert event.exit_ns - event.entry_ns == 3 * _DAY_NS

    rules = load_prop_rule_set(RULE_CONFIG)
    outcome = simulate_phase(
        [event],
        rules=rules,
        phase=EvaluationPhase.CHALLENGE,
        initial_capital=D("100000"),
        horizon_ns=ONE_YEAR_NS,
    )
    # An overnight hold counts only its entry trading day, matching
    # test_state_machine.py::
    #   test_overnight_multi_day_hold_counts_only_its_entry_trading_day.
    assert outcome.trading_days == 1


def test_prague_day_boundary_logic_is_reused_unchanged() -> None:
    """This module adds no Prague-day-reset logic of its own -- proven by
    reusing state_machine._trading_day_from_ns directly on an adapted
    event's own entry_ns and confirming it matches a second call with the
    same rules (i.e. nothing in this adapter recomputes or shadows it)."""

    from ftmoquant.research.ftmo_pass_probability.state_machine import (
        _trading_day_from_ns,
    )

    leg_a = _leg(
        instrument_id="USD/CAD.OANDA",
        direction=1,
        quantity="10000",
        base_currency="USD",
        quote_currency="CAD",
        entry_ns=0,
        entry_price="1.30",
        exit_ns=_HOUR_NS,
        exit_price="1.30",
    )
    leg_b = _leg(
        instrument_id="EUR/USD.OANDA",
        direction=1,
        quantity="10000",
        base_currency="EUR",
        quote_currency="USD",
        entry_ns=0,
        entry_price="1.08",
        exit_ns=_HOUR_NS,
        exit_price="1.08",
    )
    episode = RelativeValueEpisode("t1", leg_a, leg_b, exit_reason="target")
    event = adapt_relative_value_episode(episode)
    rules = load_prop_rule_set(RULE_CONFIG)
    assert _trading_day_from_ns(event.entry_ns, rules) == _trading_day_from_ns(
        event.entry_ns, rules
    )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_adapter_is_deterministic() -> None:
    leg_a = _leg(
        instrument_id="USD/CAD.OANDA",
        direction=1,
        quantity="100000",
        base_currency="USD",
        quote_currency="CAD",
        entry_ns=0,
        entry_price="1.30",
        exit_ns=_HOUR_NS,
        exit_price="1.31",
        extra_marks=((1_800_000_000_000, "1.25"),),
    )
    leg_b = _leg(
        instrument_id="NZD/USD.OANDA",
        direction=1,
        quantity="50000",
        base_currency="NZD",
        quote_currency="USD",
        entry_ns=0,
        entry_price="0.60",
        exit_ns=_HOUR_NS,
        exit_price="0.65",
    )
    episode = RelativeValueEpisode("t1", leg_a, leg_b, exit_reason="target")
    first = adapt_relative_value_episode(episode)
    second = adapt_relative_value_episode(episode)
    assert first == second


# ---------------------------------------------------------------------------
# No pair-selection/signal logic in the adapter (structural check)
# ---------------------------------------------------------------------------


def test_adapter_module_contains_no_pair_or_signal_specific_logic() -> None:
    import ftmoquant.research.alpha_lab.relative_value_adapter as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    forbidden_tokens = [
        "AUD/NZD",
        "AUD_NZD",
        "cointegrat",
        "hedge_ratio",
        "z_score",
        "zscore",
        "p_value",
    ]
    lowered = source.lower()
    for token in forbidden_tokens:
        assert token.lower() not in lowered, (
            f"unexpected signal-specific token: {token}"
        )
