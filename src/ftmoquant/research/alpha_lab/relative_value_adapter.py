"""Generic multi-leg relative-value trade adapter, required by B3F1
(cross-instrument spread mean reversion) in the frozen Batch 3 v2
preregistration (``config/research/batch3_methodology_preregistration_v2.json``,
semantic hash
``e5cd74527004585cfd24bea55549d84e9cb66b05ffc6498360dac5007b651f7c``).

This module knows nothing about B3F1's actual hypothesis, pairs, hedge
ratios, or thresholds -- it is pure plumbing: given two already-executed FX
legs that together make up one coordinated relative-value position, it
builds the account-currency, intratrade-path-aware representation the
existing FTMO pass-probability machinery needs, and hands it off unchanged
to that machinery. No new Monte Carlo state machine is introduced.

Design decision (discovered, not assumed): the existing single-instrument
causal representation,
:class:`ftmoquant.research.ftmo_pass_probability.path_extraction.TradeRecord`
(entry/exit + a currency-invariant ``net_r`` + a *predetermined* stop-based
risk budget), is NOT a safe target for a generic two-leg adapter. Its
downstream sizing rescale
(:func:`ftmoquant.research.ftmo_pass_probability.sizing.apply_sizing`)
derives ``floor_equity_delta`` from ``exit_reason`` alone (``min(pnl, 0)``
for a stop-exit, ``-risk_budget`` for a target-exit) -- an exactness
argument that holds only because a single continuously-monitored
instrument's hard stop order guarantees the exit *is* the worst mark
reached. A two-leg spread's true worst combined mark is a genuine path
property of both legs together and is not recoverable from entry/exit
alone (see the adversarial fixture in
``tests/research/alpha_lab/test_relative_value_adapter.py::
test_adversarial_intraday_loss_then_recovery_still_breaches``, required by
the B3.1 task brief). This module therefore targets
:class:`ftmoquant.research.ftmo_pass_probability.state_machine.TradeEvent`
directly instead -- the other existing generic "path representation" in
this repo, which already carries an explicit ``floor_equity_delta`` and is
consumed unchanged by
:func:`ftmoquant.research.ftmo_pass_probability.state_machine.simulate_phase`.
Reusing it means every existing daily-loss/max-loss/Prague-day-boundary
rule in ``simulate_phase`` applies to a relative-value episode with zero
new breach-detection code.

Not addressed here (explicitly out of scope, a B3.2+ concern once B3F1 has
a real signal): rescaling a relative-value episode across the
DEVELOPMENT-only ``SIZING_GRID``/refinement-grid sizing-selection procedure.
That machinery is built around ``TradeRecord``'s predetermined-risk-budget
model; a relative-value family's own signal layer (not this generic
adapter) will need to supply its own notion of intended risk per spread
trade before that reuse is possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from ftmoquant.research.ftmo_pass_probability.state_machine import TradeEvent
from ftmoquant.research.mean_reversion_h1_development import (
    _convert_to_account_currency as convert_to_account_currency,
)

ZERO = Decimal("0")
_ACCOUNT_CURRENCY = "USD"

Direction = Literal[1, -1]


class RelativeValueAdapterError(ValueError):
    """Raised whenever a relative-value episode cannot be safely
    represented -- including every fail-closed case in this module. Never
    silently substitutes, drops, or approximates a value."""


class IncompleteAtomicEntryError(RelativeValueAdapterError):
    """Raised when an attempted two-leg entry did not complete on both
    legs. A one-leg fill is never treated as a completed spread trade."""


class IncompleteAtomicExitError(RelativeValueAdapterError):
    """Raised when an attempted two-leg exit did not complete on both
    legs. A one-leg close is never treated as a completed spread exit."""


@dataclass(frozen=True, slots=True)
class IncompleteLegFill:
    """Records that one leg of an attempted two-leg entry or exit could
    not be completed. A first-class failure record, not a silently
    dropped or approximated fill."""

    instrument_id: str
    attempted_direction: Direction
    attempted_quantity: Decimal
    reason: str
    at_ns: int


def attempt_relative_value_entry(
    leg_a: RelativeValueLeg | IncompleteLegFill,
    leg_b: RelativeValueLeg | IncompleteLegFill,
) -> tuple[RelativeValueLeg, RelativeValueLeg]:
    """Confirm a coordinated two-leg entry actually completed atomically.

    Raises :class:`IncompleteAtomicEntryError`, naming which leg(s) failed
    and why, if either leg did not fill. Never returns a partially-filled
    pair -- a caller that only has one confirmed fill cannot construct a
    :class:`RelativeValueEpisode` from it via this function.
    """

    incomplete = [leg for leg in (leg_a, leg_b) if isinstance(leg, IncompleteLegFill)]
    if incomplete:
        raise IncompleteAtomicEntryError(
            "relative-value entry is not atomic: "
            f"{len(incomplete)} of 2 legs failed to fill "
            f"({[record.reason for record in incomplete]!r}); a one-leg fill "
            "is never treated as a completed spread trade"
        )
    assert isinstance(leg_a, RelativeValueLeg)
    assert isinstance(leg_b, RelativeValueLeg)
    return leg_a, leg_b


def attempt_relative_value_exit(
    leg_a_closed: bool,
    leg_b_closed: bool,
    *,
    leg_a_instrument_id: str,
    leg_b_instrument_id: str,
    at_ns: int,
) -> None:
    """Confirm a coordinated two-leg exit actually completed atomically.

    Raises :class:`IncompleteAtomicExitError` if exactly one leg closed.
    Both-closed and neither-closed are the only outcomes this function
    accepts silently (neither-closed simply means the episode is not yet
    over, which is the caller's business, not an error here).
    """

    if leg_a_closed != leg_b_closed:
        stuck = leg_a_instrument_id if leg_b_closed else leg_b_instrument_id
        raise IncompleteAtomicExitError(
            f"relative-value exit is not atomic: {stuck} did not close "
            f"while its counterpart did, at ts_ns={at_ns}; a one-leg close "
            "is never treated as a completed spread exit"
        )


@dataclass(frozen=True, slots=True)
class LegMark:
    """One conservative mark observation for one leg, at or between that
    leg's own entry and exit. ``price`` must already be whichever BID/ASK
    side would actually be crossed to close this leg at this instant -- an
    execution-engine/caller responsibility; this adapter does not choose
    sides and carries no BID/ASK duality of its own (that is
    :mod:`ftmoquant.research.alpha_lab.cost_stress`'s concern, deliberately
    decoupled from this module)."""

    ts_ns: int
    price: Decimal

    def __post_init__(self) -> None:
        if not self.price.is_finite() or self.price <= ZERO:
            raise RelativeValueAdapterError(
                f"leg mark price must be positive and finite, got {self.price}"
            )


@dataclass(frozen=True, slots=True)
class RelativeValueLeg:
    """One executed leg of a coordinated two-leg relative-value trade.

    ``marks`` must be the leg's complete, exhaustive, chronological price
    path from entry to exit INCLUSIVE: the first mark is exactly
    ``(entry_ns, entry_price)`` and the last is exactly
    ``(exit_ns, exit_price)``. No interpolation, no rolling estimate, no
    future observation -- each mark is whatever the caller's own execution
    stream actually observed. ``base_currency``/``quote_currency`` are the
    pair's own ISO codes (e.g. base="USD", quote="CAD" for USD/CAD).
    """

    instrument_id: str
    direction: Direction
    quantity: Decimal
    base_currency: str
    quote_currency: str
    entry_ns: int
    entry_price: Decimal
    exit_ns: int
    exit_price: Decimal
    marks: tuple[LegMark, ...]

    def __post_init__(self) -> None:
        if self.direction not in (1, -1):
            raise RelativeValueAdapterError("direction must be 1 (long) or -1 (short)")
        if not self.quantity.is_finite() or self.quantity <= ZERO:
            raise RelativeValueAdapterError("quantity must be positive and finite")
        if self.base_currency == self.quote_currency:
            raise RelativeValueAdapterError(
                "base_currency and quote_currency must differ"
            )
        if not self.entry_price.is_finite() or self.entry_price <= ZERO:
            raise RelativeValueAdapterError("entry_price must be positive and finite")
        if not self.exit_price.is_finite() or self.exit_price <= ZERO:
            raise RelativeValueAdapterError("exit_price must be positive and finite")
        if self.exit_ns <= self.entry_ns:
            raise RelativeValueAdapterError("leg exit must be strictly after entry")
        if len(self.marks) < 2:
            raise RelativeValueAdapterError(
                "marks must include at least the entry and exit observations"
            )
        if (
            self.marks[0].ts_ns != self.entry_ns
            or self.marks[0].price != self.entry_price
        ):
            raise RelativeValueAdapterError(
                "first mark must equal (entry_ns, entry_price)"
            )
        if (
            self.marks[-1].ts_ns != self.exit_ns
            or self.marks[-1].price != self.exit_price
        ):
            raise RelativeValueAdapterError(
                "last mark must equal (exit_ns, exit_price)"
            )
        for earlier, later in zip(self.marks, self.marks[1:], strict=False):
            if later.ts_ns <= earlier.ts_ns:
                raise RelativeValueAdapterError("marks must be strictly chronological")
        if self.marks[0].ts_ns < self.entry_ns or self.marks[-1].ts_ns > self.exit_ns:
            raise RelativeValueAdapterError("marks must lie within [entry_ns, exit_ns]")

    def pnl_usd_at(self, price: Decimal) -> Decimal:
        """This leg's unrealized/realized P&L in account currency (USD) if
        marked at ``price``, using the SAME single-conversion-at-entry-price
        convention already used throughout this repo (see
        ``ftmoquant.research.mean_reversion_h1_development.
        _convert_to_account_currency`` and
        ``ftmoquant.research.ftmo_pass_probability.path_extraction``'s own
        docstring) -- the FX conversion rate is fixed at this leg's own
        entry price for the whole leg, never re-marked continuously."""

        pnl_quote = self.direction * self.quantity * (price - self.entry_price)
        return convert_to_account_currency(
            pnl_quote,
            self.quote_currency,
            base_currency=self.base_currency,
            quote_currency=self.quote_currency,
            conversion_price=self.entry_price,
        )

    def contribution_at(self, ts_ns: int) -> Decimal:
        """This leg's USD contribution to combined equity at ``ts_ns``:
        zero before this leg has entered; the last known mark at-or-before
        ``ts_ns`` while open (a step function -- no interpolation); frozen
        at its own exit mark once ``ts_ns`` is at or after this leg's own
        exit (since no mark exists past ``exit_ns``, "last mark <= ts_ns"
        already resolves to the exit mark for any ``ts_ns >= exit_ns``, so
        a closed leg's realized P&L persists forward automatically)."""

        if ts_ns < self.entry_ns:
            return ZERO
        candidate = self.marks[0]
        for mark in self.marks:
            if mark.ts_ns > ts_ns:
                break
            candidate = mark
        return self.pnl_usd_at(candidate.price)


class RelativeValuePositionState(StrEnum):
    """Reported open-position state of a two-leg episode at a given
    instant -- exposed so callers/tests can prove legging exposure is
    never silently hidden."""

    NOT_YET_OPEN = "not_yet_open"
    LEGGING_IN = "legging_in"
    BOTH_LEGS_OPEN = "both_legs_open"
    LEGGING_OUT = "legging_out"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class RelativeValueEpisode:
    """ONE logical relative-value (two-leg) trade.

    ``exit_reason`` is caller-supplied and opaque to this adapter -- this
    module carries it through without interpreting it; deciding why a
    B3F1 spread trade exited belongs to that family's future signal layer,
    never to this generic plumbing.
    """

    logical_trade_id: str
    leg_a: RelativeValueLeg
    leg_b: RelativeValueLeg
    exit_reason: str

    def __post_init__(self) -> None:
        if self.leg_a.instrument_id == self.leg_b.instrument_id:
            raise RelativeValueAdapterError(
                "leg_a and leg_b must be different instruments"
            )
        if self.both_legs_open_from_ns > self.both_legs_closed_by_ns:
            raise RelativeValueAdapterError(
                "legs never overlap in time -- this is two unrelated "
                "single-leg trades, not one coordinated spread trade"
            )

    @property
    def entry_ns(self) -> int:
        """First instant either leg carries exposure (legging-in start)."""

        return min(self.leg_a.entry_ns, self.leg_b.entry_ns)

    @property
    def exit_ns(self) -> int:
        """Last instant either leg still carries exposure (legging-out end)."""

        return max(self.leg_a.exit_ns, self.leg_b.exit_ns)

    @property
    def both_legs_open_from_ns(self) -> int:
        """Instant the hedge becomes complete (later of the two entries)."""

        return max(self.leg_a.entry_ns, self.leg_b.entry_ns)

    @property
    def both_legs_closed_by_ns(self) -> int:
        """Instant the hedge stops being complete (earlier of the two exits)."""

        return min(self.leg_a.exit_ns, self.leg_b.exit_ns)

    def position_state_at(self, ts_ns: int) -> RelativeValuePositionState:
        if ts_ns < self.entry_ns:
            return RelativeValuePositionState.NOT_YET_OPEN
        if ts_ns > self.exit_ns:
            return RelativeValuePositionState.CLOSED
        if ts_ns < self.both_legs_open_from_ns:
            return RelativeValuePositionState.LEGGING_IN
        if ts_ns > self.both_legs_closed_by_ns:
            return RelativeValuePositionState.LEGGING_OUT
        return RelativeValuePositionState.BOTH_LEGS_OPEN

    def combined_pnl_usd_at(self, ts_ns: int) -> Decimal:
        """Combined account-currency P&L at ``ts_ns``, summing each leg's
        own :meth:`RelativeValueLeg.contribution_at` -- both legs already
        converted to USD independently, never summed in native quote
        currency (never ``JPY + USD``, ``CAD + USD``, ``CHF + USD``, ...).
        Legging exposure (only one leg open) is not a special case: it
        falls out automatically, since the not-yet-open/already-closed leg
        simply contributes zero/its frozen realized amount."""

        return self.leg_a.contribution_at(ts_ns) + self.leg_b.contribution_at(ts_ns)

    def combined_pnl_path(self) -> tuple[tuple[int, Decimal], ...]:
        """The full, deterministic, chronological combined-equity path
        over the union of both legs' own mark timestamps -- the
        intratrade path information this adapter exists to preserve."""

        timestamps = sorted(
            {mark.ts_ns for mark in self.leg_a.marks}
            | {mark.ts_ns for mark in self.leg_b.marks}
        )
        return tuple((ts_ns, self.combined_pnl_usd_at(ts_ns)) for ts_ns in timestamps)

    def floor_equity_delta(self) -> Decimal:
        """The worst (most negative) combined equity offset reached at any
        observed mark across the whole episode, relative to the balance at
        ``entry_ns`` (== 0 P&L). Guaranteed <= 0 because the path always
        includes at least one point at/near entry where combined P&L is 0
        or worse."""

        return min(ZERO, *(pnl for _, pnl in self.combined_pnl_path()))

    def realized_pnl(self) -> Decimal:
        """Final combined P&L once both legs are closed."""

        return self.combined_pnl_usd_at(self.exit_ns)


def adapt_relative_value_episode(episode: RelativeValueEpisode) -> TradeEvent:
    """The sole bridge from a :class:`RelativeValueEpisode` to the
    existing, unmodified FTMO path-replay machinery: builds one
    :class:`ftmoquant.research.ftmo_pass_probability.state_machine.TradeEvent`
    whose ``floor_equity_delta`` is the GENUINE worst combined mark
    observed across both legs' real price paths (not a single-leg
    stop/target heuristic), and whose ``entry_ns``/``exit_ns`` bound the
    full legging-in-to-legging-out window. Feed the result straight into
    :func:`ftmoquant.research.ftmo_pass_probability.state_machine.
    simulate_phase` unchanged -- every daily-loss/max-loss/Prague-day-
    boundary rule already implemented there applies with no new code.
    """

    return TradeEvent(
        entry_ns=episode.entry_ns,
        exit_ns=episode.exit_ns,
        floor_equity_delta=episode.floor_equity_delta(),
        realized_pnl=episode.realized_pnl(),
    )
