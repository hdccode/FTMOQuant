"""B3F1 execution: turns causal :class:`B3F1TradeIntent` decisions into
genuine two-leg, native-BID/ASK-executed
:class:`~ftmoquant.research.alpha_lab.relative_value_adapter.RelativeValueEpisode`
trades.

Reuses, rather than re-derives:

- :mod:`ftmoquant.research.alpha_lab.relative_value_adapter` for the
  atomic-fill, account-currency, intratrade-path-preserving two-leg
  representation (Component B, B3.1) -- this module never re-implements
  currency conversion or breach-detection logic. Once both legs have a
  confirmed fill, :func:`~ftmoquant.research.alpha_lab.
  relative_value_adapter.attempt_relative_value_entry`/
  ``attempt_relative_value_exit`` are the actual gate a
  :class:`RelativeValueEpisode` must pass through -- this module never
  constructs one directly.
- :mod:`ftmoquant.research.alpha_lab.cost_stress` for the pre-execution
  spread-widening transform (Component A, B3.1) -- applied to the raw M1
  BID/ASK frames BEFORE any fill price is read, never as a post-hoc P&L
  haircut.
- The same "first strictly later paired M1 observation" ``bisect``
  convention already used by
  ``ftmoquant.research.alpha_lab.wick_fvg_squeeze_execution.simulate_trades``
  and
  ``ftmoquant.research.mean_reversion_h1_development._first_strictly_later_paired_ns``
  -- reimplemented locally (both those functions are private/coupled to
  their own single-instrument bar shape) using the identical semantics.

Section 10's leg-sizing formula is implemented and DIMENSIONALLY AUDITED
here (see :func:`usd_gross_to_quantity`): the naive
``quantity = gross_usd / entry_price`` is only correct for quote=USD pairs
(EUR/USD, GBP/USD, AUD/USD, NZD/USD). For the three base=USD pairs in the
frozen universe (USD/CAD, USD/CHF, USD/JPY), 1 unit of the traded quantity
already equals 1 USD of notional (the base leg IS the account currency), so
dividing by price would silently under-size the position by a factor equal
to that pair's own price (e.g. ~150x for USD/JPY). This module implements
the correct base-vs-quote branch rather than assuming quote=USD.

Two distinct "cannot execute" cases are modelled separately, deliberately:
a :class:`B3F1SkipRecord` when the M1 data itself has no later observation
for one or both legs (a data-availability boundary, not a broker
rejection -- mirrors ``wick_fvg_squeeze_execution.SkipRecord``'s
``"no_later_m1_observation"``), versus the B3.1 adapter's own
``IncompleteAtomicEntryError``/``IncompleteAtomicExitError`` (a genuine
attempted-but-failed fill on one already-known leg), which this module
raises through when it has two already-priced legs to hand the adapter.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

import pandas as pd  # type: ignore[import-untyped]

from ftmoquant.data.instruments import InstrumentSpec
from ftmoquant.research.alpha_lab.b3f1_spread_signals import (
    B3F1TradeIntent,
    SpreadSide,
)
from ftmoquant.research.alpha_lab.cost_stress import widen_bid_ask_frame
from ftmoquant.research.alpha_lab.relative_value_adapter import (
    LegMark,
    RelativeValueEpisode,
    RelativeValueLeg,
    attempt_relative_value_entry,
    attempt_relative_value_exit,
)

GROSS_NOTIONAL_USD = Decimal("100000")
_ACCOUNT_CURRENCY = "USD"
_LONG = 1
_SHORT = -1


class B3F1ExecutionError(ValueError):
    """Raised on any violation of the frozen B3F1 execution contract."""


@dataclass(frozen=True, slots=True)
class B3F1SkipRecord:
    """One logical trade intent that could not be executed -- never
    silently dropped, always explicit about which stage and why."""

    sleeve_id: str
    entry_ts: pd.Timestamp
    reason: str


def compute_leg_weights(beta: Decimal) -> tuple[Decimal, Decimal]:
    """The frozen, deterministic gross-notional split (section 10):
    ``weight_y = 1 / (1 + |beta|)``, ``weight_x = |beta| / (1 + |beta|)``.
    Always sums to exactly 1; both weights are strictly between 0 and 1 for
    any finite positive beta (beta<=0 is already rejected at formation)."""

    if not beta.is_finite() or beta <= 0:
        raise B3F1ExecutionError("beta must be finite and positive to size legs")
    abs_beta = abs(beta)
    weight_y = Decimal(1) / (Decimal(1) + abs_beta)
    weight_x = abs_beta / (Decimal(1) + abs_beta)
    return weight_y, weight_x


def usd_gross_to_quantity(
    gross_usd: Decimal, entry_price: Decimal, *, base_currency: str, quote_currency: str
) -> Decimal:
    """Convert a USD gross-notional target into the instrument's own
    native quantity, causally (using only this leg's own entry price).

    DIMENSIONAL AUDIT (section 10): for a base=USD pair (USD/CAD, USD/CHF,
    USD/JPY), one unit of quantity already equals one USD of notional --
    dividing by price here would be wrong (see module docstring). For a
    quote=USD pair (EUR/USD, GBP/USD, AUD/USD, NZD/USD), quantity is in
    units of the base currency, each worth ``entry_price`` USD, so
    ``quantity = gross_usd / entry_price`` is the correct conversion.
    """

    if base_currency == quote_currency:
        raise B3F1ExecutionError("base_currency and quote_currency must differ")
    if not entry_price.is_finite() or entry_price <= 0:
        raise B3F1ExecutionError("entry_price must be positive and finite")
    if base_currency == _ACCOUNT_CURRENCY:
        return gross_usd
    if quote_currency == _ACCOUNT_CURRENCY:
        return gross_usd / entry_price
    raise B3F1ExecutionError(
        f"cannot size a {base_currency}/{quote_currency} leg without a "
        f"direct {_ACCOUNT_CURRENCY} rate -- neither leg is {_ACCOUNT_CURRENCY}"
    )


def _first_strictly_later_ns(paired_ns: Sequence[int], decision_ns: int) -> int | None:
    """Same "first strictly later" semantics as
    ``wick_fvg_squeeze_execution.simulate_trades``'s inline ``bisect``
    usage and ``mean_reversion_h1_development._first_strictly_later_paired_ns``
    -- fails closed (returns ``None``) rather than fabricating a fill when
    no later observation exists."""

    index = bisect.bisect_right(paired_ns, decision_ns)
    if index >= len(paired_ns):
        return None
    return paired_ns[index]


def _direction_for_side(side: SpreadSide) -> tuple[int, int]:
    """(y_direction, x_direction) per section 10: RICH -> short Y / long X;
    CHEAP -> long Y / short X."""

    if side is SpreadSide.RICH:
        return _SHORT, _LONG
    return _LONG, _SHORT


def _fill_price(
    direction: int, *, is_entry: bool, bid_close: float, ask_close: float
) -> Decimal:
    """Section 13/14's exact side convention: LONG buys ASK / sells BID;
    SHORT sells BID / buys ASK -- entry and exit use the SAME rule (a long
    leg always transacts at ASK to open and BID to close, symmetric for a
    short leg), so both call sites share this one function."""

    crosses_ask = (direction == _LONG) == is_entry
    return Decimal(str(ask_close if crosses_ask else bid_close))


def _leg_marks(
    *,
    entry_ns: int,
    entry_price: Decimal,
    exit_ns: int,
    exit_price: Decimal,
    direction: int,
    bid_m1: pd.DataFrame,
    ask_m1: pd.DataFrame,
) -> tuple[LegMark, ...]:
    """The leg's real, observed M1-close intratrade mark path -- always the
    conservative side (the side you would actually realize by closing
    right then: BID for a long leg, ASK for a short leg), strictly between
    entry and exit. No interpolation: only M1 closes that genuinely exist
    in the paired index are used."""

    conservative = ask_m1["close"] if direction == _SHORT else bid_m1["close"]
    ns_index = conservative.index.as_unit("ns").asi8
    between = conservative.loc[(ns_index > entry_ns) & (ns_index < exit_ns)]
    marks = [LegMark(entry_ns, entry_price)]
    marks.extend(
        LegMark(int(ts.value), Decimal(str(price))) for ts, price in between.items()
    )
    marks.append(LegMark(exit_ns, exit_price))
    return tuple(marks)


@dataclass(frozen=True, slots=True)
class _LegFill:
    entry_ns: int
    entry_price: Decimal
    exit_ns: int
    exit_price: Decimal


def _find_fill(
    *,
    entry_decision_ns: int,
    exit_decision_ns: int,
    direction: int,
    bid_m1: pd.DataFrame,
    ask_m1: pd.DataFrame,
) -> _LegFill | None:
    """Locate one leg's entry and exit fills, or ``None`` if either the
    entry or the exit has no later M1 observation available -- a
    data-availability boundary, handled by the caller as a
    :class:`B3F1SkipRecord`, distinct from the adapter's own
    attempted-and-failed-fill error path."""

    ns_index = bid_m1.index.as_unit("ns").asi8.tolist()
    entry_ns = _first_strictly_later_ns(ns_index, entry_decision_ns)
    if entry_ns is None:
        return None
    exit_ns = _first_strictly_later_ns(ns_index, exit_decision_ns)
    if exit_ns is None:
        return None
    entry_ts = pd.Timestamp(entry_ns, tz="UTC")
    exit_ts = pd.Timestamp(exit_ns, tz="UTC")
    entry_price = _fill_price(
        direction,
        is_entry=True,
        bid_close=bid_m1.at[entry_ts, "close"],
        ask_close=ask_m1.at[entry_ts, "close"],
    )
    exit_price = _fill_price(
        direction,
        is_entry=False,
        bid_close=bid_m1.at[exit_ts, "close"],
        ask_close=ask_m1.at[exit_ts, "close"],
    )
    return _LegFill(
        entry_ns=entry_ns,
        entry_price=entry_price,
        exit_ns=exit_ns,
        exit_price=exit_price,
    )


def simulate_b3f1_intents(
    intents: Sequence[B3F1TradeIntent],
    *,
    y_spec: InstrumentSpec,
    x_spec: InstrumentSpec,
    y_bid_m1: pd.DataFrame,
    y_ask_m1: pd.DataFrame,
    x_bid_m1: pd.DataFrame,
    x_ask_m1: pd.DataFrame,
    gross_notional_usd: Decimal = GROSS_NOTIONAL_USD,
    cost_stress_multiplier: Decimal = Decimal("1"),
) -> tuple[tuple[RelativeValueEpisode, ...], tuple[B3F1SkipRecord, ...]]:
    """Execute a chronological sequence of causal :class:`B3F1TradeIntent`
    decisions against genuine M1 BID/ASK data, one
    :class:`RelativeValueEpisode` per intent.

    ``cost_stress_multiplier`` is applied to the raw M1 BID/ASK frames via
    :func:`~ftmoquant.research.alpha_lab.cost_stress.widen_bid_ask_frame`
    BEFORE any fill price is read (section 15) -- signal formation is
    untouched by this parameter, since intents are already frozen causal
    decisions computed upstream from unstressed mid prices.

    Each leg's entry/exit fill is the first strictly-later paired M1
    observation after the intent's own H1 decision timestamp (section 13).
    If the M1 data itself runs out for either leg's entry or exit, the
    whole logical trade is skipped with an explicit reason. Once both legs
    DO have a fill, the trade is confirmed through
    :func:`~ftmoquant.research.alpha_lab.relative_value_adapter.
    attempt_relative_value_entry`/``attempt_relative_value_exit`` -- the
    real atomicity gate, not re-implemented here.
    """

    if not y_bid_m1.index.equals(y_ask_m1.index):
        raise B3F1ExecutionError("y_bid_m1 and y_ask_m1 must share an identical index")
    if not x_bid_m1.index.equals(x_ask_m1.index):
        raise B3F1ExecutionError("x_bid_m1 and x_ask_m1 must share an identical index")

    if cost_stress_multiplier != Decimal("1"):
        y_bid_m1, y_ask_m1 = widen_bid_ask_frame(
            y_bid_m1, y_ask_m1, float(cost_stress_multiplier)
        )
        x_bid_m1, x_ask_m1 = widen_bid_ask_frame(
            x_bid_m1, x_ask_m1, float(cost_stress_multiplier)
        )

    episodes: list[RelativeValueEpisode] = []
    skips: list[B3F1SkipRecord] = []
    busy_until_ns: int | None = None

    for intent in intents:
        entry_decision_ns = int(intent.entry_ts.value)
        exit_decision_ns = int(intent.exit_ts.value)
        if busy_until_ns is not None and entry_decision_ns <= busy_until_ns:
            skips.append(
                B3F1SkipRecord(
                    intent.sleeve_id, intent.entry_ts, "signal_during_open_trade"
                )
            )
            continue

        y_direction, x_direction = _direction_for_side(intent.side)
        y_fill = _find_fill(
            entry_decision_ns=entry_decision_ns,
            exit_decision_ns=exit_decision_ns,
            direction=y_direction,
            bid_m1=y_bid_m1,
            ask_m1=y_ask_m1,
        )
        x_fill = _find_fill(
            entry_decision_ns=entry_decision_ns,
            exit_decision_ns=exit_decision_ns,
            direction=x_direction,
            bid_m1=x_bid_m1,
            ask_m1=x_ask_m1,
        )
        if y_fill is None or x_fill is None:
            skips.append(
                B3F1SkipRecord(
                    intent.sleeve_id, intent.entry_ts, "no_later_m1_observation"
                )
            )
            continue

        weight_y, weight_x = compute_leg_weights(Decimal(str(intent.frozen_beta)))
        quantity_y = usd_gross_to_quantity(
            gross_notional_usd * weight_y,
            y_fill.entry_price,
            base_currency=y_spec.base_currency,
            quote_currency=y_spec.quote_currency,
        )
        quantity_x = usd_gross_to_quantity(
            gross_notional_usd * weight_x,
            x_fill.entry_price,
            base_currency=x_spec.base_currency,
            quote_currency=x_spec.quote_currency,
        )

        leg_y = RelativeValueLeg(
            instrument_id=y_spec.instrument_id,
            direction=y_direction,  # type: ignore[arg-type]
            quantity=quantity_y,
            base_currency=y_spec.base_currency,
            quote_currency=y_spec.quote_currency,
            entry_ns=y_fill.entry_ns,
            entry_price=y_fill.entry_price,
            exit_ns=y_fill.exit_ns,
            exit_price=y_fill.exit_price,
            marks=_leg_marks(
                entry_ns=y_fill.entry_ns,
                entry_price=y_fill.entry_price,
                exit_ns=y_fill.exit_ns,
                exit_price=y_fill.exit_price,
                direction=y_direction,
                bid_m1=y_bid_m1,
                ask_m1=y_ask_m1,
            ),
        )
        leg_x = RelativeValueLeg(
            instrument_id=x_spec.instrument_id,
            direction=x_direction,  # type: ignore[arg-type]
            quantity=quantity_x,
            base_currency=x_spec.base_currency,
            quote_currency=x_spec.quote_currency,
            entry_ns=x_fill.entry_ns,
            entry_price=x_fill.entry_price,
            exit_ns=x_fill.exit_ns,
            exit_price=x_fill.exit_price,
            marks=_leg_marks(
                entry_ns=x_fill.entry_ns,
                entry_price=x_fill.entry_price,
                exit_ns=x_fill.exit_ns,
                exit_price=x_fill.exit_price,
                direction=x_direction,
                bid_m1=x_bid_m1,
                ask_m1=x_ask_m1,
            ),
        )

        confirmed_y, confirmed_x = attempt_relative_value_entry(leg_y, leg_x)
        attempt_relative_value_exit(
            leg_a_closed=True,
            leg_b_closed=True,
            leg_a_instrument_id=y_spec.instrument_id,
            leg_b_instrument_id=x_spec.instrument_id,
            at_ns=max(y_fill.exit_ns, x_fill.exit_ns),
        )
        episode = RelativeValueEpisode(
            logical_trade_id=f"{intent.sleeve_id}:{entry_decision_ns}",
            leg_a=confirmed_y,
            leg_b=confirmed_x,
            exit_reason=intent.exit_reason,
        )
        episodes.append(episode)
        busy_until_ns = max(y_fill.exit_ns, x_fill.exit_ns)

    return tuple(episodes), tuple(skips)
