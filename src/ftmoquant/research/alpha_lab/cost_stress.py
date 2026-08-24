"""Generic native-BID/ASK spread-widening cost-stress transform.

Required by the frozen Batch 3 v2 preregistration's
``development_gates.transaction_cost_sensitivity`` gate
(``config/research/batch3_methodology_preregistration_v2.json``, semantic
hash ``e5cd74527004585cfd24bea55549d84e9cb66b05ffc6498360dac5007b651f7c``):
B3F1 must survive 1.5x spread stress; B3F2 and B3F3 must survive both 1.5x
and 2.0x. That gate explicitly forbids a post-hoc fee subtraction -- the
widened quotes must be produced BEFORE execution, and the identical,
unmodified signal/execution logic re-run against them.

This module has exactly one job: given an original paired BID/ASK quote (or
bar), produce a synthetically wider paired quote (or bar) around the SAME
midpoint. It knows nothing about any strategy, signal, instrument, or
Batch 3 family -- callers re-run their own execution engine against the
widened output.

Two levels are provided:

- :func:`widen_bid_ask_quotes` -- the exact, unambiguous scalar transform
  (Decimal in, Decimal out) for a single paired point-in-time quote.
- :func:`widen_bid_ask_bar` / :func:`widen_bid_ask_frame` -- a conservative
  extension to M1 OHLC BID/ASK bars, whose exact semantics and the
  discovered ambiguity motivating them are documented on
  :func:`widen_bid_ask_bar`. Read that docstring before using it.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

ZERO = Decimal("0")
ONE = Decimal("1")

#: The only multipliers the frozen v2 preregistration actually requires.
#: The scalar/bar transforms below accept any Decimal/float >= 1 -- this
#: tuple is Batch 3's own frozen choice of which values callers must use,
#: not a constraint enforced by the generic transform itself.
REQUIRED_BATCH3_MULTIPLIERS: tuple[Decimal, ...] = (
    Decimal("1.0"),
    Decimal("1.5"),
    Decimal("2.0"),
)


class CostStressError(ValueError):
    """Raised on any non-finite, crossed, or sub-1.0-multiplier input.

    Fails closed rather than clipping or silently substituting a value --
    a bad input to a cost-stress transform must never be quietly narrowed
    into something that understates transaction cost.
    """


@dataclass(frozen=True, slots=True)
class WidenedQuote:
    bid: Decimal
    ask: Decimal


def _validate_quote(bid: Decimal, ask: Decimal) -> None:
    if not bid.is_finite() or not ask.is_finite():
        raise CostStressError(
            f"bid/ask must be finite Decimals, got bid={bid}, ask={ask}"
        )
    if ask < bid:
        raise CostStressError(f"crossed market rejected: ask ({ask}) < bid ({bid})")


def _validate_multiplier(multiplier: Decimal) -> None:
    if not multiplier.is_finite():
        raise CostStressError(f"multiplier must be a finite Decimal, got {multiplier}")
    if multiplier < ONE:
        raise CostStressError(
            "multiplier must be >= 1 (narrowing the spread is forbidden), "
            f"got {multiplier}"
        )


def widen_bid_ask_quotes(
    bid: Decimal, ask: Decimal, multiplier: Decimal
) -> WidenedQuote:
    """Widen one paired point-in-time BID/ASK quote around its own midpoint.

    ``stressed_bid = mid - multiplier * half_spread``
    ``stressed_ask = mid + multiplier * half_spread``

    where ``mid = (bid + ask) / 2`` and ``half_spread = (ask - bid) / 2``.

    Exact and unambiguous: a single paired quote has exactly one midpoint,
    so there is no synchronization assumption to make (contrast
    :func:`widen_bid_ask_bar`, where this is not true). ``multiplier == 1``
    reproduces the original quote exactly. Zero spread stays zero at any
    multiplier. Fails closed (raises :class:`CostStressError`) on a
    non-finite quote, a crossed market (``ask < bid``), or a multiplier
    below 1 -- never clips or substitutes a value.
    """

    _validate_quote(bid, ask)
    _validate_multiplier(multiplier)
    mid = (bid + ask) / 2
    half_spread = (ask - bid) / 2
    return WidenedQuote(
        bid=mid - multiplier * half_spread, ask=mid + multiplier * half_spread
    )


@dataclass(frozen=True, slots=True)
class Bar:
    """A single OHLC bar for one side (BID or ASK)."""

    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal

    def __post_init__(self) -> None:
        for field_name, value in (
            ("open", self.open),
            ("high", self.high),
            ("low", self.low),
            ("close", self.close),
        ):
            if not value.is_finite():
                raise CostStressError(f"bar.{field_name} must be finite, got {value}")
        if self.low > self.high:
            raise CostStressError(
                f"bar invariant violated: low ({self.low}) > high ({self.high})"
            )
        if not (self.low <= self.open <= self.high):
            raise CostStressError("bar invariant violated: open outside [low, high]")
        if not (self.low <= self.close <= self.high):
            raise CostStressError("bar invariant violated: close outside [low, high]")


@dataclass(frozen=True, slots=True)
class WidenedBarPair:
    bid: Bar
    ask: Bar


def widen_bid_ask_bar(
    bid_bar: Bar, ask_bar: Bar, multiplier: Decimal
) -> WidenedBarPair:
    """Conservatively widen one M1 BID/ASK OHLC bar pair around the bar's
    CLOSE-based midpoint.

    DISCOVERED AMBIGUITY (see repo audit): the M1 BID and ASK OHLC bars
    consumed by the execution engines in this repo
    (``ftmoquant.research.alpha_lab.wick_fvg_squeeze_execution.load_m1_bidask``,
    ``ftmoquant.data.derived_bars``) are derived INDEPENDENTLY per side from
    each side's own tick stream (``derived_bars.py``'s ``_SIDES = ("BID",
    "ASK")`` accumulators are built and validated separately, only checked
    for matching *timestamp coverage*, never for matching *intrabar
    timing* of their own extremes). This means ``bid_bar.high`` and
    ``ask_bar.high`` are not guaranteed to have occurred at the same
    intraminute instant, and likewise for the two sides' lows. Naively
    widening each OHLC field around a per-field cross-side midpoint
    (``(bid_high + ask_high) / 2``, ``(bid_low + ask_low) / 2``, ...) would
    therefore assume a simultaneity the source data does not guarantee --
    exactly the ambiguity the B3.1 task brief asked to be surfaced rather
    than silently resolved in a possibly-favorable direction.

    RESOLUTION CHOSEN (conservative, not the naive one): only the bar's
    CLOSE is treated as a genuinely paired, single instant -- this is not
    a new assumption; it is the same one the existing execution engine
    already relies on (``wick_fvg_squeeze_execution.simulate_trades`` uses
    ``ask_m1["close"]``/``bid_m1["close"]`` as *the* executable quote for
    entries). From that one well-defined midpoint
    (``mid_close = (bid_bar.close + ask_bar.close) / 2``) and its implied
    half-spread (``half_spread_close = (ask_bar.close - bid_bar.close) /
    2``), every field of the BID bar is shifted down by a single constant
    offset ``(multiplier - 1) * half_spread_close`` and every field of the
    ASK bar is shifted up by the same constant. This is a pure translation
    (not an independent per-field rescale), so it:

    - preserves every OHLC ordering invariant exactly (``low <= open,
      close <= high`` is invariant under adding the same constant to all
      four fields);
    - preserves the bar's own close-based midpoint exactly
      (``stressed_mid_close == mid_close``);
    - NEVER narrows any field's distance from that midpoint: the offset is
      strictly >= 0 for ``multiplier >= 1``, so ``stressed_bid.low <=
      bid_bar.low``, ``stressed_bid.high <= bid_bar.high`` is NOT claimed
      (high moves down too, since the whole bar translates), but every
      field's *signed distance from mid_close* strictly increases (or
      stays equal at multiplier == 1) -- i.e. transaction cost, however
      the caller's execution engine measures it against this bar, is never
      reduced by this transform.

    This is a documented, deterministic, midpoint-preserving-where-defined,
    conservative choice -- not a claim that ``stressed_bid.high``/``.low``
    reproduce a real widened tick that would actually have printed.
    Callers relying on exact per-tick intrabar stop/target geometry beyond
    what this translation provides should treat that as a known,
    documented limitation of M1-bar-level stress testing, not a defect to
    silently work around.
    """

    _validate_quote(bid_bar.close, ask_bar.close)
    _validate_multiplier(multiplier)
    half_spread_close = (ask_bar.close - bid_bar.close) / 2
    offset = (multiplier - ONE) * half_spread_close

    def _shift(bar: Bar, delta: Decimal) -> Bar:
        return Bar(
            open=bar.open + delta,
            high=bar.high + delta,
            low=bar.low + delta,
            close=bar.close + delta,
        )

    return WidenedBarPair(bid=_shift(bid_bar, -offset), ask=_shift(ask_bar, offset))


_OHLC_COLUMNS = ("open", "high", "low", "close")


def _to_decimal_series(column: pd.Series) -> pd.Series:
    """``Decimal(str(x))`` per element -- the SAME conversion
    :func:`widen_bid_ask_bar` uses per field -- applied via ``.map`` rather
    than a ``for`` loop. ``.map(str)`` calls Python's own built-in ``str``
    (never NumPy's/pandas' own float formatting, which can differ), so this
    is the identical decimal value the scalar/bar path would have used."""

    return column.map(str).map(Decimal)


def widen_bid_ask_frame(
    bid_m1: pd.DataFrame, ask_m1: pd.DataFrame, multiplier: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frame-level counterpart of :func:`widen_bid_ask_bar`, operating on
    the exact BID/ASK DataFrame shape produced by
    ``ftmoquant.research.alpha_lab.wick_fvg_squeeze_execution.load_m1_bidask``
    (float ``open``/``high``/``low``/``close`` columns, a shared
    ``DatetimeIndex``). Each row is transformed independently -- no rolling
    window, no future observation, no period-wide average spread -- so
    prefix invariance holds exactly: widening rows ``[0, k)`` alone
    produces identical output to widening rows ``[0, n)`` and truncating to
    ``[0, k)``.

    Applies the same close-anchored translation as :func:`widen_bid_ask_bar`
    -- vectorized (pandas ``Series``-level Decimal arithmetic, no
    ``.iterrows()``, no per-row ``Bar``/dict construction) rather than
    row-by-row, but using the IDENTICAL ``Decimal(str(x))`` conversion and
    arithmetic ordering, so the result is provably bit-for-bit identical to
    the original row-by-row implementation -- see
    ``tests/research/alpha_lab/test_cost_stress.py``'s parity tests against
    a preserved reference implementation of the original algorithm
    (required because native float64 arithmetic on the SAME inputs was
    empirically shown, during the B3.2c performance audit, to disagree
    with the ``Decimal(str(x))``-mediated result on a meaningful fraction
    of realistic FX-precision rows -- so plain float64 vectorization was
    rejected as not observably equivalent, per that audit's explicit
    exact-equality standard). Batch-validates every row's finiteness and
    OHLC/crossed-market invariants BEFORE constructing any output, so a bad
    row anywhere in the frame still fails closed exactly as before -- no
    partial result is ever returned.
    """

    if not bid_m1.index.equals(ask_m1.index):
        raise CostStressError("bid_m1 and ask_m1 must share the identical paired index")
    for frame, label in ((bid_m1, "bid_m1"), (ask_m1, "ask_m1")):
        missing = [column for column in _OHLC_COLUMNS if column not in frame.columns]
        if missing:
            raise CostStressError(f"{label} is missing OHLC column(s): {missing}")

    decimal_multiplier = Decimal(str(multiplier))
    _validate_multiplier(decimal_multiplier)

    for frame, label in ((bid_m1, "bid_m1"), (ask_m1, "ask_m1")):
        finite = np.isfinite(frame[list(_OHLC_COLUMNS)].to_numpy(dtype=float))
        if not finite.all():
            raise CostStressError(f"{label} contains a non-finite OHLC value")
        if (frame["low"] > frame["high"]).any():
            raise CostStressError(f"{label} bar invariant violated: low > high")
        if ((frame["open"] < frame["low"]) | (frame["open"] > frame["high"])).any():
            raise CostStressError(
                f"{label} bar invariant violated: open outside [low, high]"
            )
        if ((frame["close"] < frame["low"]) | (frame["close"] > frame["high"])).any():
            raise CostStressError(
                f"{label} bar invariant violated: close outside [low, high]"
            )
    if (ask_m1["close"] < bid_m1["close"]).any():
        raise CostStressError(
            "crossed market rejected: ask close < bid close for one or more bars"
        )

    bid_close_dec = _to_decimal_series(bid_m1["close"])
    ask_close_dec = _to_decimal_series(ask_m1["close"])
    half_spread_close = (ask_close_dec - bid_close_dec) / 2
    offset = (decimal_multiplier - ONE) * half_spread_close

    stressed_bid: dict[str, pd.Series] = {}
    stressed_ask: dict[str, pd.Series] = {}
    for column in _OHLC_COLUMNS:
        bid_col_dec = _to_decimal_series(bid_m1[column])
        ask_col_dec = _to_decimal_series(ask_m1[column])
        stressed_bid[column] = (bid_col_dec - offset).map(float)
        stressed_ask[column] = (ask_col_dec + offset).map(float)

    return (
        pd.DataFrame(stressed_bid, index=bid_m1.index),
        pd.DataFrame(stressed_ask, index=ask_m1.index),
    )


def required_multiplier_for_family(
    family_id: Literal["B3F1", "B3F2", "B3F3"],
) -> tuple[Decimal, ...]:
    """The exact frozen cost-stress multipliers each Batch 3 family must
    survive, per the v2 preregistration's
    ``development_gates.transaction_cost_sensitivity.family_requirements``.
    A thin, literal lookup -- carries no signal or pair knowledge.
    """

    if family_id == "B3F1":
        return (Decimal("1.5"),)
    if family_id in ("B3F2", "B3F3"):
        return (Decimal("1.5"), Decimal("2.0"))
    raise CostStressError(f"unknown Batch 3 family id: {family_id!r}")
