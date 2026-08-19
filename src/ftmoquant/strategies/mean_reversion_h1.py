"""Execution-promotion signal layer for the frozen, validated short-horizon
mean-reversion family (H1, lookback 40, z_entry 2.0), promoted from
:func:`ftmoquant.research.alpha_lab.families.mean_reversion_signals`.

This module is deliberately narrow: it reproduces that exact frozen signal
*causally* (bar-by-bar, using only information available at or before each
bar's close) and exposes a decision/execution time split so a Nautilus
``Strategy`` can submit orders no earlier than the bar strictly after the one
that produced the signal. It does not simulate fills, P&L, spread, or costs
-- those remain owned by the existing Nautilus execution harness
(:mod:`ftmoquant.backtest.execution_harness`) and its private
``_add_venue``/``_engine_config``/``_fee_model``/``_rollover_modules``
helpers, exactly as ``eurusd_tsm_development.py`` already does. This is not
a second backtester.

Position sizing reuses :mod:`ftmoquant.research.g1.normalization` (the
causal 1%-annualized-volatility target already used identically by every
other G1-family validated strategy) unchanged.

Frozen parameters -- see docs/research/mean_reversion_h1_v1_execution_promotion.md
for the full audit and design freeze this module implements:

    family:    short-horizon mean reversion
    timeframe: H1
    lookback:  40
    z_entry:   2.0
    universe:  the seven OANDA alpha-lab pairs (AUD/USD, EUR/USD, GBP/USD,
               NZD/USD, USD/CAD, USD/CHF, USD/JPY)

None of these may be changed without re-running the frozen DEVELOPMENT/
VALIDATION screening evidence this promotion is based on.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ftmoquant.research.g1.normalization import (
    CausalEwmaDailyVolatility,
    CausalExposureDecision,
    G1VolatilityNormalizer,
)
from ftmoquant.research.stage_g import HOLDOUT_START

FROZEN_TIMEFRAME = "H1"
FROZEN_LOOKBACK = 40
FROZEN_Z_ENTRY = 2.0
FROZEN_UNIVERSE: tuple[str, ...] = (
    "AUD/USD.OANDA",
    "EUR/USD.OANDA",
    "GBP/USD.OANDA",
    "NZD/USD.OANDA",
    "USD/CAD.OANDA",
    "USD/CHF.OANDA",
    "USD/JPY.OANDA",
)

#: Execution-timing policy frozen in the design doc: a signal is decided
#: from bar t's close; the earliest an order may be submitted/filled is
#: strictly after that instant (i.e. no earlier than the next bar/tick).
#: Enforced here by construction: on_bar(t) only ever reveals state *for*
#: bar t once bar t has fully closed, and the caller (a Nautilus Strategy)
#: can only act on that state once its own on_bar(t) handler runs, which
#: Nautilus itself only invokes at or after bar t's close -- any resulting
#: order is processed by the venue strictly after that dispatch.
EXECUTION_TIMING_POLICY = (
    "decision_time = bar_close_time; "
    "earliest_executable_time = strictly_after(decision_time)"
)


class MeanReversionSignalError(ValueError):
    """Raised on any violation of the frozen mean-reversion promotion
    contract: wrong parameters, non-causal access, or holdout access."""


@dataclass(frozen=True, slots=True)
class CausalSignalBar:
    """The causal state of one instrument immediately after processing one
    closed bar. ``position`` is the signal's target direction (-1/0/+1)
    *as of* ``timestamp_ns`` -- an execution layer may only act on it from
    a strictly later timestamp."""

    timestamp_ns: int
    close: float
    z_score: float | None
    position: int


class CausalMeanReversionSignal:
    """Bar-by-bar reproduction of
    :func:`ftmoquant.research.alpha_lab.families.mean_reversion_signals`'s
    exact math (rolling mean/std with ``ddof=1`` over the last
    ``lookback`` closes, current bar included) and exact state-transition
    rules (fade beyond ``z_entry``, exit on mean-cross, reverse on an
    opposite-direction trigger) -- reimplemented incrementally because
    vectorbt's vectorized rolling arrays cannot be replayed causally bar by
    bar. Numerical/behavioral equivalence to the vectorized original is
    proven in ``tests/strategies/test_mean_reversion_h1.py``.
    """

    def __init__(
        self, *, lookback: int = FROZEN_LOOKBACK, z_entry: float = FROZEN_Z_ENTRY
    ) -> None:
        if lookback != FROZEN_LOOKBACK or z_entry != FROZEN_Z_ENTRY:
            raise MeanReversionSignalError(
                "mean-reversion promotion parameters are frozen at "
                f"lookback={FROZEN_LOOKBACK}, z_entry={FROZEN_Z_ENTRY}"
            )
        self._lookback = lookback
        self._z_entry = z_entry
        self._window: deque[float] = deque(maxlen=lookback)
        self._position = 0
        self._last_timestamp_ns: int | None = None

    def on_bar(self, timestamp_ns: int, close: float) -> CausalSignalBar:
        """Process exactly one newly closed bar, strictly after any prior
        one. Never looks at, and cannot be given, a later bar first."""

        if (
            self._last_timestamp_ns is not None
            and timestamp_ns <= self._last_timestamp_ns
        ):
            raise MeanReversionSignalError(
                "bars must be processed in strictly increasing timestamp order"
            )
        self._last_timestamp_ns = timestamp_ns
        self._window.append(close)

        z_score: float | None = None
        if len(self._window) == self._lookback:
            mean = sum(self._window) / self._lookback
            variance = sum((x - mean) ** 2 for x in self._window) / (self._lookback - 1)
            std = variance**0.5
            if std > 0.0:
                z_score = (close - mean) / std

        self._position = _next_position(self._position, z_score, self._z_entry)
        return CausalSignalBar(timestamp_ns, close, z_score, self._position)


def _next_position(position: int, z_score: float | None, z_entry: float) -> int:
    """Exact state-transition table matching
    ``families.mean_reversion_signals``'s four boolean conditions
    (``z < -z_entry`` / ``z > z_entry`` / ``z >= 0`` / ``z <= 0``) plus
    vectorbt's default reverse-on-opposite-entry handling.

    One case is genuinely ambiguous in the source vectorbt implementation
    itself and is resolved deterministically here rather than silently:
    a single bar can simultaneously satisfy an exit condition *and* an
    opposite-direction entry condition (e.g. while long, z jumping past
    +z_entry in one bar also satisfies ``z >= 0``). vectorbt's own
    same-bar conflict resolution for this exact combination is not
    verified here; this implementation always treats it as a direct
    reversal (net effect identical either way -- see the design doc).
    """

    if z_score is None:
        return position
    if position == 0:
        if z_score < -z_entry:
            return 1
        if z_score > z_entry:
            return -1
        return 0
    if position == 1:
        if z_score > z_entry:
            return -1  # opposite entry -> reverse (documented ambiguous case)
        if z_score >= 0:
            return 0
        return 1
    if z_score < -z_entry:
        return 1  # opposite entry -> reverse (documented ambiguous case)
    if z_score <= 0:
        return 0
    return -1


def causal_signal_stream(
    bars: Iterable[tuple[int, float]],
    *,
    lookback: int = FROZEN_LOOKBACK,
    z_entry: float = FROZEN_Z_ENTRY,
) -> tuple[CausalSignalBar, ...]:
    """Convenience wrapper: process a whole chronological bar stream."""

    signal = CausalMeanReversionSignal(lookback=lookback, z_entry=z_entry)
    return tuple(signal.on_bar(ts, close) for ts, close in bars)


def reject_final_holdout(timestamp_ns: int) -> None:
    """Fail closed on any timestamp at or after the frozen final-holdout
    boundary. This promotion stage is authorized only through the
    already-observed VALIDATION period."""

    holdout_start_ns = int(HOLDOUT_START.timestamp() * 1_000_000_000)
    if timestamp_ns >= holdout_start_ns:
        raise MeanReversionSignalError(
            "final holdout is not accessible during execution promotion"
        )


def size_positions_across_pairs(
    *,
    signals: Mapping[str, CausalSignalBar],
    estimators: Mapping[str, CausalEwmaDailyVolatility],
    decision_time_ns: int,
    normalizer: G1VolatilityNormalizer | None = None,
) -> dict[str, CausalExposureDecision]:
    """Size every frozen pair's causal signal to the same ex-ante
    volatility target, reusing :class:`G1VolatilityNormalizer` unchanged.
    Requires exactly the seven frozen pairs -- no pair may be silently
    excluded from the aggregate."""

    if set(signals) != set(FROZEN_UNIVERSE) or set(estimators) != set(FROZEN_UNIVERSE):
        raise MeanReversionSignalError(
            "position sizing requires exactly the seven frozen pairs, no exclusions"
        )
    active_normalizer = normalizer or G1VolatilityNormalizer()
    return {
        instrument_id: active_normalizer.target_at(
            float(signals[instrument_id].position),
            decision_time_ns,
            estimators[instrument_id],
        )
        for instrument_id in FROZEN_UNIVERSE
    }
