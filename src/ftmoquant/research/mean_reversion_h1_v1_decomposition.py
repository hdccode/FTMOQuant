"""Fixed diagnostic decomposition of the mean_reversion_h1_v1 DEVELOPMENT
degradation between the already-validated VectorBT screening result and the
already-observed real Nautilus execution-promotion result.

This is attribution only, not a new search or optimization stage: every
variant below reuses the identical, unchanged, frozen causal signal
(``ftmoquant.strategies.mean_reversion_h1.CausalMeanReversionSignal`` /
``ftmoquant.research.alpha_lab.families.mean_reversion_signals`` -- the same
function, same ``lookback=40``/``z_entry=2.0``) and differs *only* in
execution timing, spread treatment, and position sizing, exactly as
specified by the five frozen variants:

    A. ORIGINAL SCREENING REFERENCE -- VectorBT semantics/sizing, same-bar
       close, existing screening fee assumption. Reused directly from
       ``ftmoquant.research.alpha_lab.screening_common`` (the exact private
       helpers ``run_family_grid`` itself calls) -- not reimplemented.
    B. TIMING-ONLY -- original screening (all-in, ``accumulate=False``,
       reverse-on-opposite-entry) sizing, unchanged signal, execution moved
       to the first genuine M1 observation strictly after the H1 decision,
       filled at that observation's MIDPOINT, zero additional costs.
    C. SPREAD-ONLY ON TOP OF B -- identical to B except execution crosses
       the genuine M1 BID/ASK (buy at ASK, sell at BID) instead of the
       midpoint.
    D. VOL-SIZING EFFECT -- current frozen ``G1VolatilityNormalizer``
       (1% annualized target, EWMA ``com=60``) sizing, strictly-later
       execution, MIDPOINT fill, zero additional costs. Reuses
       ``mean_reversion_h1_development.run_frozen_signal_backtest``
       unchanged -- fed synthetic zero-spread M1 bars (``bid.close ==
       ask.close == genuine mid``) instead of the real BID/ASK stream, so
       every other mechanic (signal, sizing, timing, the real Nautilus
       ``BacktestEngine``) is the exact, unmodified production code path.
    E. CURRENT PROMOTION IMPLEMENTATION -- current sizing, strictly-later
       execution, genuine BID/ASK crossing, the frozen
       ``canonical_execution_profile()``. This *is*
       ``run_mean_reversion_h1_development``'s own per-pair engine call,
       reused unchanged (just orchestrated per pair here to also capture
       per-pair detail without writing partition artifacts).

B/C's "original screening sizing, strictly-later execution" combination
does not exist anywhere else in the repo (VectorBT has no concept of
execution delay; the current Nautilus promotion path only ever sizes with
G1VolatilityNormalizer) -- :func:`_resolve_direction_transitions` and
:func:`_compound_all_in_equity` are the one tiny, deterministic calculator
this decomposition adds, built specifically to isolate timing/spread at
VectorBT's own all-in sizing convention. They reuse
``mean_reversion_h1_development``'s already-tested strictly-later-timing
primitives (``_paired_timestamps``, ``_first_strictly_later_paired_ns``,
``_paired_midpoints``) and the same frozen ``CausalMeanReversionSignal``
directly -- no new signal or timing logic, only a new all-in equity
compounding rule, validated in this module's own test suite against
VectorBT's own same-bar-fill result (see
``tests/research/test_mean_reversion_h1_v1_decomposition.py``).

Every variant's return/Sharpe/profitable-pair-count is computed through
the identical, already-existing statistics pipeline
(``mean_reversion_h1_development._pair_daily_returns`` /
``_pair_performance`` / ``_aggregate_performance``) against an independent
$100,000-equivalent sleeve per pair, so results are directly comparable
across A-E without any further capital-basis normalization.

DEVELOPMENT-only. Never opens VALIDATION or final-holdout data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from nautilus_trader.model import Bar, CurrencyPair
from nautilus_trader.persistence import ParquetDataCatalog

import ftmoquant.research.mean_reversion_h1_development as dev
from ftmoquant.backtest.execution_harness import (
    _instrument_for_profile,
    canonical_execution_profile,
)
from ftmoquant.data.instruments import oanda_symbol
from ftmoquant.research.alpha_lab.data import AlphaLabDataset, load_alpha_lab_dataset
from ftmoquant.research.alpha_lab.families import mean_reversion_signals
from ftmoquant.research.alpha_lab.screening_common import (
    _build_portfolio,
    screening_cost_per_instrument,
)
from ftmoquant.research.eurusd_tsm_development import EquityPoint
from ftmoquant.research.stage_g import DEVELOPMENT_END_EXCLUSIVE, DEVELOPMENT_START
from ftmoquant.strategies.mean_reversion_h1 import (
    FROZEN_LOOKBACK,
    FROZEN_UNIVERSE,
    FROZEN_Z_ENTRY,
    CausalMeanReversionSignal,
    reject_final_holdout,
)

#: The frozen OANDA alpha-lab DEVELOPMENT lineage's own canonical paths --
#: identical to mean_reversion_h1_development.py's own production catalog
#: roots. VALIDATION/holdout paths are never referenced anywhere in this
#: module.
DEVELOPMENT_CATALOG_ROOT = Path(
    "/Users/Shared/FTMOQuant-data/oanda_fx_alpha_lab_v1/canonical"
)
DEVELOPMENT_READINESS_PATH = Path(
    "/Users/Shared/FTMOQuant-data/oanda_fx_alpha_lab_v1/readiness/"
    "ftmoquant_oanda_alpha_lab_readiness.json"
)

#: A common $100,000-equivalent independent sleeve per pair for every
#: variant (Section 3: common return basis) -- matches
#: mean_reversion_h1_development.AccountParameters.initial_capital exactly,
#: so D/E's real Nautilus sleeves and A/B/C's equity curves are on the
#: identical capital basis without further rescaling.
DIAGNOSTIC_INITIAL_CAPITAL = Decimal("100000")


class DecompositionError(ValueError):
    """Raised on any diagnostic-decomposition data or logic failure --
    never silently substitutes a default or invents a rescue value."""


def load_development_dataset() -> AlphaLabDataset:
    """The same aligned H1 mid/bid/ask DEVELOPMENT matrices the frozen
    VectorBT screening itself consumed -- ``dataset.close`` is exactly
    ``(bid.close + ask.close) / 2``, identical to what
    ``mean_reversion_h1_development._precompute_h1_decisions`` computes
    from the real H1 catalog bars, so A and B/C/D/E all decide from the
    same causal input by construction."""

    return load_alpha_lab_dataset(
        readiness_path=DEVELOPMENT_READINESS_PATH,
        development_root_dir=DEVELOPMENT_CATALOG_ROOT,
        timeframe="H1",
        source="oanda",
    )


# ---------------------------------------------------------------------------
# Common result types + the one shared statistics pipeline (Section 3):
# every variant reduces to (equity_points_by_pair, realized_cost_by_pair),
# then reuses mean_reversion_h1_development's own, already-tested
# pair/aggregate performance functions unchanged.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VariantResult:
    label: str
    pair_performance: dict[str, dev.PairPerformance]
    aggregate: dev.AggregatePerformance


def _variant_result(
    label: str,
    equity_points_by_pair: Mapping[str, tuple[EquityPoint, ...]],
    realized_cost_by_pair: Mapping[str, Decimal],
) -> VariantResult:
    if set(equity_points_by_pair) != set(FROZEN_UNIVERSE):
        raise DecompositionError(
            f"{label}: exactly the seven frozen pairs are required"
        )
    pair_performance: dict[str, dev.PairPerformance] = {}
    daily_returns_by_pair: dict[str, tuple[tuple[Any, float], ...]] = {}
    for instrument_id in FROZEN_UNIVERSE:
        points = equity_points_by_pair[instrument_id]
        daily = dev._pair_daily_returns(points, DIAGNOSTIC_INITIAL_CAPITAL)
        daily_returns_by_pair[instrument_id] = daily
        pair_performance[instrument_id] = dev._pair_performance(
            instrument_id,
            points,
            daily,
            realized_cost_by_pair.get(instrument_id, Decimal(0)),
            DIAGNOSTIC_INITIAL_CAPITAL,
        )
    aggregate = dev._aggregate_performance(pair_performance, daily_returns_by_pair)
    return VariantResult(
        label=label, pair_performance=pair_performance, aggregate=aggregate
    )


def _last_of_day(points: Sequence[EquityPoint]) -> tuple[EquityPoint, ...]:
    """Reduce an arbitrary (possibly sub-daily) chronological EquityPoint
    sequence to one mark per UTC calendar day, keeping the very first point
    as the unconditional seed -- matches
    ``_MeanReversionH1Executor``'s own daily-mark convention exactly, so
    every variant's Sharpe is computed at the identical granularity."""

    if not points:
        raise DecompositionError("no equity points to reduce")
    seed = points[0]
    by_day: dict[Any, EquityPoint] = {}
    for point in points[1:]:
        day = datetime.fromtimestamp(
            point.information_time_ns / 1_000_000_000, tz=UTC
        ).date()
        by_day[day] = point
    return (seed, *(by_day[day] for day in sorted(by_day)))


# ---------------------------------------------------------------------------
# Variant A: original VectorBT screening reference -- reused unchanged.
# ---------------------------------------------------------------------------


def variant_a_screening_reference(dataset: AlphaLabDataset) -> VariantResult:
    """Same-bar-close, original all-in VectorBT sizing, the existing
    screening fee assumption (``screening_cost_per_instrument`` /
    ``SCREENING_COST_LABEL``) -- ``_build_portfolio`` is the exact private
    helper ``run_family_grid`` itself calls per instrument; nothing here
    reimplements VectorBT's own sizing/reversal/fee mechanics."""

    entries, exits, short_entries, short_exits = mean_reversion_signals(
        dataset.close, FROZEN_LOOKBACK, FROZEN_Z_ENTRY
    )
    cost = screening_cost_per_instrument(dataset)
    equity_points_by_pair: dict[str, tuple[EquityPoint, ...]] = {}
    for instrument_id in FROZEN_UNIVERSE:
        fee = float(cost[instrument_id])
        portfolio = _build_portfolio(
            dataset.close[instrument_id],
            entries[instrument_id],
            exits[instrument_id],
            short_entries[instrument_id],
            short_exits[instrument_id],
            fees=fee,
            freq="1h",
        )
        value = portfolio.value()
        # VectorBT's own init_cash=100.0 (vbt.settings.portfolio["init_cash"]),
        # rescaled onto DIAGNOSTIC_INITIAL_CAPITAL -- valid because the
        # sizing rule (100% of currently-available cash) is scale-invariant
        # in percentage-return terms; see the module docstring.
        scale = DIAGNOSTIC_INITIAL_CAPITAL / Decimal("100")
        hourly_points = tuple(
            EquityPoint(
                int(ts.timestamp() * 1_000_000_000), Decimal(str(equity)) * scale
            )
            for ts, equity in value.items()
        )
        equity_points_by_pair[instrument_id] = _last_of_day(hourly_points)
    zero_cost = dict.fromkeys(FROZEN_UNIVERSE, Decimal(0))
    return _variant_result("A_screening_reference", equity_points_by_pair, zero_cost)


def target_sequence_signature(dataset: AlphaLabDataset) -> dict[str, tuple[int, ...]]:
    """VectorBT's own realized target-direction sequence per pair (sign of
    cumulative signed asset flow), for the Section 4 identity proof against
    the causal signal's own position sequence."""

    import numpy as np
    import vectorbt as vbt  # type: ignore[import-untyped]

    entries, exits, short_entries, short_exits = mean_reversion_signals(
        dataset.close, FROZEN_LOOKBACK, FROZEN_Z_ENTRY
    )
    cost = screening_cost_per_instrument(dataset)
    result: dict[str, tuple[int, ...]] = {}
    for instrument_id in FROZEN_UNIVERSE:
        portfolio = vbt.Portfolio.from_signals(
            dataset.close[instrument_id],
            entries[instrument_id],
            exits[instrument_id],
            short_entries=short_entries[instrument_id],
            short_exits=short_exits[instrument_id],
            fees=float(cost[instrument_id]),
            freq="1h",
        )
        flow = portfolio.asset_flow(direction="both").cumsum()
        result[instrument_id] = tuple(int(np.sign(v)) for v in flow)
    return result


def causal_position_sequence(dataset: AlphaLabDataset) -> dict[str, tuple[int, ...]]:
    """The frozen causal signal's own position sequence (identical
    lookback/z_entry, identical mid-price input) -- the shared signal input
    for B, C, D, and E."""

    result: dict[str, tuple[int, ...]] = {}
    for instrument_id in dataset.instrument_ids:
        signal = CausalMeanReversionSignal()
        positions = []
        for ts, close in dataset.close[instrument_id].items():
            bar = signal.on_bar(int(ts.timestamp() * 1_000_000_000), float(close))
            positions.append(bar.position)
        result[instrument_id] = tuple(positions)
    return result


# ---------------------------------------------------------------------------
# Real M1 catalog loading (shared by B/C/D/E) -- same bar-type conventions
# already established and tested in mean_reversion_h1_development.py.
# ---------------------------------------------------------------------------


def _load_pair_bars(
    instrument_id: str, dataset_symbol: str
) -> tuple[CurrencyPair, dict[str, tuple[Bar, ...]], dict[str, tuple[Bar, ...]]]:
    """One pair's frozen CurrencyPair instrument, real H1 BID/ASK bars
    (offline signal input only), and real M1 BID/ASK bars (the sole
    engine-injected execution stream) -- identical loading convention to
    ``run_mean_reversion_h1_development``'s own per-pair loop."""

    root = DEVELOPMENT_CATALOG_ROOT / oanda_symbol(dataset_symbol)
    catalog = ParquetDataCatalog(str(root / "catalog"))
    found = catalog.instruments([instrument_id])
    if len(found) != 1 or not isinstance(found[0], CurrencyPair):
        raise DecompositionError(f"frozen CurrencyPair is unavailable: {instrument_id}")
    profile = canonical_execution_profile()
    instrument = _instrument_for_profile(found[0], profile.fee)

    start_ns = dev._ns(DEVELOPMENT_START)
    end_ns = dev._ns(DEVELOPMENT_END_EXCLUSIVE)

    h1_by_side: dict[str, tuple[Bar, ...]] = {}
    for side in ("BID", "ASK"):
        bar_type = (
            f"{instrument_id}-{dev._H1_BAR_TIMEFRAME}-{side}-"
            f"{dev._CATALOG_BAR_AGGREGATION}"
        )
        bars = tuple(catalog.query_bars([bar_type], start=start_ns, end=end_ns - 1))
        dev._validate_bar_stream(bars, bar_type, start_ns, end_ns)
        h1_by_side[side] = bars

    m1_by_side: dict[str, tuple[Bar, ...]] = {}
    for side in ("BID", "ASK"):
        bar_type = (
            f"{instrument_id}-{dev._M1_BAR_TIMEFRAME}-{side}-"
            f"{dev._ENGINE_BAR_AGGREGATION}"
        )
        bars = tuple(catalog.query_bars([bar_type], start=start_ns, end=end_ns - 1))
        dev._validate_bar_stream(bars, bar_type, start_ns, end_ns)
        m1_by_side[side] = bars

    return instrument, h1_by_side, m1_by_side


def _dataset_symbol_by_instrument() -> dict[str, str]:
    from ftmoquant.data.oanda_alpha_lab_development import load_oanda_alpha_lab_config

    config = load_oanda_alpha_lab_config()
    return {
        instrument_id: config.instrument(instrument_id).dataset_symbol
        for instrument_id in FROZEN_UNIVERSE
    }


# ---------------------------------------------------------------------------
# B/C: the one small deterministic calculator this decomposition adds --
# original screening (all-in, no re-sizing while direction is unchanged,
# reverse-on-opposite-entry) sizing at strictly-later execution. Reuses
# the frozen CausalMeanReversionSignal and mean_reversion_h1_development's
# own strictly-later-timing primitives; the only new logic is the all-in
# equity compounding rule below.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _DirectionTransition:
    decision_ns: int
    execution_ns: int
    new_direction: int
    execution_mid: Decimal
    execution_bid: Decimal
    execution_ask: Decimal


def _resolve_direction_transitions(
    h1_bid_bars: tuple[Bar, ...],
    h1_ask_bars: tuple[Bar, ...],
    m1_bid_bars: tuple[Bar, ...],
    m1_ask_bars: tuple[Bar, ...],
    start_ns: int,
    end_exclusive_ns: int,
) -> tuple[_DirectionTransition, ...]:
    """Every raw causal-signal direction change (-1/0/+1), resolved to its
    strictly-later M1 execution observation -- the direction-only sibling
    of ``mean_reversion_h1_development._precompute_h1_decisions``, reusing
    its exact timing-resolution primitives, differing only in tracking raw
    ``position`` instead of a G1-sized ``target_units`` (B/C use VectorBT's
    own all-in sizing, computed downstream in
    :func:`_compound_all_in_equity`, not G1)."""

    by_ts: dict[int, dict[str, Bar]] = {}
    for bar in h1_bid_bars:
        by_ts.setdefault(bar.ts_event, {})["BID"] = bar
    for bar in h1_ask_bars:
        by_ts.setdefault(bar.ts_event, {})["ASK"] = bar

    paired_m1_ns = dev._paired_timestamps(m1_bid_bars, m1_ask_bars)
    paired_m1_mid = dev._paired_midpoints(m1_bid_bars, m1_ask_bars)
    m1_bid_by_ts = {bar.ts_event: bar.close.as_decimal() for bar in m1_bid_bars}
    m1_ask_by_ts = {bar.ts_event: bar.close.as_decimal() for bar in m1_ask_bars}

    signal = CausalMeanReversionSignal()
    direction = 0
    transitions: list[_DirectionTransition] = []
    for decision_ns in sorted(by_ts):
        if decision_ns < start_ns or decision_ns >= end_exclusive_ns:
            continue
        sides = by_ts[decision_ns]
        if "BID" not in sides or "ASK" not in sides:
            continue
        reject_final_holdout(decision_ns)

        mid = (float(sides["BID"].close) + float(sides["ASK"].close)) / 2.0
        signal_bar = signal.on_bar(decision_ns, mid)
        if signal_bar.position == direction:
            continue

        execution_ns = dev._first_strictly_later_paired_ns(paired_m1_ns, decision_ns)
        if execution_ns is None or execution_ns >= end_exclusive_ns:
            continue
        if transitions and transitions[-1].execution_ns == execution_ns:
            raise DecompositionError(
                "multiple direction transitions mapped onto the same M1 "
                "execution observation"
            )
        transitions.append(
            _DirectionTransition(
                decision_ns=decision_ns,
                execution_ns=execution_ns,
                new_direction=signal_bar.position,
                execution_mid=paired_m1_mid[execution_ns],
                execution_bid=m1_bid_by_ts[execution_ns],
                execution_ask=m1_ask_by_ts[execution_ns],
            )
        )
        direction = signal_bar.position
    return tuple(transitions)


def _compound_all_in_equity(
    h1_bid_bars: tuple[Bar, ...],
    h1_ask_bars: tuple[Bar, ...],
    transitions: tuple[_DirectionTransition, ...],
    *,
    use_spread: bool,
    initial_capital: Decimal,
) -> tuple[EquityPoint, ...]:
    """VectorBT-equivalent all-in, ``accumulate=False``,
    reverse-on-opposite-entry compounding: 100% of current equity is held
    long or short (or flat) at all times; a transition neither adds to nor
    partially reduces the existing position, it fully reverses/opens/closes
    it in one fill. Marks continuously against the H1 mid between
    transitions (identical information to VectorBT's own bar-close marking)
    and, on a transition, splits the interval at the transition's own
    strictly-later execution price/timestamp: the OLD direction is marked
    up to execution, the NEW direction from execution onward -- the direct,
    literal consequence of delaying execution, not an approximation.

    Validated against VectorBT's own ``portfolio.value()`` in same-bar-fill
    mode (transitions collapsed onto their own decision bar) in
    ``tests/research/test_mean_reversion_h1_v1_decomposition.py`` before
    being trusted for the delayed-execution B/C variants.
    """

    by_ts: dict[int, dict[str, Bar]] = {}
    for bar in h1_bid_bars:
        by_ts.setdefault(bar.ts_event, {})["BID"] = bar
    for bar in h1_ask_bars:
        by_ts.setdefault(bar.ts_event, {})["ASK"] = bar
    h1_mid_by_ts = tuple(
        (ts, (Decimal(str(sides["BID"].close)) + Decimal(str(sides["ASK"].close))) / 2)
        for ts, sides in sorted(by_ts.items())
        if "BID" in sides and "ASK" in sides
    )
    if not h1_mid_by_ts:
        raise DecompositionError("no paired H1 bars to compound equity over")

    events: list[tuple[int, int, Decimal, int | None]] = []
    # kind: 0 == execute (processed before a mark at the identical
    # timestamp, though this essentially never occurs at M1 granularity),
    # 1 == mark.
    for ts, mid in h1_mid_by_ts:
        events.append((ts, 1, mid, None))
    for transition in transitions:
        events.append(
            (
                transition.execution_ns,
                0,
                transition.execution_mid,
                transition.new_direction,
            )
        )
    events.sort(key=lambda item: (item[0], item[1]))

    execution_price_by_ns = {t.execution_ns: t for t in transitions}

    direction = 0
    last_ns, last_price = h1_mid_by_ts[0]
    equity = initial_capital
    points = [EquityPoint(last_ns, equity)]
    for ts, kind, mark_price, new_direction in events:
        if ts < last_ns:
            continue
        price = mark_price
        if kind == 0 and use_spread:
            transition = execution_price_by_ns[ts]
            side_is_buy = transition.new_direction > direction
            price = (
                transition.execution_ask if side_is_buy else transition.execution_bid
            )
        if ts > last_ns and direction != 0:
            equity = equity * (price / last_price) ** direction
        last_ns, last_price = ts, price
        if kind == 0:
            assert new_direction is not None
            direction = new_direction
        else:
            points.append(EquityPoint(ts, equity))
    return tuple(points)


def _bc_variant(
    dataset: AlphaLabDataset, *, use_spread: bool, label: str
) -> VariantResult:
    dataset_symbols = _dataset_symbol_by_instrument()
    equity_points_by_pair: dict[str, tuple[EquityPoint, ...]] = {}
    start_ns = dev._ns(DEVELOPMENT_START)
    end_ns = dev._ns(DEVELOPMENT_END_EXCLUSIVE)
    for instrument_id in FROZEN_UNIVERSE:
        _, h1_by_side, m1_by_side = _load_pair_bars(
            instrument_id, dataset_symbols[instrument_id]
        )
        transitions = _resolve_direction_transitions(
            h1_by_side["BID"],
            h1_by_side["ASK"],
            m1_by_side["BID"],
            m1_by_side["ASK"],
            start_ns,
            end_ns,
        )
        raw_points = _compound_all_in_equity(
            h1_by_side["BID"],
            h1_by_side["ASK"],
            transitions,
            use_spread=use_spread,
            initial_capital=DIAGNOSTIC_INITIAL_CAPITAL,
        )
        equity_points_by_pair[instrument_id] = _last_of_day(raw_points)
    return _variant_result(
        label, equity_points_by_pair, dict.fromkeys(FROZEN_UNIVERSE, Decimal(0))
    )


def variant_b_timing_only(dataset: AlphaLabDataset) -> VariantResult:
    return _bc_variant(dataset, use_spread=False, label="B_timing_only_midpoint")


def variant_c_spread_only(dataset: AlphaLabDataset) -> VariantResult:
    return _bc_variant(dataset, use_spread=True, label="C_spread_only")


# ---------------------------------------------------------------------------
# D/E: reuse run_frozen_signal_backtest UNCHANGED -- current frozen
# G1VolatilityNormalizer sizing and the real Nautilus BacktestEngine own
# every fill/timing/sizing mechanic. D's only difference from E is that the
# M1 stream fed to the engine has its spread synthetically zeroed
# (bid.close == ask.close == genuine mid) so bar_execution fills exactly at
# the midpoint; the engine, signal, and sizing code paths are byte-for-byte
# identical to production.
# ---------------------------------------------------------------------------


def _zero_spread_bars(
    bid_bars: tuple[Bar, ...], ask_bars: tuple[Bar, ...]
) -> tuple[tuple[Bar, ...], tuple[Bar, ...]]:
    """Rebuild each paired M1 BID/ASK observation with both sides set to
    the genuine mid -- same timestamps, same BarType, same OHLCV shape,
    only the BID/ASK spread is zeroed. Never invents a price: the mid used
    is exactly ``(real_bid.close + real_ask.close) / 2``."""

    bid_by_ts = {bar.ts_event: bar for bar in bid_bars}
    ask_by_ts = {bar.ts_event: bar for bar in ask_bars}
    shared = sorted(bid_by_ts.keys() & ask_by_ts.keys())
    new_bid: list[Bar] = []
    new_ask: list[Bar] = []
    for ts in shared:
        bid_bar, ask_bar = bid_by_ts[ts], ask_by_ts[ts]
        # Format at the instrument's own declared price precision -- a
        # mismatched-precision Price silently halts the Nautilus matching
        # engine after 20 consecutive events (see
        # mean_reversion_h1_development.py's own precision-mismatch
        # history); the raw string form of a Decimal average does not
        # reliably preserve the original precision.
        precision = bid_bar.close.precision
        mid_value = (bid_bar.close.as_decimal() + ask_bar.close.as_decimal()) / 2
        mid_price = type(bid_bar.close).from_str(f"{mid_value:.{precision}f}")
        new_bid.append(
            Bar(
                bar_type=bid_bar.bar_type,
                open=mid_price,
                high=mid_price,
                low=mid_price,
                close=mid_price,
                volume=bid_bar.volume,
                ts_event=ts,
                ts_init=bid_bar.ts_init,
            )
        )
        new_ask.append(
            Bar(
                bar_type=ask_bar.bar_type,
                open=mid_price,
                high=mid_price,
                low=mid_price,
                close=mid_price,
                volume=ask_bar.volume,
                ts_event=ts,
                ts_init=ask_bar.ts_init,
            )
        )
    return tuple(new_bid), tuple(new_ask)


def _de_variant(
    *, zero_spread: bool, label: str
) -> VariantResult:
    dataset_symbols = _dataset_symbol_by_instrument()
    equity_points_by_pair: dict[str, tuple[EquityPoint, ...]] = {}
    realized_cost_by_pair: dict[str, Decimal] = {}
    for instrument_id in FROZEN_UNIVERSE:
        instrument, h1_by_side, m1_by_side = _load_pair_bars(
            instrument_id, dataset_symbols[instrument_id]
        )
        if zero_spread:
            bid_bars, ask_bars = _zero_spread_bars(
                m1_by_side["BID"], m1_by_side["ASK"]
            )
            m1_by_side = {"BID": bid_bars, "ASK": ask_bars}

        outcome = dev.run_frozen_signal_backtest(
            start=DEVELOPMENT_START,
            end_exclusive=DEVELOPMENT_END_EXCLUSIVE,
            instruments={instrument_id: instrument},
            h1_bars_by_instrument={instrument_id: h1_by_side},
            m1_bars_by_instrument={instrument_id: m1_by_side},
        )
        equity_points_by_pair[instrument_id] = outcome.equity_points[instrument_id]
        realized_cost_by_pair[instrument_id] = outcome.realized_variable_cost[
            instrument_id
        ]
    return _variant_result(label, equity_points_by_pair, realized_cost_by_pair)


def variant_d_vol_sizing_midpoint() -> VariantResult:
    return _de_variant(zero_spread=True, label="D_vol_sizing_midpoint")


def variant_e_current_promotion() -> VariantResult:
    return _de_variant(zero_spread=False, label="E_current_promotion")


# ---------------------------------------------------------------------------
# Orchestration + classification (Section 5's decision rule, applied
# mechanically -- never a rescue rule invented after seeing results).
# ---------------------------------------------------------------------------

DecompositionClassification = str

SAME_BAR_EXECUTION_DEPENDENT: DecompositionClassification = (
    "SAME_BAR_EXECUTION_DEPENDENT"
)
NATIVE_SPREAD_INTOLERANT: DecompositionClassification = "NATIVE_SPREAD_INTOLERANT"
RISK_SIZING_TRANSFORMATION: DecompositionClassification = "RISK_SIZING_TRANSFORMATION"
PROCEED_TO_FROZEN_PROTOCOL: DecompositionClassification = "PROCEED_TO_FROZEN_PROTOCOL"


def classify(variants: Mapping[str, VariantResult]) -> DecompositionClassification:
    a = variants["A"].aggregate.equal_weight_net_return
    b = variants["B"].aggregate.equal_weight_net_return
    c = variants["C"].aggregate.equal_weight_net_return
    d = variants["D"].aggregate.equal_weight_net_return
    e = variants["E"].aggregate.equal_weight_net_return

    if a > 0 and b <= 0:
        return SAME_BAR_EXECUTION_DEPENDENT
    if b > 0 and c <= 0:
        return NATIVE_SPREAD_INTOLERANT
    if b > 0 and c > 0 and (d <= 0 or e <= 0):
        return RISK_SIZING_TRANSFORMATION
    if e > 0:
        return PROCEED_TO_FROZEN_PROTOCOL
    raise DecompositionError(
        "decomposition result does not match any Section 5 classification rule"
    )


def run_decomposition() -> dict[str, VariantResult]:
    dataset = load_development_dataset()
    return {
        "A": variant_a_screening_reference(dataset),
        "B": variant_b_timing_only(dataset),
        "C": variant_c_spread_only(dataset),
        "D": variant_d_vol_sizing_midpoint(),
        "E": variant_e_current_promotion(),
    }
