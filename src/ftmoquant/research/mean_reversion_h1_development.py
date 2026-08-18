"""Nautilus execution-promotion wiring for the frozen, validated
short-horizon mean-reversion H1 signal
(:mod:`ftmoquant.strategies.mean_reversion_h1`).

Mirrors ``eurusd_tsm_development.py``'s wiring pattern (engine construction,
delta-to-target market-order execution, ``G1VolatilityNormalizer`` sizing,
``g1/artifacts.py`` provenance) but is deliberately much smaller: there is no
walk-forward search/selection here (``g1/search.py``, ``g1/selector.py``,
``g1/guards.py``, ``stage_g``'s frozen 2-pair Dukascopy universe machinery)
because the signal's parameters are already frozen -- nothing is optimized
or selected by this module. It reuses the execution harness's private
``_engine_config``/``_fee_model``/``_rollover_modules``/
``_instrument_for_profile``/``canonical_execution_profile`` unchanged.

``_add_venue`` itself is not reused unchanged: it hardcodes
``Venue("DUKASCOPY")`` internally and takes no venue parameter, so this
module defines ``_add_oanda_venue`` as a venue-parameterized copy of its
body (every other flag identical) rather than monkeypatching or modifying
the frozen harness. This is not a second backtester: all fills, spread
crossing, cost, and latency mechanics remain entirely owned by the real
Nautilus ``BacktestEngine``.

Supports exactly two partitions, both already-frozen and neither ever
selected/tuned by this module: DEVELOPMENT ``[2019-03-11, 2023-04-11)`` and
the already-observed VALIDATION ``[2023-04-11, 2024-08-21)``. Final holdout
(``>= stage_g.HOLDOUT_START``) is never reachable -- guarded at the path,
partition-boundary, and per-bar level.

Execution timing (amended -- see ``EXECUTION_TIMING_LABEL``): the H1 signal
is decided from the completed H1 bar's close exactly as frozen, but the
resulting order is never executed on that same H1 bar -- it is precomputed
offline (mirroring ``eurusd_tsm_development.py``'s own higher-timeframe-
decision -> lower-timeframe-strictly-later-execution precomputation
pattern) and executed on the first genuine M1 BID/ASK observation strictly
after the decision instant. H1 bars are read only to precompute decisions;
the engine is only ever fed the M1 stream.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any

from nautilus_trader.backtest import BacktestEngine
from nautilus_trader.execution import ProbabilisticFillModel, StaticLatencyModel
from nautilus_trader.model import (
    AccountType,
    Bar,
    BookType,
    Currency,
    CurrencyPair,
    InstrumentId,
    Money,
    OmsType,
    OrderFilled,
    OrderSide,
    OtoTriggerMode,
    StrategyId,
    Venue,
)
from nautilus_trader.persistence import ParquetDataCatalog
from nautilus_trader.trading import Strategy, StrategyConfig

from ftmoquant.backtest.execution_harness import (
    AccountParameters,
    ExecutionProfile,
    _engine_config,
    _fee_model,
    _git_commit,
    _instrument_for_profile,
    _money_text,
    _rollover_modules,
    _sha256_json,
    canonical_execution_profile,
)
from ftmoquant.data.instruments import oanda_symbol
from ftmoquant.data.oanda_alpha_lab_development import (
    load_oanda_alpha_lab_config,
)
from ftmoquant.data.oanda_alpha_lab_validation import (
    VALIDATION_READINESS_VERSION,
    load_oanda_alpha_lab_validation_config,
)
from ftmoquant.research.eurusd_tsm_development import EquityPoint
from ftmoquant.research.g1.artifacts import write_runtime_provenance
from ftmoquant.research.g1.normalization import (
    CausalEwmaDailyVolatility,
    CompletedDailyMidpoint,
    G1VolatilityNormalizer,
    completed_daily_log_returns,
)
from ftmoquant.research.stage_g import (
    DEVELOPMENT_END_EXCLUSIVE,
    DEVELOPMENT_START,
    HOLDOUT_START,
    VALIDATION_START,
)
from ftmoquant.research.ts_momentum_development import _annualized_sharpe
from ftmoquant.strategies.mean_reversion_h1 import (
    FROZEN_LOOKBACK,
    FROZEN_UNIVERSE,
    FROZEN_Z_ENTRY,
    CausalMeanReversionSignal,
    reject_final_holdout,
)

RUN_MODULE_NAME = "ftmoquant.research.mean_reversion_h1_development"
STRATEGY_SEMANTIC_ID = "mean_reversion_h1_v1"
VENUE = Venue("OANDA")
#: Reference notional the causal exposure decision scales against, matching
#: eurusd_tsm_development.py's BASE_RESEARCH_UNITS convention.
BASE_RESEARCH_UNITS = Decimal("100000")
#: Execution-timing correction (frozen before any real DEVELOPMENT/VALIDATION
#: return is observed): the H1 decision is computed offline from the
#: completed H1 close exactly as before, but the resulting order is executed
#: -- never decided -- on the first genuine M1 BID/ASK observation strictly
#: after that decision instant. Same canonical cost profile
#: (canonical_execution_profile(): zero latency/commission/slippage, genuine
#: BID/ASK spread crossing, rollover disabled) as every other detail of §5 of
#: the design doc; only the timing is new, hence this label is deliberately
#: distinct from "canonical_execution_profile" (the cost identity, unchanged)
#: recorded alongside it.
EXECUTION_TIMING_LABEL = "native_spread_crossing_strictly_later_execution"
#: Performance-measurement-only addition (frozen §8/§9 gates unevaluated by
#: this module -- it reports numbers, never a pass/fail verdict). The
#: screening-stage 1.5x cost stress (screening_common.STRESSED_COST_MULTIPLIER)
#: re-simulates vectorbt with a proportional fee scaled by 1.5x -- there is
#: no such fee here to re-simulate: cost is realized natively via genuine
#: BID/ASK spread crossing. The faithful, already-established (not invented
#: here) mapping for exactly this native-spread-crossing-vs-proportional-fee
#: mismatch is reused unchanged from eurusd_tsm_development.py's own
#: ``_fold_metrics`` (``stressed_total = total_net_return - 0.5 * cost_return``)
#: and ts_momentum_development.py's per-day ``cost_stress_1_5x_return =
#: net_return - realized_cost / 2``: each fill's realized half-spread cost
#: (``quantity * |fill_price - execution_midpoint|``, plus any commission)
#: is tracked directly, and the 1.5x-stressed return subtracts half of that
#: same realized cost again -- the same linear decomposition already
#: accepted twice in this repo for this exact situation.
COST_STRESS_METHODOLOGY_LABEL = (
    "native_spread_crossing_realized_cost_half_stress"
)
#: The venue account's base/reporting currency (AccountParameters.currency
#: in run_frozen_signal_backtest / _add_oanda_venue) -- always USD,
#: regardless of any individual pair's own quote currency (e.g. JPY for
#: USD/JPY, CHF for USD/CHF). Every performance-measurement amount below
#: must be expressed in THIS currency before being divided by a USD
#: initial_capital or summed across pairs.
_ACCOUNT_CURRENCY = "USD"
_H1_BAR_TIMEFRAME = "1-HOUR"
_M1_BAR_TIMEFRAME = "1-MINUTE"
#: The OANDA alpha-lab catalog stores derived H1 bars tagged INTERNAL (they
#: were produced by an offline aggregation pipeline for pandas/vectorbt
#: consumption -- see ftmoquant.data.derived_bars). H1 bars are never
#: injected into the engine any more (see EXECUTION_TIMING_LABEL above) --
#: they are only read to precompute decisions offline -- so this tag never
#: needs correcting. Native M1 bars are already stored EXTERNAL
#: (InstrumentSpec.bar_type's own default), which is exactly the tag the
#: engine requires for injection, so no M1 retagging is needed either.
_CATALOG_BAR_AGGREGATION = "INTERNAL"
_ENGINE_BAR_AGGREGATION = "EXTERNAL"


class MeanReversionH1DevelopmentError(ValueError):
    """Raised when the mean-reversion H1 execution-promotion run cannot
    proceed: wrong partition, forbidden path, non-frozen universe, or a
    readiness/provenance mismatch."""


class Partition(StrEnum):
    DEVELOPMENT = "development"
    VALIDATION = "validation"


_PARTITION_BOUNDS: dict[Partition, tuple[datetime, datetime]] = {
    Partition.DEVELOPMENT: (DEVELOPMENT_START, DEVELOPMENT_END_EXCLUSIVE),
    Partition.VALIDATION: (VALIDATION_START, HOLDOUT_START),
}
assert all(end <= HOLDOUT_START for _, end in _PARTITION_BOUNDS.values())


def parse_partition(value: str) -> Partition:
    """Fail closed on anything other than the two frozen partitions."""

    try:
        return Partition(value)
    except ValueError as error:
        raise MeanReversionH1DevelopmentError(
            f"unsupported partition: {value!r}; must be 'development' or 'validation'"
        ) from error


def partition_bounds(partition: Partition) -> tuple[datetime, datetime]:
    start, end_exclusive = _PARTITION_BOUNDS[partition]
    return start, end_exclusive


def _ns(value: datetime) -> int:
    return int(value.timestamp() * 1_000_000_000)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _reject_sealed_path(root: Path, *, partition: Partition) -> None:
    """Holdout is always forbidden. ``validation``-labeled paths are only
    forbidden for a DEVELOPMENT run -- a VALIDATION run legitimately opens
    a validation-labeled catalog root."""

    lowered = tuple(part.casefold() for part in root.parts)
    always_forbidden = ("holdout", "final_holdout", "final-holdout", "final_test")
    if any(marker in part for marker in always_forbidden for part in lowered):
        raise MeanReversionH1DevelopmentError(f"holdout path is forbidden: {root}")
    if partition is Partition.DEVELOPMENT and any(
        "validation" in part for part in lowered
    ):
        raise MeanReversionH1DevelopmentError(
            f"DEVELOPMENT run cannot access a validation-labeled path: {root}"
        )


def _resolve_catalog_roots(
    *, partition: Partition, catalog_root: Path, universe_readiness_path: Path
) -> dict[str, Path]:
    """Discover the seven frozen pairs' canonical catalog roots from the
    partition-appropriate readiness artifact -- never invents an eighth
    pair and never silently drops one."""

    try:
        document = json.loads(universe_readiness_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MeanReversionH1DevelopmentError(
            f"could not read readiness document: {error}"
        ) from error
    if not isinstance(document, dict) or document.get("holdout_accessed") is not False:
        raise MeanReversionH1DevelopmentError("readiness document is invalid")

    dataset_symbol_by_instrument: dict[str, str]
    if partition is Partition.DEVELOPMENT:
        if document.get("readiness_version") != "oanda-alpha-lab-readiness-1":
            raise MeanReversionH1DevelopmentError(
                "DEVELOPMENT readiness document has the wrong readiness_version"
            )
        development_config = load_oanda_alpha_lab_config()
        dataset_symbol_by_instrument = {
            instrument_id: development_config.instrument(instrument_id).dataset_symbol
            for instrument_id in FROZEN_UNIVERSE
        }
    else:
        if (
            document.get("readiness_version") != VALIDATION_READINESS_VERSION
            or document.get("partition") != "VALIDATION"
        ):
            raise MeanReversionH1DevelopmentError(
                "VALIDATION readiness document has the wrong "
                "readiness_version/partition"
            )
        validation_config = load_oanda_alpha_lab_validation_config()
        dataset_symbol_by_instrument = {
            instrument_id: validation_config.instrument(instrument_id).dataset_symbol
            for instrument_id in FROZEN_UNIVERSE
        }

    statuses = document.get("per_instrument_status")
    if not isinstance(statuses, dict):
        raise MeanReversionH1DevelopmentError("readiness document is malformed")
    ready = {
        instrument_id
        for instrument_id, status in statuses.items()
        if status == "research_ready"
    }
    if set(FROZEN_UNIVERSE) - ready:
        raise MeanReversionH1DevelopmentError(
            "not all seven frozen pairs are research_ready; no pair may be excluded"
        )

    roots: dict[str, Path] = {}
    for instrument_id in FROZEN_UNIVERSE:
        dataset_symbol = dataset_symbol_by_instrument[instrument_id]
        root = catalog_root / oanda_symbol(dataset_symbol)
        _reject_sealed_path(root, partition=partition)
        roots[instrument_id] = root
    return roots


# ---------------------------------------------------------------------------
# Venue wiring: canonical_execution_profile() and every other harness helper
# reused unchanged; only _add_venue's hardcoded Venue("DUKASCOPY") cannot be
# reused as-is, so its body is copied here with VENUE substituted.
# ---------------------------------------------------------------------------


def _add_oanda_venue(
    engine: BacktestEngine,
    account: AccountParameters,
    profile: ExecutionProfile,
    fill_model: ProbabilisticFillModel,
    latency_model: StaticLatencyModel,
    fee_model: Any,
    modules: Sequence[Any],
) -> None:
    engine.add_venue(
        venue=VENUE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[
            Money.from_str(_money_text(account.initial_capital, account.currency))
        ],
        base_currency=Currency.from_str(account.currency),
        default_leverage=account.leverage,
        leverages=None,
        margin_model=None,
        fill_model=fill_model,
        fee_model=fee_model,
        latency_model=latency_model,
        modules=modules,
        book_type=BookType.L1_MBP,
        routing=False,
        reject_stop_orders=True,
        support_gtd_orders=True,
        support_contingent_orders=True,
        use_position_ids=True,
        use_random_ids=False,
        use_reduce_only=True,
        use_message_queue=True,
        use_market_order_acks=False,
        bar_execution=True,
        bar_adaptive_high_low_ordering=profile.adaptive_high_low_ordering,
        trade_execution=False,
        liquidity_consumption=False,
        queue_position=False,
        allow_cash_borrowing=False,
        frozen_account=False,
        oto_trigger_mode=OtoTriggerMode.PARTIAL,
        price_protection_points=0,
        settlement_prices=None,
        liquidation_enabled=False,
        liquidation_trigger_ratio=1.0,
        liquidation_cancel_open_orders=True,
    )


def _validate_bar_stream(
    bars: tuple[Bar, ...], bar_type: str, start_ns: int, end_ns: int
) -> None:
    if not bars:
        raise MeanReversionH1DevelopmentError(f"no bars returned for {bar_type}")
    previous_ts: int | None = None
    for bar in bars:
        if str(bar.bar_type) != bar_type:
            raise MeanReversionH1DevelopmentError(f"bar type mismatch: {bar_type}")
        if not start_ns <= bar.ts_event < end_ns:
            raise MeanReversionH1DevelopmentError(
                f"{bar_type} bar outside the requested partition range"
            )
        if previous_ts is not None and bar.ts_event <= previous_ts:
            raise MeanReversionH1DevelopmentError(
                f"{bar_type} bars are not strictly increasing"
            )
        previous_ts = bar.ts_event


def _daily_midpoints_from_h1_bars(
    bid_bars: tuple[Bar, ...], ask_bars: tuple[Bar, ...]
) -> tuple[CompletedDailyMidpoint, ...]:
    """The last paired BID/ASK H1 bar of each UTC calendar day becomes that
    day's completed midpoint -- a batch preparation step, exactly mirroring
    eurusd_tsm_development.py's own daily-midpoint construction. Causality
    for any individual sizing decision is still enforced downstream by
    CausalEwmaDailyVolatility.estimate_before(), which only ever consults
    endpoints strictly before the decision time."""

    by_ts: dict[int, dict[str, float]] = {}
    for bar in bid_bars:
        by_ts.setdefault(bar.ts_event, {})["bid"] = float(bar.close)
    for bar in ask_bars:
        by_ts.setdefault(bar.ts_event, {})["ask"] = float(bar.close)

    latest_by_day: dict[date, tuple[int, float]] = {}
    for ts_event in sorted(by_ts):
        sides = by_ts[ts_event]
        if "bid" not in sides or "ask" not in sides:
            continue
        midpoint = (sides["bid"] + sides["ask"]) / 2.0
        day = datetime.fromtimestamp(ts_event / 1_000_000_000, tz=UTC).date()
        latest_by_day[day] = (ts_event, midpoint)

    return tuple(
        CompletedDailyMidpoint(information_time_ns=ts, midpoint=midpoint)
        for day, (ts, midpoint) in sorted(latest_by_day.items())
    )


# ---------------------------------------------------------------------------
# Strategy: a mechanical delta-to-target executor, mirroring
# eurusd_tsm_development.py's _ScaledTargetExecutor pattern. The causal
# signal math and state machine live entirely in
# ftmoquant.strategies.mean_reversion_h1 -- this class only turns that
# already-decided state into Nautilus order submissions.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OrderSubmissionRecord:
    """One decision-to-order-submission event, for parity assertions.

    ``decision_ns`` is the completed H1 bar close that produced the target
    (Section 1, unchanged); ``execution_ns`` is the strictly-later M1
    timestamp the order was actually submitted/filled on (Section 2, the
    subject of this amendment). ``execution_ns > decision_ns`` always --
    never equal, unlike the superseded same-H1-bar behavior this replaces.
    """

    instrument_id: str
    decision_ns: int
    execution_ns: int
    target_units: str
    side: str


@dataclass(frozen=True, slots=True)
class FillRecord:
    """One native venue fill event, for parity/timing assertions.
    ``last_px`` is the native venue fill price -- used to prove genuine
    spread crossing (buy at ASK, sell at BID), not a fee approximation."""

    instrument_id: str
    fill_ns: int
    side: str
    last_px: str


@dataclass(frozen=True, slots=True)
class _ScaledInstruction:
    """One precomputed decision->execution instruction: the frozen H1 signal
    decided a new target at ``decision_ns``; it may only be submitted at
    ``execution_ns``, the first genuine paired M1 BID/ASK observation
    strictly after ``decision_ns`` (never the H1 decision bar itself).
    ``execution_midpoint`` is that same M1 observation's mid price -- used
    only for performance measurement (realized spread-cost decomposition,
    see COST_STRESS_METHODOLOGY_LABEL), never for order sizing or timing."""

    decision_ns: int
    execution_ns: int
    target_units: Decimal
    execution_midpoint: Decimal


def _paired_timestamps(
    bid_bars: tuple[Bar, ...], ask_bars: tuple[Bar, ...]
) -> tuple[int, ...]:
    """The genuine executable observations: timestamps where *both* a BID
    and an ASK bar exist. No interpolation, no synthesis -- a side missing
    at a given timestamp (a real M1 gap) simply is not a candidate."""

    bid_ts = {bar.ts_event for bar in bid_bars}
    ask_ts = {bar.ts_event for bar in ask_bars}
    return tuple(sorted(bid_ts & ask_ts))


def _paired_midpoints(
    bid_bars: tuple[Bar, ...], ask_bars: tuple[Bar, ...]
) -> dict[int, Decimal]:
    """Mid price at every genuine paired M1 observation -- performance
    measurement only (see ``_ScaledInstruction.execution_midpoint``)."""

    bid_by_ts = {bar.ts_event: bar.close for bar in bid_bars}
    ask_by_ts = {bar.ts_event: bar.close for bar in ask_bars}
    return {
        ts: (bid_by_ts[ts].as_decimal() + ask_by_ts[ts].as_decimal()) / 2
        for ts in bid_by_ts.keys() & ask_by_ts.keys()
    }


def _first_strictly_later_paired_ns(
    paired_ns: Sequence[int], decision_ns: int
) -> int | None:
    """Production ``bisect``-based specialization of the same "first
    strictly later" semantics as
    ``carver_trend_carry_ftmo5_development.first_strictly_later_execution``
    (proven equivalent for representative sequences in
    ``tests/research/test_mean_reversion_h1_development.py``) -- operating
    on integer nanosecond timestamps directly instead of constructing a
    ``datetime`` per candidate, since ``paired_ns`` here can be a full
    partition's worth of M1 observations. Returns ``None`` (Section 2: fail
    closed) when no later executable observation exists, e.g. at the
    partition boundary."""

    index = bisect.bisect_right(paired_ns, decision_ns)
    if index >= len(paired_ns):
        return None
    return paired_ns[index]


def _precompute_h1_decisions(
    *,
    h1_bid_bars: tuple[Bar, ...],
    h1_ask_bars: tuple[Bar, ...],
    m1_bid_bars: tuple[Bar, ...],
    m1_ask_bars: tuple[Bar, ...],
    start_ns: int,
    end_exclusive_ns: int,
    estimator: CausalEwmaDailyVolatility,
) -> tuple[_ScaledInstruction, ...]:
    """Offline precomputation, mirroring ``eurusd_tsm_development.py``'s own
    higher-timeframe-decision -> lower-timeframe-execution precomputation
    pattern: replay the frozen, unchanged causal H1 signal
    (``CausalMeanReversionSignal``) and sizing (``G1VolatilityNormalizer``)
    bar by bar to get the exact target-unit sequence Section 1 requires,
    then resolve each transition's execution timestamp as the first genuine
    paired M1 BID/ASK observation strictly after that H1 decision instant.
    A transition with no such observation before the partition boundary is
    dropped -- it is never executed in this run (Section 2: fail closed)."""

    by_ts: dict[int, dict[str, Bar]] = {}
    for bar in h1_bid_bars:
        by_ts.setdefault(bar.ts_event, {})["BID"] = bar
    for bar in h1_ask_bars:
        by_ts.setdefault(bar.ts_event, {})["ASK"] = bar

    paired_m1_ns = _paired_timestamps(m1_bid_bars, m1_ask_bars)
    paired_m1_midpoints = _paired_midpoints(m1_bid_bars, m1_ask_bars)

    signal = CausalMeanReversionSignal()
    normalizer = G1VolatilityNormalizer()
    target_units = Decimal(0)
    instructions: list[_ScaledInstruction] = []
    for decision_ns in sorted(by_ts):
        if decision_ns < start_ns or decision_ns >= end_exclusive_ns:
            continue
        sides = by_ts[decision_ns]
        if "BID" not in sides or "ASK" not in sides:
            continue
        reject_final_holdout(decision_ns)

        mid = (float(sides["BID"].close) + float(sides["ASK"].close)) / 2.0
        signal_bar = signal.on_bar(decision_ns, mid)
        decision = normalizer.target_at(
            float(signal_bar.position), decision_ns, estimator
        )
        new_target = decision.target_base_units(BASE_RESEARCH_UNITS).quantize(
            Decimal("0.00000001")
        )
        if new_target == target_units:
            continue
        target_units = new_target

        execution_ns = _first_strictly_later_paired_ns(paired_m1_ns, decision_ns)
        if execution_ns is None or execution_ns >= end_exclusive_ns:
            continue
        if instructions and instructions[-1].execution_ns == execution_ns:
            raise MeanReversionH1DevelopmentError(
                "multiple H1 decisions mapped onto the same M1 execution "
                "observation; a real M1 gap spans more than one H1 decision"
            )
        instructions.append(
            _ScaledInstruction(
                decision_ns=decision_ns,
                execution_ns=execution_ns,
                target_units=target_units,
                execution_midpoint=paired_m1_midpoints[execution_ns],
            )
        )
    return tuple(instructions)


def _convert_to_account_currency(
    amount: Decimal,
    amount_currency: str,
    *,
    base_currency: str,
    quote_currency: str,
    conversion_price: Decimal,
) -> Decimal:
    """Convert ``amount`` (denominated in ``amount_currency``, which must be
    either the pair's own ``base_currency`` or ``quote_currency``) into
    ``_ACCOUNT_CURRENCY``, using ``conversion_price`` -- the *same
    instrument's own* quote/base price (``quote_currency`` per unit of
    ``base_currency``) at the causally relevant instant -- as the FX rate.

    General and dimensionally exact for every frozen pair because the
    universe is, by construction, entirely USD/XXX or XXX/USD (every pair
    has USD as either its base or its quote leg -- see FROZEN_UNIVERSE):
    an amount already in the account currency needs no conversion; an
    amount in the pair's quote currency converts to its base currency by
    dividing by the quote/base price; an amount in the pair's base
    currency converts to its quote currency by multiplying by that same
    price. No third-currency cross rate is ever required. Fails closed
    (raises) rather than guessing for any other combination -- e.g. an
    amount whose currency is neither leg of the pair -- so a currency
    normalization gap can never be silently swallowed."""

    if amount_currency == _ACCOUNT_CURRENCY:
        return amount
    if amount_currency == quote_currency and base_currency == _ACCOUNT_CURRENCY:
        return amount / conversion_price
    if amount_currency == base_currency and quote_currency == _ACCOUNT_CURRENCY:
        return amount * conversion_price
    raise MeanReversionH1DevelopmentError(
        f"cannot convert a {amount_currency} amount to {_ACCOUNT_CURRENCY} "
        f"using only this pair's own base={base_currency}/quote={quote_currency} "
        "price -- no direct rate is available"
    )


class _MeanReversionH1Executor(Strategy):
    def __new__(
        cls,
        instruments: Mapping[str, CurrencyPair],
        instructions: Mapping[str, tuple[_ScaledInstruction, ...]],
        start_ns: int,
        end_exclusive_ns: int,
        initial_capital: Decimal,
    ) -> _MeanReversionH1Executor:
        del instruments, instructions, start_ns, end_exclusive_ns, initial_capital
        return super().__new__(cls)

    def __init__(
        self,
        instruments: Mapping[str, CurrencyPair],
        instructions: Mapping[str, tuple[_ScaledInstruction, ...]],
        start_ns: int,
        end_exclusive_ns: int,
        initial_capital: Decimal,
    ) -> None:
        super().__init__(
            StrategyConfig(
                strategy_id=StrategyId("MEANREV-H1-V1-DEVELOPMENT-001"),
                log_events=False,
                log_commands=False,
            )
        )
        self._instruments = dict(instruments)
        # The causal signal/sizing decision is already fully resolved
        # offline (see _precompute_h1_decisions) -- this strategy is a
        # mechanical player of precomputed (execution_ns, target_units)
        # instructions against the live M1 BID/ASK stream, exactly mirroring
        # eurusd_tsm_development.py's _ScaledTargetExecutor pattern.
        self._instructions = {
            instrument_id: list(sequence)
            for instrument_id, sequence in instructions.items()
        }
        self._next_index: dict[str, int] = {
            instrument_id: 0 for instrument_id in instruments
        }
        self._start_ns = start_ns
        self._end_exclusive_ns = end_exclusive_ns
        self._pending: dict[str, dict[str, Bar]] = {
            instrument_id: {} for instrument_id in instruments
        }
        self._target_units: dict[str, Decimal] = {
            instrument_id: Decimal(0) for instrument_id in instruments
        }
        self.transition_counts: dict[str, int] = {
            instrument_id: 0 for instrument_id in instruments
        }
        self.submissions: list[OrderSubmissionRecord] = []
        self.fills: list[FillRecord] = []

        # Performance-measurement-only state (does not affect signal,
        # sizing, timing, or order submission above): per-pair realized
        # spread cost (for the §4/COST_STRESS_METHODOLOGY_LABEL 1.5x-stress
        # mapping) and a daily equity mark series (for net return / Sharpe),
        # mirroring eurusd_tsm_development.py's _ScaledTargetExecutor
        # exactly (_order_context / realized_variable_cost / equity_points).
        self._order_context: dict[str, _ScaledInstruction] = {}
        self.realized_variable_cost: dict[str, Decimal] = {
            instrument_id: Decimal(0) for instrument_id in instruments
        }
        self.equity_points: dict[str, list[EquityPoint]] = {
            instrument_id: [EquityPoint(start_ns, initial_capital)]
            for instrument_id in instruments
        }
        self._current_day: dict[str, date] = {}
        self._last_bar_ns: dict[str, int] = {}

    def on_start(self) -> None:
        for instrument_id in self._instruments:
            for side in ("BID", "ASK"):
                self.subscribe_bars(
                    _bar_type(
                        f"{instrument_id}-{_M1_BAR_TIMEFRAME}-{side}-"
                        f"{_ENGINE_BAR_AGGREGATION}"
                    )
                )

    def on_bar(self, bar: Bar) -> None:
        if bar.ts_event < self._start_ns or bar.ts_event >= self._end_exclusive_ns:
            return
        reject_final_holdout(bar.ts_event)
        instrument_id = str(bar.bar_type.instrument_id)
        side = bar.bar_type.spec.price_type.name
        frame = self._pending[instrument_id]
        frame[side] = bar
        if "BID" not in frame or "ASK" not in frame:
            return
        bid_bar = frame["BID"]
        ask_bar = frame["ASK"]
        if bid_bar.ts_event != ask_bar.ts_event:
            return
        self._pending[instrument_id] = {}
        ts_event = bid_bar.ts_event

        # One equity mark per UTC calendar day: on the first paired M1
        # observation of a NEW day, flush a mark for the PRIOR day using
        # the last state observed on that prior day (performance
        # measurement only -- does not affect execution).
        day = datetime.fromtimestamp(ts_event / 1_000_000_000, tz=UTC).date()
        prior_day = self._current_day.get(instrument_id)
        if prior_day is not None and day != prior_day:
            self._append_equity_mark(instrument_id, self._last_bar_ns[instrument_id])
        self._current_day[instrument_id] = day

        sequence = self._instructions[instrument_id]
        index = self._next_index[instrument_id]
        while index < len(sequence) and sequence[index].execution_ns == ts_event:
            instruction = sequence[index]
            self._submit_target(instrument_id, instruction)
            index += 1
        self._next_index[instrument_id] = index

        self._last_bar_ns[instrument_id] = ts_event

    def on_stop(self) -> None:
        for instrument_id, ts_event in self._last_bar_ns.items():
            self._append_equity_mark(instrument_id, ts_event)

    def _submit_target(
        self, instrument_id: str, instruction: _ScaledInstruction
    ) -> None:
        target_units = instruction.target_units
        delta = target_units - self._target_units[instrument_id]
        if delta == 0:
            return
        instrument = self._instruments[instrument_id]
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        quantity = instrument.make_qty(float(abs(delta)))
        order = self.order_factory.market(
            instrument_id=InstrumentId.from_str(instrument_id),
            order_side=side,
            quantity=quantity,
            tags=(
                STRATEGY_SEMANTIC_ID,
                f"decision_ns={instruction.decision_ns}",
                f"execution_ns={instruction.execution_ns}",
                f"target_units={target_units}",
            ),
        )
        self._order_context[order.client_order_id.value] = instruction
        self._target_units[instrument_id] = target_units
        self.transition_counts[instrument_id] += 1
        self.submissions.append(
            OrderSubmissionRecord(
                instrument_id=instrument_id,
                decision_ns=instruction.decision_ns,
                execution_ns=instruction.execution_ns,
                target_units=str(target_units),
                side=side.name,
            )
        )
        self.submit_order(order)

    def on_order_filled(self, event: OrderFilled) -> None:
        self.fills.append(
            FillRecord(
                instrument_id=str(event.instrument_id),
                fill_ns=event.ts_event,
                side=event.order_side.name,
                last_px=str(event.last_px),
            )
        )

        # Performance-measurement-only: the realized half-spread cost of
        # this fill, relative to the M1 execution midpoint -- see
        # COST_STRESS_METHODOLOGY_LABEL. Never affects the order/target
        # already submitted above.
        instruction = self._order_context.get(event.client_order_id.value)
        if instruction is None:
            raise MeanReversionH1DevelopmentError(
                "native fill lacks a precomputed instruction context"
            )
        instrument_id = str(event.instrument_id)
        instrument = self._instruments[instrument_id]
        base_currency = instrument.base_currency.code
        quote_currency = instrument.quote_currency.code

        quantity = event.last_qty.as_decimal()
        fill_price = event.last_px.as_decimal()
        # quantity is denominated in the pair's BASE currency and
        # fill_price/execution_midpoint are QUOTE-per-BASE, so this
        # difference is always in the pair's QUOTE currency (e.g. JPY for
        # USD/JPY) -- NOT necessarily the USD account currency. Only
        # quote_currency=="USD" pairs (EUR/USD, GBP/USD, AUD/USD, NZD/USD)
        # happen to need no further conversion; base_currency=="USD" pairs
        # (USD/JPY, USD/CAD, USD/CHF) do not, and must be converted below.
        spread_cost_quote = (
            quantity * (fill_price - instruction.execution_midpoint)
            if event.order_side is OrderSide.BUY
            else quantity * (instruction.execution_midpoint - fill_price)
        )
        spread_cost = _convert_to_account_currency(
            spread_cost_quote,
            quote_currency,
            base_currency=base_currency,
            quote_currency=quote_currency,
            conversion_price=instruction.execution_midpoint,
        )

        commission = Decimal(0)
        if event.commission is not None:
            commission_amount = abs(event.commission.as_decimal())
            if commission_amount != 0:
                commission = _convert_to_account_currency(
                    commission_amount,
                    event.commission.currency.code,
                    base_currency=base_currency,
                    quote_currency=quote_currency,
                    conversion_price=instruction.execution_midpoint,
                )

        self.realized_variable_cost[instrument_id] += spread_cost + commission

    def _append_equity_mark(self, instrument_id: str, ts_event: int) -> None:
        # Reuse Nautilus's OWN currency-converted total account equity
        # (balance + unrealized P&L of every open position, natively
        # converted to the venue account's base/reporting currency) rather
        # than reconstructing it: nautilus_trader.portfolio.Portfolio
        # already tracks this per venue. This was audited against a manual
        # reconstruction using position.unrealized_pnl(), which returns
        # Money in the POSITION's own settlement/quote currency (e.g. JPY
        # for USD/JPY) rather than the account currency -- summing that
        # directly into a USD balance without conversion was the root
        # cause of a currency-unit bug for every base=USD pair (USD/JPY,
        # USD/CAD, USD/CHF). Portfolio.equity() has no such defect: it is
        # Nautilus's own accounting, not a reimplementation.
        # Portfolio.equity(venue) is venue-wide, not instrument-scoped --
        # this is exact for the production design (run_frozen_signal_backtest
        # is always called with exactly one instrument per account/venue;
        # see run_mean_reversion_h1_development's independent-equal-capital-
        # sleeve comment). It would double-count across instruments if this
        # executor were ever driven with more than one instrument sharing
        # one account, which nothing in this module does.
        instrument = self._instruments[instrument_id]
        equity_by_currency = self.portfolio.equity(venue=instrument.id.venue)
        equity_money = equity_by_currency.get(Currency.from_str(_ACCOUNT_CURRENCY))
        if equity_money is None:
            raise MeanReversionH1DevelopmentError(
                f"native portfolio equity unavailable in {_ACCOUNT_CURRENCY}"
            )
        equity = equity_money.as_decimal()
        points = self.equity_points[instrument_id]
        if ts_event > points[-1].information_time_ns:
            points.append(EquityPoint(ts_event, equity))


def _bar_type(value: str) -> Any:
    from nautilus_trader.model import BarType

    return BarType.from_str(value)


# ---------------------------------------------------------------------------
# Performance measurement (frozen §8/§9 gates NOT evaluated here -- this
# section only computes the numbers a human later reads against those
# gates; it never emits a pass/fail verdict, and never touches signal,
# sizing, timing, or order-submission logic above). Each pair's own
# independent equal-capital sleeve stays independent throughout: no
# function below pools capital or cash across pairs.
# ---------------------------------------------------------------------------


def _pair_daily_returns(
    equity_points: tuple[EquityPoint, ...], initial_capital: Decimal
) -> tuple[tuple[date, float], ...]:
    """One causal daily return per UTC-day equity mark transition (see
    ``_MeanReversionH1Executor``'s day-boundary marking), converted to a
    fraction of the sleeve's own starting capital -- mirrors
    eurusd_tsm_development.py's own equity-point-to-return conversion
    (``_fold_metrics``)."""

    if initial_capital <= 0:
        raise MeanReversionH1DevelopmentError("initial capital must be positive")
    return tuple(
        (
            datetime.fromtimestamp(
                current.information_time_ns / 1_000_000_000, tz=UTC
            ).date(),
            float((current.equity - previous.equity) / initial_capital),
        )
        for previous, current in zip(equity_points, equity_points[1:])
    )


@dataclass(frozen=True, slots=True)
class PairPerformance:
    """One independent sleeve's performance summary. Measurement only --
    no promotion verdict is computed or implied by this dataclass."""

    instrument_id: str
    initial_capital: str
    net_return: float
    realized_variable_cost: str
    cost_stress_1_5x_return: float
    annualized_sharpe: float | None
    daily_return_count: int


def _pair_performance(
    instrument_id: str,
    equity_points: tuple[EquityPoint, ...],
    daily_returns: tuple[tuple[date, float], ...],
    realized_variable_cost: Decimal,
    initial_capital: Decimal,
) -> PairPerformance:
    if not equity_points:
        raise MeanReversionH1DevelopmentError(
            f"{instrument_id} has no equity points; at least the seed mark is required"
        )
    net_return = float(
        (equity_points[-1].equity - equity_points[0].equity) / initial_capital
    )
    cost_return = float(realized_variable_cost / initial_capital)
    stressed_return = net_return - 0.5 * cost_return
    sharpe = _annualized_sharpe([value for _, value in daily_returns])
    return PairPerformance(
        instrument_id=instrument_id,
        initial_capital=str(initial_capital),
        net_return=net_return,
        realized_variable_cost=str(realized_variable_cost),
        cost_stress_1_5x_return=stressed_return,
        annualized_sharpe=sharpe,
        daily_return_count=len(daily_returns),
    )


@dataclass(frozen=True, slots=True)
class AggregatePerformance:
    """Equal-weight aggregate across the seven independent sleeves --
    the simple average of their own returns, matching
    screening_common._build_aggregate_portfolio's cash_sharing=False
    convention (see run_mean_reversion_h1_development's own capital-
    semantics comment). Measurement only -- no promotion verdict."""

    pair_count: int
    equal_weight_net_return: float
    equal_weight_cost_stress_1_5x_return: float
    profitable_pair_count: int
    annualized_sharpe: float | None
    daily_return_count: int
    cost_stress_methodology: str


def _aggregate_daily_returns(
    daily_returns_by_pair: Mapping[str, tuple[tuple[date, float], ...]]
) -> tuple[float, ...]:
    """Causal equal-weight aggregation of the independent sleeves' own daily
    return series: for every UTC day observed by *any* pair, average that
    day's return across all seven pairs (0.0 for a pair with no mark that
    day -- no observed price movement for that sleeve that day). This is
    the same arithmetic identity behind
    screening_common._build_aggregate_portfolio's equal-init-cash summed-
    equity aggregate: summing equal-capital dollar P&L and dividing by the
    summed equal capital is identical to simply averaging the per-sleeve
    returns."""

    pair_count = len(daily_returns_by_pair)
    by_day: dict[date, dict[str, float]] = {}
    for instrument_id, rows in daily_returns_by_pair.items():
        for day, value in rows:
            by_day.setdefault(day, {})[instrument_id] = value
    pair_ids = tuple(daily_returns_by_pair)
    return tuple(
        sum(by_day[day].get(instrument_id, 0.0) for instrument_id in pair_ids)
        / pair_count
        for day in sorted(by_day)
    )


def _aggregate_performance(
    pair_performance: Mapping[str, PairPerformance],
    daily_returns_by_pair: Mapping[str, tuple[tuple[date, float], ...]],
) -> AggregatePerformance:
    if set(pair_performance) != set(daily_returns_by_pair):
        raise MeanReversionH1DevelopmentError(
            "pair performance and daily-return series must cover the same pairs"
        )
    pair_count = len(pair_performance)
    if pair_count == 0:
        raise MeanReversionH1DevelopmentError(
            "aggregate performance requires at least one pair"
        )
    equal_weight_net_return = (
        sum(item.net_return for item in pair_performance.values()) / pair_count
    )
    equal_weight_stressed_return = (
        sum(item.cost_stress_1_5x_return for item in pair_performance.values())
        / pair_count
    )
    profitable_pair_count = sum(
        1 for item in pair_performance.values() if item.net_return > 0
    )
    aggregate_daily = _aggregate_daily_returns(daily_returns_by_pair)
    return AggregatePerformance(
        pair_count=pair_count,
        equal_weight_net_return=equal_weight_net_return,
        equal_weight_cost_stress_1_5x_return=equal_weight_stressed_return,
        profitable_pair_count=profitable_pair_count,
        annualized_sharpe=_annualized_sharpe(aggregate_daily),
        daily_return_count=len(aggregate_daily),
        cost_stress_methodology=COST_STRESS_METHODOLOGY_LABEL,
    )


# ---------------------------------------------------------------------------
# Orchestration + provenance.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MeanReversionH1RunResult:
    partition: str
    start_utc: str
    end_exclusive_utc: str
    instrument_ids: tuple[str, ...]
    transition_counts: dict[str, int]
    order_report_rows: int
    fill_report_rows: int
    position_report_rows: int
    execution_profile: str
    execution_timing: str
    risk_normalization: str
    #: Performance measurement only (§8/§9 gates remain unevaluated by this
    #: module -- these are the raw numbers a human later reads against the
    #: frozen gate thresholds, never a pass/fail verdict).
    pair_performance: dict[str, PairPerformance]
    aggregate_performance: AggregatePerformance


@dataclass(frozen=True, slots=True)
class EngineRunOutcome:
    """Raw output of one real Nautilus engine run -- no partition/provenance
    concept, so this is directly usable by both the frozen production
    orchestrator and synthetic parity tests."""

    transition_counts: dict[str, int]
    submissions: tuple[OrderSubmissionRecord, ...]
    fills: tuple[FillRecord, ...]
    order_report_rows: int
    fill_report_rows: int
    position_report_rows: int
    #: Performance-measurement-only additions (do not affect execution):
    #: one daily-marked equity curve and one realized-spread-cost total per
    #: instrument, plus the shared sleeve's starting capital.
    equity_points: dict[str, tuple[EquityPoint, ...]]
    realized_variable_cost: dict[str, Decimal]
    initial_capital: Decimal


def run_frozen_signal_backtest(
    *,
    start: datetime,
    end_exclusive: datetime,
    instruments: Mapping[str, CurrencyPair],
    h1_bars_by_instrument: Mapping[str, Mapping[str, tuple[Bar, ...]]],
    m1_bars_by_instrument: Mapping[str, Mapping[str, tuple[Bar, ...]]],
) -> EngineRunOutcome:
    """Pure engine-wiring core: build and run one real Nautilus
    ``BacktestEngine`` over caller-supplied instruments/bars for an
    arbitrary ``[start, end_exclusive)`` window. Has no concept of
    "partition", never opens a catalog or readiness document, and never
    resolves dates itself -- this is what makes it directly callable from
    synthetic parity tests with a handful of fabricated bars.

    ``h1_bars_by_instrument`` is used only to precompute the frozen causal
    signal/sizing decisions offline (Section 1, unchanged) -- it is never
    injected into the engine. ``m1_bars_by_instrument`` is the sole stream
    the engine actually sees; every order is submitted and filled on a
    genuine M1 BID/ASK observation strictly after the H1 decision that
    produced it (Section 2), never on the H1 decision bar itself.

    This function must NEVER be exposed through the CLI or given real
    catalog data with caller-chosen dates in production: the only
    production entry point is :func:`run_mean_reversion_h1_development`,
    which always resolves ``start``/``end_exclusive`` via the frozen,
    non-overridable :func:`partition_bounds`. Extracting this core does not
    weaken that guard -- it is unreachable from the CLI and takes no
    partition argument at all.
    """

    if set(instruments) != set(h1_bars_by_instrument) or set(instruments) != set(
        m1_bars_by_instrument
    ):
        raise MeanReversionH1DevelopmentError(
            "instruments, h1_bars_by_instrument, and m1_bars_by_instrument "
            "must all cover the same pairs"
        )
    start_ns, end_ns = _ns(start), _ns(end_exclusive)
    profile = canonical_execution_profile()
    account = AccountParameters(
        initial_capital=Decimal("100000"), currency="USD", leverage=Decimal(1)
    )

    instructions_by_instrument: dict[str, tuple[_ScaledInstruction, ...]] = {}
    all_bars: list[Bar] = []
    for instrument_id in instruments:
        h1_by_side = h1_bars_by_instrument[instrument_id]
        m1_by_side = m1_bars_by_instrument[instrument_id]
        for side in ("BID", "ASK"):
            h1_bar_type = (
                f"{instrument_id}-{_H1_BAR_TIMEFRAME}-{side}-"
                f"{_CATALOG_BAR_AGGREGATION}"
            )
            _validate_bar_stream(h1_by_side[side], h1_bar_type, start_ns, end_ns)
            m1_bar_type = (
                f"{instrument_id}-{_M1_BAR_TIMEFRAME}-{side}-"
                f"{_ENGINE_BAR_AGGREGATION}"
            )
            _validate_bar_stream(m1_by_side[side], m1_bar_type, start_ns, end_ns)
            all_bars.extend(m1_by_side[side])
        estimator = CausalEwmaDailyVolatility(
            completed_daily_log_returns(
                _daily_midpoints_from_h1_bars(h1_by_side["BID"], h1_by_side["ASK"])
            )
        )
        instructions_by_instrument[instrument_id] = _precompute_h1_decisions(
            h1_bid_bars=h1_by_side["BID"],
            h1_ask_bars=h1_by_side["ASK"],
            m1_bid_bars=m1_by_side["BID"],
            m1_ask_bars=m1_by_side["ASK"],
            start_ns=start_ns,
            end_exclusive_ns=end_ns,
            estimator=estimator,
        )
    all_bars.sort(key=lambda bar: (bar.ts_event, str(bar.bar_type)))

    identity = _sha256_json(
        {
            "module": RUN_MODULE_NAME,
            "instruments": sorted(instruments),
            "lookback": FROZEN_LOOKBACK,
            "z_entry": FROZEN_Z_ENTRY,
            "start_utc": _iso(start),
            "end_exclusive_utc": _iso(end_exclusive),
            "execution_timing": EXECUTION_TIMING_LABEL,
        }
    )
    engine = BacktestEngine(_engine_config(identity))
    executor = _MeanReversionH1Executor(
        instruments=instruments,
        instructions=instructions_by_instrument,
        start_ns=start_ns,
        end_exclusive_ns=end_ns,
        initial_capital=account.initial_capital,
    )
    try:
        fill_model = ProbabilisticFillModel(
            prob_fill_on_limit=float(profile.fill_on_limit_probability),
            prob_slippage=float(profile.adverse_slippage_probability),
            random_seed=profile.random_seed,
        )
        latency_model = StaticLatencyModel(
            base_latency_nanos=profile.base_latency_ns,
            insert_latency_nanos=profile.insert_latency_ns,
            update_latency_nanos=profile.update_latency_ns,
            cancel_latency_nanos=profile.cancel_latency_ns,
        )
        _add_oanda_venue(
            engine,
            account,
            profile,
            fill_model,
            latency_model,
            _fee_model(profile.fee),
            _rollover_modules(profile.rollover),
        )
        for instrument in instruments.values():
            engine.add_instrument(instrument)
        engine.add_data(all_bars, validate=True, sort=True)
        engine.add_strategy(executor)
        engine.run(
            start=start_ns, end=end_ns - 1, run_config_id=f"MEANREV-H1-{identity[:20]}"
        )
        order_report = engine.generate_orders_report()
        fill_report = engine.generate_fills_report()
        position_report = engine.generate_positions_report()
    finally:
        engine.dispose()

    return EngineRunOutcome(
        transition_counts=dict(executor.transition_counts),
        submissions=tuple(executor.submissions),
        fills=tuple(executor.fills),
        order_report_rows=len(order_report),
        fill_report_rows=len(fill_report),
        position_report_rows=len(position_report),
        equity_points={
            instrument_id: tuple(points)
            for instrument_id, points in executor.equity_points.items()
        },
        realized_variable_cost=dict(executor.realized_variable_cost),
        initial_capital=account.initial_capital,
    )


def run_mean_reversion_h1_development(
    *,
    partition: Partition,
    catalog_roots: Mapping[str, Path],
    output_dir: Path,
) -> MeanReversionH1RunResult:
    """Run the frozen mean-reversion H1 signal through a real Nautilus
    ``BacktestEngine`` for exactly one partition, across exactly the seven
    frozen pairs. Requires the caller to have already resolved
    ``catalog_roots`` via :func:`_resolve_catalog_roots` (or an equivalent
    readiness-checked mapping) -- this function re-validates the universe
    but does not itself open any readiness document. Dates are always
    resolved via the frozen :func:`partition_bounds`; there is no override."""

    if set(catalog_roots) != set(FROZEN_UNIVERSE):
        raise MeanReversionH1DevelopmentError(
            "exactly the seven frozen pairs are required; no pair may be excluded"
        )
    for root in catalog_roots.values():
        _reject_sealed_path(root, partition=partition)
    start, end_exclusive = partition_bounds(partition)
    start_ns, end_ns = _ns(start), _ns(end_exclusive)
    if output_dir.exists():
        raise MeanReversionH1DevelopmentError(
            f"output directory already exists: {output_dir}"
        )

    profile = canonical_execution_profile()

    # Section 5 (capital semantics), frozen: each pair gets its own
    # independent equal-capital sleeve -- one dedicated BacktestEngine and
    # account per instrument, matching the already-validated VectorBT
    # screening's own convention exactly
    # (screening_common._build_aggregate_portfolio: cash_sharing=False,
    # every instrument its own equal init_cash, summed/averaged for the
    # aggregate -- never one shared margin account competing for capital
    # across all seven pairs). run_frozen_signal_backtest already builds a
    # fresh AccountParameters/engine on every call, so achieving this only
    # requires calling it once per pair rather than once with all seven.
    transition_counts: dict[str, int] = {}
    order_report_rows = 0
    fill_report_rows = 0
    position_report_rows = 0
    pair_performance: dict[str, PairPerformance] = {}
    daily_returns_by_pair: dict[str, tuple[tuple[date, float], ...]] = {}
    for instrument_id in FROZEN_UNIVERSE:
        root = catalog_roots[instrument_id]
        catalog = ParquetDataCatalog(str(root / "catalog"))
        found = catalog.instruments([instrument_id])
        if len(found) != 1 or not isinstance(found[0], CurrencyPair):
            raise MeanReversionH1DevelopmentError(
                f"frozen CurrencyPair is unavailable: {instrument_id}"
            )
        instrument = _instrument_for_profile(found[0], profile.fee)

        # H1 -- read-only, offline decision precomputation. Never injected
        # into the engine, so the catalog's native INTERNAL tag is used as-is.
        h1_by_side: dict[str, tuple[Bar, ...]] = {}
        for side in ("BID", "ASK"):
            h1_bar_type = (
                f"{instrument_id}-{_H1_BAR_TIMEFRAME}-{side}-"
                f"{_CATALOG_BAR_AGGREGATION}"
            )
            bars = tuple(
                catalog.query_bars([h1_bar_type], start=start_ns, end=end_ns - 1)
            )
            _validate_bar_stream(bars, h1_bar_type, start_ns, end_ns)
            h1_by_side[side] = bars

        # M1 -- the sole engine-injected execution stream (Section 2). The
        # OANDA alpha-lab catalog already stores native M1 bars tagged
        # EXTERNAL, so no retagging is needed here.
        m1_by_side: dict[str, tuple[Bar, ...]] = {}
        for side in ("BID", "ASK"):
            m1_bar_type = (
                f"{instrument_id}-{_M1_BAR_TIMEFRAME}-{side}-"
                f"{_ENGINE_BAR_AGGREGATION}"
            )
            bars = tuple(
                catalog.query_bars([m1_bar_type], start=start_ns, end=end_ns - 1)
            )
            _validate_bar_stream(bars, m1_bar_type, start_ns, end_ns)
            m1_by_side[side] = bars

        outcome = run_frozen_signal_backtest(
            start=start,
            end_exclusive=end_exclusive,
            instruments={instrument_id: instrument},
            h1_bars_by_instrument={instrument_id: h1_by_side},
            m1_bars_by_instrument={instrument_id: m1_by_side},
        )
        transition_counts[instrument_id] = outcome.transition_counts[instrument_id]
        order_report_rows += outcome.order_report_rows
        fill_report_rows += outcome.fill_report_rows
        position_report_rows += outcome.position_report_rows

        pair_equity_points = outcome.equity_points[instrument_id]
        pair_daily_returns = _pair_daily_returns(
            pair_equity_points, outcome.initial_capital
        )
        daily_returns_by_pair[instrument_id] = pair_daily_returns
        pair_performance[instrument_id] = _pair_performance(
            instrument_id,
            pair_equity_points,
            pair_daily_returns,
            outcome.realized_variable_cost[instrument_id],
            outcome.initial_capital,
        )

    aggregate_performance = _aggregate_performance(
        pair_performance, daily_returns_by_pair
    )

    result = MeanReversionH1RunResult(
        partition=partition.value,
        start_utc=_iso(start),
        end_exclusive_utc=_iso(end_exclusive),
        instrument_ids=FROZEN_UNIVERSE,
        transition_counts=transition_counts,
        order_report_rows=order_report_rows,
        fill_report_rows=fill_report_rows,
        position_report_rows=position_report_rows,
        execution_profile="canonical_execution_profile",
        execution_timing=EXECUTION_TIMING_LABEL,
        risk_normalization="G1VolatilityNormalizer_1pct_annualized_ewma60",
        pair_performance=pair_performance,
        aggregate_performance=aggregate_performance,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_result_artifacts(output_dir, result)
    return result


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _write_result_artifacts(output_dir: Path, result: MeanReversionH1RunResult) -> None:
    result_bytes = _canonical_bytes(asdict(result)) + b"\n"
    result_path = output_dir / "mean_reversion_h1_result.json"
    result_path.write_bytes(result_bytes)

    write_runtime_provenance(
        output_dir / "runtime_provenance.json",
        {
            "git_commit": _git_commit(),
            "python_module": RUN_MODULE_NAME,
            "strategy_semantic_id": STRATEGY_SEMANTIC_ID,
            "execution_profile": "canonical_execution_profile",
            "execution_timing": result.execution_timing,
            "risk_normalization": "G1VolatilityNormalizer_1pct_annualized_ewma60",
            "cost_stress_methodology": (
                result.aggregate_performance.cost_stress_methodology
            ),
            "promotion_gates_evaluated": False,
            "validation_accessed": result.partition == Partition.VALIDATION.value,
            "final_holdout_accessed": False,
        },
    )

    manifest = {result_path.name: hashlib.sha256(result_bytes).hexdigest()}
    (output_dir / "artifact_hashes.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the frozen mean_reversion_h1_v1 signal (H1, lookback "
            "40, z_entry 2.0, all seven OANDA alpha-lab pairs) through the "
            "real Nautilus execution harness for one already-frozen "
            "partition."
        )
    )
    parser.add_argument(
        "--partition", choices=("development", "validation"), required=True
    )
    parser.add_argument(
        "--catalog-root",
        type=Path,
        required=True,
        help=(
            "directory containing one <OANDA_SYMBOL> subdirectory per pair "
            "(e.g. EUR_USD, GBP_USD, ...), each already canonicalized and "
            "derived to H1"
        ),
    )
    parser.add_argument("--universe-readiness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    partition = parse_partition(args.partition)
    catalog_roots = _resolve_catalog_roots(
        partition=partition,
        catalog_root=args.catalog_root,
        universe_readiness_path=args.universe_readiness,
    )
    result = run_mean_reversion_h1_development(
        partition=partition, catalog_roots=catalog_roots, output_dir=args.output
    )
    print(json.dumps(asdict(result), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
