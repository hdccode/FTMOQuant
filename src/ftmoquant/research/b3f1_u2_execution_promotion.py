"""Nautilus execution-promotion wiring for the frozen, VALIDATED B3F1
underpowered candidate U2 (:mod:`ftmoquant.research.alpha_lab.
b3f1_underpowered_u2_validation`).

Reproduces U2's exact Alpha-Lab native-BID/ASK two-leg spread lifecycle in
the canonical Nautilus ``BacktestEngine`` and measures execution
degradation between the two -- a MEASUREMENT stage, not a new gate.

Reuses, rather than re-derives:

- :mod:`ftmoquant.research.alpha_lab.b3f1_spread_signals` (``compute_
  formation_series``, ``generate_b3f1_decisions``) -- completely unchanged;
  this is the ONLY place U2's causal decisions are computed, so signal
  parity with Alpha-Lab is guaranteed by construction, not proven after
  the fact.
- :mod:`ftmoquant.research.alpha_lab.b3f1_spread_execution` (``simulate_
  b3f1_intents``, ``compute_leg_weights``, ``usd_gross_to_quantity``,
  ``GROSS_NOTIONAL_USD``, the private ``_first_strictly_later_ns``) --
  this module's ALPHA-LAB reference numbers are produced by literally
  calling ``simulate_b3f1_intents`` unchanged; this module never
  reimplements the Alpha-Lab lifecycle.
- :mod:`ftmoquant.research.alpha_lab.b3f1_underpowered_u2_validation` for
  U2's frozen candidate identity (``Y_SPEC``/``X_SPEC``/``FORMATION_
  WINDOW``/``Z_ENTRY``/``Z_STOP``) and preregistration hash verification
  (``verify_preregistration``) -- imported directly so this module is
  structurally incapable of drifting from the exact validated candidate.
- :mod:`ftmoquant.research.mean_reversion_h1_development`'s execution-
  promotion pattern (``Partition``/``parse_partition``/``partition_
  bounds``/``_reject_sealed_path``/``_add_oanda_venue``/``_convert_to_
  account_currency``/``_ns``/``_iso``) -- imported and reused UNCHANGED
  rather than copied, since none of it is single-instrument-specific.
- :mod:`ftmoquant.backtest.execution_harness` (``canonical_execution_
  profile``, ``AccountParameters``, ``ExecutionProfile``, the private
  ``_engine_config``/``_fee_model``/``_rollover_modules``/``_instrument_
  for_profile``/``_money_text``/``_git_commit``/``_sha256_json``) --
  reused unchanged, exactly as ``mean_reversion_h1_development.py``
  already does. This is not a second backtester: every fill, spread
  crossing, cost, and latency mechanic remains entirely owned by the real
  Nautilus ``BacktestEngine``.
- :func:`ftmoquant.research.alpha_lab.b3f1_spread_screen.
  expectancy_and_profit_factor` and :func:`ftmoquant.research.alpha_lab.
  wick_fvg_squeeze_screen._annualized_sharpe` for BOTH sides' performance
  metrics, so the Alpha-Lab-vs-Nautilus comparison uses identical
  arithmetic on each side, never two different formulas.

Sizing (section 5): each Nautilus leg order's quantity is taken directly
from the ALREADY-COMPUTED Alpha-Lab reference episode's own leg quantity
(``RelativeValueLeg.quantity``, itself ``usd_gross_to_quantity`` applied to
the SAME beta-derived weight and the SAME precomputed entry price) -- never
recomputed, retargeted, or optimized for the Nautilus run. This isolates
execution degradation only: Nautilus receives the exact same order sizes
Alpha-Lab used and independently determines its own fill prices/timing
from its own native BID/ASK bar stream.

Supports exactly two partitions, both frozen and neither ever selected/
tuned by this module: DEVELOPMENT and the already-observed VALIDATION.
Final holdout is never reachable -- guarded at the path, partition-
boundary, and per-bar level, mirroring ``mean_reversion_h1_development.py``
exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from nautilus_trader.backtest import BacktestEngine
from nautilus_trader.execution import ProbabilisticFillModel, StaticLatencyModel
from nautilus_trader.model import (
    Bar,
    BarType,
    CurrencyPair,
    InstrumentId,
    OrderFilled,
    OrderSide,
    StrategyId,
)
from nautilus_trader.persistence import ParquetDataCatalog
from nautilus_trader.trading import Strategy, StrategyConfig

from ftmoquant.backtest.execution_harness import (
    AccountParameters,
    _engine_config,
    _fee_model,
    _git_commit,
    _instrument_for_profile,
    _rollover_modules,
    _sha256_json,
    canonical_execution_profile,
)
from ftmoquant.data.instruments import oanda_symbol
from ftmoquant.data.oanda_alpha_lab_development import (
    OandaAlphaLabConfig,
    load_oanda_alpha_lab_config,
)
from ftmoquant.data.oanda_alpha_lab_validation import (
    VALIDATION_READINESS_VERSION,
    OandaAlphaLabValidationConfig,
    load_oanda_alpha_lab_validation_config,
)
from ftmoquant.research.alpha_lab.b3f1_spread_execution import (
    GROSS_NOTIONAL_USD,
    simulate_b3f1_intents,
)
from ftmoquant.research.alpha_lab.b3f1_spread_screen import (
    _rows_from_episodes,
    expectancy_and_profit_factor,
)
from ftmoquant.research.alpha_lab.b3f1_spread_signals import (
    compute_formation_series,
    generate_b3f1_decisions,
)
from ftmoquant.research.alpha_lab.b3f1_underpowered_u2_validation import (
    FAMILY_ID,
    FORMATION_WINDOW,
    FROZEN_PREREGISTRATION_SHA256,
    PREREGISTRATION_PATH,
    SLEEVE_ID,
    X_SPEC,
    Y_SPEC,
    Z_ENTRY,
    Z_STOP,
    verify_preregistration,
)
from ftmoquant.research.alpha_lab.data import load_alpha_lab_dataset
from ftmoquant.research.alpha_lab.relative_value_adapter import RelativeValueEpisode
from ftmoquant.research.alpha_lab.validation import load_validation_dataset
from ftmoquant.research.alpha_lab.wick_fvg_squeeze_execution import load_m1_bidask
from ftmoquant.research.alpha_lab.wick_fvg_squeeze_screen import _annualized_sharpe
from ftmoquant.research.g1.artifacts import write_runtime_provenance
from ftmoquant.research.mean_reversion_h1_development import (
    Partition,
    _add_oanda_venue,
    _convert_to_account_currency,
    _iso,
    _ns,
    _reject_sealed_path,
    parse_partition,
    partition_bounds,
)
from ftmoquant.research.stage_g import HOLDOUT_START

RUN_MODULE_NAME = "ftmoquant.research.b3f1_u2_execution_promotion"
STRATEGY_SEMANTIC_ID = "b3f1_u2_execution_promotion_v1"
_ACCOUNT_CURRENCY = "USD"
_H1_BAR_TIMEFRAME = "1-HOUR"
_M1_BAR_TIMEFRAME = "1-MINUTE"
_CATALOG_BAR_AGGREGATION = "INTERNAL"
_ENGINE_BAR_AGGREGATION = "EXTERNAL"
EXECUTION_TIMING_LABEL = "native_spread_crossing_first_strictly_later_execution"

_LEGS: tuple[str, ...] = ("Y", "X")
_LEG_SPEC = {"Y": Y_SPEC, "X": X_SPEC}


class B3F1U2ExecutionPromotionError(ValueError):
    """Raised on any violation of this execution-promotion contract."""


# ---------------------------------------------------------------------------
# Catalog/readiness resolution -- mirrors mean_reversion_h1_development's
# _resolve_catalog_roots, narrowed to U2's exactly two legs.
# ---------------------------------------------------------------------------


def resolve_leg_roots(
    *, partition: Partition, catalog_root: Path, universe_readiness_path: Path
) -> dict[str, Path]:
    try:
        document = json.loads(universe_readiness_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise B3F1U2ExecutionPromotionError(
            f"could not read readiness document: {error}"
        ) from error
    if not isinstance(document, dict) or document.get("holdout_accessed") is not False:
        raise B3F1U2ExecutionPromotionError("readiness document is invalid")

    config: OandaAlphaLabConfig | OandaAlphaLabValidationConfig
    if partition is Partition.DEVELOPMENT:
        if document.get("readiness_version") != "oanda-alpha-lab-readiness-1":
            raise B3F1U2ExecutionPromotionError(
                "DEVELOPMENT readiness document has the wrong readiness_version"
            )
        config = load_oanda_alpha_lab_config()
    else:
        if (
            document.get("readiness_version") != VALIDATION_READINESS_VERSION
            or document.get("partition") != "VALIDATION"
        ):
            raise B3F1U2ExecutionPromotionError(
                "VALIDATION readiness document has the wrong "
                "readiness_version/partition"
            )
        config = load_oanda_alpha_lab_validation_config()

    statuses = document.get("per_instrument_status")
    if not isinstance(statuses, dict):
        raise B3F1U2ExecutionPromotionError("readiness document is malformed")
    ready = {
        instrument_id
        for instrument_id, status in statuses.items()
        if status == "research_ready"
    }
    required = {Y_SPEC.instrument_id, X_SPEC.instrument_id}
    if required - ready:
        raise B3F1U2ExecutionPromotionError(
            "both U2 legs must be research_ready; no leg may be excluded"
        )

    roots: dict[str, Path] = {}
    for leg, spec in _LEG_SPEC.items():
        dataset_symbol = config.instrument(spec.instrument_id).dataset_symbol
        root = catalog_root / oanda_symbol(dataset_symbol)
        _reject_sealed_path(root, partition=partition)
        roots[leg] = root
    return roots


def _leg_h1_log_close(
    *, partition: Partition, catalog_root: Path, universe_readiness_path: Path
) -> tuple[Any, Any]:
    """Y/X legs' H1 mid-close log series, reusing the same aligned-dataset
    loaders B3F1's own DEVELOPMENT/VALIDATION orchestrators already use
    unchanged."""

    import numpy as np

    if partition is Partition.DEVELOPMENT:
        dataset = load_alpha_lab_dataset(
            readiness_path=universe_readiness_path,
            development_root_dir=catalog_root,
            timeframe="H1",
            source="oanda",
        )
    else:
        dataset = load_validation_dataset(
            validation_root=catalog_root,
            universe_readiness_path=universe_readiness_path,
            timeframe="H1",
        )
    log_close = np.log(dataset.close)
    return log_close[Y_SPEC.instrument_id], log_close[X_SPEC.instrument_id]


# ---------------------------------------------------------------------------
# Two-leg Nautilus executor: a mechanical player of ALREADY-DECIDED Alpha-
# Lab reference episodes (same decision, same size), submitting a market
# order per leg at that leg's own precomputed entry/exit execution
# timestamp and tracking the logical spread trade's lifecycle across
# independent per-leg fills.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TwoLegOrderSubmissionRecord:
    logical_trade_id: str
    leg: str
    instrument_id: str
    action: str  # "entry" or "exit"
    execution_ns: int
    side: str
    quantity: str


@dataclass(frozen=True, slots=True)
class TwoLegFillRecord:
    logical_trade_id: str
    leg: str
    instrument_id: str
    action: str
    fill_ns: int
    side: str
    last_px: str


@dataclass(frozen=True, slots=True)
class LogicalTradeRecord:
    logical_trade_id: str
    entry_leg_fill_ns: dict[str, int]
    exit_leg_fill_ns: dict[str, int]
    both_legs_open_ns: int
    both_legs_closed_ns: int
    leg_pnl_usd: dict[str, str]
    total_pnl_usd: str


@dataclass(frozen=True, slots=True)
class _TwoLegInstruction:
    """One precomputed decision->execution instruction pair, extracted
    directly from an already-simulated Alpha-Lab reference episode (never
    recomputed): exactly the same entry/exit timestamps, sides, and
    quantities Alpha-Lab used for both legs."""

    logical_trade_id: str
    entry_ns: dict[str, int]
    exit_ns: dict[str, int]
    direction: dict[str, int]
    quantity: dict[str, Decimal]


def _instructions_from_episodes(
    episodes: Sequence[RelativeValueEpisode],
) -> tuple[_TwoLegInstruction, ...]:
    instructions = []
    for episode in episodes:
        instructions.append(
            _TwoLegInstruction(
                logical_trade_id=episode.logical_trade_id,
                entry_ns={"Y": episode.leg_a.entry_ns, "X": episode.leg_b.entry_ns},
                exit_ns={"Y": episode.leg_a.exit_ns, "X": episode.leg_b.exit_ns},
                direction={
                    "Y": episode.leg_a.direction,
                    "X": episode.leg_b.direction,
                },
                quantity={
                    "Y": episode.leg_a.quantity,
                    "X": episode.leg_b.quantity,
                },
            )
        )
    return tuple(instructions)


class _B3F1U2Executor(Strategy):
    def __new__(
        cls,
        *,
        instruments: Mapping[str, CurrencyPair],
        instructions: Sequence[_TwoLegInstruction],
        start_ns: int,
        end_exclusive_ns: int,
    ) -> _B3F1U2Executor:
        del instruments, instructions, start_ns, end_exclusive_ns
        return super().__new__(cls)

    def __init__(
        self,
        *,
        instruments: Mapping[str, CurrencyPair],
        instructions: Sequence[_TwoLegInstruction],
        start_ns: int,
        end_exclusive_ns: int,
    ) -> None:
        super().__init__(
            StrategyConfig(
                strategy_id=StrategyId("B3F1-U2-EXECUTION-PROMOTION-001"),
                log_events=False,
                log_commands=False,
            )
        )
        self._instruments = dict(instruments)
        self._start_ns = start_ns
        self._end_exclusive_ns = end_exclusive_ns
        self._pending: dict[str, dict[str, Bar]] = {leg: {} for leg in _LEGS}

        # Two schedules keyed by (leg, ts) -> instruction, built ONCE from
        # the already-decided Alpha-Lab reference episodes.
        self._entry_schedule: dict[tuple[str, int], _TwoLegInstruction] = {}
        self._exit_schedule: dict[tuple[str, int], _TwoLegInstruction] = {}
        for instruction in instructions:
            for leg in _LEGS:
                self._entry_schedule[(leg, instruction.entry_ns[leg])] = instruction
                self._exit_schedule[(leg, instruction.exit_ns[leg])] = instruction

        self.submissions: list[TwoLegOrderSubmissionRecord] = []
        self.fills: list[TwoLegFillRecord] = []
        self.logical_trades: list[LogicalTradeRecord] = []

        self._order_context: dict[
            str, tuple[str, str, str]
        ] = {}  # order_id -> (trade_id, leg, action)
        self._entry_fill_ns: dict[str, dict[str, int]] = {}
        self._exit_fill_ns: dict[str, dict[str, int]] = {}
        self._entry_fill_price: dict[str, dict[str, Decimal]] = {}

    def on_start(self) -> None:
        for leg, spec in _LEG_SPEC.items():
            for side in ("BID", "ASK"):
                self.subscribe_bars(
                    BarType.from_str(
                        f"{spec.instrument_id}-{_M1_BAR_TIMEFRAME}-{side}-"
                        f"{_ENGINE_BAR_AGGREGATION}"
                    )
                )

    def on_bar(self, bar: Bar) -> None:
        if bar.ts_event < self._start_ns or bar.ts_event >= self._end_exclusive_ns:
            return
        if bar.ts_event >= int(HOLDOUT_START.timestamp() * 1_000_000_000):
            raise B3F1U2ExecutionPromotionError(
                "final holdout is not accessible during execution promotion"
            )
        instrument_id = str(bar.bar_type.instrument_id)
        leg = "Y" if instrument_id == Y_SPEC.instrument_id else "X"
        side = bar.bar_type.spec.price_type.name
        frame = self._pending[leg]
        frame[side] = bar
        if "BID" not in frame or "ASK" not in frame:
            return
        bid_bar, ask_bar = frame["BID"], frame["ASK"]
        if bid_bar.ts_event != ask_bar.ts_event:
            return
        self._pending[leg] = {}
        ts_event = bid_bar.ts_event

        entry_instruction = self._entry_schedule.get((leg, ts_event))
        if entry_instruction is not None:
            self._submit_leg(leg, entry_instruction, "entry")
        exit_instruction = self._exit_schedule.get((leg, ts_event))
        if exit_instruction is not None:
            self._submit_leg(leg, exit_instruction, "exit")

    def _submit_leg(
        self, leg: str, instruction: _TwoLegInstruction, action: str
    ) -> None:
        direction = instruction.direction[leg]
        quantity_value = instruction.quantity[leg]
        spec = _LEG_SPEC[leg]
        instrument = self._instruments[spec.instrument_id]
        if action == "entry":
            side = OrderSide.BUY if direction == 1 else OrderSide.SELL
        else:
            side = OrderSide.SELL if direction == 1 else OrderSide.BUY
        quantity = instrument.make_qty(float(quantity_value))
        order = self.order_factory.market(
            instrument_id=InstrumentId.from_str(spec.instrument_id),
            order_side=side,
            quantity=quantity,
            tags=(
                STRATEGY_SEMANTIC_ID,
                f"logical_trade_id={instruction.logical_trade_id}",
                f"leg={leg}",
                f"action={action}",
            ),
        )
        self._order_context[order.client_order_id.value] = (
            instruction.logical_trade_id,
            leg,
            action,
        )
        self.submissions.append(
            TwoLegOrderSubmissionRecord(
                logical_trade_id=instruction.logical_trade_id,
                leg=leg,
                instrument_id=spec.instrument_id,
                action=action,
                execution_ns=(
                    instruction.entry_ns[leg]
                    if action == "entry"
                    else instruction.exit_ns[leg]
                ),
                side=side.name,
                quantity=str(quantity_value),
            )
        )
        self.submit_order(order)

    def on_order_filled(self, event: OrderFilled) -> None:
        context = self._order_context.get(event.client_order_id.value)
        if context is None:
            raise B3F1U2ExecutionPromotionError(
                "native fill lacks a precomputed instruction context"
            )
        trade_id, leg, action = context
        spec = _LEG_SPEC[leg]
        self.fills.append(
            TwoLegFillRecord(
                logical_trade_id=trade_id,
                leg=leg,
                instrument_id=spec.instrument_id,
                action=action,
                fill_ns=event.ts_event,
                side=event.order_side.name,
                last_px=str(event.last_px),
            )
        )
        fill_price = event.last_px.as_decimal()
        if action == "entry":
            # Both this leg's own entry-fill bookkeeping and the shared
            # native margin account's equity are already updated by
            # Nautilus itself the instant this single fill lands -- a
            # one-legged spread's unrealized exposure is therefore
            # genuinely visible in account equity even before the second
            # leg fills, with no extra tracking needed here.
            self._entry_fill_ns.setdefault(trade_id, {})[leg] = event.ts_event
            self._entry_fill_price.setdefault(trade_id, {})[leg] = fill_price
        else:
            self._exit_fill_ns.setdefault(trade_id, {})[leg] = event.ts_event
            if len(self._exit_fill_ns[trade_id]) == len(_LEGS) and trade_id not in {
                t.logical_trade_id for t in self.logical_trades
            }:
                self._close_logical_trade(trade_id, fill_price, leg)

    def _close_logical_trade(
        self, trade_id: str, exit_price: Decimal, closing_leg: str
    ) -> None:
        leg_pnl: dict[str, Decimal] = {}
        for leg in _LEGS:
            spec = _LEG_SPEC[leg]
            entry_price = self._entry_fill_price[trade_id][leg]
            # Direction is recovered from this leg's own ENTRY submission
            # record (frozen at signal-decision time), never re-derived
            # from whichever fill happens to close the trade last.
            instruction = next(
                s
                for s in self.submissions
                if s.logical_trade_id == trade_id
                and s.leg == leg
                and s.action == "entry"
            )
            entry_side = instruction.side
            original_direction = 1 if entry_side == "BUY" else -1
            leg_exit_price = (
                exit_price
                if leg == closing_leg
                else self._leg_exit_price(trade_id, leg)
            )
            pnl_quote = (
                original_direction
                * Decimal(instruction.quantity)
                * (leg_exit_price - entry_price)
            )
            leg_pnl[leg] = _convert_to_account_currency(
                pnl_quote,
                spec.quote_currency,
                base_currency=spec.base_currency,
                quote_currency=spec.quote_currency,
                conversion_price=entry_price,
            )
        total_pnl = sum(leg_pnl.values(), Decimal(0))
        self.logical_trades.append(
            LogicalTradeRecord(
                logical_trade_id=trade_id,
                entry_leg_fill_ns=dict(self._entry_fill_ns[trade_id]),
                exit_leg_fill_ns=dict(self._exit_fill_ns[trade_id]),
                both_legs_open_ns=max(self._entry_fill_ns[trade_id].values()),
                both_legs_closed_ns=max(self._exit_fill_ns[trade_id].values()),
                leg_pnl_usd={leg: str(value) for leg, value in leg_pnl.items()},
                total_pnl_usd=str(total_pnl),
            )
        )

    def _leg_exit_price(self, trade_id: str, leg: str) -> Decimal:
        for fill in self.fills:
            if (
                fill.logical_trade_id == trade_id
                and fill.leg == leg
                and fill.action == "exit"
            ):
                return Decimal(fill.last_px)
        raise B3F1U2ExecutionPromotionError(f"exit fill missing for {trade_id}/{leg}")


# ---------------------------------------------------------------------------
# Pure engine-wiring core -- directly callable from synthetic parity tests
# with fabricated bars, exactly mirroring run_frozen_signal_backtest's own
# separation of concerns.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EngineRunOutcome:
    submissions: tuple[TwoLegOrderSubmissionRecord, ...]
    fills: tuple[TwoLegFillRecord, ...]
    logical_trades: tuple[LogicalTradeRecord, ...]
    order_report_rows: int
    fill_report_rows: int
    position_report_rows: int


def run_frozen_u2_backtest(
    *,
    start: datetime,
    end_exclusive: datetime,
    instruments: Mapping[str, CurrencyPair],
    instructions: Sequence[_TwoLegInstruction],
    m1_bars: Sequence[Bar],
) -> EngineRunOutcome:
    """Build and run one real Nautilus ``BacktestEngine`` for U2's two legs
    over an arbitrary window, given ALREADY-DECIDED instructions (never
    computed here) and an already-loaded M1 BID/ASK bar stream. Has no
    concept of partition, never opens a catalog or readiness document --
    the sole production entry point is :func:`run_b3f1_u2_execution_promotion`,
    which always resolves dates via the frozen, non-overridable
    :func:`partition_bounds`."""

    if set(instruments) != {Y_SPEC.instrument_id, X_SPEC.instrument_id}:
        raise B3F1U2ExecutionPromotionError("exactly U2's two frozen legs are required")
    start_ns, end_ns = _ns(start), _ns(end_exclusive)
    profile = canonical_execution_profile()
    account = AccountParameters(
        initial_capital=Decimal("100000"),
        currency=_ACCOUNT_CURRENCY,
        leverage=Decimal(1),
    )

    identity = _sha256_json(
        {
            "module": RUN_MODULE_NAME,
            "sleeve_id": SLEEVE_ID,
            "formation_window": FORMATION_WINDOW,
            "z_entry": str(Z_ENTRY),
            "z_stop": str(Z_STOP),
            "start_utc": _iso(start),
            "end_exclusive_utc": _iso(end_exclusive),
            "execution_timing": EXECUTION_TIMING_LABEL,
            "instruction_count": len(instructions),
        }
    )
    engine = BacktestEngine(_engine_config(identity))
    executor = _B3F1U2Executor(
        instruments=instruments,
        instructions=instructions,
        start_ns=start_ns,
        end_exclusive_ns=end_ns,
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
        for instrument_id, instrument in instruments.items():
            engine.add_instrument(_instrument_for_profile(instrument, profile.fee))
        engine.add_data(
            sorted(m1_bars, key=lambda bar: (bar.ts_event, str(bar.bar_type))),
            validate=True,
            sort=True,
        )
        engine.add_strategy(executor)
        engine.run(
            start=start_ns,
            end=end_ns - 1,
            run_config_id=f"B3F1-U2-{identity[:20]}",
        )
        order_report = engine.generate_orders_report()
        fill_report = engine.generate_fills_report()
        position_report = engine.generate_positions_report()
    finally:
        engine.dispose()

    return EngineRunOutcome(
        submissions=tuple(executor.submissions),
        fills=tuple(executor.fills),
        logical_trades=tuple(executor.logical_trades),
        order_report_rows=len(order_report),
        fill_report_rows=len(fill_report),
        position_report_rows=len(position_report),
    )


# ---------------------------------------------------------------------------
# Parity / degradation measurement (section 7) -- descriptive only, no gate.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SideMetrics:
    trade_count: int
    net_pnl_usd: str
    expectancy_usd: str
    profit_factor: str
    annualized_sharpe: float | None
    maximum_drawdown: float


def _alpha_lab_side_metrics(
    episodes: Sequence[RelativeValueEpisode], initial_capital: Decimal
) -> SideMetrics:
    rows = _rows_from_episodes(episodes)
    if not rows:
        return SideMetrics(0, "0", "0", "0", None, 0.0)
    pnls = [row.pnl for row in rows]
    expectancy, pf = expectancy_and_profit_factor(pnls)
    net_pnl = sum(pnls, Decimal(0))
    equity = float(initial_capital)
    peak = equity
    max_dd = 0.0
    daily: dict[date, float] = {}
    for row in rows:
        equity += float(row.pnl)
        peak = max(peak, equity)
        max_dd = min(max_dd, equity / peak - 1.0)
        daily[row.exit_ts.date()] = equity
    values = sorted(daily.items())
    daily_returns = [
        (values[i][1] / values[i - 1][1]) - 1.0 for i in range(1, len(values))
    ]
    sharpe = _annualized_sharpe(daily_returns)
    return SideMetrics(
        trade_count=len(rows),
        net_pnl_usd=str(net_pnl),
        expectancy_usd=str(expectancy),
        profit_factor=str(pf),
        annualized_sharpe=sharpe,
        maximum_drawdown=max_dd,
    )


def _nautilus_side_metrics(
    logical_trades: Sequence[LogicalTradeRecord], initial_capital: Decimal
) -> SideMetrics:
    if not logical_trades:
        return SideMetrics(0, "0", "0", "0", None, 0.0)
    pnls = [Decimal(trade.total_pnl_usd) for trade in logical_trades]
    expectancy, pf = expectancy_and_profit_factor(pnls)
    net_pnl = sum(pnls, Decimal(0))
    ordered = sorted(
        zip(logical_trades, pnls, strict=True),
        key=lambda item: item[0].both_legs_closed_ns,
    )
    equity = float(initial_capital)
    peak = equity
    max_dd = 0.0
    daily: dict[date, float] = {}
    for trade, pnl in ordered:
        equity += float(pnl)
        peak = max(peak, equity)
        max_dd = min(max_dd, equity / peak - 1.0)
        day = datetime.fromtimestamp(
            trade.both_legs_closed_ns / 1_000_000_000, tz=UTC
        ).date()
        daily[day] = equity
    values = sorted(daily.items())
    daily_returns = [
        (values[i][1] / values[i - 1][1]) - 1.0 for i in range(1, len(values))
    ]
    sharpe = _annualized_sharpe(daily_returns)
    return SideMetrics(
        trade_count=len(logical_trades),
        net_pnl_usd=str(net_pnl),
        expectancy_usd=str(expectancy),
        profit_factor=str(pf),
        annualized_sharpe=sharpe,
        maximum_drawdown=max_dd,
    )


@dataclass(frozen=True, slots=True)
class ParityDiagnostics:
    alpha_lab: SideMetrics
    nautilus: SideMetrics
    trade_count_difference: int
    entry_timestamp_mismatches: int
    exit_timestamp_mismatches: int
    per_leg_fill_price_differences: dict[str, str]
    logical_trade_pnl_differences: dict[str, str]
    total_return_degradation_usd: str
    expectancy_degradation_usd: str
    profit_factor_degradation: str
    sharpe_degradation: float | None


def compute_parity_diagnostics(
    *,
    episodes: Sequence[RelativeValueEpisode],
    logical_trades: Sequence[LogicalTradeRecord],
    initial_capital: Decimal,
) -> ParityDiagnostics:
    alpha_lab = _alpha_lab_side_metrics(episodes, initial_capital)
    nautilus = _nautilus_side_metrics(logical_trades, initial_capital)

    episodes_by_id = {episode.logical_trade_id: episode for episode in episodes}
    trades_by_id = {trade.logical_trade_id: trade for trade in logical_trades}

    entry_mismatches = 0
    exit_mismatches = 0
    price_diffs: dict[str, str] = {}
    pnl_diffs: dict[str, str] = {}
    for trade_id, episode in episodes_by_id.items():
        nautilus_trade = trades_by_id.get(trade_id)
        if nautilus_trade is None:
            continue
        alpha_lab_entry = {"Y": episode.leg_a.entry_ns, "X": episode.leg_b.entry_ns}
        alpha_lab_exit = {"Y": episode.leg_a.exit_ns, "X": episode.leg_b.exit_ns}
        for leg in _LEGS:
            if alpha_lab_entry[leg] != nautilus_trade.entry_leg_fill_ns.get(leg):
                entry_mismatches += 1
            if alpha_lab_exit[leg] != nautilus_trade.exit_leg_fill_ns.get(leg):
                exit_mismatches += 1
        alpha_lab_pnl = episode.realized_pnl()
        nautilus_pnl = Decimal(nautilus_trade.total_pnl_usd)
        pnl_diffs[trade_id] = str(nautilus_pnl - alpha_lab_pnl)

    return ParityDiagnostics(
        alpha_lab=alpha_lab,
        nautilus=nautilus,
        trade_count_difference=nautilus.trade_count - alpha_lab.trade_count,
        entry_timestamp_mismatches=entry_mismatches,
        exit_timestamp_mismatches=exit_mismatches,
        per_leg_fill_price_differences=price_diffs,
        logical_trade_pnl_differences=pnl_diffs,
        total_return_degradation_usd=str(
            Decimal(nautilus.net_pnl_usd) - Decimal(alpha_lab.net_pnl_usd)
        ),
        expectancy_degradation_usd=str(
            Decimal(nautilus.expectancy_usd) - Decimal(alpha_lab.expectancy_usd)
        ),
        profit_factor_degradation=str(
            _safe_decimal(nautilus.profit_factor)
            - _safe_decimal(alpha_lab.profit_factor)
        ),
        sharpe_degradation=(
            nautilus.annualized_sharpe - alpha_lab.annualized_sharpe
            if nautilus.annualized_sharpe is not None
            and alpha_lab.annualized_sharpe is not None
            else None
        ),
    )


def _safe_decimal(value: str) -> Decimal:
    parsed = Decimal(value)
    return parsed if parsed.is_finite() else Decimal(0)


# ---------------------------------------------------------------------------
# Production orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExecutionPromotionResult:
    partition: str
    start_utc: str
    end_exclusive_utc: str
    candidate_id: str
    family: str
    sleeve_id: str
    formation_window: int
    z_entry: str
    z_stop: str
    execution_profile: str
    execution_timing: str
    calibration_status: str
    parity: ParityDiagnostics


def run_b3f1_u2_execution_promotion(
    *,
    partition: Partition,
    catalog_root: Path,
    universe_readiness_path: Path,
    output_dir: Path,
) -> ExecutionPromotionResult:
    """The full one-candidate execution-promotion run for exactly one
    partition. Write-once check happens FIRST; preregistration hash
    verification happens before any market-data path is opened."""

    if output_dir.exists():
        raise B3F1U2ExecutionPromotionError(
            f"output directory already exists: {output_dir}"
        )
    verify_preregistration(PREREGISTRATION_PATH)

    leg_roots = resolve_leg_roots(
        partition=partition,
        catalog_root=catalog_root,
        universe_readiness_path=universe_readiness_path,
    )
    start, end_exclusive = partition_bounds(partition)
    start_ns, end_ns = _ns(start), _ns(end_exclusive)

    log_y, log_x = _leg_h1_log_close(
        partition=partition,
        catalog_root=catalog_root,
        universe_readiness_path=universe_readiness_path,
    )
    formation = compute_formation_series(log_y, log_x, FORMATION_WINDOW)
    decisions = generate_b3f1_decisions(
        formation, log_y, log_x, sleeve_id=SLEEVE_ID, z_entry=Z_ENTRY, z_stop=Z_STOP
    )

    y_bid, y_ask = load_m1_bidask(
        instrument_id=Y_SPEC.instrument_id,
        root=leg_roots["Y"],
        start_utc=start,
        end_exclusive_utc=end_exclusive,
    )
    x_bid, x_ask = load_m1_bidask(
        instrument_id=X_SPEC.instrument_id,
        root=leg_roots["X"],
        start_utc=start,
        end_exclusive_utc=end_exclusive,
    )
    native_episodes, _skips = simulate_b3f1_intents(
        decisions,
        y_spec=Y_SPEC,
        x_spec=X_SPEC,
        y_bid_m1=y_bid,
        y_ask_m1=y_ask,
        x_bid_m1=x_bid,
        x_ask_m1=x_ask,
        gross_notional_usd=GROSS_NOTIONAL_USD,
        cost_stress_multiplier=Decimal("1"),
    )
    instructions = _instructions_from_episodes(native_episodes)

    catalog_y = ParquetDataCatalog(str(leg_roots["Y"] / "catalog"))
    catalog_x = ParquetDataCatalog(str(leg_roots["X"] / "catalog"))
    m1_bars: list[Bar] = []
    for spec, catalog in ((Y_SPEC, catalog_y), (X_SPEC, catalog_x)):
        for side in ("BID", "ASK"):
            bar_type = (
                f"{spec.instrument_id}-{_M1_BAR_TIMEFRAME}-{side}-"
                f"{_ENGINE_BAR_AGGREGATION}"
            )
            m1_bars.extend(
                catalog.query_bars([bar_type], start=start_ns, end=end_ns - 1)
            )

    instruments = {
        Y_SPEC.instrument_id: _load_nautilus_instrument(
            catalog_y, Y_SPEC.instrument_id
        ),
        X_SPEC.instrument_id: _load_nautilus_instrument(
            catalog_x, X_SPEC.instrument_id
        ),
    }

    outcome = run_frozen_u2_backtest(
        start=start,
        end_exclusive=end_exclusive,
        instruments=instruments,
        instructions=instructions,
        m1_bars=m1_bars,
    )

    parity = compute_parity_diagnostics(
        episodes=native_episodes,
        logical_trades=outcome.logical_trades,
        initial_capital=Decimal("100000"),
    )

    profile = canonical_execution_profile()
    result = ExecutionPromotionResult(
        partition=partition.value,
        start_utc=_iso(start),
        end_exclusive_utc=_iso(end_exclusive),
        candidate_id="U2",
        family=FAMILY_ID,
        sleeve_id=SLEEVE_ID,
        formation_window=FORMATION_WINDOW,
        z_entry=str(Z_ENTRY),
        z_stop=str(Z_STOP),
        execution_profile="canonical_execution_profile",
        execution_timing=EXECUTION_TIMING_LABEL,
        calibration_status=profile.calibration_status.value,
        parity=parity,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    _write_result_artifacts(output_dir, result, native_episodes, outcome)
    return result


def _load_nautilus_instrument(
    catalog: ParquetDataCatalog, instrument_id: str
) -> CurrencyPair:
    found = catalog.instruments([instrument_id])
    if len(found) != 1 or not isinstance(found[0], CurrencyPair):
        raise B3F1U2ExecutionPromotionError(
            f"frozen CurrencyPair is unavailable: {instrument_id}"
        )
    return found[0]


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode()


def _write_result_artifacts(
    output_dir: Path,
    result: ExecutionPromotionResult,
    episodes: Sequence[RelativeValueEpisode],
    outcome: EngineRunOutcome,
) -> None:
    result_dict = _jsonable(asdict(result))
    result_bytes = _canonical_bytes(result_dict) + b"\n"
    result_path = output_dir / "result.json"
    result_path.write_bytes(result_bytes)

    trades_path = output_dir / "trades.csv"
    import csv

    with trades_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "logical_trade_id",
                "source",
                "y_entry_ns",
                "x_entry_ns",
                "y_exit_ns",
                "x_exit_ns",
                "pnl_usd",
            ]
        )
        for episode in episodes:
            writer.writerow(
                [
                    episode.logical_trade_id,
                    "alpha_lab",
                    episode.leg_a.entry_ns,
                    episode.leg_b.entry_ns,
                    episode.leg_a.exit_ns,
                    episode.leg_b.exit_ns,
                    str(episode.realized_pnl()),
                ]
            )
        for trade in outcome.logical_trades:
            writer.writerow(
                [
                    trade.logical_trade_id,
                    "nautilus",
                    trade.entry_leg_fill_ns.get("Y"),
                    trade.entry_leg_fill_ns.get("X"),
                    trade.exit_leg_fill_ns.get("Y"),
                    trade.exit_leg_fill_ns.get("X"),
                    trade.total_pnl_usd,
                ]
            )

    write_runtime_provenance(
        output_dir / "runtime_provenance.json",
        {
            "git_commit": _git_commit(),
            "python_module": RUN_MODULE_NAME,
            "strategy_semantic_id": STRATEGY_SEMANTIC_ID,
            "preregistration_path": str(PREREGISTRATION_PATH),
            "preregistration_sha256": FROZEN_PREREGISTRATION_SHA256,
            "candidate_identity": {
                "candidate_id": "U2",
                "family": FAMILY_ID,
                "sleeve_id": SLEEVE_ID,
                "formation_window": FORMATION_WINDOW,
                "z_entry": str(Z_ENTRY),
                "z_stop": str(Z_STOP),
            },
            "execution_profile": "canonical_execution_profile",
            "execution_profile_identity": {
                "calibration_status": result.calibration_status,
                "native_bid_ask_crossing": True,
                "modeled_latency_ns": 0,
                "modeled_commission": "0",
                "modeled_slippage_probability": "0",
                "rollover": "disabled",
            },
            "execution_timing": result.execution_timing,
            "promotion_gates_evaluated": False,
            "sizing": "frozen_beta_weighted_100k_gross_no_optimization",
            "validation_accessed": result.partition == Partition.VALIDATION.value,
            "final_holdout_accessed": False,
        },
    )

    hashes = {
        "result.json": hashlib.sha256(result_bytes).hexdigest(),
        "trades.csv": hashlib.sha256(trades_path.read_bytes()).hexdigest(),
        "runtime_provenance.json": hashlib.sha256(
            (output_dir / "runtime_provenance.json").read_bytes()
        ).hexdigest(),
    }
    (output_dir / "artifact_hashes.json").write_text(
        json.dumps(hashes, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# CLI -- no signal/pair/parameter override flags exist.
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the frozen, VALIDATED B3F1 candidate U2's Alpha-Lab "
            "native-BID/ASK two-leg lifecycle through the real Nautilus "
            "execution harness for one already-frozen partition, and "
            "measure execution degradation."
        )
    )
    parser.add_argument(
        "--partition", choices=("development", "validation"), required=True
    )
    parser.add_argument("--catalog-root", type=Path, required=True)
    parser.add_argument("--universe-readiness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    partition = parse_partition(args.partition)
    result = run_b3f1_u2_execution_promotion(
        partition=partition,
        catalog_root=args.catalog_root,
        universe_readiness_path=args.universe_readiness,
        output_dir=args.output,
    )
    print(json.dumps(_jsonable(asdict(result)), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
