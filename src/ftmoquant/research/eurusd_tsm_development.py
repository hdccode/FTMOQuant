"""Production DEVELOPMENT evaluator and exact-grid runner for ``eurusd_tsm_v1``.

The module has no validation or holdout loader.  Real execution is opt-in only
through :func:`main`; importing and testing it never opens the frozen catalogs.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import pandas as pd  # type: ignore[import-untyped]
from nautilus_trader.backtest import BacktestEngine
from nautilus_trader.execution import ProbabilisticFillModel, StaticLatencyModel
from nautilus_trader.model import (
    Bar,
    Currency,
    CurrencyPair,
    InstrumentId,
    OrderFilled,
    OrderSide,
    Position,
    Price,
    StrategyId,
)
from nautilus_trader.persistence import ParquetDataCatalog
from nautilus_trader.trading import Strategy, StrategyConfig

from ftmoquant.backtest.execution_harness import (
    _add_venue,
    _engine_config,
    _fee_model,
    _instrument_for_profile,
    _rollover_modules,
    _sha256_json,
    canonical_execution_profile,
)
from ftmoquant.data.dukascopy import NAUTILUS_VERSION
from ftmoquant.research.eurusd_tsm_spec import (
    EURUSD_TSM_SEMANTIC_SHA256,
    EURUSD_TSM_SPEC_PATH,
    EurusdTsmSpec,
    load_eurusd_tsm_spec,
)
from ftmoquant.research.g1.artifacts import (
    deterministic_artifact_bytes,
    write_runtime_provenance,
)
from ftmoquant.research.g1.guards import (
    DevelopmentAccessPolicy,
    DevelopmentSearchContext,
    WalkForwardWindow,
)
from ftmoquant.research.g1.normalization import (
    CausalEwmaDailyVolatility,
    CompletedDailyMidpoint,
    G1VolatilityNormalizer,
    completed_daily_log_returns,
)
from ftmoquant.research.g1.parameter_space import ParameterMapping
from ftmoquant.research.g1.search import (
    SearchConfig,
    SearchMode,
    SearchResult,
    run_search,
)
from ftmoquant.research.g1.selector import SelectionDecision, select_candidate
from ftmoquant.research.g1.trials import Attribution, FoldMetrics
from ftmoquant.research.stage_g import (
    DEVELOPMENT_END_EXCLUSIVE,
    DEVELOPMENT_START,
    FROZEN_INSTRUMENT_IDS,
    DevelopmentResearchContext,
    ResearchPartition,
    frozen_development_folds,
    open_development_context,
)
from ftmoquant.research.ts_momentum_development import (
    DevelopmentEvaluationConfig,
    _annualized_sharpe,
    _maximum_drawdown,
    _validate_catalog_trees,
    load_development_evaluation_config,
)
from ftmoquant.strategies.eurusd_tsm import EurusdTsmFamily
from ftmoquant.strategies.trend_pullback import CompletedPair, PriceBar, Timeframe
from ftmoquant.strategies.ts_momentum import RawDirectionalTarget

EVALUATOR_VERSION = "eurusd-tsm-v1-development-evaluator-1"
RESULT_SCHEMA = "ftmoquant.g1-eurusd-tsm-development-results"
BASE_RESEARCH_UNITS = Decimal("100000")
_EURUSD = "EUR/USD.DUKASCOPY"
_MINUTE_NS = 60_000_000_000
_OBSERVATION_DELAY_NS = 1
_SESSION_ZONE = ZoneInfo("America/New_York")


class EurusdTsmEvaluationError(ValueError):
    """Raised when the frozen production evaluation cannot be honored exactly."""


@dataclass(frozen=True, slots=True)
class PreparedEurusdTsmMarketData:
    instrument: CurrencyPair
    minute_bars: tuple[Bar, ...]
    h1_pairs: tuple[CompletedPair, ...]
    h4_pairs: tuple[CompletedPair, ...]
    daily_midpoints: tuple[CompletedDailyMidpoint, ...]
    source_query_count: int = 0


@dataclass(frozen=True, slots=True)
class ScaledTargetInstruction:
    execution_event_ns: int
    execution_information_ns: int
    decision_information_ns: int
    raw_target: RawDirectionalTarget
    target_base_units: Decimal
    execution_midpoint: Decimal
    count_alpha_transition: bool = True
    reason: str = "scheduled_refresh"


@dataclass(frozen=True, slots=True)
class EquityPoint:
    information_time_ns: int
    equity: Decimal


@dataclass(frozen=True, slots=True)
class NativeFoldResult:
    fold_id: str
    executed_raw_alpha_transitions: int
    turnover_base_units: Decimal
    realized_variable_cost: Decimal
    equity_points: tuple[EquityPoint, ...]
    order_report: pd.DataFrame
    fill_report: pd.DataFrame
    position_report: pd.DataFrame


class ExecutedAlphaTransitionTracker:
    """Count raw-state changes only when a native position-changing fill occurs."""

    def __init__(self) -> None:
        self.expressed_raw_target = RawDirectionalTarget.FLAT
        self.count = 0

    def observe_fill(self, instruction: ScaledTargetInstruction) -> bool:
        if (
            not instruction.count_alpha_transition
            or instruction.raw_target is self.expressed_raw_target
        ):
            return False
        self.expressed_raw_target = instruction.raw_target
        self.count += 1
        return True


class EurusdTsmTrialEvaluator:
    """Callable generic-G1 evaluator backed by one prepared immutable data load."""

    def __init__(
        self,
        prepared: PreparedEurusdTsmMarketData,
        evaluation_config: DevelopmentEvaluationConfig,
        spec: EurusdTsmSpec,
        search_context: DevelopmentSearchContext | None = None,
    ) -> None:
        self.prepared = prepared
        self.evaluation_config = evaluation_config
        self.spec = spec
        self.search_context = (
            _search_context() if search_context is None else search_context
        )

    def __call__(
        self,
        family: Any,
        parameters: Mapping[str, str | int | float | bool | None],
        window: WalkForwardWindow,
    ) -> FoldMetrics:
        if not isinstance(family, EurusdTsmFamily):
            raise EurusdTsmEvaluationError("evaluator received the wrong family")
        family.validate_parameters(parameters)
        _validate_window(window, self.search_context)
        pairs = (
            self.prepared.h1_pairs
            if parameters["timeframe"] == "H1"
            else self.prepared.h4_pairs
        )
        sliced_pairs = tuple(
            pair
            for pair in pairs
            if _ns(window.train_start_utc)
            <= pair.info_time_ns
            < _ns(window.evaluate_end_exclusive_utc)
        )
        minute_bars = tuple(
            bar
            for bar in self.prepared.minute_bars
            if _ns(window.evaluate_start_utc)
            <= bar.ts_init
            < _ns(window.evaluate_end_exclusive_utc)
        )
        daily = tuple(
            item
            for item in self.prepared.daily_midpoints
            if _ns(window.train_start_utc)
            <= item.information_time_ns
            < _ns(window.evaluate_end_exclusive_utc)
        )
        if not sliced_pairs or not minute_bars or len(daily) < 2:
            raise EurusdTsmEvaluationError(
                f"{window.fold_id} has insufficient prepared market inputs"
            )
        instructions = _build_scaled_instructions(
            family,
            parameters,
            sliced_pairs,
            daily,
            minute_bars,
            window,
        )
        native = _run_native_fold(
            fold_id=window.fold_id,
            instrument=self.prepared.instrument,
            minute_bars=minute_bars,
            instructions=instructions,
            config=self.evaluation_config,
            start_ns=_ns(window.evaluate_start_utc),
            end_exclusive_ns=_ns(window.evaluate_end_exclusive_utc),
        )
        return _fold_metrics(native, window)


def prepare_eurusd_tsm_market_data(
    context: DevelopmentResearchContext,
    development_roots: Mapping[str, Path],
) -> PreparedEurusdTsmMarketData:
    """Load authoritative EUR/USD DEVELOPMENT streams exactly once."""

    context.require_range(
        DEVELOPMENT_START,
        DEVELOPMENT_END_EXCLUSIVE,
        partition=ResearchPartition.DEVELOPMENT,
    )
    _validate_catalog_trees(context, development_roots)
    root = development_roots[_EURUSD].resolve()
    _reject_sealed_path(root)
    catalog = ParquetDataCatalog(str(root / "catalog"))
    found = catalog.instruments([_EURUSD])
    if len(found) != 1 or not isinstance(found[0], CurrencyPair):
        raise EurusdTsmEvaluationError("frozen EUR/USD CurrencyPair is unavailable")
    profile = canonical_execution_profile()
    instrument = _instrument_for_profile(found[0], profile.fee)
    start = _ns(DEVELOPMENT_START)
    end = _ns(DEVELOPMENT_END_EXCLUSIVE)
    queried: dict[str, tuple[Bar, ...]] = {}
    for bar_type in (
        f"{_EURUSD}-1-MINUTE-BID-EXTERNAL",
        f"{_EURUSD}-1-MINUTE-ASK-EXTERNAL",
        f"{_EURUSD}-1-HOUR-BID-INTERNAL",
        f"{_EURUSD}-1-HOUR-ASK-INTERNAL",
        f"{_EURUSD}-4-HOUR-BID-INTERNAL",
        f"{_EURUSD}-4-HOUR-ASK-INTERNAL",
    ):
        bars = tuple(catalog.query_bars([bar_type], start=start, end=end - 1))
        _validate_bar_stream(bars, bar_type, start, end)
        queried[bar_type] = bars
    minute = tuple(
        sorted(
            (
                *queried[f"{_EURUSD}-1-MINUTE-BID-EXTERNAL"],
                *queried[f"{_EURUSD}-1-MINUTE-ASK-EXTERNAL"],
            ),
            key=lambda bar: (bar.ts_event, str(bar.bar_type)),
        )
    )
    h1 = _pair_native_bars(
        queried[f"{_EURUSD}-1-HOUR-BID-INTERNAL"],
        queried[f"{_EURUSD}-1-HOUR-ASK-INTERNAL"],
        Timeframe.HOUR,
    )
    h4 = _pair_native_bars(
        queried[f"{_EURUSD}-4-HOUR-BID-INTERNAL"],
        queried[f"{_EURUSD}-4-HOUR-ASK-INTERNAL"],
        Timeframe.FOUR_HOURS,
    )
    daily = _daily_midpoints_from_minute_bars(minute)
    _validate_catalog_trees(context, development_roots)
    return PreparedEurusdTsmMarketData(instrument, minute, h1, h4, daily, 6)


def run_eurusd_tsm_development(
    *,
    spec_path: Path,
    universe_readiness_path: Path,
    development_roots: Mapping[str, Path],
    evaluation_config_path: Path,
    output_dir: Path,
) -> tuple[SearchResult, SelectionDecision, dict[str, Any]]:
    """Execute the exact 90-cell DEVELOPMENT grid and persist deterministic output."""

    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    _require_clean_worktree()
    spec = load_eurusd_tsm_spec(spec_path)
    if spec.semantic_sha256 != EURUSD_TSM_SEMANTIC_SHA256:
        raise EurusdTsmEvaluationError("semantic SHA mismatch blocks runner")
    for path in development_roots.values():
        _reject_sealed_path(path)
    context = open_development_context(universe_readiness_path, development_roots)
    prepared = prepare_eurusd_tsm_market_data(context, development_roots)
    config = load_development_evaluation_config(evaluation_config_path)
    family = EurusdTsmFamily(spec)
    search_context = _search_context()
    result = run_search(
        family=family,
        context=search_context,
        config=SearchConfig(mode=SearchMode.EXACT_GRID, seed=0),
        evaluator=EurusdTsmTrialEvaluator(
            prepared, config, spec, search_context=search_context
        ),
    )
    if result.exact_trial_count != 90:
        raise EurusdTsmEvaluationError("production search did not attempt 90 cells")
    selection = select_candidate(
        family=family, registry=result.registry, policy=spec.selection_policy
    )
    outcome = (
        "ALPHA_REJECTED"
        if selection.selected_trial_id is None
        else "DEVELOPMENT_CANDIDATE_SELECTED"
    )
    provenance = _provenance(spec, context, development_roots)
    summary = {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "family_id": spec.family_id,
        "family_version": spec.version,
        "family_semantic_sha256": spec.semantic_sha256,
        "outcome": outcome,
        "selected_trial_id": selection.selected_trial_id,
        "search_landscape": _search_landscape(result, selection),
        "provenance": provenance,
    }
    _write_result_artifacts(output_dir, result, selection, summary)
    return result, selection, summary


def _build_scaled_instructions(
    family: EurusdTsmFamily,
    parameters: ParameterMapping,
    pairs: Sequence[CompletedPair],
    daily_midpoints: Sequence[CompletedDailyMidpoint],
    minute_bars: Sequence[Bar],
    window: WalkForwardWindow,
) -> tuple[ScaledTargetInstruction, ...]:
    state = family.create_state(parameters)
    estimator = CausalEwmaDailyVolatility(completed_daily_log_returns(daily_midpoints))
    normalizer = G1VolatilityNormalizer()
    execution_frames = _minute_execution_frames(minute_bars)
    execution_information = tuple(item[1] for item in execution_frames)
    evaluate_start = _ns(window.evaluate_start_utc)
    evaluate_end = _ns(window.evaluate_end_exclusive_utc)
    planned_units = Decimal(0)
    result: list[ScaledTargetInstruction] = []
    for pair in pairs:
        refresh = state.on_target_refresh(pair)
        if refresh is None or not (
            evaluate_start <= refresh.signal_information_ns < evaluate_end
        ):
            continue
        decision = normalizer.target_at(
            float(refresh.target), refresh.signal_information_ns, estimator
        )
        target_units = decision.target_base_units(BASE_RESEARCH_UNITS).quantize(
            Decimal("0.00000001")
        )
        if target_units == planned_units:
            continue
        execution = _strictly_later_frame(
            execution_frames,
            execution_information,
            refresh.signal_information_ns,
            evaluate_end,
        )
        if execution is None:
            continue
        result.append(
            ScaledTargetInstruction(
                execution_event_ns=execution[0],
                execution_information_ns=execution[1],
                decision_information_ns=refresh.signal_information_ns,
                raw_target=refresh.target,
                target_base_units=target_units,
                execution_midpoint=execution[2],
            )
        )
        planned_units = target_units
    if planned_units != 0:
        final = next(
            (item for item in reversed(execution_frames) if item[1] < evaluate_end),
            None,
        )
        if final is None or (result and final[0] <= result[-1].execution_event_ns):
            raise EurusdTsmEvaluationError(
                "no safe final pre-boundary observation can liquidate exposure"
            )
        result.append(
            ScaledTargetInstruction(
                execution_event_ns=final[0],
                execution_information_ns=final[1],
                decision_information_ns=final[1] - 1,
                raw_target=state.latest_target or RawDirectionalTarget.FLAT,
                target_base_units=Decimal(0),
                execution_midpoint=final[2],
                count_alpha_transition=False,
                reason="fold_end_liquidation",
            )
        )
    if len({item.execution_event_ns for item in result}) != len(result):
        raise EurusdTsmEvaluationError("multiple targets mapped to one execution frame")
    return tuple(result)


class _ScaledTargetExecutor(Strategy):
    def __new__(
        cls,
        instructions: Sequence[ScaledTargetInstruction],
        instrument: CurrencyPair,
        evaluation_config: DevelopmentEvaluationConfig,
        start_ns: int,
        end_exclusive_ns: int,
    ) -> _ScaledTargetExecutor:
        del instructions, instrument, evaluation_config, start_ns, end_exclusive_ns
        return super().__new__(cls)

    def __init__(
        self,
        instructions: Sequence[ScaledTargetInstruction],
        instrument: CurrencyPair,
        evaluation_config: DevelopmentEvaluationConfig,
        start_ns: int,
        end_exclusive_ns: int,
    ) -> None:
        super().__init__(
            StrategyConfig(
                strategy_id=StrategyId("EURUSD-TSM-V1-DEVELOPMENT-001"),
                log_events=False,
                log_commands=False,
            )
        )
        self._instructions = {item.execution_event_ns: item for item in instructions}
        if len(self._instructions) != len(instructions):
            raise EurusdTsmEvaluationError("duplicate execution instruction")
        self._instrument = instrument
        self._config = evaluation_config
        self._start_ns = start_ns
        self._end_exclusive_ns = end_exclusive_ns
        self._target_units = Decimal(0)
        self._order_context: dict[str, ScaledTargetInstruction] = {}
        self._counted_orders: set[str] = set()
        self._tracker = ExecutedAlphaTransitionTracker()
        self._bars: dict[int, dict[str, Bar]] = {}
        self._last_pair: tuple[int, Price, Price] | None = None
        self.turnover_base_units = Decimal(0)
        self.realized_variable_cost = Decimal(0)
        self.equity_points: list[EquityPoint] = [
            EquityPoint(start_ns, evaluation_config.account.initial_capital)
        ]

    @property
    def transition_count(self) -> int:
        return self._tracker.count

    def on_start(self) -> None:
        for side in ("BID", "ASK"):
            self.subscribe_bars(_bar_type(f"{_EURUSD}-1-MINUTE-{side}-EXTERNAL"))

    def on_bar(self, bar: Bar) -> None:
        if bar.ts_init < self._start_ns or bar.ts_init >= self._end_exclusive_ns:
            return
        side = bar.bar_type.spec.price_type.name
        frame = self._bars.setdefault(bar.ts_event, {})
        if side in frame:
            raise RuntimeError("duplicate native side bar")
        frame[side] = bar
        if set(frame) != {"BID", "ASK"}:
            return
        bid = frame["BID"].close
        ask = frame["ASK"].close
        info = max(frame["BID"].ts_init, frame["ASK"].ts_init)
        self._last_pair = (info, bid, ask)
        instruction = self._instructions.get(bar.ts_event)
        if instruction is not None:
            if info != instruction.execution_information_ns:
                raise RuntimeError("execution information timestamp drifted")
            self._submit_target(instruction)
        local = _datetime(info).astimezone(_SESSION_ZONE)
        if local.weekday() <= 4 and local.hour == 17 and local.minute == 0:
            self.clock.set_time_alert_ns(
                name=f"eurusd-tsm-mark-{bar.ts_event}",
                alert_time_ns=info + _OBSERVATION_DELAY_NS,
                callback=self._on_mark,
            )
        del self._bars[bar.ts_event]

    def on_stop(self) -> None:
        if self._target_units != 0:
            raise RuntimeError("fold ended with nonzero target exposure")
        if self._last_pair is not None:
            self._append_equity(*self._last_pair)

    def on_order_filled(self, event: OrderFilled) -> None:
        instruction = self._order_context.get(event.client_order_id.value)
        if instruction is None:
            raise RuntimeError("native fill lacks scaled target context")
        quantity = event.last_qty.as_decimal()
        fill = event.last_px.as_decimal()
        spread = (
            quantity * (fill - instruction.execution_midpoint)
            if event.order_side is OrderSide.BUY
            else quantity * (instruction.execution_midpoint - fill)
        )
        commission = (
            Decimal(0)
            if event.commission is None
            else abs(event.commission.as_decimal())
        )
        cost = spread + commission
        if cost < 0:
            raise RuntimeError("native fill improved beyond execution midpoint")
        self.turnover_base_units += quantity
        self.realized_variable_cost += cost
        order_id = event.client_order_id.value
        if order_id not in self._counted_orders:
            self._counted_orders.add(order_id)
            self._tracker.observe_fill(instruction)

    def _submit_target(self, instruction: ScaledTargetInstruction) -> None:
        delta = instruction.target_base_units - self._target_units
        if delta == 0:
            return
        if abs(instruction.target_base_units) > Decimal("3000000"):
            raise RuntimeError("frozen EUR currency-exposure limit breached")
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        quantity = self._instrument.make_qty(float(abs(delta)))
        order = self.order_factory.market(
            instrument_id=InstrumentId.from_str(_EURUSD),
            order_side=side,
            quantity=quantity,
            tags=(
                "eurusd_tsm_v1",
                f"decision_information_ns={instruction.decision_information_ns}",
                f"raw_target={int(instruction.raw_target)}",
                f"reason={instruction.reason}",
            ),
        )
        self._order_context[order.client_order_id.value] = instruction
        self._target_units = instruction.target_base_units
        self.submit_order(order)

    def _on_mark(self, _: Any) -> None:
        if self._last_pair is not None:
            self._append_equity(*self._last_pair)

    def _append_equity(self, info: int, bid: Price, ask: Price) -> None:
        account = self.cache.account_for_venue(self._instrument.id.venue)
        if account is None:
            raise RuntimeError("native account unavailable")
        balance = account.balance_total(Currency.from_str("USD"))
        if balance is None:
            raise RuntimeError("native USD balance unavailable")
        equity = balance.as_decimal()
        for position in cast(list[Position], self.cache.positions_open()):
            mark = bid if position.is_long else ask
            equity += position.unrealized_pnl(mark).as_decimal()
        point = EquityPoint(info, equity)
        if point.information_time_ns > self.equity_points[-1].information_time_ns:
            self.equity_points.append(point)


def _run_native_fold(
    *,
    fold_id: str,
    instrument: CurrencyPair,
    minute_bars: Sequence[Bar],
    instructions: Sequence[ScaledTargetInstruction],
    config: DevelopmentEvaluationConfig,
    start_ns: int,
    end_exclusive_ns: int,
) -> NativeFoldResult:
    if version("nautilus_trader") != NAUTILUS_VERSION:
        raise RuntimeError("NautilusTrader version drifted")
    profile = config.cost_models[0].execution_profile
    identity = _sha256_json(
        {
            "family_semantic_sha256": EURUSD_TSM_SEMANTIC_SHA256,
            "fold_id": fold_id,
            "instructions": [asdict(item) for item in instructions],
        }
    )
    engine = BacktestEngine(_engine_config(identity))
    executor = _ScaledTargetExecutor(
        instructions, instrument, config, start_ns, end_exclusive_ns
    )
    try:
        _add_venue(
            engine,
            config.account,
            profile,
            ProbabilisticFillModel(
                prob_fill_on_limit=float(profile.fill_on_limit_probability),
                prob_slippage=float(profile.adverse_slippage_probability),
                random_seed=profile.random_seed,
            ),
            StaticLatencyModel(
                base_latency_nanos=profile.base_latency_ns,
                insert_latency_nanos=profile.insert_latency_ns,
                update_latency_nanos=profile.update_latency_ns,
                cancel_latency_nanos=profile.cancel_latency_ns,
            ),
            _fee_model(profile.fee),
            _rollover_modules(profile.rollover),
        )
        engine.add_instrument(instrument)
        engine.add_data(list(minute_bars), validate=True, sort=True)
        engine.add_strategy(executor)
        engine.run(
            start=start_ns,
            end=end_exclusive_ns - 1,
            run_config_id=f"G1TSM-{identity[:20]}",
        )
        return NativeFoldResult(
            fold_id=fold_id,
            executed_raw_alpha_transitions=executor.transition_count,
            turnover_base_units=executor.turnover_base_units,
            realized_variable_cost=executor.realized_variable_cost,
            equity_points=tuple(executor.equity_points),
            order_report=cast(pd.DataFrame, engine.generate_orders_report()),
            fill_report=cast(pd.DataFrame, engine.generate_fills_report()),
            position_report=cast(pd.DataFrame, engine.generate_positions_report()),
        )
    finally:
        engine.dispose()


def _fold_metrics(native: NativeFoldResult, window: WalkForwardWindow) -> FoldMetrics:
    points = native.equity_points
    if len(points) < 2:
        points = (
            *points,
            EquityPoint(_ns(window.evaluate_end_exclusive_utc) - 1, points[-1].equity),
        )
    returns = tuple(
        float((current.equity - previous.equity) / BASE_RESEARCH_UNITS)
        for previous, current in zip(points, points[1:])
    )
    total_net_return = float(
        (points[-1].equity - points[0].equity) / BASE_RESEARCH_UNITS
    )
    cost_return = float(native.realized_variable_cost / BASE_RESEARCH_UNITS)
    stressed_total = total_net_return - 0.5 * cost_return
    count = native.executed_raw_alpha_transitions
    expectancy = total_net_return / count if count else 0.0
    stressed_expectancy = stressed_total / count if count else 0.0
    yearly: dict[str, float] = {}
    for previous, current in zip(points, points[1:]):
        label = str(_datetime(current.information_time_ns).year)
        yearly[label] = yearly.get(label, 0.0) + float(
            (current.equity - previous.equity) / BASE_RESEARCH_UNITS
        )
    return FoldMetrics(
        fold_id=window.fold_id,
        trade_count=count,
        net_expectancy=expectancy,
        sharpe=_annualized_sharpe(returns) or 0.0,
        drawdown=_maximum_drawdown(returns),
        turnover=float(native.turnover_base_units / BASE_RESEARCH_UNITS),
        cost_stressed_expectancy=stressed_expectancy,
        net_daily_mean=sum(returns) / len(returns),
        yearly_attribution=tuple(
            Attribution(label, value) for label, value in sorted(yearly.items())
        ),
        execution_perturbed_expectancy=None,
    )


def _pair_native_bars(
    bid_bars: Sequence[Bar], ask_bars: Sequence[Bar], timeframe: Timeframe
) -> tuple[CompletedPair, ...]:
    bids = {bar.ts_event: bar for bar in bid_bars}
    asks = {bar.ts_event: bar for bar in ask_bars}
    if (
        len(bids) != len(bid_bars)
        or len(asks) != len(ask_bars)
        or set(bids) != set(asks)
    ):
        raise EurusdTsmEvaluationError("derived BID/ASK timestamps are not one-to-one")
    return tuple(
        CompletedPair(
            timeframe=timeframe,
            bid=_price_bar(bids[timestamp]),
            ask=_price_bar(asks[timestamp]),
            ts_event=timestamp,
            info_time_ns=max(bids[timestamp].ts_init, asks[timestamp].ts_init),
            contiguous=True,
        )
        for timestamp in sorted(bids)
    )


def _daily_midpoints_from_minute_bars(
    minute_bars: Sequence[Bar],
) -> tuple[CompletedDailyMidpoint, ...]:
    result: list[CompletedDailyMidpoint] = []
    for _, information, midpoint in _minute_execution_frames(minute_bars):
        local = _datetime(information).astimezone(_SESSION_ZONE)
        if local.weekday() <= 4 and local.hour == 17 and local.minute == 0:
            result.append(CompletedDailyMidpoint(information, float(midpoint)))
    return tuple(result)


def _minute_execution_frames(
    minute_bars: Sequence[Bar],
) -> tuple[tuple[int, int, Decimal], ...]:
    by_event: dict[int, dict[str, Bar]] = {}
    for bar in minute_bars:
        side = bar.bar_type.spec.price_type.name
        if side not in {"BID", "ASK"}:
            raise EurusdTsmEvaluationError("minute execution bar side is invalid")
        sides = by_event.setdefault(bar.ts_event, {})
        if side in sides:
            raise EurusdTsmEvaluationError("duplicate minute execution side")
        sides[side] = bar
    result: list[tuple[int, int, Decimal]] = []
    for event in sorted(by_event):
        sides = by_event[event]
        if set(sides) != {"BID", "ASK"}:
            continue
        info = max(sides["BID"].ts_init, sides["ASK"].ts_init)
        midpoint = (
            sides["BID"].close.as_decimal() + sides["ASK"].close.as_decimal()
        ) / Decimal(2)
        result.append((event, info, midpoint))
    return tuple(result)


def _strictly_later_frame(
    frames: Sequence[tuple[int, int, Decimal]],
    information_times: Sequence[int],
    decision_ns: int,
    end_exclusive_ns: int,
) -> tuple[int, int, Decimal] | None:
    index = bisect.bisect_right(information_times, decision_ns)
    if index >= len(frames) or frames[index][1] >= end_exclusive_ns:
        return None
    return frames[index]


def _validate_bar_stream(
    bars: Sequence[Bar], bar_type: str, start_ns: int, end_ns: int
) -> None:
    if not bars:
        raise EurusdTsmEvaluationError(f"DEVELOPMENT has no {bar_type} bars")
    keys = tuple((bar.ts_event, bar.ts_init) for bar in bars)
    if len(set(keys)) != len(keys) or any(
        current <= previous for previous, current in zip(keys, keys[1:])
    ):
        raise EurusdTsmEvaluationError(f"{bar_type} timestamps are not monotonic")
    if any(
        str(bar.bar_type) != bar_type
        or bar.ts_event < start_ns
        or bar.ts_init >= end_ns
        for bar in bars
    ):
        raise EurusdTsmEvaluationError(f"{bar_type} query admitted invalid rows")


def _price_bar(bar: Bar) -> PriceBar:
    return PriceBar(
        bar.open.as_decimal(),
        bar.high.as_decimal(),
        bar.low.as_decimal(),
        bar.close.as_decimal(),
    )


def _search_context() -> DevelopmentSearchContext:
    folds = frozen_development_folds()
    return DevelopmentSearchContext(
        access_policy=DevelopmentAccessPolicy(
            DEVELOPMENT_START, DEVELOPMENT_END_EXCLUSIVE
        ),
        windows=tuple(
            WalkForwardWindow(
                fold.fold_id,
                fold.train_start_utc,
                fold.train_end_exclusive_utc,
                fold.compare_start_utc,
                fold.compare_end_exclusive_utc,
            )
            for fold in folds.folds
        ),
    )


def _validate_window(
    window: WalkForwardWindow, context: DevelopmentSearchContext
) -> None:
    expected = {item.fold_id: item for item in context.windows}
    if window.fold_id not in expected or window != expected[window.fold_id]:
        raise EurusdTsmEvaluationError("requested fold is not frozen")


def _provenance(
    spec: EurusdTsmSpec,
    context: DevelopmentResearchContext,
    development_roots: Mapping[str, Path],
) -> dict[str, Any]:
    _validate_catalog_trees(context, development_roots)
    return {
        "git_commit": _git_commit(),
        "evaluator_version": EVALUATOR_VERSION,
        "family_semantic_sha256": spec.semantic_sha256,
        "universe_readiness_sha256": context.universe.readiness_sha256,
        "universe_plan_sha256": context.universe.plan.semantic_sha256,
        "development_artifacts": [asdict(item) for item in context.artifacts],
        "validation_accessed": False,
        "final_holdout_accessed": False,
    }


def _write_result_artifacts(
    output_dir: Path,
    result: SearchResult,
    selection: SelectionDecision,
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    registry_bytes = deterministic_artifact_bytes(result, selection)
    registry_path = output_dir / "trial_registry.json"
    registry_path.write_bytes(registry_bytes)
    registry_sha = hashlib.sha256(registry_bytes).hexdigest()
    deterministic = {
        **summary,
        "trial_registry_path": registry_path.name,
        "trial_registry_sha256": registry_sha,
    }
    encoded = _canonical_bytes(deterministic)
    result_path = output_dir / "development_result.json"
    result_bytes = encoded + b"\n"
    result_path.write_bytes(result_bytes)
    manifest = {
        "development_result.json": hashlib.sha256(result_bytes).hexdigest(),
        "trial_registry.json": registry_sha,
    }
    write_runtime_provenance(
        output_dir / "runtime_provenance.json",
        {
            "git_commit": _git_commit(),
            "python_module": "ftmoquant.research.eurusd_tsm_development",
        },
    )
    if selection.selected_trial_id is not None:
        record = result.registry.get(selection.selected_trial_id)
        candidate = {
            "family_id": result.family_id,
            "family_version": result.family_version,
            "family_semantic_sha256": EURUSD_TSM_SEMANTIC_SHA256,
            "selected_trial_id": record.trial_id,
            "parameters": record.parameter_mapping,
            "validation_accessed": False,
        }
        candidate_bytes = _canonical_bytes(candidate) + b"\n"
        (output_dir / "selected_candidate.json").write_bytes(candidate_bytes)
        manifest["selected_candidate.json"] = hashlib.sha256(
            candidate_bytes
        ).hexdigest()
    (output_dir / "artifact_hashes.json").write_bytes(
        _canonical_bytes(manifest) + b"\n"
    )


def _search_landscape(
    result: SearchResult, selection: SelectionDecision
) -> dict[str, int]:
    return {
        "attempted": result.registry.total_attempted_configurations,
        "unique": result.registry.total_unique_configurations,
        "complete": result.registry.valid_configurations,
        "invalid": result.registry.invalid_configurations,
        "failed": result.registry.failed_configurations,
        "hard_gate_eligible": (selection.configurations_surviving_eligibility_gates),
        "selector_eligible": (selection.configurations_considered_by_final_selector),
    }


def _reject_sealed_path(path: Path) -> None:
    lowered = tuple(part.casefold() for part in path.parts)
    forbidden = ("validation", "holdout", "final_holdout", "final-holdout")
    if any(marker in part for marker in forbidden for part in lowered):
        raise EurusdTsmEvaluationError("validation/final-holdout path is forbidden")


def _require_clean_worktree() -> None:
    completed = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout.strip():
        raise EurusdTsmEvaluationError("production runner requires a clean worktree")


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _ns(value: datetime) -> int:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise EurusdTsmEvaluationError("timestamp must be timezone-aware UTC")
    return int(value.timestamp() * 1_000_000_000)


def _datetime(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1_000_000_000, tz=UTC)


def _bar_type(value: str) -> Any:
    from nautilus_trader.model import BarType

    return BarType.from_str(value)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str
    ).encode()


def _development_root(value: str) -> tuple[str, Path]:
    instrument, separator, raw = value.partition("=")
    if not separator or instrument not in FROZEN_INSTRUMENT_IDS or not raw:
        raise argparse.ArgumentTypeError("development root must be INSTRUMENT_ID=PATH")
    path = Path(raw)
    try:
        _reject_sealed_path(path)
    except EurusdTsmEvaluationError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return instrument, path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute frozen eurusd_tsm_v1 exact DEVELOPMENT grid"
    )
    parser.add_argument("--spec", type=Path, default=EURUSD_TSM_SPEC_PATH)
    parser.add_argument("--universe-readiness", type=Path, required=True)
    parser.add_argument(
        "--development-root", type=_development_root, action="append", required=True
    )
    parser.add_argument("--evaluation-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    pairs = cast(list[tuple[str, Path]], args.development_root)
    roots = dict(pairs)
    if len(roots) != len(pairs):
        raise EurusdTsmEvaluationError("duplicate development-root instrument")
    _, _, summary = run_eurusd_tsm_development(
        spec_path=cast(Path, args.spec),
        universe_readiness_path=cast(Path, args.universe_readiness),
        development_roots=roots,
        evaluation_config_path=cast(Path, args.evaluation_config),
        output_dir=cast(Path, args.output),
    )
    print(_canonical_bytes(summary).decode())


if __name__ == "__main__":
    main()
