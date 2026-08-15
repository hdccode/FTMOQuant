"""G1.4C Phase 2 DEVELOPMENT-only evaluator for ``ts_momentum_v1``.

This module is orchestration around the frozen Stage G boundary, the existing
G0.7 Nautilus execution adapters, and the existing ``arch`` wrappers. It has no
validation or holdout loader and admits exactly one frozen candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import pandas as pd  # type: ignore[import-untyped]
from nautilus_trader.backtest import BacktestEngine
from nautilus_trader.common import TimeEvent
from nautilus_trader.execution import ProbabilisticFillModel, StaticLatencyModel
from nautilus_trader.model import (
    Bar,
    BarType,
    Currency,
    CurrencyPair,
    InstrumentId,
    OrderFilled,
    OrderSide,
    Position,
    StrategyId,
)
from nautilus_trader.persistence import ParquetDataCatalog
from nautilus_trader.trading import Strategy, StrategyConfig

from ftmoquant.backtest.execution_harness import (
    AccountParameters,
    _add_venue,
    _engine_config,
    _fee_model,
    _instrument_for_profile,
    _profile_dict,
    _rollover_modules,
    _sha256_json,
    _sha256_tree,
    canonical_execution_profile,
)
from ftmoquant.data.dukascopy import NAUTILUS_VERSION
from ftmoquant.research.stage_g import (
    DEVELOPMENT_END_EXCLUSIVE,
    DEVELOPMENT_START,
    FROZEN_INSTRUMENT_IDS,
    CurrencyExposureLimits,
    DevelopmentFold,
    DevelopmentResearchContext,
    FxPosition,
    InstrumentBarObservation,
    MissingBarPolicy,
    PerInstrumentCostModel,
    ResearchPartition,
    SynchronizedClockFrame,
    aggregate_currency_exposures,
    frozen_development_folds,
    open_development_context,
    synchronize_universe_clock,
    validate_cost_models,
)
from ftmoquant.research.statistics import (
    SPAConfig,
    StationaryBootstrapConfig,
    result_as_dict,
    spa_reality_check,
    stationary_bootstrap_confidence_interval,
)
from ftmoquant.research.tournament_registry import (
    candidate_registry,
    preregistered_selection_contract,
)
from ftmoquant.research.ts_momentum_spec import (
    TS_MOMENTUM_CONFIG_SHA256,
    load_ts_momentum_spec,
    ts_momentum_config_sha256,
)
from ftmoquant.strategies.ts_momentum import (
    RawDirectionalTarget,
    TsMomentumDevelopmentFold,
    derive_daily_midpoint_closes,
)

EVALUATION_CONFIG_PATH = Path("config/research/ts_momentum_development_v1.json")
RESERVED_COST_MODELS_PATH = Path(".artifacts/g1_4c/phase2_cost_models.json")
EVALUATION_CONFIG_SHA256 = (
    "6764e2bc41cc92cecd050419bb003da82db2960c3f9feb909b75e0be139a2ba2"
)
EVALUATOR_VERSION = "g1.4c-ts-momentum-development-evaluator-1"
_SESSION_ZONE = ZoneInfo("America/New_York")
_MINUTE_NS = 60_000_000_000
_OBSERVATION_DELAY_NS = 1


class TsMomentumEvaluationError(ValueError):
    """Raised when the frozen DEVELOPMENT evaluation cannot be proven safe."""


@dataclass(frozen=True, slots=True)
class DevelopmentEvaluationConfig:
    semantic_sha256: str
    account: AccountParameters
    fixed_base_units: Decimal
    daily_return_denominator: Decimal
    exposure_limits: CurrencyExposureLimits
    cost_models: tuple[PerInstrumentCostModel, ...]
    canonical_document: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TargetInstruction:
    instrument_id: str
    target: RawDirectionalTarget
    event_time_ns: int
    information_time_ns: int
    midpoint: Decimal
    frame_midpoints: tuple[tuple[str, Decimal], ...]
    execution_price: Decimal | None = None
    exit_reason: str | None = None


@dataclass(frozen=True, slots=True)
class DailyEquity:
    information_time_ns: int
    equity: Decimal


@dataclass(frozen=True, slots=True)
class RealizedExecutionCost:
    information_time_ns: int
    instrument_id: str
    quantity: Decimal
    midpoint: Decimal
    fill_price: Decimal
    commission: Decimal
    cost_usd: Decimal


@dataclass(frozen=True, slots=True)
class FoldEngineResult:
    fold_id: str
    daily_equity: tuple[DailyEquity, ...]
    realized_costs: tuple[RealizedExecutionCost, ...]
    exposure_breach_count: int
    order_report: pd.DataFrame
    fill_report: pd.DataFrame
    position_report: pd.DataFrame


@dataclass(frozen=True, slots=True)
class PreparedDevelopmentMarketData:
    """One verified DEVELOPMENT load, sliced in-memory for immutable folds."""

    instruments: tuple[CurrencyPair, ...]
    bars: tuple[Bar, ...]
    frames: tuple[SynchronizedClockFrame, ...]
    source_query_count: int


def load_development_evaluation_config(
    path: Path = EVALUATION_CONFIG_PATH,
) -> DevelopmentEvaluationConfig:
    """Load the exact Phase 2 contract and derive costs from canonical G0.7."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TsMomentumEvaluationError(
            f"invalid evaluation config: {error}"
        ) from error
    if not isinstance(raw, dict):
        raise TsMomentumEvaluationError("evaluation config must be an object")
    document = cast(dict[str, Any], raw)
    semantic = document.get("semantic_sha256")
    payload = {
        key: value for key, value in document.items() if key != "semantic_sha256"
    }
    calculated = _semantic_sha256(payload)
    if semantic != calculated or semantic != EVALUATION_CONFIG_SHA256:
        raise TsMomentumEvaluationError("evaluation config semantic hash is not frozen")
    expected_keys = {
        "schema",
        "schema_version",
        "account",
        "cost_models",
        "fixed_research_exposure",
        "currency_exposure_limits",
        "semantic_sha256",
    }
    if set(document) != expected_keys:
        raise TsMomentumEvaluationError("evaluation config schema is not exact")
    account_raw = _mapping(document["account"], "account")
    exposure_raw = _mapping(
        document["fixed_research_exposure"], "fixed research exposure"
    )
    limits_raw = _mapping(document["currency_exposure_limits"], "exposure limits")
    limit_values = _mapping(limits_raw.get("max_abs_amount"), "limit amounts")
    required = {
        "schema": document["schema"],
        "schema_version": document["schema_version"],
        "account": account_raw,
        "exposure": exposure_raw,
        "limits_derivation": limits_raw.get("derivation"),
        "limit_currencies": tuple(limit_values),
    }
    if required != {
        "schema": "ftmoquant.ts-momentum-development-evaluation",
        "schema_version": 1,
        "account": {
            "currency": "USD",
            "initial_capital": "100000",
            "leverage": "30",
        },
        "exposure": {
            "base_units_per_nonzero_target": "100000",
            "derivation": "existing_g0_7_canonical_round_trip_probe_quantity",
            "portfolio_daily_return_denominator": "100000",
        },
        "limits_derivation": ("initial_capital_times_leverage_in_each_frozen_currency"),
        "limit_currencies": ("EUR", "GBP", "USD"),
    }:
        raise TsMomentumEvaluationError("evaluation config changed frozen semantics")
    account = AccountParameters(
        initial_capital=Decimal(account_raw["initial_capital"]),
        currency=cast(str, account_raw["currency"]),
        leverage=Decimal(account_raw["leverage"]),
    )
    leverage_limit = account.initial_capital * account.leverage
    limits = CurrencyExposureLimits(
        {
            currency: Decimal(cast(str, amount))
            for currency, amount in limit_values.items()
        }
    )
    if any(value != leverage_limit for value in limits.max_abs_amount.values()):
        raise TsMomentumEvaluationError("exposure limits are not canonically derived")
    costs_raw = document["cost_models"]
    if not isinstance(costs_raw, list):
        raise TsMomentumEvaluationError("cost_models must be ordered")
    profile = canonical_execution_profile()
    cost_models: list[PerInstrumentCostModel] = []
    for instrument_id, item_raw in zip(FROZEN_INSTRUMENT_IDS, costs_raw, strict=True):
        item = _mapping(item_raw, "cost model")
        if item != {
            "instrument_id": instrument_id,
            "execution_profile_source": "canonical_g0_7_uncalibrated_baseline",
            "execution_profile_sha256": (
                "5bfb0349fff97a08a8b5d97a6e79b39c694a9038730e1be1e5dcb3f66b3f29cc"
            ),
            "spread_source": "observed_paired_bid_ask",
        }:
            raise TsMomentumEvaluationError("cost model is not canonical G0.7")
        if _sha256_json(_profile_dict(profile)) != item["execution_profile_sha256"]:
            raise TsMomentumEvaluationError("canonical G0.7 profile hash drifted")
        cost_models.append(
            PerInstrumentCostModel(
                instrument_id=instrument_id,
                execution_profile=profile,
                spread_source=cast(str, item["spread_source"]),
            )
        )
    validate_cost_models(cost_models)
    return DevelopmentEvaluationConfig(
        semantic_sha256=cast(str, semantic),
        account=account,
        fixed_base_units=Decimal(
            cast(str, exposure_raw["base_units_per_nonzero_target"])
        ),
        daily_return_denominator=Decimal(
            cast(str, exposure_raw["portfolio_daily_return_denominator"])
        ),
        exposure_limits=limits,
        cost_models=tuple(cost_models),
        canonical_document=document,
    )


def materialize_canonical_cost_models(path: Path) -> Path:
    """Write the reserved artifact only as a byte copy of the tracked freeze."""

    if path.resolve() != RESERVED_COST_MODELS_PATH.resolve():
        raise TsMomentumEvaluationError(
            "missing cost artifact can be generated only at the reserved path"
        )
    if path.exists():
        load_development_evaluation_config(path)
        return path
    content = EVALUATION_CONFIG_PATH.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    load_development_evaluation_config(path)
    return path


def evaluate_ts_momentum_development(
    *,
    spec_path: Path,
    universe_readiness_path: Path,
    development_roots: Mapping[str, Path],
    cost_models_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run the frozen candidate on DEVELOPMENT and write deterministic artifacts."""

    spec = load_ts_momentum_spec(spec_path)
    if ts_momentum_config_sha256(spec) != TS_MOMENTUM_CONFIG_SHA256:
        raise TsMomentumEvaluationError("ts_momentum_v1 semantic SHA drifted")
    return evaluate_frozen_development_candidate(
        strategy_id=spec.strategy_id,
        strategy_config_sha256=TS_MOMENTUM_CONFIG_SHA256,
        spec_path=spec_path,
        universe_readiness_path=universe_readiness_path,
        development_roots=development_roots,
        cost_models_path=cost_models_path,
        output_dir=output_dir,
        target_instruction_builder=_build_target_instructions,
        evaluator_version=EVALUATOR_VERSION,
        result_schema="ftmoquant.ts-momentum-development-results",
        implementation_paths=(Path("src/ftmoquant/strategies/ts_momentum.py"),),
    )


TargetInstructionBuilder = Callable[
    [
        DevelopmentResearchContext,
        DevelopmentFold,
        tuple[SynchronizedClockFrame, ...],
        Path,
    ],
    tuple[TargetInstruction, ...],
]

PreparedMarketDataLoader = Callable[
    [DevelopmentResearchContext, Mapping[str, Path]], PreparedDevelopmentMarketData
]


def evaluate_frozen_development_candidate(
    *,
    strategy_id: str,
    strategy_config_sha256: str,
    spec_path: Path,
    universe_readiness_path: Path,
    development_roots: Mapping[str, Path],
    cost_models_path: Path,
    output_dir: Path,
    target_instruction_builder: TargetInstructionBuilder,
    evaluator_version: str,
    result_schema: str,
    implementation_paths: tuple[Path, ...],
    prepared_market_data_loader: PreparedMarketDataLoader | None = None,
    bar_interval_minutes: int = 1,
    instruction_timestamp: str = "event",
) -> dict[str, Any]:
    """Evaluate one frozen Stage G candidate through the common native boundary."""

    if output_dir.exists():
        raise FileExistsError(f"evaluation output already exists: {output_dir}")
    if not cost_models_path.exists():
        materialize_canonical_cost_models(cost_models_path)
    config = load_development_evaluation_config(cost_models_path)
    context = open_development_context(universe_readiness_path, development_roots)
    context.require_range(
        DEVELOPMENT_START,
        DEVELOPMENT_END_EXCLUSIVE,
        partition=ResearchPartition.DEVELOPMENT,
    )
    folds = frozen_development_folds()
    selection = preregistered_selection_contract()
    registry = candidate_registry()
    registry_entry = next(
        item for item in registry.ordered_entries if item.candidate_id == strategy_id
    )
    if registry_entry.implementation_status.value != "implemented_not_evaluated":
        raise TsMomentumEvaluationError(
            "candidate registry status is not Phase 2 ready"
        )

    _validate_catalog_trees(context, development_roots)

    prepared = (
        _load_development_market_data_once(context, development_roots)
        if prepared_market_data_loader is None
        else prepared_market_data_loader(context, development_roots)
    )
    logging.getLogger(__name__).info(
        "Stage G DEVELOPMENT source load complete: %d catalog queries, %d bars, "
        "%d frames",
        prepared.source_query_count,
        len(prepared.bars),
        len(prepared.frames),
    )
    fold_results: list[FoldEngineResult] = []
    fold_summaries: list[dict[str, Any]] = []
    pooled_rows: list[dict[str, Any]] = []
    for fold in folds.folds:
        instruments, bars, frames = _slice_prepared_market_data(prepared, fold)
        logging.getLogger(__name__).info("Evaluating DEVELOPMENT fold %s", fold.fold_id)
        instructions = target_instruction_builder(context, fold, frames, spec_path)
        result = _run_fold_engine(
            fold=fold,
            instruments=instruments,
            bars=bars,
            instructions=instructions,
            config=config,
            strategy_id=strategy_id,
            strategy_config_sha256=strategy_config_sha256,
            bar_interval_minutes=bar_interval_minutes,
            instruction_timestamp=instruction_timestamp,
        )
        rows = _daily_return_rows(result, fold, config.daily_return_denominator)
        summary = _fold_statistics(result, rows, config.daily_return_denominator)
        fold_results.append(result)
        fold_summaries.append(summary)
        pooled_rows.extend(rows)

    pooled = _pooled_statistics(pooled_rows, strategy_id=strategy_id)
    finite_sharpes = [
        cast(float, item["annualized_net_sharpe"])
        for item in fold_summaries
        if item["annualized_net_sharpe"] is not None
    ]
    ordered_sharpes = sorted(finite_sharpes)
    median_sharpe = (
        ordered_sharpes[len(ordered_sharpes) // 2]
        if len(ordered_sharpes) == len(fold_summaries)
        else None
    )
    tournament_statistics = {
        "median_development_fold_annualized_net_sharpe": median_sharpe,
        "worst_fold_annualized_net_sharpe": (
            min(finite_sharpes) if len(finite_sharpes) == len(fold_summaries) else None
        ),
        "positive_fold_count": sum(value > 0.0 for value in finite_sharpes),
        "pooled_development_mean_daily_net_return": pooled["mean_daily_net_return"],
    }
    provenance = {
        "evaluator_version": evaluator_version,
        "git_commit": _git_commit(),
        "nautilus_version": version("nautilus_trader"),
        "arch_version": version("arch"),
        "strategy_id": strategy_id,
        "strategy_config_semantic_sha256": strategy_config_sha256,
        "evaluation_config_semantic_sha256": config.semantic_sha256,
        "evaluation_profile": _profile_dict(config.cost_models[0].execution_profile),
        "evaluation_profile_sha256": _sha256_json(
            _profile_dict(config.cost_models[0].execution_profile)
        ),
        "universe_readiness_semantic_sha256": context.universe.readiness_sha256,
        "universe_plan_semantic_sha256": context.universe.plan.semantic_sha256,
        "development_artifacts": [
            {
                "instrument_id": item.instrument_id,
                "manifest_sha256": item.manifest_sha256,
                "catalog_tree_sha256": item.catalog_tree_sha256,
            }
            for item in context.artifacts
        ],
        "development_folds_semantic_sha256": folds.semantic_sha256,
        "candidate_registry_semantic_sha256": registry.semantic_sha256,
        "selection_contract_semantic_sha256": selection.semantic_sha256,
        "ordered_instruments": list(FROZEN_INSTRUMENT_IDS),
        "partition": "development",
        "validation_accessed": False,
        "final_holdout_accessed": False,
        "parameter_optimization_occurred": False,
        "candidate_family": [strategy_id],
        "implementation_file_sha256": {
            spec_path.as_posix(): _sha256_file(spec_path),
            cost_models_path.as_posix(): _sha256_file(cost_models_path),
            "src/ftmoquant/research/ts_momentum_development.py": _sha256_file(
                Path("src/ftmoquant/research/ts_momentum_development.py")
            ),
            **{path.as_posix(): _sha256_file(path) for path in implementation_paths},
        },
    }
    manifest: dict[str, Any] = {
        "schema": result_schema,
        "schema_version": 1,
        "provenance": provenance,
        "folds": fold_summaries,
        "pooled": pooled,
        "preregistered_tournament_statistics": tournament_statistics,
        "mcs": {
            "status": "not_applicable_single_candidate",
            "reason": "the adopted Stage G wrapper requires at least two models",
        },
        "hard_failures": [
            failure
            for summary in fold_summaries
            for failure in cast(list[str], summary["hard_failures"])
        ],
    }
    manifest["semantic_sha256"] = _semantic_sha256(manifest)
    _validate_catalog_trees(context, development_roots)
    _write_artifacts(output_dir, manifest, fold_results, pooled_rows)
    return manifest


def _load_fold_market_data(
    *,
    context: DevelopmentResearchContext,
    fold: DevelopmentFold,
    development_roots: Mapping[str, Path],
) -> tuple[
    tuple[CurrencyPair, ...], tuple[Bar, ...], tuple[SynchronizedClockFrame, ...]
]:
    prepared = _load_development_market_data_once(context, development_roots)
    return _slice_prepared_market_data(prepared, fold)


def _load_development_market_data_once(
    context: DevelopmentResearchContext,
    development_roots: Mapping[str, Path],
) -> PreparedDevelopmentMarketData:
    """Query every frozen DEVELOPMENT source stream once, never once per fold."""
    context.require_range(
        DEVELOPMENT_START,
        DEVELOPMENT_END_EXCLUSIVE,
        partition=ResearchPartition.DEVELOPMENT,
    )
    start_ns = _ns(DEVELOPMENT_START)
    end_ns = _ns(DEVELOPMENT_END_EXCLUSIVE)
    instruments: list[CurrencyPair] = []
    native_bars: list[Bar] = []
    streams: dict[str, tuple[InstrumentBarObservation, ...]] = {}
    source_query_count = 0
    for instrument_id in FROZEN_INSTRUMENT_IDS:
        catalog = ParquetDataCatalog(str(development_roots[instrument_id] / "catalog"))
        found = catalog.instruments([instrument_id])
        if len(found) != 1 or not isinstance(found[0], CurrencyPair):
            raise TsMomentumEvaluationError(
                f"frozen CurrencyPair missing from DEVELOPMENT: {instrument_id}"
            )
        profile = canonical_execution_profile()
        instruments.append(_instrument_for_profile(found[0], profile.fee))
        by_side: dict[str, dict[int, Bar]] = {}
        for side in ("BID", "ASK"):
            bar_type = f"{instrument_id}-1-MINUTE-{side}-EXTERNAL"
            queried = catalog.query_bars([bar_type], start=start_ns, end=end_ns - 1)
            source_query_count += 1
            selected = {
                bar.ts_event: bar
                for bar in queried
                if start_ns <= bar.ts_event
                and bar.ts_init < end_ns
                and str(bar.bar_type) == bar_type
            }
            if not selected:
                raise TsMomentumEvaluationError(f"DEVELOPMENT has no {bar_type} data")
            if len(selected) != len(queried):
                raise TsMomentumEvaluationError(
                    "DEVELOPMENT query admitted duplicate/out-of-range bars"
                )
            by_side[side] = selected
        observations: list[InstrumentBarObservation] = []
        for timestamp in sorted(set(by_side["BID"]) & set(by_side["ASK"])):
            bid = by_side["BID"][timestamp]
            ask = by_side["ASK"][timestamp]
            if bid.ts_init != ask.ts_init or bid.ts_init != timestamp + _MINUTE_NS:
                raise TsMomentumEvaluationError("paired one-minute timing drifted")
            observation = InstrumentBarObservation(
                instrument_id=instrument_id,
                timestamp_utc=_datetime(timestamp),
                available_at_utc=_datetime(bid.ts_init),
                bid=bid.close.as_decimal(),
                ask=ask.close.as_decimal(),
            )
            observation.validate()
            observations.append(observation)
            native_bars.extend((bid, ask))
        streams[instrument_id] = tuple(observations)
    frames = synchronize_universe_clock(
        streams, missing_policy=MissingBarPolicy.EMIT_NONTRADABLE_GAP
    )
    if not frames:
        raise TsMomentumEvaluationError("DEVELOPMENT has no clock frames")
    return PreparedDevelopmentMarketData(
        tuple(instruments), tuple(native_bars), frames, source_query_count
    )


def _slice_prepared_market_data(
    prepared: PreparedDevelopmentMarketData, fold: DevelopmentFold
) -> tuple[
    tuple[CurrencyPair, ...], tuple[Bar, ...], tuple[SynchronizedClockFrame, ...]
]:
    start_ns = _ns(fold.train_start_utc)
    end_ns = _ns(fold.compare_end_exclusive_utc)
    bars = tuple(bar for bar in prepared.bars if start_ns <= bar.ts_init < end_ns)
    frames = tuple(
        frame
        for frame in prepared.frames
        if fold.train_start_utc
        <= frame.available_at_utc
        < fold.compare_end_exclusive_utc
    )
    if not bars or not frames:
        raise TsMomentumEvaluationError(
            f"fold {fold.fold_id} has no prepared market data"
        )
    return prepared.instruments, bars, frames


def _build_target_instructions(
    context: DevelopmentResearchContext,
    fold: DevelopmentFold,
    frames: tuple[SynchronizedClockFrame, ...],
    spec_path: Path,
) -> tuple[TargetInstruction, ...]:
    spec = load_ts_momentum_spec(spec_path)
    candidate = TsMomentumDevelopmentFold(context, fold, spec)
    closes_by_information: dict[datetime, list[Any]] = {}
    for close in derive_daily_midpoint_closes(frames):
        closes_by_information.setdefault(close.information_time_utc, []).append(close)
    result: list[TargetInstruction] = []
    for frame in frames:
        for close in closes_by_information.get(frame.available_at_utc, []):
            candidate.on_daily_close(close, partition=ResearchPartition.DEVELOPMENT)
        executable = candidate.on_execution_frame(
            frame, partition=ResearchPartition.DEVELOPMENT
        )
        if not executable:
            continue
        result.extend(_target_instructions_at_frame(executable, frame))
    return tuple(result)


def _target_instructions_at_frame(
    executable: Sequence[Any], frame: SynchronizedClockFrame
) -> tuple[TargetInstruction, ...]:
    """Convert candidate raw targets through the shared native order boundary."""

    observations = {
        item.instrument_id: item for item in frame.observations if item is not None
    }
    if set(observations) != set(FROZEN_INSTRUMENT_IDS):
        raise TsMomentumEvaluationError("executable target lacks synchronized prices")
    return tuple(
        TargetInstruction(
            instrument_id=target.instrument_id,
            target=target.target,
            event_time_ns=_ns(target.execution_event_time_utc),
            information_time_ns=_ns(target.execution_information_time_utc),
            midpoint=(
                observations[target.instrument_id].bid
                + observations[target.instrument_id].ask
            )
            / Decimal(2),
            frame_midpoints=tuple(
                (
                    instrument_id,
                    (observations[instrument_id].bid + observations[instrument_id].ask)
                    / Decimal(2),
                )
                for instrument_id in FROZEN_INSTRUMENT_IDS
            ),
        )
        for target in executable
    )


class _NautilusTargetExecutor(Strategy):
    """Thin adapter from frozen raw targets to native market orders and reports."""

    def __new__(
        cls,
        instructions: Sequence[TargetInstruction],
        instruments: Sequence[CurrencyPair],
        config: DevelopmentEvaluationConfig,
        strategy_id: str,
        instrument_ids: Sequence[str],
        bar_interval_minutes: int,
        instruction_timestamp: str,
    ) -> _NautilusTargetExecutor:
        del (
            instructions,
            instruments,
            config,
            strategy_id,
            instrument_ids,
            bar_interval_minutes,
            instruction_timestamp,
        )
        return super().__new__(cls)

    def __init__(
        self,
        instructions: Sequence[TargetInstruction],
        instruments: Sequence[CurrencyPair],
        config: DevelopmentEvaluationConfig,
        strategy_id: str,
        instrument_ids: Sequence[str],
        bar_interval_minutes: int,
        instruction_timestamp: str,
    ) -> None:
        super().__init__(
            StrategyConfig(
                strategy_id=StrategyId(
                    f"{strategy_id.upper().replace('_', '-')}-DEVELOPMENT-001"
                ),
                log_events=False,
                log_commands=False,
            )
        )
        self._instructions = {
            (item.event_time_ns, item.instrument_id): item for item in instructions
        }
        if len(self._instructions) != len(instructions):
            raise TsMomentumEvaluationError("duplicate target instruction")
        self._instruments = {str(item.id): item for item in instruments}
        self._config = config
        self._strategy_id = strategy_id
        self._instrument_ids = tuple(instrument_ids)
        self._bar_interval_minutes = bar_interval_minutes
        self._instruction_timestamp = instruction_timestamp
        if not self._instrument_ids or instruction_timestamp not in {"event", "init"}:
            raise TsMomentumEvaluationError("native executor input is invalid")
        self._targets = {
            instrument_id: RawDirectionalTarget.FLAT
            for instrument_id in self._instrument_ids
        }
        self._closes: dict[int, dict[str, dict[str, Any]]] = {}
        self._scheduled: set[int] = set()
        self._order_context: dict[str, TargetInstruction] = {}
        self.daily_equity: list[DailyEquity] = []
        self.realized_costs: list[RealizedExecutionCost] = []
        self.exposure_breach_count = 0

    def on_start(self) -> None:
        for instrument_id in self._instrument_ids:
            for side in ("BID", "ASK"):
                self.subscribe_bars(
                    BarType.from_str(
                        f"{instrument_id}-{self._bar_interval_minutes}-MINUTE-{side}-EXTERNAL"
                    )
                )

    def on_stop(self) -> None:
        for instrument_id in self._instrument_ids:
            for side in ("BID", "ASK"):
                self.unsubscribe_bars(
                    BarType.from_str(f"{instrument_id}-1-MINUTE-{side}-EXTERNAL")
                )

    def on_bar(self, bar: Bar) -> None:
        instrument_id = str(bar.bar_type.instrument_id)
        side = bar.bar_type.spec.price_type.name
        if instrument_id not in self._instruments or side not in {"BID", "ASK"}:
            return
        for timestamp in tuple(self._closes):
            if timestamp < bar.ts_event and timestamp not in self._scheduled:
                del self._closes[timestamp]
        frame = self._closes.setdefault(bar.ts_event, {})
        frame.setdefault(instrument_id, {})[side] = bar.close
        timestamp = (
            bar.ts_event if self._instruction_timestamp == "event" else bar.ts_init
        )
        instruction = self._instructions.get((timestamp, instrument_id))
        if instruction is not None and side == "ASK":
            self._execute(instruction)
        if bar.ts_event not in self._scheduled and all(
            set(frame.get(item, {})) == {"BID", "ASK"} for item in self._instrument_ids
        ):
            local = _datetime(bar.ts_init).astimezone(_SESSION_ZONE)
            if local.weekday() <= 4 and local.hour == 17 and local.minute == 0:
                self._scheduled.add(bar.ts_event)
                self.clock.set_time_alert_ns(
                    name=f"ts-momentum-mark-{bar.ts_event}",
                    alert_time_ns=bar.ts_init + _OBSERVATION_DELAY_NS,
                    callback=self.on_time_event,
                )
            else:
                del self._closes[bar.ts_event]

    def on_time_event(self, event: TimeEvent) -> None:
        timestamp = (
            event.ts_event
            - _OBSERVATION_DELAY_NS
            - self._bar_interval_minutes * _MINUTE_NS
        )
        frame = self._closes.pop(timestamp, None)
        self._scheduled.discard(timestamp)
        if frame is None:
            raise RuntimeError("missing complete frame at valuation callback")
        information_time = timestamp + self._bar_interval_minutes * _MINUTE_NS
        local = _datetime(information_time).astimezone(_SESSION_ZONE)
        if local.weekday() > 4 or local.hour != 17 or local.minute != 0:
            return
        equity = self._native_liquidation_equity(frame)
        self.daily_equity.append(DailyEquity(information_time, equity))

    def on_order_filled(self, event: OrderFilled) -> None:
        instruction = self._order_context.pop(event.client_order_id.value, None)
        if instruction is None:
            raise RuntimeError("native fill lacks frozen target context")
        quantity = event.last_qty.as_decimal()
        fill_price = event.last_px.as_decimal()
        if event.order_side is OrderSide.BUY:
            spread_cost = quantity * (fill_price - instruction.midpoint)
        else:
            spread_cost = quantity * (instruction.midpoint - fill_price)
        commission = (
            Decimal(0)
            if event.commission is None
            else abs(event.commission.as_decimal())
        )
        cost = spread_cost + commission
        if cost < 0:
            raise RuntimeError("native execution improved beyond midpoint unexpectedly")
        self.realized_costs.append(
            RealizedExecutionCost(
                information_time_ns=instruction.information_time_ns,
                instrument_id=instruction.instrument_id,
                quantity=quantity,
                midpoint=instruction.midpoint,
                fill_price=fill_price,
                commission=commission,
                cost_usd=cost,
            )
        )

    def _execute(self, instruction: TargetInstruction) -> None:
        prior = self._targets[instruction.instrument_id]
        delta = int(instruction.target) - int(prior)
        if delta == 0:
            raise RuntimeError("unchanged target reached native execution")
        proposed = dict(self._targets)
        proposed[instruction.instrument_id] = instruction.target
        prices = dict(instruction.frame_midpoints)
        positions = tuple(
            FxPosition(
                instrument_id=instrument_id,
                signed_base_units=Decimal(int(target)) * self._config.fixed_base_units,
                price=prices[instrument_id],
            )
            for instrument_id, target in proposed.items()
            if target is not RawDirectionalTarget.FLAT
        )
        breaches = self._config.exposure_limits.breaches(
            aggregate_currency_exposures(positions)
        )
        if breaches:
            self.exposure_breach_count += len(breaches)
            raise RuntimeError("frozen currency-exposure limit breached")
        instrument = self._instruments[instruction.instrument_id]
        tags = [
            self._strategy_id,
            f"signal_execution_information_ns={instruction.information_time_ns}",
        ]
        if instruction.exit_reason is not None:
            tags.append(f"leo_exit_reason={instruction.exit_reason}")
        if instruction.execution_price is not None:
            tags.append(f"leo_intended_exit_price={instruction.execution_price}")
        instrument_id = InstrumentId.from_str(instruction.instrument_id)
        order_side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        quantity = instrument.make_qty(
            float(abs(delta) * self._config.fixed_base_units)
        )
        order = (
            self.order_factory.market(
                instrument_id=instrument_id,
                order_side=order_side,
                quantity=quantity,
                tags=tuple(tags),
            )
            if instruction.execution_price is None
            else self.order_factory.limit(
                instrument_id=instrument_id,
                order_side=order_side,
                quantity=quantity,
                price=instrument.make_price(float(instruction.execution_price)),
                tags=tuple(tags),
            )
        )
        self._order_context[order.client_order_id.value] = instruction
        self._targets[instruction.instrument_id] = instruction.target
        self.submit_order(order)

    def _native_liquidation_equity(
        self, frame: Mapping[str, Mapping[str, Any]]
    ) -> Decimal:
        venue = next(iter(self._instruments.values())).id.venue
        account = self.cache.account_for_venue(venue)
        if account is None:
            raise RuntimeError("native account is unavailable")
        currency = Currency.from_str(self._config.account.currency)
        balance = account.balance_total(currency)
        if balance is None:
            raise RuntimeError("native USD balance is unavailable")
        equity = balance.as_decimal()
        positions = cast(list[Position], self.cache.positions_open())
        for position in positions:
            instrument_id = str(position.instrument_id)
            marks = frame.get(instrument_id)
            if marks is None or set(marks) != {"BID", "ASK"}:
                raise RuntimeError("open position lacks synchronized liquidation mark")
            mark = marks["BID"] if position.is_long else marks["ASK"]
            pnl = position.unrealized_pnl(mark)
            if pnl.currency.code != self._config.account.currency:
                raise RuntimeError("native P/L is not normalized to account currency")
            equity += pnl.as_decimal()
        return cast(Decimal, equity)


def _run_fold_engine(
    *,
    fold: DevelopmentFold,
    instruments: tuple[CurrencyPair, ...],
    bars: tuple[Bar, ...],
    instructions: tuple[TargetInstruction, ...],
    config: DevelopmentEvaluationConfig,
    strategy_id: str = "ts_momentum_v1",
    strategy_config_sha256: str = TS_MOMENTUM_CONFIG_SHA256,
    bar_interval_minutes: int = 1,
    instruction_timestamp: str = "event",
) -> FoldEngineResult:
    if version("nautilus_trader") != NAUTILUS_VERSION:
        raise RuntimeError(
            f"expected NautilusTrader {NAUTILUS_VERSION}, found "
            f"{version('nautilus_trader')}"
        )
    profiles = tuple(item.execution_profile for item in config.cost_models)
    if any(_profile_dict(item) != _profile_dict(profiles[0]) for item in profiles):
        raise TsMomentumEvaluationError(
            "one native venue cannot admit differing per-instrument profiles"
        )
    profile = profiles[0]
    identity = _semantic_sha256(
        {
            "fold_id": fold.fold_id,
            "strategy_sha256": strategy_config_sha256,
            "evaluation_config_sha256": config.semantic_sha256,
        }
    )
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
    engine = BacktestEngine(_engine_config(identity))
    executor = _NautilusTargetExecutor(
        instructions,
        instruments,
        config,
        strategy_id,
        tuple(str(item.id) for item in instruments),
        bar_interval_minutes,
        instruction_timestamp,
    )
    try:
        _add_venue(
            engine,
            config.account,
            profile,
            fill_model,
            latency_model,
            _fee_model(profile.fee),
            _rollover_modules(profile.rollover),
        )
        for instrument in instruments:
            engine.add_instrument(instrument)
        engine.add_data(bars, validate=True, sort=True)
        engine.add_strategy(executor)
        engine.run(
            start=_ns(fold.train_start_utc),
            end=max(bar.ts_init for bar in bars) + _OBSERVATION_DELAY_NS,
            run_config_id=f"G14-{identity[:20]}",
        )
        return FoldEngineResult(
            fold_id=fold.fold_id,
            daily_equity=tuple(executor.daily_equity),
            realized_costs=tuple(executor.realized_costs),
            exposure_breach_count=executor.exposure_breach_count,
            order_report=cast(pd.DataFrame, engine.generate_orders_report()),
            fill_report=cast(pd.DataFrame, engine.generate_fills_report()),
            position_report=cast(pd.DataFrame, engine.generate_positions_report()),
        )
    finally:
        engine.dispose()


def _daily_return_rows(
    result: FoldEngineResult,
    fold: DevelopmentFold,
    denominator: Decimal,
) -> list[dict[str, Any]]:
    if denominator <= 0:
        raise TsMomentumEvaluationError("daily return denominator must be positive")
    costs_by_day: dict[str, Decimal] = {}
    for cost_item in result.realized_costs:
        day = (
            _datetime(cost_item.information_time_ns)
            .astimezone(_SESSION_ZONE)
            .date()
            .isoformat()
        )
        costs_by_day[day] = costs_by_day.get(day, Decimal(0)) + cost_item.cost_usd
    rows: list[dict[str, Any]] = []
    previous: DailyEquity | None = None
    for equity_item in result.daily_equity:
        information = _datetime(equity_item.information_time_ns)
        if (
            previous is not None
            and fold.compare_start_utc <= information < fold.compare_end_exclusive_utc
        ):
            day = information.astimezone(_SESSION_ZONE).date().isoformat()
            net_return = (equity_item.equity - previous.equity) / denominator
            realized_cost = costs_by_day.get(day, Decimal(0)) / denominator
            rows.append(
                {
                    "fold_id": fold.fold_id,
                    "session_date": day,
                    "information_time_utc": information.isoformat().replace(
                        "+00:00", "Z"
                    ),
                    "net_return": float(net_return),
                    "realized_cost_return": float(realized_cost),
                    "cost_stress_1_5x_return": float(net_return - realized_cost / 2),
                }
            )
        previous = equity_item
    if not rows:
        raise TsMomentumEvaluationError(f"fold {fold.fold_id} has no daily returns")
    return rows


def _fold_statistics(
    result: FoldEngineResult,
    rows: Sequence[Mapping[str, Any]],
    denominator: Decimal,
) -> dict[str, Any]:
    values = [float(item["net_return"]) for item in rows]
    stressed = [float(item["cost_stress_1_5x_return"]) for item in rows]
    turnover = sum(
        float(item.quantity * item.midpoint / denominator)
        for item in result.realized_costs
    )
    sharpe = _annualized_sharpe(values)
    hard_failures: list[str] = []
    if sharpe is None:
        hard_failures.append(f"{result.fold_id}:non_finite_sharpe")
    if result.exposure_breach_count:
        hard_failures.append(f"{result.fold_id}:currency_exposure_limit_breach")
    return {
        "fold_id": result.fold_id,
        "observation_count": len(values),
        "annualized_net_sharpe": sharpe,
        "mean_daily_net_return": sum(values) / len(values),
        "maximum_drawdown": _maximum_drawdown(values),
        "turnover": turnover,
        "mean_cost_stress_1_5x_return": sum(stressed) / len(stressed),
        "positive_cost_stress_1_5x_mean": sum(stressed) / len(stressed) > 0.0,
        "currency_exposure_limit_breach_count": result.exposure_breach_count,
        "hard_failures": hard_failures,
    }


def _pooled_statistics(
    rows: Sequence[Mapping[str, Any]], *, strategy_id: str
) -> dict[str, Any]:
    index = pd.Index(
        [f"{item['fold_id']}:{item['session_date']}" for item in rows],
        name="fold_session",
    )
    returns = pd.Series(
        [float(item["net_return"]) for item in rows],
        index=index,
        name=f"{strategy_id}_daily_net_return",
    )
    if len(returns) < 20:
        raise TsMomentumEvaluationError(
            "pooled DEVELOPMENT series is shorter than frozen block size 20"
        )
    bootstrap = stationary_bootstrap_confidence_interval(
        returns,
        StationaryBootstrapConfig(
            block_size=20,
            repetitions=10_000,
            seed=14_042_026,
            confidence_level=0.95,
            method="basic",
        ),
    )
    benchmark = pd.Series(0.0, index=index, name="zero_net_return_benchmark_loss")
    model_losses = pd.DataFrame({strategy_id: -returns.to_numpy()}, index=index)
    spa = spa_reality_check(
        benchmark,
        model_losses,
        SPAConfig(
            procedure="SPA",
            block_size=20,
            repetitions=10_000,
            seed=14_042_026,
            bootstrap="stationary",
            studentize=True,
            nested=False,
            significance_level=0.05,
            pvalue_type="consistent",
        ),
    )
    return {
        "observation_count": len(returns),
        "mean_daily_net_return": float(returns.mean()),
        "stationary_bootstrap_mean_confidence_interval": result_as_dict(bootstrap),
        "spa_zero_return_comparison": result_as_dict(spa),
    }


def _annualized_sharpe(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    if variance <= 0.0 or not math.isfinite(variance):
        return None
    result = math.sqrt(252.0) * mean / math.sqrt(variance)
    return result if math.isfinite(result) else None


def _maximum_drawdown(values: Sequence[float]) -> float:
    wealth = 1.0
    peak = 1.0
    drawdown = 0.0
    for value in values:
        wealth *= 1.0 + value
        peak = max(peak, wealth)
        drawdown = max(drawdown, (peak - wealth) / peak)
    return drawdown


def _write_artifacts(
    output_dir: Path,
    manifest: dict[str, Any],
    fold_results: Sequence[FoldEngineResult],
    pooled_rows: Sequence[Mapping[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(pooled_rows).to_csv(
        output_dir / "daily_returns.csv", index=False, lineterminator="\n"
    )
    for result in fold_results:
        fold_dir = output_dir / result.fold_id
        fold_dir.mkdir()
        for name, report in (
            ("orders.csv", result.order_report),
            ("fills.csv", result.fill_report),
            ("positions.csv", result.position_report),
        ):
            report.sort_index().to_csv(fold_dir / name, lineterminator="\n")
    result_hashes = {
        str(path.relative_to(output_dir)): _sha256_file(path)
        for path in sorted(output_dir.rglob("*.csv"))
    }
    manifest["result_file_sha256"] = result_hashes
    manifest["semantic_sha256"] = _semantic_sha256(
        {key: value for key, value in manifest.items() if key != "semantic_sha256"}
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _validate_catalog_trees(
    context: DevelopmentResearchContext,
    development_roots: Mapping[str, Path],
) -> None:
    statuses = {item.instrument_id: item for item in context.artifacts}
    for instrument_id in FROZEN_INSTRUMENT_IDS:
        actual = _sha256_tree(development_roots[instrument_id] / "catalog")
        if actual != statuses[instrument_id].catalog_tree_sha256:
            raise TsMomentumEvaluationError(
                f"DEVELOPMENT catalog tree hash drifted: {instrument_id}"
            )


def _mapping(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TsMomentumEvaluationError(f"{description} must be an object")
    return cast(dict[str, Any], value)


def _semantic_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _ns(value: datetime) -> int:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise TsMomentumEvaluationError("timestamp must be timezone-aware UTC")
    return int(value.timestamp() * 1_000_000_000)


def _datetime(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1_000_000_000, tz=UTC)


def _development_root(value: str) -> tuple[str, Path]:
    instrument_id, separator, raw_path = value.partition("=")
    if not separator or instrument_id not in FROZEN_INSTRUMENT_IDS or not raw_path:
        raise argparse.ArgumentTypeError(
            "development root must be FROZEN_INSTRUMENT_ID=PATH"
        )
    lowered = raw_path.lower()
    if "validation" in lowered or "holdout" in lowered:
        raise argparse.ArgumentTypeError("validation and holdout paths are forbidden")
    return instrument_id, Path(raw_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen ts_momentum_v1 on Stage G DEVELOPMENT only"
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--universe-readiness", type=Path, required=True)
    parser.add_argument(
        "--development-root", type=_development_root, action="append", required=True
    )
    parser.add_argument("--cost-models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    roots = dict(cast(list[tuple[str, Path]], args.development_root))
    if len(roots) != len(cast(list[tuple[str, Path]], args.development_root)):
        raise TsMomentumEvaluationError("duplicate development root instrument")
    evaluate_ts_momentum_development(
        spec_path=cast(Path, args.spec),
        universe_readiness_path=cast(Path, args.universe_readiness),
        development_roots=roots,
        cost_models_path=cast(Path, args.cost_models),
        output_dir=cast(Path, args.output),
    )


if __name__ == "__main__":
    main()
