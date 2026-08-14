"""Canonical G0.7 Nautilus EUR/USD execution/backtest harness.

The only strategy in this module is a deterministic execution probe.  It is
deliberately scripted and must not be interpreted as alpha.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal
from enum import StrEnum
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]
from nautilus_trader.backtest import (
    BacktestDataConfig,
    BacktestEngine,
    BacktestEngineConfig,
    BacktestRunConfig,
    BacktestVenueConfig,
    FXRolloverInterestModule,
    InterestRateRecord,
)
from nautilus_trader.common import LoggerConfig, LogLevel
from nautilus_trader.core import UUID4
from nautilus_trader.execution import (
    FixedFeeModel,
    MakerTakerFeeModel,
    PerContractFeeModel,
    ProbabilisticFillModel,
    StaticLatencyModel,
)
from nautilus_trader.model import (
    AccountType,
    Bar,
    BarType,
    BookType,
    Currency,
    CurrencyPair,
    InstrumentId,
    MarginAccount,
    Money,
    OmsType,
    OrderFilled,
    OrderSide,
    OtoTriggerMode,
    PositionOpened,
    Price,
    Quantity,
    StrategyId,
    Venue,
)
from nautilus_trader.persistence import ParquetDataCatalog
from nautilus_trader.trading import Strategy, StrategyConfig

from ftmoquant.data.derived_bars import DERIVED_MANIFEST_FILENAME
from ftmoquant.data.dukascopy import INSTRUMENT_ID, NAUTILUS_VERSION
from ftmoquant.prop_rules import EvaluationPhase, PropRuleSet, load_prop_rule_set
from ftmoquant.risk import (
    FTMO_OBSERVATION_DELAY_NS,
    FTMO_OBSERVATION_VERSION,
    FTMO_VALUATION_VERSION,
    FtmoObservation,
    FtmoRuntimeConfig,
    NativeAccountSnapshot,
    NautilusAccountSnapshotSource,
    NautilusFtmoBridge,
    NautilusFtmoOverlay,
    PairedBarLiquidationSnapshotSource,
)

EXECUTION_CONFIG_VERSION = 1
EXECUTION_PROFILE_VERSION = "g0.7-1"
RUN_MANIFEST_FILENAME = "ftmoquant_execution_run.json"
G0_5_MANIFEST_FILENAME = "ftmoquant_provenance.json"

_VENUE = Venue("DUKASCOPY")
_INSTRUMENT = InstrumentId.from_str(INSTRUMENT_ID)
_MINUTE_NS = 60_000_000_000
_SOURCE_BAR_TYPES = (
    f"{INSTRUMENT_ID}-1-MINUTE-BID-EXTERNAL",
    f"{INSTRUMENT_ID}-1-MINUTE-ASK-EXTERNAL",
)
_REPORT_FILENAMES = {
    "account": "account.csv",
    "orders": "orders.csv",
    "order_fills": "order_fills.csv",
    "fills": "fills.csv",
    "positions": "positions.csv",
}


class ExecutionValidationError(ValueError):
    """Raised before a run when execution inputs are unsafe or ambiguous."""


class CalibrationStatus(StrEnum):
    """Evidence level for an execution profile."""

    UNCALIBRATED = "uncalibrated"
    PROXY = "proxy"
    BROKER_CALIBRATED = "broker_calibrated"


class FeeModelKind(StrEnum):
    """Supported native Nautilus fee models."""

    FIXED = "fixed"
    PER_CONTRACT = "per_contract"
    MAKER_TAKER = "maker_taker"


class RolloverMode(StrEnum):
    """Supported native rollover modes."""

    DISABLED = "disabled"
    FX_INTEREST = "fx_rollover_interest"


class RunMode(StrEnum):
    """Research is provenance-gated; test/probe mode accepts fixtures."""

    TEST = "test"
    RESEARCH = "research"


class ProbeOrderKind(StrEnum):
    """Scripted probe actions, never trading signals."""

    MARKET = "market"
    LIMIT = "limit"
    DUAL_LIMIT = "dual_limit"


class ProbeSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True, slots=True)
class InterestRateInput:
    """One explicit input for Nautilus ``InterestRateRecord``."""

    location: str
    time: str
    value: Decimal


@dataclass(frozen=True, slots=True)
class FeeModelConfig:
    """Explicit configuration for one built-in Nautilus fee model."""

    kind: FeeModelKind
    commission: Decimal | None = None
    currency: str | None = None
    charge_commission_once: bool | None = None
    maker_rate: Decimal | None = None
    taker_rate: Decimal | None = None


@dataclass(frozen=True, slots=True)
class RolloverConfig:
    """Explicit native FX rollover configuration."""

    mode: RolloverMode
    records: tuple[InterestRateInput, ...] = ()


@dataclass(frozen=True, slots=True)
class ExecutionProfile:
    """Complete, hashable execution assumptions for one G0.7 run."""

    fill_on_limit_probability: Decimal
    adverse_slippage_probability: Decimal
    random_seed: int
    base_latency_ns: int
    insert_latency_ns: int
    update_latency_ns: int
    cancel_latency_ns: int
    fee: FeeModelConfig
    rollover: RolloverConfig
    adaptive_high_low_ordering: bool
    calibration_status: CalibrationStatus
    broker_evidence_sha256: str | None = None
    version: str = EXECUTION_PROFILE_VERSION


@dataclass(frozen=True, slots=True)
class AccountParameters:
    """Runtime account parameters; no FTMO values are assumed."""

    initial_capital: Decimal
    currency: str
    leverage: Decimal


@dataclass(frozen=True, slots=True)
class ProbePlan:
    """Predetermined orders for execution integration validation only."""

    order_kind: ProbeOrderKind
    side: ProbeSide
    quantity: Decimal
    entry_bar_index: int
    exit_bar_index: int | None = None
    limit_price: Decimal | None = None
    second_limit_price: Decimal | None = None


@dataclass(frozen=True, slots=True)
class ExecutionRunRequest:
    """All mutable inputs to one canonical run."""

    dataset_root: Path
    output_dir: Path
    prop_rule_config_path: Path
    profile: ExecutionProfile
    account: AccountParameters
    probe: ProbePlan
    requested_start_ns: int
    requested_end_ns: int
    ftmo_phase: EvaluationPhase
    mode: RunMode = RunMode.TEST


@dataclass(frozen=True, slots=True)
class ExecutionRunResult:
    """Stable result surface backed by native Nautilus reports/state."""

    manifest_path: Path
    report_paths: Mapping[str, Path]
    run_config_id: str
    profile_sha256: str
    result_summary: Mapping[str, Any]
    ftmo_evaluation: Mapping[str, Any]
    ftmo_observations: tuple[FtmoObservation, ...]


class _ExecutionProbe(Strategy):
    """A clocked order script used only to verify execution plumbing."""

    def __init__(self) -> None:
        super().__init__(
            StrategyConfig(
                strategy_id=StrategyId("G07-PROBE-001"),
                log_events=False,
                log_commands=False,
            )
        )
        self._plan: ProbePlan | None = None
        self._instrument: CurrencyPair | None = None
        self._ask_bar_index = -1
        self._ftmo_runtime: FtmoRuntimeConfig | None = None
        self._ftmo_rules: PropRuleSet | None = None
        self._ftmo_bridge: NautilusFtmoBridge | None = None

    def configure(
        self,
        plan: ProbePlan,
        instrument: CurrencyPair,
        ftmo_runtime: FtmoRuntimeConfig,
        ftmo_rules: PropRuleSet,
    ) -> None:
        """Supply the already-validated immutable probe configuration."""

        self._plan = plan
        self._instrument = instrument
        self._ftmo_runtime = ftmo_runtime
        self._ftmo_rules = ftmo_rules

    @property
    def ftmo_bridge(self) -> NautilusFtmoBridge:
        """Return the initialized reusable FTMO bridge."""

        if self._ftmo_bridge is None:
            raise RuntimeError("FTMO bridge is not initialized")
        return self._ftmo_bridge

    def on_start(self) -> None:
        if self._ftmo_runtime is None or self._ftmo_rules is None:
            raise RuntimeError("execution probe FTMO configuration is missing")
        native_source = NautilusAccountSnapshotSource(
            cache=self.cache,
            portfolio=self.portfolio,
            venue=_VENUE,
            currency=Currency.from_str(self._ftmo_runtime.currency),
        )
        source = PairedBarLiquidationSnapshotSource(
            cache=self.cache,
            native_source=native_source,
            instrument_id=_INSTRUMENT,
            account_currency=Currency.from_str(self._ftmo_runtime.currency),
        )
        overlay = NautilusFtmoOverlay(
            runtime=self._ftmo_runtime,
            rules=self._ftmo_rules,
            clock=self.clock,
            account_source=source,
        )
        self._ftmo_bridge = NautilusFtmoBridge(
            clock=self.clock,
            overlay=overlay,
            account_source=source,
        )
        self.subscribe_bars(BarType.from_str(_SOURCE_BAR_TYPES[0]))
        self.subscribe_bars(BarType.from_str(_SOURCE_BAR_TYPES[1]))

    def on_bar(self, bar: Bar) -> None:
        if self._plan is None:
            raise RuntimeError("execution probe is not configured")
        self.ftmo_bridge.on_bar(bar)
        if str(bar.bar_type) != _SOURCE_BAR_TYPES[1]:
            return
        self._ask_bar_index += 1
        if self._ask_bar_index == self._plan.entry_bar_index:
            self._submit_entry()
        if self._ask_bar_index == self._plan.exit_bar_index:
            self.close_all_positions(_INSTRUMENT)

    def on_order_filled(self, event: OrderFilled) -> None:
        self.ftmo_bridge.on_order_filled(event)

    def on_position_opened(self, event: PositionOpened) -> None:
        self.ftmo_bridge.on_position_opened(event)

    def _submit_entry(self) -> None:
        if self._plan is None or self._instrument is None:
            raise RuntimeError("execution probe is not configured")
        side = (
            OrderSide.BUY if self._plan.side is ProbeSide.BUY else OrderSide.SELL
        )
        quantity = _quantity(self._plan.quantity, self._instrument.size_precision)
        if self._plan.order_kind is ProbeOrderKind.MARKET:
            self.submit_order(
                self.order_factory.market(
                    instrument_id=_INSTRUMENT,
                    order_side=side,
                    quantity=quantity,
                )
            )
            return

        if self._plan.limit_price is None:
            raise RuntimeError("validated limit probe is missing its price")
        self.submit_order(
            self.order_factory.limit(
                instrument_id=_INSTRUMENT,
                order_side=side,
                quantity=quantity,
                price=_price(
                    self._plan.limit_price,
                    self._instrument.price_precision,
                ),
            )
        )
        if self._plan.order_kind is ProbeOrderKind.DUAL_LIMIT:
            if self._plan.second_limit_price is None:
                raise RuntimeError("validated dual-limit probe is missing its price")
            opposite = OrderSide.SELL if side is OrderSide.BUY else OrderSide.BUY
            self.submit_order(
                self.order_factory.limit(
                    instrument_id=_INSTRUMENT,
                    order_side=opposite,
                    quantity=quantity,
                    price=_price(
                        self._plan.second_limit_price,
                        self._instrument.price_precision,
                    ),
                )
            )


def run_eurusd_execution(request: ExecutionRunRequest) -> ExecutionRunResult:
    """Run the deterministic native Nautilus execution harness once.

    The output directory is append-never: an existing path is rejected rather
    than overwritten.  The G0.5 catalog is only read and its content hash is
    checked again after the run.
    """

    _validate_runtime(request)
    root = request.dataset_root.resolve()
    output_dir = request.output_dir.resolve()
    catalog_path = root / "catalog"
    g0_5_path = root / G0_5_MANIFEST_FILENAME
    g0_6_path = root / DERIVED_MANIFEST_FILENAME

    g0_5 = _load_json_object(g0_5_path, "G0.5 manifest")
    _validate_g0_5_manifest(g0_5)
    g0_5_sha256 = _sha256_file(g0_5_path)
    g0_6_sha256 = _validate_research_gate(
        request.mode, g0_6_path, g0_5_sha256
    )
    ftmo_rules = load_prop_rule_set(request.prop_rule_config_path)
    prop_rule_version = ftmo_rules.version
    git_commit = _git_commit()

    catalog = ParquetDataCatalog(str(catalog_path))
    instrument = _load_instrument(catalog)
    bid, ask = _load_execution_bars(catalog, g0_5)
    selected_bid, selected_ask = _select_range(
        bid,
        ask,
        request.requested_start_ns,
        request.requested_end_ns,
    )
    _validate_probe(request.probe, len(selected_ask), instrument)
    catalog_sha256_before = _sha256_tree(catalog_path)

    profile_dict = _profile_dict(request.profile)
    profile_sha256 = _sha256_json(profile_dict)
    run_identity = _run_identity(
        request,
        profile_sha256,
        g0_5_sha256,
        g0_6_sha256,
        selected_bid,
        selected_ask,
    )
    run_config_id = f"G07-{run_identity[:20]}"
    effective_instrument = _instrument_for_profile(instrument, request.profile.fee)
    fill_model = ProbabilisticFillModel(
        prob_fill_on_limit=float(request.profile.fill_on_limit_probability),
        prob_slippage=float(request.profile.adverse_slippage_probability),
        random_seed=request.profile.random_seed,
    )
    latency_model = StaticLatencyModel(
        base_latency_nanos=request.profile.base_latency_ns,
        insert_latency_nanos=request.profile.insert_latency_ns,
        update_latency_nanos=request.profile.update_latency_ns,
        cancel_latency_nanos=request.profile.cancel_latency_ns,
    )
    fee_model = _fee_model(request.profile.fee)
    modules = _rollover_modules(request.profile.rollover)
    engine_config = _engine_config(run_identity)
    venue_config = _venue_config(
        request.account,
        request.profile,
        fill_model,
        latency_model,
        fee_model,
        modules,
    )
    data_config = BacktestDataConfig(
        data_type="Bar",
        catalog_path=str(catalog_path),
        instrument_id=_INSTRUMENT,
        start_time=request.requested_start_ns,
        end_time=request.requested_end_ns,
        bar_types=list(_SOURCE_BAR_TYPES),
        optimize_file_loading=True,
    )
    run_config = BacktestRunConfig(
        venues=[venue_config],
        data=[data_config],
        engine=engine_config,
        id=run_config_id,
        raise_exception=True,
        dispose_on_completion=False,
        start=request.requested_start_ns,
        end=request.requested_end_ns + FTMO_OBSERVATION_DELAY_NS,
    )
    if run_config.id != run_config_id:
        raise RuntimeError("Nautilus changed the deterministic run-config ID")

    engine = BacktestEngine(engine_config)
    try:
        _add_venue(
            engine,
            request.account,
            request.profile,
            fill_model,
            latency_model,
            fee_model,
            modules,
        )
        engine.add_instrument(effective_instrument)
        engine.add_data((*selected_bid, *selected_ask), validate=True, sort=True)
        probe = _ExecutionProbe()
        probe.configure(
            request.probe,
            effective_instrument,
            FtmoRuntimeConfig(
                initial_capital=request.account.initial_capital,
                currency=request.account.currency,
                active_phase=request.ftmo_phase,
            ),
            ftmo_rules,
        )
        engine.add_strategy(probe)
        engine.run(
            start=request.requested_start_ns,
            end=request.requested_end_ns + FTMO_OBSERVATION_DELAY_NS,
            run_config_id=run_config_id,
        )
        ftmo_bridge = probe.ftmo_bridge
        native_snapshot = ftmo_bridge.snapshot()
        reports = _native_reports(engine)
        result_summary = _result_summary(
            engine,
            reports,
            request.account,
            native_snapshot,
        )
        ftmo_observations = ftmo_bridge.observations
        ftmo_evaluation = _ftmo_evaluation_dict(
            request.ftmo_phase,
            ftmo_bridge,
            native_snapshot,
        )
    finally:
        engine.dispose()

    catalog_sha256_after = _sha256_tree(catalog_path)
    if catalog_sha256_after != catalog_sha256_before:
        raise RuntimeError("source catalog changed during the execution run")

    output_dir.mkdir(parents=True, exist_ok=False)
    report_paths, report_hashes = _write_reports(output_dir, reports)

    manifest = {
        "schema": "ftmoquant.execution-run-provenance",
        "schema_version": EXECUTION_CONFIG_VERSION,
        "execution_profile_version": request.profile.version,
        "git_commit": git_commit,
        "nautilus_version": NAUTILUS_VERSION,
        "ftmo_rule_config_version": prop_rule_version,
        "ftmo_evaluation": ftmo_evaluation,
        "g0_5_manifest": G0_5_MANIFEST_FILENAME,
        "g0_5_manifest_sha256": g0_5_sha256,
        "g0_6_manifest": (
            DERIVED_MANIFEST_FILENAME if g0_6_sha256 is not None else None
        ),
        "g0_6_manifest_sha256": g0_6_sha256,
        "run_mode": request.mode.value,
        "research_readiness_required": request.mode is RunMode.RESEARCH,
        "execution_profile": profile_dict,
        "execution_profile_sha256": profile_sha256,
        "calibration_status": request.profile.calibration_status.value,
        "account_parameters": _account_dict(request.account),
        "random_seed": request.profile.random_seed,
        "requested_start_ns": request.requested_start_ns,
        "requested_end_ns": request.requested_end_ns,
        "source_bar_types": list(_SOURCE_BAR_TYPES),
        "source_bar_counts": {
            "BID": len(selected_bid),
            "ASK": len(selected_ask),
        },
        "source_catalog_sha256": catalog_sha256_after,
        "run_config_id": run_config_id,
        "run_config": _run_config_dict(request, run_config_id),
        "probe": _jsonable(asdict(request.probe)),
        "report_hashes": report_hashes,
        "result_summary": result_summary,
    }
    manifest_path = output_dir / RUN_MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path.chmod(0o444)
    return ExecutionRunResult(
        manifest_path=manifest_path,
        report_paths=report_paths,
        run_config_id=run_config_id,
        profile_sha256=profile_sha256,
        result_summary=result_summary,
        ftmo_evaluation=ftmo_evaluation,
        ftmo_observations=ftmo_observations,
    )


def _validate_runtime(request: ExecutionRunRequest) -> None:
    if version("nautilus-trader") != NAUTILUS_VERSION:
        raise RuntimeError(f"expected NautilusTrader {NAUTILUS_VERSION}")
    if request.output_dir.exists():
        raise FileExistsError(
            f"immutable execution output already exists: {request.output_dir}"
        )
    if not request.dataset_root.is_dir():
        raise ExecutionValidationError(
            f"dataset root does not exist: {request.dataset_root}"
        )
    if not (request.dataset_root / "catalog").is_dir():
        raise ExecutionValidationError("missing G0.5 catalog")
    output = request.output_dir.resolve()
    for protected in (
        request.dataset_root.resolve() / "catalog",
        request.dataset_root.resolve() / "source",
    ):
        if output == protected or protected in output.parents:
            raise ExecutionValidationError(
                "execution output cannot be inside G0.5 source data or catalog"
            )
    if not request.prop_rule_config_path.is_file():
        raise ExecutionValidationError("missing FTMO rule config")
    if request.requested_start_ns >= request.requested_end_ns:
        raise ExecutionValidationError("requested start must be before end")
    if not isinstance(request.ftmo_phase, EvaluationPhase):
        raise ExecutionValidationError("ftmo_phase must be explicit")
    _validate_account(request.account)
    _validate_profile(request.profile)


def _validate_profile(profile: ExecutionProfile) -> None:
    if profile.version != EXECUTION_PROFILE_VERSION:
        raise ExecutionValidationError(
            f"execution profile version must be {EXECUTION_PROFILE_VERSION}"
        )
    for name, probability in (
        ("fill_on_limit_probability", profile.fill_on_limit_probability),
        ("adverse_slippage_probability", profile.adverse_slippage_probability),
    ):
        if not probability.is_finite() or not Decimal(0) <= probability <= Decimal(1):
            raise ExecutionValidationError(f"{name} must be between 0 and 1")
    if isinstance(profile.random_seed, bool) or not isinstance(
        profile.random_seed, int
    ):
        raise ExecutionValidationError("random_seed must be an explicit integer")
    for name, latency in (
        ("base_latency_ns", profile.base_latency_ns),
        ("insert_latency_ns", profile.insert_latency_ns),
        ("update_latency_ns", profile.update_latency_ns),
        ("cancel_latency_ns", profile.cancel_latency_ns),
    ):
        if isinstance(latency, bool) or not isinstance(latency, int) or latency < 0:
            raise ExecutionValidationError(f"{name} must be a non-negative integer")
    if profile.calibration_status is CalibrationStatus.BROKER_CALIBRATED:
        evidence = profile.broker_evidence_sha256
        if evidence is None or len(evidence) != 64 or not _is_hex(evidence):
            raise ExecutionValidationError(
                "broker_calibrated requires external broker evidence SHA-256"
            )
    elif profile.broker_evidence_sha256 is not None:
        raise ExecutionValidationError(
            "broker evidence is only valid for broker_calibrated profiles"
        )
    _validate_fee(profile.fee)
    if profile.rollover.mode is RolloverMode.DISABLED:
        if profile.rollover.records:
            raise ExecutionValidationError(
                "disabled rollover cannot contain interest-rate records"
            )
    else:
        if not profile.rollover.records:
            raise ExecutionValidationError(
                "FX rollover requires explicit InterestRateRecord inputs"
            )
        if profile.calibration_status is CalibrationStatus.UNCALIBRATED:
            raise ExecutionValidationError(
                "FXRolloverInterestModule inputs must be labelled proxy or "
                "broker_calibrated"
            )
        for record in profile.rollover.records:
            if not record.location or not record.time or not record.value.is_finite():
                raise ExecutionValidationError("invalid interest-rate record")


def validate_execution_profile(profile: ExecutionProfile) -> None:
    """Validate a reusable G0.7 execution profile without starting a run."""

    _validate_profile(profile)


def _validate_fee(config: FeeModelConfig) -> None:
    if config.kind in (FeeModelKind.FIXED, FeeModelKind.PER_CONTRACT):
        if config.commission is None or config.currency is None:
            raise ExecutionValidationError(
                f"{config.kind.value} fee requires commission and currency"
            )
        if not config.commission.is_finite() or config.commission < 0:
            raise ExecutionValidationError("fee commission must be non-negative")
        Currency.from_str(config.currency)
        if config.maker_rate is not None or config.taker_rate is not None:
            raise ExecutionValidationError("fixed fee configuration has rate fields")
        if (
            config.kind is FeeModelKind.FIXED
            and config.charge_commission_once is None
        ):
            raise ExecutionValidationError(
                "fixed fee requires explicit charge_commission_once"
            )
        if (
            config.kind is FeeModelKind.PER_CONTRACT
            and config.charge_commission_once is not None
        ):
            raise ExecutionValidationError(
                "per-contract fee does not support charge_commission_once"
            )
        return
    if config.maker_rate is None or config.taker_rate is None:
        raise ExecutionValidationError(
            "maker_taker fee requires explicit maker_rate and taker_rate"
        )
    if not config.maker_rate.is_finite() or not config.taker_rate.is_finite():
        raise ExecutionValidationError("maker/taker rates must be finite")
    if any(
        value is not None
        for value in (
            config.commission,
            config.currency,
            config.charge_commission_once,
        )
    ):
        raise ExecutionValidationError("maker_taker fee has fixed-fee fields")


def _validate_account(account: AccountParameters) -> None:
    if not account.initial_capital.is_finite() or account.initial_capital <= 0:
        raise ExecutionValidationError("initial capital must be positive")
    if not account.leverage.is_finite() or account.leverage <= 0:
        raise ExecutionValidationError("leverage must be positive")
    Currency.from_str(account.currency)


def _validate_probe(
    probe: ProbePlan, bar_count: int, instrument: CurrencyPair
) -> None:
    if not probe.quantity.is_finite() or probe.quantity <= 0:
        raise ExecutionValidationError("probe quantity must be positive")
    _quantity(probe.quantity, instrument.size_precision)
    if probe.entry_bar_index < 0 or probe.entry_bar_index >= bar_count:
        raise ExecutionValidationError("probe entry index is outside source bars")
    if probe.exit_bar_index is not None:
        if probe.exit_bar_index <= probe.entry_bar_index:
            raise ExecutionValidationError("probe exit must follow entry")
        if probe.exit_bar_index >= bar_count:
            raise ExecutionValidationError("probe exit index is outside source bars")
    if probe.order_kind is ProbeOrderKind.MARKET:
        if probe.limit_price is not None or probe.second_limit_price is not None:
            raise ExecutionValidationError("market probe cannot contain limit prices")
        return
    if probe.limit_price is None:
        raise ExecutionValidationError("limit probe requires limit_price")
    _price(probe.limit_price, instrument.price_precision)
    if probe.order_kind is ProbeOrderKind.DUAL_LIMIT:
        if probe.second_limit_price is None:
            raise ExecutionValidationError(
                "dual-limit probe requires second_limit_price"
            )
        _price(probe.second_limit_price, instrument.price_precision)
    elif probe.second_limit_price is not None:
        raise ExecutionValidationError(
            "single-limit probe cannot contain second_limit_price"
        )


def _validate_g0_5_manifest(manifest: Mapping[str, Any]) -> None:
    expected = {
        "instrument_id": INSTRUMENT_ID,
        "source_granularity": "1-minute",
        "price_sides": ["BID", "ASK"],
        "nautilus_version": NAUTILUS_VERSION,
        "ingestion_version": "g0.5-1",
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ExecutionValidationError(
                f"G0.5 manifest has incompatible {field}"
            )


def _validate_research_gate(
    mode: RunMode, path: Path, g0_5_sha256: str
) -> str | None:
    if not path.is_file():
        if mode is RunMode.RESEARCH:
            raise ExecutionValidationError(
                "research mode requires the G0.6 derived provenance manifest"
            )
        return None
    manifest = _load_json_object(path, "G0.6 manifest")
    if mode is RunMode.RESEARCH:
        if manifest.get("parent_g0_5_manifest_sha256") != g0_5_sha256:
            raise ExecutionValidationError(
                "research mode requires G0.6 provenance for this G0.5 manifest"
            )
        if manifest.get("derived_bar_integrity_valid") is not True:
            raise ExecutionValidationError(
                "research mode requires valid G0.6 derived-bar integrity"
            )
        if manifest.get("research_ready") is not True:
            raise ExecutionValidationError(
                "research mode requires G0.6 research_ready=true"
            )
    return _sha256_file(path)


def _load_instrument(catalog: ParquetDataCatalog) -> CurrencyPair:
    instruments = catalog.instruments([INSTRUMENT_ID])
    if len(instruments) != 1 or not isinstance(instruments[0], CurrencyPair):
        raise ExecutionValidationError(
            f"catalog must contain exactly one CurrencyPair for {INSTRUMENT_ID}"
        )
    return instruments[0]


def _load_execution_bars(
    catalog: ParquetDataCatalog, manifest: Mapping[str, Any]
) -> tuple[tuple[Bar, ...], tuple[Bar, ...]]:
    bid = tuple(catalog.query_bars([_SOURCE_BAR_TYPES[0]]))
    ask = tuple(catalog.query_bars([_SOURCE_BAR_TYPES[1]]))
    qa = manifest.get("qa")
    if not isinstance(qa, dict):
        raise ExecutionValidationError("G0.5 manifest has invalid qa")
    if qa.get("bid_bar_count") != len(bid) or qa.get("ask_bar_count") != len(ask):
        raise ExecutionValidationError(
            "catalog source counts do not match the G0.5 manifest"
        )
    for side, expected_type, bars in (
        ("BID", _SOURCE_BAR_TYPES[0], bid),
        ("ASK", _SOURCE_BAR_TYPES[1], ask),
    ):
        previous: int | None = None
        for bar in bars:
            if str(bar.bar_type) != expected_type:
                raise ExecutionValidationError(f"invalid {side} source bar type")
            if bar.ts_event % _MINUTE_NS != 0:
                raise ExecutionValidationError(f"{side} bar is not minute-aligned")
            if bar.ts_init != bar.ts_event + _MINUTE_NS:
                raise ExecutionValidationError(
                    f"{side} bar does not use G0.5 close availability"
                )
            if previous is not None and bar.ts_event <= previous:
                raise ExecutionValidationError(
                    f"{side} bars are not strictly chronological"
                )
            previous = bar.ts_event
    if tuple((bar.ts_event, bar.ts_init) for bar in bid) != tuple(
        (bar.ts_event, bar.ts_init) for bar in ask
    ):
        raise ExecutionValidationError(
            "paired BID/ASK bars require exactly matching timestamps"
        )
    return bid, ask


def _select_range(
    bid: Sequence[Bar], ask: Sequence[Bar], start_ns: int, end_ns: int
) -> tuple[tuple[Bar, ...], tuple[Bar, ...]]:
    selected_bid = tuple(
        bar for bar in bid if start_ns <= bar.ts_init <= end_ns
    )
    selected_ask = tuple(
        bar for bar in ask if start_ns <= bar.ts_init <= end_ns
    )
    if not selected_bid or not selected_ask:
        raise ExecutionValidationError("requested range has no paired source bars")
    if tuple(bar.ts_init for bar in selected_bid) != tuple(
        bar.ts_init for bar in selected_ask
    ):
        raise ExecutionValidationError("requested range has unpaired BID/ASK bars")
    return selected_bid, selected_ask


def _engine_config(identity: str) -> BacktestEngineConfig:
    uuid_text = str(uuid.UUID(hex=identity[:32], version=4))
    return BacktestEngineConfig(
        trader_id=None,
        load_state=False,
        save_state=False,
        shutdown_on_error=True,
        bypass_logging=True,
        run_analysis=False,
        logging=LoggerConfig(
            stdout_level=LogLevel.OFF,
            print_config=False,
            bypass_logging=True,
        ),
        instance_id=UUID4.from_str(uuid_text),
    )


def _venue_config(
    account: AccountParameters,
    profile: ExecutionProfile,
    fill_model: ProbabilisticFillModel,
    latency_model: StaticLatencyModel,
    fee_model: FixedFeeModel | PerContractFeeModel | MakerTakerFeeModel,
    modules: Sequence[FXRolloverInterestModule],
) -> BacktestVenueConfig:
    return BacktestVenueConfig(
        name=str(_VENUE),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        book_type=BookType.L1_MBP,
        starting_balances=[_money_text(account.initial_capital, account.currency)],
        routing=False,
        frozen_account=False,
        reject_stop_orders=True,
        support_gtd_orders=True,
        support_contingent_orders=True,
        use_position_ids=True,
        use_random_ids=False,
        use_reduce_only=True,
        bar_execution=True,
        bar_adaptive_high_low_ordering=profile.adaptive_high_low_ordering,
        trade_execution=False,
        use_market_order_acks=False,
        liquidity_consumption=False,
        allow_cash_borrowing=False,
        queue_position=False,
        oto_trigger_mode=OtoTriggerMode.PARTIAL,
        base_currency=Currency.from_str(account.currency),
        default_leverage=account.leverage,
        leverages=None,
        margin_model=None,
        modules=modules,
        fill_model=fill_model,
        latency_model=latency_model,
        fee_model=fee_model,
        price_protection_points=0,
        settlement_prices=None,
        liquidation_enabled=False,
        liquidation_trigger_ratio=1.0,
        liquidation_cancel_open_orders=True,
    )


def _add_venue(
    engine: BacktestEngine,
    account: AccountParameters,
    profile: ExecutionProfile,
    fill_model: ProbabilisticFillModel,
    latency_model: StaticLatencyModel,
    fee_model: FixedFeeModel | PerContractFeeModel | MakerTakerFeeModel,
    modules: Sequence[FXRolloverInterestModule],
) -> None:
    engine.add_venue(
        venue=_VENUE,
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


def _fee_model(
    config: FeeModelConfig,
) -> FixedFeeModel | PerContractFeeModel | MakerTakerFeeModel:
    if config.kind is FeeModelKind.MAKER_TAKER:
        return MakerTakerFeeModel()
    if config.commission is None or config.currency is None:
        raise RuntimeError("validated fee configuration is incomplete")
    commission = Money.from_str(_money_text(config.commission, config.currency))
    if config.kind is FeeModelKind.FIXED:
        return FixedFeeModel(
            commission,
            charge_commission_once=cast(bool, config.charge_commission_once),
        )
    return PerContractFeeModel(commission)


def _instrument_for_profile(
    instrument: CurrencyPair, config: FeeModelConfig
) -> CurrencyPair:
    if config.kind is not FeeModelKind.MAKER_TAKER:
        return instrument
    values = instrument.to_dict()
    values["maker_fee"] = str(config.maker_rate)
    values["taker_fee"] = str(config.taker_rate)
    cloned = CurrencyPair.from_dict(values)
    if cloned is None:
        raise RuntimeError("Nautilus could not clone the fee-configured instrument")
    return cloned


def _rollover_modules(
    config: RolloverConfig,
) -> tuple[FXRolloverInterestModule, ...]:
    if config.mode is RolloverMode.DISABLED:
        return ()
    records = [
        InterestRateRecord(
            location=record.location,
            time=record.time,
            value=float(record.value),
        )
        for record in config.records
    ]
    return (FXRolloverInterestModule(records),)


def _native_reports(engine: BacktestEngine) -> dict[str, pd.DataFrame]:
    return {
        "account": cast(pd.DataFrame, engine.generate_account_report(venue=_VENUE)),
        "orders": cast(pd.DataFrame, engine.generate_orders_report()),
        "order_fills": cast(pd.DataFrame, engine.generate_order_fills_report()),
        "fills": cast(pd.DataFrame, engine.generate_fills_report()),
        "positions": cast(pd.DataFrame, engine.generate_positions_report()),
    }


def _write_reports(
    output_dir: Path, reports: Mapping[str, pd.DataFrame]
) -> tuple[dict[str, Path], dict[str, str]]:
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for name, filename in _REPORT_FILENAMES.items():
        path = output_dir / filename
        report = reports[name]
        stable = _stable_report_frame(report)
        stable.to_csv(path, index=True, lineterminator="\n")
        path.chmod(0o444)
        paths[name] = path
        hashes[name] = _sha256_file(path)
    return paths, hashes


def _result_summary(
    engine: BacktestEngine,
    reports: Mapping[str, pd.DataFrame],
    account_parameters: AccountParameters,
    native_snapshot: NativeAccountSnapshot,
) -> dict[str, Any]:
    account = cast(MarginAccount | None, engine.cache.account_for_venue(_VENUE))
    if account is None:
        raise RuntimeError("Nautilus margin account is unavailable")
    currency = Currency.from_str(account_parameters.currency)
    balance = account.balance_total(currency)
    if balance is None:
        raise RuntimeError("Nautilus ending balance is unavailable")
    fills = _stable_report_frame(reports["fills"])
    fill_rows = _stable_records(fills)
    positions = _stable_report_frame(reports["positions"])
    return {
        "ending_balance": str(balance.as_decimal()),
        "ending_equity": str(native_snapshot.equity),
        "ending_currency": account_parameters.currency,
        "open_position_ids": sorted(native_snapshot.open_position_ids),
        "order_count": len(reports["orders"]),
        "fill_count": len(fills),
        "position_count": len(positions),
        "fills": fill_rows,
        "positions_sha256": _sha256_json(_stable_records(positions)),
        "account_sha256": _sha256_json(_stable_records(reports["account"])),
    }


def _ftmo_evaluation_dict(
    phase: EvaluationPhase,
    bridge: NautilusFtmoBridge,
    snapshot: NativeAccountSnapshot,
) -> dict[str, Any]:
    state = bridge.overlay.state
    breach = state.breach
    valuation = bridge.valuation()
    observations = [_jsonable(asdict(item)) for item in bridge.observations]
    return {
        "observation_mechanism": (
            "fresh-pair native fill/position callbacks; pre-bar callbacks "
            "coalesced into the complete BID/ASK pair post-settlement "
            "native-clock alert"
        ),
        "observation_mechanism_version": FTMO_OBSERVATION_VERSION,
        "valuation_mechanism": (
            "native account balance plus native Position.unrealized_pnl at "
            "completed-pair liquidation-side close"
        ),
        "valuation_mechanism_version": FTMO_VALUATION_VERSION,
        "valuation_resolution": "1-minute paired-bar close",
        "continuous_intraminute_compliance_exact": False,
        "post_settlement_delay_ns": FTMO_OBSERVATION_DELAY_NS,
        "active_phase": phase.value,
        "current_ftmo_trading_day": state.current_trading_day.isoformat(),
        "midnight_reset_balance": str(state.midnight_reset_balance),
        "daily_loss_floor": str(state.daily_loss_floor),
        "maximum_loss_floor": str(state.maximum_loss_floor),
        "counted_trading_days": sorted(day.isoformat() for day in state.trading_days),
        "terminal_status": state.status.value,
        "breach": (
            None
            if breach is None
            else {
                "reason": breach.reason.value,
                "equity": str(breach.equity),
                "floor": str(breach.floor),
                "timestamp_ns": breach.timestamp_ns,
            }
        ),
        "final_native_snapshot": {
            "balance": str(snapshot.balance),
            "equity": str(snapshot.equity),
            "native_portfolio_equity": str(valuation.native_portfolio_equity),
            "native_minus_ftmo_equity": str(
                valuation.native_portfolio_equity - snapshot.equity
            ),
            "open_position_ids": sorted(snapshot.open_position_ids),
            "completed_mark_timestamp_ns": (
                valuation.completed_mark_timestamp_ns
            ),
            "bid_close": (
                None if valuation.bid_close is None else str(valuation.bid_close)
            ),
            "ask_close": (
                None if valuation.ask_close is None else str(valuation.ask_close)
            ),
        },
        "observation_count": len(observations),
        "observations_sha256": _sha256_json(observations),
    }


def _stable_report_frame(report: pd.DataFrame) -> pd.DataFrame:
    # rc2 generates fresh OrderInitialized and event UUIDs even when venue
    # order/trade/position IDs are deterministic. They are transport metadata,
    # not execution outcomes, so exclude them from immutable semantic reports.
    stable = report.drop(columns=["init_id", "event_id"], errors="ignore").copy()
    if "events" in stable.columns:
        stable["events"] = stable["events"].map(_strip_volatile_event_ids)
    return stable


def _strip_volatile_event_ids(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_volatile_event_ids(item)
            for key, item in value.items()
            if key not in {"event_id", "init_id"}
        }
    if isinstance(value, list):
        return [_strip_volatile_event_ids(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_volatile_event_ids(item) for item in value)
    return value


def _stable_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    reset = frame.reset_index()
    records = cast(list[dict[str, Any]], reset.to_dict(orient="records"))
    return [
        {str(key): _scalar_jsonable(value) for key, value in sorted(row.items())}
        for row in records
    ]


def _run_identity(
    request: ExecutionRunRequest,
    profile_sha256: str,
    g0_5_sha256: str,
    g0_6_sha256: str | None,
    bid: Sequence[Bar],
    ask: Sequence[Bar],
) -> str:
    value = {
        "profile_sha256": profile_sha256,
        "g0_5_manifest_sha256": g0_5_sha256,
        "g0_6_manifest_sha256": g0_6_sha256,
        "account": _account_dict(request.account),
        "probe": _jsonable(asdict(request.probe)),
        "mode": request.mode.value,
        "ftmo_phase": request.ftmo_phase.value,
        "start": request.requested_start_ns,
        "end": request.requested_end_ns,
        "bid": [_bar_content(bar) for bar in bid],
        "ask": [_bar_content(bar) for bar in ask],
    }
    return _sha256_json(value)


def _run_config_dict(
    request: ExecutionRunRequest, run_config_id: str
) -> dict[str, Any]:
    return {
        "id": run_config_id,
        "oms_type": "NETTING",
        "account_type": "MARGIN",
        "book_type": "L1_MBP",
        "bar_execution": True,
        "bar_adaptive_high_low_ordering": (
            request.profile.adaptive_high_low_ordering
        ),
        "trade_execution": False,
        "routing": False,
        "use_random_ids": False,
        "use_position_ids": True,
        "use_reduce_only": True,
        "use_market_order_acks": False,
        "liquidity_consumption": False,
        "queue_position": False,
        "allow_cash_borrowing": False,
        "frozen_account": False,
        "source": "paired external 1-minute BID/ASK bars",
        "spread_adjustment": "none",
        "ftmo_observation_version": FTMO_OBSERVATION_VERSION,
        "ftmo_valuation_version": FTMO_VALUATION_VERSION,
        "ftmo_post_settlement_delay_ns": FTMO_OBSERVATION_DELAY_NS,
    }


def _profile_dict(profile: ExecutionProfile) -> dict[str, Any]:
    return cast(dict[str, Any], _jsonable(asdict(profile)))


def _account_dict(account: AccountParameters) -> dict[str, str]:
    return {
        "initial_capital": str(account.initial_capital),
        "currency": account.currency,
        "leverage": str(account.leverage),
    }


def _bar_content(bar: Bar) -> list[Any]:
    return [
        str(bar.bar_type),
        str(bar.open),
        str(bar.high),
        str(bar.low),
        str(bar.close),
        str(bar.volume),
        bar.ts_event,
        bar.ts_init,
    ]


def _quantity(value: Decimal, precision: int) -> Quantity:
    quantum = Decimal(1).scaleb(-precision)
    if value != value.quantize(quantum):
        raise ExecutionValidationError(
            f"quantity exceeds instrument size precision {precision}"
        )
    return Quantity.from_str(f"{value:.{precision}f}")


def _price(value: Decimal, precision: int) -> Price:
    if not value.is_finite() or value <= 0:
        raise ExecutionValidationError("limit price must be positive")
    quantum = Decimal(1).scaleb(-precision)
    if value != value.quantize(quantum):
        raise ExecutionValidationError(
            f"limit price exceeds instrument precision {precision}"
        )
    return Price.from_str(f"{value:.{precision}f}")


def _money_text(amount: Decimal, currency: str) -> str:
    return f"{format(amount, 'f')} {currency}"


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExecutionValidationError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise ExecutionValidationError(f"{label} must be an object")
    return cast(dict[str, Any], value)


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(_sha256_file(item).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value), separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _scalar_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _scalar_jsonable(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (tuple, list)):
        return [_scalar_jsonable(item) for item in value]
    if isinstance(value, float):
        if pd.isna(value):
            return None
        return format(value, ".17g")
    missing = pd.isna(value)
    if isinstance(missing, bool) and missing:
        return None
    return str(value)


def _is_hex(value: str) -> bool:
    try:
        int(value, 16)
    except ValueError:
        return False
    return True
