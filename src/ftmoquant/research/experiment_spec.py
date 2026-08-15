"""Strict G1.1 preregistration models for ``trend_pullback_v1``.

This module validates a single research hypothesis and its not-yet-run registry
entry.  It intentionally contains no indicators, signals, orders, backtest
runner, parameter search, or performance calculation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml

_SEMVER_PATTERN = re.compile(r"[1-9]\d*\.\d+\.\d+")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")

REGISTRY_COLUMNS = (
    "experiment_id",
    "strategy_id",
    "strategy_version",
    "status",
    "spec_path",
    "config_path",
    "strategy_config_sha256",
    "git_commit",
    "data_manifest_sha256",
    "execution_profile_sha256",
    "engine_version",
    "run_seed",
    "started_at",
    "completed_at",
    "primary_metric",
    "primary_metric_value",
    "decision",
)

_REQUIRED_PROVENANCE_FIELDS = (
    "git_commit",
    "strategy_config_sha256",
    "data_manifest_sha256",
    "execution_profile_sha256",
    "engine_version",
    "run_seed",
)


class StrategySpecValidationError(ValueError):
    """Raised when a strategy preregistration is ambiguous or invalid."""


class ExperimentRegistryValidationError(ValueError):
    """Raised when registry state overstates available experiment evidence."""


@dataclass(frozen=True, slots=True)
class InstrumentSpec:
    """The single market admitted by the first hypothesis."""

    instrument_id: str
    asset_class: str


@dataclass(frozen=True, slots=True)
class DataSemantics:
    """Market-data identity, timing, and fail-closed policies."""

    price_input: str
    bar_completion: str
    timestamp_policy: str
    missing_data_policy: str
    warmup_boundary_policy: str
    signal_bid_bar_type: str
    signal_ask_bar_type: str
    trend_bid_bar_type: str
    trend_ask_bar_type: str


@dataclass(frozen=True, slots=True)
class TrendDefinition:
    """Completed 4H EMA trend filter."""

    timeframe: str
    fast_ema_period: int
    slow_ema_period: int
    long_condition: str
    short_condition: str
    equality_policy: str


@dataclass(frozen=True, slots=True)
class PullbackDefinition:
    """Completed 1H pullback arming and continuation trigger."""

    timeframe: str
    ema_period: int
    max_armed_bars: int
    long_arm: str
    long_trigger: str
    short_arm: str
    short_trigger: str
    expiry_policy: str


@dataclass(frozen=True, slots=True)
class VolatilityDefinition:
    """ATR inputs fixed before strategy evaluation."""

    timeframe: str
    atr_period: int
    atr_moving_average: str
    atr_value_at_signal: str


@dataclass(frozen=True, slots=True)
class SignalDefinition:
    """Complete trend, pullback, and volatility signal contract."""

    trend: TrendDefinition
    pullback: PullbackDefinition
    volatility: VolatilityDefinition


@dataclass(frozen=True, slots=True)
class ExecutionDefinition:
    """Signal-to-order timing without implementing an order."""

    order_type: str
    earliest_entry: str
    execution_market: str
    signal_expiry_minutes: int


@dataclass(frozen=True, slots=True)
class RiskDefinition:
    """Alpha-research sizing independent of FTMO optimization."""

    sizing_convention: str
    risk_unit: Decimal
    compounding: bool
    ftmo_adjustments: bool


@dataclass(frozen=True, slots=True)
class ExitDefinition:
    """Fixed baseline exits and conservative ambiguous-path behavior."""

    stop_atr_multiple: Decimal
    target_r_multiple: Decimal
    time_exit_bars: int
    opposite_signal_policy: str
    same_bar_stop_target_policy: str


@dataclass(frozen=True, slots=True)
class PositionDefinition:
    """Position concurrency, signal, and re-entry policy."""

    max_open_positions: int
    pyramiding: bool
    signals_while_open: str
    reentry: str


@dataclass(frozen=True, slots=True)
class SplitDefinition:
    """Chronological development, validation, and sealed holdout split."""

    development_fraction: Decimal
    validation_fraction: Decimal
    final_holdout_fraction: Decimal
    boundary_rule: str
    final_holdout_policy: str


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    """Predeclared primary and supporting metrics."""

    primary: str
    supporting: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GateDefinition:
    """G1 evidence thresholds fixed before observing returns."""

    development_mean_net_r_gt: Decimal
    validation_mean_net_r_gt: Decimal
    validation_ci_level: Decimal
    validation_ci_lower_bound_gt: Decimal
    validation_ci_method: str
    validation_ci_block_size: int
    validation_ci_repetitions: int
    validation_ci_seed: int
    min_development_trades: int
    min_validation_trades: int
    minimum_validation_profit_factor_gt: Decimal


@dataclass(frozen=True, slots=True)
class ParameterFamily:
    """The only parameter family admitted by this preregistration."""

    mode: str
    permitted_variants: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FalsificationDefinition:
    """Outcomes which cannot be relabeled as a passing G1 result."""

    insufficient_trade_count: str
    nonpositive_development: str
    nonpositive_validation: str
    nonpositive_validation_ci: str
    maximum_single_year_net_r_share: Decimal


@dataclass(frozen=True, slots=True)
class ResearchProtocol:
    """Split, metrics, gates, parameter family, and falsification rules."""

    split: SplitDefinition
    metrics: MetricDefinition
    gates: GateDefinition
    parameter_family: ParameterFamily
    falsification: FalsificationDefinition


@dataclass(frozen=True, slots=True)
class ProvenanceRequirements:
    """Runtime evidence required before an experiment can be completed."""

    required_fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TrendPullbackSpec:
    """Immutable, complete G1.1 specification for ``trend_pullback_v1``."""

    schema_version: int
    strategy_id: str
    version: str
    status: str
    instrument: InstrumentSpec
    data_semantics: DataSemantics
    signal: SignalDefinition
    execution: ExecutionDefinition
    risk: RiskDefinition
    exits: ExitDefinition
    positions: PositionDefinition
    research_protocol: ResearchProtocol
    provenance: ProvenanceRequirements


@dataclass(frozen=True, slots=True)
class ExperimentRegistryEntry:
    """Validated registry row; blank runtime fields remain explicit blanks."""

    experiment_id: str
    strategy_id: str
    strategy_version: str
    status: str
    spec_path: str
    config_path: str
    strategy_config_sha256: str
    git_commit: str
    data_manifest_sha256: str
    execution_profile_sha256: str
    engine_version: str
    run_seed: str
    started_at: str
    completed_at: str
    primary_metric: str
    primary_metric_value: str
    decision: str


def load_trend_pullback_spec(path: str | Path) -> TrendPullbackSpec:
    """Load the one admitted G1.1 YAML document with an exact schema."""

    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise StrategySpecValidationError(
            f"could not load {config_path}: {error}"
        ) from error

    root = _mapping(raw, "configuration")
    _keys(
        root,
        {
            "schema_version",
            "strategy_id",
            "version",
            "status",
            "instrument",
            "data_semantics",
            "signal",
            "execution",
            "risk",
            "exits",
            "positions",
            "research_protocol",
            "provenance",
        },
        "configuration",
    )

    schema_version = _positive_integer(root["schema_version"], "schema_version")
    if schema_version != 1:
        raise StrategySpecValidationError("schema_version must be 1")
    strategy_id = _literal(root["strategy_id"], "trend_pullback_v1", "strategy_id")
    version = _string(root["version"], "version")
    if _SEMVER_PATTERN.fullmatch(version) is None:
        raise StrategySpecValidationError("version must use MAJOR.MINOR.PATCH format")

    spec = TrendPullbackSpec(
        schema_version=schema_version,
        strategy_id=strategy_id,
        version=version,
        status=_literal(root["status"], "specified_not_run", "status"),
        instrument=_load_instrument(root["instrument"]),
        data_semantics=_load_data_semantics(root["data_semantics"]),
        signal=_load_signal(root["signal"]),
        execution=_load_execution(root["execution"]),
        risk=_load_risk(root["risk"]),
        exits=_load_exits(root["exits"]),
        positions=_load_positions(root["positions"]),
        research_protocol=_load_research_protocol(root["research_protocol"]),
        provenance=_load_provenance(root["provenance"]),
    )
    _validate_cross_field_invariants(spec)
    return spec


def canonical_spec_json(spec: TrendPullbackSpec) -> str:
    """Return deterministic semantic JSON for hashing and provenance."""

    return json.dumps(
        asdict(spec),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=_json_default,
    )


def strategy_config_sha256(spec: TrendPullbackSpec) -> str:
    """Hash the canonical semantic configuration, not YAML formatting."""

    return hashlib.sha256(canonical_spec_json(spec).encode("utf-8")).hexdigest()


def load_experiment_registry(path: str | Path) -> tuple[ExperimentRegistryEntry, ...]:
    """Load registry rows and reject states which imply nonexistent evidence."""

    registry_path = Path(path)
    try:
        with registry_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != REGISTRY_COLUMNS:
                raise ExperimentRegistryValidationError(
                    "registry columns must exactly match REGISTRY_COLUMNS"
                )
            rows = list(reader)
    except OSError as error:
        raise ExperimentRegistryValidationError(
            f"could not load {registry_path}: {error}"
        ) from error

    if not rows:
        raise ExperimentRegistryValidationError(
            "registry must contain at least one row"
        )

    entries: list[ExperimentRegistryEntry] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows, start=2):
        if None in row or any(row[column] is None for column in REGISTRY_COLUMNS):
            raise ExperimentRegistryValidationError(
                f"registry row {index} has unexpected columns"
            )
        entry = ExperimentRegistryEntry(
            **{column: row[column] for column in REGISTRY_COLUMNS}
        )
        _validate_registry_entry(entry, index)
        if entry.experiment_id in seen_ids:
            raise ExperimentRegistryValidationError(
                f"registry row {index} duplicates experiment_id"
            )
        seen_ids.add(entry.experiment_id)
        entries.append(entry)
    return tuple(entries)


def _load_instrument(value: object) -> InstrumentSpec:
    item = _mapping(value, "instrument")
    _keys(item, {"instrument_id", "asset_class"}, "instrument")
    return InstrumentSpec(
        instrument_id=_literal(
            item["instrument_id"], "EUR/USD.DUKASCOPY", "instrument.instrument_id"
        ),
        asset_class=_literal(item["asset_class"], "fx", "instrument.asset_class"),
    )


def _load_data_semantics(value: object) -> DataSemantics:
    item = _mapping(value, "data_semantics")
    _keys(
        item,
        {
            "price_input",
            "bar_completion",
            "timestamp_policy",
            "missing_data_policy",
            "warmup_boundary_policy",
            "bars",
        },
        "data_semantics",
    )
    bars = _mapping(item["bars"], "data_semantics.bars")
    _keys(
        bars,
        {"signal_bid", "signal_ask", "trend_bid", "trend_ask"},
        "data_semantics.bars",
    )
    return DataSemantics(
        price_input=_literal(
            item["price_input"],
            "synchronized_bid_ask_ohlc_midpoint",
            "data_semantics.price_input",
        ),
        bar_completion=_literal(
            item["bar_completion"], "completed_only", "data_semantics.bar_completion"
        ),
        timestamp_policy=_literal(
            item["timestamp_policy"],
            "decision_at_completed_bar_ts_init",
            "data_semantics.timestamp_policy",
        ),
        missing_data_policy=_literal(
            item["missing_data_policy"],
            "fail_closed_and_emit_no_signal",
            "data_semantics.missing_data_policy",
        ),
        warmup_boundary_policy=_literal(
            item["warmup_boundary_policy"],
            "prior_split_context_allowed_no_prior_split_trades",
            "data_semantics.warmup_boundary_policy",
        ),
        signal_bid_bar_type=_literal(
            bars["signal_bid"],
            "EUR/USD.DUKASCOPY-1-HOUR-BID-INTERNAL",
            "data_semantics.bars.signal_bid",
        ),
        signal_ask_bar_type=_literal(
            bars["signal_ask"],
            "EUR/USD.DUKASCOPY-1-HOUR-ASK-INTERNAL",
            "data_semantics.bars.signal_ask",
        ),
        trend_bid_bar_type=_literal(
            bars["trend_bid"],
            "EUR/USD.DUKASCOPY-4-HOUR-BID-INTERNAL",
            "data_semantics.bars.trend_bid",
        ),
        trend_ask_bar_type=_literal(
            bars["trend_ask"],
            "EUR/USD.DUKASCOPY-4-HOUR-ASK-INTERNAL",
            "data_semantics.bars.trend_ask",
        ),
    )


def _load_signal(value: object) -> SignalDefinition:
    item = _mapping(value, "signal")
    _keys(item, {"trend", "pullback", "volatility"}, "signal")
    return SignalDefinition(
        trend=_load_trend(item["trend"]),
        pullback=_load_pullback(item["pullback"]),
        volatility=_load_volatility(item["volatility"]),
    )


def _load_trend(value: object) -> TrendDefinition:
    item = _mapping(value, "signal.trend")
    _keys(
        item,
        {
            "timeframe",
            "fast_ema_period",
            "slow_ema_period",
            "long_condition",
            "short_condition",
            "equality_policy",
        },
        "signal.trend",
    )
    return TrendDefinition(
        timeframe=_literal(item["timeframe"], "4h", "signal.trend.timeframe"),
        fast_ema_period=_positive_integer(
            item["fast_ema_period"], "signal.trend.fast_ema_period"
        ),
        slow_ema_period=_positive_integer(
            item["slow_ema_period"], "signal.trend.slow_ema_period"
        ),
        long_condition=_literal(
            item["long_condition"],
            "fast_ema_gt_slow_ema",
            "signal.trend.long_condition",
        ),
        short_condition=_literal(
            item["short_condition"],
            "fast_ema_lt_slow_ema",
            "signal.trend.short_condition",
        ),
        equality_policy=_literal(
            item["equality_policy"], "no_trend", "signal.trend.equality_policy"
        ),
    )


def _load_pullback(value: object) -> PullbackDefinition:
    item = _mapping(value, "signal.pullback")
    _keys(
        item,
        {
            "timeframe",
            "ema_period",
            "max_armed_bars",
            "long_arm",
            "long_trigger",
            "short_arm",
            "short_trigger",
            "expiry_policy",
        },
        "signal.pullback",
    )
    return PullbackDefinition(
        timeframe=_literal(item["timeframe"], "1h", "signal.pullback.timeframe"),
        ema_period=_positive_integer(item["ema_period"], "signal.pullback.ema_period"),
        max_armed_bars=_positive_integer(
            item["max_armed_bars"], "signal.pullback.max_armed_bars"
        ),
        long_arm=_literal(
            item["long_arm"],
            "previous_close_gt_previous_ema_and_close_lte_ema",
            "signal.pullback.long_arm",
        ),
        long_trigger=_literal(
            item["long_trigger"],
            "armed_and_previous_close_lte_previous_ema_and_close_gt_ema",
            "signal.pullback.long_trigger",
        ),
        short_arm=_literal(
            item["short_arm"],
            "previous_close_lt_previous_ema_and_close_gte_ema",
            "signal.pullback.short_arm",
        ),
        short_trigger=_literal(
            item["short_trigger"],
            "armed_and_previous_close_gte_previous_ema_and_close_lt_ema",
            "signal.pullback.short_trigger",
        ),
        expiry_policy=_literal(
            item["expiry_policy"],
            "expire_after_max_armed_bars_or_trend_loss",
            "signal.pullback.expiry_policy",
        ),
    )


def _load_volatility(value: object) -> VolatilityDefinition:
    item = _mapping(value, "signal.volatility")
    _keys(
        item,
        {"timeframe", "atr_period", "atr_moving_average", "atr_value_at_signal"},
        "signal.volatility",
    )
    return VolatilityDefinition(
        timeframe=_literal(item["timeframe"], "1h", "signal.volatility.timeframe"),
        atr_period=_positive_integer(
            item["atr_period"], "signal.volatility.atr_period"
        ),
        atr_moving_average=_literal(
            item["atr_moving_average"],
            "exponential",
            "signal.volatility.atr_moving_average",
        ),
        atr_value_at_signal=_literal(
            item["atr_value_at_signal"],
            "completed_trigger_bar",
            "signal.volatility.atr_value_at_signal",
        ),
    )


def _load_execution(value: object) -> ExecutionDefinition:
    item = _mapping(value, "execution")
    _keys(
        item,
        {"order_type", "earliest_entry", "execution_market", "signal_expiry_minutes"},
        "execution",
    )
    return ExecutionDefinition(
        order_type=_literal(item["order_type"], "market", "execution.order_type"),
        earliest_entry=_literal(
            item["earliest_entry"],
            "first_synchronized_1m_pair_strictly_after_signal_close",
            "execution.earliest_entry",
        ),
        execution_market=_literal(
            item["execution_market"],
            "canonical_g0_7_bid_ask_harness",
            "execution.execution_market",
        ),
        signal_expiry_minutes=_positive_integer(
            item["signal_expiry_minutes"], "execution.signal_expiry_minutes"
        ),
    )


def _load_risk(value: object) -> RiskDefinition:
    item = _mapping(value, "risk")
    _keys(
        item,
        {"sizing_convention", "risk_unit", "compounding", "ftmo_adjustments"},
        "risk",
    )
    return RiskDefinition(
        sizing_convention=_literal(
            item["sizing_convention"],
            "quantity_sizes_initial_stop_loss_to_one_r_before_costs",
            "risk.sizing_convention",
        ),
        risk_unit=_positive_decimal(item["risk_unit"], "risk.risk_unit"),
        compounding=_boolean(item["compounding"], "risk.compounding"),
        ftmo_adjustments=_boolean(item["ftmo_adjustments"], "risk.ftmo_adjustments"),
    )


def _load_exits(value: object) -> ExitDefinition:
    item = _mapping(value, "exits")
    _keys(
        item,
        {
            "stop_atr_multiple",
            "target_r_multiple",
            "time_exit_bars",
            "opposite_signal_policy",
            "same_bar_stop_target_policy",
        },
        "exits",
    )
    return ExitDefinition(
        stop_atr_multiple=_positive_decimal(
            item["stop_atr_multiple"], "exits.stop_atr_multiple"
        ),
        target_r_multiple=_positive_decimal(
            item["target_r_multiple"], "exits.target_r_multiple"
        ),
        time_exit_bars=_positive_integer(
            item["time_exit_bars"], "exits.time_exit_bars"
        ),
        opposite_signal_policy=_literal(
            item["opposite_signal_policy"],
            "ignore_while_position_open",
            "exits.opposite_signal_policy",
        ),
        same_bar_stop_target_policy=_literal(
            item["same_bar_stop_target_policy"],
            "lower_timeframe_path_else_stop_first",
            "exits.same_bar_stop_target_policy",
        ),
    )


def _load_positions(value: object) -> PositionDefinition:
    item = _mapping(value, "positions")
    _keys(
        item,
        {"max_open_positions", "pyramiding", "signals_while_open", "reentry"},
        "positions",
    )
    return PositionDefinition(
        max_open_positions=_positive_integer(
            item["max_open_positions"], "positions.max_open_positions"
        ),
        pyramiding=_boolean(item["pyramiding"], "positions.pyramiding"),
        signals_while_open=_literal(
            item["signals_while_open"],
            "discard",
            "positions.signals_while_open",
        ),
        reentry=_literal(
            item["reentry"],
            "requires_new_arm_and_trigger_after_flat",
            "positions.reentry",
        ),
    )


def _load_research_protocol(value: object) -> ResearchProtocol:
    item = _mapping(value, "research_protocol")
    _keys(
        item,
        {"split", "metrics", "gates", "parameter_family", "falsification"},
        "research_protocol",
    )
    return ResearchProtocol(
        split=_load_split(item["split"]),
        metrics=_load_metrics(item["metrics"]),
        gates=_load_gates(item["gates"]),
        parameter_family=_load_parameter_family(item["parameter_family"]),
        falsification=_load_falsification(item["falsification"]),
    )


def _load_split(value: object) -> SplitDefinition:
    item = _mapping(value, "research_protocol.split")
    _keys(
        item,
        {
            "development_fraction",
            "validation_fraction",
            "final_holdout_fraction",
            "boundary_rule",
            "final_holdout_policy",
        },
        "research_protocol.split",
    )
    return SplitDefinition(
        development_fraction=_unit_fraction(
            item["development_fraction"], "research_protocol.split.development_fraction"
        ),
        validation_fraction=_unit_fraction(
            item["validation_fraction"], "research_protocol.split.validation_fraction"
        ),
        final_holdout_fraction=_unit_fraction(
            item["final_holdout_fraction"],
            "research_protocol.split.final_holdout_fraction",
        ),
        boundary_rule=_literal(
            item["boundary_rule"],
            "chronological_whole_utc_days_floor",
            "research_protocol.split.boundary_rule",
        ),
        final_holdout_policy=_literal(
            item["final_holdout_policy"],
            "sealed_until_g2",
            "research_protocol.split.final_holdout_policy",
        ),
    )


def _load_metrics(value: object) -> MetricDefinition:
    item = _mapping(value, "research_protocol.metrics")
    _keys(item, {"primary", "supporting"}, "research_protocol.metrics")
    supporting = _string_tuple(
        item["supporting"], "research_protocol.metrics.supporting", allow_empty=False
    )
    if len(set(supporting)) != len(supporting):
        raise StrategySpecValidationError("supporting metrics must be unique")
    return MetricDefinition(
        primary=_literal(
            item["primary"],
            "validation_mean_net_r_per_trade",
            "research_protocol.metrics.primary",
        ),
        supporting=supporting,
    )


def _load_gates(value: object) -> GateDefinition:
    item = _mapping(value, "research_protocol.gates")
    _keys(
        item,
        {
            "development_mean_net_r_gt",
            "validation_mean_net_r_gt",
            "validation_ci_level",
            "validation_ci_lower_bound_gt",
            "validation_ci_method",
            "validation_ci_block_size",
            "validation_ci_repetitions",
            "validation_ci_seed",
            "min_development_trades",
            "min_validation_trades",
            "minimum_validation_profit_factor_gt",
        },
        "research_protocol.gates",
    )
    return GateDefinition(
        development_mean_net_r_gt=_finite_decimal(
            item["development_mean_net_r_gt"],
            "research_protocol.gates.development_mean_net_r_gt",
        ),
        validation_mean_net_r_gt=_finite_decimal(
            item["validation_mean_net_r_gt"],
            "research_protocol.gates.validation_mean_net_r_gt",
        ),
        validation_ci_level=_unit_fraction(
            item["validation_ci_level"], "research_protocol.gates.validation_ci_level"
        ),
        validation_ci_lower_bound_gt=_finite_decimal(
            item["validation_ci_lower_bound_gt"],
            "research_protocol.gates.validation_ci_lower_bound_gt",
        ),
        validation_ci_method=_literal(
            item["validation_ci_method"],
            "bca",
            "research_protocol.gates.validation_ci_method",
        ),
        validation_ci_block_size=_positive_integer(
            item["validation_ci_block_size"],
            "research_protocol.gates.validation_ci_block_size",
        ),
        validation_ci_repetitions=_positive_integer(
            item["validation_ci_repetitions"],
            "research_protocol.gates.validation_ci_repetitions",
        ),
        validation_ci_seed=_nonnegative_integer(
            item["validation_ci_seed"],
            "research_protocol.gates.validation_ci_seed",
        ),
        min_development_trades=_positive_integer(
            item["min_development_trades"],
            "research_protocol.gates.min_development_trades",
        ),
        min_validation_trades=_positive_integer(
            item["min_validation_trades"],
            "research_protocol.gates.min_validation_trades",
        ),
        minimum_validation_profit_factor_gt=_positive_decimal(
            item["minimum_validation_profit_factor_gt"],
            "research_protocol.gates.minimum_validation_profit_factor_gt",
        ),
    )


def _load_parameter_family(value: object) -> ParameterFamily:
    item = _mapping(value, "research_protocol.parameter_family")
    _keys(item, {"mode", "permitted_variants"}, "research_protocol.parameter_family")
    return ParameterFamily(
        mode=_literal(
            item["mode"], "baseline_only", "research_protocol.parameter_family.mode"
        ),
        permitted_variants=_string_tuple(
            item["permitted_variants"],
            "research_protocol.parameter_family.permitted_variants",
            allow_empty=True,
        ),
    )


def _load_falsification(value: object) -> FalsificationDefinition:
    item = _mapping(value, "research_protocol.falsification")
    _keys(
        item,
        {
            "insufficient_trade_count",
            "nonpositive_development",
            "nonpositive_validation",
            "nonpositive_validation_ci",
            "maximum_single_year_net_r_share",
        },
        "research_protocol.falsification",
    )
    return FalsificationDefinition(
        insufficient_trade_count=_literal(
            item["insufficient_trade_count"],
            "unresolved_not_pass",
            "research_protocol.falsification.insufficient_trade_count",
        ),
        nonpositive_development=_literal(
            item["nonpositive_development"],
            "fail",
            "research_protocol.falsification.nonpositive_development",
        ),
        nonpositive_validation=_literal(
            item["nonpositive_validation"],
            "fail",
            "research_protocol.falsification.nonpositive_validation",
        ),
        nonpositive_validation_ci=_literal(
            item["nonpositive_validation_ci"],
            "fail",
            "research_protocol.falsification.nonpositive_validation_ci",
        ),
        maximum_single_year_net_r_share=_unit_fraction(
            item["maximum_single_year_net_r_share"],
            "research_protocol.falsification.maximum_single_year_net_r_share",
        ),
    )


def _load_provenance(value: object) -> ProvenanceRequirements:
    item = _mapping(value, "provenance")
    _keys(item, {"required_fields"}, "provenance")
    fields = _string_tuple(item["required_fields"], "provenance.required_fields", False)
    if fields != _REQUIRED_PROVENANCE_FIELDS:
        raise StrategySpecValidationError(
            "provenance.required_fields must use the complete ordered G1 set"
        )
    return ProvenanceRequirements(required_fields=fields)


def _validate_cross_field_invariants(spec: TrendPullbackSpec) -> None:
    if spec.signal.trend.fast_ema_period >= spec.signal.trend.slow_ema_period:
        raise StrategySpecValidationError(
            "fast_ema_period must be less than slow_ema_period"
        )
    if spec.risk.risk_unit != Decimal("1"):
        raise StrategySpecValidationError("risk_unit must be exactly one R")
    if spec.risk.compounding:
        raise StrategySpecValidationError("compounding must be disabled in G1")
    if spec.risk.ftmo_adjustments:
        raise StrategySpecValidationError("FTMO adjustments must be disabled in G1")
    if spec.positions.max_open_positions != 1:
        raise StrategySpecValidationError("max_open_positions must be exactly 1")
    if spec.positions.pyramiding:
        raise StrategySpecValidationError("pyramiding must be disabled")
    split = spec.research_protocol.split
    if (
        split.development_fraction
        + split.validation_fraction
        + split.final_holdout_fraction
        != Decimal("1")
    ):
        raise StrategySpecValidationError("research split fractions must sum to 1")
    family = spec.research_protocol.parameter_family
    if family.mode == "baseline_only" and family.permitted_variants:
        raise StrategySpecValidationError(
            "baseline_only parameter family cannot contain permitted variants"
        )


def _validate_registry_entry(entry: ExperimentRegistryEntry, row_number: int) -> None:
    prefix = f"registry row {row_number}"
    for field in (
        "experiment_id",
        "strategy_id",
        "strategy_version",
        "spec_path",
        "config_path",
        "primary_metric",
        "decision",
    ):
        if not getattr(entry, field).strip():
            raise ExperimentRegistryValidationError(f"{prefix} {field} cannot be blank")
    if entry.strategy_id not in {"trend_pullback_v1", "leo_gbpusd_v1"}:
        raise ExperimentRegistryValidationError(f"{prefix} has an unknown strategy_id")
    if entry.primary_metric != "validation_mean_net_r_per_trade":
        raise ExperimentRegistryValidationError(
            f"{prefix} has an unknown primary_metric"
        )
    if _SEMVER_PATTERN.fullmatch(entry.strategy_version) is None:
        raise ExperimentRegistryValidationError(
            f"{prefix} has an invalid strategy_version"
        )
    if _SHA256_PATTERN.fullmatch(entry.strategy_config_sha256) is None:
        raise ExperimentRegistryValidationError(
            f"{prefix} strategy_config_sha256 must be a lowercase SHA-256"
        )

    runtime_fields = (
        "git_commit",
        "data_manifest_sha256",
        "execution_profile_sha256",
        "engine_version",
        "run_seed",
        "started_at",
        "completed_at",
        "primary_metric_value",
    )
    if entry.status == "specified_not_run":
        if entry.decision != "not_evaluated":
            raise ExperimentRegistryValidationError(
                f"{prefix} specified_not_run decision must be not_evaluated"
            )
        populated = [field for field in runtime_fields if getattr(entry, field)]
        if populated:
            raise ExperimentRegistryValidationError(
                f"{prefix} specified_not_run cannot contain runtime evidence: "
                + ", ".join(populated)
            )
        return
    if entry.status != "completed":
        raise ExperimentRegistryValidationError(
            f"{prefix} status must be specified_not_run or completed"
        )
    missing = [field for field in runtime_fields if not getattr(entry, field)]
    if missing:
        raise ExperimentRegistryValidationError(
            f"{prefix} completed experiment is missing evidence: " + ", ".join(missing)
        )
    if _GIT_COMMIT_PATTERN.fullmatch(entry.git_commit) is None:
        raise ExperimentRegistryValidationError(
            f"{prefix} git_commit must be a full Git object ID"
        )
    for field in ("data_manifest_sha256", "execution_profile_sha256"):
        if _SHA256_PATTERN.fullmatch(getattr(entry, field)) is None:
            raise ExperimentRegistryValidationError(f"{prefix} {field} must be SHA-256")
    try:
        metric = float(entry.primary_metric_value)
        seed = int(entry.run_seed)
    except ValueError as error:
        raise ExperimentRegistryValidationError(
            f"{prefix} completed numeric evidence is invalid"
        ) from error
    if not math.isfinite(metric):
        raise ExperimentRegistryValidationError(
            f"{prefix} primary_metric_value must be finite"
        )
    if seed < 0:
        raise ExperimentRegistryValidationError(
            f"{prefix} run_seed must be non-negative"
        )
    if entry.decision not in {"pass", "fail", "unresolved"}:
        raise ExperimentRegistryValidationError(
            f"{prefix} completed decision must be pass, fail, or unresolved"
        )


def _mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise StrategySpecValidationError(f"{field} must be a mapping with string keys")
    return value


def _keys(values: dict[str, object], expected: set[str], field: str) -> None:
    actual = set(values)
    missing = expected - actual
    unexpected = actual - expected
    if missing:
        raise StrategySpecValidationError(
            f"{field} is missing fields: {', '.join(sorted(missing))}"
        )
    if unexpected:
        raise StrategySpecValidationError(
            f"{field} has unexpected fields: {', '.join(sorted(unexpected))}"
        )


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StrategySpecValidationError(f"{field} must be a non-empty string")
    return value


def _literal(value: object, expected: str, field: str) -> str:
    actual = _string(value, field)
    if actual != expected:
        raise StrategySpecValidationError(f"{field} must be {expected}")
    return actual


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise StrategySpecValidationError(f"{field} must be a positive integer")
    return value


def _nonnegative_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StrategySpecValidationError(f"{field} must be a non-negative integer")
    return value


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise StrategySpecValidationError(f"{field} must be a boolean")
    return value


def _finite_decimal(value: object, field: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise StrategySpecValidationError(f"{field} must be a finite decimal")
    try:
        result = Decimal(str(value))
    except InvalidOperation as error:
        raise StrategySpecValidationError(
            f"{field} must be a finite decimal"
        ) from error
    if not result.is_finite():
        raise StrategySpecValidationError(f"{field} must be a finite decimal")
    return result


def _positive_decimal(value: object, field: str) -> Decimal:
    result = _finite_decimal(value, field)
    if result <= 0:
        raise StrategySpecValidationError(f"{field} must be positive")
    return result


def _unit_fraction(value: object, field: str) -> Decimal:
    result = _finite_decimal(value, field)
    if not Decimal("0") < result < Decimal("1"):
        raise StrategySpecValidationError(f"{field} must be strictly between 0 and 1")
    return result


def _string_tuple(value: object, field: str, allow_empty: bool) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        requirement = "a list" if allow_empty else "a non-empty list"
        raise StrategySpecValidationError(f"{field} must be {requirement}")
    return tuple(_string(item, f"{field}[{index}]") for index, item in enumerate(value))


def _json_default(value: Any) -> str:
    if isinstance(value, Decimal):
        return format(value, "f")
    raise TypeError(f"cannot serialize {type(value).__name__}")
