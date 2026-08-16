"""Frozen DEVELOPMENT-only Carver FTMO5 evaluator and synthetic-safe core."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]
from nautilus_trader.model import Bar, CurrencyPair
from nautilus_trader.persistence import ParquetDataCatalog

from ftmoquant.backtest.execution_harness import _sha256_tree
from ftmoquant.prop_rules.g0_8_cfd_economics import (
    G0_8_CFD_ECONOMICS_SHA256,
    G08CfdEconomics,
    G08CfdEconomicsError,
    load_g08_cfd_economics,
)
from ftmoquant.research.carver_trend_carry_ftmo5_spec import (
    CARVER_TREND_CARRY_FTMO5_CONFIG_SHA256,
    CarverTrendCarryFtmo5Spec,
    load_carver_trend_carry_ftmo5_spec,
)
from ftmoquant.research.stage_g import (
    DEVELOPMENT_END_EXCLUSIVE,
    DEVELOPMENT_START,
    frozen_development_folds,
)
from ftmoquant.research.statistics import (
    StationaryBootstrapConfig,
    result_as_dict,
    stationary_bootstrap_confidence_interval,
)

EVALUATOR_VERSION = "g1.4g-carver-trend-carry-ftmo5-development-2"
_MAPPING = {
    "EUR/USD.DUKASCOPY": "EUR/USD",
    "XAU_USD.OANDA": "XAU/USD",
    "SPX500_USD.OANDA": "US500.cash",
    "WTICO_USD.OANDA": "USOIL.cash",
    "SOYBN_USD.OANDA": "SOYBEAN.c",
}
_EXECUTION_PRICE_SCALE_TO_FTMO = {"SOYBN_USD.OANDA": Decimal("100")}


class CarverTrendCarryFtmo5EvaluationError(ValueError):
    """Raised when a DEVELOPMENT request cannot preserve frozen semantics."""


@dataclass(frozen=True, slots=True)
class CarverForecasts:
    trend_16_64: pd.Series
    trend_32_128: pd.Series
    trend_64_256: pd.Series
    carry: pd.Series
    combined: pd.Series


@dataclass(frozen=True, slots=True)
class ExecutionObservation:
    """One genuine provider BID/ASK close and its causal availability time."""

    candle_start_utc: datetime
    available_at_utc: datetime
    bid_close: Decimal
    ask_close: Decimal


@dataclass(frozen=True, slots=True)
class DesiredPosition:
    instrument: str
    signal_timestamp_utc: datetime
    combined_forecast: Decimal
    daily_proxy_price_volatility: Decimal
    desired_lots: Decimal


@dataclass(frozen=True, slots=True)
class ExecutedTransition:
    instrument: str
    signal_timestamp_utc: datetime
    execution_timestamp_utc: datetime
    prior_lots: Decimal
    desired_lots: Decimal
    delta_lots: Decimal
    bid: Decimal
    ask: Decimal
    fill_price: Decimal
    observed_spread_cost: Decimal
    commission: Decimal
    traded_notional: Decimal


@dataclass(frozen=True, slots=True)
class SyntheticFoldInput:
    """Fully artificial fold input used by tests and the dry-run only."""

    fold_id: str
    desired_positions: tuple[DesiredPosition, ...]
    execution_observations: Mapping[str, tuple[ExecutionObservation, ...]]
    compare_start_utc: datetime
    compare_end_exclusive_utc: datetime


@dataclass(slots=True)
class _InstrumentState:
    lots: Decimal = Decimal(0)
    last_mid: Decimal | None = None
    cumulative_gross_pnl: Decimal = Decimal(0)
    cumulative_cost: Decimal = Decimal(0)
    cumulative_turnover: Decimal = Decimal(0)


def verify_reference_sources(
    root: Path, spec: CarverTrendCarryFtmo5Spec
) -> dict[str, str]:
    """Verify the ten frozen reference inputs without parsing their content."""
    sources = spec.canonical_document["provenance"]["source_sha256"]
    actual: dict[str, str] = {}
    for relative, expected in sorted(sources.items()):
        path = root / relative
        if not path.is_file():
            raise CarverTrendCarryFtmo5EvaluationError(
                f"missing reference source: {relative}"
            )
        digest = _sha256(path)
        if digest != expected:
            raise CarverTrendCarryFtmo5EvaluationError(
                f"reference SHA mismatch: {relative}"
            )
        actual[relative] = digest
    return actual


def causal_ewmac(price: pd.Series, fast: int, slow: int, scalar: float) -> pd.Series:
    """Pinned-Carver EWMAC form using causal EWMAs and causal return volatility."""
    _price_series(price)
    volatility = pinned_mixed_daily_price_volatility(price)
    raw = (
        price.ewm(span=fast, adjust=True, min_periods=1).mean()
        - price.ewm(span=slow, adjust=True, min_periods=1).mean()
    ) / volatility
    return (raw * scalar).clip(lower=-20.0, upper=20.0)


def causal_carry(
    multiple_prices: pd.DataFrame, adjusted_price: pd.Series | None = None
) -> pd.Series:
    """Pinned-Carver annualised roll/return-vol form; no future rows are used."""
    required = {"PRICE", "CARRY", "PRICE_CONTRACT", "CARRY_CONTRACT"}
    if set(multiple_prices.columns) != required:
        raise CarverTrendCarryFtmo5EvaluationError(
            "multiple-price columns are not exact"
        )
    frame = multiple_prices.astype(float)
    price = _daily_last(
        frame["PRICE"] if adjusted_price is None else adjusted_price.astype(float)
    )
    price_contract = frame["PRICE_CONTRACT"]
    carry_contract = frame["CARRY_CONTRACT"]
    difference = _contract_year_fraction(carry_contract) - _contract_year_fraction(
        price_contract
    )
    minimum = 1.0 / 365.25
    difference = difference.where(difference.abs() >= minimum, minimum)
    raw_roll = (frame["PRICE"] - frame["CARRY"]).replace(0.0, float("nan"))
    annualised_roll = (raw_roll / difference).resample("1D").mean()
    annualised_vol = pinned_mixed_daily_price_volatility(price) * 16.0
    raw = annualised_roll / annualised_vol
    return (raw.ewm(span=90, adjust=False, min_periods=90).mean() * 30.0).clip(
        -20.0, 20.0
    )


def pinned_mixed_daily_price_volatility(price: pd.Series) -> pd.Series:
    """Return pinned mixed price-unit volatility without noncausal backfill."""
    _price_series(price)
    short = price.diff().ewm(adjust=True, span=35, min_periods=10).std()
    slow = short.ewm(span=20 * 256, adjust=True).mean()
    mixed = slow * 0.35 + short * 0.65
    mixed = mixed.where(mixed >= 1e-10, 1e-10)
    return mixed.ffill()


def combine_forecasts(
    adjusted_price: pd.Series, multiple_prices: pd.DataFrame
) -> CarverForecasts:
    """Apply the frozen weights/FDM; final clipping is intentionally not added."""
    daily_price = _daily_last(adjusted_price.astype(float))
    first = causal_ewmac(daily_price, 16, 64, 3.75)
    second = causal_ewmac(daily_price, 32, 128, 2.65)
    third = causal_ewmac(daily_price, 64, 256, 1.87)
    carry = causal_carry(multiple_prices, daily_price)
    combined = (first * 0.21 + second * 0.08 + third * 0.21 + carry * 0.50) * 1.31
    return CarverForecasts(first, second, third, carry, combined)


def comparison_fold(timestamp: datetime) -> str | None:
    """Warm-up timestamps return None; comparison timestamps get one frozen fold."""
    timestamp = timestamp.astimezone(UTC)
    matches = [
        fold.fold_id
        for fold in frozen_development_folds().folds
        if fold.compare_start_utc <= timestamp < fold.compare_end_exclusive_utc
    ]
    if len(matches) > 1:
        raise CarverTrendCarryFtmo5EvaluationError("ambiguous DEVELOPMENT fold")
    return matches[0] if matches else None


def first_strictly_later_execution(
    signal_timestamp: datetime, execution_timestamps: Sequence[datetime]
) -> datetime | None:
    """Choose the first executable CFD observation strictly after signal time."""
    ordered = tuple(execution_timestamps)
    if any(item.tzinfo is None for item in ordered) or any(
        a >= b for a, b in zip(ordered, ordered[1:])
    ):
        raise CarverTrendCarryFtmo5EvaluationError(
            "execution timestamps must be strictly increasing UTC"
        )
    return next((item for item in ordered if item > signal_timestamp), None)


def first_eligible_cfd_execution(
    signal_timestamp: datetime,
    execution_timestamps: Sequence[datetime],
    execution_instrument: str,
    economics: G08CfdEconomics,
) -> datetime | None:
    """Require strict later causality and a G0.8 FTMO trading-session opening."""
    economics_symbol = _economics_symbol(execution_instrument)
    if first_strictly_later_execution(signal_timestamp, execution_timestamps) is None:
        return None
    eligible = [
        timestamp
        for timestamp in execution_timestamps
        if timestamp > signal_timestamp
        and economics.is_session_eligible(economics_symbol, timestamp)
    ]
    return eligible[0] if eligible else None


def cfd_net_pnl(
    economics: G08CfdEconomics,
    execution_instrument: str,
    direction: int,
    entry: Decimal,
    exit: Decimal,
    lots: Decimal,
) -> Decimal:
    """Gross CFD P/L less independent entry/exit commissions; spread is observed."""
    symbol = _economics_symbol(execution_instrument)
    entry = normalize_execution_price(execution_instrument, entry)
    exit = normalize_execution_price(execution_instrument, exit)
    return (
        economics.gross_pnl(symbol, direction, entry, exit, lots)
        - economics.side_commission(symbol, entry, lots)
        - economics.side_commission(symbol, exit, lots)
    )


def cfd_margin_requirement(
    economics: G08CfdEconomics, execution_instrument: str, price: Decimal, lots: Decimal
) -> Decimal:
    """Use frozen swing margin without quantizing continuous research lots."""
    return economics.margin_requirement(
        _economics_symbol(execution_instrument),
        normalize_execution_price(execution_instrument, price),
        lots,
    )


def normalize_execution_price(execution_instrument: str, raw_price: Decimal) -> Decimal:
    """Convert the frozen provider price quote to the FTMO G0.8 economics scale."""
    _economics_symbol(execution_instrument)
    if raw_price <= 0:
        raise CarverTrendCarryFtmo5EvaluationError("execution price must be positive")
    return raw_price * _EXECUTION_PRICE_SCALE_TO_FTMO.get(
        execution_instrument, Decimal("1")
    )


def stressed_cost(base_cost: float) -> tuple[float, float]:
    """Return frozen base and 1.5x stress cost without a new cost assumption."""
    if base_cost < 0:
        raise CarverTrendCarryFtmo5EvaluationError("base cost must be nonnegative")
    return base_cost, base_cost * 1.5


def forecast_to_desired_lots(
    *,
    combined_forecast: Decimal,
    daily_proxy_price_volatility: Decimal,
    contract_size: Decimal,
    instrument_weight: Decimal,
    spec: CarverTrendCarryFtmo5Spec,
) -> Decimal:
    """Apply the frozen Carver forecast-to-continuous-CFD-lots equation."""
    if (
        not combined_forecast.is_finite()
        or not daily_proxy_price_volatility.is_finite()
        or daily_proxy_price_volatility <= 0
        or contract_size <= 0
    ):
        raise CarverTrendCarryFtmo5EvaluationError("invalid position-sizing input")
    evaluator = spec.evaluator
    daily_cash_target = (
        evaluator.research_capital
        * evaluator.annual_volatility_target
        / Decimal(evaluator.business_days_per_year).sqrt()
    )
    average_position = daily_cash_target / (
        daily_proxy_price_volatility * contract_size
    )
    return (
        average_position
        * combined_forecast
        / evaluator.average_absolute_forecast
        * instrument_weight
        * evaluator.instrument_diversification_multiplier
    )


def build_desired_positions(
    instrument: str,
    forecasts: pd.Series,
    proxy_daily_volatility: pd.Series,
    economics: G08CfdEconomics,
    spec: CarverTrendCarryFtmo5Spec,
) -> tuple[DesiredPosition, ...]:
    """Create one causal target at the next UTC midnight for each completed day."""
    symbol = _economics_symbol(instrument)
    futures_code = _futures_code(instrument)
    weight = dict(spec.evaluator.instrument_weights)[futures_code]
    contract_size = economics.contract(symbol).contract_size
    result: list[DesiredPosition] = []
    for timestamp, forecast in forecasts.items():
        if pd.isna(forecast):
            continue
        day = _as_utc_datetime(timestamp)
        volatility = proxy_daily_volatility.get(day)
        if volatility is None or pd.isna(volatility):
            continue
        signal_timestamp = datetime.combine(
            day.date() + timedelta(days=1), time.min, tzinfo=UTC
        )
        if not DEVELOPMENT_START <= signal_timestamp < DEVELOPMENT_END_EXCLUSIVE:
            continue
        forecast_decimal = Decimal(str(float(forecast)))
        volatility_decimal = Decimal(str(float(volatility)))
        result.append(
            DesiredPosition(
                instrument=instrument,
                signal_timestamp_utc=signal_timestamp,
                combined_forecast=forecast_decimal,
                daily_proxy_price_volatility=volatility_decimal,
                desired_lots=forecast_to_desired_lots(
                    combined_forecast=forecast_decimal,
                    daily_proxy_price_volatility=volatility_decimal,
                    contract_size=contract_size,
                    instrument_weight=weight,
                    spec=spec,
                ),
            )
        )
    return tuple(result)


def evaluate_synthetic_fold(
    fold_input: SyntheticFoldInput,
    *,
    spec: CarverTrendCarryFtmo5Spec | None = None,
    economics: G08CfdEconomics | None = None,
) -> dict[str, Any]:
    """Run the complete accounting pipeline on caller-supplied artificial data."""
    frozen_spec = spec or load_carver_trend_carry_ftmo5_spec()
    frozen_economics = economics or _frozen_g08_economics()
    if fold_input.compare_start_utc >= fold_input.compare_end_exclusive_utc:
        raise CarverTrendCarryFtmo5EvaluationError("invalid synthetic fold boundary")
    observations = {
        instrument: _validated_observations(instrument, values)
        for instrument, values in fold_input.execution_observations.items()
    }
    if set(observations) != set(_MAPPING):
        raise CarverTrendCarryFtmo5EvaluationError(
            "synthetic execution observations must cover all five instruments"
        )
    scheduled: list[tuple[datetime, str, DesiredPosition, ExecutionObservation]] = []
    for desired in sorted(
        fold_input.desired_positions,
        key=lambda item: (item.signal_timestamp_utc, item.instrument),
    ):
        found = _first_eligible_observation(
            desired.signal_timestamp_utc,
            observations[desired.instrument],
            desired.instrument,
            frozen_economics,
        )
        if found is None:
            raise CarverTrendCarryFtmo5EvaluationError(
                f"no later eligible execution for {desired.instrument}"
            )
        if found.available_at_utc >= fold_input.compare_end_exclusive_utc:
            continue
        scheduled.append((found.available_at_utc, desired.instrument, desired, found))
    marks = _midnight_marks(
        fold_input.compare_start_utc, fold_input.compare_end_exclusive_utc
    )
    timeline: list[tuple[datetime, int, str, Any]] = [
        (timestamp, 0, "", None) for timestamp in marks
    ]
    timeline.extend(
        (timestamp, 1, instrument, (desired, observation))
        for timestamp, instrument, desired, observation in scheduled
    )
    timeline.sort(key=lambda item: (item[0], item[1], item[2]))
    states = {instrument: _InstrumentState() for instrument in _MAPPING}
    transitions: list[ExecutedTransition] = []
    snapshots: list[tuple[datetime, dict[str, Decimal], dict[str, Decimal]]] = []
    for timestamp, kind, instrument, payload in timeline:
        if kind == 0:
            pnl, costs = _liquidation_snapshot(
                timestamp,
                observations,
                states,
                frozen_economics,
                final=timestamp == fold_input.compare_end_exclusive_utc,
            )
            snapshots.append((timestamp, pnl, costs))
            continue
        desired, observation = cast(
            tuple[DesiredPosition, ExecutionObservation], payload
        )
        _refresh_open_states_at(timestamp, observations, states, frozen_economics)
        transition = _execute_transition(
            desired,
            observation,
            states,
            frozen_economics,
            frozen_spec.evaluator.research_capital,
        )
        if transition is not None:
            transitions.append(transition)
    rows = _daily_rows_from_snapshots(
        fold_input.fold_id,
        snapshots,
        frozen_spec.evaluator.research_capital,
    )
    if not rows:
        raise CarverTrendCarryFtmo5EvaluationError("fold has no scored daily returns")
    summary = _summarize_fold(fold_input.fold_id, rows, transitions, frozen_spec)
    return {
        "fold_id": fold_input.fold_id,
        "daily_rows": rows,
        "transitions": [_transition_dict(item) for item in transitions],
        "summary": summary,
    }


def summarize_development(
    fold_results: Sequence[Mapping[str, Any]],
    *,
    spec: CarverTrendCarryFtmo5Spec | None = None,
) -> dict[str, Any]:
    """Pool three frozen synthetic/real fold outputs and apply the mechanical gate."""
    frozen_spec = spec or load_carver_trend_carry_ftmo5_spec()
    expected = tuple(fold.fold_id for fold in frozen_development_folds().folds)
    if tuple(str(item["fold_id"]) for item in fold_results) != expected:
        raise CarverTrendCarryFtmo5EvaluationError("fold results are not exact/ordered")
    rows = [
        row
        for result in fold_results
        for row in cast(list[dict[str, Any]], result["daily_rows"])
    ]
    values = [float(row["net_return"]) for row in rows]
    stressed = [float(row["cost_stress_1_5x_return"]) for row in rows]
    if len(values) < frozen_spec.evaluator.bootstrap_block_size:
        raise CarverTrendCarryFtmo5EvaluationError(
            "pooled series is shorter than frozen bootstrap block size"
        )
    series = pd.Series(
        values,
        index=pd.Index(
            [f"{row['fold_id']}:{row['session_date']}" for row in rows],
            name="fold_session",
        ),
        name="carver_trend_carry_ftmo5_v1_daily_net_return",
    )
    bootstrap = stationary_bootstrap_confidence_interval(
        series,
        StationaryBootstrapConfig(
            block_size=frozen_spec.evaluator.bootstrap_block_size,
            repetitions=frozen_spec.evaluator.bootstrap_repetitions,
            seed=frozen_spec.evaluator.bootstrap_seed,
            confidence_level=frozen_spec.evaluator.bootstrap_confidence_level,
            method=cast(Any, frozen_spec.evaluator.bootstrap_method),
        ),
    )
    summaries = [cast(dict[str, Any], item["summary"]) for item in fold_results]
    fold_means = [float(item["mean_daily_net_return"]) for item in summaries]
    pooled_mean = sum(values) / len(values)
    stressed_mean = sum(stressed) / len(stressed)
    ordered_fold_means = sorted(fold_means)
    per_instrument = {
        instrument: sum(
            float(
                cast(Mapping[str, float], row["per_instrument_net_return"])[instrument]
            )
            for row in rows
        )
        for instrument in _MAPPING
    }
    cumulative_net = sum(per_instrument.values())
    hard_failures = [
        failure
        for summary in summaries
        for failure in cast(list[str], summary["hard_failures"])
    ]
    checks = {
        "pooled_mean_net_daily_return_gt_zero": pooled_mean > 0.0,
        "positive_fold_means_at_least_two": sum(item > 0.0 for item in fold_means) >= 2,
        "median_fold_mean_gt_zero": ordered_fold_means[1] > 0.0,
        "pooled_mean_under_1_5x_cost_gt_zero": stressed_mean > 0.0,
        "no_implementation_or_data_integrity_failure": not hard_failures,
    }
    outcome = "PASS_DEVELOPMENT" if all(checks.values()) else "REJECT_RETIRE"
    return {
        "pooled": {
            "observation_count": len(values),
            "mean_daily_net_return": pooled_mean,
            "median_daily_net_return": float(pd.Series(values).median()),
            "annualized_net_sharpe": _annualized_sharpe(
                values, frozen_spec.evaluator.sharpe_annualisation_days
            ),
            "maximum_drawdown": _maximum_drawdown(values),
            "turnover": sum(float(item.get("turnover", 0.0)) for item in summaries),
            "mean_cost_stress_1_5x_return": stressed_mean,
            "bootstrap_mean_confidence_interval": result_as_dict(bootstrap),
            "per_instrument_cumulative_net_return": per_instrument,
            "per_instrument_share_of_net_pnl": {
                key: (None if cumulative_net == 0 else value / cumulative_net)
                for key, value in per_instrument.items()
            },
        },
        "folds": summaries,
        "positive_fold_mean_count": sum(item > 0.0 for item in fold_means),
        "median_fold_mean": ordered_fold_means[1],
        "worst_fold_by_mean": summaries[fold_means.index(min(fold_means))]["fold_id"],
        "gate_checks": checks,
        "hard_failures": hard_failures,
        "outcome": outcome,
    }


def validate_development_request(
    reference_root: Path, execution_roots: Mapping[str, Path]
) -> dict[str, str]:
    """Validate frozen provenance and G0.8 mapping without loading price data."""
    spec = load_carver_trend_carry_ftmo5_spec()
    if spec.semantic_sha256 != CARVER_TREND_CARRY_FTMO5_CONFIG_SHA256:
        raise CarverTrendCarryFtmo5EvaluationError("candidate semantic SHA drifted")
    verify_reference_sources(reference_root, spec)
    if set(execution_roots) != set(_MAPPING):
        raise CarverTrendCarryFtmo5EvaluationError(
            "execution roots do not match frozen five-CFD mapping"
        )
    economics = _frozen_g08_economics()
    for instrument, symbol in _MAPPING.items():
        try:
            economics.contract(symbol)
        except G08CfdEconomicsError as error:
            raise CarverTrendCarryFtmo5EvaluationError(
                f"missing frozen G0.8 economics for {instrument}"
            ) from error
    return {
        "g0_8_semantic_sha256": economics.semantic_sha256,
        "rollover_status": "UNMODELLED",
        "rollover_warning": economics.rollover_warning,
        "results_economics_status": "pre_rollover_not_fully_deployment_calibrated",
    }


def verify_frozen_development_inputs(
    reference_root: Path,
    eur_development_root: Path,
    oanda_cache_root: Path,
) -> dict[str, Any]:
    """Verify every preregistered DEVELOPMENT input without reading returns."""
    spec = load_carver_trend_carry_ftmo5_spec()
    if spec.semantic_sha256 != CARVER_TREND_CARRY_FTMO5_CONFIG_SHA256:
        raise CarverTrendCarryFtmo5EvaluationError("candidate semantic SHA drifted")
    hashes = verify_reference_sources(reference_root, spec)
    contract = cast(
        dict[str, Any],
        spec.canonical_document["development_evaluator"]["input_contract"],
    )
    _forbid_sealed_path(eur_development_root)
    _forbid_sealed_path(oanda_cache_root)
    expected_eur = Path(str(contract["eur_development_root"])).resolve()
    expected_oanda = Path(str(contract["oanda_cache_root"])).resolve()
    if eur_development_root.resolve() != expected_eur:
        raise CarverTrendCarryFtmo5EvaluationError("EUR DEVELOPMENT root drifted")
    if oanda_cache_root.resolve() != expected_oanda:
        raise CarverTrendCarryFtmo5EvaluationError("OANDA DEVELOPMENT root drifted")
    split_manifest = eur_development_root / "ftmoquant_split_view.json"
    if _sha256(split_manifest) != contract["eur_split_manifest_sha256"]:
        raise CarverTrendCarryFtmo5EvaluationError("EUR split manifest SHA drifted")
    split = _json_object(split_manifest)
    split_range = cast(dict[str, Any], split.get("range"))
    if (
        split_range.get("start_utc") != "2019-03-11T00:00:00Z"
        or split_range.get("end_exclusive_utc") != "2023-04-11T00:00:00Z"
        or split.get("holdout_rows") != 0
        or split.get("split") != "development"
        or split.get("access_policy") != "candidate_read_only"
    ):
        raise CarverTrendCarryFtmo5EvaluationError("EUR split boundary drifted")
    catalog_hash = _sha256_tree(eur_development_root / "catalog")
    if catalog_hash != contract["eur_catalog_tree_sha256"]:
        raise CarverTrendCarryFtmo5EvaluationError("EUR catalog tree SHA drifted")
    manifest = oanda_cache_root / "oanda_development_manifest.json"
    if _sha256(manifest) != contract["oanda_manifest_sha256"]:
        raise CarverTrendCarryFtmo5EvaluationError("OANDA manifest SHA drifted")
    manifest_document = _json_object(manifest)
    if (
        manifest_document.get("carver_semantic_sha256")
        != contract["oanda_acquisition_carver_semantic_sha256"]
    ):
        raise CarverTrendCarryFtmo5EvaluationError(
            "OANDA acquisition-time Carver SHA drifted"
        )
    processed_expected = cast(dict[str, str], contract["oanda_processed_sha256"])
    qa_expected = cast(dict[str, str], contract["oanda_qa_sha256"])
    oanda: dict[str, Any] = {}
    for provider in ("XAU_USD", "SPX500_USD", "WTICO_USD", "SOYBN_USD"):
        processed = oanda_cache_root / "processed" / f"{provider}_M1_bid_ask.csv"
        qa = oanda_cache_root / "qa" / f"{provider}_M1_bid_ask_qa.json"
        processed_hash = _sha256(processed)
        qa_hash = _sha256(qa)
        report = _json_object(qa)
        if (
            processed_hash != processed_expected[provider]
            or qa_hash != qa_expected[provider]
            or report.get("processed_sha256") != processed_hash
            or report.get("research_ready") is not True
            or cast(dict[str, Any], report.get("acquisition_completeness")).get(
                "complete"
            )
            is not True
            or cast(dict[str, Any], report.get("strategy_observation_usability")).get(
                "usable_without_semantic_change"
            )
            is not True
        ):
            raise CarverTrendCarryFtmo5EvaluationError(
                f"OANDA frozen readiness/hash drifted: {provider}"
            )
        oanda[provider] = {
            "processed_sha256": processed_hash,
            "qa_sha256": qa_hash,
        }
    return {
        "reference_source_sha256": hashes,
        "eur_split_manifest_sha256": _sha256(split_manifest),
        "eur_catalog_tree_sha256": catalog_hash,
        "oanda_manifest_sha256": _sha256(manifest),
        "oanda": oanda,
    }


def evaluate_carver_development(
    *,
    reference_root: Path,
    eur_development_root: Path,
    oanda_cache_root: Path,
    output_dir: Path,
    run_timestamp_utc: datetime,
) -> dict[str, Any]:
    """Run the frozen real DEVELOPMENT evaluator; callers own authorization."""
    if output_dir.exists():
        raise FileExistsError(f"evaluation output already exists: {output_dir}")
    if run_timestamp_utc.tzinfo is None or run_timestamp_utc.utcoffset() != timedelta(
        0
    ):
        raise CarverTrendCarryFtmo5EvaluationError("run timestamp must be UTC")
    spec = load_carver_trend_carry_ftmo5_spec()
    inputs = verify_frozen_development_inputs(
        reference_root, eur_development_root, oanda_cache_root
    )
    economics = _frozen_g08_economics()
    observations = _load_execution_observations(eur_development_root, oanda_cache_root)
    all_desired: list[DesiredPosition] = []
    for futures_code, instrument in cast(
        dict[str, str], spec.canonical_document["universe"]["futures_to_execution"]
    ).items():
        adjusted, multiple = _load_reference_market(reference_root, futures_code)
        forecasts = combine_forecasts(adjusted, multiple).combined
        proxy_vol = _proxy_daily_volatility(instrument, observations[instrument])
        all_desired.extend(
            build_desired_positions(instrument, forecasts, proxy_vol, economics, spec)
        )
    fold_results = [
        evaluate_synthetic_fold(
            SyntheticFoldInput(
                fold_id=fold.fold_id,
                desired_positions=tuple(all_desired),
                execution_observations=observations,
                compare_start_utc=fold.compare_start_utc,
                compare_end_exclusive_utc=fold.compare_end_exclusive_utc,
            ),
            spec=spec,
            economics=economics,
        )
        for fold in frozen_development_folds().folds
    ]
    summary = summarize_development(fold_results, spec=spec)
    result = _write_result_artifacts(
        output_dir=output_dir,
        spec=spec,
        economics=economics,
        inputs=inputs,
        fold_results=fold_results,
        summary=summary,
        run_timestamp_utc=run_timestamp_utc,
    )
    return result


def _load_reference_market(
    root: Path, futures_code: str
) -> tuple[pd.Series, pd.DataFrame]:
    adjusted_path = root / "adjusted_prices_csv" / f"{futures_code}.csv"
    multiple_path = root / "multiple_prices_csv" / f"{futures_code}.csv"
    adjusted_frame = pd.read_csv(adjusted_path)
    multiple = pd.read_csv(multiple_path)
    if tuple(adjusted_frame.columns) != ("DATETIME", "price") or tuple(
        multiple.columns
    ) != (
        "DATETIME",
        "CARRY",
        "CARRY_CONTRACT",
        "PRICE",
        "PRICE_CONTRACT",
        "FORWARD",
        "FORWARD_CONTRACT",
    ):
        raise CarverTrendCarryFtmo5EvaluationError(
            f"reference columns drifted: {futures_code}"
        )
    adjusted_frame.index = _utc_index(adjusted_frame.pop("DATETIME"))
    multiple.index = _utc_index(multiple.pop("DATETIME"))
    adjusted = adjusted_frame["price"].astype(float)
    if (
        not adjusted.index.is_monotonic_increasing
        or not multiple.index.is_monotonic_increasing
    ):
        raise CarverTrendCarryFtmo5EvaluationError(
            f"reference timestamps are unordered: {futures_code}"
        )
    adjusted = adjusted.loc[adjusted.index < DEVELOPMENT_END_EXCLUSIVE]
    multiple = multiple.loc[multiple.index < DEVELOPMENT_END_EXCLUSIVE]
    if adjusted.empty or multiple.empty:
        raise CarverTrendCarryFtmo5EvaluationError(
            f"reference data has no pre-endpoint rows: {futures_code}"
        )
    return adjusted, multiple


def _load_execution_observations(
    eur_root: Path, oanda_root: Path
) -> dict[str, tuple[ExecutionObservation, ...]]:
    economics = _frozen_g08_economics()
    result = {"EUR/USD.DUKASCOPY": _load_eur_observations(eur_root)}
    provider_to_instrument = {
        "XAU_USD": "XAU_USD.OANDA",
        "SPX500_USD": "SPX500_USD.OANDA",
        "WTICO_USD": "WTICO_USD.OANDA",
        "SOYBN_USD": "SOYBN_USD.OANDA",
    }
    for provider, instrument in provider_to_instrument.items():
        path = oanda_root / "processed" / f"{provider}_M1_bid_ask.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            expected = (
                "timestamp_utc",
                "bid_open",
                "bid_high",
                "bid_low",
                "bid_close",
                "ask_open",
                "ask_high",
                "ask_low",
                "ask_close",
                "volume",
            )
            if tuple(reader.fieldnames or ()) != expected:
                raise CarverTrendCarryFtmo5EvaluationError(
                    f"OANDA processed columns drifted: {provider}"
                )
            result[instrument] = _evaluation_observation_view(
                instrument, _oanda_observations(reader), economics
            )
    return result


def _load_eur_observations(root: Path) -> tuple[ExecutionObservation, ...]:
    instrument_id = "EUR/USD.DUKASCOPY"
    catalog = ParquetDataCatalog(str(root / "catalog"))
    instruments = catalog.instruments([instrument_id])
    if len(instruments) != 1 or not isinstance(instruments[0], CurrencyPair):
        raise CarverTrendCarryFtmo5EvaluationError("EUR DEVELOPMENT instrument missing")
    start_ns = int(DEVELOPMENT_START.timestamp() * 1_000_000_000)
    end_ns = int(DEVELOPMENT_END_EXCLUSIVE.timestamp() * 1_000_000_000)
    by_side: dict[str, dict[int, Bar]] = {}
    for side in ("BID", "ASK"):
        bar_type = f"{instrument_id}-1-MINUTE-{side}-EXTERNAL"
        queried = catalog.query_bars([bar_type], start=start_ns, end=end_ns - 1)
        selected = {bar.ts_event: bar for bar in queried}
        if len(selected) != len(queried) or not selected:
            raise CarverTrendCarryFtmo5EvaluationError(
                f"EUR {side} bars are duplicate or absent"
            )
        by_side[side] = selected
    if set(by_side["BID"]) != set(by_side["ASK"]):
        raise CarverTrendCarryFtmo5EvaluationError("EUR BID/ASK timestamps differ")
    return _evaluation_observation_view(
        instrument_id,
        _eur_observations(by_side),
        _frozen_g08_economics(),
    )


def _evaluation_observation_view(
    instrument: str,
    values: Iterable[ExecutionObservation],
    economics: G08CfdEconomics,
) -> tuple[ExecutionObservation, ...]:
    """Retain only observations that can be a midnight signal fill or daily mark."""
    _economics_symbol(instrument)
    symbol = _economics_symbol(instrument)
    first_eligible: dict[date, ExecutionObservation] = {}
    last_genuine: dict[date, ExecutionObservation] = {}
    previous: datetime | None = None
    count = 0
    for item in values:
        if (
            item.candle_start_utc.tzinfo is None
            or item.available_at_utc.tzinfo is None
            or item.available_at_utc != item.candle_start_utc + timedelta(minutes=1)
            or item.bid_close <= 0
            or item.ask_close < item.bid_close
            or (previous is not None and item.available_at_utc <= previous)
        ):
            raise CarverTrendCarryFtmo5EvaluationError(
                f"invalid or unordered execution observation: {instrument}"
            )
        day = item.available_at_utc.date()
        last_genuine[day] = item
        if day not in first_eligible and economics.is_session_eligible(
            symbol, item.available_at_utc
        ):
            first_eligible[day] = item
        previous = item.available_at_utc
        count += 1
    if count == 0:
        raise CarverTrendCarryFtmo5EvaluationError(
            f"no execution observations: {instrument}"
        )
    selected = {
        item.available_at_utc: item
        for item in (*first_eligible.values(), *last_genuine.values())
    }
    return tuple(selected[key] for key in sorted(selected))


def _oanda_observations(
    reader: Iterable[Mapping[str, str | None]],
) -> Iterable[ExecutionObservation]:
    for row in reader:
        start = _parse_utc(str(row["timestamp_utc"]))
        yield ExecutionObservation(
            start,
            start + timedelta(minutes=1),
            Decimal(str(row["bid_close"])),
            Decimal(str(row["ask_close"])),
        )


def _eur_observations(
    by_side: Mapping[str, Mapping[int, Bar]],
) -> Iterable[ExecutionObservation]:
    for timestamp in sorted(by_side["BID"]):
        bid = by_side["BID"][timestamp]
        ask = by_side["ASK"][timestamp]
        if bid.ts_init != ask.ts_init or bid.ts_init != timestamp + 60_000_000_000:
            raise CarverTrendCarryFtmo5EvaluationError("EUR bar availability drifted")
        yield ExecutionObservation(
            datetime.fromtimestamp(timestamp / 1_000_000_000, tz=UTC),
            datetime.fromtimestamp(bid.ts_init / 1_000_000_000, tz=UTC),
            bid.close.as_decimal(),
            ask.close.as_decimal(),
        )


def _proxy_daily_volatility(
    instrument: str, observations: Sequence[ExecutionObservation]
) -> pd.Series:
    index = pd.DatetimeIndex([item.available_at_utc for item in observations])
    mid = pd.Series(
        [
            float(sum(_normalized_quote(instrument, item), Decimal(0)) / Decimal(2))
            for item in observations
        ],
        index=index,
    )
    daily = mid.resample("1D").last().dropna()
    return pinned_mixed_daily_price_volatility(daily)


def _write_result_artifacts(
    *,
    output_dir: Path,
    spec: CarverTrendCarryFtmo5Spec,
    economics: G08CfdEconomics,
    inputs: Mapping[str, Any],
    fold_results: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    run_timestamp_utc: datetime,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    daily_rows = [
        row
        for result in fold_results
        for row in cast(list[dict[str, Any]], result["daily_rows"])
    ]
    trade_rows = [
        row
        for result in fold_results
        for row in cast(list[dict[str, Any]], result["transitions"])
    ]
    daily_path = output_dir / "daily_returns.csv"
    trade_path = output_dir / "trades.csv"
    pd.DataFrame(
        [
            {
                **{
                    key: value
                    for key, value in row.items()
                    if key != "per_instrument_net_return"
                },
                **{
                    f"{instrument}_net_return": cast(
                        Mapping[str, float], row["per_instrument_net_return"]
                    )[instrument]
                    for instrument in _MAPPING
                },
            }
            for row in daily_rows
        ]
    ).to_csv(daily_path, index=False, lineterminator="\n")
    pd.DataFrame(trade_rows).to_csv(trade_path, index=False, lineterminator="\n")
    folds = frozen_development_folds()
    result: dict[str, Any] = {
        "schema": spec.evaluator.result_schema,
        "schema_version": 1,
        "provenance": {
            "git_commit": _git_commit(),
            "evaluator_version": EVALUATOR_VERSION,
            "carver_semantic_sha256": spec.semantic_sha256,
            "g0_8_semantic_sha256": economics.semantic_sha256,
            "pysystemtrade_commit": spec.canonical_document["provenance"][
                "pysystemtrade_commit"
            ],
            "development_start_utc": _utc(DEVELOPMENT_START),
            "development_end_exclusive_utc": _utc(DEVELOPMENT_END_EXCLUSIVE),
            "development_folds_sha256": folds.semantic_sha256,
            "input_hashes": inputs,
            "implementation_sha256": {
                "strategy_yaml": _sha256(
                    Path("config/strategies/carver_trend_carry_ftmo5_v1.yaml")
                ),
                "typed_spec": _sha256(
                    Path("src/ftmoquant/research/carver_trend_carry_ftmo5_spec.py")
                ),
                "evaluator": _sha256(
                    Path(
                        "src/ftmoquant/research/carver_trend_carry_ftmo5_development.py"
                    )
                ),
                "g0_8_config": _sha256(
                    Path("config/prop/g0_8_ftmo_cfd_economics_2026-08-15.yaml")
                ),
            },
            "validation_accessed": False,
            "final_holdout_accessed": False,
            "parameter_optimization_occurred": False,
            "rollover_status": "UNMODELLED",
            "rollover_warning": economics.rollover_warning,
            "economics_status": "pre_rollover_not_fully_deployment_calibrated",
            "basis_mismatch": "explicit_no_favorable_basis_assumption",
        },
        "summary": summary,
        "fold_results": [
            cast(dict[str, Any], item["summary"]) for item in fold_results
        ],
        "result_file_sha256": {
            "daily_returns.csv": _sha256(daily_path),
            "trades.csv": _sha256(trade_path),
        },
    }
    result["semantic_sha256"] = _semantic_sha256(result)
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    run_provenance = {
        "schema": "ftmoquant.carver-trend-carry-ftmo5-development-run-v1",
        "execution_timestamp_utc": _utc(run_timestamp_utc),
        "git_commit": _git_commit(),
        "result_json_sha256": _sha256(result_path),
        "deterministic_result_semantic_sha256": result["semantic_sha256"],
    }
    (output_dir / "run_provenance.json").write_text(
        json.dumps(run_provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _frozen_g08_economics() -> G08CfdEconomics:
    economics = load_g08_cfd_economics()
    if economics.semantic_sha256 != G0_8_CFD_ECONOMICS_SHA256:
        raise CarverTrendCarryFtmo5EvaluationError("frozen G0.8 economics SHA drifted")
    return economics


def _validated_observations(
    instrument: str, values: Sequence[ExecutionObservation]
) -> tuple[ExecutionObservation, ...]:
    _economics_symbol(instrument)
    result = tuple(values)
    previous: datetime | None = None
    for item in result:
        if (
            item.candle_start_utc.tzinfo is None
            or item.available_at_utc.tzinfo is None
            or item.available_at_utc != item.candle_start_utc + timedelta(minutes=1)
            or item.bid_close <= 0
            or item.ask_close < item.bid_close
            or (previous is not None and item.available_at_utc <= previous)
        ):
            raise CarverTrendCarryFtmo5EvaluationError(
                f"invalid or unordered execution observation: {instrument}"
            )
        previous = item.available_at_utc
    if not result:
        raise CarverTrendCarryFtmo5EvaluationError(
            f"no execution observations: {instrument}"
        )
    return result


def _first_eligible_observation(
    signal_timestamp: datetime,
    observations: Sequence[ExecutionObservation],
    instrument: str,
    economics: G08CfdEconomics,
) -> ExecutionObservation | None:
    symbol = _economics_symbol(instrument)
    index = _observation_insertion_index(observations, signal_timestamp, right=True)
    for position in range(index, len(observations)):
        item = observations[position]
        if economics.is_session_eligible(symbol, item.available_at_utc):
            return item
    return None


def _midnight_marks(start: datetime, end: datetime) -> tuple[datetime, ...]:
    if start.tzinfo is None or end.tzinfo is None:
        raise CarverTrendCarryFtmo5EvaluationError("fold marks require aware UTC")
    first = datetime.combine(start.date(), time.min, tzinfo=UTC)
    if first < start:
        first += timedelta(days=1)
    result: list[datetime] = []
    current = first
    while current <= end:
        result.append(current)
        current += timedelta(days=1)
    if not result or result[0] != start or result[-1] != end:
        raise CarverTrendCarryFtmo5EvaluationError(
            "comparison folds must start/end at UTC midnight"
        )
    return tuple(result)


def _latest_observation(
    observations: Sequence[ExecutionObservation], timestamp: datetime
) -> ExecutionObservation | None:
    index = _observation_insertion_index(observations, timestamp, right=False) - 1
    return None if index < 0 else observations[index]


def _latest_observation_at_or_before(
    observations: Sequence[ExecutionObservation], timestamp: datetime
) -> ExecutionObservation | None:
    index = _observation_insertion_index(observations, timestamp, right=True) - 1
    return None if index < 0 else observations[index]


def _observation_insertion_index(
    observations: Sequence[ExecutionObservation],
    timestamp: datetime,
    *,
    right: bool,
) -> int:
    lower = 0
    upper = len(observations)
    while lower < upper:
        middle = (lower + upper) // 2
        candidate = observations[middle].available_at_utc
        if candidate < timestamp or (right and candidate == timestamp):
            lower = middle + 1
        else:
            upper = middle
    return lower


def _refresh_open_states_at(
    timestamp: datetime,
    observations: Mapping[str, Sequence[ExecutionObservation]],
    states: Mapping[str, _InstrumentState],
    economics: G08CfdEconomics,
) -> None:
    for instrument, state in states.items():
        if state.lots == 0:
            continue
        observation = _latest_observation_at_or_before(
            observations[instrument], timestamp
        )
        if observation is None:
            raise CarverTrendCarryFtmo5EvaluationError(
                f"open position has no causal mark: {instrument}"
            )
        bid, ask = _normalized_quote(instrument, observation)
        _mark_to_mid(
            state,
            (bid + ask) / 2,
            economics.contract(_economics_symbol(instrument)).contract_size,
        )


def _mark_to_mid(state: _InstrumentState, mid: Decimal, contract_size: Decimal) -> None:
    if state.last_mid is not None and state.lots != 0:
        state.cumulative_gross_pnl += (
            (mid - state.last_mid) * contract_size * state.lots
        )
    state.last_mid = mid


def _execute_transition(
    desired: DesiredPosition,
    observation: ExecutionObservation,
    states: Mapping[str, _InstrumentState],
    economics: G08CfdEconomics,
    capital: Decimal,
) -> ExecutedTransition | None:
    instrument = desired.instrument
    state = states[instrument]
    bid, ask = _normalized_quote(instrument, observation)
    mid = (bid + ask) / 2
    contract = economics.contract(_economics_symbol(instrument))
    _mark_to_mid(state, mid, contract.contract_size)
    delta = desired.desired_lots - state.lots
    if delta == 0:
        return None
    proposed = {key: item.lots for key, item in states.items()}
    proposed[instrument] = desired.desired_lots
    margin = Decimal(0)
    for key, lots in proposed.items():
        if lots == 0:
            continue
        latest = states[key].last_mid
        if latest is None:
            raise CarverTrendCarryFtmo5EvaluationError(
                f"margin calculation lacks causal mark: {key}"
            )
        margin += economics.margin_requirement(
            _economics_symbol(key), latest, abs(lots)
        )
    if margin > capital:
        raise CarverTrendCarryFtmo5EvaluationError(
            "aggregate frozen G0.8 swing margin exceeds research capital"
        )
    fill = ask if delta > 0 else bid
    spread_cost = abs(delta) * contract.contract_size * abs(fill - mid)
    commission = economics.side_commission(contract.symbol, fill, abs(delta))
    traded_notional = abs(fill * contract.contract_size * delta)
    state.cumulative_cost += spread_cost + commission
    state.cumulative_turnover += traded_notional
    prior = state.lots
    state.lots = desired.desired_lots
    return ExecutedTransition(
        instrument=instrument,
        signal_timestamp_utc=desired.signal_timestamp_utc,
        execution_timestamp_utc=observation.available_at_utc,
        prior_lots=prior,
        desired_lots=desired.desired_lots,
        delta_lots=delta,
        bid=bid,
        ask=ask,
        fill_price=fill,
        observed_spread_cost=spread_cost,
        commission=commission,
        traded_notional=traded_notional,
    )


def _liquidation_snapshot(
    timestamp: datetime,
    observations: Mapping[str, Sequence[ExecutionObservation]],
    states: Mapping[str, _InstrumentState],
    economics: G08CfdEconomics,
    *,
    final: bool,
) -> tuple[dict[str, Decimal], dict[str, Decimal]]:
    pnl: dict[str, Decimal] = {}
    costs: dict[str, Decimal] = {}
    for instrument, state in states.items():
        contract = economics.contract(_economics_symbol(instrument))
        observation = _latest_observation(observations[instrument], timestamp)
        if observation is not None:
            bid, ask = _normalized_quote(instrument, observation)
            mid = (bid + ask) / 2
            _mark_to_mid(state, mid, contract.contract_size)
        elif state.lots != 0:
            raise CarverTrendCarryFtmo5EvaluationError(
                f"open position has no daily mark: {instrument}"
            )
        liquidation_adjustment = Decimal(0)
        final_commission = Decimal(0)
        final_spread = Decimal(0)
        if state.lots != 0:
            if observation is None:
                raise AssertionError("nonzero position without observation")
            bid, ask = _normalized_quote(instrument, observation)
            mid = (bid + ask) / 2
            liquidation = bid if state.lots > 0 else ask
            liquidation_adjustment = (
                (liquidation - mid) * contract.contract_size * state.lots
            )
            if final:
                final_spread = abs(liquidation_adjustment)
                final_commission = economics.side_commission(
                    contract.symbol, liquidation, abs(state.lots)
                )
        pnl[instrument] = (
            state.cumulative_gross_pnl
            - state.cumulative_cost
            + liquidation_adjustment
            - final_commission
        )
        costs[instrument] = state.cumulative_cost + final_spread + final_commission
    return pnl, costs


def _daily_rows_from_snapshots(
    fold_id: str,
    snapshots: Sequence[tuple[datetime, Mapping[str, Decimal], Mapping[str, Decimal]]],
    capital: Decimal,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for previous, current in zip(snapshots, snapshots[1:]):
        timestamp, pnl, costs = current
        _, prior_pnl, prior_costs = previous
        by_instrument = {
            instrument: (pnl[instrument] - prior_pnl[instrument]) / capital
            for instrument in _MAPPING
        }
        net_return = sum(by_instrument.values(), Decimal(0))
        realized_cost_return = sum(
            (costs[instrument] - prior_costs[instrument]) / capital
            for instrument in _MAPPING
        )
        rows.append(
            {
                "fold_id": fold_id,
                "session_date": (timestamp - timedelta(microseconds=1))
                .date()
                .isoformat(),
                "mark_boundary_utc": _utc(timestamp),
                "net_return": float(net_return),
                "realized_spread_and_commission_cost_return": float(
                    realized_cost_return
                ),
                "cost_stress_1_5x_return": float(
                    net_return - realized_cost_return / Decimal(2)
                ),
                "per_instrument_net_return": {
                    key: float(value) for key, value in by_instrument.items()
                },
            }
        )
    return rows


def _summarize_fold(
    fold_id: str,
    rows: Sequence[Mapping[str, Any]],
    transitions: Sequence[ExecutedTransition],
    spec: CarverTrendCarryFtmo5Spec,
) -> dict[str, Any]:
    values = [float(item["net_return"]) for item in rows]
    stressed = [float(item["cost_stress_1_5x_return"]) for item in rows]
    capital = spec.evaluator.research_capital
    contributions = {
        instrument: sum(
            float(
                cast(Mapping[str, float], row["per_instrument_net_return"])[instrument]
            )
            for row in rows
        )
        for instrument in _MAPPING
    }
    pooled = sum(contributions.values())
    return {
        "fold_id": fold_id,
        "observation_count": len(values),
        "mean_daily_net_return": sum(values) / len(values),
        "median_daily_net_return": float(pd.Series(values).median()),
        "annualized_net_sharpe": _annualized_sharpe(
            values, spec.evaluator.sharpe_annualisation_days
        ),
        "maximum_drawdown": _maximum_drawdown(values),
        "turnover": float(
            sum((item.traded_notional for item in transitions), Decimal(0)) / capital
        ),
        "mean_cost_stress_1_5x_return": sum(stressed) / len(stressed),
        "per_instrument_cumulative_net_return": contributions,
        "per_instrument_share_of_net_pnl": {
            key: (None if pooled == 0 else value / pooled)
            for key, value in contributions.items()
        },
        "trend_carry_decomposition": "not_reported_not_gate_required",
        "hard_failures": [],
    }


def _annualized_sharpe(
    values: Sequence[float], annualisation_days: int
) -> float | None:
    if len(values) < 2:
        return None
    series = pd.Series(values, dtype=float)
    standard_deviation = float(series.std(ddof=1))
    if not math.isfinite(standard_deviation) or standard_deviation <= 0:
        return None
    value = math.sqrt(annualisation_days) * float(series.mean()) / standard_deviation
    return value if math.isfinite(value) else None


def _maximum_drawdown(values: Sequence[float]) -> float:
    wealth = 1.0
    peak = 1.0
    result = 0.0
    for value in values:
        wealth *= 1.0 + value
        peak = max(peak, wealth)
        result = max(result, (peak - wealth) / peak)
    return result


def _transition_dict(item: ExecutedTransition) -> dict[str, Any]:
    return {
        "instrument": item.instrument,
        "signal_timestamp_utc": _utc(item.signal_timestamp_utc),
        "execution_timestamp_utc": _utc(item.execution_timestamp_utc),
        "prior_lots": str(item.prior_lots),
        "desired_lots": str(item.desired_lots),
        "delta_lots": str(item.delta_lots),
        "bid": str(item.bid),
        "ask": str(item.ask),
        "fill_price": str(item.fill_price),
        "observed_spread_cost": str(item.observed_spread_cost),
        "commission": str(item.commission),
        "traded_notional": str(item.traded_notional),
    }


def _normalized_quote(
    instrument: str, observation: ExecutionObservation
) -> tuple[Decimal, Decimal]:
    bid = normalize_execution_price(instrument, observation.bid_close)
    ask = normalize_execution_price(instrument, observation.ask_close)
    if ask < bid:
        raise CarverTrendCarryFtmo5EvaluationError("normalized ask is below bid")
    return bid, ask


def _economics_symbol(execution_instrument: str) -> str:
    try:
        return _MAPPING[execution_instrument]
    except KeyError as error:
        raise CarverTrendCarryFtmo5EvaluationError(
            f"unknown frozen execution instrument: {execution_instrument}"
        ) from error


def main(argv: Sequence[str] | None = None) -> None:
    """Run the exact frozen DEVELOPMENT evaluator; no sealed-data arguments exist."""
    parser = argparse.ArgumentParser(
        description="Evaluate frozen Carver FTMO5 on DEVELOPMENT only"
    )
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--eur-development-root", type=Path, required=True)
    parser.add_argument("--oanda-cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-timestamp-utc", type=_parse_utc, required=True)
    args = parser.parse_args(argv)
    evaluate_carver_development(
        reference_root=cast(Path, args.reference_root),
        eur_development_root=cast(Path, args.eur_development_root),
        oanda_cache_root=cast(Path, args.oanda_cache_root),
        output_dir=cast(Path, args.output),
        run_timestamp_utc=cast(datetime, args.run_timestamp_utc),
    )


def _price_series(value: pd.Series) -> None:
    if (
        not value.index.is_monotonic_increasing
        or value.index.has_duplicates
        or (value <= 0).any()
    ):
        raise CarverTrendCarryFtmo5EvaluationError(
            "adjusted futures price series is not causal/valid"
        )


def _daily_last(value: pd.Series) -> pd.Series:
    if not isinstance(value.index, pd.DatetimeIndex):
        raise CarverTrendCarryFtmo5EvaluationError("price index must be datetime")
    return value.resample("1D").last().dropna()


def _contract_year_fraction(value: pd.Series) -> pd.Series:
    numeric = value.astype(float)
    years = numeric.floordiv(10000)
    months = numeric.mod(10000) / 100.0
    return years + months / 12.0


def _futures_code(execution_instrument: str) -> str:
    reverse = {
        "EUR/USD.DUKASCOPY": "EUR",
        "XAU_USD.OANDA": "GOLD",
        "SPX500_USD.OANDA": "SP500",
        "WTICO_USD.OANDA": "CRUDE_W",
        "SOYBN_USD.OANDA": "SOYBEAN",
    }
    try:
        return reverse[execution_instrument]
    except KeyError as error:
        raise CarverTrendCarryFtmo5EvaluationError(
            f"unknown frozen execution instrument: {execution_instrument}"
        ) from error


def _utc_index(values: pd.Series) -> pd.DatetimeIndex:
    try:
        result = pd.DatetimeIndex(pd.to_datetime(values, utc=True, errors="raise"))
    except (TypeError, ValueError) as error:
        raise CarverTrendCarryFtmo5EvaluationError(
            "reference timestamp parsing failed"
        ) from error
    if result.has_duplicates:
        raise CarverTrendCarryFtmo5EvaluationError("reference timestamps duplicate")
    return result


def _as_utc_datetime(value: Any) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(UTC)
    else:
        timestamp = timestamp.tz_convert(UTC)
    return cast(datetime, timestamp.to_pydatetime())


def _parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("timestamp must be RFC3339 UTC") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise argparse.ArgumentTypeError("timestamp must be RFC3339 UTC")
    return parsed.astimezone(UTC)


def _utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise CarverTrendCarryFtmo5EvaluationError("timestamp must be aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CarverTrendCarryFtmo5EvaluationError(
            f"invalid frozen JSON artifact: {path}"
        ) from error
    if not isinstance(value, dict):
        raise CarverTrendCarryFtmo5EvaluationError(
            f"frozen JSON artifact is not an object: {path}"
        )
    return cast(dict[str, Any], value)


def _forbid_sealed_path(path: Path) -> None:
    lowered = str(path).lower()
    if "validation" in lowered or "holdout" in lowered:
        raise CarverTrendCarryFtmo5EvaluationError(
            "validation and holdout paths are forbidden"
        )


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _semantic_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
