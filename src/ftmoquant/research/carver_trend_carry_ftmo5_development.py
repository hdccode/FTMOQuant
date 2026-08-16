"""DEVELOPMENT-only Carver signal primitives and G0.8 CFD preflight."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

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
from ftmoquant.research.stage_g import frozen_development_folds

EVALUATOR_VERSION = "g1.4g-carver-trend-carry-ftmo5-development-1"
_MAPPING = {
    "EUR/USD.DUKASCOPY": "EUR/USD",
    "XAU_USD.OANDA": "XAU/USD",
    "SPX500_USD.OANDA": "US500.cash",
    "WTICO_USD.OANDA": "USOIL.cash",
    "SOYBN_USD.OANDA": "SOYBEAN.c",
}


class CarverTrendCarryFtmo5EvaluationError(ValueError):
    """Raised when a DEVELOPMENT request cannot preserve frozen semantics."""


@dataclass(frozen=True, slots=True)
class CarverForecasts:
    trend_16_64: pd.Series
    trend_32_128: pd.Series
    trend_64_256: pd.Series
    carry: pd.Series
    combined: pd.Series


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
    volatility = price.pct_change().ewm(span=35, adjust=False, min_periods=2).std()
    raw = (
        price.ewm(span=fast, adjust=False, min_periods=fast).mean()
        - price.ewm(span=slow, adjust=False, min_periods=slow).mean()
    ) / volatility
    return (raw * scalar).clip(lower=-20.0, upper=20.0)


def causal_carry(multiple_prices: pd.DataFrame) -> pd.Series:
    """Pinned-Carver annualised roll/return-vol form; no future rows are used."""
    required = {"PRICE", "CARRY", "PRICE_CONTRACT", "CARRY_CONTRACT"}
    if set(multiple_prices.columns) != required:
        raise CarverTrendCarryFtmo5EvaluationError(
            "multiple-price columns are not exact"
        )
    price = multiple_prices["PRICE"].astype(float)
    carry = multiple_prices["CARRY"].astype(float)
    months = (
        multiple_prices["CARRY_CONTRACT"].astype(float)
        - multiple_prices["PRICE_CONTRACT"].astype(float)
    ) / 100.0
    annualised_roll = (carry - price) / months * 12.0
    annualised_vol = price.pct_change().ewm(
        span=35, adjust=False, min_periods=2
    ).std() * (256.0**0.5)
    raw = annualised_roll / annualised_vol
    return (raw.ewm(span=90, adjust=False, min_periods=90).mean() * 30.0).clip(
        -20.0, 20.0
    )


def combine_forecasts(
    adjusted_price: pd.Series, multiple_prices: pd.DataFrame
) -> CarverForecasts:
    """Apply the frozen weights/FDM; final clipping is intentionally not added."""
    first = causal_ewmac(adjusted_price, 16, 64, 3.75)
    second = causal_ewmac(adjusted_price, 32, 128, 2.65)
    third = causal_ewmac(adjusted_price, 64, 256, 1.87)
    carry = causal_carry(multiple_prices)
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
        _economics_symbol(execution_instrument), price, lots
    )


def stressed_cost(base_cost: float) -> tuple[float, float]:
    """Return frozen base and 1.5x stress cost without a new cost assumption."""
    if base_cost < 0:
        raise CarverTrendCarryFtmo5EvaluationError("base cost must be nonnegative")
    return base_cost, base_cost * 1.5


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


def _frozen_g08_economics() -> G08CfdEconomics:
    economics = load_g08_cfd_economics()
    if economics.semantic_sha256 != G0_8_CFD_ECONOMICS_SHA256:
        raise CarverTrendCarryFtmo5EvaluationError("frozen G0.8 economics SHA drifted")
    return economics


def _economics_symbol(execution_instrument: str) -> str:
    try:
        return _MAPPING[execution_instrument]
    except KeyError as error:
        raise CarverTrendCarryFtmo5EvaluationError(
            f"unknown frozen execution instrument: {execution_instrument}"
        ) from error


def _root(value: str) -> tuple[str, Path]:
    instrument, separator, path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("execution root must be INSTRUMENT=PATH")
    return instrument, Path(path)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the provenance/execution preflight; CFD execution remains fail-closed."""
    parser = argparse.ArgumentParser(
        description="Preflight frozen Carver DEVELOPMENT execution compatibility"
    )
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--execution-root", action="append", type=_root, required=True)
    args = parser.parse_args(argv)
    validate_development_request(args.reference_root, dict(args.execution_root))


def _price_series(value: pd.Series) -> None:
    if (
        not value.index.is_monotonic_increasing
        or value.index.has_duplicates
        or (value <= 0).any()
    ):
        raise CarverTrendCarryFtmo5EvaluationError(
            "adjusted futures price series is not causal/valid"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
