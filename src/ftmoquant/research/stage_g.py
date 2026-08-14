"""Leakage-resistant, strategy-free infrastructure for the G1.4B tournament.

This module deliberately has no candidate signals, performance readers, backtest
runner, validation accessor, or holdout accessor.  Its data-facing surface is
limited to frozen readiness and DEVELOPMENT split metadata.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from ftmoquant.backtest.execution_harness import (
    ExecutionProfile,
    validate_execution_profile,
)
from ftmoquant.data.universe_plan import (
    ResearchUniversePlan,
    load_research_universe_plan,
)
from ftmoquant.data.universe_readiness import (
    SPLIT_VIEW_FILENAME,
    UNIVERSE_READINESS_FILENAME,
)

STAGE_G_VERSION = "g1.4b-stage-g-1"
FROZEN_UNIVERSE_ID = "g1_4_fx_usd_liquid_v1"
FROZEN_UNIVERSE_PLAN_SHA256 = (
    "16c29df7bae3bde0a9e64e2cc7758f158186182a3d7a464347791400abcfff69"
)
FROZEN_UNIVERSE_READINESS_SHA256 = (
    "e6d511bb7bbea6b80c42b00c3f1b4c79ea35cd4532d5f71b69ee8f43bea3f8a0"
)
FROZEN_INSTRUMENT_IDS = (
    "EUR/USD.DUKASCOPY",
    "GBP/USD.DUKASCOPY",
)
DEVELOPMENT_START = datetime(2019, 3, 11, tzinfo=UTC)
DEVELOPMENT_END_EXCLUSIVE = datetime(2023, 4, 11, tzinfo=UTC)
VALIDATION_START = DEVELOPMENT_END_EXCLUSIVE
HOLDOUT_START = datetime(2024, 8, 21, tzinfo=UTC)

_SHA256 = re.compile(r"[0-9a-f]{64}")


class StageGValidationError(ValueError):
    """Raised when Stage G cannot prove its frozen or causal contract."""


class ResearchPartition(StrEnum):
    DEVELOPMENT = "development"
    VALIDATION = "validation"
    HOLDOUT = "holdout"


class MissingBarPolicy(StrEnum):
    DROP_INCOMPLETE_TIMESTAMP = "drop_incomplete_timestamp"
    EMIT_NONTRADABLE_GAP = "emit_nontradable_gap"


@dataclass(frozen=True, slots=True)
class FrozenInstrumentArtifact:
    instrument_id: str
    development_split_sha256: str
    catalog_tree_sha256: str


@dataclass(frozen=True, slots=True)
class FrozenUniverse:
    """Validated identity of the one admitted G1.4 universe."""

    readiness_path: Path
    readiness_sha256: str
    plan: ResearchUniversePlan
    ordered_artifacts: tuple[FrozenInstrumentArtifact, ...]

    @property
    def ordered_instrument_ids(self) -> tuple[str, ...]:
        return tuple(item.instrument_id for item in self.ordered_artifacts)


@dataclass(frozen=True, slots=True)
class DevelopmentArtifactStatus:
    instrument_id: str
    manifest_path: Path
    manifest_sha256: str
    catalog_tree_sha256: str
    catalog_directory_present: bool
    declared_bar_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class DevelopmentResearchContext:
    """The only candidate-development data context exposed by Stage G."""

    universe: FrozenUniverse
    artifacts: tuple[DevelopmentArtifactStatus, ...]
    start_utc: datetime = DEVELOPMENT_START
    end_exclusive_utc: datetime = DEVELOPMENT_END_EXCLUSIVE

    def require_range(
        self,
        start_utc: datetime,
        end_exclusive_utc: datetime,
        *,
        partition: ResearchPartition = ResearchPartition.DEVELOPMENT,
    ) -> tuple[datetime, datetime]:
        """Admit one half-open DEVELOPMENT request and reject every other split."""

        if partition is not ResearchPartition.DEVELOPMENT:
            raise StageGValidationError(
                f"{partition.value} is LOCKED in the development research API"
            )
        start = _aware_utc(start_utc, "start_utc")
        end = _aware_utc(end_exclusive_utc, "end_exclusive_utc")
        if start >= end:
            raise StageGValidationError("research range must be non-empty")
        if start < self.start_utc or end > self.end_exclusive_utc:
            boundary = (
                "validation"
                if start < HOLDOUT_START and end > self.end_exclusive_utc
                else "holdout"
            )
            raise StageGValidationError(
                f"request reaches locked {boundary} data; DEVELOPMENT ends "
                f"{_format_utc(self.end_exclusive_utc)}"
            )
        return start, end

    @property
    def synchronization_metadata_ready(self) -> bool:
        return len(self.artifacts) == len(FROZEN_INSTRUMENT_IDS) and all(
            item.catalog_directory_present for item in self.artifacts
        )


def load_frozen_universe(
    readiness_path: Path,
    plan_path: Path = Path("config/data/g1_4_fx_usd_liquid_v1.yaml"),
) -> FrozenUniverse:
    """Load only the exact frozen universe manifest and fail closed on drift."""

    plan = load_research_universe_plan(plan_path)
    if (
        plan.universe_id != FROZEN_UNIVERSE_ID
        or plan.semantic_sha256 != FROZEN_UNIVERSE_PLAN_SHA256
        or tuple(item.instrument_id for item in plan.instruments)
        != FROZEN_INSTRUMENT_IDS
    ):
        raise StageGValidationError("universe plan does not match the Stage G freeze")
    document = _semantic_json(readiness_path, "universe readiness")
    if document["semantic_sha256"] != FROZEN_UNIVERSE_READINESS_SHA256:
        raise StageGValidationError("universe readiness hash is not the frozen hash")
    required = {
        "readiness_version",
        "universe_id",
        "universe_plan_sha256",
        "ordered_instrument_ids",
        "exact_currency_set",
        "permitted_utc_interval",
        "instrument_artifacts",
        "per_instrument_status",
        "cross_sectional_factor_status",
        "minimum_cross_sectional_instruments",
        "research_ready",
        "holdout_accessed",
        "holdout_rows_admitted",
        "strategy_return_accessed",
        "semantic_sha256",
    }
    if set(document) != required:
        raise StageGValidationError("universe readiness schema is not exact")
    if (
        document["readiness_version"] != "g1.4a-universe-readiness-1"
        or document["universe_id"] != FROZEN_UNIVERSE_ID
        or document["universe_plan_sha256"] != FROZEN_UNIVERSE_PLAN_SHA256
        or tuple(document["ordered_instrument_ids"]) != FROZEN_INSTRUMENT_IDS
        or document["exact_currency_set"] != ["EUR", "GBP", "USD"]
        or document["research_ready"] is not True
        or document["holdout_accessed"] is not False
        or document["holdout_rows_admitted"] != 0
        or document["strategy_return_accessed"] is not False
        or document["cross_sectional_factor_status"]
        != "not_eligible_insufficient_universe"
        or document["minimum_cross_sectional_instruments"] != 4
    ):
        raise StageGValidationError("universe readiness gates are incompatible")
    expected_range = {
        "start_utc": _format_utc(DEVELOPMENT_START),
        "end_exclusive_utc": _format_utc(HOLDOUT_START),
    }
    if document["permitted_utc_interval"] != expected_range:
        raise StageGValidationError("universe readiness range is incompatible")
    statuses = document["per_instrument_status"]
    if not isinstance(statuses, dict) or statuses != {
        item: "research_ready" for item in FROZEN_INSTRUMENT_IDS
    }:
        raise StageGValidationError("instrument readiness status is incomplete")
    raw_artifacts = document["instrument_artifacts"]
    if not isinstance(raw_artifacts, list):
        raise StageGValidationError("instrument artifacts must be ordered")
    artifact_ids = tuple(
        item.get("instrument_id") if isinstance(item, dict) else None
        for item in raw_artifacts
    )
    if artifact_ids != FROZEN_INSTRUMENT_IDS:
        raise StageGValidationError("missing, extra, or reordered instrument artifact")
    artifacts: list[FrozenInstrumentArtifact] = []
    for raw in raw_artifacts:
        item = cast(dict[str, Any], raw)
        expected_fields = {
            "instrument_id",
            "canonical_sha256",
            "derived_sha256",
            "coverage_sha256",
            "reconciliation_sha256",
            "correction_sha256",
            "readiness_sha256",
            "catalog_tree_sha256",
            "development_split_sha256",
            "validation_split_sha256",
        }
        if set(item) != expected_fields:
            raise StageGValidationError("instrument artifact schema is not exact")
        for key, value in item.items():
            if key == "instrument_id" or (key == "correction_sha256" and value is None):
                continue
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise StageGValidationError(f"invalid instrument artifact {key}")
        artifacts.append(
            FrozenInstrumentArtifact(
                instrument_id=cast(str, item["instrument_id"]),
                development_split_sha256=cast(str, item["development_split_sha256"]),
                catalog_tree_sha256=cast(str, item["catalog_tree_sha256"]),
            )
        )
    return FrozenUniverse(
        readiness_path=readiness_path.resolve(),
        readiness_sha256=FROZEN_UNIVERSE_READINESS_SHA256,
        plan=plan,
        ordered_artifacts=tuple(artifacts),
    )


def open_development_context(
    readiness_path: Path,
    development_roots: Mapping[str, Path],
    plan_path: Path = Path("config/data/g1_4_fx_usd_liquid_v1.yaml"),
) -> DevelopmentResearchContext:
    """Validate DEVELOPMENT manifests only; no catalog rows are read here."""

    universe = load_frozen_universe(readiness_path, plan_path)
    if set(development_roots) != set(FROZEN_INSTRUMENT_IDS) or len(
        development_roots
    ) != len(FROZEN_INSTRUMENT_IDS):
        raise StageGValidationError("development roots have missing/extra instruments")
    expected_range = {
        "start_utc": _format_utc(DEVELOPMENT_START),
        "end_exclusive_utc": _format_utc(DEVELOPMENT_END_EXCLUSIVE),
    }
    statuses: list[DevelopmentArtifactStatus] = []
    refs = {item.instrument_id: item for item in universe.ordered_artifacts}
    for instrument_id in FROZEN_INSTRUMENT_IDS:
        root = development_roots[instrument_id].resolve()
        path = root / SPLIT_VIEW_FILENAME
        document = _semantic_json(path, f"{instrument_id} development split")
        if (
            document["semantic_sha256"] != refs[instrument_id].development_split_sha256
            or document.get("instrument_id") != instrument_id
            or document.get("universe_plan_sha256") != FROZEN_UNIVERSE_PLAN_SHA256
            or document.get("split") != "development"
            or document.get("range") != expected_range
            or document.get("holdout_rows") != 0
            or document.get("access_policy") != "candidate_read_only"
            or document.get("network_access_required") is not False
        ):
            raise StageGValidationError(
                f"development split is incompatible or locked: {instrument_id}"
            )
        counts = document.get("bar_counts")
        if (
            not isinstance(counts, dict)
            or not counts
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in counts.values()
            )
        ):
            raise StageGValidationError("development split bar counts are invalid")
        catalog_tree = document.get("catalog_tree_sha256")
        if not isinstance(catalog_tree, str) or _SHA256.fullmatch(catalog_tree) is None:
            raise StageGValidationError("development catalog hash is invalid")
        statuses.append(
            DevelopmentArtifactStatus(
                instrument_id=instrument_id,
                manifest_path=path,
                manifest_sha256=cast(str, document["semantic_sha256"]),
                catalog_tree_sha256=catalog_tree,
                catalog_directory_present=(root / "catalog").is_dir(),
                declared_bar_counts=tuple(sorted(cast(dict[str, int], counts).items())),
            )
        )
    return DevelopmentResearchContext(universe=universe, artifacts=tuple(statuses))


@dataclass(frozen=True, slots=True)
class InstrumentBarObservation:
    """One actually observed paired BID/ASK close and its causal availability."""

    instrument_id: str
    timestamp_utc: datetime
    available_at_utc: datetime
    bid: Decimal
    ask: Decimal

    def validate(self) -> None:
        timestamp = _aware_utc(self.timestamp_utc, "timestamp_utc")
        available = _aware_utc(self.available_at_utc, "available_at_utc")
        if self.instrument_id not in FROZEN_INSTRUMENT_IDS:
            raise StageGValidationError(
                "observation instrument is outside the universe"
            )
        if available < timestamp:
            raise StageGValidationError("observation cannot be available before event")
        if (
            not self.bid.is_finite()
            or not self.ask.is_finite()
            or self.bid <= 0
            or self.ask < self.bid
        ):
            raise StageGValidationError("observation has invalid BID/ASK prices")


@dataclass(frozen=True, slots=True)
class SynchronizedClockFrame:
    timestamp_utc: datetime
    available_at_utc: datetime
    observations: tuple[InstrumentBarObservation | None, ...]
    tradable: bool


def synchronize_universe_clock(
    streams: Mapping[str, Sequence[InstrumentBarObservation]],
    *,
    missing_policy: MissingBarPolicy,
) -> tuple[SynchronizedClockFrame, ...]:
    """Align actual observations deterministically without fill or synthesis."""

    if set(streams) != set(FROZEN_INSTRUMENT_IDS) or len(streams) != len(
        FROZEN_INSTRUMENT_IDS
    ):
        raise StageGValidationError("clock streams have missing/extra instruments")
    by_instrument: dict[str, dict[datetime, InstrumentBarObservation]] = {}
    timestamps: set[datetime] = set()
    for instrument_id in FROZEN_INSTRUMENT_IDS:
        indexed: dict[datetime, InstrumentBarObservation] = {}
        previous: datetime | None = None
        for observation in streams[instrument_id]:
            observation.validate()
            if observation.instrument_id != instrument_id:
                raise StageGValidationError("clock stream contains wrong instrument")
            timestamp = _aware_utc(observation.timestamp_utc, "timestamp_utc")
            if previous is not None and timestamp <= previous:
                raise StageGValidationError("clock streams must be strictly ordered")
            indexed[timestamp] = observation
            timestamps.add(timestamp)
            previous = timestamp
        by_instrument[instrument_id] = indexed
    frames: list[SynchronizedClockFrame] = []
    for timestamp in sorted(timestamps):
        observations = tuple(
            by_instrument[instrument_id].get(timestamp)
            for instrument_id in FROZEN_INSTRUMENT_IDS
        )
        complete = all(item is not None for item in observations)
        if (
            not complete
            and missing_policy is MissingBarPolicy.DROP_INCOMPLETE_TIMESTAMP
        ):
            continue
        available = max(
            item.available_at_utc for item in observations if item is not None
        )
        frames.append(
            SynchronizedClockFrame(
                timestamp_utc=timestamp,
                available_at_utc=available,
                observations=observations,
                tradable=complete,
            )
        )
    return tuple(frames)


@dataclass(frozen=True, slots=True)
class NormalizedInstrumentReturn:
    instrument_id: str
    base_currency: str
    quote_currency: str
    price_return: Decimal


@dataclass(frozen=True, slots=True)
class FxInstrumentCurrencies:
    """Explicit currency identity for future instruments outside the frozen v1."""

    instrument_id: str
    base_currency: str
    quote_currency: str

    def validate(self) -> None:
        if (
            len(self.base_currency) != 3
            or len(self.quote_currency) != 3
            or not self.base_currency.isalpha()
            or not self.quote_currency.isalpha()
            or not self.base_currency.isupper()
            or not self.quote_currency.isupper()
            or self.base_currency == self.quote_currency
        ):
            raise StageGValidationError("currency metadata must use distinct ISO codes")


def normalized_price_return(
    instrument_id: str,
    entry_price: Decimal,
    exit_price: Decimal,
    *,
    currencies: FxInstrumentCurrencies | None = None,
) -> NormalizedInstrumentReturn:
    """Represent a price return with explicit base/quote identity."""

    base, quote = _currencies(instrument_id, currencies)
    if any(not item.is_finite() or item <= 0 for item in (entry_price, exit_price)):
        raise StageGValidationError("return prices must be finite and positive")
    return NormalizedInstrumentReturn(
        instrument_id, base, quote, exit_price / entry_price - Decimal(1)
    )


def liquidation_pnl_usd(
    instrument_id: str,
    signed_base_units: Decimal,
    entry_price: Decimal,
    liquidation_price: Decimal,
    *,
    quote_to_usd: Decimal | None = None,
    currencies: FxInstrumentCurrencies | None = None,
) -> Decimal:
    """Liquidate signed base units, converting quote P/L explicitly when needed."""

    _, quote = _currencies(instrument_id, currencies)
    values = (signed_base_units, entry_price, liquidation_price)
    if (
        any(not item.is_finite() for item in values)
        or entry_price <= 0
        or liquidation_price <= 0
    ):
        raise StageGValidationError(
            "liquidation inputs must be finite with positive prices"
        )
    pnl_quote = signed_base_units * (liquidation_price - entry_price)
    if quote == "USD":
        if quote_to_usd not in (None, Decimal(1)):
            raise StageGValidationError("USD quote conversion must be omitted or one")
        return pnl_quote
    if quote_to_usd is None or not quote_to_usd.is_finite() or quote_to_usd <= 0:
        raise StageGValidationError("non-USD quote requires a positive PIT USD rate")
    return pnl_quote * quote_to_usd


@dataclass(frozen=True, slots=True)
class FxPosition:
    instrument_id: str
    signed_base_units: Decimal
    price: Decimal
    currencies: FxInstrumentCurrencies | None = None


@dataclass(frozen=True, slots=True)
class CurrencyExposure:
    currency: str
    amount: Decimal


def aggregate_currency_exposures(
    positions: Sequence[FxPosition],
) -> tuple[CurrencyExposure, ...]:
    """Aggregate FX incidence: +base units and -quote notional per position."""

    totals: dict[str, Decimal] = {}
    for position in positions:
        base, quote = _currencies(position.instrument_id, position.currencies)
        if (
            not position.signed_base_units.is_finite()
            or not position.price.is_finite()
            or position.price <= 0
        ):
            raise StageGValidationError("position exposure inputs are invalid")
        totals[base] = totals.get(base, Decimal(0)) + position.signed_base_units
        totals[quote] = totals.get(quote, Decimal(0)) - (
            position.signed_base_units * position.price
        )
    return tuple(CurrencyExposure(key, totals[key]) for key in sorted(totals))


@dataclass(frozen=True, slots=True)
class CurrencyExposureLimits:
    max_abs_amount: Mapping[str, Decimal]

    def breaches(
        self, exposures: Sequence[CurrencyExposure]
    ) -> tuple[CurrencyExposure, ...]:
        """Return deterministic per-currency breaches; absent limits fail closed."""

        limits = dict(self.max_abs_amount)
        if not limits or any(
            not value.is_finite() or value <= 0 for value in limits.values()
        ):
            raise StageGValidationError("exposure limits must be finite and positive")
        breaches: list[CurrencyExposure] = []
        for exposure in exposures:
            limit = limits.get(exposure.currency)
            if limit is None:
                raise StageGValidationError(
                    f"missing exposure limit for {exposure.currency}"
                )
            if abs(exposure.amount) > limit:
                breaches.append(exposure)
        return tuple(breaches)


@dataclass(frozen=True, slots=True)
class PerInstrumentCostModel:
    """Bind observed spread and the existing G0.7 native execution profile."""

    instrument_id: str
    execution_profile: ExecutionProfile
    spread_source: str = "observed_paired_bid_ask"

    def validate(self) -> None:
        _currencies(self.instrument_id)
        if self.spread_source != "observed_paired_bid_ask":
            raise StageGValidationError("spread must come from observed paired BID/ASK")
        validate_execution_profile(self.execution_profile)


def validate_cost_models(models: Sequence[PerInstrumentCostModel]) -> None:
    if tuple(item.instrument_id for item in models) != FROZEN_INSTRUMENT_IDS:
        raise StageGValidationError("cost models must match frozen instrument order")
    for model in models:
        model.validate()


@dataclass(frozen=True, slots=True)
class DevelopmentFold:
    fold_id: str
    train_start_utc: datetime
    train_end_exclusive_utc: datetime
    compare_start_utc: datetime
    compare_end_exclusive_utc: datetime


@dataclass(frozen=True, slots=True)
class DevelopmentFoldPlan:
    version: str
    folds: tuple[DevelopmentFold, ...]
    semantic_sha256: str


def frozen_development_folds() -> DevelopmentFoldPlan:
    """Return preregistered expanding-window folds contained in DEVELOPMENT."""

    boundaries = (
        ("dev_fold_1", "2020-04-11T00:00:00Z", "2021-04-11T00:00:00Z"),
        ("dev_fold_2", "2021-04-11T00:00:00Z", "2022-04-11T00:00:00Z"),
        ("dev_fold_3", "2022-04-11T00:00:00Z", "2023-04-11T00:00:00Z"),
    )
    folds = tuple(
        DevelopmentFold(
            fold_id=fold_id,
            train_start_utc=DEVELOPMENT_START,
            train_end_exclusive_utc=_parse_utc(compare_start),
            compare_start_utc=_parse_utc(compare_start),
            compare_end_exclusive_utc=_parse_utc(compare_end),
        )
        for fold_id, compare_start, compare_end in boundaries
    )
    for fold in folds:
        if not (
            DEVELOPMENT_START
            <= fold.train_start_utc
            < fold.train_end_exclusive_utc
            == fold.compare_start_utc
            < fold.compare_end_exclusive_utc
            <= DEVELOPMENT_END_EXCLUSIVE
        ):
            raise StageGValidationError("development fold escaped DEVELOPMENT")
    payload = {
        "version": "g1.4b-development-folds-1",
        "folds": [
            {
                key: _format_utc(value) if isinstance(value, datetime) else value
                for key, value in asdict(fold).items()
            }
            for fold in folds
        ],
    }
    return DevelopmentFoldPlan(
        version=cast(str, payload["version"]),
        folds=folds,
        semantic_sha256=_semantic_sha256(payload),
    )


def _currencies(
    instrument_id: str, explicit: FxInstrumentCurrencies | None = None
) -> tuple[str, str]:
    if explicit is not None:
        explicit.validate()
        if explicit.instrument_id != instrument_id:
            raise StageGValidationError("currency metadata instrument mismatch")
        result = (explicit.base_currency, explicit.quote_currency)
        if instrument_id in FROZEN_INSTRUMENT_IDS:
            symbol = instrument_id.split(".", maxsplit=1)[0]
            if result != tuple(symbol.split("/", maxsplit=1)):
                raise StageGValidationError("frozen instrument currency metadata drift")
        return result
    if instrument_id not in FROZEN_INSTRUMENT_IDS:
        raise StageGValidationError(
            "future instrument requires explicit base/quote currency metadata"
        )
    symbol = instrument_id.split(".", maxsplit=1)[0]
    base, quote = symbol.split("/", maxsplit=1)
    return base, quote


def _semantic_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StageGValidationError(f"invalid {description}: {error}") from error
    if not isinstance(value, dict):
        raise StageGValidationError(f"{description} must be an object")
    document = cast(dict[str, Any], value)
    semantic = document.get("semantic_sha256")
    payload = {key: item for key, item in document.items() if key != "semantic_sha256"}
    if not isinstance(semantic, str) or semantic != _semantic_sha256(payload):
        raise StageGValidationError(f"{description} semantic hash mismatch")
    return document


def _semantic_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _aware_utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise StageGValidationError(f"{name} must be timezone-aware UTC")
    return value.astimezone(UTC)


def _format_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


DEFAULT_UNIVERSE_READINESS_PATH = (
    Path(".artifacts/g1_4b/universe") / UNIVERSE_READINESS_FILENAME
)
DEFAULT_DEVELOPMENT_ROOTS: Mapping[str, Path] = {
    "EUR/USD.DUKASCOPY": Path(".artifacts/g1_4b/development/EURUSD"),
    "GBP/USD.DUKASCOPY": Path(".artifacts/g1_4b/development/GBPUSD"),
}
