"""Fail-closed reconciliation of expected-open Dukascopy EUR/USD gaps."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import re
import struct
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

from tradedesk_dukascopy.export import (  # type: ignore[import-untyped]
    _decode_ticks,
    _dukascopy_tick_url,
    _probe_price_format,
)

from ftmoquant.data.canonical_source import (
    CORRECTED_INGESTION_VERSION,
    GENERIC_CORRECTED_INGESTION_VERSION,
    HF_INGESTION_VERSION,
    validate_canonical_eurusd_source_manifest,
    validate_canonical_source_manifest,
)
from ftmoquant.data.derived_bars import PARENT_MANIFEST_FILENAME
from ftmoquant.data.dukascopy import INSTRUMENT_ID, UPSTREAM_COMMIT, UPSTREAM_VERSION
from ftmoquant.data.instruments import EURUSD_SPEC, InstrumentSpec
from ftmoquant.data.research_plan import (
    ResearchDataPlan,
    ResearchSplit,
    load_research_data_plan,
)
from ftmoquant.data.session_coverage import (
    COVERAGE_MANIFEST_FILENAME,
    COVERAGE_QA_VERSION,
    SESSION_POLICY_VERSION,
    is_eurusd_expected_open,
)
from ftmoquant.data.universe_plan import load_research_universe_plan

RECONCILIATION_MANIFEST_FILENAME = "ftmoquant_session_reconciliation.json"
RECONCILIATION_VERSION = "g1-session-reconciliation-1"
OFFLINE_EVIDENCE_VERSION = "dukascopy-jforex-offline-domains-v1"
DIRECT_VERIFICATION_VERSION = "dukascopy-bi5-exact-minute-v1"
DIRECT_SOURCE_URL = "https://datafeed.dukascopy.com/datafeed"

_MINUTE = timedelta(minutes=1)
_HOUR = timedelta(hours=1)
_TICK_RECORD_SIZE = 20
_SHA256_LENGTH = 64
_CLASS_PROVIDER_OFFLINE = "provider_offline"
_CLASS_VERIFIED_NO_TICK = "verified_no_tick"
_CLASS_UNEXPLAINED = "unexplained_missing"
_OUTCOME_VERIFIED = "verified"
_OUTCOME_FAILED = "retrieval_failed"
_EXPECTED_MISSING_SIDES = ("BID", "ASK")


class ReconciliationValidationError(ValueError):
    """Raised when reconciliation evidence cannot support a safe conclusion."""


@dataclass(frozen=True, slots=True)
class OfflineDomain:
    """One exact half-open provider offline domain from JForex."""

    evidence_id: str
    start_utc: datetime
    end_exclusive_utc: datetime


@dataclass(frozen=True, slots=True)
class HourVerification:
    """Validated direct-source outcome for one UTC BI5 hour."""

    hour_start_utc: datetime
    source_url: str
    outcome: str
    response_status: int | None
    tick_minutes: tuple[datetime, ...]
    tick_count: int | None
    content_size: int | None
    content_sha256: str | None
    cache_relative_path: str | None
    failure_reason: str | None


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """Small injectable HTTP boundary used by deterministic acquisition tests."""

    status: int
    body: bytes


@dataclass(frozen=True, slots=True)
class ReconciledInterval:
    """Inclusive interval of identically classified missing source minutes."""

    classification: str
    start_utc: str
    end_utc: str
    minutes: int
    missing_sides: tuple[str, ...]
    reason: str
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    """Persisted path and final readiness outcome."""

    manifest_path: Path
    session_aware_research_ready: bool
    semantic_sha256: str
    counts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class AcquisitionBatchResult:
    """Outcome of one bounded cache-warming batch."""

    selected_hour_count: int
    verified_hour_count: int
    failed_hour_count: int
    remaining_uncached_hour_count: int


@dataclass(frozen=True, slots=True)
class _MissingMinute:
    timestamp: datetime
    missing_sides: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _MinuteDecision:
    timestamp: datetime
    missing_sides: tuple[str, ...]
    classification: str
    reason: str
    evidence_ref: str | None


Transport = Callable[[str], HttpResponse]


class DirectDukascopyAcquirer:
    """Acquire, validate, and cache exact hourly Dukascopy BI5 source files."""

    def __init__(
        self,
        cache_root: Path,
        *,
        symbol: str = "EURUSD",
        transport: Transport | None = None,
        retries: int = 3,
        retry_delay_seconds: float = 0.8,
    ) -> None:
        if retries <= 0:
            raise ValueError("retries must be positive")
        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must not be negative")
        self.cache_root = cache_root.resolve()
        if re.fullmatch(r"[A-Z]{6}", symbol) is None:
            raise ValueError("symbol must be exactly six uppercase letters")
        self.symbol = symbol
        self.transport = transport or _http_get
        self.retries = retries
        self.retry_delay_seconds = retry_delay_seconds

    def has_complete_cache_entry(self, hour_start_utc: datetime) -> bool:
        """Return whether both files for one cache entry exist."""

        hour = _require_utc_hour(hour_start_utc, "hour_start_utc")
        raw_path, meta_path = self._cache_paths(hour)
        return raw_path.is_file() and meta_path.is_file()

    def acquire(self, hour_start_utc: datetime, cutoff: datetime) -> HourVerification:
        """Return a cache-stable verification outcome for one permitted hour."""

        hour = _require_utc_hour(hour_start_utc, "hour_start_utc")
        cutoff = _require_utc_minute(cutoff, "cutoff")
        if hour + _HOUR > cutoff:
            raise ReconciliationValidationError(
                "direct Dukascopy query would cross the sealed holdout boundary"
            )
        source_url = _dukascopy_tick_url(self.symbol, hour)
        raw_path, meta_path = self._cache_paths(hour)
        if raw_path.exists() or meta_path.exists():
            return self._load_cache(hour, source_url, raw_path, meta_path)

        last_reason = "transport_failure"
        last_status: int | None = None
        for attempt in range(1, self.retries + 1):
            try:
                response = self.transport(source_url)
                last_status = response.status
                if response.status != 200:
                    last_reason = f"http_status_{response.status}"
                else:
                    verification = _verification_from_payload(
                        hour,
                        source_url,
                        response.body,
                        _relative_cache_path(hour, self.symbol),
                    )
                    self._write_cache(raw_path, meta_path, verification, response.body)
                    return verification
            except (OSError, ReconciliationValidationError, ValueError, lzma.LZMAError):
                last_reason = "transport_or_payload_failure"
            if attempt < self.retries and self.retry_delay_seconds:
                time.sleep(self.retry_delay_seconds * (2 ** (attempt - 1)))
        return HourVerification(
            hour_start_utc=hour,
            source_url=source_url,
            outcome=_OUTCOME_FAILED,
            response_status=last_status,
            tick_minutes=(),
            tick_count=None,
            content_size=None,
            content_sha256=None,
            cache_relative_path=None,
            failure_reason=last_reason,
        )

    def _cache_paths(self, hour: datetime) -> tuple[Path, Path]:
        relative = Path(_relative_cache_path(hour, self.symbol))
        raw_path = self.cache_root / relative
        return raw_path, raw_path.with_suffix(".bi5.meta.json")

    def _load_cache(
        self,
        hour: datetime,
        source_url: str,
        raw_path: Path,
        meta_path: Path,
    ) -> HourVerification:
        if not raw_path.is_file() or not meta_path.is_file():
            return _failed_verification(hour, source_url, "incomplete_cache_entry")
        try:
            metadata = _load_json_object(meta_path, "BI5 cache metadata")
            payload = raw_path.read_bytes()
            expected = {
                "cache_schema_version": DIRECT_VERIFICATION_VERSION,
                "hour_start_utc": _format_utc(hour),
                "source_url": source_url,
                "response_status": 200,
                "content_size": len(payload),
                "content_sha256": hashlib.sha256(payload).hexdigest(),
            }
            if any(metadata.get(key) != value for key, value in expected.items()):
                raise ReconciliationValidationError("BI5 cache metadata mismatch")
            verification = _verification_from_payload(
                hour, source_url, payload, _relative_cache_path(hour, self.symbol)
            )
            if metadata.get("tick_count") != verification.tick_count:
                raise ReconciliationValidationError("BI5 cache tick count mismatch")
            return verification
        except (OSError, ValueError, lzma.LZMAError) as error:
            return _failed_verification(
                hour, source_url, f"invalid_cache:{type(error).__name__}"
            )

    def _write_cache(
        self,
        raw_path: Path,
        meta_path: Path,
        verification: HourVerification,
        payload: bytes,
    ) -> None:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_temp = raw_path.with_suffix(".bi5.tmp")
        meta_temp = meta_path.with_suffix(".json.tmp")
        raw_temp.write_bytes(payload)
        metadata = {
            "cache_schema_version": DIRECT_VERIFICATION_VERSION,
            "hour_start_utc": _format_utc(verification.hour_start_utc),
            "source_url": verification.source_url,
            "response_status": 200,
            "content_size": verification.content_size,
            "content_sha256": verification.content_sha256,
            "tick_count": verification.tick_count,
        }
        meta_temp.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raw_temp.replace(raw_path)
        meta_temp.replace(meta_path)


def run_eurusd_session_reconciliation(
    plan_path: Path,
    output_root: Path,
    cache_root: Path,
    *,
    offline_evidence_path: Path | None = None,
    workers: int = 8,
    acquirer: DirectDukascopyAcquirer | None = None,
) -> ReconciliationResult:
    """Reconcile the immutable expected-open gap report and persist a new artifact."""

    if workers <= 0:
        raise ValueError("workers must be positive")
    plan = load_research_data_plan(plan_path)
    root = output_root.resolve()
    provenance_path = root / PARENT_MANIFEST_FILENAME
    coverage_path = root / COVERAGE_MANIFEST_FILENAME
    provenance = _load_json_object(provenance_path, "canonical provenance")
    coverage = _load_json_object(coverage_path, "session coverage report")
    missing = _validate_inputs(
        plan,
        plan_path,
        provenance,
        provenance_path,
        coverage,
        coverage_path,
    )
    offline_domains, offline_provenance = _load_offline_evidence(
        offline_evidence_path, plan
    )

    required_hours = sorted(
        {
            _floor_hour(item.timestamp)
            for item in missing
            if item.missing_sides == _EXPECTED_MISSING_SIDES
            and _offline_for_minute(item.timestamp, offline_domains) is None
        }
    )
    verifier = acquirer or DirectDukascopyAcquirer(cache_root)
    hour_results = _acquire_hours(
        required_hours, verifier, plan.g1_allowed_end_exclusive, workers
    )

    payload = build_reconciliation_payload(
        plan=plan,
        plan_file_sha256=_sha256(plan_path),
        provenance=provenance,
        provenance_file_sha256=_sha256(provenance_path),
        coverage=coverage,
        coverage_file_sha256=_sha256(coverage_path),
        missing_minutes=missing,
        offline_domains=offline_domains,
        offline_provenance=offline_provenance,
        hour_results=hour_results,
    )
    semantic_sha256 = _semantic_sha256(payload)
    manifest = {**payload, "semantic_sha256": semantic_sha256}
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    manifest_path = root / RECONCILIATION_MANIFEST_FILENAME
    if manifest_path.exists() and manifest_path.read_bytes() != manifest_bytes:
        raise ReconciliationValidationError(
            f"existing session reconciliation conflicts with this run: {manifest_path}"
        )
    if not manifest_path.exists():
        manifest_path.write_bytes(manifest_bytes)
    return ReconciliationResult(
        manifest_path=manifest_path,
        session_aware_research_ready=cast(
            bool, manifest["session_aware_research_ready"]
        ),
        semantic_sha256=semantic_sha256,
        counts=cast(Mapping[str, int], manifest["counts"]),
    )


def run_instrument_reconciliation(
    plan_path: Path,
    instrument_id: str,
    output_root: Path,
    cache_root: Path,
    *,
    offline_evidence_path: Path | None = None,
    workers: int = 4,
    acquirer: DirectDukascopyAcquirer | None = None,
) -> ReconciliationResult:
    """Reconcile one universe member with symbol-specific direct BI5 evidence."""

    if workers <= 0 or workers > 4:
        raise ValueError("multi-instrument reconciliation workers must be in 1..4")
    universe = load_research_universe_plan(plan_path)
    instrument = universe.instrument(instrument_id)
    plan = ResearchDataPlan(
        dataset_id=universe.universe_id,
        market_data_origin=universe.market_data_origin,
        distribution_source=universe.distribution_source,
        huggingface_repo=universe.huggingface_repo,
        huggingface_revision=universe.huggingface_revision,
        instrument=instrument.instrument_id,
        full_start=universe.development.start,
        full_end=universe.validation.end,
        boundary_rule="explicit_whole_utc_days",
        development=universe.development,
        validation=universe.validation,
        final_holdout=ResearchSplit(
            start=universe.holdout_cutoff_utc.date(),
            end=universe.holdout_cutoff_utc.date(),
            days=1,
        ),
        g1_allowed_end_exclusive=universe.holdout_cutoff_utc,
        semantic_sha256=universe.semantic_sha256,
    )
    root = output_root.resolve()
    provenance_path = root / PARENT_MANIFEST_FILENAME
    coverage_path = root / COVERAGE_MANIFEST_FILENAME
    provenance = _load_json_object(provenance_path, "canonical provenance")
    coverage = _load_json_object(coverage_path, "session coverage report")
    missing = _validate_inputs(
        plan,
        plan_path,
        provenance,
        provenance_path,
        coverage,
        coverage_path,
        instrument_spec=instrument,
        session_policy_version=instrument.session_policy_id,
        legacy=False,
    )
    offline_domains, offline_provenance = _load_offline_evidence(
        offline_evidence_path, plan
    )
    required_hours = sorted(
        {
            _floor_hour(item.timestamp)
            for item in missing
            if item.missing_sides == _EXPECTED_MISSING_SIDES
            and _offline_for_minute(item.timestamp, offline_domains) is None
        }
    )
    verifier = acquirer or DirectDukascopyAcquirer(
        cache_root, symbol=instrument.dataset_symbol
    )
    if verifier.symbol != instrument.dataset_symbol:
        raise ReconciliationValidationError(
            "BI5 acquirer symbol does not match instrument"
        )
    hour_results = _acquire_hours(
        required_hours,
        verifier,
        plan.g1_allowed_end_exclusive,
        workers,
        symbol=instrument.dataset_symbol,
    )
    payload = build_reconciliation_payload(
        plan=plan,
        plan_file_sha256=_sha256(plan_path),
        provenance=provenance,
        provenance_file_sha256=_sha256(provenance_path),
        coverage=coverage,
        coverage_file_sha256=_sha256(coverage_path),
        missing_minutes=missing,
        offline_domains=offline_domains,
        offline_provenance=offline_provenance,
        hour_results=hour_results,
        instrument_id=instrument.instrument_id,
        symbol=instrument.dataset_symbol,
    )
    semantic_sha256 = _semantic_sha256(payload)
    manifest = {**payload, "semantic_sha256": semantic_sha256}
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    manifest_path = root / RECONCILIATION_MANIFEST_FILENAME
    if manifest_path.exists() and manifest_path.read_bytes() != manifest_bytes:
        raise ReconciliationValidationError(
            f"existing session reconciliation conflicts with this run: {manifest_path}"
        )
    if not manifest_path.exists():
        manifest_path.write_bytes(manifest_bytes)
    return ReconciliationResult(
        manifest_path=manifest_path,
        session_aware_research_ready=cast(
            bool, manifest["session_aware_research_ready"]
        ),
        semantic_sha256=semantic_sha256,
        counts=cast(Mapping[str, int], manifest["counts"]),
    )


def acquire_instrument_reconciliation_batch(
    plan_path: Path,
    instrument_id: str,
    output_root: Path,
    cache_root: Path,
    *,
    offline_evidence_path: Path | None = None,
    workers: int = 4,
    max_uncached_hours: int = 200,
    acquirer: DirectDukascopyAcquirer | None = None,
) -> AcquisitionBatchResult:
    """Warm a bounded instrument cache batch without writing a final manifest."""

    if workers <= 0 or workers > 4:
        raise ValueError("multi-instrument reconciliation workers must be in 1..4")
    if max_uncached_hours <= 0:
        raise ValueError("max_uncached_hours must be positive")
    universe = load_research_universe_plan(plan_path)
    instrument = universe.instrument(instrument_id)
    plan = ResearchDataPlan(
        dataset_id=universe.universe_id,
        market_data_origin=universe.market_data_origin,
        distribution_source=universe.distribution_source,
        huggingface_repo=universe.huggingface_repo,
        huggingface_revision=universe.huggingface_revision,
        instrument=instrument.instrument_id,
        full_start=universe.development.start,
        full_end=universe.validation.end,
        boundary_rule="explicit_whole_utc_days",
        development=universe.development,
        validation=universe.validation,
        final_holdout=ResearchSplit(
            start=universe.holdout_cutoff_utc.date(),
            end=universe.holdout_cutoff_utc.date(),
            days=1,
        ),
        g1_allowed_end_exclusive=universe.holdout_cutoff_utc,
        semantic_sha256=universe.semantic_sha256,
    )
    root = output_root.resolve()
    provenance_path = root / PARENT_MANIFEST_FILENAME
    coverage_path = root / COVERAGE_MANIFEST_FILENAME
    provenance = _load_json_object(provenance_path, "canonical provenance")
    coverage = _load_json_object(coverage_path, "session coverage report")
    missing = _validate_inputs(
        plan,
        plan_path,
        provenance,
        provenance_path,
        coverage,
        coverage_path,
        instrument_spec=instrument,
        session_policy_version=instrument.session_policy_id,
        legacy=False,
    )
    offline_domains, _ = _load_offline_evidence(offline_evidence_path, plan)
    required_hours = sorted(
        {
            _floor_hour(item.timestamp)
            for item in missing
            if item.missing_sides == _EXPECTED_MISSING_SIDES
            and _offline_for_minute(item.timestamp, offline_domains) is None
        }
    )
    verifier = acquirer or DirectDukascopyAcquirer(
        cache_root, symbol=instrument.dataset_symbol
    )
    if verifier.symbol != instrument.dataset_symbol:
        raise ReconciliationValidationError(
            "BI5 acquirer symbol does not match instrument"
        )
    uncached = [
        hour for hour in required_hours if not verifier.has_complete_cache_entry(hour)
    ]
    selected = uncached[:max_uncached_hours]
    results = _acquire_hours(
        selected,
        verifier,
        plan.g1_allowed_end_exclusive,
        workers,
        symbol=instrument.dataset_symbol,
    )
    verified = sum(result.outcome == _OUTCOME_VERIFIED for result in results.values())
    return AcquisitionBatchResult(
        selected_hour_count=len(selected),
        verified_hour_count=verified,
        failed_hour_count=len(selected) - verified,
        remaining_uncached_hour_count=len(uncached) - verified,
    )


def acquire_eurusd_reconciliation_batch(
    plan_path: Path,
    output_root: Path,
    cache_root: Path,
    *,
    offline_evidence_path: Path | None = None,
    workers: int = 8,
    max_uncached_hours: int = 200,
    acquirer: DirectDukascopyAcquirer | None = None,
) -> AcquisitionBatchResult:
    """Warm a bounded cache batch without writing a partial final artifact."""

    if workers <= 0:
        raise ValueError("workers must be positive")
    if max_uncached_hours <= 0:
        raise ValueError("max_uncached_hours must be positive")
    plan = load_research_data_plan(plan_path)
    root = output_root.resolve()
    provenance_path = root / PARENT_MANIFEST_FILENAME
    coverage_path = root / COVERAGE_MANIFEST_FILENAME
    provenance = _load_json_object(provenance_path, "canonical provenance")
    coverage = _load_json_object(coverage_path, "session coverage report")
    missing = _validate_inputs(
        plan,
        plan_path,
        provenance,
        provenance_path,
        coverage,
        coverage_path,
    )
    offline_domains, _ = _load_offline_evidence(offline_evidence_path, plan)
    required_hours = sorted(
        {
            _floor_hour(item.timestamp)
            for item in missing
            if item.missing_sides == _EXPECTED_MISSING_SIDES
            and _offline_for_minute(item.timestamp, offline_domains) is None
        }
    )
    verifier = acquirer or DirectDukascopyAcquirer(cache_root)
    uncached = [
        hour for hour in required_hours if not verifier.has_complete_cache_entry(hour)
    ]
    selected = uncached[:max_uncached_hours]
    results = _acquire_hours(selected, verifier, plan.g1_allowed_end_exclusive, workers)
    verified = sum(result.outcome == _OUTCOME_VERIFIED for result in results.values())
    return AcquisitionBatchResult(
        selected_hour_count=len(selected),
        verified_hour_count=verified,
        failed_hour_count=len(selected) - verified,
        remaining_uncached_hour_count=len(uncached) - verified,
    )


def _acquire_hours(
    hours: Sequence[datetime],
    verifier: DirectDukascopyAcquirer,
    cutoff: datetime,
    workers: int,
    symbol: str = "EURUSD",
) -> dict[datetime, HourVerification]:
    results: dict[datetime, HourVerification] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(verifier.acquire, hour, cutoff): hour for hour in hours
        }
        for future in as_completed(futures):
            hour = futures[future]
            try:
                result = future.result()
            except Exception:
                result = _failed_verification(
                    hour,
                    _dukascopy_tick_url(symbol, hour),
                    "worker_failure",
                )
            results[hour] = result
    return results


def build_reconciliation_payload(
    *,
    plan: ResearchDataPlan,
    plan_file_sha256: str,
    provenance: dict[str, Any],
    provenance_file_sha256: str,
    coverage: dict[str, Any],
    coverage_file_sha256: str,
    missing_minutes: Sequence[_MissingMinute],
    offline_domains: Sequence[OfflineDomain],
    offline_provenance: dict[str, Any],
    hour_results: Mapping[datetime, HourVerification],
    instrument_id: str = INSTRUMENT_ID,
    symbol: str = "EURUSD",
) -> dict[str, Any]:
    """Build the deterministic semantic payload from already validated inputs."""

    cutoff = plan.g1_allowed_end_exclusive
    for hour, result in hour_results.items():
        _require_utc_hour(hour, "hour result key")
        if hour + _HOUR > cutoff or result.hour_start_utc != hour:
            raise ReconciliationValidationError(
                "independent verification result crosses the holdout boundary"
            )
        _validate_hour_result(result, symbol)

    decisions = tuple(
        _classify_minute(item, offline_domains, hour_results)
        for item in missing_minutes
    )
    intervals = _compress_decisions(decisions)
    by_class = {
        classification: tuple(
            item for item in intervals if item.classification == classification
        )
        for classification in (
            _CLASS_PROVIDER_OFFLINE,
            _CLASS_VERIFIED_NO_TICK,
            _CLASS_UNEXPLAINED,
        )
    }
    minute_counts = {
        classification: sum(item.minutes for item in items)
        for classification, items in by_class.items()
    }
    interval_counts = {
        classification: len(items) for classification, items in by_class.items()
    }
    original_minutes = len(missing_minutes)
    if sum(minute_counts.values()) != original_minutes:
        raise AssertionError("reconciliation classifications do not partition gaps")
    structural_valid = coverage.get("source_derived_structural_integrity_valid") is True
    holdout_rows = cast(dict[str, Any], provenance["qa"]).get("holdout_rows_admitted")
    ready = (
        structural_valid
        and minute_counts[_CLASS_UNEXPLAINED] == 0
        and holdout_rows == 0
    )
    query_rows = [_hour_provenance(hour_results[key]) for key in sorted(hour_results)]
    original_interval_count = len(
        cast(list[object], coverage["unexplained_missing_intervals_utc"])
    )
    return {
        "reconciliation_version": RECONCILIATION_VERSION,
        "instrument_id": instrument_id,
        "research_plan": {
            "file_sha256": plan_file_sha256,
            "semantic_sha256": plan.semantic_sha256,
            "dataset_id": plan.dataset_id,
        },
        "canonical_provenance": {
            "filename": PARENT_MANIFEST_FILENAME,
            "file_sha256": provenance_file_sha256,
            "semantic_sha256": provenance.get("semantic_sha256"),
            "ingestion_version": provenance.get("ingestion_version"),
            "distribution_source": provenance.get("distribution_source"),
            "dataset_repo": provenance.get("dataset_repo"),
            "dataset_revision": provenance.get("dataset_revision"),
        },
        "session_coverage": {
            "filename": COVERAGE_MANIFEST_FILENAME,
            "file_sha256": coverage_file_sha256,
            "semantic_sha256": coverage.get("semantic_sha256"),
            "coverage_qa_version": coverage.get("coverage_qa_version"),
            "session_policy_version": cast(dict[str, Any], coverage["session_policy"])[
                "version"
            ],
            "immutable": True,
        },
        "permitted_utc_interval": {
            "start_utc": datetime.combine(
                plan.development.start, datetime.min.time(), tzinfo=UTC
            )
            .isoformat()
            .replace("+00:00", "Z"),
            "end_exclusive_utc": _format_utc(cutoff),
        },
        "classification_semantics": {
            "expected_closure": (
                "existing recurring weekly closure; retained in the immutable "
                "session coverage artifact and not reclassified here"
            ),
            "provider_offline": (
                "missing expected-open paired minute exactly covered by a validated "
                "Dukascopy JForex offline domain"
            ),
            "verified_no_tick": (
                "missing expected-open paired minute whose successfully retrieved "
                "and validated direct Dukascopy BI5 hour contains zero ticks"
            ),
            "unexplained_missing": (
                "missing expected-open minute with a canonical side mismatch, direct "
                "tick, absent verification, or failed/ambiguous retrieval"
            ),
            "classification_order": [
                "canonical_side_mismatch",
                "independent_tick_present",
                "provider_offline",
                "verified_no_tick",
                "unexplained_missing",
            ],
        },
        "provider_offline_evidence": offline_provenance,
        "independent_tick_verification": {
            "version": DIRECT_VERIFICATION_VERSION,
            "market_data_origin": "Dukascopy",
            "distribution_source": "direct Dukascopy hourly BI5",
            "base_url": DIRECT_SOURCE_URL,
            "adopted_package": "tradedesk-dukascopy",
            "package_version": version("tradedesk-dukascopy"),
            "package_revision": UPSTREAM_COMMIT,
            "package_license": "Apache-2.0",
            "query_granularity": (
                "one provider-native hourly BI5 object per unique gap hour"
            ),
            "http_404_is_zero_tick_proof": False,
            "retrieval_failure_is_zero_tick_proof": False,
            "hour_query_count": len(query_rows),
            "hours": query_rows,
        },
        "counts": {
            "original_unexplained_interval_count": original_interval_count,
            "original_unexplained_minute_count": original_minutes,
            "provider_offline_interval_count": interval_counts[_CLASS_PROVIDER_OFFLINE],
            "provider_offline_minute_count": minute_counts[_CLASS_PROVIDER_OFFLINE],
            "verified_no_tick_interval_count": interval_counts[_CLASS_VERIFIED_NO_TICK],
            "verified_no_tick_minute_count": minute_counts[_CLASS_VERIFIED_NO_TICK],
            "unexplained_missing_interval_count": interval_counts[_CLASS_UNEXPLAINED],
            "unexplained_missing_minute_count": minute_counts[_CLASS_UNEXPLAINED],
        },
        "provider_offline_intervals_utc": [
            asdict(item) for item in by_class[_CLASS_PROVIDER_OFFLINE]
        ],
        "verified_no_tick_intervals_utc": [
            asdict(item) for item in by_class[_CLASS_VERIFIED_NO_TICK]
        ],
        "unexplained_missing_intervals_utc": [
            asdict(item) for item in by_class[_CLASS_UNEXPLAINED]
        ],
        "source_derived_structural_integrity_valid": structural_valid,
        "holdout_rows_admitted": holdout_rows,
        "holdout_accessed": False,
        "fills_or_interpolation": False,
        "canonical_or_derived_data_modified": False,
        "session_aware_research_ready": ready,
        "semantic_hash_contract": (
            "SHA-256 of canonical JSON for every other manifest field; no fetch or "
            "wall-clock timestamp and no cache-hit state is included"
        ),
    }


def _validate_inputs(
    plan: ResearchDataPlan,
    plan_path: Path,
    provenance: dict[str, Any],
    provenance_path: Path,
    coverage: dict[str, Any],
    coverage_path: Path,
    *,
    instrument_spec: InstrumentSpec = EURUSD_SPEC,
    session_policy_version: str = SESSION_POLICY_VERSION,
    legacy: bool = True,
) -> tuple[_MissingMinute, ...]:
    if version("tradedesk-dukascopy") != UPSTREAM_VERSION:
        raise ReconciliationValidationError(
            f"expected tradedesk-dukascopy {UPSTREAM_VERSION}"
        )
    identity = (
        validate_canonical_eurusd_source_manifest(provenance)
        if legacy
        else validate_canonical_source_manifest(provenance, instrument_spec)
    )
    if identity.ingestion_version not in {
        HF_INGESTION_VERSION,
        CORRECTED_INGESTION_VERSION,
        GENERIC_CORRECTED_INGESTION_VERSION,
    }:
        raise ReconciliationValidationError(
            "reconciliation requires the frozen or explicitly corrected G1 source"
        )
    expected_provenance = {
        "dataset_id": plan.dataset_id,
        "dataset_plan_sha256": plan.semantic_sha256,
        "dataset_repo": plan.huggingface_repo,
        "dataset_revision": plan.huggingface_revision,
        "g1_cutoff_end_exclusive_utc": _format_utc(plan.g1_allowed_end_exclusive),
    }
    for field, expected in expected_provenance.items():
        if provenance.get(field) != expected:
            raise ReconciliationValidationError(
                f"canonical provenance does not match research plan field {field}"
            )
    qa = provenance.get("qa")
    if not isinstance(qa, dict) or qa.get("holdout_rows_admitted") != 0:
        raise ReconciliationValidationError(
            "canonical provenance does not prove zero holdout admission"
        )
    requested = provenance.get("requested_utc_range")
    if requested != {
        "start_date": plan.development.start.isoformat(),
        "end_date_inclusive": plan.validation.end.isoformat(),
    }:
        raise ReconciliationValidationError(
            "canonical provenance range is not exactly development plus validation"
        )
    _validate_semantic_hash(coverage, "session coverage report")
    if coverage.get("coverage_qa_version") != COVERAGE_QA_VERSION:
        raise ReconciliationValidationError("session coverage version is incompatible")
    session_policy = coverage.get("session_policy")
    if (
        not isinstance(session_policy, dict)
        or session_policy.get("version") != session_policy_version
    ):
        raise ReconciliationValidationError("session policy version is incompatible")
    if coverage.get("parent_g0_5_manifest_sha256") != _sha256(provenance_path):
        raise ReconciliationValidationError(
            "session coverage is not bound to canonical provenance"
        )
    expected_range = {
        "start_utc": _format_utc(
            datetime.combine(plan.development.start, datetime.min.time(), tzinfo=UTC)
        ),
        "end_exclusive_utc": _format_utc(plan.g1_allowed_end_exclusive),
    }
    if coverage.get("requested_utc_interval") != expected_range:
        raise ReconciliationValidationError(
            "session coverage range is not exactly development plus validation"
        )
    counts = coverage.get("counts")
    raw_intervals = coverage.get("unexplained_missing_intervals_utc")
    if not isinstance(counts, dict) or not isinstance(raw_intervals, list):
        raise ReconciliationValidationError("session coverage gap structure is invalid")
    missing = _expand_missing_intervals(raw_intervals, plan.g1_allowed_end_exclusive)
    expected_missing = counts.get("unexplained_missing_minute_count")
    if expected_missing != len(missing):
        raise ReconciliationValidationError(
            "session coverage unexplained count does not match exact intervals"
        )
    open_count = counts.get("expected_open_minute_count")
    paired_open_count = counts.get("observed_paired_expected_open_minute_count")
    paired_count = counts.get("observed_paired_minute_count")
    expected_closed_count = counts.get("expected_closure_minute_count")
    requested_count = counts.get("requested_minute_count")
    requested_minutes = int(
        (
            plan.g1_allowed_end_exclusive
            - datetime.combine(plan.development.start, datetime.min.time(), tzinfo=UTC)
        )
        / _MINUTE
    )
    if (
        not isinstance(open_count, int)
        or isinstance(open_count, bool)
        or not isinstance(paired_open_count, int)
        or isinstance(paired_open_count, bool)
        or not isinstance(paired_count, int)
        or isinstance(paired_count, bool)
        or not isinstance(expected_closed_count, int)
        or isinstance(expected_closed_count, bool)
        or requested_count != requested_minutes
        or paired_open_count + len(missing) != open_count
        or paired_count + expected_closed_count + len(missing) != requested_count
        or paired_count != qa.get("bid_bar_count")
        or paired_count != qa.get("ask_bar_count")
    ):
        raise ReconciliationValidationError(
            "session coverage counts do not reconcile with canonical provenance"
        )
    if coverage.get("source_derived_structural_integrity_valid") is not True:
        raise ReconciliationValidationError(
            "source/derived structural integrity is not valid"
        )
    return missing


def _expand_missing_intervals(
    raw_intervals: Sequence[object], cutoff: datetime
) -> tuple[_MissingMinute, ...]:
    expanded: list[_MissingMinute] = []
    previous: datetime | None = None
    for raw in raw_intervals:
        if not isinstance(raw, dict):
            raise ReconciliationValidationError("coverage interval must be an object")
        item = cast(dict[str, Any], raw)
        if item.get("classification") != _CLASS_UNEXPLAINED:
            raise ReconciliationValidationError(
                "coverage gap has invalid classification"
            )
        start = _parse_utc_minute(item.get("start_utc"), "coverage interval start")
        end = _parse_utc_minute(item.get("end_utc"), "coverage interval end")
        if end < start or end >= cutoff:
            raise ReconciliationValidationError(
                "coverage interval is invalid or reaches sealed holdout"
            )
        minutes = int((end - start) / _MINUTE) + 1
        if item.get("minutes") != minutes:
            raise ReconciliationValidationError(
                "coverage interval minute count mismatch"
            )
        raw_sides = item.get("missing_sides")
        if (
            not isinstance(raw_sides, list)
            or not raw_sides
            or not all(side in _EXPECTED_MISSING_SIDES for side in raw_sides)
            or len(set(raw_sides)) != len(raw_sides)
        ):
            raise ReconciliationValidationError(
                "coverage interval missing sides invalid"
            )
        sides = tuple(cast(list[str], raw_sides))
        current = start
        while current <= end:
            if previous is not None and current <= previous:
                raise ReconciliationValidationError(
                    "coverage intervals overlap or are not ordered"
                )
            if not is_eurusd_expected_open(current):
                raise ReconciliationValidationError(
                    "coverage unexplained interval includes an expected closure"
                )
            expanded.append(_MissingMinute(current, sides))
            previous = current
            current += _MINUTE
    return tuple(expanded)


def _load_offline_evidence(
    path: Path | None, plan: ResearchDataPlan
) -> tuple[tuple[OfflineDomain, ...], dict[str, Any]]:
    if path is None:
        return (), {
            "used": False,
            "authoritative_interface": (
                "Dukascopy JForex IDataService.getOfflineTimeDomains"
            ),
            "reason": "no exact semantically hashed offline-domain export supplied",
            "domains": [],
        }
    document = _load_json_object(path, "provider offline evidence")
    _validate_semantic_hash(document, "provider offline evidence")
    expected = {
        "schema_version": OFFLINE_EVIDENCE_VERSION,
        "provider": "Dukascopy",
        "interface": "IDataService.getOfflineTimeDomains",
        "instrument_scope": "provider-wide",
        "requested_start_utc": _format_utc(
            datetime.combine(plan.development.start, datetime.min.time(), tzinfo=UTC)
        ),
        "requested_end_exclusive_utc": _format_utc(plan.g1_allowed_end_exclusive),
    }
    for field, value in expected.items():
        if document.get(field) != value:
            raise ReconciliationValidationError(
                f"provider offline evidence has invalid {field}"
            )
    api_version = document.get("api_version")
    retrieval = document.get("retrieval_details")
    raw_domains = document.get("domains")
    if (
        not isinstance(api_version, str)
        or not api_version
        or not isinstance(retrieval, str)
        or not retrieval
        or not isinstance(raw_domains, list)
    ):
        raise ReconciliationValidationError(
            "provider offline evidence lacks retrieval provenance"
        )
    domains: list[OfflineDomain] = []
    previous_end: datetime | None = None
    for index, raw in enumerate(raw_domains):
        if not isinstance(raw, dict):
            raise ReconciliationValidationError("offline domain must be an object")
        item = cast(dict[str, Any], raw)
        start = _parse_utc_minute(item.get("start_utc"), "offline domain start")
        end = _parse_utc_minute(item.get("end_exclusive_utc"), "offline domain end")
        if end <= start or end > plan.g1_allowed_end_exclusive:
            raise ReconciliationValidationError(
                "offline domain is empty or reaches beyond the permitted range"
            )
        if previous_end is not None and start < previous_end:
            raise ReconciliationValidationError(
                "offline domains overlap or are unordered"
            )
        evidence_id = item.get("evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id:
            evidence_id = f"jforex-domain-{index:05d}"
        domains.append(OfflineDomain(evidence_id, start, end))
        previous_end = end
    return tuple(domains), {
        "used": True,
        "filename": path.name,
        "file_sha256": _sha256(path),
        "semantic_sha256": document["semantic_sha256"],
        "schema_version": OFFLINE_EVIDENCE_VERSION,
        "provider": "Dukascopy",
        "interface": "IDataService.getOfflineTimeDomains",
        "api_version": api_version,
        "retrieval_details": retrieval,
        "domains": [
            {
                "evidence_id": item.evidence_id,
                "start_utc": _format_utc(item.start_utc),
                "end_exclusive_utc": _format_utc(item.end_exclusive_utc),
            }
            for item in domains
        ],
    }


def _classify_minute(
    item: _MissingMinute,
    offline_domains: Sequence[OfflineDomain],
    hour_results: Mapping[datetime, HourVerification],
) -> _MinuteDecision:
    if item.missing_sides != _EXPECTED_MISSING_SIDES:
        return _MinuteDecision(
            item.timestamp,
            item.missing_sides,
            _CLASS_UNEXPLAINED,
            "canonical_bid_ask_coverage_mismatch",
            None,
        )
    hour = _floor_hour(item.timestamp)
    verification = hour_results.get(hour)
    if (
        verification is not None
        and verification.outcome == _OUTCOME_VERIFIED
        and item.timestamp in verification.tick_minutes
    ):
        return _MinuteDecision(
            item.timestamp,
            item.missing_sides,
            _CLASS_UNEXPLAINED,
            "independent_direct_source_contains_tick",
            _hour_evidence_id(hour),
        )
    offline = _offline_for_minute(item.timestamp, offline_domains)
    if offline is not None:
        return _MinuteDecision(
            item.timestamp,
            item.missing_sides,
            _CLASS_PROVIDER_OFFLINE,
            "exact_authoritative_offline_domain",
            offline.evidence_id,
        )
    if verification is None:
        return _MinuteDecision(
            item.timestamp,
            item.missing_sides,
            _CLASS_UNEXPLAINED,
            "independent_verification_missing",
            None,
        )
    if verification.outcome != _OUTCOME_VERIFIED:
        return _MinuteDecision(
            item.timestamp,
            item.missing_sides,
            _CLASS_UNEXPLAINED,
            verification.failure_reason or "independent_retrieval_failed",
            _hour_evidence_id(hour),
        )
    return _MinuteDecision(
        item.timestamp,
        item.missing_sides,
        _CLASS_VERIFIED_NO_TICK,
        "validated_direct_bi5_contains_zero_ticks_for_minute",
        _hour_evidence_id(hour),
    )


def _compress_decisions(
    decisions: Sequence[_MinuteDecision],
) -> tuple[ReconciledInterval, ...]:
    if not decisions:
        return ()
    intervals: list[ReconciledInterval] = []
    start = decisions[0].timestamp
    end = start
    classification = decisions[0].classification
    sides = decisions[0].missing_sides
    reason = decisions[0].reason
    refs: list[str] = []
    if decisions[0].evidence_ref is not None:
        refs.append(decisions[0].evidence_ref)
    for item in decisions[1:]:
        contiguous = item.timestamp == end + _MINUTE
        same = (
            item.classification == classification
            and item.missing_sides == sides
            and item.reason == reason
        )
        if contiguous and same:
            end = item.timestamp
            if item.evidence_ref is not None and item.evidence_ref not in refs:
                refs.append(item.evidence_ref)
            continue
        intervals.append(
            _reconciled_interval(classification, start, end, sides, reason, refs)
        )
        start = end = item.timestamp
        classification = item.classification
        sides = item.missing_sides
        reason = item.reason
        refs = [item.evidence_ref] if item.evidence_ref is not None else []
    intervals.append(
        _reconciled_interval(classification, start, end, sides, reason, refs)
    )
    return tuple(intervals)


def _reconciled_interval(
    classification: str,
    start: datetime,
    end: datetime,
    sides: tuple[str, ...],
    reason: str,
    refs: Sequence[str],
) -> ReconciledInterval:
    return ReconciledInterval(
        classification=classification,
        start_utc=_format_utc(start),
        end_utc=_format_utc(end),
        minutes=int((end - start) / _MINUTE) + 1,
        missing_sides=sides,
        reason=reason,
        evidence_refs=tuple(refs),
    )


def _verification_from_payload(
    hour: datetime,
    source_url: str,
    payload: bytes,
    cache_relative_path: str,
) -> HourVerification:
    tick_minutes: set[datetime] = set()
    tick_count = 0
    if payload:
        raw = lzma.decompress(payload)
        if not raw or len(raw) % _TICK_RECORD_SIZE:
            raise ReconciliationValidationError(
                "invalid decompressed BI5 record length"
            )
        previous_ms: int | None = None
        for offset in range(0, len(raw), _TICK_RECORD_SIZE):
            milliseconds = struct.unpack_from(">i", raw, offset)[0]
            if not 0 <= milliseconds < 3_600_000:
                raise ReconciliationValidationError(
                    "BI5 tick offset is outside its hour"
                )
            if previous_ms is not None and milliseconds < previous_ms:
                raise ReconciliationValidationError("BI5 ticks are not chronological")
            previous_ms = milliseconds
            tick_minutes.add(hour + timedelta(minutes=milliseconds // 60_000))
            tick_count += 1
        price_format = _probe_price_format(payload)
        decoded = _decode_ticks(
            hour, payload, price_format=price_format, price_divisor=100_000.0
        )
        if len(decoded) != tick_count:
            raise ReconciliationValidationError("BI5 decoder tick count mismatch")
        for tick in decoded:
            if (
                not hour <= tick.ts < hour + _HOUR
                or not all(
                    math.isfinite(value)
                    for value in (tick.bid, tick.ask, tick.bid_vol, tick.ask_vol)
                )
                or tick.ask < tick.bid
                or tick.bid_vol < 0
                or tick.ask_vol < 0
            ):
                raise ReconciliationValidationError(
                    "BI5 tick lacks valid paired BID/ASK source values"
                )
    return HourVerification(
        hour_start_utc=hour,
        source_url=source_url,
        outcome=_OUTCOME_VERIFIED,
        response_status=200,
        tick_minutes=tuple(sorted(tick_minutes)),
        tick_count=tick_count,
        content_size=len(payload),
        content_sha256=hashlib.sha256(payload).hexdigest(),
        cache_relative_path=cache_relative_path,
        failure_reason=None,
    )


def _validate_hour_result(result: HourVerification, symbol: str = "EURUSD") -> None:
    hour = _require_utc_hour(result.hour_start_utc, "verification hour")
    if result.source_url != _dukascopy_tick_url(symbol, hour):
        raise ReconciliationValidationError("verification source URL mismatch")
    if result.outcome == _OUTCOME_VERIFIED:
        if (
            result.response_status != 200
            or result.tick_count is None
            or result.tick_count < 0
            or result.content_size is None
            or result.content_size < 0
            or not _is_sha256(result.content_sha256)
            or result.failure_reason is not None
        ):
            raise ReconciliationValidationError("verified BI5 outcome is incomplete")
        for minute in result.tick_minutes:
            _require_utc_minute(minute, "verified tick minute")
            if not hour <= minute < hour + _HOUR:
                raise ReconciliationValidationError(
                    "verified tick minute falls outside its hour"
                )
    elif result.outcome == _OUTCOME_FAILED:
        if result.failure_reason is None or result.tick_minutes:
            raise ReconciliationValidationError("failed BI5 outcome is invalid")
    else:
        raise ReconciliationValidationError("unknown BI5 verification outcome")


def _hour_provenance(result: HourVerification) -> dict[str, Any]:
    return {
        "evidence_id": _hour_evidence_id(result.hour_start_utc),
        "hour_start_utc": _format_utc(result.hour_start_utc),
        "source_url": result.source_url,
        "outcome": result.outcome,
        "response_status": result.response_status,
        "tick_count": result.tick_count,
        "content_size": result.content_size,
        "content_sha256": result.content_sha256,
        "cache_relative_path": result.cache_relative_path,
        "failure_reason": result.failure_reason,
    }


def _failed_verification(
    hour: datetime, source_url: str, reason: str
) -> HourVerification:
    return HourVerification(
        hour_start_utc=hour,
        source_url=source_url,
        outcome=_OUTCOME_FAILED,
        response_status=None,
        tick_minutes=(),
        tick_count=None,
        content_size=None,
        content_sha256=None,
        cache_relative_path=None,
        failure_reason=reason,
    )


def _offline_for_minute(
    minute: datetime, domains: Sequence[OfflineDomain]
) -> OfflineDomain | None:
    for domain in domains:
        if domain.start_utc <= minute < domain.end_exclusive_utc:
            return domain
        if domain.start_utc > minute:
            break
    return None


def _http_get(url: str) -> HttpResponse:
    request = urllib.request.Request(  # noqa: S310 - fixed HTTPS provider URL
        url,
        headers={
            "User-Agent": (
                "FTMOQuant-session-reconciliation/1 "
                "(tradedesk-dukascopy-compatible BI5 verifier)"
            )
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=20.0) as response:  # noqa: S310
            return HttpResponse(status=int(response.status), body=response.read())
    except urllib.error.HTTPError as error:
        return HttpResponse(status=error.code, body=b"")


def _validate_semantic_hash(document: dict[str, Any], description: str) -> None:
    actual = document.get("semantic_sha256")
    if not _is_sha256(actual):
        raise ReconciliationValidationError(f"{description} lacks semantic SHA-256")
    payload = {
        key: value for key, value in document.items() if key != "semantic_sha256"
    }
    if actual != _semantic_sha256(payload):
        raise ReconciliationValidationError(
            f"{description} semantic SHA-256 does not match"
        )


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReconciliationValidationError(
            f"invalid {description}: {error}"
        ) from error
    if not isinstance(raw, dict):
        raise ReconciliationValidationError(f"{description} must be an object")
    return cast(dict[str, Any], raw)


def _parse_utc_minute(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ReconciliationValidationError(f"{field} must be an ISO UTC minute")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ReconciliationValidationError(
            f"{field} must be an ISO UTC minute"
        ) from error
    return _require_utc_minute(parsed, field)


def _require_utc_minute(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ReconciliationValidationError(f"{field} must be timezone-aware UTC")
    normalized = value.astimezone(UTC)
    if value.utcoffset() != timedelta(0) or normalized.second or normalized.microsecond:
        raise ReconciliationValidationError(f"{field} must be an exact UTC minute")
    return normalized


def _require_utc_hour(value: datetime, field: str) -> datetime:
    normalized = _require_utc_minute(value, field)
    if normalized.minute:
        raise ReconciliationValidationError(f"{field} must be an exact UTC hour")
    return normalized


def _floor_hour(value: datetime) -> datetime:
    return value.replace(minute=0, second=0, microsecond=0)


def _relative_cache_path(hour: datetime, symbol: str = "EURUSD") -> str:
    return (
        f"{symbol}/{hour.year:04d}/{hour.month - 1:02d}/{hour.day:02d}/"
        f"{hour.hour:02d}h_ticks.bi5"
    )


def _hour_evidence_id(hour: datetime) -> str:
    return "direct-bi5-" + hour.strftime("%Y%m%dT%H00Z")


def _format_utc(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _semantic_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconcile Dukascopy EUR/USD expected-open source gaps"
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--provider-offline-evidence", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--acquire-only-max-hours",
        type=int,
        help=(
            "validate inputs and warm at most this many uncached BI5 hours; "
            "do not write a reconciliation artifact"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run deterministic expected-open gap reconciliation."""

    args = _build_parser().parse_args(argv)
    acquire_only = cast(int | None, args.acquire_only_max_hours)
    if acquire_only is not None:
        batch = acquire_eurusd_reconciliation_batch(
            cast(Path, args.plan),
            cast(Path, args.output_root),
            cast(Path, args.cache_root),
            offline_evidence_path=cast(Path | None, args.provider_offline_evidence),
            workers=cast(int, args.workers),
            max_uncached_hours=acquire_only,
        )
        print(f"selected hours: {batch.selected_hour_count}")
        print(f"verified hours: {batch.verified_hour_count}")
        print(f"failed hours: {batch.failed_hour_count}")
        print(f"remaining uncached hours: {batch.remaining_uncached_hour_count}")
        print("reconciliation artifact written: false")
        return 0
    result = run_eurusd_session_reconciliation(
        cast(Path, args.plan),
        cast(Path, args.output_root),
        cast(Path, args.cache_root),
        offline_evidence_path=cast(Path | None, args.provider_offline_evidence),
        workers=cast(int, args.workers),
    )
    counts = result.counts
    print(
        "original unexplained: "
        f"{counts['original_unexplained_interval_count']} intervals / "
        f"{counts['original_unexplained_minute_count']} minutes"
    )
    print(
        "provider_offline: "
        f"{counts['provider_offline_interval_count']} intervals / "
        f"{counts['provider_offline_minute_count']} minutes"
    )
    print(
        "verified_no_tick: "
        f"{counts['verified_no_tick_interval_count']} intervals / "
        f"{counts['verified_no_tick_minute_count']} minutes"
    )
    print(
        "still unexplained: "
        f"{counts['unexplained_missing_interval_count']} intervals / "
        f"{counts['unexplained_missing_minute_count']} minutes"
    )
    print(
        "session_aware_research_ready: "
        f"{str(result.session_aware_research_ready).lower()}"
    )
    print(f"reconciliation artifact: {result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
