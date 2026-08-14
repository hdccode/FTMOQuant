"""Immutable correction of the seven independently proven EUR/USD omissions."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from nautilus_trader.model import Bar
from nautilus_trader.persistence import ParquetDataCatalog
from tradedesk_dukascopy.export import (  # type: ignore[import-untyped]
    _decode_ticks,
    _probe_price_format,
)

from ftmoquant.data.canonical_source import (
    CORRECTED_DISTRIBUTION_SOURCE,
    CORRECTED_INGESTION_VERSION,
    HF_INGESTION_VERSION,
    validate_canonical_eurusd_source_manifest,
)
from ftmoquant.data.derived_bars import (
    DERIVED_MANIFEST_FILENAME,
    PARENT_MANIFEST_FILENAME,
    _update_bar_digest,
)
from ftmoquant.data.dukascopy import INSTRUMENT_ID, SourceBar, _to_nautilus_bars
from ftmoquant.data.hf_dukascopy import (
    _advance_minute_aggregate,
    _datetime_ms,
    _decimal_float,
    _MinuteAggregate,
    _source_bar_pair,
    _validate_paired_bars,
)
from ftmoquant.data.research_plan import ResearchDataPlan, load_research_data_plan
from ftmoquant.data.session_coverage import COVERAGE_MANIFEST_FILENAME
from ftmoquant.data.session_reconciliation import (
    RECONCILIATION_MANIFEST_FILENAME,
    DirectDukascopyAcquirer,
)

CORRECTION_ID = CORRECTED_INGESTION_VERSION
READINESS_VERSION = "g1-research-readiness-1"
READINESS_MANIFEST_FILENAME = "ftmoquant_research_readiness.json"
CORRECTION_IMPLEMENTATION = (
    "ftmoquant.data.hf_dukascopy._advance_minute_aggregate+_source_bar_pair"
)

_MINUTE = timedelta(minutes=1)
_MINUTE_NS = 60_000_000_000
_SCAN_CHUNK_MINUTES = 10_000
_SIDES = ("BID", "ASK")
_EXPECTED_TICK_COUNTS = {
    "2023-11-14T13:31:00Z": 537,
    "2023-11-29T14:29:00Z": 68,
    "2023-11-29T14:30:00Z": 127,
    "2023-11-29T14:31:00Z": 140,
    "2023-11-29T14:32:00Z": 182,
    "2023-11-29T14:33:00Z": 174,
    "2023-11-29T14:34:00Z": 153,
}
_EXPECTED_MINUTES = tuple(
    datetime.fromisoformat(value.replace("Z", "+00:00"))
    for value in _EXPECTED_TICK_COUNTS
)


class CanonicalCorrectionValidationError(ValueError):
    """Raised when correction evidence or equivalence cannot be proven."""


@dataclass(frozen=True, slots=True)
class CorrectionResult:
    """Stable result of one immutable canonical correction."""

    output_root: Path
    manifest_path: Path
    semantic_sha256: str
    correction_semantic_sha256: str
    bid_bar_count: int
    ask_bar_count: int
    bid_content_sha256: str
    ask_content_sha256: str


@dataclass(frozen=True, slots=True)
class ReadinessResult:
    """Persisted final G1 readiness state."""

    manifest_path: Path
    semantic_sha256: str
    research_ready: bool


def build_corrected_eurusd_dataset(
    plan_path: Path,
    parent_root: Path,
    output_root: Path,
    cache_root: Path,
) -> CorrectionResult:
    """Create a new canonical root containing only the seven proven corrections."""

    plan = load_research_data_plan(plan_path)
    parent = parent_root.resolve()
    output = output_root.resolve()
    cache = cache_root.resolve()
    _validate_roots(parent, output)
    parent_manifest_path = parent / PARENT_MANIFEST_FILENAME
    reconciliation_path = parent / RECONCILIATION_MANIFEST_FILENAME
    parent_manifest = _load_semantic_json(parent_manifest_path, "parent provenance")
    identity = validate_canonical_eurusd_source_manifest(parent_manifest)
    if identity.ingestion_version != HF_INGESTION_VERSION:
        raise CanonicalCorrectionValidationError(
            "correction parent must be the frozen Hugging Face canonical version"
        )
    _validate_parent_plan(parent_manifest, plan)
    reconciliation = _load_semantic_json(
        reconciliation_path, "parent reconciliation"
    )
    _validate_reconciliation(
        reconciliation, _sha256(parent_manifest_path), plan
    )
    parent_tree_before = _sha256_tree(parent)
    minute_bars, minute_evidence = _correction_bars(
        reconciliation, cache, plan.g1_allowed_end_exclusive
    )

    parent_catalog_path = parent / "catalog"
    parent_catalog = ParquetDataCatalog(str(parent_catalog_path))
    _reject_existing_minutes(parent_catalog, minute_bars)
    _copy_canonical_catalog(parent_catalog_path, output / "catalog")
    corrected_catalog = ParquetDataCatalog(str(output / "catalog"))
    bid_bars = _to_nautilus_bars(
        [minute_bars[value][0] for value in _EXPECTED_MINUTES], "BID"
    )
    ask_bars = _to_nautilus_bars(
        [minute_bars[value][1] for value in _EXPECTED_MINUTES], "ASK"
    )
    _validate_paired_bars(bid_bars, ask_bars)
    _populate_corrected_catalog(
        parent_catalog,
        corrected_catalog,
        parent_manifest,
        bid_bars,
        ask_bars,
    )

    equivalence, content = _prove_parent_equivalence(
        parent_catalog,
        corrected_catalog,
        parent_manifest,
        bid_bars,
        ask_bars,
    )
    corrected_payload = _corrected_manifest_payload(
        parent_manifest=parent_manifest,
        parent_manifest_sha256=_sha256(parent_manifest_path),
        plan_file_sha256=_sha256(plan_path),
        reconciliation_file_sha256=_sha256(reconciliation_path),
        reconciliation_semantic_sha256=cast(str, reconciliation["semantic_sha256"]),
        evidence=minute_evidence,
        equivalence=equivalence,
        content=content,
    )
    semantic_sha256 = _semantic_sha256(corrected_payload)
    manifest = {**corrected_payload, "semantic_sha256": semantic_sha256}
    manifest_path = output / PARENT_MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validate_canonical_eurusd_source_manifest(manifest)
    parent_tree_after = _sha256_tree(parent)
    if parent_tree_after != parent_tree_before:
        raise CanonicalCorrectionValidationError(
            "parent dataset changed during correction build"
        )
    return CorrectionResult(
        output_root=output,
        manifest_path=manifest_path,
        semantic_sha256=semantic_sha256,
        correction_semantic_sha256=cast(
            str, cast(dict[str, Any], manifest["correction"])["semantic_sha256"]
        ),
        bid_bar_count=cast(int, cast(dict[str, Any], manifest["qa"])["bid_bar_count"]),
        ask_bar_count=cast(int, cast(dict[str, Any], manifest["qa"])["ask_bar_count"]),
        bid_content_sha256=cast(
            str, cast(dict[str, Any], manifest["canonical_content"])["BID"]
        ),
        ask_content_sha256=cast(
            str, cast(dict[str, Any], manifest["canonical_content"])["ASK"]
        ),
    )


def freeze_eurusd_research_readiness(
    plan_path: Path, output_root: Path
) -> ReadinessResult:
    """Write final readiness only after every corrected-data gate is proven."""

    plan = load_research_data_plan(plan_path)
    root = output_root.resolve()
    canonical_path = root / PARENT_MANIFEST_FILENAME
    derived_path = root / DERIVED_MANIFEST_FILENAME
    coverage_path = root / COVERAGE_MANIFEST_FILENAME
    reconciliation_path = root / RECONCILIATION_MANIFEST_FILENAME
    canonical = _load_semantic_json(canonical_path, "corrected canonical provenance")
    identity = validate_canonical_eurusd_source_manifest(canonical)
    if identity.ingestion_version != CORRECTED_INGESTION_VERSION:
        raise CanonicalCorrectionValidationError(
            "readiness requires the corrected canonical identity"
        )
    _validate_parent_plan(canonical, plan)
    derived = _load_json(derived_path, "derived provenance")
    coverage = _load_semantic_json(coverage_path, "session coverage")
    reconciliation = _load_semantic_json(
        reconciliation_path, "session reconciliation"
    )
    canonical_sha = _sha256(canonical_path)
    derived_sha = _sha256(derived_path)
    coverage_sha = _sha256(coverage_path)
    reconciliation_sha = _sha256(reconciliation_path)
    counts = cast(dict[str, Any], reconciliation.get("counts"))
    coverage_counts = cast(dict[str, Any], coverage.get("counts"))
    verified_no_tick_count = counts.get("verified_no_tick_minute_count")
    provider_offline_count = counts.get("provider_offline_minute_count")
    correction = cast(dict[str, Any], canonical["correction"])
    equivalence = cast(dict[str, Any], canonical["parent_equivalence"])
    reconciliation_canonical = reconciliation.get("canonical_provenance")
    reconciliation_coverage = reconciliation.get("session_coverage")
    gates = {
        "corrected_canonical_valid": True,
        "parent_correction_equivalence_valid": (
            equivalence.get("parent_bid_bars_unchanged") is True
            and equivalence.get("parent_ask_bars_unchanged") is True
            and equivalence.get("parent_timestamps_removed") == 0
            and equivalence.get("new_bid_timestamp_count") == 7
            and equivalence.get("new_ask_timestamp_count") == 7
            and equivalence.get("other_changed_or_added_bar_count") == 0
        ),
        "derived_integrity_valid": (
            derived.get("parent_g0_5_manifest_sha256") == canonical_sha
            and derived.get("derived_bar_integrity_valid") is True
            and derived.get("bid_ask_derived_coverage_matches") is True
        ),
        "coverage_integrity_valid": (
            coverage.get("parent_g0_5_manifest_sha256") == canonical_sha
            and coverage.get("structural_g0_6_manifest_sha256") == derived_sha
            and coverage.get("source_derived_structural_integrity_valid") is True
        ),
        "zero_unexplained_omissions": (
            counts.get("unexplained_missing_minute_count") == 0
        ),
        "all_remaining_missing_reconciled": (
            isinstance(verified_no_tick_count, int)
            and isinstance(provider_offline_count, int)
            and
            coverage_counts.get("unexplained_missing_minute_count")
            == verified_no_tick_count + provider_offline_count
        ),
        "verified_no_tick_unsynthesized": (
            counts.get("verified_no_tick_minute_count") == 17_728
            and coverage_counts.get("observed_paired_minute_count")
            == cast(dict[str, Any], canonical["qa"]).get("bid_bar_count")
        ),
        "reconciliation_ready": (
            reconciliation.get("session_aware_research_ready") is True
            and reconciliation.get("source_derived_structural_integrity_valid")
            is True
            and isinstance(reconciliation_canonical, dict)
            and reconciliation_canonical.get("file_sha256") == canonical_sha
            and isinstance(reconciliation_coverage, dict)
            and reconciliation_coverage.get("file_sha256") == coverage_sha
        ),
        "holdout_not_accessed": (
            canonical.get("holdout_accessed") is False
            and correction.get("holdout_accessed") is False
            and reconciliation.get("holdout_accessed") is False
        ),
        "zero_holdout_rows": (
            cast(dict[str, Any], canonical["qa"]).get("holdout_rows_admitted") == 0
            and correction.get("holdout_rows_admitted") == 0
            and reconciliation.get("holdout_rows_admitted") == 0
        ),
    }
    if not all(gates.values()):
        failed = ", ".join(name for name, passed in gates.items() if not passed)
        raise CanonicalCorrectionValidationError(
            f"research readiness remains fail-closed: {failed}"
        )
    payload = {
        "readiness_version": READINESS_VERSION,
        "dataset_identity": CORRECTION_ID,
        "canonical": {
            "filename": PARENT_MANIFEST_FILENAME,
            "file_sha256": canonical_sha,
            "semantic_sha256": canonical["semantic_sha256"],
            "correction_provenance_sha256": correction["semantic_sha256"],
            "content_sha256": canonical["canonical_content"],
        },
        "derived": {
            "filename": DERIVED_MANIFEST_FILENAME,
            "file_sha256": derived_sha,
            "series": {
                key: {
                    "emitted_bar_count": value["emitted_bar_count"],
                    "content_sha256": value["content_sha256"],
                }
                for key, value in cast(
                    dict[str, dict[str, Any]], derived["series"]
                ).items()
            },
        },
        "session_coverage": {
            "filename": COVERAGE_MANIFEST_FILENAME,
            "file_sha256": coverage_sha,
            "semantic_sha256": coverage["semantic_sha256"],
        },
        "session_reconciliation": {
            "filename": RECONCILIATION_MANIFEST_FILENAME,
            "file_sha256": reconciliation_sha,
            "semantic_sha256": reconciliation["semantic_sha256"],
            "counts": counts,
        },
        "permitted_utc_interval": {
            "start_utc": f"{plan.development.start.isoformat()}T00:00:00Z",
            "end_exclusive_utc": _format_utc(plan.g1_allowed_end_exclusive),
        },
        "holdout_cutoff_utc": _format_utc(plan.g1_allowed_end_exclusive),
        "gates": gates,
        "research_ready": True,
        "holdout_accessed": False,
        "holdout_rows_admitted": 0,
        "strategy_return_accessed": False,
        "g1_3_run": False,
        "semantic_hash_contract": (
            "SHA-256 of canonical JSON for every other manifest field"
        ),
    }
    semantic_sha256 = _semantic_sha256(payload)
    manifest = {**payload, "semantic_sha256": semantic_sha256}
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    path = root / READINESS_MANIFEST_FILENAME
    if path.exists() and path.read_bytes() != manifest_bytes:
        raise CanonicalCorrectionValidationError(
            "existing readiness artifact conflicts with validated state"
        )
    if not path.exists():
        path.write_bytes(manifest_bytes)
    return ReadinessResult(path, semantic_sha256, True)


def _validate_roots(parent: Path, output: Path) -> None:
    if not parent.is_dir():
        raise CanonicalCorrectionValidationError("parent dataset root is missing")
    if output.exists():
        raise CanonicalCorrectionValidationError(
            "corrected output root must not already exist"
        )
    if output == parent or parent in output.parents or output in parent.parents:
        raise CanonicalCorrectionValidationError(
            "parent and corrected dataset roots must be disjoint"
        )


def _validate_parent_plan(manifest: dict[str, Any], plan: ResearchDataPlan) -> None:
    expected = {
        "dataset_plan_sha256": plan.semantic_sha256,
        "dataset_repo": plan.huggingface_repo,
        "dataset_revision": plan.huggingface_revision,
        "g1_cutoff_end_exclusive_utc": _format_utc(plan.g1_allowed_end_exclusive),
        "requested_utc_range": {
            "start_date": plan.development.start.isoformat(),
            "end_date_inclusive": plan.validation.end.isoformat(),
        },
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise CanonicalCorrectionValidationError(
                f"canonical provenance does not match frozen plan field {field}"
            )
    qa = manifest.get("qa")
    if (
        not isinstance(qa, dict)
        or qa.get("holdout_rows_admitted") != 0
        or manifest.get("holdout_accessed", False) is not False
    ):
        raise CanonicalCorrectionValidationError(
            "canonical provenance does not prove the holdout remained sealed"
        )


def _validate_reconciliation(
    manifest: dict[str, Any], parent_sha256: str, plan: ResearchDataPlan
) -> None:
    canonical = manifest.get("canonical_provenance")
    counts = manifest.get("counts")
    intervals = manifest.get("unexplained_missing_intervals_utc")
    if (
        not isinstance(canonical, dict)
        or canonical.get("file_sha256") != parent_sha256
        or not isinstance(counts, dict)
        or counts.get("unexplained_missing_minute_count") != 7
        or counts.get("verified_no_tick_minute_count") != 17_728
        or manifest.get("session_aware_research_ready") is not False
        or manifest.get("holdout_accessed") is not False
        or manifest.get("holdout_rows_admitted") != 0
    ):
        raise CanonicalCorrectionValidationError(
            "parent reconciliation is not the required fail-closed seven-minute state"
        )
    found: set[str] = set()
    if not isinstance(intervals, list):
        raise CanonicalCorrectionValidationError(
            "parent reconciliation lacks exact unexplained intervals"
        )
    for raw in intervals:
        if not isinstance(raw, dict) or raw.get("reason") != (
            "independent_direct_source_contains_tick"
        ):
            raise CanonicalCorrectionValidationError(
                "parent unexplained interval is not a proven direct-source omission"
            )
        start = _parse_minute(raw.get("start_utc"))
        end = _parse_minute(raw.get("end_utc"))
        while start <= end:
            found.add(_format_utc(start))
            start += _MINUTE
    if found != set(_EXPECTED_TICK_COUNTS):
        raise CanonicalCorrectionValidationError(
            "parent reconciliation omissions differ from the exact seven minutes"
        )
    permitted = manifest.get("permitted_utc_interval")
    if not isinstance(permitted, dict) or permitted.get("end_exclusive_utc") != (
        _format_utc(plan.g1_allowed_end_exclusive)
    ):
        raise CanonicalCorrectionValidationError(
            "parent reconciliation has an incompatible holdout cutoff"
        )


def _correction_bars(
    reconciliation: dict[str, Any], cache_root: Path, cutoff: datetime
) -> tuple[
    dict[datetime, tuple[SourceBar, SourceBar]], tuple[dict[str, Any], ...]
]:
    verification = reconciliation.get("independent_tick_verification")
    hours_raw = (
        None if not isinstance(verification, dict) else verification.get("hours")
    )
    if not isinstance(hours_raw, list):
        raise CanonicalCorrectionValidationError("reconciliation lacks BI5 evidence")
    evidence_by_hour = {
        cast(str, item["hour_start_utc"]): item
        for item in hours_raw
        if isinstance(item, dict) and isinstance(item.get("hour_start_utc"), str)
    }
    expected_by_hour: dict[datetime, list[datetime]] = {}
    for minute in _EXPECTED_MINUTES:
        if minute >= cutoff:
            raise CanonicalCorrectionValidationError(
                "correction minute reaches the sealed holdout boundary"
            )
        expected_by_hour.setdefault(minute.replace(minute=0), []).append(minute)

    bars: dict[datetime, tuple[SourceBar, SourceBar]] = {}
    evidence_rows: list[dict[str, Any]] = []
    acquirer = DirectDukascopyAcquirer(cache_root)
    for hour, minutes in sorted(expected_by_hour.items()):
        result = acquirer.acquire(hour, cutoff)
        if result.outcome != "verified" or result.tick_count is None:
            raise CanonicalCorrectionValidationError(
                f"missing or corrupt BI5 evidence for {_format_utc(hour)}"
            )
        reconciled = evidence_by_hour.get(_format_utc(hour))
        if (
            not isinstance(reconciled, dict)
            or reconciled.get("content_sha256") != result.content_sha256
            or reconciled.get("content_size") != result.content_size
            or reconciled.get("tick_count") != result.tick_count
            or reconciled.get("cache_relative_path") != result.cache_relative_path
        ):
            raise CanonicalCorrectionValidationError(
                "BI5 cache evidence differs from the frozen reconciliation"
            )
        if result.cache_relative_path is None:
            raise CanonicalCorrectionValidationError("BI5 cache path is absent")
        payload_path = cache_root / result.cache_relative_path
        payload = payload_path.read_bytes()
        decoded = _decode_ticks(
            hour,
            payload,
            price_format=_probe_price_format(payload),
            price_divisor=100_000.0,
        )
        aggregate: _MinuteAggregate | None = None
        completed: list[_MinuteAggregate] = []
        counts: dict[datetime, int] = {}
        for tick in decoded:
            minute = tick.ts.replace(second=0, microsecond=0)
            counts[minute] = counts.get(minute, 0) + 1
            aggregate, emitted = _advance_minute_aggregate(
                aggregate,
                _datetime_ms(tick.ts),
                _decimal_float(tick.bid),
                _decimal_float(tick.ask),
                _decimal_float(tick.bid_vol),
                _decimal_float(tick.ask_vol),
            )
            if emitted is not None:
                completed.append(emitted)
        if aggregate is not None:
            completed.append(aggregate)
        by_minute = {
            datetime.fromtimestamp(item.minute_ms / 1_000, tz=UTC): item
            for item in completed
        }
        minute_rows: list[dict[str, Any]] = []
        for minute in minutes:
            expected_count = _EXPECTED_TICK_COUNTS[_format_utc(minute)]
            if counts.get(minute) != expected_count or minute not in by_minute:
                raise CanonicalCorrectionValidationError(
                    f"BI5 tick-count mismatch for {_format_utc(minute)}"
                )
            pair = _source_bar_pair(by_minute[minute])
            _validate_source_pair(pair)
            bars[minute] = pair
            minute_rows.append(
                {"timestamp_utc": _format_utc(minute), "tick_count": expected_count}
            )
        evidence_rows.append(
            {
                "evidence_id": reconciled["evidence_id"],
                "hour_start_utc": _format_utc(hour),
                "source_url": result.source_url,
                "cache_relative_path": result.cache_relative_path,
                "content_size": result.content_size,
                "content_sha256": result.content_sha256,
                "hour_tick_count": result.tick_count,
                "corrected_minutes": minute_rows,
            }
        )
    if set(bars) != set(_EXPECTED_MINUTES):
        raise CanonicalCorrectionValidationError(
            "BI5 aggregation did not produce exactly seven correction minutes"
        )
    return bars, tuple(evidence_rows)


def _validate_source_pair(pair: tuple[SourceBar, SourceBar]) -> None:
    bid, ask = pair
    if bid.timestamp != ask.timestamp:
        raise CanonicalCorrectionValidationError("corrected BID/ASK timestamps differ")
    for field in ("open", "high", "low", "close"):
        if getattr(ask, field) < getattr(bid, field):
            raise CanonicalCorrectionValidationError(
                f"corrected ASK {field} is below BID"
            )
    if bid.volume < 0 or ask.volume < 0:
        raise CanonicalCorrectionValidationError("corrected volume is negative")


def _reject_existing_minutes(
    catalog: ParquetDataCatalog,
    minute_bars: Mapping[datetime, tuple[SourceBar, SourceBar]],
) -> None:
    for minute in minute_bars:
        timestamp_ns = _datetime_ns(minute)
        for side in _SIDES:
            existing = tuple(
                bar
                for bar in catalog.query_bars(
                    [_source_bar_type(side)],
                    start=timestamp_ns,
                    end=timestamp_ns + _MINUTE_NS,
                )
                if bar.ts_event == timestamp_ns
            )
            if existing:
                raise CanonicalCorrectionValidationError(
                    f"parent already contains {_format_utc(minute)} {side}"
                )


def _copy_canonical_catalog(source: Path, destination: Path) -> None:
    source_data = source / "data"
    instrument = source_data / "instruments" / "EURUSD.DUKASCOPY"
    if not instrument.is_dir():
        raise CanonicalCorrectionValidationError("parent instrument catalog is missing")
    for side in _SIDES:
        path = source_data / "bars" / _catalog_directory(side)
        if not path.is_dir():
            raise CanonicalCorrectionValidationError(
                f"parent {side} canonical catalog is missing"
            )
    shutil.copytree(
        instrument, destination / "data" / "instruments" / instrument.name
    )


def _populate_corrected_catalog(
    parent: ParquetDataCatalog,
    corrected: ParquetDataCatalog,
    parent_manifest: dict[str, Any],
    expected_bid: Sequence[Bar],
    expected_ask: Sequence[Bar],
) -> None:
    """Stream unchanged parent bars plus corrections into disjoint child files."""

    start_ns, end_ns = _manifest_range_ns(parent_manifest)
    expected = {"BID": expected_bid, "ASK": expected_ask}
    chunk_ns = _SCAN_CHUNK_MINUTES * _MINUTE_NS
    for side in _SIDES:
        additions = {bar.ts_event: bar for bar in expected[side]}
        written_additions: set[int] = set()
        chunk_start = start_ns
        while chunk_start < end_ns:
            chunk_end = min(chunk_start + chunk_ns, end_ns)
            parent_bars = tuple(
                bar
                for bar in parent.query_bars(
                    [_source_bar_type(side)], start=chunk_start, end=chunk_end
                )
                if chunk_start <= bar.ts_event < chunk_end
            )
            chunk_additions = tuple(
                bar
                for timestamp, bar in additions.items()
                if chunk_start <= timestamp < chunk_end
            )
            combined = tuple(
                sorted((*parent_bars, *chunk_additions), key=lambda bar: bar.ts_event)
            )
            if len({bar.ts_event for bar in combined}) != len(combined):
                raise CanonicalCorrectionValidationError(
                    f"corrected {side} write would create a duplicate timestamp"
                )
            if combined:
                corrected.write_bars(combined)
            written_additions.update(bar.ts_event for bar in chunk_additions)
            chunk_start = chunk_end
        if written_additions != set(additions):
            raise CanonicalCorrectionValidationError(
                f"corrected {side} write omitted a required addition"
            )


def _prove_parent_equivalence(
    parent: ParquetDataCatalog,
    corrected: ParquetDataCatalog,
    parent_manifest: dict[str, Any],
    expected_bid: Sequence[Bar],
    expected_ask: Sequence[Bar],
) -> tuple[dict[str, Any], dict[str, str]]:
    start_ns, end_ns = _manifest_range_ns(parent_manifest)
    expected = {
        "BID": {bar.ts_event: bar for bar in expected_bid},
        "ASK": {bar.ts_event: bar for bar in expected_ask},
    }
    extra_counts = {"BID": 0, "ASK": 0}
    unchanged = {"BID": 0, "ASK": 0}
    parent_digests = {side: hashlib.sha256() for side in _SIDES}
    corrected_digests = {side: hashlib.sha256() for side in _SIDES}
    chunk = _SCAN_CHUNK_MINUTES * _MINUTE_NS
    chunk_start = start_ns
    while chunk_start < end_ns:
        chunk_end = min(chunk_start + chunk, end_ns)
        for side in _SIDES:
            parent_bars = tuple(
                bar
                for bar in parent.query_bars(
                    [_source_bar_type(side)], start=chunk_start, end=chunk_end
                )
                if chunk_start <= bar.ts_event < chunk_end
            )
            corrected_bars = tuple(
                bar
                for bar in corrected.query_bars(
                    [_source_bar_type(side)], start=chunk_start, end=chunk_end
                )
                if chunk_start <= bar.ts_event < chunk_end
            )
            corrected_by_time = {bar.ts_event: bar for bar in corrected_bars}
            if len(corrected_by_time) != len(corrected_bars):
                raise CanonicalCorrectionValidationError(
                    f"corrected {side} series contains duplicate timestamps"
                )
            parent_times = {bar.ts_event for bar in parent_bars}
            for bar in parent_bars:
                if corrected_by_time.get(bar.ts_event) != bar:
                    raise CanonicalCorrectionValidationError(
                        f"parent {side} bar changed or disappeared"
                    )
                unchanged[side] += 1
                _update_bar_digest(parent_digests[side], bar)
            for bar in corrected_bars:
                _update_bar_digest(corrected_digests[side], bar)
                if bar.ts_event not in parent_times:
                    if expected[side].get(bar.ts_event) != bar:
                        raise CanonicalCorrectionValidationError(
                            f"unexpected corrected {side} bar appeared"
                        )
                    extra_counts[side] += 1
        chunk_start = chunk_end
    parent_counts = cast(dict[str, Any], parent_manifest["qa"])
    if any(
        unchanged[side] != parent_counts[f"{side.lower()}_bar_count"]
        or extra_counts[side] != 7
        for side in _SIDES
    ):
        raise CanonicalCorrectionValidationError(
            "parent/correction bar counts do not prove exact equivalence"
        )
    equivalence = {
        "parent_bid_bars_unchanged": True,
        "parent_ask_bars_unchanged": True,
        "parent_timestamps_removed": 0,
        "new_bid_timestamp_count": 7,
        "new_ask_timestamp_count": 7,
        "other_changed_or_added_bar_count": 0,
        "parent_counts": {side: unchanged[side] for side in _SIDES},
        "parent_content_sha256": {
            side: parent_digests[side].hexdigest() for side in _SIDES
        },
    }
    content = {
        side: corrected_digests[side].hexdigest() for side in _SIDES
    }
    return equivalence, content


def _manifest_range_ns(manifest: dict[str, Any]) -> tuple[int, int]:
    requested = cast(dict[str, Any], manifest["requested_utc_range"])
    start = datetime.fromisoformat(cast(str, requested["start_date"])).replace(
        tzinfo=UTC
    )
    end = (
        datetime.fromisoformat(cast(str, requested["end_date_inclusive"])).replace(
            tzinfo=UTC
        )
        + timedelta(days=1)
    )
    if end <= start:
        raise CanonicalCorrectionValidationError(
            "canonical requested range is empty"
        )
    return _datetime_ns(start), _datetime_ns(end)


def _corrected_manifest_payload(
    *,
    parent_manifest: dict[str, Any],
    parent_manifest_sha256: str,
    plan_file_sha256: str,
    reconciliation_file_sha256: str,
    reconciliation_semantic_sha256: str,
    evidence: Sequence[dict[str, Any]],
    equivalence: dict[str, Any],
    content: dict[str, str],
) -> dict[str, Any]:
    payload = deepcopy(parent_manifest)
    payload.pop("semantic_sha256", None)
    parent_qa = cast(dict[str, Any], parent_manifest["qa"])
    missing = _subtract_corrected_minutes(
        cast(list[object], parent_qa["missing_intervals_utc"])
    )
    corrected_minutes = [
        {"timestamp_utc": value, "tick_count": count}
        for value, count in _EXPECTED_TICK_COUNTS.items()
    ]
    correction_payload = {
        "identity": CORRECTION_ID,
        "research_plan_file_sha256": plan_file_sha256,
        "research_plan_semantic_sha256": parent_manifest["dataset_plan_sha256"],
        "reconciliation_file_sha256": reconciliation_file_sha256,
        "reconciliation_semantic_sha256": reconciliation_semantic_sha256,
        "corrected_minutes": corrected_minutes,
        "direct_dukascopy_bi5": list(evidence),
        "aggregation_implementation": CORRECTION_IMPLEMENTATION,
        "aggregation_version": HF_INGESTION_VERSION,
        "fills_or_interpolation": False,
        "synthetic_minutes_added": 0,
        "holdout_accessed": False,
        "holdout_rows_admitted": 0,
    }
    correction = {
        **correction_payload,
        "semantic_sha256": _semantic_sha256(correction_payload),
    }
    payload.update(
        {
            "dataset_identity": CORRECTION_ID,
            "distribution_source": CORRECTED_DISTRIBUTION_SOURCE,
            "ingestion_version": CORRECTED_INGESTION_VERSION,
            "parent_canonical": {
                "filename": PARENT_MANIFEST_FILENAME,
                "file_sha256": parent_manifest_sha256,
                "semantic_sha256": parent_manifest["semantic_sha256"],
                "ingestion_version": parent_manifest["ingestion_version"],
                "dataset_id": parent_manifest["dataset_id"],
            },
            "correction": correction,
            "parent_equivalence": equivalence,
            "canonical_content": content,
            "holdout_accessed": False,
            "qa": {
                **parent_qa,
                "admitted_tick_count": parent_qa["admitted_tick_count"]
                + sum(_EXPECTED_TICK_COUNTS.values()),
                "bid_bar_count": parent_qa["bid_bar_count"] + 7,
                "ask_bar_count": parent_qa["ask_bar_count"] + 7,
                "missing_interval_count": parent_qa["missing_interval_count"] - 7,
                "missing_intervals_utc": missing,
                "holdout_rows_admitted": 0,
                "gaps_filled": False,
            },
            "fills_or_interpolation": False,
            "semantic_hash_contract": (
                "SHA-256 of canonical JSON for every other manifest field"
            ),
        }
    )
    return payload


def _subtract_corrected_minutes(intervals: Sequence[object]) -> list[dict[str, Any]]:
    excluded = set(_EXPECTED_MINUTES)
    result: list[dict[str, Any]] = []
    for raw in intervals:
        if not isinstance(raw, dict):
            raise CanonicalCorrectionValidationError(
                "parent missing interval is invalid"
            )
        start = _parse_minute(raw.get("start_utc"))
        end = _parse_minute(raw.get("end_utc"))
        cursor = start
        active: datetime | None = None
        while cursor <= end:
            if cursor not in excluded and active is None:
                active = cursor
            if cursor in excluded and active is not None:
                result.append(_missing_interval(active, cursor - _MINUTE))
                active = None
            cursor += _MINUTE
        if active is not None:
            result.append(_missing_interval(active, end))
    if sum(cast(int, item["minutes"]) for item in result) != (
        sum(cast(int, cast(dict[str, Any], item)["minutes"]) for item in intervals) - 7
    ):
        raise CanonicalCorrectionValidationError(
            "corrected missing-interval provenance does not remove exactly seven"
        )
    return result


def _missing_interval(start: datetime, end: datetime) -> dict[str, Any]:
    return {
        "start_utc": _format_utc(start),
        "end_utc": _format_utc(end),
        "minutes": int((end - start) / _MINUTE) + 1,
    }


def _load_semantic_json(path: Path, description: str) -> dict[str, Any]:
    value = _load_json(path, description)
    semantic = value.get("semantic_sha256")
    payload = {key: item for key, item in value.items() if key != "semantic_sha256"}
    if semantic != _semantic_sha256(payload):
        raise CanonicalCorrectionValidationError(
            f"{description} semantic SHA-256 does not match"
        )
    return value


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CanonicalCorrectionValidationError(
            f"invalid {description}: {error}"
        ) from error
    if not isinstance(raw, dict):
        raise CanonicalCorrectionValidationError(f"{description} must be an object")
    return cast(dict[str, Any], raw)


def _parse_minute(value: object) -> datetime:
    if not isinstance(value, str):
        raise CanonicalCorrectionValidationError("minute must be an ISO UTC string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise CanonicalCorrectionValidationError(
            "minute is not valid ISO UTC"
        ) from error
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() != timedelta(0)
        or parsed.second != 0
        or parsed.microsecond != 0
        or not value.endswith("Z")
    ):
        raise CanonicalCorrectionValidationError("minute is not exact UTC")
    return parsed.astimezone(UTC)


def _source_bar_type(side: str) -> str:
    return f"{INSTRUMENT_ID}-1-MINUTE-{side}-EXTERNAL"


def _catalog_directory(side: str) -> str:
    return f"EURUSD.DUKASCOPY-1-MINUTE-{side}-EXTERNAL"


def _datetime_ns(value: datetime) -> int:
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    delta = value - epoch
    return (delta.days * 86_400 + delta.seconds) * 1_000_000_000


def _semantic_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _format_utc(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or freeze the corrected EUR/USD G1 canonical dataset"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--plan", required=True, type=Path)
    build.add_argument("--parent-root", required=True, type=Path)
    build.add_argument("--output-root", required=True, type=Path)
    build.add_argument("--cache-root", required=True, type=Path)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--plan", required=True, type=Path)
    freeze.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the immutable correction or final readiness freeze."""

    args = _build_parser().parse_args(argv)
    if args.command == "build":
        correction_result = build_corrected_eurusd_dataset(
            cast(Path, args.plan),
            cast(Path, args.parent_root),
            cast(Path, args.output_root),
            cast(Path, args.cache_root),
        )
        print(correction_result.manifest_path)
        return 0
    readiness_result = freeze_eurusd_research_readiness(
        cast(Path, args.plan), cast(Path, args.output_root)
    )
    print(readiness_result.manifest_path)
    print(f"research_ready: {str(readiness_result.research_ready).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
