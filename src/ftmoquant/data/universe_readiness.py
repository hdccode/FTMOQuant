"""Immutable correction, split, and readiness sealing for G1.4 universes."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from nautilus_trader.persistence import ParquetDataCatalog

from ftmoquant.data.canonical_source import (
    HF_INGESTION_VERSION,
    iter_paired_source_chunks,
    validate_canonical_source_manifest,
)
from ftmoquant.data.derived_bars import (
    DERIVED_MANIFEST_FILENAME,
    PARENT_MANIFEST_FILENAME,
)
from ftmoquant.data.dukascopy import SourceBar
from ftmoquant.data.instruments import InstrumentSpec, to_nautilus_bars
from ftmoquant.data.session_coverage import COVERAGE_MANIFEST_FILENAME
from ftmoquant.data.session_reconciliation import RECONCILIATION_MANIFEST_FILENAME
from ftmoquant.data.universe_plan import (
    load_research_universe_plan,
)

GENERIC_CORRECTION_VERSION = "g1.4a-dukascopy-corrected-1"
INSTRUMENT_READINESS_VERSION = "g1.4a-instrument-readiness-1"
UNIVERSE_READINESS_VERSION = "g1.4a-universe-readiness-1"
SPLIT_VIEW_VERSION = "g1.4a-split-view-1"
INSTRUMENT_READINESS_FILENAME = "ftmoquant_instrument_readiness.json"
UNIVERSE_READINESS_FILENAME = "ftmoquant_universe_readiness.json"
SPLIT_VIEW_FILENAME = "ftmoquant_split_view.json"

_MINUTE = timedelta(minutes=1)
_MINUTE_NS = 60_000_000_000
_SCAN_MINUTES = 10_000
_SHA256 = re.compile(r"[0-9a-f]{64}")


class UniverseReadinessValidationError(ValueError):
    """Raised when any lineage, split, or universe gate fails closed."""


@dataclass(frozen=True, slots=True)
class InstrumentArtifactRef:
    """Path-independent identity for one research-ready instrument."""

    instrument_id: str
    canonical_sha256: str
    derived_sha256: str
    coverage_sha256: str
    reconciliation_sha256: str
    correction_sha256: str | None
    readiness_sha256: str
    catalog_tree_sha256: str
    development_split_sha256: str
    validation_split_sha256: str


@dataclass(frozen=True, slots=True)
class UniverseReadinessManifest:
    """All-or-nothing universe state returned after an immutable freeze."""

    manifest_path: Path
    semantic_sha256: str
    ordered_artifacts: tuple[InstrumentArtifactRef, ...]
    exact_currency_set: tuple[str, ...]
    permitted_start_utc: str
    permitted_end_exclusive_utc: str
    per_instrument_status: tuple[tuple[str, str], ...]
    research_ready: bool


@dataclass(frozen=True, slots=True)
class SplitViewResult:
    split: str
    manifest_path: Path
    semantic_sha256: str
    catalog_tree_sha256: str


def build_corrected_instrument_dataset(
    instrument_spec: InstrumentSpec,
    parent_root: Path,
    output_root: Path,
    corrections: Mapping[datetime, tuple[SourceBar, SourceBar]],
) -> Path:
    """Create an immutable canonical child for the exact reconciliation gap set."""

    instrument_spec.validate()
    parent = parent_root.resolve()
    output = output_root.resolve()
    if output.exists() or parent == output or parent in output.parents:
        raise UniverseReadinessValidationError(
            "correction output must be a new disjoint immutable root"
        )
    parent_manifest_path = parent / PARENT_MANIFEST_FILENAME
    reconciliation_path = parent / RECONCILIATION_MANIFEST_FILENAME
    parent_manifest = _semantic_json(parent_manifest_path, "parent canonical")
    identity = validate_canonical_source_manifest(parent_manifest, instrument_spec)
    if identity.ingestion_version != HF_INGESTION_VERSION:
        raise UniverseReadinessValidationError(
            "generic correction requires the pinned HF canonical parent"
        )
    reconciliation = _semantic_json(reconciliation_path, "reconciliation")
    if reconciliation.get("instrument_id") != instrument_spec.instrument_id:
        raise UniverseReadinessValidationError("reconciliation instrument mismatch")
    expected = _proven_correction_minutes(reconciliation)
    if set(corrections) != expected or not expected:
        raise UniverseReadinessValidationError(
            "corrections must equal the exact reconciliation-proven missing set"
        )
    cutoff = _utc(reconciliation["permitted_utc_interval"]["end_exclusive_utc"])
    if any(timestamp >= cutoff for timestamp in corrections):
        raise UniverseReadinessValidationError("correction reaches the holdout cutoff")

    parent_catalog = ParquetDataCatalog(str(parent / "catalog"))
    output.mkdir(parents=True)
    output_catalog_path = output / "catalog"
    output_catalog_path.mkdir()
    destination = ParquetDataCatalog(str(output_catalog_path))
    instruments = parent_catalog.instruments([instrument_spec.instrument_id])
    if len(instruments) != 1:
        raise UniverseReadinessValidationError("parent catalog instrument is missing")
    destination.write_instruments(instruments)
    start_ns, end_ns = _manifest_range_ns(parent_manifest)
    correction_ns = {int(item.timestamp() * 1_000_000_000) for item in expected}
    copied = 0
    for chunk in iter_paired_source_chunks(
        parent_catalog,
        parent_manifest,
        instrument_spec,
        start_ns,
        end_ns,
        chunk_minutes=_SCAN_MINUTES,
    ):
        if any(bar.ts_event in correction_ns for bar in chunk.bid_bars):
            raise UniverseReadinessValidationError(
                "parent already contains a correction"
            )
        if chunk.bid_bars:
            destination.write_bars(chunk.bid_bars)
            destination.write_bars(chunk.ask_bars)
            copied += len(chunk.bid_bars)
    ordered = sorted(corrections.items())
    bid_rows = [pair[0] for _, pair in ordered]
    ask_rows = [pair[1] for _, pair in ordered]
    _validate_correction_pairs(ordered, instrument_spec)
    destination.write_bars(to_nautilus_bars(bid_rows, "BID", instrument_spec))
    destination.write_bars(to_nautilus_bars(ask_rows, "ASK", instrument_spec))

    correction_payload = {
        "identity": GENERIC_CORRECTION_VERSION,
        "reconciliation_file_sha256": _sha256(reconciliation_path),
        "reconciliation_semantic_sha256": reconciliation["semantic_sha256"],
        "corrected_minutes_utc": [_format_utc(item) for item in sorted(expected)],
        "fills_or_interpolation": False,
        "holdout_accessed": False,
        "holdout_rows_admitted": 0,
    }
    correction = {
        **correction_payload,
        "semantic_sha256": _semantic_sha256(correction_payload),
    }
    payload = {
        **{
            key: value
            for key, value in parent_manifest.items()
            if key != "semantic_sha256"
        },
        "distribution_source": "Hugging Face + direct Dukascopy BI5 correction",
        "ingestion_version": GENERIC_CORRECTION_VERSION,
        "parent_canonical": {
            "file_sha256": _sha256(parent_manifest_path),
            "semantic_sha256": parent_manifest["semantic_sha256"],
            "ingestion_version": identity.ingestion_version,
        },
        "correction": correction,
        "parent_equivalence": {
            "parent_bid_bars_unchanged": True,
            "parent_ask_bars_unchanged": True,
            "parent_timestamps_removed": 0,
            "new_bid_timestamp_count": len(expected),
            "new_ask_timestamp_count": len(expected),
            "other_changed_or_added_bar_count": 0,
        },
        "qa": {
            **cast(dict[str, Any], parent_manifest["qa"]),
            "bid_bar_count": copied + len(expected),
            "ask_bar_count": copied + len(expected),
            "holdout_rows_admitted": 0,
            "gaps_filled": False,
        },
    }
    semantic = _semantic_sha256(payload)
    path = output / PARENT_MANIFEST_FILENAME
    path.write_text(
        json.dumps({**payload, "semantic_sha256": semantic}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return path


def materialize_instrument_split_views(
    plan_path: Path,
    instrument_id: str,
    canonical_root: Path,
    output_root: Path,
) -> tuple[SplitViewResult, SplitViewResult]:
    """Write immutable, physically separate development and validation catalogs."""

    plan = load_research_universe_plan(plan_path)
    instrument = plan.instrument(instrument_id)
    root = canonical_root.resolve()
    destination_root = output_root.resolve()
    if destination_root.exists():
        raise UniverseReadinessValidationError("split output root already exists")
    manifest = _semantic_json(root / PARENT_MANIFEST_FILENAME, "canonical")
    validate_canonical_source_manifest(manifest, instrument)
    start_ns, end_ns = _manifest_range_ns(manifest)
    if start_ns != _ns(plan.permitted_start_utc) or end_ns != _ns(
        plan.permitted_end_exclusive_utc
    ):
        raise UniverseReadinessValidationError(
            "canonical range does not match universe"
        )
    source = ParquetDataCatalog(str(root / "catalog"))
    results = []
    ranges = (
        (
            "development",
            plan.permitted_start_utc,
            datetime.combine(
                plan.development.end + timedelta(days=1),
                datetime.min.time(),
                tzinfo=UTC,
            ),
        ),
        (
            "validation",
            datetime.combine(plan.validation.start, datetime.min.time(), tzinfo=UTC),
            plan.permitted_end_exclusive_utc,
        ),
    )
    for name, split_start, split_end in ranges:
        split_root = destination_root / name
        catalog_path = split_root / "catalog"
        catalog_path.mkdir(parents=True)
        destination = ParquetDataCatalog(str(catalog_path))
        native = source.instruments([instrument.instrument_id])
        if len(native) != 1:
            raise UniverseReadinessValidationError("canonical instrument is missing")
        destination.write_instruments(native)
        counts: dict[str, int] = {}
        for minutes, aggregation in (
            (1, "EXTERNAL"),
            (60, "INTERNAL"),
            (240, "INTERNAL"),
        ):
            for side in ("BID", "ASK"):
                bar_type = instrument.bar_type(
                    minutes=minutes, side=side, aggregation=aggregation
                )
                count = _copy_bar_range(
                    source, destination, bar_type, _ns(split_start), _ns(split_end)
                )
                if count <= 0:
                    raise UniverseReadinessValidationError(
                        f"split {name} is missing required {bar_type}"
                    )
                counts[bar_type] = count
        tree_sha = _sha256_tree(catalog_path)
        payload = {
            "split_view_version": SPLIT_VIEW_VERSION,
            "universe_plan_sha256": plan.semantic_sha256,
            "instrument_id": instrument.instrument_id,
            "split": name,
            "range": {
                "start_utc": _format_utc(split_start),
                "end_exclusive_utc": _format_utc(split_end),
            },
            "canonical_parent": {
                "file_sha256": _sha256(root / PARENT_MANIFEST_FILENAME),
                "semantic_sha256": manifest["semantic_sha256"],
            },
            "catalog_tree_sha256": tree_sha,
            "bar_counts": counts,
            "holdout_rows": 0,
            "network_access_required": False,
            "access_policy": (
                "candidate_read_only"
                if name == "development"
                else "encrypted_normally_unmounted_validation_runner_read_only"
            ),
        }
        semantic = _semantic_sha256(payload)
        manifest_path = split_root / SPLIT_VIEW_FILENAME
        manifest_path.write_text(
            json.dumps(
                {**payload, "semantic_sha256": semantic}, indent=2, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )
        results.append(SplitViewResult(name, manifest_path, semantic, tree_sha))
    return cast(tuple[SplitViewResult, SplitViewResult], tuple(results))


def freeze_instrument_readiness(
    plan_path: Path,
    instrument_id: str,
    artifact_root: Path,
    split_root: Path,
) -> InstrumentArtifactRef:
    """Seal one instrument only after every data and holdout gate passes."""

    plan = load_research_universe_plan(plan_path)
    instrument = plan.instrument(instrument_id)
    root = artifact_root.resolve()
    documents = {
        "canonical": _semantic_json(root / PARENT_MANIFEST_FILENAME, "canonical"),
        "derived": _json(root / DERIVED_MANIFEST_FILENAME, "derived"),
        "coverage": _semantic_json(root / COVERAGE_MANIFEST_FILENAME, "coverage"),
        "reconciliation": _semantic_json(
            root / RECONCILIATION_MANIFEST_FILENAME, "reconciliation"
        ),
    }
    validate_canonical_source_manifest(documents["canonical"], instrument)
    expected_range = {
        "start_utc": _format_utc(plan.permitted_start_utc),
        "end_exclusive_utc": _format_utc(plan.permitted_end_exclusive_utc),
    }
    canonical_start, canonical_end = _manifest_range_ns(documents["canonical"])
    split_docs = {
        name: _semantic_json(split_root / name / SPLIT_VIEW_FILENAME, f"{name} split")
        for name in ("development", "validation")
    }
    expected_split_ranges = {
        "development": {
            "start_utc": _format_utc(plan.permitted_start_utc),
            "end_exclusive_utc": _format_utc(
                datetime.combine(
                    plan.development.end + timedelta(days=1),
                    datetime.min.time(),
                    tzinfo=UTC,
                )
            ),
        },
        "validation": {
            "start_utc": _format_utc(
                datetime.combine(plan.validation.start, datetime.min.time(), tzinfo=UTC)
            ),
            "end_exclusive_utc": _format_utc(plan.permitted_end_exclusive_utc),
        },
    }
    hashes = {
        name: _sha256(root / filename)
        for name, filename in (
            ("canonical", PARENT_MANIFEST_FILENAME),
            ("derived", DERIVED_MANIFEST_FILENAME),
            ("coverage", COVERAGE_MANIFEST_FILENAME),
            ("reconciliation", RECONCILIATION_MANIFEST_FILENAME),
        )
    }
    reconciliation_counts = documents["reconciliation"].get("counts")
    gates = {
        "exact_instrument": all(
            item.get("instrument_id") == instrument.instrument_id
            for item in documents.values()
        ),
        "exact_range": (
            canonical_start == _ns(plan.permitted_start_utc)
            and canonical_end == _ns(plan.permitted_end_exclusive_utc)
            and documents["coverage"].get("requested_utc_interval") == expected_range
            and documents["reconciliation"].get("permitted_utc_interval")
            == expected_range
        ),
        "derived_parent_bound": documents["derived"].get("parent_g0_5_manifest_sha256")
        == hashes["canonical"],
        "coverage_bound": (
            documents["coverage"].get("parent_g0_5_manifest_sha256")
            == hashes["canonical"]
            and documents["coverage"].get("structural_g0_6_manifest_sha256")
            == hashes["derived"]
        ),
        "reconciliation_ready": (
            documents["reconciliation"].get("session_aware_research_ready") is True
            and isinstance(reconciliation_counts, dict)
            and reconciliation_counts.get("unexplained_missing_minute_count") == 0
        ),
        "structural_ready": (
            documents["derived"].get("derived_bar_integrity_valid") is True
            and documents["coverage"].get("source_derived_structural_integrity_valid")
            is True
        ),
        "zero_holdout": (
            cast(dict[str, Any], documents["canonical"].get("qa", {})).get(
                "holdout_rows_admitted"
            )
            == 0
            and documents["reconciliation"].get("holdout_rows_admitted") == 0
            and documents["reconciliation"].get("holdout_accessed") is False
        ),
        "split_views_exact": all(
            item.get("instrument_id") == instrument.instrument_id
            and item.get("holdout_rows") == 0
            and item.get("universe_plan_sha256") == plan.semantic_sha256
            and item.get("range") == expected_split_ranges[name]
            and item.get("catalog_tree_sha256")
            == _sha256_tree(split_root / name / "catalog")
            for name, item in split_docs.items()
        ),
    }
    if not all(gates.values()):
        failed = ", ".join(name for name, value in gates.items() if not value)
        raise UniverseReadinessValidationError(f"instrument readiness failed: {failed}")
    artifact_entries = {
        name: {
            "file_sha256": hashes[name],
            "semantic_sha256": documents[name].get("semantic_sha256"),
        }
        for name in hashes
    }
    correction = documents["canonical"].get("correction")
    if isinstance(correction, dict):
        artifact_entries["correction"] = {
            "file_sha256": correction.get("semantic_sha256"),
            "semantic_sha256": correction.get("semantic_sha256"),
        }
    payload = {
        "readiness_version": INSTRUMENT_READINESS_VERSION,
        "universe_plan_sha256": plan.semantic_sha256,
        "instrument_id": instrument.instrument_id,
        "exact_currency_set": sorted(
            {instrument.base_currency, instrument.quote_currency}
        ),
        "permitted_utc_interval": expected_range,
        "artifacts": artifact_entries,
        "catalog_tree_sha256": _sha256_tree(root / "catalog"),
        "split_views": {
            name: {
                "semantic_sha256": split_docs[name]["semantic_sha256"],
                "catalog_tree_sha256": split_docs[name]["catalog_tree_sha256"],
            }
            for name in split_docs
        },
        "gates": gates,
        "research_ready": True,
        "holdout_accessed": False,
        "holdout_rows_admitted": 0,
        "strategy_return_accessed": False,
    }
    semantic = _semantic_sha256(payload)
    path = root / INSTRUMENT_READINESS_FILENAME
    encoded = (
        json.dumps({**payload, "semantic_sha256": semantic}, indent=2, sort_keys=True)
        + "\n"
    ).encode()
    if path.exists() and path.read_bytes() != encoded:
        raise UniverseReadinessValidationError("instrument readiness conflicts")
    if not path.exists():
        path.write_bytes(encoded)
    return _artifact_ref(root, split_docs, hashes, semantic, documents)


def freeze_universe_readiness(
    plan_path: Path,
    instrument_roots: Mapping[str, Path],
    output_root: Path,
) -> UniverseReadinessManifest:
    """Freeze the exact ordered universe; one bad/missing/extra child fails all."""

    plan = load_research_universe_plan(plan_path)
    expected_ids = tuple(item.instrument_id for item in plan.instruments)
    if set(instrument_roots) != set(expected_ids) or len(instrument_roots) != len(
        expected_ids
    ):
        raise UniverseReadinessValidationError(
            "universe roots contain missing, extra, or duplicate instruments"
        )
    refs: list[InstrumentArtifactRef] = []
    statuses: list[tuple[str, str]] = []
    for instrument_id in expected_ids:
        root = instrument_roots[instrument_id].resolve()
        document = _semantic_json(
            root / INSTRUMENT_READINESS_FILENAME, "instrument readiness"
        )
        if (
            document.get("instrument_id") != instrument_id
            or document.get("universe_plan_sha256") != plan.semantic_sha256
            or document.get("research_ready") is not True
            or document.get("permitted_utc_interval")
            != {
                "start_utc": _format_utc(plan.permitted_start_utc),
                "end_exclusive_utc": _format_utc(plan.permitted_end_exclusive_utc),
            }
        ):
            raise UniverseReadinessValidationError(
                f"instrument readiness is false or incompatible: {instrument_id}"
            )
        reference = _artifact_ref_from_readiness(document)
        for field, value in asdict(reference).items():
            if field == "instrument_id" or value is None:
                continue
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise UniverseReadinessValidationError(
                    f"instrument readiness has invalid {field}: {instrument_id}"
                )
        refs.append(reference)
        statuses.append((instrument_id, "research_ready"))
    payload = {
        "readiness_version": UNIVERSE_READINESS_VERSION,
        "universe_id": plan.universe_id,
        "universe_plan_sha256": plan.semantic_sha256,
        "ordered_instrument_ids": list(expected_ids),
        "exact_currency_set": list(plan.exact_currency_set),
        "permitted_utc_interval": {
            "start_utc": _format_utc(plan.permitted_start_utc),
            "end_exclusive_utc": _format_utc(plan.permitted_end_exclusive_utc),
        },
        "instrument_artifacts": [asdict(item) for item in refs],
        "per_instrument_status": dict(statuses),
        "cross_sectional_factor_status": "not_eligible_insufficient_universe",
        "minimum_cross_sectional_instruments": 4,
        "research_ready": True,
        "holdout_accessed": False,
        "holdout_rows_admitted": 0,
        "strategy_return_accessed": False,
    }
    semantic = _semantic_sha256(payload)
    output = output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / UNIVERSE_READINESS_FILENAME
    encoded = (
        json.dumps({**payload, "semantic_sha256": semantic}, indent=2, sort_keys=True)
        + "\n"
    ).encode()
    if path.exists() and path.read_bytes() != encoded:
        raise UniverseReadinessValidationError("universe readiness conflicts")
    if not path.exists():
        path.write_bytes(encoded)
    return UniverseReadinessManifest(
        path,
        semantic,
        tuple(refs),
        plan.exact_currency_set,
        _format_utc(plan.permitted_start_utc),
        _format_utc(plan.permitted_end_exclusive_utc),
        tuple(statuses),
        True,
    )


def _artifact_ref(
    root: Path,
    split_docs: Mapping[str, dict[str, Any]],
    hashes: Mapping[str, str],
    readiness_sha: str,
    documents: Mapping[str, dict[str, Any]],
) -> InstrumentArtifactRef:
    correction = documents["canonical"].get("correction")
    return InstrumentArtifactRef(
        instrument_id=cast(str, documents["canonical"]["instrument_id"]),
        canonical_sha256=hashes["canonical"],
        derived_sha256=hashes["derived"],
        coverage_sha256=hashes["coverage"],
        reconciliation_sha256=hashes["reconciliation"],
        correction_sha256=(
            None
            if not isinstance(correction, dict)
            else cast(str, correction.get("semantic_sha256"))
        ),
        readiness_sha256=readiness_sha,
        catalog_tree_sha256=_sha256_tree(root / "catalog"),
        development_split_sha256=cast(
            str, split_docs["development"]["semantic_sha256"]
        ),
        validation_split_sha256=cast(str, split_docs["validation"]["semantic_sha256"]),
    )


def _artifact_ref_from_readiness(document: dict[str, Any]) -> InstrumentArtifactRef:
    artifacts = cast(dict[str, dict[str, Any]], document["artifacts"])
    splits = cast(dict[str, dict[str, Any]], document["split_views"])
    correction = artifacts.get("correction")
    return InstrumentArtifactRef(
        instrument_id=cast(str, document["instrument_id"]),
        canonical_sha256=artifacts["canonical"]["file_sha256"],
        derived_sha256=artifacts["derived"]["file_sha256"],
        coverage_sha256=artifacts["coverage"]["file_sha256"],
        reconciliation_sha256=artifacts["reconciliation"]["file_sha256"],
        correction_sha256=None if correction is None else correction["file_sha256"],
        readiness_sha256=cast(str, document["semantic_sha256"]),
        catalog_tree_sha256=cast(str, document["catalog_tree_sha256"]),
        development_split_sha256=splits["development"]["semantic_sha256"],
        validation_split_sha256=splits["validation"]["semantic_sha256"],
    )


def _copy_bar_range(
    source: ParquetDataCatalog,
    destination: ParquetDataCatalog,
    bar_type: str,
    start_ns: int,
    end_ns: int,
) -> int:
    count = 0
    chunk_ns = _SCAN_MINUTES * _MINUTE_NS
    cursor = start_ns
    while cursor < end_ns:
        stop = min(cursor + chunk_ns, end_ns)
        bars = tuple(
            bar
            for bar in source.query_bars([bar_type], start=cursor, end=stop)
            if cursor <= bar.ts_event < stop and bar.ts_init <= end_ns
        )
        if any(str(bar.bar_type) != bar_type for bar in bars):
            raise UniverseReadinessValidationError("catalog returned a wrong bar type")
        if bars:
            destination.write_bars(bars)
            count += len(bars)
        cursor = stop
    first = destination.query_first_timestamp("bars", bar_type)
    last = destination.query_last_timestamp("bars", bar_type)
    if first is None or last is None or first < start_ns or last > end_ns:
        raise UniverseReadinessValidationError("split catalog range proof failed")
    return count


def _proven_correction_minutes(reconciliation: dict[str, Any]) -> set[datetime]:
    raw = reconciliation.get("unexplained_missing_intervals_utc")
    if not isinstance(raw, list):
        raise UniverseReadinessValidationError("reconciliation lacks gap intervals")
    result: set[datetime] = set()
    for value in raw:
        if not isinstance(value, dict):
            raise UniverseReadinessValidationError("invalid reconciliation interval")
        if value.get("reason") != "independent_direct_source_contains_tick":
            continue
        if value.get("missing_sides") != ["BID", "ASK"] or not value.get(
            "evidence_refs"
        ):
            raise UniverseReadinessValidationError(
                "correction interval lacks paired direct-source evidence"
            )
        start = _utc(value.get("start_utc"))
        end = _utc(value.get("end_utc"))
        current = start
        while current <= end:
            if current in result:
                raise UniverseReadinessValidationError(
                    "overlapping reconciliation gaps"
                )
            result.add(current)
            current += _MINUTE
    return result


def _validate_correction_pairs(
    items: Sequence[tuple[datetime, tuple[SourceBar, SourceBar]]],
    instrument: InstrumentSpec,
) -> None:
    previous: datetime | None = None
    for timestamp, (bid, ask) in items:
        if timestamp != bid.timestamp or timestamp != ask.timestamp:
            raise UniverseReadinessValidationError("correction timestamps differ")
        if previous is not None and timestamp <= previous:
            raise UniverseReadinessValidationError("corrections are not ordered")
        if any(
            getattr(ask, field) < getattr(bid, field)
            for field in ("open", "high", "low", "close")
        ):
            raise UniverseReadinessValidationError("correction ASK is below BID")
        previous = timestamp
    # Encoding is part of validation and rejects incompatible precision.
    to_nautilus_bars([pair[0] for _, pair in items], "BID", instrument)
    to_nautilus_bars([pair[1] for _, pair in items], "ASK", instrument)


def _manifest_range_ns(manifest: dict[str, Any]) -> tuple[int, int]:
    requested = manifest.get("requested_utc_range")
    if not isinstance(requested, dict):
        raise UniverseReadinessValidationError("canonical range is missing")
    try:
        start = datetime.fromisoformat(cast(str, requested["start_date"]))
        end = datetime.fromisoformat(
            cast(str, requested["end_date_inclusive"])
        ) + timedelta(days=1)
    except (KeyError, TypeError, ValueError) as error:
        raise UniverseReadinessValidationError("canonical range is invalid") from error
    return _ns(start.replace(tzinfo=UTC)), _ns(end.replace(tzinfo=UTC))


def _semantic_json(path: Path, description: str) -> dict[str, Any]:
    document = _json(path, description)
    semantic = document.get("semantic_sha256")
    payload = {
        key: value for key, value in document.items() if key != "semantic_sha256"
    }
    if not isinstance(semantic, str) or semantic != _semantic_sha256(payload):
        raise UniverseReadinessValidationError(f"{description} semantic hash mismatch")
    return document


def _json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UniverseReadinessValidationError(
            f"invalid {description}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise UniverseReadinessValidationError(f"{description} must be an object")
    return cast(dict[str, Any], value)


def _utc(value: object) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise UniverseReadinessValidationError("timestamp must be UTC")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise UniverseReadinessValidationError("timestamp must be UTC") from error
    if parsed.utcoffset() != timedelta(0) or parsed.second or parsed.microsecond:
        raise UniverseReadinessValidationError("timestamp must be minute-aligned UTC")
    return parsed.astimezone(UTC)


def _ns(value: datetime) -> int:
    return int(value.timestamp() * 1_000_000_000)


def _format_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


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
        digest.update(_sha256(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()
