"""NEW, separate OANDA alpha-lab DEVELOPMENT lineage.

Acquisition (from ``oanda_development.py``'s already-generic low-level
helpers), canonicalization into the existing Nautilus catalog shape, derived
M30/H1/H4 bars (via the existing generic ``derive_instrument_bars``), and a
small alpha-lab-specific readiness freeze -- entirely independent of the
frozen Carver OANDA acquisition path and the rigorous Dukascopy DEVELOPMENT
lineage. Neither is read, modified, or replaced by anything in this module.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import yaml
from nautilus_trader.persistence import ParquetDataCatalog

from ftmoquant.backtest.execution_harness import _sha256_tree
from ftmoquant.data.canonical_source import (
    OANDA_ALPHA_LAB_DISTRIBUTION_SOURCE,
    OANDA_ALPHA_LAB_INGESTION_VERSION,
)
from ftmoquant.data.derived_bars import (
    DERIVED_MANIFEST_FILENAME,
    PARENT_MANIFEST_FILENAME,
)
from ftmoquant.data.dukascopy import NAUTILUS_VERSION, SourceBar
from ftmoquant.data.instruments import (
    OANDA_ALPHA_LAB_SPECS,
    InstrumentSpec,
    oanda_symbol,
    to_nautilus_bars,
)
from ftmoquant.data.oanda_development import _CSV_FIELDS as _OANDA_CSV_FIELDS
from ftmoquant.data.oanda_development import (
    InstrumentAcquisitionResult as _AcquisitionResult,
)
from ftmoquant.data.oanda_development import (
    MissingInterval,
    OandaDevelopmentDataError,
    _access_token_from_environment,
    _acquire_instrument,
    _candle_row,
    _fetch_response,
    _interval,
    _load_json_object,
    _load_raw_document,
    _missing_buckets,
    _parse_utc,
    _reject_worktree,
    _segment_stem,
    _segments,
    _sha256,
    _validate_cached_request,
    _validate_processed_row,
    _validate_request_partition,
    _write_idempotent_json,
    _write_replaceable_json,
)

CONFIG_PATH = Path("config/data/oanda_fx_alpha_lab_v1.yaml")
LINEAGE_ID = "oanda_fx_alpha_lab_v1"
READINESS_VERSION = "oanda-alpha-lab-readiness-1"
DEVELOPMENT_MANIFEST_FILENAME = "oanda_alpha_lab_development_manifest.json"
READINESS_FILENAME = "ftmoquant_oanda_alpha_lab_readiness.json"
#: Reuses the exact corrected principle already established for the frozen
#: Carver OANDA QA (carver-oanda-development-availability-qa-2): absent
#: calendar minutes are provider-observation availability, not an
#: acquisition-completeness defect, and never block research_ready alone.
AVAILABILITY_QA_VERSION = "oanda-fx-alpha-lab-availability-qa-2"


class OandaAlphaLabConfigError(ValueError):
    """Raised when the OANDA alpha-lab config cannot be trusted."""


class OandaAlphaLabReadinessError(ValueError):
    """Raised when the OANDA alpha-lab readiness cannot be proven."""


@dataclass(frozen=True, slots=True)
class OandaAlphaLabInstrument:
    dataset_symbol: str
    instrument_id: str
    oanda_instrument: str


@dataclass(frozen=True, slots=True)
class CachedInstrumentAvailabilityAudit:
    """Cache-only audit result: acquisition-completeness proof (gates
    ``research_ready``) kept strictly separate from raw provider-observation
    availability (never gates it alone)."""

    oanda_instrument: str
    request_record_count: int
    raw_response_count: int
    acquisition_defect_count: int
    processed_row_count: int
    first_timestamp_utc: str
    last_timestamp_utc: str
    missing_interval_count: int
    missing_minutes: int
    availability_ratio: float
    gap_buckets: dict[str, int]
    start_boundary_absence_minutes: int
    end_boundary_absence_minutes: int
    research_ready: bool
    qa_path: Path


@dataclass(frozen=True, slots=True)
class OandaAlphaLabConfig:
    lineage_id: str
    version: int
    development_start_utc: datetime
    development_end_exclusive_utc: datetime
    instruments: tuple[OandaAlphaLabInstrument, ...]
    semantic_sha256: str

    def instrument(self, instrument_id: str) -> OandaAlphaLabInstrument:
        matches = tuple(
            item for item in self.instruments if item.instrument_id == instrument_id
        )
        if len(matches) != 1:
            raise OandaAlphaLabConfigError(
                f"config does not contain exactly one {instrument_id}"
            )
        return matches[0]


def load_oanda_alpha_lab_config(path: Path = CONFIG_PATH) -> OandaAlphaLabConfig:
    """Load the exact, non-strategy OANDA alpha-lab lineage config."""

    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise OandaAlphaLabConfigError(f"could not load config: {error}") from error
    if not isinstance(document, dict):
        raise OandaAlphaLabConfigError("config must be a mapping")
    if document.get("lineage_id") != LINEAGE_ID or document.get("version") != 1:
        raise OandaAlphaLabConfigError("config identity/version is not recognized")
    provider = document.get("provider")
    if not isinstance(provider, dict) or provider != {
        "name": "OANDA_v20_practice",
        "environment": "practice",
        "api_version": "v3",
    }:
        raise OandaAlphaLabConfigError("config provider block is incompatible")
    development = document.get("development")
    if not isinstance(development, dict):
        raise OandaAlphaLabConfigError("config development block is invalid")
    start = _require_utc(development.get("start_utc"), "development.start_utc")
    end = _require_utc(
        development.get("end_exclusive_utc"), "development.end_exclusive_utc"
    )
    if end <= start:
        raise OandaAlphaLabConfigError("development range must be non-empty")
    if document.get("granularity") != "M1" or document.get("price_components") != "BA":
        raise OandaAlphaLabConfigError(
            "config granularity/price_components must be M1/BA"
        )
    policy = document.get("missing_observation_policy")
    if not isinstance(policy, dict) or policy != {
        "fills_or_interpolation": False,
        "synthetic_fill_permitted": False,
        "absent_minutes_during_genuine_market_closure_are_not_defects": True,
        "missing_observations_reported_explicitly": True,
    }:
        raise OandaAlphaLabConfigError(
            "config missing-observation policy is incompatible"
        )
    if document.get("derived_timeframes") != ["M30", "H1", "H4"]:
        raise OandaAlphaLabConfigError(
            "config derived_timeframes must be [M30, H1, H4]"
        )
    raw_instruments = document.get("instruments")
    if not isinstance(raw_instruments, list) or not raw_instruments:
        raise OandaAlphaLabConfigError("config instruments must be a non-empty list")
    instruments = tuple(_instrument(item) for item in raw_instruments)
    ids = [item.instrument_id for item in instruments]
    if len(ids) != len(set(ids)):
        raise OandaAlphaLabConfigError("config has duplicate instrument IDs")
    known_ids = {spec.instrument_id for spec in OANDA_ALPHA_LAB_SPECS}
    if set(ids) != known_ids:
        raise OandaAlphaLabConfigError(
            "config instruments do not match the known OANDA alpha-lab specs"
        )
    semantic_sha256 = hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return OandaAlphaLabConfig(
        lineage_id=LINEAGE_ID,
        version=1,
        development_start_utc=start,
        development_end_exclusive_utc=end,
        instruments=instruments,
        semantic_sha256=semantic_sha256,
    )


def _instrument(value: object) -> OandaAlphaLabInstrument:
    if not isinstance(value, dict):
        raise OandaAlphaLabConfigError("each config instrument must be a mapping")
    keys = {"dataset_symbol", "instrument_id", "oanda_instrument"}
    if set(value) != keys:
        raise OandaAlphaLabConfigError(
            f"config instrument must contain exactly: {keys}"
        )
    dataset_symbol = value["dataset_symbol"]
    instrument_id = value["instrument_id"]
    oanda_instrument = value["oanda_instrument"]
    if (
        not isinstance(dataset_symbol, str)
        or not isinstance(instrument_id, str)
        or not isinstance(oanda_instrument, str)
        or oanda_symbol(dataset_symbol) != oanda_instrument
    ):
        raise OandaAlphaLabConfigError("config instrument fields are inconsistent")
    return OandaAlphaLabInstrument(dataset_symbol, instrument_id, oanda_instrument)


def _require_utc(value: object, name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise OandaAlphaLabConfigError(f"{name} must be an ISO UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise OandaAlphaLabConfigError(
            f"{name} must be an ISO UTC timestamp"
        ) from error
    if parsed.utcoffset() != timedelta(0):
        raise OandaAlphaLabConfigError(f"{name} must be UTC")
    return parsed.astimezone(UTC)


# --------------------------------------------------------------------------
# Acquisition -- reuses oanda_development.py's generic low-level machinery
# unchanged; never touches _INSTRUMENTS or the frozen Carver path.
# --------------------------------------------------------------------------


def acquire_oanda_alpha_lab_development(
    output_root: Path, *, config_path: Path = CONFIG_PATH
) -> tuple[_AcquisitionResult, ...]:
    """Acquire the frozen OANDA alpha-lab DEVELOPMENT interval for all seven
    FX pairs and run QA, reusing the existing generic acquisition helpers."""

    config = load_oanda_alpha_lab_config(config_path)
    token = _access_token_from_environment()
    if not token:
        raise OandaDevelopmentDataError(
            "an OANDA access-token environment variable is required for acquisition"
        )
    root = output_root.resolve()
    _reject_worktree(root)
    results = tuple(
        _acquire_instrument(
            root,
            item.oanda_instrument,
            config.development_start_utc,
            config.development_end_exclusive_utc,
            token,
            _fetch_response,
        )
        for item in config.instruments
    )
    _write_idempotent_json(
        root / DEVELOPMENT_MANIFEST_FILENAME,
        {
            "lineage_id": LINEAGE_ID,
            "provider": "OANDA_v20_practice",
            "granularity": "M1",
            "price_components": "BA",
            "requested_start_utc": _iso(config.development_start_utc),
            "requested_end_exclusive_utc": _iso(config.development_end_exclusive_utc),
            "alpha_lab_config_sha256": config.semantic_sha256,
            "instruments": [
                {
                    "oanda_instrument": item.oanda_instrument,
                    "processed_path": str(result.processed_path.relative_to(root)),
                    "processed_sha256": result.processed_sha256,
                    "qa_path": str(result.qa_path.relative_to(root)),
                }
                for item, result in zip(config.instruments, results, strict=True)
            ],
        },
    )
    return results


def audit_cached_oanda_alpha_lab_development(
    output_root: Path, *, config_path: Path = CONFIG_PATH
) -> tuple[CachedInstrumentAvailabilityAudit, ...]:
    """Cache-only, no-network re-verification of already-acquired OANDA
    alpha-lab DEVELOPMENT data. Reuses the exact corrected distinction
    already established for the frozen Carver OANDA QA
    (``carver-oanda-development-availability-qa-2``): acquisition-
    completeness defects gate ``research_ready``; absent calendar minutes
    (provider-observation availability) are reported in full but never do.

    Replaces only the seven ``qa/*_M1_bid_ask_qa.json`` files. Never touches
    raw responses, request metadata, processed CSVs, or the acquisition
    manifest.
    """

    config = load_oanda_alpha_lab_config(config_path)
    root = output_root.resolve()
    manifest = _load_json_object(root / DEVELOPMENT_MANIFEST_FILENAME)
    _validate_cached_development_manifest(manifest, config)
    records = cast(list[dict[str, Any]], manifest["instruments"])
    by_instrument = {record["oanda_instrument"]: record for record in records}
    return tuple(
        _audit_cached_alpha_lab_instrument(
            root,
            item.oanda_instrument,
            config.development_start_utc,
            config.development_end_exclusive_utc,
            by_instrument[item.oanda_instrument],
        )
        for item in config.instruments
    )


def _validate_cached_development_manifest(
    manifest: dict[str, Any], config: OandaAlphaLabConfig
) -> None:
    if (
        manifest.get("lineage_id") != LINEAGE_ID
        or manifest.get("provider") != "OANDA_v20_practice"
        or manifest.get("granularity") != "M1"
        or manifest.get("price_components") != "BA"
        or manifest.get("requested_start_utc") != _iso(config.development_start_utc)
        or manifest.get("requested_end_exclusive_utc")
        != _iso(config.development_end_exclusive_utc)
        or manifest.get("alpha_lab_config_sha256") != config.semantic_sha256
    ):
        raise OandaDevelopmentDataError("cached acquisition manifest drifted")
    records = manifest.get("instruments")
    if not isinstance(records, list) or [
        record.get("oanda_instrument") if isinstance(record, dict) else None
        for record in records
    ] != [item.oanda_instrument for item in config.instruments]:
        raise OandaDevelopmentDataError("cached acquisition instruments drifted")


def _audit_cached_alpha_lab_instrument(
    root: Path,
    oanda_instrument: str,
    start: datetime,
    end: datetime,
    manifest_record: dict[str, Any],
) -> CachedInstrumentAvailabilityAudit:
    raw_root = root / "raw" / oanda_instrument
    processed = root / "processed" / f"{oanda_instrument}_M1_bid_ask.csv"
    qa_path = root / "qa" / f"{oanda_instrument}_M1_bid_ask_qa.json"
    prior_qa = _load_json_object(qa_path)
    segments = tuple(_segments(start, end))
    _validate_request_partition(segments, start, end)
    expected_responses = {
        _segment_stem(index, segment_start, segment_end) + ".json"
        for index, (segment_start, segment_end) in enumerate(segments)
    }
    expected_requests = {
        name.removesuffix(".json") + ".request.json" for name in expected_responses
    }
    actual_requests = {path.name for path in raw_root.glob("*.request.json")}
    actual_responses = {
        path.name
        for path in raw_root.glob("*.json")
        if not path.name.endswith(".request.json")
    }
    if actual_requests != expected_requests:
        raise OandaDevelopmentDataError(
            f"{oanda_instrument} skipped, overlapping, duplicated, or extra "
            "request metadata"
        )
    if actual_responses != expected_responses:
        raise OandaDevelopmentDataError(
            f"{oanda_instrument} skipped, duplicated, or extra raw response chunks"
        )

    prior_raw_hashes = prior_qa.get("raw_response_sha256")
    if not isinstance(prior_raw_hashes, dict):
        raise OandaDevelopmentDataError(
            f"{oanda_instrument} prior raw hashes are missing"
        )
    raw_hashes: dict[str, str] = {}
    missing: list[MissingInterval] = []
    previous: datetime | None = None
    minimum: datetime | None = None
    maximum: datetime | None = None
    returned_count = 0
    try:
        processed_handle = processed.open(newline="", encoding="utf-8")
    except OSError as error:
        raise OandaDevelopmentDataError(
            f"could not read processed data for {oanda_instrument}: {error}"
        ) from error
    with processed_handle:
        reader = csv.DictReader(processed_handle)
        if tuple(reader.fieldnames or ()) != _OANDA_CSV_FIELDS:
            raise OandaDevelopmentDataError(
                f"{oanda_instrument} processed columns are not exact"
            )
        for index, (segment_start, segment_end) in enumerate(segments):
            stem = _segment_stem(index, segment_start, segment_end)
            request_path = raw_root / f"{stem}.request.json"
            response_path = raw_root / f"{stem}.json"
            _validate_cached_request(
                request_path, oanda_instrument, segment_start, segment_end
            )
            raw_hashes[response_path.name] = _sha256(response_path)
            document = _load_raw_document(response_path)
            if document.get("instrument") != oanda_instrument:
                raise OandaDevelopmentDataError(
                    f"{oanda_instrument} raw response instrument mismatch"
                )
            if document.get("granularity") != "M1":
                raise OandaDevelopmentDataError(
                    f"{oanda_instrument} raw response granularity mismatch"
                )
            candles = document.get("candles")
            if not isinstance(candles, list):
                raise OandaDevelopmentDataError(
                    f"{oanda_instrument} raw response candles must be a list"
                )
            for candle in candles:
                row = _candle_row(candle)
                timestamp = _parse_utc(row["timestamp_utc"])
                if not segment_start <= timestamp < segment_end:
                    raise OandaDevelopmentDataError(
                        f"{oanda_instrument} returned timestamp outside recorded "
                        "request"
                    )
                _validate_processed_row(row, timestamp, start, end, previous)
                if previous is None and timestamp > start:
                    missing.append(_interval(start, timestamp))
                elif previous is not None and timestamp > previous + timedelta(
                    minutes=1
                ):
                    missing.append(
                        _interval(previous + timedelta(minutes=1), timestamp)
                    )
                processed_row = next(reader, None)
                if processed_row is None:
                    raise OandaDevelopmentDataError(
                        f"{oanda_instrument} processing dropped a returned candle"
                    )
                if processed_row != row:
                    raise OandaDevelopmentDataError(
                        f"{oanda_instrument} processed row differs from raw candle"
                    )
                minimum = minimum or timestamp
                maximum = timestamp
                previous = timestamp
                returned_count += 1
        if next(reader, None) is not None:
            raise OandaDevelopmentDataError(
                f"{oanda_instrument} processed data contains a non-source row"
            )
    if previous is None or minimum is None or maximum is None:
        raise OandaDevelopmentDataError(f"{oanda_instrument} has no returned candles")
    if previous + timedelta(minutes=1) < end:
        missing.append(_interval(previous + timedelta(minutes=1), end))
    processed_sha256 = _sha256(processed)
    if raw_hashes != prior_raw_hashes:
        raise OandaDevelopmentDataError(
            f"{oanda_instrument} raw-response hashes differ from recorded QA"
        )
    if manifest_record.get("processed_sha256") != processed_sha256:
        raise OandaDevelopmentDataError(
            f"{oanda_instrument} processed hash differs from acquisition manifest"
        )
    if prior_qa.get("processed_sha256") != processed_sha256:
        raise OandaDevelopmentDataError(
            f"{oanda_instrument} processed hash differs from recorded QA"
        )
    if returned_count != prior_qa.get("processed_row_count"):
        raise OandaDevelopmentDataError(
            f"{oanda_instrument} processed count differs from recorded QA"
        )

    missing_minutes = sum(item.minutes for item in missing)
    requested_minutes = int((end - start).total_seconds() // 60)
    if returned_count + missing_minutes != requested_minutes:
        raise OandaDevelopmentDataError(
            f"{oanda_instrument} availability does not partition the requested "
            "minutes"
        )
    availability_ratio = returned_count / requested_minutes
    gap_buckets = _missing_buckets(missing)
    start_boundary_absence = int((minimum - start).total_seconds() // 60)
    end_boundary_absence = int(
        (end - (maximum + timedelta(minutes=1))).total_seconds() // 60
    )

    report = {
        "qa_version": AVAILABILITY_QA_VERSION,
        "provider": "OANDA_v20_practice",
        "instrument": oanda_instrument,
        "requested_start_utc": _iso(start),
        "requested_end_exclusive_utc": _iso(end),
        "request_partition": {
            "window_count": len(segments),
            "exact_contiguous_half_open_partition": True,
            "skipped_windows": 0,
            "overlapping_windows": 0,
            "duplicated_chunks": 0,
            "accidental_early_stop": False,
        },
        "acquisition_completeness": {
            "complete": True,
            "request_metadata_matches_expected": True,
            "raw_response_correspondence_valid": True,
            "all_returned_timestamps_within_request_window": True,
            "all_candles_complete": True,
            "raw_hashes_match_prior_qa": True,
            "processed_hash_matches_prior_qa_and_manifest": True,
            "processing_preserved_every_returned_candle": True,
            "defects": [],
        },
        "provider_observation_availability": {
            "requested_calendar_minutes": requested_minutes,
            "returned_complete_candles": returned_count,
            "calendar_minutes_without_returned_candle": missing_minutes,
            "availability_ratio": availability_ratio,
            "missing_interval_count": len(missing),
            "missing_interval_duration_buckets": gap_buckets,
            "missing_intervals_utc": [asdict(item) for item in missing],
            "filled_or_interpolated_minutes": 0,
        },
        "boundaries": {
            "first_observation_utc": _iso(minimum),
            "last_observation_utc": _iso(maximum),
            "minutes_absent_at_start": start_boundary_absence,
            "minutes_absent_at_end": end_boundary_absence,
        },
        "processed_row_count": returned_count,
        "min_timestamp_utc": _iso(minimum),
        "max_timestamp_utc": _iso(maximum),
        "requested_range_respected": True,
        "duplicate_timestamps": 0,
        "timestamp_ordering": "strictly_increasing",
        "bid_ask_integrity": "validated_all_ohlc_fields",
        "research_ready": True,
        "readiness_gate": (
            "acquisition_complete_and_raw_processed_identity_proven; absent "
            "calendar minutes during genuine market closure are provider-"
            "observation availability, not an acquisition-completeness defect, "
            "and do not alone block research_ready"
        ),
        "no_synthetic_fill_or_interpolation": True,
        "raw_response_sha256": raw_hashes,
        "processed_sha256": processed_sha256,
        "raw_prices_preserved_unscaled": True,
    }
    _write_replaceable_json(qa_path, report)
    return CachedInstrumentAvailabilityAudit(
        oanda_instrument=oanda_instrument,
        request_record_count=len(segments),
        raw_response_count=len(segments),
        acquisition_defect_count=0,
        processed_row_count=returned_count,
        first_timestamp_utc=_iso(minimum),
        last_timestamp_utc=_iso(maximum),
        missing_interval_count=len(missing),
        missing_minutes=missing_minutes,
        availability_ratio=availability_ratio,
        gap_buckets=gap_buckets,
        start_boundary_absence_minutes=start_boundary_absence,
        end_boundary_absence_minutes=end_boundary_absence,
        research_ready=True,
        qa_path=qa_path,
    )


# --------------------------------------------------------------------------
# Canonicalization -- OANDA processed M1 BID/ASK CSV -> the existing
# Nautilus ParquetDataCatalog shape consumed by derive_instrument_bars().
# --------------------------------------------------------------------------


def canonicalize_oanda_instrument(
    *,
    processed_csv_path: Path,
    instrument_spec: InstrumentSpec,
    output_root: Path,
    alpha_lab_config_sha256: str,
    start_utc: datetime,
    end_exclusive_utc: datetime,
) -> Path:
    """Write one OANDA processed CSV as a fresh canonical 1-minute catalog +
    ``ftmoquant_provenance.json``, reusing InstrumentSpec/to_nautilus_bars/
    ParquetDataCatalog exactly as the Dukascopy path does. Does not invent a
    second bar-storage format."""

    instrument_spec.validate()
    root = output_root.resolve()
    catalog_path = root / "catalog"
    manifest_path = root / PARENT_MANIFEST_FILENAME
    if catalog_path.exists() or manifest_path.exists():
        raise OandaDevelopmentDataError(
            "canonicalization output root already contains artifacts"
        )
    bid_rows: list[SourceBar] = []
    ask_rows: list[SourceBar] = []
    with processed_csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            timestamp = datetime.fromisoformat(
                row["timestamp_utc"].replace("Z", "+00:00")
            ).astimezone(UTC)
            if not start_utc <= timestamp < end_exclusive_utc:
                continue
            volume = Decimal(row["volume"])
            bid_rows.append(
                SourceBar(
                    timestamp=timestamp,
                    open=Decimal(row["bid_open"]),
                    high=Decimal(row["bid_high"]),
                    low=Decimal(row["bid_low"]),
                    close=Decimal(row["bid_close"]),
                    volume=volume,
                )
            )
            ask_rows.append(
                SourceBar(
                    timestamp=timestamp,
                    open=Decimal(row["ask_open"]),
                    high=Decimal(row["ask_high"]),
                    low=Decimal(row["ask_low"]),
                    close=Decimal(row["ask_close"]),
                    volume=volume,
                )
            )
    if not bid_rows:
        raise OandaDevelopmentDataError(
            f"no admitted rows for {instrument_spec.instrument_id} in DEVELOPMENT"
        )
    catalog_path.mkdir(parents=True)
    catalog = ParquetDataCatalog(str(catalog_path))
    catalog.write_instruments([instrument_spec.nautilus_instrument()])
    catalog.write_bars(to_nautilus_bars(tuple(bid_rows), "BID", instrument_spec))
    catalog.write_bars(to_nautilus_bars(tuple(ask_rows), "ASK", instrument_spec))

    payload = {
        "instrument_id": instrument_spec.instrument_id,
        "source_granularity": "1-minute",
        "price_sides": ["BID", "ASK"],
        "nautilus_version": NAUTILUS_VERSION,
        "ingestion_version": OANDA_ALPHA_LAB_INGESTION_VERSION,
        "distribution_source": OANDA_ALPHA_LAB_DISTRIBUTION_SOURCE,
        "oanda_instrument": oanda_symbol(instrument_spec.dataset_symbol),
        "symbol": instrument_spec.dataset_symbol,
        "alpha_lab_config_sha256": alpha_lab_config_sha256,
        "fills_or_interpolation": False,
        "requested_utc_range": {
            "start_date": start_utc.date().isoformat(),
            "end_date_inclusive": (
                end_exclusive_utc.date() - timedelta(days=1)
            ).isoformat(),
        },
        "qa": {
            "bid_bar_count": len(bid_rows),
            "ask_bar_count": len(ask_rows),
            "holdout_rows_admitted": 0,
            "gaps_filled": False,
        },
    }
    semantic_sha256 = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(
            {**payload, "semantic_sha256": semantic_sha256}, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


# --------------------------------------------------------------------------
# Readiness -- a small, alpha-lab-specific readiness manifest. Deliberately
# NOT routed through universe_readiness.py (that would require touching the
# frozen Dukascopy universe hashes).
# --------------------------------------------------------------------------


def freeze_oanda_alpha_lab_readiness(
    *,
    development_roots: Mapping[str, Path],
    output_root: Path,
    config_path: Path = CONFIG_PATH,
) -> Path:
    """Freeze OANDA alpha-lab DEVELOPMENT readiness from already-derived
    per-instrument catalogs. Readiness is gated on derive_instrument_bars()'s
    own closure-aware research_ready flag (absent minutes during genuine
    market closure are not treated as defects there), not on a naive
    zero-missing-minutes count from raw acquisition QA."""

    config = load_oanda_alpha_lab_config(config_path)
    if set(development_roots) != {item.instrument_id for item in config.instruments}:
        raise OandaAlphaLabReadinessError(
            "development roots have missing/extra instruments"
        )
    spec_by_id = {spec.instrument_id: spec for spec in OANDA_ALPHA_LAB_SPECS}
    artifacts: list[dict[str, Any]] = []
    statuses: dict[str, str] = {}
    for instrument_id in sorted(development_roots):
        root = development_roots[instrument_id].resolve()
        spec = spec_by_id[instrument_id]
        parent = _read_json(root / PARENT_MANIFEST_FILENAME)
        if (
            parent.get("instrument_id") != instrument_id
            or parent.get("ingestion_version") != OANDA_ALPHA_LAB_INGESTION_VERSION
            or parent.get("alpha_lab_config_sha256") != config.semantic_sha256
        ):
            raise OandaAlphaLabReadinessError(
                f"canonical manifest is incompatible: {instrument_id}"
            )
        derived = _read_json(root / DERIVED_MANIFEST_FILENAME)
        if derived.get("instrument_id") != instrument_id:
            raise OandaAlphaLabReadinessError(
                f"derived manifest instrument mismatch: {instrument_id}"
            )
        research_ready = derived.get("research_ready") is True
        statuses[instrument_id] = "research_ready" if research_ready else "not_ready"
        artifacts.append(
            {
                "instrument_id": instrument_id,
                "dataset_symbol": spec.dataset_symbol,
                "canonical_sha256": _sha256(root / PARENT_MANIFEST_FILENAME),
                "derived_sha256": _sha256(root / DERIVED_MANIFEST_FILENAME),
                "catalog_tree_sha256": _sha256_tree(root / "catalog"),
                "coverage_status": derived.get("coverage_status"),
                "research_ready": research_ready,
            }
        )
    ordered_ids = tuple(sorted(development_roots))
    payload = {
        "readiness_version": READINESS_VERSION,
        "lineage_id": LINEAGE_ID,
        "alpha_lab_config_sha256": config.semantic_sha256,
        "development_start_utc": _iso(config.development_start_utc),
        "development_end_exclusive_utc": _iso(config.development_end_exclusive_utc),
        "ordered_instrument_ids": list(ordered_ids),
        "per_instrument_status": statuses,
        "instrument_artifacts": artifacts,
        "holdout_accessed": False,
        "holdout_rows_admitted": 0,
        "research_ready": all(
            status == "research_ready" for status in statuses.values()
        ),
    }
    semantic_sha256 = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    root = output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    readiness_path = root / READINESS_FILENAME
    readiness_path.write_text(
        json.dumps(
            {**payload, "semantic_sha256": semantic_sha256}, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    return readiness_path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OandaAlphaLabReadinessError(f"could not read {path}: {error}") from error
    if not isinstance(value, dict):
        raise OandaAlphaLabReadinessError(f"{path} is not a JSON object")
    return cast(dict[str, Any], value)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Acquire the OANDA alpha-lab DEVELOPMENT M1 BID/ASK lineage"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--cache-only-qa",
        action="store_true",
        help="audit existing raw/processed artifacts without provider access",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.cache_only_qa:
        audits = audit_cached_oanda_alpha_lab_development(
            args.output_root, config_path=args.config
        )
        for audit in audits:
            print(
                f"{audit.oanda_instrument} research_ready={audit.research_ready} "
                f"missing_minutes={audit.missing_minutes} "
                f"availability_ratio={audit.availability_ratio:.6f}"
            )
        return
    results = acquire_oanda_alpha_lab_development(
        args.output_root, config_path=args.config
    )
    print(args.output_root.resolve() / DEVELOPMENT_MANIFEST_FILENAME)
    print(f"acquired_instruments={len(results)}")


def build_canonicalize_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Canonicalize one already-acquired OANDA alpha-lab processed M1 "
            "BID/ASK CSV into the Nautilus catalog shape used by "
            "derive_instrument_bars()"
        )
    )
    parser.add_argument("--processed-csv", type=Path, required=True)
    parser.add_argument("--instrument-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    return parser


def canonicalize_main(argv: list[str] | None = None) -> None:
    args = build_canonicalize_parser().parse_args(argv)
    config = load_oanda_alpha_lab_config(args.config)
    config.instrument(args.instrument_id)  # fail closed if not in this lineage
    spec_by_id = {spec.instrument_id: spec for spec in OANDA_ALPHA_LAB_SPECS}
    spec = spec_by_id.get(args.instrument_id)
    if spec is None:
        raise OandaAlphaLabConfigError(
            f"no OANDA alpha-lab InstrumentSpec for {args.instrument_id}"
        )
    manifest_path = canonicalize_oanda_instrument(
        processed_csv_path=args.processed_csv,
        instrument_spec=spec,
        output_root=args.output_root,
        alpha_lab_config_sha256=config.semantic_sha256,
        start_utc=config.development_start_utc,
        end_exclusive_utc=config.development_end_exclusive_utc,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(manifest_path)
    print(f"instrument_id={manifest['instrument_id']}")
    print(f"bid_bar_count={manifest['qa']['bid_bar_count']}")
    print(f"ask_bar_count={manifest['qa']['ask_bar_count']}")
    print(f"semantic_sha256={manifest['semantic_sha256']}")


if __name__ == "__main__":
    main()
