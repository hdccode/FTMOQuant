"""Frozen DEVELOPMENT-only OANDA M1 BID/ASK acquisition and quality assurance.

This module intentionally stores provider responses before parsing them.  It is
an acquisition/QA boundary only: it neither calculates signals nor evaluates a
candidate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Protocol, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ftmoquant.research.carver_trend_carry_ftmo5_spec import (
    CARVER_TREND_CARRY_FTMO5_CONFIG_SHA256,
    load_carver_trend_carry_ftmo5_spec,
)

ACQUISITION_VERSION = "carver-oanda-development-m1-bid-ask-1"
QA_VERSION = "carver-oanda-development-availability-qa-2"
_CHUNK_MINUTES = 5_000
_INSTRUMENTS = ("XAU_USD", "SPX500_USD", "WTICO_USD", "SOYBN_USD")
_CSV_FIELDS = (
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
_OHLC_FIELD_NAMES = {"o": "open", "h": "high", "l": "low", "c": "close"}


class OandaDevelopmentDataError(ValueError):
    """Raised when OANDA acquisition or its raw evidence is not trustworthy."""


class ResponseFetcher(Protocol):
    def __call__(self, url: str, authorization: str) -> bytes: ...


@dataclass(frozen=True, slots=True)
class MissingInterval:
    start_utc: str
    end_exclusive_utc: str
    minutes: int


@dataclass(frozen=True, slots=True)
class InstrumentAcquisitionResult:
    instrument: str
    raw_response_count: int
    raw_response_sha256: dict[str, str]
    processed_path: Path
    processed_sha256: str
    qa_path: Path
    row_count: int
    missing_intervals: tuple[MissingInterval, ...]


@dataclass(frozen=True, slots=True)
class CachedInstrumentAudit:
    instrument: str
    acquisition_complete: bool
    raw_response_count: int
    returned_candle_count: int
    processed_sha256: str
    missing_interval_count: int
    missing_minutes: int
    research_ready: bool
    qa_path: Path


def acquire_carver_oanda_development(
    output_root: Path,
) -> tuple[InstrumentAcquisitionResult, ...]:
    """Acquire only the frozen Carver DEVELOPMENT interval and run QA."""
    spec = load_carver_trend_carry_ftmo5_spec()
    if spec.semantic_sha256 != CARVER_TREND_CARRY_FTMO5_CONFIG_SHA256:
        raise OandaDevelopmentDataError("Carver semantic SHA drifted")
    boundary = spec.canonical_document["research_boundary"]
    start = _parse_utc(str(boundary["development_start_utc"]))
    end = _parse_utc(str(boundary["development_end_exclusive_utc"]))
    token = _access_token_from_environment()
    if not token:
        raise OandaDevelopmentDataError(
            "an OANDA access-token environment variable is required for acquisition"
        )
    environment = os.environ.get("OANDA_ENVIRONMENT", "practice")
    if environment != "practice":
        raise OandaDevelopmentDataError("only OANDA practice environment is permitted")
    root = output_root.resolve()
    _reject_worktree(root)
    results = tuple(
        _acquire_instrument(root, instrument, start, end, token, _fetch_response)
        for instrument in _INSTRUMENTS
    )
    _write_idempotent_json(
        root / "oanda_development_manifest.json",
        {
            "acquisition_version": ACQUISITION_VERSION,
            "provider": "OANDA_v20_practice",
            "granularity": "M1",
            "price_components": "BA",
            "requested_start_utc": _utc(start),
            "requested_end_exclusive_utc": _utc(end),
            "carver_semantic_sha256": spec.semantic_sha256,
            "instruments": [
                {
                    "instrument": item.instrument,
                    "processed_path": str(item.processed_path.relative_to(root)),
                    "processed_sha256": item.processed_sha256,
                    "qa_path": str(item.qa_path.relative_to(root)),
                }
                for item in results
            ],
            "raw_prices_preserved_unscaled": True,
            "soybean_scale_transform": "not_applied_in_acquisition",
        },
    )
    audit_cached_oanda_development(root)
    return results


def audit_cached_oanda_development(
    output_root: Path,
) -> tuple[CachedInstrumentAudit, ...]:
    """Prove cached request/processing completeness without any provider access."""
    spec = load_carver_trend_carry_ftmo5_spec()
    if spec.semantic_sha256 != CARVER_TREND_CARRY_FTMO5_CONFIG_SHA256:
        raise OandaDevelopmentDataError("Carver semantic SHA drifted")
    boundary = spec.canonical_document["research_boundary"]
    start = _parse_utc(str(boundary["development_start_utc"]))
    end = _parse_utc(str(boundary["development_end_exclusive_utc"]))
    root = output_root.resolve()
    manifest = _load_json_object(root / "oanda_development_manifest.json")
    _validate_cached_manifest(manifest, spec.semantic_sha256, start, end)
    records = manifest.get("instruments")
    assert isinstance(records, list)
    by_instrument = {
        str(record["instrument"]): record
        for record in records
        if isinstance(record, dict) and "instrument" in record
    }
    return tuple(
        _audit_cached_instrument(
            root, instrument, start, end, by_instrument[instrument]
        )
        for instrument in _INSTRUMENTS
    )


def _acquire_instrument(
    root: Path,
    instrument: str,
    start: datetime,
    end: datetime,
    token: str,
    fetcher: ResponseFetcher,
) -> InstrumentAcquisitionResult:
    raw_root = root / "raw" / instrument
    raw_root.mkdir(parents=True, exist_ok=True)
    segments = tuple(_segments(start, end))
    raw_hashes: dict[str, str] = {}
    for index, (segment_start, segment_end) in enumerate(segments):
        stem = f"{index:06d}_{segment_start:%Y%m%dT%H%MZ}_{segment_end:%Y%m%dT%H%MZ}"
        response_path = raw_root / f"{stem}.json"
        request_path = raw_root / f"{stem}.request.json"
        url = _candles_url(instrument, segment_start, segment_end)
        if response_path.exists():
            payload = response_path.read_bytes()
        else:
            payload = fetcher(url, f"Bearer {token}")
            _write_idempotent_bytes(response_path, payload)
        _validate_raw_response(payload, response_path)
        _write_idempotent_json(
            request_path,
            {
                "provider": "OANDA_v20_practice",
                "instrument": instrument,
                "url": url,
                "requested_start_utc": _utc(segment_start),
                "requested_end_exclusive_utc": _utc(segment_end),
                "granularity": "M1",
                "price_components": "BA",
                "include_first": True,
                "authorization": "redacted",
            },
        )
        raw_hashes[response_path.name] = _sha256(response_path)
    processed = root / "processed" / f"{instrument}_M1_bid_ask.csv"
    qa_path = root / "qa" / f"{instrument}_M1_bid_ask_qa.json"
    rows = _iter_raw_rows(raw_root, segments, start, end)
    row_count, missing = _write_processed_and_qa(
        processed, qa_path, instrument, rows, start, end, raw_hashes
    )
    return InstrumentAcquisitionResult(
        instrument,
        len(segments),
        raw_hashes,
        processed,
        _sha256(processed),
        qa_path,
        row_count,
        missing,
    )


def _write_processed_and_qa(
    processed: Path,
    qa_path: Path,
    instrument: str,
    rows: Iterable[dict[str, str]],
    start: datetime,
    end: datetime,
    raw_hashes: dict[str, str],
) -> tuple[int, tuple[MissingInterval, ...]]:
    processed.parent.mkdir(parents=True, exist_ok=True)
    temporary = processed.with_suffix(".csv.tmp")
    count = 0
    previous: datetime | None = None
    minimum: datetime | None = None
    maximum: datetime | None = None
    missing: list[MissingInterval] = []
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            timestamp = _parse_utc(row["timestamp_utc"])
            _validate_processed_row(row, timestamp, start, end, previous)
            if previous is None and timestamp > start:
                missing.append(_interval(start, timestamp))
            elif previous is not None and timestamp > previous + timedelta(minutes=1):
                missing.append(_interval(previous + timedelta(minutes=1), timestamp))
            writer.writerow(row)
            minimum = minimum or timestamp
            maximum = timestamp
            previous = timestamp
            count += 1
    if previous is None:
        temporary.unlink(missing_ok=True)
        raise OandaDevelopmentDataError(f"no admitted candles for {instrument}")
    assert minimum is not None and maximum is not None
    if previous + timedelta(minutes=1) < end:
        missing.append(_interval(previous + timedelta(minutes=1), end))
    _replace_idempotently(temporary, processed)
    qa_payload = {
        "acquisition_version": ACQUISITION_VERSION,
        "provider": "OANDA_v20_practice",
        "instrument": instrument,
        "requested_start_utc": _utc(start),
        "requested_end_exclusive_utc": _utc(end),
        "processed_row_count": count,
        "min_timestamp_utc": _utc(minimum),
        "max_timestamp_utc": _utc(maximum),
        "requested_range_respected": minimum >= start and maximum < end,
        "duplicate_timestamps": 0,
        "timestamp_ordering": "strictly_increasing",
        "bid_ask_integrity": "validated_all_ohlc_fields",
        "missing_interval_count": len(missing),
        "missing_minutes": sum(item.minutes for item in missing),
        "missing_intervals_utc": [asdict(item) for item in missing],
        "research_ready": len(missing) == 0,
        "readiness_reason": (
            "no_missing_minutes"
            if not missing
            else "missing_minutes_require_explicit_provider-availability_review"
        ),
        "raw_response_sha256": raw_hashes,
        "processed_sha256": _sha256(processed),
        "raw_prices_preserved_unscaled": True,
        "soybean_scale_transform": "not_applied_in_acquisition",
    }
    _write_replaceable_json(qa_path, qa_payload)
    return count, tuple(missing)


def _audit_cached_instrument(
    root: Path,
    instrument: str,
    start: datetime,
    end: datetime,
    manifest_record: dict[str, Any],
) -> CachedInstrumentAudit:
    evaluator_version = _verify_irregular_observation_semantics(start)
    raw_root = root / "raw" / instrument
    processed = root / "processed" / f"{instrument}_M1_bid_ask.csv"
    qa_path = root / "qa" / f"{instrument}_M1_bid_ask_qa.json"
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
            f"{instrument} skipped, overlapping, duplicated, or extra request metadata"
        )
    if actual_responses != expected_responses:
        raise OandaDevelopmentDataError(
            f"{instrument} skipped, duplicated, or extra raw response chunks"
        )

    prior_raw_hashes = prior_qa.get("raw_response_sha256")
    if not isinstance(prior_raw_hashes, dict):
        raise OandaDevelopmentDataError(f"{instrument} prior raw hashes are missing")
    raw_hashes: dict[str, str] = {}
    missing: list[MissingInterval] = []
    previous: datetime | None = None
    minimum: datetime | None = None
    maximum: datetime | None = None
    returned_count = 0
    processed.parent.mkdir(parents=True, exist_ok=True)
    try:
        processed_handle = processed.open(newline="", encoding="utf-8")
    except OSError as error:
        raise OandaDevelopmentDataError(
            f"could not read processed data for {instrument}: {error}"
        ) from error
    with processed_handle:
        reader = csv.DictReader(processed_handle)
        if tuple(reader.fieldnames or ()) != _CSV_FIELDS:
            raise OandaDevelopmentDataError(
                f"{instrument} processed columns are not exact"
            )
        for index, (segment_start, segment_end) in enumerate(segments):
            stem = _segment_stem(index, segment_start, segment_end)
            request_path = raw_root / f"{stem}.request.json"
            response_path = raw_root / f"{stem}.json"
            _validate_cached_request(
                request_path, instrument, segment_start, segment_end
            )
            raw_hashes[response_path.name] = _sha256(response_path)
            document = _load_raw_document(response_path)
            if document.get("instrument") != instrument:
                raise OandaDevelopmentDataError(
                    f"{instrument} raw response instrument mismatch"
                )
            if document.get("granularity") != "M1":
                raise OandaDevelopmentDataError(
                    f"{instrument} raw response granularity mismatch"
                )
            candles = document.get("candles")
            if not isinstance(candles, list):
                raise OandaDevelopmentDataError(
                    f"{instrument} raw response candles must be a list"
                )
            for candle in candles:
                row = _candle_row(candle)
                timestamp = _parse_utc(row["timestamp_utc"])
                if not segment_start <= timestamp < segment_end:
                    raise OandaDevelopmentDataError(
                        f"{instrument} returned timestamp outside recorded request"
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
                        f"{instrument} processing dropped a returned candle"
                    )
                if processed_row != row:
                    raise OandaDevelopmentDataError(
                        f"{instrument} processed row differs from raw candle"
                    )
                minimum = minimum or timestamp
                maximum = timestamp
                previous = timestamp
                returned_count += 1
        if next(reader, None) is not None:
            raise OandaDevelopmentDataError(
                f"{instrument} processed data contains a non-source row"
            )
    if previous is None or minimum is None or maximum is None:
        raise OandaDevelopmentDataError(f"{instrument} has no returned candles")
    if previous + timedelta(minutes=1) < end:
        missing.append(_interval(previous + timedelta(minutes=1), end))
    processed_sha256 = _sha256(processed)
    if raw_hashes != prior_raw_hashes:
        raise OandaDevelopmentDataError(
            f"{instrument} raw-response hashes differ from recorded QA"
        )
    if prior_qa.get("processed_sha256") != processed_sha256:
        raise OandaDevelopmentDataError(
            f"{instrument} processed hash differs from recorded QA"
        )
    if manifest_record.get("processed_sha256") != processed_sha256:
        raise OandaDevelopmentDataError(
            f"{instrument} processed hash differs from acquisition manifest"
        )
    if returned_count != prior_qa.get("processed_row_count"):
        raise OandaDevelopmentDataError(
            f"{instrument} processed count differs from recorded QA"
        )

    missing_minutes = sum(item.minutes for item in missing)
    requested_minutes = int((end - start).total_seconds() // 60)
    if returned_count + missing_minutes != requested_minutes:
        raise OandaDevelopmentDataError(
            f"{instrument} availability does not partition the requested minutes"
        )
    report = {
        "qa_version": QA_VERSION,
        "acquisition_version": ACQUISITION_VERSION,
        "provider": "OANDA_v20_practice",
        "instrument": instrument,
        "requested_start_utc": _utc(start),
        "requested_end_exclusive_utc": _utc(end),
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
            "availability_ratio": returned_count / requested_minutes,
            "missing_interval_count": len(missing),
            "missing_interval_duration_buckets": _missing_buckets(missing),
            "missing_intervals_utc": [asdict(item) for item in missing],
            "filled_or_interpolated_minutes": 0,
        },
        "boundaries": {
            "first_observation_utc": _utc(minimum),
            "last_observation_utc": _utc(maximum),
            "minutes_absent_at_start": int((minimum - start).total_seconds() // 60),
            "minutes_absent_at_end": int(
                (end - (maximum + timedelta(minutes=1))).total_seconds() // 60
            ),
        },
        "processed_row_count": returned_count,
        "min_timestamp_utc": _utc(minimum),
        "max_timestamp_utc": _utc(maximum),
        "requested_range_respected": True,
        "duplicate_timestamps": 0,
        "timestamp_ordering": "strictly_increasing",
        "bid_ask_integrity": "validated_all_ohlc_fields",
        "strategy_observation_usability": {
            "usable_without_semantic_change": True,
            "evaluator_version": evaluator_version,
            "reason": (
                "frozen evaluator selects the first genuine eligible observation "
                "strictly later than the signal from a strictly increasing sequence"
            ),
        },
        "research_ready": True,
        "readiness_gate": (
            "acquisition_complete_and_raw_processed_identity_proven_and_"
            "irregular_observations_usable_without_strategy_change"
        ),
        "raw_response_sha256": raw_hashes,
        "processed_sha256": processed_sha256,
        "raw_prices_preserved_unscaled": True,
        "soybean_scale_transform": "not_applied_in_acquisition",
    }
    _write_replaceable_json(qa_path, report)
    return CachedInstrumentAudit(
        instrument,
        True,
        len(segments),
        returned_count,
        processed_sha256,
        len(missing),
        missing_minutes,
        True,
        qa_path,
    )


def _verify_irregular_observation_semantics(signal: datetime) -> str:
    from ftmoquant.research.carver_trend_carry_ftmo5_development import (
        EVALUATOR_VERSION,
        first_strictly_later_execution,
    )

    sparse_next = signal + timedelta(minutes=15)
    if first_strictly_later_execution(signal, (signal, sparse_next)) != sparse_next:
        raise OandaDevelopmentDataError(
            "frozen evaluator cannot consume sparse genuine observations"
        )
    return EVALUATOR_VERSION


def _validate_request_partition(
    windows: Sequence[tuple[datetime, datetime]],
    start: datetime,
    end: datetime,
) -> None:
    if not windows or windows[0][0] != start:
        raise OandaDevelopmentDataError("request partition has a skipped start")
    previous_end = start
    for window_start, window_end in windows:
        if window_end <= window_start:
            raise OandaDevelopmentDataError("request partition has an invalid window")
        if window_start > previous_end:
            raise OandaDevelopmentDataError("request partition has a skipped window")
        if window_start < previous_end:
            raise OandaDevelopmentDataError("request partition has overlapping windows")
        previous_end = window_end
    if previous_end != end:
        raise OandaDevelopmentDataError("request partition has a skipped end")


def _validate_cached_request(
    path: Path,
    instrument: str,
    start: datetime,
    end: datetime,
) -> None:
    value = _load_json_object(path)
    expected = {
        "provider": "OANDA_v20_practice",
        "instrument": instrument,
        "url": _candles_url(instrument, start, end),
        "requested_start_utc": _utc(start),
        "requested_end_exclusive_utc": _utc(end),
        "granularity": "M1",
        "price_components": "BA",
        "include_first": True,
        "authorization": "redacted",
    }
    if value != expected:
        raise OandaDevelopmentDataError(
            f"{instrument} raw response does not match recorded request metadata"
        )


def _validate_cached_manifest(
    manifest: dict[str, Any],
    carver_sha256: str,
    start: datetime,
    end: datetime,
) -> None:
    if (
        manifest.get("acquisition_version") != ACQUISITION_VERSION
        or manifest.get("provider") != "OANDA_v20_practice"
        or manifest.get("granularity") != "M1"
        or manifest.get("price_components") != "BA"
        or manifest.get("requested_start_utc") != _utc(start)
        or manifest.get("requested_end_exclusive_utc") != _utc(end)
        or manifest.get("carver_semantic_sha256") != carver_sha256
        or manifest.get("raw_prices_preserved_unscaled") is not True
        or manifest.get("soybean_scale_transform") != "not_applied_in_acquisition"
    ):
        raise OandaDevelopmentDataError("cached acquisition manifest drifted")
    records = manifest.get("instruments")
    if not isinstance(records, list) or [
        record.get("instrument") if isinstance(record, dict) else None
        for record in records
    ] != list(_INSTRUMENTS):
        raise OandaDevelopmentDataError("cached acquisition instruments drifted")


def _missing_buckets(intervals: Sequence[MissingInterval]) -> dict[str, int]:
    buckets = {"1_minute": 0, "2_to_5": 0, "6_to_15": 0, "16_to_60": 0, "over_60": 0}
    for interval in intervals:
        if interval.minutes == 1:
            buckets["1_minute"] += 1
        elif interval.minutes <= 5:
            buckets["2_to_5"] += 1
        elif interval.minutes <= 15:
            buckets["6_to_15"] += 1
        elif interval.minutes <= 60:
            buckets["16_to_60"] += 1
        else:
            buckets["over_60"] += 1
    return buckets


def _iter_raw_rows(
    raw_root: Path,
    segments: Sequence[tuple[datetime, datetime]],
    start: datetime,
    end: datetime,
) -> Iterable[dict[str, str]]:
    for index, (segment_start, segment_end) in enumerate(segments):
        stem = f"{index:06d}_{segment_start:%Y%m%dT%H%MZ}_{segment_end:%Y%m%dT%H%MZ}"
        document = _load_raw_document(raw_root / f"{stem}.json")
        candles = document.get("candles")
        if not isinstance(candles, list):
            raise OandaDevelopmentDataError("raw response candles must be a list")
        for candle in candles:
            row = _candle_row(candle)
            timestamp = _parse_utc(row["timestamp_utc"])
            if start <= timestamp < end:
                yield row


def _candle_row(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or value.get("complete") is not True:
        raise OandaDevelopmentDataError("OANDA candle must be a complete object")
    timestamp = value.get("time")
    bid = value.get("bid")
    ask = value.get("ask")
    volume = value.get("volume")
    if (
        not isinstance(timestamp, str)
        or not isinstance(bid, dict)
        or not isinstance(ask, dict)
    ):
        raise OandaDevelopmentDataError("OANDA candle is missing time, bid, or ask")
    row = {"timestamp_utc": _utc(_parse_utc(timestamp)), "volume": str(volume)}
    for side, source in (("bid", bid), ("ask", ask)):
        for field in ("o", "h", "l", "c"):
            item = source.get(field)
            if not isinstance(item, str):
                raise OandaDevelopmentDataError("OANDA OHLC field must be a string")
            row[f"{side}_{_OHLC_FIELD_NAMES[field]}"] = item
    return row


def _validate_processed_row(
    row: dict[str, str],
    timestamp: datetime,
    start: datetime,
    end: datetime,
    previous: datetime | None,
) -> None:
    if timestamp.second or timestamp.microsecond or not start <= timestamp < end:
        raise OandaDevelopmentDataError(
            "candle timestamp is outside the frozen minute range"
        )
    if previous is not None and timestamp <= previous:
        raise OandaDevelopmentDataError("duplicate or unordered OANDA candle timestamp")
    values: dict[str, Decimal] = {}
    for field in _CSV_FIELDS[1:-1]:
        values[field] = _decimal(row[field], field)
    volume = _decimal(row["volume"], "volume", allow_zero=True)
    if volume < 0:
        raise OandaDevelopmentDataError("OANDA candle volume is negative")
    for side in ("bid", "ask"):
        low, high = values[f"{side}_low"], values[f"{side}_high"]
        if (
            low > high
            or not low <= values[f"{side}_open"] <= high
            or not low <= values[f"{side}_close"] <= high
        ):
            raise OandaDevelopmentDataError(f"invalid {side} OHLC")
    for field in ("open", "high", "low", "close"):
        if values[f"ask_{field}"] < values[f"bid_{field}"]:
            raise OandaDevelopmentDataError("OANDA ASK is below BID")


def _segments(start: datetime, end: datetime) -> Iterable[tuple[datetime, datetime]]:
    current = start
    while current < end:
        following = min(current + timedelta(minutes=_CHUNK_MINUTES), end)
        yield current, following
        current = following


def _segment_stem(index: int, start: datetime, end: datetime) -> str:
    return f"{index:06d}_{start:%Y%m%dT%H%MZ}_{end:%Y%m%dT%H%MZ}"


def _candles_url(instrument: str, start: datetime, end: datetime) -> str:
    query = urlencode(
        {
            "from": _utc(start),
            "to": _utc(end),
            "granularity": "M1",
            "price": "BA",
            "includeFirst": "true",
        }
    )
    return (
        f"https://api-fxpractice.oanda.com/v3/instruments/{instrument}/candles?{query}"
    )


def _fetch_response(url: str, authorization: str) -> bytes:
    request = Request(
        url,
        headers={"Authorization": authorization, "Accept-Datetime-Format": "RFC3339"},
    )
    try:
        with urlopen(request, timeout=60) as response:  # noqa: S310 - fixed HTTPS OANDA host
            return cast(bytes, response.read())
    except (HTTPError, URLError) as error:
        raise OandaDevelopmentDataError(f"OANDA request failed: {error}") from error


def _access_token_from_environment() -> str | None:
    """Use existing secret wiring without persisting or printing its value."""
    for name in ("OANDA_ACCESS_TOKEN", "OANDA_API_TOKEN", "OANDA_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value
    return None


def _validate_raw_response(payload: bytes, path: Path) -> None:
    try:
        _load_raw_document_from_bytes(payload)
    except OandaDevelopmentDataError as error:
        raise OandaDevelopmentDataError(
            f"invalid raw OANDA response {path.name}: {error}"
        ) from error


def _load_raw_document(path: Path) -> dict[str, Any]:
    return _load_raw_document_from_bytes(path.read_bytes())


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OandaDevelopmentDataError(
            f"could not read JSON artifact {path}"
        ) from error
    if not isinstance(value, dict):
        raise OandaDevelopmentDataError(f"JSON artifact is not an object: {path}")
    return value


def _load_raw_document_from_bytes(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OandaDevelopmentDataError("response is not JSON") from error
    if not isinstance(value, dict):
        raise OandaDevelopmentDataError("response must be a JSON object")
    return value


def _write_idempotent_json(path: Path, value: dict[str, Any]) -> None:
    _write_idempotent_bytes(
        path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    )


def _write_replaceable_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically replace derived QA while leaving raw evidence immutable."""
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == payload:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _write_idempotent_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise OandaDevelopmentDataError(f"immutable artifact conflicts: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _replace_idempotently(temporary: Path, target: Path) -> None:
    payload = temporary.read_bytes()
    if target.exists() and target.read_bytes() == payload:
        temporary.unlink()
    elif target.exists():
        temporary.unlink()
        raise OandaDevelopmentDataError(f"immutable artifact conflicts: {target}")
    else:
        temporary.replace(target)


def _decimal(value: str, field: str, *, allow_zero: bool = False) -> Decimal:
    try:
        result = Decimal(value)
    except InvalidOperation as error:
        raise OandaDevelopmentDataError(f"malformed numeric {field}") from error
    if not result.is_finite() or result < 0 or (result == 0 and not allow_zero):
        raise OandaDevelopmentDataError(f"invalid numeric {field}")
    return result


def _interval(start: datetime, end: datetime) -> MissingInterval:
    return MissingInterval(
        _utc(start), _utc(end), int((end - start).total_seconds() // 60)
    )


def _parse_utc(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise OandaDevelopmentDataError("timestamp is not RFC3339 UTC") from error
    if not value.endswith("Z") or result.utcoffset() != UTC.utcoffset(result):
        raise OandaDevelopmentDataError("timestamp is not UTC")
    return result.astimezone(UTC)


def _utc(value: datetime | None) -> str:
    assert value is not None
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reject_worktree(root: Path) -> None:
    if (root / ".git").exists():
        raise OandaDevelopmentDataError("output root must be outside the Git worktree")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Acquire frozen Carver OANDA DEVELOPMENT M1 BID/ASK data"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--cache-only-qa",
        action="store_true",
        help="audit existing raw/processed artifacts without provider access",
    )
    args = parser.parse_args(argv)
    result = (
        audit_cached_oanda_development(args.output_root)
        if args.cache_only_qa
        else acquire_carver_oanda_development(args.output_root)
    )
    print(args.output_root.resolve() / "oanda_development_manifest.json")
    print(f"qaed_instruments={len(result)}")


if __name__ == "__main__":
    main()
