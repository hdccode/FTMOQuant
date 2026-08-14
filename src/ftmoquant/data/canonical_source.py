"""Shared fail-closed validation for canonical EUR/USD source manifests."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, cast

from nautilus_trader.model import Bar
from nautilus_trader.persistence import ParquetDataCatalog

from ftmoquant.data.dukascopy import INSTRUMENT_ID, NAUTILUS_VERSION

LEGACY_INGESTION_VERSION = "g0.5-1"
HF_INGESTION_VERSION = "g1-hf-dukascopy-1"
TRUSTED_INGESTION_VERSIONS = frozenset({LEGACY_INGESTION_VERSION, HF_INGESTION_VERSION})
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_MINUTE_NS = 60_000_000_000
_SIDES = ("BID", "ASK")


class CanonicalSourceValidationError(ValueError):
    """Raised when source provenance cannot prove the canonical contract."""


@dataclass(frozen=True, slots=True)
class CanonicalSourceIdentity:
    """The trusted ingestion identity established by a source manifest."""

    ingestion_version: str
    distribution_source: str


@dataclass(frozen=True, slots=True)
class PairedSourceChunk:
    """One bounded half-open UTC range of validated paired source bars."""

    start_ns: int
    end_ns: int
    bid_bars: tuple[Bar, ...]
    ask_bars: tuple[Bar, ...]


def iter_paired_eurusd_source_chunks(
    catalog: ParquetDataCatalog,
    manifest: dict[str, Any],
    start_ns: int,
    end_ns: int,
    *,
    chunk_minutes: int,
) -> Iterator[PairedSourceChunk]:
    """Yield strictly validated paired bars without materializing the catalog."""

    if start_ns % _MINUTE_NS or end_ns % _MINUTE_NS or end_ns <= start_ns:
        raise CanonicalSourceValidationError(
            "canonical source scan requires a non-empty minute-aligned range"
        )
    if chunk_minutes <= 0:
        raise CanonicalSourceValidationError(
            "source scan chunk_minutes must be positive"
        )
    qa_raw = manifest.get("qa")
    if not isinstance(qa_raw, dict):
        raise CanonicalSourceValidationError("canonical source manifest has invalid qa")
    qa = cast(dict[str, Any], qa_raw)
    expected_counts: dict[str, int] = {}
    for side in _SIDES:
        value = qa.get(f"{side.lower()}_bar_count")
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise CanonicalSourceValidationError(
                f"canonical source manifest has invalid {side} bar count"
            )
        expected_counts[side] = value

    for side in _SIDES:
        identifier = _source_bar_type(side)
        first_init = catalog.query_first_timestamp("bars", identifier)
        last_init = catalog.query_last_timestamp("bars", identifier)
        if first_init is None or last_init is None:
            raise CanonicalSourceValidationError(
                f"canonical {side} source series is empty"
            )
        if first_init < start_ns + _MINUTE_NS or last_init > end_ns:
            raise CanonicalSourceValidationError(
                f"canonical {side} source bars fall outside the requested UTC range"
            )

    previous: dict[str, int | None] = {side: None for side in _SIDES}
    counts = {side: 0 for side in _SIDES}
    chunk_ns = chunk_minutes * _MINUTE_NS
    chunk_start = start_ns
    while chunk_start < end_ns:
        chunk_end = min(chunk_start + chunk_ns, end_ns)
        by_side: dict[str, tuple[Bar, ...]] = {}
        for side in _SIDES:
            identifier = _source_bar_type(side)
            queried = catalog.query_bars([identifier], start=chunk_start, end=chunk_end)
            bars = tuple(
                bar for bar in queried if chunk_start <= bar.ts_event < chunk_end
            )
            for bar in bars:
                if str(bar.bar_type) != identifier:
                    raise CanonicalSourceValidationError(
                        f"catalog returned an invalid {side} source bar type"
                    )
                if bar.ts_event % _MINUTE_NS:
                    raise CanonicalSourceValidationError(
                        f"canonical {side} source bar is not UTC-minute aligned"
                    )
                if bar.ts_init != bar.ts_event + _MINUTE_NS:
                    raise CanonicalSourceValidationError(
                        f"canonical {side} source bar has invalid timestamps"
                    )
                if previous[side] is not None and bar.ts_event <= cast(
                    int, previous[side]
                ):
                    raise CanonicalSourceValidationError(
                        f"canonical {side} source bars are not strictly chronological"
                    )
                previous[side] = bar.ts_event
            counts[side] += len(bars)
            by_side[side] = bars
        bid_times = tuple(bar.ts_event for bar in by_side["BID"])
        ask_times = tuple(bar.ts_event for bar in by_side["ASK"])
        if bid_times != ask_times:
            raise CanonicalSourceValidationError(
                "BID/ASK one-minute source coverage mismatch; not research-ready"
            )
        yield PairedSourceChunk(
            start_ns=chunk_start,
            end_ns=chunk_end,
            bid_bars=by_side["BID"],
            ask_bars=by_side["ASK"],
        )
        chunk_start = chunk_end

    for side in _SIDES:
        if counts[side] != expected_counts[side]:
            raise CanonicalSourceValidationError(
                f"{side} catalog count does not match the canonical source manifest"
            )


def _source_bar_type(side: str) -> str:
    return f"{INSTRUMENT_ID}-1-MINUTE-{side}-EXTERNAL"


def validate_canonical_eurusd_source_manifest(
    manifest: dict[str, Any],
) -> CanonicalSourceIdentity:
    """Validate common canonical fields and one of two explicit provenances."""

    expected = {
        "instrument_id": INSTRUMENT_ID,
        "source_granularity": "1-minute",
        "price_sides": ["BID", "ASK"],
        "nautilus_version": NAUTILUS_VERSION,
    }
    for field, expected_value in expected.items():
        if manifest.get(field) != expected_value:
            raise CanonicalSourceValidationError(
                f"canonical source manifest has incompatible {field}"
            )
    for field, expected_value in {"provider": "Dukascopy", "symbol": "EURUSD"}.items():
        if field in manifest and manifest[field] != expected_value:
            raise CanonicalSourceValidationError(
                f"canonical source manifest has incompatible {field}"
            )
    encoding = manifest.get("nautilus_encoding")
    if encoding is not None:
        if not isinstance(encoding, dict):
            raise CanonicalSourceValidationError(
                "canonical source manifest has invalid nautilus_encoding"
            )
        typed_encoding = cast(dict[str, Any], encoding)
        expected_encoding = {
            "price_precision": 5,
            "size_precision": 8,
            "ts_event": "bar_open_utc",
            "ts_init": "bar_close_utc",
        }
        for field, value in expected_encoding.items():
            if typed_encoding.get(field) != value:
                raise CanonicalSourceValidationError(
                    "canonical source manifest has incompatible "
                    f"nautilus_encoding.{field}"
                )
    qa = manifest.get("qa")
    if not isinstance(qa, dict):
        raise CanonicalSourceValidationError("canonical source manifest has invalid qa")
    ingestion_version = manifest.get("ingestion_version")
    if ingestion_version == LEGACY_INGESTION_VERSION:
        upstream = manifest.get("upstream_downloader")
        if upstream is not None and (
            not isinstance(upstream, dict)
            or upstream.get("package") != "tradedesk-dukascopy"
        ):
            raise CanonicalSourceValidationError(
                "legacy source manifest lacks tradedesk-dukascopy provenance"
            )
        if "gaps_filled" in qa and qa.get("gaps_filled") is not False:
            raise CanonicalSourceValidationError(
                "legacy source manifest has incompatible gaps_filled"
            )
        return CanonicalSourceIdentity(ingestion_version, "tradedesk-dukascopy")
    if ingestion_version == HF_INGESTION_VERSION:
        _validate_hf_identity(manifest, cast(dict[str, Any], qa))
        return CanonicalSourceIdentity(ingestion_version, "Hugging Face")
    raise CanonicalSourceValidationError(
        "canonical source manifest has an untrusted ingestion_version"
    )


def _validate_hf_identity(manifest: dict[str, Any], qa: dict[str, Any]) -> None:
    expected = {
        "distribution_source": "Hugging Face",
        "dataset_repo": "mito0o852/dukascopy-ticks",
        "market_data_origin": "Dukascopy",
        "source_timestamp_convention": "raw_unix_epoch_milliseconds_to_bar_open_utc",
        "provider": "Dukascopy",
        "symbol": "EURUSD",
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise CanonicalSourceValidationError(
                f"Hugging Face source manifest has incompatible {field}"
            )
    revision = manifest.get("dataset_revision")
    if not isinstance(revision, str) or _GIT_SHA.fullmatch(revision) is None:
        raise CanonicalSourceValidationError(
            "Hugging Face source manifest lacks an immutable dataset revision"
        )
    plan_sha = manifest.get("dataset_plan_sha256")
    semantic_sha = manifest.get("semantic_sha256")
    if not isinstance(plan_sha, str) or _SHA256.fullmatch(plan_sha) is None:
        raise CanonicalSourceValidationError(
            "Hugging Face source manifest lacks a dataset-plan SHA-256"
        )
    if not isinstance(semantic_sha, str) or _SHA256.fullmatch(semantic_sha) is None:
        raise CanonicalSourceValidationError(
            "Hugging Face source manifest lacks a semantic SHA-256"
        )
    semantic_payload = {
        key: value for key, value in manifest.items() if key != "semantic_sha256"
    }
    expected_semantic_sha = hashlib.sha256(
        json.dumps(semantic_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if semantic_sha != expected_semantic_sha:
        raise CanonicalSourceValidationError(
            "Hugging Face source manifest semantic SHA-256 does not match"
        )
    if qa.get("holdout_rows_admitted") != 0:
        raise CanonicalSourceValidationError(
            "Hugging Face source manifest does not prove zero holdout admission"
        )
    if qa.get("gaps_filled") is not False:
        raise CanonicalSourceValidationError(
            "Hugging Face source manifest must prove gaps_filled=false"
        )
    if manifest.get("fills_or_interpolation") is not False:
        raise CanonicalSourceValidationError(
            "Hugging Face source manifest must prove no fills or interpolation"
        )
    source_files = manifest.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise CanonicalSourceValidationError(
            "Hugging Face source manifest lacks selected source files"
        )
    for item in source_files:
        if not isinstance(item, dict):
            raise CanonicalSourceValidationError(
                "Hugging Face source manifest has invalid source file provenance"
            )
        path = item.get("repo_path")
        file_sha = item.get("downloaded_sha256")
        size = item.get("downloaded_size")
        if (
            not isinstance(path, str)
            or not path.startswith("data/EURUSD/")
            or not isinstance(file_sha, str)
            or _SHA256.fullmatch(file_sha) is None
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
        ):
            raise CanonicalSourceValidationError(
                "Hugging Face source manifest has invalid source file provenance"
            )
