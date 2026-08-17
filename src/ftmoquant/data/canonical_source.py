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
from ftmoquant.data.instruments import EURUSD_SPEC, InstrumentSpec, oanda_symbol

LEGACY_INGESTION_VERSION = "g0.5-1"
HF_INGESTION_VERSION = "g1-hf-dukascopy-1"
CORRECTED_INGESTION_VERSION = "g1-dukascopy-corrected-1"
CORRECTED_DISTRIBUTION_SOURCE = "Hugging Face + direct Dukascopy BI5 correction"
GENERIC_CORRECTED_INGESTION_VERSION = "g1.4a-dukascopy-corrected-1"
#: A separate, additive canonical-source lineage for the OANDA alpha-lab
#: screening universe -- never produced by, or accepted in place of, any
#: Dukascopy ingestion path above.
OANDA_ALPHA_LAB_INGESTION_VERSION = "oanda-alpha-lab-development-1"
OANDA_ALPHA_LAB_DISTRIBUTION_SOURCE = "OANDA v20 practice M1 BID/ASK candles"
TRUSTED_INGESTION_VERSIONS = frozenset(
    {
        LEGACY_INGESTION_VERSION,
        HF_INGESTION_VERSION,
        CORRECTED_INGESTION_VERSION,
        GENERIC_CORRECTED_INGESTION_VERSION,
        OANDA_ALPHA_LAB_INGESTION_VERSION,
    }
)
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

    yield from iter_paired_source_chunks(
        catalog,
        manifest,
        EURUSD_SPEC,
        start_ns,
        end_ns,
        chunk_minutes=chunk_minutes,
    )


def iter_paired_source_chunks(
    catalog: ParquetDataCatalog,
    manifest: dict[str, Any],
    instrument: InstrumentSpec,
    start_ns: int,
    end_ns: int,
    *,
    chunk_minutes: int,
) -> Iterator[PairedSourceChunk]:
    """Yield a bounded paired BID/ASK stream for one exact instrument."""

    instrument.validate()

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
        identifier = instrument.bar_type(minutes=1, side=side, aggregation="EXTERNAL")
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
            identifier = instrument.bar_type(
                minutes=1, side=side, aggregation="EXTERNAL"
            )
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

    identity = validate_canonical_source_manifest(manifest, EURUSD_SPEC)
    if identity.ingestion_version == CORRECTED_INGESTION_VERSION:
        _validate_correction_identity(manifest)
    return identity


def validate_canonical_source_manifest(
    manifest: dict[str, Any], instrument: InstrumentSpec
) -> CanonicalSourceIdentity:
    """Validate common canonical fields against an explicit instrument spec."""

    instrument.validate()

    expected = {
        "instrument_id": instrument.instrument_id,
        "source_granularity": "1-minute",
        "price_sides": ["BID", "ASK"],
        "nautilus_version": NAUTILUS_VERSION,
    }
    for field, expected_value in expected.items():
        if manifest.get(field) != expected_value:
            raise CanonicalSourceValidationError(
                f"canonical source manifest has incompatible {field}"
            )
    for field, expected_value in {
        "provider": "Dukascopy",
        "symbol": instrument.dataset_symbol,
    }.items():
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
            "price_precision": instrument.price_precision,
            "size_precision": instrument.size_precision,
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
        distribution = manifest.get("distribution_source")
        if distribution not in {
            "Hugging Face",
            "Hugging Face + direct Dukascopy cutoff completion",
        }:
            raise CanonicalSourceValidationError(
                "Hugging Face source manifest has incompatible distribution_source"
            )
        _validate_hf_identity(
            manifest,
            cast(dict[str, Any], qa),
            instrument=instrument,
            distribution_source=cast(str, distribution),
        )
        if distribution != "Hugging Face":
            _validate_hybrid_segments(manifest)
        return CanonicalSourceIdentity(ingestion_version, cast(str, distribution))
    if ingestion_version == CORRECTED_INGESTION_VERSION:
        _validate_hf_identity(
            manifest,
            cast(dict[str, Any], qa),
            distribution_source=CORRECTED_DISTRIBUTION_SOURCE,
            instrument=instrument,
        )
        return CanonicalSourceIdentity(ingestion_version, CORRECTED_DISTRIBUTION_SOURCE)
    if ingestion_version == GENERIC_CORRECTED_INGESTION_VERSION:
        _validate_hf_identity(
            manifest,
            cast(dict[str, Any], qa),
            distribution_source=CORRECTED_DISTRIBUTION_SOURCE,
            instrument=instrument,
        )
        _validate_generic_correction_identity(manifest)
        return CanonicalSourceIdentity(ingestion_version, CORRECTED_DISTRIBUTION_SOURCE)
    if ingestion_version == OANDA_ALPHA_LAB_INGESTION_VERSION:
        _validate_oanda_identity(
            manifest, cast(dict[str, Any], qa), instrument=instrument
        )
        return CanonicalSourceIdentity(
            ingestion_version, OANDA_ALPHA_LAB_DISTRIBUTION_SOURCE
        )
    raise CanonicalSourceValidationError(
        "canonical source manifest has an untrusted ingestion_version"
    )


def _validate_hf_identity(
    manifest: dict[str, Any],
    qa: dict[str, Any],
    *,
    distribution_source: str = "Hugging Face",
    instrument: InstrumentSpec = EURUSD_SPEC,
) -> None:
    expected = {
        "distribution_source": distribution_source,
        "dataset_repo": "mito0o852/dukascopy-ticks",
        "market_data_origin": "Dukascopy",
        "source_timestamp_convention": "raw_unix_epoch_milliseconds_to_bar_open_utc",
        "provider": "Dukascopy",
        "symbol": instrument.dataset_symbol,
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
    if manifest.get("dataset_id") == "g1_4_fx_usd_liquid_v1":
        rights = manifest.get("data_use_rights")
        if (
            not isinstance(rights, dict)
            or not isinstance(rights.get("evidence_sha256"), str)
            or _SHA256.fullmatch(cast(str, rights["evidence_sha256"])) is None
            or rights.get("release_blocker_resolved_for_this_acquisition") is not True
        ):
            raise CanonicalSourceValidationError(
                "G1.4 source manifest lacks immutable data-use-rights evidence"
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
            or not path.startswith(f"data/{instrument.dataset_symbol}/")
            or not isinstance(file_sha, str)
            or _SHA256.fullmatch(file_sha) is None
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
        ):
            raise CanonicalSourceValidationError(
                "Hugging Face source manifest has invalid source file provenance"
            )


def _validate_oanda_identity(
    manifest: dict[str, Any], qa: dict[str, Any], *, instrument: InstrumentSpec
) -> None:
    """Validate the OANDA alpha-lab canonical-source lineage (see
    ``oanda_alpha_lab_development.py``). No HF/Dukascopy-specific fields are
    required or accepted here; this is a separate, additive provenance."""

    if manifest.get("distribution_source") != OANDA_ALPHA_LAB_DISTRIBUTION_SOURCE:
        raise CanonicalSourceValidationError(
            "OANDA source manifest has incompatible distribution_source"
        )
    if manifest.get("oanda_instrument") != oanda_symbol(instrument.dataset_symbol):
        raise CanonicalSourceValidationError(
            "OANDA source manifest instrument does not match the declared pair"
        )
    config_sha = manifest.get("alpha_lab_config_sha256")
    semantic_sha = manifest.get("semantic_sha256")
    if not isinstance(config_sha, str) or _SHA256.fullmatch(config_sha) is None:
        raise CanonicalSourceValidationError(
            "OANDA source manifest lacks an alpha-lab config SHA-256"
        )
    if not isinstance(semantic_sha, str) or _SHA256.fullmatch(semantic_sha) is None:
        raise CanonicalSourceValidationError(
            "OANDA source manifest lacks a semantic SHA-256"
        )
    semantic_payload = {
        key: value for key, value in manifest.items() if key != "semantic_sha256"
    }
    expected_semantic_sha = hashlib.sha256(
        json.dumps(semantic_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if semantic_sha != expected_semantic_sha:
        raise CanonicalSourceValidationError(
            "OANDA source manifest semantic SHA-256 does not match"
        )
    if qa.get("holdout_rows_admitted") != 0:
        raise CanonicalSourceValidationError(
            "OANDA source manifest does not prove zero holdout admission"
        )
    if qa.get("gaps_filled") is not False:
        raise CanonicalSourceValidationError(
            "OANDA source manifest must prove gaps_filled=false"
        )
    if manifest.get("fills_or_interpolation") is not False:
        raise CanonicalSourceValidationError(
            "OANDA source manifest must prove no fills or interpolation"
        )


def _validate_correction_identity(manifest: dict[str, Any]) -> None:
    parent = manifest.get("parent_canonical")
    correction = manifest.get("correction")
    equivalence = manifest.get("parent_equivalence")
    if not isinstance(parent, dict) or not isinstance(correction, dict):
        raise CanonicalSourceValidationError(
            "corrected source manifest lacks parent/correction provenance"
        )
    if not isinstance(equivalence, dict):
        raise CanonicalSourceValidationError(
            "corrected source manifest lacks parent equivalence proof"
        )
    if (
        parent.get("ingestion_version") != HF_INGESTION_VERSION
        or not isinstance(parent.get("file_sha256"), str)
        or _SHA256.fullmatch(cast(str, parent["file_sha256"])) is None
        or not isinstance(parent.get("semantic_sha256"), str)
        or _SHA256.fullmatch(cast(str, parent["semantic_sha256"])) is None
    ):
        raise CanonicalSourceValidationError(
            "corrected source manifest has invalid parent identity"
        )
    expected_minutes = {
        "2023-11-14T13:31:00Z",
        "2023-11-29T14:29:00Z",
        "2023-11-29T14:30:00Z",
        "2023-11-29T14:31:00Z",
        "2023-11-29T14:32:00Z",
        "2023-11-29T14:33:00Z",
        "2023-11-29T14:34:00Z",
    }
    raw_minutes = correction.get("corrected_minutes")
    if (
        correction.get("identity") != CORRECTED_INGESTION_VERSION
        or correction.get("fills_or_interpolation") is not False
        or correction.get("holdout_accessed") is not False
        or correction.get("holdout_rows_admitted") != 0
        or not isinstance(raw_minutes, list)
        or {item.get("timestamp_utc") for item in raw_minutes if isinstance(item, dict)}
        != expected_minutes
        or len(raw_minutes) != len(expected_minutes)
    ):
        raise CanonicalSourceValidationError(
            "corrected source manifest has invalid exact-minute provenance"
        )
    if (
        equivalence.get("parent_bid_bars_unchanged") is not True
        or equivalence.get("parent_ask_bars_unchanged") is not True
        or equivalence.get("parent_timestamps_removed") != 0
        or equivalence.get("new_bid_timestamp_count") != 7
        or equivalence.get("new_ask_timestamp_count") != 7
        or equivalence.get("other_changed_or_added_bar_count") != 0
    ):
        raise CanonicalSourceValidationError(
            "corrected source manifest lacks an exact parent equivalence proof"
        )


def _validate_hybrid_segments(manifest: dict[str, Any]) -> None:
    segments = manifest.get("source_segments")
    if not isinstance(segments, list) or len(segments) not in {1, 2}:
        raise CanonicalSourceValidationError(
            "hybrid source manifest has invalid segments"
        )
    if not isinstance(segments[0], dict) or segments[0].get("kind") != (
        "pinned_hugging_face_parquet"
    ):
        raise CanonicalSourceValidationError(
            "hybrid source lacks its pinned HF segment"
        )
    if segments[0].get("overlaps_next_segment") is not False:
        raise CanonicalSourceValidationError("hybrid source segments may not overlap")
    if len(segments) == 2:
        direct = segments[1]
        if (
            not isinstance(direct, dict)
            or direct.get("kind") != "direct_dukascopy_bi5_cutoff_completion"
            or direct.get("overlaps_previous_segment") is not False
            or direct.get("start_utc") != segments[0].get("end_exclusive_utc")
            or not isinstance(direct.get("evidence_sha256"), str)
            or _SHA256.fullmatch(cast(str, direct["evidence_sha256"])) is None
        ):
            raise CanonicalSourceValidationError(
                "hybrid direct segment is not contiguous and immutable"
            )


def _validate_generic_correction_identity(manifest: dict[str, Any]) -> None:
    parent = manifest.get("parent_canonical")
    correction = manifest.get("correction")
    equivalence = manifest.get("parent_equivalence")
    if not all(isinstance(item, dict) for item in (parent, correction, equivalence)):
        raise CanonicalSourceValidationError(
            "generic correction lacks parent, correction, or equivalence lineage"
        )
    typed_parent = cast(dict[str, Any], parent)
    typed_correction = cast(dict[str, Any], correction)
    typed_equivalence = cast(dict[str, Any], equivalence)
    minutes = typed_correction.get("corrected_minutes_utc")
    if (
        typed_parent.get("ingestion_version") != HF_INGESTION_VERSION
        or not isinstance(typed_parent.get("file_sha256"), str)
        or _SHA256.fullmatch(cast(str, typed_parent["file_sha256"])) is None
        or typed_correction.get("identity") != GENERIC_CORRECTED_INGESTION_VERSION
        or not isinstance(minutes, list)
        or not minutes
        or len(minutes) != len(set(minutes))
        or typed_correction.get("fills_or_interpolation") is not False
        or typed_correction.get("holdout_accessed") is not False
        or typed_correction.get("holdout_rows_admitted") != 0
        or typed_equivalence.get("parent_bid_bars_unchanged") is not True
        or typed_equivalence.get("parent_ask_bars_unchanged") is not True
        or typed_equivalence.get("parent_timestamps_removed") != 0
        or typed_equivalence.get("new_bid_timestamp_count") != len(minutes)
        or typed_equivalence.get("new_ask_timestamp_count") != len(minutes)
        or typed_equivalence.get("other_changed_or_added_bar_count") != 0
    ):
        raise CanonicalSourceValidationError("generic correction lineage is invalid")
