"""Session-aware Dukascopy EUR/USD source coverage quality assurance."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

from nautilus_trader.model import Bar
from nautilus_trader.persistence import ParquetDataCatalog

from ftmoquant.data.derived_bars import (
    DERIVATION_VERSION,
    DERIVED_MANIFEST_FILENAME,
    PARENT_MANIFEST_FILENAME,
)
from ftmoquant.data.dukascopy import (
    INGESTION_VERSION,
    INSTRUMENT_ID,
    NAUTILUS_VERSION,
)

COVERAGE_MANIFEST_FILENAME = "ftmoquant_session_coverage.json"
COVERAGE_QA_VERSION = "g1-data-readiness-1"
SESSION_POLICY_VERSION = "dukascopy-eurusd-ny-close-v1"
SESSION_TIMEZONE = "America/New_York"
SESSION_POLICY_VALID_FROM_UTC = datetime(2019, 3, 10, 21, tzinfo=UTC)

_MINUTE = timedelta(minutes=1)
_MINUTE_NS = 60_000_000_000
_UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_SIDES = ("BID", "ASK")
_CLASS_EXPECTED_CLOSED = "expected_market_closed"
_CLASS_UNEXPLAINED = "unexplained_missing"
_NEW_YORK_CLOSE = time(17, 0)
_SESSION_ZONE = ZoneInfo(SESSION_TIMEZONE)

_AUTHORITATIVE_SOURCES: tuple[dict[str, str], ...] = (
    {
        "description": (
            "Dukascopy general trading hours: most instruments trade from Sunday "
            "21:00 GMT in summer / 22:00 GMT in winter until the corresponding "
            "Friday close"
        ),
        "url": (
            "https://www.dukascopy.com/swiss/english/forex/"
            "forex-trading-accounts/link/"
        ),
    },
    {
        "description": (
            "Dukascopy 2019 DST notice: the FX trading day ends at 17:00 New York "
            "time and market opening/settlement moves between 22:00 and 21:00 GMT "
            "with US Eastern DST"
        ),
        "url": (
            "https://www.dukascopy.com/swiss/english/about/ournews/"
            "change-to-daylight-saving-time-dbl201384"
        ),
    },
    {
        "description": (
            "Dukascopy JForex IDataService market-hours documentation: historical "
            "and upcoming offline periods are available as ITimeDomain values"
        ),
        "url": (
            "https://www.dukascopy.com/wiki/en/development/strategy-api/"
            "instruments/market-hours/"
        ),
    },
)


class CoverageValidationError(ValueError):
    """Raised when coverage inputs or their bound provenance are not trustworthy."""


@dataclass(frozen=True, slots=True)
class CoverageInterval:
    """An inclusive range of incomplete paired one-minute source updates."""

    classification: str
    start_utc: str
    end_utc: str
    minutes: int
    missing_sides: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CoverageAssessment:
    """Deterministic session classification for one half-open UTC interval."""

    requested_start_utc: str
    requested_end_exclusive_utc: str
    requested_minute_count: int
    expected_open_minute_count: int
    observed_paired_minute_count: int
    observed_paired_expected_open_minute_count: int
    expected_closure_minute_count: int
    unexplained_missing_minute_count: int
    expected_closure_intervals: tuple[CoverageInterval, ...]
    unexplained_missing_intervals: tuple[CoverageInterval, ...]


@dataclass(frozen=True, slots=True)
class SessionCoverageResult:
    """Location and readiness outcome from a persisted coverage bridge run."""

    manifest_path: Path
    session_aware_research_ready: bool
    semantic_sha256: str


def is_eurusd_expected_open(timestamp: datetime) -> bool:
    """Return whether a UTC minute is inside Dukascopy's recurring FX week."""

    timestamp = _validate_utc_minute(timestamp, "timestamp")
    if timestamp < SESSION_POLICY_VALID_FROM_UTC:
        raise CoverageValidationError(
            "the session policy is not authoritative before "
            f"{_format_utc(SESSION_POLICY_VALID_FROM_UTC)}"
        )
    local = timestamp.astimezone(_SESSION_ZONE)
    weekday = local.weekday()
    if weekday < 4:
        return True
    if weekday == 4:
        return local.timetz().replace(tzinfo=None) < _NEW_YORK_CLOSE
    if weekday == 5:
        return False
    return local.timetz().replace(tzinfo=None) >= _NEW_YORK_CLOSE


def assess_eurusd_source_coverage(
    start_utc: datetime,
    end_exclusive_utc: datetime,
    bid_timestamps: Sequence[datetime],
    ask_timestamps: Sequence[datetime],
) -> CoverageAssessment:
    """Classify every absent paired source minute without filling any data."""

    start = _validate_utc_minute(start_utc, "start_utc")
    end = _validate_utc_minute(end_exclusive_utc, "end_exclusive_utc")
    if end <= start:
        raise CoverageValidationError("requested UTC interval must be non-empty")
    if start < SESSION_POLICY_VALID_FROM_UTC:
        raise CoverageValidationError(
            "the requested interval predates authoritative session policy support"
        )

    bids = _timestamp_set(bid_timestamps, start, end, "BID")
    asks = _timestamp_set(ask_timestamps, start, end, "ASK")
    paired = bids & asks

    expected_open_count = 0
    paired_open_count = 0
    expected_closed_count = 0
    unexplained_count = 0
    intervals: list[CoverageInterval] = []
    active: tuple[str, tuple[str, ...], datetime, datetime] | None = None

    current = start
    while current < end:
        expected_open = is_eurusd_expected_open(current)
        if expected_open:
            expected_open_count += 1
        if current in paired:
            if expected_open:
                paired_open_count += 1
            if active is not None:
                intervals.append(_coverage_interval(*active))
                active = None
        else:
            classification = (
                _CLASS_UNEXPLAINED if expected_open else _CLASS_EXPECTED_CLOSED
            )
            missing_sides = tuple(
                side
                for side, observed in (("BID", bids), ("ASK", asks))
                if current not in observed
            )
            if classification == _CLASS_UNEXPLAINED:
                unexplained_count += 1
            else:
                expected_closed_count += 1
            if (
                active is not None
                and active[0] == classification
                and active[1] == missing_sides
                and current == active[3] + _MINUTE
            ):
                active = (active[0], active[1], active[2], current)
            else:
                if active is not None:
                    intervals.append(_coverage_interval(*active))
                active = (classification, missing_sides, current, current)
        current += _MINUTE

    if active is not None:
        intervals.append(_coverage_interval(*active))
    requested_count = int((end - start) / _MINUTE)
    if len(paired) + expected_closed_count + unexplained_count != requested_count:
        raise AssertionError("coverage classification does not partition the request")

    return CoverageAssessment(
        requested_start_utc=_format_utc(start),
        requested_end_exclusive_utc=_format_utc(end),
        requested_minute_count=requested_count,
        expected_open_minute_count=expected_open_count,
        observed_paired_minute_count=len(paired),
        observed_paired_expected_open_minute_count=paired_open_count,
        expected_closure_minute_count=expected_closed_count,
        unexplained_missing_minute_count=unexplained_count,
        expected_closure_intervals=tuple(
            item for item in intervals if item.classification == _CLASS_EXPECTED_CLOSED
        ),
        unexplained_missing_intervals=tuple(
            item for item in intervals if item.classification == _CLASS_UNEXPLAINED
        ),
    )


def run_eurusd_session_coverage_qa(output_root: Path) -> SessionCoverageResult:
    """Run and persist the bridge over existing G0.5 and G0.6 artifacts."""

    root = output_root.resolve()
    parent_path = root / PARENT_MANIFEST_FILENAME
    derived_path = root / DERIVED_MANIFEST_FILENAME
    catalog_path = root / "catalog"
    parent = _load_json_object(parent_path, "G0.5 source manifest")
    _validate_parent_manifest(parent)
    start, end = _requested_interval(parent)
    if not catalog_path.is_dir():
        raise CoverageValidationError(f"missing G0.5 catalog: {catalog_path}")

    catalog = ParquetDataCatalog(str(catalog_path))
    sources = {
        side: tuple(catalog.query_bars([_source_bar_type(side)])) for side in _SIDES
    }
    _validate_source_bars(sources, parent, start, end)
    assessment = assess_eurusd_source_coverage(
        start,
        end,
        tuple(_bar_timestamp(bar) for bar in sources["BID"]),
        tuple(_bar_timestamp(bar) for bar in sources["ASK"]),
    )

    parent_sha256 = _sha256(parent_path)
    derived = _load_json_object(derived_path, "G0.6 derived manifest")
    structural_valid = _derived_structure_valid(derived, parent_sha256, sources)
    payload = _coverage_manifest(
        assessment=assessment,
        parent_sha256=parent_sha256,
        derived_sha256=_sha256(derived_path),
        structural_valid=structural_valid,
    )
    semantic_sha256 = _semantic_sha256(payload)
    manifest = {**payload, "semantic_sha256": semantic_sha256}
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    manifest_path = root / COVERAGE_MANIFEST_FILENAME
    if manifest_path.exists() and manifest_path.read_bytes() != manifest_bytes:
        raise CoverageValidationError(
            "existing session coverage manifest conflicts with this run: "
            f"{manifest_path}"
        )
    if not manifest_path.exists():
        manifest_path.write_bytes(manifest_bytes)
    return SessionCoverageResult(
        manifest_path=manifest_path,
        session_aware_research_ready=cast(
            bool, manifest["session_aware_research_ready"]
        ),
        semantic_sha256=semantic_sha256,
    )


def _coverage_manifest(
    *,
    assessment: CoverageAssessment,
    parent_sha256: str,
    derived_sha256: str,
    structural_valid: bool,
) -> dict[str, Any]:
    return {
        "coverage_qa_version": COVERAGE_QA_VERSION,
        "provider": "Dukascopy",
        "instrument": "EUR/USD",
        "instrument_id": INSTRUMENT_ID,
        "session_policy": {
            "version": SESSION_POLICY_VERSION,
            "valid_from_utc": _format_utc(SESSION_POLICY_VALID_FROM_UTC),
            "timezone": SESSION_TIMEZONE,
            "timezone_semantics": (
                "IANA America/New_York wall time via standard-library zoneinfo; "
                "US Eastern DST therefore maps 17:00 to 21:00 UTC in summer and "
                "22:00 UTC in winter"
            ),
            "weekly_session": (
                "expected open from Sunday 17:00 America/New_York inclusive to "
                "Friday 17:00 America/New_York exclusive"
            ),
            "offline_exception_policy": (
                "no historical IDataService ITimeDomain evidence is embedded; all "
                "missing expected-open minutes, including holidays or exceptional "
                "provider outages, remain unexplained"
            ),
        },
        "authoritative_sources": [dict(item) for item in _AUTHORITATIVE_SOURCES],
        "nautilus_reference": {
            "version": NAUTILUS_VERSION,
            "inspected_helpers": [
                "ForexSession.NEW_YORK",
                "fx_local_from_utc",
                "fx_next_end",
                "fx_next_start",
                "fx_prev_end",
                "fx_prev_start",
            ],
            "adopted_for_provider_classification": False,
            "reason": (
                "helpers model weekday regional sessions, not Dukascopy's weekly "
                "provider session or historical offline domains"
            ),
        },
        "requested_utc_interval": {
            "start_utc": assessment.requested_start_utc,
            "end_exclusive_utc": assessment.requested_end_exclusive_utc,
        },
        "parent_g0_5_manifest": PARENT_MANIFEST_FILENAME,
        "parent_g0_5_manifest_sha256": parent_sha256,
        "structural_g0_6_manifest": DERIVED_MANIFEST_FILENAME,
        "structural_g0_6_manifest_sha256": derived_sha256,
        "counts": {
            "requested_minute_count": assessment.requested_minute_count,
            "expected_open_minute_count": assessment.expected_open_minute_count,
            "observed_paired_minute_count": assessment.observed_paired_minute_count,
            "observed_paired_expected_open_minute_count": (
                assessment.observed_paired_expected_open_minute_count
            ),
            "expected_closure_minute_count": (
                assessment.expected_closure_minute_count
            ),
            "unexplained_missing_minute_count": (
                assessment.unexplained_missing_minute_count
            ),
        },
        "expected_closure_intervals_utc": [
            asdict(item) for item in assessment.expected_closure_intervals
        ],
        "unexplained_missing_intervals_utc": [
            asdict(item) for item in assessment.unexplained_missing_intervals
        ],
        "source_derived_structural_integrity_valid": structural_valid,
        "session_aware_research_ready": (
            structural_valid and assessment.unexplained_missing_minute_count == 0
        ),
        "fills_or_interpolation": False,
        "semantic_hash_contract": (
            "SHA-256 of canonical JSON for every other manifest field; no fetch or "
            "wall-clock timestamp is included"
        ),
    }


def _validate_parent_manifest(manifest: dict[str, Any]) -> None:
    expected = {
        "provider": "Dukascopy",
        "symbol": "EURUSD",
        "instrument_id": INSTRUMENT_ID,
        "source_granularity": "1-minute",
        "price_sides": ["BID", "ASK"],
        "ingestion_version": INGESTION_VERSION,
        "nautilus_version": NAUTILUS_VERSION,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise CoverageValidationError(f"G0.5 source manifest has invalid {field}")


def _requested_interval(manifest: dict[str, Any]) -> tuple[datetime, datetime]:
    raw = manifest.get("requested_utc_range")
    if not isinstance(raw, dict):
        raise CoverageValidationError("G0.5 source manifest has invalid UTC range")
    requested = cast(dict[str, Any], raw)
    try:
        start_date = date.fromisoformat(str(requested["start_date"]))
        end_date = date.fromisoformat(str(requested["end_date_inclusive"]))
    except (KeyError, ValueError) as error:
        raise CoverageValidationError(
            "G0.5 source manifest has invalid UTC dates"
        ) from error
    start = datetime.combine(start_date, datetime.min.time(), tzinfo=UTC)
    end = datetime.combine(
        end_date + timedelta(days=1), datetime.min.time(), tzinfo=UTC
    )
    _validate_utc_minute(start, "requested start")
    _validate_utc_minute(end, "requested end")
    if end <= start:
        raise CoverageValidationError("G0.5 requested UTC interval is empty")
    return start, end


def _validate_source_bars(
    sources: dict[str, tuple[Bar, ...]],
    parent: dict[str, Any],
    start: datetime,
    end: datetime,
) -> None:
    qa_raw = parent.get("qa")
    if not isinstance(qa_raw, dict):
        raise CoverageValidationError("G0.5 source manifest has invalid qa")
    qa = cast(dict[str, Any], qa_raw)
    timestamps: dict[str, tuple[int, ...]] = {}
    for side in _SIDES:
        bars = sources[side]
        if not bars:
            raise CoverageValidationError(f"G0.5 {side} source series is empty")
        if qa.get(f"{side.lower()}_bar_count") != len(bars):
            raise CoverageValidationError(
                f"{side} catalog count does not match the G0.5 source manifest"
            )
        previous: int | None = None
        for bar in bars:
            if str(bar.bar_type) != _source_bar_type(side):
                raise CoverageValidationError(f"catalog contains an invalid {side} bar")
            if bar.ts_event % _MINUTE_NS != 0:
                raise CoverageValidationError(
                    f"{side} source bar is not minute aligned"
                )
            if bar.ts_init != bar.ts_event + _MINUTE_NS:
                raise CoverageValidationError(
                    f"{side} source bar violates G0.5 timestamp semantics"
                )
            if previous is not None and bar.ts_event <= previous:
                raise CoverageValidationError(
                    f"{side} source bars are not strictly chronological"
                )
            timestamp = _bar_timestamp(bar)
            if not start <= timestamp < end:
                raise CoverageValidationError(
                    f"{side} source bar is outside the requested UTC interval"
                )
            previous = bar.ts_event
        timestamps[side] = tuple(bar.ts_event for bar in bars)
    if timestamps["BID"] != timestamps["ASK"]:
        raise CoverageValidationError("G0.5 BID/ASK source coverage is incompatible")


def _derived_structure_valid(
    manifest: dict[str, Any],
    parent_sha256: str,
    sources: dict[str, tuple[Bar, ...]],
) -> bool:
    expected = {
        "derivation_version": DERIVATION_VERSION,
        "parent_g0_5_manifest": PARENT_MANIFEST_FILENAME,
        "parent_g0_5_manifest_sha256": parent_sha256,
        "parent_ingestion_version": INGESTION_VERSION,
        "nautilus_version": NAUTILUS_VERSION,
        "instrument_id": INSTRUMENT_ID,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise CoverageValidationError(f"G0.6 derived manifest has invalid {field}")
    counts_raw = manifest.get("counts")
    if not isinstance(counts_raw, dict):
        raise CoverageValidationError("G0.6 derived manifest has invalid counts")
    source_raw = cast(dict[str, Any], counts_raw).get("source_1_minute")
    if not isinstance(source_raw, dict):
        raise CoverageValidationError("G0.6 derived manifest has invalid source counts")
    for side in _SIDES:
        if cast(dict[str, Any], source_raw).get(side) != len(sources[side]):
            raise CoverageValidationError(
                f"G0.6 {side} source count does not match the catalog"
            )
    emitted_raw = cast(dict[str, Any], counts_raw).get("emitted")
    if not isinstance(emitted_raw, dict):
        raise CoverageValidationError(
            "G0.6 derived manifest has invalid emitted counts"
        )
    all_required_series_nonempty = True
    for timeframe in ("1H", "4H"):
        timeframe_raw = cast(dict[str, Any], emitted_raw).get(timeframe)
        if not isinstance(timeframe_raw, dict):
            raise CoverageValidationError(
                f"G0.6 derived manifest has invalid {timeframe} counts"
            )
        for side in _SIDES:
            count = cast(dict[str, Any], timeframe_raw).get(side)
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise CoverageValidationError(
                    f"G0.6 derived manifest has invalid {timeframe} {side} count"
                )
            all_required_series_nonempty = all_required_series_nonempty and count > 0
    return (
        manifest.get("derived_bar_integrity_valid") is True
        and manifest.get("bid_ask_derived_coverage_matches") is True
        and all_required_series_nonempty
    )


def _timestamp_set(
    values: Sequence[datetime],
    start: datetime,
    end: datetime,
    side: str,
) -> set[datetime]:
    normalized = [_validate_utc_minute(value, f"{side} timestamp") for value in values]
    if len(set(normalized)) != len(normalized):
        raise CoverageValidationError(f"{side} timestamps contain duplicates")
    if any(value < start or value >= end for value in normalized):
        raise CoverageValidationError(
            f"{side} timestamp is outside the requested UTC interval"
        )
    return set(normalized)


def _validate_utc_minute(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise CoverageValidationError(f"{field} must be timezone-aware UTC")
    if value.utcoffset() != timedelta(0):
        raise CoverageValidationError(f"{field} must use UTC")
    normalized = value.astimezone(UTC)
    if normalized.second != 0 or normalized.microsecond != 0:
        raise CoverageValidationError(f"{field} must be exactly minute aligned")
    return normalized


def _coverage_interval(
    classification: str,
    missing_sides: tuple[str, ...],
    start: datetime,
    end: datetime,
) -> CoverageInterval:
    return CoverageInterval(
        classification=classification,
        start_utc=_format_utc(start),
        end_utc=_format_utc(end),
        minutes=int((end - start) / _MINUTE) + 1,
        missing_sides=missing_sides,
    )


def _bar_timestamp(bar: Bar) -> datetime:
    seconds, nanos = divmod(bar.ts_event, 1_000_000_000)
    if nanos:
        raise CoverageValidationError("source timestamp is not second aligned")
    return _UNIX_EPOCH + timedelta(seconds=seconds)


def _source_bar_type(side: str) -> str:
    return f"{INSTRUMENT_ID}-1-MINUTE-{side}-EXTERNAL"


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CoverageValidationError(f"invalid {description}: {error}") from error
    if not isinstance(raw, dict):
        raise CoverageValidationError(f"{description} must be an object")
    return cast(dict[str, Any], raw)


def _semantic_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _format_utc(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify Dukascopy EUR/USD source coverage by provider session"
    )
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the session-aware EUR/USD coverage command."""

    args = _build_parser().parse_args(argv)
    result = run_eurusd_session_coverage_qa(cast(Path, args.output_root))
    print(result.manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
