"""Point-in-time importer for the documented Forex Factory/Hanover calendar CSVs.

This module intentionally does not calculate surprises, returns, or trade signals.
It preserves the published strings alongside conservative numeric parsing so a later
preregistered event study can choose its definitions without rewriting the source.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

IMPORTER_VERSION = "macro-events-ptit-1"
DEFAULT_TIMEZONE = "America/New_York"
NORMALIZATION_RULE_VERSION = "us-nfp-cpi-exact-labels-1"

_MISSING_NUMERIC = frozenset({"", "-", "—", "n/a", "na", "null", "none"})
_HEADER_ALIASES = {
    "date": ("date",),
    "time": ("time",),
    "currency": ("currency",),
    "impact": ("impact",),
    "event_description": ("news description", "event", "event description", "title"),
    "actual": ("actual",),
    "forecast": ("forecast",),
    "previous": ("previous",),
    "revised_previous": ("revised from", "revised previous", "revised"),
    "event_id": ("ff event id", "event id", "id"),
    "group_id": ("group id", "event group id"),
    "actual_better_worse": ("actual better/worse", "actual bw", "actual color"),
    "previous_better_worse": ("previous better/worse", "previous bw", "previous color"),
}
_NUMBER = re.compile(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))(?:\s*)([%KMBT])?$", re.I)


class MacroEventImportError(ValueError):
    """Raised when a source cannot support a deterministic PIT import."""


@dataclass(frozen=True, slots=True)
class ImportResult:
    normalized_rows: int
    quarantined_rows: int
    output_dir: Path


def import_historical_calendar(
    inputs: list[Path],
    output_dir: Path,
    *,
    source_name: str,
    source_url: str,
    timezone_name: str = DEFAULT_TIMEZONE,
    access_date: str | None = None,
    secondary_input: Path | None = None,
) -> ImportResult:
    """Import local, immutable calendar files and persist deterministic artifacts."""

    if not inputs:
        raise MacroEventImportError("at least one input CSV is required")
    zone = _load_zone(timezone_name)
    accessed = access_date or date.today().isoformat()
    _parse_date(accessed, "access_date")
    raw_rows: list[dict[str, Any]] = []
    source_files: list[dict[str, str]] = []
    for input_path in sorted((path.resolve() for path in inputs), key=str):
        if not input_path.is_file():
            raise MacroEventImportError(f"input is not a file: {input_path}")
        source_files.append(
            {"path": str(input_path), "sha256": _sha256_file(input_path)}
        )
        raw_rows.extend(_read_csv(input_path))

    normalized: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    scope_excluded = 0
    for row in raw_rows:
        candidate, reason = _normalize_row(row, zone, timezone_name)
        if reason == "out_of_scope":
            scope_excluded += 1
        elif reason is not None:
            quarantined.append({"reason": reason, "raw": row})
        elif candidate is not None:
            normalized.append(candidate)

    normalized.sort(
        key=lambda row: (
            row["timestamp_utc"],
            row["event_family"],
            row["source_file"],
            row["source_row_number"],
        )
    )
    retained, duplicate_quarantine = _quarantine_exact_duplicates(normalized)
    quarantined.extend(duplicate_quarantine)
    qa = _qa_report(raw_rows, retained, quarantined, scope_excluded)
    cross_check = _cross_check(retained, secondary_input, zone, timezone_name)
    qa["secondary_cross_check"] = cross_check

    destination = output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    normalized_path = destination / "normalized_events.jsonl"
    quarantine_path = destination / "quarantined_events.jsonl"
    _write_jsonl(normalized_path, retained)
    _write_jsonl(quarantine_path, quarantined)
    qa["output_sha256"] = _sha256_file(normalized_path)
    _write_json(destination / "qa_report.json", qa)
    manifest = {
        "importer_version": IMPORTER_VERSION,
        "source": {
            "name": source_name,
            "url": source_url,
            "files": source_files,
            "access_date": accessed,
        },
        "timezone_semantics": {
            "source_timezone": timezone_name,
            "source_timestamp": (
                "Date + Time fields interpreted as wall-clock time in source_timezone"
            ),
            "utc_timestamp": (
                "timezone-aware conversion; ambiguous or nonexistent local "
                "timestamps are quarantined"
            ),
        },
        "normalization": {
            "rule_version": NORMALIZATION_RULE_VERSION,
            "scope": (
                "USD NFP headline employment change and distinct US CPI "
                "headline/core m/m and y/y labels only"
            ),
            "rules": _normalization_rules(),
        },
        "outputs": {
            "normalized_events": normalized_path.name,
            "normalized_events_sha256": qa["output_sha256"],
            "qa_report": "qa_report.json",
            "quarantine": quarantine_path.name,
        },
    }
    _write_json(destination / "provenance_manifest.json", manifest)
    return ImportResult(len(retained), len(quarantined), destination)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise MacroEventImportError(f"CSV has no header: {path}")
            mapping = _map_headers(reader.fieldnames, path)
            rows = []
            for row_number, source in enumerate(reader, start=2):
                raw = {
                    key: (value or "").strip()
                    for key, value in source.items()
                    if key is not None
                }
                fields = {
                    field: raw.get(header, "") for field, header in mapping.items()
                }
                rows.append(
                    {
                        "source_file": str(path.resolve()),
                        "source_row_number": row_number,
                        "fields": fields,
                        "raw_columns": raw,
                    }
                )
            return rows
    except UnicodeDecodeError as error:
        raise MacroEventImportError(f"CSV must be UTF-8: {path}") from error


def _map_headers(headers: Sequence[str], path: Path) -> dict[str, str]:
    by_canonical = {_canon_header(header): header for header in headers}
    mapped: dict[str, str] = {}
    for field, aliases in _HEADER_ALIASES.items():
        for alias in aliases:
            if alias in by_canonical:
                mapped[field] = by_canonical[alias]
                break
    required = (
        "date",
        "time",
        "currency",
        "event_description",
        "actual",
        "forecast",
        "previous",
    )
    missing = [field for field in required if field not in mapped]
    if missing:
        raise MacroEventImportError(f"CSV missing required columns {missing}: {path}")
    return mapped


def _normalize_row(
    row: dict[str, Any], zone: ZoneInfo, timezone_name: str
) -> tuple[dict[str, Any] | None, str | None]:
    fields = row["fields"]
    currency = fields["currency"].upper()
    description = fields["event_description"]
    family = _event_family(currency, description)
    if family is None:
        return None, "out_of_scope"
    timestamp, timestamp_reason = _parse_timestamp(fields["date"], fields["time"], zone)
    if timestamp_reason:
        return None, timestamp_reason
    parsed = {
        name: _parse_numeric(fields.get(name, ""))
        for name in ("actual", "forecast", "previous", "revised_previous")
    }
    return {
        "timestamp_utc": timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "timestamp_local": timestamp.isoformat(),
        "timezone_provenance": timezone_name,
        "currency": currency,
        "impact": fields.get("impact", ""),
        "event_description_raw": description,
        "event_family": family,
        "actual_raw": fields.get("actual", ""),
        "forecast_raw": fields.get("forecast", ""),
        "previous_raw": fields.get("previous", ""),
        "revised_previous_raw": fields.get("revised_previous", ""),
        "event_id_raw": fields.get("event_id", ""),
        "group_id_raw": fields.get("group_id", ""),
        "actual_better_worse_raw": fields.get("actual_better_worse", ""),
        "previous_better_worse_raw": fields.get("previous_better_worse", ""),
        "actual_parsed": parsed["actual"],
        "forecast_parsed": parsed["forecast"],
        "previous_parsed": parsed["previous"],
        "revised_previous_parsed": parsed["revised_previous"],
        "source_file": row["source_file"],
        "source_row_number": row["source_row_number"],
        "raw_source_fields": row["raw_columns"],
    }, None


def _normalization_rules() -> dict[str, list[str]]:
    return {
        "US_NFP_HEADLINE_EMPLOYMENT_CHANGE": [
            "Non-Farm Employment Change",
            "Nonfarm Employment Change",
            "Non-Farm Payrolls",
            "Nonfarm Payrolls",
        ],
        "US_CPI_HEADLINE_M_M": ["CPI m/m"],
        "US_CPI_HEADLINE_Y_Y": ["CPI y/y"],
        "US_CPI_CORE_M_M": ["Core CPI m/m"],
        "US_CPI_CORE_Y_Y": ["Core CPI y/y"],
    }


def _event_family(currency: str, description: str) -> str | None:
    if currency != "USD":
        return None
    canonical = " ".join(description.split()).casefold()
    for family, labels in _normalization_rules().items():
        if canonical in {" ".join(label.split()).casefold() for label in labels}:
            return family
    return None


def _parse_timestamp(
    raw_date: str, raw_time: str, zone: ZoneInfo
) -> tuple[datetime, str | None]:
    if raw_time.strip().casefold() in {"all day", "tentative", "", "-"}:
        return datetime.min.replace(tzinfo=UTC), "missing_or_nonexact_timestamp"
    parsed_date: date | None = None
    for fmt in ("%Y-%m-%d", "%m-%d-%Y", "%m/%d/%Y", "%b %d, %Y", "%d %b %Y"):
        try:
            parsed_date = datetime.strptime(raw_date.strip(), fmt).date()
            break
        except ValueError:
            pass
    if parsed_date is None:
        return datetime.min.replace(tzinfo=UTC), "malformed_date"
    parsed_time: datetime | None = None
    for fmt in ("%H:%M", "%I:%M%p", "%I:%M %p"):
        try:
            parsed_time = datetime.strptime(raw_time.strip().upper(), fmt)
            break
        except ValueError:
            pass
    if parsed_time is None:
        return datetime.min.replace(tzinfo=UTC), "malformed_time"
    naive = datetime.combine(parsed_date, parsed_time.time())
    candidates = []
    for fold in (0, 1):
        candidate = naive.replace(tzinfo=zone, fold=fold)
        if candidate.astimezone(UTC).astimezone(zone).replace(tzinfo=None) == naive:
            candidates.append(candidate)
    if not candidates:
        return datetime.min.replace(tzinfo=UTC), "nonexistent_local_timestamp"
    if len({item.utcoffset() for item in candidates}) > 1:
        return datetime.min.replace(tzinfo=UTC), "ambiguous_local_timestamp"
    return candidates[0], None


def _parse_numeric(value: str) -> dict[str, Any]:
    raw = value.strip()
    if raw.casefold() in _MISSING_NUMERIC:
        return {"status": "missing", "value": None, "unit": None}
    compact = raw.replace(",", "").replace(" ", "")
    match = _NUMBER.fullmatch(compact)
    if not match:
        return {"status": "malformed", "value": None, "unit": None}
    unit = (match.group(2) or "number").upper()
    return {
        "status": "parsed",
        "value": match.group(1),
        "unit": "percent" if unit == "%" else unit.lower(),
    }


def _quarantine_exact_duplicates(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seen: dict[str, dict[str, Any]] = {}
    kept: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    for row in rows:
        identity = json.dumps(
            {
                key: row[key]
                for key in (
                    "timestamp_utc",
                    "currency",
                    "event_description_raw",
                    "actual_raw",
                    "forecast_raw",
                    "previous_raw",
                    "revised_previous_raw",
                    "event_id_raw",
                    "group_id_raw",
                )
            },
            sort_keys=True,
        )
        if identity in seen:
            quarantined.append(
                {
                    "reason": "exact_duplicate",
                    "duplicate_of": {
                        "source_file": seen[identity]["source_file"],
                        "source_row_number": seen[identity]["source_row_number"],
                    },
                    "raw": row,
                }
            )
        else:
            seen[identity] = row
            kept.append(row)
    return kept, quarantined


def _qa_report(
    raw: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    quarantined: list[dict[str, Any]],
    scope_excluded: int,
) -> dict[str, Any]:
    by_family = Counter(row["event_family"] for row in rows)
    parsed = [row["forecast_parsed"] for row in rows]
    actual = [row["actual_parsed"] for row in rows]
    names = defaultdict(set)
    for row in rows:
        names[row["event_family"]].add(row["event_description_raw"])
    midnight = [row for row in rows if row["timestamp_local"][11:16] == "00:00"]
    revisions = [row for row in rows if row["revised_previous_raw"]]
    malformed = sum(
        value["status"] == "malformed"
        for row in rows
        for value in (
            row["actual_parsed"],
            row["forecast_parsed"],
            row["previous_parsed"],
            row["revised_previous_parsed"],
        )
    )
    dates = [row["timestamp_utc"][:10] for row in rows]
    reasons = Counter(item["reason"] for item in quarantined)
    return {
        "qa_version": IMPORTER_VERSION,
        "total_source_rows": len(raw),
        "in_scope_rows": len(rows) + sum(reasons.values()),
        "scope_excluded_rows": scope_excluded,
        "normalized_rows": len(rows),
        "date_coverage_utc": {
            "start": min(dates) if dates else None,
            "end": max(dates) if dates else None,
        },
        "counts_by_event_family": dict(sorted(by_family.items())),
        "missing_forecast_percent": _percent(
            sum(value["status"] == "missing" for value in parsed), len(parsed)
        ),
        "missing_actual_percent": _percent(
            sum(value["status"] == "missing" for value in actual), len(actual)
        ),
        "malformed_numeric_field_count": malformed,
        "exact_duplicate_count": reasons["exact_duplicate"],
        "timestamp_anomalies": {
            reason: count
            for reason, count in sorted(reasons.items())
            if "timestamp" in reason or reason in {"malformed_date", "malformed_time"}
        }
        | {"suspicious_midnight_count": len(midnight)},
        "revision_prevalence_percent": _percent(len(revisions), len(rows)),
        "revision_count": len(revisions),
        "rejected_or_quarantined_rows": len(quarantined),
        "quarantine_by_reason": dict(sorted(reasons.items())),
        "event_name_drift_examples": {
            family: sorted(labels)
            for family, labels in sorted(names.items())
            if len(labels) > 1
        },
    }


def _cross_check(
    rows: list[dict[str, Any]],
    secondary: Path | None,
    zone: ZoneInfo,
    timezone_name: str,
) -> dict[str, Any]:
    if secondary is None:
        return {
            "status": "not_run",
            "reason": (
                "no secondary free source was documented in the repository or "
                "supplied to this import"
            ),
        }
    secondary_rows = []
    for raw in _read_csv(secondary.resolve()):
        candidate, reason = _normalize_row(raw, zone, timezone_name)
        if candidate is not None and reason is None:
            secondary_rows.append(candidate)
    primary_keys = {
        (
            row["timestamp_utc"],
            row["event_family"],
            row["actual_raw"],
            row["forecast_raw"],
        )
        for row in rows
    }
    secondary_keys = {
        (
            row["timestamp_utc"],
            row["event_family"],
            row["actual_raw"],
            row["forecast_raw"],
        )
        for row in secondary_rows
    }
    sample = sorted(primary_keys)[:10]
    return {
        "status": "completed_non_authoritative",
        "secondary_file": str(secondary.resolve()),
        "fixed_sample_size": len(sample),
        "matching_rows": sum(key in secondary_keys for key in sample),
        "timestamp_qa": (
            "accepted only exact, timezone-aware timestamps; unmatched rows do "
            "not alter primary output"
        ),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canon_header(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def _load_zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except Exception as error:
        raise MacroEventImportError(f"invalid IANA timezone: {name}") from error


def _parse_date(value: str, label: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise MacroEventImportError(f"{label} must be YYYY-MM-DD") from error


def _percent(numerator: int, denominator: int) -> float:
    return round(100 * numerator / denominator, 6) if denominator else 0.0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Import PIT US NFP/CPI Forex Factory/Hanover calendar CSVs"
    )
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--source-name", default="Forex Factory historical calendar / Hanover archive"
    )
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--access-date")
    parser.add_argument("--secondary-input", type=Path)
    args = parser.parse_args(argv)
    result = import_historical_calendar(
        args.input,
        args.output_dir,
        source_name=args.source_name,
        source_url=args.source_url,
        timezone_name=args.timezone,
        access_date=args.access_date,
        secondary_input=args.secondary_input,
    )
    print(
        json.dumps(
            {
                "normalized_rows": result.normalized_rows,
                "quarantined_rows": result.quarantined_rows,
                "output_dir": str(result.output_dir),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
