"""Frozen causal-availability amendment for Batch 5 CFTC observations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ftmoquant.research.alpha_lab.batch5_preregistration import (
    EXPECTED_PREREGISTRATION_SEMANTIC_SHA256,
    verify_preregistration,
)

AMENDMENT_PATH = Path(
    "config/research/batch5_cftc_causal_availability_amendment_v1.json"
)
AMENDMENT_VERSION = "batch5-cftc-causal-availability-amendment-v1"
EXPECTED_AMENDMENT_SEMANTIC_SHA256 = (
    "57c406f9c3c84ab983b3fb0ba3cca7e5e7b0ee9d39c274fa7c58f93bc766a613"
)
STATUS_VERIFIED = "VERIFIED_OFFICIAL"
STATUS_STANDARD = "STANDARD_OFFICIAL_SCHEDULE"
STATUS_UNRESOLVED = "UNRESOLVED"
_NEW_YORK = ZoneInfo("America/New_York")


class Batch5CftcAvailabilityError(RuntimeError):
    """Raised when the amendment or a causal timestamp cannot be trusted."""


def _semantic_sha256(document: dict[str, Any]) -> str:
    semantic = {k: v for k, v in document.items() if k != "amendment_semantic_sha256"}
    payload = json.dumps(
        semantic, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_amendment(
    path: Path = AMENDMENT_PATH, *, verify_original: bool = True
) -> dict[str, Any]:
    """Verify the amendment identity and, by default, both frozen documents."""

    if verify_original:
        verify_preregistration()
    try:
        document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Batch5CftcAvailabilityError("could not read amendment") from error
    actual = document.get("amendment_semantic_sha256")
    if actual != _semantic_sha256(document):
        raise Batch5CftcAvailabilityError("amendment_semantic_sha256 mismatch")
    if actual != EXPECTED_AMENDMENT_SEMANTIC_SHA256:
        raise Batch5CftcAvailabilityError("frozen amendment identity mismatch")
    if document.get("amendment_version") != AMENDMENT_VERSION:
        raise Batch5CftcAvailabilityError("unexpected amendment version")
    if (
        document.get("original_preregistration_semantic_sha256")
        != EXPECTED_PREREGISTRATION_SEMANTIC_SHA256
    ):
        raise Batch5CftcAvailabilityError("original preregistration identity mismatch")
    counts = document.get("status_counts", {})
    if sum(counts.values()) != document["audit_window"]["expected_week_count"]:
        raise Batch5CftcAvailabilityError("status counts do not cover audit window")
    return document


def availability_for_report_date(
    report_date: date, *, amendment: dict[str, Any] | None = None
) -> tuple[str, datetime | None]:
    """Apply the frozen exception-first policy to one official report date."""

    document = amendment if amendment is not None else verify_amendment()
    exceptions = {
        date.fromisoformat(row["report_date"]): row
        for row in document["exception_calendar"]
    }
    if report_date in exceptions:
        row = exceptions[report_date]
        timestamp = row["availability_timestamp"]
        return row["availability_status"], (
            datetime.fromisoformat(timestamp) if timestamp is not None else None
        )
    if report_date.weekday() != 1:
        raise Batch5CftcAvailabilityError(
            "ordinary report dates must be Tuesday; exceptional dates must be frozen"
        )
    friday = report_date + timedelta(days=3)
    return STATUS_STANDARD, datetime.combine(friday, time(15, 30), _NEW_YORK)


def is_visible_at(availability_timestamp: datetime | None, formation: datetime) -> bool:
    """Fail closed for unresolved observations and reject naive timestamps."""

    if formation.tzinfo is None:
        raise Batch5CftcAvailabilityError("formation timestamp must be timezone-aware")
    return availability_timestamp is not None and availability_timestamp <= formation


def latest_visible_by_key(
    observations: Iterable[dict[str, Any]],
    formation: datetime,
    *,
    key_field: str = "currency",
) -> dict[str, dict[str, Any]]:
    """Return each key's newest released row without filling or interpolation."""

    if formation.tzinfo is None:
        raise Batch5CftcAvailabilityError("formation timestamp must be timezone-aware")
    visible: dict[str, dict[str, Any]] = {}
    for row in observations:
        timestamp = row.get("availability_timestamp")
        if not isinstance(timestamp, str) or not timestamp:
            continue
        parsed = datetime.fromisoformat(timestamp)
        if parsed.tzinfo is None:
            raise Batch5CftcAvailabilityError("availability timestamp must be aware")
        if parsed > formation:
            continue
        key = str(row[key_field])
        incumbent = visible.get(key)
        if incumbent is None or str(row["report_date"]) > str(incumbent["report_date"]):
            visible[key] = row
    return visible
