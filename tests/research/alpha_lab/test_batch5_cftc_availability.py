from __future__ import annotations

import copy
import json
from datetime import UTC, date, datetime

import pytest

from ftmoquant.research.alpha_lab.batch5_cftc_availability import (
    AMENDMENT_PATH,
    EXPECTED_AMENDMENT_SEMANTIC_SHA256,
    STATUS_STANDARD,
    STATUS_UNRESOLVED,
    STATUS_VERIFIED,
    Batch5CftcAvailabilityError,
    _semantic_sha256,
    availability_for_report_date,
    is_visible_at,
    latest_visible_by_key,
    verify_amendment,
)


def test_frozen_amendment_verifies_both_documents() -> None:
    document = verify_amendment()
    assert document["amendment_semantic_sha256"] == EXPECTED_AMENDMENT_SEMANTIC_SHA256
    assert document["status_counts"] == {
        STATUS_STANDARD: 193,
        STATUS_UNRESOLVED: 12,
        STATUS_VERIFIED: 9,
    }


def test_exception_priority_and_standard_schedule() -> None:
    document = verify_amendment()
    status, timestamp = availability_for_report_date(
        date(2023, 2, 7), amendment=document
    )
    assert status == STATUS_VERIFIED
    assert (
        timestamp is not None and timestamp.isoformat() == "2023-03-03T15:30:00-05:00"
    )
    status, timestamp = availability_for_report_date(
        date(2022, 11, 22), amendment=document
    )
    assert status == STATUS_UNRESOLVED and timestamp is None
    status, timestamp = availability_for_report_date(
        date(2022, 10, 4), amendment=document
    )
    assert status == STATUS_STANDARD
    assert (
        timestamp is not None and timestamp.isoformat() == "2022-10-07T15:30:00-04:00"
    )


def test_month_end_visibility_fails_closed() -> None:
    assert not is_visible_at(None, datetime(2021, 11, 30, tzinfo=UTC))
    assert not is_visible_at(
        datetime(2023, 3, 3, 15, 30, tzinfo=UTC),
        datetime(2023, 3, 3, 15, 29, tzinfo=UTC),
    )


def test_asof_lookup_uses_latest_earlier_public_vintage_without_fill() -> None:
    observations = [
        {
            "currency": "EUR",
            "report_date": "2023-02-07",
            "availability_timestamp": "2023-03-03T15:30:00-05:00",
        },
        {
            "currency": "EUR",
            "report_date": "2023-02-14",
            "availability_timestamp": "2023-03-08T15:30:00-05:00",
        },
        {
            "currency": "JPY",
            "report_date": "2022-11-22",
            "availability_timestamp": "",
        },
    ]
    visible = latest_visible_by_key(
        observations, datetime(2023, 3, 7, 23, 59, tzinfo=UTC)
    )
    assert visible["EUR"]["report_date"] == "2023-02-07"
    assert "JPY" not in visible


def test_mutation_and_mutation_plus_rehash_are_rejected(tmp_path) -> None:
    document = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
    document["policy"]["ordinary_week_schedule"]["time"] = "00:00:00"
    path = tmp_path / "amendment.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(Batch5CftcAvailabilityError, match="semantic_sha256"):
        verify_amendment(path, verify_original=False)
    rehashed = copy.deepcopy(document)
    rehashed["amendment_semantic_sha256"] = _semantic_sha256(rehashed)
    path.write_text(json.dumps(rehashed), encoding="utf-8")
    with pytest.raises(Batch5CftcAvailabilityError, match="identity"):
        verify_amendment(path, verify_original=False)
