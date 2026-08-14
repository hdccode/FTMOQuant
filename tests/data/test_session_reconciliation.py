import hashlib
import json
import lzma
import struct
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import NoReturn

import pytest

import ftmoquant.data.session_reconciliation as reconciliation
from ftmoquant.data.research_plan import ResearchDataPlan, load_research_data_plan

PLAN = Path("config/data/eurusd_research_v1.yaml")
CUTOFF = datetime(2024, 8, 21, tzinfo=UTC)


def test_new_year_provider_offline_requires_exact_evidence() -> None:
    start = _dt("2023-12-31T22:00:00Z")
    missing = _missing(start, 3)

    unsupported = _build(missing, (), {})
    supported = _build(
        missing,
        (
            reconciliation.OfflineDomain(
                "jforex-new-year",
                start,
                start + timedelta(minutes=3),
            ),
        ),
        {},
    )

    assert unsupported["counts"]["provider_offline_minute_count"] == 0
    assert unsupported["counts"]["unexplained_missing_minute_count"] == 3
    assert supported["counts"]["provider_offline_minute_count"] == 3
    assert supported["counts"]["unexplained_missing_minute_count"] == 0
    assert supported["session_aware_research_ready"] is True


def test_unsupported_holiday_looking_interval_remains_unexplained() -> None:
    missing = _missing(_dt("2020-12-25T08:00:00Z"), 2)

    result = _build(missing, (), {})

    assert result["provider_offline_intervals_utc"] == []
    assert result["unexplained_missing_intervals_utc"][0]["minutes"] == 2
    assert result["session_aware_research_ready"] is False


def test_offline_domain_export_requires_valid_semantic_provenance(
    tmp_path: Path,
) -> None:
    plan = load_research_data_plan(PLAN)
    payload: dict[str, object] = {
        "schema_version": "dukascopy-jforex-offline-domains-v1",
        "provider": "Dukascopy",
        "interface": "IDataService.getOfflineTimeDomains",
        "instrument_scope": "provider-wide",
        "api_version": "2.12.46",
        "retrieval_details": "JForex strategy export fixture",
        "requested_start_utc": "2019-03-11T00:00:00Z",
        "requested_end_exclusive_utc": "2024-08-21T00:00:00Z",
        "domains": [
            {
                "evidence_id": "new-year",
                "start_utc": "2023-12-31T22:00:00Z",
                "end_exclusive_utc": "2024-01-01T22:00:00Z",
            }
        ],
    }
    path = tmp_path / "offline.json"
    _write_json(path, _with_semantic_hash(payload))

    domains, provenance = reconciliation._load_offline_evidence(path, plan)

    assert domains[0].evidence_id == "new-year"
    assert provenance["file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["domains"][0]["end_exclusive_utc"] = "2024-01-01T21:59:00Z"
    _write_json(path, tampered)
    with pytest.raises(
        reconciliation.ReconciliationValidationError, match="semantic SHA-256"
    ):
        reconciliation._load_offline_evidence(path, plan)


def test_verified_zero_tick_minute_becomes_verified_no_tick() -> None:
    minute = _dt("2024-01-02T12:03:00Z")
    hour = minute.replace(minute=0)

    result = _build(_missing(minute, 1), (), {hour: _verified(hour, ())})

    assert result["counts"]["verified_no_tick_minute_count"] == 1
    assert result["counts"]["unexplained_missing_minute_count"] == 0
    assert result["session_aware_research_ready"] is True


def test_direct_tick_for_canonical_missing_minute_blocks_readiness() -> None:
    minute = _dt("2024-01-02T12:03:00Z")
    hour = minute.replace(minute=0)

    result = _build(_missing(minute, 1), (), {hour: _verified(hour, (minute,))})

    interval = result["unexplained_missing_intervals_utc"][0]
    assert interval["reason"] == "independent_direct_source_contains_tick"
    assert result["counts"]["unexplained_missing_minute_count"] == 1
    assert result["session_aware_research_ready"] is False


def test_partial_offline_overlap_only_classifies_supported_minutes() -> None:
    start = _dt("2024-01-02T12:00:00Z")
    hour = start.replace(minute=0)
    domain = reconciliation.OfflineDomain(
        "partial-domain",
        start + timedelta(minutes=1),
        start + timedelta(minutes=3),
    )

    result = _build(_missing(start, 4), (domain,), {hour: _verified(hour, ())})

    assert result["counts"]["provider_offline_minute_count"] == 2
    assert result["counts"]["verified_no_tick_minute_count"] == 2
    assert result["counts"]["unexplained_missing_minute_count"] == 0
    assert result["provider_offline_intervals_utc"][0]["start_utc"] == (
        "2024-01-02T12:01:00Z"
    )
    assert result["provider_offline_intervals_utc"][0]["end_utc"] == (
        "2024-01-02T12:02:00Z"
    )


def test_canonical_side_mismatch_fails_closed() -> None:
    minute = _dt("2024-01-02T12:03:00Z")
    hour = minute.replace(minute=0)
    missing = (reconciliation._MissingMinute(minute, ("BID",)),)

    result = _build(missing, (), {hour: _verified(hour, ())})

    assert result["unexplained_missing_intervals_utc"][0]["reason"] == (
        "canonical_bid_ask_coverage_mismatch"
    )
    assert result["session_aware_research_ready"] is False


def test_malformed_direct_bid_ask_values_fail_closed(tmp_path: Path) -> None:
    hour = _dt("2024-01-02T12:00:00Z")
    malformed = _bi5_payload(60_000, ask=109_990, bid=110_000)
    acquirer = reconciliation.DirectDukascopyAcquirer(
        tmp_path,
        transport=lambda _: reconciliation.HttpResponse(200, malformed),
        retries=1,
        retry_delay_seconds=0,
    )

    verification = acquirer.acquire(hour, CUTOFF)
    result = _build(_missing(hour + timedelta(minutes=1), 1), (), {hour: verification})

    assert verification.outcome == "retrieval_failed"
    assert result["counts"]["unexplained_missing_minute_count"] == 1
    assert result["session_aware_research_ready"] is False


def test_cache_retry_and_resume_do_not_change_verified_result(tmp_path: Path) -> None:
    hour = _dt("2024-01-02T12:00:00Z")
    payload = _bi5_payload(60_000)
    calls = 0

    def flaky(_: str) -> reconciliation.HttpResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("temporary")
        return reconciliation.HttpResponse(200, payload)

    first = reconciliation.DirectDukascopyAcquirer(
        tmp_path,
        transport=flaky,
        retries=2,
        retry_delay_seconds=0,
    ).acquire(hour, CUTOFF)

    def must_not_fetch(_: str) -> NoReturn:
        raise AssertionError("cache was not resumed")

    resumed = reconciliation.DirectDukascopyAcquirer(
        tmp_path,
        transport=must_not_fetch,
        retries=1,
        retry_delay_seconds=0,
    ).acquire(hour, CUTOFF)

    assert calls == 2
    assert resumed == first
    assert first.tick_minutes == (hour + timedelta(minutes=1),)
    assert first.content_sha256 == hashlib.sha256(payload).hexdigest()


def test_classification_order_does_not_hide_direct_tick_with_offline_domain() -> None:
    minute = _dt("2024-01-02T12:03:00Z")
    hour = minute.replace(minute=0)
    domain = reconciliation.OfflineDomain(
        "conflicting-domain", minute, minute + timedelta(minutes=1)
    )

    result = _build(
        _missing(minute, 1),
        (domain,),
        {hour: _verified(hour, (minute,))},
    )

    assert result["counts"]["provider_offline_minute_count"] == 0
    assert result["unexplained_missing_intervals_utc"][0]["reason"] == (
        "independent_direct_source_contains_tick"
    )


def test_retrieval_failure_with_empty_tick_set_is_not_zero_tick_proof(
    tmp_path: Path,
) -> None:
    hour = _dt("2024-01-02T12:00:00Z")

    def failed(_: str) -> NoReturn:
        raise OSError("offline")

    verification = reconciliation.DirectDukascopyAcquirer(
        tmp_path,
        transport=failed,
        retries=1,
        retry_delay_seconds=0,
    ).acquire(hour, CUTOFF)
    result = _build(_missing(hour, 1), (), {hour: verification})

    assert verification.tick_minutes == ()
    assert verification.outcome == "retrieval_failed"
    assert result["counts"]["verified_no_tick_minute_count"] == 0
    assert result["counts"]["unexplained_missing_minute_count"] == 1


def test_http_404_is_not_zero_tick_proof(tmp_path: Path) -> None:
    hour = _dt("2024-01-02T12:00:00Z")
    verification = reconciliation.DirectDukascopyAcquirer(
        tmp_path,
        transport=lambda _: reconciliation.HttpResponse(404, b""),
        retries=1,
        retry_delay_seconds=0,
    ).acquire(hour, CUTOFF)

    result = _build(_missing(hour, 1), (), {hour: verification})

    assert verification.response_status == 404
    assert result["counts"]["verified_no_tick_minute_count"] == 0
    assert result["session_aware_research_ready"] is False


def test_holdout_boundary_is_strict_and_never_calls_transport(tmp_path: Path) -> None:
    called = False

    def transport(_: str) -> NoReturn:
        nonlocal called
        called = True
        raise AssertionError("holdout network access attempted")

    acquirer = reconciliation.DirectDukascopyAcquirer(
        tmp_path, transport=transport, retries=1, retry_delay_seconds=0
    )

    with pytest.raises(
        reconciliation.ReconciliationValidationError, match="holdout"
    ):
        acquirer.acquire(CUTOFF, CUTOFF)

    assert called is False


def test_first_tick_at_holdout_boundary_cannot_be_admitted() -> None:
    hour = _dt("2024-08-20T23:00:00Z")
    payload = _bi5_payload(3_600_000)

    with pytest.raises(
        reconciliation.ReconciliationValidationError, match="outside its hour"
    ):
        reconciliation._verification_from_payload(
            hour,
            reconciliation._dukascopy_tick_url("EURUSD", hour),
            payload,
            "fixture.bi5",
        )


def test_semantic_hash_is_deterministic_across_result_mapping_order() -> None:
    first_minute = _dt("2024-01-02T12:03:00Z")
    second_minute = _dt("2024-01-02T13:04:00Z")
    first_hour = first_minute.replace(minute=0)
    second_hour = second_minute.replace(minute=0)
    missing = (
        reconciliation._MissingMinute(first_minute, ("BID", "ASK")),
        reconciliation._MissingMinute(second_minute, ("BID", "ASK")),
    )
    forward = {
        first_hour: _verified(first_hour, ()),
        second_hour: _verified(second_hour, ()),
    }
    reverse = dict(reversed(tuple(forward.items())))

    first = _build(missing, (), forward)
    second = _build(missing, (), reverse)

    assert first == second
    assert reconciliation._semantic_sha256(first) == (
        reconciliation._semantic_sha256(second)
    )


def test_live_command_writes_only_reconciliation_artifact(
    tmp_path: Path,
) -> None:
    plan = load_research_data_plan(PLAN)
    root = tmp_path / "canonical"
    root.mkdir()
    provenance = _canonical_provenance(plan)
    provenance_path = root / reconciliation.PARENT_MANIFEST_FILENAME
    _write_json(provenance_path, provenance)
    gap = _dt("2024-01-02T12:03:00Z")
    coverage = _coverage_report(plan, provenance_path, gap)
    _write_json(root / reconciliation.COVERAGE_MANIFEST_FILENAME, coverage)
    catalog_sentinel = root / "catalog" / "source.bin"
    catalog_sentinel.parent.mkdir()
    catalog_sentinel.write_bytes(b"canonical-source-unchanged")
    derived_sentinel = root / "ftmoquant_derived_provenance.json"
    derived_sentinel.write_bytes(b"derived-artifact-unchanged")
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (provenance_path, catalog_sentinel, derived_sentinel)
    }
    hour = gap.replace(minute=0)

    class FakeAcquirer:
        def acquire(
            self, hour_start_utc: datetime, cutoff: datetime
        ) -> reconciliation.HourVerification:
            assert hour_start_utc == hour
            assert cutoff == CUTOFF
            return _verified(hour, ())

    result = reconciliation.run_eurusd_session_reconciliation(
        PLAN,
        root,
        tmp_path / "cache",
        workers=1,
        acquirer=FakeAcquirer(),  # type: ignore[arg-type]
    )

    after = {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before
    }
    assert after == before
    assert result.session_aware_research_ready is True
    assert result.manifest_path.name == (
        reconciliation.RECONCILIATION_MANIFEST_FILENAME
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["fills_or_interpolation"] is False
    assert manifest["canonical_or_derived_data_modified"] is False
    assert manifest["holdout_accessed"] is False


def test_bounded_acquisition_batch_never_writes_partial_artifact(
    tmp_path: Path,
) -> None:
    plan = load_research_data_plan(PLAN)
    root = tmp_path / "canonical"
    root.mkdir()
    provenance_path = root / reconciliation.PARENT_MANIFEST_FILENAME
    _write_json(provenance_path, _canonical_provenance(plan))
    gap = _dt("2024-01-02T12:03:00Z")
    _write_json(
        root / reconciliation.COVERAGE_MANIFEST_FILENAME,
        _coverage_report(plan, provenance_path, gap),
    )
    hour = gap.replace(minute=0)

    class FakeBatchAcquirer(reconciliation.DirectDukascopyAcquirer):
        def has_complete_cache_entry(self, hour_start_utc: datetime) -> bool:
            assert hour_start_utc == hour
            return False

        def acquire(
            self, hour_start_utc: datetime, cutoff: datetime
        ) -> reconciliation.HourVerification:
            assert hour_start_utc == hour
            assert cutoff == CUTOFF
            return _verified(hour, ())

    result = reconciliation.acquire_eurusd_reconciliation_batch(
        PLAN,
        root,
        tmp_path / "cache",
        workers=1,
        max_uncached_hours=1,
        acquirer=FakeBatchAcquirer(tmp_path / "unused"),
    )

    assert result == reconciliation.AcquisitionBatchResult(1, 1, 0, 0)
    assert not (root / reconciliation.RECONCILIATION_MANIFEST_FILENAME).exists()


def _build(
    missing: tuple[reconciliation._MissingMinute, ...],
    domains: tuple[reconciliation.OfflineDomain, ...],
    hours: dict[datetime, reconciliation.HourVerification],
) -> dict[str, object]:
    plan = load_research_data_plan(PLAN)
    provenance = {"qa": {"holdout_rows_admitted": 0}}
    coverage = {
        "source_derived_structural_integrity_valid": True,
        "session_policy": {"version": "dukascopy-eurusd-ny-close-v1"},
        "coverage_qa_version": "g1-data-readiness-1",
        "semantic_sha256": "b" * 64,
        "unexplained_missing_intervals_utc": [object()] * len(missing),
    }
    return reconciliation.build_reconciliation_payload(
        plan=plan,
        plan_file_sha256="a" * 64,
        provenance=provenance,
        provenance_file_sha256="c" * 64,
        coverage=coverage,
        coverage_file_sha256="d" * 64,
        missing_minutes=missing,
        offline_domains=domains,
        offline_provenance={"used": bool(domains)},
        hour_results=hours,
    )


def _missing(
    start: datetime, count: int
) -> tuple[reconciliation._MissingMinute, ...]:
    return tuple(
        reconciliation._MissingMinute(
            start + timedelta(minutes=index), ("BID", "ASK")
        )
        for index in range(count)
    )


def _verified(
    hour: datetime, tick_minutes: tuple[datetime, ...]
) -> reconciliation.HourVerification:
    return reconciliation.HourVerification(
        hour_start_utc=hour,
        source_url=reconciliation._dukascopy_tick_url("EURUSD", hour),
        outcome="verified",
        response_status=200,
        tick_minutes=tick_minutes,
        tick_count=len(tick_minutes),
        content_size=1,
        content_sha256="e" * 64,
        cache_relative_path="fixture.bi5",
        failure_reason=None,
    )


def _bi5_payload(
    milliseconds: int,
    *,
    ask: int = 110_020,
    bid: int = 110_000,
) -> bytes:
    raw = struct.pack(">i i i f f", milliseconds, ask, bid, 1.0, 1.0)
    return lzma.compress(raw)


def _canonical_provenance(plan: ResearchDataPlan) -> dict[str, object]:
    payload: dict[str, object] = {
        "dataset_id": plan.dataset_id,
        "dataset_plan_sha256": plan.semantic_sha256,
        "dataset_repo": plan.huggingface_repo,
        "dataset_revision": plan.huggingface_revision,
        "distribution_source": "Hugging Face",
        "fills_or_interpolation": False,
        "g1_cutoff_end_exclusive_utc": "2024-08-21T00:00:00Z",
        "ingestion_version": "g1-hf-dukascopy-1",
        "instrument_id": "EUR/USD.DUKASCOPY",
        "market_data_origin": "Dukascopy",
        "nautilus_version": "2.0.0rc2",
        "price_sides": ["BID", "ASK"],
        "provider": "Dukascopy",
        "qa": {
            "ask_bar_count": 1,
            "bid_bar_count": 1,
            "gaps_filled": False,
            "holdout_rows_admitted": 0,
        },
        "requested_utc_range": {
            "start_date": "2019-03-11",
            "end_date_inclusive": "2024-08-20",
        },
        "source_files": [
            {
                "repo_path": "data/EURUSD/2019-03-11_2019-04-09.parquet",
                "downloaded_sha256": "f" * 64,
                "downloaded_size": 1,
            }
        ],
        "source_granularity": "1-minute",
        "source_timestamp_convention": (
            "raw_unix_epoch_milliseconds_to_bar_open_utc"
        ),
        "symbol": "EURUSD",
    }
    return _with_semantic_hash(payload)


def _coverage_report(
    plan: ResearchDataPlan, provenance_path: Path, gap: datetime
) -> dict[str, object]:
    start = datetime.combine(plan.development.start, datetime.min.time(), tzinfo=UTC)
    requested_minutes = int((CUTOFF - start) / timedelta(minutes=1))
    payload: dict[str, object] = {
        "coverage_qa_version": "g1-data-readiness-1",
        "session_policy": {"version": "dukascopy-eurusd-ny-close-v1"},
        "parent_g0_5_manifest_sha256": hashlib.sha256(
            provenance_path.read_bytes()
        ).hexdigest(),
        "requested_utc_interval": {
            "start_utc": "2019-03-11T00:00:00Z",
            "end_exclusive_utc": "2024-08-21T00:00:00Z",
        },
        "counts": {
            "requested_minute_count": requested_minutes,
            "expected_open_minute_count": 2,
            "observed_paired_expected_open_minute_count": 1,
            "observed_paired_minute_count": 1,
            "expected_closure_minute_count": requested_minutes - 2,
            "unexplained_missing_minute_count": 1,
        },
        "unexplained_missing_intervals_utc": [
            {
                "classification": "unexplained_missing",
                "start_utc": reconciliation._format_utc(gap),
                "end_utc": reconciliation._format_utc(gap),
                "minutes": 1,
                "missing_sides": ["BID", "ASK"],
            }
        ],
        "source_derived_structural_integrity_valid": True,
        "session_aware_research_ready": False,
    }
    return _with_semantic_hash(payload)


def _with_semantic_hash(payload: dict[str, object]) -> dict[str, object]:
    return {**payload, "semantic_sha256": reconciliation._semantic_sha256(payload)}


def _write_json(path: Path, document: dict[str, object]) -> None:
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
