import hashlib
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from nautilus_trader.model import Bar
from nautilus_trader.persistence import ParquetDataCatalog

import ftmoquant.data.session_coverage as coverage
from ftmoquant.data.dukascopy import SourceBar, _eurusd_instrument, _to_nautilus_bars


def test_ordinary_weekday_minutes_are_expected_open() -> None:
    start = _dt("2024-01-09T12:00:00Z")
    observed = _minutes(start, 3)

    result = coverage.assess_eurusd_source_coverage(
        start, start + timedelta(minutes=3), observed, observed
    )

    assert result.expected_open_minute_count == 3
    assert result.observed_paired_minute_count == 3
    assert result.expected_closure_minute_count == 0
    assert result.unexplained_missing_minute_count == 0


@pytest.mark.parametrize(
    ("timestamp", "expected"),
    [
        ("2024-07-05T20:59:00Z", True),
        ("2024-07-05T21:00:00Z", False),
        ("2024-07-07T20:59:00Z", False),
        ("2024-07-07T21:00:00Z", True),
        ("2024-01-05T21:59:00Z", True),
        ("2024-01-05T22:00:00Z", False),
        ("2024-01-07T21:59:00Z", False),
        ("2024-01-07T22:00:00Z", True),
    ],
)
def test_summer_and_winter_friday_sunday_boundaries(
    timestamp: str, expected: bool
) -> None:
    assert coverage.is_eurusd_expected_open(_dt(timestamp)) is expected


def test_spring_dst_weekend_is_forty_seven_hours() -> None:
    start = _dt("2024-03-08T22:00:00Z")
    end = _dt("2024-03-10T21:00:00Z")

    result = coverage.assess_eurusd_source_coverage(start, end, (), ())

    assert result.expected_closure_minute_count == 47 * 60
    assert result.unexplained_missing_minute_count == 0
    assert result.expected_closure_intervals[0].start_utc == ("2024-03-08T22:00:00Z")
    assert result.expected_closure_intervals[0].end_utc == "2024-03-10T20:59:00Z"


def test_autumn_dst_weekend_is_forty_nine_hours() -> None:
    start = _dt("2024-11-01T21:00:00Z")
    end = _dt("2024-11-03T22:00:00Z")

    result = coverage.assess_eurusd_source_coverage(start, end, (), ())

    assert result.expected_closure_minute_count == 49 * 60
    assert result.unexplained_missing_minute_count == 0
    assert result.expected_closure_intervals[0].minutes == 49 * 60


def test_full_summer_weekend_is_one_exact_expected_closure() -> None:
    start = _dt("2024-07-05T21:00:00Z")
    end = _dt("2024-07-07T21:00:00Z")

    result = coverage.assess_eurusd_source_coverage(start, end, (), ())

    assert result.expected_open_minute_count == 0
    assert result.expected_closure_minute_count == 48 * 60
    assert result.unexplained_missing_intervals == ()
    assert result.expected_closure_intervals == (
        coverage.CoverageInterval(
            classification="expected_market_closed",
            start_utc="2024-07-05T21:00:00Z",
            end_utc="2024-07-07T20:59:00Z",
            minutes=48 * 60,
            missing_sides=("BID", "ASK"),
        ),
    )


def test_one_missing_open_minute_is_unexplained() -> None:
    start = _dt("2024-01-09T12:00:00Z")
    observed = (start, start + timedelta(minutes=2))

    result = coverage.assess_eurusd_source_coverage(
        start, start + timedelta(minutes=3), observed, observed
    )

    assert result.unexplained_missing_minute_count == 1
    assert result.unexplained_missing_intervals == (
        coverage.CoverageInterval(
            classification="unexplained_missing",
            start_utc="2024-01-09T12:01:00Z",
            end_utc="2024-01-09T12:01:00Z",
            minutes=1,
            missing_sides=("BID", "ASK"),
        ),
    )


@pytest.mark.parametrize(
    ("bid_missing", "expected_missing_side"),
    [(True, "BID"), (False, "ASK")],
)
def test_one_missing_bid_or_ask_side_is_unexplained(
    bid_missing: bool, expected_missing_side: str
) -> None:
    start = _dt("2024-01-09T12:00:00Z")
    complete = _minutes(start, 3)
    incomplete = (complete[0], complete[2])

    result = coverage.assess_eurusd_source_coverage(
        start,
        start + timedelta(minutes=3),
        incomplete if bid_missing else complete,
        complete if bid_missing else incomplete,
    )

    assert result.unexplained_missing_minute_count == 1
    assert result.unexplained_missing_intervals[0].missing_sides == (
        expected_missing_side,
    )


def test_legitimate_closure_is_never_unexplained() -> None:
    start = _dt("2024-01-05T22:00:00Z")

    result = coverage.assess_eurusd_source_coverage(
        start, start + timedelta(minutes=2), (), ()
    )

    assert result.expected_closure_minute_count == 2
    assert result.unexplained_missing_minute_count == 0


def test_weekend_closure_does_not_block_otherwise_valid_readiness() -> None:
    start = _dt("2024-07-05T20:59:00Z")
    end = _dt("2024-07-07T21:01:00Z")
    observed = (start, end - timedelta(minutes=1))
    assessment = coverage.assess_eurusd_source_coverage(start, end, observed, observed)

    manifest = coverage._coverage_manifest(
        assessment=assessment,
        parent_sha256="a" * 64,
        derived_sha256="b" * 64,
        structural_valid=True,
    )

    assert assessment.expected_closure_minute_count == 48 * 60
    assert assessment.unexplained_missing_minute_count == 0
    assert manifest["session_aware_research_ready"] is True


def test_interval_before_authoritative_policy_fails_closed() -> None:
    with pytest.raises(coverage.CoverageValidationError, match="predates"):
        coverage.assess_eurusd_source_coverage(
            _dt("2018-01-02T00:00:00Z"),
            _dt("2018-01-02T00:01:00Z"),
            (),
            (),
        )


def test_manifest_and_semantic_hash_are_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _complete_weekday_root(tmp_path)
    catalog = ParquetDataCatalog(str(root / "catalog"))
    reference = coverage.assess_eurusd_source_coverage(
        _dt("2024-01-09T00:00:00Z"),
        _dt("2024-01-10T00:00:00Z"),
        tuple(
            coverage._bar_timestamp(bar)
            for bar in catalog.query_bars([coverage._source_bar_type("BID")])
        ),
        tuple(
            coverage._bar_timestamp(bar)
            for bar in catalog.query_bars([coverage._source_bar_type("ASK")])
        ),
    )
    parent_path = root / coverage.PARENT_MANIFEST_FILENAME
    derived_path = root / coverage.DERIVED_MANIFEST_FILENAME
    expected_payload = coverage._coverage_manifest(
        assessment=reference,
        parent_sha256=hashlib.sha256(parent_path.read_bytes()).hexdigest(),
        derived_sha256=hashlib.sha256(derived_path.read_bytes()).hexdigest(),
        structural_valid=True,
    )
    expected_manifest = {
        **expected_payload,
        "semantic_sha256": coverage._semantic_sha256(expected_payload),
    }
    monkeypatch.setattr(coverage, "_SOURCE_QUERY_CHUNK_MINUTES", 37)

    first = coverage.run_eurusd_session_coverage_qa(root)
    original = first.manifest_path.read_bytes()
    repeated = coverage.run_eurusd_session_coverage_qa(root)
    manifest = json.loads(original)
    payload = {
        key: value for key, value in manifest.items() if key != "semantic_sha256"
    }

    assert repeated.manifest_path.read_bytes() == original
    assert first.semantic_sha256 == repeated.semantic_sha256
    assert (
        first.semantic_sha256
        == hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    assert first.session_aware_research_ready is True
    assert manifest == expected_manifest
    assert manifest["provider"] == "Dukascopy"
    assert manifest["instrument"] == "EUR/USD"
    assert manifest["session_policy"]["version"] == ("dukascopy-eurusd-ny-close-v1")
    assert manifest["counts"] == {
        "expected_closure_minute_count": 0,
        "expected_open_minute_count": 1440,
        "observed_paired_expected_open_minute_count": 1440,
        "observed_paired_minute_count": 1440,
        "requested_minute_count": 1440,
        "unexplained_missing_minute_count": 0,
    }
    assert "timestamp" not in manifest


def test_weekend_classification_is_identical_across_chunk_boundaries() -> None:
    start = _dt("2024-07-05T20:59:00Z")
    end = _dt("2024-07-07T21:01:00Z")
    observed = (start, end - timedelta(minutes=1))
    reference = coverage.assess_eurusd_source_coverage(start, end, observed, observed)
    accumulator = coverage._CoverageAccumulator(start=start, end=end, next_minute=start)
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(minutes=37), end)
        chunk_observed = tuple(
            timestamp for timestamp in observed if chunk_start <= timestamp < chunk_end
        )
        coverage._process_coverage_chunk(
            accumulator,
            chunk_start,
            chunk_end,
            chunk_observed,
            chunk_observed,
        )
        chunk_start = chunk_end

    streamed = coverage._finish_coverage(accumulator)

    assert streamed == reference


def test_structural_failure_prevents_session_aware_readiness() -> None:
    start = _dt("2024-01-09T12:00:00Z")
    observed = _minutes(start, 2)
    assessment = coverage.assess_eurusd_source_coverage(
        start, start + timedelta(minutes=2), observed, observed
    )

    manifest = coverage._coverage_manifest(
        assessment=assessment,
        parent_sha256="a" * 64,
        derived_sha256="b" * 64,
        structural_valid=False,
    )

    assert manifest["session_aware_research_ready"] is False


def test_streaming_catalog_bid_ask_mismatch_fails_closed(tmp_path: Path) -> None:
    start = _dt("2024-01-09T00:00:00Z")
    bids = _minutes(start, 60)
    asks = bids[:-1]
    catalog_path = tmp_path / "catalog"
    catalog_path.mkdir(parents=True)
    catalog = ParquetDataCatalog(str(catalog_path))
    catalog.write_instruments([_eurusd_instrument()])
    catalog.write_bars(_bars(bids, "BID"))
    catalog.write_bars(_bars(asks, "ASK"))
    parent = {
        "provider": "Dukascopy",
        "symbol": "EURUSD",
        "instrument_id": coverage.INSTRUMENT_ID,
        "requested_utc_range": {
            "start_date": "2024-01-09",
            "end_date_inclusive": "2024-01-09",
        },
        "source_granularity": "1-minute",
        "price_sides": ["BID", "ASK"],
        "ingestion_version": "g0.5-1",
        "nautilus_version": "2.0.0rc2",
        "qa": {"bid_bar_count": len(bids), "ask_bar_count": len(asks)},
    }
    (tmp_path / coverage.PARENT_MANIFEST_FILENAME).write_text(
        json.dumps(parent, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(coverage.CoverageValidationError, match="coverage mismatch"):
        coverage.run_eurusd_session_coverage_qa(tmp_path)


def _complete_weekday_root(root: Path) -> Path:
    start = _dt("2024-01-09T00:00:00Z")
    timestamps = _minutes(start, 24 * 60)
    catalog_path = root / "catalog"
    catalog_path.mkdir(parents=True)
    catalog = ParquetDataCatalog(str(catalog_path))
    catalog.write_instruments([_eurusd_instrument()])
    catalog.write_bars(_bars(timestamps, "BID"))
    catalog.write_bars(_bars(timestamps, "ASK"))

    parent = {
        "provider": "Dukascopy",
        "symbol": "EURUSD",
        "instrument_id": coverage.INSTRUMENT_ID,
        "requested_utc_range": {
            "start_date": "2024-01-09",
            "end_date_inclusive": "2024-01-09",
        },
        "source_granularity": "1-minute",
        "price_sides": ["BID", "ASK"],
        "ingestion_version": "g0.5-1",
        "nautilus_version": "2.0.0rc2",
        "qa": {"bid_bar_count": 1440, "ask_bar_count": 1440},
    }
    parent_path = root / coverage.PARENT_MANIFEST_FILENAME
    parent_path.write_text(
        json.dumps(parent, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    parent_sha256 = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    derived = {
        "derivation_version": "g0.6-1",
        "parent_g0_5_manifest": coverage.PARENT_MANIFEST_FILENAME,
        "parent_g0_5_manifest_sha256": parent_sha256,
        "parent_ingestion_version": "g0.5-1",
        "nautilus_version": "2.0.0rc2",
        "instrument_id": coverage.INSTRUMENT_ID,
        "counts": {
            "source_1_minute": {"BID": 1440, "ASK": 1440},
            "emitted": {
                "1H": {"BID": 24, "ASK": 24},
                "4H": {"BID": 6, "ASK": 6},
            },
        },
        "derived_bar_integrity_valid": True,
        "bid_ask_derived_coverage_matches": True,
        "coverage_status": "complete",
        "research_ready": True,
    }
    (root / coverage.DERIVED_MANIFEST_FILENAME).write_text(
        json.dumps(derived, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


def _bars(timestamps: tuple[datetime, ...], side: str) -> list[Bar]:
    spread = Decimal("0.00020") if side == "ASK" else Decimal(0)
    rows = [
        SourceBar(
            timestamp=timestamp,
            open=Decimal("1.10000") + spread,
            high=Decimal("1.10010") + spread,
            low=Decimal("1.09990") + spread,
            close=Decimal("1.10005") + spread,
            volume=Decimal(1),
        )
        for timestamp in timestamps
    ]
    return _to_nautilus_bars(rows, side)


def _minutes(start: datetime, count: int) -> tuple[datetime, ...]:
    return tuple(start + timedelta(minutes=index) for index in range(count))


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
