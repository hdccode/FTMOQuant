from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from nautilus_trader.persistence import ParquetDataCatalog

import ftmoquant.data.canonical_correction as correction
import ftmoquant.data.derived_bars as derived
import ftmoquant.data.hf_dukascopy as hf
from ftmoquant.data.canonical_source import (
    CORRECTED_INGESTION_VERSION,
    validate_canonical_eurusd_source_manifest,
)
from ftmoquant.data.dukascopy import SourceBar, _eurusd_instrument, _to_nautilus_bars


def test_correction_reuses_hf_tick_aggregation_semantics() -> None:
    start = _dt("2023-11-14T13:31:00Z")
    current: hf._MinuteAggregate | None = None
    ticks = (
        (start + timedelta(seconds=1), "1.10000", "1.10020", "2", "3"),
        (start + timedelta(seconds=2), "1.10010", "1.10030", "5", "7"),
        (start + timedelta(seconds=3), "1.09990", "1.10010", "11", "13"),
    )
    for timestamp, bid, ask, bid_volume, ask_volume in ticks:
        current, emitted = hf._advance_minute_aggregate(
            current,
            hf._datetime_ms(timestamp),
            Decimal(bid),
            Decimal(ask),
            Decimal(bid_volume),
            Decimal(ask_volume),
        )
        assert emitted is None
    assert current is not None

    bid, ask = hf._source_bar_pair(current)
    assert (bid.open, bid.high, bid.low, bid.close, bid.volume) == (
        Decimal("1.10000"),
        Decimal("1.10010"),
        Decimal("1.09990"),
        Decimal("1.09990"),
        Decimal("18"),
    )
    assert (ask.open, ask.high, ask.low, ask.close, ask.volume) == (
        Decimal("1.10020"),
        Decimal("1.10030"),
        Decimal("1.10010"),
        Decimal("1.10010"),
        Decimal("23"),
    )
    encoded = _to_nautilus_bars([bid], "BID")[0]
    assert encoded.ts_event == _ns(start)
    assert encoded.ts_init == _ns(start + timedelta(minutes=1))


def test_missing_intervals_remove_exactly_seven_and_nothing_else() -> None:
    intervals: list[object] = [
        {
            "start_utc": "2023-11-14T13:30:00Z",
            "end_utc": "2023-11-14T13:32:00Z",
            "minutes": 3,
        },
        {
            "start_utc": "2023-11-29T14:28:00Z",
            "end_utc": "2023-11-29T14:35:00Z",
            "minutes": 8,
        },
        {
            "start_utc": "2020-01-02T12:03:00Z",
            "end_utc": "2020-01-02T12:03:00Z",
            "minutes": 1,
        },
    ]

    result = correction._subtract_corrected_minutes(intervals)

    assert sum(item["minutes"] for item in result) == 5
    assert {_expand(item) for item in result} == {
        ("2023-11-14T13:30:00Z",),
        ("2023-11-14T13:32:00Z",),
        ("2023-11-29T14:28:00Z",),
        ("2023-11-29T14:35:00Z",),
        ("2020-01-02T12:03:00Z",),
    }


def test_duplicate_already_existing_correction_minute_is_rejected(
    tmp_path: Path,
) -> None:
    minute = correction._EXPECTED_MINUTES[0]
    catalog = _catalog(tmp_path / "catalog", [minute])
    pair = _source_pair(minute)

    with pytest.raises(
        correction.CanonicalCorrectionValidationError, match="already contains"
    ):
        correction._reject_existing_minutes(catalog, {minute: pair})


def test_parent_equivalence_proves_only_seven_paired_additions(
    tmp_path: Path,
) -> None:
    parent_times = [
        _dt("2023-11-14T13:30:00Z"),
        _dt("2023-11-29T14:35:00Z"),
    ]
    parent = _catalog(tmp_path / "parent", parent_times)
    corrected = _catalog(
        tmp_path / "corrected", [*parent_times, *correction._EXPECTED_MINUTES]
    )
    expected_bid = _bars(correction._EXPECTED_MINUTES, "BID")
    expected_ask = _bars(correction._EXPECTED_MINUTES, "ASK")

    proof, hashes = correction._prove_parent_equivalence(
        parent,
        corrected,
        _parent_manifest(len(parent_times)),
        expected_bid,
        expected_ask,
    )

    assert proof["parent_timestamps_removed"] == 0
    assert proof["new_bid_timestamp_count"] == 7
    assert proof["new_ask_timestamp_count"] == 7
    assert proof["other_changed_or_added_bar_count"] == 0
    assert set(hashes) == {"BID", "ASK"}


def test_parent_equivalence_rejects_any_eighth_bar(tmp_path: Path) -> None:
    parent_times = [_dt("2023-11-14T13:30:00Z")]
    extra = _dt("2023-11-29T14:35:00Z")
    parent = _catalog(tmp_path / "parent", parent_times)
    corrected = _catalog(
        tmp_path / "corrected",
        [*parent_times, *correction._EXPECTED_MINUTES, extra],
    )

    with pytest.raises(
        correction.CanonicalCorrectionValidationError, match="unexpected corrected"
    ):
        correction._prove_parent_equivalence(
            parent,
            corrected,
            _parent_manifest(1),
            _bars(correction._EXPECTED_MINUTES, "BID"),
            _bars(correction._EXPECTED_MINUTES, "ASK"),
        )


def test_parent_equivalence_rejects_altered_parent_bar(tmp_path: Path) -> None:
    minute = _dt("2023-11-14T13:30:00Z")
    parent = _catalog(tmp_path / "parent", [minute])
    corrected_path = tmp_path / "corrected"
    corrected = _catalog(corrected_path, correction._EXPECTED_MINUTES)
    corrected.write_bars(
        _to_nautilus_bars(
            [
                SourceBar(
                    minute,
                    Decimal("1.2"),
                    Decimal("1.2"),
                    Decimal("1.2"),
                    Decimal("1.2"),
                    Decimal("1"),
                )
            ],
            "BID",
        )
    )
    corrected.write_bars(_bars([minute], "ASK"))

    with pytest.raises(
        correction.CanonicalCorrectionValidationError, match="changed or disappeared"
    ):
        correction._prove_parent_equivalence(
            parent,
            corrected,
            _parent_manifest(1),
            _bars(correction._EXPECTED_MINUTES, "BID"),
            _bars(correction._EXPECTED_MINUTES, "ASK"),
        )


def test_copy_never_mutates_parent_catalog(tmp_path: Path) -> None:
    parent_path = tmp_path / "parent"
    _catalog(parent_path, [_dt("2023-11-14T13:30:00Z")])
    before = correction._sha256_tree(parent_path)

    correction._copy_canonical_catalog(parent_path, tmp_path / "corrected")

    assert correction._sha256_tree(parent_path) == before


def test_missing_or_corrupt_bi5_evidence_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailedAcquirer:
        def __init__(self, _root: Path) -> None:
            pass

        def acquire(
            self, hour_start_utc: datetime, _cutoff: datetime
        ) -> Any:
            return type(
                "Failed",
                (),
                {
                    "outcome": "retrieval_failed",
                    "tick_count": None,
                },
            )()

    monkeypatch.setattr(correction, "DirectDukascopyAcquirer", FailedAcquirer)

    with pytest.raises(
        correction.CanonicalCorrectionValidationError, match="missing or corrupt"
    ):
        correction._correction_bars(
            _reconciliation_evidence(),
            tmp_path,
            _dt("2024-08-21T00:00:00Z"),
        )


def test_source_hash_or_tick_count_mismatch_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class MismatchedAcquirer:
        def __init__(self, _root: Path) -> None:
            pass

        def acquire(
            self, hour_start_utc: datetime, _cutoff: datetime
        ) -> Any:
            return SimpleNamespace(
                outcome="verified",
                tick_count=999,
                content_sha256="d" * 64,
                content_size=2,
                cache_relative_path="fixture.bi5",
                source_url="https://example.invalid",
            )

    monkeypatch.setattr(
        correction, "DirectDukascopyAcquirer", MismatchedAcquirer
    )

    with pytest.raises(
        correction.CanonicalCorrectionValidationError,
        match="differs from the frozen reconciliation",
    ):
        correction._correction_bars(
            _reconciliation_evidence(),
            tmp_path,
            _dt("2024-08-21T00:00:00Z"),
        )


def test_corrected_bid_ask_integrity_failure_is_rejected() -> None:
    minute = correction._EXPECTED_MINUTES[0]
    bid, ask = _source_pair(minute)
    invalid_ask = SourceBar(
        minute,
        bid.open - Decimal("0.00001"),
        ask.high,
        ask.low,
        ask.close,
        ask.volume,
    )

    with pytest.raises(
        correction.CanonicalCorrectionValidationError, match="ASK open is below BID"
    ):
        correction._validate_source_pair((bid, invalid_ask))


def test_holdout_boundary_is_rejected_including_exact_cutoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cutoff = _dt("2024-08-21T00:00:00Z")
    monkeypatch.setattr(correction, "_EXPECTED_MINUTES", (cutoff,))
    monkeypatch.setattr(
        correction, "_EXPECTED_TICK_COUNTS", {_utc(cutoff): 1}
    )

    with pytest.raises(
        correction.CanonicalCorrectionValidationError, match="holdout boundary"
    ):
        correction._correction_bars(
            {"independent_tick_verification": {"hours": []}},
            tmp_path,
            cutoff,
        )


def test_corrected_derived_window_appears_only_from_complete_source() -> None:
    start = _dt("2023-11-14T13:00:00Z")
    incomplete_times = [
        start + timedelta(minutes=index) for index in range(60) if index != 31
    ]
    complete_times = [start + timedelta(minutes=index) for index in range(60)]
    incomplete, dropped = derived._eligible_source_bars(
        _bars(incomplete_times, "BID"),
        60,
        60,
        range_start_ns=_ns(start),
        range_end_ns=_ns(start + timedelta(hours=1)),
    )
    complete, corrected_dropped = derived._eligible_source_bars(
        _bars(complete_times, "BID"),
        60,
        60,
        range_start_ns=_ns(start),
        range_end_ns=_ns(start + timedelta(hours=1)),
    )

    assert incomplete == ()
    assert dropped[0].observed_minutes == 59
    assert len(complete) == 60
    assert corrected_dropped == ()


def test_final_readiness_requires_zero_unexplained_omissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical = {
        "correction": {
            "semantic_sha256": "c" * 64,
            "holdout_accessed": False,
            "holdout_rows_admitted": 0,
        },
        "parent_equivalence": {
            "parent_bid_bars_unchanged": True,
            "parent_ask_bars_unchanged": True,
            "parent_timestamps_removed": 0,
            "new_bid_timestamp_count": 7,
            "new_ask_timestamp_count": 7,
            "other_changed_or_added_bar_count": 0,
        },
        "canonical_content": {"BID": "b" * 64, "ASK": "a" * 64},
        "qa": {
            "bid_bar_count": 10,
            "holdout_rows_admitted": 0,
        },
        "holdout_accessed": False,
        "semantic_sha256": "d" * 64,
    }
    derived_manifest = {
        "parent_g0_5_manifest_sha256": "1" * 64,
        "derived_bar_integrity_valid": True,
        "bid_ask_derived_coverage_matches": True,
    }
    coverage = {
        "parent_g0_5_manifest_sha256": "1" * 64,
        "structural_g0_6_manifest_sha256": "2" * 64,
        "source_derived_structural_integrity_valid": True,
        "counts": {
            "unexplained_missing_minute_count": 17_728,
            "observed_paired_minute_count": 10,
        },
    }
    reconciliation = {
        "counts": {
            "unexplained_missing_minute_count": 1,
            "verified_no_tick_minute_count": 17_728,
            "provider_offline_minute_count": 0,
        },
        "session_aware_research_ready": False,
        "source_derived_structural_integrity_valid": True,
        "holdout_accessed": False,
        "holdout_rows_admitted": 0,
    }

    def load_semantic(path: Path, _description: str) -> dict[str, Any]:
        by_name = {
            correction.PARENT_MANIFEST_FILENAME: canonical,
            correction.COVERAGE_MANIFEST_FILENAME: coverage,
            correction.RECONCILIATION_MANIFEST_FILENAME: reconciliation,
        }
        return by_name[path.name]

    monkeypatch.setattr(correction, "_load_semantic_json", load_semantic)
    monkeypatch.setattr(
        correction,
        "_load_json",
        lambda _path, _description: derived_manifest,
    )
    monkeypatch.setattr(
        correction,
        "_sha256",
        lambda path: {
            correction.PARENT_MANIFEST_FILENAME: "1" * 64,
            correction.DERIVED_MANIFEST_FILENAME: "2" * 64,
            correction.COVERAGE_MANIFEST_FILENAME: "3" * 64,
            correction.RECONCILIATION_MANIFEST_FILENAME: "4" * 64,
        }[path.name],
    )
    monkeypatch.setattr(
        correction,
        "validate_canonical_eurusd_source_manifest",
        lambda _manifest: SimpleNamespace(
            ingestion_version=CORRECTED_INGESTION_VERSION
        ),
    )
    monkeypatch.setattr(correction, "_validate_parent_plan", lambda *_args: None)

    with pytest.raises(
        correction.CanonicalCorrectionValidationError,
        match="zero_unexplained_omissions",
    ):
        correction.freeze_eurusd_research_readiness(
            Path("config/data/eurusd_research_v1.yaml"), tmp_path
        )


def test_corrected_manifest_hash_and_identity_are_deterministic() -> None:
    parent = _parent_manifest(1)
    parent["semantic_sha256"] = correction._semantic_sha256(parent)
    evidence = (
        {
            "evidence_id": "direct-bi5-20231114T1300Z",
            "content_sha256": "a" * 64,
        },
    )
    equivalence = {
        "parent_bid_bars_unchanged": True,
        "parent_ask_bars_unchanged": True,
        "parent_timestamps_removed": 0,
        "new_bid_timestamp_count": 7,
        "new_ask_timestamp_count": 7,
        "other_changed_or_added_bar_count": 0,
    }
    content = {"BID": "b" * 64, "ASK": "c" * 64}
    kwargs = {
        "parent_manifest": parent,
        "parent_manifest_sha256": "d" * 64,
        "plan_file_sha256": "e" * 64,
        "reconciliation_file_sha256": "f" * 64,
        "reconciliation_semantic_sha256": "1" * 64,
        "evidence": evidence,
        "equivalence": equivalence,
        "content": content,
    }

    first = correction._corrected_manifest_payload(**kwargs)  # type: ignore[arg-type]
    second = correction._corrected_manifest_payload(**kwargs)  # type: ignore[arg-type]
    first_manifest = {**first, "semantic_sha256": correction._semantic_sha256(first)}

    assert first == second
    assert correction._semantic_sha256(first) == correction._semantic_sha256(second)
    assert validate_canonical_eurusd_source_manifest(
        first_manifest
    ).ingestion_version == CORRECTED_INGESTION_VERSION


def test_generated_market_data_and_cache_patterns_are_git_ignored() -> None:
    ignored = Path(".gitignore").read_text(encoding="utf-8")

    assert "/catalog/" in ignored
    assert "/market-data/" in ignored
    assert "/data/" in ignored


def _catalog(
    path: Path, timestamps: list[datetime] | tuple[datetime, ...]
) -> ParquetDataCatalog:
    path.mkdir(parents=True)
    catalog = ParquetDataCatalog(str(path))
    catalog.write_instruments([_eurusd_instrument()])
    ordered = sorted(timestamps)
    catalog.write_bars(_bars(ordered, "BID"))
    catalog.write_bars(_bars(ordered, "ASK"))
    return catalog


def _bars(timestamps: Any, side: str) -> list[Any]:
    spread = Decimal("0.00020") if side == "ASK" else Decimal(0)
    return _to_nautilus_bars(
        [
            SourceBar(
                timestamp,
                Decimal("1.10000") + spread,
                Decimal("1.10010") + spread,
                Decimal("1.09990") + spread,
                Decimal("1.10005") + spread,
                Decimal("1"),
            )
            for timestamp in timestamps
        ],
        side,
    )


def _source_pair(timestamp: datetime) -> tuple[SourceBar, SourceBar]:
    bars = []
    for side in ("BID", "ASK"):
        spread = Decimal("0.00020") if side == "ASK" else Decimal(0)
        bars.append(
            SourceBar(
                timestamp,
                Decimal("1.10000") + spread,
                Decimal("1.10010") + spread,
                Decimal("1.09990") + spread,
                Decimal("1.10005") + spread,
                Decimal("1"),
            )
        )
    return bars[0], bars[1]


def _parent_manifest(count: int) -> dict[str, Any]:
    missing = [
        {
            "start_utc": timestamp,
            "end_utc": timestamp,
            "minutes": 1,
        }
        for timestamp in correction._EXPECTED_TICK_COUNTS
    ]
    return {
        "aggregation": {},
        "dataset_id": "eurusd_research_v1",
        "dataset_plan_sha256": "a" * 64,
        "dataset_repo": "mito0o852/dukascopy-ticks",
        "dataset_revision": "b" * 40,
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
            "admitted_tick_count": count,
            "ask_bar_count": count,
            "bid_bar_count": count,
            "gaps_filled": False,
            "holdout_rows_admitted": 0,
            "missing_interval_count": 7,
            "missing_intervals_utc": missing,
        },
        "requested_utc_range": {
            "start_date": "2023-11-14",
            "end_date_inclusive": "2023-11-29",
        },
        "source_files": [
            {
                "repo_path": "data/EURUSD/fixture.parquet",
                "downloaded_sha256": "c" * 64,
                "downloaded_size": 1,
            }
        ],
        "source_granularity": "1-minute",
        "source_timestamp_convention": (
            "raw_unix_epoch_milliseconds_to_bar_open_utc"
        ),
        "symbol": "EURUSD",
    }


def _reconciliation_evidence() -> dict[str, Any]:
    hours = []
    target_hours = {value.replace(minute=0) for value in correction._EXPECTED_MINUTES}
    for hour in sorted(target_hours):
        hours.append(
            {
                "hour_start_utc": _utc(hour),
                "content_sha256": "a" * 64,
                "content_size": 1,
                "tick_count": 1,
                "cache_relative_path": "fixture.bi5",
            }
        )
    return {"independent_tick_verification": {"hours": hours}}


def _expand(item: dict[str, Any]) -> tuple[str, ...]:
    start = _dt(item["start_utc"])
    return tuple(
        _utc(start + timedelta(minutes=index)) for index in range(item["minutes"])
    )


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _utc(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _ns(value: datetime) -> int:
    return int(value.timestamp() * 1_000_000_000)
