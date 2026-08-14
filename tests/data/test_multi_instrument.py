import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from nautilus_trader.persistence import ParquetDataCatalog

from ftmoquant.data import hf_dukascopy as hf
from ftmoquant.data import universe_readiness
from ftmoquant.data.canonical_source import (
    iter_paired_source_chunks,
    validate_canonical_source_manifest,
)
from ftmoquant.data.derived_bars import derive_instrument_bars
from ftmoquant.data.instruments import (
    GBPUSD_SPEC,
    InstrumentSpec,
    InstrumentSpecValidationError,
)
from ftmoquant.data.research_plan import load_research_data_plan
from ftmoquant.data.session_coverage import run_instrument_session_coverage
from ftmoquant.data.session_reconciliation import (
    DirectDukascopyAcquirer,
    HourVerification,
)
from ftmoquant.data.universe_plan import (
    ResearchUniversePlanValidationError,
    load_research_universe_plan,
)
from ftmoquant.data.universe_readiness import (
    INSTRUMENT_READINESS_FILENAME,
    UniverseReadinessValidationError,
    freeze_universe_readiness,
)

PLAN = Path("config/data/g1_4_fx_usd_liquid_v1.yaml")
LEGACY_PLAN = Path("config/data/eurusd_research_v1.yaml")
REVISION = "bf19dbd89c732f010e20db7c148922ba02b2e33b"
DAY = datetime(2019, 3, 11, tzinfo=UTC)


class FakeApi:
    def __init__(self, files: list[str]) -> None:
        self.files = files

    def dataset_info(self, *_args: object, **_kwargs: object) -> Any:
        siblings = [
            SimpleNamespace(rfilename=item, lfs=None, size=None) for item in self.files
        ]
        return SimpleNamespace(sha=REVISION, siblings=siblings)

    def list_repo_files(self, *_args: object, **_kwargs: object) -> list[str]:
        return self.files


def test_universe_plan_is_ordered_strict_and_legacy_hash_is_unchanged() -> None:
    plan = load_research_universe_plan(PLAN)

    assert plan.universe_id == "g1_4_fx_usd_liquid_v1"
    assert tuple(item.instrument_id for item in plan.instruments) == (
        "EUR/USD.DUKASCOPY",
        "GBP/USD.DUKASCOPY",
    )
    assert plan.exact_currency_set == ("EUR", "GBP", "USD")
    assert plan.permitted_end_exclusive_utc == datetime(2024, 8, 21, tzinfo=UTC)
    assert load_research_data_plan(LEGACY_PLAN).semantic_sha256 == (
        "3f45f6b7b896f1bf7922c9c743051857847d156d2c4fad6a35aa8307c9a0e365"
    )


@pytest.mark.parametrize("mutation", ["duplicate_id", "wrong_precision", "cutoff"])
def test_universe_plan_fails_closed(tmp_path: Path, mutation: str) -> None:
    document = yaml.safe_load(PLAN.read_text(encoding="utf-8"))
    if mutation == "duplicate_id":
        document["instruments"][1] = dict(document["instruments"][0])
    elif mutation == "wrong_precision":
        document["instruments"][1]["price_increment"] = "0.001"
    else:
        document["permitted_range"]["end_exclusive_utc"] = "2024-08-22T00:00:00Z"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(ResearchUniversePlanValidationError):
        load_research_universe_plan(path)


def test_instrument_spec_owns_bar_types_precision_and_native_identity() -> None:
    assert (
        GBPUSD_SPEC.bar_type(minutes=1, side="bid", aggregation="external")
        == "GBP/USD.DUKASCOPY-1-MINUTE-BID-EXTERNAL"
    )
    assert (
        GBPUSD_SPEC.composite_bar_type(hours=4, side="ASK")
        == "GBP/USD.DUKASCOPY-4-HOUR-ASK-INTERNAL@1-MINUTE-EXTERNAL"
    )
    assert str(GBPUSD_SPEC.nautilus_instrument().id) == "GBP/USD.DUKASCOPY"

    invalid_jpy = InstrumentSpec(
        dataset_symbol="USDJPY",
        instrument_id="USD/JPY.DUKASCOPY",
        base_currency="USD",
        quote_currency="JPY",
        price_precision=3,
        price_increment="0.00001",
        size_precision=8,
        size_increment="0.00000001",
        session_policy_id="dukascopy-fx-ny-close-v1",
    )
    with pytest.raises(InstrumentSpecValidationError, match="price_increment"):
        invalid_jpy.validate()


def test_holdout_straddling_shard_is_excluded_for_gbp() -> None:
    files = [
        "data/GBPUSD/2024-07-17_2024-08-15.parquet",
        "data/GBPUSD/2024-08-16_2024-09-14.parquet",
    ]
    selected = hf.discover_instrument_source_plan(
        GBPUSD_SPEC,
        files,
        REVISION,
        datetime(2024, 8, 1, tzinfo=UTC),
        datetime(2024, 8, 21, tzinfo=UTC),
        reject_straddling_end=True,
    )

    assert [item.repo_path for item in selected] == files[:1]


def test_generic_acquisition_is_blocked_without_rights_evidence(
    tmp_path: Path,
) -> None:
    with pytest.raises(hf.HfDukascopyValidationError, match="data-use-rights"):
        hf.ingest_hf_instrument(
            PLAN,
            GBPUSD_SPEC.instrument_id,
            tmp_path / "blocked",
            requested_start=DAY,
            requested_end_exclusive=DAY + timedelta(days=1),
            api=FakeApi([]),
        )


def test_synthetic_gbpusd_ingest_derive_and_coverage(tmp_path: Path) -> None:
    source_path = tmp_path / "gbp.parquet"
    timestamps = [
        int((DAY + timedelta(minutes=index)).timestamp() * 1_000)
        for index in range(24 * 60)
    ]
    table = pa.table(
        {
            "timestamp": pa.array(timestamps, type=pa.int64()),
            "askPrice": pa.array([1.2502] * len(timestamps), type=pa.float64()),
            "bidPrice": pa.array([1.25] * len(timestamps), type=pa.float64()),
            "askVolume": pa.array([2.0] * len(timestamps), type=pa.float64()),
            "bidVolume": pa.array([3.0] * len(timestamps), type=pa.float64()),
        }
    )
    pq.write_table(table, source_path)
    repo_path = "data/GBPUSD/2019-03-11_2019-04-09.parquet"
    root = tmp_path / "canonical"

    result = hf.ingest_hf_instrument(
        PLAN,
        GBPUSD_SPEC.instrument_id,
        root,
        requested_start=DAY,
        requested_end_exclusive=DAY + timedelta(days=1),
        data_use_rights_evidence_sha256="f" * 64,
        api=FakeApi([repo_path]),
        downloader=lambda **_kwargs: str(source_path),
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    validate_canonical_source_manifest(manifest, GBPUSD_SPEC)
    catalog = ParquetDataCatalog(str(result.catalog_path))
    chunks = tuple(
        iter_paired_source_chunks(
            catalog,
            manifest,
            GBPUSD_SPEC,
            int(DAY.timestamp() * 1_000_000_000),
            int((DAY + timedelta(days=1)).timestamp() * 1_000_000_000),
            chunk_minutes=500,
        )
    )
    assert sum(len(item.bid_bars) for item in chunks) == 1440
    assert max(bar.ts_event for item in chunks for bar in item.bid_bars) < int(
        (DAY + timedelta(days=1)).timestamp() * 1_000_000_000
    )

    derived = derive_instrument_bars(root, GBPUSD_SPEC)
    coverage = run_instrument_session_coverage(root, GBPUSD_SPEC)
    assert derived.emitted_counts["GBP/USD.DUKASCOPY-1-HOUR-BID-INTERNAL"] == 24
    assert coverage.session_aware_research_ready is True


def test_direct_cache_and_url_are_symbol_specific(tmp_path: Path) -> None:
    acquirer = DirectDukascopyAcquirer(tmp_path, symbol="GBPUSD")
    raw, _ = acquirer._cache_paths(datetime(2024, 8, 16, tzinfo=UTC))
    assert raw.relative_to(tmp_path).parts[0] == "GBPUSD"


def test_direct_cutoff_segment_keeps_verified_zero_tick_hours_empty(
    tmp_path: Path,
) -> None:
    relative = "GBPUSD/2019/02/11/00h_ticks.bi5"
    payload = tmp_path / relative
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"")

    class EmptyAcquirer:
        symbol = "GBPUSD"

        def acquire(self, hour: datetime, _cutoff: datetime) -> HourVerification:
            return HourVerification(
                hour_start_utc=hour,
                source_url=(
                    "https://datafeed.dukascopy.com/datafeed/GBPUSD/"
                    "2019/02/11/00h_ticks.bi5"
                ),
                outcome="verified",
                response_status=200,
                tick_minutes=(),
                tick_count=0,
                content_size=0,
                content_sha256=hashlib.sha256(b"").hexdigest(),
                cache_relative_path=relative,
                failure_reason=None,
            )

    segment = hf.acquire_direct_cutoff_segment(
        GBPUSD_SPEC,
        DAY,
        DAY + timedelta(hours=1),
        tmp_path,
        workers=1,
        acquirer=EmptyAcquirer(),  # type: ignore[arg-type]
    )

    assert segment.bid_bars == ()
    assert segment.ask_bars == ()
    assert len(segment.evidence_sha256) == 64


def test_direct_tail_scheduler_skips_closed_weekend_hours() -> None:
    start = datetime(2024, 8, 16, tzinfo=UTC)
    end = datetime(2024, 8, 19, tzinfo=UTC)

    hours = hf._direct_expected_open_hours(GBPUSD_SPEC, start, end)

    assert hours[0] == start
    assert hours[-1] == datetime(2024, 8, 18, 23, tzinfo=UTC)
    assert datetime(2024, 8, 17, 8, tzinfo=UTC) not in hours
    assert datetime(2024, 8, 18, 20, tzinfo=UTC) not in hours
    assert datetime(2024, 8, 18, 21, tzinfo=UTC) in hours


def test_generic_correction_accepts_only_direct_tick_evidence() -> None:
    reconciliation = {
        "unexplained_missing_intervals_utc": [
            {
                "start_utc": "2024-08-19T10:00:00Z",
                "end_utc": "2024-08-19T10:01:00Z",
                "missing_sides": ["BID", "ASK"],
                "reason": "independent_direct_source_contains_tick",
                "evidence_refs": ["bi5-hour"],
            },
            {
                "start_utc": "2024-08-19T11:00:00Z",
                "end_utc": "2024-08-19T11:00:00Z",
                "missing_sides": ["BID", "ASK"],
                "reason": "transport_failure",
                "evidence_refs": [],
            },
        ]
    }

    assert universe_readiness._proven_correction_minutes(reconciliation) == {
        datetime(2024, 8, 19, 10, 0, tzinfo=UTC),
        datetime(2024, 8, 19, 10, 1, tzinfo=UTC),
    }


def test_universe_freeze_is_ordered_path_independent_and_fail_closed(
    tmp_path: Path,
) -> None:
    plan = load_research_universe_plan(PLAN)
    roots: dict[str, Path] = {}
    for instrument in plan.instruments:
        root = tmp_path / instrument.dataset_symbol
        root.mkdir()
        document = _readiness_document(plan.semantic_sha256, instrument.instrument_id)
        _write_semantic(root / INSTRUMENT_READINESS_FILENAME, document)
        roots[instrument.instrument_id] = root

    result = freeze_universe_readiness(PLAN, roots, tmp_path / "universe")
    manifest_text = result.manifest_path.read_text(encoding="utf-8")
    assert [item.instrument_id for item in result.ordered_artifacts] == [
        "EUR/USD.DUKASCOPY",
        "GBP/USD.DUKASCOPY",
    ]
    assert "not_eligible_insufficient_universe" in manifest_text
    assert str(tmp_path) not in manifest_text

    with pytest.raises(UniverseReadinessValidationError, match="missing"):
        freeze_universe_readiness(
            PLAN,
            {"EUR/USD.DUKASCOPY": roots["EUR/USD.DUKASCOPY"]},
            tmp_path / "failed",
        )


def _readiness_document(plan_sha: str, instrument_id: str) -> dict[str, Any]:
    artifacts = {
        name: {"file_sha256": character * 64, "semantic_sha256": character * 64}
        for name, character in (
            ("canonical", "a"),
            ("derived", "b"),
            ("coverage", "c"),
            ("reconciliation", "d"),
        )
    }
    return {
        "readiness_version": "g1.4a-instrument-readiness-1",
        "universe_plan_sha256": plan_sha,
        "instrument_id": instrument_id,
        "permitted_utc_interval": {
            "start_utc": "2019-03-11T00:00:00Z",
            "end_exclusive_utc": "2024-08-21T00:00:00Z",
        },
        "artifacts": artifacts,
        "catalog_tree_sha256": "e" * 64,
        "split_views": {
            "development": {
                "semantic_sha256": "f" * 64,
                "catalog_tree_sha256": "1" * 64,
            },
            "validation": {
                "semantic_sha256": "2" * 64,
                "catalog_tree_sha256": "3" * 64,
            },
        },
        "research_ready": True,
    }


def _write_semantic(path: Path, payload: dict[str, Any]) -> None:
    semantic = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path.write_text(
        json.dumps({**payload, "semantic_sha256": semantic}, sort_keys=True),
        encoding="utf-8",
    )
