import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from nautilus_trader.persistence import ParquetDataCatalog

import ftmoquant.data.hf_dukascopy as hf
from ftmoquant.data.derived_bars import derive_eurusd_bars
from ftmoquant.data.research_plan import (
    ResearchPlanValidationError,
    load_research_data_plan,
)
from ftmoquant.data.session_coverage import run_eurusd_session_coverage_qa

PLAN = Path("config/data/eurusd_research_v1.yaml")
REVISION = "bf19dbd89c732f010e20db7c148922ba02b2e33b"
DAY = datetime(2019, 3, 11, tzinfo=UTC)
PATH = "data/EURUSD/2019-03-11_2019-04-09.parquet"


class FakeApi:
    def __init__(self, files: list[str], revision: str = REVISION) -> None:
        self.files = files
        self.revision = revision
        self.called = False

    def dataset_info(self, *_args: object, **_kwargs: object) -> Any:
        self.called = True
        return SimpleNamespace(sha=self.revision, siblings=[])

    def list_repo_files(self, *_args: object, **_kwargs: object) -> list[str]:
        self.called = True
        return self.files


def test_frozen_dataset_plan_dates_splits_and_hash() -> None:
    plan = load_research_data_plan(PLAN)

    assert (plan.full_start, plan.full_end) == (
        date(2019, 3, 11),
        date(2025, 12, 31),
    )
    assert (plan.development.days, plan.validation.days, plan.final_holdout.days) == (
        1492,
        498,
        498,
    )
    assert plan.development.end == date(2023, 4, 10)
    assert plan.validation.start == date(2023, 4, 11)
    assert plan.validation.end == date(2024, 8, 20)
    assert plan.final_holdout.start == date(2024, 8, 21)
    assert plan.huggingface_revision == REVISION
    assert plan.semantic_sha256 == (
        "3f45f6b7b896f1bf7922c9c743051857847d156d2c4fad6a35aa8307c9a0e365"
    )
    assert load_research_data_plan(PLAN).semantic_sha256 == plan.semantic_sha256


def test_invalid_dataset_split_is_rejected(tmp_path: Path) -> None:
    document = yaml.safe_load(PLAN.read_text(encoding="utf-8"))
    document["split"]["validation"]["days"] = 499
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ResearchPlanValidationError, match="split is not frozen"):
        load_research_data_plan(invalid)


def test_inventory_selects_only_eurusd_in_date_order() -> None:
    paths = [
        "README.md",
        "data/USDJPY/2019-03-11_2019-04-09.parquet",
        "data/EURUSD/2019-04-10_2019-05-09.parquet",
        PATH,
    ]

    selected = hf.discover_source_file_plan(
        paths, REVISION, DAY, DAY + timedelta(days=31)
    )

    assert [item.repo_path for item in selected] == paths[3:] + paths[2:3]
    assert all(item.repo_path.startswith("data/EURUSD/") for item in selected)


@pytest.mark.parametrize(
    ("paths", "message"),
    [
        (["data/EURUSD/not-a-range.parquet"], "malformed"),
        (
            [
                PATH,
                "data/EURUSD/2019-04-09_2019-05-08.parquet",
            ],
            "overlapping",
        ),
        (
            [
                PATH,
                "data/EURUSD/2019-04-11_2019-05-10.parquet",
            ],
            "date gap",
        ),
        ([PATH, PATH], "duplicate"),
    ],
)
def test_invalid_inventory_is_rejected(paths: list[str], message: str) -> None:
    with pytest.raises(hf.HfDukascopyValidationError, match=message):
        hf.discover_source_file_plan(paths, REVISION, DAY, DAY + timedelta(days=1))


def test_inventory_requires_immutable_revision() -> None:
    with pytest.raises(hf.HfDukascopyValidationError, match="revision"):
        hf.discover_source_file_plan([PATH], "main", DAY, DAY + timedelta(days=1))


def test_cutoff_shard_is_marked_as_straddling() -> None:
    path = "data/EURUSD/2024-08-16_2024-09-14.parquet"
    selected = hf.discover_source_file_plan(
        [path],
        REVISION,
        datetime(2024, 8, 20, tzinfo=UTC),
        datetime(2024, 8, 21, tzinfo=UTC),
    )

    assert selected[0].straddles_research_split_boundary is True


def test_correct_schema_is_accepted() -> None:
    hf._validate_schema(_table(_valid_rows()).schema, PATH)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda table: table.drop(["askVolume"]), "exactly"),
        (
            lambda table: table.set_column(
                0, "timestamp", pa.array([1, 2, 3], type=pa.int32())
            ),
            "primitive type",
        ),
    ],
)
def test_missing_or_wrong_schema_is_rejected(mutate: Any, message: str) -> None:
    with pytest.raises(hf.HfDukascopyValidationError, match=message):
        hf._validate_schema(mutate(_table(_valid_rows())).schema, PATH)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("askPrice", float("nan"), "non-finite"),
        ("bidPrice", float("inf"), "non-finite"),
        ("askPrice", 0.0, "nonpositive"),
        ("bidPrice", -1.0, "nonpositive"),
        ("askVolume", -0.1, "negative volume"),
        ("bidVolume", -0.1, "negative volume"),
        ("askPrice", 1.0, "ASK below BID"),
    ],
)
def test_invalid_tick_values_are_rejected(
    field: str, value: float, message: str
) -> None:
    rows = _valid_rows()
    rows[field][0] = value
    source = hf.SourceFilePlan(
        PATH,
        date(2019, 3, 11),
        date(2019, 4, 9),
        REVISION,
        None,
        None,
        True,
        False,
    )
    with pytest.raises(hf.HfDukascopyValidationError, match=message):
        hf._validate_admitted_batch(_table(rows).to_batches()[0], source)


def test_null_required_field_is_rejected() -> None:
    rows = _valid_rows()
    rows["askVolume"][0] = None
    source = _source_plan()
    with pytest.raises(hf.HfDukascopyValidationError, match="null"):
        hf._validate_admitted_batch(_table(rows).to_batches()[0], source)


def test_timestamp_milliseconds_are_proved_and_wrong_unit_is_rejected() -> None:
    source = _source_plan()
    hf._prove_millisecond_timestamp_unit(_ms(DAY), source)

    with pytest.raises(hf.HfDukascopyValidationError, match="unambiguously"):
        hf._prove_millisecond_timestamp_unit(int(DAY.timestamp()), source)


def test_backwards_ticks_are_rejected(tmp_path: Path) -> None:
    rows = _valid_rows()
    rows["timestamp"] = [
        rows["timestamp"][2],
        rows["timestamp"][0],
        rows["timestamp"][1],
    ]
    parquet = _write_table(tmp_path / "bad.parquet", rows)

    with pytest.raises(hf.HfDukascopyValidationError, match="monotonic"):
        _ingest(tmp_path / "out", parquet)


def test_tick_aggregation_is_side_correct_sparse_and_tie_deterministic(
    tmp_path: Path,
) -> None:
    rows = _valid_rows()
    rows["timestamp"][2] = _ms(DAY + timedelta(seconds=59))
    for field, value in (
        ("timestamp", _ms(DAY + timedelta(minutes=2, seconds=5))),
        ("askPrice", 1.1003),
        ("bidPrice", 1.1001),
        ("askVolume", 40.0),
        ("bidVolume", 4.0),
    ):
        rows[field].append(value)
    parquet = _write_table(tmp_path / "ticks.parquet", rows)

    result = _ingest(tmp_path / "out", parquet, batch_size=2)

    catalog = ParquetDataCatalog(str(result.catalog_path))
    bid = catalog.query_bars([_bar_type("BID")])
    ask = catalog.query_bars([_bar_type("ASK")])
    assert len(bid) == len(ask) == 2
    assert [bar.ts_event for bar in bid] == [_ns(DAY), _ns(DAY + timedelta(minutes=2))]
    assert str(bid[0].open) == "1.10000"
    assert str(bid[0].high) == "1.10020"
    assert str(bid[0].low) == "1.09990"
    assert str(bid[0].close) == "1.09990"
    assert str(bid[0].volume) == "6.00000000"
    assert str(ask[0].open) == "1.10020"
    assert str(ask[0].high) == "1.10040"
    assert str(ask[0].low) == "1.10010"
    assert str(ask[0].close) == "1.10010"
    assert str(ask[0].volume) == "60.00000000"
    assert bid[0].ts_init == bid[0].ts_event + 60_000_000_000
    assert str(bid[0].bar_type) == _bar_type("BID")
    assert str(ask[0].bar_type) == _bar_type("ASK")
    assert result.missing_intervals[0].start_utc == "2019-03-11T00:01:00Z"


def test_straddling_shard_filters_holdout_before_callback_and_reporting(
    tmp_path: Path,
) -> None:
    cutoff = datetime(2024, 8, 21, tzinfo=UTC)
    rows = _valid_rows(start=cutoff - timedelta(seconds=1))
    rows["timestamp"] = [
        _ms(cutoff - timedelta(seconds=1)),
        _ms(cutoff),
        _ms(cutoff + timedelta(seconds=1)),
    ]
    parquet = _write_table(tmp_path / "straddle.parquet", rows)
    seen: list[list[int]] = []
    path = "data/EURUSD/2024-08-16_2024-09-14.parquet"

    result = _ingest(
        tmp_path / "out",
        parquet,
        path=path,
        start=datetime(2024, 8, 20, tzinfo=UTC),
        end=cutoff,
        callback=lambda batch: seen.append(batch.column(0).to_pylist()),
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert seen == [[_ms(cutoff - timedelta(seconds=1))]]
    assert result.admitted_tick_count == 1
    assert manifest["qa"]["holdout_rows_admitted"] == 0
    assert manifest["qa"]["admitted_tick_count"] == 1
    assert manifest["source_files"][0]["admitted_max_timestamp_utc"] < (
        "2024-08-21T00:00:00.000Z"
    )
    assert all(
        bar.ts_event < _ns(cutoff)
        for bar in ParquetDataCatalog(str(result.catalog_path)).query_bars(
            [_bar_type("BID")]
        )
    )


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (
            datetime(2024, 8, 21, tzinfo=UTC),
            datetime(2024, 8, 22, tzinfo=UTC),
        ),
        (
            datetime(2024, 8, 20, tzinfo=UTC),
            datetime(2024, 8, 22, tzinfo=UTC),
        ),
    ],
)
def test_holdout_requests_fail_before_remote_processing(
    tmp_path: Path, start: datetime, end: datetime
) -> None:
    api = FakeApi([PATH])

    with pytest.raises(hf.HfDukascopyValidationError, match="holdout"):
        hf.ingest_hf_eurusd(
            PLAN,
            tmp_path / "out",
            requested_start=start,
            requested_end_exclusive=end,
            api=api,
        )

    assert api.called is False


def test_provenance_hashes_are_deterministic(tmp_path: Path) -> None:
    parquet = _write_table(tmp_path / "ticks.parquet", _valid_rows())
    first = _ingest(tmp_path / "first", parquet)
    second = _ingest(tmp_path / "second", parquet)
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))

    assert first.semantic_sha256 == second.semantic_sha256
    assert manifest["dataset_revision"] == REVISION
    assert (
        manifest["dataset_plan_sha256"] == load_research_data_plan(PLAN).semantic_sha256
    )
    assert (
        manifest["source_files"][0]["downloaded_sha256"]
        == hashlib.sha256(parquet.read_bytes()).hexdigest()
    )
    payload = {
        key: value for key, value in manifest.items() if key != "semantic_sha256"
    }
    assert (
        manifest["semantic_sha256"]
        == hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def test_bounded_bar_chunks_append_without_losing_same_day_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = {
        "timestamp": [_ms(DAY + timedelta(minutes=index)) for index in range(4)],
        "askPrice": [1.1002] * 4,
        "bidPrice": [1.1] * 4,
        "askVolume": [1.0] * 4,
        "bidVolume": [1.0] * 4,
    }
    parquet = _write_table(tmp_path / "ticks.parquet", rows)
    monkeypatch.setattr(hf, "_WRITE_CHUNK_BARS", 2)

    result = _ingest(tmp_path / "out", parquet)

    catalog = ParquetDataCatalog(str(result.catalog_path))
    assert len(catalog.query_bars([_bar_type("BID")])) == 4
    assert len(catalog.query_bars([_bar_type("ASK")])) == 4


def test_minute_split_across_physical_files_matches_single_file(
    tmp_path: Path,
) -> None:
    rows = {
        "timestamp": [_ms(DAY + timedelta(seconds=1)), _ms(DAY + timedelta(seconds=2))],
        "askPrice": [1.1002, 1.1003],
        "bidPrice": [1.1, 1.1001],
        "askVolume": [2.0, 3.0],
        "bidVolume": [4.0, 5.0],
    }
    combined = _write_table(tmp_path / "combined.parquet", rows)
    expected_result = _ingest(tmp_path / "combined", combined)
    expected_catalog = ParquetDataCatalog(str(expected_result.catalog_path))

    first = _write_table(
        tmp_path / "first.parquet", {name: values[:1] for name, values in rows.items()}
    )
    second = _write_table(
        tmp_path / "second.parquet", {name: values[1:] for name, values in rows.items()}
    )
    catalog_path = tmp_path / "split" / "catalog"
    catalog_path.mkdir(parents=True)
    catalog = ParquetDataCatalog(str(catalog_path))
    catalog.write_instruments([hf._eurusd_instrument()])
    state = hf._StreamState(
        catalog=catalog,
        current=None,
        previous_tick_ms=None,
        previous_file_last_ms=None,
        last_emitted_minute_ms=None,
        bid_pending=[],
        ask_pending=[],
        bar_count=0,
        next_expected_minute=DAY,
        missing_intervals=[],
    )
    source = _source_plan()
    hf._process_source_file(
        first,
        source,
        DAY,
        DAY + timedelta(days=1),
        state,
        batch_size=1,
        on_admitted_batch=None,
    )
    hf._process_source_file(
        second,
        source,
        DAY,
        DAY + timedelta(days=1),
        state,
        batch_size=1,
        on_admitted_batch=None,
    )
    hf._emit_current(state)
    hf._write_pending(state)

    for side in ("BID", "ASK"):
        assert catalog.query_bars([_bar_type(side)]) == expected_catalog.query_bars(
            [_bar_type(side)]
        )


def test_hf_source_is_accepted_by_g06_and_session_coverage(tmp_path: Path) -> None:
    timestamps = [
        _ms(DAY + timedelta(minutes=index, seconds=1)) for index in range(240)
    ]
    rows = {
        "timestamp": timestamps,
        "askPrice": [1.1002] * 240,
        "bidPrice": [1.1] * 240,
        "askVolume": [1.0] * 240,
        "bidVolume": [1.0] * 240,
    }
    parquet = _write_table(tmp_path / "ticks.parquet", rows)
    root = tmp_path / "out"
    _ingest(root, parquet)

    derived = derive_eurusd_bars(root)
    coverage = run_eurusd_session_coverage_qa(root)

    assert derived.emitted_counts[f"{hf.INSTRUMENT_ID}-1-HOUR-BID-INTERNAL"] == 4
    assert coverage.manifest_path.is_file()
    coverage_manifest = json.loads(coverage.manifest_path.read_text(encoding="utf-8"))
    assert coverage_manifest["parent_ingestion_version"] == hf.HF_INGESTION_VERSION


def _valid_rows(start: datetime = DAY) -> dict[str, list[Any]]:
    return {
        "timestamp": [
            _ms(start + timedelta(seconds=1)),
            _ms(start + timedelta(seconds=1)),
            _ms(start + timedelta(minutes=2, seconds=5)),
        ],
        "askPrice": [1.1002, 1.1004, 1.1001],
        "bidPrice": [1.1, 1.1002, 1.0999],
        "askVolume": [10.0, 20.0, 30.0],
        "bidVolume": [1.0, 2.0, 3.0],
    }


def _table(rows: dict[str, list[Any]]) -> pa.Table:
    return pa.table(
        {
            "timestamp": pa.array(rows["timestamp"], type=pa.int64()),
            "askPrice": pa.array(rows["askPrice"], type=pa.float64()),
            "bidPrice": pa.array(rows["bidPrice"], type=pa.float64()),
            "askVolume": pa.array(rows["askVolume"], type=pa.float64()),
            "bidVolume": pa.array(rows["bidVolume"], type=pa.float64()),
        }
    )


def _write_table(path: Path, rows: dict[str, list[Any]]) -> Path:
    pq.write_table(_table(rows), path, row_group_size=2)
    return path


def _ingest(
    output: Path,
    parquet: Path,
    *,
    path: str = PATH,
    start: datetime = DAY,
    end: datetime = DAY + timedelta(days=1),
    batch_size: int = 65_536,
    callback: Any = None,
) -> hf.HfIngestionResult:
    return hf.ingest_hf_eurusd(
        PLAN,
        output,
        requested_start=start,
        requested_end_exclusive=end,
        api=FakeApi([path]),
        downloader=lambda **_kwargs: str(parquet),
        batch_size=batch_size,
        on_admitted_batch=callback,
    )


def _source_plan() -> hf.SourceFilePlan:
    return hf.SourceFilePlan(
        PATH,
        date(2019, 3, 11),
        date(2019, 4, 9),
        REVISION,
        None,
        None,
        True,
        False,
    )


def _ms(value: datetime) -> int:
    return int(value.timestamp() * 1_000)


def _ns(value: datetime) -> int:
    return int(value.timestamp() * 1_000_000_000)


def _bar_type(side: str) -> str:
    return f"{hf.INSTRUMENT_ID}-1-MINUTE-{side}-EXTERNAL"
