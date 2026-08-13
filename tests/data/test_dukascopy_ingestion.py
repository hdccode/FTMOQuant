import csv
import hashlib
import json
from collections.abc import Sequence
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from nautilus_trader.persistence import ParquetDataCatalog

import ftmoquant.data.dukascopy as pipeline
from ftmoquant.data.dukascopy import IngestionValidationError, ingest_eurusd

START = date(2024, 1, 2)
END = date(2024, 1, 2)
DIVISOR = Decimal("1000")
BID_ROWS = [
    ["2024-01-02 00:00:00+00:00", "1.10000", "1.10010", "1.09990", "1.10005", "10.5"],
    ["2024-01-02 00:01:00+00:00", "1.10005", "1.10015", "1.10000", "1.10010", "11.25"],
    ["2024-01-02 00:03:00+00:00", "1.10010", "1.10020", "1.10005", "1.10015", "9.75"],
]
ASK_ROWS = [
    ["2024-01-02 00:00:00+00:00", "1.10020", "1.10030", "1.10010", "1.10025", "9.5"],
    ["2024-01-02 00:01:00+00:00", "1.10025", "1.10035", "1.10020", "1.10030", "10.25"],
    ["2024-01-02 00:03:00+00:00", "1.10030", "1.10040", "1.10025", "1.10035", "8.75"],
]


def test_valid_local_fixture_writes_nautilus_catalog_and_provenance(
    tmp_path: Path,
) -> None:
    _write_source(tmp_path)

    result = ingest_eurusd(START, END, tmp_path, DIVISOR, acquire=False)

    assert result.bid_bar_count == 3
    assert result.ask_bar_count == 3
    catalog = ParquetDataCatalog(str(result.catalog_path))
    assert len(catalog.instruments([pipeline.INSTRUMENT_ID])) == 1
    bid = catalog.query_bars([_bar_type("BID")])
    ask = catalog.query_bars([_bar_type("ASK")])
    assert len(bid) == len(ask) == 3
    assert str(bid[0].bar_type) == _bar_type("BID")
    assert bid[0].ts_init - bid[0].ts_event == 60_000_000_000

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["provider"] == "Dukascopy"
    assert manifest["instrument_id"] == pipeline.INSTRUMENT_ID
    assert manifest["upstream_downloader"]["version"] == "1.0.0"
    assert manifest["upstream_downloader"]["commit"] == pipeline.UPSTREAM_COMMIT
    assert manifest["nautilus_version"] == "2.0.0rc2"
    assert manifest["config_version"] == 1
    assert manifest["source_metadata_sidecar_filenames"] == [
        "source/EURUSD_1MIN_bid.csv.meta.json",
        "source/EURUSD_1MIN_ask.csv.meta.json",
    ]
    for name in manifest["source_filenames"]:
        source = tmp_path / name
        assert manifest["source_sha256"][name] == hashlib.sha256(
            source.read_bytes()
        ).hexdigest()


def test_duplicate_timestamp_is_rejected(tmp_path: Path) -> None:
    duplicate_bid = [BID_ROWS[0], BID_ROWS[0], BID_ROWS[1]]
    _write_source(tmp_path, bid_rows=duplicate_bid, ask_rows=duplicate_bid)

    with pytest.raises(IngestionValidationError, match="duplicate timestamps"):
        ingest_eurusd(START, END, tmp_path, DIVISOR, acquire=False)


def test_invalid_ohlc_is_rejected(tmp_path: Path) -> None:
    invalid = _copy_rows(BID_ROWS)
    invalid[1][2] = "1.09900"
    _write_source(tmp_path, bid_rows=invalid)

    with pytest.raises(IngestionValidationError, match="invalid OHLC ordering"):
        ingest_eurusd(START, END, tmp_path, DIVISOR, acquire=False)


def test_ask_below_bid_is_rejected(tmp_path: Path) -> None:
    invalid = _copy_rows(ASK_ROWS)
    invalid[1][1:5] = ["1.09905", "1.09915", "1.09900", "1.09910"]
    _write_source(tmp_path, ask_rows=invalid)

    with pytest.raises(IngestionValidationError, match="ASK open is below BID"):
        ingest_eurusd(START, END, tmp_path, DIVISOR, acquire=False)


def test_timezone_invalid_input_is_rejected(tmp_path: Path) -> None:
    invalid = _copy_rows(BID_ROWS)
    invalid[0][0] = "2024-01-02 00:00:00"
    _write_source(tmp_path, bid_rows=invalid)

    with pytest.raises(IngestionValidationError, match="exact UTC minute"):
        ingest_eurusd(START, END, tmp_path, DIVISOR, acquire=False)


def test_missing_intervals_are_reported_and_not_filled(tmp_path: Path) -> None:
    _write_source(tmp_path)

    result = ingest_eurusd(START, END, tmp_path, DIVISOR, acquire=False)

    assert result.missing_intervals[0].start_utc == "2024-01-02T00:02:00Z"
    assert result.missing_intervals[0].end_utc == "2024-01-02T00:02:00Z"
    assert result.missing_intervals[0].minutes == 1
    assert sum(item.minutes for item in result.missing_intervals) == 1437
    catalog = ParquetDataCatalog(str(result.catalog_path))
    assert len(catalog.query_bars([_bar_type("BID")])) == len(BID_ROWS)


def test_identical_source_is_reproducible_and_same_root_is_idempotent(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _write_source(first_root)
    _write_source(second_root)

    first = ingest_eurusd(START, END, first_root, DIVISOR, acquire=False)
    second = ingest_eurusd(START, END, second_root, DIVISOR, acquire=False)
    repeated = ingest_eurusd(START, END, first_root, DIVISOR, acquire=False)

    first_catalog = ParquetDataCatalog(str(first.catalog_path))
    second_catalog = ParquetDataCatalog(str(second.catalog_path))
    for side in ("BID", "ASK"):
        assert first_catalog.query_bars([_bar_type(side)]) == second_catalog.query_bars(
            [_bar_type(side)]
        )
        assert first_catalog.query_bars([_bar_type(side)]) == ParquetDataCatalog(
            str(repeated.catalog_path)
        ).query_bars([_bar_type(side)])


def test_incompatible_bid_ask_coverage_is_rejected(tmp_path: Path) -> None:
    _write_source(tmp_path, ask_rows=ASK_ROWS[:-1])

    with pytest.raises(IngestionValidationError, match="coverage is incompatible"):
        ingest_eurusd(START, END, tmp_path, DIVISOR, acquire=False)


def test_obviously_corrupt_price_scale_is_rejected(tmp_path: Path) -> None:
    corrupt = _copy_rows(BID_ROWS)
    for row in corrupt:
        for index in range(1, 5):
            row[index] = str(Decimal(row[index]) * Decimal("100000"))
    _write_source(tmp_path, bid_rows=corrupt, ask_rows=corrupt)

    with pytest.raises(IngestionValidationError, match="price scale is uncertain"):
        ingest_eurusd(START, END, tmp_path, DIVISOR, acquire=False)


def test_empty_source_is_rejected(tmp_path: Path) -> None:
    _write_source(tmp_path, bid_rows=[])

    with pytest.raises(IngestionValidationError, match="dataset is empty"):
        ingest_eurusd(START, END, tmp_path, DIVISOR, acquire=False)


def test_acquisition_delegates_to_pinned_upstream_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    def fake_upstream(arguments: Sequence[str]) -> int:
        captured.extend(arguments)
        output = Path(arguments[arguments.index("--out") + 1]).parent
        _write_source(output)
        return 0

    monkeypatch.setattr(pipeline, "dukascopy_export_main", fake_upstream)

    ingest_eurusd(START, END, tmp_path, DIVISOR)

    assert captured[captured.index("--symbols") + 1] == "EURUSD"
    assert captured[captured.index("--resample") + 1] == "1min"
    assert captured[captured.index("--price-divisor") + 1] == "1000"


def test_output_root_inside_git_worktree_is_rejected(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    (repository / ".git").mkdir(parents=True)

    with pytest.raises(ValueError, match="outside a Git worktree"):
        ingest_eurusd(START, END, repository / "data", DIVISOR, acquire=False)


def _write_source(
    root: Path,
    *,
    bid_rows: Sequence[Sequence[str]] = BID_ROWS,
    ask_rows: Sequence[Sequence[str]] = ASK_ROWS,
) -> None:
    source = root / "source"
    source.mkdir(parents=True, exist_ok=True)
    for side, rows in (("bid", bid_rows), ("ask", ask_rows)):
        csv_path = source / f"EURUSD_1MIN_{side}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
            writer.writerows(rows)
        metadata = {
            "data_type": "candles",
            "generated_at": "2024-01-03T00:00:00Z",
            "params": {
                "date_from": START.isoformat(),
                "date_to": END.isoformat(),
                "price_side": side,
                "resample": "1min",
            },
            "price_divisor": float(DIVISOR),
            "schema_version": "1",
            "source": "dukascopy",
            "symbol": "EURUSD",
            "timestamp_format": "iso8601_utc",
        }
        csv_path.with_suffix(".csv.meta.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _copy_rows(rows: Sequence[Sequence[str]]) -> list[list[str]]:
    return [list(row) for row in rows]


def _bar_type(side: str) -> str:
    return f"{pipeline.INSTRUMENT_ID}-1-MINUTE-{side}-EXTERNAL"
