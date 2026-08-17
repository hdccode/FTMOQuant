import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from nautilus_trader.model import Bar, BarType
from nautilus_trader.persistence import ParquetDataCatalog

import ftmoquant.data.derived_bars as derived
from ftmoquant.data.derived_bars import (
    DerivationValidationError,
    derive_eurusd_bars,
)
from ftmoquant.data.dukascopy import SourceBar, _eurusd_instrument, _to_nautilus_bars

START = datetime(2024, 1, 2, tzinfo=UTC)


def test_sixty_contiguous_minutes_emit_exactly_one_hour(tmp_path: Path) -> None:
    root = _catalog_root(tmp_path, range(60))

    result = derive_eurusd_bars(root)

    catalog = ParquetDataCatalog(str(result.catalog_path))
    for side in ("BID", "ASK"):
        bars = catalog.query_bars([_target(1, side)])
        assert len(bars) == 1
        assert bars[0].ts_event == _ns(START + timedelta(hours=1))
        assert bars[0].ts_init == bars[0].ts_event
    manifest = _manifest(result.manifest_path)
    assert manifest["derived_bar_integrity_valid"] is True
    assert manifest["coverage_status"] == "incomplete"
    assert manifest["research_ready"] is False
    assert result.research_ready is manifest["research_ready"]


def test_fifty_nine_minutes_emit_no_hour(tmp_path: Path) -> None:
    root = _catalog_root(tmp_path, range(59))

    result = derive_eurusd_bars(root)

    assert result.emitted_counts[_target(1, "BID")] == 0
    manifest = _manifest(result.manifest_path)
    qa = manifest["series"][_target(1, "BID")]
    assert qa["dropped_incomplete_window_count"] == 1
    assert qa["dropped_window_details"][0]["observed_minutes"] == 59
    assert manifest["coverage_status"] == "incomplete"
    assert manifest["research_ready"] is False
    assert result.research_ready is manifest["research_ready"]


def test_two_hundred_forty_minutes_emit_one_four_hour_bar(
    tmp_path: Path,
) -> None:
    root = _catalog_root(tmp_path, range(240))

    result = derive_eurusd_bars(root)

    catalog = ParquetDataCatalog(str(result.catalog_path))
    for side in ("BID", "ASK"):
        bars = catalog.query_bars([_target(4, side)])
        assert len(bars) == 1
        assert bars[0].ts_event == _ns(START + timedelta(hours=4))


def test_thirty_contiguous_minutes_emit_exactly_one_m30_bar(tmp_path: Path) -> None:
    root = _catalog_root(tmp_path, range(30))

    result = derive_eurusd_bars(root)

    catalog = ParquetDataCatalog(str(result.catalog_path))
    for side in ("BID", "ASK"):
        target = derived._target_bar_type(30, side)
        assert target == f"{derived.INSTRUMENT_ID}-30-MINUTE-{side}-INTERNAL"
        bars = catalog.query_bars([target])
        assert len(bars) == 1
        assert bars[0].ts_event == _ns(START + timedelta(minutes=30))
    manifest = _manifest(result.manifest_path)
    assert manifest["counts"]["emitted"]["M30"] == {"BID": 1, "ASK": 1}
    assert "M30" in manifest["utc_alignment"]


def test_twenty_nine_minutes_emit_no_m30_bar(tmp_path: Path) -> None:
    root = _catalog_root(tmp_path, range(29))

    result = derive_eurusd_bars(root)

    assert result.emitted_counts[derived._target_bar_type(30, "BID")] == 0
    manifest = _manifest(result.manifest_path)
    detail = manifest["series"][derived._target_bar_type(30, "BID")]
    assert detail["dropped_incomplete_window_count"] == 1
    assert detail["dropped_window_details"][0]["observed_minutes"] == 29


def test_m30_h1_h4_all_emit_from_one_complete_day(tmp_path: Path) -> None:
    root = _catalog_root(tmp_path, range(24 * 60))

    result = derive_eurusd_bars(root)

    catalog = ParquetDataCatalog(str(result.catalog_path))
    for minutes, expected_count in ((30, 48), (60, 24), (240, 6)):
        for side in ("BID", "ASK"):
            bars = catalog.query_bars([derived._target_bar_type(minutes, side)])
            assert len(bars) == expected_count
    manifest = _manifest(result.manifest_path)
    assert manifest["utc_alignment"]["M30"] == "00:00, 00:30, ..., 23:30 UTC closes"
    assert manifest["utc_alignment"]["1H"] == "00:00, 01:00, ..., 23:00 UTC closes"
    assert manifest["utc_alignment"]["4H"] == "00:00, 04:00, ..., 20:00 UTC closes"


def test_complete_dataset_is_research_ready(tmp_path: Path) -> None:
    root = _catalog_root(tmp_path, range(24 * 60))

    result = derive_eurusd_bars(root)

    manifest = _manifest(result.manifest_path)
    assert manifest["derived_bar_integrity_valid"] is True
    assert manifest["coverage_status"] == "complete"
    assert manifest["research_ready"] is True
    assert manifest["research_readiness_reasons"] == []
    assert result.research_ready is manifest["research_ready"]


def test_missing_minute_inside_window_drops_window(tmp_path: Path) -> None:
    root = _catalog_root(tmp_path, [minute for minute in range(60) if minute != 31])

    result = derive_eurusd_bars(root)

    assert result.emitted_counts[_target(1, "BID")] == 0
    detail = _manifest(result.manifest_path)["series"][_target(1, "BID")]
    assert detail["dropped_incomplete_window_count"] == 1
    assert detail["dropped_window_details"][0]["reason"] == (
        "incomplete_or_nonconsecutive_minutes"
    )


def test_partial_first_window_is_skipped_but_next_full_window_emits(
    tmp_path: Path,
) -> None:
    root = _catalog_root(tmp_path, range(30, 120))

    result = derive_eurusd_bars(root)

    bars = ParquetDataCatalog(str(result.catalog_path)).query_bars([_target(1, "BID")])
    assert [bar.ts_event for bar in bars] == [_ns(START + timedelta(hours=2))]
    details = _manifest(result.manifest_path)["series"][_target(1, "BID")]
    assert details["dropped_incomplete_window_count"] == 1
    assert details["dropped_window_details"][0]["observed_minutes"] == 30


def test_bid_ask_source_coverage_mismatch_is_rejected(tmp_path: Path) -> None:
    root = _catalog_root(tmp_path, range(60), ask_minutes=range(59))

    with pytest.raises(DerivationValidationError, match="coverage mismatch"):
        derive_eurusd_bars(root)


def test_weekend_no_update_intervals_never_emit_bars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    friday = datetime(2024, 1, 5, 20, tzinfo=UTC)
    monday = datetime(2024, 1, 8, 0, tzinfo=UTC)
    timestamps = [friday + timedelta(minutes=index) for index in range(240)]
    timestamps.extend(monday + timedelta(minutes=index) for index in range(240))
    root = _catalog_root_from_timestamps(tmp_path, timestamps)
    monkeypatch.setattr(derived, "_SOURCE_QUERY_CHUNK_MINUTES", 61)

    result = derive_eurusd_bars(root)

    bars = ParquetDataCatalog(str(result.catalog_path)).query_bars([_target(1, "BID")])
    assert [bar.ts_event for bar in bars] == [
        *[_ns(friday + timedelta(hours=hour)) for hour in range(1, 5)],
        *[_ns(monday + timedelta(hours=hour)) for hour in range(1, 5)],
    ]
    qa = _manifest(result.manifest_path)["series"][_target(1, "BID")]
    assert qa["no_update_window_count"] == 88
    manifest = _manifest(result.manifest_path)
    assert manifest["coverage_status"] == "unclassified"
    assert manifest["research_ready"] is False
    assert result.research_ready is manifest["research_ready"]


@pytest.mark.parametrize(
    ("chunk_minutes", "source_minutes", "hours", "expected"),
    [(37, 120, 1, 2), (73, 480, 4, 2)],
)
def test_complete_window_crossing_query_chunk_matches_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    chunk_minutes: int,
    source_minutes: int,
    hours: int,
    expected: int,
) -> None:
    root = _catalog_root(tmp_path, range(source_minutes))
    reference = _reference_behavior(root)
    monkeypatch.setattr(derived, "_SOURCE_QUERY_CHUNK_MINUTES", chunk_minutes)

    result = derive_eurusd_bars(root)

    catalog = ParquetDataCatalog(str(result.catalog_path))
    for side in ("BID", "ASK"):
        target = _target(hours, side)
        assert len(catalog.query_bars([target])) == expected
        assert tuple(catalog.query_bars([target])) == reference[target][0]


def test_missing_minute_at_query_boundary_is_dropped_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    minutes = [minute for minute in range(60) if minute != 31]
    root = _catalog_root(tmp_path, minutes)
    monkeypatch.setattr(derived, "_SOURCE_QUERY_CHUNK_MINUTES", 31)

    derive_eurusd_bars(root)

    detail = _manifest(root / derived.DERIVED_MANIFEST_FILENAME)["series"][
        _target(1, "BID")
    ]
    assert detail["dropped_incomplete_window_count"] == 1
    assert detail["dropped_window_details"][0]["observed_minutes"] == 59


def test_streaming_manifest_bars_and_callbacks_match_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    minutes = [minute for minute in range(24 * 60) if minute not in {37, 238, 601}]
    root = _catalog_root(tmp_path, minutes)
    reference = _reference_behavior(root)
    reference_manifest = _reference_manifest(root, reference)
    native = derived._aggregate_native
    captured: dict[str, list[tuple[int, int]]] = {}

    def recording_native(
        *,
        instrument: object,
        source_bars: Sequence[Bar],
        side: str,
        minutes: int,
    ) -> tuple[tuple[Bar, ...], tuple[int, ...]]:
        bars, callbacks = native(
            instrument=instrument,
            source_bars=source_bars,
            side=side,
            minutes=minutes,
        )
        captured.setdefault(derived._target_bar_type(minutes, side), []).extend(
            (bar.ts_event, callback)
            for bar, callback in zip(bars, callbacks, strict=True)
        )
        return bars, callbacks

    monkeypatch.setattr(derived, "_SOURCE_QUERY_CHUNK_MINUTES", 37)
    monkeypatch.setattr(derived, "_aggregate_native", recording_native)

    result = derive_eurusd_bars(root)

    assert _manifest(result.manifest_path) == reference_manifest
    catalog = ParquetDataCatalog(str(result.catalog_path))
    for target, (bars, callbacks, _) in reference.items():
        assert tuple(catalog.query_bars([target])) == bars
        assert captured.get(target, []) == [
            (bar.ts_event, callback)
            for bar, callback in zip(bars, callbacks, strict=True)
        ]


def test_utc_alignment_and_native_ohlcv_are_preserved(tmp_path: Path) -> None:
    root = _catalog_root(tmp_path, range(240))
    source_catalog = ParquetDataCatalog(str(root / "catalog"))
    source = tuple(source_catalog.query_bars([_source("BID")]))
    instrument = source_catalog.instruments([derived.INSTRUMENT_ID])[0]
    eligible, _ = derived._eligible_source_bars(source, 240, 240)
    native, callbacks = derived._aggregate_native(
        instrument=instrument,
        source_bars=eligible,
        side="BID",
        minutes=240,
    )

    result = derive_eurusd_bars(root)

    stored = ParquetDataCatalog(str(result.catalog_path)).query_bars(
        [_target(4, "BID")]
    )
    assert stored == list(native)
    assert len(callbacks) == 1
    assert str(stored[0].open) == "1.10000"
    assert str(stored[0].high) == "1.10249"
    assert str(stored[0].low) == "1.09990"
    assert str(stored[0].close) == "1.10244"
    assert str(stored[0].volume) == "240.00000000"
    assert stored[0].ts_event % (4 * 60 * derived._MINUTE_NS) == 0


def test_bar_is_not_visible_before_final_source_minute_close(
    tmp_path: Path,
) -> None:
    root = _catalog_root(tmp_path, range(60))
    catalog = ParquetDataCatalog(str(root / "catalog"))
    source = tuple(catalog.query_bars([_source("BID")]))
    eligible, _ = derived._eligible_source_bars(source, 60, 60)

    bars, callbacks = derived._aggregate_native(
        instrument=catalog.instruments([derived.INSTRUMENT_ID])[0],
        source_bars=eligible,
        side="BID",
        minutes=60,
    )

    assert bars[0].ts_event == source[-1].ts_init
    assert callbacks[0] == source[-1].ts_init + 1_000


def test_repeated_derivation_is_deterministic_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_root = _catalog_root(tmp_path / "first", range(240))
    second_root = _catalog_root(tmp_path / "second", range(240))
    monkeypatch.setattr(derived, "_SOURCE_QUERY_CHUNK_MINUTES", 37)

    first = derive_eurusd_bars(first_root)
    original_manifest = first.manifest_path.read_bytes()
    repeated = derive_eurusd_bars(first_root)
    second = derive_eurusd_bars(second_root)

    assert repeated.manifest_path.read_bytes() == original_manifest
    assert second.manifest_path.read_bytes() == original_manifest
    first_catalog = ParquetDataCatalog(str(first.catalog_path))
    second_catalog = ParquetDataCatalog(str(second.catalog_path))
    for hours in (1, 4):
        for side in ("BID", "ASK"):
            assert first_catalog.query_bars(
                [_target(hours, side)]
            ) == second_catalog.query_bars([_target(hours, side)])


def test_conflicting_existing_derived_data_fails_before_writes(
    tmp_path: Path,
) -> None:
    root = _catalog_root(tmp_path, range(60))
    catalog = ParquetDataCatalog(str(root / "catalog"))
    source = catalog.query_bars([_source("BID")])[0]
    conflict = Bar(
        bar_type=BarType.from_str(_target(1, "BID")),
        open=source.open,
        high=source.high,
        low=source.low,
        close=source.close,
        volume=source.volume,
        ts_event=_ns(START + timedelta(hours=1)),
        ts_init=_ns(START + timedelta(hours=1)),
    )
    catalog.write_bars([conflict])

    with pytest.raises(DerivationValidationError, match="conflicting data"):
        derive_eurusd_bars(root)

    assert not (root / derived.DERIVED_MANIFEST_FILENAME).exists()


def test_manifest_records_parent_hash_config_and_series_hashes(
    tmp_path: Path,
) -> None:
    root = _catalog_root(tmp_path, range(60))
    parent = root / derived.PARENT_MANIFEST_FILENAME

    result = derive_eurusd_bars(root)

    manifest = _manifest(result.manifest_path)
    assert (
        manifest["parent_g0_5_manifest_sha256"]
        == hashlib.sha256(parent.read_bytes()).hexdigest()
    )
    assert manifest["nautilus_version"] == "2.0.0rc2"
    assert manifest["derivation_version"] == "g0.6-2"
    assert manifest["aggregation_configuration"] == {
        "api": "Nautilus BacktestEngine composite BarType bar-to-bar aggregation",
        "time_bars_build_delay_microseconds": 1,
        "time_bars_build_with_no_updates": False,
        "time_bars_interval_type": "LEFT_OPEN",
        "time_bars_origin_offset": {},
        "time_bars_skip_first_non_full_bar": True,
        "time_bars_timestamp_on_close": True,
    }
    assert manifest["series"][_target(1, "BID")]["content_sha256"]
    assert manifest["bid_ask_derived_coverage_matches"] is True


def _catalog_root(
    root: Path,
    bid_minutes: Sequence[int] | range,
    *,
    ask_minutes: Sequence[int] | range | None = None,
) -> Path:
    ask_minutes = bid_minutes if ask_minutes is None else ask_minutes
    return _catalog_root_from_timestamps(
        root,
        [START + timedelta(minutes=minute) for minute in bid_minutes],
        ask_timestamps=[START + timedelta(minutes=minute) for minute in ask_minutes],
    )


def _catalog_root_from_timestamps(
    root: Path,
    bid_timestamps: Sequence[datetime],
    *,
    ask_timestamps: Sequence[datetime] | None = None,
) -> Path:
    ask_timestamps = bid_timestamps if ask_timestamps is None else ask_timestamps
    catalog_path = root / "catalog"
    catalog_path.mkdir(parents=True)
    catalog = ParquetDataCatalog(str(catalog_path))
    catalog.write_instruments([_eurusd_instrument()])
    catalog.write_bars(_bars(bid_timestamps, "BID"))
    catalog.write_bars(_bars(ask_timestamps, "ASK"))
    parent = {
        "instrument_id": derived.INSTRUMENT_ID,
        "source_granularity": "1-minute",
        "price_sides": ["BID", "ASK"],
        "nautilus_version": "2.0.0rc2",
        "ingestion_version": "g0.5-1",
        "requested_utc_range": {
            "start_date": min(bid_timestamps).date().isoformat(),
            "end_date_inclusive": max(bid_timestamps).date().isoformat(),
        },
        "qa": {
            "bid_bar_count": len(bid_timestamps),
            "ask_bar_count": len(ask_timestamps),
        },
    }
    (root / derived.PARENT_MANIFEST_FILENAME).write_text(
        json.dumps(parent, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root


def _bars(timestamps: Sequence[datetime], side: str) -> list[Bar]:
    spread = Decimal("0.00020") if side == "ASK" else Decimal(0)
    rows = []
    for index, timestamp in enumerate(timestamps):
        base = Decimal("1.10000") + Decimal(index) * Decimal("0.00001") + spread
        rows.append(
            SourceBar(
                timestamp=timestamp,
                open=base,
                high=base + Decimal("0.00010"),
                low=base - Decimal("0.00010"),
                close=base + Decimal("0.00005"),
                volume=Decimal(1),
            )
        )
    return _to_nautilus_bars(rows, side)


def _reference_behavior(
    root: Path,
) -> dict[
    str,
    tuple[tuple[Bar, ...], tuple[int, ...], tuple[derived.DroppedWindow, ...]],
]:
    catalog = ParquetDataCatalog(str(root / "catalog"))
    parent = derived._load_parent_manifest(root / derived.PARENT_MANIFEST_FILENAME)
    start_ns, end_ns = derived._requested_range_ns(parent)
    instrument = catalog.instruments([derived.INSTRUMENT_ID])[0]
    output = {}
    for minutes, expected_minutes in derived._TARGETS:
        for side in derived._SIDES:
            source = tuple(catalog.query_bars([_source(side)]))
            eligible, dropped = derived._eligible_source_bars(
                source,
                minutes,
                expected_minutes,
                range_start_ns=start_ns,
                range_end_ns=end_ns,
            )
            bars, callbacks = derived._aggregate_native(
                instrument=instrument,
                source_bars=eligible,
                side=side,
                minutes=minutes,
            )
            output[derived._target_bar_type(minutes, side)] = (bars, callbacks, dropped)
    return output


def _reference_manifest(
    root: Path,
    reference: dict[
        str,
        tuple[tuple[Bar, ...], tuple[int, ...], tuple[derived.DroppedWindow, ...]],
    ],
) -> dict[str, object]:
    parent_path = root / derived.PARENT_MANIFEST_FILENAME
    parent = derived._load_parent_manifest(parent_path)
    series = []
    for minutes, expected_minutes in derived._TARGETS:
        for side in derived._SIDES:
            target = derived._target_bar_type(minutes, side)
            bars, _, dropped = reference[target]
            coverage_digest = hashlib.sha256()
            for bar in bars:
                coverage_digest.update(f"{bar.ts_event}\n".encode())
            source_count = parent["qa"][f"{side.lower()}_bar_count"]
            series.append(
                derived.SeriesResult(
                    source_bar_type=_source(side),
                    composite_bar_type=derived._composite_bar_type(minutes, side),
                    target_bar_type=target,
                    source_bar_count=source_count,
                    eligible_source_bar_count=(len(bars) * expected_minutes),
                    emitted_bar_count=len(bars),
                    content_sha256=derived._bars_sha256(bars),
                    coverage_sha256=coverage_digest.hexdigest(),
                    dropped_windows=dropped,
                )
            )
    coverage_status, ready, reasons = derived._research_readiness(series)
    return derived._derived_manifest(
        hashlib.sha256(parent_path.read_bytes()).hexdigest(),
        parent,
        series,
        root / "catalog",
        root,
        coverage_status,
        ready,
        reasons,
    )


def _target(hours: int, side: str) -> str:
    return f"{derived.INSTRUMENT_ID}-{hours}-HOUR-{side}-INTERNAL"


def _source(side: str) -> str:
    return f"{derived.INSTRUMENT_ID}-1-MINUTE-{side}-EXTERNAL"


def _ns(value: datetime) -> int:
    return int(value.timestamp() * 1_000_000_000)


def _manifest(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value
