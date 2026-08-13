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


def test_fifty_nine_minutes_emit_no_hour(tmp_path: Path) -> None:
    root = _catalog_root(tmp_path, range(59))

    result = derive_eurusd_bars(root)

    assert result.emitted_counts[_target(1, "BID")] == 0
    manifest = _manifest(result.manifest_path)
    qa = manifest["series"][_target(1, "BID")]
    assert qa["dropped_incomplete_window_count"] == 1
    assert qa["dropped_window_details"][0]["observed_minutes"] == 59


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

    bars = ParquetDataCatalog(str(result.catalog_path)).query_bars(
        [_target(1, "BID")]
    )
    assert [bar.ts_event for bar in bars] == [_ns(START + timedelta(hours=2))]
    details = _manifest(result.manifest_path)["series"][_target(1, "BID")]
    assert details["dropped_incomplete_window_count"] == 1
    assert details["dropped_window_details"][0]["observed_minutes"] == 30


def test_bid_ask_source_coverage_mismatch_is_rejected(tmp_path: Path) -> None:
    root = _catalog_root(tmp_path, range(60), ask_minutes=range(59))

    with pytest.raises(DerivationValidationError, match="coverage mismatch"):
        derive_eurusd_bars(root)


def test_weekend_no_update_intervals_never_emit_bars(tmp_path: Path) -> None:
    friday = datetime(2024, 1, 5, 20, tzinfo=UTC)
    monday = datetime(2024, 1, 8, 0, tzinfo=UTC)
    timestamps = [friday + timedelta(minutes=index) for index in range(60)]
    timestamps.extend(monday + timedelta(minutes=index) for index in range(60))
    root = _catalog_root_from_timestamps(tmp_path, timestamps)

    result = derive_eurusd_bars(root)

    bars = ParquetDataCatalog(str(result.catalog_path)).query_bars(
        [_target(1, "BID")]
    )
    assert [bar.ts_event for bar in bars] == [
        _ns(friday + timedelta(hours=1)),
        _ns(monday + timedelta(hours=1)),
    ]
    qa = _manifest(result.manifest_path)["series"][_target(1, "BID")]
    assert qa["no_update_window_count"] == 94


def test_utc_alignment_and_native_ohlcv_are_preserved(tmp_path: Path) -> None:
    root = _catalog_root(tmp_path, range(240))
    source_catalog = ParquetDataCatalog(str(root / "catalog"))
    source = tuple(source_catalog.query_bars([_source("BID")]))
    instrument = source_catalog.instruments([derived.INSTRUMENT_ID])[0]
    eligible, _ = derived._eligible_source_bars(source, 4, 240)
    native, callbacks = derived._aggregate_native(
        instrument=instrument,
        source_bars=eligible,
        side="BID",
        hours=4,
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
    eligible, _ = derived._eligible_source_bars(source, 1, 60)

    bars, callbacks = derived._aggregate_native(
        instrument=catalog.instruments([derived.INSTRUMENT_ID])[0],
        source_bars=eligible,
        side="BID",
        hours=1,
    )

    assert bars[0].ts_event == source[-1].ts_init
    assert callbacks[0] == source[-1].ts_init + 1_000


def test_repeated_derivation_is_deterministic_and_idempotent(
    tmp_path: Path,
) -> None:
    first_root = _catalog_root(tmp_path / "first", range(240))
    second_root = _catalog_root(tmp_path / "second", range(240))

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
    assert manifest["parent_g0_5_manifest_sha256"] == hashlib.sha256(
        parent.read_bytes()
    ).hexdigest()
    assert manifest["nautilus_version"] == "2.0.0rc2"
    assert manifest["derivation_version"] == "g0.6-1"
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
