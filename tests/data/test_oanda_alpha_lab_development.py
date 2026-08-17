import csv
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
import yaml
from nautilus_trader.persistence import ParquetDataCatalog

from ftmoquant.data import oanda_alpha_lab_development as oanda_lab
from ftmoquant.data.derived_bars import derive_instrument_bars
from ftmoquant.data.instruments import (
    NZDUSD_OANDA_SPEC,
    OANDA_ALPHA_LAB_SPECS,
    USDCAD_OANDA_SPEC,
    USDJPY_OANDA_SPEC,
    InstrumentSpec,
    InstrumentSpecValidationError,
)

CONFIG_PATH = oanda_lab.CONFIG_PATH
START = datetime(2019, 3, 11, tzinfo=UTC)
_CSV_FIELDS = (
    "timestamp_utc",
    "bid_open",
    "bid_high",
    "bid_low",
    "bid_close",
    "ask_open",
    "ask_high",
    "ask_low",
    "ask_close",
    "volume",
)


def test_config_declares_exactly_seven_instruments() -> None:
    config = oanda_lab.load_oanda_alpha_lab_config()
    assert tuple(item.instrument_id for item in config.instruments) == (
        "EUR/USD.OANDA",
        "GBP/USD.OANDA",
        "AUD/USD.OANDA",
        "USD/CHF.OANDA",
        "USD/JPY.OANDA",
        "USD/CAD.OANDA",
        "NZD/USD.OANDA",
    )
    assert config.development_start_utc == START
    assert config.development_end_exclusive_utc == datetime(
        2023, 4, 11, tzinfo=UTC
    )


def test_config_semantic_hash_is_deterministic() -> None:
    first = oanda_lab.load_oanda_alpha_lab_config()
    second = oanda_lab.load_oanda_alpha_lab_config()
    assert first.semantic_sha256 == second.semantic_sha256
    assert len(first.semantic_sha256) == 64


def test_config_rejects_wrong_instrument_list(tmp_path: Path) -> None:
    document = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    document["instruments"] = document["instruments"][:-1]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    with pytest.raises(oanda_lab.OandaAlphaLabConfigError):
        oanda_lab.load_oanda_alpha_lab_config(path)


def test_config_rejects_fill_or_interpolation_flip(tmp_path: Path) -> None:
    document = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    document["missing_observation_policy"]["fills_or_interpolation"] = True
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    with pytest.raises(oanda_lab.OandaAlphaLabConfigError):
        oanda_lab.load_oanda_alpha_lab_config(path)


def test_usdcad_and_nzdusd_oanda_specs_validate() -> None:
    for spec in (USDCAD_OANDA_SPEC, NZDUSD_OANDA_SPEC):
        spec.validate()
        assert spec.instrument_id.endswith(".OANDA")


def test_usdjpy_oanda_precision_is_three_decimals() -> None:
    assert USDJPY_OANDA_SPEC.price_precision == 3
    assert USDJPY_OANDA_SPEC.price_increment == "0.001"
    for spec in OANDA_ALPHA_LAB_SPECS:
        if spec is not USDJPY_OANDA_SPEC:
            assert spec.price_precision == 5
            assert spec.price_increment == "0.00001"


def test_instrument_spec_rejects_unknown_source_suffix() -> None:
    bad = InstrumentSpec(
        dataset_symbol="EURUSD",
        instrument_id="EUR/USD.UNKNOWN",
        base_currency="EUR",
        quote_currency="USD",
        price_precision=5,
        price_increment="0.00001",
        size_precision=8,
        size_increment="0.00000001",
        session_policy_id="dukascopy-fx-ny-close-v1",
    )
    with pytest.raises(InstrumentSpecValidationError):
        bad.validate()


def _write_processed_csv(path: Path, spec: InstrumentSpec, minutes: int) -> None:
    quantum = Decimal(1).scaleb(-spec.price_precision)
    base = Decimal("110") if spec.price_precision == 3 else Decimal("1.1")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for index in range(minutes):
            timestamp = START + timedelta(minutes=index)

            def fmt(value: Decimal) -> str:
                return str(value.quantize(quantum))

            writer.writerow(
                {
                    "timestamp_utc": timestamp.isoformat().replace("+00:00", "Z"),
                    "bid_open": fmt(base),
                    "bid_high": fmt(base + quantum * 10),
                    "bid_low": fmt(base - quantum * 10),
                    "bid_close": fmt(base + quantum * 5),
                    "ask_open": fmt(base + quantum * 20),
                    "ask_high": fmt(base + quantum * 30),
                    "ask_low": fmt(base + quantum * 10),
                    "ask_close": fmt(base + quantum * 25),
                    "volume": "1",
                }
            )


def test_canonicalize_converts_oanda_csv_to_nautilus_catalog(tmp_path: Path) -> None:
    spec = OANDA_ALPHA_LAB_SPECS[0]
    csv_path = tmp_path / "processed.csv"
    minutes = 120
    _write_processed_csv(csv_path, spec, minutes)
    output_root = tmp_path / "canonical"

    manifest_path = oanda_lab.canonicalize_oanda_instrument(
        processed_csv_path=csv_path,
        instrument_spec=spec,
        output_root=output_root,
        alpha_lab_config_sha256="a" * 64,
        start_utc=START,
        end_exclusive_utc=START + timedelta(minutes=minutes),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["instrument_id"] == spec.instrument_id
    assert manifest["ingestion_version"] == oanda_lab.OANDA_ALPHA_LAB_INGESTION_VERSION
    assert manifest["fills_or_interpolation"] is False
    assert manifest["qa"]["bid_bar_count"] == minutes
    assert manifest["qa"]["ask_bar_count"] == minutes
    assert manifest["qa"]["holdout_rows_admitted"] == 0

    catalog = ParquetDataCatalog(str(output_root / "catalog"))
    instruments = catalog.instruments([spec.instrument_id])
    assert len(instruments) == 1
    bid_type = spec.bar_type(minutes=1, side="BID", aggregation="EXTERNAL")
    bars = catalog.query_bars([bid_type])
    assert len(bars) == minutes


def test_canonicalize_drops_rows_outside_declared_range_without_filling(
    tmp_path: Path,
) -> None:
    """Rows outside [start, end) are excluded outright -- never filled,
    never included as validation/holdout leakage."""

    spec = OANDA_ALPHA_LAB_SPECS[0]
    csv_path = tmp_path / "processed.csv"
    minutes = 120
    _write_processed_csv(csv_path, spec, minutes)
    output_root = tmp_path / "canonical"
    narrow_minutes = 60

    oanda_lab.canonicalize_oanda_instrument(
        processed_csv_path=csv_path,
        instrument_spec=spec,
        output_root=output_root,
        alpha_lab_config_sha256="a" * 64,
        start_utc=START,
        end_exclusive_utc=START + timedelta(minutes=narrow_minutes),
    )
    catalog = ParquetDataCatalog(str(output_root / "catalog"))
    bid_type = spec.bar_type(minutes=1, side="BID", aggregation="EXTERNAL")
    bars = catalog.query_bars([bid_type])
    assert len(bars) == narrow_minutes


def test_canonicalize_refuses_to_overwrite_existing_root(tmp_path: Path) -> None:
    spec = OANDA_ALPHA_LAB_SPECS[0]
    csv_path = tmp_path / "processed.csv"
    _write_processed_csv(csv_path, spec, 60)
    output_root = tmp_path / "canonical"
    oanda_lab.canonicalize_oanda_instrument(
        processed_csv_path=csv_path,
        instrument_spec=spec,
        output_root=output_root,
        alpha_lab_config_sha256="a" * 64,
        start_utc=START,
        end_exclusive_utc=START + timedelta(minutes=60),
    )
    with pytest.raises(Exception):  # noqa: B017 - fail-closed re-run guard
        oanda_lab.canonicalize_oanda_instrument(
            processed_csv_path=csv_path,
            instrument_spec=spec,
            output_root=output_root,
            alpha_lab_config_sha256="a" * 64,
            start_utc=START,
            end_exclusive_utc=START + timedelta(minutes=60),
        )


def _canonicalize_and_derive(
    root: Path, spec: InstrumentSpec, whole_days: int
) -> Path:
    """``requested_utc_range`` is whole-UTC-day granular (matching the
    Dukascopy convention derive_instrument_bars() expects), so every
    derive-bars-dependent fixture must span an exact number of whole days."""

    minutes = whole_days * 24 * 60
    csv_path = root / f"{spec.dataset_symbol}.csv"
    _write_processed_csv(csv_path, spec, minutes)
    canon_root = root / "canonical" / spec.dataset_symbol
    config = oanda_lab.load_oanda_alpha_lab_config()
    oanda_lab.canonicalize_oanda_instrument(
        processed_csv_path=csv_path,
        instrument_spec=spec,
        output_root=canon_root,
        alpha_lab_config_sha256=config.semantic_sha256,
        start_utc=START,
        end_exclusive_utc=START + timedelta(minutes=minutes),
    )
    derive_instrument_bars(canon_root, spec)
    return canon_root


def test_m30_h1_h4_derive_from_canonicalized_oanda_data(tmp_path: Path) -> None:
    spec = OANDA_ALPHA_LAB_SPECS[0]
    canon_root = _canonicalize_and_derive(tmp_path, spec, 1)
    catalog = ParquetDataCatalog(str(canon_root / "catalog"))
    for minutes, expected_count in ((30, 48), (60, 24), (240, 6)):
        bar_type = spec.bar_type(minutes=minutes, side="BID", aggregation="INTERNAL")
        assert len(catalog.query_bars([bar_type])) == expected_count


def test_readiness_freeze_and_alpha_lab_loading_round_trip(tmp_path: Path) -> None:
    development_roots = {}
    for spec in OANDA_ALPHA_LAB_SPECS:
        development_roots[spec.instrument_id] = _canonicalize_and_derive(
            tmp_path, spec, 1
        )

    readiness_path = oanda_lab.freeze_oanda_alpha_lab_readiness(
        development_roots=development_roots,
        output_root=tmp_path / "readiness",
    )
    document = json.loads(readiness_path.read_text(encoding="utf-8"))
    assert document["research_ready"] is True
    assert document["holdout_accessed"] is False
    assert document["holdout_rows_admitted"] == 0
    assert len(document["ordered_instrument_ids"]) == 7
    assert document["ordered_instrument_ids"] == sorted(
        document["ordered_instrument_ids"]
    )

    from ftmoquant.research.alpha_lab.data import load_alpha_lab_dataset

    dataset = load_alpha_lab_dataset(
        readiness_path=readiness_path,
        development_root_dir=tmp_path / "canonical",
        timeframe="M30",
        source="oanda",
    )
    assert dataset.instrument_ids == tuple(sorted(development_roots))
    assert dataset.close.shape[1] == 7
    assert not dataset.close.isna().any().any()
    assert str(dataset.close.index.tz) == "UTC"


def test_readiness_freeze_requires_all_seven_instruments(tmp_path: Path) -> None:
    spec = OANDA_ALPHA_LAB_SPECS[0]
    root = _canonicalize_and_derive(tmp_path, spec, 1)
    with pytest.raises(oanda_lab.OandaAlphaLabReadinessError):
        oanda_lab.freeze_oanda_alpha_lab_readiness(
            development_roots={spec.instrument_id: root},
            output_root=tmp_path / "readiness",
        )


def test_readiness_document_and_config_hashes_are_deterministic(
    tmp_path: Path,
) -> None:
    development_roots = {}
    for spec in OANDA_ALPHA_LAB_SPECS:
        development_roots[spec.instrument_id] = _canonicalize_and_derive(
            tmp_path, spec, 1
        )
    first = oanda_lab.freeze_oanda_alpha_lab_readiness(
        development_roots=development_roots, output_root=tmp_path / "r1"
    )
    second = oanda_lab.freeze_oanda_alpha_lab_readiness(
        development_roots=development_roots, output_root=tmp_path / "r2"
    )
    first_doc = json.loads(first.read_text(encoding="utf-8"))
    second_doc = json.loads(second.read_text(encoding="utf-8"))
    assert first_doc["semantic_sha256"] == second_doc["semantic_sha256"]
