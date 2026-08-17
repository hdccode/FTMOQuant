import csv
import json
import urllib.parse
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
from ftmoquant.data.oanda_development import (
    OandaDevelopmentDataError,
    _acquire_instrument,
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


# --------------------------------------------------------------------------
# Corrected cache-only availability QA: acquisition-completeness defects
# (must fail research_ready) kept strictly separate from provider-
# observation availability gaps like weekends (must NOT fail research_ready).
# --------------------------------------------------------------------------

_BID = {"o": "1.10000", "h": "1.10010", "l": "1.09990", "c": "1.10005"}
_ASK = {"o": "1.10020", "h": "1.10030", "l": "1.10010", "c": "1.10025"}


def _fake_fetcher(candles_by_minute: dict, oanda_instrument: str):
    def fetcher(url: str, authorization: str) -> bytes:
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        seg_start = datetime.fromisoformat(query["from"][0].replace("Z", "+00:00"))
        seg_end = datetime.fromisoformat(query["to"][0].replace("Z", "+00:00"))
        candles = [
            {
                "complete": True,
                "time": minute.isoformat().replace("+00:00", "Z"),
                "bid": _BID,
                "ask": _ASK,
                "volume": 1,
            }
            for minute in sorted(candles_by_minute)
            if seg_start <= minute < seg_end
        ]
        payload = {
            "instrument": oanda_instrument,
            "granularity": "M1",
            "candles": candles,
        }
        return json.dumps(payload).encode()

    return fetcher


def _weekday_minutes(start: datetime, end: datetime) -> dict:
    """Every minute in [start, end) except Sat/Sun -- a realistic FX
    weekend gap, present in every real cache."""

    minutes = {}
    cursor = start
    while cursor < end:
        if cursor.weekday() < 5:
            minutes[cursor] = True
        cursor += timedelta(minutes=1)
    return minutes


def _build_synthetic_cache(
    root: Path, oanda_instrument: str, start: datetime, end: datetime, present: dict
):
    fetcher = _fake_fetcher(present, oanda_instrument)
    return _acquire_instrument(
        root, oanda_instrument, start, end, "test-token", fetcher
    )


def _manifest_record(root: Path, result) -> dict:
    return {
        "oanda_instrument": result.instrument,
        "processed_path": str(result.processed_path.relative_to(root)),
        "processed_sha256": result.processed_sha256,
        "qa_path": str(result.qa_path.relative_to(root)),
    }


def test_weekend_gaps_do_not_fail_readiness(tmp_path: Path) -> None:
    start = datetime(2023, 1, 6, tzinfo=UTC)  # Friday
    end = datetime(2023, 1, 10, tzinfo=UTC)  # following Tuesday
    present = _weekday_minutes(start, end)
    result = _build_synthetic_cache(tmp_path, "EUR_USD", start, end, present)

    audit = oanda_lab._audit_cached_alpha_lab_instrument(
        tmp_path, "EUR_USD", start, end, _manifest_record(tmp_path, result)
    )

    assert audit.research_ready is True
    assert audit.acquisition_defect_count == 0
    assert audit.missing_interval_count > 0
    assert audit.missing_minutes > 0
    document = json.loads(result.qa_path.read_text(encoding="utf-8"))
    assert document["qa_version"] == oanda_lab.AVAILABILITY_QA_VERSION
    assert document["research_ready"] is True
    assert document["provider_observation_availability"]["missing_interval_count"] > 0


def test_genuinely_skipped_request_window_fails(tmp_path: Path) -> None:
    start = datetime(2023, 1, 6, tzinfo=UTC)
    end = datetime(2023, 1, 10, tzinfo=UTC)
    present = _weekday_minutes(start, end)
    result = _build_synthetic_cache(tmp_path, "EUR_USD", start, end, present)
    raw_root = tmp_path / "raw" / "EUR_USD"
    response = next(iter(sorted(raw_root.glob("*.json"))))
    response.unlink()

    with pytest.raises(OandaDevelopmentDataError, match="skipped"):
        oanda_lab._audit_cached_alpha_lab_instrument(
            tmp_path, "EUR_USD", start, end, _manifest_record(tmp_path, result)
        )


def test_duplicated_request_window_fails(tmp_path: Path) -> None:
    start = datetime(2023, 1, 6, tzinfo=UTC)
    end = datetime(2023, 1, 10, tzinfo=UTC)
    present = _weekday_minutes(start, end)
    result = _build_synthetic_cache(tmp_path, "EUR_USD", start, end, present)
    raw_root = tmp_path / "raw" / "EUR_USD"
    response = next(iter(sorted(raw_root.glob("*.json"))))
    extra = response.with_name("999999_extra.json")
    extra.write_bytes(response.read_bytes())

    with pytest.raises(OandaDevelopmentDataError, match="skipped|extra"):
        oanda_lab._audit_cached_alpha_lab_instrument(
            tmp_path, "EUR_USD", start, end, _manifest_record(tmp_path, result)
        )


def test_raw_request_mismatch_fails(tmp_path: Path) -> None:
    start = datetime(2023, 1, 6, tzinfo=UTC)
    end = datetime(2023, 1, 10, tzinfo=UTC)
    present = _weekday_minutes(start, end)
    result = _build_synthetic_cache(tmp_path, "EUR_USD", start, end, present)
    raw_root = tmp_path / "raw" / "EUR_USD"
    request_path = next(iter(sorted(raw_root.glob("*.request.json"))))
    mutated = json.loads(request_path.read_text(encoding="utf-8"))
    mutated["url"] = mutated["url"] + "&tampered=true"
    request_path.write_text(json.dumps(mutated), encoding="utf-8")

    with pytest.raises(OandaDevelopmentDataError, match="does not match"):
        oanda_lab._audit_cached_alpha_lab_instrument(
            tmp_path, "EUR_USD", start, end, _manifest_record(tmp_path, result)
        )


def test_processed_hash_drift_fails(tmp_path: Path) -> None:
    """Any processed-CSV tampering must fail closed. A row-level edit is
    caught by the raw/processed correspondence check before the explicit
    processed_sha256 comparison is even reached -- both guard the same
    tamper-evidence contract, so either failing proves it."""

    start = datetime(2023, 1, 6, tzinfo=UTC)
    end = datetime(2023, 1, 10, tzinfo=UTC)
    present = _weekday_minutes(start, end)
    result = _build_synthetic_cache(tmp_path, "EUR_USD", start, end, present)
    content = result.processed_path.read_text(encoding="utf-8")
    tampered = content.replace("1.10005", "1.10099", 1)
    assert tampered != content
    result.processed_path.write_text(tampered, encoding="utf-8")

    with pytest.raises(OandaDevelopmentDataError):
        oanda_lab._audit_cached_alpha_lab_instrument(
            tmp_path, "EUR_USD", start, end, _manifest_record(tmp_path, result)
        )


def test_no_fill_or_interpolation_and_full_missing_reporting(tmp_path: Path) -> None:
    start = datetime(2023, 1, 6, tzinfo=UTC)
    end = datetime(2023, 1, 10, tzinfo=UTC)
    present = _weekday_minutes(start, end)
    result = _build_synthetic_cache(tmp_path, "EUR_USD", start, end, present)

    audit = oanda_lab._audit_cached_alpha_lab_instrument(
        tmp_path, "EUR_USD", start, end, _manifest_record(tmp_path, result)
    )

    total_minutes = int((end - start).total_seconds() // 60)
    assert audit.processed_row_count == len(present)
    assert audit.processed_row_count + audit.missing_minutes == total_minutes
    document = json.loads(result.qa_path.read_text(encoding="utf-8"))
    assert document["no_synthetic_fill_or_interpolation"] is True
    assert (
        document["provider_observation_availability"]["filled_or_interpolated_minutes"]
        == 0
    )
    buckets = document["provider_observation_availability"][
        "missing_interval_duration_buckets"
    ]
    assert sum(buckets.values()) == document["provider_observation_availability"][
        "missing_interval_count"
    ]


# --------------------------------------------------------------------------
# Tiny CLI wrapper for canonicalize_oanda_instrument() -- no new conversion
# logic, just argument parsing + one direct call.
# --------------------------------------------------------------------------


def test_canonicalize_cli_calls_canonicalize_oanda_instrument_directly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = OANDA_ALPHA_LAB_SPECS[0]
    csv_path = tmp_path / "processed.csv"
    _write_processed_csv(csv_path, spec, 120)
    output_root = tmp_path / "canonical"

    oanda_lab.canonicalize_main(
        [
            "--processed-csv",
            str(csv_path),
            "--instrument-id",
            spec.instrument_id,
            "--output-root",
            str(output_root),
        ]
    )

    manifest_path = output_root / oanda_lab.PARENT_MANIFEST_FILENAME
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["instrument_id"] == spec.instrument_id
    assert manifest["qa"]["bid_bar_count"] == 120

    captured = capsys.readouterr()
    assert str(manifest_path) in captured.out
    assert f"instrument_id={spec.instrument_id}" in captured.out
    assert "bid_bar_count=120" in captured.out


def test_canonicalize_cli_rejects_unknown_instrument_id(tmp_path: Path) -> None:
    csv_path = tmp_path / "processed.csv"
    csv_path.write_text("timestamp_utc\n", encoding="utf-8")

    with pytest.raises(Exception):  # noqa: B017 - fail-closed on unknown instrument
        oanda_lab.canonicalize_main(
            [
                "--processed-csv",
                str(csv_path),
                "--instrument-id",
                "XXX/YYY.OANDA",
                "--output-root",
                str(tmp_path / "canonical"),
            ]
        )


def test_canonicalize_cli_fails_if_output_root_already_populated(
    tmp_path: Path,
) -> None:
    spec = OANDA_ALPHA_LAB_SPECS[0]
    csv_path = tmp_path / "processed.csv"
    _write_processed_csv(csv_path, spec, 60)
    output_root = tmp_path / "canonical"

    oanda_lab.canonicalize_main(
        [
            "--processed-csv",
            str(csv_path),
            "--instrument-id",
            spec.instrument_id,
            "--output-root",
            str(output_root),
        ]
    )
    with pytest.raises(OandaDevelopmentDataError, match="already contains"):
        oanda_lab.canonicalize_main(
            [
                "--processed-csv",
                str(csv_path),
                "--instrument-id",
                spec.instrument_id,
                "--output-root",
                str(output_root),
            ]
        )
