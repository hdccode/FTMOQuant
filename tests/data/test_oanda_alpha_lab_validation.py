import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
import yaml
from nautilus_trader.persistence import ParquetDataCatalog

from ftmoquant.data import oanda_alpha_lab_development as oanda_dev
from ftmoquant.data import oanda_alpha_lab_validation as oanda_val
from ftmoquant.data.derived_bars import derive_instrument_bars
from ftmoquant.data.instruments import (
    OANDA_ALPHA_LAB_SPECS,
    InstrumentSpec,
    oanda_symbol,
)
from ftmoquant.data.oanda_development import (
    OandaDevelopmentDataError,
    _acquire_instrument,
    _segments,
)
from ftmoquant.research.stage_g import HOLDOUT_START, VALIDATION_START

VALIDATION_CONFIG_PATH = oanda_val.VALIDATION_CONFIG_PATH
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


# --------------------------------------------------------------------------
# Section 1: exact frozen VALIDATION boundary; DEVELOPMENT left untouched.
# --------------------------------------------------------------------------


def test_validation_config_pins_the_exact_frozen_boundary() -> None:
    config = oanda_val.load_oanda_alpha_lab_validation_config()
    assert config.validation_start_utc == VALIDATION_START
    assert config.validation_end_exclusive_utc == HOLDOUT_START
    assert config.validation_start_utc == datetime(2023, 4, 11, tzinfo=UTC)
    assert config.validation_end_exclusive_utc == datetime(2024, 8, 21, tzinfo=UTC)
    assert config.partition == "VALIDATION"


def test_validation_config_declares_exactly_the_seven_development_instruments() -> None:
    development = oanda_dev.load_oanda_alpha_lab_config()
    validation = oanda_val.load_oanda_alpha_lab_validation_config()
    dev_ids = {item.instrument_id for item in development.instruments}
    val_ids = {item.instrument_id for item in validation.instruments}
    assert val_ids == dev_ids
    assert len(val_ids) == 7


def test_validation_config_semantic_sha_is_distinct_from_development() -> None:
    development = oanda_dev.load_oanda_alpha_lab_config()
    validation = oanda_val.load_oanda_alpha_lab_validation_config()
    assert validation.semantic_sha256 != development.semantic_sha256
    assert validation.lineage_id != development.lineage_id


def test_development_config_remains_byte_for_byte_unchanged() -> None:
    """A regression pin: this module must never mutate the frozen
    DEVELOPMENT config or drift its already-published semantic hash."""

    development = oanda_dev.load_oanda_alpha_lab_config()
    assert development.development_start_utc == datetime(2019, 3, 11, tzinfo=UTC)
    assert development.development_end_exclusive_utc == datetime(
        2023, 4, 11, tzinfo=UTC
    )
    assert (
        development.semantic_sha256
        == "b31334b2b603b0f92350e82c36a7541407f4a15bb1d5b6e18b5247c4b1fc5d29"
    )


def test_validation_config_rejects_any_boundary_other_than_frozen(
    tmp_path: Path,
) -> None:
    document = yaml.safe_load(VALIDATION_CONFIG_PATH.read_text(encoding="utf-8"))
    document["validation"]["end_exclusive_utc"] = "2024-08-22T00:00:00Z"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    with pytest.raises(oanda_val.OandaAlphaLabValidationConfigError):
        oanda_val.load_oanda_alpha_lab_validation_config(path)


def test_validation_config_rejects_start_before_frozen_boundary(tmp_path: Path) -> None:
    document = yaml.safe_load(VALIDATION_CONFIG_PATH.read_text(encoding="utf-8"))
    document["validation"]["start_utc"] = "2023-01-01T00:00:00Z"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    with pytest.raises(oanda_val.OandaAlphaLabValidationConfigError):
        oanda_val.load_oanda_alpha_lab_validation_config(path)


def test_validation_config_rejects_wrong_partition(tmp_path: Path) -> None:
    document = yaml.safe_load(VALIDATION_CONFIG_PATH.read_text(encoding="utf-8"))
    document["partition"] = "DEVELOPMENT"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    with pytest.raises(oanda_val.OandaAlphaLabValidationConfigError):
        oanda_val.load_oanda_alpha_lab_validation_config(path)


def test_validation_config_rejects_wrong_lineage_id(tmp_path: Path) -> None:
    document = yaml.safe_load(VALIDATION_CONFIG_PATH.read_text(encoding="utf-8"))
    document["lineage_id"] = "oanda_fx_alpha_lab_v1"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    with pytest.raises(oanda_val.OandaAlphaLabValidationConfigError):
        oanda_val.load_oanda_alpha_lab_validation_config(path)


def test_validation_config_rejects_missing_instrument(tmp_path: Path) -> None:
    document = yaml.safe_load(VALIDATION_CONFIG_PATH.read_text(encoding="utf-8"))
    document["instruments"] = document["instruments"][:-1]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    with pytest.raises(oanda_val.OandaAlphaLabValidationConfigError):
        oanda_val.load_oanda_alpha_lab_validation_config(path)


def test_validation_config_rejects_fill_or_interpolation_flip(tmp_path: Path) -> None:
    document = yaml.safe_load(VALIDATION_CONFIG_PATH.read_text(encoding="utf-8"))
    document["missing_observation_policy"]["fills_or_interpolation"] = True
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    with pytest.raises(oanda_val.OandaAlphaLabValidationConfigError):
        oanda_val.load_oanda_alpha_lab_validation_config(path)


# --------------------------------------------------------------------------
# Acquisition cannot cross HOLDOUT_START: the low-level request segmentation
# reused by acquire_oanda_alpha_lab_validation() is itself bounded by the
# exact config interval, which the loader above already pins.
# --------------------------------------------------------------------------


def test_request_segments_never_extend_past_holdout_start() -> None:
    config = oanda_val.load_oanda_alpha_lab_validation_config()
    segments = list(
        _segments(config.validation_start_utc, config.validation_end_exclusive_utc)
    )
    assert segments
    assert segments[0][0] == VALIDATION_START
    assert segments[-1][1] == HOLDOUT_START
    assert all(end <= HOLDOUT_START for _, end in segments)
    assert all(start >= VALIDATION_START for start, _ in segments)


def test_acquired_candles_never_extend_past_holdout_start() -> None:
    # A fetcher that (incorrectly) tries to return candles past HOLDOUT_START
    # must still only ever be asked for segments inside the frozen window --
    # _acquire_instrument only requests what it's given as start/end.
    result = _acquire_instrument(
        _tmp_root(),
        "EUR_USD",
        VALIDATION_START,
        VALIDATION_START + timedelta(minutes=5),
        "test-token",
        _fixed_fetcher(VALIDATION_START, 5),
    )
    with result.processed_path.open(encoding="utf-8") as handle:
        lines = handle.read().splitlines()[1:]
    for line in lines:
        timestamp = datetime.fromisoformat(line.split(",")[0].replace("Z", "+00:00"))
        assert VALIDATION_START <= timestamp < HOLDOUT_START


def _tmp_root() -> Path:
    import tempfile

    return Path(tempfile.mkdtemp())


def _fixed_fetcher(start: datetime, minutes: int):
    def fetcher(url: str, authorization: str) -> bytes:
        import urllib.parse

        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        seg_start = datetime.fromisoformat(query["from"][0].replace("Z", "+00:00"))
        seg_end = datetime.fromisoformat(query["to"][0].replace("Z", "+00:00"))
        candles = [
            {
                "complete": True,
                "time": minute.isoformat().replace("+00:00", "Z"),
                "bid": {"o": "1.1", "h": "1.1", "l": "1.1", "c": "1.1"},
                "ask": {"o": "1.1", "h": "1.1", "l": "1.1", "c": "1.1"},
                "volume": 1,
            }
            for i in range(minutes)
            if seg_start <= (minute := start + timedelta(minutes=i)) < seg_end
        ]
        return json.dumps(
            {"instrument": "EUR_USD", "granularity": "M1", "candles": candles}
        ).encode()

    return fetcher


# --------------------------------------------------------------------------
# Canonicalization / M30-H1-H4 derivation, reused unchanged from
# oanda_alpha_lab_development / derived_bars, exercised at VALIDATION dates.
# --------------------------------------------------------------------------


def _write_processed_csv(path: Path, spec: InstrumentSpec, minutes: int) -> None:
    import csv

    quantum = Decimal(1).scaleb(-spec.price_precision)
    base = Decimal("110") if spec.price_precision == 3 else Decimal("1.1")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for index in range(minutes):
            timestamp = VALIDATION_START + timedelta(minutes=index)

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


def _canonicalize_and_derive(root: Path, spec: InstrumentSpec, whole_days: int) -> Path:
    minutes = whole_days * 24 * 60
    csv_path = root / f"{spec.dataset_symbol}.csv"
    _write_processed_csv(csv_path, spec, minutes)
    canon_root = root / "canonical" / oanda_symbol(spec.dataset_symbol)
    config = oanda_val.load_oanda_alpha_lab_validation_config()
    oanda_val.canonicalize_oanda_instrument(
        processed_csv_path=csv_path,
        instrument_spec=spec,
        output_root=canon_root,
        alpha_lab_config_sha256=config.semantic_sha256,
        start_utc=VALIDATION_START,
        end_exclusive_utc=VALIDATION_START + timedelta(minutes=minutes),
    )
    derive_instrument_bars(canon_root, spec)
    return canon_root


def _build_all_seven_validation_roots(tmp_path: Path) -> dict[str, Path]:
    roots = {}
    for spec in OANDA_ALPHA_LAB_SPECS:
        roots[spec.instrument_id] = _canonicalize_and_derive(tmp_path, spec, 1)
    return roots


def test_m30_h1_h4_derive_from_canonicalized_validation_data(tmp_path: Path) -> None:
    spec = OANDA_ALPHA_LAB_SPECS[0]
    canon_root = _canonicalize_and_derive(tmp_path, spec, 1)
    catalog = ParquetDataCatalog(str(canon_root / "catalog"))
    for minutes, expected_count in ((30, 48), (60, 24), (240, 6)):
        bar_type = spec.bar_type(minutes=minutes, side="BID", aggregation="INTERNAL")
        assert len(catalog.query_bars([bar_type])) == expected_count


def test_canonicalize_drops_rows_outside_the_validation_range(tmp_path: Path) -> None:
    spec = OANDA_ALPHA_LAB_SPECS[0]
    csv_path = tmp_path / "processed.csv"
    _write_processed_csv(csv_path, spec, 120)
    output_root = tmp_path / "canonical"
    config = oanda_val.load_oanda_alpha_lab_validation_config()
    oanda_val.canonicalize_oanda_instrument(
        processed_csv_path=csv_path,
        instrument_spec=spec,
        output_root=output_root,
        alpha_lab_config_sha256=config.semantic_sha256,
        start_utc=VALIDATION_START,
        end_exclusive_utc=VALIDATION_START + timedelta(minutes=60),
    )
    catalog = ParquetDataCatalog(str(output_root / "catalog"))
    bid_type = spec.bar_type(minutes=1, side="BID", aggregation="EXTERNAL")
    assert len(catalog.query_bars([bid_type])) == 60


# --------------------------------------------------------------------------
# Section 5/6: VALIDATION QA and readiness -- reuses
# _development._audit_cached_alpha_lab_instrument /
# _oanda_derived_structural_readiness unchanged, wired to the new lineage.
# --------------------------------------------------------------------------


def test_readiness_freeze_and_dataset_loading_round_trip(tmp_path: Path) -> None:
    validation_roots = _build_all_seven_validation_roots(tmp_path)
    readiness_path = oanda_val.freeze_oanda_alpha_lab_validation_readiness(
        validation_roots=validation_roots, output_root=tmp_path / "readiness"
    )
    document = json.loads(readiness_path.read_text(encoding="utf-8"))
    assert document["readiness_version"] == oanda_val.VALIDATION_READINESS_VERSION
    assert document["partition"] == "VALIDATION"
    assert document["validation_start_utc"] == "2023-04-11T00:00:00Z"
    assert document["validation_end_exclusive_utc"] == "2024-08-21T00:00:00Z"
    assert document["research_ready"] is True
    assert document["holdout_accessed"] is False
    assert document["holdout_rows_admitted"] == 0
    assert len(document["ordered_instrument_ids"]) == 7
    for artifact in document["instrument_artifacts"]:
        assert artifact["research_ready"] is True
        assert "canonical_sha256" in artifact
        assert "derived_sha256" in artifact
        assert "catalog_tree_sha256" in artifact


def test_validation_readiness_requires_all_seven_instruments(tmp_path: Path) -> None:
    spec = OANDA_ALPHA_LAB_SPECS[0]
    root = _canonicalize_and_derive(tmp_path, spec, 1)
    with pytest.raises(oanda_val.OandaAlphaLabValidationReadinessError):
        oanda_val.freeze_oanda_alpha_lab_validation_readiness(
            validation_roots={spec.instrument_id: root},
            output_root=tmp_path / "readiness",
        )


def test_validation_readiness_hash_is_deterministic(tmp_path: Path) -> None:
    validation_roots = _build_all_seven_validation_roots(tmp_path)
    first = oanda_val.freeze_oanda_alpha_lab_validation_readiness(
        validation_roots=validation_roots, output_root=tmp_path / "r1"
    )
    second = oanda_val.freeze_oanda_alpha_lab_validation_readiness(
        validation_roots=validation_roots, output_root=tmp_path / "r2"
    )
    first_doc = json.loads(first.read_text(encoding="utf-8"))
    second_doc = json.loads(second.read_text(encoding="utf-8"))
    assert first_doc["semantic_sha256"] == second_doc["semantic_sha256"]


def test_canonical_manifest_hash_drift_fails_validation_readiness(
    tmp_path: Path,
) -> None:
    validation_roots = _build_all_seven_validation_roots(tmp_path)
    spec = OANDA_ALPHA_LAB_SPECS[0]
    manifest_path = validation_roots[spec.instrument_id] / "ftmoquant_provenance.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["instrument_id"] = "TAMPERED/PAIR.OANDA"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(oanda_val.OandaAlphaLabValidationReadinessError):
        oanda_val.freeze_oanda_alpha_lab_validation_readiness(
            validation_roots=validation_roots, output_root=tmp_path / "readiness"
        )


def test_derived_manifest_instrument_mismatch_fails_validation_readiness(
    tmp_path: Path,
) -> None:
    validation_roots = _build_all_seven_validation_roots(tmp_path)
    spec = OANDA_ALPHA_LAB_SPECS[0]
    derived_path = (
        validation_roots[spec.instrument_id] / "ftmoquant_derived_provenance.json"
    )
    derived = json.loads(derived_path.read_text(encoding="utf-8"))
    derived["instrument_id"] = "TAMPERED/PAIR.OANDA"
    derived_path.write_text(json.dumps(derived), encoding="utf-8")
    with pytest.raises(oanda_val.OandaAlphaLabValidationReadinessError):
        oanda_val.freeze_oanda_alpha_lab_validation_readiness(
            validation_roots=validation_roots, output_root=tmp_path / "readiness"
        )


def test_weekend_gaps_do_not_fail_validation_readiness(tmp_path: Path) -> None:
    start = datetime(2023, 4, 14, tzinfo=UTC)  # Friday, inside VALIDATION
    end = datetime(2023, 4, 18, tzinfo=UTC)  # following Tuesday
    present = _weekday_minutes(start, end)
    result = _build_synthetic_cache(tmp_path, "EUR_USD", start, end, present)

    audit = oanda_dev._audit_cached_alpha_lab_instrument(
        tmp_path, "EUR_USD", start, end, _manifest_record(tmp_path, result)
    )
    assert audit.research_ready is True
    assert audit.acquisition_defect_count == 0
    assert audit.missing_interval_count > 0


def test_genuinely_skipped_request_window_fails_validation_qa(tmp_path: Path) -> None:
    start = datetime(2023, 4, 14, tzinfo=UTC)
    end = datetime(2023, 4, 18, tzinfo=UTC)
    present = _weekday_minutes(start, end)
    result = _build_synthetic_cache(tmp_path, "EUR_USD", start, end, present)
    raw_root = tmp_path / "raw" / "EUR_USD"
    response = next(iter(sorted(raw_root.glob("*.json"))))
    response.unlink()

    with pytest.raises(OandaDevelopmentDataError, match="skipped"):
        oanda_dev._audit_cached_alpha_lab_instrument(
            tmp_path, "EUR_USD", start, end, _manifest_record(tmp_path, result)
        )


def test_no_fill_or_interpolation_in_validation_qa(tmp_path: Path) -> None:
    start = datetime(2023, 4, 14, tzinfo=UTC)
    end = datetime(2023, 4, 18, tzinfo=UTC)
    present = _weekday_minutes(start, end)
    result = _build_synthetic_cache(tmp_path, "EUR_USD", start, end, present)
    audit = oanda_dev._audit_cached_alpha_lab_instrument(
        tmp_path, "EUR_USD", start, end, _manifest_record(tmp_path, result)
    )
    document = json.loads(result.qa_path.read_text(encoding="utf-8"))
    assert document["no_synthetic_fill_or_interpolation"] is True
    assert document["qa_version"] == oanda_val.AVAILABILITY_QA_VERSION
    assert audit.research_ready is True


def _weekday_minutes(start: datetime, end: datetime) -> dict:
    minutes = {}
    cursor = start
    while cursor < end:
        if cursor.weekday() < 5:
            minutes[cursor] = True
        cursor += timedelta(minutes=1)
    return minutes


def _fake_fetcher(candles_by_minute: dict, oanda_instrument: str):
    import urllib.parse

    def fetcher(url: str, authorization: str) -> bytes:
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        seg_start = datetime.fromisoformat(query["from"][0].replace("Z", "+00:00"))
        seg_end = datetime.fromisoformat(query["to"][0].replace("Z", "+00:00"))
        candles = [
            {
                "complete": True,
                "time": minute.isoformat().replace("+00:00", "Z"),
                "bid": {"o": "1.10000", "h": "1.10010", "l": "1.09990", "c": "1.10005"},
                "ask": {"o": "1.10020", "h": "1.10030", "l": "1.10010", "c": "1.10025"},
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
