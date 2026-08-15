from __future__ import annotations

import csv
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml

from ftmoquant.research import (
    REGISTRY_COLUMNS,
    ExperimentRegistryValidationError,
    StrategySpecValidationError,
    canonical_spec_json,
    load_experiment_registry,
    load_trend_pullback_spec,
    strategy_config_sha256,
)

CONFIG_PATH = Path("config/strategies/trend_pullback_v1.yaml")
REGISTRY_PATH = Path("experiments/registry.csv")
EXPECTED_CONFIG_SHA256 = (
    "d21100fc8412f4d258efdc90b2f1a936c0eb27cd6f88081ae082f24ae6d4cc5e"
)


def test_loads_preregistered_baseline() -> None:
    spec = load_trend_pullback_spec(CONFIG_PATH)

    assert spec.strategy_id == "trend_pullback_v1"
    assert spec.version == "1.0.0"
    assert spec.status == "specified_not_run"
    assert spec.instrument.instrument_id == "EUR/USD.DUKASCOPY"
    assert spec.signal.trend.fast_ema_period == 50
    assert spec.signal.trend.slow_ema_period == 200
    assert spec.signal.pullback.ema_period == 20
    assert spec.signal.volatility.atr_period == 14
    assert spec.risk.risk_unit == Decimal("1.0")
    assert not spec.risk.compounding
    assert not spec.risk.ftmo_adjustments
    assert spec.research_protocol.parameter_family.permitted_variants == ()
    assert strategy_config_sha256(spec) == EXPECTED_CONFIG_SHA256


def test_canonical_hash_ignores_yaml_format_and_key_order(tmp_path: Path) -> None:
    original = load_trend_pullback_spec(CONFIG_PATH)
    reordered_path = tmp_path / "reordered.yaml"
    reordered_path.write_text(
        yaml.safe_dump(_read_config(), sort_keys=True), encoding="utf-8"
    )
    reordered = load_trend_pullback_spec(reordered_path)

    assert canonical_spec_json(reordered) == canonical_spec_json(original)
    assert strategy_config_sha256(reordered) == strategy_config_sha256(original)


@pytest.mark.parametrize(
    ("path", "value", "error"),
    [
        (("signal", "trend", "timeframe"), "1h", "must be 4h"),
        (("signal", "pullback", "ema_period"), 0, "positive integer"),
        (("exits", "stop_atr_multiple"), "0", "must be positive"),
        (("exits", "target_r_multiple"), ".nan", "finite decimal"),
        (
            ("research_protocol", "split", "validation_fraction"),
            "1.2",
            "strictly between 0 and 1",
        ),
        (
            ("research_protocol", "gates", "validation_ci_level"),
            float("nan"),
            "finite decimal",
        ),
        (("positions", "max_open_positions"), 0, "positive integer"),
    ],
)
def test_rejects_invalid_values(
    tmp_path: Path, path: tuple[str, ...], value: object, error: str
) -> None:
    config = _read_config()
    _set_nested(config, path, value)
    invalid_path = _write_config(tmp_path, config)

    with pytest.raises(StrategySpecValidationError, match=error):
        load_trend_pullback_spec(invalid_path)


def test_rejects_missing_and_unknown_fields(tmp_path: Path) -> None:
    missing = _read_config()
    del missing["execution"]["earliest_entry"]
    with pytest.raises(
        StrategySpecValidationError, match="missing fields: earliest_entry"
    ):
        load_trend_pullback_spec(_write_config(tmp_path, missing, "missing.yaml"))

    unknown = _read_config()
    unknown["signal"]["trend"]["lookahead"] = True
    with pytest.raises(
        StrategySpecValidationError, match="unexpected fields: lookahead"
    ):
        load_trend_pullback_spec(_write_config(tmp_path, unknown, "unknown.yaml"))


@pytest.mark.parametrize(
    ("path", "value", "error"),
    [
        (("signal", "trend", "fast_ema_period"), 200, "fast_ema_period"),
        (("risk", "compounding"), True, "compounding must be disabled"),
        (("risk", "ftmo_adjustments"), True, "FTMO adjustments"),
        (("positions", "pyramiding"), True, "pyramiding must be disabled"),
        (
            ("research_protocol", "split", "development_fraction"),
            "0.50",
            "fractions must sum to 1",
        ),
        (
            ("research_protocol", "parameter_family", "permitted_variants"),
            ["ema_40_160"],
            "cannot contain permitted variants",
        ),
    ],
)
def test_rejects_cross_field_inconsistency(
    tmp_path: Path, path: tuple[str, ...], value: object, error: str
) -> None:
    config = _read_config()
    _set_nested(config, path, value)

    with pytest.raises(StrategySpecValidationError, match=error):
        load_trend_pullback_spec(_write_config(tmp_path, config))


def test_registry_declares_no_runtime_evidence() -> None:
    entries = load_experiment_registry(REGISTRY_PATH)

    assert len(entries) == 2
    entry = next(item for item in entries if item.strategy_id == "trend_pullback_v1")
    assert entry.status == "specified_not_run"
    assert entry.strategy_config_sha256 == EXPECTED_CONFIG_SHA256
    assert entry.git_commit == ""
    assert entry.data_manifest_sha256 == ""
    assert entry.execution_profile_sha256 == ""
    assert entry.engine_version == ""
    assert entry.run_seed == ""
    assert entry.primary_metric_value == ""
    assert entry.decision == "not_evaluated"


def test_leo_preregistration_registry_row_declares_no_runtime_evidence() -> None:
    entries = load_experiment_registry(REGISTRY_PATH)
    entry = next(item for item in entries if item.strategy_id == "leo_gbpusd_v1")

    assert entry.status == "specified_not_run"
    assert entry.strategy_config_sha256 == (
        "5c89c825db340c49f10c18fdc3982b03f23ade718a5889e4e2ef662081abbf4c"
    )
    assert all(
        getattr(entry, field) == ""
        for field in (
            "git_commit",
            "data_manifest_sha256",
            "execution_profile_sha256",
            "engine_version",
            "run_seed",
            "started_at",
            "completed_at",
            "primary_metric_value",
        )
    )
    assert entry.decision == "not_evaluated"


def test_not_run_row_cannot_contain_result(tmp_path: Path) -> None:
    row = _registry_row()
    row["primary_metric_value"] = "0.25"
    path = _write_registry(tmp_path, row)

    with pytest.raises(
        ExperimentRegistryValidationError, match="cannot contain runtime evidence"
    ):
        load_experiment_registry(path)


def test_completed_row_requires_complete_finite_evidence(tmp_path: Path) -> None:
    row = _registry_row()
    row["status"] = "completed"
    row["decision"] = "pass"
    path = _write_registry(tmp_path, row)

    with pytest.raises(
        ExperimentRegistryValidationError,
        match="completed experiment is missing evidence",
    ):
        load_experiment_registry(path)


def test_completed_row_accepts_full_evidence(tmp_path: Path) -> None:
    row = _registry_row()
    row.update(
        {
            "status": "completed",
            "git_commit": "a" * 40,
            "data_manifest_sha256": "b" * 64,
            "execution_profile_sha256": "c" * 64,
            "engine_version": "2.0.0rc2",
            "run_seed": "1729",
            "started_at": "2026-08-13T12:00:00Z",
            "completed_at": "2026-08-13T12:10:00Z",
            "primary_metric_value": "0.10",
            "decision": "pass",
        }
    )

    entries = load_experiment_registry(_write_registry(tmp_path, row))

    assert entries[0].status == "completed"
    assert entries[0].git_commit == "a" * 40


def _read_config() -> dict[str, Any]:
    loaded = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return deepcopy(loaded)


def _set_nested(config: dict[str, Any], path: tuple[str, ...], value: object) -> None:
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def _write_config(
    tmp_path: Path, config: dict[str, Any], name: str = "invalid.yaml"
) -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _registry_row() -> dict[str, str]:
    with REGISTRY_PATH.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    return row


def _write_registry(tmp_path: Path, row: dict[str, str]) -> Path:
    path = tmp_path / "registry.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    return path
