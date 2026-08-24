"""Native AUD/CAD and EUR/JPY Batch-5 acquisition preparation only."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from ftmoquant.backtest.execution_harness import _sha256_tree
from ftmoquant.data.derived_bars import (
    DERIVED_MANIFEST_FILENAME,
    PARENT_MANIFEST_FILENAME,
    derive_instrument_bars,
)
from ftmoquant.data.instruments import OANDA_BATCH5_CROSS_SPECS, oanda_symbol
from ftmoquant.data.oanda_alpha_lab_development import (
    _audit_cached_alpha_lab_instrument,
    _oanda_derived_structural_readiness,
    canonicalize_oanda_instrument,
)
from ftmoquant.data.oanda_development import (
    InstrumentAcquisitionResult,
    OandaDevelopmentDataError,
    _access_token_from_environment,
    _acquire_instrument,
    _fetch_response,
    _reject_worktree,
    _write_idempotent_json,
)
from ftmoquant.research.alpha_lab.batch5_preregistration import (
    EXPECTED_PREREGISTRATION_SEMANTIC_SHA256,
    verify_preregistration,
)

CONFIG_PATH = Path("config/data/oanda_batch5_native_crosses_v1.yaml")
LINEAGE_ID = "oanda_batch5_native_crosses_v1"
MANIFEST_FILENAME = "oanda_batch5_native_crosses_acquisition_manifest.json"
PREREG_SHA = EXPECTED_PREREGISTRATION_SEMANTIC_SHA256


class OandaBatch5CrossConfigError(ValueError):
    """Raised when the frozen two-cross acquisition config drifts."""


@dataclass(frozen=True, slots=True)
class CrossConfig:
    semantic_sha256: str
    acquisition_start_utc: datetime
    development_start_utc: datetime
    development_end_exclusive_utc: datetime
    instruments: tuple[str, ...]


def _utc(value: object) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise OandaBatch5CrossConfigError("config timestamps must be UTC Z strings")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() is None or parsed.astimezone(UTC) != parsed:
        raise OandaBatch5CrossConfigError("config timestamps must be UTC")
    return parsed


def load_config(path: Path = CONFIG_PATH) -> CrossConfig:
    verify_preregistration()
    try:
        document: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise OandaBatch5CrossConfigError("could not load config") from error
    if document.get("lineage_id") != LINEAGE_ID or document.get("version") != 1:
        raise OandaBatch5CrossConfigError("config identity/version mismatch")
    if document.get("provider") != {
        "name": "OANDA_v20_practice",
        "environment": "practice",
        "api_version": "v3",
    }:
        raise OandaBatch5CrossConfigError("provider mismatch")
    if (
        document.get("granularity") != "M1"
        or document.get("price_components") != "BA"
        or document.get("native_only") is not True
        or document.get("synthetic_crosses_permitted") is not False
        or document.get("fills_or_interpolation") is not False
    ):
        raise OandaBatch5CrossConfigError("native M1 BID/ASK contract mismatch")
    raw_instruments = document.get("instruments")
    if not isinstance(raw_instruments, list):
        raise OandaBatch5CrossConfigError("instruments must be a list")
    expected = {
        spec.instrument_id: (
            spec.dataset_symbol,
            oanda_symbol(spec.dataset_symbol),
            spec.price_precision,
        )
        for spec in OANDA_BATCH5_CROSS_SPECS
    }
    observed: dict[str, tuple[str, str, int]] = {}
    for row in raw_instruments:
        if not isinstance(row, dict):
            raise OandaBatch5CrossConfigError("instrument entry must be a mapping")
        observed[str(row.get("instrument_id"))] = (
            str(row.get("dataset_symbol")),
            str(row.get("oanda_instrument")),
            int(row.get("expected_display_precision", -1)),
        )
    if observed != expected:
        raise OandaBatch5CrossConfigError(
            "config must contain exactly native AUD/CAD and EUR/JPY"
        )
    firewall = document.get("firewall")
    if firewall != {
        "validation_accessed": False,
        "holdout_accessed": False,
        "performance_fields_permitted": False,
    }:
        raise OandaBatch5CrossConfigError("firewall mismatch")
    acquisition = document.get("acquisition")
    if not isinstance(acquisition, dict):
        raise OandaBatch5CrossConfigError("acquisition block missing")
    start = _utc(acquisition.get("start_utc"))
    development_start = _utc(acquisition.get("development_start_utc"))
    end = _utc(acquisition.get("development_end_exclusive_utc"))
    if not start < development_start < end:
        raise OandaBatch5CrossConfigError("acquisition boundaries are invalid")
    semantic = hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return CrossConfig(semantic, start, development_start, end, tuple(sorted(observed)))


def acquire_native_crosses(
    output_root: Path, *, config_path: Path = CONFIG_PATH
) -> tuple[InstrumentAcquisitionResult, ...]:
    """Reuse the existing append/resume-safe immutable OANDA acquisition."""

    config = load_config(config_path)
    token = _access_token_from_environment()
    if not token:
        raise OandaDevelopmentDataError(
            "an OANDA access-token environment variable is required"
        )
    root = output_root.resolve()
    _reject_worktree(root)
    spec_by_id = {spec.instrument_id: spec for spec in OANDA_BATCH5_CROSS_SPECS}
    results = tuple(
        _acquire_instrument(
            root,
            oanda_symbol(spec_by_id[instrument_id].dataset_symbol),
            config.acquisition_start_utc,
            config.development_end_exclusive_utc,
            token,
            _fetch_response,
        )
        for instrument_id in config.instruments
    )
    _write_idempotent_json(
        root / MANIFEST_FILENAME,
        {
            "lineage_id": LINEAGE_ID,
            "provider": "OANDA_v20_practice",
            "granularity": "M1",
            "price_components": "BA",
            "native_instruments_only": True,
            "synthetic_crosses_used": False,
            "config_semantic_sha256": config.semantic_sha256,
            "batch5_preregistration_semantic_sha256": PREREG_SHA,
            "acquisition_start_utc": config.acquisition_start_utc.isoformat(),
            "development_start_utc": config.development_start_utc.isoformat(),
            "development_end_exclusive_utc": (
                config.development_end_exclusive_utc.isoformat()
            ),
            "instruments": [
                {
                    "instrument_id": instrument_id,
                    "oanda_instrument": oanda_symbol(
                        spec_by_id[instrument_id].dataset_symbol
                    ),
                    "processed_path": str(result.processed_path.relative_to(root)),
                    "processed_sha256": result.processed_sha256,
                    "qa_path": str(result.qa_path.relative_to(root)),
                    "raw_response_count": result.raw_response_count,
                    "raw_response_sha256": result.raw_response_sha256,
                    "row_count": result.row_count,
                }
                for instrument_id, result in zip(
                    config.instruments, results, strict=True
                )
            ],
            "validation_accessed": False,
            "holdout_accessed": False,
            "performance_accessed": False,
        },
    )
    for instrument_id, result in zip(config.instruments, results, strict=True):
        spec = spec_by_id[instrument_id]
        _audit_cached_alpha_lab_instrument(
            root,
            oanda_symbol(spec.dataset_symbol),
            config.acquisition_start_utc,
            config.development_end_exclusive_utc,
            {
                "oanda_instrument": oanda_symbol(spec.dataset_symbol),
                "processed_path": str(result.processed_path.relative_to(root)),
                "processed_sha256": result.processed_sha256,
                "qa_path": str(result.qa_path.relative_to(root)),
            },
        )
    return results


def _read_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OandaBatch5CrossConfigError(f"could not read artifact: {path}") from error
    if not isinstance(document, dict):
        raise OandaBatch5CrossConfigError(f"artifact is not a mapping: {path}")
    return document


def canonicalize_native_crosses(
    output_root: Path, *, config_path: Path = CONFIG_PATH
) -> tuple[Path, ...]:
    """Reuse the existing OANDA M1 CSV-to-Nautilus catalog implementation."""

    config = load_config(config_path)
    root = output_root.resolve()
    manifest = _read_json(root / MANIFEST_FILENAME)
    if (
        manifest.get("lineage_id") != LINEAGE_ID
        or manifest.get("config_semantic_sha256") != config.semantic_sha256
        or manifest.get("batch5_preregistration_semantic_sha256")
        != EXPECTED_PREREGISTRATION_SEMANTIC_SHA256
    ):
        raise OandaBatch5CrossConfigError("acquisition manifest identity mismatch")
    records = {row["instrument_id"]: row for row in manifest["instruments"]}
    spec_by_id = {spec.instrument_id: spec for spec in OANDA_BATCH5_CROSS_SPECS}
    return tuple(
        canonicalize_oanda_instrument(
            processed_csv_path=root / records[instrument_id]["processed_path"],
            instrument_spec=spec_by_id[instrument_id],
            output_root=root
            / "canonical"
            / oanda_symbol(spec_by_id[instrument_id].dataset_symbol),
            alpha_lab_config_sha256=config.semantic_sha256,
            start_utc=config.acquisition_start_utc,
            end_exclusive_utc=config.development_end_exclusive_utc,
        )
        for instrument_id in config.instruments
    )


def derive_native_crosses(
    output_root: Path, *, config_path: Path = CONFIG_PATH
) -> tuple[Path, ...]:
    """Reuse existing deterministic M30/H1/H4 derivation for both crosses."""

    config = load_config(config_path)
    root = output_root.resolve()
    spec_by_id = {spec.instrument_id: spec for spec in OANDA_BATCH5_CROSS_SPECS}
    return tuple(
        derive_instrument_bars(
            root / "canonical" / oanda_symbol(spec_by_id[instrument_id].dataset_symbol),
            spec_by_id[instrument_id],
        ).manifest_path
        for instrument_id in config.instruments
    )


def freeze_readiness(output_root: Path, *, config_path: Path = CONFIG_PATH) -> Path:
    """Freeze deterministic per-cross source, QA, catalog, and firewall lineage."""

    config = load_config(config_path)
    root = output_root.resolve()
    acquisition = _read_json(root / MANIFEST_FILENAME)
    records = {row["instrument_id"]: row for row in acquisition["instruments"]}
    spec_by_id = {spec.instrument_id: spec for spec in OANDA_BATCH5_CROSS_SPECS}
    artifacts: list[dict[str, Any]] = []
    for instrument_id in config.instruments:
        spec = spec_by_id[instrument_id]
        native = oanda_symbol(spec.dataset_symbol)
        canonical_root = root / "canonical" / native
        parent_path = canonical_root / PARENT_MANIFEST_FILENAME
        derived_path = canonical_root / DERIVED_MANIFEST_FILENAME
        parent = _read_json(parent_path)
        derived = _read_json(derived_path)
        qa_path = root / "qa" / f"{native}_M1_bid_ask_qa.json"
        qa = _read_json(qa_path)
        record = records[instrument_id]
        ready = (
            parent.get("instrument_id") == instrument_id
            and parent.get("alpha_lab_config_sha256") == config.semantic_sha256
            and derived.get("instrument_id") == instrument_id
            and _oanda_derived_structural_readiness(derived)
            and qa.get("instrument") == native
            and qa.get("research_ready") is True
        )
        artifacts.append(
            {
                "instrument_id": instrument_id,
                "oanda_instrument": native,
                "source_identity": "OANDA_v20_practice_M1_BA_native",
                "raw_response_count": record["raw_response_count"],
                "raw_response_sha256": record["raw_response_sha256"],
                "normalized_processed_sha256": record["processed_sha256"],
                "normalized_row_count": record["row_count"],
                "acquisition_qa_sha256": _file_sha256(qa_path),
                "gap_diagnostics": qa.get("provider_observation_availability", qa),
                "canonical_manifest_sha256": _file_sha256(parent_path),
                "derived_manifest_sha256": _file_sha256(derived_path),
                "catalog_tree_sha256": _sha256_tree(canonical_root / "catalog"),
                "schema_version": "OANDA_M1_PAIRED_BID_ASK_V1",
                "research_ready": ready,
            }
        )
    payload = {
        "lineage_id": LINEAGE_ID,
        "config_semantic_sha256": config.semantic_sha256,
        "batch5_preregistration_semantic_sha256": PREREG_SHA,
        "acquisition_start_utc": config.acquisition_start_utc.isoformat(),
        "development_start_utc": config.development_start_utc.isoformat(),
        "development_end_exclusive_utc": (
            config.development_end_exclusive_utc.isoformat()
        ),
        "instrument_artifacts": artifacts,
        "research_ready": all(row["research_ready"] for row in artifacts),
        "validation_accessed": False,
        "holdout_accessed": False,
        "performance_accessed": False,
    }
    payload["semantic_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    target = root / "readiness" / "oanda_batch5_native_crosses_readiness.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Acquire exactly two native Batch-5 OANDA crosses"
    )
    parser.add_argument(
        "command", choices=("acquire", "canonicalize", "derive", "readiness")
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "acquire":
        results = acquire_native_crosses(args.output_root, config_path=args.config)
        print(args.output_root.resolve() / MANIFEST_FILENAME)
        print(f"acquired_instruments={len(results)}")
    elif args.command == "canonicalize":
        for path in canonicalize_native_crosses(
            args.output_root, config_path=args.config
        ):
            print(path)
    elif args.command == "derive":
        for path in derive_native_crosses(args.output_root, config_path=args.config):
            print(path)
    else:
        print(freeze_readiness(args.output_root, config_path=args.config))


if __name__ == "__main__":
    main()
