"""NEW, separate OANDA alpha-lab VALIDATION lineage.

Sibling to :mod:`ftmoquant.data.oanda_alpha_lab_development` (DEVELOPMENT),
not a modification of it -- that module and its frozen historical artifacts
are never read, written, or hashed by anything here. Reuses every genuinely
generic piece of the existing OANDA alpha-lab pipeline directly and
unchanged: the low-level acquisition/QA primitives from ``oanda_development``
(``_acquire_instrument``, ``_fetch_response``, per-candle validation, hash
utilities, ...), ``canonicalize_oanda_instrument`` (already parameterized by
explicit ``start_utc``/``end_exclusive_utc``, not DEVELOPMENT-hardcoded),
``derive_instrument_bars`` (fully generic), the cached per-instrument QA
audit ``_audit_cached_alpha_lab_instrument`` (also already parameterized by
explicit start/end, not DEVELOPMENT-hardcoded), and the derived-bar
structural-readiness gate ``_oanda_derived_structural_readiness`` (also
fully generic). Only the lineage-identity orchestration that is genuinely
DEVELOPMENT-specific (config loader, acquisition/QA-manifest identity,
readiness-freeze payload shape) is duplicated in miniature here, with its
own lineage id, config path, and readiness schema -- so that no code path
touching DEVELOPMENT can be affected by anything in this module, and vice
versa.

The VALIDATION interval is not invented here: it is pinned, at config-load
time, to the one interval already frozen anywhere in this repository --
``ftmoquant.research.stage_g.VALIDATION_START`` -> ``HOLDOUT_START``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import yaml

from ftmoquant.backtest.execution_harness import _sha256_tree
from ftmoquant.data import oanda_alpha_lab_development as _development
from ftmoquant.data.canonical_source import OANDA_ALPHA_LAB_INGESTION_VERSION
from ftmoquant.data.derived_bars import (
    DERIVED_MANIFEST_FILENAME,
    PARENT_MANIFEST_FILENAME,
    derive_instrument_bars,
)
from ftmoquant.data.instruments import (
    OANDA_ALPHA_LAB_SPECS,
    InstrumentSpec,
    oanda_symbol,
)
from ftmoquant.data.oanda_alpha_lab_development import (
    CachedInstrumentAvailabilityAudit,
    OandaAlphaLabInstrument,
    canonicalize_oanda_instrument,
)
from ftmoquant.data.oanda_development import (
    InstrumentAcquisitionResult as _AcquisitionResult,
)
from ftmoquant.data.oanda_development import (
    OandaDevelopmentDataError,
    _access_token_from_environment,
    _acquire_instrument,
    _fetch_response,
    _load_json_object,
    _reject_worktree,
    _sha256,
    _write_idempotent_json,
)
from ftmoquant.research.stage_g import HOLDOUT_START, VALIDATION_START

VALIDATION_CONFIG_PATH = Path("config/data/oanda_fx_alpha_lab_validation_v1.yaml")
VALIDATION_LINEAGE_ID = "oanda_fx_alpha_lab_validation_v1"
VALIDATION_READINESS_VERSION = "oanda-alpha-lab-validation-readiness-1"
VALIDATION_MANIFEST_FILENAME = "oanda_alpha_lab_validation_manifest.json"
VALIDATION_READINESS_FILENAME = "ftmoquant_oanda_alpha_lab_validation_readiness.json"
#: Same corrected QA principle and version lineage as DEVELOPMENT's
#: ``oanda-fx-alpha-lab-availability-qa-2``, reused via
#: ``_development._audit_cached_alpha_lab_instrument`` directly (it already
#: writes this exact constant into its report -- not redefined here).
AVAILABILITY_QA_VERSION = _development.AVAILABILITY_QA_VERSION


class OandaAlphaLabValidationConfigError(ValueError):
    """Raised when the OANDA alpha-lab VALIDATION config cannot be trusted."""


class OandaAlphaLabValidationReadinessError(ValueError):
    """Raised when the OANDA alpha-lab VALIDATION readiness cannot be proven."""


@dataclass(frozen=True, slots=True)
class OandaAlphaLabValidationConfig:
    lineage_id: str
    version: int
    partition: str
    validation_start_utc: datetime
    validation_end_exclusive_utc: datetime
    instruments: tuple[OandaAlphaLabInstrument, ...]
    semantic_sha256: str

    def instrument(self, instrument_id: str) -> OandaAlphaLabInstrument:
        matches = tuple(
            item for item in self.instruments if item.instrument_id == instrument_id
        )
        if len(matches) != 1:
            raise OandaAlphaLabValidationConfigError(
                f"config does not contain exactly one {instrument_id}"
            )
        return matches[0]


def load_oanda_alpha_lab_validation_config(
    path: Path = VALIDATION_CONFIG_PATH,
) -> OandaAlphaLabValidationConfig:
    """Load the exact, non-strategy OANDA alpha-lab VALIDATION lineage
    config. Pins ``validation.start_utc``/``end_exclusive_utc`` to exactly
    ``[VALIDATION_START, HOLDOUT_START)`` -- a config declaring any other
    interval is rejected outright, so this boundary cannot be re-invented by
    editing the YAML."""

    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise OandaAlphaLabValidationConfigError(
            f"could not load config: {error}"
        ) from error
    if not isinstance(document, dict):
        raise OandaAlphaLabValidationConfigError("config must be a mapping")
    if (
        document.get("lineage_id") != VALIDATION_LINEAGE_ID
        or document.get("version") != 1
        or document.get("partition") != "VALIDATION"
    ):
        raise OandaAlphaLabValidationConfigError(
            "config identity/version/partition is not recognized"
        )
    provider = document.get("provider")
    if not isinstance(provider, dict) or provider != {
        "name": "OANDA_v20_practice",
        "environment": "practice",
        "api_version": "v3",
    }:
        raise OandaAlphaLabValidationConfigError(
            "config provider block is incompatible"
        )
    validation = document.get("validation")
    if not isinstance(validation, dict):
        raise OandaAlphaLabValidationConfigError("config validation block is invalid")
    start = _require_utc(validation.get("start_utc"), "validation.start_utc")
    end = _require_utc(
        validation.get("end_exclusive_utc"), "validation.end_exclusive_utc"
    )
    if start != VALIDATION_START or end != HOLDOUT_START:
        raise OandaAlphaLabValidationConfigError(
            "config validation range must be exactly "
            "[stage_g.VALIDATION_START, stage_g.HOLDOUT_START)"
        )
    if document.get("granularity") != "M1" or document.get("price_components") != "BA":
        raise OandaAlphaLabValidationConfigError(
            "config granularity/price_components must be M1/BA"
        )
    policy = document.get("missing_observation_policy")
    if not isinstance(policy, dict) or policy != {
        "fills_or_interpolation": False,
        "synthetic_fill_permitted": False,
        "absent_minutes_during_genuine_market_closure_are_not_defects": True,
        "missing_observations_reported_explicitly": True,
    }:
        raise OandaAlphaLabValidationConfigError(
            "config missing-observation policy is incompatible"
        )
    if document.get("derived_timeframes") != ["M30", "H1", "H4"]:
        raise OandaAlphaLabValidationConfigError(
            "config derived_timeframes must be [M30, H1, H4]"
        )
    raw_instruments = document.get("instruments")
    if not isinstance(raw_instruments, list) or not raw_instruments:
        raise OandaAlphaLabValidationConfigError(
            "config instruments must be a non-empty list"
        )
    instruments = tuple(_instrument(item) for item in raw_instruments)
    ids = [item.instrument_id for item in instruments]
    if len(ids) != len(set(ids)):
        raise OandaAlphaLabValidationConfigError("config has duplicate instrument IDs")
    known_ids = {spec.instrument_id for spec in OANDA_ALPHA_LAB_SPECS}
    if set(ids) != known_ids:
        raise OandaAlphaLabValidationConfigError(
            "config instruments do not match the known OANDA alpha-lab specs"
        )
    if len(ids) != 7:
        raise OandaAlphaLabValidationConfigError(
            "config must declare exactly the seven OANDA alpha-lab instruments"
        )
    semantic_sha256 = hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return OandaAlphaLabValidationConfig(
        lineage_id=VALIDATION_LINEAGE_ID,
        version=1,
        partition="VALIDATION",
        validation_start_utc=start,
        validation_end_exclusive_utc=end,
        instruments=instruments,
        semantic_sha256=semantic_sha256,
    )


def _instrument(value: object) -> OandaAlphaLabInstrument:
    if not isinstance(value, dict):
        raise OandaAlphaLabValidationConfigError(
            "each config instrument must be a mapping"
        )
    keys = {"dataset_symbol", "instrument_id", "oanda_instrument"}
    if set(value) != keys:
        raise OandaAlphaLabValidationConfigError(
            f"config instrument must contain exactly: {keys}"
        )
    dataset_symbol = value["dataset_symbol"]
    instrument_id = value["instrument_id"]
    oanda_instrument = value["oanda_instrument"]
    if (
        not isinstance(dataset_symbol, str)
        or not isinstance(instrument_id, str)
        or not isinstance(oanda_instrument, str)
        or oanda_symbol(dataset_symbol) != oanda_instrument
    ):
        raise OandaAlphaLabValidationConfigError(
            "config instrument fields are inconsistent"
        )
    return OandaAlphaLabInstrument(dataset_symbol, instrument_id, oanda_instrument)


def _require_utc(value: object, name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise OandaAlphaLabValidationConfigError(f"{name} must be an ISO UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise OandaAlphaLabValidationConfigError(
            f"{name} must be an ISO UTC timestamp"
        ) from error
    if parsed.utcoffset() != timedelta(0):
        raise OandaAlphaLabValidationConfigError(f"{name} must be UTC")
    return parsed.astimezone(UTC)


# --------------------------------------------------------------------------
# Acquisition -- reuses oanda_development.py's generic low-level machinery
# unchanged, exactly as the DEVELOPMENT orchestrator does.
# --------------------------------------------------------------------------


def acquire_oanda_alpha_lab_validation(
    output_root: Path, *, config_path: Path = VALIDATION_CONFIG_PATH
) -> tuple[_AcquisitionResult, ...]:
    """Acquire the frozen OANDA alpha-lab VALIDATION interval for all seven
    FX pairs and run QA, reusing the existing generic acquisition helpers."""

    config = load_oanda_alpha_lab_validation_config(config_path)
    token = _access_token_from_environment()
    if not token:
        raise OandaDevelopmentDataError(
            "an OANDA access-token environment variable is required for acquisition"
        )
    root = output_root.resolve()
    _reject_worktree(root)
    results = tuple(
        _acquire_instrument(
            root,
            item.oanda_instrument,
            config.validation_start_utc,
            config.validation_end_exclusive_utc,
            token,
            _fetch_response,
        )
        for item in config.instruments
    )
    _write_idempotent_json(
        root / VALIDATION_MANIFEST_FILENAME,
        {
            "lineage_id": VALIDATION_LINEAGE_ID,
            "partition": "VALIDATION",
            "provider": "OANDA_v20_practice",
            "granularity": "M1",
            "price_components": "BA",
            "requested_start_utc": _iso(config.validation_start_utc),
            "requested_end_exclusive_utc": _iso(config.validation_end_exclusive_utc),
            "alpha_lab_config_sha256": config.semantic_sha256,
            "instruments": [
                {
                    "oanda_instrument": item.oanda_instrument,
                    "processed_path": str(result.processed_path.relative_to(root)),
                    "processed_sha256": result.processed_sha256,
                    "qa_path": str(result.qa_path.relative_to(root)),
                }
                for item, result in zip(config.instruments, results, strict=True)
            ],
        },
    )
    return results


def audit_cached_oanda_alpha_lab_validation(
    output_root: Path, *, config_path: Path = VALIDATION_CONFIG_PATH
) -> tuple[CachedInstrumentAvailabilityAudit, ...]:
    """Cache-only, no-network re-verification of already-acquired OANDA
    alpha-lab VALIDATION data. Reuses
    :func:`ftmoquant.data.oanda_alpha_lab_development._audit_cached_alpha_lab_instrument`
    directly and unchanged -- that function is already parameterized by
    explicit ``start``/``end`` and touches only ``output_root``, so it
    carries no DEVELOPMENT-specific behavior to duplicate."""

    config = load_oanda_alpha_lab_validation_config(config_path)
    root = output_root.resolve()
    manifest = _load_json_object(root / VALIDATION_MANIFEST_FILENAME)
    _validate_cached_validation_manifest(manifest, config)
    records = cast(list[dict[str, Any]], manifest["instruments"])
    by_instrument = {record["oanda_instrument"]: record for record in records}
    return tuple(
        _development._audit_cached_alpha_lab_instrument(
            root,
            item.oanda_instrument,
            config.validation_start_utc,
            config.validation_end_exclusive_utc,
            by_instrument[item.oanda_instrument],
        )
        for item in config.instruments
    )


def _validate_cached_validation_manifest(
    manifest: dict[str, Any], config: OandaAlphaLabValidationConfig
) -> None:
    if (
        manifest.get("lineage_id") != VALIDATION_LINEAGE_ID
        or manifest.get("partition") != "VALIDATION"
        or manifest.get("provider") != "OANDA_v20_practice"
        or manifest.get("granularity") != "M1"
        or manifest.get("price_components") != "BA"
        or manifest.get("requested_start_utc") != _iso(config.validation_start_utc)
        or manifest.get("requested_end_exclusive_utc")
        != _iso(config.validation_end_exclusive_utc)
        or manifest.get("alpha_lab_config_sha256") != config.semantic_sha256
    ):
        raise OandaDevelopmentDataError(
            "cached VALIDATION acquisition manifest drifted"
        )
    records = manifest.get("instruments")
    if not isinstance(records, list) or [
        record.get("oanda_instrument") if isinstance(record, dict) else None
        for record in records
    ] != [item.oanda_instrument for item in config.instruments]:
        raise OandaDevelopmentDataError(
            "cached VALIDATION acquisition instruments drifted"
        )


# --------------------------------------------------------------------------
# Canonicalization / M30-H1-H4 derivation -- both already fully generic
# (parameterized by explicit start/end or by spec+root); reused directly,
# unchanged, from oanda_alpha_lab_development / derived_bars. See
# canonicalize_main / derive_main below.
# --------------------------------------------------------------------------


def resolve_oanda_alpha_lab_validation_spec(
    config: OandaAlphaLabValidationConfig, instrument_id: str
) -> InstrumentSpec:
    """Fail closed unless ``instrument_id`` belongs to the frozen seven-pair
    OANDA alpha-lab lineage, then resolve its exact InstrumentSpec."""

    config.instrument(instrument_id)  # fail closed if not in this lineage
    spec_by_id = {spec.instrument_id: spec for spec in OANDA_ALPHA_LAB_SPECS}
    spec = spec_by_id.get(instrument_id)
    if spec is None:
        raise OandaAlphaLabValidationConfigError(
            f"no OANDA alpha-lab InstrumentSpec for {instrument_id}"
        )
    return spec


# --------------------------------------------------------------------------
# Readiness -- a small, VALIDATION-specific readiness manifest, structurally
# parallel to (but never derived from, and never overwriting) the
# DEVELOPMENT readiness artifact.
# --------------------------------------------------------------------------


def freeze_oanda_alpha_lab_validation_readiness(
    *,
    validation_roots: Mapping[str, Path],
    output_root: Path,
    config_path: Path = VALIDATION_CONFIG_PATH,
    acquisition_qa_root: Path | None = None,
) -> Path:
    """Freeze OANDA alpha-lab VALIDATION readiness from already-derived
    per-instrument catalogs. Gated on
    :func:`ftmoquant.data.oanda_alpha_lab_development._oanda_derived_structural_readiness`
    (reused unchanged; see its docstring for why DEVELOPMENT readiness uses
    the structural gate rather than ``derive_instrument_bars()``'s own,
    permanently-``False``-for-real-data flag -- the same is true here)."""

    config = load_oanda_alpha_lab_validation_config(config_path)
    if set(validation_roots) != {item.instrument_id for item in config.instruments}:
        raise OandaAlphaLabValidationReadinessError(
            "validation roots have missing/extra instruments"
        )
    spec_by_id = {spec.instrument_id: spec for spec in OANDA_ALPHA_LAB_SPECS}
    artifacts: list[dict[str, Any]] = []
    statuses: dict[str, str] = {}
    for instrument_id in sorted(validation_roots):
        root = validation_roots[instrument_id].resolve()
        spec = spec_by_id[instrument_id]
        parent = _read_json(root / PARENT_MANIFEST_FILENAME)
        if (
            parent.get("instrument_id") != instrument_id
            or parent.get("ingestion_version") != OANDA_ALPHA_LAB_INGESTION_VERSION
            or parent.get("alpha_lab_config_sha256") != config.semantic_sha256
        ):
            raise OandaAlphaLabValidationReadinessError(
                f"canonical manifest is incompatible: {instrument_id}"
            )
        derived = _read_json(root / DERIVED_MANIFEST_FILENAME)
        if derived.get("instrument_id") != instrument_id:
            raise OandaAlphaLabValidationReadinessError(
                f"derived manifest instrument mismatch: {instrument_id}"
            )
        research_ready = _development._oanda_derived_structural_readiness(derived)
        acquisition_research_ready: bool | None = None
        if acquisition_qa_root is not None:
            oanda_instrument = oanda_symbol(spec.dataset_symbol)
            qa_path = acquisition_qa_root / f"{oanda_instrument}_M1_bid_ask_qa.json"
            qa = _read_json(qa_path)
            acquisition_research_ready = (
                qa.get("instrument") == oanda_instrument
                and qa.get("research_ready") is True
            )
            research_ready = research_ready and acquisition_research_ready
        statuses[instrument_id] = "research_ready" if research_ready else "not_ready"
        artifacts.append(
            {
                "instrument_id": instrument_id,
                "dataset_symbol": spec.dataset_symbol,
                "canonical_sha256": _sha256(root / PARENT_MANIFEST_FILENAME),
                "derived_sha256": _sha256(root / DERIVED_MANIFEST_FILENAME),
                "catalog_tree_sha256": _sha256_tree(root / "catalog"),
                "coverage_status": derived.get("coverage_status"),
                "derived_research_ready_raw": derived.get("research_ready"),
                "acquisition_research_ready": acquisition_research_ready,
                "research_ready": research_ready,
            }
        )
    if len(artifacts) != 7:
        raise OandaAlphaLabValidationReadinessError(
            "VALIDATION readiness requires exactly seven instruments"
        )
    ordered_ids = tuple(sorted(validation_roots))
    payload = {
        "readiness_version": VALIDATION_READINESS_VERSION,
        "lineage_id": VALIDATION_LINEAGE_ID,
        "partition": "VALIDATION",
        "alpha_lab_config_sha256": config.semantic_sha256,
        "validation_start_utc": _iso(config.validation_start_utc),
        "validation_end_exclusive_utc": _iso(config.validation_end_exclusive_utc),
        "ordered_instrument_ids": list(ordered_ids),
        "per_instrument_status": statuses,
        "instrument_artifacts": artifacts,
        "holdout_accessed": False,
        "holdout_rows_admitted": 0,
        "research_ready": all(
            status == "research_ready" for status in statuses.values()
        ),
    }
    semantic_sha256 = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    root = output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    readiness_path = root / VALIDATION_READINESS_FILENAME
    readiness_path.write_text(
        json.dumps(
            {**payload, "semantic_sha256": semantic_sha256}, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    return readiness_path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OandaAlphaLabValidationReadinessError(
            f"could not read {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise OandaAlphaLabValidationReadinessError(f"{path} is not a JSON object")
    return cast(dict[str, Any], value)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------
# CLIs -- thin argument parsing over the functions above, mirroring
# oanda_alpha_lab_development.py's CLI shape exactly.
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Acquire the OANDA alpha-lab VALIDATION M1 BID/ASK lineage"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=VALIDATION_CONFIG_PATH)
    parser.add_argument(
        "--cache-only-qa",
        action="store_true",
        help="audit existing raw/processed artifacts without provider access",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.cache_only_qa:
        audits = audit_cached_oanda_alpha_lab_validation(
            args.output_root, config_path=args.config
        )
        for audit in audits:
            print(
                f"{audit.oanda_instrument} research_ready={audit.research_ready} "
                f"missing_minutes={audit.missing_minutes} "
                f"availability_ratio={audit.availability_ratio:.6f}"
            )
        return
    results = acquire_oanda_alpha_lab_validation(
        args.output_root, config_path=args.config
    )
    print(args.output_root.resolve() / VALIDATION_MANIFEST_FILENAME)
    print(f"acquired_instruments={len(results)}")


def build_canonicalize_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Canonicalize one already-acquired OANDA alpha-lab VALIDATION "
            "processed M1 BID/ASK CSV into the Nautilus catalog shape used "
            "by derive_instrument_bars()"
        )
    )
    parser.add_argument("--processed-csv", type=Path, required=True)
    parser.add_argument("--instrument-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=VALIDATION_CONFIG_PATH)
    return parser


def canonicalize_main(argv: list[str] | None = None) -> None:
    args = build_canonicalize_parser().parse_args(argv)
    config = load_oanda_alpha_lab_validation_config(args.config)
    spec = resolve_oanda_alpha_lab_validation_spec(config, args.instrument_id)
    manifest_path = canonicalize_oanda_instrument(
        processed_csv_path=args.processed_csv,
        instrument_spec=spec,
        output_root=args.output_root,
        alpha_lab_config_sha256=config.semantic_sha256,
        start_utc=config.validation_start_utc,
        end_exclusive_utc=config.validation_end_exclusive_utc,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(manifest_path)
    print(f"instrument_id={manifest['instrument_id']}")
    print(f"bid_bar_count={manifest['qa']['bid_bar_count']}")
    print(f"ask_bar_count={manifest['qa']['ask_bar_count']}")
    print(f"semantic_sha256={manifest['semantic_sha256']}")


def build_derive_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Derive M30/H1/H4 bars for one OANDA alpha-lab VALIDATION "
            "canonical instrument, using the existing generic "
            "derive_instrument_bars()"
        )
    )
    parser.add_argument("--config", type=Path, default=VALIDATION_CONFIG_PATH)
    parser.add_argument("--instrument-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def derive_main(argv: list[str] | None = None) -> None:
    args = build_derive_parser().parse_args(argv)
    config = load_oanda_alpha_lab_validation_config(args.config)
    spec = resolve_oanda_alpha_lab_validation_spec(config, args.instrument_id)
    result = derive_instrument_bars(args.output_root, spec)
    print(result.manifest_path)


def build_readiness_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze OANDA alpha-lab VALIDATION readiness across all seven "
            "already-derived canonical instruments"
        )
    )
    parser.add_argument("--config", type=Path, default=VALIDATION_CONFIG_PATH)
    parser.add_argument(
        "--canonical-root",
        type=Path,
        required=True,
        help=(
            "directory containing one <EUR_USD-style oanda_symbol> "
            "subdirectory per instrument, each already canonicalized and "
            "derived over the VALIDATION interval"
        ),
    )
    parser.add_argument(
        "--acquisition-qa-root",
        type=Path,
        default=None,
        help=(
            "optional: directory containing the cache-only acquisition QA "
            "JSONs; if given, readiness also requires each instrument's "
            "acquisition QA to show research_ready=true"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def readiness_main(argv: list[str] | None = None) -> None:
    args = build_readiness_parser().parse_args(argv)
    config = load_oanda_alpha_lab_validation_config(args.config)
    validation_roots = {
        item.instrument_id: args.canonical_root / oanda_symbol(item.dataset_symbol)
        for item in config.instruments
    }
    readiness_path = freeze_oanda_alpha_lab_validation_readiness(
        validation_roots=validation_roots,
        output_root=args.output,
        config_path=args.config,
        acquisition_qa_root=args.acquisition_qa_root,
    )
    print(readiness_path)


if __name__ == "__main__":
    main()
