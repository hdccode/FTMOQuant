"""Deterministic G1 result artifacts separated from runtime provenance."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from ftmoquant.research.g1.search import SearchResult
from ftmoquant.research.g1.selector import SelectionDecision


class ArtifactError(ValueError):
    """Raised when an artifact payload cannot be made deterministic."""


def deterministic_artifact_bytes(
    result: SearchResult, selection: SelectionDecision
) -> bytes:
    payload = {
        "artifact_version": "g1-family-trial-registry-1",
        "family_id": result.family_id,
        "family_version": result.family_version,
        "search": {
            "mode": result.config.mode.value,
            "seed": result.config.seed,
            "optuna_trials": result.config.optuna_trials,
            "semantic_sha256": result.config.semantic_sha256,
        },
        "multiple_testing": {
            "total_attempted_configurations": (
                result.registry.total_attempted_configurations
            ),
            "total_unique_configurations": (
                result.registry.total_unique_configurations
            ),
            "valid_configurations": result.registry.valid_configurations,
            "invalid_configurations": result.registry.invalid_configurations,
            "failed_configurations": result.registry.failed_configurations,
            "configurations_surviving_eligibility_gates": (
                selection.configurations_surviving_eligibility_gates
            ),
            "configurations_considered_by_final_selector": (
                selection.configurations_considered_by_final_selector
            ),
            "deflated_sharpe_ratio": "not_implemented_extension_point",
            "probability_of_backtest_overfitting": ("not_implemented_extension_point"),
        },
        "trials": result.registry.records,
        "selection": selection,
    }
    normalized = _jsonable(payload)
    semantic = hashlib.sha256(_canonical_bytes(normalized)).hexdigest()
    return _canonical_bytes({**normalized, "semantic_sha256": semantic}) + b"\n"


def write_deterministic_artifact(
    path: Path, result: SearchResult, selection: SelectionDecision
) -> str:
    content = deterministic_artifact_bytes(result, selection)
    _atomic_write(path, content)
    return hashlib.sha256(content).hexdigest()


def write_runtime_provenance(path: Path, provenance: dict[str, object]) -> None:
    """Write wall-clock/runtime metadata separately from deterministic results."""

    _atomic_write(path, _canonical_bytes(_jsonable(provenance)) + b"\n")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ArtifactError("artifact mapping keys must be strings")
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ArtifactError(f"unsupported artifact value: {type(value).__name__}")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
