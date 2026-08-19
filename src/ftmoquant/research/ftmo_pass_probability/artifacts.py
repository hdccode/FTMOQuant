"""Deterministic, self-hashed, append-never JSON/CSV artifact writers.

Mirrors the write-once convention already used throughout this repo (e.g.
``execution_harness.py``'s ``RUN_MANIFEST_FILENAME`` and
``research/g1/artifacts.py``): canonical (sorted-key, no whitespace-drift)
JSON, a SHA-256 embedded in the payload itself, atomic write, then a
read-only file mode. An existing path is rejected rather than overwritten.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any


class ArtifactError(ValueError):
    """Raised when an artifact would silently overwrite an existing one."""


def read_json_artifact(path: Path) -> dict[str, Any]:
    """Read back a previously-written JSON artifact, e.g. for embedding one
    artifact's already-computed result inside a later, comparison-only one.
    """

    if not path.is_file():
        raise ArtifactError(f"missing artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ArtifactError(f"{path} must contain a JSON object")
    return value


def write_json_artifact(path: Path, payload: Mapping[str, Any]) -> str:
    """Write one canonical, self-hashed JSON artifact. Refuses overwrite."""

    if path.exists():
        raise ArtifactError(f"refusing to overwrite existing artifact: {path}")
    normalized = _jsonable(dict(payload))
    content_sha256 = hashlib.sha256(_canonical_bytes(normalized)).hexdigest()
    final = {**normalized, "content_sha256": content_sha256}
    content = _canonical_bytes(final) + b"\n"
    _atomic_write(path, content)
    path.chmod(0o444)
    return content_sha256


def write_csv_artifact(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> str:
    """Write one deterministic CSV artifact. Refuses overwrite."""

    if path.exists():
        raise ArtifactError(f"refusing to overwrite existing artifact: {path}")
    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(fieldnames), lineterminator="\n"
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _scalar(value) for key, value in row.items()})
        os.replace(tmp_path, path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise
    path.chmod(0o444)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
        os.replace(tmp_path, path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    return value


def _scalar(value: Any) -> Any:
    if isinstance(value, (Decimal, Enum)):
        return str(value) if isinstance(value, Decimal) else value.value
    return value
