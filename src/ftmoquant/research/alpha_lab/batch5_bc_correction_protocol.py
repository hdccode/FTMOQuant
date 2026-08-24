"""Identity verification for the Batch 5 B5B/B5C implementation correction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ftmoquant.research.alpha_lab.batch5_cftc_availability import (
    EXPECTED_AMENDMENT_SEMANTIC_SHA256,
)
from ftmoquant.research.alpha_lab.batch5_preregistration import (
    EXPECTED_PREREGISTRATION_SEMANTIC_SHA256,
    FAMILY_B5A,
    FAMILY_B5B,
    FAMILY_B5C,
)

CORRECTION_PROTOCOL_PATH = Path(
    "config/research/batch5_bc_implementation_correction_protocol_v1.json"
)
CORRECTION_PROTOCOL_VERSION = "batch5-bc-implementation-correction-v1"
EXPECTED_CORRECTION_PROTOCOL_SEMANTIC_SHA256 = (
    "0d534da4717487fa667259ed556a5bc52214fe36e2e3c3d526d096f784a4b454"
)
ORIGINAL_ARTIFACT_HASH_MANIFEST_SHA256 = (
    "8400ac4093e07a98d7e4ac673a83cc8b11bd259bf1597cd96d067fbdd617c878"
)


class Batch5CorrectionProtocolError(RuntimeError):
    """Raised when the correction protocol identity or scope drifts."""


def semantic_sha256(document: dict[str, Any]) -> str:
    semantic = {
        key: value
        for key, value in document.items()
        if key != "correction_protocol_semantic_sha256"
    }
    return hashlib.sha256(
        json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def verify_correction_protocol(
    path: Path = CORRECTION_PROTOCOL_PATH,
) -> dict[str, Any]:
    try:
        document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Batch5CorrectionProtocolError(
            f"could not read correction protocol: {path}"
        ) from error
    actual = document.get("correction_protocol_semantic_sha256")
    if actual != semantic_sha256(document):
        raise Batch5CorrectionProtocolError("correction protocol semantic hash drift")
    if actual != EXPECTED_CORRECTION_PROTOCOL_SEMANTIC_SHA256:
        raise Batch5CorrectionProtocolError("unexpected correction protocol identity")
    if document.get("correction_protocol_version") != CORRECTION_PROTOCOL_VERSION:
        raise Batch5CorrectionProtocolError("unexpected correction protocol version")
    identities = document.get("frozen_identities", {})
    if (
        identities.get("original_batch5_preregistration_semantic_sha256")
        != EXPECTED_PREREGISTRATION_SEMANTIC_SHA256
        or identities.get("cftc_availability_amendment_semantic_sha256")
        != EXPECTED_AMENDMENT_SEMANTIC_SHA256
        or identities.get("original_development_artifact_hash_manifest_sha256")
        != ORIGINAL_ARTIFACT_HASH_MANIFEST_SHA256
    ):
        raise Batch5CorrectionProtocolError("frozen source identity drift")
    scope = document.get("scope", {})
    if (
        tuple(scope.get("included_families", ())) != (FAMILY_B5B, FAMILY_B5C)
        or scope.get("excluded_family") != FAMILY_B5A
        or document.get("forensic_audit", {}).get("demonstrated_defect_count") != 3
    ):
        raise Batch5CorrectionProtocolError("correction scope drift")
    return document
