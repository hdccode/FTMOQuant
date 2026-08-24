"""Identity verifier for the provider-aware Batch 5B/5C FX-day amendment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ftmoquant.research.alpha_lab.batch5_bc_correction_protocol import (
    EXPECTED_CORRECTION_PROTOCOL_SEMANTIC_SHA256,
)
from ftmoquant.research.alpha_lab.batch5_preregistration import (
    EXPECTED_PREREGISTRATION_SEMANTIC_SHA256,
    FAMILY_B5A,
    FAMILY_B5B,
    FAMILY_B5C,
)

AMENDMENT_PATH = Path(
    "config/research/batch5_bc_provider_aware_fx_day_amendment_v1.json"
)
AMENDMENT_VERSION = "batch5-bc-provider-aware-fx-day-amendment-v1"
EXPECTED_AMENDMENT_SEMANTIC_SHA256 = (
    "fbdf910c9a89f25743f89c219db1a4b82d5233de63343d8f308955e791bfbb0f"
)
PRIOR_EXACT_BOUNDARY_ARTIFACT_HASH_MANIFEST_SHA256 = (
    "ae104501f281bd3eb1e004b030375a2807c284a16e45ec0efcc1eb1d23a1b045"
)


class Batch5FxDayAmendmentError(RuntimeError):
    """Raised when the frozen provider-aware amendment identity drifts."""


def semantic_sha256(document: dict[str, Any]) -> str:
    semantic = {
        key: value
        for key, value in document.items()
        if key != "amendment_semantic_sha256"
    }
    return hashlib.sha256(
        json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def verify_fx_day_amendment(path: Path = AMENDMENT_PATH) -> dict[str, Any]:
    try:
        document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Batch5FxDayAmendmentError(
            f"could not read provider-aware FX-day amendment: {path}"
        ) from error
    actual = document.get("amendment_semantic_sha256")
    if actual != semantic_sha256(document):
        raise Batch5FxDayAmendmentError("FX-day amendment semantic hash drift")
    if actual != EXPECTED_AMENDMENT_SEMANTIC_SHA256:
        raise Batch5FxDayAmendmentError("unexpected FX-day amendment identity")
    if document.get("amendment_version") != AMENDMENT_VERSION:
        raise Batch5FxDayAmendmentError("unexpected FX-day amendment version")
    identities = document.get("frozen_identities", {})
    if (
        identities.get("original_batch5_preregistration_semantic_sha256")
        != EXPECTED_PREREGISTRATION_SEMANTIC_SHA256
        or identities.get(
            "prior_implementation_correction_protocol_semantic_sha256"
        )
        != EXPECTED_CORRECTION_PROTOCOL_SEMANTIC_SHA256
        or identities.get(
            "prior_exact_boundary_corrected_artifact_hash_manifest_sha256"
        )
        != PRIOR_EXACT_BOUNDARY_ARTIFACT_HASH_MANIFEST_SHA256
    ):
        raise Batch5FxDayAmendmentError("frozen predecessor identity drift")
    scope = document.get("scope", {})
    governance = document.get("governance_disclosure", {})
    if (
        tuple(scope.get("included_families", ())) != (FAMILY_B5B, FAMILY_B5C)
        or scope.get("excluded_family") != FAMILY_B5A
        or governance.get("maximum_provider_aware_corrected_development_runs") != 1
        or governance.get("independent_preregistration_evidence") is not False
        or governance.get("validation_access_prohibited") is not True
        or governance.get("holdout_access_prohibited") is not True
    ):
        raise Batch5FxDayAmendmentError("FX-day amendment governance drift")
    return document
