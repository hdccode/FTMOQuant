from __future__ import annotations

import json

import pytest

from ftmoquant.research.alpha_lab.batch5_bc_fx_day_amendment import (
    AMENDMENT_PATH,
    EXPECTED_AMENDMENT_SEMANTIC_SHA256,
    Batch5FxDayAmendmentError,
    semantic_sha256,
    verify_fx_day_amendment,
)


def test_provider_aware_amendment_identity_scope_and_governance() -> None:
    document = verify_fx_day_amendment()
    assert semantic_sha256(document) == EXPECTED_AMENDMENT_SEMANTIC_SHA256
    assert document["scope"]["excluded_family"].startswith("B5A_")
    governance = document["governance_disclosure"]
    assert governance["created_after_original_development_screen"] is True
    assert governance["truncated_b5b_b5c_performance_already_observed"] is True
    assert governance["alternative_boundary_convention_evaluated_economically"] is False
    assert governance["independent_preregistration_evidence"] is False
    assert governance["maximum_provider_aware_corrected_development_runs"] == 1


def test_provider_aware_amendment_fails_closed_on_semantic_drift(tmp_path) -> None:
    document = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
    document["calendar_membership"]["tunable_tolerance_minutes"] = 5
    path = tmp_path / "drifted.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(Batch5FxDayAmendmentError, match="semantic hash drift"):
        verify_fx_day_amendment(path)
