from __future__ import annotations

import json

import pytest

from ftmoquant.research.alpha_lab.batch5_bc_correction_protocol import (
    CORRECTION_PROTOCOL_PATH,
    EXPECTED_CORRECTION_PROTOCOL_SEMANTIC_SHA256,
    Batch5CorrectionProtocolError,
    semantic_sha256,
    verify_correction_protocol,
)


def test_correction_protocol_identity_scope_and_exact_three_defects() -> None:
    document = verify_correction_protocol()
    assert semantic_sha256(document) == EXPECTED_CORRECTION_PROTOCOL_SEMANTIC_SHA256
    assert document["forensic_audit"]["demonstrated_defect_count"] == 3
    assert document["scope"]["b5a_policy"].startswith("immutable_reference_only")
    assert document["governance"]["original_gates_and_breadth_unchanged"] is True


def test_correction_protocol_fails_closed_on_semantic_drift(tmp_path) -> None:
    document = json.loads(CORRECTION_PROTOCOL_PATH.read_text(encoding="utf-8"))
    document["governance"]["not_parameter_tuning"] = False
    path = tmp_path / "drifted.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(Batch5CorrectionProtocolError, match="semantic hash drift"):
        verify_correction_protocol(path)
