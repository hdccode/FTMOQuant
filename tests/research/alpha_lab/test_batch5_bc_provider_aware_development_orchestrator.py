from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ftmoquant.research.alpha_lab import (
    batch5_bc_provider_aware_development_orchestrator as mod,
)
from ftmoquant.research.alpha_lab.batch5_bc_corrected_development_orchestrator import (
    Batch5BCCorrectedOrchestratorError,
)
from ftmoquant.research.alpha_lab.batch5_daily import (
    build_provider_aware_fx_days_with_diagnostics,
)


def test_provider_aware_cli_has_only_frozen_data_roots_and_output() -> None:
    parser = mod.build_parser()
    options = {action.dest for action in parser._actions} - {"help"}
    assert options == {
        "development_root",
        "universe_readiness",
        "batch5_cross_root",
        "output",
    }


def test_provider_aware_runner_rejects_alternate_output_before_execution(
    tmp_path: Path,
) -> None:
    with pytest.raises(Batch5BCCorrectedOrchestratorError, match="output is frozen"):
        mod.run_provider_aware_development(
            development_root=tmp_path / "canonical",
            universe_readiness=tmp_path / "readiness.json",
            batch5_cross_root=tmp_path / "crosses",
            output_dir=tmp_path / "alternate-output",
        )


def test_provider_aware_runner_passes_only_frozen_builder_and_amendment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}
    amendment = {"amendment_semantic_sha256": "a" * 64}
    monkeypatch.setattr(mod, "verify_fx_day_amendment", lambda: amendment)
    monkeypatch.setattr(
        mod,
        "_sha256_file",
        lambda _path: mod.PRIOR_EXACT_BOUNDARY_ARTIFACT_HASH_MANIFEST_SHA256,
    )
    monkeypatch.setattr(Path, "is_file", lambda _path: True)
    monkeypatch.setattr(
        mod,
        "run_corrected_development",
        lambda **kwargs: captured.update(kwargs),
    )

    mod.run_provider_aware_development(
        development_root=Path("canonical"),
        universe_readiness=Path("readiness.json"),
        batch5_cross_root=Path("crosses"),
        output_dir=mod.ARTIFACT_ROOT,
    )

    assert captured["daily_builder"] is build_provider_aware_fx_days_with_diagnostics
    assert captured["methodology_amendment"] is amendment
    assert captured["stage"] == mod.STAGE


def test_provider_aware_runner_imports_no_b5a_validation_or_holdout_module() -> None:
    source = Path(mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    assert not any(
        "batch5a" in name or "validation" in name or "holdout" in name
        for name in imported
    )
