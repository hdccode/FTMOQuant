from __future__ import annotations

import ast
from pathlib import Path

from ftmoquant.research.alpha_lab.development_resurrection_audit import (
    CandidateEvidence,
    ForensicClass,
    classify_forensic,
)


def test_validation_exposed_candidate_is_never_resurrection_eligible() -> None:
    evidence = CandidateEvidence(
        validation_exposed=True,
        original_preregistered_pass=None,
        native_expectancy=100.0,
        native_profit_factor=5.0,
        trade_count=1000,
        min_trade_count_requirement=50,
    )
    result = classify_forensic(evidence)
    assert result.forensic_class is ForensicClass.VALIDATION_EXPOSED

    # Even a candidate that would otherwise look like a robust survivor
    # must still be excluded once exposed to VALIDATION.
    passed_but_exposed = CandidateEvidence(
        validation_exposed=True, original_preregistered_pass=True
    )
    assert (
        classify_forensic(passed_but_exposed).forensic_class
        is ForensicClass.VALIDATION_EXPOSED
    )


def test_economic_failure_is_dead_regardless_of_trade_count() -> None:
    evidence = CandidateEvidence(
        native_expectancy=-5.0,
        native_profit_factor=0.8,
        trade_count=10_000,  # power is fine -- economics are not
        min_trade_count_requirement=50,
    )
    result = classify_forensic(evidence)
    assert result.forensic_class is ForensicClass.ECONOMICALLY_DEAD


def test_underpowered_near_miss_requires_every_available_economic_gate_to_pass() -> (
    None
):
    clean_economics = CandidateEvidence(
        native_expectancy=12.5,
        native_profit_factor=1.8,
        stress_expectancy=3.0,
        stress_label="1.5x",
        best_5pct_removed_expectancy=2.0,
        quarter_concentration=0.2,
        quarter_concentration_limit=0.4,
        trade_count=40,
        min_trade_count_requirement=50,
    )
    result = classify_forensic(clean_economics)
    assert result.forensic_class is ForensicClass.UNDERPOWERED_NEAR_MISS

    # Same power shortfall, but PF also fails -> must NOT be a near miss.
    weak_pf = CandidateEvidence(
        native_expectancy=12.5,
        native_profit_factor=1.05,
        stress_expectancy=3.0,
        best_5pct_removed_expectancy=2.0,
        quarter_concentration=0.2,
        quarter_concentration_limit=0.4,
        trade_count=40,
        min_trade_count_requirement=50,
    )
    weak_result = classify_forensic(weak_pf)
    assert weak_result.forensic_class is not ForensicClass.UNDERPOWERED_NEAR_MISS
    assert weak_result.forensic_class is not ForensicClass.CREDIBLE_NEAR_MISS


def test_credible_near_miss_allows_at_most_two_pool_gate_failures() -> None:
    two_failures = CandidateEvidence(
        native_expectancy=12.5,
        native_profit_factor=1.8,
        stress_expectancy=3.0,
        trade_count=40,
        min_trade_count_requirement=50,
        positive_fold_count=1,
        fold_requirement=3,
    )
    assert classify_forensic(two_failures).forensic_class is (
        ForensicClass.CREDIBLE_NEAR_MISS
    )

    three_failures = CandidateEvidence(
        native_expectancy=12.5,
        native_profit_factor=1.8,
        stress_expectancy=3.0,
        trade_count=40,
        min_trade_count_requirement=50,
        positive_fold_count=1,
        fold_requirement=3,
        connected_region_size=1,
        connected_region_requirement=2,
    )
    result = classify_forensic(three_failures)
    assert result.forensic_class is ForensicClass.ECONOMICALLY_DEAD
    assert "budget" in result.reason


def test_missing_metrics_yield_not_auditable_never_invented() -> None:
    evidence = CandidateEvidence()
    result = classify_forensic(evidence)
    assert result.forensic_class is ForensicClass.NOT_AUDITABLE


def test_original_survivor_is_robust_survivor() -> None:
    evidence = CandidateEvidence(
        original_preregistered_pass=True,
        native_expectancy=-1.0,  # even if the retained evidence looks weak
    )
    assert classify_forensic(evidence).forensic_class is ForensicClass.ROBUST_SURVIVOR


def test_classification_is_deterministic() -> None:
    evidence = CandidateEvidence(
        native_expectancy=12.5,
        native_profit_factor=1.8,
        stress_expectancy=3.0,
        trade_count=40,
        min_trade_count_requirement=50,
    )
    results = {classify_forensic(evidence).forensic_class for _ in range(50)}
    assert len(results) == 1


def test_module_reads_no_validation_or_holdout_paths() -> None:
    import ftmoquant.research.alpha_lab.development_resurrection_audit as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    forbidden = (
        "ftmoquant.data.oanda_alpha_lab_validation",
        "ftmoquant.data.canonical_source",
        "ftmoquant.research.ftmo_pass_probability.validation_diagnostic",
    )
    for name in forbidden:
        assert name not in imported
    assert "validation_scorecard" not in source
    assert "family_validation_summary" not in source
    assert "candidate_validation_summary" not in source
    assert "VALIDATION_START" not in source
    assert "HOLDOUT_START" not in source
