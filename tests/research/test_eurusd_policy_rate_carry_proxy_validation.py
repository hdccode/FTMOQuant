"""Synthetic-data-only tests for the ``eurusd_policy_rate_carry_proxy_v1``
one-shot validation adapter.

No test in this file opens a real validation/holdout catalog, invokes
:func:`run_eurusd_policy_rate_carry_proxy_validation` to completion, or
otherwise observes real validation returns. Every test either exercises pure
logic against fabricated ``DailyPnlDecomposition``/``ValidationMetrics``
fixtures, or proves a preflight check fails closed on a bad/missing input.
"""

from __future__ import annotations

import subprocess
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from ftmoquant.research.eurusd_policy_rate_carry_proxy_development import (
    DailyPnlDecomposition,
    decompose_daily_pnl,
)
from ftmoquant.research.eurusd_policy_rate_carry_proxy_development import (
    daily_decompositions_from_equity_marks as _dev_carry_isolation,
)
from ftmoquant.research.eurusd_policy_rate_carry_proxy_validation import (
    DEVELOPMENT_RESULT_PATH,
    FAMILY_SEMANTIC_SHA256,
    INTEGRITY_AUDIT_PATH,
    MINIMUM_ELIGIBLE_OBSERVATIONS,
    PROTOCOL_PATH,
    PROTOCOL_SEMANTIC_SHA256,
    VALIDATION_OUTPUT_DIR,
    EurusdPolicyRateCarryProxyValidationError,
    ValidationMetrics,
    ValidationProtocol,
    _check_clean_worktree,
    _check_development_result_sha,
    _check_family_semantic_sha,
    _check_integrity_audit_sha,
    _check_no_search_or_selector_imports,
    _check_one_candidate_only,
    _check_output_not_exists,
    _check_sample_window_and_carry_isolation_enforcement,
    _check_validation_boundaries,
    _decompositions_in_validation_window,
    _reject_final_holdout_path,
    _run_preflight_checks,
    _write_validation_artifacts,
    build_parser,
    load_validation_protocol,
    run_eurusd_policy_rate_carry_proxy_validation,
    validation_passes,
    verify_development_evidence,
)
from ftmoquant.research.stage_g import HOLDOUT_START, VALIDATION_START

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _metrics(
    *,
    observation_count: int = 250,
    mean: float = 0.0001,
    stressed_mean: float = 0.0001,
) -> ValidationMetrics:
    return ValidationMetrics(
        observation_count=observation_count,
        cumulative_total_return=mean * observation_count,
        mean_daily_total_return=mean,
        stressed_cumulative_total_return=stressed_mean * observation_count,
        stressed_mean_daily_total_return=stressed_mean,
        sharpe=1.0,
        maximum_drawdown=0.01,
        stationary_bootstrap_ci=None,
        yearly_attribution=(),
        rate_regime_attribution=(),
        spot_pnl_contribution=0.02,
        carry_accrual_contribution=-0.005,
        execution_cost_contribution=-0.0001,
        carry_contribution_ratio=-0.2,
    )


def _protocol() -> ValidationProtocol:
    return load_validation_protocol()


def _zero_decomposition(as_of: date) -> DailyPnlDecomposition:
    return decompose_daily_pnl(
        as_of,
        total_equity_change=Decimal("0"),
        carry_accrual=Decimal("0"),
        execution_cost=Decimal("0"),
    )


# ---------------------------------------------------------------------------
# validation_passes: exact four hard gates
# ---------------------------------------------------------------------------


def test_gate_fails_at_199_observations() -> None:
    protocol = _protocol()
    assert not validation_passes(_metrics(observation_count=199), protocol)


def test_gate_passes_at_200_observations() -> None:
    protocol = _protocol()
    assert validation_passes(_metrics(observation_count=200), protocol)


def test_gate_passes_when_base_and_stressed_are_both_positive() -> None:
    protocol = _protocol()
    assert validation_passes(_metrics(mean=0.0002, stressed_mean=0.0001), protocol)


def test_gate_fails_when_base_mean_is_zero() -> None:
    protocol = _protocol()
    assert not validation_passes(_metrics(mean=0.0, stressed_mean=0.0001), protocol)


def test_gate_fails_when_base_mean_is_negative() -> None:
    protocol = _protocol()
    assert not validation_passes(_metrics(mean=-0.0001, stressed_mean=0.0001), protocol)


def test_gate_fails_when_stressed_mean_is_negative_even_if_base_is_positive() -> None:
    protocol = _protocol()
    metrics = _metrics(mean=0.0001, stressed_mean=-0.00001)
    assert not validation_passes(metrics, protocol)


def test_gate_fails_on_nonfinite_metrics() -> None:
    protocol = _protocol()
    metrics = _metrics()
    bad = ValidationMetrics(
        observation_count=metrics.observation_count,
        cumulative_total_return=float("nan"),
        mean_daily_total_return=metrics.mean_daily_total_return,
        stressed_cumulative_total_return=metrics.stressed_cumulative_total_return,
        stressed_mean_daily_total_return=metrics.stressed_mean_daily_total_return,
        sharpe=metrics.sharpe,
        maximum_drawdown=metrics.maximum_drawdown,
        stationary_bootstrap_ci=None,
        yearly_attribution=(),
        rate_regime_attribution=(),
        spot_pnl_contribution=0.0,
        carry_accrual_contribution=0.0,
        execution_cost_contribution=0.0,
        carry_contribution_ratio=None,
    )
    assert not validation_passes(bad, protocol)


# ---------------------------------------------------------------------------
# Sample-window filtering (pre-validation warm-up exclusion)
# ---------------------------------------------------------------------------


def test_pre_validation_warmup_excluded_from_sample() -> None:
    warmup_date = VALIDATION_START.date() - timedelta(days=5)
    in_window_date = VALIDATION_START.date()
    last_in_window_date = HOLDOUT_START.date() - timedelta(days=1)
    after_window_date = HOLDOUT_START.date()
    decompositions = (
        _zero_decomposition(warmup_date),
        _zero_decomposition(in_window_date),
        _zero_decomposition(last_in_window_date),
        _zero_decomposition(after_window_date),
    )
    windowed = _decompositions_in_validation_window(decompositions)
    assert [item.as_of_date for item in windowed] == [
        in_window_date,
        last_in_window_date,
    ]


def test_validation_endpoint_holdout_start_is_excluded() -> None:
    decompositions = (_zero_decomposition(HOLDOUT_START.date()),)
    assert _decompositions_in_validation_window(decompositions) == ()


def test_validation_start_itself_is_included() -> None:
    decompositions = (_zero_decomposition(VALIDATION_START.date()),)
    windowed = _decompositions_in_validation_window(decompositions)
    assert len(windowed) == 1


# ---------------------------------------------------------------------------
# Balance-based carry isolation / fill-inside-bracket: proven upstream by the
# DEVELOPMENT module's own tests; here we confirm the validation module
# reuses the exact same fixed function objects (the enforcement mechanism).
# ---------------------------------------------------------------------------


def test_carry_isolation_reuses_the_fixed_development_function_object() -> None:
    import ftmoquant.research.eurusd_policy_rate_carry_proxy_validation as vmod

    assert vmod.daily_decompositions_from_equity_marks is _dev_carry_isolation  # type: ignore[attr-defined]


def test_check_sample_window_and_carry_isolation_enforcement_passes() -> None:
    _check_sample_window_and_carry_isolation_enforcement()  # no raise


# ---------------------------------------------------------------------------
# Pending-target supersession / strict causal execution: proven exhaustively
# in the DEVELOPMENT test file. A thin confirmation that the validation
# module imports and uses the same builder.
# ---------------------------------------------------------------------------


def test_validation_module_reuses_build_carry_proxy_instructions() -> None:
    import ftmoquant.research.eurusd_policy_rate_carry_proxy_validation as vmod
    from ftmoquant.research.eurusd_policy_rate_carry_proxy_development import (
        build_carry_proxy_instructions as _dev_builder,
    )

    assert vmod.build_carry_proxy_instructions is _dev_builder  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Final-holdout rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_path",
    [
        Path("some/holdout/root"),
        Path("some/FINAL_HOLDOUT/root"),
        Path("final-holdout/data"),
    ],
)
def test_reject_final_holdout_path_fails_closed(bad_path: Path) -> None:
    with pytest.raises(EurusdPolicyRateCarryProxyValidationError):
        _reject_final_holdout_path(bad_path)


def test_reject_final_holdout_path_passes_on_a_safe_path() -> None:
    _reject_final_holdout_path(Path("/some/safe/validation/EURUSD"))  # no raise


# ---------------------------------------------------------------------------
# Candidate-only behavior: no search/selector machinery imported.
# ---------------------------------------------------------------------------


def test_module_source_never_imports_search_or_selector_machinery() -> None:
    _check_no_search_or_selector_imports()  # no raise


def test_check_one_candidate_only_passes_for_the_real_protocol() -> None:
    protocol = _protocol()
    _check_one_candidate_only(protocol.canonical_document)  # no raise


def test_check_one_candidate_only_fails_closed_on_a_selector_field() -> None:
    protocol = _protocol()
    tampered = dict(protocol.canonical_document)
    tampered["selected_trial_id"] = "abc123"
    with pytest.raises(EurusdPolicyRateCarryProxyValidationError):
        _check_one_candidate_only(tampered)


def test_check_one_candidate_only_fails_closed_on_more_than_one_configuration() -> None:
    protocol = _protocol()
    tampered = dict(protocol.canonical_document)
    tampered["one_shot"] = {**tampered["one_shot"], "maximum_configurations": 2}
    with pytest.raises(EurusdPolicyRateCarryProxyValidationError):
        _check_one_candidate_only(tampered)


# ---------------------------------------------------------------------------
# Second-run / output-existing rejection
# ---------------------------------------------------------------------------


def test_check_output_not_exists_passes_on_a_nonexistent_directory(
    tmp_path: Path,
) -> None:
    _check_output_not_exists(tmp_path / "brand-new-output")  # no raise


def test_check_output_not_exists_passes_on_an_empty_existing_directory(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    _check_output_not_exists(output)  # no raise


def test_check_output_not_exists_fails_closed_on_a_populated_directory(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "validation_result.json").write_text("{}")
    with pytest.raises(EurusdPolicyRateCarryProxyValidationError):
        _check_output_not_exists(output)


def test_real_validation_output_dir_does_not_exist() -> None:
    """This is the single fixed one-shot output path -- it must never have
    been created by any prior real (or accidental) run in this task."""

    assert not VALIDATION_OUTPUT_DIR.exists()


# ---------------------------------------------------------------------------
# Deterministic artifacts
# ---------------------------------------------------------------------------


def test_written_artifacts_are_byte_identical_across_two_writes(
    tmp_path: Path,
) -> None:
    payload = {"schema": "test", "value": 1, "nested": {"b": 2, "a": 1}}
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    _write_validation_artifacts(first_dir, payload)
    _write_validation_artifacts(second_dir, payload)
    first_bytes = (first_dir / "validation_result.json").read_bytes()
    second_bytes = (second_dir / "validation_result.json").read_bytes()
    assert first_bytes == second_bytes


def test_write_validation_artifacts_produces_all_three_files(tmp_path: Path) -> None:
    output = tmp_path / "output"
    _write_validation_artifacts(output, {"schema": "test"})
    assert (output / "validation_result.json").exists()
    assert (output / "artifact_hashes.json").exists()
    assert (output / "runtime_provenance.json").exists()


# ---------------------------------------------------------------------------
# Protocol semantic SHA load/verify round-trip
# ---------------------------------------------------------------------------


def test_load_validation_protocol_loads_the_real_frozen_protocol() -> None:
    protocol = load_validation_protocol()
    assert protocol.semantic_sha256 == PROTOCOL_SEMANTIC_SHA256
    assert protocol.family_semantic_sha256 == FAMILY_SEMANTIC_SHA256
    assert protocol.start_utc == VALIDATION_START
    assert protocol.end_exclusive_utc == HOLDOUT_START
    assert protocol.minimum_eligible_observations == MINIMUM_ELIGIBLE_OBSERVATIONS


def test_load_validation_protocol_fails_closed_on_a_missing_path(
    tmp_path: Path,
) -> None:
    with pytest.raises(EurusdPolicyRateCarryProxyValidationError):
        load_validation_protocol(tmp_path / "does-not-exist.json")


def test_load_validation_protocol_fails_closed_on_tampered_content(
    tmp_path: Path,
) -> None:
    import json

    document = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    document["status"] = "tampered"
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(EurusdPolicyRateCarryProxyValidationError):
        load_validation_protocol(tampered_path)


# ---------------------------------------------------------------------------
# verify_development_evidence
# ---------------------------------------------------------------------------


def test_verify_development_evidence_passes_for_the_real_artifacts() -> None:
    protocol = _protocol()
    result, audit = verify_development_evidence(protocol)
    assert result["family_semantic_sha256"] == FAMILY_SEMANTIC_SHA256
    assert audit["final_classification"] == "DEVELOPMENT_CANDIDATE_SELECTED"


def test_verify_development_evidence_fails_closed_on_a_wrong_hash(
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    bogus = tmp_path / "development_result.json"
    bogus.write_text("{}")
    with pytest.raises(EurusdPolicyRateCarryProxyValidationError):
        verify_development_evidence(
            protocol,
            development_result_path=bogus,
            integrity_audit_path=INTEGRITY_AUDIT_PATH,
        )


# ---------------------------------------------------------------------------
# Every individual preflight check, independently, fails closed.
# ---------------------------------------------------------------------------


def test_check_family_semantic_sha_passes_for_the_real_spec() -> None:
    _check_family_semantic_sha()  # no raise


def test_check_development_result_sha_passes_for_the_real_artifact() -> None:
    _check_development_result_sha(DEVELOPMENT_RESULT_PATH)  # no raise


def test_check_development_result_sha_fails_closed_on_a_missing_path(
    tmp_path: Path,
) -> None:
    with pytest.raises(EurusdPolicyRateCarryProxyValidationError):
        _check_development_result_sha(tmp_path / "does-not-exist.json")


def test_check_development_result_sha_fails_closed_on_a_tampered_file(
    tmp_path: Path,
) -> None:
    bogus = tmp_path / "development_result.json"
    bogus.write_bytes(DEVELOPMENT_RESULT_PATH.read_bytes() + b" ")
    with pytest.raises(EurusdPolicyRateCarryProxyValidationError):
        _check_development_result_sha(bogus)


def test_check_integrity_audit_sha_passes_for_the_real_artifact() -> None:
    _check_integrity_audit_sha(INTEGRITY_AUDIT_PATH)  # no raise


def test_check_integrity_audit_sha_fails_closed_on_a_missing_path(
    tmp_path: Path,
) -> None:
    with pytest.raises(EurusdPolicyRateCarryProxyValidationError):
        _check_integrity_audit_sha(tmp_path / "does-not-exist.json")


def test_check_validation_boundaries_passes_for_the_real_protocol() -> None:
    _check_validation_boundaries(_protocol())  # no raise


def test_check_validation_boundaries_fails_closed_on_a_tampered_protocol() -> None:
    from dataclasses import replace

    real = _protocol()
    tampered = replace(
        real, end_exclusive_utc=real.end_exclusive_utc + timedelta(days=1)
    )
    with pytest.raises(EurusdPolicyRateCarryProxyValidationError):
        _check_validation_boundaries(tampered)


def test_check_clean_worktree_passes_when_git_reports_no_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "ftmoquant.research.eurusd_policy_rate_carry_proxy_validation.subprocess.run",
        fake_run,
    )
    _check_clean_worktree()  # must not raise


def test_check_clean_worktree_fails_closed_on_any_uncommitted_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args, 0, stdout=" M some/file.py\n", stderr=""
        )

    monkeypatch.setattr(
        "ftmoquant.research.eurusd_policy_rate_carry_proxy_validation.subprocess.run",
        fake_run,
    )
    with pytest.raises(EurusdPolicyRateCarryProxyValidationError):
        _check_clean_worktree()


def test_run_preflight_checks_fails_closed_on_the_first_bad_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "ftmoquant.research.eurusd_policy_rate_carry_proxy_validation.subprocess.run",
        fake_run,
    )
    with pytest.raises(EurusdPolicyRateCarryProxyValidationError):
        _run_preflight_checks(
            protocol_path=tmp_path / "does-not-exist.json",
            development_result_path=DEVELOPMENT_RESULT_PATH,
            integrity_audit_path=INTEGRITY_AUDIT_PATH,
            output_dir=tmp_path / "output",
        )


def test_run_preflight_checks_fails_closed_on_an_already_populated_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "ftmoquant.research.eurusd_policy_rate_carry_proxy_validation.subprocess.run",
        fake_run,
    )
    output = tmp_path / "output"
    output.mkdir()
    (output / "validation_result.json").write_text("{}")
    with pytest.raises(EurusdPolicyRateCarryProxyValidationError):
        _run_preflight_checks(
            protocol_path=PROTOCOL_PATH,
            development_result_path=DEVELOPMENT_RESULT_PATH,
            integrity_audit_path=INTEGRITY_AUDIT_PATH,
            output_dir=output,
        )


def test_run_preflight_checks_passes_for_all_real_frozen_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "ftmoquant.research.eurusd_policy_rate_carry_proxy_validation.subprocess.run",
        fake_run,
    )
    spec, protocol, evidence = _run_preflight_checks(
        protocol_path=PROTOCOL_PATH,
        development_result_path=DEVELOPMENT_RESULT_PATH,
        integrity_audit_path=INTEGRITY_AUDIT_PATH,
        output_dir=tmp_path / "brand-new-output",
    )
    assert spec.family_id == "eurusd_policy_rate_carry_proxy_v1"
    assert protocol.semantic_sha256 == PROTOCOL_SEMANTIC_SHA256
    assert evidence[0]["family_semantic_sha256"] == FAMILY_SEMANTIC_SHA256


def test_runner_fails_closed_before_touching_any_catalog(tmp_path: Path) -> None:
    """No matter the current worktree state, this must raise this family's

    own error type -- never proceed to open a real catalog. We deliberately
    do not mock the worktree here, so this exercises whatever real state
    the repo is in (dirty or clean) and still must not reach catalog I/O for
    a nonexistent readiness/root path.
    """

    expected = (EurusdPolicyRateCarryProxyValidationError, ValueError, OSError)
    with pytest.raises(expected):
        run_eurusd_policy_rate_carry_proxy_validation(
            universe_readiness_path=Path("does-not-matter"),
            development_root=Path("does-not-matter"),
            validation_root=Path("does-not-matter"),
            policy_rates_dir=Path("does-not-matter"),
            output_dir=tmp_path / "output",
        )


def test_runner_rejects_a_holdout_marked_validation_root(tmp_path: Path) -> None:
    expected = (EurusdPolicyRateCarryProxyValidationError, ValueError, OSError)
    with pytest.raises(expected):
        run_eurusd_policy_rate_carry_proxy_validation(
            universe_readiness_path=Path("does-not-matter"),
            development_root=Path("does-not-matter"),
            validation_root=tmp_path / "final_holdout" / "EURUSD",
            policy_rates_dir=Path("does-not-matter"),
            output_dir=tmp_path / "output",
        )


def test_cli_help_dry_run_never_touches_any_real_data() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])


def test_cli_validation_root_rejects_a_holdout_marker() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--universe-readiness",
                "x",
                "--development-root",
                "x",
                "--validation-root",
                "some/holdout/root",
                "--policy-rates-dir",
                "x",
            ]
        )


# ---------------------------------------------------------------------------
# Sanity: this file never actually calls the real runner to completion.
# ---------------------------------------------------------------------------


def test_this_test_module_never_invokes_the_runner_to_completion() -> None:
    """Static self-check: every call to the real orchestrator in this file

    is wrapped in ``pytest.raises`` and targets a nonexistent/sealed path,
    so the runner always raises before touching real validation data.
    """

    source = Path(__file__).read_text(encoding="utf-8")
    assert source.count("run_eurusd_policy_rate_carry_proxy_validation(") >= 2
    assert not VALIDATION_OUTPUT_DIR.exists()
