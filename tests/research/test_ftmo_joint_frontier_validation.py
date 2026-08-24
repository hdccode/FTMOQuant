from __future__ import annotations

import ast
import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

import ftmoquant.research.ftmo_joint_frontier as joint
import ftmoquant.research.ftmo_joint_frontier_validation as validation
from ftmoquant.prop_rules.loader import load_prop_rule_set
from ftmoquant.research.ftmo_pass_probability.path_extraction import (
    TradeRecord,
    ValidationTradePath,
)
from ftmoquant.research.ftmo_pass_probability.reporting import (
    BinomialEstimate,
    CertaintyTier,
    PolicySummary,
)
from ftmoquant.research.stage_g import HOLDOUT_START, VALIDATION_START

RULES = load_prop_rule_set(joint.FTMO_RULES_PATH)


def _estimate(value: float, *, lower: float | None = None) -> BinomialEstimate:
    return BinomialEstimate(
        successes=round(value * 100_000),
        trials=100_000,
        estimate=value,
        ci_lower_95=value if lower is None else lower,
        ci_upper_95=value,
    )


def _method_result(
    method: str,
    *,
    pass_both: float = 0.70,
    pass_both_lower: float = 0.65,
    fail_daily: float = 0.01,
    fail_max: float = 0.25,
    p90_days: float = 170,
    p95_drawdown: float = 0.17,
) -> validation.ValidationMethodResult:
    summary = PolicySummary(
        policy_id=validation.FROZEN_POLICY.policy_id,
        method=method,
        replications=100_000,
        pass_challenge=_estimate(0.80),
        pass_verification_given_challenge=_estimate(0.875),
        pass_both=_estimate(pass_both, lower=pass_both_lower),
        fail_daily_loss=_estimate(fail_daily),
        fail_max_loss=_estimate(fail_max),
        censoring_rate=_estimate(0.04),
        median_trading_days_to_pass_both=80,
        p90_trading_days_to_pass_both=p90_days,
        p95_trading_days_to_pass_both=190,
        median_max_drawdown=0.09,
        p95_max_drawdown=p95_drawdown,
        certainty_tier=CertaintyTier.PLAUSIBLE,
    )
    return validation.ValidationMethodResult(
        method=method,  # type: ignore[arg-type]
        run=joint.JointMethodRun(summary, 48, 120, 0.14),
    )


def _trade() -> TradeRecord:
    start_ns = int(VALIDATION_START.timestamp() * 1_000_000_000)
    return TradeRecord(
        trade_index=0,
        entry_ns=start_ns + 1,
        exit_ns=start_ns + 2,
        exit_reason="target",
        net_r=Decimal("1"),
        original_realized_pnl=Decimal("100"),
        original_risk_budget=Decimal("100"),
        usd_risk_per_unit=Decimal("0.001"),
    )


def _path() -> ValidationTradePath:
    return ValidationTradePath((_trade(),), Path("validation"), "a" * 64, "b" * 64)


def test_exact_single_frozen_policy_and_methodology() -> None:
    assert validation.FROZEN_POLICIES == (validation.FROZEN_POLICY,)
    assert validation.FROZEN_POLICY.policy_id == "A2.20x_U21.25x"
    assert validation.FROZEN_POLICY.a_multiplier == Decimal("2.20")
    assert validation.FROZEN_POLICY.u2_multiplier == Decimal("1.25")
    assert validation.PATH_COUNT == 100_000
    assert validation.SEED == 20260819
    assert validation.METHODS == ("stationary", "circular")


def test_cli_has_no_policy_multiplier_method_seed_or_path_overrides() -> None:
    options = {
        option
        for action in validation.build_parser()._actions  # noqa: SLF001
        for option in action.option_strings
    }
    assert options == {
        "-h",
        "--help",
        "--validation-root",
        "--universe-readiness",
        "--output",
    }


def test_exact_validation_partition_boundaries() -> None:
    assert VALIDATION_START.isoformat() == "2023-04-11T00:00:00+00:00"
    assert HOLDOUT_START.isoformat() == "2024-08-21T00:00:00+00:00"


def test_partition_guard_rejects_development_and_holdout_events() -> None:
    valid = _path()
    validation.validate_partition_events(valid, ())
    holdout_ns = int(HOLDOUT_START.timestamp() * 1_000_000_000)
    with pytest.raises(joint.JointFrontierError, match="outside"):
        validation.validate_partition_events(
            valid, (SimpleNamespace(entry_ns=holdout_ns, exit_ns=holdout_ns + 1),)
        )


def test_development_reference_is_read_and_remains_not_eligible() -> None:
    reference = validation.load_development_reference()
    assert reference["pass_both"] == 0.72798
    assert reference["median_trading_days_to_pass_both"] == 78
    assert reference["p90_trading_days_to_pass_both"] == 158
    assert reference["development_classification"] == "NOT_ELIGIBLE"


def test_confirmation_thresholds_and_classification_are_exact() -> None:
    assert validation.CONFIRMATION_THRESHOLDS == {
        "stationary_pass_both_ge": 0.65,
        "stationary_pass_both_lower_ci_ge": 0.60,
        "stationary_fail_daily_loss_le": 0.02,
        "stationary_fail_max_loss_le": 0.30,
        "stationary_p90_days_le": 180.0,
        "stationary_p95_max_drawdown_le": 0.18,
        "circular_pass_both_ge": 0.60,
        "circular_fail_daily_loss_le": 0.03,
        "circular_fail_max_loss_le": 0.35,
    }
    confirmed = validation.classify_validation(
        (_method_result("stationary"), _method_result("circular"))
    )
    assert confirmed.classification == "VALIDATION_CONFIRMED"
    failed = validation.classify_validation(
        (
            _method_result("stationary", pass_both_lower=0.5999),
            _method_result("circular"),
        )
    )
    assert failed.classification == "VALIDATION_NOT_CONFIRMED"
    assert failed.failed_checks == ("B_stationary_pass_both_lower_ci_ge_0_60",)


def test_shared_streaming_engine_receives_only_frozen_policy_and_methods(
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_prepare(groups):
        return ()

    def fake_scaled(*args, **kwargs):
        return ()

    def fake_run(**kwargs):
        calls.append(kwargs)
        return _method_result(kwargs["method"]).run

    monkeypatch.setattr(joint, "prepare_joint_groups_scaling", fake_prepare)
    monkeypatch.setattr(joint, "precompute_prepared_scaled_events", fake_scaled)
    monkeypatch.setattr(joint, "run_joint_policy_method_streaming", fake_run)
    results = validation.run_validation_methods(
        groups=(), rules=RULES, block_size=2
    )
    assert tuple(result.method for result in results) == validation.METHODS
    assert {call["policy"] for call in calls} == {validation.FROZEN_POLICY}
    assert {call["path_count"] for call in calls} == {100_000}
    assert {call["seed"] for call in calls} == {20260819}


def test_no_rescue_or_independent_bootstrap_structure() -> None:
    source = Path(validation.__file__).read_text(encoding="utf-8")
    assert "draw_index_path(" not in source
    assert "simulate_phase(" not in source
    assert "run_refinement_screen(" not in source
    assert "A2.20x_U21.00x" not in source
    assert "A2.10x_U21.25x" not in source
    assert "A2.30x_U21.00x" not in source
    tree = ast.parse(source)
    assert tree is not None


def test_output_refusal_precedes_validation_loading_and_mc(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "existing"
    output.mkdir()

    def forbidden(*args, **kwargs):
        raise AssertionError("long work reached")

    monkeypatch.setattr(validation, "load_strategy_a_validation", forbidden)
    monkeypatch.setattr(validation, "run_validation_methods", forbidden)
    with pytest.raises(joint.JointFrontierError, match="refusing to overwrite"):
        validation.main(
            ["--validation-root", str(tmp_path / "validation"), "--output", str(output)]
        )


def test_seven_artifact_schema_and_no_rescue_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(validation, "_file_sha256", lambda path: "c" * 64)
    monkeypatch.setattr(validation.refinement, "_git_commit", lambda: "d" * 40)
    monkeypatch.setattr(
        validation.refinement,
        "_dependency_versions",
        lambda: {"python": "3.12.test"},
    )
    results = (_method_result("stationary"), _method_result("circular"))
    development = {
        "policy_id": validation.FROZEN_POLICY.policy_id,
        "paths": 100_000,
        "method": "stationary",
        "development_classification": "NOT_ELIGIBLE",
        "failed_development_criteria": ["median", "p90"],
        "pass_challenge": 0.80,
        "pass_both": 0.70,
        "fail_daily_loss": 0.0,
        "fail_max_loss": 0.20,
        "censoring_rate": 0.05,
        "median_trading_days_to_pass_both": 78.0,
        "p90_trading_days_to_pass_both": 158.0,
        "p95_trading_days_to_pass_both": 184.0,
        "median_max_drawdown": 0.09,
        "p95_max_drawdown": 0.15,
    }
    diagnostics = joint.DiversificationDiagnostics(
        aligned_daily_correlation=0.1,
        downside_correlation=0.2,
        correlation_conditional_on_a_losing=0.3,
        correlation_conditional_on_u2_losing=0.4,
        fraction_a_losing_days_with_u2_positive=0.5,
        fraction_u2_losing_days_with_a_positive=0.6,
        maximum_same_day_combined_loss_usd="-1000",
        overlap_group_count=1,
        total_group_count=2,
        fraction_a_trades_overlapping_u2=0.5,
        fraction_u2_trades_overlapping_a=0.5,
    )
    output = tmp_path / "out"
    validation.write_results(
        output_dir=output,
        results=results,
        development=development,
        block_length=joint.FrozenJointBlockLength(2.2, 2.5, 2, 10, 0.1, 1),
        diagnostics=diagnostics,
        a_path=_path(),
        u2_episode_count=1,
        validation_readiness_path=tmp_path / "readiness.json",
    )
    assert {path.name for path in output.iterdir()} == set(
        validation.EXPECTED_ARTIFACT_FILENAMES
    )
    classification = json.loads(
        (output / "validation_classification.json").read_text(encoding="utf-8")
    )
    assert classification["no_rescue"] is True
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["final_holdout_accessed"] is False
    assert metadata["policy_count"] == 1
