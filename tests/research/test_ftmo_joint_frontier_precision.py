from __future__ import annotations

import ast
import csv
import json
from decimal import Decimal
from pathlib import Path

import pytest

import ftmoquant.research.ftmo_joint_frontier as joint
import ftmoquant.research.ftmo_joint_frontier_precision as precision
import ftmoquant.research.ftmo_pass_probability.bootstrap as bootstrap
from ftmoquant.prop_rules.loader import load_prop_rule_set
from ftmoquant.research.ftmo_pass_probability.path_extraction import TradeRecord
from ftmoquant.research.ftmo_pass_probability.reporting import (
    BinomialEstimate,
    CertaintyTier,
    PolicySummary,
    wilson_score_interval,
)

RULES = load_prop_rule_set(joint.FTMO_RULES_PATH)
DAY_NS = 86_400_000_000_000
HOUR_NS = 3_600_000_000_000


def _trade(index: int, net_r: str) -> TradeRecord:
    risk = Decimal("1000")
    return TradeRecord(
        trade_index=index,
        entry_ns=index * DAY_NS,
        exit_ns=index * DAY_NS + HOUR_NS,
        exit_reason="stop" if Decimal(net_r) < 0 else "target",
        net_r=Decimal(net_r),
        original_realized_pnl=Decimal(net_r) * risk,
        original_risk_budget=risk,
        usd_risk_per_unit=Decimal("100"),
    )


def _estimate(value: float, *, lower: float | None = None, upper: float | None = None):
    return BinomialEstimate(
        successes=round(value * 100_000),
        trials=100_000,
        estimate=value,
        ci_lower_95=value if lower is None else lower,
        ci_upper_95=value if upper is None else upper,
    )


def _result(
    policy_id: str = "A2.20x_U21.25x",
    *,
    pass_both: float = 0.72,
    pass_both_lower: float = 0.71,
    median_days: float = 74,
    p90_days: float = 149,
    fail_daily: float = 0.0,
    fail_daily_upper: float = 0.001,
    fail_max: float = 0.24,
    fail_max_upper: float = 0.245,
    p95_drawdown: float = 0.15,
) -> precision.PrecisionResult:
    policy = next(p for p in precision.FROZEN_POLICIES if p.policy_id == policy_id)
    summary = PolicySummary(
        policy_id=policy_id,
        method="stationary",
        replications=100_000,
        pass_challenge=_estimate(0.82),
        pass_verification_given_challenge=_estimate(0.88),
        pass_both=_estimate(pass_both, lower=pass_both_lower),
        fail_daily_loss=_estimate(fail_daily, upper=fail_daily_upper),
        fail_max_loss=_estimate(fail_max, upper=fail_max_upper),
        censoring_rate=_estimate(0.03),
        median_trading_days_to_pass_both=median_days,
        p90_trading_days_to_pass_both=p90_days,
        p95_trading_days_to_pass_both=p90_days + 25,
        median_max_drawdown=0.09,
        p95_max_drawdown=p95_drawdown,
        certainty_tier=CertaintyTier.PLAUSIBLE,
    )
    return precision.PrecisionResult(
        policy=policy,
        run=joint.JointMethodRun(
            summary=summary,
            median_trading_days_to_pass_challenge=45,
            p75_trading_days_to_pass_both=105,
            p90_max_drawdown=0.13,
        ),
    )


def test_exact_three_frozen_policies_method_paths_and_seed() -> None:
    assert precision.FROZEN_POLICY_IDS == (
        "A2.20x_U21.25x",
        "A2.30x_U21.00x",
        "A2.30x_U21.25x",
    )
    assert [(p.a_multiplier, p.u2_multiplier) for p in precision.FROZEN_POLICIES] == [
        (Decimal("2.20"), Decimal("1.25")),
        (Decimal("2.30"), Decimal("1.00")),
        (Decimal("2.30"), Decimal("1.25")),
    ]
    assert precision.PRECISION_PATH_COUNT == 100_000
    assert precision.PRECISION_METHOD == "stationary"
    assert precision.PRECISION_SEED == 20260819


def test_cli_has_no_policy_multiplier_path_method_or_grid_override() -> None:
    options = {
        option
        for action in precision.build_parser()._actions  # noqa: SLF001
        for option in action.option_strings
    }
    assert options == {
        "-h",
        "--help",
        "--catalog-root",
        "--universe-readiness",
        "--output",
    }


def test_precision_batch_routes_exact_constants_through_shared_joint_engine(
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        return _result(kwargs["policy"].policy_id).run

    monkeypatch.setattr(joint, "run_joint_policy_method_streaming", fake_run)
    results = precision.run_precision_batch(groups=(), rules=RULES, block_size=2)
    assert tuple(result.policy.policy_id for result in results) == (
        precision.FROZEN_POLICY_IDS
    )
    assert len(calls) == 3
    assert {call["method"] for call in calls} == {"stationary"}
    assert {call["path_count"] for call in calls} == {100_000}
    assert {call["seed"] for call in calls} == {20260819}
    assert {call["challenge_horizon_ns"] for call in calls} == {
        joint.DEFAULT_HORIZON_NS
    }
    assert {call["verification_horizon_ns"] for call in calls} == {
        joint.DEFAULT_HORIZON_NS
    }


def test_same_jointgroup_bootstrap_and_ftmo_engine_is_reused_not_forked() -> None:
    source = Path(precision.__file__).read_text(encoding="utf-8")
    assert "draw_index_path(" not in source
    assert "simulate_phase(" not in source
    assert "simulate_two_phase_joint_path(" not in source
    assert "scale_joint_group(" not in source
    assert "summarize_policy(" not in source
    assert "joint.run_joint_policy_method_streaming(" in source
    assert joint.draw_index_path is bootstrap.draw_index_path


def test_bootstrap_indices_are_exact_for_1000_precision_replication_ids() -> None:
    for replication in range(1000):
        seed = precision.PRECISION_SEED + replication
        oracle = bootstrap.draw_index_path(
            289,
            method="stationary",
            block_size=2,
            seed=seed,
            min_length=868,
        )
        production = joint.draw_index_path(
            289,
            method="stationary",
            block_size=2,
            seed=seed,
            min_length=868,
        )
        assert production == oracle


def test_streaming_aggregation_is_exactly_equal_to_materialized_oracle() -> None:
    groups = joint.build_joint_groups(
        tuple(_trade(i, "1.2" if i % 3 else "-0.5") for i in range(12)),
        (),
    )
    policy = precision.FROZEN_POLICIES[0]
    kwargs = {
        "groups": groups,
        "rules": RULES,
        "block_size": 2,
        "policy": policy,
        "method": "stationary",
        "path_count": 100,
        "seed": precision.PRECISION_SEED,
        "challenge_horizon_ns": 90 * DAY_NS,
        "verification_horizon_ns": 90 * DAY_NS,
    }
    oracle = joint.run_joint_policy_method(**kwargs)
    streamed = joint.run_joint_policy_method_streaming(**kwargs)
    assert streamed == oracle


def test_historical_comparison_reads_exact_stationary_rows(tmp_path: Path) -> None:
    path = tmp_path / "refinement.csv"
    fieldnames = [
        "policy_id",
        "method",
        "pass_both",
        "fail_daily_loss",
        "fail_max_loss",
        "median_trading_days_to_pass_both",
        "p90_trading_days_to_pass_both",
        "p95_max_drawdown",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, policy_id in enumerate(precision.FROZEN_POLICY_IDS):
            for method in ("stationary", "circular"):
                writer.writerow(
                    {
                        "policy_id": policy_id,
                        "method": method,
                        "pass_both": 0.70 + index / 100,
                        "fail_daily_loss": 0,
                        "fail_max_loss": 0.20 + index / 100,
                        "median_trading_days_to_pass_both": 75 + index,
                        "p90_trading_days_to_pass_both": 150 + index,
                        "p95_max_drawdown": 0.14 + index / 100,
                    }
                )
    loaded = precision.load_historical_metrics(path)
    assert set(loaded) == set(precision.FROZEN_POLICY_IDS)
    assert loaded["A2.30x_U21.00x"].pass_both == pytest.approx(0.71)
    rows = precision._convergence_rows(  # noqa: SLF001
        tuple(_result(policy_id) for policy_id in precision.FROZEN_POLICY_IDS),
        loaded,
    )
    assert rows[0]["delta_pass_both"] == pytest.approx(0.02)
    assert rows[1]["delta_p90_days"] == -2


def test_repository_historical_values_are_not_approximations() -> None:
    loaded = precision.load_historical_metrics()
    assert loaded["A2.20x_U21.25x"].pass_both == 0.7268
    assert loaded["A2.20x_U21.25x"].fail_max_loss == 0.24035
    assert loaded["A2.30x_U21.00x"].median_days == 75
    assert loaded["A2.30x_U21.25x"].p90_days == 151


def test_wilson_interval_known_value_and_exact_counts() -> None:
    estimate = wilson_score_interval(700, 1000)
    assert estimate.successes == 700
    assert estimate.trials == 1000
    assert estimate.estimate == 0.7
    assert estimate.ci_lower_95 == pytest.approx(0.6708761390827828)
    assert estimate.ci_upper_95 == pytest.approx(0.7275931575229951)


def test_original_eligibility_and_robust_diagnostic_are_separate_and_exact() -> None:
    assert precision.ELIGIBILITY_RULE == {
        "stationary_pass_both_ge": 0.70,
        "stationary_median_trading_days_to_pass_both_le": 75,
        "stationary_p90_trading_days_to_pass_both_le": 150,
        "stationary_fail_daily_loss_le": 0.02,
        "stationary_fail_max_loss_le": 0.25,
    }
    eligible_not_robust = _result(pass_both_lower=0.6999)
    verdict = precision.evaluate_precision(eligible_not_robust)
    assert verdict.point_estimate_eligible is True
    assert verdict.robust_to_mc_uncertainty is False
    assert verdict.classification == "DEPLOYMENT_ELIGIBLE"
    assert verdict.failed_robust_criteria == ("pass_both_lower_ci_ge_0_70",)


def test_classification_never_rescues_single_or_multiple_misses() -> None:
    borderline = precision.evaluate_precision(_result(p90_days=151))
    assert borderline.point_estimate_eligible is False
    assert borderline.classification == "PRECISION_BORDERLINE"
    rejected = precision.evaluate_precision(_result(median_days=76, p90_days=151))
    assert rejected.point_estimate_eligible is False
    assert rejected.classification == "NOT_ELIGIBLE"


def test_frozen_final_ranking_and_null_when_none_eligible() -> None:
    slower = _result("A2.20x_U21.25x", median_days=75, pass_both=0.74)
    faster = _result("A2.30x_U21.00x", median_days=74, pass_both=0.71)
    assert precision.select_policy((slower, faster)) is faster
    assert precision.SELECTION_RULE == (
        "lowest median trading days to pass both",
        "highest pass_both",
        "lowest fail_max_loss",
        "lowest p95 max drawdown",
        "lower total gross multiplier",
        "lexicographic policy_id",
    )
    assert precision.select_policy(
        (_result(p90_days=151), _result("A2.30x_U21.00x", median_days=76))
    ) is None


def test_output_refusal_precedes_data_loading_and_monte_carlo(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "existing"
    output.mkdir()

    def forbidden(*args, **kwargs):
        raise AssertionError("work was reached")

    monkeypatch.setattr(precision, "load_historical_metrics", forbidden)
    monkeypatch.setattr(precision, "run_precision_batch", forbidden)
    with pytest.raises(joint.JointFrontierError, match="refusing to overwrite"):
        precision.main(
            [
                "--catalog-root",
                str(tmp_path / "catalog"),
                "--universe-readiness",
                str(tmp_path / "readiness.json"),
                "--output",
                str(output),
            ]
        )


def test_validation_holdout_and_automatic_refinement_are_inaccessible() -> None:
    source = Path(precision.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "ftmoquant.research.alpha_lab.validation" not in imports
    assert "ftmoquant.data.oanda_alpha_lab_validation" not in imports
    assert "Partition.VALIDATION" not in source
    assert "holdout_path" not in source
    assert "run_refinement_screen(" not in source
    assert precision.DEVELOPMENT_ONLY_STATEMENT.startswith("Precision-only")


def test_expected_five_artifacts_and_probability_counts(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(precision.refinement, "_git_commit", lambda: "a" * 40)
    monkeypatch.setattr(
        precision.refinement,
        "_dependency_versions",
        lambda: {"python": "3.12.test"},
    )
    monkeypatch.setattr(precision, "_sha256", lambda path: "b" * 64)
    output = tmp_path / "out"
    historical = {
        policy_id: precision.HistoricalMetrics(
            policy_id, 0.70, 0.0, 0.20, 75, 150, 0.14
        )
        for policy_id in precision.FROZEN_POLICY_IDS
    }
    precision.write_results(
        output_dir=output,
        results=tuple(_result(policy_id) for policy_id in precision.FROZEN_POLICY_IDS),
        historical=historical,
        parity={"same_joint_path_builder": True},
        block_length=joint.FrozenJointBlockLength(2.2, 2.5, 2, 289, 0.1, 1),
    )
    assert {path.name for path in output.iterdir()} == set(
        precision.EXPECTED_ARTIFACT_FILENAMES
    )
    with (output / "precision_results.csv").open(newline="", encoding="utf-8") as f:
        first = next(csv.DictReader(f))
    assert first["pass_both_successes"] == "72000"
    assert first["pass_both_trials"] == "100000"
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["frozen_policy_ids"] == list(precision.FROZEN_POLICY_IDS)
    assert metadata["validation_accessed"] is False
    assert metadata["holdout_accessed"] is False
    assert metadata["no_automatic_further_refinement"] is True
