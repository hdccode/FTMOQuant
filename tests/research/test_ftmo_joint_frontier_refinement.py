from __future__ import annotations

import ast
import csv
import json
from decimal import Decimal
from pathlib import Path

import pytest

import ftmoquant.research.ftmo_joint_frontier as coarse
import ftmoquant.research.ftmo_joint_frontier_refinement as refinement
from ftmoquant.prop_rules.loader import load_prop_rule_set
from ftmoquant.research.ftmo_pass_probability.path_extraction import TradeRecord
from ftmoquant.research.ftmo_pass_probability.reporting import (
    BinomialEstimate,
    CertaintyTier,
    PolicySummary,
)

RULES = load_prop_rule_set(coarse.FTMO_RULES_PATH)
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


def _estimate(value: float) -> BinomialEstimate:
    return BinomialEstimate(
        successes=round(value * 1000),
        trials=1000,
        estimate=value,
        ci_lower_95=value,
        ci_upper_95=value,
    )


def _summary(
    policy_id: str = "A2.10x_U20.75x",
    *,
    pass_both: float = 0.72,
    median_days: float = 70,
    p90_days: float = 140,
    fail_daily: float = 0.01,
    fail_max: float = 0.20,
    p95_drawdown: float = 0.15,
) -> coarse.JointPolicySummary:
    def policy_summary(method: str) -> PolicySummary:
        return PolicySummary(
            policy_id=policy_id,
            method=method,
            replications=1000,
            pass_challenge=_estimate(0.80),
            pass_verification_given_challenge=_estimate(0.90),
            pass_both=_estimate(pass_both),
            fail_daily_loss=_estimate(fail_daily),
            fail_max_loss=_estimate(fail_max),
            censoring_rate=_estimate(0.07),
            median_trading_days_to_pass_both=median_days,
            p90_trading_days_to_pass_both=p90_days,
            p95_trading_days_to_pass_both=p90_days + 20,
            median_max_drawdown=0.08,
            p95_max_drawdown=p95_drawdown,
            certainty_tier=CertaintyTier.STRONG,
        )

    policy = next(
        policy
        for policy in refinement.REFINEMENT_POLICY_GRID
        if policy.policy_id == policy_id
    )
    return coarse.JointPolicySummary(
        policy=policy,
        stationary=policy_summary("stationary"),
        circular=policy_summary("circular"),
        median_trading_days_to_pass_challenge=45,
        p75_trading_days_to_pass_both=100,
    )


def _references() -> dict[tuple[str, str], refinement.CoarseReferenceMetrics]:
    result: dict[tuple[str, str], refinement.CoarseReferenceMetrics] = {}
    for policy_id, pass_both in (
        ("A2.0x_U21.25x", 0.75),
        ("A2.5x_U20.75x", 0.68),
    ):
        for method in ("stationary", "circular"):
            result[(policy_id, method)] = refinement.CoarseReferenceMetrics(
                policy_id=policy_id,
                method=method,
                pass_both=pass_both,
                median_days=88 if policy_id.startswith("A2.0") else 68,
                p90_days=173 if policy_id.startswith("A2.0") else 140,
                fail_daily_loss=0.0,
                fail_max_loss=0.20 if policy_id.startswith("A2.0") else 0.30,
            )
    return result


def test_exact_frozen_12_policy_grid_and_ids() -> None:
    assert refinement.REFINEMENT_A_MULTIPLIERS == tuple(
        Decimal(value) for value in ("2.10", "2.20", "2.30", "2.40")
    )
    assert refinement.REFINEMENT_U2_MULTIPLIERS == tuple(
        Decimal(value) for value in ("0.75", "1.00", "1.25")
    )
    assert len(refinement.REFINEMENT_POLICY_GRID) == 12
    assert refinement.REFINEMENT_POLICY_GRID[0].policy_id == "A2.10x_U20.75x"
    assert refinement.REFINEMENT_POLICY_GRID[-1].policy_id == "A2.40x_U21.25x"
    assert {
        policy.a_multiplier for policy in refinement.REFINEMENT_POLICY_GRID
    } == set(refinement.REFINEMENT_A_MULTIPLIERS)
    assert {
        policy.u2_multiplier for policy in refinement.REFINEMENT_POLICY_GRID
    } == set(refinement.REFINEMENT_U2_MULTIPLIERS)


def test_cli_has_no_grid_seed_path_count_or_method_overrides() -> None:
    options = {
        option
        for action in refinement.build_parser()._actions  # noqa: SLF001
        for option in action.option_strings
    }
    assert options == {
        "-h",
        "--help",
        "--catalog-root",
        "--universe-readiness",
        "--output",
    }


def test_refinement_routes_only_its_grid_through_shared_engine(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_shared(**kwargs):
        captured.update(kwargs)
        return ()

    monkeypatch.setattr(coarse, "run_joint_sizing_screen", fake_shared)
    result = refinement.run_refinement_screen(
        groups=(), rules=RULES, block_size=2, path_count=1
    )
    assert result == ()
    assert captured["policies"] is refinement.REFINEMENT_POLICY_GRID
    assert captured["path_count"] == 1
    assert captured["seed"] == 20260819


def test_coarse_policy_has_identical_deterministic_result_via_shared_path() -> None:
    groups = coarse.build_joint_groups(
        tuple(_trade(i, "1.2" if i % 2 == 0 else "-0.4") for i in range(8)),
        (),
    )
    policy = coarse.JointPolicy("coarse-control", Decimal("2.0"), Decimal("0.00"))
    kwargs = {
        "groups": groups,
        "rules": RULES,
        "block_size": 2,
        "path_count": 3,
        "seed": coarse.SCREEN_SEED,
        "challenge_horizon_ns": 30 * DAY_NS,
        "verification_horizon_ns": 30 * DAY_NS,
    }
    direct = coarse.run_joint_sizing_screen(policies=(policy,), **kwargs)
    routed = refinement._run_shared_screen(policies=(policy,), **kwargs)  # noqa: SLF001
    assert routed == direct


def test_shared_machinery_and_frozen_rules_are_not_forked() -> None:
    source = Path(refinement.__file__).read_text(encoding="utf-8")
    assert "draw_index_path(" not in source
    assert "simulate_phase(" not in source
    assert "scale_joint_group(" not in source
    assert "summarize_policy(" not in source
    assert refinement.ELIGIBILITY_RULE == {
        "stationary_pass_both_ge": 0.70,
        "stationary_median_trading_days_to_pass_both_le": 75,
        "stationary_p90_trading_days_to_pass_both_le": 150,
        "stationary_fail_daily_loss_le": 0.02,
        "stationary_fail_max_loss_le": 0.25,
    }
    assert refinement.SELECTION_RULE == (
        "lowest median trading days to pass both",
        "highest pass_both",
        "lowest fail_max_loss",
        "lowest p95 max drawdown",
        "lower total gross multiplier",
        "lexicographic policy_id",
    )
    assert coarse.compute_pareto_frontier is refinement.joint.compute_pareto_frontier


def test_multiplier_scaling_remains_exactly_shared_and_linear() -> None:
    group = coarse.build_joint_groups((_trade(0, "-1"),), ())[0]
    event = coarse.scale_joint_group(
        group, a_multiplier=Decimal("2.30"), u2_multiplier=Decimal("1.25")
    )
    baseline = coarse.scale_joint_group(
        group, a_multiplier=Decimal("1.00"), u2_multiplier=Decimal("0.00")
    )
    assert event.realized_pnl == baseline.realized_pnl * Decimal("2.30")
    assert event.floor_equity_delta == baseline.floor_equity_delta * Decimal("2.30")


def test_coarse_reference_rows_and_deltas_are_read_from_artifact(
    tmp_path: Path,
) -> None:
    path = tmp_path / "coarse.csv"
    fieldnames = [
        "policy_id",
        "method",
        "pass_both",
        "median_trading_days_to_pass_both",
        "p90_trading_days_to_pass_both",
        "fail_daily_loss",
        "fail_max_loss",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for reference in _references().values():
            writer.writerow(
                {
                    "policy_id": reference.policy_id,
                    "method": reference.method,
                    "pass_both": reference.pass_both,
                    "median_trading_days_to_pass_both": reference.median_days,
                    "p90_trading_days_to_pass_both": reference.p90_days,
                    "fail_daily_loss": reference.fail_daily_loss,
                    "fail_max_loss": reference.fail_max_loss,
                }
            )
    loaded = refinement.load_coarse_references(path)
    assert loaded[("A2.0x_U21.25x", "stationary")].pass_both == 0.75
    row = refinement._screen_rows((_summary(),), loaded)[0]  # noqa: SLF001
    assert row["delta_vs_A2.0x_U21.25x_pass_both"] == pytest.approx(-0.03)
    assert row["delta_vs_A2.5x_U20.75x_median_days"] == 2
    assert row["delta_vs_A2.5x_U20.75x_fail_max_loss"] == pytest.approx(-0.10)


def test_daily_loss_cliff_diagnostic_boundaries() -> None:
    near = refinement._stationary_diagnostic(  # noqa: SLF001
        _summary(fail_daily=0.05, fail_max=0.10)
    )
    assert near["combined_failure_rate"] == pytest.approx(0.15)
    assert near["distance_from_daily_failure_eligibility_ceiling"] == pytest.approx(
        -0.03
    )
    assert near["near_daily_loss_cliff"] is True
    zero = refinement._stationary_diagnostic(  # noqa: SLF001
        _summary(fail_daily=0.0)
    )
    assert zero["near_daily_loss_cliff"] is False


def test_output_refusal_precedes_data_loading_and_monte_carlo(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "existing"
    output.mkdir()

    def forbidden(*args, **kwargs):
        raise AssertionError("expensive work was reached")

    monkeypatch.setattr(refinement, "load_coarse_references", forbidden)
    monkeypatch.setattr(coarse, "load_strategy_a_development", forbidden)
    monkeypatch.setattr(refinement, "run_refinement_screen", forbidden)
    with pytest.raises(coarse.JointFrontierError, match="refusing to overwrite"):
        refinement.main(
            [
                "--catalog-root",
                str(tmp_path / "catalog"),
                "--universe-readiness",
                str(tmp_path / "readiness.json"),
                "--output",
                str(output),
            ]
        )


def test_refinement_imports_no_validation_or_holdout_machinery() -> None:
    source = Path(refinement.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    forbidden = {
        "ftmoquant.data.oanda_alpha_lab_validation",
        "ftmoquant.research.alpha_lab.validation",
        "ftmoquant.research.ftmo_pass_probability.validation_diagnostic",
    }
    assert imported.isdisjoint(forbidden)
    assert "Partition.VALIDATION" not in source
    assert "holdout_path" not in source


def test_all_six_artifacts_include_rules_diagnostics_and_precision_plan(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(refinement, "_git_commit", lambda: "a" * 40)
    monkeypatch.setattr(
        refinement, "_dependency_versions", lambda: {"python": "3.12.test"}
    )
    output = tmp_path / "out"
    block = coarse.FrozenJointBlockLength(2.2, 2.5, 2, 10, 0.1, 1)
    refinement._write_results(  # noqa: SLF001
        output_dir=output,
        summaries=(_summary(),),
        references=_references(),
        parity={"same_joint_path_builder": True},
        block_length=block,
    )
    assert {path.name for path in output.iterdir()} == set(
        refinement.EXPECTED_ARTIFACT_FILENAMES
    )
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["label"] == refinement.LABEL
    assert metadata["forensic_audit_verdict"] == "CLIFF_CONFIRMED_MECHANICAL"
    assert metadata["validation_accessed"] is False
    assert metadata["holdout_accessed"] is False
    assert len(metadata["grid"]) == 12
    selection = json.loads(
        (output / "selection_summary.json").read_text(encoding="utf-8")
    )
    assert selection["selected_policy_id"] == "A2.10x_U20.75x"
    assert selection["precision_stage"]["path_count"] == 100_000
    assert (
        selection["precision_stage"]["predetermined_comparison_control_policy_id"]
        == "A2.0x_U21.25x"
    )
