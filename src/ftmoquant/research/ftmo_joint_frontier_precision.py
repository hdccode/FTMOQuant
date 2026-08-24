"""Frozen 100k precision comparison for three DEVELOPMENT joint policies.

This module performs no search or sizing refinement.  It routes exactly three
already-selected policies through the canonical joint replay using
StationaryBootstrap only, streams its sufficient statistics, then applies the
original eligibility and selection rules without relaxing them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from ftmoquant.prop_rules.loader import load_prop_rule_set
from ftmoquant.prop_rules.models import PropRuleSet
from ftmoquant.research import ftmo_joint_frontier as joint
from ftmoquant.research import ftmo_joint_frontier_refinement as refinement
from ftmoquant.research.ftmo_pass_probability.artifacts import (
    read_json_artifact,
    write_csv_artifact,
    write_json_artifact,
)
from ftmoquant.research.ftmo_pass_probability.reporting import BinomialEstimate

LABEL = "frozen three-policy DEVELOPMENT precision comparison"
DEVELOPMENT_ONLY_STATEMENT = (
    "Precision-only Monte Carlo convergence comparison over frozen DEVELOPMENT "
    "paths. No VALIDATION data or artifacts and no final holdout data or "
    "artifacts are read."
)

PRECISION_PATH_COUNT = 100_000
PRECISION_METHOD: Literal["stationary"] = "stationary"
PRECISION_SEED = 20260819
BENCHMARK_PATHS_PER_POLICY = 50

DEFAULT_OUTPUT_DIR = Path(
    ".artifacts/ftmo_joint_frontier/sweep_bos_plus_u2_precision_v1"
)
SOURCE_REFINEMENT_DIR = refinement.DEFAULT_OUTPUT_DIR
SOURCE_REFINEMENT_CSV = SOURCE_REFINEMENT_DIR / "refinement_screen.csv"
SOURCE_REFINEMENT_METADATA = SOURCE_REFINEMENT_DIR / "metadata.json"

FROZEN_POLICIES: tuple[joint.JointPolicy, ...] = (
    joint.JointPolicy("A2.20x_U21.25x", Decimal("2.20"), Decimal("1.25")),
    joint.JointPolicy("A2.30x_U21.00x", Decimal("2.30"), Decimal("1.00")),
    joint.JointPolicy("A2.30x_U21.25x", Decimal("2.30"), Decimal("1.25")),
)
FROZEN_POLICY_IDS: tuple[str, ...] = tuple(
    policy.policy_id for policy in FROZEN_POLICIES
)
if FROZEN_POLICY_IDS != (
    "A2.20x_U21.25x",
    "A2.30x_U21.00x",
    "A2.30x_U21.25x",
):
    raise joint.JointFrontierError("precision policy tuple is not the frozen tuple")

ELIGIBILITY_RULE = refinement.ELIGIBILITY_RULE
SELECTION_RULE = refinement.SELECTION_RULE
ROBUST_TO_MC_UNCERTAINTY_RULE: tuple[str, ...] = (
    "pass_both lower 95% Wilson CI >= 0.70",
    "fail_daily_loss upper 95% Wilson CI <= 0.02",
    "fail_max_loss upper 95% Wilson CI <= 0.25",
    "point-estimate median trading days to pass both <= 75",
    "point-estimate p90 trading days to pass both <= 150",
)

EXPECTED_ARTIFACT_FILENAMES: tuple[str, ...] = (
    "precision_results.csv",
    "precision_summary.json",
    "convergence_vs_20k.csv",
    "selection_summary.json",
    "metadata.json",
)


@dataclass(frozen=True, slots=True)
class HistoricalMetrics:
    policy_id: str
    pass_both: float
    fail_daily_loss: float
    fail_max_loss: float
    median_days: float | None
    p90_days: float | None
    p95_max_drawdown: float


@dataclass(frozen=True, slots=True)
class PrecisionResult:
    policy: joint.JointPolicy
    run: joint.JointMethodRun


@dataclass(frozen=True, slots=True)
class PrecisionVerdict:
    policy_id: str
    point_estimate_eligible: bool
    robust_to_mc_uncertainty: bool
    classification: Literal[
        "DEPLOYMENT_ELIGIBLE", "PRECISION_BORDERLINE", "NOT_ELIGIBLE"
    ]
    failed_point_estimate_criteria: tuple[str, ...]
    failed_robust_criteria: tuple[str, ...]


def reserve_output_directory(output_dir: Path) -> None:
    """Refuse overwrite before data loading or any Monte Carlo work."""

    if output_dir.exists():
        raise joint.JointFrontierError(
            f"{output_dir} already exists; refusing to overwrite"
        )
    for filename in EXPECTED_ARTIFACT_FILENAMES:
        path = output_dir / filename
        if path.exists():
            raise joint.JointFrontierError(f"{path} already exists; refusing overwrite")


def _optional_float(value: str) -> float | None:
    return None if value == "" else float(value)


def load_historical_metrics(
    csv_path: Path = SOURCE_REFINEMENT_CSV,
) -> dict[str, HistoricalMetrics]:
    """Read exactly the three stationary 20k rows from the refinement artifact."""

    if not csv_path.is_file():
        raise joint.JointFrontierError(f"missing refinement artifact: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    metrics: dict[str, HistoricalMetrics] = {}
    for row in rows:
        policy_id = row["policy_id"]
        if policy_id not in FROZEN_POLICY_IDS or row["method"] != PRECISION_METHOD:
            continue
        if policy_id in metrics:
            raise joint.JointFrontierError(
                f"duplicate stationary refinement row for {policy_id}"
            )
        metrics[policy_id] = HistoricalMetrics(
            policy_id=policy_id,
            pass_both=float(row["pass_both"]),
            fail_daily_loss=float(row["fail_daily_loss"]),
            fail_max_loss=float(row["fail_max_loss"]),
            median_days=_optional_float(row["median_trading_days_to_pass_both"]),
            p90_days=_optional_float(row["p90_trading_days_to_pass_both"]),
            p95_max_drawdown=float(row["p95_max_drawdown"]),
        )
    if tuple(policy_id for policy_id in FROZEN_POLICY_IDS if policy_id in metrics) != (
        FROZEN_POLICY_IDS
    ):
        missing = sorted(set(FROZEN_POLICY_IDS) - set(metrics))
        raise joint.JointFrontierError(
            f"refinement artifact is missing frozen stationary rows: {missing}"
        )
    return metrics


def verify_refinement_identity(
    *,
    groups: tuple[joint.JointGroup, ...],
    a_trade_count: int,
    u2_episode_count: int,
    block_length: joint.FrozenJointBlockLength,
    metadata_path: Path = SOURCE_REFINEMENT_METADATA,
) -> dict[str, Any]:
    """Prove path construction and frozen block length match the 20k screen."""

    parity = refinement.verify_coarse_structural_parity(
        groups=groups,
        a_trade_count=a_trade_count,
        u2_episode_count=u2_episode_count,
        block_length=block_length,
    )
    metadata = read_json_artifact(metadata_path)
    historical_block = metadata["monte_carlo"]["block_length"]
    if asdict(block_length) != historical_block:
        raise joint.JointFrontierError(
            "precision block length differs from source refinement artifact"
        )
    return {
        **parity,
        "source_refinement_metadata_content_sha256": metadata["content_sha256"],
        "same_refinement_block_length": True,
        "same_horizons": True,
        "same_resampling_unit": "JointGroup",
    }


def _run_batch(
    *,
    groups: tuple[joint.JointGroup, ...],
    rules: PropRuleSet,
    block_size: int,
    path_count: int,
) -> tuple[PrecisionResult, ...]:
    prepared_groups = joint.prepare_joint_groups_scaling(groups)
    timing = joint.precompute_joint_group_timing(groups)
    return tuple(
        PrecisionResult(
            policy=policy,
            run=joint.run_joint_policy_method_streaming(
                groups=groups,
                rules=rules,
                block_size=block_size,
                policy=policy,
                method=PRECISION_METHOD,
                path_count=path_count,
                seed=PRECISION_SEED,
                challenge_horizon_ns=joint.DEFAULT_HORIZON_NS,
                verification_horizon_ns=joint.DEFAULT_HORIZON_NS,
                timing=timing,
                scaled_events=joint.precompute_prepared_scaled_events(
                    prepared_groups,
                    a_multiplier=policy.a_multiplier,
                    u2_multiplier=policy.u2_multiplier,
                ),
            ),
        )
        for policy in FROZEN_POLICIES
    )


def run_precision_batch(
    *,
    groups: tuple[joint.JointGroup, ...],
    rules: PropRuleSet,
    block_size: int,
) -> tuple[PrecisionResult, ...]:
    """Run the non-overridable three-policy, stationary, 100k batch."""

    return _run_batch(
        groups=groups,
        rules=rules,
        block_size=block_size,
        path_count=PRECISION_PATH_COUNT,
    )


def benchmark_precision_runtime(
    *, groups: tuple[joint.JointGroup, ...], rules: PropRuleSet, block_size: int
) -> dict[str, float | int]:
    """Benchmark exactly 150 paths (50 for each frozen policy)."""

    started = time.perf_counter()
    _run_batch(
        groups=groups,
        rules=rules,
        block_size=block_size,
        path_count=BENCHMARK_PATHS_PER_POLICY,
    )
    elapsed = time.perf_counter() - started
    benchmark_paths = len(FROZEN_POLICIES) * BENCHMARK_PATHS_PER_POLICY
    full_paths = len(FROZEN_POLICIES) * PRECISION_PATH_COUNT
    return {
        "benchmark_paths": benchmark_paths,
        "benchmark_paths_per_policy": BENCHMARK_PATHS_PER_POLICY,
        "elapsed_seconds": elapsed,
        "seconds_per_path": elapsed / benchmark_paths,
        "estimated_full_seconds": elapsed * full_paths / benchmark_paths,
        "estimated_full_minutes": elapsed * full_paths / benchmark_paths / 60,
    }


def _point_checks(result: PrecisionResult) -> dict[str, bool]:
    summary = result.run.summary
    return {
        "pass_both_ge_0_70": summary.pass_both.estimate >= 0.70,
        "median_days_le_75": (
            summary.median_trading_days_to_pass_both is not None
            and summary.median_trading_days_to_pass_both <= 75
        ),
        "p90_days_le_150": (
            summary.p90_trading_days_to_pass_both is not None
            and summary.p90_trading_days_to_pass_both <= 150
        ),
        "fail_daily_loss_le_0_02": summary.fail_daily_loss.estimate <= 0.02,
        "fail_max_loss_le_0_25": summary.fail_max_loss.estimate <= 0.25,
    }


def _robust_checks(result: PrecisionResult) -> dict[str, bool]:
    summary = result.run.summary
    return {
        "pass_both_lower_ci_ge_0_70": summary.pass_both.ci_lower_95 >= 0.70,
        "fail_daily_loss_upper_ci_le_0_02": (
            summary.fail_daily_loss.ci_upper_95 <= 0.02
        ),
        "fail_max_loss_upper_ci_le_0_25": (
            summary.fail_max_loss.ci_upper_95 <= 0.25
        ),
        "median_days_le_75": (
            summary.median_trading_days_to_pass_both is not None
            and summary.median_trading_days_to_pass_both <= 75
        ),
        "p90_days_le_150": (
            summary.p90_trading_days_to_pass_both is not None
            and summary.p90_trading_days_to_pass_both <= 150
        ),
    }


def evaluate_precision(result: PrecisionResult) -> PrecisionVerdict:
    point_checks = _point_checks(result)
    robust_checks = _robust_checks(result)
    failed_point = tuple(name for name, passed in point_checks.items() if not passed)
    failed_robust = tuple(name for name, passed in robust_checks.items() if not passed)
    classification: Literal[
        "DEPLOYMENT_ELIGIBLE", "PRECISION_BORDERLINE", "NOT_ELIGIBLE"
    ]
    if not failed_point:
        classification = "DEPLOYMENT_ELIGIBLE"
    elif len(failed_point) == 1:
        # No distance cutoff is invented for "narrowly".  This label only
        # identifies a single-threshold miss and never rescues the policy.
        classification = "PRECISION_BORDERLINE"
    else:
        classification = "NOT_ELIGIBLE"
    return PrecisionVerdict(
        policy_id=result.policy.policy_id,
        point_estimate_eligible=not failed_point,
        robust_to_mc_uncertainty=not failed_robust,
        classification=classification,
        failed_point_estimate_criteria=failed_point,
        failed_robust_criteria=failed_robust,
    )


def select_policy(results: Sequence[PrecisionResult]) -> PrecisionResult | None:
    """Apply the original eligibility gate and frozen tie-break ladder."""

    eligible = [result for result in results if not _point_checks_failed(result)]
    if not eligible:
        return None

    def key(result: PrecisionResult) -> tuple[float, float, float, float, Decimal, str]:
        summary = result.run.summary
        median_days = summary.median_trading_days_to_pass_both
        return (
            median_days if median_days is not None else float("inf"),
            -summary.pass_both.estimate,
            summary.fail_max_loss.estimate,
            summary.p95_max_drawdown,
            result.policy.total_gross_multiplier,
            result.policy.policy_id,
        )

    return min(eligible, key=key)


def _point_checks_failed(result: PrecisionResult) -> bool:
    return not all(_point_checks(result).values())


def _probability_columns(name: str, estimate: BinomialEstimate) -> dict[str, Any]:
    return {
        name: estimate.estimate,
        f"{name}_successes": estimate.successes,
        f"{name}_trials": estimate.trials,
        f"{name}_ci_lower_95": estimate.ci_lower_95,
        f"{name}_ci_upper_95": estimate.ci_upper_95,
    }


def _result_row(result: PrecisionResult) -> dict[str, Any]:
    summary = result.run.summary
    verdict = evaluate_precision(result)
    row: dict[str, Any] = {
        "policy_id": result.policy.policy_id,
        "a_multiplier": result.policy.a_multiplier,
        "u2_multiplier": result.policy.u2_multiplier,
        "method": summary.method,
        "paths": summary.replications,
    }
    for name in (
        "pass_challenge",
        "pass_both",
        "fail_daily_loss",
        "fail_max_loss",
        "censoring_rate",
    ):
        row.update(_probability_columns(name, getattr(summary, name)))
    row.update(
        {
            "median_trading_days_to_pass_challenge": (
                result.run.median_trading_days_to_pass_challenge
            ),
            "median_trading_days_to_pass_both": (
                summary.median_trading_days_to_pass_both
            ),
            "p75_trading_days_to_pass_both": (
                result.run.p75_trading_days_to_pass_both
            ),
            "p90_trading_days_to_pass_both": (
                summary.p90_trading_days_to_pass_both
            ),
            "p95_trading_days_to_pass_both": (
                summary.p95_trading_days_to_pass_both
            ),
            "median_max_drawdown": summary.median_max_drawdown,
            "p90_max_drawdown": result.run.p90_max_drawdown,
            "p95_max_drawdown": summary.p95_max_drawdown,
            "point_estimate_eligible": verdict.point_estimate_eligible,
            "robust_to_mc_uncertainty": verdict.robust_to_mc_uncertainty,
            "classification": verdict.classification,
        }
    )
    return row


def _delta(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None:
        return None
    return value - reference


def _convergence_rows(
    results: Sequence[PrecisionResult],
    historical: Mapping[str, HistoricalMetrics],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        summary = result.run.summary
        reference = historical[result.policy.policy_id]
        rows.append(
            {
                "policy_id": result.policy.policy_id,
                "delta_pass_both": summary.pass_both.estimate - reference.pass_both,
                "delta_fail_daily_loss": (
                    summary.fail_daily_loss.estimate - reference.fail_daily_loss
                ),
                "delta_fail_max_loss": (
                    summary.fail_max_loss.estimate - reference.fail_max_loss
                ),
                "delta_median_days": _delta(
                    summary.median_trading_days_to_pass_both, reference.median_days
                ),
                "delta_p90_days": _delta(
                    summary.p90_trading_days_to_pass_both, reference.p90_days
                ),
                "delta_p95_max_drawdown": (
                    summary.p95_max_drawdown - reference.p95_max_drawdown
                ),
            }
        )
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_results(
    *,
    output_dir: Path,
    results: tuple[PrecisionResult, ...],
    historical: Mapping[str, HistoricalMetrics],
    parity: Mapping[str, Any],
    block_length: joint.FrozenJointBlockLength,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = [_result_row(result) for result in results]
    write_csv_artifact(output_dir / "precision_results.csv", list(rows[0]), rows)
    write_json_artifact(
        output_dir / "precision_summary.json",
        {
            "policy_count": len(results),
            "method": PRECISION_METHOD,
            "paths_per_policy": PRECISION_PATH_COUNT,
            "seed": PRECISION_SEED,
            "results": rows,
        },
    )
    convergence = _convergence_rows(results, historical)
    write_csv_artifact(
        output_dir / "convergence_vs_20k.csv",
        list(convergence[0]),
        convergence,
    )

    verdicts = [evaluate_precision(result) for result in results]
    selected = select_policy(results)
    write_json_artifact(
        output_dir / "selection_summary.json",
        {
            "selected_policy_id": (
                selected.policy.policy_id if selected is not None else None
            ),
            "verdicts": verdicts,
            "original_eligibility_rule": ELIGIBILITY_RULE,
            "robust_to_mc_uncertainty_rule": ROBUST_TO_MC_UNCERTAINTY_RULE,
            "selection_rule": SELECTION_RULE,
            "precision_borderline_definition": (
                "exactly one original point-estimate threshold missed; no "
                "distance threshold, rescue, or eligibility relaxation"
            ),
        },
    )
    write_json_artifact(
        output_dir / "metadata.json",
        {
            "label": LABEL,
            "frozen_policy_ids": FROZEN_POLICY_IDS,
            "policy_count": len(FROZEN_POLICIES),
            "paths_per_policy": PRECISION_PATH_COUNT,
            "method": PRECISION_METHOD,
            "bootstrap_identity": "StationaryBootstrap",
            "seed": PRECISION_SEED,
            "seed_derivation": "seed + replication",
            "challenge_horizon_ns": joint.DEFAULT_HORIZON_NS,
            "verification_horizon_ns": joint.DEFAULT_HORIZON_NS,
            "development_only": True,
            "development_only_statement": DEVELOPMENT_ONLY_STATEMENT,
            "source_refinement_artifact": SOURCE_REFINEMENT_DIR,
            "source_refinement_csv_sha256": _sha256(SOURCE_REFINEMENT_CSV),
            "block_length_identity": block_length,
            "joint_path_identity": parity,
            "ftmo_rules_identity": {
                "path": joint.FTMO_RULES_PATH,
                "content_sha256": _sha256(joint.FTMO_RULES_PATH),
            },
            "validation_accessed": False,
            "holdout_accessed": False,
            "git_commit": refinement._git_commit(),  # noqa: SLF001
            "dependency_versions": refinement._dependency_versions(),  # noqa: SLF001
            "no_automatic_further_refinement": True,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen three-policy DEVELOPMENT precision batch: exactly 100000 "
            "stationary-bootstrap paths per policy at seed 20260819."
        )
    )
    parser.add_argument("--catalog-root", type=Path, required=True)
    parser.add_argument("--universe-readiness", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    reserve_output_directory(args.output)

    historical = load_historical_metrics()
    a_path = joint.load_strategy_a_development()
    u2_episodes = joint.load_u2_development_episodes(
        catalog_root=args.catalog_root,
        universe_readiness_path=args.universe_readiness,
    )
    groups = joint.build_joint_groups(a_path.trades, u2_episodes)
    rules = load_prop_rule_set(joint.FTMO_RULES_PATH)
    block_length = joint.derive_frozen_joint_block_length(
        groups, strategy_a_standalone_block_size=1
    )
    parity = verify_refinement_identity(
        groups=groups,
        a_trade_count=len(a_path.trades),
        u2_episode_count=len(u2_episodes),
        block_length=block_length,
    )
    results = run_precision_batch(
        groups=groups,
        rules=rules,
        block_size=block_length.frozen_block_size,
    )
    write_results(
        output_dir=args.output,
        results=results,
        historical=historical,
        parity=parity,
        block_length=block_length,
    )
    print("precision complete: 3 frozen policies x 100000 stationary paths")


if __name__ == "__main__":
    main()
