"""One-shot VALIDATION diagnostic for the frozen A2.20x/U2=1.25x portfolio.

This module is confirmation-only.  It exposes no strategy, multiplier,
bootstrap, seed, path-count, horizon, or threshold override and contains no
fallback/rescue policy.  Final-holdout access remains prohibited by the
partition-typed source loaders.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from ftmoquant.prop_rules.loader import load_prop_rule_set
from ftmoquant.prop_rules.models import PropRuleSet
from ftmoquant.research import ftmo_joint_frontier as joint
from ftmoquant.research import ftmo_joint_frontier_precision as precision
from ftmoquant.research import ftmo_joint_frontier_refinement as refinement
from ftmoquant.research.alpha_lab.b3f1_spread_execution import (
    GROSS_NOTIONAL_USD,
    simulate_b3f1_intents,
)
from ftmoquant.research.alpha_lab.b3f1_spread_signals import (
    compute_formation_series,
    generate_b3f1_decisions,
)
from ftmoquant.research.alpha_lab.b3f1_underpowered_u2_validation import (
    DEFAULT_VALIDATION_READINESS_PATH,
    SLEEVE_ID,
    X_SPEC,
    Y_SPEC,
    Z_ENTRY,
    Z_STOP,
    load_u2_validation_data,
)
from ftmoquant.research.alpha_lab.relative_value_adapter import RelativeValueEpisode
from ftmoquant.research.ftmo_pass_probability.artifacts import (
    read_json_artifact,
    write_csv_artifact,
    write_json_artifact,
)
from ftmoquant.research.ftmo_pass_probability.path_extraction import (
    ValidationTradePath,
    load_validation_trade_path,
)
from ftmoquant.research.ftmo_pass_probability.reporting import BinomialEstimate
from ftmoquant.research.stage_g import HOLDOUT_START, VALIDATION_START

LABEL = "frozen joint-portfolio one-shot VALIDATION diagnostic"
FROZEN_POLICY = joint.JointPolicy(
    "A2.20x_U21.25x", Decimal("2.20"), Decimal("1.25")
)
FROZEN_POLICIES: tuple[joint.JointPolicy, ...] = (FROZEN_POLICY,)
PATH_COUNT = 100_000
SEED = 20260819
METHODS: tuple[Literal["stationary", "circular"], ...] = (
    "stationary",
    "circular",
)

STRATEGY_A_VALIDATION_EXECUTION_DIR = Path(
    ".artifacts/usdcad_sweep_bos_retest_v1/validation_execution"
)
DEVELOPMENT_PRECISION_DIR = precision.DEFAULT_OUTPUT_DIR
DEVELOPMENT_PRECISION_CSV = DEVELOPMENT_PRECISION_DIR / "precision_results.csv"
DEVELOPMENT_SELECTION = DEVELOPMENT_PRECISION_DIR / "selection_summary.json"
DEFAULT_OUTPUT_DIR = Path(
    ".artifacts/ftmo_joint_frontier/sweep_bos_plus_u2_validation_v1"
)

CONFIRMATION_THRESHOLDS: dict[str, float] = {
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

EXPECTED_ARTIFACT_FILENAMES: tuple[str, ...] = (
    "validation_joint_path_diagnostics.json",
    "validation_results.csv",
    "validation_summary.json",
    "development_comparison.json",
    "validation_classification.json",
    "metadata.json",
    "artifact_hashes.json",
)


@dataclass(frozen=True, slots=True)
class ValidationMethodResult:
    method: Literal["stationary", "circular"]
    run: joint.JointMethodRun


@dataclass(frozen=True, slots=True)
class ValidationClassification:
    classification: Literal["VALIDATION_CONFIRMED", "VALIDATION_NOT_CONFIRMED"]
    checks: dict[str, bool]
    failed_checks: tuple[str, ...]


def reserve_output_directory(output_dir: Path) -> None:
    if output_dir.exists():
        raise joint.JointFrontierError(
            f"{output_dir} already exists; refusing to overwrite"
        )
    for filename in EXPECTED_ARTIFACT_FILENAMES:
        path = output_dir / filename
        if path.exists():
            raise joint.JointFrontierError(f"{path} exists; refusing to overwrite")


def load_strategy_a_validation() -> ValidationTradePath:
    """Load only the frozen Strategy-A VALIDATION execution artifact."""

    return load_validation_trade_path(STRATEGY_A_VALIDATION_EXECUTION_DIR)


def load_u2_validation_episodes(
    *, validation_root: Path, universe_readiness_path: Path
) -> tuple[RelativeValueEpisode, ...]:
    """Rebuild exact U2 VALIDATION episodes with M1 two-leg mark paths."""

    data = load_u2_validation_data(
        validation_root=validation_root,
        universe_readiness_path=universe_readiness_path,
    )
    formation = compute_formation_series(
        data.log_y, data.log_x, joint.FROZEN_U2_FORMATION_WINDOW
    )
    decisions = generate_b3f1_decisions(
        formation,
        data.log_y,
        data.log_x,
        sleeve_id=SLEEVE_ID,
        z_entry=Z_ENTRY,
        z_stop=Z_STOP,
    )
    episodes, _skips = simulate_b3f1_intents(
        decisions,
        y_spec=Y_SPEC,
        x_spec=X_SPEC,
        y_bid_m1=data.y_bid_m1,
        y_ask_m1=data.y_ask_m1,
        x_bid_m1=data.x_bid_m1,
        x_ask_m1=data.x_ask_m1,
        gross_notional_usd=GROSS_NOTIONAL_USD,
        cost_stress_multiplier=Decimal("1"),
    )
    return episodes


def validate_partition_events(
    a_path: ValidationTradePath, u2_episodes: Sequence[RelativeValueEpisode]
) -> None:
    start_ns = int(VALIDATION_START.timestamp() * 1_000_000_000)
    end_ns = int(HOLDOUT_START.timestamp() * 1_000_000_000)
    spans = [(trade.entry_ns, trade.exit_ns) for trade in a_path.trades]
    spans.extend((episode.entry_ns, episode.exit_ns) for episode in u2_episodes)
    if not spans:
        raise joint.JointFrontierError("VALIDATION joint path is empty")
    if any(entry < start_ns or exit >= end_ns for entry, exit in spans):
        raise joint.JointFrontierError(
            "joint event falls outside [VALIDATION_START, HOLDOUT_START)"
        )


def load_development_reference(
    csv_path: Path = DEVELOPMENT_PRECISION_CSV,
    selection_path: Path = DEVELOPMENT_SELECTION,
) -> dict[str, Any]:
    """Read the exact frozen 100k DEVELOPMENT stationary row and verdict."""

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    matches = [
        row
        for row in rows
        if row["policy_id"] == FROZEN_POLICY.policy_id
        and row["method"] == "stationary"
        and int(row["paths"]) == PATH_COUNT
    ]
    if len(matches) != 1:
        raise joint.JointFrontierError(
            "expected exactly one frozen 100k DEVELOPMENT stationary row"
        )
    row = matches[0]
    selection = read_json_artifact(selection_path)
    verdicts = {
        verdict["policy_id"]: verdict for verdict in selection["verdicts"]
    }
    verdict = verdicts.get(FROZEN_POLICY.policy_id)
    if verdict is None or verdict["classification"] != "NOT_ELIGIBLE":
        raise joint.JointFrontierError(
            "frozen practical candidate must remain DEVELOPMENT NOT_ELIGIBLE"
        )
    fields = (
        "pass_challenge",
        "pass_both",
        "fail_daily_loss",
        "fail_max_loss",
        "censoring_rate",
        "median_trading_days_to_pass_both",
        "p90_trading_days_to_pass_both",
        "p95_trading_days_to_pass_both",
        "median_max_drawdown",
        "p95_max_drawdown",
    )
    return {
        "policy_id": FROZEN_POLICY.policy_id,
        "paths": PATH_COUNT,
        "method": "stationary",
        "development_classification": verdict["classification"],
        "failed_development_criteria": verdict["failed_point_estimate_criteria"],
        **{field: float(row[field]) for field in fields},
    }


def run_validation_methods(
    *,
    groups: tuple[joint.JointGroup, ...],
    rules: PropRuleSet,
    block_size: int,
    path_count: int = PATH_COUNT,
) -> tuple[ValidationMethodResult, ...]:
    """Run the single frozen policy under the two frozen methods."""

    prepared = joint.prepare_joint_groups_scaling(groups)
    scaled_events = joint.precompute_prepared_scaled_events(
        prepared,
        a_multiplier=FROZEN_POLICY.a_multiplier,
        u2_multiplier=FROZEN_POLICY.u2_multiplier,
    )
    timing = joint.precompute_joint_group_timing(groups)
    return tuple(
        ValidationMethodResult(
            method=method,
            run=joint.run_joint_policy_method_streaming(
                groups=groups,
                rules=rules,
                block_size=block_size,
                policy=FROZEN_POLICY,
                method=method,
                path_count=path_count,
                seed=SEED,
                challenge_horizon_ns=joint.DEFAULT_HORIZON_NS,
                verification_horizon_ns=joint.DEFAULT_HORIZON_NS,
                timing=timing,
                scaled_events=scaled_events,
            ),
        )
        for method in METHODS
    )


def classify_validation(
    results: Sequence[ValidationMethodResult],
) -> ValidationClassification:
    by_method = {result.method: result.run.summary for result in results}
    if set(by_method) != set(METHODS):
        raise joint.JointFrontierError("both frozen bootstrap methods are required")
    stationary = by_method["stationary"]
    circular = by_method["circular"]
    checks = {
        "A_stationary_pass_both_ge_0_65": stationary.pass_both.estimate >= 0.65,
        "B_stationary_pass_both_lower_ci_ge_0_60": (
            stationary.pass_both.ci_lower_95 >= 0.60
        ),
        "C_stationary_fail_daily_loss_le_0_02": (
            stationary.fail_daily_loss.estimate <= 0.02
        ),
        "D_stationary_fail_max_loss_le_0_30": (
            stationary.fail_max_loss.estimate <= 0.30
        ),
        "E_stationary_p90_days_le_180": (
            stationary.p90_trading_days_to_pass_both is not None
            and stationary.p90_trading_days_to_pass_both <= 180
        ),
        "F_stationary_p95_max_drawdown_le_0_18": (
            stationary.p95_max_drawdown <= 0.18
        ),
        "G_circular_pass_both_ge_0_60": circular.pass_both.estimate >= 0.60,
        "H_circular_fail_daily_loss_le_0_03": (
            circular.fail_daily_loss.estimate <= 0.03
        ),
        "I_circular_fail_max_loss_le_0_35": (
            circular.fail_max_loss.estimate <= 0.35
        ),
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    return ValidationClassification(
        classification=(
            "VALIDATION_CONFIRMED" if not failed else "VALIDATION_NOT_CONFIRMED"
        ),
        checks=checks,
        failed_checks=failed,
    )


def _probability_columns(name: str, estimate: BinomialEstimate) -> dict[str, Any]:
    return {
        name: estimate.estimate,
        f"{name}_successes": estimate.successes,
        f"{name}_trials": estimate.trials,
        f"{name}_ci_lower_95": estimate.ci_lower_95,
        f"{name}_ci_upper_95": estimate.ci_upper_95,
    }


def result_row(result: ValidationMethodResult) -> dict[str, Any]:
    summary = result.run.summary
    row: dict[str, Any] = {
        "policy_id": FROZEN_POLICY.policy_id,
        "a_multiplier": FROZEN_POLICY.a_multiplier,
        "u2_multiplier": FROZEN_POLICY.u2_multiplier,
        "method": result.method,
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
            "p75_trading_days_to_pass_both": result.run.p75_trading_days_to_pass_both,
            "p90_trading_days_to_pass_both": summary.p90_trading_days_to_pass_both,
            "p95_trading_days_to_pass_both": summary.p95_trading_days_to_pass_both,
            "median_max_drawdown": summary.median_max_drawdown,
            "p90_max_drawdown": result.run.p90_max_drawdown,
            "p95_max_drawdown": summary.p95_max_drawdown,
        }
    )
    return row


def development_comparison(
    stationary: ValidationMethodResult, development: Mapping[str, Any]
) -> dict[str, Any]:
    row = result_row(stationary)
    fields = (
        "pass_challenge",
        "pass_both",
        "fail_daily_loss",
        "fail_max_loss",
        "censoring_rate",
        "median_trading_days_to_pass_both",
        "p90_trading_days_to_pass_both",
        "p95_trading_days_to_pass_both",
        "median_max_drawdown",
        "p95_max_drawdown",
    )
    return {
        "policy_id": FROZEN_POLICY.policy_id,
        "comparison_is_descriptive_only": True,
        "development_classification_remains": "NOT_ELIGIBLE",
        "development": dict(development),
        "validation_stationary": {field: row[field] for field in fields},
        "absolute_deltas_validation_minus_development": {
            field: float(row[field]) - float(development[field]) for field in fields
        },
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_results(
    *,
    output_dir: Path,
    results: tuple[ValidationMethodResult, ...],
    development: Mapping[str, Any],
    block_length: joint.FrozenJointBlockLength,
    diagnostics: joint.DiversificationDiagnostics,
    a_path: ValidationTradePath,
    u2_episode_count: int,
    validation_readiness_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json_artifact(
        output_dir / "validation_joint_path_diagnostics.json",
        {
            "strategy_a_trade_count": len(a_path.trades),
            "u2_trade_count": u2_episode_count,
            "joint_group_count": diagnostics.total_group_count,
            "overlap_group_count": diagnostics.overlap_group_count,
            "fraction_a_trades_overlapping_u2": (
                diagnostics.fraction_a_trades_overlapping_u2
            ),
            "fraction_u2_trades_overlapping_a": (
                diagnostics.fraction_u2_trades_overlapping_a
            ),
            "aligned_daily_return_correlation": diagnostics.aligned_daily_correlation,
            "downside_correlation": diagnostics.downside_correlation,
            "maximum_observed_same_day_combined_loss_usd": (
                diagnostics.maximum_same_day_combined_loss_usd
            ),
            "block_length": block_length,
        },
    )
    rows = [result_row(result) for result in results]
    write_csv_artifact(output_dir / "validation_results.csv", list(rows[0]), rows)
    write_json_artifact(
        output_dir / "validation_summary.json",
        {
            "policy": FROZEN_POLICY,
            "paths_per_method": PATH_COUNT,
            "seed": SEED,
            "methods": METHODS,
            "results": rows,
        },
    )
    stationary = next(result for result in results if result.method == "stationary")
    write_json_artifact(
        output_dir / "development_comparison.json",
        development_comparison(stationary, development),
    )
    classification = classify_validation(results)
    write_json_artifact(
        output_dir / "validation_classification.json",
        {
            **asdict(classification),
            "confirmation_thresholds": CONFIRMATION_THRESHOLDS,
            "no_rescue": True,
            "on_not_confirmed": "report failure and stop; evaluate no alternative",
        },
    )
    write_json_artifact(
        output_dir / "metadata.json",
        {
            "label": LABEL,
            "portfolio_was_selected_after_development_analysis": True,
            "development_classification": "NOT_ELIGIBLE",
            "practical_candidate_rationale": (
                "preferred risk/speed compromise among investigated "
                "DEVELOPMENT policies"
            ),
            "frozen_policy": FROZEN_POLICY,
            "policy_count": 1,
            "validation_partition": {
                "start_utc": VALIDATION_START.isoformat().replace("+00:00", "Z"),
                "end_exclusive_utc": HOLDOUT_START.isoformat().replace(
                    "+00:00", "Z"
                ),
            },
            "monte_carlo": {
                "methods": METHODS,
                "paths_per_method": PATH_COUNT,
                "seed": SEED,
                "seed_derivation": "seed + replication",
                "block_length": block_length,
                "resampling_unit": "JointGroup",
            },
            "strategy_a_validation_artifact": STRATEGY_A_VALIDATION_EXECUTION_DIR,
            "strategy_a_trades_sha256": a_path.trades_csv_sha256,
            "u2_validation_readiness": validation_readiness_path,
            "u2_validation_readiness_sha256": _file_sha256(validation_readiness_path),
            "source_development_precision": DEVELOPMENT_PRECISION_DIR,
            "ftmo_rules": joint.FTMO_RULES_PATH,
            "ftmo_rules_sha256": _file_sha256(joint.FTMO_RULES_PATH),
            "validation_accessed": True,
            "final_holdout_accessed": False,
            "no_alternative_policy_path": True,
            "git_commit": refinement._git_commit(),  # noqa: SLF001
            "dependency_versions": refinement._dependency_versions(),  # noqa: SLF001
        },
    )
    hashed_files = EXPECTED_ARTIFACT_FILENAMES[:-1]
    write_json_artifact(
        output_dir / "artifact_hashes.json",
        {
            "algorithm": "sha256",
            "files": {
                filename: _file_sha256(output_dir / filename)
                for filename in hashed_files
            },
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "One-shot frozen A2.20x/U2=1.25x VALIDATION diagnostic: "
            "100000 stationary + 100000 circular paths at seed 20260819."
        )
    )
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument(
        "--universe-readiness",
        type=Path,
        default=DEFAULT_VALIDATION_READINESS_PATH,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    reserve_output_directory(args.output)
    development = load_development_reference()
    a_path = load_strategy_a_validation()
    u2_episodes = load_u2_validation_episodes(
        validation_root=args.validation_root,
        universe_readiness_path=args.universe_readiness,
    )
    validate_partition_events(a_path, u2_episodes)
    groups = joint.build_joint_groups(a_path.trades, u2_episodes)
    block_length = joint.derive_frozen_joint_block_length(
        groups, strategy_a_standalone_block_size=1
    )
    diagnostics = joint.compute_diversification_diagnostics(
        groups, a_path.trades, u2_episodes
    )
    rules = load_prop_rule_set(joint.FTMO_RULES_PATH)
    results = run_validation_methods(
        groups=groups,
        rules=rules,
        block_size=block_length.frozen_block_size,
    )
    write_results(
        output_dir=args.output,
        results=results,
        development=development,
        block_length=block_length,
        diagnostics=diagnostics,
        a_path=a_path,
        u2_episode_count=len(u2_episodes),
        validation_readiness_path=args.universe_readiness,
    )
    print("VALIDATION complete: one frozen policy, 100000 stationary + 100000 circular")


if __name__ == "__main__":
    main()
