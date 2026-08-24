"""Frozen post-screen DEVELOPMENT refinement for the joint FTMO frontier.

This module is intentionally thin.  Path construction, scaling, bootstrap
draws, FTMO replay, reporting, eligibility, selection, and Pareto dominance
all remain in :mod:`ftmoquant.research.ftmo_joint_frontier` and are called
directly here.  Only the frozen 12-policy grid and refinement-specific
artifact presentation live in this module.
"""

from __future__ import annotations

import argparse
import csv
import platform
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ftmoquant.prop_rules.loader import load_prop_rule_set
from ftmoquant.prop_rules.models import PropRuleSet
from ftmoquant.research import ftmo_joint_frontier as joint
from ftmoquant.research.ftmo_pass_probability.artifacts import (
    read_json_artifact,
    write_csv_artifact,
    write_json_artifact,
)

LABEL = "post-screen DEVELOPMENT local refinement"
FORENSIC_AUDIT_VERDICT = "CLIFF_CONFIRMED_MECHANICAL"
DEVELOPMENT_ONLY_STATEMENT = (
    "Post-screen DEVELOPMENT local refinement; not independent evidence. "
    "No VALIDATION data or artifacts and no final holdout data or artifacts are read."
)

DEFAULT_OUTPUT_DIR = Path(
    ".artifacts/ftmo_joint_frontier/sweep_bos_plus_u2_refinement_v1"
)
COARSE_SCREEN_DIR = Path(
    ".artifacts/ftmo_joint_frontier/sweep_bos_plus_u2_v1"
)
COARSE_SCREEN_CSV = COARSE_SCREEN_DIR / "joint_sizing_screen.csv"
COARSE_PATH_DIAGNOSTICS = COARSE_SCREEN_DIR / "joint_path_diagnostics.json"

REFINEMENT_A_MULTIPLIERS: tuple[Decimal, ...] = (
    Decimal("2.10"),
    Decimal("2.20"),
    Decimal("2.30"),
    Decimal("2.40"),
)
REFINEMENT_U2_MULTIPLIERS: tuple[Decimal, ...] = (
    Decimal("0.75"),
    Decimal("1.00"),
    Decimal("1.25"),
)
REFINEMENT_POLICY_GRID: tuple[joint.JointPolicy, ...] = tuple(
    joint.JointPolicy(joint._policy_id(a_mult, u2_mult), a_mult, u2_mult)
    for a_mult in REFINEMENT_A_MULTIPLIERS
    for u2_mult in REFINEMENT_U2_MULTIPLIERS
)
if len(REFINEMENT_POLICY_GRID) != 12:
    raise joint.JointFrontierError(
        f"expected exactly 12 refinement policies, got {len(REFINEMENT_POLICY_GRID)}"
    )
if len({policy.policy_id for policy in REFINEMENT_POLICY_GRID}) != 12:
    raise joint.JointFrontierError("refinement policy_ids must be unique")

COARSE_REFERENCE_POLICY_IDS: tuple[str, str] = (
    "A2.0x_U21.25x",
    "A2.5x_U20.75x",
)
PRECISION_CONTROL_POLICY_ID = "A2.0x_U21.25x"

ELIGIBILITY_RULE: dict[str, float | int] = {
    "stationary_pass_both_ge": joint.ELIGIBILITY_PASS_BOTH_GE,
    "stationary_median_trading_days_to_pass_both_le": (
        joint.ELIGIBILITY_MEDIAN_DAYS_LE
    ),
    "stationary_p90_trading_days_to_pass_both_le": joint.ELIGIBILITY_P90_DAYS_LE,
    "stationary_fail_daily_loss_le": joint.ELIGIBILITY_FAIL_DAILY_LOSS_LE,
    "stationary_fail_max_loss_le": joint.ELIGIBILITY_FAIL_MAX_LOSS_LE,
}
SELECTION_RULE: tuple[str, ...] = (
    "lowest median trading days to pass both",
    "highest pass_both",
    "lowest fail_max_loss",
    "lowest p95 max drawdown",
    "lower total gross multiplier",
    "lexicographic policy_id",
)
PARETO_RULE: dict[str, tuple[str, ...]] = {
    "maximize": ("pass_both",),
    "minimize": (
        "median_trading_days_to_pass_both",
        "p90_trading_days_to_pass_both",
        "fail_max_loss",
        "p95_max_drawdown",
    ),
}

EXPECTED_ARTIFACT_FILENAMES: tuple[str, ...] = (
    "refinement_grid.json",
    "refinement_screen.csv",
    "pareto_frontier.csv",
    "selection_summary.json",
    "refinement_diagnostics.json",
    "metadata.json",
)


@dataclass(frozen=True, slots=True)
class CoarseReferenceMetrics:
    policy_id: str
    method: str
    pass_both: float
    median_days: float | None
    p90_days: float | None
    fail_daily_loss: float
    fail_max_loss: float


def reserve_output_directory(output_dir: Path) -> None:
    """Refuse an existing output before loading data or drawing paths."""

    if output_dir.exists():
        raise joint.JointFrontierError(
            f"{output_dir} already exists; refusing to overwrite"
        )
    for filename in EXPECTED_ARTIFACT_FILENAMES:
        if (output_dir / filename).exists():
            raise joint.JointFrontierError(
                f"{output_dir / filename} already exists; refusing to overwrite"
            )


def _optional_float(value: str) -> float | None:
    return None if value == "" else float(value)


def load_coarse_references(
    csv_path: Path = COARSE_SCREEN_CSV,
) -> dict[tuple[str, str], CoarseReferenceMetrics]:
    """Read the two predetermined coarse policies; never hardcode metrics."""

    if not csv_path.is_file():
        raise joint.JointFrontierError(f"missing coarse screen artifact: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    references: dict[tuple[str, str], CoarseReferenceMetrics] = {}
    for row in rows:
        policy_id = row["policy_id"]
        method = row["method"]
        if policy_id not in COARSE_REFERENCE_POLICY_IDS:
            continue
        key = (policy_id, method)
        if key in references:
            raise joint.JointFrontierError(f"duplicate coarse reference row: {key}")
        references[key] = CoarseReferenceMetrics(
            policy_id=policy_id,
            method=method,
            pass_both=float(row["pass_both"]),
            median_days=_optional_float(row["median_trading_days_to_pass_both"]),
            p90_days=_optional_float(row["p90_trading_days_to_pass_both"]),
            fail_daily_loss=float(row["fail_daily_loss"]),
            fail_max_loss=float(row["fail_max_loss"]),
        )

    expected = {
        (policy_id, method)
        for policy_id in COARSE_REFERENCE_POLICY_IDS
        for method in ("stationary", "circular")
    }
    if set(references) != expected:
        missing = sorted(expected - set(references))
        raise joint.JointFrontierError(
            f"coarse screen is missing predetermined reference rows: {missing}"
        )
    return references


def _delta(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None:
        return None
    return value - reference


def _screen_rows(
    summaries: Sequence[joint.JointPolicySummary],
    references: Mapping[tuple[str, str], CoarseReferenceMetrics],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        for method, policy_summary in (
            ("stationary", summary.stationary),
            ("circular", summary.circular),
        ):
            row: dict[str, Any] = {
                "policy_id": summary.policy.policy_id,
                "a_multiplier": summary.policy.a_multiplier,
                "u2_multiplier": summary.policy.u2_multiplier,
                "method": method,
                "pass_challenge": policy_summary.pass_challenge.estimate,
                "pass_both": policy_summary.pass_both.estimate,
                "fail_daily_loss": policy_summary.fail_daily_loss.estimate,
                "fail_max_loss": policy_summary.fail_max_loss.estimate,
                "censoring_rate": policy_summary.censoring_rate.estimate,
                "median_trading_days_to_pass_both": (
                    policy_summary.median_trading_days_to_pass_both
                ),
                "p90_trading_days_to_pass_both": (
                    policy_summary.p90_trading_days_to_pass_both
                ),
                "p95_trading_days_to_pass_both": (
                    policy_summary.p95_trading_days_to_pass_both
                ),
                "median_max_drawdown": policy_summary.median_max_drawdown,
                "p95_max_drawdown": policy_summary.p95_max_drawdown,
            }
            for reference_id in COARSE_REFERENCE_POLICY_IDS:
                reference = references[(reference_id, method)]
                prefix = f"delta_vs_{reference_id}_"
                row[f"{prefix}pass_both"] = (
                    policy_summary.pass_both.estimate - reference.pass_both
                )
                row[f"{prefix}median_days"] = _delta(
                    policy_summary.median_trading_days_to_pass_both,
                    reference.median_days,
                )
                row[f"{prefix}p90_days"] = _delta(
                    policy_summary.p90_trading_days_to_pass_both,
                    reference.p90_days,
                )
                row[f"{prefix}fail_daily_loss"] = (
                    policy_summary.fail_daily_loss.estimate
                    - reference.fail_daily_loss
                )
                row[f"{prefix}fail_max_loss"] = (
                    policy_summary.fail_max_loss.estimate - reference.fail_max_loss
                )
            rows.append(row)
    return rows


def _stationary_diagnostic(summary: joint.JointPolicySummary) -> dict[str, Any]:
    stationary = summary.stationary
    daily = stationary.fail_daily_loss.estimate
    maximum = stationary.fail_max_loss.estimate
    return {
        "policy_id": summary.policy.policy_id,
        "fail_daily_loss": daily,
        "fail_max_loss": maximum,
        "combined_failure_rate": daily + maximum,
        "p95_max_drawdown": stationary.p95_max_drawdown,
        "distance_from_daily_failure_eligibility_ceiling": (
            joint.ELIGIBILITY_FAIL_DAILY_LOSS_LE - daily
        ),
        "near_daily_loss_cliff": 0 < daily <= 0.05,
    }


def run_refinement_screen(
    *,
    groups: tuple[joint.JointGroup, ...],
    rules: PropRuleSet,
    block_size: int,
    path_count: int = joint.SCREEN_PATH_COUNT,
    seed: int = joint.SCREEN_SEED,
    challenge_horizon_ns: int = joint.DEFAULT_HORIZON_NS,
    verification_horizon_ns: int = joint.DEFAULT_HORIZON_NS,
) -> tuple[joint.JointPolicySummary, ...]:
    """Route the exact refinement tuple through the shared coarse engine."""

    return _run_shared_screen(
        policies=REFINEMENT_POLICY_GRID,
        groups=groups,
        rules=rules,
        block_size=block_size,
        path_count=path_count,
        seed=seed,
        challenge_horizon_ns=challenge_horizon_ns,
        verification_horizon_ns=verification_horizon_ns,
    )


def _run_shared_screen(
    *,
    policies: Sequence[joint.JointPolicy],
    groups: tuple[joint.JointGroup, ...],
    rules: PropRuleSet,
    block_size: int,
    path_count: int,
    seed: int,
    challenge_horizon_ns: int,
    verification_horizon_ns: int,
) -> tuple[joint.JointPolicySummary, ...]:
    return joint.run_joint_sizing_screen(
        groups=groups,
        rules=rules,
        block_size=block_size,
        path_count=path_count,
        seed=seed,
        challenge_horizon_ns=challenge_horizon_ns,
        verification_horizon_ns=verification_horizon_ns,
        policies=policies,
    )


def verify_coarse_structural_parity(
    *,
    groups: tuple[joint.JointGroup, ...],
    a_trade_count: int,
    u2_episode_count: int,
    block_length: joint.FrozenJointBlockLength,
    diagnostics_path: Path = COARSE_PATH_DIAGNOSTICS,
) -> dict[str, Any]:
    """Fail if the reconstructed DEVELOPMENT path differs from the coarse run."""

    coarse = read_json_artifact(diagnostics_path)
    checks = {
        "joint_group_count": len(groups) == int(coarse["joint_group_count"]),
        "a_trade_count": a_trade_count == int(coarse["a_trade_count"]),
        "u2_episode_count": u2_episode_count == int(coarse["u2_episode_count"]),
        "block_length": asdict(block_length) == coarse["block_length"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise joint.JointFrontierError(
            "refinement joint DEVELOPMENT path is not structurally identical "
            f"to coarse screen: {failed}"
        )
    return {
        "coarse_joint_path_diagnostics_content_sha256": coarse["content_sha256"],
        "checks": checks,
        "same_joint_path_builder": True,
        "same_block_length_derivation": True,
        "same_bootstrap_draw_function": True,
        "same_seed_derivation": "seed + replication",
        "same_ftmo_state_machine": True,
        "same_multiplier_scaling": True,
        "same_policy_reporting": True,
    }


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _dependency_versions() -> dict[str, str]:
    distributions = (
        "arch",
        "ftmoquant",
        "nautilus-trader",
        "numpy",
        "pandas",
        "pyarrow",
        "PyYAML",
        "tradedesk-dukascopy",
        "vectorbt",
    )
    resolved: dict[str, str] = {"python": platform.python_version()}
    for distribution in distributions:
        try:
            resolved[distribution] = version(distribution)
        except PackageNotFoundError:
            resolved[distribution] = "not-installed"
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen 12-policy post-screen DEVELOPMENT local refinement. "
            f"Seed {joint.SCREEN_SEED} and {joint.SCREEN_PATH_COUNT} paths per "
            "method are fixed and have no CLI overrides."
        )
    )
    parser.add_argument("--catalog-root", type=Path, required=True)
    parser.add_argument("--universe-readiness", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def _write_results(
    *,
    output_dir: Path,
    summaries: tuple[joint.JointPolicySummary, ...],
    references: Mapping[tuple[str, str], CoarseReferenceMetrics],
    parity: Mapping[str, Any],
    block_length: joint.FrozenJointBlockLength,
) -> None:
    write_json_artifact(
        output_dir / "refinement_grid.json",
        {
            "a_multipliers": REFINEMENT_A_MULTIPLIERS,
            "u2_multipliers": REFINEMENT_U2_MULTIPLIERS,
            "policy_count": len(REFINEMENT_POLICY_GRID),
            "policies": REFINEMENT_POLICY_GRID,
        },
    )
    rows = _screen_rows(summaries, references)
    write_csv_artifact(output_dir / "refinement_screen.csv", list(rows[0]), rows)

    pareto_ids = joint.compute_pareto_frontier(summaries)
    by_id = {summary.policy.policy_id: summary for summary in summaries}
    pareto_rows = [
        {
            "policy_id": policy_id,
            "pass_both": by_id[policy_id].stationary.pass_both.estimate,
            "median_trading_days_to_pass_both": (
                by_id[policy_id].stationary.median_trading_days_to_pass_both
            ),
            "p90_trading_days_to_pass_both": (
                by_id[policy_id].stationary.p90_trading_days_to_pass_both
            ),
            "fail_max_loss": by_id[policy_id].stationary.fail_max_loss.estimate,
            "p95_max_drawdown": by_id[policy_id].stationary.p95_max_drawdown,
        }
        for policy_id in pareto_ids
    ]
    write_csv_artifact(
        output_dir / "pareto_frontier.csv",
        [
            "policy_id",
            "pass_both",
            "median_trading_days_to_pass_both",
            "p90_trading_days_to_pass_both",
            "fail_max_loss",
            "p95_max_drawdown",
        ],
        pareto_rows,
    )

    selected = joint.select_policy(summaries)
    selected_id = selected.policy.policy_id if selected is not None else None
    write_json_artifact(
        output_dir / "selection_summary.json",
        {
            "selected_policy_id": selected_id,
            "eligibility_verdicts": [
                asdict(joint.evaluate_eligibility(summary)) for summary in summaries
            ],
            "selection_rule": SELECTION_RULE,
            "pareto_frontier": pareto_ids,
            "precision_stage": {
                "implemented_not_run": True,
                "method": "stationary",
                "path_count": 100_000,
                "seed": joint.SCREEN_SEED,
                "selected_refinement_policy_id": selected_id,
                "predetermined_comparison_control_policy_id": (
                    PRECISION_CONTROL_POLICY_ID
                ),
            },
        },
    )
    write_json_artifact(
        output_dir / "refinement_diagnostics.json",
        {
            "stationary_daily_loss_cliff_diagnostics": [
                _stationary_diagnostic(summary) for summary in summaries
            ],
            "coarse_reference_policy_ids": COARSE_REFERENCE_POLICY_IDS,
            "coarse_comparisons_are_descriptive_only": True,
            "structural_parity": parity,
        },
    )
    write_json_artifact(
        output_dir / "metadata.json",
        {
            "label": LABEL,
            "coarse_screen_artifact_path": COARSE_SCREEN_DIR,
            "forensic_audit_verdict": FORENSIC_AUDIT_VERDICT,
            "refinement_rationale": (
                "The coarse DEVELOPMENT screen showed the A=2.0 region safer "
                "but too slow and the A=2.5 region fast enough but riskier. The "
                "forensic audit confirmed a genuine daily-loss threshold cliff "
                "near A=2.5. This refinement searches only the immediate "
                "transition zone below that cliff."
            ),
            "frozen_strategies": {
                "strategy_a": {
                    "identity": joint.STRATEGY_A_IDENTITY,
                    "instrument": "USD/CAD",
                    "timeframe": "M30",
                    "swing_lookback": 40,
                    "rr": 2.0,
                },
                "strategy_b": {
                    "identity": "B3F1 U2 spread mean reversion",
                    "pair": "USD/CAD.OANDA__USD/CHF.OANDA",
                    "formation_window": joint.FROZEN_U2_FORMATION_WINDOW,
                    "z_entry": joint.FROZEN_U2_Z_ENTRY,
                    "z_stop": joint.FROZEN_U2_Z_STOP,
                },
            },
            "grid": REFINEMENT_POLICY_GRID,
            "eligibility_rule": ELIGIBILITY_RULE,
            "selection_rule": SELECTION_RULE,
            "pareto_rule": PARETO_RULE,
            "monte_carlo": {
                "primary": "StationaryBootstrap",
                "secondary": "CircularBlockBootstrap",
                "paths_per_method": joint.SCREEN_PATH_COUNT,
                "seed": joint.SCREEN_SEED,
                "seed_derivation": "seed + replication",
                "block_length": block_length,
            },
            "development_only_statement": DEVELOPMENT_ONLY_STATEMENT,
            "independent_evidence": False,
            "validation_accessed": False,
            "holdout_accessed": False,
            "git_commit": _git_commit(),
            "dependency_versions": _dependency_versions(),
        },
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    reserve_output_directory(args.output)

    references = load_coarse_references()
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
    parity = verify_coarse_structural_parity(
        groups=groups,
        a_trade_count=len(a_path.trades),
        u2_episode_count=len(u2_episodes),
        block_length=block_length,
    )
    summaries = run_refinement_screen(
        groups=groups,
        rules=rules,
        block_size=block_length.frozen_block_size,
    )
    _write_results(
        output_dir=args.output,
        summaries=summaries,
        references=references,
        parity=parity,
        block_length=block_length,
    )
    print(f"refinement complete: {len(summaries)} policies evaluated")


if __name__ == "__main__":
    main()
