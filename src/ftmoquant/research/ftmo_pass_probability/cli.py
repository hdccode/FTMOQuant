"""CLI entry points for the FTMO pass-probability sizing/Monte Carlo layer.

Five subcommands, matching task §21's required exact Terminal commands:

- ``path-diagnostics``: DEVELOPMENT-only, cheap (no Monte Carlo). Writes
  ``ftmo_rules_snapshot.json``, ``resampling_spec.json``, ``sizing_grid.json``,
  ``development_path_diagnostics.json``.
- ``sizing-screen``: runs the 20,000-path-per-policy-per-method Monte Carlo
  screen across all 8 frozen sizing candidates. Writes ``sizing_screen.csv``.
- ``sizing-screen-refinement``: a distinct, explicitly-labelled follow-up
  screen over ``sizing.NOTIONAL_REFINEMENT_GRID`` only (9 fixed-notional
  multipliers, 1.50x-2.50x) -- a local refinement around the optimum
  observed in ``sizing_screen.csv``. Reuses the exact same rules, bootstrap
  methodology, block length, and seed-derivation as ``sizing-screen``.
  Writes ``sizing_screen_refinement.csv``; never touches ``sizing_screen.csv``.
- ``precision-run``: runs a 100,000-path precision Monte Carlo for one
  already-selected policy. Writes ``selected_policy_precision.json``.
- ``validation-diagnostic``: read-only VALIDATION replay of the single,
  hard-frozen ``fixed_notional_2_0x``/``stationary`` policy selected from
  DEVELOPMENT. Takes **no** ``--policy-id``/``--method`` flags -- see
  ``validation_diagnostic.py`` for the anti-tuning rule this enforces.
  Writes ``frozen_policy_validation_diagnostic.json``.

Deliberately does not select a policy: none of these commands picks a
winner -- a human (or a separate, explicit follow-up step) chooses the
``--policy-id``/``--method`` given to ``precision-run``, and only that
already-frozen choice may ever reach ``validation-diagnostic``.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path
from statistics import mean, pstdev

from ftmoquant.prop_rules import load_prop_rule_set
from ftmoquant.prop_rules.models import EvaluationPhase, PropRuleSet
from ftmoquant.research.ftmo_pass_probability.alpha_diagnostic import (
    compare_trade_distributions,
    diagnostic_a_trade_distribution,
    diagnostic_b_chronological_path,
    diagnostic_c_temporal_stability,
    diagnostic_d_subgroups,
    diagnostic_e_chronological_replay,
    frozen_policy_pnl_series,
    subgroup_eligibility,
)
from ftmoquant.research.ftmo_pass_probability.artifacts import (
    read_json_artifact,
    write_csv_artifact,
    write_json_artifact,
)
from ftmoquant.research.ftmo_pass_probability.bootstrap import (
    derive_frozen_block_length,
)
from ftmoquant.research.ftmo_pass_probability.monte_carlo import (
    TradeTiming,
    TwoPhaseOutcome,
    precompute_trade_timing,
    simulate_two_phase_path,
)
from ftmoquant.research.ftmo_pass_probability.path_extraction import (
    FROZEN_CANDIDATE_IDENTITY,
    TradeRecord,
    load_development_trade_path,
    load_validation_trade_path,
)
from ftmoquant.research.ftmo_pass_probability.reporting import (
    rank_policies,
    summarize_policy,
)
from ftmoquant.research.ftmo_pass_probability.sizing import (
    NOTIONAL_REFINEMENT_GRID,
    SIZING_GRID,
    SizingPolicy,
    apply_sizing,
)
from ftmoquant.research.ftmo_pass_probability.state_machine import (
    TradeEvent,
    simulate_phase,
)
from ftmoquant.research.ftmo_pass_probability.validation_diagnostic import (
    frozen_policy,
    run_validation_diagnostic,
)

DEFAULT_RULE_CONFIG = Path("config/prop/ftmo_2step_swing_2026-08.yaml")
DEFAULT_EXECUTION_DIR = Path(
    ".artifacts/usdcad_sweep_bos_retest_v1/development_execution"
)
DEFAULT_VALIDATION_EXECUTION_DIR = Path(
    ".artifacts/usdcad_sweep_bos_retest_v1/validation_execution"
)
DEFAULT_OUTPUT_ROOT = Path(
    ".artifacts/ftmo_pass_probability/usdcad_sweep_bos_retest_v1"
)
DEFAULT_DEVELOPMENT_PRECISION_PATH = (
    DEFAULT_OUTPUT_ROOT / "selected_policy_precision.json"
)
DEFAULT_INITIAL_CAPITAL = Decimal("100000")
#: task §12: "suggested max 3 synthetic years per phase"
DEFAULT_HORIZON_NS = 3 * 365 * 24 * 60 * 60 * 1_000_000_000
PRIMARY_METHODS = ("stationary", "circular")
MODULE_VERSION = "ftmo-pass-probability-v1"


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _provenance() -> dict[str, object]:
    return {
        "module_version": MODULE_VERSION,
        "git_commit": _git_commit(),
        "arch_version": version("arch"),
        "python_version": platform.python_version(),
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }


def path_diagnostics_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-dir", type=Path, default=DEFAULT_EXECUTION_DIR)
    parser.add_argument("--rule-config", type=Path, default=DEFAULT_RULE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--initial-capital", type=Decimal, default=DEFAULT_INITIAL_CAPITAL
    )
    args = parser.parse_args(argv)

    rules = load_prop_rule_set(args.rule_config)
    path = load_development_trade_path(args.execution_dir)
    timing = precompute_trade_timing(path.trades)
    block_length = derive_frozen_block_length(path.trades)

    net_r = [float(trade.net_r) for trade in path.trades]
    diagnostics = {
        "frozen_candidate_identity": FROZEN_CANDIDATE_IDENTITY,
        "development_source_dir": str(args.execution_dir),
        "trades_csv_sha256": path.trades_csv_sha256,
        "trade_count": len(path.trades),
        "net_r_mean": mean(net_r),
        "net_r_stdev": pstdev(net_r) if len(net_r) > 1 else 0.0,
        "net_r_lag1_autocorrelation": _lag1_autocorrelation(net_r),
        "win_loss_streaks": _streak_lengths(path.trades),
        "exit_reason_counts": {
            "stop": sum(1 for t in path.trades if t.exit_reason == "stop"),
            "target": sum(1 for t in path.trades if t.exit_reason == "target"),
        },
        "holding_ns_stats": _stats([t.holding_ns for t in timing]),
        "gap_before_ns_stats": _stats([t.gap_before_ns for t in timing]),
        "replay_proof": _proof_replay(path.trades, timing, rules, args.initial_capital),
        **_provenance(),
    }
    rules_snapshot = {
        "schema_version": rules.schema_version,
        "rule_set_id": rules.rule_set_id,
        "version": rules.version,
        "provider": rules.provider.value,
        "program": rules.program.value,
        "account_type": rules.account_type.value,
        "verified_on": rules.verified_on.isoformat(),
        "phases": [
            {
                "phase": phase.phase.value,
                "profit_target": str(phase.profit_target),
                "minimum_trading_days": phase.minimum_trading_days,
                "evaluation_period_days": phase.evaluation_period_days,
            }
            for phase in rules.phases
        ],
        "loss_limits": {
            "maximum_daily_loss": str(rules.loss_limits.maximum_daily_loss),
            "maximum_loss": str(rules.loss_limits.maximum_loss),
            "maximum_loss_type": rules.loss_limits.maximum_loss_type.value,
        },
        "daily_reset": {
            "time": rules.daily_reset.time.isoformat(),
            "timezone": str(rules.daily_reset.timezone),
        },
        "source_urls": list(rules.source_urls),
        "loaded_from": str(args.rule_config),
        **_provenance(),
    }
    resampling_spec = {
        "primary_method": "stationary",
        "secondary_method": "circular",
        "diagnostic_only_method": "iid_diagnostic",
        "resampling_unit": "consecutive_complete_trade_episodes",
        "stationary_block_length_estimate": block_length.stationary_block_length,
        "circular_block_length_estimate": block_length.circular_block_length,
        "frozen_block_size": block_length.frozen_block_size,
        "observation_count": block_length.observation_count,
        "input_content_sha256": block_length.input_content_sha256,
        "development_only": True,
        **_provenance(),
    }
    sizing_grid = {
        "policies": [asdict(policy) for policy in SIZING_GRID],
        "policy_count": len(SIZING_GRID),
        **_provenance(),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_artifact(args.output_dir / "ftmo_rules_snapshot.json", rules_snapshot)
    write_json_artifact(args.output_dir / "resampling_spec.json", resampling_spec)
    write_json_artifact(args.output_dir / "sizing_grid.json", sizing_grid)
    write_json_artifact(
        args.output_dir / "development_path_diagnostics.json", diagnostics
    )
    print(f"wrote path diagnostics to {args.output_dir}")


def sizing_screen_main(argv: list[str] | None = None) -> None:
    _run_sizing_screen(argv, grid=SIZING_GRID, output_filename="sizing_screen.csv")


def sizing_screen_refinement_main(argv: list[str] | None = None) -> None:
    """Local refinement of the fixed-notional family only, around the
    apparent optimum observed in the frozen ``SIZING_GRID`` screen (task
    follow-up: 1.50x-2.50x). Reuses ``_run_sizing_screen`` unchanged --
    identical FTMO rules, bootstrap methodology, block length, seed
    derivation, and horizons -- so the only thing that differs from
    ``sizing_screen_main`` is which grid is scored and which file is
    written. Never touches ``sizing_screen.csv`` (the frozen 8-candidate
    benchmark output).
    """

    _run_sizing_screen(
        argv,
        grid=NOTIONAL_REFINEMENT_GRID,
        output_filename="sizing_screen_refinement.csv",
    )


def _run_sizing_screen(
    argv: list[str] | None, *, grid: tuple[SizingPolicy, ...], output_filename: str
) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-dir", type=Path, default=DEFAULT_EXECUTION_DIR)
    parser.add_argument("--rule-config", type=Path, default=DEFAULT_RULE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--initial-capital", type=Decimal, default=DEFAULT_INITIAL_CAPITAL
    )
    parser.add_argument("--paths", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--challenge-horizon-ns", type=int, default=DEFAULT_HORIZON_NS)
    parser.add_argument(
        "--verification-horizon-ns", type=int, default=DEFAULT_HORIZON_NS
    )
    args = parser.parse_args(argv)

    rules = load_prop_rule_set(args.rule_config)
    path = load_development_trade_path(args.execution_dir)
    timing = precompute_trade_timing(path.trades)
    block_length = derive_frozen_block_length(path.trades)

    rows: list[dict[str, object]] = []
    summaries = []
    for method in PRIMARY_METHODS:
        for policy in grid:
            outcomes: list[TwoPhaseOutcome] = []
            for replication in range(args.paths):
                seed = _derive_seed(args.seed, method, policy.policy_id, replication)
                outcomes.append(
                    simulate_two_phase_path(
                        path.trades,
                        timing,
                        method=method,  # type: ignore[arg-type]
                        block_size=block_length.frozen_block_size,
                        policy=policy,
                        rules=rules,
                        initial_capital=args.initial_capital,
                        challenge_horizon_ns=args.challenge_horizon_ns,
                        verification_horizon_ns=args.verification_horizon_ns,
                        seed=seed,
                    )
                )
            summary = summarize_policy(policy.policy_id, method, tuple(outcomes))
            summaries.append(summary)
            rows.append(
                {
                    "policy_id": summary.policy_id,
                    "method": summary.method,
                    "replications": summary.replications,
                    "pass_challenge": summary.pass_challenge.estimate,
                    "pass_both_estimate": summary.pass_both.estimate,
                    "pass_both_ci_lower_95": summary.pass_both.ci_lower_95,
                    "pass_both_ci_upper_95": summary.pass_both.ci_upper_95,
                    "fail_daily_loss": summary.fail_daily_loss.estimate,
                    "fail_max_loss": summary.fail_max_loss.estimate,
                    "censoring_rate": summary.censoring_rate.estimate,
                    "median_trading_days_to_pass_both": (
                        summary.median_trading_days_to_pass_both
                    ),
                    "p90_trading_days_to_pass_both": (
                        summary.p90_trading_days_to_pass_both
                    ),
                    "p95_trading_days_to_pass_both": (
                        summary.p95_trading_days_to_pass_both
                    ),
                    "median_max_drawdown": summary.median_max_drawdown,
                    "p95_max_drawdown": summary.p95_max_drawdown,
                    "certainty_tier": summary.certainty_tier.value,
                }
            )

    ranked = rank_policies(tuple(summaries))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_artifact(args.output_dir / output_filename, list(rows[0].keys()), rows)
    print("ranked policies (best first):")
    for summary in ranked:
        pass_both = summary.pass_both.estimate
        print(f"  {summary.method}/{summary.policy_id}: pass_both={pass_both:.4f}")


def precision_run_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-dir", type=Path, default=DEFAULT_EXECUTION_DIR)
    parser.add_argument("--rule-config", type=Path, default=DEFAULT_RULE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--initial-capital", type=Decimal, default=DEFAULT_INITIAL_CAPITAL
    )
    parser.add_argument("--policy-id", type=str, required=True)
    parser.add_argument("--method", type=str, choices=PRIMARY_METHODS, required=True)
    parser.add_argument("--paths", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--challenge-horizon-ns", type=int, default=DEFAULT_HORIZON_NS)
    parser.add_argument(
        "--verification-horizon-ns", type=int, default=DEFAULT_HORIZON_NS
    )
    args = parser.parse_args(argv)

    rules = load_prop_rule_set(args.rule_config)
    path = load_development_trade_path(args.execution_dir)
    timing = precompute_trade_timing(path.trades)
    block_length = derive_frozen_block_length(path.trades)
    policy = next((p for p in SIZING_GRID if p.policy_id == args.policy_id), None)
    if policy is None:
        raise SystemExit(f"unknown policy_id {args.policy_id!r}")

    outcomes = tuple(
        simulate_two_phase_path(
            path.trades,
            timing,
            method=args.method,
            block_size=block_length.frozen_block_size,
            policy=policy,
            rules=rules,
            initial_capital=args.initial_capital,
            challenge_horizon_ns=args.challenge_horizon_ns,
            verification_horizon_ns=args.verification_horizon_ns,
            seed=_derive_seed(args.seed, args.method, policy.policy_id, replication),
        )
        for replication in range(args.paths)
    )
    summary = summarize_policy(policy.policy_id, args.method, outcomes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_artifact(
        args.output_dir / "selected_policy_precision.json",
        {
            "policy_id": summary.policy_id,
            "method": summary.method,
            "replications": summary.replications,
            "pass_challenge": asdict(summary.pass_challenge),
            "pass_both": asdict(summary.pass_both),
            "fail_daily_loss": asdict(summary.fail_daily_loss),
            "fail_max_loss": asdict(summary.fail_max_loss),
            "censoring_rate": asdict(summary.censoring_rate),
            "median_trading_days_to_pass_both": (
                summary.median_trading_days_to_pass_both
            ),
            "p90_trading_days_to_pass_both": summary.p90_trading_days_to_pass_both,
            "p95_trading_days_to_pass_both": summary.p95_trading_days_to_pass_both,
            "median_max_drawdown": summary.median_max_drawdown,
            "p95_max_drawdown": summary.p95_max_drawdown,
            "certainty_tier": summary.certainty_tier.value,
            **_provenance(),
        },
    )
    print(
        f"pass_both = {summary.pass_both.estimate:.4f} [{summary.certainty_tier.value}]"
    )


def validation_diagnostic_main(argv: list[str] | None = None) -> None:
    """Read-only VALIDATION replay of the hard-frozen fixed_notional_2_0x /
    stationary policy. Deliberately exposes no ``--policy-id``/``--method``
    flag -- see ``validation_diagnostic.py`` for why.
    """

    parser = argparse.ArgumentParser(description=validation_diagnostic_main.__doc__)
    parser.add_argument(
        "--development-execution-dir", type=Path, default=DEFAULT_EXECUTION_DIR
    )
    parser.add_argument(
        "--validation-execution-dir",
        type=Path,
        default=DEFAULT_VALIDATION_EXECUTION_DIR,
    )
    parser.add_argument("--rule-config", type=Path, default=DEFAULT_RULE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--development-precision-path",
        type=Path,
        default=DEFAULT_DEVELOPMENT_PRECISION_PATH,
    )
    parser.add_argument(
        "--initial-capital", type=Decimal, default=DEFAULT_INITIAL_CAPITAL
    )
    parser.add_argument("--paths", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--challenge-horizon-ns", type=int, default=DEFAULT_HORIZON_NS)
    parser.add_argument(
        "--verification-horizon-ns", type=int, default=DEFAULT_HORIZON_NS
    )
    args = parser.parse_args(argv)

    rules = load_prop_rule_set(args.rule_config)
    result = run_validation_diagnostic(
        development_execution_dir=args.development_execution_dir,
        validation_execution_dir=args.validation_execution_dir,
        rules=rules,
        initial_capital=args.initial_capital,
        paths=args.paths,
        seed=args.seed,
        challenge_horizon_ns=args.challenge_horizon_ns,
        verification_horizon_ns=args.verification_horizon_ns,
        derive_seed=_derive_seed,
    )
    summary = result.summary
    development_precision = read_json_artifact(args.development_precision_path)
    development_pass_both = development_precision.get("pass_both", {}).get("estimate")
    pass_both_abs_difference = (
        None
        if development_pass_both is None
        else abs(summary.pass_both.estimate - float(development_pass_both))
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_artifact(
        args.output_dir / "frozen_policy_validation_diagnostic.json",
        {
            "diagnostic_only": True,
            "policy_selected_on_validation": False,
            "source_partition": "validation",
            "policy_id": summary.policy_id,
            "method": summary.method,
            "paths": args.paths,
            "seed": args.seed,
            "replications": summary.replications,
            "block_size": {
                "frozen_block_size": result.block_length.frozen_block_size,
                "provenance": (
                    "derived from DEVELOPMENT trades only via "
                    "bootstrap.derive_frozen_block_length, reused unchanged "
                    "for this VALIDATION replay -- never re-estimated from "
                    "VALIDATION"
                ),
                "stationary_block_length_estimate": (
                    result.block_length.stationary_block_length
                ),
                "circular_block_length_estimate": (
                    result.block_length.circular_block_length
                ),
                "development_observation_count": (
                    result.block_length.observation_count
                ),
                "development_input_content_sha256": (
                    result.block_length.input_content_sha256
                ),
            },
            "pass_challenge": asdict(summary.pass_challenge),
            "pass_both": asdict(summary.pass_both),
            "fail_daily_loss": asdict(summary.fail_daily_loss),
            "fail_max_loss": asdict(summary.fail_max_loss),
            "censoring_rate": asdict(summary.censoring_rate),
            "median_trading_days_to_pass_both": (
                summary.median_trading_days_to_pass_both
            ),
            "p90_trading_days_to_pass_both": summary.p90_trading_days_to_pass_both,
            "p95_trading_days_to_pass_both": summary.p95_trading_days_to_pass_both,
            "median_max_drawdown": summary.median_max_drawdown,
            "p95_max_drawdown": summary.p95_max_drawdown,
            "certainty_tier": summary.certainty_tier.value,
            "development_trade_count": result.development_trade_count,
            "validation_trade_count": result.validation_trade_count,
            "development_trades_csv_sha256": (result.development_trades_csv_sha256),
            "validation_trades_csv_sha256": result.validation_trades_csv_sha256,
            "development_precision_result": development_precision,
            "development_precision_source_path": str(args.development_precision_path),
            "pass_both_absolute_difference_vs_development": (pass_both_abs_difference),
            "note": (
                "Diagnostic evidence only: reports how the already-frozen "
                "DEVELOPMENT-selected policy behaves on VALIDATION. No "
                "pass/fail gate is defined or evaluated from this "
                "comparison, and VALIDATION was not used to select, rank, "
                "or re-derive this policy, method, or block length."
            ),
            **_provenance(),
        },
    )
    print(
        f"VALIDATION pass_both = {summary.pass_both.estimate:.4f} "
        f"(DEVELOPMENT = {development_pass_both}, "
        f"|diff| = {pass_both_abs_difference})"
    )


def alpha_diagnostic_main(argv: list[str] | None = None) -> None:
    """Read-only DEVELOPMENT-vs-VALIDATION alpha/distribution diagnostic
    (task: "alpha first, FTMO optimization second"). No Monte Carlo, no
    bootstrap, no sizing/method selection -- see ``alpha_diagnostic.py``.
    """

    parser = argparse.ArgumentParser(description=alpha_diagnostic_main.__doc__)
    parser.add_argument(
        "--development-execution-dir", type=Path, default=DEFAULT_EXECUTION_DIR
    )
    parser.add_argument(
        "--validation-execution-dir",
        type=Path,
        default=DEFAULT_VALIDATION_EXECUTION_DIR,
    )
    parser.add_argument("--rule-config", type=Path, default=DEFAULT_RULE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--initial-capital", type=Decimal, default=DEFAULT_INITIAL_CAPITAL
    )
    args = parser.parse_args(argv)

    rules = load_prop_rule_set(args.rule_config)
    development = load_development_trade_path(args.development_execution_dir)
    validation = load_validation_trade_path(args.validation_execution_dir)

    development_pnl = frozen_policy_pnl_series(development.trades)
    validation_pnl = frozen_policy_pnl_series(validation.trades)

    distribution_development = diagnostic_a_trade_distribution(
        development.trades, development_pnl
    )
    distribution_validation = diagnostic_a_trade_distribution(
        validation.trades, validation_pnl
    )
    distribution_comparison = compare_trade_distributions(
        distribution_development, distribution_validation
    )

    path_development, rolling_series_development = diagnostic_b_chronological_path(
        development.trades, development_pnl, args.initial_capital
    )
    path_validation, rolling_series_validation = diagnostic_b_chronological_path(
        validation.trades, validation_pnl, args.initial_capital
    )

    stability_development = diagnostic_c_temporal_stability(
        development.trades, development_pnl
    )
    stability_validation = diagnostic_c_temporal_stability(
        validation.trades, validation_pnl
    )

    subgroups_development = diagnostic_d_subgroups(
        args.development_execution_dir, development.trades, development_pnl
    )
    subgroups_validation = diagnostic_d_subgroups(
        args.validation_execution_dir, validation.trades, validation_pnl
    )

    replay_development = diagnostic_e_chronological_replay(
        development.trades, rules, args.initial_capital
    )
    replay_validation = diagnostic_e_chronological_replay(
        validation.trades, rules, args.initial_capital
    )

    observations = _largest_changes(distribution_comparison)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_artifact(
        args.output_dir / "development_validation_alpha_diagnostic.json",
        {
            "diagnostic_only": True,
            "validation_used_for_selection": False,
            "monte_carlo_or_bootstrap_used": False,
            "final_holdout_accessed": False,
            "frozen_policy_id": frozen_policy().policy_id,
            "frozen_candidate_identity": FROZEN_CANDIDATE_IDENTITY,
            "development_trade_count": len(development.trades),
            "validation_trade_count": len(validation.trades),
            "development_trades_csv_sha256": development.trades_csv_sha256,
            "validation_trades_csv_sha256": validation.trades_csv_sha256,
            "diagnostic_a_trade_distribution": {
                "development": distribution_development,
                "validation": distribution_validation,
                "comparison": distribution_comparison,
            },
            "diagnostic_b_chronological_path": {
                "development": path_development,
                "validation": path_validation,
                "rolling_series_note": (
                    "full rolling 30/50-trade expectancy series written to "
                    "development_validation_alpha_diagnostic_rolling.csv"
                ),
            },
            "diagnostic_c_temporal_stability": {
                "development": stability_development,
                "validation": stability_validation,
            },
            "diagnostic_d_subgroups": {
                "dimensions_inspected": subgroup_eligibility(),
                "development": subgroups_development,
                "validation": subgroups_validation,
            },
            "diagnostic_e_chronological_replay": {
                "development": replay_development,
                "validation": replay_validation,
            },
            "observations": observations,
            **_provenance(),
        },
    )

    rolling_rows = [
        {
            "partition": partition,
            "window_size": window,
            "index": index + 1,
            "expectancy": value,
        }
        for partition, series_by_window in (
            ("development", rolling_series_development),
            ("validation", rolling_series_validation),
        )
        for window, series in series_by_window.items()
        for index, value in enumerate(series)
    ]
    if rolling_rows:
        write_csv_artifact(
            args.output_dir / "development_validation_alpha_diagnostic_rolling.csv",
            list(rolling_rows[0].keys()),
            rolling_rows,
        )

    print(
        "wrote alpha diagnostic to "
        f"{args.output_dir / 'development_validation_alpha_diagnostic.json'}"
    )


def _largest_changes(
    comparison: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    """Purely descriptive ranking of the largest DEVELOPMENT->VALIDATION
    changes by absolute relative difference. Reports, does not recommend."""

    ranked = sorted(
        (
            {"metric": metric, **fields}
            for metric, fields in comparison.items()
            if fields.get("relative_difference") is not None
        ),
        key=lambda row: abs(row["relative_difference"]),  # type: ignore[arg-type]
        reverse=True,
    )
    return ranked[:5]


def _derive_seed(base_seed: int, method: str, policy_id: str, replication: int) -> int:
    import hashlib

    digest = hashlib.sha256(
        f"{base_seed}:{method}:{policy_id}:{replication}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _lag1_autocorrelation(values: list[float]) -> float | None:
    if len(values) < 3:
        return None
    average = mean(values)
    numerator = sum(
        (values[i] - average) * (values[i + 1] - average)
        for i in range(len(values) - 1)
    )
    denominator = sum((value - average) ** 2 for value in values)
    return numerator / denominator if denominator != 0 else None


def _streak_lengths(trades: tuple[TradeRecord, ...]) -> dict[str, object]:
    streaks: list[int] = []
    current_sign = None
    current_length = 0
    for trade in trades:
        sign = "win" if trade.net_r > 0 else "loss"
        if sign == current_sign:
            current_length += 1
        else:
            if current_sign is not None:
                streaks.append(current_length)
            current_sign = sign
            current_length = 1
    if current_sign is not None:
        streaks.append(current_length)
    return {
        "max_streak_length": max(streaks) if streaks else 0,
        "mean_streak_length": mean(streaks) if streaks else 0.0,
        "streak_count": len(streaks),
    }


def _stats(values: list[int]) -> dict[str, float]:
    floats = [float(value) for value in values]
    return {
        "mean": mean(floats),
        "min": min(floats),
        "max": max(floats),
        "stdev": pstdev(floats) if len(floats) > 1 else 0.0,
    }


def _proof_replay(
    trades: tuple[TradeRecord, ...],
    timing: tuple[TradeTiming, ...],
    rules: PropRuleSet,
    initial_capital: Decimal,
) -> dict[str, object]:
    """Replay the real (unresampled) DEVELOPMENT trades at 1x frozen notional
    through the pure state machine and report what the real path implies for
    an account of ``initial_capital`` -- proves the extraction pipeline
    reproduces genuine trading-day counts and final balance, and reports
    (does not select on) whether the real path would have breached.
    """

    policy_1x = next(p for p in SIZING_GRID if p.policy_id == "fixed_notional_1_0x")
    balance = initial_capital
    events = []
    for trade in trades:
        sized = apply_sizing(policy_1x, trade, balance)
        events.append(
            TradeEvent(
                entry_ns=trade.entry_ns,
                exit_ns=trade.exit_ns,
                floor_equity_delta=sized.floor_equity_delta,
                realized_pnl=sized.realized_pnl,
            )
        )
        balance += sized.realized_pnl
    total_span_ns = trades[-1].exit_ns - trades[0].entry_ns + 1
    outcome = simulate_phase(
        events,
        rules=rules,
        phase=EvaluationPhase.CHALLENGE,
        initial_capital=initial_capital,
        horizon_ns=total_span_ns,
    )
    naive_final_balance = initial_capital + sum(
        (t.realized_pnl for t in events), Decimal("0")
    )
    return {
        "naive_final_balance_no_breach_stop": str(naive_final_balance),
        "state_machine_status": outcome.status.value,
        "state_machine_ending_balance": str(outcome.ending_balance),
        "state_machine_trading_days": outcome.trading_days,
        "state_machine_trades_replayed": outcome.trades_replayed,
        "note": (
            "Uses the fixed_notional_1_0x sizing candidate (same 100,000-unit "
            "notional as the frozen DEVELOPMENT execution) so "
            "naive_final_balance reproduces the execution artifact's own "
            "net_return exactly when no breach occurs; state_machine_* fields "
            "report what an FTMO account of this initial_capital would "
            "actually experience, including any breach precedence, which "
            "the raw execution artifact never evaluated."
        ),
    }
