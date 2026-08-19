"""Read-only VALIDATION Monte Carlo diagnostic for the already-frozen,
DEVELOPMENT-selected sizing policy.

This is evaluation, not another research/search stage. It reuses every
piece of the existing DEVELOPMENT machinery unchanged --
:func:`ftmoquant.research.ftmo_pass_probability.monte_carlo.
simulate_two_phase_path`, :func:`~.reporting.summarize_policy`, the same
:class:`~.state_machine.FtmoPathStatus` semantics -- and adds nothing new
except where the DEVELOPMENT VALIDATION firewall previously refused entry.

Hard freezes (task's anti-tuning rule): the sizing policy and bootstrap
method are Python constants, not parameters. Nothing in this module (or the
CLI built on it) accepts a policy id or resampling method as input.
VALIDATION trades are used only to *replay* this already-chosen policy
under this already-chosen method -- never to rank, compare, or re-derive
either. The block length is derived from DEVELOPMENT trades only (never
from VALIDATION) via the same
:func:`~.bootstrap.derive_frozen_block_length` used by the frozen
DEVELOPMENT precision run, and is recorded in the output artifact together
with that provenance so the reuse is auditable, not just asserted.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from ftmoquant.prop_rules.models import PropRuleSet
from ftmoquant.research.ftmo_pass_probability.bootstrap import (
    FrozenBlockLength,
    derive_frozen_block_length,
)
from ftmoquant.research.ftmo_pass_probability.monte_carlo import (
    TwoPhaseOutcome,
    precompute_trade_timing,
    simulate_two_phase_path,
)
from ftmoquant.research.ftmo_pass_probability.path_extraction import (
    DevelopmentTradePath,
    ValidationTradePath,
    load_development_trade_path,
    load_validation_trade_path,
)
from ftmoquant.research.ftmo_pass_probability.reporting import (
    PolicySummary,
    summarize_policy,
)
from ftmoquant.research.ftmo_pass_probability.sizing import SIZING_GRID, SizingPolicy

#: Hard-frozen by the already-completed DEVELOPMENT selection. Not a CLI
#: parameter -- there is deliberately no code path that accepts a
#: different value for either constant.
FROZEN_POLICY_ID = "fixed_notional_2_0x"
FROZEN_METHOD = "stationary"


class ValidationDiagnosticError(ValueError):
    """Raised when the frozen policy/method cannot be resolved safely."""


def frozen_policy() -> SizingPolicy:
    """Return the single, hard-frozen sizing policy this diagnostic may use."""

    for policy in SIZING_GRID:
        if policy.policy_id == FROZEN_POLICY_ID:
            return policy
    raise ValidationDiagnosticError(
        f"frozen policy {FROZEN_POLICY_ID!r} is missing from SIZING_GRID"
    )


@dataclass(frozen=True, slots=True)
class ValidationDiagnosticResult:
    """Everything needed to report the VALIDATION diagnostic artifact."""

    summary: PolicySummary
    block_length: FrozenBlockLength
    development_trade_count: int
    validation_trade_count: int
    development_trades_csv_sha256: str
    validation_trades_csv_sha256: str


def run_validation_diagnostic(
    *,
    development_execution_dir: Path,
    validation_execution_dir: Path,
    rules: PropRuleSet,
    initial_capital: Decimal,
    paths: int,
    seed: int,
    challenge_horizon_ns: int,
    verification_horizon_ns: int,
    derive_seed: Callable[[int, str, str, int], int],
) -> ValidationDiagnosticResult:
    """Replay VALIDATION trades under the frozen policy/method only.

    ``derive_seed`` is the caller's existing deterministic
    ``(base_seed, method, policy_id, replication) -> int`` seed-derivation
    callable -- passed in rather than imported privately from ``cli.py`` so
    this module has no dependency in the other direction; it must be the
    exact same function used by the DEVELOPMENT precision run for the
    methodology to be preserved.
    """

    if paths <= 0:
        raise ValidationDiagnosticError("paths must be positive")

    development: DevelopmentTradePath = load_development_trade_path(
        development_execution_dir
    )
    validation: ValidationTradePath = load_validation_trade_path(
        validation_execution_dir
    )

    # Block length is derived from DEVELOPMENT only -- never re-estimated
    # from VALIDATION -- per the task's frozen-methodology requirement.
    block_length = derive_frozen_block_length(development.trades)

    policy = frozen_policy()
    timing = precompute_trade_timing(validation.trades)

    outcomes: list[TwoPhaseOutcome] = []
    for replication in range(paths):
        replication_seed = derive_seed(
            seed, FROZEN_METHOD, policy.policy_id, replication
        )
        outcomes.append(
            simulate_two_phase_path(
                validation.trades,
                timing,
                method=FROZEN_METHOD,  # type: ignore[arg-type]
                block_size=block_length.frozen_block_size,
                policy=policy,
                rules=rules,
                initial_capital=initial_capital,
                challenge_horizon_ns=challenge_horizon_ns,
                verification_horizon_ns=verification_horizon_ns,
                seed=replication_seed,
            )
        )

    summary = summarize_policy(policy.policy_id, FROZEN_METHOD, tuple(outcomes))
    return ValidationDiagnosticResult(
        summary=summary,
        block_length=block_length,
        development_trade_count=len(development.trades),
        validation_trade_count=len(validation.trades),
        development_trades_csv_sha256=development.trades_csv_sha256,
        validation_trades_csv_sha256=validation.trades_csv_sha256,
    )
