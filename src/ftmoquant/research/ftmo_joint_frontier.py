"""Joint two-strategy FTMO sizing-frontier infrastructure: Strategy A
(``usdcad_sweep_bos_retest_v1``) + Strategy B (B3F1 underpowered candidate
U2), DEVELOPMENT-only.

This module builds the framework -- it never runs the real 20,000-path
screen itself (that is a separate, explicitly non-Codex CLI invocation; see
:func:`build_parser`).

Reuses, rather than re-derives:

- :mod:`ftmoquant.research.ftmo_pass_probability.state_machine`
  (``simulate_phase``, ``TradeEvent``) -- completely UNCHANGED. The joint
  account path is reduced to the exact same ``TradeEvent`` shape that
  single-strategy module already consumes; this module contains no second
  FTMO breach/pass state machine.
- :mod:`ftmoquant.research.ftmo_pass_probability.bootstrap`
  (``draw_index_path``, ``estimate_optimal_block_length`` via
  ``derive_frozen_block_length``'s own pattern) -- UNCHANGED. The
  dependence-preserving resampling unit is simply a longer index sequence
  fed to the same ``StationaryBootstrap``/``CircularBlockBootstrap``
  wiring; no new resampling algorithm is implemented here.
- :mod:`ftmoquant.research.ftmo_pass_probability.sizing` (``SizingPolicy``,
  ``SizingFamily``, ``apply_sizing``) -- UNCHANGED, for Strategy A's own
  fixed-notional-multiplier scaling.
- :mod:`ftmoquant.research.ftmo_pass_probability.path_extraction`
  (``load_development_trade_path``, ``TradeRecord``) -- UNCHANGED, for
  Strategy A's frozen DEVELOPMENT trade sequence.
- :mod:`ftmoquant.research.ftmo_pass_probability.reporting``
  (``summarize_policy``, ``wilson_score_interval``, ``PolicySummary``) --
  UNCHANGED; this module's Monte Carlo replications are handed to
  ``summarize_policy`` exactly as the single-strategy screen already does.
- :mod:`ftmoquant.research.alpha_lab.relative_value_adapter``
  (``RelativeValueEpisode.combined_pnl_usd_at``,
  ``RelativeValueLeg.contribution_at``) -- UNCHANGED, for U2's exact
  intratrade mark-to-market path (this is what makes U2's temporary
  legging exposure exactly, not approximately, representable).
- :mod:`ftmoquant.research.b3f1_u2_execution_promotion` (``Y_SPEC``,
  ``X_SPEC``, ``FORMATION_WINDOW``, ``Z_ENTRY``, ``Z_STOP``,
  ``resolve_leg_roots``, signal/execution pipeline pieces) -- UNCHANGED,
  for U2's frozen DEVELOPMENT episode sequence.

No second FTMO simulator, no second bootstrap engine, no second sizing
family, and no second breach state machine exist anywhere in this module.

==============================================================================
Joint account path representation (task section 3)
==============================================================================

The two strategies' raw trade/episode sequences are merged into
:class:`JointGroup` objects: a maximal, TRANSITIVELY connected cluster of
overlapping [entry_ns, exit_ns) intervals drawn from BOTH strategies (a
graph connected-components construction over interval overlap). Whenever
neither strategy has an open position, the account is flat by construction
-- the boundary between two groups is therefore always a genuine flat
instant, never an arbitrary cut through open risk. A period where only one
strategy is ever active reduces to a group of size one, holding exactly
that strategy's own already-known trade -- this is what makes the "U2=0
reproduces Strategy-A standalone" and "A multiplier alone reproduces the
existing standalone benchmark" invariants hold structurally, not by
coincidence (see the module's own tests).

Within one group, the worst combined mark-to-market offset
(``floor_equity_delta``, task section 13: "FTMO breach logic must operate
on total account equity") is evaluated at the union of every candidate
worst-instant available from either strategy:

- every U2 episode's own mark timestamps (``combined_pnl_path()``) --
  EXACT, since U2's real intratrade path is preserved via
  ``relative_value_adapter.py`` unchanged;
- every Strategy-A trade's own entry/exit boundary, at which its own
  scalar ``floor_equity_delta`` (already, per ``path_extraction.py``,
  either EXACT for a stop-exit or a proven never-optimistic BOUND for a
  target-exit) is used for the WHOLE span that trade is open -- since that
  bound already dominates (is <=) every true intratrade mark of that trade
  by construction, using it at any candidate instant within the trade's
  span remains a valid, never-optimistic bound on the true combined
  equity at that instant.

At each candidate instant t, the combined offset is the SUM of each
strategy's own already-USD-converted contribution at t (each stated
account-currency amount, never a raw CAD/CHF quantity -- task section 13)
-- both strategies can simultaneously carry USD/CAD exposure (Strategy A's
own USD/CAD position and U2's own USD/CAD leg), and this construction
never nets that exposure at the instrument level; it only ever sums
already-independently-USD-converted dollar contributions, which is exactly
how the real Nautilus margin account's own total equity is computed.
"""

from __future__ import annotations

import argparse
import bisect
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from decimal import Decimal
from pathlib import Path
from statistics import median
from typing import Any, Literal

import pandas as pd  # type: ignore[import-untyped]

from ftmoquant.prop_rules.loader import load_prop_rule_set
from ftmoquant.prop_rules.models import EvaluationPhase, PropRuleSet
from ftmoquant.research.alpha_lab.b3f1_spread_execution import (
    GROSS_NOTIONAL_USD,
    simulate_b3f1_intents,
)
from ftmoquant.research.alpha_lab.b3f1_spread_signals import (
    compute_formation_series,
    generate_b3f1_decisions,
)
from ftmoquant.research.alpha_lab.b3f1_underpowered_u2_validation import (
    FORMATION_WINDOW as U2_FORMATION_WINDOW,
)
from ftmoquant.research.alpha_lab.b3f1_underpowered_u2_validation import (
    SLEEVE_ID as U2_SLEEVE_ID,
)
from ftmoquant.research.alpha_lab.b3f1_underpowered_u2_validation import (
    X_SPEC as U2_X_SPEC,
)
from ftmoquant.research.alpha_lab.b3f1_underpowered_u2_validation import (
    Y_SPEC as U2_Y_SPEC,
)
from ftmoquant.research.alpha_lab.b3f1_underpowered_u2_validation import (
    Z_ENTRY as U2_Z_ENTRY,
)
from ftmoquant.research.alpha_lab.b3f1_underpowered_u2_validation import (
    Z_STOP as U2_Z_STOP,
)
from ftmoquant.research.alpha_lab.relative_value_adapter import (
    RelativeValueEpisode,
    RelativeValueLeg,
)
from ftmoquant.research.b3f1_u2_execution_promotion import (
    _leg_h1_log_close,
    resolve_leg_roots,
)
from ftmoquant.research.ftmo_pass_probability.artifacts import (
    write_csv_artifact,
    write_json_artifact,
)
from ftmoquant.research.ftmo_pass_probability.bootstrap import (
    ResamplingMethod,
    draw_index_path,
)
from ftmoquant.research.ftmo_pass_probability.path_extraction import (
    DevelopmentTradePath,
    TradeRecord,
    load_development_trade_path,
)
from ftmoquant.research.ftmo_pass_probability.reporting import (
    PolicySummary,
    _percentile,
    summarize_policy,
    summarize_policy_statistics,
)
from ftmoquant.research.ftmo_pass_probability.sizing import (
    SizingFamily,
    SizingPolicy,
    apply_sizing,
)
from ftmoquant.research.ftmo_pass_probability.state_machine import (
    FtmoPathStatus,
    TradeEvent,
    simulate_phase,
)
from ftmoquant.research.mean_reversion_h1_development import Partition
from ftmoquant.research.stage_g import DEVELOPMENT_END_EXCLUSIVE, DEVELOPMENT_START
from ftmoquant.research.statistics import estimate_optimal_block_length

ZERO = Decimal("0")

# Public strategy-identity aliases for downstream screens that reuse this
# module without importing the candidate-definition module independently.
FROZEN_U2_FORMATION_WINDOW = U2_FORMATION_WINDOW
FROZEN_U2_Z_ENTRY = U2_Z_ENTRY
FROZEN_U2_Z_STOP = U2_Z_STOP

STRATEGY_A_IDENTITY = "usdcad_sweep_bos_retest_v1"
STRATEGY_A_DEVELOPMENT_EXECUTION_DIR = Path(
    ".artifacts/usdcad_sweep_bos_retest_v1/development_execution"
)
STRATEGY_A_STANDALONE_BENCHMARK = {
    "sizing_policy_id": "fixed_notional_2_0x",
    "pass_both": 0.74596,
    "median_trading_days_to_pass_both": 101,
    "p90_trading_days_to_pass_both": 194,
    "p95_trading_days_to_pass_both": 222,
}

FTMO_RULES_PATH = Path("config/prop/ftmo_2step_swing_2026-08.yaml")

DEFAULT_OUTPUT_DIR = Path(".artifacts/ftmo_joint_frontier/sweep_bos_plus_u2_v1")

SCREEN_SEED = 20260819
SCREEN_PATH_COUNT = 20_000

INITIAL_CAPITAL = Decimal("100000")


class JointFrontierError(ValueError):
    """Raised on any violation of this joint-frontier framework's contract."""


# ---------------------------------------------------------------------------
# Section 5: the frozen 2-D sizing grid -- 36 policies, never expanded.
# 4.0x is an intentional aggressive boundary point, included now precisely
# so the screen is never post-hoc re-expanded if the speed/risk frontier is
# still improving at 3.5x. Do not extend beyond 4.0x in this screen.
# ---------------------------------------------------------------------------

A_MULTIPLIERS: tuple[Decimal, ...] = (
    Decimal("1.5"),
    Decimal("2.0"),
    Decimal("2.5"),
    Decimal("3.0"),
    Decimal("3.5"),
    Decimal("4.0"),
)
U2_MULTIPLIERS: tuple[Decimal, ...] = (
    Decimal("0.00"),
    Decimal("0.25"),
    Decimal("0.50"),
    Decimal("0.75"),
    Decimal("1.00"),
    Decimal("1.25"),
)


@dataclass(frozen=True, slots=True)
class JointPolicy:
    """One frozen (Strategy-A multiplier, U2 multiplier) portfolio policy."""

    policy_id: str
    a_multiplier: Decimal
    u2_multiplier: Decimal

    @property
    def a_sizing_policy(self) -> SizingPolicy:
        return SizingPolicy(
            f"joint_a_{self.a_multiplier}x",
            SizingFamily.FIXED_NOTIONAL_MULTIPLIER,
            notional_multiplier=self.a_multiplier,
        )

    @property
    def total_gross_multiplier(self) -> Decimal:
        return self.a_multiplier + self.u2_multiplier


def _policy_id(a_multiplier: Decimal, u2_multiplier: Decimal) -> str:
    return f"A{a_multiplier}x_U2{u2_multiplier}x"


JOINT_POLICY_GRID: tuple[JointPolicy, ...] = tuple(
    JointPolicy(_policy_id(a_mult, u2_mult), a_mult, u2_mult)
    for a_mult in A_MULTIPLIERS
    for u2_mult in U2_MULTIPLIERS
)
if len(JOINT_POLICY_GRID) != 36:
    raise JointFrontierError(
        f"expected exactly 36 frozen policies, got {len(JOINT_POLICY_GRID)}"
    )
if len({p.policy_id for p in JOINT_POLICY_GRID}) != 36:
    raise JointFrontierError("policy_ids must be unique")

#: Control policies (task section 6): A alone, U2=0, at every frozen A
#: multiplier (1.5x through the new 4.0x boundary point). A=2.0x/U2=0.00x
#: is the CRITICAL control -- it must reproduce the existing Strategy-A
#: standalone benchmark within Monte Carlo sampling error (proven
#: structurally by test_a_2_0_u2_0_reproduces_standalone_semantics).
CONTROL_POLICY_IDS: tuple[str, ...] = tuple(
    _policy_id(a_mult, Decimal("0.00")) for a_mult in A_MULTIPLIERS
)
CRITICAL_CONTROL_POLICY_ID = _policy_id(Decimal("2.0"), Decimal("0.00"))


# ---------------------------------------------------------------------------
# Section 9: eligibility rule -- frozen before results.
# ---------------------------------------------------------------------------

ELIGIBILITY_PASS_BOTH_GE = 0.70
ELIGIBILITY_MEDIAN_DAYS_LE = 75
ELIGIBILITY_P90_DAYS_LE = 150
ELIGIBILITY_FAIL_DAILY_LOSS_LE = 0.02
ELIGIBILITY_FAIL_MAX_LOSS_LE = 0.25


@dataclass(frozen=True, slots=True)
class JointPolicySummary:
    """Everything a policy's stationary-bootstrap screen row needs beyond
    what :class:`~ftmoquant.research.ftmo_pass_probability.reporting.
    PolicySummary` already reports."""

    policy: JointPolicy
    stationary: PolicySummary
    circular: PolicySummary
    median_trading_days_to_pass_challenge: float | None
    p75_trading_days_to_pass_both: float | None


@dataclass(frozen=True, slots=True)
class EligibilityVerdict:
    policy_id: str
    eligible: bool
    failed_criteria: tuple[str, ...]


def evaluate_eligibility(summary: JointPolicySummary) -> EligibilityVerdict:
    stationary = summary.stationary
    checks: dict[str, bool] = {
        "A_stationary_pass_both_ge_0_70": (
            stationary.pass_both.estimate >= ELIGIBILITY_PASS_BOTH_GE
        ),
        "B_stationary_median_days_le_75": (
            stationary.median_trading_days_to_pass_both is not None
            and stationary.median_trading_days_to_pass_both
            <= ELIGIBILITY_MEDIAN_DAYS_LE
        ),
        "C_stationary_p90_days_le_150": (
            stationary.p90_trading_days_to_pass_both is not None
            and stationary.p90_trading_days_to_pass_both <= ELIGIBILITY_P90_DAYS_LE
        ),
        "D_fail_daily_loss_le_0_02": (
            stationary.fail_daily_loss.estimate <= ELIGIBILITY_FAIL_DAILY_LOSS_LE
        ),
        "E_fail_max_loss_le_0_25": (
            stationary.fail_max_loss.estimate <= ELIGIBILITY_FAIL_MAX_LOSS_LE
        ),
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    return EligibilityVerdict(
        policy_id=summary.policy.policy_id, eligible=not failed, failed_criteria=failed
    )


# ---------------------------------------------------------------------------
# Section 10: selection rule -- exact tie-break ladder, never a weighted score.
# ---------------------------------------------------------------------------


def select_policy(
    summaries: Sequence[JointPolicySummary],
) -> JointPolicySummary | None:
    """Among SPEED_ELIGIBLE policies, apply the exact frozen tie-break
    ladder. Returns ``None`` (never a relaxed-threshold fallback) if no
    policy is eligible -- callers must then report the Pareto frontier
    instead (task section 10)."""

    eligible = [s for s in summaries if evaluate_eligibility(s).eligible]
    if not eligible:
        return None

    def sort_key(
        summary: JointPolicySummary,
    ) -> tuple[float, float, float, float, Decimal, str]:
        stationary = summary.stationary
        median_days = stationary.median_trading_days_to_pass_both
        return (
            median_days if median_days is not None else math.inf,
            -stationary.pass_both.estimate,
            stationary.fail_max_loss.estimate,
            stationary.p95_max_drawdown,
            summary.policy.total_gross_multiplier,
            summary.policy.policy_id,
        )

    return min(eligible, key=sort_key)


# ---------------------------------------------------------------------------
# Section 11: Pareto frontier -- maximize pass_both; minimize median days,
# p90 days, fail_max_loss, p95 drawdown.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParetoObjectives:
    policy_id: str
    pass_both: float
    median_days: float
    p90_days: float
    fail_max_loss: float
    p95_drawdown: float


def _objectives(summary: JointPolicySummary) -> ParetoObjectives | None:
    stationary = summary.stationary
    if (
        stationary.median_trading_days_to_pass_both is None
        or stationary.p90_trading_days_to_pass_both is None
    ):
        return None
    return ParetoObjectives(
        policy_id=summary.policy.policy_id,
        pass_both=stationary.pass_both.estimate,
        median_days=stationary.median_trading_days_to_pass_both,
        p90_days=stationary.p90_trading_days_to_pass_both,
        fail_max_loss=stationary.fail_max_loss.estimate,
        p95_drawdown=stationary.p95_max_drawdown,
    )


def _dominates(a: ParetoObjectives, b: ParetoObjectives) -> bool:
    """``a`` dominates ``b`` iff ``a`` is at least as good on every
    objective and strictly better on at least one."""

    at_least_as_good = (
        a.pass_both >= b.pass_both
        and a.median_days <= b.median_days
        and a.p90_days <= b.p90_days
        and a.fail_max_loss <= b.fail_max_loss
        and a.p95_drawdown <= b.p95_drawdown
    )
    strictly_better = (
        a.pass_both > b.pass_both
        or a.median_days < b.median_days
        or a.p90_days < b.p90_days
        or a.fail_max_loss < b.fail_max_loss
        or a.p95_drawdown < b.p95_drawdown
    )
    return at_least_as_good and strictly_better


def compute_pareto_frontier(
    summaries: Sequence[JointPolicySummary],
) -> tuple[str, ...]:
    """Return the policy_ids of every non-dominated (Pareto-efficient)
    policy, in the order they first appear in ``summaries``."""

    candidates = [o for o in (_objectives(s) for s in summaries) if o is not None]
    frontier = []
    for candidate in candidates:
        if not any(
            other.policy_id != candidate.policy_id and _dominates(other, candidate)
            for other in candidates
        ):
            frontier.append(candidate.policy_id)
    return tuple(frontier)


# ---------------------------------------------------------------------------
# Section 3/4: joint account path construction.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ACandidate:
    kind: Literal["A"]
    entry_ns: int
    exit_ns: int
    trade: TradeRecord


@dataclass(frozen=True, slots=True)
class _U2Candidate:
    kind: Literal["U2"]
    entry_ns: int
    exit_ns: int
    episode: RelativeValueEpisode


_Candidate = _ACandidate | _U2Candidate


@dataclass(frozen=True, slots=True)
class JointGroup:
    """One maximal, transitively-overlapping cluster of Strategy-A trades
    and U2 episodes -- the joint account's own atomic resampling unit
    (task section 4). Group boundaries are always genuine flat instants:
    neither strategy holds an open position immediately before a group
    starts or immediately after it ends."""

    entry_ns: int
    exit_ns: int
    a_trades: tuple[TradeRecord, ...]
    u2_episodes: tuple[RelativeValueEpisode, ...]

    @property
    def a_trade_count(self) -> int:
        return len(self.a_trades)

    @property
    def u2_episode_count(self) -> int:
        return len(self.u2_episodes)


def build_joint_groups(
    a_trades: Sequence[TradeRecord], u2_episodes: Sequence[RelativeValueEpisode]
) -> tuple[JointGroup, ...]:
    """Merge both strategies' chronological trade sequences into
    transitively-overlapping :class:`JointGroup` clusters (a standard
    interval-merge / union-find over BOTH strategies' [entry_ns, exit_ns)
    spans together, never independently per strategy)."""

    candidates: list[_Candidate] = [
        _ACandidate("A", trade.entry_ns, trade.exit_ns, trade) for trade in a_trades
    ] + [
        _U2Candidate("U2", episode.entry_ns, episode.exit_ns, episode)
        for episode in u2_episodes
    ]
    if not candidates:
        raise JointFrontierError("at least one trade/episode is required")
    ordered = sorted(candidates, key=lambda c: c.entry_ns)

    groups: list[JointGroup] = []
    current_a: list[TradeRecord] = []
    current_u2: list[RelativeValueEpisode] = []
    current_start: int | None = None
    current_end: int | None = None

    def _flush() -> None:
        nonlocal current_a, current_u2, current_start, current_end
        if current_start is None or current_end is None:
            return
        groups.append(
            JointGroup(
                entry_ns=current_start,
                exit_ns=current_end,
                a_trades=tuple(current_a),
                u2_episodes=tuple(current_u2),
            )
        )
        current_a = []
        current_u2 = []
        current_start = None
        current_end = None

    for candidate in ordered:
        if current_end is not None and candidate.entry_ns < current_end:
            # overlaps the current cluster -- extend it.
            current_end = max(current_end, candidate.exit_ns)
        else:
            _flush()
            current_start = candidate.entry_ns
            current_end = candidate.exit_ns
        if candidate.kind == "A":
            current_a.append(candidate.trade)
        else:
            current_u2.append(candidate.episode)
    _flush()
    return tuple(groups)


def _candidate_instants(group: JointGroup) -> tuple[int, ...]:
    instants: set[int] = {group.entry_ns, group.exit_ns}
    for trade in group.a_trades:
        instants.add(trade.entry_ns)
        instants.add(trade.exit_ns)
    for episode in group.u2_episodes:
        for ts_ns, _price in episode.combined_pnl_path():
            instants.add(ts_ns)
    return tuple(sorted(instants))


def _a_component_at(
    ts_ns: int,
    a_trades: Sequence[TradeRecord],
    a_multiplier: Decimal,
    sized_by_index: dict[int, tuple[Decimal, Decimal]],
) -> Decimal:
    """Realized-so-far (all fully-closed A trades) plus the currently-open
    A trade's own scaled floor bound (a never-optimistic value at ANY
    instant within that trade's own span -- see module docstring)."""

    total = ZERO
    for index, trade in enumerate(a_trades):
        floor_delta, realized = sized_by_index[index]
        if trade.exit_ns <= ts_ns:
            total += realized
        elif trade.entry_ns <= ts_ns:
            total += floor_delta
    return total


def _u2_component_at(
    ts_ns: int,
    u2_episodes: Sequence[RelativeValueEpisode],
    u2_multiplier: Decimal,
) -> Decimal:
    total = ZERO
    for episode in u2_episodes:
        if episode.exit_ns <= ts_ns:
            total += u2_multiplier * episode.realized_pnl()
        elif episode.entry_ns <= ts_ns:
            total += u2_multiplier * episode.combined_pnl_usd_at(ts_ns)
    return total


def scale_joint_group(
    group: JointGroup, *, a_multiplier: Decimal, u2_multiplier: Decimal
) -> TradeEvent:
    """Linearly scale ``group`` by each strategy's own frozen multiplier
    and reduce it to exactly one :class:`TradeEvent` -- ready, unmodified,
    for :func:`state_machine.simulate_phase` (task section 14: the
    multiplier scales the intratrade floor/legging path, not merely the
    terminal P&L)."""

    sized_by_index: dict[int, tuple[Decimal, Decimal]] = {}
    for index, trade in enumerate(group.a_trades):
        if a_multiplier == ZERO:
            sized_by_index[index] = (ZERO, ZERO)
            continue
        policy = SizingPolicy(
            f"joint_a_{a_multiplier}x",
            SizingFamily.FIXED_NOTIONAL_MULTIPLIER,
            notional_multiplier=a_multiplier,
        )
        sized = apply_sizing(policy, trade, INITIAL_CAPITAL)
        sized_by_index[index] = (sized.floor_equity_delta, sized.realized_pnl)

    instants = _candidate_instants(group)
    combined_at = {
        ts_ns: (
            _a_component_at(ts_ns, group.a_trades, a_multiplier, sized_by_index)
            + _u2_component_at(ts_ns, group.u2_episodes, u2_multiplier)
        )
        for ts_ns in instants
    }
    worst = min(combined_at.values())
    total_realized = combined_at[group.exit_ns]
    floor_equity_delta = min(worst, ZERO)
    if floor_equity_delta == ZERO and total_realized < ZERO:
        # the group's own terminal realized loss is itself a valid lower
        # bound on the worst intratrade mark (equity cannot be higher at
        # the close of a losing period than the true minimum along the way).
        floor_equity_delta = total_realized
    return TradeEvent(
        entry_ns=group.entry_ns,
        exit_ns=group.exit_ns,
        floor_equity_delta=floor_equity_delta,
        realized_pnl=total_realized,
    )


@dataclass(frozen=True, slots=True)
class PreparedJointGroupScaling:
    """Policy-independent inputs for exact joint-group scaling.

    U2 mark lookup is the expensive part of ``scale_joint_group``.  These
    values depend only on the immutable DEVELOPMENT episode paths, so they
    are computed once and then multiplied in the original episode order for
    every policy.  The final policy-specific reduction still constructs the
    canonical :class:`TradeEvent` consumed by ``simulate_phase``.
    """

    group: JointGroup
    candidate_instants: tuple[int, ...]
    u2_components_by_instant: tuple[tuple[Decimal, ...], ...]


def _leg_contribution_at_bisect(
    ts_ns: int,
    leg: RelativeValueLeg,
    mark_timestamps: tuple[int, ...] | None = None,
) -> Decimal:
    """Exact ``RelativeValueLeg.contribution_at`` lookup without a linear scan."""

    if ts_ns < leg.entry_ns:
        return ZERO
    if mark_timestamps is None:
        mark_timestamps = tuple(mark.ts_ns for mark in leg.marks)
    index = bisect.bisect_right(mark_timestamps, ts_ns) - 1
    return leg.pnl_usd_at(leg.marks[index].price)


def _episode_component_at_bisect(
    ts_ns: int, episode: RelativeValueEpisode
) -> Decimal:
    return _leg_contribution_at_bisect(
        ts_ns, episode.leg_a
    ) + _leg_contribution_at_bisect(ts_ns, episode.leg_b)


def prepare_joint_group_scaling(group: JointGroup) -> PreparedJointGroupScaling:
    """Prepare exact policy-independent U2 components for one joint group."""

    instants: set[int] = {group.entry_ns, group.exit_ns}
    for trade in group.a_trades:
        instants.add(trade.entry_ns)
        instants.add(trade.exit_ns)
    for episode in group.u2_episodes:
        instants.update(mark.ts_ns for mark in episode.leg_a.marks)
        instants.update(mark.ts_ns for mark in episode.leg_b.marks)
    candidate_instants = tuple(sorted(instants))
    episode_lookups = tuple(
        (
            episode,
            tuple(mark.ts_ns for mark in episode.leg_a.marks),
            tuple(mark.ts_ns for mark in episode.leg_b.marks),
        )
        for episode in group.u2_episodes
    )

    def components_at(ts_ns: int) -> tuple[Decimal, ...]:
        return tuple(
            _leg_contribution_at_bisect(ts_ns, episode.leg_a, leg_a_timestamps)
            + _leg_contribution_at_bisect(ts_ns, episode.leg_b, leg_b_timestamps)
            for episode, leg_a_timestamps, leg_b_timestamps in episode_lookups
        )

    return PreparedJointGroupScaling(
        group=group,
        candidate_instants=candidate_instants,
        u2_components_by_instant=tuple(
            components_at(ts_ns) for ts_ns in candidate_instants
        ),
    )


def prepare_joint_groups_scaling(
    groups: tuple[JointGroup, ...],
) -> tuple[PreparedJointGroupScaling, ...]:
    return tuple(prepare_joint_group_scaling(group) for group in groups)


def scale_prepared_joint_group(
    prepared: PreparedJointGroupScaling,
    *,
    a_multiplier: Decimal,
    u2_multiplier: Decimal,
) -> TradeEvent:
    """Scale a prepared group with operation ordering matching the oracle."""

    group = prepared.group
    sized_by_index: dict[int, tuple[Decimal, Decimal]] = {}
    for index, trade in enumerate(group.a_trades):
        if a_multiplier == ZERO:
            sized_by_index[index] = (ZERO, ZERO)
            continue
        policy = SizingPolicy(
            f"joint_a_{a_multiplier}x",
            SizingFamily.FIXED_NOTIONAL_MULTIPLIER,
            notional_multiplier=a_multiplier,
        )
        sized = apply_sizing(policy, trade, INITIAL_CAPITAL)
        sized_by_index[index] = (sized.floor_equity_delta, sized.realized_pnl)

    combined_at: dict[int, Decimal] = {}
    for ts_ns, u2_components in zip(
        prepared.candidate_instants,
        prepared.u2_components_by_instant,
        strict=True,
    ):
        u2_total = ZERO
        for component in u2_components:
            u2_total += u2_multiplier * component
        combined_at[ts_ns] = _a_component_at(
            ts_ns, group.a_trades, a_multiplier, sized_by_index
        ) + u2_total

    worst = min(combined_at.values())
    total_realized = combined_at[group.exit_ns]
    floor_equity_delta = min(worst, ZERO)
    if floor_equity_delta == ZERO and total_realized < ZERO:
        floor_equity_delta = total_realized
    return TradeEvent(
        entry_ns=group.entry_ns,
        exit_ns=group.exit_ns,
        floor_equity_delta=floor_equity_delta,
        realized_pnl=total_realized,
    )


def precompute_prepared_scaled_events(
    prepared_groups: tuple[PreparedJointGroupScaling, ...],
    *,
    a_multiplier: Decimal,
    u2_multiplier: Decimal,
) -> tuple[TradeEvent, ...]:
    return tuple(
        scale_prepared_joint_group(
            prepared,
            a_multiplier=a_multiplier,
            u2_multiplier=u2_multiplier,
        )
        for prepared in prepared_groups
    )


# ---------------------------------------------------------------------------
# Section 15: block-length derivation, DEVELOPMENT joint path only.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrozenJointBlockLength:
    stationary_block_length: float
    circular_block_length: float
    frozen_block_size: int
    observation_count: int
    lag1_autocorrelation: float | None
    strategy_a_standalone_block_size: int


def derive_frozen_joint_block_length(
    groups: tuple[JointGroup, ...], *, strategy_a_standalone_block_size: int
) -> FrozenJointBlockLength:
    """Estimate and freeze the joint-path block length from DEVELOPMENT
    joint groups only, at the reference (1x/1x) sizing -- never VALIDATION.
    """

    if len(groups) < 2:
        raise JointFrontierError(
            "block-length estimation requires at least two joint groups"
        )
    reference_pnls = [
        float(
            scale_joint_group(
                group, a_multiplier=Decimal("1.0"), u2_multiplier=Decimal("1.0")
            ).realized_pnl
        )
        for group in groups
    ]
    series = pd.Series(reference_pnls, name="joint_path_reference_pnl")
    result = estimate_optimal_block_length(series)
    frozen_block_size = max(1, round(result.stationary))
    autocorrelation = float(series.autocorr(lag=1)) if len(series) > 2 else None
    return FrozenJointBlockLength(
        stationary_block_length=result.stationary,
        circular_block_length=result.circular,
        frozen_block_size=frozen_block_size,
        observation_count=result.observation_count,
        lag1_autocorrelation=autocorrelation,
        strategy_a_standalone_block_size=strategy_a_standalone_block_size,
    )


# ---------------------------------------------------------------------------
# Timing + Monte Carlo replay -- mirrors monte_carlo.py's own precompute/
# placement/replay separation exactly, generalized from single trades to
# joint groups.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class JointGroupTiming:
    gap_before_ns: int
    holding_ns: int


def precompute_joint_group_timing(
    groups: tuple[JointGroup, ...],
) -> tuple[JointGroupTiming, ...]:
    timing: list[JointGroupTiming] = []
    previous_exit_ns: int | None = None
    for group in groups:
        gap_before_ns = (
            0 if previous_exit_ns is None else group.entry_ns - previous_exit_ns
        )
        timing.append(
            JointGroupTiming(
                gap_before_ns=max(gap_before_ns, 0),
                holding_ns=group.exit_ns - group.entry_ns,
            )
        )
        previous_exit_ns = group.exit_ns
    return tuple(timing)


def _estimate_required_group_count(
    timing: tuple[JointGroupTiming, ...],
    horizon_ns: int,
    safety_factor: Decimal = Decimal("2"),
) -> int:
    if not timing:
        raise JointFrontierError("timing must not be empty")
    average_cycle_ns = sum(t.gap_before_ns + t.holding_ns for t in timing) / len(timing)
    if average_cycle_ns <= 0:
        average_cycle_ns = 1.0
    return max(
        1, math.ceil(Decimal(horizon_ns) / Decimal(average_cycle_ns) * safety_factor)
    )


def precompute_scaled_events(
    groups: tuple[JointGroup, ...], *, a_multiplier: Decimal, u2_multiplier: Decimal
) -> tuple[TradeEvent, ...]:
    """Scale EVERY group exactly once for one (a_multiplier, u2_multiplier)
    policy -- ``scale_joint_group`` is a pure function of ``(group, a_mult,
    u2_mult)`` only, so recomputing it separately for every placement in
    every one of 20,000 resampled paths (as a naive per-path loop would)
    repeats identical work potentially tens of thousands of times per
    group. Precomputing once per policy and reusing the result across
    every path is the single highest-value performance optimization here
    (task section 20) -- it changes nothing about which floor/realized
    values are produced, only how many times each is computed."""

    return tuple(
        scale_joint_group(group, a_multiplier=a_multiplier, u2_multiplier=u2_multiplier)
        for group in groups
    )


def _build_synthetic_group_placements(
    scaled_events: tuple[TradeEvent, ...],
    timing: tuple[JointGroupTiming, ...],
    index_path: Sequence[int],
    *,
    horizon_ns: int,
) -> tuple[tuple[TradeEvent, int, int], ...]:
    placements: list[tuple[TradeEvent, int, int]] = []
    clock_ns = 0
    for index in index_path:
        clock_ns += timing[index].gap_before_ns
        if clock_ns >= horizon_ns:
            break
        entry_ns = clock_ns
        exit_ns = clock_ns + timing[index].holding_ns
        placements.append((scaled_events[index], entry_ns, exit_ns))
        clock_ns = exit_ns
    return tuple(placements)


@dataclass(frozen=True, slots=True)
class JointTwoPhaseOutcome:
    challenge: Any
    verification: Any
    passed_both: bool


def _censor_if_active(outcome: Any) -> Any:
    if outcome.status is FtmoPathStatus.ACTIVE:
        return replace(outcome, status=FtmoPathStatus.CENSORED_NOT_PASSED)
    return outcome


def simulate_two_phase_joint_path(
    groups: tuple[JointGroup, ...],
    timing: tuple[JointGroupTiming, ...],
    *,
    method: ResamplingMethod,
    block_size: int,
    policy: JointPolicy,
    rules: PropRuleSet,
    initial_capital: Decimal,
    challenge_horizon_ns: int,
    verification_horizon_ns: int,
    seed: int,
    scaled_events: tuple[TradeEvent, ...] | None = None,
) -> JointTwoPhaseOutcome:
    """Joint-path analogue of ``monte_carlo.simulate_two_phase_path``: draw
    ONE resampled index path into the JOINT group sequence (never two
    independently-shuffled per-strategy paths -- task section 4), place
    the resampled groups on a synthetic clock using each group's own
    historical pacing, scale by ``policy``'s two frozen multipliers, and
    replay through the UNCHANGED ``state_machine.simulate_phase``.

    ``scaled_events`` may be supplied pre-computed (see
    :func:`precompute_scaled_events`) so a full Monte Carlo screen never
    repeats identical ``scale_joint_group`` work across paths; when omitted
    (e.g. a single ad-hoc call) it is computed once here instead."""

    if scaled_events is None:
        scaled_events = precompute_scaled_events(
            groups, a_multiplier=policy.a_multiplier, u2_multiplier=policy.u2_multiplier
        )

    required = _estimate_required_group_count(
        timing, challenge_horizon_ns + verification_horizon_ns
    )
    index_path = draw_index_path(
        len(groups),
        method=method,
        block_size=block_size,
        seed=seed,
        min_length=required,
    )

    challenge_placements = _build_synthetic_group_placements(
        scaled_events, timing, index_path, horizon_ns=challenge_horizon_ns
    )
    challenge_events = _retime(challenge_placements)
    challenge_outcome = _censor_if_active(
        simulate_phase(
            challenge_events,
            rules=rules,
            phase=EvaluationPhase.CHALLENGE,
            initial_capital=initial_capital,
            horizon_ns=challenge_horizon_ns,
        )
    )
    if challenge_outcome.status is not FtmoPathStatus.PASSED:
        return JointTwoPhaseOutcome(
            challenge=challenge_outcome, verification=None, passed_both=False
        )

    remaining_index_path = index_path[challenge_outcome.trades_replayed :]
    verification_placements = _build_synthetic_group_placements(
        scaled_events, timing, remaining_index_path, horizon_ns=verification_horizon_ns
    )
    verification_events = _retime(verification_placements)
    verification_outcome = _censor_if_active(
        simulate_phase(
            verification_events,
            rules=rules,
            phase=EvaluationPhase.VERIFICATION,
            initial_capital=initial_capital,
            horizon_ns=verification_horizon_ns,
        )
    )
    return JointTwoPhaseOutcome(
        challenge=challenge_outcome,
        verification=verification_outcome,
        passed_both=verification_outcome.status is FtmoPathStatus.PASSED,
    )


def _retime(
    placements: tuple[tuple[TradeEvent, int, int], ...],
) -> tuple[TradeEvent, ...]:
    """``scale_joint_group``/``precompute_scaled_events`` preserve each
    group's OWN original entry/exit timestamps; the synthetic clock
    instead needs the PLACED (entry_ns, exit_ns) -- rewritten here without
    touching the already-computed floor/realized figures."""

    return tuple(
        replace(event, entry_ns=entry_ns, exit_ns=exit_ns)
        for event, entry_ns, exit_ns in placements
    )


# ---------------------------------------------------------------------------
# Data loading -- Strategy A reused unchanged; U2 loaded fresh (with real
# intratrade marks) via the frozen b3f1_u2_execution_promotion pipeline.
# ---------------------------------------------------------------------------


def load_strategy_a_development(
    execution_dir: Path = STRATEGY_A_DEVELOPMENT_EXECUTION_DIR,
) -> DevelopmentTradePath:
    return load_development_trade_path(execution_dir)


def load_u2_development_episodes(
    *, catalog_root: Path, universe_readiness_path: Path
) -> tuple[RelativeValueEpisode, ...]:
    """Load U2's frozen DEVELOPMENT episode sequence -- WITH real intratrade
    marks preserved (unlike ``b3f1_u2_execution_promotion``'s own flattened
    ``trades.csv``) -- by calling that module's exact frozen signal/
    execution pipeline pieces unchanged. DEVELOPMENT partition only; never
    accepts a partition argument, structurally incapable of reading
    VALIDATION or holdout."""

    from ftmoquant.research.alpha_lab.wick_fvg_squeeze_execution import load_m1_bidask

    leg_roots = resolve_leg_roots(
        partition=Partition.DEVELOPMENT,
        catalog_root=catalog_root,
        universe_readiness_path=universe_readiness_path,
    )
    log_y, log_x = _leg_h1_log_close(
        partition=Partition.DEVELOPMENT,
        catalog_root=catalog_root,
        universe_readiness_path=universe_readiness_path,
    )
    formation = compute_formation_series(log_y, log_x, U2_FORMATION_WINDOW)
    decisions = generate_b3f1_decisions(
        formation,
        log_y,
        log_x,
        sleeve_id=U2_SLEEVE_ID,
        z_entry=U2_Z_ENTRY,
        z_stop=U2_Z_STOP,
    )
    y_bid, y_ask = load_m1_bidask(
        instrument_id=U2_Y_SPEC.instrument_id,
        root=leg_roots["Y"],
        start_utc=DEVELOPMENT_START,
        end_exclusive_utc=DEVELOPMENT_END_EXCLUSIVE,
    )
    x_bid, x_ask = load_m1_bidask(
        instrument_id=U2_X_SPEC.instrument_id,
        root=leg_roots["X"],
        start_utc=DEVELOPMENT_START,
        end_exclusive_utc=DEVELOPMENT_END_EXCLUSIVE,
    )
    episodes, _skips = simulate_b3f1_intents(
        decisions,
        y_spec=U2_Y_SPEC,
        x_spec=U2_X_SPEC,
        y_bid_m1=y_bid,
        y_ask_m1=y_ask,
        x_bid_m1=x_bid,
        x_ask_m1=x_ask,
        gross_notional_usd=GROSS_NOTIONAL_USD,
        cost_stress_multiplier=Decimal("1"),
    )
    return episodes


# ---------------------------------------------------------------------------
# Section 12: diversification diagnostics -- DEVELOPMENT only, no MC needed.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DiversificationDiagnostics:
    aligned_daily_correlation: float | None
    downside_correlation: float | None
    correlation_conditional_on_a_losing: float | None
    correlation_conditional_on_u2_losing: float | None
    fraction_a_losing_days_with_u2_positive: float | None
    fraction_u2_losing_days_with_a_positive: float | None
    maximum_same_day_combined_loss_usd: str
    overlap_group_count: int
    total_group_count: int
    fraction_a_trades_overlapping_u2: float
    fraction_u2_trades_overlapping_a: float


def _daily_pnl_series(
    a_trades: Sequence[TradeRecord], u2_episodes: Sequence[RelativeValueEpisode]
) -> pd.DataFrame:
    from datetime import UTC, datetime

    a_daily: dict[Any, float] = {}
    for trade in a_trades:
        day = datetime.fromtimestamp(trade.exit_ns / 1_000_000_000, tz=UTC).date()
        a_daily[day] = a_daily.get(day, 0.0) + float(trade.original_realized_pnl)
    u2_daily: dict[Any, float] = {}
    for episode in u2_episodes:
        day = datetime.fromtimestamp(episode.exit_ns / 1_000_000_000, tz=UTC).date()
        u2_daily[day] = u2_daily.get(day, 0.0) + float(episode.realized_pnl())

    all_days = sorted(set(a_daily) | set(u2_daily))
    return pd.DataFrame(
        {
            "a_pnl": [a_daily.get(day, 0.0) for day in all_days],
            "u2_pnl": [u2_daily.get(day, 0.0) for day in all_days],
        },
        index=all_days,
    )


def compute_diversification_diagnostics(
    groups: tuple[JointGroup, ...],
    a_trades: Sequence[TradeRecord],
    u2_episodes: Sequence[RelativeValueEpisode],
) -> DiversificationDiagnostics:
    daily = _daily_pnl_series(a_trades, u2_episodes)
    aligned_corr = (
        float(daily["a_pnl"].corr(daily["u2_pnl"])) if len(daily) >= 3 else None
    )
    downside = daily[(daily["a_pnl"] < 0) | (daily["u2_pnl"] < 0)]
    downside_corr = (
        float(downside["a_pnl"].corr(downside["u2_pnl"]))
        if len(downside) >= 3
        else None
    )
    a_losing = daily[daily["a_pnl"] < 0]
    u2_losing = daily[daily["u2_pnl"] < 0]
    corr_given_a_losing = (
        float(a_losing["a_pnl"].corr(a_losing["u2_pnl"]))
        if len(a_losing) >= 3
        else None
    )
    corr_given_u2_losing = (
        float(u2_losing["a_pnl"].corr(u2_losing["u2_pnl"]))
        if len(u2_losing) >= 3
        else None
    )
    fraction_a_losing_u2_positive = (
        float((a_losing["u2_pnl"] > 0).mean()) if len(a_losing) > 0 else None
    )
    fraction_u2_losing_a_positive = (
        float((u2_losing["a_pnl"] > 0).mean()) if len(u2_losing) > 0 else None
    )
    max_combined_loss = (
        float((daily["a_pnl"] + daily["u2_pnl"]).min()) if len(daily) > 0 else 0.0
    )

    overlap_groups = [
        g for g in groups if g.a_trade_count > 0 and g.u2_episode_count > 0
    ]
    a_trade_ids_overlapping = {
        id(trade) for g in overlap_groups for trade in g.a_trades
    }
    u2_episode_ids_overlapping = {
        id(episode) for g in overlap_groups for episode in g.u2_episodes
    }
    return DiversificationDiagnostics(
        aligned_daily_correlation=aligned_corr,
        downside_correlation=downside_corr,
        correlation_conditional_on_a_losing=corr_given_a_losing,
        correlation_conditional_on_u2_losing=corr_given_u2_losing,
        fraction_a_losing_days_with_u2_positive=fraction_a_losing_u2_positive,
        fraction_u2_losing_days_with_a_positive=fraction_u2_losing_a_positive,
        maximum_same_day_combined_loss_usd=str(max_combined_loss),
        overlap_group_count=len(overlap_groups),
        total_group_count=len(groups),
        fraction_a_trades_overlapping_u2=(
            len(a_trade_ids_overlapping) / len(a_trades) if a_trades else 0.0
        ),
        fraction_u2_trades_overlapping_a=(
            len(u2_episode_ids_overlapping) / len(u2_episodes) if u2_episodes else 0.0
        ),
    )


# ---------------------------------------------------------------------------
# Additional speed metrics reporting.PolicySummary does not already carry
# (median days to pass Challenge alone; p75 days to pass both) -- computed
# directly from the same outcome list, reusing reporting._percentile.
# ---------------------------------------------------------------------------


def _extra_speed_metrics(
    outcomes: tuple[JointTwoPhaseOutcome, ...],
) -> tuple[float | None, float | None]:
    challenge_days = [
        o.challenge.trading_days
        for o in outcomes
        if o.challenge.status.value == "passed"
    ]
    both_days = [
        o.challenge.trading_days
        + (o.verification.trading_days if o.verification else 0)
        for o in outcomes
        if o.passed_both
    ]
    median_challenge = median(challenge_days) if challenge_days else None
    p75_both = _percentile(both_days, 0.75) if both_days else None
    return median_challenge, p75_both


@dataclass(frozen=True, slots=True)
class JointMethodRun:
    """One policy evaluated under one existing joint bootstrap method.

    This is the method-level unit shared by the coarse/refinement screens and
    the later stationary-only precision stage.  Keeping the draw/replay loop
    here prevents downstream stages from implementing a second Monte Carlo
    engine merely to select one of the already-supported methods.
    """

    summary: PolicySummary
    median_trading_days_to_pass_challenge: float | None
    p75_trading_days_to_pass_both: float | None
    p90_max_drawdown: float


def run_joint_policy_method(
    *,
    groups: tuple[JointGroup, ...],
    rules: PropRuleSet,
    block_size: int,
    policy: JointPolicy,
    method: ResamplingMethod,
    path_count: int,
    seed: int,
    challenge_horizon_ns: int,
    verification_horizon_ns: int,
    timing: tuple[JointGroupTiming, ...] | None = None,
    scaled_events: tuple[TradeEvent, ...] | None = None,
) -> JointMethodRun:
    """Run one policy/method through the canonical joint replay machinery."""

    if timing is None:
        timing = precompute_joint_group_timing(groups)
    if scaled_events is None:
        scaled_events = precompute_scaled_events(
            groups,
            a_multiplier=policy.a_multiplier,
            u2_multiplier=policy.u2_multiplier,
        )
    outcomes = tuple(
        simulate_two_phase_joint_path(
            groups,
            timing,
            method=method,
            block_size=block_size,
            policy=policy,
            rules=rules,
            initial_capital=INITIAL_CAPITAL,
            challenge_horizon_ns=challenge_horizon_ns,
            verification_horizon_ns=verification_horizon_ns,
            seed=seed + replication,
            scaled_events=scaled_events,
        )
        for replication in range(path_count)
    )
    median_challenge_days, p75_both_days = _extra_speed_metrics(outcomes)
    max_drawdowns = [
        max(
            float(outcome.challenge.max_drawdown),
            float(outcome.verification.max_drawdown)
            if outcome.verification is not None
            else 0.0,
        )
        for outcome in outcomes
    ]
    return JointMethodRun(
        summary=summarize_policy(
            policy.policy_id,
            method,
            outcomes,  # type: ignore[arg-type]
        ),
        median_trading_days_to_pass_challenge=median_challenge_days,
        p75_trading_days_to_pass_both=p75_both_days,
        p90_max_drawdown=_percentile(max_drawdowns, 0.90),
    )


def run_joint_policy_method_streaming(
    *,
    groups: tuple[JointGroup, ...],
    rules: PropRuleSet,
    block_size: int,
    policy: JointPolicy,
    method: ResamplingMethod,
    path_count: int,
    seed: int,
    challenge_horizon_ns: int,
    verification_horizon_ns: int,
    timing: tuple[JointGroupTiming, ...] | None = None,
    scaled_events: tuple[TradeEvent, ...] | None = None,
) -> JointMethodRun:
    """Canonical replay with streamed counters and minimal percentile arrays."""

    if timing is None:
        timing = precompute_joint_group_timing(groups)
    if scaled_events is None:
        scaled_events = precompute_scaled_events(
            groups,
            a_multiplier=policy.a_multiplier,
            u2_multiplier=policy.u2_multiplier,
        )

    pass_challenge_count = 0
    reached_verification_count = 0
    pass_verification_count = 0
    pass_both_count = 0
    fail_daily_loss_count = 0
    fail_max_loss_count = 0
    censored_count = 0
    challenge_days: list[int] = []
    both_days: list[int] = []
    max_drawdowns: list[float] = []

    for replication in range(path_count):
        outcome = simulate_two_phase_joint_path(
            groups,
            timing,
            method=method,
            block_size=block_size,
            policy=policy,
            rules=rules,
            initial_capital=INITIAL_CAPITAL,
            challenge_horizon_ns=challenge_horizon_ns,
            verification_horizon_ns=verification_horizon_ns,
            seed=seed + replication,
            scaled_events=scaled_events,
        )
        challenge_status = outcome.challenge.status
        verification_status = (
            outcome.verification.status
            if outcome.verification is not None
            else None
        )
        if challenge_status is FtmoPathStatus.PASSED:
            pass_challenge_count += 1
            challenge_days.append(outcome.challenge.trading_days)
        if outcome.verification is not None:
            reached_verification_count += 1
        if verification_status is FtmoPathStatus.PASSED:
            pass_verification_count += 1
        if outcome.passed_both:
            pass_both_count += 1
            both_days.append(
                outcome.challenge.trading_days
                + (
                    outcome.verification.trading_days
                    if outcome.verification is not None
                    else 0
                )
            )
        statuses = (challenge_status, verification_status)
        if FtmoPathStatus.FAILED_DAILY_LOSS in statuses:
            fail_daily_loss_count += 1
        if FtmoPathStatus.FAILED_MAX_LOSS in statuses:
            fail_max_loss_count += 1
        if FtmoPathStatus.CENSORED_NOT_PASSED in statuses:
            censored_count += 1
        max_drawdowns.append(
            max(
                float(outcome.challenge.max_drawdown),
                float(outcome.verification.max_drawdown)
                if outcome.verification is not None
                else 0.0,
            )
        )

    summary = summarize_policy_statistics(
        policy_id=policy.policy_id,
        method=method,
        trials=path_count,
        pass_challenge_count=pass_challenge_count,
        reached_verification_count=reached_verification_count,
        pass_verification_count=pass_verification_count,
        pass_both_count=pass_both_count,
        fail_daily_loss_count=fail_daily_loss_count,
        fail_max_loss_count=fail_max_loss_count,
        censored_count=censored_count,
        trading_days_to_pass_both=both_days,
        max_drawdowns=max_drawdowns,
    )
    return JointMethodRun(
        summary=summary,
        median_trading_days_to_pass_challenge=(
            median(challenge_days) if challenge_days else None
        ),
        p75_trading_days_to_pass_both=(
            _percentile(both_days, 0.75) if both_days else None
        ),
        p90_max_drawdown=_percentile(max_drawdowns, 0.90),
    )


# ---------------------------------------------------------------------------
# Section 18: artifacts (write-once, refused BEFORE any Monte Carlo work).
# ---------------------------------------------------------------------------

EXPECTED_ARTIFACT_FILENAMES: tuple[str, ...] = (
    "joint_path_diagnostics.json",
    "joint_sizing_grid.json",
    "joint_sizing_screen.csv",
    "pareto_frontier.csv",
    "selection_summary.json",
    "diversification_diagnostics.json",
)


def reserve_output_directory(output_dir: Path) -> None:
    """Fail closed BEFORE any Monte Carlo path is drawn."""

    if output_dir.exists():
        raise JointFrontierError(f"{output_dir} already exists; refusing to overwrite")
    for filename in EXPECTED_ARTIFACT_FILENAMES:
        path = output_dir / filename
        if path.exists():
            raise JointFrontierError(f"{path} already exists; refusing to overwrite")


def write_joint_sizing_grid(output_dir: Path) -> None:
    write_json_artifact(
        output_dir / "joint_sizing_grid.json",
        {
            "a_multipliers": [str(m) for m in A_MULTIPLIERS],
            "u2_multipliers": [str(m) for m in U2_MULTIPLIERS],
            "policy_count": len(JOINT_POLICY_GRID),
            "policies": [asdict(p) for p in JOINT_POLICY_GRID],
            "control_policy_ids": list(CONTROL_POLICY_IDS),
            "critical_control_policy_id": CRITICAL_CONTROL_POLICY_ID,
            "eligibility_rule": {
                "A_pass_both_ge": ELIGIBILITY_PASS_BOTH_GE,
                "B_median_days_le": ELIGIBILITY_MEDIAN_DAYS_LE,
                "C_p90_days_le": ELIGIBILITY_P90_DAYS_LE,
                "D_fail_daily_loss_le": ELIGIBILITY_FAIL_DAILY_LOSS_LE,
                "E_fail_max_loss_le": ELIGIBILITY_FAIL_MAX_LOSS_LE,
            },
        },
    )


# ---------------------------------------------------------------------------
# Section 16/precision-stage command builder -- implemented, never executed.
# ---------------------------------------------------------------------------


def build_precision_commands(selected_policy_id: str | None) -> tuple[str, ...]:
    commands = [
        (
            "ftmoquant-joint-frontier-precision --policy "
            f"{CRITICAL_CONTROL_POLICY_ID} --paths 100000 --seed "
            f"{SCREEN_SEED} --method stationary"
        )
    ]
    if selected_policy_id is not None:
        commands.append(
            "ftmoquant-joint-frontier-precision --policy "
            f"{selected_policy_id} --paths 100000 --seed {SCREEN_SEED} "
            "--method stationary"
        )
    return tuple(commands)


# ---------------------------------------------------------------------------
# CLI -- builds the framework end to end; the real 20,000-path screen is a
# separate, explicit invocation this function documents but never executes.
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Joint Strategy-A + U2 FTMO sizing-frontier DEVELOPMENT screen. "
            "Frozen 36-policy grid, frozen seed "
            f"{SCREEN_SEED}, frozen {SCREEN_PATH_COUNT}-path count -- none "
            "overridable by flag."
        )
    )
    parser.add_argument("--catalog-root", type=Path, required=True)
    parser.add_argument("--universe-readiness", type=Path, required=True)
    parser.add_argument(
        "--strategy-a-execution-dir",
        type=Path,
        default=STRATEGY_A_DEVELOPMENT_EXECUTION_DIR,
    )
    parser.add_argument("--ftmo-rules", type=Path, default=FTMO_RULES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


DEFAULT_HORIZON_NS = 3 * 365 * 24 * 60 * 60 * 1_000_000_000
_METHODS: tuple[ResamplingMethod, ...] = ("stationary", "circular")


def run_joint_sizing_screen(
    *,
    groups: tuple[JointGroup, ...],
    rules: PropRuleSet,
    block_size: int,
    path_count: int = SCREEN_PATH_COUNT,
    seed: int = SCREEN_SEED,
    challenge_horizon_ns: int = DEFAULT_HORIZON_NS,
    verification_horizon_ns: int = DEFAULT_HORIZON_NS,
    policies: Sequence[JointPolicy] = JOINT_POLICY_GRID,
) -> tuple[JointPolicySummary, ...]:
    """Run the supplied frozen policies under both bootstrap methods. This is the
    expensive step the real 20,000-path DEVELOPMENT screen invokes -- never
    called with a large ``path_count`` inside Codex (see the module
    docstring and ``build_parser``). ``policies`` defaults to the original
    coarse 36-policy grid; the post-screen refinement passes its separately
    frozen 12-policy tuple through this same underlying engine."""

    timing = precompute_joint_group_timing(groups)
    summaries: list[JointPolicySummary] = []
    for policy in policies:
        scaled_events = precompute_scaled_events(
            groups,
            a_multiplier=policy.a_multiplier,
            u2_multiplier=policy.u2_multiplier,
        )
        method_runs = {
            method: run_joint_policy_method(
                groups=groups,
                rules=rules,
                block_size=block_size,
                policy=policy,
                method=method,
                path_count=path_count,
                seed=seed,
                challenge_horizon_ns=challenge_horizon_ns,
                verification_horizon_ns=verification_horizon_ns,
                timing=timing,
                scaled_events=scaled_events,
            )
            for method in _METHODS
        }
        stationary_run = method_runs["stationary"]
        circular_run = method_runs["circular"]
        summaries.append(
            JointPolicySummary(
                policy=policy,
                stationary=stationary_run.summary,
                circular=circular_run.summary,
                median_trading_days_to_pass_challenge=(
                    stationary_run.median_trading_days_to_pass_challenge
                ),
                p75_trading_days_to_pass_both=(
                    stationary_run.p75_trading_days_to_pass_both
                ),
            )
        )
    return tuple(summaries)


def write_joint_sizing_screen_csv(
    output_dir: Path, summaries: tuple[JointPolicySummary, ...]
) -> None:
    rows = []
    for summary in summaries:
        for label, ps in (
            ("stationary", summary.stationary),
            ("circular", summary.circular),
        ):
            rows.append(
                {
                    "policy_id": summary.policy.policy_id,
                    "a_multiplier": str(summary.policy.a_multiplier),
                    "u2_multiplier": str(summary.policy.u2_multiplier),
                    "method": label,
                    "pass_challenge": ps.pass_challenge.estimate,
                    "pass_both": ps.pass_both.estimate,
                    "fail_daily_loss": ps.fail_daily_loss.estimate,
                    "fail_max_loss": ps.fail_max_loss.estimate,
                    "censoring_rate": ps.censoring_rate.estimate,
                    "median_trading_days_to_pass_challenge": (
                        summary.median_trading_days_to_pass_challenge
                        if label == "stationary"
                        else ""
                    ),
                    "median_trading_days_to_pass_both": (
                        ps.median_trading_days_to_pass_both
                    ),
                    "p75_trading_days_to_pass_both": (
                        summary.p75_trading_days_to_pass_both
                        if label == "stationary"
                        else ""
                    ),
                    "p90_trading_days_to_pass_both": ps.p90_trading_days_to_pass_both,
                    "p95_trading_days_to_pass_both": ps.p95_trading_days_to_pass_both,
                    "median_max_drawdown": ps.median_max_drawdown,
                    "p95_max_drawdown": ps.p95_max_drawdown,
                }
            )
    write_csv_artifact(
        output_dir / "joint_sizing_screen.csv", list(rows[0].keys()), rows
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    reserve_output_directory(args.output)

    a_path = load_strategy_a_development(args.strategy_a_execution_dir)
    u2_episodes = load_u2_development_episodes(
        catalog_root=args.catalog_root, universe_readiness_path=args.universe_readiness
    )
    groups = build_joint_groups(a_path.trades, u2_episodes)
    rules = load_prop_rule_set(args.ftmo_rules)
    block_length = derive_frozen_joint_block_length(
        groups, strategy_a_standalone_block_size=1
    )

    write_joint_sizing_grid(args.output)
    write_json_artifact(
        args.output / "joint_path_diagnostics.json",
        {
            "joint_group_count": len(groups),
            "a_trade_count": len(a_path.trades),
            "u2_episode_count": len(u2_episodes),
            "block_length": asdict(block_length),
        },
    )
    diagnostics = compute_diversification_diagnostics(
        groups, a_path.trades, u2_episodes
    )
    write_json_artifact(
        args.output / "diversification_diagnostics.json", asdict(diagnostics)
    )

    summaries = run_joint_sizing_screen(
        groups=groups, rules=rules, block_size=block_length.frozen_block_size
    )
    write_joint_sizing_screen_csv(args.output, summaries)

    pareto_ids = compute_pareto_frontier(summaries)
    write_csv_artifact(
        args.output / "pareto_frontier.csv",
        ["policy_id"],
        [{"policy_id": policy_id} for policy_id in pareto_ids],
    )

    selected = select_policy(summaries)
    write_json_artifact(
        args.output / "selection_summary.json",
        {
            "selected_policy_id": selected.policy.policy_id if selected else None,
            "pareto_frontier": list(pareto_ids),
            "precision_commands": build_precision_commands(
                selected.policy.policy_id if selected else None
            ),
        },
    )
    print(f"joint sizing screen complete: {len(summaries)} policies evaluated")


if __name__ == "__main__":
    main()
