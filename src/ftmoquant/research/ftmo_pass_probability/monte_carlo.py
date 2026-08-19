"""Monte Carlo orchestration: resampled trade path -> sized events -> FTMO
state-machine outcome.

Sizing is genuinely causal here: because ``sizing.SizingFamily.
FIXED_FRACTIONAL_RISK`` depends on the account balance *at the moment each
trade opens*, and that balance itself depends on every prior trade's sized
outcome on this specific resampled path, sizing cannot be precomputed
independently of path replay (unlike the frozen 1x DEVELOPMENT artifact,
whose fixed 100,000-unit notional never depended on balance).
:func:`size_synthetic_path` therefore walks the resampled sequence once,
causally, before handing the fully-sized events to
:func:`ftmoquant.research.ftmo_pass_probability.state_machine.simulate_phase`
for authoritative breach/pass detection -- state_machine.py itself stays
free of any sizing concept, which is what keeps it independently testable.

Calendar reconstruction: a block-bootstrap resample reorders real trades, so
their original absolute timestamps no longer apply. Each resampled trade
instance reuses *its own* historical gap-before-entry and holding duration
(see :func:`precompute_trade_timing`) chained onto a synthetic clock. This
is a disclosed simplification -- cross-block transitions splice two
real historical gaps that never actually occurred back to back -- but it
preserves each trade's own realistic pacing rather than inventing one, and
keeps local (within-block) calendar structure exactly as observed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from decimal import Decimal

from ftmoquant.prop_rules.models import EvaluationPhase, PropRuleSet
from ftmoquant.research.ftmo_pass_probability.bootstrap import (
    ResamplingMethod,
    draw_index_path,
)
from ftmoquant.research.ftmo_pass_probability.path_extraction import TradeRecord
from ftmoquant.research.ftmo_pass_probability.sizing import SizingPolicy, apply_sizing
from ftmoquant.research.ftmo_pass_probability.state_machine import (
    FtmoPathStatus,
    PhaseOutcome,
    TradeEvent,
    simulate_phase,
)


@dataclass(frozen=True, slots=True)
class TradeTiming:
    """Original-history pacing reused when placing a trade synthetically."""

    gap_before_ns: int
    holding_ns: int


@dataclass(frozen=True, slots=True)
class TwoPhaseOutcome:
    """Sequential Challenge -> Verification outcome on one resampled path."""

    challenge: PhaseOutcome
    verification: PhaseOutcome | None
    passed_both: bool


def precompute_trade_timing(trades: tuple[TradeRecord, ...]) -> tuple[TradeTiming, ...]:
    """Derive each DEVELOPMENT trade's own gap-before and holding duration."""

    timing: list[TradeTiming] = []
    previous_exit_ns: int | None = None
    for trade in trades:
        gap_before_ns = (
            0 if previous_exit_ns is None else trade.entry_ns - previous_exit_ns
        )
        timing.append(
            TradeTiming(
                gap_before_ns=max(gap_before_ns, 0),
                holding_ns=trade.exit_ns - trade.entry_ns,
            )
        )
        previous_exit_ns = trade.exit_ns
    return tuple(timing)


def estimate_required_trade_count(
    timing: tuple[TradeTiming, ...],
    horizon_ns: int,
    safety_factor: Decimal = Decimal("2"),
) -> int:
    """Estimate how many resampled trades are needed to plausibly span a
    horizon, from the DEVELOPMENT path's own average trade cycle time.

    A heuristic sizing of the resampling draw only -- it never affects which
    trades are chosen or in what order, only how many are drawn before
    ``build_synthetic_trade_placements`` truncates at the true horizon.
    """

    if not timing:
        raise ValueError("timing must not be empty")
    average_cycle_ns = sum(t.gap_before_ns + t.holding_ns for t in timing) / len(timing)
    if average_cycle_ns <= 0:
        average_cycle_ns = 1.0
    return max(
        1, math.ceil(Decimal(horizon_ns) / Decimal(average_cycle_ns) * safety_factor)
    )


def build_synthetic_trade_placements(
    trades: tuple[TradeRecord, ...],
    timing: tuple[TradeTiming, ...],
    index_path: Sequence[int],
    *,
    horizon_ns: int,
) -> tuple[tuple[TradeRecord, int, int], ...]:
    """Place resampled trade indices on a synthetic clock, truncated to
    the first trade whose entry would fall at or beyond ``horizon_ns``.
    """

    placements: list[tuple[TradeRecord, int, int]] = []
    clock_ns = 0
    for index in index_path:
        clock_ns += timing[index].gap_before_ns
        if clock_ns >= horizon_ns:
            break
        entry_ns = clock_ns
        exit_ns = clock_ns + timing[index].holding_ns
        placements.append((trades[index], entry_ns, exit_ns))
        clock_ns = exit_ns
    return tuple(placements)


def size_synthetic_path(
    policy: SizingPolicy,
    placements: tuple[tuple[TradeRecord, int, int], ...],
    initial_capital: Decimal,
) -> tuple[TradeEvent, ...]:
    """Causally size a synthetic path, one trade at a time, in order.

    Stops early if balance is exhausted (<= 0): no further trade can be
    sized, and the state machine will already have flagged a max-loss
    breach well before this point on any realistic path.
    """

    events: list[TradeEvent] = []
    balance = initial_capital
    for trade, entry_ns, exit_ns in placements:
        if balance <= 0:
            break
        sized = apply_sizing(policy, trade, balance)
        events.append(
            TradeEvent(
                entry_ns=entry_ns,
                exit_ns=exit_ns,
                floor_equity_delta=sized.floor_equity_delta,
                realized_pnl=sized.realized_pnl,
            )
        )
        balance += sized.realized_pnl
    return tuple(events)


def _censor_if_active(outcome: PhaseOutcome) -> PhaseOutcome:
    if outcome.status is FtmoPathStatus.ACTIVE:
        return replace(outcome, status=FtmoPathStatus.CENSORED_NOT_PASSED)
    return outcome


def simulate_two_phase_path(
    trades: tuple[TradeRecord, ...],
    timing: tuple[TradeTiming, ...],
    *,
    method: ResamplingMethod,
    block_size: int,
    policy: SizingPolicy,
    rules: PropRuleSet,
    initial_capital: Decimal,
    challenge_horizon_ns: int,
    verification_horizon_ns: int,
    seed: int,
) -> TwoPhaseOutcome:
    """Simulate one continuous resampled market path through both phases.

    Verification (if reached) continues consuming the *same* resampled
    trade sequence where Challenge left off, with a fresh capital reset --
    modeling a trader who keeps trading the same alpha into Verification
    immediately after passing Challenge, rather than an independent
    unrelated market redraw. This is a disclosed design choice, not a
    result-driven one: it was fixed before any pass-probability screen ran.
    """

    required = estimate_required_trade_count(
        timing, challenge_horizon_ns + verification_horizon_ns
    )
    index_path = draw_index_path(
        len(trades),
        method=method,
        block_size=block_size,
        seed=seed,
        min_length=required,
    )

    challenge_placements = build_synthetic_trade_placements(
        trades, timing, index_path, horizon_ns=challenge_horizon_ns
    )
    challenge_events = size_synthetic_path(
        policy, challenge_placements, initial_capital
    )
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
        return TwoPhaseOutcome(
            challenge=challenge_outcome, verification=None, passed_both=False
        )

    remaining_index_path = index_path[challenge_outcome.trades_replayed :]
    verification_placements = build_synthetic_trade_placements(
        trades, timing, remaining_index_path, horizon_ns=verification_horizon_ns
    )
    verification_events = size_synthetic_path(
        policy, verification_placements, initial_capital
    )
    verification_outcome = _censor_if_active(
        simulate_phase(
            verification_events,
            rules=rules,
            phase=EvaluationPhase.VERIFICATION,
            initial_capital=initial_capital,
            horizon_ns=verification_horizon_ns,
        )
    )
    return TwoPhaseOutcome(
        challenge=challenge_outcome,
        verification=verification_outcome,
        passed_both=verification_outcome.status is FtmoPathStatus.PASSED,
    )
