"""Aggregate Monte Carlo replications into pass-probability reporting.

Ranking objective (task §10): maximize P(pass BOTH phases); tie-break by
(1) lower P(fail daily loss), (2) lower P(fail max loss), (3) lower median
trading days to complete both phases. Nothing here optimizes headline
return or Sharpe, and nothing here selects a policy -- :func:`rank_policies`
only orders already-computed summaries; the caller decides whether to act
on the order.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from statistics import median

from ftmoquant.research.ftmo_pass_probability.monte_carlo import TwoPhaseOutcome
from ftmoquant.research.ftmo_pass_probability.state_machine import FtmoPathStatus

_Z_95 = 1.959963984540054  # two-sided 95% normal quantile


class CertaintyTier(StrEnum):
    """Reporting-only threshold tiers (task §11). Never used to pick sizing."""

    HIGH_CERTAINTY = "high_certainty"
    STRONG = "strong"
    PLAUSIBLE = "plausible"
    INSUFFICIENT_FOR_HIGH_CERTAINTY = "insufficient_for_high_certainty"


@dataclass(frozen=True, slots=True)
class BinomialEstimate:
    """A Monte Carlo proportion with a Wilson-score 95% confidence interval.

    This interval captures simulation (sampling) uncertainty only -- it
    does not capture future structural market change, model
    misspecification, slippage/commission uncertainty, or OANDA-vs-FTMO
    execution differences (task §14).
    """

    successes: int
    trials: int
    estimate: float
    ci_lower_95: float
    ci_upper_95: float


@dataclass(frozen=True, slots=True)
class PolicySummary:
    """Complete Monte Carlo summary for one sizing policy at one screen."""

    policy_id: str
    method: str
    replications: int
    pass_challenge: BinomialEstimate
    pass_verification_given_challenge: BinomialEstimate
    pass_both: BinomialEstimate
    fail_daily_loss: BinomialEstimate
    fail_max_loss: BinomialEstimate
    censoring_rate: BinomialEstimate
    median_trading_days_to_pass_both: float | None
    p90_trading_days_to_pass_both: float | None
    p95_trading_days_to_pass_both: float | None
    median_max_drawdown: float
    p95_max_drawdown: float
    certainty_tier: CertaintyTier


def wilson_score_interval(successes: int, trials: int) -> BinomialEstimate:
    """Wilson score 95% CI for a binomial proportion (stable at p near 0/1)."""

    if trials <= 0:
        raise ValueError("trials must be positive")
    if not (0 <= successes <= trials):
        raise ValueError("successes must be between 0 and trials")
    p_hat = successes / trials
    denominator = 1 + _Z_95**2 / trials
    center = (p_hat + _Z_95**2 / (2 * trials)) / denominator
    half_width = (
        _Z_95
        * math.sqrt((p_hat * (1 - p_hat) + _Z_95**2 / (4 * trials)) / trials)
        / denominator
    )
    return BinomialEstimate(
        successes=successes,
        trials=trials,
        estimate=p_hat,
        ci_lower_95=max(0.0, center - half_width),
        ci_upper_95=min(1.0, center + half_width),
    )


def certainty_tier(pass_both_probability: float) -> CertaintyTier:
    if pass_both_probability >= 0.90:
        return CertaintyTier.HIGH_CERTAINTY
    if pass_both_probability >= 0.80:
        return CertaintyTier.STRONG
    if pass_both_probability >= 0.65:
        return CertaintyTier.PLAUSIBLE
    return CertaintyTier.INSUFFICIENT_FOR_HIGH_CERTAINTY


def summarize_policy(
    policy_id: str, method: str, outcomes: tuple[TwoPhaseOutcome, ...]
) -> PolicySummary:
    trials = len(outcomes)
    if trials == 0:
        raise ValueError("outcomes must not be empty")

    pass_challenge_count = sum(
        1 for o in outcomes if o.challenge.status is FtmoPathStatus.PASSED
    )
    reached_verification = [o for o in outcomes if o.verification is not None]
    pass_verification_count = sum(
        1
        for o in reached_verification
        if o.verification is not None and o.verification.status is FtmoPathStatus.PASSED
    )
    pass_both_count = sum(1 for o in outcomes if o.passed_both)
    fail_daily_loss_count = sum(
        1 for o in outcomes if _any_phase_status(o, FtmoPathStatus.FAILED_DAILY_LOSS)
    )
    fail_max_loss_count = sum(
        1 for o in outcomes if _any_phase_status(o, FtmoPathStatus.FAILED_MAX_LOSS)
    )
    censored_count = sum(
        1 for o in outcomes if _any_phase_status(o, FtmoPathStatus.CENSORED_NOT_PASSED)
    )

    trading_days_to_pass_both = [
        o.challenge.trading_days
        + (o.verification.trading_days if o.verification else 0)
        for o in outcomes
        if o.passed_both
    ]
    max_drawdowns = [
        max(
            float(o.challenge.max_drawdown),
            float(o.verification.max_drawdown) if o.verification else 0.0,
        )
        for o in outcomes
    ]

    pass_both = wilson_score_interval(pass_both_count, trials)
    return PolicySummary(
        policy_id=policy_id,
        method=method,
        replications=trials,
        pass_challenge=wilson_score_interval(pass_challenge_count, trials),
        pass_verification_given_challenge=(
            wilson_score_interval(pass_verification_count, len(reached_verification))
            if reached_verification
            else wilson_score_interval(0, trials)
        ),
        pass_both=pass_both,
        fail_daily_loss=wilson_score_interval(fail_daily_loss_count, trials),
        fail_max_loss=wilson_score_interval(fail_max_loss_count, trials),
        censoring_rate=wilson_score_interval(censored_count, trials),
        median_trading_days_to_pass_both=(
            median(trading_days_to_pass_both) if trading_days_to_pass_both else None
        ),
        p90_trading_days_to_pass_both=(
            _percentile(trading_days_to_pass_both, 0.90)
            if trading_days_to_pass_both
            else None
        ),
        p95_trading_days_to_pass_both=(
            _percentile(trading_days_to_pass_both, 0.95)
            if trading_days_to_pass_both
            else None
        ),
        median_max_drawdown=median(max_drawdowns),
        p95_max_drawdown=_percentile(max_drawdowns, 0.95),
        certainty_tier=certainty_tier(pass_both.estimate),
    )


def rank_policies(summaries: tuple[PolicySummary, ...]) -> tuple[PolicySummary, ...]:
    """Order summaries by the frozen §10 objective and tie-breakers only."""

    def sort_key(summary: PolicySummary) -> tuple[float, float, float, float]:
        median_days = summary.median_trading_days_to_pass_both
        return (
            -summary.pass_both.estimate,
            summary.fail_daily_loss.estimate,
            summary.fail_max_loss.estimate,
            median_days if median_days is not None else math.inf,
        )

    return tuple(sorted(summaries, key=sort_key))


def _any_phase_status(outcome: TwoPhaseOutcome, status: FtmoPathStatus) -> bool:
    if outcome.challenge.status is status:
        return True
    return outcome.verification is not None and outcome.verification.status is status


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = quantile * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
