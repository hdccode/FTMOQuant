from __future__ import annotations

from decimal import Decimal

import pytest

from ftmoquant.research.ftmo_pass_probability.monte_carlo import TwoPhaseOutcome
from ftmoquant.research.ftmo_pass_probability.reporting import (
    CertaintyTier,
    certainty_tier,
    rank_policies,
    summarize_policy,
    wilson_score_interval,
)
from ftmoquant.research.ftmo_pass_probability.state_machine import (
    FtmoPathStatus,
    PhaseOutcome,
)


def _phase(
    status: FtmoPathStatus, days: int = 4, drawdown: str = "0.02"
) -> PhaseOutcome:
    return PhaseOutcome(
        status=status,
        ending_balance=Decimal("101000"),
        trading_days=days,
        trades_replayed=days,
        breach_trade_index=None,
        passed_trade_index=None,
        max_drawdown=Decimal(drawdown),
    )


@pytest.mark.parametrize(
    "probability,expected",
    [
        (0.95, CertaintyTier.HIGH_CERTAINTY),
        (0.90, CertaintyTier.HIGH_CERTAINTY),
        (0.85, CertaintyTier.STRONG),
        (0.80, CertaintyTier.STRONG),
        (0.70, CertaintyTier.PLAUSIBLE),
        (0.65, CertaintyTier.PLAUSIBLE),
        (0.50, CertaintyTier.INSUFFICIENT_FOR_HIGH_CERTAINTY),
    ],
)
def test_certainty_tier_thresholds(probability, expected) -> None:
    assert certainty_tier(probability) is expected


def test_wilson_interval_is_centered_and_bounded() -> None:
    estimate = wilson_score_interval(9000, 10000)
    assert (
        0.0 <= estimate.ci_lower_95 <= estimate.estimate <= estimate.ci_upper_95 <= 1.0
    )
    assert estimate.estimate == pytest.approx(0.9, rel=1e-6)


def test_wilson_interval_handles_zero_successes() -> None:
    estimate = wilson_score_interval(0, 500)
    assert estimate.estimate == 0.0
    assert estimate.ci_lower_95 == pytest.approx(0.0, abs=1e-12)
    assert estimate.ci_upper_95 > 0.0


def test_summarize_policy_computes_pass_both_and_failure_rates() -> None:
    outcomes = (
        TwoPhaseOutcome(
            _phase(FtmoPathStatus.PASSED), _phase(FtmoPathStatus.PASSED), True
        ),
        TwoPhaseOutcome(
            _phase(FtmoPathStatus.PASSED),
            _phase(FtmoPathStatus.FAILED_DAILY_LOSS),
            False,
        ),
        TwoPhaseOutcome(_phase(FtmoPathStatus.FAILED_MAX_LOSS), None, False),
        TwoPhaseOutcome(_phase(FtmoPathStatus.CENSORED_NOT_PASSED), None, False),
    )
    summary = summarize_policy("test_policy", "stationary", outcomes)
    assert summary.pass_both.successes == 1
    assert summary.pass_both.trials == 4
    assert summary.fail_daily_loss.successes == 1
    assert summary.fail_max_loss.successes == 1
    assert summary.censoring_rate.successes == 1
    assert summary.median_trading_days_to_pass_both == 8


def test_rank_policies_orders_by_pass_both_then_tiebreakers() -> None:
    better = summarize_policy(
        "better",
        "stationary",
        (
            TwoPhaseOutcome(
                _phase(FtmoPathStatus.PASSED), _phase(FtmoPathStatus.PASSED), True
            ),
        )
        * 9
        + (TwoPhaseOutcome(_phase(FtmoPathStatus.FAILED_MAX_LOSS), None, False),),
    )
    worse = summarize_policy(
        "worse",
        "stationary",
        (
            TwoPhaseOutcome(
                _phase(FtmoPathStatus.PASSED), _phase(FtmoPathStatus.PASSED), True
            ),
        )
        * 3
        + (TwoPhaseOutcome(_phase(FtmoPathStatus.FAILED_MAX_LOSS), None, False),) * 7,
    )
    ranked = rank_policies((worse, better))
    assert ranked[0].policy_id == "better"
