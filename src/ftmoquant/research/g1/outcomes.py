"""Formal candidate outcome taxonomy, independent of evaluation machinery."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class CandidateOutcome(StrEnum):
    ALPHA_REJECTED = "ALPHA_REJECTED"
    VALIDATION_REJECTED = "VALIDATION_REJECTED"
    ROBUSTNESS_REJECTED = "ROBUSTNESS_REJECTED"
    DEPLOYMENT_FEASIBILITY_BLOCKED = "DEPLOYMENT_FEASIBILITY_BLOCKED"
    IMPLEMENTATION_OR_DATA_FAILURE = "IMPLEMENTATION_OR_DATA_FAILURE"


@dataclass(frozen=True, slots=True)
class CandidateOutcomeRecord:
    candidate_id: str
    outcome: CandidateOutcome
    alpha_evaluated: bool
    reason: str


TREND_PULLBACK_V1_OUTCOME = CandidateOutcomeRecord(
    candidate_id="trend_pullback_v1",
    outcome=CandidateOutcome.VALIDATION_REJECTED,
    alpha_evaluated=True,
    reason=(
        "frozen G1.3 baseline (g1.3-trend_pullback_v1-first-frozen-baseline, "
        "commit 4bb4ec507cfd5bf4cb1a47cd8c6972eeafb6c248): overall_verdict=FAIL. "
        "DEVELOPMENT 460 trades, mean net R -0.15120, profit factor 0.794, win "
        "rate 30.2% vs 35.3% breakeven, negative both directions/4-of-5 years/"
        "all sessions/all volatility quartiles (development_mean_net_r_gt_0: "
        "FAIL). VALIDATION was already accessed in the same run: 154 trades, "
        "mean net R -0.14682, profit factor 0.806, 95% BCa stationary-"
        "bootstrap lower bound FAIL, calendar concentration FAIL (2024 net R "
        "-21.78 vs 2023 -0.83); only the trade-count gates passed "
        "(development >=100, validation >=50). Postmortem "
        "(commit 2801445ec1ac3bc55fcc99e5a58af663cc8e4509) classifies the "
        "failure as INSUFFICIENT_WIN_PROBABILITY and sets "
        "RETAINED_ONLY_AS_RESEARCH_REFERENCE; do not create trend_pullback_v1.1 "
        "or reuse the already-observed validation period as evidence for a new "
        "hypothesis. Final holdout was never accessed (>= 2024-08-21 remains "
        "sealed)"
    ),
)

LEO_GBPUSD_V1_OUTCOME = CandidateOutcomeRecord(
    candidate_id="leo_gbpusd_v1",
    outcome=CandidateOutcome.ALPHA_REJECTED,
    alpha_evaluated=True,
    reason=(
        "real DEVELOPMENT run twice at commit "
        "86e8755fe7cdbe5df691ac898f7b1a024c5cef8e (implementation changed "
        "between runs, 'development_run' then 'development_run_exit_fixed'): "
        "run 1: 1/3 positive folds, worst-fold annualized net Sharpe -1.568, "
        "pooled mean daily net return -8.68e-05, no fold passes 1.5x cost "
        "stress; run 2: 1/3 positive folds, worst-fold Sharpe -1.133, pooled "
        "mean daily net return -1.166e-04, one fold passes cost stress. "
        "Neither run ever produced a formal decision record; this is the "
        "first explicit disposition, assigned from the already-observed "
        "evidence without a third run. validation_accessed=false and "
        "final_holdout_accessed=false in both runs"
    ),
)

TS_MOMENTUM_V1_OUTCOME = CandidateOutcomeRecord(
    candidate_id="ts_momentum_v1",
    outcome=CandidateOutcome.ALPHA_REJECTED,
    alpha_evaluated=True,
    reason=(
        "real DEVELOPMENT run at commit 86e8755fe7cdbe5df691ac898f7b1a024c5cef8e "
        "('Implement G1.4B tournament infrastructure'): 1/3 positive folds, "
        "worst-fold annualized net Sharpe -1.568, pooled mean daily net "
        "return -8.68e-05. No postmortem or formal decision record was ever "
        "produced; this is the first explicit disposition, assigned from the "
        "already-observed evidence without a rerun. validation_accessed=false "
        "and final_holdout_accessed=false. Distinct from the later, "
        "independently-run eurusd_tsm_v1 (VALIDATION_REJECTED); the two must "
        "not be conflated"
    ),
)

CARVER_V1_OUTCOME = CandidateOutcomeRecord(
    candidate_id="carver_trend_carry_ftmo5_v1",
    outcome=CandidateOutcome.DEPLOYMENT_FEASIBILITY_BLOCKED,
    alpha_evaluated=False,
    reason="frozen aggregate Swing margin constraint breached before returns",
)

EURUSD_LIQUIDITY_SHOCK_REVERSION_V1_OUTCOME = CandidateOutcomeRecord(
    candidate_id="eurusd_liquidity_shock_reversion_v1",
    outcome=CandidateOutcome.ALPHA_REJECTED,
    alpha_evaluated=True,
    reason=(
        "exact 36-cell DEVELOPMENT grid (baseline_prior_returns x "
        "shock_multiple x hold_eligible_minutes) over the frozen 3-fold "
        "DEVELOPMENT window: 36/36 configurations completed, 0/36 pooled "
        "net-profitable, 0/36 pooled cost-stressed-profitable, 0/36 met all "
        "hard eligibility gates (pooled expectancy > 0, stressed pooled "
        "expectancy > 0, >=2/3 positive folds, >=100 pooled trades); no "
        "candidate selected. The overlapping 60/5.0/15 cell (matching the "
        "retired liquidity_shock_reversion_v1 baseline) was retained "
        "unmodified in the grid and failed the same gates as every other "
        "cell"
    ),
)

EURUSD_SESSION_RANGE_EXPANSION_V1_OUTCOME = CandidateOutcomeRecord(
    candidate_id="eurusd_session_range_expansion_v1",
    outcome=CandidateOutcome.VALIDATION_REJECTED,
    alpha_evaluated=True,
    reason=(
        "one-shot validation of trial "
        "cce6c53bd97b188ae3bf735f3ce52b9bfbdc3dc2ff7aaa456a40758a21b4bc69 "
        "(breakout_window_end=11:00, scheduled_exit=17:00) over "
        "[2023-04-11T00:00:00Z, 2024-08-21T00:00:00Z): 65 completed executed "
        "session-breakout round trips, net return -0.48637%, "
        "1.5x-cost-stressed return -0.51974%, net expectancy -0.748 "
        "bp/trade, stressed expectancy -0.800 bp/trade, Sharpe -0.6082, max "
        "drawdown 1.129%; four of six frozen PASS gates failed (net return, "
        "stressed return, net expectancy, stressed expectancy)"
    ),
)

EURUSD_TSM_V1_OUTCOME = CandidateOutcomeRecord(
    candidate_id="eurusd_tsm_v1",
    outcome=CandidateOutcome.VALIDATION_REJECTED,
    alpha_evaluated=True,
    reason=(
        "one-shot validation of trial "
        "0b84330c4e13b930c6d3a2ef0a3d16424210c6f61414bd5a8dad8842dbca964a "
        "(H4, lookback=6, deadband=0.25, refresh=3) over "
        "[2023-04-11T00:00:00Z, 2024-08-21T00:00:00Z): 251 executed raw alpha "
        "transitions, net return -0.83574%, 1.5x-cost-stressed return "
        "-0.87874%, net expectancy -0.333 bp/transition, stressed expectancy "
        "-0.350 bp/transition, Sharpe -0.6776, max drawdown 1.5258%; four of "
        "six frozen PASS gates failed (net return, stressed return, net "
        "expectancy, stressed expectancy)"
    ),
)
