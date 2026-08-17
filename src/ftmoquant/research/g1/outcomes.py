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
