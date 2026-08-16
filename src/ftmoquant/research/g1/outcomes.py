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
