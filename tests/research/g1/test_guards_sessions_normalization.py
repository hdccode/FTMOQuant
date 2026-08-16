from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from ftmoquant.research.g1.guards import (
    DevelopmentAccessPolicy,
    ResearchAccessError,
    ResearchPartition,
)
from ftmoquant.research.g1.normalization import (
    G1VolatilityNormalizer,
    VolatilityNormalizationError,
)
from ftmoquant.research.g1.outcomes import CARVER_V1_OUTCOME, CandidateOutcome
from ftmoquant.research.g1.sessions import (
    SessionDefinitionError,
    SessionId,
    is_session_eligible,
)
from ftmoquant.research.g1.trials import FoldMetrics, TrialRegistryError


def test_development_guard_has_no_validation_or_holdout_access() -> None:
    policy = DevelopmentAccessPolicy(
        datetime(2010, 1, 1, tzinfo=UTC), datetime(2020, 1, 1, tzinfo=UTC)
    )
    assert policy.require_interval(
        datetime(2011, 1, 1, tzinfo=UTC), datetime(2012, 1, 1, tzinfo=UTC)
    )
    for partition in (ResearchPartition.VALIDATION, ResearchPartition.FINAL_HOLDOUT):
        with pytest.raises(ResearchAccessError, match="sealed"):
            policy.require_interval(
                datetime(2011, 1, 1, tzinfo=UTC),
                datetime(2012, 1, 1, tzinfo=UTC),
                partition=partition,
            )
    with pytest.raises(ResearchAccessError, match="sealed partition"):
        policy.require_development_path(Path("artifacts/validation/bars.parquet"))
    with pytest.raises(ResearchAccessError, match="leaves DEVELOPMENT"):
        policy.require_interval(
            datetime(2019, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
        )


def test_sessions_use_local_time_and_dst_safe_overlap() -> None:
    assert is_session_eligible(datetime(2026, 1, 5, 0, 30, tzinfo=UTC), SessionId.ASIAN)
    assert is_session_eligible(
        datetime(2026, 1, 5, 8, 30, tzinfo=UTC), SessionId.LONDON
    )
    assert is_session_eligible(
        datetime(2026, 3, 20, 13, 30, tzinfo=UTC),
        SessionId.LONDON_NEW_YORK_OVERLAP,
    )
    assert is_session_eligible(
        datetime(2026, 4, 1, 12, 30, tzinfo=UTC),
        SessionId.LONDON_NEW_YORK_OVERLAP,
    )
    assert not is_session_eligible(
        datetime(2026, 4, 1, 11, 30, tzinfo=UTC),
        SessionId.LONDON_NEW_YORK_OVERLAP,
    )
    with pytest.raises(SessionDefinitionError, match="timezone-aware"):
        is_session_eligible(datetime(2026, 1, 1), SessionId.ALL)


def test_g1_risk_normalization_is_fixed_at_one_percent_and_fails_closed() -> None:
    returns = (-0.001, 0.0, 0.001, 0.002)
    normalized = G1VolatilityNormalizer().normalize(returns, periods_per_year=252)
    observed = float(np.asarray(normalized).std(ddof=1) * np.sqrt(252))

    assert observed == pytest.approx(0.01)
    with pytest.raises(VolatilityNormalizationError, match="fixed"):
        G1VolatilityNormalizer(0.02)
    with pytest.raises(VolatilityNormalizationError, match="positive"):
        G1VolatilityNormalizer().scale_for(0.0)
    with pytest.raises(TrialRegistryError, match="1% annualized"):
        FoldMetrics(
            "fold",
            10,
            0.1,
            1.0,
            0.1,
            1.0,
            0.08,
            annualized_volatility_target=0.02,
        )


def test_failure_taxonomy_and_frozen_carver_metadata() -> None:
    assert {item.value for item in CandidateOutcome} == {
        "ALPHA_REJECTED",
        "VALIDATION_REJECTED",
        "ROBUSTNESS_REJECTED",
        "DEPLOYMENT_FEASIBILITY_BLOCKED",
        "IMPLEMENTATION_OR_DATA_FAILURE",
    }
    assert CARVER_V1_OUTCOME.candidate_id == "carver_trend_carry_ftmo5_v1"
    assert CARVER_V1_OUTCOME.outcome is (
        CandidateOutcome.DEPLOYMENT_FEASIBILITY_BLOCKED
    )
    assert CARVER_V1_OUTCOME.alpha_evaluated is False
