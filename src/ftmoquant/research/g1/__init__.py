"""Generic, DEVELOPMENT-only EUR/USD G1 alpha-research foundation."""

from ftmoquant.research.g1.family import (
    FamilyMetadata,
    FamilyRegistry,
    StrategyFamily,
)
from ftmoquant.research.g1.normalization import (
    CausalEwmaDailyVolatility,
    CausalExposureDecision,
    CompletedDailyLogReturn,
    CompletedDailyMidpoint,
    G1VolatilityNormalizer,
    completed_daily_log_returns,
)
from ftmoquant.research.g1.outcomes import CandidateOutcome
from ftmoquant.research.g1.search import SearchConfig, SearchMode, run_search

__all__ = [
    "CandidateOutcome",
    "CausalEwmaDailyVolatility",
    "CausalExposureDecision",
    "CompletedDailyLogReturn",
    "CompletedDailyMidpoint",
    "FamilyMetadata",
    "FamilyRegistry",
    "G1VolatilityNormalizer",
    "SearchConfig",
    "SearchMode",
    "StrategyFamily",
    "completed_daily_log_returns",
    "run_search",
]
