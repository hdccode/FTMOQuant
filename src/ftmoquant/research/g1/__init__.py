"""Generic, DEVELOPMENT-only EUR/USD G1 alpha-research foundation."""

from ftmoquant.research.g1.family import (
    FamilyMetadata,
    FamilyRegistry,
    StrategyFamily,
)
from ftmoquant.research.g1.outcomes import CandidateOutcome
from ftmoquant.research.g1.search import SearchConfig, SearchMode, run_search

__all__ = [
    "CandidateOutcome",
    "FamilyMetadata",
    "FamilyRegistry",
    "SearchConfig",
    "SearchMode",
    "StrategyFamily",
    "run_search",
]
