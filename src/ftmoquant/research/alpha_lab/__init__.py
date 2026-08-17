"""Thin VectorBT-based DEVELOPMENT-only screening layer.

Canonical DEVELOPMENT FX data -> aligned pandas matrices -> vectorbt ->
standardized screening results. Not a backtester; no validation or holdout
access.
"""

from ftmoquant.research.alpha_lab.data import (
    AlphaLabDataError,
    AlphaLabDataset,
    discover_development_instrument_ids,
    load_alpha_lab_dataset,
)
from ftmoquant.research.alpha_lab.smoke import (
    ScreeningResultRow,
    SmokeRunResult,
    run_sma_crossover_smoke,
)

__all__ = [
    "AlphaLabDataError",
    "AlphaLabDataset",
    "ScreeningResultRow",
    "SmokeRunResult",
    "discover_development_instrument_ids",
    "load_alpha_lab_dataset",
    "run_sma_crossover_smoke",
]
