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
from ftmoquant.research.alpha_lab.screening_batch import (
    GridConfig,
    ScreeningBatchResult,
    build_grid,
    run_screening_batch,
)
from ftmoquant.research.alpha_lab.screening_common import (
    ScreeningResultRow,
)
from ftmoquant.research.alpha_lab.smoke import (
    ScreeningResultRow as SmokeScreeningResultRow,
)
from ftmoquant.research.alpha_lab.smoke import (
    SmokeRunResult,
    run_sma_crossover_smoke,
)

__all__ = [
    "AlphaLabDataError",
    "AlphaLabDataset",
    "GridConfig",
    "ScreeningBatchResult",
    "ScreeningResultRow",
    "SmokeRunResult",
    "SmokeScreeningResultRow",
    "build_grid",
    "discover_development_instrument_ids",
    "load_alpha_lab_dataset",
    "run_screening_batch",
    "run_sma_crossover_smoke",
]
