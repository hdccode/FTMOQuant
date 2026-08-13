"""Market data ingestion interfaces."""

from ftmoquant.data.dukascopy import (
    IngestionResult,
    IngestionValidationError,
    ingest_eurusd,
    probe_eurusd_scale,
)

__all__ = [
    "IngestionResult",
    "IngestionValidationError",
    "ingest_eurusd",
    "probe_eurusd_scale",
]
