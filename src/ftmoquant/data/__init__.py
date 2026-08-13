"""Market data ingestion interfaces."""

from ftmoquant.data.dukascopy import (
    IngestionResult,
    IngestionValidationError,
    ingest_eurusd,
    probe_eurusd_scale,
)
from ftmoquant.data.session_coverage import (
    CoverageAssessment,
    CoverageInterval,
    CoverageValidationError,
    SessionCoverageResult,
    assess_eurusd_source_coverage,
    is_eurusd_expected_open,
    run_eurusd_session_coverage_qa,
)

__all__ = [
    "IngestionResult",
    "IngestionValidationError",
    "CoverageAssessment",
    "CoverageInterval",
    "CoverageValidationError",
    "SessionCoverageResult",
    "assess_eurusd_source_coverage",
    "ingest_eurusd",
    "is_eurusd_expected_open",
    "probe_eurusd_scale",
    "run_eurusd_session_coverage_qa",
]
