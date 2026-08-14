"""Market data ingestion interfaces."""

from ftmoquant.data.canonical_source import iter_paired_source_chunks
from ftmoquant.data.derived_bars import derive_instrument_bars
from ftmoquant.data.dukascopy import (
    IngestionResult,
    IngestionValidationError,
    ingest_eurusd,
    probe_eurusd_scale,
)
from ftmoquant.data.hf_dukascopy import (
    acquire_direct_cutoff_segment,
    discover_instrument_source_plan,
    ingest_hf_instrument,
)
from ftmoquant.data.instruments import EURUSD_SPEC, GBPUSD_SPEC, InstrumentSpec
from ftmoquant.data.session_coverage import (
    CoverageAssessment,
    CoverageInterval,
    CoverageValidationError,
    SessionCoverageResult,
    assess_eurusd_source_coverage,
    is_eurusd_expected_open,
    run_eurusd_session_coverage_qa,
    run_instrument_session_coverage,
)
from ftmoquant.data.session_reconciliation import (
    acquire_instrument_reconciliation_batch,
    run_instrument_reconciliation,
)
from ftmoquant.data.universe_plan import (
    ResearchUniversePlan,
    load_research_universe_plan,
)
from ftmoquant.data.universe_readiness import (
    InstrumentArtifactRef,
    UniverseReadinessManifest,
    build_corrected_instrument_dataset,
    freeze_instrument_readiness,
    freeze_universe_readiness,
    materialize_instrument_split_views,
)

__all__ = [
    "acquire_direct_cutoff_segment",
    "acquire_instrument_reconciliation_batch",
    "IngestionResult",
    "IngestionValidationError",
    "InstrumentArtifactRef",
    "InstrumentSpec",
    "EURUSD_SPEC",
    "GBPUSD_SPEC",
    "CoverageAssessment",
    "CoverageInterval",
    "CoverageValidationError",
    "build_corrected_instrument_dataset",
    "derive_instrument_bars",
    "discover_instrument_source_plan",
    "SessionCoverageResult",
    "ResearchUniversePlan",
    "UniverseReadinessManifest",
    "assess_eurusd_source_coverage",
    "ingest_eurusd",
    "ingest_hf_instrument",
    "iter_paired_source_chunks",
    "is_eurusd_expected_open",
    "probe_eurusd_scale",
    "freeze_instrument_readiness",
    "freeze_universe_readiness",
    "load_research_universe_plan",
    "materialize_instrument_split_views",
    "run_eurusd_session_coverage_qa",
    "run_instrument_reconciliation",
    "run_instrument_session_coverage",
]
