# G1.4A multi-instrument FX data

`g1_4_fx_usd_liquid_v1` is the first fail-closed multi-instrument universe. Its
ordered membership is exactly `EUR/USD.DUKASCOPY`, then `GBP/USD.DUKASCOPY`.
The universe is intentionally too small for cross-sectional factor evidence;
every universe readiness manifest records
`not_eligible_insufficient_universe` with a minimum of four instruments.

The frozen configuration is
`config/data/g1_4_fx_usd_liquid_v1.yaml`. It is independent of the legacy
`eurusd_research_v1` semantic document. Legacy EURUSD commands, source
validation, seven-minute correction, and hashes remain compatibility paths.

## Fail-closed acquisition

The generic importer requires a SHA-256 identity for data-use-rights evidence
before it performs repository metadata calls or downloads. The upstream dataset
does not declare a license, so this evidence is a release prerequisite, not a
placeholder to invent. The project owner's scoped personal, non-commercial-use
decision is recorded in
`config/data/evidence/g1_4a_hf_data_use_decision_v1.json`; it explicitly makes no
repository-wide licensing claim and contains no fabricated permission URL.

The importer refuses the HF shard beginning 2024-08-16 when the requested end is
the 2024-08-21 holdout cutoff. A full-period run must supply a prevalidated
`DirectSourceSegment` for the exact 2024-08-16 through 2024-08-20 tail. The two
source segments must be contiguous and non-overlapping. Row admission and the
final manifest both bind the exclusive cutoff; no zero-tick minute is filled.

Metadata inspection is available through `report_instrument_inventory` and does
not download shards. Live acquisition is intentionally not part of the code
implementation run which introduced this module.

## Sequential pipeline

Run one instrument at a time. Generic console entrypoints coexist with all
legacy EURUSD commands:

1. `ftmoquant-ingest-hf-instrument`
2. `ftmoquant-derive-instrument-bars`
3. `ftmoquant-qa-instrument-coverage`
4. `ftmoquant-reconcile-instrument-gaps` (one to four workers)
5. If direct ticks prove exact omissions, create an immutable child with
   `build_corrected_instrument_dataset`, then repeat derivation and QA.
6. `ftmoquant-materialize-instrument-splits`
7. `ftmoquant-freeze-instrument-readiness`
8. `ftmoquant-freeze-universe-readiness`

Correction input must equal the complete reconciliation-proven missing-minute
set. Split catalogs are physically distinct. Development is marked for
candidate read-only access; validation is marked for encrypted,
normally-unmounted, validation-runner-only access. The sealing code verifies
their ranges and tree hashes but infrastructure operators remain responsible for
mount permissions, encryption, key custody, and disabling validation-runner
network access.

Universe freeze accepts no missing or extra roots and binds ordered artifact,
catalog, split, and readiness hashes without absolute paths or wall-clock
timestamps. No strategy or return calculation is invoked by this pipeline.

## Deferred prerequisites

The live GBPUSD acquisition/readiness run remains blocked until data-use rights
are documented. The multi-instrument tournament harness remains a separate task
after a real two-instrument universe manifest is frozen. That task must add the
synchronized universe clock, currency-incidence exposure aggregation,
portfolio-level limits, and its own G1.4 registry before any candidate strategy
is implemented or evaluated.
