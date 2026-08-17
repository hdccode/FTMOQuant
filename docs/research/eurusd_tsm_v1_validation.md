# eurusd_tsm_v1 one-shot validation protocol

Status: `frozen_pre_validation_unobserved`. The authoritative machine-readable
protocol is
[`config/validation/eurusd_tsm_v1_one_shot.json`](../../config/validation/eurusd_tsm_v1_one_shot.json),
with semantic SHA-256
`580244b36e5f542d6739deb662e7bbf889edd6e52a147b698728e50f5cac554a`.
No validation market row or return was accessed while freezing it.

The only admitted trial is
`0b84330c4e13b930c6d3a2ef0a3d16424210c6f61414bd5a8dad8842dbca964a`:
H4, six-bar lookback, 0.25 deadband, and three-bar refresh. Validation is the
498-day half-open interval `[2023-04-11T00:00:00Z,
2024-08-21T00:00:00Z)`. The final holdout begins at the exclusive endpoint and
remains locked.

Passing requires a deterministic, numerically valid completion, at least 100
executed raw alpha transitions, and positive base and 1.5×-cost-stressed net
returns and expectancies. Sharpe, drawdown, yearly concentration, comparisons
with DEVELOPMENT magnitude, and FTMO Challenge rules are diagnostics only.

The runner warms H4 signal and causal daily EWMA state from DEVELOPMENT
information, resets account and P&L at validation start, reuses the DEVELOPMENT
native execution path, and writes only to the fixed one-shot directory
`.artifacts/g1_4h/eurusd_tsm_v1/validation_run`. It has no search or selector:

```console
uv run python -m ftmoquant.research.eurusd_tsm_validation \
  --selected-candidate .artifacts/g1_4h/eurusd_tsm_v1/development_run/selected_candidate.json \
  --trial-registry .artifacts/g1_4h/eurusd_tsm_v1/development_run/trial_registry.json \
  --development-result .artifacts/g1_4h/eurusd_tsm_v1/development_run/development_result.json \
  --universe-readiness .artifacts/g1_4b/universe/ftmoquant_universe_readiness.json \
  --development-root 'EUR/USD.DUKASCOPY=PATH' \
  --development-root 'GBP/USD.DUKASCOPY=PATH' \
  --validation-root PATH_TO_SEALED_EURUSD_VALIDATION \
  --evaluation-config .artifacts/g1_4c/phase2_cost_models.json
```

If the candidate fails, the outcome is `VALIDATION_REJECTED`; no alternate
DEVELOPMENT candidate may be tried.

## Validation decision record

Status: **VALIDATION_REJECTED / closed**. The one-shot run executed on
2026-08-16 against the sealed EUR/USD validation split
(`/Users/Shared/FTMOQuant-data/g1_4a/eurusd_corrected_v1_splits/validation`,
manifest SHA-256 `9cc2bc9d...bca964a` matching the frozen protocol) and wrote
`.artifacts/g1_4h/eurusd_tsm_v1/validation_run/{validation_result.json,
artifact_hashes.json, runtime_provenance.json}`.

The candidate executed 251 raw alpha transitions (≥ 100 required) but failed
four of the six frozen PASS gates: net return `-0.83574%`, 1.5×-cost-stressed
return `-0.87874%`, net expectancy `-3.3296e-05` (`≈ -0.333 bp/transition`),
and cost-stressed expectancy `-3.5009e-05` (`≈ -0.350 bp/transition`) were all
non-positive. Diagnostic-only figures: Sharpe `-0.6776` (vs. DEVELOPMENT mean
fold Sharpe `0.7131`), maximum drawdown `1.5258%` (vs. DEVELOPMENT
`0.9002%`). Yearly net expectancy was positive in 2023 (`+0.0022197`) and
negative in 2024 (`-0.0105771`).

This is a terminal record: no alternate DEVELOPMENT candidate was tried, no
tuning followed from these numbers, the final holdout was not accessed
(`final_holdout_accessed: false`), and DEVELOPMENT was not rerun (its three
frozen artifacts remain byte-identical to the DEVELOPMENT report). See
[`g1/outcomes.py`](../../src/ftmoquant/research/g1/outcomes.py)'s
`EURUSD_TSM_V1_OUTCOME` for the formal outcome record.
