# `ts_momentum_v1` Phase 1

Status: **DEVELOPMENT_BURNED / `ALPHA_REJECTED`** (corrected 2026-08-17; see
"Result / lifecycle correction" below). The YAML at
`config/strategies/ts_momentum_v1.yaml` still literally reads
`status: implemented_not_evaluated` — **that field is frozen and must not be
edited**: `ts_momentum_config_sha256` hashes the entire YAML document, and
`load_ts_momentum_spec` additionally requires the loaded document's identity
tuple to exactly equal `(1, "ts_momentum_v1", "1.0.0",
"implemented_not_evaluated")`, hardcoded in
`src/ftmoquant/research/ts_momentum_spec.py`. Changing the field would break
both the semantic hash
(`edcbe2e4afe631e5fde1223558122ecf4d796abd0610729313ebbb32a468ccd5`) and the
loader. The authoritative current status is recorded here and in
[`TS_MOMENTUM_V1_OUTCOME`](../../src/ftmoquant/research/g1/outcomes.py).

## Result / lifecycle correction

Despite the text below stating *"No strategy return has been read"* and
*"the real command has not been run,"* a real DEVELOPMENT run exists at
`.artifacts/g1_4c/ts_momentum_v1/development/manifest.json` (code commit
`86e8755fe7cdbe5df691ac898f7b1a024c5cef8e`, "Implement G1.4B tournament
infrastructure"): 1/3 positive folds, worst-fold annualized net Sharpe
`-1.568`, pooled mean daily net return `-8.68e-05`. `validation_accessed:
false` and `final_holdout_accessed: false` — **validation and final holdout
remain untouched**. No postmortem or formal `decision` field was ever
recorded for this run; this correction is the first explicit disposition.
Given negative pooled evidence and only 1 of 3 folds positive, this family
is recorded as **`ALPHA_REJECTED`**: it did not survive DEVELOPMENT and was
never promoted to validation. It must not be rerun based on this
already-observed evidence, and any future EUR/USD time-series-momentum work
should be understood as building on an already-tested, already-negative
daily sign-following baseline (note: `eurusd_tsm_v1`, a *different*, later,
H1/H4-lookback-based family, was independently preregistered, run, and
`VALIDATION_REJECTED` — see
[`docs/research/eurusd_tsm_v1.md`](eurusd_tsm_v1.md) — and should not be
conflated with this one).

## Frozen hypothesis

For EUR/USD and GBP/USD independently, the candidate selects only an actually
observed completed one-minute paired BID/ASK close whose information time is
17:00 `America/New_York` on Monday through Friday. This reuses the frozen
Dukascopy session convention and naturally maps to 22:00 UTC under EST and
21:00 UTC under EDT. The daily close is `(BID + ASK) / 2`. A missing instrument
does not invalidate the other instrument’s close. Nothing is filled,
interpolated, or forwarded into daily history.

Each instrument owns a pinned NautilusTrader
`RateOfChange(period=253, use_log=True)`. Nautilus period is the complete window
length, so 253 observations calculate exactly
`ln(C_t / C_(t-252))`: the current close against 252 prior eligible daily
observations. The first 252 valid observations emit no signal. Missing, zero,
negative, NaN, or infinite prices emit no signal and do not enter eligible
history.

The sign maps directly to the sole outputs:

- positive: `+1`
- negative: `-1`
- exactly zero: `0`

No unchanged target is re-emitted. The existing executable target remains held
until a changed daily target reaches the first synchronized, tradable Stage G
frame whose information time is strictly later than the signal information
time. The implementation emits no quantity, order, stop, target, volatility
scaling, or FTMO adjustment. Nautilus and Stage G remain the execution, cost,
currency exposure, portfolio-limit, fold, and tournament boundaries.

## Fold and data isolation

Every frozen development fold receives a fresh candidate instance. Its train
interval may initialize the 252-observation history but cannot create a pending
or executable target. Targets may originate only in that fold’s comparison
interval. All timestamps pass through `DevelopmentResearchContext`; validation
and final holdout partitions remain locked.

Phase 1 contains no catalog reader, performance metric, P/L calculation,
backtest loop, SPA/MCS/bootstrap invocation, parameter sweep, or winner logic.
Only synthetic fixtures were used to verify the implementation.

## Phase 2 evaluator

Phase 2 is implemented but has not been run. It adds the DEVELOPMENT-only
evaluator around these target outputs, freezes the per-instrument G0.7 execution
profiles before reading returns, and writes deterministic result provenance.
It does not unlock validation or holdout. Full semantics are documented in
`docs/research/ts_momentum_v1_phase2.md`. The reserved one-line invocation is:

```console
UV_CACHE_DIR=.uv-cache UV_PYTHON_INSTALL_DIR=.uv-python uv run ftmoquant-evaluate-ts-momentum-development --spec config/strategies/ts_momentum_v1.yaml --universe-readiness .artifacts/g1_4b/universe/ftmoquant_universe_readiness.json --development-root 'EUR/USD.DUKASCOPY=.artifacts/g1_4b/development/EURUSD' --development-root 'GBP/USD.DUKASCOPY=.artifacts/g1_4b/development/GBPUSD' --cost-models .artifacts/g1_4c/phase2_cost_models.json --output .artifacts/g1_4c/ts_momentum_v1/development
```

If the reserved cost-model artifact is absent, the evaluator materializes it as
an exact byte copy of the tracked frozen Phase 2 configuration. The real command
has not been run.
