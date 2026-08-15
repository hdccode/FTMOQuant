# `ts_momentum_v1` Phase 1

Status: **implemented, not evaluated**. No strategy return has been read. The
machine-readable source of truth is
`config/strategies/ts_momentum_v1.yaml`, with semantic SHA-256
`edcbe2e4afe631e5fde1223558122ecf4d796abd0610729313ebbb32a468ccd5`.

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
