# `session_range_expansion_v1` Phase 1

Status: **DEVELOPMENT failed / retired**. The machine-readable source of truth is
`config/strategies/session_range_expansion_v1.yaml`, semantic SHA-256
`5303094db45b3d9164d1854787c39d9ce0e69974875689dc51742e4495b9e472`.

For EUR/USD and GBP/USD independently, the candidate consumes only observed,
causally available Stage G one-minute BID/ASK pairs and calculates their
midpoint. Session timestamps are converted with DST-aware `Europe/London`.
Exactly 480 consecutive completed midpoint closes whose London completion times
are in `[00:00, 08:00)` define a valid session range. The first close in
`[08:00, 12:00)` strictly above its high emits `+1`; the first strictly below
its low emits `-1`. An incomplete or invalid range emits no entry, and no
breakout remains flat.

Each instrument consumes at most one entry per London calendar day. A live
target is replaced with flat when the first observed close at or after 16:00
London produces the exit signal. Every target—entry or exit—becomes executable
only at the first complete Stage G frame whose information time is strictly
later than the signal information time.

The fold adapter creates fresh state for every frozen DEVELOPMENT fold and
does not process its training interval. It therefore carries neither ranges,
pending targets, nor positions across a fold boundary. Validation and final
holdout partitions are rejected through the existing Stage G context.

This phase contains no catalog reader, evaluator, order submission, sizing,
cost model, P/L calculation, normalization, portfolio model, statistics,
parameter sweep, or FTMO-specific adjustment. Only synthetic tests were used.

## DEVELOPMENT evaluator wiring

The frozen candidate now has a DEVELOPMENT-only command adapter. It verifies
the strategy semantic SHA before entering the shared Stage G evaluator, admits
only the frozen DEVELOPMENT roots, materializes the existing canonical cost
artifact if it is absent, and reuses the same native execution, currency limits,
statistics, and deterministic provenance as `ts_momentum_v1`.

## DEVELOPMENT decision record

The frozen DEVELOPMENT result is **failed / retired**. It does not advance to
validation or final holdout: only two of three folds were positive, the worst
fold Sharpe was negative, two of three folds failed the fixed 1.5× cost-stress
check, the stationary-bootstrap confidence interval crossed zero, and SPA did
not reject the zero-return benchmark. This records the completed DEVELOPMENT
decision only; it introduces no tuning, strategy change, or FTMO optimization.

The reserved DEVELOPMENT-only invocation is:

```console
UV_CACHE_DIR=.uv-cache UV_PYTHON_INSTALL_DIR=.uv-python uv run ftmoquant-evaluate-session-range-expansion-development --spec config/strategies/session_range_expansion_v1.yaml --universe-readiness .artifacts/g1_4b/universe/ftmoquant_universe_readiness.json --development-root 'EUR/USD.DUKASCOPY=.artifacts/g1_4b/development/EURUSD' --development-root 'GBP/USD.DUKASCOPY=.artifacts/g1_4b/development/GBPUSD' --cost-models .artifacts/g1_4c/phase2_cost_models.json --output .artifacts/g1_4d/session_range_expansion_v1/development
```
