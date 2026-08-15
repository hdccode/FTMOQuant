# `ts_momentum_v1` Phase 2 DEVELOPMENT evaluator

Status: **implemented, not evaluated**. The real DEVELOPMENT command has not
been run and no strategy return has been inspected.

The evaluator is a thin orchestration layer. It loads the frozen strategy SHA,
the exact Stage G universe/readiness manifests and three DEVELOPMENT folds,
builds causal synchronized frames, and passes changed raw targets to one native
NautilusTrader `BacktestEngine`. It reuses G0.7 venue, fee, fill, latency,
rollover, native report, and native position-P/L facilities. It does not contain
a second backtester, execution model, cost ledger, portfolio engine, or
bootstrap implementation.

The tracked evaluation/cost source of truth is
`config/research/ts_momentum_development_v1.json`, semantic SHA-256
`6764e2bc41cc92cecd050419bb003da82db2960c3f9feb909b75e0be139a2ba2`.
The reserved runtime artifact
`.artifacts/g1_4c/phase2_cost_models.json` is created, if absent, as an exact
byte copy of that tracked freeze. Any existing artifact with different semantic
content fails closed.

Both instruments use the repository's sole canonical G0.7 uncalibrated
profile: observed paired BID/ASK spread, zero added commission, adverse
slippage, latency, and rollover. A nonzero raw target maps to the existing
100,000-base-unit G0.7 probe quantity. Daily net P/L is divided by the fixed
100,000 research denominator. The EUR, GBP, and USD incidence limits are each
3,000,000, derived mechanically from the existing 100,000 USD account and 30×
leverage; they are not FTMO optimization parameters.

For every fold, train data initializes candidate history but cannot originate a
target. A changed target is executed only at its preregistered first complete
synchronized frame strictly after signal information time. Daily equity uses
native account balance plus native `Position.unrealized_pnl` at liquidation-side
BID for longs and ASK for shorts. The evaluator emits per-fold native reports,
aligned session-day net returns, fold Sharpe/mean/drawdown/turnover/cost-stress
and exposure diagnostics, pooled mean, the adopted stationary-bootstrap mean
interval, and the adopted SPA zero-return comparison. MCS is recorded as not
applicable because the adopted wrapper correctly refuses a one-model input.

Before and after the run, catalog tree hashes must equal the frozen DEVELOPMENT
manifests. The output records all frozen hashes, source hashes, package versions,
input catalog hashes, and explicit false flags for validation access, final
holdout access, and parameter optimization. The output directory must not
already exist.

The only real evaluation invocation is:

```console
UV_CACHE_DIR=.uv-cache UV_PYTHON_INSTALL_DIR=.uv-python uv run ftmoquant-evaluate-ts-momentum-development --spec config/strategies/ts_momentum_v1.yaml --universe-readiness .artifacts/g1_4b/universe/ftmoquant_universe_readiness.json --development-root 'EUR/USD.DUKASCOPY=.artifacts/g1_4b/development/EURUSD' --development-root 'GBP/USD.DUKASCOPY=.artifacts/g1_4b/development/GBPUSD' --cost-models .artifacts/g1_4c/phase2_cost_models.json --output .artifacts/g1_4c/ts_momentum_v1/development
```

Validation and final holdout have no loader or command option in this module.
