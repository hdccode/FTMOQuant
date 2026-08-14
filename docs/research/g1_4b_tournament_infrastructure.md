# G1.4B Stage G tournament infrastructure

Stage G is the strategy-free development boundary for the multi-strategy
tournament. It validates the exact frozen `g1_4_fx_usd_liquid_v1` identity,
provides causal multi-instrument primitives, freezes candidate eligibility and
comparison policy, and supplies a read-only marimo research view. It does not
implement a candidate, run a backtest, or read strategy returns.

## Architecture and isolation

`ftmoquant.research.stage_g` owns the exact readiness and plan hashes, ordered
instrument loader, DEVELOPMENT context, synchronized clock, return/currency
normalization, currency-incidence exposure model, per-instrument execution-cost
binding, and deterministic development folds. The loader accepts the frozen
readiness SHA
`e6d511bb7bbea6b80c42b00c3f1b4c79ea35cd4532d5f71b69ee8f43bea3f8a0`
and plan SHA
`16c29df7bae3bde0a9e64e2cc7758f158186182a3d7a464347791400abcfff69`
only. Missing, extra, or reordered instruments fail closed.

The candidate-facing context can admit only the half-open interval
`2019-03-11T00:00:00Z` through `2023-04-11T00:00:00Z`. Requests labelled
validation or holdout fail explicitly. A split path is admitted only when its
semantic manifest says `split=development`, has `candidate_read_only` access,
matches the frozen per-instrument development split SHA, and declares zero
holdout rows. Stage G has no validation catalog loader, holdout loader, return
artifact reader, or fallback split.

The synchronized clock takes already-observed paired BID/ASK closes and
preserves their causal availability. Its two explicit missing-bar policies
either drop an incomplete timestamp or retain it as a nontradable frame with a
literal `None` for the absent instrument. It never forwards, interpolates, or
synthesizes a price.

FX positions produce `+base` and `-quote` currency incidence. Portfolio limits
are configured per currency and missing limits fail closed. Quote-currency P/L
is USD directly for the current EUR/USD and GBP/USD instruments. A future
non-USD quote requires an explicit positive point-in-time quote-to-USD rate;
there is no global USD-quote assumption.

Per-instrument costs reuse the complete G0.7 `ExecutionProfile` and its public
validator. Spread is always the observed paired BID/ASK spread. Commission,
adverse slippage, latency, and swap/rollover therefore retain the existing
NautilusTrader-backed semantics rather than acquiring a second cost framework.

## Folds, registry, and selection

Three expanding-window comparisons are frozen wholly inside DEVELOPMENT. Their
versioned canonical hash is exposed with the registry and selection-contract
hashes. `ftmoquant.research.tournament_registry` lists, in order:

1. `ts_momentum_v1`
2. `carry_momentum_v1`
3. `carry_momentum_value_v1`
4. `session_range_expansion_v1`
5. `liquidity_shock_reversion_v1`
6. `session_regime_hybrid_v1`

Time-series/session candidates whose infrastructure prerequisites are present
are eligible for later implementation, not approved or evaluated. Carry stays
blocked until valid point-in-time carry inputs exist. Carry/value stays blocked
for missing point-in-time carry and value inputs and because cross-sectional
ranking requires at least four instruments. Price data may not substitute for
carry or value inputs.

The preregistered selection contract fixes primary and robustness metrics, fold
aggregation, hard-failure semantics, deterministic tie-breaking, and a
multiple-testing policy using the existing `arch==8.0.0` SPA, MCS, and
stationary bootstrap facilities. At most one candidate can advance, and only
after every frozen gate passes. Otherwise none advances. A composite is allowed
only if constituents and fixed weights were registered before returns were
seen and the composite independently passes the same gates. Validation remains
an independent runner handoff; Stage G cannot unlock it.

Future candidate implementations plug in by preserving the registered
candidate ID, implementing signal/execution logic outside the notebook,
consuming only a `DevelopmentResearchContext`, supplying all frozen fold
outputs, and recording any failure rather than dropping it. Adding a candidate,
changing a fold, tuning a metric, or altering a composite after return access
requires a new version and preregistration.

## marimo workflow

marimo is pinned as the uv development dependency `marimo==0.23.15`. From the
repository root, open the tracked notebook with the verified installed syntax:

```console
UV_CACHE_DIR=.uv-cache UV_PYTHON_INSTALL_DIR=.uv-python uv run marimo edit research/g1_4_tournament.py
```

Read-only app mode is:

```console
UV_CACHE_DIR=.uv-cache UV_PYTHON_INSTALL_DIR=.uv-python uv run marimo run research/g1_4_tournament.py
```

The notebook uses fixed ignored local mount points:

- `.artifacts/g1_4b/universe/ftmoquant_universe_readiness.json`
- `.artifacts/g1_4b/development/EURUSD/`
- `.artifacts/g1_4b/development/GBPUSD/`

Each development directory must contain its exact
`ftmoquant_split_view.json` and `catalog/`. There is no path widget. If an exact
mount is unavailable or incompatible, the notebook displays a fail-closed
unavailable status. It does not silently substitute validation. Codex can pair
later by editing the tested FTMOQuant modules and notebook presentation cells;
substantive research logic remains outside the notebook.

Stage G itself has inspected no strategy returns. During implementation it did
not access validation data, final-holdout data, or market rows at or after the
holdout cutoff.
