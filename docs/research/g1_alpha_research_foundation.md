# Generic EUR/USD G1 alpha-research foundation

G1 is alpha discovery only. This foundation replaces the architectural pattern
of adding another strategy-specific evaluator for every hypothesis. It does not
run historical returns, search real parameters, inspect validation or final
holdout, optimize an FTMO Challenge, create Carver V2, or implement the parked
supply/demand strategy.

## Boundary and authority

The existing EUR/USD canonical/derived bars and Stage G DEVELOPMENT manifests
remain the data authority. G0.7/Nautilus remains the sole BID/ASK execution,
spread, commission, latency, causality, and event-ordering authority. Family
implementations provide causal signal/state plus a bounded parameter
declaration; a caller-supplied authoritative evaluator returns common metrics.

All family comparisons use fixed 1% annualized volatility normalization. The
normalizer rejects zero, negative, non-finite, or numerically pathological
inputs. The target is immutable and is not an FTMO leverage, margin, pass-rate,
or live-risk target.

`DevelopmentSearchContext` exposes only a configurable DEVELOPMENT interval and
explicit rolling train/evaluation windows. Requests outside it, explicit
validation/final-holdout partitions, and paths containing sealed partition
names fail closed. A search has no validation or holdout accessor.

## Family contract

`StrategyFamily` owns:

- `FamilyMetadata`: ID, semantic version, economic hypothesis, supported
  timeframes, and eligible sessions;
- one typed bounded `ParameterSpace`;
- family-specific parameter validity constraints;
- causal `build_signals` implementation;
- optional family-defined adjacent parameter configurations.

The common engine owns data admission, folds, execution callback orchestration,
risk normalization, search, canonical parameters, trial identity, metrics,
complete registry, plateau analysis, selection, and artifacts. `FamilyRegistry`
provides deterministic `(family_id, version)` registration without making a
large plugin framework.

## Search and trial registry

Small finite spaces use declaration-order Cartesian enumeration. Continuous or
materially larger bounded spaces may explicitly select seeded Optuna/TPE. The
mode, seed, and Optuna trial count are immutable config with a semantic hash;
Optuna is not the default. A repeated Optuna configuration fails the run rather
than evaluating or silently discarding a duplicate.

Every attempted unique configuration becomes one `TrialRecord`, including
losers, family-invalid cells, evaluator-invalid cells, and implementation/data
failures. A complete record holds each fold's trade count, net expectancy,
optional daily mean, Sharpe, drawdown, turnover, cost-stressed expectancy,
positive/negative result, optional year/regime attribution, and optional
execution-perturbed expectancy. Canonical sorted JSON and SHA-256 over family,
version, and parameters give deterministic identities.

Artifacts explicitly count attempted, unique, valid, invalid, failed, hard-gate
survivor, and final-selector configurations. DSR and PBO are labeled
`not_implemented_extension_point`; no multiple-testing correction is claimed.
Deterministic results and semantic hashes exclude wall-clock metadata, which has
a separate provenance writer.

## Mechanical selection and plateau support

Selection is frozen and mechanical:

1. hard gates require positive pooled and cost-stressed expectancy, adequate
   trades/positive folds, and bounded drawdown;
2. robustness filters enforce configured year concentration, execution
   sensitivity, and evaluated-neighbour/plateau requirements where available;
3. survivors are ranked lexicographically by positive-fold fraction, acceptable
   neighbour fraction, low concentration, low execution sensitivity, worst-fold
   expectancy, cost-stressed expectancy, pooled expectancy, and drawdown;
4. the SHA-256 trial ID is the deterministic final tie-breaker.

There is no weighted composite score. A family, not the generic engine, defines
whether adjacent EMA periods, windows, thresholds, multipliers, or holding
periods are neighbours. The engine only matches those declarations to complete
registry cells and measures how many hard-gate-acceptable neighbours surround a
candidate.

## Sessions

Sessions are centralized as All, Asian (09:00–18:00 Asia/Tokyo), London
(08:00–17:00 Europe/London), New York (08:00–17:00 America/New_York), and the
actual intersection of London and New York. IANA timezone conversion handles
each region's DST and the weeks when their transitions differ. These are
eligibility intervals, not a holiday/provider-open calendar.

## Future family fit

The interface can later host time-series momentum/trend, trend pullback,
Donchian/volatility breakout, session opening-range breakout, short-horizon mean
reversion, failed-breakout/liquidity-sweep fade, and
`supply_demand_mtf_v1`. None was newly implemented here.

The parked `supply_demand_mtf_v1` shape remains: causal M30/H1/H4 zone retests,
substantially higher-timeframe directional alignment, ATR(50) exits, roughly
0.5 ATR target and 1.0–1.5 ATR stop candidates, narrow-zone preference, one
position, and both directions. Its zone/direction algorithms and bounded
DEVELOPMENT space must later be transparent. Challenge sizing and
stop-after-one-loss behavior are explicitly excluded from G1.

## Candidate outcome record

The formal outcomes are `ALPHA_REJECTED`, `VALIDATION_REJECTED`,
`ROBUSTNESS_REJECTED`, `DEPLOYMENT_FEASIBILITY_BLOCKED`, and
`IMPLEMENTATION_OR_DATA_FAILURE`.

Carver V1 remains frozen and was not rerun:

```yaml
candidate_id: carver_trend_carry_ftmo5_v1
outcome: DEPLOYMENT_FEASIBILITY_BLOCKED
alpha_evaluated: false
reason: frozen aggregate Swing margin constraint breached before returns
```
