# eurusd_tsm_v1 — preregistered exact-grid family

Status: `preregistered_not_run`. The authoritative machine-readable protocol is
[`config/strategies/eurusd_tsm_v1.yaml`](../../config/strategies/eurusd_tsm_v1.yaml),
with semantic SHA-256
`6f2cdf187e40e9bc9c065c7e7befa16f89d1fe380c8626ab853474658466f116`.
No historical strategy return was calculated while creating this protocol.

## Hypothesis and reuse

EUR/USD may exhibit time-series persistence from intraday through multiweek
horizons, so causal past price movement may predict subsequent returns after
realistic costs. This is price-only: no discretionary filter, macro input,
machine learning, stop, target, or configurable signal strength.

The implementation reuses three existing FTMOQuant boundaries rather than
duplicating them:

- `RawDirectionalTarget` and hold-until-changed semantics from
  `ts_momentum_v1`;
- H1/H4 synchronized completed BID/ASK `CompletedPair` semantics from
  `trend_pullback_v1` and the existing G0.6 derived bars;
- strictly-later native G0.7/Nautilus execution and the established 1.5×
  realized-cost stress convention for the future evaluator.

The old TSM implementation's single daily 252-observation ROC is intentionally
not reused as the family signal because it cannot express the frozen H1/H4
grid, deadband, refresh, or volatility normalization. Carver's mixed daily
price-unit volatility is also not used: its horizon is wrong here and its
positive floor conflicts with this experiment's fail-closed volatility rule.
NumPy's sample standard deviation is used directly over a causal rolling window
of one-bar log returns; no estimator source was copied.

## Frozen signal

For each contiguous completed BID/ASK bar, the signal uses midpoint close. For
lookback `L`, trailing return is `ln(C_t/C_(t-L))`. One-bar log-return sample
volatility uses the last `max(20, L)` returns with `ddof=1`; trailing-return
volatility is that estimate times `sqrt(L)`. The normalized trailing return is
the former divided by the latter.

The state emits `+1` above the positive deadband, `-1` below the negative
deadband, and `0` otherwise. It evaluates first when the full causal window is
available and then every configured refresh interval. Unchanged targets are
held and not re-emitted. Missing/noncontiguous data resets signal history; it is
never filled. Unavailable, zero, or non-finite volatility emits no signal and
is never replaced by a tiny value. Positions will execute only at the first
synchronized tradable one-minute observation strictly after signal information
time through the existing BID/ASK engine.

## Exact family experiment

Session is `All` and is not optimized. Each timeframe uses only its own
lookbacks and refresh intervals:

| Timeframe | Lookbacks | Refresh intervals | Deadbands |
| --- | --- | --- | --- |
| H1 | 24, 72, 120, 240, 480 | 1, 4, 24 | 0.0, 0.25, 0.5 |
| H4 | 6, 18, 30, 60, 120 | 1, 3, 6 | 0.0, 0.25, 0.5 |

The conditional Cartesian product contains exactly `45 + 45 = 90` unique
configurations. Search mode is exact grid, seed metadata is `0`, and Optuna is
forbidden. The 90 cells are one declared family experiment, not 90 selectively
reported discoveries. DSR and PBO remain unimplemented extension points.

## DEVELOPMENT and costs

The existing `g1.4b-development-folds-1` expanding windows remain frozen:

| Fold | Train `[start, end)` | Evaluate `[start, end)` |
| --- | --- | --- |
| `dev_fold_1` | 2019-03-11 – 2020-04-11 | 2020-04-11 – 2021-04-11 |
| `dev_fold_2` | 2019-03-11 – 2021-04-11 | 2021-04-11 – 2022-04-11 |
| `dev_fold_3` | 2019-03-11 – 2022-04-11 | 2022-04-11 – 2023-04-11 |

The future run must use observed EUR/USD BID/ASK spread, canonical G0.7
commission/slippage/latency semantics, and deterministic event order. The base
result uses the existing realistic profile. The 1.5× result subtracts another
half of realized base variable costs, matching the existing TSM evaluator.
There is no additional execution perturbation because the generic engine does
not yet implement one.

Raw target returns are normalized by the common G1 normalizer to 1% annualized
volatility for family comparison. This is not causal live sizing and introduces
no Challenge margin, daily-loss, pass-probability, or funded-account rule.

## Eligibility and selection

A cell must complete all three required folds, have at least 100 pooled executed
target transitions, have positive pooled net expectancy, remain positive under
1.5× cost, have at least two positive folds, and pass numerical validity checks.
The sample threshold reuses the existing preregistered trend-pullback
DEVELOPMENT minimum. Maximum drawdown, year concentration, and plateau quality
are not hard gates.

The existing generic selector is reused exactly. It ranks hard-gate survivors
lexicographically by:

1. positive fold fraction;
2. eligible evaluated-neighbour fraction;
3. lower absolute yearly concentration, missing last;
4. lower execution sensitivity, missing last;
5. higher worst-fold expectancy;
6. higher cost-stressed pooled expectancy;
7. higher pooled expectancy;
8. lower maximum drawdown;
9. deterministic SHA-256 trial ID.

No weighted score or maximum-Sharpe selection is used. Neighbours have the same
timeframe and differ by one adjacent step in exactly one of lookback, deadband,
or refresh interval. H1 and H4 cells are never neighbours.

If no cell survives in the future DEVELOPMENT run, the outcome is
`ALPHA_REJECTED`. Otherwise exactly one mechanically selected configuration is
frozen before a later, separate validation action. Validation and final holdout
remain locked.
