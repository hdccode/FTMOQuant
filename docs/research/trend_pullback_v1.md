# `trend_pullback_v1` preregistration

Status: **specified, not run**. Version: **1.0.0**.

This document freezes the first G1 hypothesis before any strategy return is
examined. The machine-readable source of truth is
`config/strategies/trend_pullback_v1.yaml`; prose below disambiguates event
ordering and research decisions. If prose and configuration disagree, the run
must stop and the specification must be versioned before any result is viewed.

## Falsifiable hypothesis

After the already-built realistic BID/ASK execution and costs, EUR/USD has
positive net expectancy when a completed 1H close returns through its EMA(20)
in the direction of a completed 4H EMA(50)/EMA(200) trend. The hypothesis is
symmetric for long and short trades. It makes no claim about FTMO pass
probability and uses no Challenge-state risk adjustment.

The chosen periods are conventional, deliberately simple defaults. They were
fixed without inspecting strategy returns. Version 1.0.0 admits no parameter
variants or sweep.

## Data and chronology

- The only instrument is `EUR/USD.DUKASCOPY`.
- Signals use G0.6 completed internal 1H and 4H BID/ASK bars. For each OHLC
  field, the signal value is `(BID + ASK) / 2` from a pair with identical bar
  type except price side and identical completed-window timestamp.
- A pair that is missing, unequal in timestamp, stale, incomplete, or not
  research-ready emits no signal. No value is filled or carried across a gap.
- A decision is made only when both bars in the pair have been delivered. Its
  information time is the later `ts_init`; a nominal `ts_event` close alone is
  insufficient.
- At a 1H decision, the trend is the most recent completed 4H midpoint bar
  whose information time is no later than the 1H decision time.
- EMA and ATR state may be warmed with observations immediately before a
  research split, but those observations cannot create trades or results in
  the later split. No indicator state may read forward across a split.
- Signals require initialized EMA(20), ATR(14), EMA(50), and EMA(200). Until
  all four are initialized, the system fails closed.

## Trend, arm, and trigger

Let `C[t]` be completed 1H midpoint close and `E[t]` its completed EMA(20).
Let `F[T]` and `S[T]` be completed 4H midpoint EMA(50) and EMA(200).

- Long trend: `F[T] > S[T]`.
- Short trend: `F[T] < S[T]`.
- Equality has no trend and expires any armed setup.
- Long arm at 1H bar `t`: long trend is active,
  `C[t-1] > E[t-1]`, and `C[t] <= E[t]`.
- Short arm: short trend is active, `C[t-1] < E[t-1]`, and
  `C[t] >= E[t]`.
- Long trigger on a later completed 1H bar: the long setup is armed, long trend
  remains active, `C[t-1] <= E[t-1]`, and `C[t] > E[t]`.
- Short trigger is the exact inverse.
- An arm remains eligible for the next three completed 1H bars. It expires
  after the third non-triggering bar or immediately when its trend is lost.
  A same-direction arm observed while already armed restarts neither clock nor
  state. Opposite arming is impossible without first losing the active trend.

## Entry, sizing, and exits

- A trigger creates one market-order intent. It may execute only on the first
  synchronized G0.5 1-minute BID/ASK pair whose information time is strictly
  after the trigger decision. If none arrives within five minutes, discard the
  intent.
- The G0.7 canonical execution harness supplies spread, fee, slippage, latency,
  rollover, and matching behavior. The strategy must not duplicate or soften
  those assumptions. Longs execute against ASK and shorts against BID through
  that harness.
- ATR(14) is exponential and includes the completed trigger bar. Initial stop
  distance is `1.5 * ATR`. Quantity is selected so the initial stop represents
  exactly `1R` before costs. There is no compounding and no FTMO adjustment.
- Target distance is `2R`. Stops and targets never move.
- If lower-timeframe observations establish the order of stop and target, use
  that order. If both are first touched within the same unresolved 1-minute
  path, record the stop first. Never choose the favorable path.
- If neither price exit occurs, exit at the first eligible 1-minute pair after
  48 completed 1H bars with information times later than the entry fill.
- Ignore and discard all signals while a position or entry intent is active.
  There is one EUR/USD position maximum, no pyramiding, and no same-signal
  reversal. After flat, re-entry requires a completely new arm and trigger.

## Research split and evidence gate

The research-ready data manifest is divided chronologically into 60%
development, 20% validation, and 20% sealed final holdout. Boundaries are
floored to whole UTC days without reordering observations. The final holdout
must not be loaded, summarized, or inspected during G1; it remains sealed until
G2.

The primary metric is validation mean net R per trade. A G1 pass requires all
of the following after canonical costs:

- more than zero development mean net R across at least 100 trades;
- more than zero validation mean net R across at least 50 trades;
- validation profit factor strictly greater than 1.0;
- a two-sided 95% BCa stationary-bootstrap confidence interval for validation
  mean net R, using block size 5, 10,000 repetitions, and seed 1729, whose
  lower bound is strictly greater than zero; and
- no single validation calendar year supplies more than 50% of positive total
  validation net R.

Insufficient trade count is `unresolved`, never a pass. Nonpositive
development expectancy, validation expectancy, or confidence lower bound is a
failure of version 1.0.0. A failed version is not repaired by examining the
sealed holdout or silently changing a parameter.

Supporting outputs are development expectancy, the validation confidence
interval, validation profit factor and count, validation maximum drawdown in
R, and net R by calendar year. No result exists yet.

## Required provenance

Before an experiment may be marked complete, retain its Git commit, canonical
strategy-config SHA-256, data-manifest SHA-256, execution-profile SHA-256,
engine version, run seed, timestamps, primary result, and decision. The initial
registry row intentionally leaves run-time fields blank and says
`not_evaluated`.

## G1.2 handoff boundary

G1.2 may implement this exact state machine using pinned Nautilus v2 native
strategy configuration, completed-bar callbacks, EMA/ATR indicators, indicator
registration, and initialization flags. G1.2 must not alter this hypothesis,
run it, add a parameter sweep, or inspect the final holdout in the same change.
