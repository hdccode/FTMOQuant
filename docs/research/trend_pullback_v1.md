# `trend_pullback_v1` preregistration

Status: **FAILED / RETIRED (`RETAINED_ONLY_AS_RESEARCH_REFERENCE`)**.
Version: **1.0.0**.

> **Current disposition vs. frozen preregistration text.** The YAML at
> `config/strategies/trend_pullback_v1.yaml` still literally reads
> `status: specified_not_run`, and the historical prose below still describes
> a not-yet-run hypothesis. **That is intentional and must not be edited.**
> `status` is a field inside the hashed preregistration document — changing
> it would change `strategy_config_sha256`
> (`d21100fc8412f4d258efdc90b2f1a936c0eb27cd6f88081ae082f24ae6d4cc5e`), which
> is hard-gated inside `src/ftmoquant/strategies/trend_pullback.py` and
> referenced by the frozen baseline/postmortem artifacts below. `
> specified_not_run` therefore records the state *at the moment of
> preregistration*, frozen for reproducibility — it is **not** a live status
> field. The authoritative current status is recorded here and in
> `EURUSD_TREND_PULLBACK_V1_OUTCOME`-equivalent
> [`TREND_PULLBACK_V1_OUTCOME`](../../src/ftmoquant/research/g1/outcomes.py).

## Result / retirement record

The family was run once, in full, under the frozen G1.3 process
(`g1.3-trend_pullback_v1-first-frozen-baseline`,
`.artifacts/g1_3_trend_pullback_v1_first_baseline.json`,
`semantic_sha256=db5c372bdbb8d549ed0baef1ba5f00ca96d98c7a7528226274270966d51afaa5`),
at commit
[`4bb4ec5`](../../../../commit/4bb4ec507cfd5bf4cb1a47cd8c6972eeafb6c248)
("Run G1.3 frozen baseline", 2026-08-14). **`overall_verdict: FAIL`.**

- DEVELOPMENT (2019-03-11 to 2023-04-11, warm-up-only before that): 460
  trades, mean net R **-0.15120** (gate `development_mean_net_r_gt_0`:
  **FAIL**; `development_trade_count_gte_100`: PASS), profit factor 0.794,
  win rate 30.2% vs. a 35.3% breakeven rate, negative both directions,
  negative in 4 of 5 calendar years, negative across all three sessions and
  all four volatility quartiles.
- VALIDATION (2023-04-11 to 2024-08-21) **was already accessed** in this same
  frozen run: 154 trades, mean net R **-0.14682** (gate
  `validation_mean_net_r_gt_0`: **FAIL**), profit factor 0.806 (`
  validation_profit_factor_gt_1`: **FAIL**), 95% BCa stationary-bootstrap
  lower bound **FAIL** (`validation_bca_95_lower_bound_gt_0`), calendar
  concentration gate **FAIL** (`validation_calendar_concentration_lte_0_50`;
  2024 alone carried net R of -21.78 vs. -0.83 in 2023), trade count gate
  PASS (`validation_trade_count_gte_50`, 154 ≥ 50).
- Final holdout: **not accessed** (`holdout_accessed: false`,
  `holdout_rows_admitted: 0`) — remains sealed.

A postmortem
(`.artifacts/g1_3_trend_pullback_v1_failure_postmortem.json`, commit
[`2801445`](../../../../commit/2801445ec1ac3bc55fcc99e5a58af663cc8e4509),
"Analyze G1.3 baseline failure", 2026-08-14) classified the failure as
`INSUFFICIENT_WIN_PROBABILITY`, found it broad rather than attributable to
one subset, and set `family_recommendation.classification:
RETAINED_ONLY_AS_RESEARCH_REFERENCE` with the explicit rationale: *"Preserve
it as a falsified reference; do not create trend_pullback_v1.1."* Its own
`future_evaluation_protocol` states any continuation of this research
direction must use a genuinely prospective dataset collected after
preregistration (recommended no earlier than 2026-08-17 UTC) and must **not**
reuse the already-observed 2023-04-11 through 2024-08-20 validation period as
clean evidence for any new hypothesis. The original final holdout
(`>= 2024-08-21`) remains sealed throughout.

**This is a terminal record.** No parameter was changed, no variant was run,
and the sealed final holdout was never opened. See
[`experiments/registry.csv`](../../experiments/registry.csv) for the
machine-readable completed-experiment row and
[`g1/outcomes.py`](../../src/ftmoquant/research/g1/outcomes.py) for the
canonical current-G1 outcome record.

---

## Historical preregistration (frozen at commit `06202b3`, unmodified below)

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
