# eurusd_policy_rate_carry_proxy_v1 — preregistration

Status: **preregistered_not_run**. The authoritative machine-readable source
of truth is
[`config/strategies/eurusd_policy_rate_carry_proxy_v1.yaml`](../../config/strategies/eurusd_policy_rate_carry_proxy_v1.yaml),
with semantic SHA-256
`b6b21d83e71e06c371645ea3dce33178e8168645c11d89c3f5ec63971c0023f9`.

## Semantic SHA change: scheduling-collision fix

The first real DEVELOPMENT attempt raised
`EurusdPolicyRateCarryProxyEvaluationError: multiple daily decisions mapped
to one execution frame` inside `build_carry_proxy_instructions`, strictly
before `_run_native_fold` was ever called -- an implementation/scheduling
failure, not an alpha result. No strategy P&L, carry accrual, fold metrics,
or gate evaluation was ever produced under the prior semantic SHA
`d200492d6c210e2b0968f8a9731fea00a6a2f273401dbce63011cf5d0cd14eae`, and no
output artifact exists for this family in `.artifacts/`.

Root cause: `causal_differential_series` emits one candidate decision per
calendar day, including weekends, but the tradable EUR/USD market is closed
across weekends (and would be similarly closed across weekday holidays or
any other multi-day data gap). Each decision was independently mapped to
the first executable frame strictly after it; with the market closed,
consecutive calendar-day decisions (e.g. Saturday and Sunday) both resolved
to the same Monday-reopen frame, producing a hard collision. Confirmed on
the real frozen `dev_fold_1` catalog using timestamp metadata only (no
price returns or P&L inspected): the first collision was the Saturday
2020-04-11 and Sunday 2020-04-12 00:00 UTC decisions, both resolving to the
execution frame at 2020-04-12T21:00:00Z (execution_information_ns
2020-04-12T21:01:00Z); fold 1 alone had 52 such colliding frames, one per
weekend.

Fix: `execution_and_costs.pending_target_policy` is now
`latest_causal_target_supersedes_unexecuted_older_target`. Daily decisions
are treated as target position states, not independent orders. When
multiple calendar-day decisions resolve to the same first-available
executable frame, only the decision with the greatest
`decision_information_ns` strictly before that frame survives; superseded
decisions produce no instruction, no order, no fill, no separate evidential
sample, and no execution cost. Resolution is driven purely by actual
executable-frame availability (no `weekday() < 5` special-casing), so it
behaves identically for weekends, weekday holidays, and arbitrary data
gaps. Every retained instruction is asserted to satisfy
`decision_information_ns < execution_information_ns`.

This changes execution-scheduling semantics only. It does not alter
`sign(ECBDFR - EFFR)`, the 00:00 UTC decision anchor, the one-business-day
rate lag, 1% causal volatility sizing, proxy carry inputs, monthly rollover
semantics, daily evidence semantics, or any DEVELOPMENT gate -- hence the
semantic SHA change is recorded here rather than treated as a silent patch.

**No DEVELOPMENT, validation, or final-holdout returns have been observed
for this family.** This document, the YAML preregistration, the strict
loader (`src/ftmoquant/research/eurusd_policy_rate_carry_proxy_spec.py`),
the production strategy/evaluator modules, and their synthetic-data unit
tests are the complete deliverable of this preregistration task. Running
real DEVELOPMENT returns is explicitly out of scope here and has not
happened.

## Hypothesis

After realistic spot execution costs and a transparently labelled proxy
carry accrual, an EUR/USD position aligned with the sign of the causal,
one-business-day-lagged ECB-vs-Fed short-rate differential earns positive
mean daily total return during DEVELOPMENT.

This is a **narrow EUR/USD pair-level policy-rate carry proxy** hypothesis.
It is explicitly **not** a claim about the general cross-sectional FX carry
factor (that would require a broader universe of currency pairs than this
repository's frozen 2-instrument universe supports), and it is explicitly
**not** a claim about actual FTMO broker swap economics (the carry accrual
here is a transparently labelled `proxy`, never `broker_calibrated`).

## Data

Two short-rate series are frozen, hash-verified, and already committed:

- `config/data/policy_rates/ECBDFR_raw.csv` (ECB Deposit Facility Rate),
  SHA-256 `a9993333c5648a3ee0cfbc6ac4cc925cb6270d9ccecfa32d0db62ea808d57fe4`.
- `config/data/policy_rates/EFFR_raw.csv` (US Effective Fed Funds Rate),
  SHA-256 `1c53e1aa230fbe2d39275775206b64ad7e56d36c8cd63e572f492278a1c79a73`.

Both are sourced from FRED; full provenance (URLs, retrieval timestamp,
units, missing-value semantics) is recorded in
`config/data/policy_rates/provenance.json`. The loader
(`ftmoquant.data.policy_rates.load_policy_rate_history`) verifies both
hashes on every load and fails closed on any drift.

`ftmoquant.data.policy_rates.causal_differential_series` derives the daily
ECBDFR-minus-EFFR differential under a strict one-business-day information
lag with weekend/holiday forward-fill. It has already been verified against
real data (July 31 2019 FOMC cut correctly propagates only from 2019-08-02
onward, exactly one business day after the cut's own effective date) and,
separately, that **across the full DEVELOPMENT window
(2019-03-11 to 2023-04-11) the differential's sign never flips** — EUR
short rates stayed below USD short rates throughout. This is a real,
load-bearing fact about the frozen source data, not something worked
around: it means the daily directional signal is expected to be
predominantly (not necessarily exclusively, since sizing can still zero
exposure during EWMA warm-up) SHORT EUR/USD across the whole DEVELOPMENT
partition, and DEVELOPMENT alone therefore cannot discriminate this
hypothesis from a naive short-EUR/USD-forever baseline. That limitation is
disclosed here rather than hidden.

## Constant-signal disclosure (pre-P&L data-mechanics fact)

This section states, as already-known pre-P&L facts about the frozen
source data (never derived by inspecting any strategy return), what the
sign-never-flips fact above implies about DEVELOPMENT specifically:

- Within the frozen DEVELOPMENT window (2019-03-11 to 2023-04-11), the
  causal ECBDFR-minus-EFFR differential's sign never changes. This was
  already verified in an earlier session purely as a fact about the two
  rate series' values, not by observing any strategy return.
- Therefore, in practice, V1 is effectively a **single persistent
  rate-directed EUR/USD exposure** (one direction throughout DEVELOPMENT)
  with ongoing volatility-driven resizing — **not** evidence of the
  strategy correctly navigating multiple independent rate-regime switches.
  Any apparent "regime attribution" diagnostic computed on DEVELOPMENT data
  will necessarily show only one regime bucket for this reason, and that
  is expected, not a bug.
- This is **not** grounds to change the strategy. No threshold, no regime
  filter, and no lookback may be added on the strength of this
  observation — the frozen sign-only rule stands exactly as
  preregistered, unmodified by this disclosure.
- Daily total-return observations remain the evidential unit regardless
  of this persistent-exposure character; the sample-count semantics
  section above is unaffected.
- The preregistered dependence-aware bootstrap CI diagnostic
  (`stationary_bootstrap_confidence_interval`) remains **required**
  specifically because of this persistent-exposure character: temporal
  dependence across DEVELOPMENT's daily observations is severe (one
  effectively continuous directional bet), not mild, so naive
  observation-count-based confidence statements would be materially
  overstated without it.

## Signal

`ftmoquant.strategies.eurusd_policy_rate_carry_proxy.EurusdPolicyRateCarryProxyState`
is a stateless, parameterless sign-only rule: `differential > 0 -> LONG`,
`differential < 0 -> SHORT`, `differential == 0 -> FLAT`. There is no
threshold, lookback, or smoothing. The daily decision anchor is **00:00
UTC**, deliberately not 17:00 America/New_York, specifically so the
strategy's own order flow never coincides with
`FXRolloverInterestModule`'s fixed 17:00 America/New_York accrual instant —
this keeps the equity-diff P&L decomposition (see below) causally clean.

## Risk normalization

Reused verbatim from every other EUR/USD family in this repository:
`ftmoquant.research.g1.normalization.G1VolatilityNormalizer` at 1%
annualized target volatility, sized from strictly-prior causal EWMA daily
log-return variance (60-trading-day center of mass, 20-observation minimum,
252-day annualization). Signal direction and risk sizing are separate:
`target_exposure = sign(differential) * 0.01 / causal_ex_ante_annualized_vol`.
Exposure is re-sized **daily even when direction is unchanged**, since
ex-ante volatility changes daily.

## Execution

Native G0.7/Nautilus execution, identical engine and BID/ASK boundary rule
used by every other family in this repository. Target changes execute only
at the first synchronized tradable BID/ASK frame strictly after the new
target's information time. No stop loss, no take profit, no FTMO-specific
optimization, no parallel backtester.

## Proxy carry accrual

`FXRolloverInterestModule` (Nautilus-native, Rust-implemented) is wired in
through `ftmoquant.backtest.execution_harness.RolloverConfig(mode=FX_INTEREST,
records=...)` with `calibration_status=CalibrationStatus.PROXY` — never
`BROKER_CALIBRATED`. This module was read in full from source before this
family was implemented; it is compatible with this task's proxy-carry
requirement:

- `InterestRateRecord(location, time, value)` — `location` is
  OECD/ISO-3166-alpha-3 or `"EA19"` (mapped internally to `"EUR"`); `"USA"`
  maps to `"USD"`. `time` is a `"YYYY-MM"` key: the module's **native
  resolution is monthly**, coarser than this strategy's daily,
  one-business-day-lagged signal. **This is a disclosed limitation, not
  silently smoothed over.**
- `ftmoquant.research.eurusd_policy_rate_carry_proxy_development.build_monthly_interest_rate_inputs`
  builds one `InterestRateInput` per currency per calendar month, using the
  raw (not precomputed-differential) ECBDFR/EFFR rate causally known as of
  the last business day of the *prior* month, under the same
  one-business-day lag frozen in `causal_differential_series`. Raw absolute
  rates are fed in specifically so the module itself economically signs the
  financing component (net signed quantity x mid price x rate / 365 / 100,
  with the module's own frozen Wednesday/Friday tripling for weekend
  rollover); **no second carry calculation is implemented anywhere in this
  family's code.**
- Accrual triggers at the module's own fixed 17:00 America/New_York
  weekday instant. The Python bindings expose no getter for applied
  rollover totals, so accrued carry is inferred by bracketing that instant
  with two equity marks per weekday (see below).

**Verified rc2 source, and quantitative lag disclosure.** The repo pins
`nautilus-trader==2.0.0rc2`. The exact commit that produced the installed
wheel's Python package version (`cad455e35c4489a3314e6fc9bad5c7b5c103ace8`,
verified via `git show cad455e35c:python/pyproject.toml`) was read in full
at `crates/backtest/src/modules/fx_rollover.rs`. This confirms, directly
from source, that `InterestRateRecord.time` accepts only two key formats —
`"YYYY-MM"` (monthly) or `"YYYY-Qn"` (quarterly fallback) — and that there
is **no daily key format and no internal forward-fill**: whoever
constructs the records is fully responsible for supplying a value for
every relevant month. `build_monthly_interest_rate_inputs` is therefore
necessary, not optional, and its bucketing mechanism was **not** changed by
this disclosure work.

This makes the carry component's effective information lag structurally
coarser than, and different in kind from, the directional signal's flat
one-business-day lag. For calendar day D in month M, the carry accrual
uses the raw ECBDFR/EFFR rate causally known as of the last business day
of month M-1. Concretely, the effective lag ranges from a **minimum of
~1 business day** (D = the 1st of month M) to a **maximum of
~1 calendar month minus 1 day, i.e. up to ~31 calendar days** (D = the
last day of month M). This range is now recorded explicitly in the frozen
YAML's `proxy_carry` block (`effective_lag_min_business_days`,
`effective_lag_max_calendar_days_approx`, `lag_disclosure`,
`verified_against`), not left implicit in prose alone.

## P&L decomposition

Three components are tracked and reported as explicitly separate
quantities: **spot P&L**, **execution cost**, and **proxy carry accrual**.
The identity `total_daily_equity_change = spot_pnl + carry_accrual -
execution_cost` is implemented as a hard, tested invariant
(`DailyPnlDecomposition.__post_init__` in
`eurusd_policy_rate_carry_proxy_development.py` raises if it is ever
violated).

Construction: execution cost is tracked directly via `on_order_filled`
(spread-vs-execution-midpoint plus any commission), exactly as every other
family in this repository does. Native account equity is marked twice per
weekday, bracketing the module's 17:00 America/New_York accrual instant at
the nearest available one-minute-bar resolution (a "pre" mark at local
16:59 and a "post" mark at local 17:00, both offset by the smallest
representable causal instant so neither can see the other side of the
accrual). Consecutive **post** marks form the daily total-equity-change
boundary (matching this repository's existing "17:00 America/New York
completed pair" daily-observation convention used elsewhere for volatility
estimation); the gap between each day's own pre/post pair isolates that
day's carry accrual. `spot_pnl` is always computed as the residual of the
identity, never independently.

## Sample / statistical semantics

The evidential unit is **one eligible causal trading-day marked TOTAL
return observation** (spot P&L + proxy carry accrual - execution costs) —
**not** a trade, order, fill, rebalance, or sign change.
`FoldMetrics.trade_count` is repurposed to hold
`eligible_daily_return_observation_count` for this family only, and is
documented as such everywhere it appears in code; no artifact or doc text
in this family calls these observations "trades."

Pooled expectancy is the arithmetic mean eligible daily total return.
Stressed expectancy applies a frozen 1.5x multiplier to the
**execution-cost component only** — the proxy carry accrual is passed
through completely unchanged (`stressed_daily_total_return` in the
evaluator module; unit-tested directly).

## DEVELOPMENT design (baseline-only, no search)

Exactly the existing frozen 3 DEVELOPMENT folds
(`ftmoquant.research.stage_g.frozen_development_folds()`). Exactly **one**
baseline configuration — no parameter grid, no selector, no plateau or
neighbour analysis. `select_candidate` and `assess_plateau` are never
imported or invoked anywhere in this family's code path
(`eurusd_policy_rate_carry_proxy_development.py`); gates are checked
directly by `evaluate_development_gates`.

Causal pre-evaluation history warms the EWMA volatility estimator and rate
information state only. No pre-fold P&L can enter a fold's evaluation
bookkeeping by construction: the instruction-building function
(`build_carry_proxy_instructions`) only emits an executable order
instruction for a decision whose 00:00 UTC information time falls inside
`[evaluate_start, evaluate_end)`; outside that window the position stays
flat, so there is nothing to reset.

### Frozen DEVELOPMENT gates (hard; never evaluated against real data in
this task)

1. Deterministic valid completion.
2. All 3 folds present.
3. Pooled eligible daily return observations >= 500.
4. Pooled mean daily total return > 0.
5. Pooled 1.5x-cost-stressed mean daily total return > 0.
6. >= 2/3 folds have positive mean daily total return.
7. All numerics finite (no NaN/inf).

### Diagnostics (reported, never hard-gated)

Sharpe, max drawdown, yearly attribution, rate-regime (differential-sign)
attribution, spot-P&L / carry-accrual / execution-cost pooled
contributions, and a `stationary_bootstrap_confidence_interval()`-based
dependence-aware CI, explicitly labelled diagnostic with commentary that
500 daily observations are not 500 independent bets given temporal
dependence.

## Lineage / integrity (Section 10)

Every prior family in this repository has already been DEVELOPMENT-burned —
see
[`docs/research/strategy_lineage_inventory.md`](strategy_lineage_inventory.md)
(audited 2026-08-17): all 10 preregistered families in `config/strategies/`
have real DEVELOPMENT evidence on disk, three of them (`eurusd_tsm_v1`,
`eurusd_session_range_expansion_v1`, `trend_pullback_v1`) also have real
validation evidence, and the inventory's own conclusion is that there was
"no preregistered-but-untouched candidate available to select for the next
experiment" without new design work.

This family is that new design work: a **genuinely new information
source** — the ECB-vs-Fed policy-rate differential — that no previous
strategy in this repository has used as its alpha signal. Every prior
family's signal was price-based (session ranges, momentum, liquidity
shocks, macro-surprise momentum, trend/pullback structure); none read a
policy-rate series. Prior failed price-based hypotheses were consulted only
at the mechanism-category level when selecting this new hypothesis (i.e.
"price-based mechanisms in this universe have repeatedly failed
DEVELOPMENT or validation, so try a fundamentally different information
source") — never at the level of their specific parameters or results. No
prior family's parameters were reused to rescue this family's performance,
because this family has not been run at all.

Validation remains locked
`[2023-04-11T00:00:00Z, 2024-08-21T00:00:00Z)`. Final holdout remains
locked (>= `2024-08-21T00:00:00Z`). `strategy_returns_accessed: false` for
this family in every sense the frozen YAML records.

## What has and has not been done

Done: data ingestion and provenance (already committed before this task);
the sign-only signal state; the strict preregistration loader and frozen
YAML with a stable semantic SHA-256; the production
strategy/evaluator module (`eurusd_policy_rate_carry_proxy_development.py`)
— importable, structurally complete, wired to the real
`FXRolloverInterestModule` via `RolloverConfig`/`CalibrationStatus.PROXY` —
and its synthetic-data unit and end-to-end tests, including one real native
`BacktestEngine` run over entirely fabricated bars that proves the
Wednesday/Friday-tripled carry accrual and the P&L decomposition identity
against real engine-produced numbers.

Not done, and out of scope for this task: any DEVELOPMENT, validation, or
final-holdout return has been computed or read for this family.
`run_eurusd_policy_rate_carry_proxy_development` exists, is importable, and
unconditionally refuses to execute (`EurusdPolicyRateCarryProxyEvaluationError`)
if called, precisely to make that boundary a hard runtime fact rather than
a convention.
