# mean_reversion_h1_v1 — execution-promotion audit and design freeze

Status: **execution-promotion design frozen. A real DEVELOPMENT run has been
performed across all seven frozen pairs, but it validated the strictly-later
execution plumbing only (order/fill counts, transition counts, report row
counts) -- it did NOT yet compute or evaluate any §8/§9 promotion metric.**
Performance measurement (net return, aggregate Sharpe, profitable-pair
count, 1.5x cost-stressed return) has since been implemented (§10 below) but
not yet run against real DEVELOPMENT data in this pass -- see §10 for the
exact rerun command and why a fresh run is required. VALIDATION has not been
run under any version of this module. This is an execution-promotion
exercise for an already-validated alpha candidate, not a new alpha search.
Nothing in this document may be used to change the frozen parameters below.

## 1. Frozen validated signal

Family: short-horizon mean reversion.
Exact configuration: `timeframe=H1`, `lookback=40`, `z_entry=2.0`.
Universe (all seven, none excluded): AUD/USD.OANDA, EUR/USD.OANDA,
GBP/USD.OANDA, NZD/USD.OANDA, USD/CAD.OANDA, USD/CHF.OANDA, USD/JPY.OANDA.

This is the representative selected by the frozen Stage-2 survivor
preregistration
(`config/validation/oanda_alpha_lab_stage2_survivors_v1_preregistration.json`,
`C_short_horizon_mean_reversion` → `lookback=40, z_entry=2.0`). No parameter
in this document changes it.

## 2. Exact frozen VectorBT signal semantics (audit)

Source: `src/ftmoquant/research/alpha_lab/families.py::mean_reversion_signals`,
executed by `src/ftmoquant/research/alpha_lab/screening_common.py::run_family_grid`.

- **Rolling mean**: `close.rolling(40).mean()` — pandas default window
  `[t-39, t]` inclusive, `center=False`.
- **Rolling standard deviation**: `close.rolling(40).std()` — pandas default
  `ddof=1` (sample standard deviation, confirmed empirically).
- **Current bar participates**: yes, in both the mean and the std — the bar
  being scored is itself one of the 40 observations. This is a deliberate,
  disclosed modeling choice (dampens a single-bar outlier's own z-score
  slightly), not a defect.
- **Z-score**: `z[t] = (close[t] - mean[t]) / std[t]`.
- **Long entry**: `z[t] < -2.0`. **Short entry**: `z[t] > 2.0`.
- **Exit / mean-cross**: long exits when `z[t] >= 0`; short exits when
  `z[t] <= 0`.
- **Signal timing**: every quantity at index `t` uses only
  `close[t-39..t]` — no `.shift(-k)`, no negative lookback, no centered
  window anywhere in the family. **No lookahead was found.**
- **Same-bar decision-and-fill**: `screening_common.run_family_grid` passes
  the identical `dataset.close` array to vectorbt both as the signal input
  and as the fill price (`vbt.Portfolio.from_signals(close, entries, ...)`).
  Empirically confirmed (see `tests/strategies/test_mean_reversion_h1.py`
  and earlier session verification): an entry signal true at index `t` fills
  at `close[t]` — the same bar, same price, same instant as the decision.
  This is a standard, disclosed vectorbt screening idealization (also true
  of every other alpha-lab family, and of Stage 1/2/Validation as already
  accepted). **It is not lookahead** (no future data is used to make the
  decision) but it *is* an execution-timing idealization no live system can
  replicate — see §5 for the causal translation this promotion applies.
- **Position persistence**: once entered, a position persists until an
  explicit exit/opposite-entry condition fires; there is no time-based or
  bar-count-based forced exit.
- **Reversal**: vectorbt's default `upon_opposite_entry='reverse'` converts
  an opposite-direction entry while already in a position into a single
  close-and-reverse order (confirmed empirically in the same-session
  Stage-1 implementation work). One edge case is genuinely ambiguous in the
  *source* vectorbt implementation and is called out rather than silently
  resolved: because `z_entry=2.0 > 0`, the short-entry condition
  (`z > 2.0`) implies the long-exit condition (`z >= 0`) is *also* true on
  the same bar whenever a long position is reversed by one extreme single-bar
  move (and symmetrically for shorts). vectorbt's own tie-break between
  "exit" and "opposite-entry-reverse" for this exact simultaneous case was
  not independently re-verified bit-for-bit; the causal port
  (`_next_position` in `mean_reversion_h1.py`) always treats it as a direct
  reversal, which is net-exposure-identical regardless of whether vectorbt
  internally posts one or two orders. This is disclosed, not silently
  patched, per the task's explicit instruction; it required a >4-sigma
  single-bar round trip to trigger and did not occur in any of the
  synthetic parity tests run.
- **Warm-up**: the first 39 bars per instrument produce `NaN` mean/std
  (`.fillna(False)` on the boolean arrays) — no position, no signal, exactly
  as `CausalMeanReversionSignal` reproduces (flat, `z_score=None`).
- **Gaps**: the rolling window is observation-count-based
  (`rolling(40)`), not calendar-time-based. An irregular gap between bar
  timestamps does not change the z-score — confirmed by
  `test_irregular_timestamp_gaps_do_not_change_the_signal`. Neither the
  original vectorbt family nor this causal port do anything special for
  gaps; both are silently insensitive to them by construction.

### Lookahead audit result

**No lookahead.** The z-score and every entry/exit condition depend only on
`close[t-39..t]`. The one caveat worth carrying forward is the same-bar
decision-and-fill idealization described above, which is a normal, disclosed
property of screening-stage vectorbt backtests (not unique to this family)
and is exactly what §5's execution-timing freeze exists to translate for a
causal execution layer. This does not invalidate the already-observed
DEVELOPMENT/Stage-2/VALIDATION screening evidence; it only means that
evidence is a *screening* result, not yet an execution-realistic one — which
is precisely the gap this promotion task closes.

## 3. Repo-first reuse audit

- **Execution harness**: `src/ftmoquant/backtest/execution_harness.py`
  provides the private `_add_venue`/`_engine_config`/`_fee_model`/
  `_rollover_modules` Nautilus wiring already reused unchanged by
  `eurusd_tsm_development.py`; there is no public generic multi-instrument
  runner today (the module's own docstring: "the only strategy in this
  module is a deterministic execution probe... must not be interpreted as
  alpha"). A real multi-pair run reuses these same private helpers rather
  than inventing new venue/engine wiring.
- **Risk normalization**: `src/ftmoquant/research/g1/normalization.py`
  (`G1VolatilityNormalizer`, `CausalEwmaDailyVolatility`) — reused
  unchanged, see §6.
- **Cost/latency profile**: `canonical_execution_profile()` in
  `execution_harness.py` — reused unchanged, see §5.
- **OANDA InstrumentSpecs / H1 derived bars**: `OANDA_ALPHA_LAB_SPECS`
  (`ftmoquant.data.instruments`) and the existing M30/H1/H4 derivation
  pipeline (`ftmoquant.data.derived_bars`, already used by both the OANDA
  alpha-lab DEVELOPMENT and VALIDATION lineages) — reused unchanged.
- **Provenance**: `ftmoquant.research.g1.artifacts`
  (`write_deterministic_artifact`, `write_runtime_provenance`) is the
  generic, already-reusable provenance writer (as opposed to the
  EUR/USD-specific manifest logic embedded in `execution_harness.py`).
- **No second backtester was built.** `mean_reversion_h1.py` only
  reproduces the causal *signal* (a pure function of price, no fills, no
  P&L, no portfolio accounting); actual order fills, spread crossing, and
  cost/latency application remain entirely owned by the existing Nautilus
  `BacktestEngine` wiring in `execution_harness.py`.

## 4. Screening-vs-Nautilus parity (synthetic) — REVISED with genuine
## end-to-end evidence

`tests/strategies/test_mean_reversion_h1.py` (pure signal layer) proves, on
multiple synthetic price paths:

- the causal z-score exactly matches `close.rolling(40).mean()/.std(ddof=1)`
  element-for-element;
- the causal position sequence (direction, timing of changes) agrees with
  the real `vbt.Portfolio.from_signals` output on >98% of bars across three
  independent random seeds, with disagreements confined to the documented
  single-bar double-cross ambiguity in §2; entry/exit *transition counts*
  differ by at most 1 between the two;
- a bar's causal output is provably invariant to any bar appended after it
  (direct no-lookahead proof, not just an absence-of-`.shift(-k)` code
  read).

**New in this revision**: `tests/research/test_mean_reversion_h1_development.py`
now also runs a genuine **real Nautilus `BacktestEngine`** (via
`run_frozen_signal_backtest`, the pure/partition-free engine core extracted
specifically for this purpose — see its docstring for why extracting it does
not weaken the production partition guards) over deterministic fabricated H1
BID/ASK bars, and directly compares its submitted-order/fill sequence
against both the causal signal and a real `vbt.Portfolio.from_signals` run
on the identical price series. Result, on the fabricated
warm-up→long-entry→mean-cross-exit→short-entry pattern:

- exactly 3 order submissions, sides `[BUY, SELL, SELL]`, matching the
  causal signal's 3 transitions exactly;
- submission `decision_ns` values match the causal transition bars exactly;
- VectorBT's own target-sequence transitions occur at the identical bar
  indices;
- **the Nautilus fill timestamp equals the decision timestamp exactly, for
  all three trades, at zero added latency.**

### Corrected finding — same-bar execution, not next-bar (SUPERSEDED — see §4b)

An earlier revision of this document asserted, from reasoning about
Nautilus's event loop rather than from a test, that "no same-bar fill is
possible in the Nautilus path." **That assertion was wrong and is retracted
here.** With `canonical_execution_profile()`'s zero added latency and
`bar_execution=True`, a market order submitted synchronously inside
`on_bar(bar_t)` is filled by Nautilus's own matching engine using `bar_t`
itself — empirically measured as `fill_ns == decision_ns` in every trade of
the synthetic parity run, with no exception. At zero latency, Nautilus and
VectorBT are execution-timing-*identical*: both decide and fill using the
same bar. There is therefore **no execution-timing degradation to quantify
at zero latency** — the two are causally equivalent at the decision/fill
level in this configuration.

This does not mean Nautilus and VectorBT-screening returns will be
identical: the difference between them comes entirely from **cost**, not
timing — VectorBT approximates cost as a proportional fee on the *mid*
price (`fixed_mean_half_spread_estimated_from_full_development`), while
Nautilus's `bar_execution` mechanics fill against the *actual* BID (for
sells) or ASK (for buys) of the same bar, i.e. it genuinely crosses the
realized spread rather than approximating it. That is the real, intended
degradation this promotion measures — not an execution-timing artifact.

**This finding was accurate but is not the frozen standard going forward —
see §4b.** Same-H1-bar decide-and-fill, while empirically not lookahead (no
future data was used), is not a defensible *realistic execution assumption*:
the completed H1 close used to make the decision must exist before an order
can be sent, so no live system could ever fill on the exact same bar whose
close produced the signal. §4b amends this before any real DEVELOPMENT/
VALIDATION return is observed.

### 4b. Execution-timing amendment — strictly-later execution (frozen,
### supersedes the same-bar finding above)

**This is the execution-timing standard actually implemented and used by
`mean_reversion_h1_development.py` going forward.** The H1 signal decision
itself is unchanged (§1, §2 above): the completed H1 bar's close still
produces the causal z-score and target exactly as validated. What changes is
*when the resulting order may be submitted*:

- **Decision-information time**: the completed H1 bar's close timestamp
  (unchanged).
- **Earliest permissible execution time**: the first genuine, paired M1
  BID/ASK observation strictly after the decision-information timestamp —
  never the H1 decision bar itself, regardless of latency configuration.
  Genuine means an actual catalog observation on both sides; no
  interpolation, synthesis, or gap-filling. If no later executable
  observation exists before the partition boundary, the instruction is
  dropped (fail closed) rather than executed against a next-partition bar.
- **Reused, not reinvented**: this is the same "higher-timeframe decision →
  lower-timeframe strictly-later execution" pattern already established by
  `eurusd_tsm_development.py` (`_ScaledTargetExecutor` / offline
  `_build_scaled_instructions` precomputation) and `trend_pullback.py`
  (`_PendingSignal` / `pair.info_time_ns > pending.info_ns`). The semantic
  "first strictly later" comparison itself reuses
  `carver_trend_carry_ftmo5_development.first_strictly_later_execution`'s
  contract — `mean_reversion_h1_development._first_strictly_later_paired_ns`
  is a `bisect`-based integer-nanosecond specialization proven equivalent to
  it in `tests/research/test_mean_reversion_h1_development.py`.
- **Mechanism**: `_precompute_h1_decisions` replays the frozen causal H1
  signal/sizing offline (H1 bars are read but never engine-injected) and
  resolves each target-unit transition's execution timestamp against the
  native M1 BID/ASK catalog. `_MeanReversionH1Executor` subscribes only to
  M1 bars and mechanically submits each precomputed instruction on its
  resolved M1 timestamp — it decides nothing live.
- **Cost profile**: unchanged from `canonical_execution_profile()` in every
  respect except this timing correction — genuine BID/ASK spread crossing,
  zero added latency, zero commission, zero adverse-slippage probability,
  rollover disabled. This combination is labeled
  `EXECUTION_TIMING_LABEL = "native_spread_crossing_strictly_later_execution"`
  in run results/provenance, deliberately distinct from the unchanged
  `"canonical_execution_profile"` cost-identity label recorded alongside
  it. This label is descriptive, not a calibration claim — no non-zero
  latency/slippage/commission evidence exists yet for any FTMOQuant OANDA
  account; a later execution-sensitivity stage may stress those separately.
- **Result**: the fill timestamp is now always strictly after the decision
  timestamp (`fill_ns == execution_ns > decision_ns`), proven directly
  against the previously-accepted same-H1-bar fixture in
  `test_synthetic_engine_parity_target_sequence_and_trade_count` (now
  asserting `fill_ns > decision_ns`, not `fill_ns == decision_ns`).
  Additional synthetic tests prove: real M1 gaps skip forward to the next
  genuine observation with no interpolation; an instruction with no later
  observation before the partition boundary is dropped rather than crossing
  into the next partition; two H1 decisions that would resolve onto the
  identical execution frame raise rather than silently collapsing; the
  resolved fill price crosses the correct spread side (buy at ASK, sell at
  BID); and the independent-equal-capital-sleeve semantics (§6 below) are
  unaffected.
- **Not changed by this amendment**: signal math, universe, sizing target,
  gates, final-holdout firewall, capital semantics.

## 5. Realistic execution spec (frozen before any real run; timing amended
## in §4b)

- **Signal decision time**: the closing timestamp of the H1 bar that
  produced the causal z-score (`CausalSignalBar.timestamp_ns`).
- **Earliest executable time (amended, §4b)**: the first genuine, paired M1
  BID/ASK observation strictly after the H1 decision timestamp — never the
  H1 decision bar itself. The same-H1-bar-fill behavior described in §4
  above was empirically accurate for the unamended wiring but is not a
  defensible realistic execution assumption (the completed close used to
  decide must exist before an order can be sent), so it is superseded by
  this strictly-later rule before any real DEVELOPMENT/VALIDATION return is
  observed. See §4b for the full mechanism and reuse rationale.
- **Order type**: market order at the next executable event (matches the
  existing `eurusd_tsm_development.py` precedent; no limit-order fill
  probability tuning).
- **Sizing rule**: §6.
- **Cost/latency/spread profile**: `canonical_execution_profile()`
  (`execution_harness.py:195`), reused **unchanged**: real BID/ASK spread
  crossing (native Nautilus fill mechanics, not a fee approximation), zero
  added latency, zero commission, zero adverse-slippage probability,
  deterministic fills (`random_seed=7`). This is the exact profile already
  used, unmodified, by every other *generic* validated FX strategy in this
  repo: `eurusd_tsm_development.py`, `ts_momentum_development.py`,
  `eurusd_session_range_expansion_development.py`,
  `eurusd_liquidity_shock_reversion_development.py`,
  `leo_gbpusd_development.py`, `trend_pullback_experiment.py`, and both
  `eurusd_tsm_validation.py`/`eurusd_session_range_expansion_validation.py`.
  The **only** exception is `eurusd_policy_rate_carry_proxy_development.py`,
  which defines its own `carry_proxy_execution_profile()` — but that family's
  entire hypothesis *is* interest-rate carry, so it specifically needs
  rollover enabled; that is a special case for a carry-specific alpha, not a
  "more realistic" generic baseline to imitate. Mean reversion has no carry
  hypothesis, so reusing the dominant, unmodified `canonical_execution_profile()`
  — exactly as seven-out-of-eight comparable FX families already do — is the
  consistent, non-cherry-picked choice.
  It is explicitly self-labeled `CalibrationStatus.UNCALIBRATED` by the
  repo's own typology (not `PROXY` or `BROKER_CALIBRATED`): its realism comes
  specifically from genuine BID/ASK spread-crossing (native Nautilus fill
  mechanics), not from any calibrated non-zero latency/slippage/commission
  evidence — none exists yet for any FTMOQuant OANDA account. Zero is the
  conservative default in the absence of that evidence, not a claim that
  real slippage/commission/latency are zero.
- **Rollover/carry treatment**: `RolloverConfig(mode=DISABLED)`, inherited
  unchanged from `canonical_execution_profile()` — the same choice already
  made for `eurusd_tsm_development.py`. This means the required-report
  "rollover contribution" (§8 of the task) will read as zero/not-modeled by
  construction; this is a reused precedent, not a new decision made to
  flatter results.
- **Liquidation boundary**: the frozen VALIDATION end
  (`stage_g.HOLDOUT_START`, 2024-08-21T00:00:00Z) — no position may be
  marked, sized, or evaluated using data at or after that instant
  (`reject_final_holdout`).

None of these were chosen by looking at realized returns; every value is
either reused unchanged from an existing frozen artifact or fixed by the
mechanics of the Nautilus event loop itself.

## 6. Risk normalization (frozen before any real run)

Reused unchanged: `G1VolatilityNormalizer` /
`CausalEwmaDailyVolatility` (`ftmoquant.research.g1.normalization`),
target = **1% annualized volatility per pair**
(`G1_ANNUAL_VOLATILITY_TARGET = 0.01`, structurally un-overridable —
`G1VolatilityNormalizer.__post_init__` raises if given any other value).
Estimator: causal EWMA of daily log returns, `com=60` days, minimum 20
completed prior daily returns before any exposure is sized (`0` otherwise —
fail closed, never filled/interpolated).

This is appropriate and is reused rather than invented because: it is
already the repo's single standard, causal, no-lookahead position-sizing
convention, used identically by every other G1-family validated strategy
(`eurusd_tsm_development.py:372-390`, plus the carry/liquidity/session-range
development modules). Applying the same common target across all seven
pairs directly satisfies §6's "common ex-ante volatility target" requirement
without inventing a second sizing convention. The target itself is not
tuned here and cannot be changed by this promotion.

### Capital semantics (frozen, corrected in this revision)

An earlier revision of `mean_reversion_h1_development.py` wired all seven
pairs into **one shared venue/account** (one `AccountParameters`, one
`engine.add_venue` call, `initial_capital=100,000 USD` shared across all
seven positions). This does **not** match what the already-validated
VectorBT screening evidence assumed: `screening_common.
_build_aggregate_portfolio` gives **every instrument its own independent,
equal `init_cash` sleeve** (`cash_sharing=False`) that never competes for
capital with any other pair; its "equal-weight aggregate" is the sum/average
of those independent sleeves' returns, not one pool's combined P&L divided
by one capital figure. Sharing one account across seven pairs would have
silently changed the capital-base denominator for every return calculation
relative to what was actually validated — exactly the kind of
unresolved semantic gap this task exists to close, not leave for the real
run to discover.

**Fixed**: `run_mean_reversion_h1_development` now runs **one independent
`BacktestEngine` (and therefore one independent account/capital sleeve) per
pair** — a loop over the seven frozen pairs, each calling
`run_frozen_signal_backtest` with exactly one instrument. This required no
new engine-wiring code: `run_frozen_signal_backtest` already constructs a
fresh `AccountParameters`/venue on every call, so achieving independent
sleeves was a matter of *how many times, and with what instrument set* it
is invoked, not new machinery. This is verified by
`test_run_calls_the_engine_core_once_per_pair_with_independent_capital`.

**Frozen going forward**: each of the seven pairs is sized against the same
`BASE_RESEARCH_UNITS = 100,000` reference notional and its own independent
capital sleeve; the equal-weight aggregate (once real returns exist) is
defined as the simple average of the seven pairs' own independent returns —
identical in spirit to `_build_aggregate_portfolio`'s already-validated
convention. No pair's fills, margin, or P&L can affect another pair's
capital availability.

## 7. Data access

Permitted and used only for design/audit and synthetic testing in this
pass: none of DEVELOPMENT, VALIDATION, or HOLDOUT price data was actually
read. `reject_final_holdout()` fails closed on any timestamp
`>= stage_g.HOLDOUT_START`. The real DEVELOPMENT/VALIDATION comparison run
(§8/§9 of the task) is designed but not executed — see the final report for
the exact command and the explicit reason it was not run in this pass.

## 8/9. Promotion gate (frozen, unevaluated)

A family execution-promotes only if, on the already-observed VALIDATION
period under the §5 realistic Nautilus execution spec: (1) aggregate net
return > 0; (2) aggregate stressed return > 0 under the existing alpha-lab
1.5x screening-cost-stress convention, if a compatible stress definition can
be mapped onto Nautilus cost output; (3) at least 4 of 7 pairs remain
profitable; (4) aggregate Sharpe > 0. All four must hold; no rescue
condition may be added after real results are read. **Not yet evaluated.**
A real DEVELOPMENT run has been performed (execution plumbing only, no
performance metrics computed at that time -- see status line and §10). No
real VALIDATION run has been performed under any version of this module,
and this section's gate has never been read against real numbers of any
kind; §10 below implements the measurement machinery this gate needs but
does not itself evaluate the gate.

## 10. Performance measurement (implemented; gate evaluation still separate)

This section documents a later addition to `mean_reversion_h1_development.py`
that computes the raw numbers §8/§9's gate needs -- it does not evaluate the
gate itself (no pass/fail verdict is computed anywhere in this module) and
does not change §1 (signal), §2/§4b (execution timing), §5 (cost profile),
§6 (sizing/capital semantics), or the final-holdout firewall.

- **Per-pair net return**: `(final_equity - initial_equity) / initial_capital`
  for each independent sleeve, from a daily equity-mark series sampled once
  per UTC calendar day (`_MeanReversionH1Executor._append_equity_mark`,
  mirroring `eurusd_tsm_development.py`'s own `EquityPoint`/`_append_equity`
  pattern exactly: venue account balance plus unrealized P&L of any open
  position, reused via direct import of `EquityPoint`).
- **Equal-weight aggregate net return**: the simple arithmetic average of
  the seven sleeves' own net returns -- mathematically identical to
  `screening_common._build_aggregate_portfolio`'s own equal-`init_cash`,
  `cash_sharing=False` summed-equity convention (summing equal-capital
  dollar P&L and dividing by the summed equal capital is the same as
  averaging the per-sleeve returns when every sleeve's capital is equal).
- **Profitable pair count**: count of the seven pairs with `net_return > 0`.
- **Aggregate daily-return Sharpe**: `_annualized_sharpe` (reused, imported
  from `ts_momentum_development.py`, already cross-reused there by
  `eurusd_tsm_development.py`) applied to a causally-formed aggregate daily
  series -- for every UTC day observed by *any* pair, the equal-weight
  average of that day's return across all seven pairs (a pair with no mark
  that day contributes `0.0` for that day, not an interpolated value).
- **1.5x cost-stressed return — faithfully mappable, and mapped**: the
  screening-stage 1.5x stress (`screening_common.STRESSED_COST_MULTIPLIER`)
  re-simulates `vbt.Portfolio.from_signals` with every fee scaled by 1.5x --
  there is no equivalent proportional fee here to re-simulate, because cost
  under `canonical_execution_profile()` is realized natively via genuine
  BID/ASK spread crossing, not a fee model. The faithful mapping reused here
  (not invented) is the decomposition already established twice in this
  repo for exactly this native-spread-crossing-vs-proportional-fee mismatch:
  `eurusd_tsm_development.py::_fold_metrics`
  (`stressed_total = total_net_return - 0.5 * cost_return`) and
  `ts_momentum_development.py`'s per-day
  `cost_stress_1_5x_return = net_return - realized_cost / 2`. Each fill's
  realized half-spread cost (`quantity * |fill_price - execution_midpoint|`,
  plus any commission) is tracked directly against the M1 execution
  midpoint already computed for strictly-later execution resolution (§4b);
  the stressed return subtracts half of that same realized cost again. The
  "0.5" coefficient is not independently chosen: it is
  `STRESSED_COST_MULTIPLIER - 1.0`, i.e. the *additional* fraction of
  already-realized cost a 1.5x-total-cost world would add. Labeled
  `COST_STRESS_METHODOLOGY_LABEL = "native_spread_crossing_realized_cost_half_stress"`
  in run results/provenance.
- **Artifacts**: `MeanReversionH1RunResult` gained `pair_performance: dict[str,
  PairPerformance]` (per pair: net return, realized cost, 1.5x-stressed
  return, annualized Sharpe, daily-observation count) and
  `aggregate_performance: AggregatePerformance` (equal-weight net return,
  equal-weight stressed return, profitable-pair count, aggregate Sharpe,
  aggregate daily-observation count, the cost-stress methodology label).
  `runtime_provenance.json` gained `cost_stress_methodology` and an explicit
  `promotion_gates_evaluated: false` flag.
- **Independent-sleeve semantics preserved**: every performance function
  above operates on one pair's own equity/cost series at a time; the only
  cross-pair step is the final equal-weight *average*, never a pooled
  capital or cash calculation -- no function in this section can make one
  pair's fills, margin, or cost affect another pair's reported return.
- **Not evaluated by this module**: whether the resulting numbers actually
  satisfy §8/§9's four gate conditions. That reading is a deliberate,
  separate step against real VALIDATION output once produced -- this module
  only ever reports numbers.
