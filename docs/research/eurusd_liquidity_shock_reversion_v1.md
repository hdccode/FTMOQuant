# eurusd_liquidity_shock_reversion_v1 — preregistration

Status: `preregistered_not_run`. The authoritative machine-readable source of
truth is
[`config/strategies/eurusd_liquidity_shock_reversion_v1.yaml`](../../config/strategies/eurusd_liquidity_shock_reversion_v1.yaml),
with semantic SHA-256
`50cd4e5460e488e06b33edd48fdb482a549199bcca4daa48625ce74da3865ad8`. No
DEVELOPMENT return, validation row, or final-holdout row was accessed while
freezing it.

## Context

This family follows the closure of
[`eurusd_tsm_v1`](eurusd_tsm_v1_validation.md), which failed one-shot
validation (`VALIDATION_REJECTED`; see the validation decision record there
and `EURUSD_TSM_V1_OUTCOME` in
[`g1/outcomes.py`](../../src/ftmoquant/research/g1/outcomes.py)). No
alternate TSM DEVELOPMENT candidate was tried; this is a new, independently
preregistered family.

## Prior lineage: `liquidity_shock_reversion_v1` (adverse prior evidence)

1. A **prior, retired candidate named `liquidity_shock_reversion_v1`**
   already exists in this repository — see
   [`docs/research/liquidity_shock_reversion_v1.md`](liquidity_shock_reversion_v1.md)
   and the frozen record at
   `.artifacts/g1_4e/liquidity_shock_reversion_v1/development/manifest.json`.
2. It used the **same core shock/fade hypothesis** at one fixed baseline
   configuration: **60 prior returns, a 5× shock threshold, and a 15-minute
   hold** — exactly the `(baseline_prior_returns=60, shock_multiple=5.0,
   hold_eligible_minutes=15)` cell of this family's grid.
3. It covered **EUR/USD and GBP/USD jointly**, under the **older Stage-G
   synchronized-frame research stack** (`LiquidityShockReversionDevelopmentFold`
   / `evaluate_frozen_development_candidate`), not the exact-grid
   `StrategyFamily`/`g1.search.run_search` engine this family uses.
4. Its frozen DEVELOPMENT record was **negative in all three folds**
   (annualized net Sharpe `-10.3922`, `-11.2375`, `-8.6647`; `0/3` positive
   folds; pooled mean daily net return `-0.0036285`; all three folds failed
   the 1.5× cost-stress check) and the candidate is **failed / retired**.
5. This prior evidence was **discovered only after
   `eurusd_liquidity_shock_reversion_v1`'s 36-cell grid had already been
   specified externally and frozen** (semantic SHA-256
   `50cd4e5460e488e06b33edd48fdb482a549199bcca4daa48625ce74da3865ad8`,
   computed and checked before this lineage note was written).
6. **No grid value, selector rule, eligibility rule, or family semantic was
   changed** after discovering the prior result — this note is
   documentation-only. The preregistration YAML's canonical document (and
   therefore its semantic SHA-256) is unaffected: the hash covers only
   `family`/`dataset`/`session`/`parameter_grid`/`signal`/
   `risk_normalization`/`execution_and_costs`/`development_folds`/
   `eligibility`/`sample_count`/`production_evaluator`/`metric_definitions`/
   `neighbours`/`selector`/`search`/`sealed_partitions`/`reuse`, none of
   which reference this markdown file or any lineage record.
7. The old result is therefore treated as **adverse prior lineage evidence,
   not ignored** — it raises the prior that this economic hypothesis may not
   survive realistic costs. It is not, however, evidence against this
   specific 36-cell grid, and it does not substitute for or bias this
   family's own DEVELOPMENT evaluation. This remains a **separately
   preregistered, EUR/USD-only G1 evaluation**, run through the current
   generic G1 engine rather than the older single-candidate stack.
8. The overlapping `(60, 5.0, 15)` cell **remains in the grid unchanged**:
   it was not removed, inverted, or otherwise modified because of the known
   old result. All 36 cells, including this one, are evaluated identically
   and only through the frozen exact-grid search and selector below.
9. **Any eventual interpretation of this family's DEVELOPMENT result must
   disclose this prior family-level testing history** — that a related,
   differently-scoped candidate using the same hypothesis at the `60/5×/15`
   baseline already failed DEVELOPMENT under the older evaluator, and that
   the `60/5×/15` cell in this grid is not an independent first test of that
   configuration.

This family otherwise reuses only the causal shock/reversion *signal logic*
pattern from
[`ftmoquant/strategies/liquidity_shock_reversion.py`](../../src/ftmoquant/strategies/liquidity_shock_reversion.py)
(median-absolute-return baseline, shock threshold, fixed hold, ignore-while-
positioned), generalized to an EUR/USD-only, three-dimensional exact grid
instead of one frozen baseline configuration. That candidate's real
DEVELOPMENT returns are historical record only; this family's
preregistration and grid were frozen without reading them, and nothing here
was tuned from them.

## Hypothesis

Very large short-horizon EUR/USD price changes can contain a temporary
liquidity/dislocation component that partially mean-reverts over the
following minutes after realistic transaction costs. No ML, no macro inputs,
no trend filter, and no session optimization are used in V1.

## Signal

On causal, completed, synchronized one-minute EUR/USD BID/ASK midpoint
closes, form `r_t = ln(C_t / C_t-1)` only over consecutive eligible closes
(no fill or interpolation; a gap resets the return chain). A shock is
`abs(r_t) > shock_multiple * median(abs(r_t-1), ..., abs(r_t-N))` over the
prior `N = baseline_prior_returns` eligible returns. A positive shock targets
short; a negative shock targets long. New shocks are ignored while a position
is open or an entry/exit is pending.

## Trade semantics

The raw target executes only at the first eligible information time strictly
after signal information time, native G0.7/Nautilus execution. It holds for
exactly `hold_eligible_minutes` later eligible completed minutes, then emits
flat at the first eligible information time strictly after that. At most one
position is open at a time; no pyramiding, stop-loss, or take-profit. Fold
adapters warm return/volatility history only in train and reset state at
fold boundaries, so no position crosses a fold boundary.

## Sample count

`FoldMetrics.trade_count` = **completed executed shock round trips**: one
entry plus its eventual exit is one evidential trade. Entries and exits are
not counted separately (the generic `ScaledTargetInstruction.
count_alpha_transition` flag is only set on the entry fill; the hold-period
flatten and any forced fold-end liquidation are excluded). Ignored shocks
while positioned, unexecuted signals, orders, fills, bars, and risk-only
rebalances are never counted.

## Risk sizing

Reuses the existing causal G1 underlying-volatility sizing:
`exposure = direction * 0.01 / causal_ex_ante_EURUSD_annualized_vol`, using
the same strictly-prior daily EWMA estimator already frozen for G1. Sizing
happens once at entry; because trades last only 5–30 minutes there is no
in-trade resizing. Exit means target exposure `= 0`. No FTMO sizing.

## Frozen exact 36-cell grid

| Dimension | Values |
|---|---|
| `baseline_prior_returns` | `30, 60, 120` |
| `shock_multiple` | `3.0, 4.0, 5.0, 6.0` |
| `hold_eligible_minutes` | `5, 15, 30` |
| session | `All` (fixed) |

`3 × 4 × 3 = 36` unconditional cells, exact-grid enumeration only (no
Optuna, no session optimization, no other filters).

## Production evaluator

`ftmoquant.research.eurusd_liquidity_shock_reversion_development` adapts the
existing causal state machine into a `StrategyFamily`
(`EurusdLiquidityShockReversionFamily`) and drives it through the same
generic G1 engine used by `eurusd_tsm_v1`:
`ftmoquant.research.g1.search.run_search`, `TrialRegistry`, `FoldMetrics`,
`ftmoquant.research.g1.selector.select_candidate`, and
`ftmoquant.research.g1.artifacts` for deterministic output. It reuses the
genuinely family-agnostic scaled-instruction shape, native-fold statistics,
and catalog/provenance validation already proven by
`eurusd_tsm_development.py` rather than duplicating the whole TSM evaluator;
only the small Nautilus execution-harness class and its DEVELOPMENT-fold
market-data loader (EUR/USD-only, 1-minute BID/ASK, no H1/H4) are
family-specific.

```console
uv run python -m ftmoquant.research.eurusd_liquidity_shock_reversion_development \
  --universe-readiness .artifacts/g1_4b/universe/ftmoquant_universe_readiness.json \
  --development-root 'EUR/USD.DUKASCOPY=PATH' \
  --development-root 'GBP/USD.DUKASCOPY=PATH' \
  --evaluation-config .artifacts/g1_4c/phase2_cost_models.json \
  --output .artifacts/g1_4h/eurusd_liquidity_shock_reversion_v1/development_run
```

## Eligibility (frozen before returns)

All 3 DEVELOPMENT folds present, `>= 100` pooled completed executed shock
round trips, pooled net expectancy `> 0`, pooled cost-stressed expectancy
`> 0`, at least 2 of 3 folds positive, and all numerical-validity checks
pass. No drawdown or FTMO hard gate.

## Neighbours / selector

Neighbours move exactly one ordered dimension one adjacent step at a time
over `[baseline_prior_returns, shock_multiple, hold_eligible_minutes]`. The
selector reuses the same robustness-oriented, non-weighted lexicographic
ranking as `eurusd_tsm_v1` (positive-fold fraction, plateau fraction, yearly
concentration, execution sensitivity, worst-fold expectancy, stressed
expectancy, pooled expectancy, drawdown, deterministic trial-ID tie-break).

## Seals

Validation is locked until one DEVELOPMENT candidate is frozen; the final
holdout remains locked. No DEVELOPMENT return has been accessed while
preregistering this family.
