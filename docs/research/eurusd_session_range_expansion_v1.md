# eurusd_session_range_expansion_v1 — preregistration

Status: `preregistered_not_run`. The authoritative machine-readable source of
truth is
[`config/strategies/eurusd_session_range_expansion_v1.yaml`](../../config/strategies/eurusd_session_range_expansion_v1.yaml),
with semantic SHA-256
`3fc1fd836fdfe6999a0ff370e7285752768078e70dca9547723a5772f8a16585`. No
DEVELOPMENT return, validation row, or final-holdout row was accessed while
freezing it.

## Context

This family follows the closure of `eurusd_liquidity_shock_reversion_v1`
(`ALPHA_REJECTED`; 36/36 complete, 0/36 profitable, 0/36 stressed-profitable,
0/36 eligible, no candidate selected — see
[`eurusd_liquidity_shock_reversion_v1.md`](eurusd_liquidity_shock_reversion_v1.md)
and `EURUSD_LIQUIDITY_SHOCK_REVERSION_V1_OUTCOME` in
[`g1/outcomes.py`](../../src/ftmoquant/research/g1/outcomes.py)). No V2 was
created and the signal was not inverted; this is a new, independently
preregistered family.

## Prior lineage: `session_range_expansion_v1` (adverse prior evidence)

Before designing this family's grid, the existing implementation
(`src/ftmoquant/strategies/session_range_expansion.py`,
`config/strategies/session_range_expansion_v1.yaml`) and the repository's
committed docs/artifacts were audited for prior returns. **Real historical
DEVELOPMENT returns for this hypothesis already exist and have already been
accessed** — this is not a virgin hypothesis:

1. A **prior, retired candidate named `session_range_expansion_v1`** already
   exists — see
   [`docs/research/session_range_expansion_v1.md`](session_range_expansion_v1.md)
   and the frozen record at
   `.artifacts/g1_4d/session_range_expansion_v1/development/manifest.json`.
2. It used the **same core hypothesis at one fixed baseline configuration**:
   range window `00:00-08:00` London, breakout window `08:00-12:00` London,
   scheduled exit `16:00` London — exactly the `(breakout_window_end=12:00,
   scheduled_exit=16:00)` cell of this family's grid.
3. It covered **EUR/USD and GBP/USD jointly**, under the **older Stage-G
   synchronized-frame research stack** (`SessionRangeExpansionDevelopmentFold`
   / `evaluate_frozen_development_candidate`), not the exact-grid
   `StrategyFamily`/`g1.search.run_search` engine this family uses.
4. Its frozen DEVELOPMENT record was **mixed but net-adverse**: only 2 of 3
   folds were positive (fold Sharpes `+0.1035`, `+1.6411`, `-0.3672`), two of
   three folds failed the fixed 1.5× cost-stress check, the pooled
   stationary-bootstrap 95% confidence interval for mean daily net return
   crossed zero (`[-0.000363, 0.000525]`), and SPA did not reject the
   zero-return benchmark (p=0.3763). The candidate is **failed / retired**
   and did not advance to validation.
5. This prior evidence was **discovered only after this task specified the
   candidate dimensions and grid size**; the range window (`00:00-08:00`)
   and breakout window (`08:00-12:00`) were fixed directly from the given
   hypothesis text, and the `breakout_window_end`/`scheduled_exit` grid
   values were chosen as the baseline plus one adjacent hour on each side —
   a structural, return-blind choice, not a search informed by the prior
   result.
6. **No grid value, selector rule, eligibility rule, or family semantic was
   changed** after reviewing the prior result. The overlapping `(12:00,
   16:00)` cell **remains in the grid unchanged** — not removed, inverted,
   or otherwise modified.
7. The old result is therefore treated as **adverse-leaning prior lineage
   evidence, not ignored** — it is mixed (2/3 folds positive) rather than
   uniformly negative, but it did not survive its own frozen decision
   criteria and does not substitute for or bias this family's own
   DEVELOPMENT evaluation. This remains a **separately preregistered,
   EUR/USD-only G1 evaluation**, run through the current generic G1 engine
   rather than the older single-candidate stack.
8. **Any eventual interpretation of this family's DEVELOPMENT result must
   disclose this prior family-level testing history** — that a related,
   differently-scoped candidate using the same hypothesis at the `12:00/16:00`
   baseline already ran DEVELOPMENT under the older evaluator with a mixed,
   ultimately non-advancing result, and that the `(12:00, 16:00)` cell in
   this grid is not an independent first test of that configuration.

## Hypothesis

The completed London overnight range (`00:00 <= Europe/London time < 08:00`)
marks an illiquid, low-participation reference band for EUR/USD. A causal
breakout beyond that band during the opening hours of the London session
(`08:00 <= Europe/London time < 12:00`, baseline) signals a directional
expansion in participation that persists into the afternoon after realistic
transaction costs. No ML, no macro inputs, no joint GBP/USD optimization in
V1.

## Signal

For EUR/USD only, on causal, completed, synchronized one-minute BID/ASK
midpoint closes: the completed London overnight range is the max/min of
exactly 480 consecutive completed 1-minute midpoints whose London local time
falls in `[00:00, 08:00)`. An incomplete or invalid range (a gap, an
invalid price, or a count `!= 480`) makes that day a no-trade day. During
`[08:00, breakout_window_end)` London time, the first completed close
strictly above the range high targets long; the first strictly below the
range low targets short. At most one entry per London calendar day. New
breakouts are ignored while a position is open or an entry/exit is pending.

## Trade semantics

The raw target executes only at the first eligible information time strictly
after signal information time, native G0.7/Nautilus execution. It holds
until the scheduled exit (`scheduled_exit` London time), which emits flat at
the first eligible information time strictly after the first observed close
at or after that time. At most one position at a time; no pyramiding,
stop-loss, or take-profit. Fold adapters reset all state (range, pending,
active) at each fold boundary, so no range or position crosses a fold
boundary.

## Sample count

`FoldMetrics.trade_count` = **completed executed session-breakout round
trips**: one entry plus its eventual scheduled exit is one evidential trade.
Entries and exits are not counted separately (the generic
`ScaledTargetInstruction.count_alpha_transition` flag is only set on the
entry fill; the scheduled-exit flatten and any forced fold-end liquidation
are excluded). Ignored second breakouts, unexecuted signals, orders, fills,
and bars are never counted.

## Risk sizing

Reuses the existing causal G1 underlying-volatility sizing:
`exposure = direction * 0.01 / causal_ex_ante_EURUSD_annualized_vol`, using
the same strictly-prior daily EWMA estimator already frozen for G1. Sizing
happens once at entry; no in-trade resizing. Exit means target exposure
`= 0`. No FTMO sizing.

## Frozen exact 9-cell grid

| Dimension | Values |
|---|---|
| `range_start` (fixed) | `00:00` London |
| `range_end` (fixed) | `08:00` London |
| `breakout_window_end` | `11:00, 12:00, 13:00` London |
| `scheduled_exit` | `15:00, 16:00, 17:00` London |
| session | `All` (fixed) |

`3 x 3 = 9` unconditional cells. `range_start`/`range_end` are held fixed at
the frozen hypothesis (the overnight range definition itself) and are not
searched — only `breakout_window_end` and `scheduled_exit` vary, each over
the retired baseline value plus one adjacent hour on either side. This is
deliberately a small baseline-plus-neighbours grid, not a broad search: the
existing baseline is already a defensible single hypothesis, so only the two
most obvious, return-blind structural robustness dimensions were added.

## Production evaluator

`ftmoquant.research.eurusd_session_range_expansion_development` adapts the
existing causal state machine into a `StrategyFamily`
(`EurusdSessionRangeExpansionFamily`) and drives it through the same generic
G1 engine used by `eurusd_tsm_v1` and `eurusd_liquidity_shock_reversion_v1`:
`ftmoquant.research.g1.search.run_search`, `TrialRegistry`, `FoldMetrics`,
`ftmoquant.research.g1.selector.select_candidate`, and
`ftmoquant.research.g1.artifacts` for deterministic output. It reuses the
genuinely family-agnostic scaled-instruction shape, native-fold statistics,
and catalog/provenance validation already proven by
`eurusd_tsm_development.py` rather than building another backtesting engine;
only the small Nautilus execution-harness class and its DEVELOPMENT-fold
market-data loader (EUR/USD-only, 1-minute BID/ASK) are family-specific.

```console
uv run python -m ftmoquant.research.eurusd_session_range_expansion_development \
  --universe-readiness .artifacts/g1_4b/universe/ftmoquant_universe_readiness.json \
  --development-root 'EUR/USD.DUKASCOPY=PATH' \
  --development-root 'GBP/USD.DUKASCOPY=PATH' \
  --evaluation-config .artifacts/g1_4c/phase2_cost_models.json \
  --output .artifacts/g1_4h/eurusd_session_range_expansion_v1/development_run
```

## Eligibility (frozen before returns)

All 3 DEVELOPMENT folds present, `>= 100` pooled completed executed
session-breakout round trips (the pooled sample threshold is reused
unchanged from the existing G1 tournament minimum, chosen before any return
was observed), pooled net expectancy `> 0`, pooled cost-stressed expectancy
`> 0`, at least 2 of 3 folds positive, and all numerical-validity checks
pass. No drawdown or FTMO hard gate.

## Neighbours / selector

Neighbours move exactly one ordered dimension one adjacent step at a time
over `[breakout_window_end, scheduled_exit]`. The selector reuses the same
robustness-oriented, non-weighted lexicographic ranking as the prior two
families (positive-fold fraction, plateau fraction, yearly concentration,
execution sensitivity, worst-fold expectancy, stressed expectancy, pooled
expectancy, drawdown, deterministic trial-ID tie-break).

## Seals

Validation is locked until one DEVELOPMENT candidate is frozen; the final
holdout remains locked. No DEVELOPMENT return has been accessed while
preregistering this family.
