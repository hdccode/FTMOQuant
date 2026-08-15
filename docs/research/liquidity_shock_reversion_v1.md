# liquidity_shock_reversion_v1 — Phase 1 preregistration

Status: **DEVELOPMENT failed / retired**. The machine-readable source of truth
is `config/strategies/liquidity_shock_reversion_v1.yaml`, semantic SHA-256
`55a2a1814a507ebf53f5713d9672cf5d4fda19790d9c808efc0a9773bed27acd`.

The candidate is restricted to EUR/USD and GBP/USD independently on causal,
synchronized, completed one-minute midpoint closes.  It forms `r_t = ln(C_t /
C_(t-1))`; a shock is strictly greater than five times the median absolute
value of the prior 60 eligible returns.  Positive shocks target short and
negative shocks target long.

The raw target executes only at the first eligible Stage G point strictly after
signal information time.  It holds for exactly 15 later eligible completed
minutes, then emits flat at the first eligible point strictly after that 15th
minute.  Each instrument has at most one open/pending position; overlapping
shocks are ignored.  Fold adapters warm history only in train, admit comparison
signals only in their own fold, and reset state at a fold boundary.

This is alpha research only. Stage G retains synchronization, execution, costs,
normalization, exposure and portfolio constraints. No tuning, strategy change,
validation access, final-holdout access, or FTMO optimization followed.

## DEVELOPMENT decision record

The frozen DEVELOPMENT result is **failed / retired** and does not advance to
validation or final holdout. All three folds were negative: Sharpes were
`-10.3922`, `-11.2375`, and `-8.6647` (positive folds: `0/3`). The pooled mean
daily net return was `-0.0036285070785070784`; its 95% stationary-bootstrap CI
was `[-0.004131127734877734, -0.003078829794079794]`. All three folds failed
the fixed 1.5× cost-stress check, and SPA found no superior model. There were
no exposure-limit breaches or infrastructure hard failures.

This is a terminal DEVELOPMENT record only: it introduces no post-hoc tuning,
strategy inversion, parameter change, validation use, final-holdout use, or
FTMO optimization.
