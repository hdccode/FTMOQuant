# Strategy lineage inventory

**Purpose:** the authoritative, evidence-based (not status-field-based)
record of which strategy candidates have already had DEVELOPMENT, validation,
or final-holdout returns observed. Built by inspecting git history and local
`.artifacts/` output directly, because repository lifecycle fields
(`status: specified_not_run`, `not_evaluated`, etc.) have repeatedly been
found stale relative to what was actually executed.

**This inventory must be checked BEFORE any future family is preregistered,
implemented, or run.** It does not rank strategies by performance and makes
no recommendation about which hypothesis to pursue next.

Audited 2026-08-17. Scope: every `config/strategies/*.yaml` file in the
repository (10 total), cross-checked against `docs/research/`,
`src/ftmoquant/strategies/`, `experiments/registry.csv`, `.artifacts/`, and
`git log`.

## Summary table

| strategy_id | declared status (file) | actual evidence | DEVELOPMENT accessed | validation accessed | final holdout accessed | disposition |
|---|---|---|:---:|:---:|:---:|---|
| `carver_trend_carry_ftmo5_v1` | (see own doc) | blocked pre-alpha on a frozen margin constraint | No | No | No | RETIRED (`DEPLOYMENT_FEASIBILITY_BLOCKED`) |
| `eurusd_tsm_v1` | (see own doc) | 90-cell DEVELOPMENT grid + one-shot validation, both real | Yes | Yes | No | RETIRED (`VALIDATION_REJECTED`) |
| `liquidity_shock_reversion_v1` (joint EUR/USD+GBP/USD, older Stage-G stack) | `implemented_not_evaluated` | real single-baseline DEVELOPMENT run, all 3 folds negative | Yes | No | No | RETIRED (development-failed) |
| `eurusd_liquidity_shock_reversion_v1` (EUR/USD-only, 36-cell) | (see own doc) | real 36-cell DEVELOPMENT grid, 0/36 eligible | Yes | No | No | RETIRED (`ALPHA_REJECTED`) |
| `session_range_expansion_v1` (joint, older Stage-G stack) | `implemented_not_evaluated` | real single-baseline DEVELOPMENT run, mixed 2/3 positive folds, failed decision gates | Yes | No | No | RETIRED (development-failed) |
| `eurusd_session_range_expansion_v1` (EUR/USD-only, 9-cell) | (see own doc) | real 9-cell DEVELOPMENT grid (9/9 eligible) + one-shot validation | Yes | Yes | No | RETIRED (`VALIDATION_REJECTED`) |
| `trend_pullback_v1` | YAML frozen at `specified_not_run` (see note below) | real G1.3 baseline: DEVELOPMENT (460 trades) **and** validation (154 trades), both FAIL | Yes | Yes | No | RETIRED (`RETAINED_ONLY_AS_RESEARCH_REFERENCE`), just corrected this task |
| `leo_gbpusd_v1` | YAML frozen at `specified_not_run` (doc corrected, see note below) | two real DEVELOPMENT runs, both pooled-negative, 1/3 positive fold each; implementation changed between runs | **Yes (rerun once)** | No | No | RETIRED (`ALPHA_REJECTED`), corrected this task |
| `ts_momentum_v1` | YAML frozen at `implemented_not_evaluated` (doc corrected, see note below) | one real DEVELOPMENT run, 1/3 positive fold, pooled mean daily net return negative | **Yes** | No | No | RETIRED (`ALPHA_REJECTED`), corrected this task |
| `usd_macro_surprise_momentum_v1` | `preregistered_not_evaluated`; doc says *"has no results"* | **stale/false.** Real DEVELOPMENT run exists at `.artifacts/g1_4g/usd_macro_surprise_momentum_v1/development_run/`, with an explicit recorded `"decision": "REJECT_RETIRE"` inside the artifact itself | **Yes** | No | No | **DEVELOPMENT_BURNED, effectively RETIRED** (decision already recorded in-artifact; doc/config never updated to reflect it) |

## Detailed evidence per newly-audited candidate

### `leo_gbpusd_v1`
- Declared: `config/strategies/leo_gbpusd_v1.yaml:4` → `status: specified_not_run`; `docs/research/leo_gbpusd_v1.md` explicitly claims no implementation/backtest/DEVELOPMENT/validation/holdout access is authorized.
- Actual: `src/ftmoquant/strategies/leo_gbpusd.py`, `src/ftmoquant/research/leo_gbpusd_{spec,development,cache}.py` all exist and are implemented.
- Two real DEVELOPMENT runs on disk (both at code commit `86e8755fe7cdbe5df691ac898f7b1a024c5cef8e`, "Implement G1.4B tournament infrastructure"):
  - `development_run/manifest.json`: `positive_fold_count: 1/3`, worst fold annualized net Sharpe `-1.568`, pooled mean daily net return `-8.68e-05`, no fold passes 1.5× cost stress.
  - `development_run_exit_fixed/manifest.json`: `positive_fold_count: 1/3`, worst fold Sharpe `-1.133`, pooled mean daily net return `-1.166e-04`; one fold now passes cost stress. The `_exit_fixed` naming and differing results indicate an implementation change between the two runs — i.e., **DEVELOPMENT was rerun**, which is itself lineage-relevant (repeated exposure to the same DEVELOPMENT partition under different code).
- `provenance.validation_accessed: false` and `final_holdout_accessed: false` in both manifests.
- No `decision`/`verdict` field is recorded in either manifest, and no postmortem exists. This candidate has real, negative DEVELOPMENT evidence on record but no formal closure — it should not be treated as virgin, and should not be rerun again without an explicit decision on the existing evidence.

### `ts_momentum_v1`
- Declared: `config/strategies/ts_momentum_v1.yaml:4` → `status: implemented_not_evaluated`; `docs/research/ts_momentum_v1.md` states *"No strategy return has been read"* and *"The real command has not been run."*
- Actual: `.artifacts/g1_4c/ts_momentum_v1/development/manifest.json` is a real, complete DEVELOPMENT result (code commit `86e8755fe7cdbe5df691ac898f7b1a024c5cef8e`): `positive_fold_count: 1/3`, worst fold annualized net Sharpe `-1.568`, pooled mean daily net return `-8.68e-05`, `validation_accessed: false`, `final_holdout_accessed: false`.
- No `decision`/`verdict` field recorded; no postmortem exists.

### `usd_macro_surprise_momentum_v1`
- Declared: `config/strategies/usd_macro_surprise_momentum_v1.yaml:4` → `status: preregistered_not_evaluated`; doc states *"has no results and does not authorize validation or final-holdout access."*
- Actual: `.artifacts/g1_4g/usd_macro_surprise_momentum_v1/development_run/manifest.json` is a real, complete DEVELOPMENT result with an event-level bootstrap CI, `positive_fold_count: 2/3`, pooled mean net event return `1.76e-05` (essentially flat; 95% CI `[-0.000606, 0.000654]` crosses zero), and — unlike the other two — an **explicit recorded decision: `"decision": "REJECT_RETIRE"`**. `validation_accessed: false`, `final_holdout_accessed: false`.
- This candidate is effectively already retired by its own recorded evidence; only the doc/config lifecycle fields were never updated to say so (the same staleness pattern found everywhere else in this audit).

## What was and was not corrected

`trend_pullback_v1`, `leo_gbpusd_v1`, and `ts_momentum_v1` have all had their
docs (`docs/research/{trend_pullback_v1,leo_gbpusd_v1,ts_momentum_v1}.md`)
and canonical outcome records (`g1/outcomes.py`) corrected to state their
true, already-observed disposition. In every case the frozen preregistration
YAML itself (and therefore its semantic hash) was left byte-for-byte
unmodified — `status` fields there are part of the hashed contract and
double-gated by hardcoded loader checks, so they intentionally still read
their original pre-run values.

`experiments/registry.csv` was corrected only for `trend_pullback_v1`. Its
row schema fixes `primary_metric` to `validation_mean_net_r_per_trade`, and
`leo_gbpusd_v1` never had validation accessed (only DEVELOPMENT) — there is
no real validation-stage number to populate, and inventing one would violate
"do not invent unavailable fields." Its registry row therefore correctly
remains `specified_not_run` / `not_evaluated` from that specific metric's
point of view, even though DEVELOPMENT was genuinely run twice; the true
disposition lives in the doc and `g1/outcomes.py` instead.
`ts_momentum_v1` is not a valid `strategy_id` in the registry's schema at all
(`_validate_registry_entry` only accepts `trend_pullback_v1`/`leo_gbpusd_v1`)
and was correctly left out of that file.

`usd_macro_surprise_momentum_v1` was audited and classified only — its
docs/config were intentionally left untouched, since its retirement decision
is already recorded machine-readably inside its own DEVELOPMENT artifact
(`"decision": "REJECT_RETIRE"`) and correcting its doc/config was outside
this task's scope.

## Genuinely eligible candidates for prospective research

**None.** Every strategy family present in `config/strategies/` has already
had real DEVELOPMENT returns observed at least once (and three of them —
`eurusd_tsm_v1`, `eurusd_session_range_expansion_v1`, `trend_pullback_v1` —
have also had validation returns observed). There is currently no
preregistered-but-untouched candidate available to select for the next
experiment; any next family would require genuinely new design work, which
is explicitly out of scope for this task.
