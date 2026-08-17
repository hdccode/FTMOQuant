# Leo GBPUSD v1 preregistration

Status: **DEVELOPMENT_BURNED / `ALPHA_REJECTED`** (corrected 2026-08-17; see
"Result / lifecycle correction" below). The YAML at
`config/strategies/leo_gbpusd_v1.yaml` still literally reads
`status: specified_not_run` — **that field is frozen and must not be
edited**: `leo_gbpusd_config_sha256` hashes the entire YAML document, and
`load_leo_gbpusd_spec` additionally rejects any document that does not
byte-for-byte equal the hardcoded `_frozen_document()` in
`src/ftmoquant/research/leo_gbpusd_spec.py`, which itself pins
`"status": "specified_not_run"`. Changing the field would break both the
semantic hash (`5c89c825db340c49f10c18fdc3982b03f23ade718a5889e4e2ef662081abbf4c`)
and the loader. The authoritative current status is recorded here and in
[`LEO_GBPUSD_V1_OUTCOME`](../../src/ftmoquant/research/g1/outcomes.py).

## Result / lifecycle correction

Despite the preregistration text below stating that *"No strategy
implementation, backtest, development-return inspection, ... is authorized,"*
an implementation exists
(`src/ftmoquant/strategies/leo_gbpusd.py`,
`src/ftmoquant/research/leo_gbpusd_{spec,development,cache}.py`) and
**DEVELOPMENT was run twice**, both with real negative pooled results, at
code commit `86e8755fe7cdbe5df691ac898f7b1a024c5cef8e` ("Implement G1.4B
tournament infrastructure"):

- `.artifacts/g1_4f/leo_gbpusd_v1/development_run/manifest.json`: 1/3
  positive folds, worst-fold annualized net Sharpe `-1.568`, pooled mean
  daily net return `-8.68e-05`; no fold passes the 1.5× cost-stress check.
- `.artifacts/g1_4f/leo_gbpusd_v1/development_run_exit_fixed/manifest.json`:
  1/3 positive folds, worst-fold annualized net Sharpe `-1.133`, pooled mean
  daily net return `-1.166e-04`; one fold now passes cost stress. The
  `_exit_fixed` naming and differing results confirm an implementation
  change occurred **between** the two runs — i.e. DEVELOPMENT was rerun once
  on the same frozen DEVELOPMENT partition.

Both runs record `validation_accessed: false` and
`final_holdout_accessed: false` — **validation and final holdout remain
untouched**. Neither run ever produced a formal pass/fail decision record
(no postmortem, no `decision` field in either manifest); this correction is
the first explicit disposition. Given uniformly negative pooled evidence
across two independent implementations and no fold-set clearing the implicit
G1 bar, this family is recorded as **`ALPHA_REJECTED`**: it did not survive
DEVELOPMENT and was never promoted to validation. It must not be rerun a
third time without an explicit reason to believe the prior two runs do not
already settle the question, and it must not be "fixed" again based on this
already-observed evidence.

---

## Historical preregistration (frozen, unmodified below)

`leo_gbpusd_v1` is a frozen mechanical approximation of FTMO trader Leo's publicly disclosed GBPUSD methodology. It is an externally sourced hypothesis, not an optimization target and not a claim that this contract reproduces discretionary trading decisions.

The contract permits GBP/USD only, completed 15-minute bars, Europe/London session boundaries with IANA DST handling, and the explicit Asia/London reference and London/New York entry windows in [the strategy config](../../config/strategies/leo_gbpusd_v1.yaml). A rejection sweeps the selected completed-session extreme and closes back inside it; the trade is opposite the sweep. Stops use the sweep-bar extreme and targets are exactly 3R.

This entry is preregistered and unevaluated. No strategy implementation, backtest, development-return inspection, validation access, or final-holdout access is authorized by this preregistration.
