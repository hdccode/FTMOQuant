# Leo GBPUSD v1 preregistration

`leo_gbpusd_v1` is a frozen mechanical approximation of FTMO trader Leo's publicly disclosed GBPUSD methodology. It is an externally sourced hypothesis, not an optimization target and not a claim that this contract reproduces discretionary trading decisions.

The contract permits GBP/USD only, completed 15-minute bars, Europe/London session boundaries with IANA DST handling, and the explicit Asia/London reference and London/New York entry windows in [the strategy config](../../config/strategies/leo_gbpusd_v1.yaml). A rejection sweeps the selected completed-session extreme and closes back inside it; the trade is opposite the sweep. Stops use the sweep-bar extreme and targets are exactly 3R.

This entry is preregistered and unevaluated. No strategy implementation, backtest, development-return inspection, validation access, or final-holdout access is authorized by this preregistration.
