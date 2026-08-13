# `trend_pullback_v1` G1.2 implementation record

Status: **implemented, not run**. Frozen strategy version: **1.0.0**.

This change implements the state machine preregistered in
`config/strategies/trend_pullback_v1.yaml`. It does not run an experiment,
inspect a return, access the sealed final holdout, add a parameter, or change
the experiment registry. The implementation refuses any strategy document
whose semantic SHA-256 differs from
`d21100fc8412f4d258efdc90b2f1a936c0eb27cd6f88081ae082f24ae6d4cc5e`.

## Runtime boundary

`TrendPullbackStrategy` is a thin Nautilus `Strategy` adapter. It subscribes to
the four G0.6 completed 1H/4H side streams and the two G0.5 external one-minute
execution streams. A fail-closed pairer admits only exact-timestamp BID/ASK
pairs, uses the later `ts_init` as information time, rejects invalid side
ordering, and never carries a missing side into another window. The strategy
also refuses to start unless the runner has established G0.6 research
readiness.

Signals use one midpoint observation per synchronized pair. Native
NautilusTrader `ExponentialMovingAverage` and `AverageTrueRange` instances own
indicator calculation, initialization state, and reset behavior. They are fed
once with midpoint OHLC through `update_raw`. Direct
`register_indicator_for_bars` is intentionally not used: registering against
BID or ASK would calculate the wrong price, while registering both would
double-update the indicator. No independent EMA or ATR implementation exists.

The state machine emits R-normalized market intents. G0.7 remains the sole
execution boundary for spread, slippage, latency, fees, rollover, and matching.
Stops and targets are bound to the actual native entry fill, remain fixed, and
are evaluated on the liquidation side (BID for longs, ASK for shorts). When a
single unresolved one-minute bar touches both, the stop is selected. Time exits
require 48 completed 1H pairs after the fill and the first later one-minute
pair. Signals are discarded while an entry or position is active; becoming
flat clears signal history so re-entry requires a new arm and trigger.

## Synthetic verification boundary

The G1.2 tests cover:

- frozen-config enforcement and native indicator initialization;
- warm-up without split-boundary trades;
- symmetric long/short arm, trigger, and strictly-later entry chronology;
- three-bar arm and five-minute entry expiry;
- gap reset and no side carry-forward;
- actual-fill fixed stop/target construction and one-R normalization;
- liquidation-side exits and conservative same-minute ambiguity;
- the 48-completed-hour time exit and post-flat re-entry reset; and
- the research-readiness start guard.

Only synthetic state-machine and adapter contract tests are permitted in G1.2.
The first strategy run remains exclusively G1.3 work.
