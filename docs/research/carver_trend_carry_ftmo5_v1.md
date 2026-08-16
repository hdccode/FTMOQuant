# carver_trend_carry_ftmo5_v1 — preregistration

This candidate is specified only; it has no DEVELOPMENT result and no evaluator.
Validation and final holdout are sealed.

It uses the pinned pysystemtrade Chapter 15 signal structure at commit
`b4a25e6e1e33a54a3ecfb45c0f6db5e2b60b84f8`. The ten frozen source-file hashes,
five futures-to-CFD mappings, all trend/carry rule parameters, causal timing,
portfolio allocation, and DEVELOPMENT gate are machine-readable in
`config/strategies/carver_trend_carry_ftmo5_v1.yaml`.

Futures series are signal inputs only. EUR uses frozen Dukascopy BID/ASK prices;
GOLD, SP500, CRUDE_W, and SOYBEAN use OANDA BID/ASK prices solely as numeric
execution-price proxies. FTMO G0.8 remains the sole research P&L and deployment
economics layer: OANDA contract size, margin, financing, commission, leverage,
and other broker economics are excluded. The CFD/futures basis mismatch must be
reported and may not be assumed favourable. A later evaluator must use the
frozen G0.7 cost framework at base and 1.5× costs, existing FTMO constraints
where compatible, and must retire the candidate after a failing DEVELOPMENT gate
without tuning.

## Pre-evaluation source-ID correction

Before any DEVELOPMENT returns were accessed, the SOYBEAN Dukascopy numeric
execution-price proxy was corrected from `SOYBEAN.CMD/USD.DUKASCOPY` to the
official source identifier `SOYBEAN.CMD/USX.DUKASCOPY`. G0.8 FTMO P&L remains
anchored to `SOYBEAN.c`; this correction does not introduce a USX-to-USD
conversion or alter any strategy rule.

## Pre-evaluation execution-price provider switch

Before any OANDA historical price data or DEVELOPMENT returns were accessed, the
non-FX numeric BID/ASK execution-price proxies were preregistered as
`XAU_USD.OANDA`, `SPX500_USD.OANDA`, `WTICO_USD.OANDA`, and `SOYBN_USD.OANDA`.
OANDA v20 practice metadata confirmed those four instruments exist. EUR remains
`EUR/USD.DUKASCOPY`. This changes only the four source identifiers;
it does not import OANDA economics or alter trend, carry, weighting, FDM,
portfolio, causality, cost-stress, folds, or the promotion gate.
