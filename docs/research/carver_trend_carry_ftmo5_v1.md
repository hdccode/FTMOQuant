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

## Pre-evaluation SOYBN quote-scale normalization

Before bulk DEVELOPMENT acquisition, the observed OANDA `SOYBN_USD` quote scale
(approximately `15.2`) and live FTMO `SOYBEAN.c` quote scale (approximately
`1180`) established a dollar-versus-cents difference. The frozen transformation
is applied only to raw `SOYBN_USD.OANDA` BID/ASK prices: multiply each by `100`
before G0.8 FTMO P&L or economics. No other execution proxy is rescaled, and
this does not change G0.8 or any strategy, portfolio, cost, causality, fold, or
promotion-gate semantics.

## DEVELOPMENT OANDA acquisition boundary

The OANDA acquisition command is restricted to the frozen half-open DEVELOPMENT
interval `2019-03-11T00:00:00Z` through `2023-04-11T00:00:00Z`. It acquires
only M1 BID/ASK candles for `XAU_USD`, `SPX500_USD`, `WTICO_USD`, and
`SOYBN_USD`; it does not request EUR, validation, or holdout data. Raw OANDA
JSON responses and redacted request metadata are immutable under `raw/`.
Unscaled, parsed BID/ASK OHLC rows are written separately under `processed/`,
and each instrument has a QA report under `qa/` recording ordering, duplicate,
BID/ASK, range-boundary, and missing-interval results. Missing minutes are
reported without filling or interpolation. This acquisition path computes no
signals, P&L, returns, or candidate-performance statistics.

### Cache-only availability QA correction

OANDA M1 candles are treated as returned price observations, not as a promise
of a synthetic candle for every wall-clock minute. QA therefore separates: (1)
proof that the 430 recorded request windows exactly partition DEVELOPMENT and
that every complete returned candle survives processing; (2) calendar minutes
for which OANDA returned no candle; and (3) strategy usability. Readiness
requires complete request/raw/processed provenance with zero acquisition
defects, plus compatibility with the frozen evaluator's existing rule of using
the first genuine eligible observation strictly later than the completed signal.
All non-observed minutes remain reported; none are filled or interpolated.
