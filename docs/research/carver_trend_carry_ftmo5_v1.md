# carver_trend_carry_ftmo5_v1 — preregistration

This candidate has a complete preregistered DEVELOPMENT evaluator but no
DEVELOPMENT result. The final evaluator freeze was completed before any strategy
return was accessed. Validation and final holdout are sealed.

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
reported and may not be assumed favourable. The evaluator uses observed BID/ASK
spread at base and 1.5× realised-cost stress, the frozen G0.8 economics and
margin constraint, and must retire the candidate after a failing DEVELOPMENT
gate without tuning.

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

## Final pre-evaluation evaluator freeze

The evaluator contract is part of the strategy semantic document. The research
input contract preserves the OANDA cache's acquisition-time Carver SHA
`489b53ab...e210d`; the new semantic SHA adds evaluator decisions without
rewriting or relabelling that immutable raw/processed cache.

The research
account is fixed USD 500,000 notional capital with a 25% annual volatility
target. These values, the 256-business-day sizing year, forecast divided by the
average absolute forecast of 10, and the order instrument-weight then IDM come
from pinned pysystemtrade. Each of the five instruments has weight 0.20. The
five-market IDM is 1.0: the pinned `rob_system` value of 2.75 belongs to a much
larger universe and was not imported. This is a performance-blind neutral
convention, not an estimated diversification uplift.

Desired continuous FTMO lots are:

`(500000 * 0.25 / sqrt(256)) / (causal daily proxy price-unit volatility * G0.8 contract size) * (combined forecast / 10) * 0.20 * 1.0`.

G0.8's frozen continuous research-target convention overrides pysystemtrade's
default whole-futures-contract rounding. No lot quantization or deployment
minimum is applied. The volatility input is calculated from causal daily closes
of each numeric CFD execution proxy using the pinned mixed-volatility form.
The pinned `rob_system` backfill option is disabled because it would expose an
early row to later volatility; unavailable warm-up volatility produces no
target instead.
Aggregate G0.8 Swing margin after a proposed transition may not exceed the fixed
research capital; a breach fails closed and is never clipped or rescaled.
Challenge loss-limit optimization is outside this G1/core-edge evaluation.

Reference rows are aggregated by UTC date: adjusted price uses the last row and
annualised futures roll uses the day's mean. The signal is complete only at the
next UTC midnight, avoiding hindsight about which intraday row was final. There
is one rebalance per completed daily signal. A one-minute close is available at
candle start plus one minute. Buys use ASK close and sells use BID close at the
first genuine G0.8-session-eligible observation strictly later than signal
completion. No missing provider minute is synthesized, filled, or interpolated.
Because signals complete only at UTC midnight and marks occur at the next UTC
boundary, the in-memory evaluation view retains only each UTC date's first
session-eligible genuine observation and last genuine observation. Every source
row is still validated by the frozen cache QA; this deterministic downselection
does not create a price or alter which observation can execute or mark.
SOYBN ×100 remains confined to the immediate G0.8 boundary.

Position changes trade only the delta to the continuous desired lots. Observed
half-spread is embedded in the side fill and G0.8 commission is charged on every
executed delta side. Daily return is the change in total liquidation equity,
including realised and unrealised P&L, divided by fixed USD 500,000. The daily
mark uses the last genuine close available before UTC midnight and values open
longs at BID and shorts at ASK. Each fold runs an independent account through
its expanding warm-up, scores only its comparison window, and virtually
liquidates remaining positions at its final pre-boundary genuine observation,
including exit commission. Rollover is explicitly unmodelled, so every artifact
is pre-rollover and not fully deployment-calibrated.

Base cost is observed spread plus G0.8 commission. The 1.5× result subtracts an
additional half of that realised cost. Turnover is absolute traded G0.8 notional
divided by fixed capital; daily Sharpe uses sqrt(252), and drawdown compounds the
sequential fixed-capital daily return series. Per-instrument contribution is
cumulative instrument net P&L divided by capital plus its share of pooled net
P&L. Trend/carry attribution is not reported: the frozen gate did not require it
and the combined forecast executes as one position, so an additive cost
allocation would be a new ambiguous assumption.

The pooled mean CI reuses FTMOQuant's `arch==8.0.0` stationary bootstrap:
two-sided 95% basic interval, block size 20 daily observations, 10,000
repetitions, and seed 14042026. The gate is purely mechanical: positive pooled
mean, at least two positive fold means, positive median fold mean, positive
pooled mean under 1.5× costs, and no implementation/data-integrity failure.

Deterministic artifacts are `daily_returns.csv`, `trades.csv`, and `result.json`.
The wall-clock execution timestamp lives only in `run_provenance.json`, so the
result files and semantic result hash are reproducible from identical frozen
inputs. The CLI exposes DEVELOPMENT inputs only and rejects any path containing
`validation` or `holdout`.
