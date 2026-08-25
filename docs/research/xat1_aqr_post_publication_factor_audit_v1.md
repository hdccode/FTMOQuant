# XAT1 AQR post-publication factor audit v1

## Pre-analysis governance and holdout consequence

This audit evaluates only AQR's already-published monthly Time Series Momentum
factor returns. It is a zero-cost viability screen, not an FTMOQuant strategy
backtest. It does not construct a trend signal, inspect any FTMOQuant strategy
performance, or access existing validation or holdout artifacts.

Inspecting AQR factor returns through the latest available 2026 observation
means that calendar years overlapping 2020--2025 can no longer honestly be
described as a completely pristine `HYPOTHESIS` holdout for the same broad
cross-asset time-series-momentum phenomenon. Those years may remain unseen for
a future, specific FTMOQuant implementation, but they are no longer unseen
evidence about cross-asset TSMOM generally. No existing sealed performance
artifact is modified by this audit.

This note and the frozen specification below were recorded before the return
workbooks were downloaded or their performance was inspected on 2026-08-25.

## Frozen specification

The primary factor is AQR's supplied all-asset TSMOM factor. The supplied
equity-index, government-bond, currency, and commodity factors are breadth
diagnostics and may not replace the primary factor.

Periods are fixed as follows:

- `SOURCE_SAMPLE`: 1985-01 through 2009-12.
- `BRIDGE`: 2010-01 through 2012-12.
- `STRICT_POST_PUBLICATION`: 2013-01 through the latest available month.
- `EARLY_POST_PUBLICATION`: 2013-01 through 2019-12.
- `RECENT`: 2020-01 through the latest available month.

The cost-hurdle screen subtracts annual drag divided by 12 from every monthly
return. Annual drags are fixed at 0%, 1%, 2%, and 3%. These are deterministic
screening hurdles, not estimates of realized implementation costs. The 2%
hurdle is primary and the 3% hurdle is the severe stress.

The stationary bootstrap is fixed at 10,000 resamples, expected block length
12 months, seed 20260825, and a one-sided 90% percentile lower confidence bound
for the arithmetic monthly mean. Reported mean bounds are annualized by 12.

XAT1 passes only if all six gates pass:

1. `STRICT_POST_PUBLICATION` all-asset arithmetic annualized mean after the 2%
   hurdle is greater than zero.
2. `EARLY_POST_PUBLICATION` all-asset arithmetic annualized mean after the 2%
   hurdle is greater than zero.
3. `RECENT` all-asset arithmetic annualized mean after the 2% hurdle is greater
   than zero.
4. `STRICT_POST_PUBLICATION` all-asset arithmetic annualized mean after the 3%
   hurdle is greater than zero.
5. At least three of four asset-class factors have positive
   `STRICT_POST_PUBLICATION` arithmetic annualized mean after the 2% hurdle.
6. The one-sided 90% stationary-bootstrap lower bound for the
   `STRICT_POST_PUBLICATION` all-asset arithmetic mean after the 2% hurdle is
   greater than zero.

The deterministic decision is
`PROCEED_FREE_IMPLEMENTATION_FEASIBILITY` only if all gates pass,
`RETIRE_XAT1` if any gate fails, and `DATA_OR_DEFINITION_FAILURE` only when a
genuine official-source or schema ambiguity prevents the test.

## Results

### Decision

**B. `RETIRE_XAT1`.** Gates 1--5 pass, but the frozen statistical Gate 6
fails. The strict-post-publication all-assets arithmetic mean is 3.37% after
the primary 2% annual hurdle, while its one-sided 90% stationary-bootstrap
lower bound is -1.02%, not greater than zero. No alternative date, asset
class, trend rule, hurdle, or bootstrap was considered.

The complete write-once machine-readable result is
`.artifacts/xat1_aqr_post_publication_factor_audit_v1/33470930e2269c0d/audit.json`
(SHA-256
`caeacde478b5defb6193f851790720c439f99de63167d7e77064c91fbb930cb9`).
It contains every factor/period/hurdle combination, all calendar-year returns,
all rolling 36-month observations, correlation matrices, and the complete list
of reconciliation difference months.

### Official source identity

Both files came from the direct download links embedded in AQR's official
dataset pages and remain outside Git under
`/Users/Shared/FTMOQuant-data/xat1_aqr_tsmom_v1/`.

| File | Downloaded UTC | Bytes | SHA-256 |
|---|---:|---:|---|
| `Time-Series-Momentum-Factors-Monthly.xlsx` | 2026-08-25 16:06:47 | 139,830 | `33470930e2269c0d97be4732ec2d9c27ddbc69ac8133b059a263e27400263eeb` |
| `Time-Series-Momentum-Original-Paper-Data.xlsx` | 2026-08-25 16:06:53 | 40,014 | `91bdbae6366ccb0693581b690236dc14862562a98ee83052c4f440f8b6ae0db8` |

The updated file is the AQR page version dated 2026-05-29. It has 497
complete monthly observations from 1985-01 through **2026-05**. Its raw date
labels run from 1985-01-31 to 2026-05-29 and are normalized to calendar month.
There are no missing values, missing months, or duplicate months.

The updated tabs are `TSMOM Factors`, `Definitions`, `Data Sources`, and
`Disclosures`. The original workbook has `TSMOM factors` and `Disclosures`.
The supplied updated columns are exactly:

- `TSMOM`: all assets, the sole primary factor;
- `TSMOM^EQ`: equity indices;
- `TSMOM^FX`: currencies;
- `TSMOM^FI`: fixed income / government bonds; and
- `TSMOM^CM`: commodities.

The updated values are stored as decimals and formatted as percentages. The
original file stores decimal values with a decimal display format. AQR calls
the series monthly excess returns of long/short factors, based on a 12-month
TSMOM rule, one-month holding period, and 58 liquid instruments. AQR does not
explicitly identify these workbook returns as gross or net of transaction
costs and supplies no separate net-of-cost series. The 0%--3% deductions below
therefore remain cost hurdles, not execution-cost estimates.

The disclosure tab says the information is informational, not advice, is not
warranted for accuracy, and past performance does not indicate future results.
AQR's website terms restrict reproduction and distribution without consent;
accordingly the raw workbooks were not committed.

### Original-paper reconciliation

The exact 1985-01 through 2009-12 overlap contains 300 months. No factor is
exactly equal: all 300 months differ for every factor. This is consistent with
AQR's workbook statement that the updated construction can differ in sources
and methodology and that AQR reconstructs the full history on each update.
The high correlations support continuity of identity; the updated file is not
rejected merely because its history was revised.

| Factor | Exact equal? | Differing months | Max absolute difference | Correlation |
|---|---:|---:|---:|---:|
| All assets | No | 300/300 | 4.579% | 0.9744 |
| Equities | No | 300/300 | 4.754% | 0.9989 |
| Currencies | No | 300/300 | 3.534% | 0.9925 |
| Fixed income | No | 300/300 | 15.015% | 0.9746 |
| Commodities | No | 300/300 | 10.023% | 0.9312 |

### Gross period results

`Mean`, `Geo`, `Vol`, `Pos`, `MDD`, `Worst`, `Best`, and bootstrap bounds are
returns; percent figures are shown in percentage units. `CI90` is the
annualized one-sided-90%-equivalent lower/upper percentile interval for the
arithmetic mean. `Growth` is terminal growth of $1. The complete artifact also
reports every corresponding result after the 1%, 2%, and 3% hurdles.

#### SOURCE_SAMPLE: 1985-01 through 2009-12

| Factor | N | Mean | Geo | Vol | Sharpe | Pos | MDD | Skew | Worst | Best | Growth | CI90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| All assets | 300 | 16.84% | 17.39% | 11.93% | 1.411 | 67.33% | -15.15% | -0.069 | 1987-10 (-10.48%) | 1990-08 (12.02%) | 55.03 | [14.55%, 19.11%] |
| Equities | 300 | 23.39% | 21.27% | 28.17% | 0.830 | 60.33% | -47.37% | -0.078 | 1987-10 (-34.77%) | 1986-03 (31.67%) | 124.02 | [15.46%, 31.41%] |
| Currencies | 300 | 14.46% | 13.55% | 18.44% | 0.784 | 60.00% | -28.50% | -0.033 | 1989-02 (-17.64%) | 2008-10 (19.84%) | 23.99 | [10.52%, 18.44%] |
| Fixed income | 300 | 20.92% | 17.84% | 29.66% | 0.705 | 59.00% | -54.89% | -0.136 | 1994-02 (-25.87%) | 1998-09 (28.86%) | 60.54 | [13.64%, 28.36%] |
| Commodities | 300 | 14.08% | 13.94% | 13.92% | 1.012 | 64.67% | -19.58% | 0.038 | 1999-03 (-9.54%) | 2008-02 (16.18%) | 26.14 | [11.58%, 16.51%] |

#### BRIDGE: 2010-01 through 2012-12

| Factor | N | Mean | Geo | Vol | Sharpe | Pos | MDD | Skew | Worst | Best | Growth | CI90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| All assets | 36 | 4.71% | 3.94% | 13.14% | 0.359 | 52.78% | -11.40% | -0.043 | 2011-09 (-9.37%) | 2012-05 (10.70%) | 1.12 | [-0.77%, 10.26%] |
| Equities | 36 | -8.18% | -10.20% | 22.71% | -0.360 | 47.22% | -37.05% | 0.039 | 2010-05 (-15.50%) | 2010-03 (15.03%) | 0.72 | [-18.75%, 2.06%] |
| Currencies | 36 | 2.14% | 0.77% | 16.73% | 0.128 | 44.44% | -22.77% | -0.086 | 2011-09 (-13.97%) | 2012-05 (12.35%) | 1.02 | [-5.97%, 10.12%] |
| Fixed income | 36 | 32.34% | 33.78% | 25.03% | 1.292 | 52.78% | -25.99% | 0.758 | 2012-06 (-7.65%) | 2010-08 (21.25%) | 2.39 | [19.88%, 44.67%] |
| Commodities | 36 | -4.13% | -5.71% | 18.42% | -0.224 | 52.78% | -31.57% | -1.234 | 2011-09 (-19.41%) | 2010-12 (8.73%) | 0.84 | [-15.16%, 7.25%] |

#### STRICT_POST_PUBLICATION: 2013-01 through 2026-05

| Factor | N | Mean | Geo | Vol | Sharpe | Pos | MDD | Skew | Worst | Best | Growth | CI90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| All assets | 161 | 5.37% | 4.62% | 13.08% | 0.411 | 51.55% | -27.91% | 0.197 | 2021-11 (-11.47%) | 2015-01 (12.96%) | 1.83 | [0.98%, 9.88%] |
| Equities | 161 | 4.69% | 0.89% | 27.00% | 0.174 | 59.01% | -80.46% | -0.779 | 2020-02 (-26.80%) | 2023-11 (20.28%) | 1.13 | [-6.62%, 16.22%] |
| Currencies | 161 | 3.73% | 2.08% | 18.45% | 0.202 | 55.90% | -47.10% | 0.517 | 2021-11 (-13.17%) | 2015-01 (24.42%) | 1.32 | [-2.00%, 9.88%] |
| Fixed income | 161 | 8.98% | 4.65% | 30.00% | 0.299 | 52.17% | -56.11% | 0.259 | 2021-02 (-23.71%) | 2022-03 (26.99%) | 1.84 | [-1.31%, 19.41%] |
| Commodities | 161 | 4.49% | 3.37% | 15.52% | 0.289 | 50.93% | -46.31% | 0.692 | 2016-04 (-10.20%) | 2020-03 (19.06%) | 1.56 | [-0.82%, 9.85%] |

#### EARLY_POST_PUBLICATION: 2013-01 through 2019-12

| Factor | N | Mean | Geo | Vol | Sharpe | Pos | MDD | Skew | Worst | Best | Growth | CI90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| All assets | 84 | 6.43% | 5.76% | 12.96% | 0.496 | 47.62% | -22.93% | 0.623 | 2015-04 (-6.58%) | 2015-01 (12.96%) | 1.48 | [0.34%, 12.60%] |
| Equities | 84 | 12.10% | 9.87% | 23.12% | 0.524 | 59.52% | -40.41% | -0.243 | 2015-08 (-18.19%) | 2015-02 (14.30%) | 1.93 | [0.94%, 23.64%] |
| Currencies | 84 | 6.94% | 5.53% | 17.96% | 0.386 | 58.33% | -25.63% | 1.018 | 2018-05 (-11.84%) | 2015-01 (24.42%) | 1.46 | [-1.97%, 16.17%] |
| Fixed income | 84 | 12.26% | 8.54% | 29.03% | 0.422 | 50.00% | -37.51% | 0.588 | 2013-05 (-19.47%) | 2019-08 (24.48%) | 1.77 | [0.13%, 24.63%] |
| Commodities | 84 | 0.89% | -0.21% | 14.92% | 0.059 | 47.62% | -41.76% | 0.314 | 2016-04 (-10.20%) | 2015-07 (13.39%) | 0.99 | [-5.75%, 7.53%] |

#### RECENT: 2020-01 through 2026-05

| Factor | N | Mean | Geo | Vol | Sharpe | Pos | MDD | Skew | Worst | Best | Growth | CI90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| All assets | 77 | 4.22% | 3.40% | 13.28% | 0.318 | 55.84% | -24.56% | -0.229 | 2021-11 (-11.47%) | 2022-03 (10.14%) | 1.24 | [-1.21%, 9.79%] |
| Equities | 77 | -3.40% | -8.07% | 30.67% | -0.111 | 58.44% | -73.86% | -0.920 | 2020-02 (-26.80%) | 2023-11 (20.28%) | 0.58 | [-18.84%, 11.70%] |
| Currencies | 77 | 0.23% | -1.55% | 19.04% | 0.012 | 53.25% | -47.10% | 0.087 | 2021-11 (-13.17%) | 2020-03 (18.14%) | 0.90 | [-7.02%, 7.47%] |
| Fixed income | 77 | 5.41% | 0.57% | 31.18% | 0.174 | 54.55% | -56.11% | -0.013 | 2021-02 (-23.71%) | 2022-03 (26.99%) | 1.04 | [-10.24%, 21.52%] |
| Commodities | 77 | 8.43% | 7.42% | 16.17% | 0.521 | 54.55% | -21.87% | 1.004 | 2020-11 (-7.99%) | 2020-03 (19.06%) | 1.58 | [1.92%, 15.02%] |

### All-assets cost-hurdle screen

Volatility is invariant to a constant monthly drag. The table gives annualized
mean, geometric return, Sharpe, maximum drawdown, terminal growth, and the
annualized bootstrap mean interval.

| Period | Hurdle | Mean | Geo | Vol | Sharpe | MDD | Growth | CI90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SOURCE_SAMPLE | Gross | 16.84% | 17.39% | 11.93% | 1.411 | -15.15% | 55.03 | [14.55%, 19.11%] |
|  | 1% | 15.84% | 16.23% | 11.93% | 1.328 | -15.51% | 42.99 | [13.55%, 18.11%] |
|  | 2% | 14.84% | 15.09% | 11.93% | 1.244 | -15.89% | 33.57 | [12.55%, 17.11%] |
|  | 3% | 13.84% | 13.96% | 11.93% | 1.160 | -16.46% | 26.22 | [11.55%, 16.11%] |
| BRIDGE | Gross | 4.71% | 3.94% | 13.14% | 0.359 | -11.40% | 1.12 | [-0.77%, 10.26%] |
|  | 1% | 3.71% | 2.91% | 13.14% | 0.283 | -11.85% | 1.09 | [-1.77%, 9.26%] |
|  | 2% | 2.71% | 1.89% | 13.14% | 0.206 | -12.30% | 1.06 | [-2.77%, 8.26%] |
|  | 3% | 1.71% | 0.88% | 13.14% | 0.130 | -12.75% | 1.03 | [-3.77%, 7.26%] |
| STRICT_POST_PUBLICATION | Gross | 5.37% | 4.62% | 13.08% | 0.411 | -27.91% | 1.83 | [0.98%, 9.88%] |
|  | 1% | 4.37% | 3.58% | 13.08% | 0.334 | -31.96% | 1.60 | [-0.02%, 8.88%] |
|  | **2%** | **3.37%** | **2.56%** | **13.08%** | **0.258** | **-35.79%** | **1.40** | **[-1.02%, 7.88%]** |
|  | 3% | 2.37% | 1.54% | 13.08% | 0.181 | -39.40% | 1.23 | [-2.02%, 6.88%] |
| EARLY_POST_PUBLICATION | Gross | 6.43% | 5.76% | 12.96% | 0.496 | -22.93% | 1.48 | [0.34%, 12.60%] |
|  | 1% | 5.43% | 4.71% | 12.96% | 0.419 | -25.22% | 1.38 | [-0.66%, 11.60%] |
|  | 2% | 4.43% | 3.67% | 12.96% | 0.342 | -27.45% | 1.29 | [-1.66%, 10.60%] |
|  | 3% | 3.43% | 2.64% | 12.96% | 0.264 | -29.62% | 1.20 | [-2.66%, 9.60%] |
| RECENT | Gross | 4.22% | 3.40% | 13.28% | 0.318 | -24.56% | 1.24 | [-1.21%, 9.79%] |
|  | 1% | 3.22% | 2.37% | 13.28% | 0.242 | -25.83% | 1.16 | [-2.21%, 8.79%] |
|  | 2% | 2.22% | 1.35% | 13.28% | 0.167 | -27.07% | 1.09 | [-3.21%, 7.79%] |
|  | 3% | 1.22% | 0.34% | 13.28% | 0.092 | -28.30% | 1.02 | [-4.21%, 6.79%] |

### Strict-post-publication breadth at the primary 2% hurdle

| Factor | Mean | Geo | Vol | Sharpe | Pos | MDD | Growth | CI90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All assets | 3.37% | 2.56% | 13.08% | 0.258 | 50.93% | -35.79% | 1.40 | [-1.02%, 7.88%] |
| Equities | 2.69% | -1.12% | 27.00% | 0.100 | 58.39% | -82.64% | 0.86 | [-8.62%, 14.22%] |
| Currencies | 1.73% | 0.06% | 18.45% | 0.094 | 55.28% | -52.92% | 1.01 | [-4.00%, 7.88%] |
| Fixed income | 6.98% | 2.58% | 30.00% | 0.233 | 50.93% | -57.33% | 1.41 | [-3.31%, 17.41%] |
| Commodities | 2.49% | 1.33% | 15.52% | 0.161 | 49.07% | -51.38% | 1.19 | [-2.82%, 7.85%] |

All four class arithmetic means are positive, so Gate 5 passes 4/4. This is a
breadth diagnostic, not an attribution claim: AQR does not publish enough
weights here to reconstruct contributions to its supplied aggregate without
arbitrary assumptions. Strict-period gross class correlations range from
-0.069 (equities/fixed income) to 0.400 (currencies/fixed income); the other
pairs are equities/commodities -0.014, equities/currencies 0.076,
commodities/fixed income 0.206, and commodities/currencies 0.333.

### Bootstrap and gates

The bootstrap uses the repository's existing `arch` wrapper with stationary
expected block length 12, 10,000 resamples, seed 20260825, and a percentile
central 80% interval. Its lower endpoint is definitionally the one-sided 90%
lower bound required by the frozen gate.

| Gate | Frozen test | Value | Result |
|---|---|---:|---:|
| G1 | Strict all-assets mean after 2% > 0 | 3.372% | PASS |
| G2 | Early all-assets mean after 2% > 0 | 4.428% | PASS |
| G3 | Recent all-assets mean after 2% > 0 | 2.220% | PASS |
| G4 | Strict all-assets mean after 3% > 0 | 2.372% | PASS |
| G5 | Positive strict class means after 2% >= 3 | 4/4 | PASS |
| G6 | Strict all-assets 2% one-sided 90% lower bound > 0 | -1.018% | **FAIL** |

### Publication decay

The following are gross-return changes from the earlier period to the later
period. `dMDD` below zero means the later period had a deeper drawdown.

| Comparison | Factor | dMean | dVol | dSharpe | dMDD | dPositive months |
|---|---|---:|---:|---:|---:|---:|
| Source to strict | All assets | -11.47 pp | +1.15 pp | -1.001 | -12.77 pp | -15.78 pp |
|  | Equities | -18.70 pp | -1.17 pp | -0.657 | -33.09 pp | -1.33 pp |
|  | Currencies | -10.73 pp | +0.01 pp | -0.582 | -18.59 pp | -4.10 pp |
|  | Fixed income | -11.94 pp | +0.34 pp | -0.406 | -1.22 pp | -6.83 pp |
|  | Commodities | -9.59 pp | +1.60 pp | -0.722 | -26.73 pp | -13.73 pp |
| Early to recent | All assets | -2.21 pp | +0.32 pp | -0.178 | -1.63 pp | +8.23 pp |
|  | Equities | -15.51 pp | +7.55 pp | -0.635 | -33.45 pp | -1.08 pp |
|  | Currencies | -6.70 pp | +1.07 pp | -0.374 | -21.47 pp | -5.09 pp |
|  | Fixed income | -6.85 pp | +2.15 pp | -0.249 | -18.60 pp | +4.55 pp |
|  | Commodities | +7.54 pp | +1.25 pp | +0.462 | +19.89 pp | +6.93 pp |

### Calendar-year and rolling diagnostics

Full calendar-year tables for all five factors at all four hurdles are in the
artifact. The post-publication calendar returns below are gross except for the
explicit all-assets 2% column; 2026 is partial through May.

| Year | All gross | All 2% | Equities | Currencies | Fixed income | Commodities |
|---:|---:|---:|---:|---:|---:|---:|
| 2013 | 18.24% | 15.92% | 75.59% | 4.03% | -9.60% | 20.64% |
| 2014 | 19.09% | 16.76% | 7.78% | 24.04% | 26.32% | 13.92% |
| 2015 | 18.14% | 15.82% | -3.63% | 27.34% | 13.47% | 21.46% |
| 2016 | -11.13% | -12.90% | -20.94% | -0.37% | 14.55% | -25.70% |
| 2017 | 6.63% | 4.52% | 60.76% | -0.34% | -7.77% | 0.40% |
| 2018 | -9.01% | -10.82% | -9.53% | -14.45% | -9.31% | -8.25% |
| 2019 | 3.18% | 1.13% | -7.86% | 4.40% | 42.97% | -13.72% |
| 2020 | -6.74% | -8.60% | -61.83% | 0.06% | 33.33% | -3.95% |
| 2021 | -4.56% | -6.46% | 47.27% | -10.06% | -49.85% | 14.33% |
| 2022 | 24.74% | 22.31% | -38.66% | 8.86% | 129.80% | 18.64% |
| 2023 | -8.14% | -9.98% | -8.33% | -4.32% | -18.11% | -8.57% |
| 2024 | 3.48% | 1.43% | 28.28% | -0.75% | -13.60% | 4.09% |
| 2025 | 1.77% | -0.24% | 34.68% | -14.45% | -10.63% | 5.47% |
| 2026 YTD | 15.35% | 14.42% | 6.70% | 13.68% | 6.75% | 21.02% |

At 2026-05, the rolling 36-month all-assets annualized mean/Sharpe are
4.91%/0.529 gross and 2.91%/0.313 after the 2% hurdle. The 2%-hurdle class
readings are equities 25.58%/0.909, currencies -3.59%/-0.203, fixed income
-8.38%/-0.348, and commodities 3.77%/0.290. The full time series in the
artifact shows that the all-assets 2%-hurdle rolling mean ranged from -10.16%
at 2019-02 to 27.97% at 1998-08; rolling Sharpe ranged from -1.010 to 2.607 at
the same endpoints. These are diagnostics and create no additional gates.

### Interpretation

The economic point estimates survive the frozen 2% and 3% hurdles, and strict
arithmetic breadth is 4/4. That is not enough under the preregistration: modern
performance is far weaker than the source sample and the primary 2%-hurdle
mean does not have a positive one-sided 90% lower bound.

This result is consistent with, rather than a rebuttal to, the recorded
critiques. An aggregate factor cannot establish strong instrument-level
predictability or separate signal skill from volatility scaling, historical
mean/sign effects, long exposure, or diversification. The low cross-class
correlations and uneven calendar returns show how diversification can improve
the aggregate even while individual classes are fragile. The cost status is
not explicit, so deterministic drags remain hurdles rather than a claim about
realized trading costs. Most importantly, the severe source-to-post-publication
decay and failed statistical gate leave no justification for paying for or
building a likely inferior CFD/free-data proxy. A positive point estimate also
would not prove FTMOQuant could reproduce the factor with CFDs.

### Exact next action

Retire XAT1 now: do not buy instrument-level futures data and do not build the
free proxy. Record `RETIRE_XAT1` in future planning/roadmap work while leaving
all existing sealed performance artifacts untouched. Preserve this audit and
the external official workbooks as provenance; do not rescue XAT1 with a new
lookback, asset subset, hurdle, volatility estimator, or trend definition.
