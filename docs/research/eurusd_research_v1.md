# EUR/USD research dataset v1

`eurusd_research_v1` freezes a whole-UTC-day universe from 11 March 2019
through 31 December 2025:

| Split | Inclusive dates | Days |
| --- | --- | ---: |
| Development | 2019-03-11 through 2023-04-10 | 1,492 |
| Validation | 2023-04-11 through 2024-08-20 | 498 |
| Final holdout | 2024-08-21 through 2025-12-31 | 498 |

G1 may admit only timestamps before `2024-08-21T00:00:00Z`. The holdout is
sealed even when a physical source shard straddles that boundary.

## Source and canonical chain

The market-data origin is Dukascopy. The distribution source is the Hugging
Face dataset `mito0o852/dukascopy-ticks`, frozen at revision
`bf19dbd89c732f010e20db7c148922ba02b2e33b`. Those are distinct provenance
roles: the importer does not claim that the legacy `tradedesk-dukascopy`
downloader acquired these files.

The deterministic transformation is:

`pinned EURUSD Parquet ticks` → `strict admitted-row validation` →
`UTC minute BID/ASK OHLC + same-side volume` → `Nautilus external 1m catalog`
→ `existing G0.6 native complete-window 1H/4H derivation`.

Only minutes containing ticks are emitted. Missing periods remain missing; no
fill, interpolation, forward-fill, holiday bar, or weekend bar is created.
Canonical one-minute timestamps use the UTC minute open as `ts_event` and the
minute close as `ts_init`.

The independently verified omissions at `2023-11-14T13:31:00Z` and
`2023-11-29T14:29:00Z` through `14:34:00Z` are represented only in the immutable
child identity `g1-dukascopy-corrected-1`. Its source is the reconciliation-bound
direct Dukascopy BI5 evidence, aggregated with the same production HF importer
semantics. The original `eurusd_research_v1` root remains unchanged; the child
adds exactly seven paired minutes and regenerates 1H/4H data from canonical 1m
bars.

## Research boundary

This work prepares and validates research data only. The integration smoke was
limited to seven development days and exercised ingestion, provenance,
session-aware gap classification, and structural 1H/4H derivation. No strategy
was run, no return or performance output was inspected, no parameter was
tuned, and no validation or final-holdout strategy result has been viewed.
