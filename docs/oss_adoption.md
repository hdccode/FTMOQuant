# OSS adoption

## Decision

NautilusTrader `2.0.0rc2` is the provisional core trading engine for FTMOQuant.
It is pinned exactly and evaluated on Python 3.12. It is not yet approved for
live capital: rc2 is explicitly a pre-release, and its v2 Python API differs
materially from v1 examples. G0.4 connects a thin FTMO-only overlay to native
account, portfolio, position/fill event, and Clock state without introducing a
second production P/L ledger.

The deterministic probe uses the v2 `BacktestEngine` with synthetic EUR/USD L1
quotes, a margin account, two market fills, and explicit fees. It confirms
access to nanosecond market/order/fill/account timestamps, orders, fills,
positions, commissions, realized P/L, balance, portfolio equity, account-state
events, and tabular order/fill/position/account reports. The same normalized
fixture result is asserted across two independent engine instances.

## Candidates reviewed

| Project | Repository | License | Intended role | Adoption status |
| --- | --- | --- | --- | --- |
| NautilusTrader | [nautechsystems/nautilus_trader](https://github.com/nautechsystems/nautilus_trader) | LGPL-3.0-or-later | Event-driven backtest/live trading core | Adopt provisionally at exactly PyPI `2.0.0rc2`; the locked macOS ARM64 wheel is SHA-256 `a271f0cfd82a75ade4da19b6ead495acb01c1b19de6b1a82da414950887cdd52` |
| arch | [bashtage/arch](https://github.com/bashtage/arch) | NCSA | Statistical bootstrap and multiple-comparison procedures | Adopt at exactly `8.0.0` for G0.8; sampling, confidence intervals, block-length estimation, SPA/Reality Check, and MCS remain native arch operations |
| tradedesk-dukascopy | [radiusred/tradedesk-dukascopy](https://github.com/radiusred/tradedesk-dukascopy) | Apache-2.0 | Dukascopy BI5 acquisition, recovery, decoding, scale probe, and 1-minute export | Adopt at exactly `1.0.0` / commit `b8fb503c9291d6e265949d008e288b76b68fb852`; FTMOQuant calls its public export CLI contract and retains its CSV metadata sidecars |
| LEAN | [QuantConnect/Lean](https://github.com/QuantConnect/Lean) | Apache-2.0 | Full event-driven trading engine alternative | Not adopted; .NET-centric runtime adds operational weight to this Python project |
| hftbacktest | [nkaz001/hftbacktest](https://github.com/nkaz001/hftbacktest) | MIT | Latency-oriented backtest architecture reference | Reference-only for G0.7; no dependency, code, matching engine, or accounting implementation was adopted |
| vectorbt | [polakowo/vectorbt](https://github.com/polakowo/vectorbt) | Apache-2.0 + Commons Clause | Vectorized research and parameter exploration | Deferred; useful research complement, not the execution/accounting core |
| Backtrader | [mementum/backtrader](https://github.com/mementum/backtrader) | GPL-3.0 | Python event-driven backtester | Not adopted; license and legacy architecture are a poor core fit |
| Zipline Reloaded | [stefan-jansen/zipline-reloaded](https://github.com/stefan-jansen/zipline-reloaded) | Apache-2.0 | Research backtesting | Not adopted; backtest-oriented rather than research-to-live execution parity |

## G0.3 oracle audit

The bespoke G0.3 implementation remains isolated on branch
`g0-ftmo-oracle` at commit `620642a`. G0.3R does not merge, modify, or extend it.

### Redundant with NautilusTrader

- Generic balance, floating P/L, equity, order, fill, fee, and position
  bookkeeping.
- Open-position identity and inconsistent open/close operation checks.
- Ordered event timestamps, deterministic event replay, and terminal position
  accounting.
- General account snapshots and execution reports.

These should come from Nautilus if it becomes the core engine, avoiding two
production ledgers.

### FTMO-specific and worth retaining

- Versioned FTMO configuration and explicit source metadata.
- Runtime account capital, currency, and active Challenge/Verification phase.
- Europe/Prague FTMO-day boundaries, including DST behavior.
- Reset-balance daily-loss floor, static maximum-loss floor, and strict
  below-floor breach semantics.
- Minimum trading-day counting, phase profit targets, all-positions-closed pass
  condition, and terminal FTMO breach/pass status.

These belong in a separate overlay consuming Nautilus events and state.

### Useful as an independent oracle

- The small immutable `Decimal` reducer is easy to reason about independently
  of Nautilus internals.
- Boundary tests cover equality, commissions/swaps, midnight resets, DST,
  simultaneous positions, and account-size scaling.
- Property-based invariants verify proportional loss floors and exactly-once
  accounting.

Retain it as a test oracle for selected event traces, not as a second runtime
execution/accounting engine.

## G0.4 source provenance

| Source | Exact revision | License | Use in FTMOQuant |
| --- | --- | --- | --- |
| [Dahshan228/FTMO-TradingBot](https://github.com/Dahshan228/FTMO-TradingBot) | [`8d1ecdff8d1b933870273794bc5b63216b4a3167`](https://github.com/Dahshan228/FTMO-TradingBot/commit/8d1ecdff8d1b933870273794bc5b63216b4a3167) | MIT, copyright 2026 Ali Eldahshan | Patterns adapted from [`src/risk/ftmo_compliance.py`](https://github.com/Dahshan228/FTMO-TradingBot/blob/8d1ecdff8d1b933870273794bc5b63216b4a3167/src/risk/ftmo_compliance.py) and [`tests/risk/test_ftmo_compliance.py`](https://github.com/Dahshan228/FTMO-TradingBot/blob/8d1ecdff8d1b933870273794bc5b63216b4a3167/tests/risk/test_ftmo_compliance.py): keep the midnight baseline separate from live equity, calculate the daily allowance from initial capital, and expose explicit daily/static floors. No source text was copied. Its float arithmetic, one-cent buffer, persistence checks, headroom API, and equality-as-breach behavior were intentionally not adapted. |
| [loadxf/quant](https://github.com/loadxf/quant) | [`043149572a63f14844bf660449105af9acdf3ea1`](https://github.com/loadxf/quant/commit/043149572a63f14844bf660449105af9acdf3ea1) | No license declared at the reviewed revision | Reference-only review of `src/quantlab/prop/evaluator.py`, `src/quantlab/prop/rules/{daily_loss,static_loss,min_days}.py`, and `tests/prop/test_evaluator.py`. No code, test text, or implementation pattern was copied. |
| FTMOQuant G0.3 oracle | `620642ac03d0abf7f019dcaae5b70a4306296d66` on local branch `g0-ftmo-oracle` | Project-owned | Preserved as a test-only independent reducer for differential assertions. It is absent from the production package and does not supply Nautilus account values. |

NautilusTrader was consumed as the exact published `2.0.0rc2` wheel rather
than adapted from a source checkout, so no upstream Git commit is claimed for
that artifact. The repository does not expose a matching immutable v2 tag in
the reviewed tag set; the lockfile records the exact wheel hashes instead.

## G0.5 data-source provenance

[`radiusred/tradedesk-dukascopy`](https://github.com/radiusred/tradedesk-dukascopy)
at commit
[`b8fb503c9291d6e265949d008e288b76b68fb852`](https://github.com/radiusred/tradedesk-dukascopy/commit/b8fb503c9291d6e265949d008e288b76b68fb852),
released as `tradedesk-dukascopy==1.0.0`, is Apache-2.0 licensed. It is the
adopted provider boundary for BI5 download, decompression, tick decoding,
retry/backoff, cache repair, gap recovery, scale probing, and UTC one-minute
BID/ASK CSV export. FTMOQuant does not reimplement those facilities.

FTMOQuant invokes the upstream `tradedesk-dc-export` entry point through its
Python CLI function. It preserves both generated CSVs and their upstream
`.meta.json` sidecars under the operator-selected output root. The configured
price divisor is explicit, checked against both sidecars, and recorded in the
FTMOQuant provenance manifest. Operators use the upstream `--probe` mode before
choosing it; FTMOQuant contains no Dukascopy divisor table.

The following QA patterns were adapted, with new FTMOQuant code rather than
copied source:

- From `scripts/dukascopy_audit.py`: align BID/ASK observations by UTC minute,
  surface absent intervals explicitly, and reject impossible spreads. The
  FTMOQuant boundary is stricter: BID and ASK coverage must match exactly and
  every comparable ASK OHLC field must be at least its BID field.
- From `scripts/audit_fx_scale.py` and `tests/test_audit_fx_scale.py`: fail on
  prices outside a deliberately broad natural FX-rate envelope. FTMOQuant uses
  this only to catch obvious EUR/USD power-of-ten corruption; it neither infers
  nor substitutes a divisor.

Validated rows are encoded as Arrow IPC and passed to NautilusTrader
`2.0.0rc2` `BarDataWrangler`. The resulting external BID/ASK bars and EUR/USD
instrument are stored only through `ParquetDataCatalog`. PyArrow is a direct
dependency solely because the v2 wrangler accepts Arrow IPC bytes rather than
a pandas DataFrame.

## G1 Hugging Face Dukascopy tick adapter

The research-data plan uses Dukascopy as the market-data origin and Hugging
Face only as the distribution source. The official `HfApi.dataset_info` call
resolved `mito0o852/dukascopy-ticks` to immutable revision
`bf19dbd89c732f010e20db7c148922ba02b2e33b` before any market-data processing.
That full SHA is frozen in `config/data/eurusd_research_v1.yaml`; runtime
metadata and every download are requested at the same SHA, never at `main`.
Updating the remote repository cannot move this plan.

`huggingface-hub==1.27.0` is adopted as the smallest official client boundary.
The importer uses `dataset_info(..., files_metadata=True)` to confirm the
resolved SHA and obtain sizes/LFS hashes, `list_repo_files(..., revision=SHA)`
for inventory, and `hf_hub_download(..., repo_type="dataset", revision=SHA)`
for one selected file at a time. The Hub cache is therefore reused on repeated
downloads. The much larger `datasets` package is not needed and was not added.
The relevant official references are the Hugging Face
[HfApi reference](https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api)
and [revision-pinned download guide](https://huggingface.co/docs/huggingface_hub/en/guides/download).

The pinned EUR/USD inventory is an unpartitioned sequence of inclusive
30-calendar-day filename ranges. It is validated for exact
`data/EURUSD/YYYY-MM-DD_YYYY-MM-DD.parquet` names, ordering, duplicate ranges,
overlaps, and date gaps before selection. Other instruments are never selected.
The G1 cutoff falls inside
`data/EURUSD/2024-08-16_2024-09-14.parquet`; selecting that packaging shard is
permitted, but an explicit raw `timestamp < 2024-08-21T00:00:00Z` predicate is
applied to every Arrow batch before conversion, validation callbacks,
aggregation, or provenance counting.

The actual pinned Parquet schema, verified on the development-only integration
shard, is exactly:

- `timestamp: int64`
- `askPrice: double`
- `bidPrice: double`
- `askVolume: double`
- `bidVolume: double`

PyArrow `22.0.0` (within the existing project constraint) is reused through
`ParquetFile.schema_arrow` and `iter_batches`. Each shard is scanned one record
batch at a time. Arrow compute filters the admitted half-open UTC range before
columns become Python values. Only the current minute aggregate, previous tick
timestamp, bounded output-bar chunk, counters, hashes, and missing-interval
state survive a batch or file boundary. This follows the official
[`ParquetFile.iter_batches`](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetFile.html)
contract; neither multiple years of ticks nor all canonical bars are retained
in Python memory.

The timestamp unit is proved rather than configured heuristically. For an
admitted raw integer in every shard, the adapter tries Unix seconds,
milliseconds, microseconds, and nanoseconds, converts each candidate to UTC,
and requires that exactly the millisecond interpretation falls inside the
filename-declared date range. Every admitted timestamp is then checked against
that range and the source stream must remain monotonic nondecreasing. Equal
milliseconds retain original Parquet row order as the first/last tie-breaker.

The tick transformation is direct and side-specific: first/max/min/last price
and summed same-side volume for each UTC minute containing at least one tick.
No empty minute is created, filled, interpolated, or forward-filled. The
existing G0.5 EUR/USD instrument, 5-digit price/8-digit size encoding,
`BarDataWrangler`, external BID/ASK bar types, spread checks, and
`ts_event=minute open` / `ts_init=minute open + 60 seconds` contract are reused.
The resulting explicit ingestion identity is `g1-hf-dukascopy-1`; it never
claims `tradedesk-dukascopy` performed the download.

### Targeted GitHub reuse audit

The pre-implementation audit searched for a reusable Hugging Face Parquet
Dukascopy adapter and reviewed these primary sources:

| Candidate | Finding |
| --- | --- |
| [huggingface/huggingface_hub](https://github.com/huggingface/huggingface_hub) and [huggingface/hub-docs streaming guidance](https://github.com/huggingface/hub-docs/blob/main/docs/hub/datasets-streaming.md) | Adopted only the official metadata, inventory, pinned-download/cache APIs and the documented Parquet row-group streaming pattern. The generic `HfFileSystem` layer and whole-repository snapshot download were unnecessary. |
| [radiusred/tradedesk-dukascopy](https://github.com/radiusred/tradedesk-dukascopy/tree/b8fb503c9291d6e265949d008e288b76b68fb852) | Existing G0.5 instrument, precision, bar encoding, catalog, timestamp, gap, and spread-validation code was reused. Its BI5 acquisition identity was not reused for the Hugging Face distribution path. |
| [theorycraft-trading/dukascopy](https://github.com/theorycraft-trading/dukascopy) | Provides an Elixir Dukascopy download/stream/resample stack, not a Python adapter for this pinned Hugging Face mirror or the sealed G1 cutoff; not adopted. |
| [cpcerrato/dukascopy-downloader](https://github.com/cpcerrato/dukascopy-downloader) | Provides a separate .NET BI5 downloader and tick aggregation path. It would duplicate both the chosen distribution source and existing canonical encoding; not adopted. |

No reviewed project supplied the required combination of immutable Hub
revision provenance, strict mirror inventory validation, pre-conversion
holdout filtering, and Nautilus `2.0.0rc2` encoding. The project-owned adapter
is consequently limited to those seams. A shared canonical source-manifest
validator now admits exactly the legacy `g0.5-1` identity and the new
`g1-hf-dukascopy-1` identity. G0.6 and the session-aware coverage bridge use
that validator without changing their aggregation or fail-closed coverage
semantics.

## Carver OANDA DEVELOPMENT acquisition reuse audit

Before implementing the OANDA acquisition boundary, FTMOQuant reviewed the
following maintained, license-compatible GitHub candidates:

| Candidate | License / finding | Decision |
| --- | --- | --- |
| [oanda/v20-python](https://github.com/oanda/v20-python) | MIT; official OANDA v20 bindings, but its small, older binding layer does not supply immutable response archival, a half-open DEVELOPMENT contract, or project QA artifacts. | Not adopted. |
| [hootnot/oanda-api-v20](https://github.com/hootnot/oanda-api-v20) | MIT; mature community wrapper with a 5,000-candle request factory. Its pagination pattern informed the fixed 5,000-minute request ceiling, but no implementation or source text was copied. | Reference-only. |

The resulting project-owned adapter deliberately uses the standard-library HTTPS
client for the one OANDA candles endpoint. This avoids adding a general trading
SDK solely to issue a fixed GET request and keeps credentials out of persisted
metadata. It stores each unmodified JSON response and a redacted request record
before parsing, does not synthesize missing candles, and does not apply the
SOYBN ×100 economics-boundary transform.

## G0.6 native derived-bar adoption

G0.6 uses NautilusTrader `2.0.0rc2` bar-to-bar composite aggregation. Its four
subscriptions use the upstream `TARGET@SOURCE` syntax directly:

- `EUR/USD.DUKASCOPY-1-HOUR-BID-INTERNAL@1-MINUTE-EXTERNAL`
- `EUR/USD.DUKASCOPY-1-HOUR-ASK-INTERNAL@1-MINUTE-EXTERNAL`
- `EUR/USD.DUKASCOPY-4-HOUR-BID-INTERNAL@1-MINUTE-EXTERNAL`
- `EUR/USD.DUKASCOPY-4-HOUR-ASK-INTERNAL@1-MINUTE-EXTERNAL`

Each target is independently fed eligible bars from the original external
one-minute series. Nautilus emits its standard target type (without the
composite suffix), and `ParquetDataCatalog.write_bars` persists those outputs.
FTMOQuant does not implement OHLC or volume aggregation. Its surrounding logic
only proves that an aligned target window contains exactly 60 or 240
consecutive source bars, excludes all other windows before engine replay,
checks BID/ASK close coverage, verifies callback time against the final source
close, hashes deterministic outputs, and enforces idempotent catalog writes.
No FX session mask, holiday calendar, fill, interpolation, or synthetic minute
is used; in particular, the `tradedesk-dukascopy` fixed-22:00-UTC mask was not
adopted.

The `DataEngineConfig` is explicit: `time_bars_timestamp_on_close=True`,
`time_bars_skip_first_non_full_bar=True`,
`time_bars_build_with_no_updates=False`, `time_bars_build_delay=1` microsecond,
`time_bars_interval_type=BarIntervalType.LEFT_OPEN`, and no origin offset.
Default UTC origins therefore close 1H bars on every UTC hour and 4H bars at
00:00, 04:00, 08:00, 12:00, 16:00, and 20:00 UTC. Stored `ts_event` and
`ts_init` are the nominal interval close.

The following upstream documentation and tests were reviewed. The public v2
release-candidate wheel has no matching immutable repository tag, so the
runtime contract was additionally checked against the installed pinned wheel's
generated Python stubs and exercised end-to-end; source links identify the
exact reviewed upstream commit `7f0e93dfa3f09ca165a5f3292a45fafbb5681561`:

- [`docs/concepts/data/index.md`](https://github.com/nautechsystems/nautilus_trader/blob/7f0e93dfa3f09ca165a5f3292a45fafbb5681561/docs/concepts/data/index.md),
  especially “Composite bars,” “Bar-to-bar example,” and “Time bar
  configuration,” for composite syntax, standard emitted targets, UTC origin
  behavior, interval semantics, and `DataEngineConfig` controls.
- [`docs/concepts/backtesting/bar-execution.md`](https://github.com/nautechsystems/nautilus_trader/blob/7f0e93dfa3f09ca165a5f3292a45fafbb5681561/docs/concepts/backtesting/bar-execution.md),
  “Internal bar aggregation timing,” for the one-microsecond close-timer delay.
- [`crates/data/src/aggregation.rs`](https://github.com/nautechsystems/nautilus_trader/blob/7f0e93dfa3f09ca165a5f3292a45fafbb5681561/crates/data/src/aggregation.rs),
  including upstream tests
  `test_time_bar_aggregator_left_open_interval`,
  `test_time_bar_aggregator_no_updates_behavior`,
  `test_time_bar_skip_first_non_full_bar_drops_partial_bar`,
  `test_aggregators_standardize_composite_bar_type`, and
  `test_composite_time_bar_aggregator_uses_standard_timer_name`.
- [`crates/data/tests/engine.rs`](https://github.com/nautechsystems/nautilus_trader/blob/7f0e93dfa3f09ca165a5f3292a45fafbb5681561/crates/data/tests/engine.rs),
  for data-engine composite subscription and response routing coverage.

Nautilus rc2 has two related timer details which G0.6 records rather than
hiding. First, a configured one-microsecond build delay also shifts the native
bar's `ts_event` and `ts_init` by one microsecond. G0.6 verifies that delayed
native callback (and therefore availability) is after the final source
one-minute `ts_init`, then subtracts only that known scheduling delay from the
two stored timestamps to restore the nominal UTC close; native OHLCV is not
modified. Second, starting replay exactly on an aligned boundary causes rc2's
`skip_first_non_full_bar` state to discard that first window even when it is
complete. Replay starts one nanosecond before the first eligible boundary so
rc2 sees the full interval while the configured skip remains enabled. Window
membership validation still independently rejects every partial first window.

Derived-bar integrity and dataset coverage readiness are separate. Passing the
native-output, complete-window, timestamp, and BID/ASK checks sets
`derived_bar_integrity_valid=true` and preserves every valid emitted bar. It
does not by itself make the dataset research-ready. Any incomplete source
window makes coverage `incomplete`; any otherwise gap-free no-update window is
`unclassified`, because G0.6 has no session or holiday calendar with which to
identify a legitimate market closure. Only datasets with no dropped windows
and at least one bar in all four required series are marked research-ready.

## G1 data-readiness session coverage bridge

The G1 bridge adds provider-aware coverage classification without changing the
G0.5 source files, catalog bars, missing-interval report, BID/ASK validation, or
the G0.6 derivation and conservative readiness fields. It writes a separate
`ftmoquant_session_coverage.json`, bound by SHA-256 to both existing manifests.
Its final `session_aware_research_ready` requires the G0.6
`derived_bar_integrity_valid` and `bid_ask_derived_coverage_matches` flags, at
least one bar in all four established 1H/4H BID/ASK series, and zero incomplete
paired source minutes during expected-open periods. Thus a complete multi-day
dataset may pass across an ordinary weekend even though the unchanged G0.6
`coverage_status` remains `unclassified`. No bar is filled, interpolated,
removed, or rewritten.

### Targeted GitHub reuse audit

The audit was performed before implementation against these exact public
revisions:

| Candidate | Reviewed revision | Finding |
| --- | --- | --- |
| NautilusTrader | [`8ecbd4b5785b15389c9e2d626a99723dcde5a0ee`](https://github.com/nautechsystems/nautilus_trader/blob/8ecbd4b5785b15389c9e2d626a99723dcde5a0ee/crates/trading/src/sessions.rs) on `develop`; installed `2.0.0rc2` stubs and runtime also inspected | `ForexSession.NEW_YORK`, `fx_local_from_utc`, and the four `fx_prev/next_start/end` helpers correctly use `America/New_York` and place the regional New York session end at 17:00 local. They model weekday regional sessions, not Dukascopy's continuous Sunday-Friday provider week or provider offline domains, so they are referenced as corroboration but are not used for classification. |
| QuantConnect LEAN | [`278fcb3d1b815b63ccadba68d7ae54422e34b792`](https://github.com/QuantConnect/Lean/blob/278fcb3d1b815b63ccadba68d7ae54422e34b792/Data/market-hours/market-hours-database.json) | Its calendars are broker-specific: Interactive Brokers uses a 17:00 New York weekly boundary, FXCM adds its own holidays and exceptions, and OANDA has daily 16:58-17:03 pauses. None establishes Dukascopy history. Adding LEAN would create a second calendar framework without provider authority, so it was not adopted. |
| tradedesk-dukascopy | [`b8fb503c9291d6e265949d008e288b76b68fb852`](https://github.com/radiusred/tradedesk-dukascopy/blob/b8fb503c9291d6e265949d008e288b76b68fb852/scripts/dukascopy_audit.py) | Its audit helper hardcodes Sunday 22:00 UTC through Friday 22:00 UTC. This ignores the summer schedule and makes both US DST-transition weekends wrong, so the mask was explicitly not reused. The pinned package remains the acquisition/decoding boundary described in G0.5. |

No new dependency or generic 24/5 calendar was adopted. The implementation
uses standard-library `zoneinfo.ZoneInfo("America/New_York")`, which is the
smallest mechanism that preserves the provider's stated US Eastern DST
semantics.

### Dukascopy session provenance and classification

Dukascopy's [general trading hours](https://www.dukascopy.com/swiss/english/forex/forex-trading-accounts/link/)
state that most instruments open Sunday at 21:00 GMT in summer / 22:00 GMT in
winter and close Friday at the matching hour. Dukascopy's
[2019 DST notice](https://www.dukascopy.com/swiss/english/about/ournews/change-to-daylight-saving-time-dbl201384)
removes the ambiguity in “summer”: the FX trading day ends at 17:00 New York
time, and the UTC change follows the US Eastern clock. Policy
`dukascopy-eurusd-ny-close-v1` therefore treats EUR/USD source minutes as
expected-open from Sunday 17:00 `America/New_York` inclusive through Friday
17:00 local exclusive. This maps to 21:00 UTC under EDT and 22:00 UTC under
EST. On a US spring-transition weekend the recurring closure is 47 hours; on
an autumn-transition weekend it is 49 hours.

The policy is conservatively valid only from the explicit 10 March 2019 change
forward. Earlier requested intervals fail closed instead of assuming that the
same rule held historically. Dukascopy's JForex
[market-hours API documentation](https://www.dukascopy.com/wiki/en/development/strategy-api/instruments/market-hours/)
also establishes that `IDataService.getOfflineTimeDomains` supplies historical
and upcoming provider offline periods. Those records are not present in the
G0.5 artifacts and are not fetched or synthesized by this bridge. Consequently
only recurring weekend minutes can be classified
`expected_market_closed`; a holiday, exceptional provider shutdown, or any
other absent expected-open minute is `unexplained_missing` until authoritative
offline-domain evidence is acquired and versioned.

Coverage is a paired-minute property. A minute counts as observed only when
both BID and ASK exist. During an expected-open period, no update, BID-only, or
ASK-only coverage is unexplained; exact intervals retain the missing side or
sides. Expected closures are reported separately with inclusive interval
boundaries and counts, and neither class is filled. The manifest records the
requested half-open UTC interval, expected-open count, paired observed count,
expected-closure count, unexplained count and exact intervals, source and
derived manifest hashes, the full session/source descriptions, and a semantic
SHA-256 over canonical JSON excluding only the hash field itself. It contains
no fetch timestamp or wall-clock timestamp.

A fixed UTC weekday mask is insufficient because it blesses one season while
misclassifying the other and cannot represent the one-hour difference between
the Friday and Sunday boundaries on US DST transition weekends. A generic
calendar's holiday list is also unsafe: provider-specific offline intervals
can differ, and unverified closure assumptions would convert missing source
data into false research readiness.

## G1 expected-open gap reconciliation

### Targeted reuse audit

The reconciliation audit was completed before implementation. It covered
Dukascopy historical offline domains, direct BI5/tick retrieval, generic FX
holiday calendars, and empty-minute handling. The reviewed state and decisions
are:

| Project or source | Exact revision/version | License | Decision and reason |
| --- | --- | --- | --- |
| [Dukascopy JForex `IDataService`](https://www.dukascopy.com/client/javadoc/com/dukascopy/api/IDataService.html) and [market-hours guide](https://www.dukascopy.com/wiki/en/development/strategy-api/instruments/market-hours/) | Published JForex API documentation `2.12.46`, retrieved 2026-08-13 | Dukascopy documentation/API terms | Authoritative reference and supported evidence format. `getOfflineTimeDomains(from, to)` is the provider interface for exact historical offline intervals. The Python reconciliation command accepts a semantically hashed export of those exact domains, but does not invent them when no authenticated JForex export is available. |
| [radiusred/tradedesk-dukascopy](https://github.com/radiusred/tradedesk-dukascopy/tree/b8fb503c9291d6e265949d008e288b76b68fb852) | PyPI `1.0.0`, commit `b8fb503c9291d6e265949d008e288b76b68fb852` | Apache-2.0 | Adopted. The existing pinned dependency remains the direct Dukascopy BI5 URL/decoding reference. Reconciliation adds a thin status-preserving cache boundary because the public export command intentionally commits old partial days and its private downloader returns the same sentinel for HTTP 404 and exhausted network retries; either ambiguity is unsafe as proof of a zero-tick minute. |
| [keyhankamyar/TickVault](https://github.com/keyhankamyar/TickVault/tree/d12cd8223a989cfce5f72b01be0120bb77899ef2) | Commit `d12cd8223a989cfce5f72b01be0120bb77899ef2` | MIT | Rejected as a dependency. It offers resume-capable hourly BI5 mirroring and SQLite status tracking, but would add `httpx`, Pydantic, NumPy/pandas, proxy, and database machinery for a small exact-window verifier already covered by the pinned provider path. |
| [Praveens1234/dukascopy-downloader](https://github.com/Praveens1234/dukascopy-downloader/tree/56e28c4f8655e801ed972a5bb0bf1a7a0fa76abb) | Commit `56e28c4f8655e801ed972a5bb0bf1a7a0fa76abb` | MIT | Rejected. Its resumable application, web UI, native-candle fallbacks, and optional inactive-period candle generation are broader than reconciliation; generated inactive candles are explicitly incompatible with the no-synthesis rule. |
| [gerrymanoim/exchange_calendars](https://github.com/gerrymanoim/exchange_calendars/tree/5308ce20578422fce74b10b43cc7d913a17e7a88) | Commit `5308ce20578422fce74b10b43cc7d913a17e7a88` | Apache-2.0 | Reference-only and rejected for classification. It provides exchange-specific regular/adhoc holiday machinery, not historical Dukascopy provider domains. A generic exchange holiday cannot prove that EUR/USD was offline at Dukascopy. |

The official Dukascopy [Trading Breaks Calendar](https://www.dukascopy.com/swiss/english/marketwatch/trading-breaks-calendar/)
and dated holiday news were also reviewed. They confirm that special schedules
exist, but the public current calendar and archived announcements inspected do
not expose a stable machine-readable historical EUR/USD interval feed. A dated
headline or a holiday name is therefore never transformed into an offline
interval. In the absence of an exact JForex domain export, an expected-open gap
can only become `verified_no_tick` after a successful, validated direct BI5
retrieval proves that exact minute contains no tick; retrieval failure, HTTP
404, malformed payloads, or a direct tick all remain fail-closed.

## G1 canonical omission correction

The seven direct-tick omissions established by reconciliation are corrected by
`g1-dukascopy-corrected-1`, a distinct immutable child of the frozen Hugging
Face canonical dataset. The command accepts only the two reconciliation-bound
BI5 cache objects, validates their content hashes and hour counts, and requires
the exact seven minute counts before any child catalog write. It reuses the HF
importer's minute transition and bar encoding helpers for side-specific
first/max/min/last, same-side volume, precision, bar types, and timestamps.

The child catalog is streamed from the parent because Nautilus catalog files
must have disjoint physical timestamp spans. Every parent `Bar` object is
written unchanged while the seven BID and seven ASK additions are merged into
bounded disjoint chunks. A second streaming merge proves all parent bars still
compare equal, no parent timestamp disappeared, and no non-target bar appeared.
The correction manifest binds the parent, research plan, reconciliation, BI5
objects, exact minute counts, canonical content hashes, and equivalence proof
with deterministic semantic hashes. It records no synthesis, interpolation, or
holdout access.

G0.6, session coverage, and reconciliation then run from the corrected
canonical root without directly patching derived data. Final readiness is a
separate semantically hashed freeze artifact and remains fail-closed unless
derived integrity passes, reconciliation has zero unexplained minutes, all
remaining expected-open gaps are independently reconciled, and the holdout and
strategy-return access flags remain false.

## Multi-year G0.6 and session-QA catalog streaming

The full G1 development-plus-validation catalog contains several million
paired one-minute bars. G0.6 and the session-aware coverage bridge therefore
must not use an unbounded `tuple(catalog.query_bars(...))` or a dataset-wide
timestamp set. The installed NautilusTrader `2.0.0rc2` generated stubs,
runtime signatures, and a local endpoint fixture were inspected before this
change.

The exact adopted `ParquetDataCatalog` APIs are:

- `query_bars(identifiers=None, start=None, end=None, where_clause=None)` for
  bounded native DataFusion reads;
- `query_first_timestamp("bars", identifier)` and
  `query_last_timestamp("bars", identifier)` for constant-memory outer-range
  checks;
- `query_files("bars", [identifier])` for an output preflight without decoding
  all stored bars; and
- `write_bars(data, start=None, end=None, skip_disjoint_check=False)` for
  incremental, disjoint checked derived output.

The rc2 fixture establishes an important detail not apparent from the Python
signature: bar `start`/`end` filtering uses stored `ts_init` and includes the
endpoints. Each 10,000-minute native query is consequently normalized with a
local `ts_event` predicate. Source chunks use `[start, end)`; stored derived
close timestamps use `(start, end]`. Adjacent chunks therefore neither lose
nor duplicate boundary bars.

The shared canonical scanner first performs a complete read-only validation
pass before any derived write. It carries only the previous BID/ASK timestamp
and counts, while preserving exact type, minute alignment, open/close
timestamp, chronology, manifest count, range, and paired-coverage checks. It
admits only `g0.5-1`, `g1-hf-dukascopy-1`, and the explicitly lineage-bound
`g1-dukascopy-corrected-1`; no generic provenance rule was added.

During the second G0.6 pass, each 1H/4H side retains only its current partial
aligned window (at most 240 source bars), complete eligible windows from the
current bounded query, rolling content/coverage hashes, counts, and the
dropped-window details which must appear in the manifest. Complete windows are
still replayed through the pinned native `BacktestEngine` composite-bar
aggregator with the unchanged G0.6 configuration. Emitted bars are validated,
hashed, and written incrementally; they are not accumulated for the full
dataset. A final bounded catalog rescan proves persisted count and content
hashes. If a failed run leaves derived files without a manifest, the next run
fails closed instead of silently adopting them.

Session QA uses the same paired scanner. Its cross-chunk state is only the next
minute, six counters, and one active missing interval. It merges bounded,
strictly ordered BID/ASK timestamps minute by minute, preserving
`expected_market_closed`, `unexplained_missing`, missing-side labels, interval
coalescing, readiness, and semantic hashes. Only completed interval details,
which are themselves required manifest output, accumulate. No whole-catalog
bar list or timestamp set remains.

### Targeted GitHub reuse audit

| Candidate | Finding |
| --- | --- |
| [NautilusTrader catalog/DataFusion path](https://github.com/nautechsystems/nautilus_trader/blob/develop/docs/integrations/databento.md#performance-considerations) | Adopted through the exact installed rc2 APIs above. Nautilus performs predicate-aware Parquet decoding into canonical `Bar` objects, retaining its schema and precision contract. |
| Installed rc2 `DataBackendSession` / `DataQueryResult` | The wheel exposes iterator types, but `ParquetDataCatalog` exposes no `backend_session` bridge in this release. Reaching into an unavailable or private session path was rejected. |
| [Apache Arrow Dataset scanner](https://github.com/apache/arrow/blob/main/python/pyarrow/dataset.py) | Supports predicate pushdown and record-batch iteration, but direct scanning would require reimplementing Nautilus catalog file discovery, metadata interpretation, and binary bar decoding. It remains the importer tool, not a second catalog reader. |
| [DuckDB streaming record batches](https://github.com/duckdb/duckdb/issues/5397) | Rejected: it adds a query engine while Nautilus already uses DataFusion, and its streaming reader has connection-lifetime constraints irrelevant to the native catalog contract. |

Differential tests retain the former pure bounded-fixture path as an oracle.
With 37-minute catalog chunks (inside both 1H and 4H windows), the streaming
path matches reference OHLCV, target timestamps, dropped windows, native
callback times, readiness fields, complete derived manifests, coverage
classifications, and coverage semantic SHA-256 values. Separate cases place a
missing minute and a weekend across query boundaries and prove deterministic
incremental writes and repeated runs.

## G0.7 native execution-harness adoption

G0.7 keeps the validated G0.5 one-minute external BID and ASK bars as the only
execution market. Nautilus `BacktestEngine` runs with `BookType.L1_MBP`,
`bar_execution=True`, and `trade_execution=False`. Its matching engine waits
for equal-`ts_init` BID/ASK bars and internally constructs the paired quote
updates used by L1 bar execution. FTMOQuant validates that pairing before the
run, but does not synthesize quote ticks, alter OHLC, or add a second spread.
G0.6 1H/4H bars never drive matching.

The typed execution profile supplies every execution assumption explicitly.
`ProbabilisticFillModel` provides limit-fill probability, adverse one-tick
slippage, and its deterministic seed. `StaticLatencyModel` provides base plus
insert/update/cancel nanosecond latency. Fees use native `FixedFeeModel`,
`PerContractFeeModel`, or `MakerTakerFeeModel`; maker/taker rates are applied to
an in-memory instrument clone, never the catalog instrument. The venue is a
native MARGIN account with runtime capital, base currency, and leverage, and
the runner persists normalized exports of Nautilus's account, orders,
order-fills, fills, and positions reports.

`FXRolloverInterestModule` is reused directly with explicit
`InterestRateRecord` inputs. Enabling it requires the profile to be labelled at
least `proxy`; it is not broker/FTMO swap calibration. A
`broker_calibrated` profile additionally requires the SHA-256 of external
broker evidence. No commission, latency, leverage, swap rate, or other FTMO
execution value is supplied by G0.7.

Research readiness and execution calibration are independent. Research mode
requires a G0.6 manifest bound to the current G0.5 manifest, with
`derived_bar_integrity_valid=true` and `research_ready=true`. Test/probe mode
may use documented synthetic inputs. The execution probe is a predetermined
market/limit entry and optional exit or overnight hold; it contains no
indicator, signal, or trading hypothesis.

The installed `2.0.0rc2` generated stubs were checked before implementation:
`nautilus_trader/backtest/__init__.pyi` for `BacktestEngine`,
`BacktestRunConfig`, `BacktestVenueConfig`, `BacktestDataConfig`, report APIs,
`FXRolloverInterestModule`, and `InterestRateRecord`; and
`nautilus_trader/execution/__init__.pyi` for the fill, fee, and latency model
constructors. The following upstream files at reviewed commit
`7f0e93dfa3f09ca165a5f3292a45fafbb5681561` supplied behavioral corroboration:

- [`docs/concepts/backtesting/bar-execution.md`](https://github.com/nautechsystems/nautilus_trader/blob/7f0e93dfa3f09ca165a5f3292a45fafbb5681561/docs/concepts/backtesting/bar-execution.md)
  for native BID/ASK bar pairing and deterministic/adaptive OHLC traversal.
- [`docs/concepts/backtesting/fill-models.md`](https://github.com/nautechsystems/nautilus_trader/blob/7f0e93dfa3f09ca165a5f3292a45fafbb5681561/docs/concepts/backtesting/fill-models.md)
  and
  [`crates/execution/src/models/fill.rs`](https://github.com/nautechsystems/nautilus_trader/blob/7f0e93dfa3f09ca165a5f3292a45fafbb5681561/crates/execution/src/models/fill.rs)
  for probability boundaries, one-tick adverse slippage, and seeded RNG.
- [`crates/execution/src/matching_engine/mod.rs`](https://github.com/nautechsystems/nautilus_trader/blob/7f0e93dfa3f09ca165a5f3292a45fafbb5681561/crates/execution/src/matching_engine/mod.rs)
  for exact paired-bar timestamp behavior and L1 matching.
- [`crates/execution/src/models/latency.rs`](https://github.com/nautechsystems/nautilus_trader/blob/7f0e93dfa3f09ca165a5f3292a45fafbb5681561/crates/execution/src/models/latency.rs)
  and
  [`crates/execution/src/models/fee.rs`](https://github.com/nautechsystems/nautilus_trader/blob/7f0e93dfa3f09ca165a5f3292a45fafbb5681561/crates/execution/src/models/fee.rs)
  for the adopted native models.
- [`crates/backtest/src/modules/fx_rollover.rs`](https://github.com/nautechsystems/nautilus_trader/blob/7f0e93dfa3f09ca165a5f3292a45fafbb5681561/crates/backtest/src/modules/fx_rollover.rs)
  and the rollover, BID/ASK bar, fee, and latency cases in
  [`crates/backtest/tests/exchange.rs`](https://github.com/nautechsystems/nautilus_trader/blob/7f0e93dfa3f09ca165a5f3292a45fafbb5681561/crates/backtest/tests/exchange.rs).

hftbacktest's latency architecture and QuantConnect LEAN were reference-only;
neither was added as a dependency or used to create another matcher or ledger.

Nautilus rc2 has one report-level reproducibility limitation: native event and
order-initialization UUIDs remain random even with `use_random_ids=False`.
G0.7 retains deterministic native venue-order, trade, and position IDs and
removes only those transport UUID columns/keys from the immutable semantic CSV
exports and hashes. Economic fill, position, and account content is unchanged.

## G0.8 statistical resampling adoption

G0.8 adopts [`bashtage/arch`](https://github.com/bashtage/arch) at exactly
`8.0.0`, under its NCSA license. The isolated research-statistics module calls
`StationaryBootstrap.conf_int`, `optimal_block_length`, `SPA` or
`RealityCheck`, and `MCS` directly. FTMOQuant validates ordering, alignment,
finite values, unique model labels, explicit seeds and resampling parameters;
it does not implement their sampling or test mathematics. Block-length
estimates are returned to the caller and are never substituted into a later
procedure automatically.

The installed `8.0.0` package was inspected before implementation. Exact API
signatures and behavior were reviewed in:

- `arch/bootstrap/base.py`: `optimal_block_length`, `IIDBootstrap.conf_int`,
  and `StationaryBootstrap`.
- `arch/bootstrap/multiple_comparison.py`: `SPA`, `RealityCheck`, and `MCS`,
  including `compute`, `pvalues`, `better_models`, `included`, and `excluded`.
- `arch/bootstrap/__init__.py`: the public exports used by FTMOQuant.
- `arch-8.0.0.dist-info/METADATA` and `LICENSE.md`: installed version and NCSA
  licensing.
- The upstream 8.0.0 API pages for
  [`StationaryBootstrap`](https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.StationaryBootstrap.html),
  [`SPA`](https://bashtage.github.io/arch/multiple-comparison/generated/arch.bootstrap.SPA.html),
  and the 8.0.0 API index entries for `optimal_block_length` and `MCS`.

Fixed-fixture tests compare every wrapper with a direct seeded arch call. No
substantive upstream test or source code was copied. The wrappers do not know
about final holdouts, rank or select models, translate losses from returns, or
turn a statistical result into a trading decision. Ordinary performance
metrics remain Nautilus analysis responsibilities.

The G0.8 mean-CI adapter is deliberately bounded to arch's nonparametric,
two-sided callback contract. Parametric and semi-parametric sampling require a
different statistic callback signature and are not exposed by this mean-only
wrapper. SPA/Reality Check procedure names and MCS methods fail closed rather
than falling through to another native procedure or method. Native optimal
block-length estimates must be finite and positive or the adapter rejects
them without substitution, and MCS requires at least two loss-model columns.

## G1.1 strategy-specification audit

The bounded G1.1 audit reviewed only NautilusTrader because a second repository
or dependency did not materially improve a static hypothesis contract or CSV
registry. The installed, pinned `nautilus-trader==2.0.0rc2` wheel was checked
for `StrategyConfig`, `ImportableStrategyConfig`, `register_indicator_for_bars`,
`AverageTrueRange`, moving-average types, `BarType`, and indicator initialized
state. These are the intended G1.2 implementation facilities; no strategy code
is present in G1.1.

For current upstream corroboration, the audit reviewed
[`nautechsystems/nautilus_trader`](https://github.com/nautechsystems/nautilus_trader)
at commit
[`1158ab32b88e6c78a03a80d6e8fb6930f1433e7d`](https://github.com/nautechsystems/nautilus_trader/commit/1158ab32b88e6c78a03a80d6e8fb6930f1433e7d),
under LGPL-3.0-or-later:

- `docs/concepts/strategies.md` for the shared backtest/live strategy source,
  explicit `StrategyConfig`, lifecycle handlers, and bar callbacks;
- `examples/backtest/example_07_using_indicators/strategy.py` for native
  moving-average creation, bar registration, and fail-closed indicator warm-up;
  and
- `python/tests/strategies/ema_cross.py` for typed configuration and explicit
  completed-bar crossover state.

No upstream source or test text was copied, no dependency was added, and the
upstream moving-average crossover is not the FTMOQuant hypothesis. G1.1 adapts
only the architectural decision to keep configuration explicit and to rely on
native bar/indicator lifecycle in G1.2. Because the reviewed current upstream
commit is newer than the pinned rc2 artifact, installed rc2 stubs remain the
runtime authority.

## G1.2 native strategy implementation audit

The bounded G1.2 audit again selected only the already-pinned
`nautilus-trader==2.0.0rc2`; no second engine or indicator dependency would
reduce bespoke code or improve fidelity. The installed rc2 generated stubs
were treated as runtime authority for `Strategy`, `StrategyConfig`, completed
bar callbacks and subscriptions, `ExponentialMovingAverage`,
`AverageTrueRange`, `MovingAverageType.Exponential`, indicator `initialized`
and `reset` state, market-order creation, fill callbacks, and native position
closure.

Current upstream corroboration reviewed
[`nautechsystems/nautilus_trader`](https://github.com/nautechsystems/nautilus_trader)
at commit
[`3e34ddb79ea2961ea6ce230cf4e80ef2fb292a32`](https://github.com/nautechsystems/nautilus_trader/commit/3e34ddb79ea2961ea6ce230cf4e80ef2fb292a32),
under LGPL-3.0-or-later, including the public indicator declarations,
`register_indicator_for_bars`, strategy lifecycle callbacks, and native order
factory interfaces. No upstream source or test text was copied.

Native bar registration cannot correctly produce this hypothesis's required
synchronized BID/ASK midpoint: registering against one raw side uses the wrong
price and registering against both updates twice. FTMOQuant therefore retains
native indicator calculation and lifecycle but feeds each indicator exactly
once through its public `update_raw` API after a valid midpoint pair. Missing
or invalid pairs reset affected indicator state rather than carrying values.
This is the smallest adapter necessary to preserve the frozen G1.1 semantics.

`empyrical-reloaded` and QuantStats remain deferred/reference-only and are not
dependencies in G0.8. Neither is needed for the adopted resampling and
multiple-comparison scope.

## G0.9 end-to-end FTMO observation

G0.9 composes the existing `NautilusFtmoOverlay` and
`NautilusAccountSnapshotSource` with the G0.7 strategy through a small
`NautilusFtmoBridge`. The bridge contains no floor, pass, matching, P/L, fee,
or rollover calculation. It forwards native `OrderFilled` and
`PositionOpened` callbacks to the overlay and reads balance and open positions
from Nautilus through the existing snapshot boundary. The runtime
`EvaluationPhase` is mandatory in every execution request; there is no default
Challenge phase.

For external one-minute bars, the bridge waits until strategy callbacks have
seen both BID and ASK with the same `ts_init`. It then schedules a native clock
alert exactly one nanosecond later. The runner extends only the observation
horizon by that one nanosecond so the final pair is also observed; source data,
requested range, matching, and scripted orders are unchanged. This mechanism
is versioned as `g0.9-3` in the execution manifest. Every observation reads
native state and the final manifest records phase, FTMO day, reset balance,
both loss floors, counted days, latched terminal status and breach evidence,
plus a deterministic observation hash.

The installed NautilusTrader `2.0.0rc2` stubs and wheel behavior were checked
before implementation. A minimal targeted engine probe with paired bars, a
fixed fee, an open FX position, an FX rollover boundary, and native clock
alerts established the following ordering:

- Portfolio subscribes to `data.bars.*EXTERNAL` before strategy callbacks, so
  each external bar has refreshed native portfolio state by `on_bar`.
- `OrderFilled` and `PositionOpened` strategy callbacks see the native cache,
  account commission, portfolio, and position updates already applied.
- A resting order triggered by an incoming external bar fills in exchange
  processing before that same bar is dispatched to the strategy. In the
  targeted rc2 probe the fill and position-open timestamps were
  `2024-01-02T00:02:00Z`, while the bridge still held the completed
  `00:01:00Z` marks; the BID and ASK callbacks subsequently completed the
  `00:02:00Z` pair before its `00:02:00.000000001Z` alert.
- Same-timestamp native timer callbacks are drained before venue modules.
- `FXRolloverInterestModule` applies its account adjustment during venue-module
  timestamp finalization. A completed-pair alert at `ts_init + 1 ns` therefore
  sees the adjusted account on the immediately following deterministic native
  clock cycle, without a later trade.
- The overlay's existing Prague reset alert captures the native account
  balance at midnight and reschedules itself; holding a position does not emit
  another `PositionOpened` or count another trading day.

Behavioral source references were reviewed at upstream commit
`7f0e93dfa3f09ca165a5f3292a45fafbb5681561`, while the installed rc2 wheel
remains authoritative:

- [`docs/concepts/backtesting/execution-flow.md`](https://github.com/nautechsystems/nautilus_trader/blob/7f0e93dfa3f09ca165a5f3292a45fafbb5681561/docs/concepts/backtesting/execution-flow.md)
  for native command, execution-event, cache, portfolio, and strategy flow.
- [`crates/portfolio/src/portfolio.rs`](https://github.com/nautechsystems/nautilus_trader/blob/7f0e93dfa3f09ca165a5f3292a45fafbb5681561/crates/portfolio/src/portfolio.rs)
  for external-bar subscription priority and `update_bar` behavior.
- [`crates/model/src/position.rs`](https://github.com/nautechsystems/nautilus_trader/blob/7f0e93dfa3f09ca165a5f3292a45fafbb5681561/crates/model/src/position.rs)
  for native `Position.unrealized_pnl(Price)` liquidation-mark valuation.
- [`crates/backtest/src/engine.rs`](https://github.com/nautechsystems/nautilus_trader/blob/7f0e93dfa3f09ca165a5f3292a45fafbb5681561/crates/backtest/src/engine.rs)
  for timer draining, timestamp finalization, venue settlement, and module
  ordering.
- [`crates/backtest/src/modules/fx_rollover.rs`](https://github.com/nautechsystems/nautilus_trader/blob/7f0e93dfa3f09ca165a5f3292a45fafbb5681561/crates/backtest/src/modules/fx_rollover.rs)
  and the rollover cases in
  [`crates/backtest/tests/exchange.rs`](https://github.com/nautechsystems/nautilus_trader/blob/7f0e93dfa3f09ca165a5f3292a45fafbb5681561/crates/backtest/tests/exchange.rs)
  for native account-adjustment settlement.

Nautilus rc2 Portfolio has an important external-bar limitation: its fallback
bar close is keyed by instrument rather than BID/ASK side. With deterministic
BID-then-ASK dispatch, native Portfolio equity therefore marks both long and
short positions at the final ASK close. This is not liquidation-side correct
for a long position and can change an FTMO floor decision.

G0.9 corrects only the observational snapshot boundary. After a complete pair,
`PairedBarLiquidationSnapshotSource` marks LONG positions at BID close and
SHORT positions at ASK close. It delegates each ephemeral open-position value
to the installed rc2 native `Position.unrealized_pnl(Price)` method, sums those
native `Money` results onto native account balance, and persists no P/L state.
Fees, realized P/L, and rollover remain native account-balance effects. The
native Portfolio fallback equity is retained separately as a diagnostic, with
the completed mark timestamp, BID/ASK closes, authoritative FTMO equity, and
their difference. Unsupported non-EUR/USD or non-USD settlement/account cases
are rejected rather than silently converted.

Native fill and position-open callbacks use immediate valuation only when the
snapshot source already holds a completed pair at least as current as the
native event timestamp. A phase-1 resting-order fill arriving before strategy
bar dispatch is queued for the matching pair's existing `+1 ns` observation;
fees and position state remain native and are read once there. The associated
`PositionOpened` CE(S)T trading day is recorded immediately from its original
native timestamp through the overlay's public day-bookkeeping API, without a
floating-equity read. The eventual paired observation records the deferred fill
and position-open timestamps for auditability. This coalesces deferred and
normal paired compliance into one evaluation and cannot present a previous
minute's liquidation mark as current.

The installed rc2 primitive was verified with a targeted long/short probe. A
100,000 EUR long opened at ASK 1.10020 and marked at BID 1.05000 produced native
`Position.unrealized_pnl` of `-5020.00 USD`, while Portfolio's ASK fallback
reported equity 95000.00; authoritative liquidation equity was 94980.00. A
short position was correspondingly marked at ASK. Closing immediately at the
same liquidation-side price reconciles the ephemeral native unrealized value
with native realized account balance when exit fees are zero.

This correction does not make one-minute bars equivalent to continuous price
monitoring. Provenance explicitly records `valuation_resolution` as
`1-minute paired-bar close` and
`continuous_intraminute_compliance_exact=false`. Intraminute FTMO breaches can
therefore remain unobserved; tick-level compliance is outside this G0 fix.

## Known evaluation limits

- The integration surface currently expects its owner to forward native
  position-open and fill events and to refresh on market/account observations;
  it is not a standalone Nautilus actor.
- G0.7 exercises synthetic rollover, latency, slippage, limit probability, and
  fee fixtures, but none is evidence of broker adapter fidelity, partial-fill
  calibration, or live reconciliation.
- Report generation requires pandas; it is installed directly instead of the
  broader Nautilus `visualization` extra.

## G1.4B Stage G tournament infrastructure audit

The targeted pre-implementation audit covered synchronized multi-instrument
clocks, FX/currency exposure, portfolio limits, per-instrument costs,
experiment registries, model-selection controls, and marimo. The adopted or
referenced projects are:

| Repository | Version / commit | License | Role and adoption | Modifications and validation |
| --- | --- | --- | --- | --- |
| [marimo-team/marimo](https://github.com/marimo-team/marimo/tree/0.23.15) | `marimo==0.23.15`; tag commit [`d8789daa177c759747fa2c36b985a7c36c37f048`](https://github.com/marimo-team/marimo/commit/d8789daa177c759747fa2c36b985a7c36c37f048) | Apache-2.0 | Development dependency and Git-tracked reactive UI only. No notebook runtime or source was copied. | The notebook calls tested FTMOQuant view-model modules and has fixed DEVELOPMENT mounts. Installed `marimo --version`, `marimo edit --help`, and `marimo run --help` verified the exact CLI. Static notebook checking and import tests validate integration. |
| [nautechsystems/nautilus_trader](https://github.com/nautechsystems/nautilus_trader/tree/7f0e93dfa3f09ca165a5f3292a45fafbb5681561) | Existing pinned `nautilus-trader==2.0.0rc2`; reviewed source commit `7f0e93dfa3f09ca165a5f3292a45fafbb5681561` | LGPL-3.0-or-later | Existing deterministic engine, native BID/ASK execution, fee, adverse-slippage, latency, and FX rollover boundary. Referenced for multi-instrument deterministic time and currency-aware domain modeling. | No upstream code copied and no second engine added. Stage G adds a public validation-only wrapper for the existing `ExecutionProfile`, then binds one validated profile per instrument with observed BID/ASK spread. Synthetic tests validate order, cost configuration, and causal alignment. |
| [bashtage/arch](https://github.com/bashtage/arch/tree/v8.0.0) | Existing pinned `arch==8.0.0`, tag `v8.0.0` | NCSA | Existing SPA, Reality Check, MCS, stationary bootstrap, and optimal block-length infrastructure. | No upstream code copied or modified. Stage G preregisters how the already-tested wrappers will be used later; it performs no model comparison or return calculation now. |

NautilusTrader deliberately remains the execution realism dependency, while
`arch` remains the multiple-comparison dependency. External experiment trackers
and portfolio/backtest frameworks were not adopted: the required registry is a
small immutable six-entry metadata contract, and adding MLflow, Sacred,
vectorbt, Backtrader, LEAN, or another portfolio engine would duplicate pinned
project facilities without improving the leakage boundary. Currency-incidence
aggregation and DEVELOPMENT manifest validation are narrow FTMOQuant seams not
provided by the pinned versions with the required frozen hashes and fail-closed
split policy.

## G1.4C Phase 1 time-series momentum audit

This audit was completed before implementing `ts_momentum_v1`. It was limited
to reusable indicator, daily-history, causal scheduling, and target/execution
facilities; no strategy returns or market-data rows were inspected.

| Repository | Version / commit | License | Role and adoption | Modifications and validation |
| --- | --- | --- | --- | --- |
| [nautechsystems/nautilus_trader](https://github.com/nautechsystems/nautilus_trader/tree/7f0e93dfa3f09ca165a5f3292a45fafbb5681561) | Existing pinned `nautilus-trader==2.0.0rc2`; source reference `7f0e93dfa3f09ca165a5f3292a45fafbb5681561`; current audit head `409214a9d2d23ecae72a7d9376b06afc1ecc7694` | LGPL-3.0-or-later | Adopted the installed native `RateOfChange(use_log=True)` indicator and retained the existing Nautilus execution boundary. | No source was copied. An installed-wheel probe established that native `period` is the total window length: `period=253` gives the frozen current-versus-252-prior-observation logarithmic change. Synthetic differential tests validate the native output against `ln(C_t/C_(t-252))`. Generic native daily time bars were not adopted because they do not encode the provider-specific 17:00 `America/New_York` close across DST. |
| [quantopian/zipline](https://github.com/quantopian/zipline/tree/014f1fc339dc8b7671d29be2d85ce57d3daec343) | `014f1fc339dc8b7671d29be2d85ce57d3daec343` | Apache-2.0 | Reference-only review of daily history windows, scheduled callbacks, and `order_target`. | Not adopted: its separate data portal, calendar, order, portfolio, and simulation stack would duplicate Stage G and Nautilus, while its APIs do not provide the frozen readiness hashes or strictly-later synchronized execution contract. |
| [polakowo/vectorbt](https://github.com/polakowo/vectorbt/tree/34b6d5935e3ea3eccd549e2592bc0f455b8045f5) | `34b6d5935e3ea3eccd549e2592bc0f455b8045f5` | Apache-2.0 with Commons Clause | Reference-only review of vectorized signal/portfolio construction. | Not adopted: it would add a parallel backtester, fill/cost model, and portfolio accounting path. No code or implementation pattern was copied. |

The remaining project code is deliberately narrow: it recognizes the frozen
Dukascopy session close from observed Stage G frames, keeps independent
per-instrument native ROC state, maps only the sign to `{-1, 0, +1}`, and holds
changed targets until the first synchronized tradable frame whose information
time is strictly later. Stage G continues to own data admission, clock
alignment, folds, costs, currency exposure, limits, and tournament statistics.

## G1.4C Phase 2 DEVELOPMENT evaluator audit

This bounded audit was completed before evaluator implementation. It covered
only the pinned engine/report and statistical APIs needed to connect the
already-frozen candidate to Stage G. No market rows, strategy returns,
validation data, or final-holdout data were opened.

| Repository | Version / commit | License | Role and adoption | Modifications and validation |
| --- | --- | --- | --- | --- |
| [nautechsystems/nautilus_trader](https://github.com/nautechsystems/nautilus_trader/tree/7f0e93dfa3f09ca165a5f3292a45fafbb5681561) | Existing pinned `nautilus-trader==2.0.0rc2`; source reference `7f0e93dfa3f09ca165a5f3292a45fafbb5681561` | LGPL-3.0-or-later | Adopted the existing low-level `BacktestEngine` pattern: add the frozen instruments and BID/ASK bars, add one thin target-to-order strategy, run deterministically, then use native account/order/fill/position state and reports. The installed native `Position.unrealized_pnl` remains the liquidation-side open-position valuation primitive. | No upstream source was copied. The evaluator reuses FTMOQuant's G0.7 engine, venue, fee, latency, slippage, rollover, and report adapters. Synthetic engine tests cover target execution and provenance. A direct native oracle proves that `RateOfChange(period=253, use_log=True)` is exactly the frozen 252-prior-observation log change. |
| [bashtage/arch](https://github.com/bashtage/arch/tree/v8.0.0) | Existing pinned `arch==8.0.0`, tag `v8.0.0` | NCSA | Reuses the existing tested FTMOQuant wrappers for the preregistered stationary-bootstrap mean interval and SPA comparison with zero return. | No source was copied or modified. The evaluator supplies deterministic DEVELOPMENT-only daily series to the wrappers. MCS is explicitly not run for a one-candidate family because the existing wrapper correctly requires at least two models. |

No additional backtester, portfolio framework, cost package, experiment tracker,
or statistics package was adopted. Such a dependency would duplicate the
already-pinned Nautilus and `arch` boundaries. The local code is restricted to
DEVELOPMENT admission, frozen-fold orchestration, raw-target order adaptation,
currency-incidence limit checks, metric presentation, and deterministic
artifact provenance.

The follow-up artifact-materialization audit reached the same reuse decision.
Nautilus owns the cost semantics, while FTMOQuant's tracked frozen Phase 2 JSON
owns their repository identity. The smallest generator is therefore an exact
byte copy to the one reserved artifact path followed by the existing semantic,
instrument-order, and canonical-profile hash validation; no serializer,
estimator, calibration step, or additional dependency was added.

## G1.4D Phase 1 session-range expansion audit

This bounded pre-implementation audit covered only timestamp/session handling,
raw target scheduling, and the existing Stage G fold boundary. No market rows,
returns, validation data, or final-holdout data were opened.

| Repository | Version / commit | License | Role and adoption | Modifications and validation |
| --- | --- | --- | --- | --- |
| [nautechsystems/nautilus_trader](https://github.com/nautechsystems/nautilus_trader/tree/7f0e93dfa3f09ca165a5f3292a45fafbb5681561) | Existing pinned `nautilus-trader==2.0.0rc2`; source reference `7f0e93dfa3f09ca165a5f3292a45fafbb5681561` | LGPL-3.0-or-later | Reference for UTC-native event timestamps and the existing Stage G/Nautilus execution boundary. | No source was copied. The candidate keeps the provider-specific London session definition in a small `ZoneInfo("Europe/London")` state machine, then reuses Stage G’s synchronized frame and strict-later execution semantics. Synthetic winter/summer tests validate DST conversion and session boundaries. |
| Python standard library `zoneinfo` | Python 3.12 runtime | PSF | Adopted for IANA `Europe/London` conversion; it adds no dependency or trading calendar layer. | No external calendar, backtester, or session framework was adopted. A fixed 480-completed-minute invariant rejects incomplete session ranges without filling prices. |

Zipline, vectorized portfolio frameworks, and external calendar/session packages
were not adopted because they would duplicate Stage G synchronization, frozen
fold admission, and the existing Nautilus execution boundary. The Phase 1 code
only records a range, maps the first breakout to a raw target, schedules the
16:00 flat target, and resets state per fold.

## G1.4D Phase 2 session-range evaluator audit

This bounded pre-implementation audit covered only the existing G1.4C
DEVELOPMENT evaluator seam, Stage G candidate/evaluation interfaces, and the
published upstream execution/statistics integrations. No market rows, returns,
validation data, or final-holdout data were opened.

| Repository | Version / commit | License | Role and adoption | Modifications and validation |
| --- | --- | --- | --- | --- |
| [nautechsystems/nautilus_trader](https://github.com/nautechsystems/nautilus_trader/tree/7f0e93dfa3f09ca165a5f3292a45fafbb5681561) | Existing pinned `nautilus-trader==2.0.0rc2`; source reference `7f0e93dfa3f09ca165a5f3292a45fafbb5681561` | LGPL-3.0-or-later | Reused through the existing G1.4C native order adapter, causal timestamp handling, G0.7 execution profile, and reports. | No source or replacement engine was added. The session adapter supplies only frozen raw targets to the shared evaluator. |
| [bashtage/arch](https://github.com/bashtage/arch) | Existing pinned project dependency | NCSA | Reused through the existing Stage G stationary-bootstrap and SPA wrappers. | No statistics implementation, parameter search, or new dependency was added. |

The adopted path was to factor the existing G1.4C orchestration into one
candidate-parameterized, DEVELOPMENT-only evaluator and add a small
session-range raw-target adapter. It auto-materializes the already-frozen
canonical cost artifact as the validated byte-identical tracked configuration.
No parallel backtester, cost model, portfolio model, provenance system, or
statistics runner was created.

### G1.4D incomplete-frame release fix

The bounded GitHub-first re-audit reconfirmed the existing Nautilus native
`BacktestEngine` as the sole execution boundary; its documented UTC,
nanosecond, timestamp-ordered processing is unchanged. No dependency or source
code was adopted. The local session adapter now withholds a pending raw target
from the shared native instruction converter until its strictly-later Stage G
frame is both tradable and has valid synchronized prices. This reuses the
existing pending-target state and preserves it across incomplete frames rather
than adding an execution queue or alternate engine.

### G1.4D metadata-label cleanup

The bounded GitHub-first audit of the existing `arch` integration confirmed that
the model label is supplied by the input series/DataFrame. The shared evaluator
now derives that label from the frozen candidate ID, preserving all numerical
inputs, bootstrap/SPA settings, and execution behavior. No dependency, source
code, or statistics implementation was added.

## Carver FTMO5 final DEVELOPMENT evaluator audit

This audit was completed before any Carver strategy return was accessed. The
exact pinned source commit was checked out solely to inspect configuration and
system semantics. No validation or final-holdout path was opened.

| Repository | Version / commit | License | Role and adoption | Modifications and validation |
| --- | --- | --- | --- | --- |
| [robcarver17/pysystemtrade](https://github.com/robcarver17/pysystemtrade/tree/b4a25e6e1e33a54a3ecfb45c0f6db5e2b60b84f8) | `b4a25e6e1e33a54a3ecfb45c0f6db5e2b60b84f8` | GPL-3.0 | Reference-only audit of `systems/provided/rob_system/config.yaml`, `systems/positionsizing.py`, `systems/portfolio.py`, `systems/forecast_combine.py`, `systems/rawdata.py`, `systems/provided/rules/ewmac.py`, `systems/provided/rules/carry.py`, `sysquant/estimators/vol.py`, `sysobjects/carry_data.py`, and the P&L calculators. Adopted the documented 500,000 USD notional capital, 25% target, 256-day cash-vol scaling, forecast/10 sizing structure, instrument-weight-before-IDM order, price-unit volatility, contract-date roll annualisation, delayed position accounting concept, and daily mark-to-market aggregation. | No upstream source text or package was copied or vendored. FTMOQuant implements a typed, independently written adapter for the already-preregistered rule structure. The upstream whole-contract rounding, large-universe weights/IDM, futures costs, and broker economics were not adopted because frozen G0.8 continuous CFD targets and five-market FTMO economics govern those boundaries. |
| Existing FTMOQuant G0.7/G0.8 and Stage G | Repository HEAD before evaluation | Project-native | Reused observed BID/ASK side selection, strict-later executable observations, liquidation-side valuation, fixed-capital daily return reporting, 1.5× realised-cost stress, deterministic folds/artifacts, and G0.8 contract, margin, commission, session, soybean-scale, and rollover-warning semantics. | No parallel broker-economics package or favourable basis/fill model was added. Synthetic tests cover every mapped contract shape and the sealed-data boundary. |
| [bashtage/arch](https://github.com/bashtage/arch/tree/v8.0.0) | Existing exact pin `arch==8.0.0` | NCSA | Reused through `ftmoquant.research.statistics` for the two-sided 95% basic stationary-bootstrap mean interval with block 20, 10,000 repetitions, and seed 14042026. | No bootstrap implementation or data-driven block selection was added. |

The pinned `rob_system` IDM of 2.75 was rejected as inapplicable because it is
estimated for a structurally different, much larger universe. The neutral
five-market IDM of 1.0, next-UTC-midnight signal completion, aggregate G0.8
margin fail-closed rule, and separation of deterministic result content from
wall-clock run metadata are explicitly new performance-blind preregistered
conventions. No backtester, parameter optimizer, financing model, or data
dependency was added.
