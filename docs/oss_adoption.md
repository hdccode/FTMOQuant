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
| tradedesk-dukascopy | [radiusred/tradedesk-dukascopy](https://github.com/radiusred/tradedesk-dukascopy) | Apache-2.0 | Dukascopy BI5 acquisition, recovery, decoding, scale probe, and 1-minute export | Adopt at exactly `1.0.0` / commit `b8fb503c9291d6e265949d008e288b76b68fb852`; FTMOQuant calls its public export CLI contract and retains its CSV metadata sidecars |
| LEAN | [QuantConnect/Lean](https://github.com/QuantConnect/Lean) | Apache-2.0 | Full event-driven trading engine alternative | Not adopted; .NET-centric runtime adds operational weight to this Python project |
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

## Known evaluation limits

- The integration surface currently expects its owner to forward native
  position-open and fill events and to refresh on market/account observations;
  it is not a standalone Nautilus actor.
- Swap/rollover data is exposed by Nautilus modules but is not exercised by the
  intraday probe.
- The probe validates one deterministic L1 margin-account round trip, not broker
  adapter fidelity, latency, slippage, partial fills, or live reconciliation.
- Report generation requires pandas; it is installed directly instead of the
  broader Nautilus `visualization` extra.
