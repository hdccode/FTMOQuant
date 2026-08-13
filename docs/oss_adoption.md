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
