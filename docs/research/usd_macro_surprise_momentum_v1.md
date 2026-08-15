# usd_macro_surprise_momentum_v1 — preregistration

This is a DEVELOPMENT-only, not-yet-evaluated macro-event hypothesis. It has no
results and does not authorize validation or final-holdout access.

The frozen event archive is the Hanover / Forex Factory UTC ZIP with SHA-256
`3fb4421df0ea63cac570b7adcd16892ec50909ad7d3c441d462443245a5d84ce`.
Only US Non-Farm Employment Change and US CPI headline m/m are admitted.

For each release, `Actual - Forecast` is the surprise: positive is USD-positive,
negative is USD-negative, and zero emits no trade. EUR/USD and GBP/USD are
shorted for USD-positive releases and bought for USD-negative releases. Their
returns form one equal-weight event portfolio before inference.

The timestamp is the frozen UTC release time. Entry is the first eligible
executable observation at/after `t + 5m`; exit is the first at/after `t + 60m`.
There are no stops, targets, trailing logic, filters, special news fills, or
same-instrument overlaps. G0.7 bid/ask execution/costs are evaluated at base
and 1.5× costs.

The full immutable contract, including event-level bootstrap reporting and the
development promotion/retirement gate, is
`config/strategies/usd_macro_surprise_momentum_v1.yaml`.
