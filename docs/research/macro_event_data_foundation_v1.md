# Macro-event data foundation v1

This is data infrastructure only. It does not define a trade, a surprise, an
event window, a return, or a backtest.

## Source contract

The primary input is the historical Forex Factory calendar CSV published by
Hanover. Its stated fields are Date, Time, Currency, Impact, News description,
Actual, Forecast, Previous, Revised from, and FF event ID; it states that its
timestamps are New York EST or DST as applicable. The importer therefore
requires the IANA timezone `America/New_York` by default, retains the supplied
wall-clock values in `raw_source_fields`, and derives (rather than replaces)
the UTC timestamp.

The archive must be downloaded into a user-controlled raw-data directory before
import. Do not rewrite the raw CSV after it has been hashed. A live Forex
Factory weekly export is useful only as a non-authoritative, small fixed-sample
cross-check; it does not replace the archived source.

## Scope and normalization

Only these exact USD labels are admitted. Whitespace and case changes are
ignored; no fuzzy matching is performed.

| Event family | Accepted labels |
| --- | --- |
| `US_NFP_HEADLINE_EMPLOYMENT_CHANGE` | Non-Farm Employment Change; Nonfarm Employment Change; Non-Farm Payrolls; Nonfarm Payrolls |
| `US_CPI_HEADLINE_M_M` | CPI m/m |
| `US_CPI_HEADLINE_Y_Y` | CPI y/y |
| `US_CPI_CORE_M_M` | Core CPI m/m |
| `US_CPI_CORE_Y_Y` | Core CPI y/y |

This intentionally excludes the unemployment rate and all rate-decision
records. A later specification may add another family only with an explicit,
versioned normalizer change.

## Import

The importer writes `normalized_events.jsonl`, `quarantined_events.jsonl`,
`qa_report.json`, and `provenance_manifest.json`. Numeric parsing is additive:
the `*_raw` values are never changed and parsed values state `parsed`,
`missing`, or `malformed`.

Exact duplicate records and malformed/non-exact/ambiguous/nonexistent local
timestamps are quarantined. Midnight timestamps are retained but reported as
suspicious. Revisions are retained as raw and parsed fields and reported, never
folded into the original previous value.

No production archive is committed to this repository, and no price split,
validation, or final-holdout data is read by this pipeline.
