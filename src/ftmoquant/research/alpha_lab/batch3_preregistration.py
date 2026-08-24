"""B3.0 methodology preregistration for Batch 3 ("prop-path-shape" families).

This module freezes the Batch 3 research protocol -- family scope, DEVELOPMENT
advancement gates, FTMO-specific DEVELOPMENT gates, the sizing-selection
procedure, and the VALIDATION policy -- BEFORE any Batch 3 DEVELOPMENT data is
loaded, any backtest is run, or any Monte Carlo simulation is performed.

Unlike every other preregistration in :mod:`ftmoquant.research.alpha_lab`
(``validation.py``, ``pair_specific_validation.py``), which freeze a
VALIDATION-stage decision immediately before VALIDATION data is touched, this
module freezes the *methodology itself* one stage earlier, before any
DEVELOPMENT screening occurs. It therefore has no dependency on any Stage-1/
Stage-2 scorecard, no dependency on ``nautilus_trader``'s ``ParquetDataCatalog``,
and performs no data I/O of any kind -- only hashing of small, already-frozen
YAML/JSON *configuration* files (never price or trade data) to bind this
document to the exact lineage/rule-set versions it reasons about.

Reused unchanged from existing infrastructure (see module docstring
cross-references below rather than reimplementing): the DEVELOPMENT/
VALIDATION/HOLDOUT partition boundaries (:mod:`ftmoquant.research.stage_g`),
the OANDA alpha-lab DEVELOPMENT lineage config
(:mod:`ftmoquant.data.oanda_alpha_lab_development`), the frozen FTMO rule set
(``config/prop/ftmo_2step_swing_2026-08.yaml``), the canonical-JSON semantic
hash convention used by every other preregistration in this package
(``_canonical_sha256`` in :mod:`ftmoquant.research.alpha_lab.validation`),
the existing ``SIZING_GRID`` / two-stage sizing-refinement precedent
(:mod:`ftmoquant.research.ftmo_pass_probability.sizing`), and the existing
N-D connected-parameter-region check
(:mod:`ftmoquant.research.alpha_lab.screening_stage2`,
:mod:`ftmoquant.research.alpha_lab.liquidity_structure_screen`).
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ftmoquant.data.oanda_alpha_lab_development import (
    CONFIG_PATH as OANDA_ALPHA_LAB_DEVELOPMENT_CONFIG_PATH,
)
from ftmoquant.data.oanda_alpha_lab_development import (
    LINEAGE_ID as OANDA_ALPHA_LAB_LINEAGE_ID,
)
from ftmoquant.research.stage_g import (
    DEVELOPMENT_END_EXCLUSIVE,
    DEVELOPMENT_START,
    HOLDOUT_START,
    VALIDATION_START,
)

PREREGISTRATION_VERSION = "batch3-methodology-preregistration-1"
STAGE_ID = "B3.0"

#: Frozen FTMO rule set this preregistration reasons about. Reused unchanged;
#: never re-derived per family.
PROP_RULE_SET_PATH = Path("config/prop/ftmo_2step_swing_2026-08.yaml")
PROP_RULE_SET_ID = "ftmo_2step_swing"

#: Where this preregistration is written. A single, human-readable document
#: per the task's own preference for "a single human-readable preregistration
#: file under the repo's existing config/research or equivalent convention".
PREREGISTRATION_PATH = Path(
    "config/research/batch3_methodology_preregistration_v1.json"
)

_DEVELOPMENT_SPAN_DAYS = (DEVELOPMENT_END_EXCLUSIVE - DEVELOPMENT_START).days
assert _DEVELOPMENT_SPAN_DAYS == 1492
assert _DEVELOPMENT_SPAN_DAYS % 4 == 0, (
    "the four temporal-stability folds require the frozen DEVELOPMENT span "
    "to divide evenly by four; if stage_g's partition boundaries ever "
    "change, this preregistration's fold boundaries must be re-derived "
    "before Batch 3 DEVELOPMENT results are observed, not after"
)

#: Four equal-duration (373-day) chronological folds spanning the full frozen
#: DEVELOPMENT interval, derived by pure calendar-day arithmetic on the
#: already-frozen ``DEVELOPMENT_START``/``DEVELOPMENT_END_EXCLUSIVE``
#: constants -- no return, price, or trade data is inspected to place these
#: boundaries.
_FOLD_STEP_DAYS = _DEVELOPMENT_SPAN_DAYS // 4
DEVELOPMENT_FOLD_BOUNDARIES: tuple[datetime, ...] = tuple(
    DEVELOPMENT_START + timedelta(days=_FOLD_STEP_DAYS * index) for index in range(5)
)
assert DEVELOPMENT_FOLD_BOUNDARIES[-1] == DEVELOPMENT_END_EXCLUSIVE
assert DEVELOPMENT_FOLD_BOUNDARIES[0] == DEVELOPMENT_START

#: Deterministic two-way chronological split of DEVELOPMENT for the new
#: cross-subperiod FTMO ``pass_both`` stability gate: the exact midpoint of
#: the frozen DEVELOPMENT interval, by calendar-day count (``//`` floor
#: division, so this is reproducible from the frozen constants alone).
_HALF_STEP_DAYS = _DEVELOPMENT_SPAN_DAYS // 2
DEVELOPMENT_SUBPERIOD_MIDPOINT = DEVELOPMENT_START + timedelta(days=_HALF_STEP_DAYS)
assert DEVELOPMENT_SUBPERIOD_MIDPOINT == DEVELOPMENT_FOLD_BOUNDARIES[2], (
    "the two-way split midpoint is required to coincide with the fold-2/"
    "fold-3 boundary above -- both are the same floor(span/2)-derived "
    "date and must never be allowed to silently drift apart"
)


class Batch3PreregistrationError(ValueError):
    """Raised when a Batch 3 preregistration document fails verification."""


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(document: dict[str, Any]) -> str:
    """Hash all document fields except the self-referential digest itself.

    Identical convention to ``_canonical_sha256`` in
    :mod:`ftmoquant.research.alpha_lab.validation` and
    :mod:`ftmoquant.research.alpha_lab.pair_specific_validation`: canonical
    (sorted-key, compact-separator) JSON encoding, SHA-256 hex digest.
    """

    payload = {
        key: value
        for key, value in document.items()
        if key != "preregistration_semantic_sha256"
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


# ---------------------------------------------------------------------------
# Family scope (frozen; see module docstring -- this is a scope freeze, not
# an implementation -- no signal code exists for B3F1/B3F2/B3F3 yet).
# ---------------------------------------------------------------------------

IN_SCOPE_FAMILIES: dict[str, dict[str, Any]] = {
    "B3F1_spread_mean_reversion": {
        "hypothesis": (
            "market-neutral, relative-value mean reversion in the spread "
            "between two cointegrated/correlated USD-cross instruments "
            "(e.g. AUD/USD vs NZD/USD, or a triangulated cross constructed "
            "from two USD legs); economically distinct from every prior "
            "batch because it is the first family in this repository built "
            "as a hedged, diversified-by-construction position rather than "
            "a single-instrument directional bet"
        ),
        "signal_concept": (
            "z-score of a rolling hedge-ratio-adjusted spread relative to "
            "its own rolling mean/std; entry when |z| exceeds a threshold, "
            "exit on reversion toward zero or a stop on further widening"
        ),
        "causality_constraint": (
            "the hedge ratio, spread mean, and spread standard deviation "
            "used to generate a signal at bar T must be estimated only from "
            "bars strictly before T (rolling, backward-looking, lagged by "
            "at least one completed bar); no contemporaneous or future "
            "price of either leg may enter formation of the hedge ratio or "
            "the spread statistic used to trigger that same bar's signal"
        ),
        "execution_semantics": "native_bid_ask_from_outset",
        "structurally_two_sided": True,
        "parameter_grid_status": (
            "not yet defined; ranges/procedure to be frozen explicitly in "
            "B3.1 before any DEVELOPMENT screening of this family occurs"
        ),
    },
    "B3F2_asian_range_fade": {
        "hypothesis": (
            "Asian-session range is typically low-volatility and "
            "range-bound; a London-open probe beyond that completed range "
            "is more often a liquidity-driven overshoot than the start of "
            "a sustained directional move -- the direct complementary "
            "hypothesis to the already-retired session-range-expansion/"
            "continuation families (which bet the breakout continues), not "
            "a cosmetic variant of them"
        ),
        "signal_concept": (
            "fade entries when price first closes back inside a causally "
            "completed Asian-session high/low range after probing beyond "
            "it; tight stop beyond the probe extreme, small fixed target "
            "back toward the range"
        ),
        "causality_constraint": (
            "the Asian-session range used for a given trading day's fade "
            "signals must be built only from bars within that day's own, "
            "already-completed Asian session window; no bar from the same "
            "session still in progress, and no later session's bars, may "
            "contribute to that day's range"
        ),
        "execution_semantics": "native_bid_ask_from_outset",
        "structurally_two_sided": True,
        "parameter_grid_status": (
            "not yet defined; ranges/procedure to be frozen explicitly in "
            "B3.1 before any DEVELOPMENT screening of this family occurs"
        ),
    },
    "B3F3_session_open_microstructure_mean_reversion": {
        "hypothesis": (
            "short-horizon (M1/M5-style) overreaction at session opens, "
            "consistent with market makers temporarily absorbing an order-"
            "flow imbalance before price reverts toward a recent anchor; "
            "very high trade frequency is the deliberate mechanism for "
            "smoothing rolling expectancy and reducing dependence on any "
            "single trade or regime, distinct from the H1-timeframe, "
            "statistical z-score mean_reversion_h1_v1 candidate by both "
            "anchor (session-relative, not a rolling statistical z-score) "
            "and holding horizon"
        ),
        "signal_concept": (
            "fade an extreme excursion in the first few minutes after a "
            "session open back toward a causally available session anchor "
            "such as the prior session's close or a same-session VWAP "
            "computed only from bars up to and including the excursion bar"
        ),
        "causality_constraint": (
            "the anchor (prior close / VWAP) used to size and trigger a "
            "given signal must be computable entirely from bars at or "
            "before that signal's own bar; VWAP accumulation may never "
            "include a bar later than the signal bar"
        ),
        "execution_semantics": "native_bid_ask_from_outset",
        "structurally_two_sided": True,
        "parameter_grid_status": (
            "not yet defined; ranges/procedure to be frozen explicitly in "
            "B3.1 before any DEVELOPMENT screening of this family occurs"
        ),
    },
}

#: Explicitly excluded from Batch 3, decided now, before any DEVELOPMENT
#: result exists for these hypotheses.
EXPLICITLY_OUT_OF_SCOPE: tuple[str, ...] = ("cross_sectional_carry_momentum_basket",)

#: Families/strategies already retired in prior batches, established from
#: documentation, configuration, and result-summary artifacts only -- no
#: DEVELOPMENT/VALIDATION/HOLDOUT price or trade data was read to build this
#: list. Sources: ``docs/research/strategy_lineage_inventory.md`` (audited
#: 2026-08-17, evidence-based, cross-checked against ``.artifacts/`` and git
#: history) for the G1-era rows; the alpha-lab Stage-2/broad-FX-VALIDATION/
#: pair-specific-VALIDATION result summaries for the Batch 1/Batch 2 rows.
RETIRED_FAMILIES: tuple[dict[str, str], ...] = (
    {
        "strategy_id": "carver_trend_carry_ftmo5_v1",
        "batch": "g1",
        "disposition": "DEPLOYMENT_FEASIBILITY_BLOCKED",
    },
    {
        "strategy_id": "eurusd_tsm_v1",
        "batch": "g1",
        "disposition": "VALIDATION_REJECTED",
    },
    {
        "strategy_id": "liquidity_shock_reversion_v1",
        "batch": "g1",
        "disposition": "development_failed",
    },
    {
        "strategy_id": "eurusd_liquidity_shock_reversion_v1",
        "batch": "g1",
        "disposition": "ALPHA_REJECTED",
    },
    {
        "strategy_id": "session_range_expansion_v1",
        "batch": "g1",
        "disposition": "development_failed",
    },
    {
        "strategy_id": "eurusd_session_range_expansion_v1",
        "batch": "g1",
        "disposition": "VALIDATION_REJECTED",
    },
    {
        "strategy_id": "trend_pullback_v1",
        "batch": "g1",
        "disposition": "RETAINED_ONLY_AS_RESEARCH_REFERENCE",
    },
    {
        "strategy_id": "leo_gbpusd_v1",
        "batch": "g1",
        "disposition": "ALPHA_REJECTED",
    },
    {
        "strategy_id": "ts_momentum_v1",
        "batch": "g1",
        "disposition": "ALPHA_REJECTED",
    },
    {
        "strategy_id": "usd_macro_surprise_momentum_v1",
        "batch": "g1",
        "disposition": "DEVELOPMENT_BURNED_REJECT_RETIRE",
    },
    {
        "strategy_id": "eurusd_policy_rate_carry_proxy_v1",
        "batch": "g1",
        "disposition": "VALIDATION_REJECTED",
    },
    {
        "strategy_id": "momentum_signals_tsm",
        "batch": "batch1_wave1",
        "disposition": "BROAD_FX_VALIDATION_REJECTED",
    },
    {
        "strategy_id": "donchian_breakout_signals",
        "batch": "batch1_wave1",
        "disposition": "BROAD_FX_VALIDATION_REJECTED",
    },
    {
        "strategy_id": "volatility_breakout_signals",
        "batch": "batch1_wave1",
        "disposition": "DID_NOT_SURVIVE_STAGE2",
    },
    {
        "strategy_id": "session_continuation_signals",
        "batch": "batch1_wave1",
        "disposition": "DID_NOT_SURVIVE_STAGE2",
    },
    {
        "strategy_id": "F1_failed_bollinger_rejection",
        "batch": "batch1_wave2",
        "disposition": "DID_NOT_SURVIVE_TO_VALIDATION",
    },
    {
        "strategy_id": "F2_fresh_fvg_mitigation",
        "batch": "batch1_wave2",
        "disposition": "PAIR_SPECIFIC_VALIDATION_REJECTED",
    },
    {
        "strategy_id": "F3_volatility_squeeze_breakout",
        "batch": "batch1_wave2",
        "disposition": "PAIR_SPECIFIC_VALIDATION_REJECTED",
    },
    {
        "strategy_id": "B2F1_sweep_bos_retest",
        "batch": "batch2",
        "disposition": "FTMO_ALPHA_DIAGNOSTIC_REJECTED",
    },
    {
        "strategy_id": "B2F2_previous_period_sweep_rejection",
        "batch": "batch2",
        "disposition": "ZERO_PASSING_DEVELOPMENT_CONFIGS",
    },
    {
        "strategy_id": "B2F3_sweep_choch_retracement",
        "batch": "batch2",
        "disposition": "ZERO_PASSING_DEVELOPMENT_CONFIGS",
    },
    {
        "strategy_id": "B2F4_compression_box_breakout",
        "batch": "batch2",
        "disposition": "ZERO_PASSING_DEVELOPMENT_CONFIGS",
    },
)

#: Not retired -- still mid-pipeline from Batch 1 -- and therefore also not
#: an eligible "new family" for Batch 3, but distinct from the RETIRED list
#: above because no rejection decision has been made for it.
MID_PIPELINE_EXCLUDED_FROM_BATCH3: tuple[dict[str, str], ...] = (
    {
        "strategy_id": "mean_reversion_h1_v1",
        "batch": "batch1_wave1",
        "status": (
            "execution-promotion design frozen; real DEVELOPMENT orders "
            "run across all seven pairs, but the post-execution net "
            "return/Sharpe/cost-stress metric has not yet been computed -- "
            "recommended to close out in B3.1 rather than duplicate as a "
            "new Batch 3 family"
        ),
    },
)


# ---------------------------------------------------------------------------
# DEVELOPMENT advancement gates (frozen; apply identically to B3F1/B3F2/B3F3)
# ---------------------------------------------------------------------------

DEVELOPMENT_GATES: dict[str, Any] = {
    "evaluation_precedence": (
        "gates are evaluated in the fixed order listed below per sleeve "
        "(one instrument x one family x one frozen parameter configuration); "
        "the opportunity_density gate is evaluated FIRST and short-circuits "
        "all subsequent gates on failure, so that rolling-stability, "
        "concentration, tail, and best-N-trade gates are never evaluated "
        "against an underpowered or zero-eligible-window sample"
    ),
    "opportunity_density": {
        "rule": "completed_trade_count >= 80",
        "scope": "per sleeve, over the full frozen DEVELOPMENT interval",
        "rationale": (
            "statistical power for the gates below, and a floor on how "
            "quickly the strategy can plausibly accumulate a challenge "
            "profit target"
        ),
    },
    "economic": {
        "expectancy_usd_per_trade_gt": 0,
        "profit_factor_gt": 1.15,
    },
    "rolling_stability": {
        "window_sizes": [30, 50],
        "windowing_convention": (
            "trailing, non-overlapping-history-required windows over "
            "trades ordered by exit timestamp, identical to "
            "ftmoquant.research.ftmo_pass_probability.alpha_diagnostic."
            "_rolling_expectancy: window w's first eligible value ends at "
            "the w-th trade (index w-1); windows requiring fewer than w "
            "completed trades do not exist and are never padded, "
            "zero-filled, or otherwise fabricated -- they are excluded "
            "from every statistic below"
        ),
        "median_eligible_window_expectancy_gt": 0,
        "fraction_of_eligible_windows_positive_gte": 0.70,
        "applies_independently_per_window_size": True,
        "undefined_case": (
            "if a sleeve has zero eligible windows for a given window size "
            "(completed_trade_count < window size), that sleeve has "
            "already failed the opportunity_density gate above (80 > 50 > "
            "30) and this gate is never reached for it"
        ),
    },
    "profit_concentration": {
        "denominator": (
            "sum of realized P&L over trades with strictly positive "
            "realized P&L only, per sleeve, over the full DEVELOPMENT "
            "interval ('total positive DEVELOPMENT profit')"
        ),
        "attribution_timestamp": "trade exit timestamp (UTC)",
        "period_boundaries": "calendar month / calendar quarter in UTC",
        "max_single_month_share": 0.15,
        "max_single_quarter_share": 0.30,
        "zero_or_negative_denominator": (
            "if total positive DEVELOPMENT profit <= 0 for a sleeve, this "
            "gate fails immediately -- mirrors the existing "
            "E_catastrophic_concentration_guard convention in "
            "config/validation/oanda_alpha_lab_stage2_survivors_v1_"
            "preregistration.json"
        ),
    },
    "exceptional_winner_dependency": {
        "rule": (
            "rank ALL completed trades in the sleeve by realized P&L "
            "descending; remove the top N = ceil(0.05 * "
            "completed_trade_count) trades (ceiling rounding, so at least "
            "one trade is always removed and the removed set is never "
            "under-sized); ties at the boundary between the Nth and "
            "(N+1)th-ranked trade by P&L are broken by removing the "
            "earlier-exit-timestamp trade first, so the removed set is "
            "always exactly N trades and fully deterministic"
        ),
        "remaining_expectancy_usd_per_trade_gt": 0,
        "remaining_profit_factor_gt": 1.05,
    },
    "temporal_stability": {
        "fold_count": 4,
        "fold_boundaries_utc": [_iso(bound) for bound in DEVELOPMENT_FOLD_BOUNDARIES],
        "fold_derivation": (
            "four equal 373-day chronological folds spanning "
            "[DEVELOPMENT_START, DEVELOPMENT_END_EXCLUSIVE) by pure "
            "calendar-day division (1492 / 4), derived without inspecting "
            "any return, price, or trade data"
        ),
        "rule": "positive net return in at least 3 of the 4 folds",
    },
    "tail_and_concentration": {
        "largest_single_winning_trade_share_of_total_positive_profit_lte": 0.20,
        "reported_not_gated": ["pnl_skewness", "pnl_kurtosis"],
    },
    "directional_robustness": {
        "declared_applicability": (
            "all three in-scope families (B3F1, B3F2, B3F3) are declared, "
            "before any DEVELOPMENT result is observed, to be structurally "
            "two-sided (each admits both long and short trades by "
            "construction: B3F1 trades the spread in both directions, "
            "B3F2 fades excursions on both sides of the Asian range, B3F3 "
            "fades excursions in both directions at session open); this "
            "gate therefore applies to all three without exception or "
            "post-hoc waiver"
        ),
        "rule": "each direction independently requires profit_factor >= 1.0",
    },
    "transaction_cost_sensitivity": {
        "primary_execution": (
            "native M1 BID/ASK trade-lifecycle execution from the outset "
            "for all three families (the existing "
            "wick_fvg_squeeze_execution.simulate_trades-style engine: buy "
            "crosses ASK, sell crosses BID, long-liquidation crosses BID, "
            "short-liquidation crosses ASK, stop-first same-bar collision, "
            "entry at the first strictly-later paired M1 observation, no "
            "interpolation) -- NOT the coarser scalar-fee vectorbt "
            "approximation used for Batch 1/2's Stage-1 screen"
        ),
        "why_a_new_stress_mechanism_is_required": (
            "the existing pair-specific VALIDATION gate already established "
            "(see STRESS_GATE_REASON in "
            "ftmoquant.research.alpha_lab.pair_specific_validation) that "
            "native-spread execution has no separate cost parameter to "
            "scale by a multiplier post hoc -- spread cost is embedded "
            "intrinsically in which side of the book each entry/exit "
            "crosses. A cost-stress gate for Batch 3 therefore requires an "
            "explicit synthetic spread-widening transform applied to the "
            "raw BID/ASK price series BEFORE execution, not a scalar "
            "applied to an already-computed return"
        ),
        "required_stress_mechanism": (
            "widen every bar's spread by a fixed multiplier around its "
            "own mid price (ask' = mid + (ask - mid) * multiplier, "
            "bid' = mid - (mid - bid) * multiplier), then re-run the "
            "identical, unmodified signal and execution logic against the "
            "widened series; compare resulting expectancy/profit_factor "
            "to the un-stressed run"
        ),
        "implementation_status": (
            "does not exist in the repository yet; must be built as "
            "generic, shared, tested infrastructure in B3.1 BEFORE any "
            "Batch 3 DEVELOPMENT screening -- it is explicitly forbidden "
            "to invent an ad hoc, per-family, unverified stress mechanism "
            "at screening time"
        ),
        "rule_1_5x": (
            "expectancy_usd_per_trade > 0 at a 1.5x spread multiplier "
            "(all three families)"
        ),
        "rule_2_0x": (
            "expectancy_usd_per_trade > 0 at a 2.0x spread multiplier "
            "(B3F2 and B3F3 only, given their higher trade frequency and "
            "smaller per-trade targets make cost more material relative "
            "to target size)"
        ),
    },
    "parameter_neighborhood_robustness": {
        "mechanism": (
            "reuse, unchanged, the existing N-D axis-adjacency + connected-"
            "component check (ftmoquant.research.alpha_lab.screening_stage2."
            "_connected_components / "
            "ftmoquant.research.alpha_lab.liquidity_structure_screen."
            "_axis_adjacent_neighbors)"
        ),
        "default_min_connected_region_size": 2,
        "amendment_rule": (
            "if a family's B3.1/B3.2 parameter grid is genuinely higher-"
            "dimensional such that a larger minimum region is warranted "
            "(as B2F1's 3-D grid required a minimum of 3), that family's "
            "minimum region size must be fixed at grid-definition time in "
            "B3.1 and recorded as an explicit amendment to this "
            "preregistration BEFORE that family's DEVELOPMENT results are "
            "observed -- never chosen after seeing results. The default of "
            "2 applies to every family unless so amended"
        ),
    },
}


# ---------------------------------------------------------------------------
# FTMO-specific DEVELOPMENT gates
# ---------------------------------------------------------------------------

FTMO_DEVELOPMENT_GATES: dict[str, Any] = {
    "full_development_pass_both": {
        "mechanism": (
            "reuse ftmoquant.research.ftmo_pass_probability.monte_carlo/"
            "bootstrap/reporting unchanged: 100,000-replication stationary "
            "block bootstrap, block length derived via "
            "derive_frozen_block_length on DEVELOPMENT trades only, at the "
            "sleeve's frozen sizing policy (see sizing_selection_procedure)"
        ),
        "rule": "pass_both >= 0.70",
    },
    "cross_subperiod_stability": {
        "split": {
            "half_1_utc": [
                _iso(DEVELOPMENT_START),
                _iso(DEVELOPMENT_SUBPERIOD_MIDPOINT),
            ],
            "half_2_utc": [
                _iso(DEVELOPMENT_SUBPERIOD_MIDPOINT),
                _iso(DEVELOPMENT_END_EXCLUSIVE),
            ],
            "derivation": (
                "deterministic floor(span/2)-day midpoint of the frozen "
                "DEVELOPMENT interval, computed from stage_g's frozen "
                "constants alone -- no return, price, or trade data "
                "inspected to place this boundary; coincides exactly with "
                "the temporal_stability fold-2/fold-3 boundary above"
            ),
        },
        "mechanism": (
            "run the identical frozen Monte Carlo methodology (same sizing "
            "policy, same bootstrap method, same 100,000-replication count) "
            "independently on each half's trades"
        ),
        "rule": (
            "pass_both(weaker_half) / pass_both(stronger_half) >= 0.60, "
            "where 'weaker'/'stronger' is determined by which half has the "
            "lower/higher pass_both value"
        ),
        "zero_denominator_handling": (
            "if the stronger half's pass_both is exactly 0, the ratio is "
            "undefined and the gate is treated as FAILED (a stronger half "
            "at 0% pass_both already implies economic failure regardless "
            "of the ratio)"
        ),
        "zero_trade_half_handling": (
            "if either half has zero completed trades for a sleeve, the "
            "gate is treated as FAILED for that sleeve -- a family that "
            "clusters all its DEVELOPMENT trades into a single half is "
            "exactly the concentration pattern this gate exists to catch"
        ),
    },
    "drawdown": {
        "mechanism": (
            "p95_max_drawdown as already emitted by "
            "ftmoquant.research.ftmo_pass_probability.reporting, at the "
            "sleeve's frozen sizing policy"
        ),
        "applicable_target": (
            "the Verification-phase profit target in dollars at the "
            "$100,000 research notional (5% = $5,000), used as the "
            "binding target because Verification is the stricter of the "
            "two chained phases a candidate must clear"
        ),
        "rule": "p95_max_drawdown_usd < 0.60 * 5000 == $3,000",
    },
}


# ---------------------------------------------------------------------------
# Sizing-selection procedure (DEVELOPMENT-only; deterministic two-stage,
# extending the existing SIZING_GRID / NOTIONAL_REFINEMENT_GRID precedent)
# ---------------------------------------------------------------------------

SIZING_SELECTION_PROCEDURE: dict[str, Any] = {
    "prohibition": "VALIDATION data must never be used to select or tune sizing",
    "stage_1": {
        "mechanism": (
            "reuse the existing, unmodified 8-policy "
            "ftmoquant.research.ftmo_pass_probability.sizing.SIZING_GRID "
            "at the existing 20,000-path exploratory replication count, on "
            "DEVELOPMENT trades only, independently per surviving sleeve"
        ),
        "selection_rule": "the single policy with the highest pass_both",
    },
    "stage_2": {
        "mechanism": (
            "a new, disjoint, 9-policy local-refinement grid, mirroring "
            "the existing NOTIONAL_REFINEMENT_GRID precedent, centered on "
            "the Stage 1-selected policy's own family and value"
        ),
        "construction_rule_fixed_notional_multiplier": (
            "if Stage 1 selects a fixed_notional_multiplier policy with "
            "value m, the refinement grid is exactly the 9 offsets "
            "{-0.50, -0.35, -0.25, -0.15, -0.05, +0.05, +0.15, +0.25, "
            "+0.50} applied to m, each floored at a minimum multiplier of "
            "0.1x"
        ),
        "construction_rule_fixed_fractional_risk": (
            "if Stage 1 selects a fixed_fractional_risk policy with value "
            "r, the refinement grid is exactly the 9 multipliers "
            "{0.50, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.50} "
            "applied to r, each floored at 0.001 (0.1%) and capped at "
            "0.05 (5%, since risking more than the daily loss limit on a "
            "single trade is economically incoherent)"
        ),
        "mechanism_run": (
            "the full 100,000-replication precision Monte Carlo across the "
            "9-policy refinement grid"
        ),
        "selection_rule": (
            "the single policy with the highest pass_both becomes the "
            "sleeve's frozen sizing policy for every remaining Batch 3 "
            "gate and for VALIDATION"
        ),
    },
    "expansion_prohibition": (
        "no further sizing grid expansion is permitted after Stage 2 "
        "results are observed, matching the existing repo-wide "
        "SIZING_GRID/NOTIONAL_REFINEMENT_GRID precedent and norm"
    ),
}


# ---------------------------------------------------------------------------
# VALIDATION policy (one-shot; precommitted now)
# ---------------------------------------------------------------------------

VALIDATION_POLICY: dict[str, Any] = {
    "access_model": "one_shot_per_frozen_candidate",
    "preconditions_before_validation_access": [
        "frozen signal implementation",
        "frozen pair/instrument universe",
        "frozen parameters",
        "frozen execution semantics",
        "frozen sizing (Stage 2 refinement policy from sizing_selection_procedure)",
        "frozen preregistration document with a verified self-hash",
    ],
    "precommitted_gates": {
        "A_native_spread_positive_return": (
            "native-spread VALIDATION net_return > 0"
        ),
        "B_native_spread_positive_sharpe": (
            "native-spread VALIDATION annualized_sharpe > 0"
        ),
        "C_ftmo_pass_both": (
            "FTMO pass_both >= 0.70 on VALIDATION trades, computed using "
            "the exact frozen sizing policy and bootstrap method selected "
            "on DEVELOPMENT -- no re-selection, mirroring the existing "
            "FROZEN_POLICY_ID/FROZEN_METHOD hardcoding precedent in "
            "ftmoquant.research.ftmo_pass_probability.validation_diagnostic"
        ),
        "validation_passed": "A AND B AND C",
    },
    "additional_diagnostics": (
        "the existing alpha_diagnostic.py-style diagnostics (rolling "
        "expectancy, monthly/quarterly stability, subgroup audit, "
        "chronological replay) may be reported for a candidate that "
        "passes A/B/C, but must NOT become an additional post-hoc pass/"
        "fail rule -- only A, B, and C gate VALIDATION"
    ),
    "on_failure": (
        "retire the candidate; no VALIDATION-informed retuning, parameter "
        "substitution, pair substitution, or rerun"
    ),
}


# ---------------------------------------------------------------------------
# Firewall
# ---------------------------------------------------------------------------

FIREWALL: dict[str, Any] = {
    "partition_boundaries": {
        "development_start_utc": _iso(DEVELOPMENT_START),
        "development_end_exclusive_utc": _iso(DEVELOPMENT_END_EXCLUSIVE),
        "validation_start_utc": _iso(VALIDATION_START),
        "holdout_start_utc": _iso(HOLDOUT_START),
        "source": "ftmoquant.research.stage_g (reused unchanged, not redefined)",
    },
    "rules": [
        (
            "Batch 3 family choice, parameters, grids, gates, and "
            "thresholds may not be altered using any Batch 3 VALIDATION "
            "result."
        ),
        (
            "final holdout remains completely sealed; no Batch 3 stage "
            "accesses it."
        ),
        (
            "prior VALIDATION observations from earlier strategy families "
            "(including B2F1's observed long/short asymmetry) must not be "
            "used to construct pair- or direction-specific Batch 3 "
            "exceptions; the directional_robustness gate above is "
            "declared symmetrically and a priori for exactly this reason."
        ),
    ],
}


# ---------------------------------------------------------------------------
# Document assembly / hashing / verification
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Batch3Preregistration:
    document: dict[str, Any]
    semantic_sha256: str


def build_preregistration(*, created_at_utc: datetime | None = None) -> dict[str, Any]:
    """Build the full B3.0 preregistration document.

    Reads only this module's own frozen Python literals plus the byte
    contents of two already-frozen configuration files (the OANDA alpha-lab
    DEVELOPMENT lineage config and the FTMO rule-set YAML) for lineage
    binding -- never DEVELOPMENT, VALIDATION, or HOLDOUT price/trade data.
    """

    timestamp = created_at_utc if created_at_utc is not None else datetime.now(UTC)
    document: dict[str, Any] = {
        "preregistration_version": PREREGISTRATION_VERSION,
        "stage": STAGE_ID,
        "created_at_utc": _iso(timestamp),
        "purpose": (
            "Freeze the approved Batch 3 research protocol before any "
            "Batch 3 DEVELOPMENT analysis occurs."
        ),
        "development_lineage": {
            "lineage_id": OANDA_ALPHA_LAB_LINEAGE_ID,
            "alpha_lab_config_sha256": _sha256_file(
                OANDA_ALPHA_LAB_DEVELOPMENT_CONFIG_PATH
            ),
            "development_start_utc": _iso(DEVELOPMENT_START),
            "development_end_exclusive_utc": _iso(DEVELOPMENT_END_EXCLUSIVE),
        },
        "validation_partition": {
            "start_utc": _iso(VALIDATION_START),
            "end_exclusive_utc": _iso(HOLDOUT_START),
        },
        "prop_rule_set": {
            "rule_set_id": PROP_RULE_SET_ID,
            "config_path": str(PROP_RULE_SET_PATH),
            "config_sha256": _sha256_file(PROP_RULE_SET_PATH),
        },
        "family_scope": {
            "in_scope": copy.deepcopy(IN_SCOPE_FAMILIES),
            "explicitly_out_of_scope": list(EXPLICITLY_OUT_OF_SCOPE),
            "retired_from_prior_batches": copy.deepcopy(list(RETIRED_FAMILIES)),
            "mid_pipeline_excluded_from_batch3": copy.deepcopy(
                list(MID_PIPELINE_EXCLUDED_FROM_BATCH3)
            ),
        },
        "development_gates": copy.deepcopy(DEVELOPMENT_GATES),
        "ftmo_development_gates": copy.deepcopy(FTMO_DEVELOPMENT_GATES),
        "sizing_selection_procedure": copy.deepcopy(SIZING_SELECTION_PROCEDURE),
        "validation_policy": copy.deepcopy(VALIDATION_POLICY),
        "firewall": copy.deepcopy(FIREWALL),
        "lifecycle": {
            "development_accessed": False,
            "validation_accessed": False,
            "holdout_accessed": False,
        },
    }
    document["preregistration_semantic_sha256"] = _canonical_sha256(document)
    return document


def write_preregistration(
    *, path: Path = PREREGISTRATION_PATH, created_at_utc: datetime | None = None
) -> Path:
    """Write the preregistration document. Refuses to overwrite an existing
    file, matching the existing repo-wide "refuses to overwrite" convention
    for frozen artifacts."""

    if path.exists():
        raise Batch3PreregistrationError(
            f"{path} already exists; refusing to overwrite"
        )
    document = build_preregistration(created_at_utc=created_at_utc)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def verify_preregistration(path: Path = PREREGISTRATION_PATH) -> dict[str, Any]:
    """Load a preregistration document and verify its self-hash, version,
    partition boundaries, and lifecycle flags. Raises
    :class:`Batch3PreregistrationError` on any mismatch."""

    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    stored_hash = document.get("preregistration_semantic_sha256")
    recomputed_hash = _canonical_sha256(document)
    if stored_hash != recomputed_hash:
        raise Batch3PreregistrationError(
            "preregistration_semantic_sha256 does not match recomputed hash "
            f"(stored={stored_hash!r}, recomputed={recomputed_hash!r})"
        )
    if document.get("preregistration_version") != PREREGISTRATION_VERSION:
        raise Batch3PreregistrationError("unexpected preregistration_version")
    if document.get("stage") != STAGE_ID:
        raise Batch3PreregistrationError("unexpected stage")

    partition = document.get("validation_partition", {})
    if partition.get("start_utc") != _iso(VALIDATION_START) or partition.get(
        "end_exclusive_utc"
    ) != _iso(HOLDOUT_START):
        raise Batch3PreregistrationError("validation_partition does not match stage_g")

    lineage = document.get("development_lineage", {})
    if lineage.get("development_start_utc") != _iso(DEVELOPMENT_START) or lineage.get(
        "development_end_exclusive_utc"
    ) != _iso(DEVELOPMENT_END_EXCLUSIVE):
        raise Batch3PreregistrationError(
            "development_lineage dates do not match stage_g"
        )

    lifecycle = document.get("lifecycle", {})
    if (
        lifecycle.get("development_accessed") is not False
        or lifecycle.get("validation_accessed") is not False
        or lifecycle.get("holdout_accessed") is not False
    ):
        raise Batch3PreregistrationError(
            "a B3.0 preregistration must declare all lifecycle flags False"
        )

    in_scope = document.get("family_scope", {}).get("in_scope", {})
    if set(in_scope) != set(IN_SCOPE_FAMILIES):
        raise Batch3PreregistrationError(
            "family_scope.in_scope does not match the frozen B3F1/B3F2/B3F3 scope"
        )
    retired_ids = {
        row["strategy_id"]
        for row in document.get("family_scope", {}).get(
            "retired_from_prior_batches", []
        )
    }
    if retired_ids & set(in_scope):
        raise Batch3PreregistrationError(
            "a retired family id also appears in the in-scope family set"
        )

    return document
