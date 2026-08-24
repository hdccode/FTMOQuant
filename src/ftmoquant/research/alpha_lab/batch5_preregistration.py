"""Write-once Batch 5 preregistration for three independent FX alpha families.

This module contains methodology metadata and integrity checks only.  It has no
market-data, result, partition, signal, execution, sizing, or optimizer imports.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PREREGISTRATION_VERSION = "batch5-three-independent-fx-alpha-families-v1"
STAGE_ID = "B5.0"
PREREGISTRATION_PATH = Path(
    "config/research/batch5_three_fx_alpha_families_preregistration_v1.json"
)
CREATED_AT_UTC = datetime(2026, 8, 24, tzinfo=UTC)

FAMILY_B5A = "B5A_cftc_dealer_demand_shock_fx"
FAMILY_B5B = "B5B_direct_audcad_mean_reversion"
FAMILY_B5C = "B5C_daily_fx_overreaction_reversal"
PRIMARY_FAMILIES = (FAMILY_B5A, FAMILY_B5B, FAMILY_B5C)

B5A_SLEEVES = {
    "JPY": "USD/JPY.OANDA",
    "CHF": "USD/CHF.OANDA",
    "EUR": "EUR/USD.OANDA",
    "GBP": "GBP/USD.OANDA",
    "CAD": "USD/CAD.OANDA",
    "AUD": "AUD/USD.OANDA",
    "NZD": "NZD/USD.OANDA",
}
B5C_INSTRUMENTS = (
    "EUR/USD.OANDA",
    "USD/JPY.OANDA",
    "USD/CAD.OANDA",
    "AUD/USD.OANDA",
    "EUR/JPY.OANDA",
)

# Filled with the semantic identity of the checked-in artifact.  Downstream
# Batch 5 code must call verify_preregistration(), which checks both a freshly
# computed hash and this independently pinned identity.  Re-hashing a mutated
# document therefore remains invalid.
EXPECTED_PREREGISTRATION_SEMANTIC_SHA256 = (
    "603c3a80427129b5c3758460c8cce5bc21947546520a3ef38a6e1c42e53fcbdf"
)


class Batch5PreregistrationError(RuntimeError):
    """Raised when the frozen Batch 5 document contract is violated."""


def _canonical_sha256(document: dict[str, Any]) -> str:
    semantic = {
        key: value
        for key, value in document.items()
        if key != "preregistration_semantic_sha256"
    }
    payload = json.dumps(
        semantic, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


SOURCE_PROVENANCE: list[dict[str, Any]] = [
    {
        "family": FAMILY_B5A,
        "title": "Breaking Parity: Equilibrium Exchange Rates and Currency Premia",
        "authors": ["Mai Chi Dao", "Pierre-Olivier Gourinchas", "Oleg Itskhoki"],
        "version_date": "2025-07-01",
        "type": "IMF_working_paper_by_established_researchers",
        "url": (
            "https://economics.princeton.edu/wp-content/uploads/2025/07/"
            "BreakingParityFX.pdf"
        ),
        "primary_claim": (
            "A rise in a dealer's scaled net-long local-currency futures position "
            "proxies a demand shift toward USD, coincides with local-currency "
            "depreciation, and is followed by smaller persistent predictable "
            "local-currency appreciation and positive returns to the post-shock "
            "position."
        ),
        "specification_support": (
            "CFTC TFF Dealer/Intermediary; monthly averages of weekly reports; "
            "dealer long minus short divided by the trailing 12-month moving "
            "average of monthly total open interest; monthly change; G7+; "
            "three-month return horizon"
        ),
        "caveat": (
            "The paper estimates equilibrium projections, not a transaction-costed "
            "trading rule; the sign-only portfolio rule is the preregistered causal "
            "operationalization of its directional prediction."
        ),
    },
    {
        "family": FAMILY_B5A,
        "title": "Commitments of Traders: explanatory notes and release schedule",
        "authors": ["U.S. Commodity Futures Trading Commission"],
        "type": "primary_regulator_documentation",
        "url": "https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm",
        "supports": (
            "Tuesday as-of positions are generally released Friday at 15:30 "
            "America/New_York; holidays can alter the schedule; TFF trader "
            "classification and revisions must use the actually published file"
        ),
    },
    {
        "family": FAMILY_B5B,
        "title": "Algorithmic Trading: Winning Strategies and Their Rationale",
        "authors": ["Ernest P. Chan"],
        "year": 2013,
        "type": "publicly_documented_practitioner_book_example",
        "url": "https://www.wiley.com/en-us/Algorithmic+Trading%3A+Winning+Strategies+and+Their+Rationale-p-9781118460146",
        "supports": (
            "Example 5.2 trades the ready-made AUD.CAD cross directly, uses daily "
            "closes and a 20-day moving average, and holds the negative sign of "
            "the close-minus-moving-average deviation with a one-day lag"
        ),
        "limitations": (
            "Practitioner evidence, not peer reviewed; published example does not "
            "include native historical bid/ask execution and is not treated as proof"
        ),
    },
    {
        "family": FAMILY_B5C,
        "title": "Daily abnormal price changes and trading strategies in the FOREX",
        "authors": ["Guglielmo Maria Caporale", "Alex Plastun"],
        "year": 2021,
        "type": "peer_reviewed_open_access",
        "journal": "Journal of Economic Studies 48(1), 211-222",
        "doi": "10.1108/JES-11-2019-0503",
        "url": "https://doi.org/10.1108/JES-11-2019-0503",
        "supports": (
            "For EURUSD, USDJPY, USDCAD, AUDUSD, and EURJPY, a day whose "
            "open-to-close return exceeds mean plus/minus two standard deviations "
            "is followed by contrarian price movement on the next day"
        ),
        "limitations": (
            "The paper omits transaction costs and does not uniquely state the "
            "estimation-window length n; its pair-specific intraday exit clocks were "
            "sample-derived and are not admitted into this preregistration"
        ),
    },
    {
        "family": FAMILY_B5C,
        "title": (
            "Intraday price-reversal patterns in the currency futures market: "
            "The impact of the introduction of GLOBEX and the euro"
        ),
        "authors": ["Joel Rentzler", "Kishore Tandon", "Susana Yu"],
        "year": 2006,
        "type": "peer_reviewed_corroboration",
        "journal": "Journal of Futures Markets 26(11), 1089-1130",
        "doi": "10.1002/fut.20226",
        "url": "https://doi.org/10.1002/fut.20226",
        "supports": (
            "Independent evidence of short-horizon reversals after large one-day "
            "moves in major CME currency futures"
        ),
    },
]


COMMON_GATES: dict[str, Any] = {
    "evaluation_order": (
        "sleeve gates, then immutable all-sleeve aggregate, then breadth"
    ),
    "native_expectancy": {"net_expectancy_account_currency_gt": 0},
    "native_net_return": {"net_return_gt": 0},
    "native_profit_factor": {"profit_factor_gt": 1.10},
    "chronological_stability": {
        "fold_count": 4,
        "fold_construction": (
            "four contiguous equal-duration DEVELOPMENT intervals fixed from the "
            "partition boundaries before prices; boundary ties go to the later fold"
        ),
        "positive_net_return_folds_gte": 3,
    },
    "largest_winners": {
        "remove": "top ceil(5%) of strictly profitable independent evaluation units",
        "tie_break": "earlier UTC exit removed first",
        "remaining_net_expectancy_gt": 0,
    },
    "period_concentration": {
        "attribution": "UTC calendar year of exit",
        "max_single_year_share_of_strictly_positive_profit_lte": 0.40,
        "nonpositive_denominator": "fail_closed",
        "profitable_calendar_year_fraction_gte": 0.60,
    },
    "costs": {
        "native": "genuine paired OANDA BID/ASK at every fill",
        "stress_multipliers": [1.5, 2.0],
        "net_expectancy_gt_at_each_stress": 0,
        "reuse": "ftmoquant.research.alpha_lab.cost_stress.widen_bid_ask_frame",
        "post_hoc_scalar_cost_subtraction_forbidden": True,
    },
    "drawdown": {
        "equal_weight_family_aggregate_maximum_drawdown_lte": 0.15,
        "equity_basis": (
            "USD 100000 initial equity per sleeve; no FTMO leverage or sizing"
        ),
    },
    "all_applicable_gates_required": True,
    "justification": (
        "Positive edge, PF 1.10, 3-of-4 time stability, 2x spread survival, "
        "winner/year robustness, and a 15% research drawdown cap are frozen as a "
        "conservative alpha screen; family observation floors differ by natural "
        "frequency."
    ),
}


def _pair_side_for_local_currency(currency: str, move: str) -> str:
    pair = B5A_SLEEVES[currency]
    base, quote_venue = pair.split("/", maxsplit=1)
    quote = quote_venue.split(".", maxsplit=1)[0]
    local_up = move == "APPRECIATES"
    buy_base = (currency == base and local_up) or (currency == quote and not local_up)
    return "BUY" if buy_base else "SELL"


def _b5a() -> dict[str, Any]:
    sleeves = [
        {
            "sleeve_id": f"B5A_{currency}",
            "currency_k": currency,
            "spot_instrument": pair,
            "positive_delta_position_side": _pair_side_for_local_currency(
                currency, "APPRECIATES"
            ),
            "negative_delta_position_side": _pair_side_for_local_currency(
                currency, "DEPRECIATES"
            ),
        }
        for currency, pair in B5A_SLEEVES.items()
    ]
    return {
        "priority": "highest",
        "hypothesis": (
            "After an increase (decrease) in scaled Dealer/Intermediary net-long "
            "futures positioning in currency k, currency k appreciates (depreciates) "
            "against USD over the subsequent three months, net of native costs."
        ),
        "source_faithful_replication": {
            "report": "CFTC Traders in Financial Futures, Futures Only",
            "market_scope": "CME currency futures versus USD",
            "trader_category": "Dealer/Intermediary long and short",
            "legacy_commercial_substitution_permitted": False,
            "position_as_of": (
                "close of report-date Tuesday (or CFTC-declared as-of date)"
            ),
            "weekly_net_position": (
                "Dealer/Intermediary long contracts minus short contracts"
            ),
            "monthly_position": (
                "arithmetic mean of all weekly net positions with CFTC as-of dates "
                "inside calendar month m"
            ),
            "monthly_open_interest": (
                "arithmetic mean of total open interest over the same weekly reports"
            ),
            "scaled_position_formula": (
                "f_star[k,m] = 100 * monthly_dealer_net[k,m] / mean("
                "monthly_open_interest[k,m-j] for j=0..11)"
            ),
            "signal": "delta_f_star[k,m] = f_star[k,m] - f_star[k,m-1]",
            "threshold": "strict sign only; zero means flat/no new cohort",
            "direction": (
                "positive delta_f_star: long currency k versus USD; negative: "
                "short currency k versus USD"
            ),
            "formation_frequency": "calendar monthly",
            "holding_period": "three calendar months from actual entry fill",
            "rebalance_frequency": "monthly independent cohorts",
            "overlap": "up to three equal-notional monthly cohorts per sleeve",
            "weighting": "equal USD gross notional per active cohort and sleeve",
            "permitted_robustness_variants": [],
        },
        "causal_publication_contract": {
            "nominal_release": "Friday 15:30 America/New_York for prior Tuesday",
            "signal_available_at": (
                "maximum actual CFTC publication timestamp among every report whose "
                "as-of date is in month m, and never before calendar month m ends"
            ),
            "entry": "first strictly-later paired OANDA M1 observation",
            "report_date_is_never_availability_date": True,
            "holidays": (
                "use actual CFTC release schedule/timestamp; never assume Friday"
            ),
            "missing_release": "do not form month until released; no interpolation",
            "revisions": (
                "use first archived vintage actually available at signal time; later "
                "revisions apply prospectively only and never rewrite old signals"
            ),
            "duplicate_or_corrected_files": (
                "retain immutable retrieval timestamp and content hash; earliest valid "
                "published vintage governs historical formation"
            ),
        },
        "sleeves": sleeves,
        "screening": {
            "independent_unit": "non-overlapping three-month cohort per sleeve",
            "minimum_monthly_formation_dates_per_sleeve": 36,
            "minimum_nonoverlapping_three_month_units_per_sleeve": 12,
            "minimum_distinct_calendar_years": 3,
        },
    }


def _b5b() -> dict[str, Any]:
    return {
        "hypothesis": (
            "The directly tradable AUD/CAD cross mean reverts around its trailing "
            "20-FX-day moving mean, so holding the opposite sign of the completed "
            "daily close deviation has positive net expectancy after native costs."
        ),
        "source_faithful_replication": {
            "instrument": "AUD/CAD.OANDA",
            "direct_instrument_required": True,
            "synthetic_two_leg_substitution_permitted": False,
            "bar_frequency": "one completed New York FX day",
            "day": "17:00 America/New_York close to next 17:00 America/New_York close",
            "formation_statistic": (
                "deviation = completed daily midpoint close minus arithmetic mean of "
                "the 20 completed daily midpoint closes including that close"
            ),
            "lookback_completed_fx_days": 20,
            "entry_threshold": "strictly nonzero deviation; no z-score threshold",
            "direction": "deviation below zero BUY AUD/CAD; above zero SELL AUD/CAD",
            "exit_rule": (
                "exit or reverse at first execution opportunity after a completed "
                "daily close whose deviation sign is zero or changes sign"
            ),
            "maximum_holding_period": None,
            "overlapping_position_policy": "one position only; never scale in",
            "signal_lag": "one completed daily observation",
            "rollover": (
                "all broker financing/rollover debits and credits must be included "
                "when supported; if unavailable, family cannot pass"
            ),
            "parameter_variants": [],
        },
        "implementation_conventions_not_literature_claims": {
            "indicator_price": "completed BID/ASK midpoint close",
            "decision_timestamp": "17:00 America/New_York daily boundary",
            "entry_and_exit": "first strictly-later paired OANDA M1 observation",
            "fills": "BUY at ASK, SELL at BID, opposite executable side on exit",
            "fixed_reference_notional_usd": 100000,
        },
        "repository_availability_audit": {
            "present_in_OANDA_ALPHA_LAB_SPECS": False,
            "present_in_oanda_fx_alpha_lab_v1_config": False,
            "required_before_future_development": (
                "add AUDCAD/AUD_CAD/AUD/CAD.OANDA as a native paired-M1 OANDA "
                "canonical instrument with provider-verified precision, immutable "
                "manifests, derived daily boundaries, and DEVELOPMENT-only readiness"
            ),
            "silent_synthesis_forbidden": True,
        },
        "screening": {
            "independent_unit": "one non-overlapping daily marked holding return",
            "minimum_daily_holding_observations": 500,
            "minimum_position_sign_changes": 20,
            "minimum_distinct_calendar_years": 3,
        },
    }


def _b5c() -> dict[str, Any]:
    return {
        "hypothesis": (
            "A two-standard-deviation open-to-close FX move overreacts; the same "
            "direct pair has positive net expectancy in the opposite direction over "
            "the next complete New York FX day."
        ),
        "literature_anchored_rule": {
            "universe": list(B5C_INSTRUMENTS),
            "day": "17:00 America/New_York close to next 17:00 America/New_York close",
            "daily_return": "completed-day midpoint close / midpoint open - 1",
            "estimation_window": (
                "30 immediately preceding completed FX-day returns, excluding the "
                "candidate event day"
            ),
            "positive_event": (
                "return_t > trailing_mean_30 + 2 * trailing_sample_std_30"
            ),
            "negative_event": (
                "return_t < trailing_mean_30 - 2 * trailing_sample_std_30"
            ),
            "entry": (
                "first strictly-later paired OANDA M1 observation after day t closes"
            ),
            "direction": "positive event SELL pair; negative event BUY pair",
            "holding_period": "one complete subsequent New York FX day",
            "exit": (
                "first strictly-later paired OANDA M1 observation after the next "
                "valid FX-day 17:00 America/New_York boundary"
            ),
            "overlap": "at most one position per instrument; overlapping event ignored",
            "weekends_and_holidays": (
                "a valid FX day requires both boundary observations and genuine paired "
                "data; otherwise advance to the next valid boundary without filling"
            ),
            "parameter_variants": [],
        },
        "implementation_conventions_not_literature_claims": {
            "new_york_day_definition": (
                "The source uses MetaQuotes daily/hourly bars without establishing "
                "a portable venue timezone; 17:00 America/New_York is frozen for "
                "causal reproducibility and is not a claimed source parameter."
            ),
            "thirty_day_window": (
                "The primary paper fixes k=2 but does not uniquely report n; 30 is the "
                "smallest conventional monthly-length causal window and is frozen "
                "without consulting repository returns."
            ),
            "full_next_day_exit": (
                "Replaces the paper's sample-derived pair-specific intraday clocks to "
                "avoid clock/session overfitting and keep B5C distinct from Batch 4."
            ),
            "indicator_price": "completed BID/ASK midpoint OHLC",
            "fills": "BUY at ASK, SELL at BID, opposite executable side on exit",
            "fixed_reference_notional_usd": 100000,
        },
        "repository_availability_audit": {
            "native_now": list(B5C_INSTRUMENTS[:4]),
            "missing_native": ["EUR/JPY.OANDA"],
            "required_before_future_development": (
                "add native paired-M1 EURJPY/EUR_JPY/EUR/JPY.OANDA through the same "
                "canonical OANDA lineage; no synthetic cross substitution"
            ),
        },
        "forbidden_filters": [
            "weekday",
            "session",
            "volatility_regime",
            "trend",
            "news",
            "spread_quantile",
        ],
        "screening": {
            "independent_unit": "completed one-day event trade",
            "minimum_events_per_sleeve": 15,
            "minimum_events_family_total": 60,
            "minimum_distinct_calendar_years_with_events": 3,
        },
    }


def build_preregistration() -> dict[str, Any]:
    document: dict[str, Any] = {
        "preregistration_version": PREREGISTRATION_VERSION,
        "stage": STAGE_ID,
        "created_at_utc": CREATED_AT_UTC.isoformat().replace("+00:00", "Z"),
        "purpose": (
            "freeze Batch 5 methodology before any DEVELOPMENT performance access"
        ),
        "family_scope": {
            "primary_exact": list(PRIMARY_FAMILIES),
            "exact_family_count": 3,
            "extra_or_deferred_families": [],
        },
        "families": {
            FAMILY_B5A: _b5a(),
            FAMILY_B5B: _b5b(),
            FAMILY_B5C: _b5c(),
        },
        "hypothesis_accounting": {
            "family_configurations": 3,
            "B5A_currency_sleeves": 7,
            "B5B_direct_instrument_sleeves": 1,
            "B5C_instrument_sleeves": 5,
            "total_executable_sleeve_hypotheses": 13,
            "robustness_variant_count": 0,
        },
        "execution_contract_for_future_implementation": {
            "status": "semantics_frozen_signals_and_execution_not_implemented",
            "market_data": "native genuine paired OANDA M1 BID/ASK",
            "entry_and_exit": (
                "first strictly-later observation after decision timestamp"
            ),
            "fills": "BUY at ASK; SELL at BID; exit at opposite executable side",
            "account_currency": "USD",
            "pnl_conversion": (
                "reuse existing account-currency conversion; direct USD quote needs "
                "none, USD base converts quote P&L once using executable conversion, "
                "and non-USD crosses require a causal native USD conversion quote"
            ),
            "reference_notional": "fixed USD 100000 gross per sleeve/cohort",
            "ftmo_sizing_or_optimization": False,
            "interpolation_or_midpoint_fills": False,
        },
        "common_development_gates": copy.deepcopy(COMMON_GATES),
        "breadth_rules": {
            FAMILY_B5A: {
                "sleeve": "one frozen currency-k versus USD mapping",
                "aggregate": (
                    "equal-weight all seven sleeves; failed sleeves are not dropped"
                ),
                "sleeves_positive_native_and_1_5x_expectancy_gte": 5,
                "sleeves_passing_all_sleeve_gates_gte": 4,
            },
            FAMILY_B5B: {
                "sleeve": "the single direct AUD/CAD instrument",
                "breadth": "not_applicable_intrinsically_single_instrument",
                "single_sleeve_must_pass_all_gates": True,
            },
            FAMILY_B5C: {
                "sleeve": "one of the five source-frozen direct pairs",
                "aggregate": (
                    "equal-weight all five sleeves; failed sleeves are not dropped"
                ),
                "sleeves_positive_native_and_1_5x_expectancy_gte": 3,
                "sleeves_passing_all_sleeve_gates_gte": 2,
            },
        },
        "development_to_validation": {
            "validation_in_this_task": False,
            "maximum_representatives_per_family": 1,
            "eligibility": (
                "all common, family-specific, aggregate, and breadth gates pass"
            ),
            "representative_selection": {
                FAMILY_B5A: "the sole frozen seven-sleeve equal-weight configuration",
                FAMILY_B5B: "the sole frozen direct AUD/CAD configuration",
                FAMILY_B5C: "the sole frozen five-sleeve equal-weight configuration",
                "tie_break_if_schema_error_creates_duplicate": (
                    "fail closed; duplicates are not ranked or selected"
                ),
            },
            "one_shot_validation_gates_all_required": {
                "native_net_return_gt": 0,
                "native_profit_factor_gt": 1.0,
                "stress_1_5x_net_expectancy_gt": 0,
                "breadth_minimum_unchanged": True,
            },
            "failure_action": (
                "retire representative and family under frozen definition"
            ),
            "rescue_forbidden": [
                "sign inversion",
                "nearby parameter",
                "alternate pair",
                "alternate holding period",
                "threshold relaxation",
                "favorable seed retry",
                "alternate data vintage",
            ],
        },
        "diagnostics": {
            "status": "report_only_never_gate_rank_filter_or_rescue",
            "fields": [
                "year and chronological-fold scorecards",
                "long-short decomposition",
                "largest one/three/five winner contribution",
                "cost as fraction of gross edge",
                "event or signal counts by year and sleeve",
                "holding duration and skipped occurrence reasons",
                "B5A release lag and revision audit",
            ],
        },
        "relationship_to_prior_research": {
            FAMILY_B5A: {
                "distinct": (
                    "uses externally released institutional positions, not prices alone"
                ),
                "diversification_mechanism": (
                    "institutional positioning and slow-moving flow"
                ),
            },
            FAMILY_B5B: {
                "distinct": (
                    "direct stationary cross-rate with no two-leg rolling OLS, ADF "
                    "selection, Johansen vector, or synthetic spread; therefore not U2"
                ),
                "diversification_mechanism": "direct cross-rate stationary reversion",
            },
            FAMILY_B5C: {
                "distinct": (
                    "contrarian response to an extreme completed daily move, not TSM, "
                    "Donchian/breakout, relative-value spread, or civil-clock/fix "
                    "effect"
                ),
                "diversification_mechanism": (
                    "behavioral short-horizon overreaction/reversal"
                ),
            },
            "empirical_correlations_used": False,
        },
        "reuse_audit": {
            "partition_firewalls": (
                "ftmoquant.research.stage_g.DevelopmentResearchContext.require_range "
                "and OANDA alpha-lab DEVELOPMENT readiness validation"
            ),
            "canonical_OANDA": (
                "ftmoquant.data.oanda_alpha_lab_development and InstrumentSpec; "
                "B5B/B5C missing crosses require additions, not parallel loaders"
            ),
            "native_bid_ask_and_first_later": (
                "ftmoquant.research.alpha_lab.batch4_execution paired-index validation "
                "and bisect_right execution conventions"
            ),
            "account_currency_pnl": (
                "Batch 4 USD-notional and quote-P&L conversion conventions, extended "
                "causally for non-USD direct crosses"
            ),
            "chronological_folds": "Batch 3/4 contiguous chronological fold semantics",
            "transaction_cost_stress": (
                "ftmoquant.research.alpha_lab.cost_stress.widen_bid_ask_frame"
            ),
            "write_once_and_hashing": "Batch 3/4 write-once semantic-hash convention",
            "screening": (
                "Batch 4 expectancy, PF, fold, concentration, winner-removal, cost, "
                "breadth, diagnostics, and no-rescue concepts with frequency-specific "
                "floors"
            ),
            "parallel_implementation_created": False,
        },
        "data_firewall": {
            "permitted_access": [
                "source code",
                "methodology configs",
                "instrument identifiers",
                "literature and regulator documentation",
            ],
            "development_prices_returns_pnl_or_performance_accessed": False,
            "validation_accessed": False,
            "final_holdout_accessed": False,
            "backtest_run": False,
            "signal_or_execution_implemented": False,
            "parameter_tuning_from_repository_data": False,
        },
        "source_provenance": copy.deepcopy(SOURCE_PROVENANCE),
        "lifecycle": {
            "development_performance_accessed": False,
            "validation_accessed": False,
            "holdout_accessed": False,
            "signals_implemented": False,
            "execution_implemented": False,
            "ftmo_optimization_run": False,
        },
    }
    document["preregistration_semantic_sha256"] = _canonical_sha256(document)
    return document


def write_preregistration(*, path: Path = PREREGISTRATION_PATH) -> Path:
    """Write once; an existing artifact is never overwritten."""

    if path.exists():
        raise Batch5PreregistrationError(
            f"{path} already exists; refusing to overwrite"
        )
    document = build_preregistration()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def verify_preregistration(path: Path = PREREGISTRATION_PATH) -> dict[str, Any]:
    """Reject mutation, mutation-plus-rehash, scope drift, or lifecycle drift."""

    try:
        document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Batch5PreregistrationError("could not read preregistration") from error
    actual = document.get("preregistration_semantic_sha256")
    if actual != _canonical_sha256(document):
        raise Batch5PreregistrationError("preregistration_semantic_sha256 mismatch")
    if actual != EXPECTED_PREREGISTRATION_SEMANTIC_SHA256:
        raise Batch5PreregistrationError("frozen preregistration identity mismatch")
    if document.get("preregistration_version") != PREREGISTRATION_VERSION:
        raise Batch5PreregistrationError("unexpected preregistration_version")
    scope = document.get("family_scope", {})
    if scope.get("primary_exact") != list(PRIMARY_FAMILIES):
        raise Batch5PreregistrationError("frozen family scope mismatch")
    if document.get("hypothesis_accounting", {}).get("family_configurations") != 3:
        raise Batch5PreregistrationError("frozen hypothesis count mismatch")
    lifecycle = document.get("lifecycle", {})
    if not lifecycle or any(value is not False for value in lifecycle.values()):
        raise Batch5PreregistrationError("preregistration lifecycle must remain false")
    return document
