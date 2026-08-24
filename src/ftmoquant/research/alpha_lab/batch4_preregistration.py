"""Write-once Batch 4 structural intraday-flow preregistration.

This module is deliberately limited to immutable methodology metadata and pure
calendar/sign helpers.  It imports no market-data, return, execution, or
partition-access code and does not implement a trading signal.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

PREREGISTRATION_VERSION = "batch4-structural-intraday-flow-preregistration-v1"
STAGE_ID = "B4.0"
PREREGISTRATION_PATH = Path(
    "config/research/batch4_structural_intraday_flow_preregistration_v1.json"
)

UNIVERSE = (
    "AUD/USD.OANDA",
    "EUR/USD.OANDA",
    "GBP/USD.OANDA",
    "NZD/USD.OANDA",
    "USD/CAD.OANDA",
    "USD/CHF.OANDA",
    "USD/JPY.OANDA",
)
PRIMARY_FAMILIES = (
    "B4F1A_local_hours_flow_seasonality",
    "B4F1B_london_fix_flow_reversal",
    "B4F1C_tokyo_fix_flow_reversal",
)
FIX_DURATIONS_MINUTES = (15, 30, 60)
Side = Literal["BUY", "SELL"]


class Batch4PreregistrationError(RuntimeError):
    """Raised when the frozen Batch 4 document contract is violated."""


def _canonical_sha256(document: dict[str, Any]) -> str:
    """Hash semantic content, excluding the self-hash field itself."""

    semantic = {
        key: value
        for key, value in document.items()
        if key != "preregistration_semantic_sha256"
    }
    payload = json.dumps(
        semantic, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def pair_side_for_currency_move(
    pair: str, currency: str, move: Literal["APPRECIATES", "DEPRECIATES"]
) -> Side:
    """Map a signed currency hypothesis to the correct USD-pair trade side."""

    if pair not in UNIVERSE:
        raise Batch4PreregistrationError(f"pair is outside frozen universe: {pair}")
    base, quote_with_venue = pair.split("/", maxsplit=1)
    quote = quote_with_venue.split(".", maxsplit=1)[0]
    if currency not in {base, quote}:
        raise Batch4PreregistrationError(f"{currency} is not represented by {pair}")
    base_up = (currency == base and move == "APPRECIATES") or (
        currency == quote and move == "DEPRECIATES"
    )
    return "BUY" if base_up else "SELL"


def local_window_utc(
    local_date: date, timezone: str, start_local: str, end_local: str
) -> tuple[datetime, datetime]:
    """Convert a frozen same-day civil-time window to aware UTC bounds."""

    start_clock = time.fromisoformat(start_local)
    end_clock = time.fromisoformat(end_local)
    if end_clock <= start_clock:
        raise Batch4PreregistrationError("Batch 4 windows must end on the same day")
    zone = ZoneInfo(timezone)
    start = datetime.combine(local_date, start_clock, tzinfo=zone)
    end = datetime.combine(local_date, end_clock, tzinfo=zone)
    return start.astimezone(UTC), end.astimezone(UTC)


_LOCAL_SLEEVE_INPUTS = (
    ("AUD", "AUD/USD.OANDA", "Australia/Sydney", "09:00", "17:00"),
    ("EUR", "EUR/USD.OANDA", "Europe/Berlin", "08:00", "16:00"),
    ("GBP", "GBP/USD.OANDA", "Europe/London", "08:00", "16:00"),
    ("NZD", "NZD/USD.OANDA", "Pacific/Auckland", "09:00", "17:00"),
    ("CAD", "USD/CAD.OANDA", "America/Toronto", "08:00", "16:00"),
    ("CHF", "USD/CHF.OANDA", "Europe/Zurich", "08:00", "16:00"),
    ("JPY", "USD/JPY.OANDA", "Asia/Tokyo", "09:00", "17:00"),
)


def _local_sleeves() -> list[dict[str, Any]]:
    return [
        {
            "sleeve_id": f"B4F1A_{currency}",
            "currency_tested": currency,
            "pair": pair,
            "timezone": timezone,
            "start_local": start,
            "end_local": end,
            "directional_hypothesis": f"{currency}_DEPRECIATES",
            "entry_side": pair_side_for_currency_move(pair, currency, "DEPRECIATES"),
            "entry_decision": "local start_local on each eligible business day",
            "exit_decision": "same local date at end_local",
        }
        for currency, pair, timezone, start, end in _LOCAL_SLEEVE_INPUTS
    ]


def _usd_side(pair: str, move: Literal["APPRECIATES", "DEPRECIATES"]) -> Side:
    return pair_side_for_currency_move(pair, "USD", move)


def _fix_configurations(timezone: str, fix_clock: str) -> list[dict[str, Any]]:
    fix_minutes = int(fix_clock[:2]) * 60 + int(fix_clock[3:])
    rows: list[dict[str, Any]] = []
    hypotheses: tuple[
        tuple[str, Literal["APPRECIATES", "DEPRECIATES"]], ...
    ] = (
        ("PRE", "APPRECIATES"),
        ("POST", "DEPRECIATES"),
    )
    for phase, usd_move in hypotheses:
        for duration in FIX_DURATIONS_MINUTES:
            start_minutes = fix_minutes - duration if phase == "PRE" else fix_minutes
            end_minutes = fix_minutes if phase == "PRE" else fix_minutes + duration

            def clock(value: int) -> str:
                return f"{value // 60:02d}:{value % 60:02d}"

            rows.append(
                {
                    "configuration_id": f"{phase}_{duration}m",
                    "phase": phase,
                    "duration_minutes": duration,
                    "timezone": timezone,
                    "start_local": clock(start_minutes),
                    "end_local": clock(end_minutes),
                    "directional_hypothesis": f"USD_{usd_move}",
                    "pair_entry_sides": {
                        pair: _usd_side(pair, usd_move) for pair in UNIVERSE
                    },
                    "eligible_pairs": list(UNIVERSE),
                }
            )
    return rows


DEVELOPMENT_GATES: dict[str, Any] = {
    "scope": "each executable pair-sleeve x frozen configuration",
    "A_opportunity_density": {"completed_trades_gte": 250},
    "B_native_expectancy": {"expectancy_usd_per_trade_gt": 0},
    "C_native_profit_factor": {"profit_factor_gt": 1.10},
    "D_temporal_stability": {
        "chronological_development_fold_count": 4,
        "positive_net_return_folds_gte": 3,
        "reuse": "Batch-3 v2 frozen chronological DEVELOPMENT folds",
    },
    "E_exceptional_winner_dependence": {
        "remove": "top ceil(5%) of profitable completed trades only",
        "ranking": "realized P&L descending; equal P&L removes earlier UTC exit first",
        "remaining_expectancy_usd_per_trade_gt": 0,
    },
    "F_quarter_concentration": {
        "attribution": "UTC calendar quarter of trade exit",
        "denominator": "total strictly positive DEVELOPMENT realized P&L",
        "max_single_quarter_share_lte": 0.40,
        "nonpositive_denominator": "fail_closed",
    },
    "G_cost_stress": {
        "required_multipliers": [1.5, 2.0],
        "expectancy_usd_per_trade_gt_at_each": 0,
    },
    "H_parameter_neighborhood": {
        "min_connected_passing_region": 2,
        "only_where_genuine_axis_exists": True,
        "B4F1A": "not_applicable_no_parameter_axis; do not fabricate one",
        "B4F1B_and_B4F1C": (
            "duration adjacency 15-30-60 within the same phase and pair only; "
            "PRE and POST are never adjacent"
        ),
    },
    "all_gates_required": True,
}

DEVELOPMENT_DIAGNOSTICS: dict[str, Any] = {
    "status": "report_only_never_gate_rank_filter_or_rescue",
    "rolling_expectancy_trade_windows": [50, 100],
    "rolling_report_fields": ["median_expectancy", "fraction_positive_windows"],
    "other_report_fields": [
        "monthly_positive_profit_concentration",
        "weekday_breakdown",
        "long_short_breakdown_where_structurally_applicable",
        "session_and_year_breakdown",
        "trade_pnl_skewness_and_kurtosis",
        "largest_winning_trade_share_of_total_positive_profit",
        "stop_target_time_exit_fractions_where_applicable",
        "mean_and_median_holding_minutes",
        "completed_trade_count_per_calendar_year",
        "spread_cost_as_percent_of_gross_edge",
    ],
}

SOURCE_PROVENANCE: list[dict[str, Any]] = [
    {
        "title": "Intraday Patterns in FX Returns and Order Flow",
        "authors": ["Francis Breedon", "Angelo Ranaldo"],
        "year": 2013,
        "type": "academic_peer_reviewed",
        "url": "https://doi.org/10.1111/jmcb.12032",
        "supports": "B4F1A",
        "directional_claim_used": (
            "a currency tends to depreciate during its own local trading hours"
        ),
        "timing_claim_used": (
            "economically defined local trading hours; the paper reports the "
            "effect is not materially sensitive to precise session endpoints"
        ),
    },
    {
        "title": "Foreign Exchange Fixings and Returns around the Clock",
        "authors": ["Ingomar Krohn", "Philippe Mueller", "Paul Whelan"],
        "year": 2024,
        "type": "academic_peer_reviewed",
        "url": "https://doi.org/10.1111/jofi.13306",
        "supports": "B4F1B and B4F1C",
        "directional_claim_used": (
            "USD appreciates before the London and Tokyo fixes and depreciates "
            "afterward"
        ),
        "timing_claim_used": "London 16:00 local; Tokyo 09:55 local",
    },
    {
        "title": "WM/Reuters FX benchmarks",
        "authors": ["FTSE Russell / LSEG"],
        "year": 2026,
        "type": "institutional_benchmark_documentation",
        "url": "https://www.lseg.com/en/ftse-russell/benchmarks/wmr-fx-benchmarks",
        "supports": "B4F1B anchor",
        "directional_claim_used": "none; benchmark-clock corroboration only",
        "timing_claim_used": "WM/R closing spot rates fixed at 16:00 London",
    },
    {
        "title": "Was the Forex Fixing Fixed?",
        "authors": ["Takatoshi Ito", "Masahiro Yamada"],
        "year": 2015,
        "type": "academic_NBER",
        "url": "https://doi.org/10.3386/w21518",
        "supports": "B4F1B and B4F1C benchmark-clock corroboration",
        "directional_claim_used": "none; clock corroboration only",
        "timing_claim_used": "Tokyo 09:55 local and London 16:00 local",
    },
    {
        "title": "Foreign Exchange Rates",
        "authors": ["Sumitomo Mitsui Banking Corporation"],
        "year": 2026,
        "type": "institutional_bank_documentation",
        "url": "https://www.smbc.co.jp/global/terms_fx/index_e.html",
        "supports": "B4F1C anchor",
        "directional_claim_used": "none; benchmark-clock corroboration only",
        "timing_claim_used": (
            "TTM is based on the actual market rate around 09:55 Tokyo"
        ),
    },
]


def build_preregistration(*, created_at_utc: datetime | None = None) -> dict[str, Any]:
    """Build the frozen document without reading any price or result data."""

    created = created_at_utc or datetime.now(UTC)
    if created.tzinfo is None:
        raise Batch4PreregistrationError("created_at_utc must be timezone-aware")
    london = _fix_configurations("Europe/London", "16:00")
    tokyo = _fix_configurations("Asia/Tokyo", "09:55")
    local = _local_sleeves()
    document: dict[str, Any] = {
        "preregistration_version": PREREGISTRATION_VERSION,
        "stage": STAGE_ID,
        "created_at_utc": created.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "purpose": (
            "freeze Batch 4 methodology before any new DEVELOPMENT performance access"
        ),
        "family_scope": {
            "primary_exact": list(PRIMARY_FAMILIES),
            "deferred_not_implemented": ["B4F2_cross_currency_lead_lag_propagation"],
            "no_extra_families": True,
        },
        "universe": list(UNIVERSE),
        "families": {
            PRIMARY_FAMILIES[0]: {
                "alpha": "clock/local institutional-customer flow only",
                "sleeves": local,
                "parameter_axis": None,
                "trade_rule": (
                    "one deterministic same-local-day time-exit trade per sleeve/day"
                ),
            },
            PRIMARY_FAMILIES[1]: {
                "institutional_anchor": "WM/R London fix",
                "fix_timezone": "Europe/London",
                "fix_local": "16:00",
                "configurations": london,
            },
            PRIMARY_FAMILIES[2]: {
                "institutional_anchor": "Tokyo fixing convention",
                "fix_timezone": "Asia/Tokyo",
                "fix_local": "09:55",
                "clock_resolution": (
                    "authoritative and academic sources agree on the 09:55 "
                    "market-rate/fix "
                    "anchor; a later publication time is not the signal clock"
                ),
                "material_definition_conflict_found": False,
                "configurations": tokyo,
            },
        },
        "execution_contract_for_future_implementation": {
            "implementation_status": (
                "frozen_semantics_only_signals_and_execution_not_implemented"
            ),
            "data": "native genuine paired OANDA M1 BID/ASK",
            "entry": (
                "first strictly-later paired M1 observation after entry decision "
                "timestamp"
            ),
            "exit": (
                "first strictly-later paired M1 observation after frozen exit timestamp"
            ),
            "fills": (
                "BUY at ASK and SELL at BID; exits use the opposite executable side"
            ),
            "same_local_date_only": True,
            "position_rule": "at most one position per pair-sleeve/configuration",
            "risk_exit": "none; time exit only",
            "sizing": "fixed USD 100000 gross notional per sleeve",
            "forbidden": [
                "interpolation",
                "midpoint_fills",
                "post_hoc_fee_subtraction",
                "price_trigger",
                "ATR_filter",
                "volatility_filter",
                "trend_filter",
                "news_filter",
                "range_filter",
                "stop_or_target_optimization",
            ],
        },
        "cost_stress": {
            "reuse_exact_function": (
                "ftmoquant.research.alpha_lab.cost_stress.widen_bid_ask_frame"
            ),
            "multipliers": [1.5, 2.0],
            "application": (
                "symmetrically widen each spread before execution and rerun "
                "identical logic"
            ),
            "post_hoc_cost_scalar_forbidden": True,
        },
        "grid_accounting": {
            "local_hours_sleeve_window_cells": len(local),
            "london_fix_timing_direction_cells_before_pairs": len(london),
            "tokyo_fix_timing_direction_cells_before_pairs": len(tokyo),
            "total_primary_cells_before_fix_pair_multiplication": len(local)
            + len(london)
            + len(tokyo),
            "london_executable_pair_cells": len(london) * len(UNIVERSE),
            "tokyo_executable_pair_cells": len(tokyo) * len(UNIVERSE),
            "total_executable_sleeve_configuration_hypotheses": len(local)
            + (len(london) + len(tokyo)) * len(UNIVERSE),
            "unique_primary_families": len(PRIMARY_FAMILIES),
        },
        "development_gates": copy.deepcopy(DEVELOPMENT_GATES),
        "development_diagnostics": copy.deepcopy(DEVELOPMENT_DIAGNOSTICS),
        "family_breadth_gate": {
            "evaluation_unit": (
                "B4F1A across its seven frozen local sleeves; B4F1B/B4F1C separately "
                "for each frozen phase-duration configuration across seven eligible "
                "pairs"
            ),
            "eligible_subset_frozen": list(UNIVERSE),
            "sleeves_meeting_breadth_metrics_gte": 3,
            "breadth_metrics_all_required": {
                "native_expectancy_usd_per_trade_gt": 0,
                "native_profit_factor_gt": 1.0,
                "stress_1_5x_expectancy_usd_per_trade_gt": 0,
            },
            "sleeves_passing_full_hard_gate_set_gte": 2,
        },
        "development_to_validation": {
            "validation_access_in_this_task": False,
            "eligibility": "mechanically selected DEVELOPMENT survivors only",
            "representative_ranking": [
                "highest median sleeve expectancy",
                "highest median sleeve profit factor",
                "lowest median sleeve max-quarter positive-profit concentration",
                "highest aggregate completed trade count",
                "lexicographically smallest strategy_id",
            ],
            "number_proceeding": 1,
            "one_shot_validation_gates_all_required": {
                "native_net_return_gt": 0,
                "native_annualized_sharpe_gt": 0,
                "stress_1_5x_expectancy_usd_per_trade_gt": 0,
            },
            "after_failure": "retire_no_rescue",
        },
        "multiple_testing_and_no_rescue": {
            "no_additional_windows": True,
            "no_sign_inversion": True,
            "no_pair_cherry_picking": True,
            "no_nearby_time_rescue": True,
            "no_weekday_rescue": True,
            "no_volatility_regime_rescue": True,
            "failed_family_action": "retire under frozen definition",
            "future_modification": "new explicitly labelled exploratory programme",
        },
        "ftmo_stage": (
            "prohibited until DEVELOPMENT, one-shot VALIDATION, and execution "
            "promotion pass"
        ),
        "source_provenance": copy.deepcopy(SOURCE_PROVENANCE),
        "reuse_audit": {
            "reuse": [
                "native paired-M1 execution conventions from Batch 3",
                "ZoneInfo-based session handling",
                "cost_stress.widen_bid_ask_frame",
                "account-currency conversion and Alpha Lab gate/report infrastructure "
                "at implementation",
                "Batch-3 v2 folds, concentration, winner-removal and "
                "connected-component semantics",
                "write-once semantic-hash artifact convention",
            ],
            "retired_session_strategies_audited": [
                "session_range_expansion_v1",
                "eurusd_session_range_expansion_v1",
                "session_continuation_signals",
                "B3F2_asian_range_fade",
                "B3F3_session_open_microstructure_mean_reversion",
            ],
            "structural_distinction": (
                "Batch 4 is triggered only by a preregistered civil clock or "
                "institutional fix; it has no range, ATR, breakout, reversal-pattern, "
                "anchor, or z-score trigger"
            ),
        },
        "deferred_B4F2": {
            "status": "recorded_only_not_part_of_B4F1_not_implemented",
            "future_requirements": [
                "synchronized M1 returns",
                "causal lagged predictor only",
                "strict first-later execution",
                "dependence-preserving pair mapping",
                "native spread from outset",
                "no contemporaneous leakage",
                "no arbitrary pair cherry-picking",
                "explicit latency sensitivity",
            ],
        },
        "data_firewall": {
            "permitted": [
                "source code",
                "methodology configs",
                "metadata",
                "literature references",
                "pure calendar/timezone calculations",
            ],
            "development_prices_or_returns_accessed": False,
            "validation_prices_or_results_accessed": False,
            "final_holdout_accessed": False,
            "backtest_run": False,
            "monte_carlo_run": False,
            "data_loader_imports_allowed_in_module": False,
        },
        "lifecycle": {
            "development_accessed": False,
            "validation_accessed": False,
            "holdout_accessed": False,
            "signals_implemented": False,
            "execution_implemented": False,
        },
    }
    document["preregistration_semantic_sha256"] = _canonical_sha256(document)
    return document


def write_preregistration(
    *, path: Path = PREREGISTRATION_PATH, created_at_utc: datetime | None = None
) -> Path:
    """Write the artifact once; an existing target is never overwritten."""

    if path.exists():
        raise Batch4PreregistrationError(
            f"{path} already exists; refusing to overwrite"
        )
    document = build_preregistration(created_at_utc=created_at_utc)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def verify_preregistration(path: Path = PREREGISTRATION_PATH) -> dict[str, Any]:
    """Fail closed on semantic tampering or lifecycle/scope drift."""

    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if document.get("preregistration_semantic_sha256") != _canonical_sha256(document):
        raise Batch4PreregistrationError("preregistration_semantic_sha256 mismatch")
    if document.get("preregistration_version") != PREREGISTRATION_VERSION:
        raise Batch4PreregistrationError("unexpected preregistration_version")
    if document.get("family_scope", {}).get("primary_exact") != list(PRIMARY_FAMILIES):
        raise Batch4PreregistrationError("frozen primary family scope mismatch")
    if document.get("universe") != list(UNIVERSE):
        raise Batch4PreregistrationError("frozen universe mismatch")
    lifecycle = document.get("lifecycle", {})
    if any(value is not False for value in lifecycle.values()):
        raise Batch4PreregistrationError(
            "preregistration lifecycle must remain entirely false"
        )
    return document
