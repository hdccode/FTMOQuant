"""B3.0 preregistration, v2: pre-DEVELOPMENT correction of the Batch 3
methodology to match the final approved gate set.

No Batch 3 DEVELOPMENT, VALIDATION, or final-holdout price/trade data had
been accessed before this amendment, so this is a legitimate methodology
correction rather than a results-informed change (see ``amendment_reason``
in :func:`build_preregistration_v2`).

Family scope, partition boundaries, temporal-fold boundaries, the
subperiod-stability split, and the FTMO rule-set/lineage bindings are
UNCHANGED from v1 and are imported and reused verbatim from
:mod:`ftmoquant.research.alpha_lab.batch3_preregistration` rather than
redefined here. Only the DEVELOPMENT hard-gate set, the
report-only-diagnostic set, the FTMO B3.4 gate, and the VALIDATION policy
differ from v1 -- see the module docstring of v1 for what stays identical
and the class-level docstrings below for exactly what changed and why.

v1 (``config/research/batch3_methodology_preregistration_v1.json``,
semantic hash ``82e00f5e5b3a7cf4269bd61163bf8ea4058a31e6414b1fed0077c820a41
7a68b``) is preserved on disk, untouched, and referenced by this document's
``supersedes_semantic_sha256`` field -- it is not overwritten and its own
historical identity is not altered.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ftmoquant.data.oanda_alpha_lab_development import (
    CONFIG_PATH as OANDA_ALPHA_LAB_DEVELOPMENT_CONFIG_PATH,
)
from ftmoquant.data.oanda_alpha_lab_development import (
    LINEAGE_ID as OANDA_ALPHA_LAB_LINEAGE_ID,
)
from ftmoquant.research.alpha_lab.batch3_preregistration import (
    DEVELOPMENT_FOLD_BOUNDARIES,
    DEVELOPMENT_SUBPERIOD_MIDPOINT,
    EXPLICITLY_OUT_OF_SCOPE,
    FIREWALL,
    IN_SCOPE_FAMILIES,
    MID_PIPELINE_EXCLUDED_FROM_BATCH3,
    PROP_RULE_SET_ID,
    PROP_RULE_SET_PATH,
    RETIRED_FAMILIES,
    Batch3PreregistrationError,
    _iso,
    _sha256_file,
)
from ftmoquant.research.alpha_lab.batch3_preregistration import (
    PREREGISTRATION_PATH as V1_PREREGISTRATION_PATH,
)
from ftmoquant.research.alpha_lab.batch3_preregistration import (
    _canonical_sha256 as _canonical_sha256_impl,
)
from ftmoquant.research.stage_g import (
    DEVELOPMENT_END_EXCLUSIVE,
    DEVELOPMENT_START,
    HOLDOUT_START,
    VALIDATION_START,
)

PREREGISTRATION_VERSION = "batch3-methodology-preregistration-v2"
STAGE_ID = "B3.0"

#: The exact v1 semantic hash this document supersedes. Verified against the
#: actual v1 artifact on disk (not merely copy-pasted) by
#: :func:`_verify_supersedes_target` at build time.
SUPERSEDES_SEMANTIC_SHA256 = (
    "82e00f5e5b3a7cf4269bd61163bf8ea4058a31e6414b1fed0077c820a417a68b"
)

AMENDMENT_REASON = (
    "Pre-DEVELOPMENT correction to match the final approved methodology; "
    "no Batch-3 market/trade data had been accessed."
)

PREREGISTRATION_PATH = Path(
    "config/research/batch3_methodology_preregistration_v2.json"
)

_canonical_sha256 = _canonical_sha256_impl


def _verify_supersedes_target(v1_path: Path = V1_PREREGISTRATION_PATH) -> None:
    """Confirm the v1 artifact this document claims to supersede is, in
    fact, still on disk with exactly the claimed hash. Reads only the small
    JSON methodology document itself -- never price or trade data."""

    if not v1_path.exists():
        raise Batch3PreregistrationError(
            f"cannot supersede {v1_path}: file does not exist"
        )
    v1_document = json.loads(v1_path.read_text(encoding="utf-8"))
    v1_stored_hash = v1_document.get("preregistration_semantic_sha256")
    if v1_stored_hash != SUPERSEDES_SEMANTIC_SHA256:
        raise Batch3PreregistrationError(
            "SUPERSEDES_SEMANTIC_SHA256 does not match the hash actually "
            f"recorded in {v1_path} (expected {SUPERSEDES_SEMANTIC_SHA256!r}, "
            f"found {v1_stored_hash!r})"
        )


# ---------------------------------------------------------------------------
# DEVELOPMENT hard gates (v2, final approved compact set)
# ---------------------------------------------------------------------------

DEVELOPMENT_GATES: dict[str, Any] = {
    "evaluation_precedence": (
        "the opportunity_density gate is evaluated FIRST, per sleeve, and "
        "short-circuits all other hard gates on failure"
    ),
    "opportunity_density": {
        "default_min_trades": 80,
        "family_overrides": {
            "B3F1_spread_mean_reversion": 50,
        },
        "override_rationale": (
            "B3F1 (cross-instrument spread mean reversion) may be "
            "structurally lower-frequency than the two session-anchored "
            "families; this exception is frozen now, before any Batch 3 "
            "DEVELOPMENT result is observed, and applies only to B3F1"
        ),
        "scope": (
            "per sleeve (instrument x family x frozen parameter "
            "configuration), over the full frozen DEVELOPMENT interval"
        ),
    },
    "economic": {
        "expectancy_usd_per_trade_gt": 0,
        "profit_factor_gt": 1.10,
    },
    "temporal_stability": {
        "fold_count": 4,
        "fold_boundaries_utc": [_iso(bound) for bound in DEVELOPMENT_FOLD_BOUNDARIES],
        "fold_derivation": (
            "unchanged from v1: four equal 373-day chronological folds "
            "spanning [DEVELOPMENT_START, DEVELOPMENT_END_EXCLUSIVE), "
            "derived by pure calendar-day division without inspecting any "
            "return, price, or trade data"
        ),
        "rule": "positive net return in at least 3 of the 4 folds",
    },
    "exceptional_winner_dependency": {
        "rule": (
            "rank only PROFITABLE completed trades (realized P&L > 0) in "
            "the sleeve by realized P&L descending; remove the top N = "
            "ceil(0.05 * profitable_trade_count) trades (ceiling rounding, "
            "so at least one profitable trade is always removed when any "
            "exist); ties at the boundary between the Nth and (N+1)th-"
            "ranked profitable trade by P&L are broken by removing the "
            "earlier-exit-timestamp trade first, so the removed set is "
            "always exactly N trades and fully deterministic"
        ),
        "remaining_expectancy_usd_per_trade_gt": 0,
        "removed_from_v1": (
            "the v1 'remaining profit_factor > 1.05' requirement is "
            "removed -- it was not part of the final approved compact "
            "hard-gate set. Only remaining expectancy is gated"
        ),
    },
    "profit_concentration": {
        "denominator": (
            "sum of realized P&L over trades with strictly positive "
            "realized P&L only, per sleeve, over the full DEVELOPMENT "
            "interval ('total positive DEVELOPMENT profit')"
        ),
        "attribution_timestamp": "trade exit timestamp (UTC)",
        "period_boundaries": "calendar quarter in UTC",
        "max_single_quarter_share": 0.40,
        "removed_from_v1": (
            "the v1 15% calendar-month hard gate is removed (monthly "
            "concentration is now report-only, see diagnostics); the v1 "
            "30% quarterly threshold is replaced by 40%, not combined "
            "with it"
        ),
        "zero_or_negative_denominator": (
            "if total positive DEVELOPMENT profit <= 0 for a sleeve, this "
            "gate fails immediately (fail closed)"
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
        "min_connected_region_size": 2,
        "amendment_rule": (
            "frozen at exactly 2 for all three families; no post-result "
            "increase or decrease of this threshold is permitted"
        ),
    },
    "transaction_cost_sensitivity": {
        "primary_execution": (
            "genuine native M1 BID/ASK trade-lifecycle execution from the "
            "outset for all three families (unchanged from v1)"
        ),
        "required_stress_mechanism": (
            "synthetically widen every bar's BID/ASK spread symmetrically "
            "around its own mid price BEFORE execution (ask' = mid + "
            "(ask - mid) * multiplier, bid' = mid - (mid - bid) * "
            "multiplier), then re-run the identical, unmodified frozen "
            "signal and execution logic against the widened series; "
            "do not fabricate a post-hoc fee subtraction against an "
            "already-computed native-spread return"
        ),
        "implementation_status": (
            "does not exist in the repository yet; must be built as "
            "generic, shared, tested infrastructure in B3.1 BEFORE any "
            "Batch 3 DEVELOPMENT screening"
        ),
        "family_requirements": {
            "B3F1_spread_mean_reversion": {
                "must_survive_multipliers": [1.5],
            },
            "B3F2_asian_range_fade": {
                "must_survive_multipliers": [1.5, 2.0],
            },
            "B3F3_session_open_microstructure_mean_reversion": {
                "must_survive_multipliers": [1.5, 2.0],
            },
        },
        "survival_rule": (
            "expectancy_usd_per_trade > 0 at each required multiplier for "
            "that family"
        ),
    },
}


# ---------------------------------------------------------------------------
# DEVELOPMENT diagnostics (v2): report-only, must NOT block B3.2/B3.3
# advancement, and must NOT be used to construct post-hoc filters.
# ---------------------------------------------------------------------------

DEVELOPMENT_DIAGNOSTICS: dict[str, Any] = {
    "status": "report_only_not_a_gate_at_any_stage_before_b3_4",
    "no_post_hoc_filtering_rule": (
        "no diagnostic value below may be used to filter, rank, select, or "
        "exclude a candidate/sleeve at B3.2 or B3.3; diagnostics may only "
        "be reported alongside a candidate that already passed every hard "
        "gate in DEVELOPMENT_GATES"
    ),
    "rolling_expectancy": {
        "window_sizes": [30, 50],
        "windowing_convention": (
            "unchanged from v1: trailing windows over trades ordered by "
            "exit timestamp; windows requiring fewer than the window size "
            "of completed trades are excluded entirely, never padded or "
            "zero-filled (matches "
            "ftmoquant.research.ftmo_pass_probability.alpha_diagnostic."
            "_rolling_expectancy)"
        ),
        "reported_fields": [
            "median_eligible_window_expectancy",
            "fraction_of_eligible_windows_positive",
        ],
        "removed_as_hard_gate_from_v1": (
            "'median > 0' and '>=70% of windows positive' are no longer "
            "gates -- reported only"
        ),
    },
    "monthly_concentration": {
        "denominator": (
            "same as the quarterly hard-gate denominator (total positive "
            "DEVELOPMENT profit)"
        ),
        "period_boundaries": "calendar month in UTC",
        "reported_field": "max_single_month_share",
        "removed_as_hard_gate_from_v1": (
            "the 15% monthly threshold is not gated at any threshold"
        ),
    },
    "tail_statistics": {
        "reported_fields": [
            "largest_winning_trade_share_of_total_positive_profit",
            "pnl_skewness",
            "pnl_kurtosis",
        ],
        "removed_as_hard_gate_from_v1": (
            "the 20% largest-single-trade threshold is not gated"
        ),
    },
    "directional_breakdown": {
        "reported_fields_per_direction": [
            "trade_count",
            "expectancy_usd_per_trade",
            "profit_factor",
            "net_return",
        ],
        "removed_as_hard_gate_from_v1": (
            "'long PF >= 1.0 and short PF >= 1.0' is not gated"
        ),
        "no_direction_specific_filtering_rule": (
            "no direction-specific filter, exception, or exclusion may be "
            "introduced after DEVELOPMENT results for this or any Batch 3 "
            "family are observed"
        ),
    },
    "drawdown": {
        "reported_field": (
            "p95_max_drawdown_usd (as already emitted by "
            "ftmoquant.research.ftmo_pass_probability.reporting, at the "
            "sleeve's DEVELOPMENT-selected sizing policy)"
        ),
        "removed_as_hard_gate_from_v1": (
            "'p95 max drawdown < $3,000' is not an alpha-stage hard gate; "
            "FTMO path suitability is evaluated later, at B3.4, using the "
            "actual FTMO state machine / Monte Carlo pass_both gate"
        ),
    },
}


# ---------------------------------------------------------------------------
# FTMO-specific gates (v2)
# ---------------------------------------------------------------------------

FTMO_DEVELOPMENT_GATES: dict[str, Any] = {
    "applicability": (
        "evaluated only for a candidate that has already cleared every "
        "hard gate in DEVELOPMENT_GATES"
    ),
    "full_development_pass_both": {
        "status": "hard_gate",
        "mechanism": (
            "reuse ftmoquant.research.ftmo_pass_probability.monte_carlo/"
            "bootstrap/reporting unchanged: 100,000-replication stationary "
            "block bootstrap, block length derived via "
            "derive_frozen_block_length on DEVELOPMENT trades only, at the "
            "sleeve's DEVELOPMENT-selected sizing policy"
        ),
        "rule": "pass_both >= 0.70",
    },
    "cross_subperiod_stability": {
        "status": "diagnostic_warning_not_an_advancement_gate",
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
                "unchanged from v1: deterministic floor(span/2)-day "
                "midpoint of the frozen DEVELOPMENT interval"
            ),
        },
        "mechanism": (
            "run the identical frozen Monte Carlo methodology (same "
            "DEVELOPMENT-selected sizing policy, same bootstrap method, "
            "same 100,000-replication count) independently on each half's "
            "trades"
        ),
        "ratio_definition": "pass_both(weaker_half) / pass_both(stronger_half)",
        "zero_denominator_handling": (
            "unchanged from v1: if the stronger half's pass_both is "
            "exactly 0, the ratio is undefined and reported as such"
        ),
        "zero_trade_half_handling": (
            "unchanged from v1: if either half has zero completed trades, "
            "the ratio is reported as undefined for that sleeve"
        ),
        "labeling_rule": {
            "ratio_gte_0_60": "stable_diagnostic",
            "ratio_lt_0_60": "instability_warning",
        },
        "removed_as_hard_gate_from_v1": (
            "the >=0.60 threshold no longer blocks B3.4 advancement by "
            "itself, and no candidate may be selected or excluded based on "
            "this ratio alone"
        ),
    },
}


# ---------------------------------------------------------------------------
# Sizing-selection procedure (v2): identical mechanics to v1, with the
# DEVELOPMENT-only / VALIDATION-untouched boundary stated explicitly.
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
            "the Stage 1-selected policy's own family and value, with its "
            "construction rule fixed now (identical formulas to v1: fixed "
            "offsets {-0.50,-0.35,-0.25,-0.15,-0.05,+0.05,+0.15,+0.25,"
            "+0.50} around a notional-multiplier center floored at 0.1x; "
            "multipliers {0.50,0.65,0.75,0.85,0.95,1.05,1.15,1.25,1.50} "
            "around a fractional-risk center floored at 0.1% and capped "
            "at 5%)"
        ),
        "mechanism_run": (
            "the full 100,000-replication precision Monte Carlo across "
            "the 9-policy refinement grid"
        ),
        "selection_rule": (
            "the single policy with the highest pass_both is the sleeve's "
            "DEVELOPMENT-selected sizing policy"
        ),
        "downstream_status": (
            "this DEVELOPMENT-selected policy is used, frozen and "
            "unmodified, for the full_development_pass_both gate, the "
            "cross_subperiod_stability diagnostic, and the VALIDATION "
            "pass_both diagnostic -- it is never re-selected using "
            "VALIDATION data"
        ),
    },
    "expansion_prohibition": (
        "no Stage 3 / further local search is permitted after Stage 2 "
        "results are observed; the procedure terminates at Stage 2"
    ),
}


# ---------------------------------------------------------------------------
# VALIDATION policy (v2)
# ---------------------------------------------------------------------------

VALIDATION_POLICY: dict[str, Any] = {
    "access_model": "one_shot_per_frozen_candidate",
    "preconditions_before_validation_access": [
        "frozen signal implementation",
        "frozen pair/instrument universe",
        "frozen parameters",
        "frozen execution semantics",
        (
            "frozen sizing (Stage 2 DEVELOPMENT-selected policy from "
            "sizing_selection_procedure)"
        ),
        "frozen preregistration document with a verified self-hash",
    ],
    "precommitted_gates": {
        "A_native_spread_positive_return": ("native-spread VALIDATION net_return > 0"),
        "B_native_spread_positive_sharpe": (
            "native-spread VALIDATION annualized_sharpe > 0"
        ),
        "validation_passed": "A AND B",
    },
    "removed_as_hard_gate_from_v1": (
        "VALIDATION pass_both >= 0.70 is no longer a hard validation gate"
    ),
    "validation_ftmo_pass_both": {
        "status": "report_only_diagnostic",
        "mechanism": (
            "computed on VALIDATION trades using the exact frozen "
            "DEVELOPMENT-selected sizing policy and bootstrap method -- no "
            "re-selection, no sizing reselection from VALIDATION"
        ),
    },
    "additional_diagnostics": (
        "the existing alpha_diagnostic.py-style diagnostics (rolling "
        "expectancy, monthly/quarterly stability, subgroup audit, "
        "chronological replay) may also be reported for a candidate that "
        "passes A/B, but must NOT become an additional post-hoc pass/fail "
        "rule -- only A and B gate VALIDATION"
    ),
    "on_failure": (
        "retire the candidate; no alternative pair/parameter/sizing "
        "rescue, and no VALIDATION-informed retuning"
    ),
}


# ---------------------------------------------------------------------------
# Document assembly / hashing / verification
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Batch3PreregistrationV2:
    document: dict[str, Any]
    semantic_sha256: str


def build_preregistration_v2(
    *, created_at_utc: datetime | None = None, verify_supersedes: bool = True
) -> dict[str, Any]:
    """Build the full v2 preregistration document.

    Reads only this module's own frozen Python literals, the byte contents
    of two already-frozen configuration files (OANDA alpha-lab DEVELOPMENT
    lineage config, FTMO rule-set YAML), and -- when ``verify_supersedes``
    is True -- the small v1 preregistration JSON document itself, to confirm
    the ``supersedes_semantic_sha256`` claim is accurate. Never DEVELOPMENT,
    VALIDATION, or HOLDOUT price/trade data.
    """

    if verify_supersedes:
        _verify_supersedes_target()

    timestamp = created_at_utc if created_at_utc is not None else datetime.now(UTC)
    document: dict[str, Any] = {
        "preregistration_version": PREREGISTRATION_VERSION,
        "stage": STAGE_ID,
        "created_at_utc": _iso(timestamp),
        "supersedes_semantic_sha256": SUPERSEDES_SEMANTIC_SHA256,
        "amendment_reason": AMENDMENT_REASON,
        "purpose": (
            "Amend the Batch 3 research protocol to the final approved "
            "methodology before any Batch 3 DEVELOPMENT analysis occurs."
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
            "unchanged_from_v1": True,
            "in_scope": copy.deepcopy(IN_SCOPE_FAMILIES),
            "explicitly_out_of_scope": list(EXPLICITLY_OUT_OF_SCOPE),
            "retired_from_prior_batches": copy.deepcopy(list(RETIRED_FAMILIES)),
            "mid_pipeline_excluded_from_batch3": copy.deepcopy(
                list(MID_PIPELINE_EXCLUDED_FROM_BATCH3)
            ),
        },
        "development_gates": copy.deepcopy(DEVELOPMENT_GATES),
        "development_diagnostics": copy.deepcopy(DEVELOPMENT_DIAGNOSTICS),
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


def write_preregistration_v2(
    *, path: Path = PREREGISTRATION_PATH, created_at_utc: datetime | None = None
) -> Path:
    """Write the v2 preregistration document. Refuses to overwrite an
    existing file. Does not touch or overwrite the v1 artifact."""

    if path.exists():
        raise Batch3PreregistrationError(
            f"{path} already exists; refusing to overwrite"
        )
    document = build_preregistration_v2(created_at_utc=created_at_utc)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def verify_preregistration_v2(path: Path = PREREGISTRATION_PATH) -> dict[str, Any]:
    """Load a v2 preregistration document and verify its self-hash,
    version, supersedes claim, partition boundaries, and lifecycle flags."""

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
    if document.get("supersedes_semantic_sha256") != SUPERSEDES_SEMANTIC_SHA256:
        raise Batch3PreregistrationError("supersedes_semantic_sha256 mismatch")

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
