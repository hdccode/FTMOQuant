from __future__ import annotations

import ast
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from ftmoquant.research.alpha_lab.b3f1_underpowered_resolution import (
    FROZEN_CANDIDATES,
    UNDERPOWERED_CONFIRMATION_ELIGIBLE,
    UNDERPOWERED_REJECTED,
    B3F1ResolutionError,
    EligibilityInputs,
    WinnerSelectionInputs,
    compute_bootstrap_diagnostics,
    evaluate_eligibility,
    frozen_fallback_block_size,
    leave_one_out,
    require_frozen_candidate,
    resolve_block_size,
    select_winner,
)

# ---------------------------------------------------------------------------
# Exact U1/U2 identities; no other candidate can enter
# ---------------------------------------------------------------------------


def test_exactly_two_frozen_candidates_with_exact_identities() -> None:
    assert set(FROZEN_CANDIDATES) == {"U1", "U2"}
    assert FROZEN_CANDIDATES["U1"] == {
        "sleeve_id": "USD/CHF.OANDA__USD/JPY.OANDA",
        "formation_window": 240,
        "z_entry": Decimal("1.5"),
        "z_stop": Decimal("3.0"),
    }
    assert FROZEN_CANDIDATES["U2"] == {
        "sleeve_id": "USD/CAD.OANDA__USD/CHF.OANDA",
        "formation_window": 240,
        "z_entry": Decimal("1.5"),
        "z_stop": Decimal("3.5"),
    }


def test_require_frozen_candidate_rejects_any_other_label() -> None:
    require_frozen_candidate("U1")
    require_frozen_candidate("U2")
    for bad_label in ("U3", "u1", "USD/CAD.OANDA__USD/JPY.OANDA", ""):
        with pytest.raises(B3F1ResolutionError):
            require_frozen_candidate(bad_label)


# ---------------------------------------------------------------------------
# Leave-one-out mechanics
# ---------------------------------------------------------------------------


def test_leave_one_out_removes_exactly_one_trade_at_a_time() -> None:
    pnls = [Decimal(v) for v in (100, 100, 100, -1000)]
    result = leave_one_out(pnls)
    # Removing the one large loser gives expectancy 100 -- the maximum
    # across all leave-one-out samples, so the MINIMUM must come from
    # removing a winner instead (still includes the -1000 loser).
    assert (
        result.minimum_expectancy == (Decimal(100) + Decimal(100) - Decimal(1000)) / 3
    )
    assert result.fraction_expectancy_positive == pytest.approx(0.25)


def test_leave_one_out_requires_at_least_two_trades() -> None:
    with pytest.raises(B3F1ResolutionError):
        leave_one_out([Decimal(100)])


def test_leave_one_out_all_positive_when_every_subsample_profitable() -> None:
    pnls = [Decimal(v) for v in (10, 20, 30, 40, 50)]
    result = leave_one_out(pnls)
    assert result.fraction_expectancy_positive == 1.0
    assert result.minimum_expectancy > 0


# ---------------------------------------------------------------------------
# Deterministic bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_is_deterministic_given_the_same_seed() -> None:
    rng = np.random.default_rng(0)
    pnls = rng.normal(loc=50, scale=200, size=40)
    first = compute_bootstrap_diagnostics(
        pnls, None, kind="stationary", block_size=3, block_size_source="test"
    )
    second = compute_bootstrap_diagnostics(
        pnls, None, kind="stationary", block_size=3, block_size_source="test"
    )
    assert first.p_expectancy_gt_0 == second.p_expectancy_gt_0
    assert first.ci_lower == second.ci_lower
    assert first.ci_upper == second.ci_upper


def test_iid_bootstrap_is_a_diagnostic_distinct_from_stationary() -> None:
    rng = np.random.default_rng(1)
    pnls = rng.normal(loc=50, scale=200, size=40)
    stationary = compute_bootstrap_diagnostics(
        pnls, None, kind="stationary", block_size=5, block_size_source="test"
    )
    iid = compute_bootstrap_diagnostics(
        pnls, None, kind="iid", block_size=None, block_size_source="not_applicable"
    )
    assert stationary.bootstrap_kind == "stationary"
    assert iid.bootstrap_kind == "iid"
    assert iid.block_size is None


def test_resolve_block_size_uses_frozen_fallback_when_unstable() -> None:
    # A non-finite / non-positive estimate must trigger the frozen fallback,
    # never a value derived from this candidate's own results.
    size, source = resolve_block_size(float("nan"), 39)
    assert source == "frozen_fallback_n_cbrt"
    assert size == frozen_fallback_block_size(39)

    size2, source2 = resolve_block_size(-1.0, 39)
    assert source2 == "frozen_fallback_n_cbrt"

    size3, source3 = resolve_block_size(3.0, 39)
    assert source3 == "arch_optimal_block_length"
    assert size3 == 3


# ---------------------------------------------------------------------------
# Eligibility thresholds -- exact
# ---------------------------------------------------------------------------


def _clean_inputs(**overrides: object) -> EligibilityInputs:
    base = dict(
        original_failure_reason_is_trade_count=True,
        native_expectancy=Decimal(50),
        native_profit_factor=Decimal("1.50"),
        stressed_1_5x_expectancy=Decimal(40),
        positive_fold_count=3,
        best_5pct_removed_expectancy=Decimal(20),
        quarter_concentration=Decimal("0.30"),
        bootstrap_p_native_expectancy_gt_0=0.85,
        bootstrap_p_stressed_expectancy_gt_0=0.80,
        leave_one_out_fraction_positive=0.95,
    )
    base.update(overrides)
    return EligibilityInputs(**base)  # type: ignore[arg-type]


def test_eligible_when_every_criterion_exactly_clears_its_threshold() -> None:
    verdict = evaluate_eligibility(_clean_inputs())
    assert verdict.verdict == UNDERPOWERED_CONFIRMATION_ELIGIBLE
    assert verdict.failed_criteria == ()


def test_bootstrap_probability_exactly_at_threshold_passes() -> None:
    verdict = evaluate_eligibility(
        _clean_inputs(
            bootstrap_p_native_expectancy_gt_0=0.80,
            bootstrap_p_stressed_expectancy_gt_0=0.75,
            leave_one_out_fraction_positive=0.90,
        )
    )
    assert verdict.verdict == UNDERPOWERED_CONFIRMATION_ELIGIBLE


def test_bootstrap_probability_just_below_threshold_rejects() -> None:
    verdict = evaluate_eligibility(
        _clean_inputs(bootstrap_p_native_expectancy_gt_0=0.7999)
    )
    assert verdict.verdict == UNDERPOWERED_REJECTED
    assert "8_bootstrap_p_native_expectancy_ge_0_80" in verdict.failed_criteria


def test_pf_exactly_at_1_10_does_not_pass_strictly_greater_rule() -> None:
    verdict = evaluate_eligibility(_clean_inputs(native_profit_factor=Decimal("1.10")))
    assert verdict.verdict == UNDERPOWERED_REJECTED
    assert "3_native_pf_gt_1_10" in verdict.failed_criteria


def test_negative_economics_reject_regardless_of_power_evidence() -> None:
    verdict = evaluate_eligibility(
        _clean_inputs(
            native_expectancy=Decimal(-5),
            bootstrap_p_native_expectancy_gt_0=0.99,
            leave_one_out_fraction_positive=1.0,
        )
    )
    assert verdict.verdict == UNDERPOWERED_REJECTED
    assert "2_native_expectancy_gt_0" in verdict.failed_criteria


def test_original_failure_not_trade_count_rejects() -> None:
    verdict = evaluate_eligibility(
        _clean_inputs(original_failure_reason_is_trade_count=False)
    )
    assert verdict.verdict == UNDERPOWERED_REJECTED
    assert "1_original_failure_is_trade_count" in verdict.failed_criteria


# ---------------------------------------------------------------------------
# Mechanical winner selection
# ---------------------------------------------------------------------------


def test_select_winner_returns_none_when_no_candidate_eligible() -> None:
    assert select_winner([]) is None


def test_select_winner_returns_the_single_eligible_candidate() -> None:
    only = WinnerSelectionInputs(
        candidate_id="U1",
        bootstrap_p_stressed_expectancy_gt_0=0.80,
        bootstrap_p_native_expectancy_gt_0=0.85,
        minimum_leave_one_out_expectancy=Decimal(30),
        stressed_profit_factor=Decimal("1.3"),
    )
    assert select_winner([only]) == "U1"


def test_select_winner_prefers_higher_p_stressed_expectancy_first() -> None:
    weaker = WinnerSelectionInputs(
        candidate_id="U1",
        bootstrap_p_stressed_expectancy_gt_0=0.80,
        bootstrap_p_native_expectancy_gt_0=0.99,
        minimum_leave_one_out_expectancy=Decimal(1000),
        stressed_profit_factor=Decimal("9.0"),
    )
    stronger = WinnerSelectionInputs(
        candidate_id="U2",
        bootstrap_p_stressed_expectancy_gt_0=0.90,
        bootstrap_p_native_expectancy_gt_0=0.10,
        minimum_leave_one_out_expectancy=Decimal(1),
        stressed_profit_factor=Decimal("1.0"),
    )
    assert select_winner([weaker, stronger]) == "U2"


def test_select_winner_falls_back_to_lexicographic_tie_break() -> None:
    tied_a = WinnerSelectionInputs(
        candidate_id="U2",
        bootstrap_p_stressed_expectancy_gt_0=0.80,
        bootstrap_p_native_expectancy_gt_0=0.80,
        minimum_leave_one_out_expectancy=Decimal(30),
        stressed_profit_factor=Decimal("1.3"),
    )
    tied_b = WinnerSelectionInputs(
        candidate_id="U1",
        bootstrap_p_stressed_expectancy_gt_0=0.80,
        bootstrap_p_native_expectancy_gt_0=0.80,
        minimum_leave_one_out_expectancy=Decimal(30),
        stressed_profit_factor=Decimal("1.3"),
    )
    assert select_winner([tied_a, tied_b]) == "U1"


def test_select_winner_never_returns_more_than_one_candidate() -> None:
    both = [
        WinnerSelectionInputs(
            candidate_id=label,
            bootstrap_p_stressed_expectancy_gt_0=0.8,
            bootstrap_p_native_expectancy_gt_0=0.8,
            minimum_leave_one_out_expectancy=Decimal(10),
            stressed_profit_factor=Decimal("1.2"),
        )
        for label in ("U1", "U2")
    ]
    winner = select_winner(both)
    assert winner in ("U1", "U2")
    assert isinstance(winner, str)


# ---------------------------------------------------------------------------
# VALIDATION / holdout firewall
# ---------------------------------------------------------------------------


def test_module_never_imports_or_references_validation_or_holdout_machinery() -> None:
    import ftmoquant.research.alpha_lab.b3f1_underpowered_resolution as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    forbidden_modules = (
        "ftmoquant.data.oanda_alpha_lab_validation",
        "ftmoquant.data.canonical_source",
        "ftmoquant.research.ftmo_pass_probability.validation_diagnostic",
        "ftmoquant.research.alpha_lab.data",
        "ftmoquant.research.alpha_lab.wick_fvg_squeeze_execution",
    )
    for name in forbidden_modules:
        assert name not in imported
    for forbidden_literal in (
        "validation_canonical",
        "validation_readiness",
        "validation_scorecard",
        "VALIDATION_START",
        "HOLDOUT_START",
        "final_holdout",
    ):
        assert forbidden_literal not in source


def test_module_performs_no_data_loading() -> None:
    """This module is pure diagnostics -- it must never itself load market
    data, open a file, or touch the network; only the (separate, one-off)
    orchestration script that calls it is permitted to load DEVELOPMENT
    data."""

    import ftmoquant.research.alpha_lab.b3f1_underpowered_resolution as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    for forbidden in ("open(", "read_csv", "read_json", "requests.", "urlopen"):
        assert forbidden not in source
