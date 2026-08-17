"""Synthetic-data-only tests for the ``eurusd_policy_rate_carry_proxy_v1``
production DEVELOPMENT evaluator.

No test in this file reads real DEVELOPMENT market data or real
DEVELOPMENT/validation/holdout returns. The one end-to-end native-engine
test (``test_native_engine_wiring_...``) exercises the full production code
path -- real ``FXRolloverInterestModule`` proxy carry accrual, real order
fills, real equity marking -- but with entirely fabricated synthetic bars
constructed in-memory, never a catalog read.
"""

from __future__ import annotations

import subprocess
from copy import deepcopy
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from ftmoquant.backtest.execution_harness import AccountParameters, InterestRateInput
from ftmoquant.data.dukascopy import SourceBar, _eurusd_instrument, _to_nautilus_bars
from ftmoquant.data.policy_rates import (
    CausalDifferentialObservation,
    DatedRate,
    PolicyRateHistory,
    PolicyRateProvenanceError,
)
from ftmoquant.research.eurusd_policy_rate_carry_proxy_development import (
    FROZEN_POLICY_RATE_CANONICAL_DATA_SHA256,
    CarryProxyInstruction,
    EurusdPolicyRateCarryProxyEvaluationError,
    _check_clean_worktree,
    _check_development_folds,
    _check_execution_profile_config,
    _check_family_identity,
    _check_output_not_exists,
    _check_partition_bounds,
    _check_policy_rate_data_hashes,
    _check_semantic_sha,
    _EquityMark,
    _midnight_utc_ns,
    _run_native_fold,
    _run_preflight_checks,
    build_carry_proxy_instructions,
    build_monthly_interest_rate_inputs,
    build_parser,
    carry_contribution_ratio,
    carry_proxy_execution_profile,
    component_contributions,
    daily_decompositions_from_equity_marks,
    decompose_daily_pnl,
    evaluate_development_gates,
    fold_metrics_from_daily_decompositions,
    run_eurusd_policy_rate_carry_proxy_development,
    stressed_daily_total_return,
    uncalibrated_baseline_execution_profile,
)
from ftmoquant.research.eurusd_policy_rate_carry_proxy_spec import (
    EURUSD_POLICY_RATE_CARRY_PROXY_SPEC_PATH,
    load_eurusd_policy_rate_carry_proxy_spec,
)
from ftmoquant.research.g1.trials import FoldMetrics
from ftmoquant.research.stage_g import VALIDATION_START
from ftmoquant.strategies.ts_momentum import RawDirectionalTarget

BASE_UNITS = Decimal("100000")


# ---------------------------------------------------------------------------
# build_monthly_interest_rate_inputs
# ---------------------------------------------------------------------------


def test_monthly_records_use_last_business_day_of_prior_month() -> None:
    history = PolicyRateHistory(
        ecbdfr=(
            DatedRate(date(2023, 11, 1), Decimal("3.50")),
            DatedRate(date(2023, 12, 29), Decimal("4.00")),
            DatedRate(date(2024, 1, 2), Decimal("4.25")),
        ),
        effr=(
            DatedRate(date(2023, 11, 1), Decimal("5.33")),
            DatedRate(date(2023, 12, 29), Decimal("5.33")),
            DatedRate(date(2024, 1, 2), Decimal("5.50")),
        ),
        canonical_data_sha256="synthetic",
    )
    records = build_monthly_interest_rate_inputs(
        history, span_start=date(2024, 1, 15), span_end=date(2024, 1, 20)
    )
    eur = {item.location: item for item in records}["EA19"]
    usd = {item.location: item for item in records}["USA"]
    assert eur.time == "2024-01"
    # Last business day of December 2023 is Fri 2023-12-29; its own 1-day
    # lag cutoff is Thu 2023-12-28. The Dec-29 and Jan-2 rows are both dated
    # strictly after that cutoff and must never be used for the 2024-01
    # record -- only the Nov-1 row (the latest one on or before Dec 28)
    # qualifies.
    assert eur.value == Decimal("3.50")
    assert usd.value == Decimal("5.33")
    assert eur.value != Decimal("4.25")
    assert usd.value != Decimal("5.50")


def test_monthly_records_never_use_the_precomputed_differential() -> None:
    history = PolicyRateHistory(
        ecbdfr=(DatedRate(date(2023, 11, 1), Decimal("4.00")),),
        effr=(DatedRate(date(2023, 11, 1), Decimal("5.33")),),
        canonical_data_sha256="synthetic",
    )
    records = build_monthly_interest_rate_inputs(
        history, span_start=date(2023, 12, 1), span_end=date(2023, 12, 31)
    )
    values = {item.location: item.value for item in records}
    # Raw absolute rates are fed in, not the -1.33 differential.
    assert values["EA19"] == Decimal("4.00")
    assert values["USA"] == Decimal("5.33")


def test_monthly_records_span_every_month_in_range() -> None:
    history = PolicyRateHistory(
        ecbdfr=(DatedRate(date(2023, 1, 1), Decimal("1.00")),),
        effr=(DatedRate(date(2023, 1, 1), Decimal("1.00")),),
        canonical_data_sha256="synthetic",
    )
    records = build_monthly_interest_rate_inputs(
        history, span_start=date(2024, 1, 15), span_end=date(2024, 3, 3)
    )
    months = sorted({item.time for item in records})
    assert months == ["2024-01", "2024-02", "2024-03"]
    assert len(records) == 6  # 3 months x 2 currencies


def test_monthly_records_reject_missing_causal_history() -> None:
    history = PolicyRateHistory(
        ecbdfr=(DatedRate(date(2024, 6, 1), Decimal("4.00")),),
        effr=(DatedRate(date(2024, 6, 1), Decimal("5.33")),),
        canonical_data_sha256="synthetic",
    )
    with pytest.raises(Exception):
        build_monthly_interest_rate_inputs(
            history, span_start=date(2024, 1, 1), span_end=date(2024, 1, 31)
        )


# ---------------------------------------------------------------------------
# P&L decomposition identity and cost stress
# ---------------------------------------------------------------------------


def test_decomposition_identity_holds_by_construction() -> None:
    decomposition = decompose_daily_pnl(
        date(2024, 1, 3),
        total_equity_change=Decimal("12.34"),
        carry_accrual=Decimal("3.21"),
        execution_cost=Decimal("0.55"),
    )
    assert (
        decomposition.spot_pnl
        + decomposition.carry_accrual
        - decomposition.execution_cost
        == decomposition.total_equity_change
    )


def test_decomposition_rejects_a_directly_constructed_inconsistent_instance() -> None:
    from ftmoquant.research.eurusd_policy_rate_carry_proxy_development import (
        DailyPnlDecomposition,
    )

    with pytest.raises(EurusdPolicyRateCarryProxyEvaluationError):
        DailyPnlDecomposition(
            as_of_date=date(2024, 1, 3),
            total_equity_change=Decimal("100"),
            carry_accrual=Decimal("1"),
            execution_cost=Decimal("1"),
            spot_pnl=Decimal("0"),  # inconsistent: 0 + 1 - 1 != 100
        )


def test_stress_leaves_carry_accrual_completely_unchanged() -> None:
    decomposition = decompose_daily_pnl(
        date(2024, 1, 3),
        total_equity_change=Decimal("10"),
        carry_accrual=Decimal("4"),
        execution_cost=Decimal("2"),
    )
    stressed = stressed_daily_total_return(decomposition)
    # spot = total(10) - carry(4) + cost(2) = 8; stressed = 8 + 4 - 1.5*2 = 9
    assert stressed == Decimal("9")
    # carry_accrual itself, as recorded on the decomposition, is untouched.
    assert decomposition.carry_accrual == Decimal("4")


def test_stress_scales_only_the_cost_component() -> None:
    zero_cost = decompose_daily_pnl(
        date(2024, 1, 3),
        total_equity_change=Decimal("5"),
        carry_accrual=Decimal("2"),
        execution_cost=Decimal("0"),
    )
    assert stressed_daily_total_return(zero_cost) == zero_cost.total_return * BASE_UNITS


# ---------------------------------------------------------------------------
# Fold metrics / sample semantics
# ---------------------------------------------------------------------------


def test_trade_count_is_the_observation_count_not_a_trade_count() -> None:
    decompositions = [
        decompose_daily_pnl(
            date(2024, 1, index + 1),
            total_equity_change=Decimal("1"),
            carry_accrual=Decimal("1"),
            execution_cost=Decimal("0"),
        )
        for index in range(7)
    ]
    metrics = fold_metrics_from_daily_decompositions("dev_fold_1", decompositions)
    assert metrics.trade_count == 7 == len(decompositions)
    # No orders/fills were ever passed to this function -- trade_count can
    # only be the observation count here.


def test_component_contributions_sum_reflects_each_bucket() -> None:
    decompositions = [
        decompose_daily_pnl(
            date(2024, 1, 1),
            total_equity_change=Decimal("10"),
            carry_accrual=Decimal("6"),
            execution_cost=Decimal("2"),
        ),
        decompose_daily_pnl(
            date(2024, 1, 2),
            total_equity_change=Decimal("4"),
            carry_accrual=Decimal("6"),
            execution_cost=Decimal("2"),
        ),
    ]
    contributions = component_contributions(decompositions)
    assert contributions["carry_accrual_contribution"] == pytest.approx(12 / 100000)
    assert contributions["execution_cost_contribution"] == pytest.approx(-4 / 100000)


# ---------------------------------------------------------------------------
# carry_contribution_ratio diagnostic (never a gate, never a selector).
# ---------------------------------------------------------------------------


def test_carry_contribution_ratio_computes_correctly_from_synthetic_components() -> (
    None
):
    decompositions = [
        decompose_daily_pnl(
            date(2024, 1, 1),
            total_equity_change=Decimal("10"),
            carry_accrual=Decimal("6"),
            execution_cost=Decimal("2"),
        ),
        decompose_daily_pnl(
            date(2024, 1, 2),
            total_equity_change=Decimal("4"),
            carry_accrual=Decimal("6"),
            execution_cost=Decimal("2"),
        ),
    ]
    # pooled spot = (10-6+2) + (4-6+2) = 6 + 0 = 6
    # pooled carry = 12, pooled cost = 4
    # ratio = 12 / (|6| + |12| + |4|) = 12 / 22
    ratio = carry_contribution_ratio(decompositions)
    assert ratio is not None
    assert ratio == pytest.approx(12 / 22)


def test_carry_contribution_ratio_is_none_when_denominator_is_zero() -> None:
    decompositions = [
        decompose_daily_pnl(
            date(2024, 1, 1),
            total_equity_change=Decimal("0"),
            carry_accrual=Decimal("0"),
            execution_cost=Decimal("0"),
        ),
    ]
    assert carry_contribution_ratio(decompositions) is None


def test_carry_contribution_ratio_rejects_empty_input() -> None:
    with pytest.raises(EurusdPolicyRateCarryProxyEvaluationError):
        carry_contribution_ratio([])


# ---------------------------------------------------------------------------
# Gate evaluation (no run_search/select_candidate/assess_plateau involved)
# ---------------------------------------------------------------------------


def _fold(fold_id: str, net: float, stressed: float, count: int) -> FoldMetrics:
    return FoldMetrics(
        fold_id=fold_id,
        trade_count=count,
        net_expectancy=net,
        sharpe=0.5,
        drawdown=0.01,
        turnover=0.0,
        cost_stressed_expectancy=stressed,
        net_daily_mean=net,
    )


def test_gates_pass_when_all_conditions_are_satisfied() -> None:
    folds = [
        _fold("dev_fold_1", 0.0002, 0.0001, 200),
        _fold("dev_fold_2", 0.0002, 0.0001, 200),
        _fold("dev_fold_3", 0.0002, 0.0001, 200),
    ]
    evaluation = evaluate_development_gates(
        "eurusd_policy_rate_carry_proxy_v1", "1.0.0", folds
    )
    assert evaluation.passed
    assert evaluation.pooled_observation_count == 600


def test_gates_fail_below_minimum_observation_count() -> None:
    folds = [
        _fold("dev_fold_1", 0.0002, 0.0001, 50),
        _fold("dev_fold_2", 0.0002, 0.0001, 50),
        _fold("dev_fold_3", 0.0002, 0.0001, 50),
    ]
    evaluation = evaluate_development_gates(
        "eurusd_policy_rate_carry_proxy_v1", "1.0.0", folds
    )
    assert not evaluation.passed
    assert not evaluation.minimum_observations_met


def test_gates_fail_when_majority_of_folds_are_negative() -> None:
    folds = [
        _fold("dev_fold_1", -0.0002, -0.0001, 200),
        _fold("dev_fold_2", -0.0002, -0.0001, 200),
        _fold("dev_fold_3", 0.0009, 0.0009, 200),
    ]
    evaluation = evaluate_development_gates(
        "eurusd_policy_rate_carry_proxy_v1", "1.0.0", folds
    )
    assert not evaluation.passed
    assert not evaluation.majority_folds_positive


def test_gates_fail_when_a_required_fold_is_missing() -> None:
    folds = [
        _fold("dev_fold_1", 0.0002, 0.0001, 300),
        _fold("dev_fold_2", 0.0002, 0.0001, 300),
    ]
    evaluation = evaluate_development_gates(
        "eurusd_policy_rate_carry_proxy_v1", "1.0.0", folds
    )
    assert not evaluation.passed
    assert not evaluation.all_folds_present


# ---------------------------------------------------------------------------
# Equity-mark bracketing -> decomposition pairing logic (pure, no engine)
# ---------------------------------------------------------------------------


def test_daily_decompositions_pair_consecutive_post_marks() -> None:
    def ns(day: int) -> int:
        stamp = datetime(2024, 1, day, 22, 0, tzinfo=UTC)
        return int(stamp.timestamp() * 1_000_000_000)

    marks = [
        _EquityMark(
            ns(8) - 60_000_000_000,
            Decimal("100000"),
            Decimal("50"),
            Decimal("100000"),
        ),
        _EquityMark(
            ns(8),
            Decimal("100010"),
            Decimal("50"),
            Decimal("100010"),
        ),  # Mon pre/post
        _EquityMark(
            ns(9) - 60_000_000_000,
            Decimal("100010"),
            Decimal("50"),
            Decimal("100010"),
        ),
        _EquityMark(
            ns(9),
            Decimal("100020"),
            Decimal("50"),
            Decimal("100020"),
        ),  # Tue pre/post
    ]
    decompositions = daily_decompositions_from_equity_marks(marks)
    assert len(decompositions) == 1
    only = decompositions[0]
    assert only.total_equity_change == Decimal("10")
    assert only.carry_accrual == Decimal("10")
    assert only.execution_cost == Decimal("0")
    assert only.spot_pnl == Decimal("0")


def test_daily_decompositions_reject_unpaired_marks() -> None:
    marks = [_EquityMark(0, Decimal("100000"), Decimal("0"), Decimal("100000"))]
    with pytest.raises(EurusdPolicyRateCarryProxyEvaluationError):
        daily_decompositions_from_equity_marks(marks)


def test_carry_isolation_uses_balance_diff_not_equity_diff_under_price_movement() -> (
    None
):
    """Section 8 fix proof: carry must isolate the native rollover cash
    event from any intervening bid/ask mark-to-market movement between the
    pre- and post-rollover marks.

    A flat-price fixture cannot distinguish the old (buggy) equity-diff
    formula from the new (fixed) balance-diff formula, because unrealized
    P&L never changes when price never moves -- that is exactly why the
    original synthetic quadrant tests, which held price flat throughout,
    never caught this. This fixture deliberately moves price between the
    pre- and post-mark of the Tuesday pair, so the two formulas would
    disagree if the bug were still present.
    """

    def ns(day: int) -> int:
        stamp = datetime(2024, 1, day, 22, 0, tzinfo=UTC)
        return int(stamp.timestamp() * 1_000_000_000)

    marks = [
        # Monday pre/post: flat, no position, establishes the anchor.
        _EquityMark(
            ns(8) - 60_000_000_000,
            Decimal("100000"),
            Decimal("50"),
            Decimal("100000"),
        ),
        _EquityMark(ns(8), Decimal("100000"), Decimal("50"), Decimal("100000")),
        # Tuesday pre: unrealized P&L is 0 relative to balance (no drift yet).
        _EquityMark(
            ns(9) - 60_000_000_000,
            Decimal("100000"),
            Decimal("50"),
            Decimal("100000"),
        ),
        # Tuesday post: balance rose by 4 (the true native rollover credit,
        # a cash event) AND price moved between pre and post, adding +50 of
        # unrealized P&L to equity that has nothing to do with rollover.
        # cumulative_cost is unchanged (50 -> 50): no fill was realized
        # inside this bracket, isolating the price-movement effect alone.
        _EquityMark(ns(9), Decimal("100054"), Decimal("50"), Decimal("100004")),
    ]
    decompositions = daily_decompositions_from_equity_marks(marks)
    assert len(decompositions) == 1
    only = decompositions[0]

    # total_equity_change is a pure post-to-post equity diff -- unaffected
    # by the carry formula, still correctly includes the full +54 move.
    assert only.total_equity_change == Decimal("54")
    # carry_accrual must be the balance-only diff (4), NOT the contaminated
    # equity diff (54) that the pre-fix formula would have produced.
    assert only.carry_accrual == Decimal("4")
    assert only.carry_accrual != only.total_equity_change
    assert only.execution_cost == Decimal("0")
    # spot_pnl (the residual) correctly absorbs the price-driven +50 move
    # that does not belong to carry: total(54) - carry(4) + cost(0) = 50.
    assert only.spot_pnl == Decimal("50")
    # Decomposition identity still holds exactly.
    assert only.spot_pnl + only.carry_accrual - only.execution_cost == (
        only.total_equity_change
    )
    # Stress still reduces to total - 0.5*cost regardless of the carry
    # formula (Section 7's algebraic invariance, reconfirmed post-fix).
    assert stressed_daily_total_return(only) == only.total_equity_change - (
        Decimal("0.5") * only.execution_cost
    )


def test_carry_isolation_rejects_a_fill_realized_inside_the_bracket() -> None:
    """If cumulative_cost changes between the pre- and post-rollover mark,
    an order fill was realized inside the bracket -- balance-diff carry
    isolation would be contaminated by that fill's own cash effect, so this
    must fail closed rather than silently misattribute."""

    def ns(day: int) -> int:
        stamp = datetime(2024, 1, day, 22, 0, tzinfo=UTC)
        return int(stamp.timestamp() * 1_000_000_000)

    marks = [
        _EquityMark(
            ns(8) - 60_000_000_000,
            Decimal("100000"),
            Decimal("50"),
            Decimal("100000"),
        ),
        _EquityMark(ns(8), Decimal("100000"), Decimal("50"), Decimal("100000")),
        # cumulative_cost changes from 50 to 55 within this single pair --
        # a fill was realized inside the pre/post bracket.
        _EquityMark(
            ns(9) - 60_000_000_000,
            Decimal("100000"),
            Decimal("50"),
            Decimal("100000"),
        ),
        _EquityMark(ns(9), Decimal("100004"), Decimal("55"), Decimal("100004")),
    ]
    with pytest.raises(EurusdPolicyRateCarryProxyEvaluationError):
        daily_decompositions_from_equity_marks(marks)


# ---------------------------------------------------------------------------
# Future-validation sample-window filtering (Section 9): warm-up
# decompositions (dated before a fold's own evaluate window) must never
# enter the sample count, metrics, or attribution.
# ---------------------------------------------------------------------------


def test_decompositions_in_evaluate_window_excludes_warmup_dates() -> None:
    from ftmoquant.research.eurusd_policy_rate_carry_proxy_development import (
        _decompositions_in_evaluate_window,
    )
    from ftmoquant.research.stage_g import frozen_development_folds

    fold = frozen_development_folds().folds[0]
    warmup_date = fold.compare_start_utc.date() - timedelta(days=5)
    in_window_date = fold.compare_start_utc.date()
    last_in_window_date = fold.compare_end_exclusive_utc.date() - timedelta(days=1)
    after_window_date = fold.compare_end_exclusive_utc.date()

    def _zero_decomposition(as_of: date) -> Any:
        return decompose_daily_pnl(
            as_of,
            total_equity_change=Decimal("0"),
            carry_accrual=Decimal("0"),
            execution_cost=Decimal("0"),
        )

    decompositions = (
        _zero_decomposition(warmup_date),
        _zero_decomposition(in_window_date),
        _zero_decomposition(last_in_window_date),
        _zero_decomposition(after_window_date),
    )
    filtered = _decompositions_in_evaluate_window(decompositions, fold)
    filtered_dates = {item.as_of_date for item in filtered}
    assert filtered_dates == {in_window_date, last_in_window_date}
    assert warmup_date not in filtered_dates
    assert after_window_date not in filtered_dates


# ---------------------------------------------------------------------------
# Real DEVELOPMENT/validation/holdout access is never authorized here.
#
# Every preflight check is independently unit-tested below with
# mocked/patched inputs (never a real dirty worktree, never real
# validation/holdout data). `run_eurusd_policy_rate_carry_proxy_development`
# itself is never actually invoked to a successful real run anywhere in
# this file -- only its preflight-gate-triggered failure paths are
# exercised, and a `--help` argparse dry run is proven not to touch any
# real data.
# ---------------------------------------------------------------------------


def test_check_semantic_sha_loads_the_real_frozen_spec() -> None:
    spec = _check_semantic_sha(EURUSD_POLICY_RATE_CARRY_PROXY_SPEC_PATH)
    assert spec.family_id == "eurusd_policy_rate_carry_proxy_v1"


def test_check_semantic_sha_fails_closed_on_a_missing_path(tmp_path: Path) -> None:
    with pytest.raises(Exception):
        _check_semantic_sha(tmp_path / "does-not-exist.yaml")


def test_check_clean_worktree_passes_when_git_reports_no_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "ftmoquant.research.eurusd_policy_rate_carry_proxy_development.subprocess.run",
        fake_run,
    )
    _check_clean_worktree()  # must not raise


def test_check_clean_worktree_fails_closed_on_any_uncommitted_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args, 0, stdout=" M some/dirty/file.py\n", stderr=""
        )

    monkeypatch.setattr(
        "ftmoquant.research.eurusd_policy_rate_carry_proxy_development.subprocess.run",
        fake_run,
    )
    with pytest.raises(EurusdPolicyRateCarryProxyEvaluationError):
        _check_clean_worktree()


def test_check_family_identity_passes_for_the_real_spec() -> None:
    _check_family_identity(load_eurusd_policy_rate_carry_proxy_spec())  # no raise


def test_check_family_identity_fails_closed_on_the_wrong_family() -> None:
    real = load_eurusd_policy_rate_carry_proxy_spec()
    from dataclasses import replace

    wrong = replace(real, family_id="some_other_family_v1")
    with pytest.raises(EurusdPolicyRateCarryProxyEvaluationError):
        _check_family_identity(wrong)


def test_check_development_folds_passes_for_the_real_spec() -> None:
    _check_development_folds(load_eurusd_policy_rate_carry_proxy_spec())  # no raise


def test_check_development_folds_fails_closed_on_a_tampered_fold_identity() -> None:
    from dataclasses import replace

    real = load_eurusd_policy_rate_carry_proxy_spec()
    document = deepcopy(real.canonical_document)
    document["development_folds"]["semantic_sha256"] = "0" * 64
    tampered = replace(real, canonical_document=document)
    with pytest.raises(EurusdPolicyRateCarryProxyEvaluationError):
        _check_development_folds(tampered)


def test_check_policy_rate_data_hashes_matches_the_pinned_audit_value() -> None:
    from ftmoquant.data.policy_rates import POLICY_RATES_DIR

    history = _check_policy_rate_data_hashes(POLICY_RATES_DIR)
    assert history.canonical_data_sha256 == FROZEN_POLICY_RATE_CANONICAL_DATA_SHA256


def test_check_policy_rate_data_hashes_fails_closed_on_a_missing_directory(
    tmp_path: Path,
) -> None:
    with pytest.raises(Exception):
        _check_policy_rate_data_hashes(tmp_path / "does-not-exist")


def test_check_execution_profile_config_passes_for_the_frozen_proxy_wiring() -> None:
    _check_execution_profile_config()  # no raise


def test_check_execution_profile_config_fails_closed_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ftmoquant.backtest.execution_harness import RolloverConfig, RolloverMode

    module = "ftmoquant.research.eurusd_policy_rate_carry_proxy_development"

    def fake_profile(*args: Any, **kwargs: Any) -> Any:
        base = uncalibrated_baseline_execution_profile()
        from dataclasses import replace

        return replace(base, rollover=RolloverConfig(mode=RolloverMode.DISABLED))

    monkeypatch.setattr(f"{module}.carry_proxy_execution_profile", fake_profile)
    with pytest.raises(EurusdPolicyRateCarryProxyEvaluationError):
        _check_execution_profile_config()


def test_check_output_not_exists_passes_on_a_nonexistent_directory(
    tmp_path: Path,
) -> None:
    _check_output_not_exists(tmp_path / "brand-new-output")  # no raise


def test_check_output_not_exists_passes_on_an_empty_existing_directory(
    tmp_path: Path,
) -> None:
    output = tmp_path / "empty-output"
    output.mkdir()
    _check_output_not_exists(output)  # no raise


def test_check_output_not_exists_fails_closed_on_a_populated_directory(
    tmp_path: Path,
) -> None:
    output = tmp_path / "populated-output"
    output.mkdir()
    (output / "development_result.json").write_text("{}")
    with pytest.raises(EurusdPolicyRateCarryProxyEvaluationError):
        _check_output_not_exists(output)


def test_check_partition_bounds_passes_for_the_real_spec_and_safe_paths(
    tmp_path: Path,
) -> None:
    _check_partition_bounds(
        load_eurusd_policy_rate_carry_proxy_spec(),
        universe_readiness_path=tmp_path / "readiness.json",
        development_root=tmp_path / "development",
        policy_rates_dir=Path("config/data/policy_rates"),
        output_dir=tmp_path / "output",
    )  # no raise


def test_check_partition_bounds_fails_closed_on_a_tampered_development_end(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    real = load_eurusd_policy_rate_carry_proxy_spec()
    document = deepcopy(real.canonical_document)
    # Push DEVELOPMENT's own declared end past VALIDATION_START.
    past_validation = VALIDATION_START + timedelta(days=400)
    document["dataset"]["development_end_exclusive_utc"] = (
        past_validation.isoformat().replace("+00:00", "Z")
    )
    tampered = replace(real, canonical_document=document)
    with pytest.raises(EurusdPolicyRateCarryProxyEvaluationError):
        _check_partition_bounds(
            tampered,
            universe_readiness_path=tmp_path / "readiness.json",
            development_root=tmp_path / "development",
            policy_rates_dir=Path("config/data/policy_rates"),
            output_dir=tmp_path / "output",
        )


@pytest.mark.parametrize(
    "bad_path",
    [
        Path("some/validation/root"),
        Path("some/HOLDOUT/root"),
        Path("final_holdout/data"),
    ],
)
def test_check_partition_bounds_fails_closed_on_a_sealed_path_marker(
    tmp_path: Path, bad_path: Path
) -> None:
    with pytest.raises(EurusdPolicyRateCarryProxyEvaluationError):
        _check_partition_bounds(
            load_eurusd_policy_rate_carry_proxy_spec(),
            universe_readiness_path=tmp_path / "readiness.json",
            development_root=bad_path,
            policy_rates_dir=Path("config/data/policy_rates"),
            output_dir=tmp_path / "output",
        )


def test_run_preflight_checks_fails_closed_on_the_first_bad_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With a clean (mocked) worktree, a bogus policy-rates dir must still

    fail closed with this family's own error type -- propagated from
    ``load_policy_rate_history`` inside ``_check_policy_rate_data_hashes``.
    """

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "ftmoquant.research.eurusd_policy_rate_carry_proxy_development.subprocess.run",
        fake_run,
    )
    with pytest.raises(PolicyRateProvenanceError):
        _run_preflight_checks(
            spec_path=EURUSD_POLICY_RATE_CARRY_PROXY_SPEC_PATH,
            universe_readiness_path=tmp_path / "readiness.json",
            development_root=tmp_path / "development",
            policy_rates_dir=tmp_path / "does-not-exist",
            output_dir=tmp_path / "output",
        )


def test_run_preflight_checks_fails_closed_on_an_already_populated_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "ftmoquant.research.eurusd_policy_rate_carry_proxy_development.subprocess.run",
        fake_run,
    )
    output = tmp_path / "output"
    output.mkdir()
    (output / "development_result.json").write_text("{}")
    with pytest.raises(EurusdPolicyRateCarryProxyEvaluationError):
        _run_preflight_checks(
            spec_path=EURUSD_POLICY_RATE_CARRY_PROXY_SPEC_PATH,
            universe_readiness_path=tmp_path / "readiness.json",
            development_root=tmp_path / "development",
            policy_rates_dir=Path("config/data/policy_rates"),
            output_dir=output,
        )


def test_runner_fails_closed_before_touching_any_catalog(tmp_path: Path) -> None:
    """No matter the current worktree state, this must raise this family's

    own error type (dirty-worktree check) or PolicyRateProvenanceError
    (bogus policy-rates dir) -- never proceed to open a real catalog.
    """

    with pytest.raises((EurusdPolicyRateCarryProxyEvaluationError, ValueError)):
        run_eurusd_policy_rate_carry_proxy_development(
            spec_path=EURUSD_POLICY_RATE_CARRY_PROXY_SPEC_PATH,
            universe_readiness_path=Path("does-not-matter"),
            development_root=Path("does-not-matter"),
            policy_rates_dir=Path("does-not-matter"),
            output_dir=Path("does-not-matter"),
        )


def test_cli_help_dry_run_never_touches_any_real_data() -> None:
    """Argparse's own ``--help`` exits before the runner is ever called."""

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])


# ---------------------------------------------------------------------------
# Full native-engine synthetic end-to-end wiring test.
# ---------------------------------------------------------------------------


def _utc(day: int, hour: int, minute: int) -> datetime:
    return datetime(2024, 1, day, hour, minute, tzinfo=UTC)


def _run_synthetic_carry_quadrant(
    *, raw_target: RawDirectionalTarget, eur_rate: Decimal, usd_rate: Decimal
) -> tuple[Any, ...]:
    """Run a tiny real BacktestEngine fold with entirely fabricated bars.

    Shared helper for the 4-quadrant long/short x EUR>USD/USD>EUR carry sign
    sanity proof. Isolates only what varies across quadrants (the raw
    directional target and the two raw rates fed into the real
    FXRolloverInterestModule via InterestRateInput) -- everything else
    (bars, timing, account, instrument) is identical across all four calls
    so cross-quadrant magnitude comparisons are meaningful.
    """

    weekdays = (8, 9, 10, 11, 12)  # Mon-Fri, Jan 2024, EST (no DST ambiguity)
    bid_timestamps = [_utc(8, 0, 1)]
    for day in weekdays:
        bid_timestamps += [_utc(day, 21, 58), _utc(day, 21, 59)]
    ask_timestamps = bid_timestamps

    bid_price = Decimal("1.10000")
    ask_price = Decimal("1.10020")
    volume = Decimal("1")
    bid_rows = [
        SourceBar(ts, bid_price, bid_price, bid_price, bid_price, volume)
        for ts in bid_timestamps
    ]
    ask_rows = [
        SourceBar(ts, ask_price, ask_price, ask_price, ask_price, volume)
        for ts in ask_timestamps
    ]
    minute_bars = [
        *_to_nautilus_bars(bid_rows, "BID"),
        *_to_nautilus_bars(ask_rows, "ASK"),
    ]

    entry_event_ns = int(_utc(8, 0, 1).timestamp() * 1_000_000_000)
    entry_info_ns = int(_utc(8, 0, 2).timestamp() * 1_000_000_000)
    instruction = CarryProxyInstruction(
        decision_information_ns=int(_utc(8, 0, 0).timestamp() * 1_000_000_000),
        execution_event_ns=entry_event_ns,
        execution_information_ns=entry_info_ns,
        raw_target=raw_target,
        target_base_units=(
            BASE_UNITS if raw_target is RawDirectionalTarget.LONG else -BASE_UNITS
        ),
        execution_midpoint=Decimal("1.10010"),
        as_of_date=date(2024, 1, 8),
    )

    profile = carry_proxy_execution_profile(
        uncalibrated_baseline_execution_profile(),
        records=(
            InterestRateInput("EA19", "2024-01", eur_rate),
            InterestRateInput("USA", "2024-01", usd_rate),
        ),
    )
    account = AccountParameters(
        initial_capital=Decimal("100000"), currency="USD", leverage=Decimal("20")
    )

    return _run_native_fold(
        fold_id="synthetic_fold",
        instrument=_eurusd_instrument(),
        minute_bars=minute_bars,
        instructions=(instruction,),
        account=account,
        profile=profile,
        start_ns=int(_utc(8, 0, 0).timestamp() * 1_000_000_000),
        end_exclusive_ns=int(
            datetime(2024, 1, 13, tzinfo=UTC).timestamp() * 1_000_000_000
        ),
    )


def _assert_clean_quadrant(decompositions: tuple[Any, ...]) -> None:
    """Shared structural assertions (dates, zero cost/spot) for every quadrant."""

    assert len(decompositions) == 4  # Tue, Wed, Thu, Fri (Mon has no prior post-mark)
    tuesday, wednesday, thursday, friday = decompositions
    assert tuesday.as_of_date == date(2024, 1, 9)
    assert wednesday.as_of_date == date(2024, 1, 10)
    assert thursday.as_of_date == date(2024, 1, 11)
    assert friday.as_of_date == date(2024, 1, 12)
    for item in decompositions:
        # No fills after Monday's entry, flat prices -> spot P&L ~ 0.
        assert item.execution_cost == Decimal("0")
        assert abs(item.spot_pnl) < Decimal("0.01")
        assert item.total_equity_change == item.carry_accrual


def test_native_engine_wiring_proxy_carry_accrues_wed_and_fri_tripled() -> None:
    """Quadrant 1: EUR(4%) > USD(1%), LONG -> positive carry every day.

    Verifies: real FXRolloverInterestModule accrual with a positive
    EUR-minus-USD differential produces positive carry for a long position;
    Wed/Fri accrual is triple a regular weekday's (the module's own frozen
    convention -- not recomputed here); with flat prices and no fills after
    entry, spot_pnl and execution_cost are ~0 on the held days, so
    total_equity_change == carry_accrual on those days, an end-to-end proof
    of the decomposition identity against real engine-produced numbers.
    """

    decompositions = _run_synthetic_carry_quadrant(
        raw_target=RawDirectionalTarget.LONG,
        eur_rate=Decimal("4"),
        usd_rate=Decimal("1"),
    )
    _assert_clean_quadrant(decompositions)
    tuesday, wednesday, thursday, friday = decompositions

    # Positive EUR-minus-USD differential on a long position -> positive carry,
    # economically signed entirely by the native module, no second calculation.
    for item in decompositions:
        assert item.carry_accrual > 0

    # Wed/Fri are tripled relative to Tue/Thu -- the module's own frozen
    # weekend/T+2 convention, not recomputed here.
    assert float(wednesday.carry_accrual) == pytest.approx(
        float(tuesday.carry_accrual) * 3, rel=1e-3
    )
    assert float(friday.carry_accrual) == pytest.approx(
        float(thursday.carry_accrual) * 3, rel=1e-3
    )
    assert float(tuesday.carry_accrual) == pytest.approx(
        float(thursday.carry_accrual), rel=1e-3
    )


def test_native_engine_wiring_eur_gt_usd_short_produces_negative_carry() -> None:
    """Quadrant 2: EUR(4%) > USD(1%), SHORT -> negative carry every day."""

    decompositions = _run_synthetic_carry_quadrant(
        raw_target=RawDirectionalTarget.SHORT,
        eur_rate=Decimal("4"),
        usd_rate=Decimal("1"),
    )
    _assert_clean_quadrant(decompositions)
    for item in decompositions:
        assert item.carry_accrual < 0


def test_native_engine_wiring_usd_gt_eur_long_produces_negative_carry() -> None:
    """Quadrant 3: USD(4%) > EUR(1%), LONG -> negative carry every day."""

    decompositions = _run_synthetic_carry_quadrant(
        raw_target=RawDirectionalTarget.LONG,
        eur_rate=Decimal("1"),
        usd_rate=Decimal("4"),
    )
    _assert_clean_quadrant(decompositions)
    for item in decompositions:
        assert item.carry_accrual < 0


def test_native_engine_wiring_usd_gt_eur_short_produces_positive_carry() -> None:
    """Quadrant 4: USD(4%) > EUR(1%), SHORT -> positive carry every day.

    Also proves magnitude symmetry against quadrant 1
    (EUR(4%) > USD(1%), LONG): both quadrants have the same |rate
    differential| (|4-1| = 3 in both directions) applied to the same
    absolute exposure, so the daily carry magnitudes should match.
    """

    decompositions = _run_synthetic_carry_quadrant(
        raw_target=RawDirectionalTarget.SHORT,
        eur_rate=Decimal("1"),
        usd_rate=Decimal("4"),
    )
    _assert_clean_quadrant(decompositions)
    for item in decompositions:
        assert item.carry_accrual > 0

    quadrant_1 = _run_synthetic_carry_quadrant(
        raw_target=RawDirectionalTarget.LONG,
        eur_rate=Decimal("4"),
        usd_rate=Decimal("1"),
    )
    for mirrored, reference in zip(decompositions, quadrant_1, strict=True):
        assert float(mirrored.carry_accrual) == pytest.approx(
            float(reference.carry_accrual), rel=1e-3
        )


# ---------------------------------------------------------------------------
# build_carry_proxy_instructions -- pending-target supersession on market
# closures (weekends, weekday holidays, arbitrary multi-day data gaps).
#
# Root cause reproduced against the real frozen dev_fold_1 catalog metadata
# (timestamps only, no price returns or P&L inspected): the first real
# collision was the Saturday 2020-04-11 and Sunday 2020-04-12 00:00 UTC
# decisions, both resolving to the execution frame at 2020-04-12T21:00:00Z.
# These tests reproduce the same mechanism with fully synthetic, minimal
# frame/decision sets so the fix is proven independent of any real market
# data.
# ---------------------------------------------------------------------------


def _obs(as_of: date, differential: str) -> CausalDifferentialObservation:
    return CausalDifferentialObservation(
        as_of_date=as_of,
        differential_percent=Decimal(differential),
        ecbdfr_observation_date=as_of - timedelta(days=1),
        effr_observation_date=as_of - timedelta(days=1),
    )


def _frame(event: datetime, info: datetime, mid: str) -> tuple[int, int, Decimal]:
    return (
        int(event.timestamp() * 1_000_000_000),
        int(info.timestamp() * 1_000_000_000),
        Decimal(mid),
    )


def test_weekend_collision_latest_causal_target_wins() -> None:
    """Section 7.A: Fri/Sat/Sun/Mon decisions, market closed Sat-Sun.

    Only Friday and Monday have executable frames. Saturday's and Sunday's
    decisions -- and Monday's own decision -- all resolve to the same
    Monday reopen frame. Exactly one instruction must be emitted there,
    carrying the latest (Monday) decision; Friday keeps its own separate,
    uncollided instruction.
    """

    friday = date(2024, 1, 5)
    saturday = date(2024, 1, 6)
    sunday = date(2024, 1, 7)
    monday = date(2024, 1, 8)

    frames = (
        _frame(
            datetime(2024, 1, 5, 12, 0, tzinfo=UTC),
            datetime(2024, 1, 5, 12, 1, tzinfo=UTC),
            "1.10000",
        ),
        _frame(
            datetime(2024, 1, 8, 22, 0, tzinfo=UTC),
            datetime(2024, 1, 8, 22, 1, tzinfo=UTC),
            "1.10500",
        ),
    )
    differentials = (
        _obs(friday, "1"),
        _obs(saturday, "1"),
        _obs(sunday, "1"),
        _obs(monday, "1"),
    )

    instructions = build_carry_proxy_instructions(
        minute_execution_frames=frames,
        daily_midpoints=(),
        differentials=differentials,
        evaluate_start_ns=_midnight_utc_ns(friday),
        evaluate_end_exclusive_ns=_midnight_utc_ns(monday) + 1,
    )

    assert len(instructions) == 2
    assert len({item.execution_event_ns for item in instructions}) == 2

    friday_instruction = next(
        item for item in instructions if item.as_of_date == friday
    )
    assert friday_instruction.decision_information_ns == _midnight_utc_ns(friday)

    monday_frame_instructions = [
        item for item in instructions if item.as_of_date != friday
    ]
    assert len(monday_frame_instructions) == 1
    winner = monday_frame_instructions[0]
    # Monday's decision (the latest of Sat/Sun/Mon) must be the survivor,
    # not Saturday's or Sunday's stale, superseded decision.
    assert winner.as_of_date == monday
    assert winner.decision_information_ns == _midnight_utc_ns(monday)
    assert winner.execution_event_ns == int(
        datetime(2024, 1, 8, 22, 0, tzinfo=UTC).timestamp() * 1_000_000_000
    )


def test_weekend_collision_changing_target_newer_wins_unequivocally() -> None:
    """Section 7.B: the superseded and surviving decisions have opposite
    signed targets -- proves supersession picks the newer target, not
    merely a timestamp label with the same direction throughout."""

    saturday = date(2024, 1, 6)
    sunday = date(2024, 1, 7)
    monday = date(2024, 1, 8)

    frames = (
        _frame(
            datetime(2024, 1, 8, 22, 0, tzinfo=UTC),
            datetime(2024, 1, 8, 22, 1, tzinfo=UTC),
            "1.10500",
        ),
    )
    # Saturday: negative differential (SHORT). Monday: positive (LONG).
    # Sunday sits in between and is also stale by the time Monday is known.
    differentials = (
        _obs(saturday, "-1"),
        _obs(sunday, "-1"),
        _obs(monday, "1"),
    )

    instructions = build_carry_proxy_instructions(
        minute_execution_frames=frames,
        daily_midpoints=(),
        differentials=differentials,
        evaluate_start_ns=_midnight_utc_ns(saturday),
        evaluate_end_exclusive_ns=_midnight_utc_ns(monday) + 1,
    )

    assert len(instructions) == 1
    winner = instructions[0]
    assert winner.as_of_date == monday
    assert winner.raw_target == RawDirectionalTarget.LONG
    assert winner.raw_target != RawDirectionalTarget.SHORT


def test_weekday_holiday_multiday_gap_collision_not_special_cased_to_weekends() -> (
    None
):
    """Section 7.C: an artificial *midweek* gap (Tue/Wed, ordinary weekdays)
    produces the identical collision-resolution behavior as the weekend
    case, proving the fix is driven by executable-frame availability, not
    a hardcoded ``weekday() < 5`` branch."""

    monday = date(2024, 1, 8)
    tuesday = date(2024, 1, 9)
    wednesday = date(2024, 1, 10)
    thursday = date(2024, 1, 11)

    frames = (
        _frame(
            datetime(2024, 1, 8, 12, 0, tzinfo=UTC),
            datetime(2024, 1, 8, 12, 1, tzinfo=UTC),
            "1.10000",
        ),
        _frame(
            datetime(2024, 1, 11, 12, 0, tzinfo=UTC),
            datetime(2024, 1, 11, 12, 1, tzinfo=UTC),
            "1.10800",
        ),
    )
    differentials = (
        _obs(monday, "1"),
        _obs(tuesday, "1"),
        _obs(wednesday, "1"),
        _obs(thursday, "1"),
    )

    instructions = build_carry_proxy_instructions(
        minute_execution_frames=frames,
        daily_midpoints=(),
        differentials=differentials,
        evaluate_start_ns=_midnight_utc_ns(monday),
        evaluate_end_exclusive_ns=_midnight_utc_ns(thursday) + 1,
    )

    assert len(instructions) == 2
    thursday_frame_instructions = [
        item for item in instructions if item.as_of_date != monday
    ]
    assert len(thursday_frame_instructions) == 1
    assert thursday_frame_instructions[0].as_of_date == thursday
    assert thursday_frame_instructions[0].decision_information_ns == _midnight_utc_ns(
        thursday
    )


def test_causality_boundary_decision_at_or_after_frame_cannot_influence_it() -> None:
    """Section 7.D: a decision known exactly at (not strictly before) a
    frame's information time must not be resolved to that frame -- it must
    roll forward to the next executable frame, and every emitted
    instruction must satisfy decision_information_ns < execution_information_ns."""

    day_one = date(2024, 1, 8)
    day_two = date(2024, 1, 9)

    frame_one_info = datetime(2024, 1, 9, 0, 0, tzinfo=UTC)  # == day_two's decision_ns
    frame_two_info = datetime(2024, 1, 9, 12, 0, tzinfo=UTC)

    frames = (
        _frame(frame_one_info, frame_one_info, "1.10000"),
        _frame(frame_two_info, frame_two_info, "1.10100"),
    )
    differentials = (_obs(day_one, "1"), _obs(day_two, "1"))

    instructions = build_carry_proxy_instructions(
        minute_execution_frames=frames,
        daily_midpoints=(),
        differentials=differentials,
        evaluate_start_ns=_midnight_utc_ns(day_one),
        evaluate_end_exclusive_ns=_midnight_utc_ns(day_two) + 1,
    )

    for item in instructions:
        assert item.decision_information_ns < item.execution_information_ns

    # day_one's decision (info time == frame_one_info) is NOT strictly
    # before frame_one, so it must resolve to frame_one only via the
    # *equality* boundary being excluded -- i.e. day_one's own decision_ns
    # (day_one 00:00) is strictly before frame_one_info (day_two 00:00),
    # so it correctly lands on frame_one.
    day_one_instruction = next(i for i in instructions if i.as_of_date == day_one)
    assert day_one_instruction.execution_event_ns == int(
        frame_one_info.timestamp() * 1_000_000_000
    )
    # day_two's decision_ns equals frame_one's information time exactly,
    # so frame_one is NOT strictly after it -- it must roll forward to
    # frame_two instead of colliding with (or wrongly preceding) frame_one.
    day_two_instruction = next(i for i in instructions if i.as_of_date == day_two)
    assert day_two_instruction.execution_event_ns == int(
        frame_two_info.timestamp() * 1_000_000_000
    )


def test_no_collision_regression_ordinary_daily_decisions_unchanged() -> None:
    """Section 7.E: with a continuously tradable market (a distinct frame
    strictly after every decision), behavior is unchanged: one instruction
    per decision, each carrying its own day's decision_information_ns."""

    days = [date(2024, 1, 8) + timedelta(days=offset) for offset in range(4)]
    frames = tuple(
        _frame(
            datetime.combine(day, datetime.min.time(), tzinfo=UTC)
            + timedelta(hours=12),
            datetime.combine(day, datetime.min.time(), tzinfo=UTC)
            + timedelta(hours=12, minutes=1),
            "1.10000",
        )
        for day in days
    )
    differentials = tuple(_obs(day, "1") for day in days)

    instructions = build_carry_proxy_instructions(
        minute_execution_frames=frames,
        daily_midpoints=(),
        differentials=differentials,
        evaluate_start_ns=_midnight_utc_ns(days[0]),
        evaluate_end_exclusive_ns=_midnight_utc_ns(days[-1]) + 1,
    )

    assert len(instructions) == len(days)
    assert len({item.execution_event_ns for item in instructions}) == len(days)
    for day, instruction in zip(days, instructions, strict=True):
        assert instruction.as_of_date == day
        assert instruction.decision_information_ns == _midnight_utc_ns(day)
