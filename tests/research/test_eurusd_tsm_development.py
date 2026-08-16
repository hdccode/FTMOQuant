from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
from nautilus_trader.model import Bar

import ftmoquant.research.eurusd_tsm_development as development_module
from ftmoquant.backtest.execution_harness import (
    _instrument_for_profile,
    canonical_execution_profile,
)
from ftmoquant.data.dukascopy import SourceBar, _eurusd_instrument, _to_nautilus_bars
from ftmoquant.research.eurusd_tsm_development import (
    EquityPoint,
    EurusdTsmTrialEvaluator,
    ExecutedAlphaTransitionTracker,
    NativeFoldResult,
    PreparedEurusdTsmMarketData,
    ScaledTargetInstruction,
    _build_scaled_instructions,
    _development_root,
    _fold_metrics,
    _run_native_fold,
    _search_landscape,
    _write_result_artifacts,
    run_eurusd_tsm_development,
)
from ftmoquant.research.eurusd_tsm_spec import load_eurusd_tsm_spec
from ftmoquant.research.g1.guards import (
    DevelopmentAccessPolicy,
    DevelopmentSearchContext,
    WalkForwardWindow,
)
from ftmoquant.research.g1.normalization import CompletedDailyMidpoint
from ftmoquant.research.g1.search import SearchConfig, SearchMode, run_search
from ftmoquant.research.g1.selector import select_candidate
from ftmoquant.research.g1.trials import (
    FoldMetrics,
    TrialRegistry,
    TrialStatus,
    make_trial_record,
    summarize_trial,
)
from ftmoquant.research.ts_momentum_development import (
    _annualized_sharpe,
    _maximum_drawdown,
    load_development_evaluation_config,
)
from ftmoquant.strategies.eurusd_tsm import EurusdTsmFamily
from ftmoquant.strategies.trend_pullback import CompletedPair, PriceBar, Timeframe
from ftmoquant.strategies.ts_momentum import RawDirectionalTarget

START = datetime(2020, 1, 2, tzinfo=UTC)


def _instruction(
    raw: RawDirectionalTarget,
    units: str,
    index: int,
    *,
    count: bool = True,
) -> ScaledTargetInstruction:
    event = int((START + timedelta(minutes=index)).timestamp() * 1_000_000_000)
    return ScaledTargetInstruction(
        execution_event_ns=event,
        execution_information_ns=event + 60_000_000_000,
        decision_information_ns=event - 1,
        raw_target=raw,
        target_base_units=Decimal(units),
        execution_midpoint=Decimal("1.10010"),
        count_alpha_transition=count,
    )


def test_executed_raw_alpha_transition_semantics_are_exact() -> None:
    tracker = ExecutedAlphaTransitionTracker()

    assert tracker.observe_fill(_instruction(RawDirectionalTarget.LONG, "100", 1))
    assert tracker.count == 1  # FLAT -> LONG
    assert not tracker.observe_fill(_instruction(RawDirectionalTarget.LONG, "150", 2))
    assert tracker.count == 1  # LONG -> LONG risk resize
    assert tracker.observe_fill(_instruction(RawDirectionalTarget.FLAT, "0", 3))
    assert tracker.count == 2  # LONG -> FLAT
    assert tracker.observe_fill(_instruction(RawDirectionalTarget.SHORT, "-100", 4))
    assert tracker.count == 3  # FLAT -> SHORT

    reversal = ExecutedAlphaTransitionTracker()
    reversal.observe_fill(_instruction(RawDirectionalTarget.LONG, "100", 1))
    before = reversal.count
    reversal.observe_fill(_instruction(RawDirectionalTarget.SHORT, "-100", 2))
    assert reversal.count == before + 1  # reversal is one transition


def test_pending_supersession_and_risk_flatten_do_not_inflate_count() -> None:
    pending_then_executed = ExecutedAlphaTransitionTracker()
    # No fill while volatility is unavailable: nothing is observed by tracker.
    assert pending_then_executed.count == 0
    pending_then_executed.observe_fill(
        _instruction(RawDirectionalTarget.LONG, "100", 2)
    )
    assert pending_then_executed.count == 1

    superseded = ExecutedAlphaTransitionTracker()
    # Pending LONG is superseded before execution; only SHORT reaches a fill.
    superseded.observe_fill(_instruction(RawDirectionalTarget.SHORT, "-100", 3))
    assert superseded.count == 1
    assert superseded.expressed_raw_target is RawDirectionalTarget.SHORT

    risk_flatten = ExecutedAlphaTransitionTracker()
    risk_flatten.observe_fill(_instruction(RawDirectionalTarget.LONG, "100", 1))
    before = risk_flatten.count
    assert not risk_flatten.observe_fill(
        _instruction(RawDirectionalTarget.LONG, "0", 2)
    )
    assert risk_flatten.count == before


def test_native_execution_counts_alpha_and_all_fill_turnover_cost() -> None:
    bars = _minute_bars(8)
    profile = canonical_execution_profile()
    instrument = _instrument_for_profile(_eurusd_instrument(), profile.fee)
    config = load_development_evaluation_config()
    instructions = (
        _instruction(RawDirectionalTarget.LONG, "10000", 1),
        _instruction(RawDirectionalTarget.LONG, "15000", 2),
        _instruction(RawDirectionalTarget.FLAT, "0", 3),
        _instruction(RawDirectionalTarget.SHORT, "-10000", 4),
        _instruction(RawDirectionalTarget.SHORT, "0", 6, count=False),
    )
    start_ns = int(START.timestamp() * 1_000_000_000)

    result = _run_native_fold(
        fold_id="synthetic_fold",
        instrument=instrument,
        minute_bars=bars,
        instructions=instructions,
        config=config,
        start_ns=start_ns,
        end_exclusive_ns=start_ns + 8 * 60_000_000_000,
    )

    assert result.executed_raw_alpha_transitions == 3
    assert result.turnover_base_units == Decimal("50000.00000000")
    assert result.realized_variable_cost > 0
    assert len(result.fill_report) == 5
    assert result.equity_points[-1].equity < result.equity_points[0].equity


def test_fold_metric_expectancy_and_stress_identities_and_zero_convention() -> None:
    window = WalkForwardWindow(
        "fold",
        START - timedelta(days=30),
        START,
        START,
        START + timedelta(days=365),
    )
    native = NativeFoldResult(
        fold_id="fold",
        executed_raw_alpha_transitions=2,
        turnover_base_units=Decimal("50000"),
        realized_variable_cost=Decimal("20"),
        equity_points=(
            EquityPoint(int(START.timestamp() * 1e9), Decimal("100000")),
            EquityPoint(
                int((START + timedelta(days=1)).timestamp() * 1e9),
                Decimal("100100"),
            ),
            EquityPoint(
                int((START + timedelta(days=2)).timestamp() * 1e9),
                Decimal("100200"),
            ),
        ),
        order_report=pd.DataFrame(),
        fill_report=pd.DataFrame(),
        position_report=pd.DataFrame(),
    )

    metrics = _fold_metrics(native, window)

    assert metrics.fold_id == "fold"
    assert metrics.trade_count == 2
    assert metrics.net_expectancy == pytest.approx(0.001)
    assert metrics.cost_stressed_expectancy == pytest.approx(0.00095)
    assert metrics.turnover == pytest.approx(0.5)
    daily_returns = (0.001, 0.001)
    assert metrics.sharpe == (_annualized_sharpe(daily_returns) or 0.0)
    assert metrics.drawdown == _maximum_drawdown(daily_returns) == 0.0
    assert metrics.yearly_attribution[0].label == "2020"

    zero = _fold_metrics(
        replace(native, executed_raw_alpha_transitions=0),
        window,
    )
    assert zero.net_expectancy == zero.cost_stressed_expectancy == 0.0


def test_generic_pooled_expectancy_is_total_return_over_total_transitions() -> None:
    registry = TrialRegistry()
    folds = (
        FoldMetrics("a", 2, 0.10, 1.0, 0.1, 1.0, 0.08),
        FoldMetrics("b", 3, -0.02, -0.2, 0.2, 2.0, -0.03),
    )
    registry.add(
        make_trial_record(
            family_id="eurusd_tsm_v1",
            family_version="1.0.0",
            parameters={"x": 1},
            status=TrialStatus.COMPLETE,
            folds=folds,
        )
    )
    summary = summarize_trial(registry.records[0])

    assert summary.pooled_expectancy == pytest.approx((0.10 * 2 - 0.02 * 3) / 5)
    assert summary.cost_stressed_expectancy == pytest.approx((0.08 * 2 - 0.03 * 3) / 5)


def test_sealed_paths_are_rejected_by_cli_parser() -> None:
    with pytest.raises(Exception, match="forbidden"):
        _development_root("EUR/USD.DUKASCOPY=/data/validation/catalog")
    with pytest.raises(Exception, match="forbidden"):
        _development_root("EUR/USD.DUKASCOPY=/data/final_holdout/catalog")


def test_full_synthetic_production_path_is_native_and_deterministic(
    tmp_path: Path,
) -> None:
    spec = load_eurusd_tsm_spec()
    family = EurusdTsmFamily(spec)
    window = WalkForwardWindow(
        "synthetic_fold",
        START - timedelta(days=40),
        START,
        START,
        START + timedelta(minutes=31),
    )
    search_context = DevelopmentSearchContext(
        DevelopmentAccessPolicy(
            window.train_start_utc, window.evaluate_end_exclusive_utc
        ),
        (window,),
    )
    profile = canonical_execution_profile()
    instrument = _instrument_for_profile(_eurusd_instrument(), profile.fee)
    daily = tuple(
        CompletedDailyMidpoint(
            int((START - timedelta(days=31 - index)).timestamp() * 1_000_000_000),
            1.09 + (0.01 if index % 2 else 0.0) + index * 0.0001,
        )
        for index in range(31)
    )
    prepared = PreparedEurusdTsmMarketData(
        instrument=instrument,
        minute_bars=_minute_bars(35),
        h1_pairs=_signal_pairs(Timeframe.HOUR),
        h4_pairs=_signal_pairs(Timeframe.FOUR_HOURS),
        daily_midpoints=daily,
    )
    config = load_development_evaluation_config()
    evaluator = EurusdTsmTrialEvaluator(
        prepared, config, spec, search_context=search_context
    )

    representative = family.enumerate_parameters()[0]
    instructions = _build_scaled_instructions(
        family,
        representative,
        prepared.h1_pairs,
        prepared.daily_midpoints,
        prepared.minute_bars,
        window,
    )
    assert instructions
    evaluation_start_ns = int(window.evaluate_start_utc.timestamp() * 1_000_000_000)
    assert prepared.h1_pairs[0].info_time_ns < evaluation_start_ns
    assert all(
        item.decision_information_ns >= evaluation_start_ns for item in instructions
    )
    assert all(
        item.execution_information_ns > item.decision_information_ns
        for item in instructions
    )
    assert all(
        item.execution_information_ns
        < int(window.evaluate_end_exclusive_utc.timestamp() * 1_000_000_000)
        for item in instructions
    )
    direct = evaluator(family, representative, window)
    assert direct.fold_id == window.fold_id
    assert direct.trade_count > 0
    assert direct.net_expectancy < 0.0

    result = run_search(
        family=family,
        context=search_context,
        config=SearchConfig(SearchMode.EXACT_GRID, seed=0),
        evaluator=evaluator,
    )
    selection = select_candidate(
        family=family, registry=result.registry, policy=spec.selection_policy
    )
    assert result.exact_trial_count == 90
    assert result.registry.valid_configurations == 90
    assert selection.selected_trial_id is None
    summary = {
        "family_semantic_sha256": spec.semantic_sha256,
        "outcome": "ALPHA_REJECTED",
        "search_landscape": _search_landscape(result, selection),
    }
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_result_artifacts(first, result, selection, summary)
    _write_result_artifacts(second, result, selection, summary)
    for name in (
        "trial_registry.json",
        "development_result.json",
        "artifact_hashes.json",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_semantic_sha_mismatch_blocks_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = Path("config/strategies/eurusd_tsm_v1.yaml").read_text()
    mismatched = tmp_path / "mismatched.yaml"
    mismatched.write_text(
        source.replace(
            development_module.EURUSD_TSM_SEMANTIC_SHA256,
            "0" * 64,
            1,
        )
    )
    monkeypatch.setattr(development_module, "_require_clean_worktree", lambda: None)

    with pytest.raises(Exception, match="semantic"):
        run_eurusd_tsm_development(
            spec_path=mismatched,
            universe_readiness_path=tmp_path / "never-opened-readiness.json",
            development_roots={},
            evaluation_config_path=tmp_path / "never-opened-config.yaml",
            output_dir=tmp_path / "output",
        )


def _signal_pairs(timeframe: Timeframe) -> tuple[CompletedPair, ...]:
    information_times = [
        START - timedelta(minutes=721 - index) for index in range(721)
    ] + [START + timedelta(minutes=2 * index) for index in range(12)]
    result = []
    for index, information in enumerate(information_times):
        close = Decimal("1.05000") + Decimal(index) * Decimal("0.00001")
        if index % 2:
            close += Decimal("0.000005")
        result.append(
            CompletedPair(
                timeframe=timeframe,
                bid=PriceBar(close, close, close, close),
                ask=PriceBar(
                    close + Decimal("0.00020"),
                    close + Decimal("0.00020"),
                    close + Decimal("0.00020"),
                    close + Decimal("0.00020"),
                ),
                ts_event=int(
                    (information - timedelta(minutes=1)).timestamp() * 1_000_000_000
                ),
                info_time_ns=int(information.timestamp() * 1_000_000_000),
                contiguous=True,
            )
        )
    return tuple(result)


def _minute_bars(count: int) -> tuple[Bar, ...]:
    timestamps = [START + timedelta(minutes=index) for index in range(count)]
    bid_rows = [
        SourceBar(
            timestamp=item,
            open=Decimal("1.10000"),
            high=Decimal("1.10000"),
            low=Decimal("1.10000"),
            close=Decimal("1.10000"),
            volume=Decimal("1000000"),
        )
        for item in timestamps
    ]
    ask_rows = [
        SourceBar(
            timestamp=item,
            open=Decimal("1.10020"),
            high=Decimal("1.10020"),
            low=Decimal("1.10020"),
            close=Decimal("1.10020"),
            volume=Decimal("1000000"),
        )
        for item in timestamps
    ]
    return tuple(
        sorted(
            (*_to_nautilus_bars(bid_rows, "BID"), *_to_nautilus_bars(ask_rows, "ASK")),
            key=lambda bar: (bar.ts_event, str(bar.bar_type)),
        )
    )
