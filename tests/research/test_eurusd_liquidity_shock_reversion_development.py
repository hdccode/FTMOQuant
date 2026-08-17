from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from nautilus_trader.model import Bar

from ftmoquant.backtest.execution_harness import (
    _instrument_for_profile,
    canonical_execution_profile,
)
from ftmoquant.data.dukascopy import SourceBar, _eurusd_instrument, _to_nautilus_bars
from ftmoquant.research import (
    eurusd_liquidity_shock_reversion_development as development_module,
)
from ftmoquant.research.eurusd_liquidity_shock_reversion_development import (
    EurusdLiquidityShockReversionFamily,
    EurusdLiquidityShockReversionTrialEvaluator,
    PreparedEurusdLiquidityShockMarketData,
    _build_scaled_instructions,
    _development_root,
    _write_result_artifacts,
    run_eurusd_liquidity_shock_reversion_development,
)
from ftmoquant.research.eurusd_liquidity_shock_reversion_spec import (
    EURUSD_LIQUIDITY_SHOCK_REVERSION_SEMANTIC_SHA256,
    load_eurusd_liquidity_shock_reversion_spec,
)
from ftmoquant.research.eurusd_tsm_development import _minute_execution_frames
from ftmoquant.research.g1.guards import (
    DevelopmentAccessPolicy,
    DevelopmentSearchContext,
    ResearchPartition,
    WalkForwardWindow,
)
from ftmoquant.research.g1.normalization import CompletedDailyMidpoint
from ftmoquant.research.g1.search import SearchConfig, SearchMode, run_search
from ftmoquant.research.g1.selector import select_candidate
from ftmoquant.research.ts_momentum_development import (
    load_development_evaluation_config,
)
from ftmoquant.strategies.ts_momentum import RawDirectionalTarget

START = datetime(2020, 1, 2, tzinfo=UTC)
_PRE_MINUTES = 130
_POST_MINUTES = 60


def test_sealed_paths_are_rejected_by_cli_parser() -> None:
    with pytest.raises(Exception, match="forbidden"):
        _development_root("EUR/USD.DUKASCOPY=/data/validation/catalog")
    with pytest.raises(Exception, match="forbidden"):
        _development_root("EUR/USD.DUKASCOPY=/data/final_holdout/catalog")


def test_full_synthetic_production_path_is_native_and_deterministic(
    tmp_path: Path,
) -> None:
    spec = load_eurusd_liquidity_shock_reversion_spec()
    family = EurusdLiquidityShockReversionFamily(spec)
    window = WalkForwardWindow(
        "synthetic_fold",
        START - timedelta(days=40),
        START,
        START,
        START + timedelta(minutes=_POST_MINUTES),
    )
    search_context = DevelopmentSearchContext(
        DevelopmentAccessPolicy(
            window.train_start_utc, window.evaluate_end_exclusive_utc
        ),
        (window,),
    )
    profile = canonical_execution_profile()
    instrument = _instrument_for_profile(_eurusd_instrument(), profile.fee)
    minute_bars = _minute_bars_with_shock()
    execution_frames = _minute_execution_frames(minute_bars)
    daily = tuple(
        CompletedDailyMidpoint(
            int((START - timedelta(days=31 - index)).timestamp() * 1_000_000_000),
            1.09 + (0.01 if index % 2 else 0.0) + index * 0.0001,
        )
        for index in range(31)
    )
    prepared = PreparedEurusdLiquidityShockMarketData(
        instrument=instrument,
        minute_bars=minute_bars,
        execution_frames=execution_frames,
        daily_midpoints=daily,
    )
    config = load_development_evaluation_config()
    evaluator = EurusdLiquidityShockReversionTrialEvaluator(
        prepared, config, spec, search_context=search_context
    )

    representative = family.enumerate_parameters()[0]
    assert representative == {
        "baseline_prior_returns": 30,
        "shock_multiple": 3.0,
        "hold_eligible_minutes": 5,
    }
    instructions = _build_scaled_instructions(
        family,
        representative,
        prepared.execution_frames,
        prepared.daily_midpoints,
        window,
    )
    assert instructions
    evaluation_start_ns = int(window.evaluate_start_utc.timestamp() * 1_000_000_000)
    evaluation_end_ns = int(
        window.evaluate_end_exclusive_utc.timestamp() * 1_000_000_000
    )
    assert all(
        item.execution_information_ns >= evaluation_start_ns for item in instructions
    )
    assert all(
        item.execution_information_ns < evaluation_end_ns for item in instructions
    )
    # Exactly one entry (counted) and its eventual exit/liquidation (not counted).
    counted = [item for item in instructions if item.count_alpha_transition]
    assert len(counted) == 1
    assert counted[0].raw_target is RawDirectionalTarget.SHORT

    direct = evaluator(family, representative, window)
    assert direct.fold_id == window.fold_id
    assert direct.trade_count == 1

    result = run_search(
        family=family,
        context=search_context,
        config=SearchConfig(SearchMode.EXACT_GRID, seed=0),
        evaluator=evaluator,
    )
    assert result.exact_trial_count == 36
    assert result.registry.valid_configurations == 36
    selection = select_candidate(
        family=family, registry=result.registry, policy=spec.selection_policy
    )
    summary = {
        "family_semantic_sha256": spec.semantic_sha256,
        "outcome": "ALPHA_REJECTED"
        if selection.selected_trial_id is None
        else "DEVELOPMENT_CANDIDATE_SELECTED",
        "search_landscape": development_module._search_landscape(result, selection),
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
    source = Path(
        "config/strategies/eurusd_liquidity_shock_reversion_v1.yaml"
    ).read_text()
    mismatched = tmp_path / "mismatched.yaml"
    mismatched.write_text(
        source.replace(
            EURUSD_LIQUIDITY_SHOCK_REVERSION_SEMANTIC_SHA256,
            "0" * 64,
            1,
        )
    )
    monkeypatch.setattr(development_module, "_require_clean_worktree", lambda: None)

    with pytest.raises(Exception, match="semantic"):
        run_eurusd_liquidity_shock_reversion_development(
            spec_path=mismatched,
            universe_readiness_path=tmp_path / "never-opened-readiness.json",
            development_roots={},
            evaluation_config_path=tmp_path / "never-opened-config.yaml",
            output_dir=tmp_path / "output",
        )


def test_cli_prints_summary_containing_paths_without_running_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary = {"artifact_path": tmp_path / "result.json", "outcome": "TEST"}

    def completed_run(**_: object) -> tuple[None, None, dict[str, object]]:
        return None, None, summary

    monkeypatch.setattr(
        development_module,
        "run_eurusd_liquidity_shock_reversion_development",
        completed_run,
    )
    development_module.main(
        [
            "--universe-readiness",
            str(tmp_path / "readiness.json"),
            "--development-root",
            f"EUR/USD.DUKASCOPY={tmp_path / 'development'}",
            "--development-root",
            f"GBP/USD.DUKASCOPY={tmp_path / 'development_gbp'}",
            "--evaluation-config",
            str(tmp_path / "config.json"),
            "--output",
            str(tmp_path / "output"),
        ]
    )

    assert json.loads(capsys.readouterr().out) == {
        "artifact_path": str(tmp_path / "result.json"),
        "outcome": "TEST",
    }


def test_open_development_context_rejects_non_development_partition() -> None:
    policy = DevelopmentAccessPolicy(
        START - timedelta(minutes=_PRE_MINUTES),
        START + timedelta(minutes=_POST_MINUTES),
    )
    with pytest.raises(Exception, match="sealed"):
        policy.require_interval(
            START, START + timedelta(minutes=1), partition=ResearchPartition.VALIDATION
        )


def _minute_bars_with_shock() -> tuple[Bar, ...]:
    """130 warm-up minutes of tiny alternating noise, a shock at minute 1 of
    the evaluate window, then flat prices so every grid cell's hold/exit can
    complete deterministically inside the fold.
    """

    total = _PRE_MINUTES + _POST_MINUTES
    timestamps = [
        START - timedelta(minutes=_PRE_MINUTES - index) for index in range(total)
    ]
    prices: list[Decimal] = []
    price = Decimal("1.10000")
    for index in range(total):
        if index == _PRE_MINUTES + 1:
            price += Decimal("0.02000")  # the shock
        else:
            price += Decimal("0.00002") if index % 2 else Decimal("0.00001")
        prices.append(price)
    bid_rows = [
        SourceBar(
            timestamp=timestamp,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=Decimal("1000000"),
        )
        for timestamp, price in zip(timestamps, prices, strict=True)
    ]
    ask_rows = [
        SourceBar(
            timestamp=timestamp,
            open=price + Decimal("0.00020"),
            high=price + Decimal("0.00020"),
            low=price + Decimal("0.00020"),
            close=price + Decimal("0.00020"),
            volume=Decimal("1000000"),
        )
        for timestamp, price in zip(timestamps, prices, strict=True)
    ]
    return tuple(
        sorted(
            (*_to_nautilus_bars(bid_rows, "BID"), *_to_nautilus_bars(ask_rows, "ASK")),
            key=lambda bar: (bar.ts_event, str(bar.bar_type)),
        )
    )
