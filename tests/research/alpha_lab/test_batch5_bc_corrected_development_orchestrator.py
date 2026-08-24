from __future__ import annotations

import ast
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

import ftmoquant.research.alpha_lab.batch5_bc_corrected_development_orchestrator as mod
from ftmoquant.research.alpha_lab.batch5_daily import (
    CompletedFxDay,
    FxDayBuildDiagnostics,
    ny_fx_boundary,
)
from ftmoquant.research.alpha_lab.batch5_development_scorecard import (
    DevelopmentSleeveInput,
    evaluate_development_sleeve,
)
from ftmoquant.research.alpha_lab.batch5_execution import Batch5TradeResult
from ftmoquant.research.alpha_lab.batch5_preregistration import (
    B5C_INSTRUMENTS,
    FAMILY_B5B,
    FAMILY_B5C,
)
from ftmoquant.research.alpha_lab.batch5_screen import FrequencyStats
from ftmoquant.research.alpha_lab.batch5c_daily_reversal_signals import (
    build_next_day_intents,
    generate_events,
)


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    timestamps = []
    closes = []
    start = date(2019, 3, 1)
    for index in range(40):
        boundary = ny_fx_boundary(start + timedelta(days=index))
        timestamps.extend(
            [boundary - timedelta(minutes=1), boundary + timedelta(minutes=1)]
        )
        closes.extend([1.0 + (index % 2) * 0.01, 1.0])
    frame_index = pd.DatetimeIndex(timestamps)
    bid = pd.DataFrame(
        {name: closes for name in ("open", "high", "low", "close")},
        index=frame_index,
    )
    return bid, bid + 0.0002


class _Cache:
    def __init__(self) -> None:
        self.pair = _frames()

    def frames(
        self, instrument_id: str, multiplier: Decimal
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        assert instrument_id in {"AUD/CAD.OANDA", "USD/CAD.OANDA"}
        assert multiplier in mod.COST_MULTIPLIERS
        return self.pair


class _WarmupCache:
    def __init__(self) -> None:
        timestamps = []
        closes = []
        start = date(2019, 1, 28)
        for index in range(100):
            boundary = ny_fx_boundary(start + timedelta(days=index))
            timestamps.extend(
                [boundary - timedelta(minutes=1), boundary + timedelta(minutes=1)]
            )
            closes.extend([1.0 + (index % 2) * 0.01, 1.0])
        full_index = pd.DatetimeIndex(timestamps)
        bid = pd.DataFrame(
            {name: closes for name in ("open", "high", "low", "close")},
            index=full_index,
        )
        self.cross = (bid, bid + 0.0002)
        development_index = full_index[
            full_index >= pd.Timestamp(mod.DEVELOPMENT_START)
        ]
        self.conversion = (
            bid.loc[development_index],
            (bid + 0.0002).loc[development_index],
        )

    def frames(
        self, instrument_id: str, multiplier: Decimal
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        assert multiplier in mod.COST_MULTIPLIERS
        if instrument_id in {"AUD/CAD.OANDA", "EUR/JPY.OANDA"}:
            return self.cross
        assert instrument_id in {"USD/CAD.OANDA", "USD/JPY.OANDA"}
        return self.conversion


def test_corrected_cli_has_no_strategy_or_parameter_overrides() -> None:
    parser = mod.build_parser()
    options = {action.dest for action in parser._actions} - {"help"}
    assert options == {
        "development_root",
        "universe_readiness",
        "batch5_cross_root",
        "output",
    }


def test_existing_output_refusal_precedes_preflight_and_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    called = False

    def forbidden(**_: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("preflight must not run")

    monkeypatch.setattr(mod, "verify_corrected_preflight", forbidden)
    with pytest.raises(mod.Batch5DevelopmentOrchestratorError, match="overwrite"):
        mod.run_corrected_development(
            development_root=tmp_path / "canonical",
            universe_readiness=tmp_path / "readiness.json",
            batch5_cross_root=tmp_path / "crosses",
            output_dir=output,
        )
    assert called is False


@pytest.mark.parametrize("token", ["validation", "holdout", "final_holdout"])
def test_forbidden_roots_rejected_before_read(
    tmp_path: Path, token: str
) -> None:
    with pytest.raises(mod.Batch5DevelopmentOrchestratorError, match="forbidden"):
        mod.verify_corrected_preflight(
            development_root=tmp_path / token,
            universe_readiness=tmp_path / "missing.json",
            batch5_cross_root=tmp_path / "crosses",
        )


def test_b5b_price_stream_generated_even_when_rollover_is_unsupported() -> None:
    sleeve, diagnostics = mod.run_corrected_b5b(_Cache())  # type: ignore[arg-type]
    assert diagnostics["days_with_20_close_statistic"] == 20
    assert diagnostics["non_flat_signal_count"] == 20
    assert diagnostics["intended_trade_count"] > 0
    assert diagnostics["completed_trade_count"] > 0
    assert diagnostics["rollover_supported"] is False
    assert sleeve.frequency.rollover_supported is False
    assert sleeve.frequency.position_sign_change_count > 0


def test_b5b_warmup_forms_indicators_but_never_reaches_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _WarmupCache()
    submitted = []
    execute = mod.execute_positions

    def traced(signals, **kwargs):  # type: ignore[no-untyped-def]
        submitted.extend(signals)
        return execute(signals, **kwargs)

    monkeypatch.setattr(mod, "execute_positions", traced)
    sleeve, diagnostics = mod.run_corrected_b5b(cache)  # type: ignore[arg-type]
    assert diagnostics["warmup_signal_count"] > 0
    assert diagnostics["days_with_20_close_statistic"] > diagnostics[
        "executable_signal_count"
    ]
    assert submitted
    assert all(row.signal_timestamp >= mod.DEVELOPMENT_START for row in submitted)
    assert cache.conversion[0].index.min() >= pd.Timestamp(mod.DEVELOPMENT_START)
    assert sleeve.native_trades
    first = sleeve.native_trades[0]
    assert first.signal_timestamp >= mod.DEVELOPMENT_START
    assert first.actual_entry_timestamp > first.signal_timestamp


def _event_day(local_date: date, value: str) -> CompletedFxDay:
    return CompletedFxDay(
        "EUR/JPY.OANDA",
        local_date,
        ny_fx_boundary(local_date - timedelta(days=1)),
        ny_fx_boundary(local_date),
        Decimal(100),
        Decimal(100) * (Decimal(1) + Decimal(value)),
    )


def test_b5c_warmup_returns_remain_but_predevelopment_event_is_not_submitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = date(2019, 2, 8)
    days = [
        _event_day(
            start + timedelta(days=index),
            "0.01" if index % 2 else "-0.01",
        )
        for index in range(30)
    ]
    days.extend(
        [
            _event_day(start + timedelta(days=30), "0.04"),
            _event_day(start + timedelta(days=31), "-0.08"),
            _event_day(start + timedelta(days=32), "0"),
        ]
    )
    all_events = generate_events(days)
    assert any(event.signal_timestamp < mod.DEVELOPMENT_START for event in all_events)
    executable = mod.partition_development_signals(all_events)
    assert executable
    assert all(event.signal_timestamp >= mod.DEVELOPMENT_START for event in executable)
    first = executable[0]
    prior = days[first.event_day_index - 30 : first.event_day_index]
    assert any(day.end_utc < mod.DEVELOPMENT_START for day in prior)
    intents = build_next_day_intents(executable, {"EUR/JPY.OANDA": days})
    assert intents
    assert all(intent.signal_timestamp >= mod.DEVELOPMENT_START for intent in intents)

    submitted = []

    def completed_days(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return tuple(days), FxDayBuildDiagnostics(
            len(days),
            len(days),
            len(days),
            0,
            0,
            0,
            days[0].local_close_date,
            days[-1].local_close_date,
        )

    def traced(events, **_kwargs):  # type: ignore[no-untyped-def]
        submitted.extend(events)
        return ()

    monkeypatch.setattr(mod, "B5C_INSTRUMENTS", ("EUR/JPY.OANDA",))
    monkeypatch.setattr(mod, "build_completed_fx_days_with_diagnostics", completed_days)
    monkeypatch.setattr(mod, "execute_events", traced)
    cache = _WarmupCache()
    mod.run_corrected_b5c(cache)  # type: ignore[arg-type]
    assert submitted
    assert all(event.signal_timestamp >= mod.DEVELOPMENT_START for event in submitted)
    assert cache.conversion[0].index.min() >= pd.Timestamp(mod.DEVELOPMENT_START)


def test_post_execution_trade_filter_remains_a_second_partition_firewall() -> None:
    escaped = Batch5TradeResult(
        FAMILY_B5B,
        "B5B_FROZEN_DIRECT_AUDCAD_MR",
        "B5B_AUDCAD",
        "AUD/CAD.OANDA",
        mod.DEVELOPMENT_START,
        mod.DEVELOPMENT_START + timedelta(minutes=1),
        mod.DEVELOPMENT_END_EXCLUSIVE,
        "BUY",
        Decimal(1),
        Decimal(1),
        Decimal(1),
        Decimal(0),
        Decimal(0),
        int(
            (
                mod.DEVELOPMENT_END_EXCLUSIVE
                - mod.DEVELOPMENT_START
                - timedelta(minutes=1)
            ).total_seconds()
        ),
        None,
    )
    assert mod._trades((escaped,)) == ()


def test_corrected_runner_has_no_validation_or_holdout_import() -> None:
    source = Path(mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    assert not any("validation" in name or "holdout" in name for name in imported)


def test_synthetic_runtime_estimate_stays_below_stop_threshold() -> None:
    result = mod.benchmark_corrected_runtime(10_000)
    assert result["unique_real_m1_streams"] == 6
    assert result["estimated_total_minutes"] < 120


def test_corrected_writer_emits_exact_six_artifacts(tmp_path: Path) -> None:
    inputs = [
        DevelopmentSleeveInput(
            FAMILY_B5B,
            "B5B_FROZEN_DIRECT_AUDCAD_MR",
            "B5B_AUDCAD",
            "AUD/CAD.OANDA",
            (),
            (),
            (),
            mod.DEVELOPMENT_FOLD_BOUNDARIES,
            FrequencyStats(rollover_supported=False),
            0,
            0,
        )
    ]
    for instrument in B5C_INSTRUMENTS:
        inputs.append(
            DevelopmentSleeveInput(
                FAMILY_B5C,
                "B5C_FROZEN_DAILY_OVERREACTION_REVERSAL",
                f"B5C_{instrument.split('.')[0].replace('/', '')}",
                instrument,
                (),
                (),
                (),
                mod.DEVELOPMENT_FOLD_BOUNDARIES,
                FrequencyStats(),
                0,
                0,
            )
        )
    scorecards = tuple(evaluate_development_sleeve(item) for item in inputs)
    families = mod.build_bc_family_summary(inputs)
    output = tmp_path / "corrected"
    mod.write_corrected_artifacts(
        scorecards=scorecards,
        family_summary=families,
        correction_summary={"status": "test"},
        diagnostics_summary={"status": "test"},
        metadata={"status": "test"},
        output_dir=output,
    )
    assert {path.name for path in output.iterdir()} == {
        "sleeve_scorecard.csv",
        "family_summary.csv",
        "correction_summary.json",
        "diagnostics_summary.json",
        "metadata.json",
        "artifact_hashes.json",
    }
