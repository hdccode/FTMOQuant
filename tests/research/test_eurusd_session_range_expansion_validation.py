from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest
from nautilus_trader.model import Bar

import ftmoquant.research.eurusd_session_range_expansion_validation as validation_module
from ftmoquant.backtest.execution_harness import (
    _instrument_for_profile,
    canonical_execution_profile,
)
from ftmoquant.data.dukascopy import SourceBar, _eurusd_instrument, _to_nautilus_bars
from ftmoquant.research.eurusd_session_range_expansion_development import (
    EurusdSessionRangeExpansionFamily,
    _build_scaled_instructions,
)
from ftmoquant.research.eurusd_session_range_expansion_spec import (
    EURUSD_SESSION_RANGE_EXPANSION_SEMANTIC_SHA256,
)
from ftmoquant.research.eurusd_session_range_expansion_validation import (
    SELECTED_TRIAL_ID,
    EurusdSessionRangeExpansionValidationError,
    ValidationMetrics,
    ValidationPreparedData,
    _validation_root,
    _validation_window,
    _verify_selected_candidate_document,
    _write_validation_artifacts,
    evaluate_prepared_validation,
    load_validation_protocol,
    run_eurusd_session_range_expansion_validation,
    validation_passes,
    verify_development_evidence,
)
from ftmoquant.research.g1.normalization import CompletedDailyMidpoint
from ftmoquant.research.stage_g import HOLDOUT_START, VALIDATION_START
from ftmoquant.research.ts_momentum_development import (
    load_development_evaluation_config,
)
from ftmoquant.strategies.ts_momentum import RawDirectionalTarget

# The first range-window bar's OPEN is one minute before London local midnight
# on the London day containing VALIDATION_START, i.e. one hour of that day's
# range (local 00:00-01:00, London BST) falls strictly inside DEVELOPMENT.
_DEV_OPEN_START = datetime(2023, 4, 10, 22, 59, tzinfo=UTC)
_TOTAL_MINUTES = 18 * 60  # past local 17:00 exit with a safety margin


def test_protocol_freezes_one_candidate_boundaries_and_threshold() -> None:
    protocol = load_validation_protocol()

    assert protocol.selected_trial_id == SELECTED_TRIAL_ID
    assert protocol.selected_parameters == {
        "breakout_window_end": "11:00",
        "scheduled_exit": "17:00",
    }
    assert protocol.start_utc == VALIDATION_START
    assert protocol.end_exclusive_utc == HOLDOUT_START
    assert protocol.duration_days == 498
    assert protocol.minimum_trades == 50
    assert not hasattr(validation_module, "run_search")
    assert not hasattr(validation_module, "select_candidate")


def test_cross_partition_warmup_and_dst_semantics_are_frozen_in_protocol() -> None:
    document = load_validation_protocol().canonical_document
    warmup = document["cross_partition_session_warmup"]
    dst = document["dst_semantics"]

    assert warmup["session_or_range_state_reset_at_validation_start"] is False
    assert warmup["development_execution_or_pnl_counted_in_validation"] is False
    assert (
        warmup["only_validation_interval_signals_may_generate_validation_trades"]
        is True
    )
    assert (
        dst["short_or_long_dst_transition_days"]
        == "remain_invalid_no_trade_not_normalized_or_fixed_for_validation"
    )


def test_only_exact_selected_candidate_identity_is_admitted() -> None:
    protocol = load_validation_protocol()
    candidate = _candidate_document(protocol)
    _verify_selected_candidate_document(candidate, protocol)

    with pytest.raises(EurusdSessionRangeExpansionValidationError, match="identity"):
        _verify_selected_candidate_document(
            {**candidate, "selected_trial_id": "f" * 64}, protocol
        )
    with pytest.raises(EurusdSessionRangeExpansionValidationError, match="identity"):
        mutated_parameters = {
            **cast(dict[str, object], candidate["parameters"]),
            "scheduled_exit": "16:00",
        }
        _verify_selected_candidate_document(
            {**candidate, "parameters": mutated_parameters},
            protocol,
        )
    with pytest.raises(EurusdSessionRangeExpansionValidationError, match="identity"):
        _verify_selected_candidate_document(
            {**candidate, "family_semantic_sha256": "0" * 64}, protocol
        )


def test_selected_artifact_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    protocol = load_validation_protocol()
    selected = tmp_path / "selected.json"
    registry = tmp_path / "registry.json"
    result = tmp_path / "result.json"
    selected.write_text("{}")
    registry.write_text("{}")
    result.write_text("{}")

    with pytest.raises(
        EurusdSessionRangeExpansionValidationError, match="artifact SHA"
    ):
        verify_development_evidence(
            protocol,
            selected_candidate_path=selected,
            trial_registry_path=registry,
            development_result_path=result,
        )


def test_validation_acceptance_gates_are_exact() -> None:
    protocol = load_validation_protocol()
    passing = _metrics()

    assert validation_passes(passing, protocol)
    assert not validation_passes(
        replace(passing, completed_executed_trades=49), protocol
    )
    assert validation_passes(replace(passing, completed_executed_trades=50), protocol)
    assert not validation_passes(
        replace(
            passing,
            stressed_net_return=-0.001,
            stressed_expectancy=-0.00001,
        ),
        protocol,
    )
    assert not validation_passes(replace(passing, net_return=0.0), protocol)


def test_short_validation_stops_before_any_data_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = replace(load_validation_protocol(), duration_days=100)
    output = tmp_path / "fixed-output"
    monkeypatch.setattr(validation_module, "VALIDATION_OUTPUT_DIR", output)
    monkeypatch.setattr(validation_module, "_require_clean_worktree", lambda: None)
    monkeypatch.setattr(
        validation_module, "load_validation_protocol", lambda _: protocol
    )
    monkeypatch.setattr(
        validation_module,
        "prepare_validation_market_data",
        lambda **_: pytest.fail("validation data preparation must not run"),
    )

    with pytest.raises(EurusdSessionRangeExpansionValidationError, match="too short"):
        run_eurusd_session_range_expansion_validation(
            protocol_path=tmp_path / "protocol.json",
            selected_candidate_path=tmp_path / "selected.json",
            trial_registry_path=tmp_path / "registry.json",
            development_result_path=tmp_path / "development.json",
            universe_readiness_path=tmp_path / "readiness.json",
            development_roots={},
            validation_root=tmp_path / "validation",
            evaluation_config_path=tmp_path / "costs.json",
            output_dir=output,
        )


def test_existing_output_directory_blocks_a_second_run(tmp_path: Path) -> None:
    output = tmp_path / "already-exists"
    output.mkdir()

    with pytest.raises(EurusdSessionRangeExpansionValidationError, match="must not"):
        run_eurusd_session_range_expansion_validation(
            protocol_path=Path("does-not-matter.json"),
            selected_candidate_path=tmp_path / "selected.json",
            trial_registry_path=tmp_path / "registry.json",
            development_result_path=tmp_path / "development.json",
            universe_readiness_path=tmp_path / "readiness.json",
            development_roots={},
            validation_root=tmp_path / "validation",
            evaluation_config_path=tmp_path / "costs.json",
            output_dir=output,
        )


def test_development_observations_warm_the_first_validation_days_partial_range() -> (
    None
):
    """DEVELOPMENT covers the London day's 00:00-01:00 portion of the range;
    VALIDATION covers 01:00-08:00, the 08:00 breakout, and the 17:00 exit.
    The candidate must still fire using the full, causally-warmed range."""
    protocol = load_validation_protocol()
    prepared = _prepared_data()
    window = _validation_window(protocol)

    # The execution-frame stream genuinely starts before validation_start.
    assert prepared.execution_frames[0][1] < _ns(VALIDATION_START)
    instructions = _build_scaled_instructions(
        _family(),
        dict(protocol.selected_parameters),
        prepared.execution_frames,
        prepared.daily_midpoints,
        window,
    )

    assert instructions
    counted = [item for item in instructions if item.count_alpha_transition]
    assert len(counted) == 1
    assert counted[0].raw_target is RawDirectionalTarget.LONG
    # Only validation-interval information may generate a validation trade.
    assert all(
        item.decision_information_ns >= _ns(VALIDATION_START) for item in instructions
    )
    assert all(
        item.execution_information_ns >= _ns(VALIDATION_START) for item in instructions
    )


def test_pre_validation_trades_and_pnl_are_excluded_and_account_resets() -> None:
    protocol = load_validation_protocol()
    prepared = _prepared_data()

    metrics, native = evaluate_prepared_validation(
        prepared, protocol, load_development_evaluation_config()
    )

    assert metrics.fold_id == "validation_one_shot"
    assert metrics.completed_executed_trades == 1
    assert native.equity_points[0].information_time_ns == _ns(VALIDATION_START)
    assert native.equity_points[0].equity == Decimal("100000")


def test_final_holdout_paths_are_rejected() -> None:
    with pytest.raises(Exception, match="holdout"):
        _validation_root("/sealed/final_holdout/EURUSD")


def test_validation_artifact_is_deterministic(tmp_path: Path) -> None:
    payload = {
        "schema": "synthetic-validation",
        "metrics": _metrics(),
        "path": Path("validation/result"),
    }
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_validation_artifacts(first, payload)
    _write_validation_artifacts(second, payload)

    for name in ("validation_result.json", "artifact_hashes.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def _family() -> EurusdSessionRangeExpansionFamily:
    return EurusdSessionRangeExpansionFamily()


def _candidate_document(protocol: object) -> dict[str, object]:
    typed = load_validation_protocol()
    assert protocol == typed
    return {
        "family_id": "eurusd_session_range_expansion_v1",
        "family_version": "1.0.0",
        "family_semantic_sha256": EURUSD_SESSION_RANGE_EXPANSION_SEMANTIC_SHA256,
        "selected_trial_id": SELECTED_TRIAL_ID,
        "parameters": dict(typed.selected_parameters),
        "validation_accessed": False,
    }


def _metrics() -> ValidationMetrics:
    return ValidationMetrics(
        fold_id="validation_one_shot",
        completed_executed_trades=100,
        normalized_turnover=5.0,
        net_return=0.01,
        stressed_net_return=0.009,
        net_expectancy=0.0001,
        stressed_expectancy=0.00009,
        sharpe=0.5,
        maximum_drawdown=0.01,
        net_daily_mean=0.00001,
        yearly_attribution=(),
        realized_variable_cost_usd=200.0,
        realized_variable_cost_return=0.002,
    )


def _prepared_data() -> ValidationPreparedData:
    profile = canonical_execution_profile()
    instrument = _instrument_for_profile(_eurusd_instrument(), profile.fee)
    daily = tuple(
        CompletedDailyMidpoint(
            _ns(VALIDATION_START - timedelta(days=31 - index)),
            1.09 + (0.01 if index % 2 else 0.0) + index * 0.0001,
        )
        for index in range(31)
    )
    execution_frames, validation_bars = _cross_partition_frames_and_bars()
    return ValidationPreparedData(
        instrument=instrument,
        validation_minute_bars=validation_bars,
        execution_frames=execution_frames,
        daily_midpoints=daily,
        validation_manifest_sha256="1" * 64,
        validation_catalog_tree_sha256="2" * 64,
    )


def _cross_partition_frames_and_bars() -> tuple[
    tuple[tuple[int, int, Decimal], ...], tuple[Bar, ...]
]:
    minute_ns = 60_000_000_000
    frames: list[tuple[int, int, Decimal]] = []
    price = Decimal("1.10000")
    open_ns = _ns(_DEV_OPEN_START)
    for index in range(_TOTAL_MINUTES):
        if index == 480:
            price += Decimal("0.02000")  # the breakout, right at local 08:00
        event_ns = open_ns + index * minute_ns
        info_ns = event_ns + minute_ns
        frames.append((event_ns, info_ns, price))

    validation_start_ns = _ns(VALIDATION_START)
    validation_frames = [item for item in frames if item[1] >= validation_start_ns]
    validation_bars = _bars_for_frames(validation_frames)
    return tuple(frames), validation_bars


def _bars_for_frames(frames: list[tuple[int, int, Decimal]]) -> tuple[Bar, ...]:
    bid_rows = [
        SourceBar(
            timestamp=_datetime(event_ns),
            open=price,
            high=price,
            low=price,
            close=price,
            volume=Decimal("1000000"),
        )
        for event_ns, _, price in frames
    ]
    ask_rows = [
        SourceBar(
            timestamp=_datetime(event_ns),
            open=price + Decimal("0.00020"),
            high=price + Decimal("0.00020"),
            low=price + Decimal("0.00020"),
            close=price + Decimal("0.00020"),
            volume=Decimal("1000000"),
        )
        for event_ns, _, price in frames
    ]
    return tuple(
        sorted(
            (*_to_nautilus_bars(bid_rows, "BID"), *_to_nautilus_bars(ask_rows, "ASK")),
            key=lambda bar: (bar.ts_event, str(bar.bar_type)),
        )
    )


def _datetime(value_ns: int) -> datetime:
    return datetime.fromtimestamp(value_ns / 1_000_000_000, tz=UTC)


def _ns(value: datetime) -> int:
    assert value.tzinfo is not None and value.utcoffset() == UTC.utcoffset(value)
    return int(value.timestamp() * 1_000_000_000)
