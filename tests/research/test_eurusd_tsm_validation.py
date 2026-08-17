from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from nautilus_trader.model import Bar

import ftmoquant.research.eurusd_tsm_validation as validation_module
from ftmoquant.backtest.execution_harness import (
    _instrument_for_profile,
    canonical_execution_profile,
)
from ftmoquant.data.dukascopy import SourceBar, _eurusd_instrument, _to_nautilus_bars
from ftmoquant.research.eurusd_tsm_development import (
    PreparedEurusdTsmMarketData,
    _build_scaled_instructions,
)
from ftmoquant.research.eurusd_tsm_spec import EURUSD_TSM_SEMANTIC_SHA256
from ftmoquant.research.eurusd_tsm_validation import (
    SELECTED_TRIAL_ID,
    EurusdTsmValidationError,
    ValidationMetrics,
    ValidationPreparedData,
    _validation_root,
    _validation_window,
    _verify_selected_candidate_document,
    _write_validation_artifacts,
    evaluate_prepared_validation,
    load_validation_protocol,
    run_eurusd_tsm_validation,
    validation_passes,
    verify_development_evidence,
)
from ftmoquant.research.g1.normalization import CompletedDailyMidpoint
from ftmoquant.research.stage_g import (
    HOLDOUT_START,
    VALIDATION_START,
)
from ftmoquant.research.ts_momentum_development import (
    load_development_evaluation_config,
)
from ftmoquant.strategies.eurusd_tsm import EurusdTsmFamily
from ftmoquant.strategies.trend_pullback import CompletedPair, PriceBar, Timeframe


def test_protocol_freezes_one_candidate_boundaries_and_threshold() -> None:
    protocol = load_validation_protocol()

    assert protocol.selected_trial_id == SELECTED_TRIAL_ID
    assert protocol.selected_parameters == {
        "timeframe": "H4",
        "lookback_bars": 6,
        "deadband": 0.25,
        "refresh_interval_bars": 3,
    }
    assert protocol.start_utc == VALIDATION_START
    assert protocol.end_exclusive_utc == HOLDOUT_START
    assert protocol.duration_days == 498
    assert protocol.minimum_transitions == 100
    assert not hasattr(validation_module, "run_search")
    assert not hasattr(validation_module, "select_candidate")


def test_only_exact_selected_candidate_identity_is_admitted() -> None:
    protocol = load_validation_protocol()
    candidate = _candidate_document(protocol)
    _verify_selected_candidate_document(candidate, protocol)

    with pytest.raises(EurusdTsmValidationError, match="identity"):
        _verify_selected_candidate_document(
            {**candidate, "selected_trial_id": "f" * 64}, protocol
        )
    with pytest.raises(EurusdTsmValidationError, match="identity"):
        _verify_selected_candidate_document(
            {
                **candidate,
                "parameters": {**candidate["parameters"], "deadband": 0.5},
            },
            protocol,
        )
    with pytest.raises(EurusdTsmValidationError, match="identity"):
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

    with pytest.raises(EurusdTsmValidationError, match="artifact SHA"):
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
        replace(passing, executed_raw_alpha_transitions=99), protocol
    )
    assert not validation_passes(
        replace(
            passing,
            stressed_net_return=-0.001,
            stressed_expectancy=-0.00001,
        ),
        protocol,
    )


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

    with pytest.raises(EurusdTsmValidationError, match="too short"):
        run_eurusd_tsm_validation(
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


def test_synthetic_warmup_native_execution_resets_pnl_and_stays_bounded() -> None:
    protocol = load_validation_protocol()
    prepared = _prepared_data()
    family = EurusdTsmFamily()
    window = _validation_window(protocol)
    instructions = _build_scaled_instructions(
        family,
        dict(protocol.selected_parameters),
        prepared.market.h4_pairs,
        prepared.market.daily_midpoints,
        prepared.market.minute_bars,
        window,
    )

    assert prepared.market.h4_pairs[0].info_time_ns < _ns(VALIDATION_START)
    assert instructions
    assert all(
        item.decision_information_ns >= _ns(VALIDATION_START) for item in instructions
    )
    assert all(
        item.execution_information_ns < _ns(HOLDOUT_START) for item in instructions
    )

    metrics, native = evaluate_prepared_validation(
        prepared, protocol, load_development_evaluation_config()
    )
    assert metrics.fold_id == "validation_one_shot"
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


def _candidate_document(protocol: object) -> dict[str, object]:
    typed = load_validation_protocol()
    assert protocol == typed
    return {
        "family_id": "eurusd_tsm_v1",
        "family_version": "1.0.0",
        "family_semantic_sha256": EURUSD_TSM_SEMANTIC_SHA256,
        "selected_trial_id": SELECTED_TRIAL_ID,
        "parameters": dict(typed.selected_parameters),
        "validation_accessed": False,
    }


def _metrics() -> ValidationMetrics:
    return ValidationMetrics(
        fold_id="validation_one_shot",
        executed_raw_alpha_transitions=100,
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
    return ValidationPreparedData(
        market=PreparedEurusdTsmMarketData(
            instrument=instrument,
            minute_bars=_minute_bars(),
            h1_pairs=(),
            h4_pairs=_signal_pairs(),
            daily_midpoints=daily,
        ),
        validation_manifest_sha256="1" * 64,
        validation_catalog_tree_sha256="2" * 64,
    )


def _signal_pairs() -> tuple[CompletedPair, ...]:
    information_times = [
        VALIDATION_START - timedelta(minutes=721 - index) for index in range(721)
    ] + [VALIDATION_START + timedelta(minutes=2 * index) for index in range(12)]
    result = []
    for index, information in enumerate(information_times):
        close = Decimal("1.05000") + Decimal(index) * Decimal("0.00001")
        if index % 2:
            close += Decimal("0.000005")
        ask = close + Decimal("0.00020")
        result.append(
            CompletedPair(
                timeframe=Timeframe.FOUR_HOURS,
                bid=PriceBar(close, close, close, close),
                ask=PriceBar(ask, ask, ask, ask),
                ts_event=_ns(information - timedelta(minutes=1)),
                info_time_ns=_ns(information),
                contiguous=True,
            )
        )
    return tuple(result)


def _minute_bars() -> tuple[Bar, ...]:
    timestamps = [VALIDATION_START + timedelta(minutes=index) for index in range(35)]
    # A quote beyond the exclusive endpoint proves it cannot be selected.
    timestamps.append(HOLDOUT_START + timedelta(minutes=1))
    bid = [
        SourceBar(
            item,
            Decimal("1.10000"),
            Decimal("1.10000"),
            Decimal("1.10000"),
            Decimal("1.10000"),
            Decimal("1000000"),
        )
        for item in timestamps
    ]
    ask = [
        SourceBar(
            item,
            Decimal("1.10020"),
            Decimal("1.10020"),
            Decimal("1.10020"),
            Decimal("1.10020"),
            Decimal("1000000"),
        )
        for item in timestamps
    ]
    return tuple(
        sorted(
            (*_to_nautilus_bars(bid, "BID"), *_to_nautilus_bars(ask, "ASK")),
            key=lambda bar: (bar.ts_event, str(bar.bar_type)),
        )
    )


def _ns(value: datetime) -> int:
    assert value.tzinfo is not None and value.utcoffset() == UTC.utcoffset(value)
    return int(value.timestamp() * 1_000_000_000)
