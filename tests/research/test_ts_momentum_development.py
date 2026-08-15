from __future__ import annotations

import json
from argparse import ArgumentTypeError
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from ftmoquant.backtest.execution_harness import (
    _profile_dict,
    canonical_execution_profile,
)
from ftmoquant.data.dukascopy import SourceBar
from ftmoquant.data.instruments import (
    EURUSD_SPEC,
    GBPUSD_SPEC,
    InstrumentSpec,
    to_nautilus_bars,
)
from ftmoquant.research import ts_momentum_development as evaluator
from ftmoquant.research.stage_g import FROZEN_INSTRUMENT_IDS, frozen_development_folds
from ftmoquant.research.ts_momentum_development import (
    EVALUATION_CONFIG_PATH,
    EVALUATION_CONFIG_SHA256,
    TargetInstruction,
    TsMomentumEvaluationError,
    _annualized_sharpe,
    _daily_return_rows,
    _development_root,
    _maximum_drawdown,
    _pooled_statistics,
    _run_fold_engine,
    load_development_evaluation_config,
    materialize_canonical_cost_models,
)
from ftmoquant.strategies.ts_momentum import RawDirectionalTarget


def test_frozen_evaluation_config_derives_existing_g07_costs_and_limits() -> None:
    config = load_development_evaluation_config()

    assert config.semantic_sha256 == EVALUATION_CONFIG_SHA256
    assert config.fixed_base_units == Decimal("100000")
    assert config.daily_return_denominator == Decimal("100000")
    assert tuple(item.instrument_id for item in config.cost_models) == (
        FROZEN_INSTRUMENT_IDS
    )
    assert all(
        _profile_dict(item.execution_profile)
        == _profile_dict(canonical_execution_profile())
        for item in config.cost_models
    )
    assert config.exposure_limits.max_abs_amount == {
        "EUR": Decimal("3000000"),
        "GBP": Decimal("3000000"),
        "USD": Decimal("3000000"),
    }


def test_evaluation_config_hash_drift_is_rejected(tmp_path: Path) -> None:
    document = json.loads(EVALUATION_CONFIG_PATH.read_text(encoding="utf-8"))
    document["account"]["leverage"] = "31"
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(TsMomentumEvaluationError, match="hash is not frozen"):
        load_development_evaluation_config(path)


def test_reserved_cost_artifact_is_exact_tracked_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reserved = tmp_path / ".artifacts/g1_4c/phase2_cost_models.json"
    monkeypatch.setattr(
        "ftmoquant.research.ts_momentum_development.RESERVED_COST_MODELS_PATH",
        reserved,
    )

    materialize_canonical_cost_models(reserved)

    assert reserved.read_bytes() == EVALUATION_CONFIG_PATH.read_bytes()
    assert load_development_evaluation_config(reserved).semantic_sha256 == (
        EVALUATION_CONFIG_SHA256
    )


@pytest.mark.parametrize("name", ["validation", "FINAL-HOLDOUT", "holdout"])
def test_cli_refuses_validation_and_holdout_roots(name: str) -> None:
    with pytest.raises(ArgumentTypeError, match="forbidden"):
        _development_root(f"EUR/USD.DUKASCOPY=.artifacts/{name}/EURUSD")


def test_synthetic_native_engine_executes_targets_and_observed_spread() -> None:
    config = load_development_evaluation_config()
    times = (
        datetime(2020, 4, 13, 20, 59, tzinfo=UTC),
        datetime(2020, 4, 14, 20, 59, tzinfo=UTC),
        datetime(2020, 4, 15, 20, 59, tzinfo=UTC),
    )
    instruments = (EURUSD_SPEC, GBPUSD_SPEC)
    native_bars = _paired_bars(instruments, times)
    frame_midpoints = tuple(
        (instrument.instrument_id, Decimal("1.10010"))
        for instrument in instruments
    )
    instructions = (
        _instruction(
            EURUSD_SPEC.instrument_id,
            RawDirectionalTarget.LONG,
            times[0],
            frame_midpoints,
        ),
        _instruction(
            EURUSD_SPEC.instrument_id,
            RawDirectionalTarget.FLAT,
            times[1],
            tuple(
                (instrument.instrument_id, Decimal("1.10020"))
                for instrument in instruments
            ),
        ),
    )

    result = _run_fold_engine(
        fold=frozen_development_folds().folds[0],
        instruments=tuple(item.nautilus_instrument() for item in instruments),
        bars=native_bars,
        instructions=instructions,
        config=config,
    )

    assert len(result.fill_report) == 2
    assert result.exposure_breach_count == 0
    assert [item.equity for item in result.daily_equity] == [
        Decimal("99980.00"),
        Decimal("99990.00"),
        Decimal("99990.00"),
    ]
    assert [item.cost_usd for item in result.realized_costs] == [
        Decimal("10.000000"),
        Decimal("10.000000"),
    ]

    rows = _daily_return_rows(
        result,
        frozen_development_folds().folds[0],
        config.daily_return_denominator,
    )
    assert [item["net_return"] for item in rows] == [0.0001, 0.0]
    assert rows[0]["cost_stress_1_5x_return"] == 0.00005


def test_metric_presentation_is_deterministic_and_fail_closed() -> None:
    values = (0.01, -0.02, 0.03, -0.01)

    assert _annualized_sharpe(values) == _annualized_sharpe(values)
    assert _annualized_sharpe((0.0, 0.0)) is None
    assert _maximum_drawdown(values) == pytest.approx(0.02)


def test_pooled_manifest_labels_derive_from_the_candidate_id(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        evaluator,
        "stationary_bootstrap_confidence_interval",
        lambda series, _config: {"series_label": series.name},
    )
    monkeypatch.setattr(
        evaluator,
        "spa_reality_check",
        lambda _benchmark, losses, _config: {
            "model_labels": list(losses.columns)
        },
    )
    monkeypatch.setattr(evaluator, "result_as_dict", lambda value: value)
    rows = [
        {
            "fold_id": "synthetic",
            "session_date": f"2020-01-{day:02d}",
            "net_return": 0.001,
        }
        for day in range(1, 21)
    ]

    session = _pooled_statistics(rows, strategy_id="session_range_expansion_v1")
    momentum = _pooled_statistics(rows, strategy_id="ts_momentum_v1")

    assert session["stationary_bootstrap_mean_confidence_interval"] == {
        "series_label": "session_range_expansion_v1_daily_net_return"
    }
    assert session["spa_zero_return_comparison"] == {
        "model_labels": ["session_range_expansion_v1"]
    }
    assert momentum["stationary_bootstrap_mean_confidence_interval"] == {
        "series_label": "ts_momentum_v1_daily_net_return"
    }
    assert momentum["spa_zero_return_comparison"] == {
        "model_labels": ["ts_momentum_v1"]
    }


def _paired_bars(
    instruments: tuple[InstrumentSpec, ...], times: tuple[datetime, ...]
) -> tuple[object, ...]:
    bars: list[object] = []
    for instrument in instruments:
        bids: list[SourceBar] = []
        asks: list[SourceBar] = []
        for index, timestamp in enumerate(times):
            bid = Decimal("1.10000") + Decimal(index) / Decimal("10000")
            ask = bid + Decimal("0.00020")
            bids.append(_source_bar(timestamp, bid))
            asks.append(_source_bar(timestamp, ask))
        bars.extend(to_nautilus_bars(bids, "BID", instrument))
        bars.extend(to_nautilus_bars(asks, "ASK", instrument))
    return tuple(bars)


def _source_bar(timestamp: datetime, price: Decimal) -> SourceBar:
    return SourceBar(timestamp, price, price, price, price, Decimal(1))


def _instruction(
    instrument_id: str,
    target: RawDirectionalTarget,
    event_time: datetime,
    frame_midpoints: tuple[tuple[str, Decimal], ...],
) -> TargetInstruction:
    information = event_time + timedelta(minutes=1)
    return TargetInstruction(
        instrument_id=instrument_id,
        target=target,
        event_time_ns=int(event_time.timestamp() * 1_000_000_000),
        information_time_ns=int(information.timestamp() * 1_000_000_000),
        midpoint=dict(frame_midpoints)[instrument_id],
        frame_midpoints=frame_midpoints,
    )
