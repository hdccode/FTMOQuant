from __future__ import annotations

from argparse import ArgumentTypeError
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from ftmoquant.research import session_range_expansion_development as evaluator
from ftmoquant.research.session_range_expansion_spec import (
    SESSION_RANGE_EXPANSION_CONFIG_SHA256,
    SESSION_RANGE_EXPANSION_SPEC_PATH,
    load_session_range_expansion_spec,
)
from ftmoquant.research.stage_g import (
    FROZEN_INSTRUMENT_IDS,
    InstrumentBarObservation,
    SynchronizedClockFrame,
)
from ftmoquant.research.ts_momentum_development import _target_instructions_at_frame
from ftmoquant.strategies.session_range_expansion import (
    SessionRangeExpansionStateMachine,
)
from ftmoquant.strategies.ts_momentum import (
    ExecutableDirectionalTarget,
    RawDirectionalTarget,
)

LONDON = ZoneInfo("Europe/London")


def test_session_evaluator_delegates_to_shared_frozen_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def fake_shared_runner(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(
        evaluator, "evaluate_frozen_development_candidate", fake_shared_runner
    )

    result = evaluator.evaluate_session_range_expansion_development(
        spec_path=SESSION_RANGE_EXPANSION_SPEC_PATH,
        universe_readiness_path=tmp_path / "universe.json",
        development_roots={
            instrument_id: tmp_path / instrument_id.replace("/", "_")
            for instrument_id in FROZEN_INSTRUMENT_IDS
        },
        cost_models_path=tmp_path / "costs.json",
        output_dir=tmp_path / "output",
    )

    assert result == {"ok": True}
    assert captured["strategy_id"] == "session_range_expansion_v1"
    assert captured["strategy_config_sha256"] == (
        SESSION_RANGE_EXPANSION_CONFIG_SHA256
    )
    assert captured["target_instruction_builder"] is (
        evaluator._build_target_instructions
    )
    assert captured["result_schema"] == evaluator.RESULT_SCHEMA


def test_session_targets_use_shared_native_instruction_shape() -> None:
    event = datetime(2020, 7, 6, 8, 1, tzinfo=UTC)
    information = event + timedelta(minutes=1)
    frame = SynchronizedClockFrame(
        timestamp_utc=event,
        available_at_utc=information,
        tradable=True,
        observations=tuple(
            InstrumentBarObservation(
                instrument_id=instrument_id,
                timestamp_utc=event,
                available_at_utc=information,
                bid=Decimal("1.1000"),
                ask=Decimal("1.1002"),
            )
            for instrument_id in FROZEN_INSTRUMENT_IDS
        ),
    )
    target = ExecutableDirectionalTarget(
        instrument_id=FROZEN_INSTRUMENT_IDS[0],
        target=RawDirectionalTarget.LONG,
        signal_information_time_utc=event,
        execution_event_time_utc=event,
        execution_information_time_utc=information,
    )

    instructions = _target_instructions_at_frame((target,), frame)

    assert instructions[0].target is RawDirectionalTarget.LONG
    assert instructions[0].event_time_ns == int(event.timestamp() * 1_000_000_000)
    assert instructions[0].information_time_ns == int(
        information.timestamp() * 1_000_000_000
    )
    assert instructions[0].frame_midpoints == tuple(
        (instrument_id, Decimal("1.1001"))
        for instrument_id in FROZEN_INSTRUMENT_IDS
    )


def test_incomplete_execution_frame_preserves_pending_entry_and_flat_target() -> None:
    machine = SessionRangeExpansionStateMachine(load_session_range_expansion_spec())
    day = date(2020, 7, 6)
    for minute in range(8 * 60):
        machine.on_frame(_session_frame(day, minute, Decimal("1.1000")))

    breakout = _session_frame(day, 8 * 60, Decimal("1.2000"))
    machine.on_frame(breakout)
    assert machine.on_execution_frame(breakout) == ()

    incomplete_entry = _session_frame(
        day, 8 * 60 + 1, Decimal("1.2000"), missing=FROZEN_INSTRUMENT_IDS[1]
    )
    machine.on_frame(incomplete_entry)
    assert not evaluator._has_synchronized_execution_prices(incomplete_entry)
    entries = machine.on_execution_frame(
        _session_frame(day, 8 * 60 + 2, Decimal("1.2000"))
    )
    assert [item.target for item in entries] == [RawDirectionalTarget.LONG] * 2

    exit_frame = _session_frame(day, 16 * 60, Decimal("1.2000"))
    machine.on_frame(exit_frame)
    assert machine.on_execution_frame(exit_frame) == ()
    incomplete_exit = _session_frame(
        day, 16 * 60 + 1, Decimal("1.2000"), missing=FROZEN_INSTRUMENT_IDS[1]
    )
    machine.on_frame(incomplete_exit)
    assert not evaluator._has_synchronized_execution_prices(incomplete_exit)
    exits = machine.on_execution_frame(
        _session_frame(day, 16 * 60 + 2, Decimal("1.2000"))
    )
    assert [item.target for item in exits] == [RawDirectionalTarget.FLAT] * 2


@pytest.mark.parametrize("name", ["validation", "FINAL-HOLDOUT", "holdout"])
def test_session_cli_refuses_validation_and_holdout_roots(name: str) -> None:
    with pytest.raises(ArgumentTypeError, match="forbidden"):
        evaluator._development_root(f"EUR/USD.DUKASCOPY=.artifacts/{name}/EURUSD")


def _session_frame(
    day: date, london_minute: int, price: Decimal, *, missing: str | None = None
) -> SynchronizedClockFrame:
    information = datetime.combine(day, time(), tzinfo=LONDON) + timedelta(
        minutes=london_minute
    )
    available = information.astimezone(UTC)
    event = available - timedelta(minutes=1)
    return SynchronizedClockFrame(
        timestamp_utc=event,
        available_at_utc=available,
        observations=tuple(
            None
            if instrument_id == missing
            else InstrumentBarObservation(
                instrument_id=instrument_id,
                timestamp_utc=event,
                available_at_utc=available,
                bid=price - Decimal("0.0001"),
                ask=price + Decimal("0.0001"),
            )
            for instrument_id in FROZEN_INSTRUMENT_IDS
        ),
        tradable=True,
    )
