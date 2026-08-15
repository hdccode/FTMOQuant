from __future__ import annotations

from argparse import ArgumentTypeError
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from ftmoquant.research import liquidity_shock_reversion_development as evaluator
from ftmoquant.research.liquidity_shock_reversion_spec import (
    LIQUIDITY_SHOCK_REVERSION_CONFIG_SHA256,
    LIQUIDITY_SHOCK_REVERSION_SPEC_PATH,
)
from ftmoquant.research.stage_g import (
    DEVELOPMENT_END_EXCLUSIVE,
    DEVELOPMENT_START,
    FROZEN_INSTRUMENT_IDS,
    DevelopmentResearchContext,
    InstrumentBarObservation,
    SynchronizedClockFrame,
    frozen_development_folds,
)
from ftmoquant.strategies.liquidity_shock_reversion import RawDirectionalTarget


def test_evaluator_delegates_only_to_shared_frozen_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def fake_shared_runner(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(
        evaluator, "evaluate_frozen_development_candidate", fake_shared_runner
    )

    result = evaluator.evaluate_liquidity_shock_reversion_development(
        spec_path=LIQUIDITY_SHOCK_REVERSION_SPEC_PATH,
        universe_readiness_path=tmp_path / "universe.json",
        development_roots={
            item: tmp_path / item.replace("/", "_") for item in FROZEN_INSTRUMENT_IDS
        },
        cost_models_path=tmp_path / "costs.json",
        output_dir=tmp_path / "output",
    )

    assert result == {"ok": True}
    assert captured["strategy_id"] == "liquidity_shock_reversion_v1"
    assert captured["strategy_config_sha256"] == LIQUIDITY_SHOCK_REVERSION_CONFIG_SHA256
    assert (
        captured["target_instruction_builder"] is evaluator._build_target_instructions
    )
    assert captured["result_schema"] == evaluator.RESULT_SCHEMA


def test_incomplete_frame_preserves_pending_entry_until_later_complete_frame() -> None:
    fold = frozen_development_folds().folds[0]
    context = DevelopmentResearchContext(
        universe=None,  # type: ignore[arg-type]
        artifacts=(),
        start_utc=DEVELOPMENT_START,
        end_exclusive_utc=DEVELOPMENT_END_EXCLUSIVE,
    )
    frames = tuple(
        _frame(
            fold.compare_start_utc + timedelta(minutes=index),
            (Decimal("0.001") * Decimal(index)).exp(),
            complete=index != 62,
        )
        if index <= 60
        else _frame(
            fold.compare_start_utc + timedelta(minutes=index),
            Decimal("0.066").exp(),
            complete=index != 62,
        )
        for index in range(64)
    )

    instructions = evaluator._build_target_instructions(
        context, fold, frames, LIQUIDITY_SHOCK_REVERSION_SPEC_PATH
    )

    assert len(instructions) == 1
    instruction = instructions[0]
    assert instruction.instrument_id == FROZEN_INSTRUMENT_IDS[0]
    assert instruction.target is RawDirectionalTarget.SHORT
    assert instruction.information_time_ns == int(
        (fold.compare_start_utc + timedelta(minutes=63)).timestamp() * 1_000_000_000
    )


@pytest.mark.parametrize("name", ["validation", "FINAL-HOLDOUT", "holdout"])
def test_cli_refuses_validation_and_holdout_roots(name: str) -> None:
    with pytest.raises(ArgumentTypeError, match="forbidden"):
        evaluator._development_root(f"EUR/USD.DUKASCOPY=.artifacts/{name}/EURUSD")


def _frame(
    information: datetime, eur_price: Decimal, *, complete: bool
) -> SynchronizedClockFrame:
    event = information - timedelta(minutes=1)
    observations = (
        InstrumentBarObservation(
            FROZEN_INSTRUMENT_IDS[0],
            event,
            information,
            eur_price - Decimal("0.0001"),
            eur_price + Decimal("0.0001"),
        ),
        InstrumentBarObservation(
            FROZEN_INSTRUMENT_IDS[1],
            event,
            information,
            Decimal("0.9999"),
            Decimal("1.0001"),
        )
        if complete
        else None,
    )
    return SynchronizedClockFrame(event, information, observations, complete)
