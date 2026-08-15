from __future__ import annotations

from argparse import ArgumentTypeError
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from ftmoquant.research import leo_gbpusd_cache as cache
from ftmoquant.research import leo_gbpusd_development as evaluator
from ftmoquant.research.leo_gbpusd_spec import (
    LEO_GBPUSD_CONFIG_SHA256,
    LEO_GBPUSD_SPEC_PATH,
)
from ftmoquant.research.stage_g import FROZEN_INSTRUMENT_IDS, DevelopmentFold
from ftmoquant.strategies.leo_gbpusd import LeoExit, LeoExitReason
from ftmoquant.strategies.trend_pullback import Direction, PriceBar

LONDON = ZoneInfo("Europe/London")


def test_evaluator_delegates_to_the_shared_development_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def fake_shared_runner(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(
        evaluator, "evaluate_frozen_development_candidate", fake_shared_runner
    )

    assert evaluator.evaluate_leo_gbpusd_development(
        spec_path=LEO_GBPUSD_SPEC_PATH,
        universe_readiness_path=tmp_path / "universe.json",
        development_roots={
            item: tmp_path / item.replace("/", "_") for item in FROZEN_INSTRUMENT_IDS
        },
        cost_models_path=tmp_path / "costs.json",
        output_dir=tmp_path / "output",
        cache_dir=tmp_path / "cache",
    ) == {"ok": True}
    assert captured["strategy_id"] == "leo_gbpusd_v1"
    assert captured["strategy_config_sha256"] == LEO_GBPUSD_CONFIG_SHA256
    assert callable(captured["target_instruction_builder"])
    assert captured["result_schema"] == evaluator.RESULT_SCHEMA


def test_complete_15m_bars_call_frozen_state_machine_and_emit_gbp_target() -> None:
    day = date(2020, 7, 6)
    start = datetime.combine(day, time(), tzinfo=LONDON)
    bars = tuple(
        _bar(start + timedelta(minutes=index), _price_for(index))
        for index in range(-15, 9 * 60 + 30, 15)
    )
    fold = DevelopmentFold(
        "synthetic",
        datetime(2020, 1, 1, tzinfo=UTC),
        start.astimezone(UTC),
        start.astimezone(UTC),
        (start + timedelta(days=1)).astimezone(UTC),
    )

    instructions = evaluator._build_target_instructions(
        None,  # type: ignore[arg-type]
        fold,
        bars,
        LEO_GBPUSD_SPEC_PATH,
    )

    # The cache supplies only complete 15m bars; a no-signal fixture remains
    # inert and therefore cannot manufacture an execution target.
    assert instructions == ()


@pytest.mark.parametrize("name", ["validation", "FINAL-HOLDOUT", "holdout"])
def test_cli_refuses_validation_and_holdout_roots(name: str) -> None:
    with pytest.raises(ArgumentTypeError, match="forbidden"):
        evaluator._development_root(f"GBP/USD.DUKASCOPY=.artifacts/{name}/GBPUSD")


def test_cache_aggregation_requires_exact_paired_complete_minutes() -> None:
    start = datetime(2020, 7, 6, tzinfo=UTC)
    bid = {
        index: _source_native_bar(
            start + timedelta(minutes=index), Decimal("1.2") + Decimal(index) / 10000
        )
        for index in range(15)
    }
    ask = {
        index: _source_native_bar(
            start + timedelta(minutes=index), Decimal("1.2002") + Decimal(index) / 10000
        )
        for index in range(15)
    }
    rows = cache._aggregate_complete_15m(
        {item.ts_event: item for item in bid.values()},
        {item.ts_event: item for item in ask.values()},
    )
    assert len(rows) == 1
    assert rows[0].bid.open == Decimal("1.2")
    assert rows[0].bid.close == Decimal("1.2014")
    assert rows[0].ask.high == Decimal("1.2016")
    del ask[7]
    assert (
        cache._aggregate_complete_15m(
            {item.ts_event: item for item in bid.values()},
            {item.ts_event: item for item in ask.values()},
        )
        == ()
    )


@pytest.mark.parametrize(
    ("reason", "price"),
    [
        (LeoExitReason.STOP_LOSS, Decimal("1.25000")),
        (LeoExitReason.TAKE_PROFIT, Decimal("1.27000")),
    ],
)
def test_adapter_preserves_priced_leo_exit(
    reason: LeoExitReason, price: Decimal
) -> None:
    bar = _bar(datetime(2020, 7, 6, tzinfo=UTC), Decimal("1.2600"))
    action = LeoExit(
        direction=Direction.LONG,
        reason=reason,
        exit_information_time_utc=bar.available_at_utc,
        exit_price=price,
        stop_price=Decimal("1.25000"),
        target_price=Decimal("1.27000"),
    )
    instruction = evaluator._instruction_from_action(action, bar)
    assert instruction is not None
    assert instruction.execution_price == price
    assert instruction.exit_reason == reason.value


def test_adapter_preserves_session_exit_reason_but_uses_market_liquidation() -> None:
    bar = _bar(datetime(2020, 7, 6, tzinfo=UTC), Decimal("1.2600"))
    action = LeoExit(
        direction=Direction.SHORT,
        reason=LeoExitReason.SESSION_WINDOW_END,
        exit_information_time_utc=bar.available_at_utc,
        exit_price=Decimal("1.2601"),
        stop_price=Decimal("1.27000"),
        target_price=Decimal("1.23000"),
    )
    instruction = evaluator._instruction_from_action(action, bar)
    assert instruction is not None
    assert instruction.execution_price is None
    assert instruction.exit_reason == "session_window_end"


def _source_native_bar(timestamp: datetime, value: Decimal) -> SimpleNamespace:
    price = SimpleNamespace(as_decimal=lambda: value)
    return SimpleNamespace(
        ts_event=int(timestamp.timestamp() * 1_000_000_000),
        ts_init=int((timestamp + timedelta(minutes=1)).timestamp() * 1_000_000_000),
        open=price,
        high=price,
        low=price,
        close=price,
    )


def _bar(local_start: datetime, price: Decimal) -> evaluator.LeoCompleted15mBar:
    event = local_start.astimezone(UTC)
    bid = PriceBar(*(price - Decimal("0.0001") for _ in range(4)))
    ask = PriceBar(*(price + Decimal("0.0001") for _ in range(4)))
    return evaluator.LeoCompleted15mBar(
        "GBP/USD.DUKASCOPY",
        event,
        event + timedelta(minutes=15),
        event + timedelta(minutes=15),
        bid,
        ask,
    )


def _price_for(index: int) -> Decimal:
    if index == -15:
        return Decimal("1.3000")
    if index == -14:
        return Decimal("1.2900")
    if index == 9 * 60:
        return Decimal("1.3010")
    if index == 9 * 60 + 14:
        return Decimal("1.2990")
    if index == 9 * 60 + 29:
        return Decimal("1.2980")
    return Decimal("1.2950")
