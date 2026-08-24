from __future__ import annotations

import ast
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pytest

from ftmoquant.research.alpha_lab.batch4_clock_scheduler import (
    ScheduledOccurrence,
    load_frozen_clock_specs,
    schedule_occurrence,
)
from ftmoquant.research.alpha_lab.batch4_execution import (
    REFERENCE_NOTIONAL_USD,
    Batch4ExecutionError,
    execute_scheduled_occurrences,
)


def _occurrence(spec_id: str, day: date = date(2026, 1, 15)) -> ScheduledOccurrence:
    spec = next(
        spec for spec in load_frozen_clock_specs() if spec.hypothesis_id == spec_id
    )
    return schedule_occurrence(spec, day)


def _frames(
    timestamps: list[pd.Timestamp],
    bids: list[float],
    asks: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.DatetimeIndex(timestamps)

    def frame(values: list[float]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "open": values,
                "high": values,
                "low": values,
                "close": values,
            },
            index=index,
        )

    return frame(bids), frame(asks)


def _decision_fixture(
    occurrence: ScheduledOccurrence,
    *,
    entry_bid: float = 1.2,
    entry_ask: float = 1.2002,
    exit_bid: float = 1.21,
    exit_ask: float = 1.2102,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    timestamps = [
        pd.Timestamp(occurrence.scheduled_entry_utc),
        pd.Timestamp(occurrence.scheduled_entry_utc + timedelta(minutes=1)),
        pd.Timestamp(occurrence.scheduled_exit_utc),
        pd.Timestamp(occurrence.scheduled_exit_utc + timedelta(minutes=1)),
    ]
    return _frames(
        timestamps,
        [entry_bid - 0.01, entry_bid, exit_bid - 0.01, exit_bid],
        [entry_ask - 0.01, entry_ask, exit_ask - 0.01, exit_ask],
    )


def test_sell_uses_bid_entry_ask_exit_and_strictly_later_observations() -> None:
    occurrence = _occurrence("B4F1A_EUR")
    bid, ask = _decision_fixture(occurrence)
    trades, skips = execute_scheduled_occurrences([occurrence], bid_m1=bid, ask_m1=ask)
    assert not skips
    trade = trades[0]
    assert trade.direction == "SELL"
    assert trade.entry_price == Decimal("1.2")
    assert trade.exit_price == Decimal("1.2102")
    assert trade.actual_entry_utc == occurrence.scheduled_entry_utc + timedelta(
        minutes=1
    )
    assert trade.actual_exit_utc == occurrence.scheduled_exit_utc + timedelta(minutes=1)
    assert trade.exit_reason == "scheduled_time_exit"


def test_buy_uses_ask_entry_bid_exit() -> None:
    occurrence = _occurrence("B4F1A_CAD")
    bid, ask = _decision_fixture(occurrence, entry_bid=1.35, entry_ask=1.3502)
    trades, skips = execute_scheduled_occurrences([occurrence], bid_m1=bid, ask_m1=ask)
    assert not skips
    assert trades[0].entry_price == Decimal("1.3502")
    assert trades[0].exit_price == Decimal("1.21")


def test_missing_entry_is_an_explicit_skip() -> None:
    occurrence = _occurrence("B4F1A_EUR")
    timestamp = pd.Timestamp(occurrence.scheduled_entry_utc - timedelta(minutes=1))
    bid, ask = _frames([timestamp], [1.0], [1.0002])
    trades, skips = execute_scheduled_occurrences([occurrence], bid_m1=bid, ask_m1=ask)
    assert not trades
    assert skips[0].reason == "no_entry_observation"
    assert skips[0].relevant_scheduled_utc == occurrence.scheduled_entry_utc


def test_missing_exit_is_an_explicit_fail_closed_skip() -> None:
    occurrence = _occurrence("B4F1A_EUR")
    timestamp = pd.Timestamp(occurrence.scheduled_entry_utc + timedelta(minutes=1))
    bid, ask = _frames([timestamp], [1.0], [1.0002])
    trades, skips = execute_scheduled_occurrences([occurrence], bid_m1=bid, ask_m1=ask)
    assert not trades
    assert skips[0].reason == "no_exit_observation"
    assert skips[0].relevant_scheduled_utc == occurrence.scheduled_exit_utc


def test_no_next_local_day_exit_rescue() -> None:
    occurrence = _occurrence("B4F1A_EUR")
    timestamps = [
        pd.Timestamp(occurrence.scheduled_entry_utc + timedelta(minutes=1)),
        pd.Timestamp(occurrence.scheduled_exit_utc + timedelta(days=1, minutes=1)),
    ]
    bid, ask = _frames(timestamps, [1.0, 1.1], [1.0002, 1.1002])
    trades, skips = execute_scheduled_occurrences([occurrence], bid_m1=bid, ask_m1=ask)
    assert not trades
    assert skips[0].reason == "exit_outside_local_date"


def test_entry_after_window_is_not_rescued() -> None:
    occurrence = _occurrence("B4F1A_EUR")
    timestamp = pd.Timestamp(occurrence.scheduled_exit_utc + timedelta(minutes=1))
    bid, ask = _frames([timestamp], [1.0], [1.0002])
    trades, skips = execute_scheduled_occurrences([occurrence], bid_m1=bid, ask_m1=ask)
    assert not trades
    assert skips[0].reason == "entry_not_before_scheduled_exit"


def test_fixed_100k_sizing_and_quote_usd_pnl() -> None:
    occurrence = _occurrence("B4F1A_EUR")
    bid, ask = _decision_fixture(
        occurrence,
        entry_bid=1.25,
        entry_ask=1.2502,
        exit_bid=1.24,
        exit_ask=1.24,
    )
    trades, _ = execute_scheduled_occurrences([occurrence], bid_m1=bid, ask_m1=ask)
    trade = trades[0]
    assert trade.reference_notional_usd == REFERENCE_NOTIONAL_USD
    assert trade.quantity == Decimal("80000")
    assert trade.pnl_account_currency == Decimal("800")
    assert trade.return_on_reference_notional == Decimal("0.008")
    assert trade.account_currency == "USD"


@pytest.mark.parametrize(
    ("spec_id", "entry", "exit_price"),
    [
        ("B4F1A_CAD", 1.25, 1.26),
        ("B4F1A_CHF", 0.8, 0.81),
        ("B4F1A_JPY", 150.0, 151.0),
    ],
)
def test_base_usd_pairs_use_100k_units_and_convert_quote_pnl_once(
    spec_id: str, entry: float, exit_price: float
) -> None:
    occurrence = _occurrence(spec_id)
    bid, ask = _decision_fixture(
        occurrence,
        entry_bid=entry - 0.0002,
        entry_ask=entry,
        exit_bid=exit_price,
        exit_ask=exit_price + 0.0002,
    )
    trades, _ = execute_scheduled_occurrences([occurrence], bid_m1=bid, ask_m1=ask)
    trade = trades[0]
    assert trade.quantity == Decimal("100000")
    expected = (
        Decimal("100000")
        * (Decimal(str(exit_price)) - Decimal(str(entry)))
        / Decimal(str(entry))
    )
    assert trade.pnl_account_currency == expected


def test_stress_keeps_schedule_and_uses_stressed_executable_sides() -> None:
    occurrence = _occurrence("B4F1A_CAD")
    bid, ask = _decision_fixture(occurrence, entry_bid=1.2, entry_ask=1.2004)
    outputs = [
        execute_scheduled_occurrences(
            [occurrence],
            bid_m1=bid,
            ask_m1=ask,
            cost_stress_multiplier=multiplier,
        )[0][0]
        for multiplier in (Decimal("1.0"), Decimal("1.5"), Decimal("2.0"))
    ]
    assert {row.scheduled_entry_utc for row in outputs} == {
        occurrence.scheduled_entry_utc
    }
    assert {row.scheduled_exit_utc for row in outputs} == {
        occurrence.scheduled_exit_utc
    }
    assert {row.actual_entry_utc for row in outputs} == {
        occurrence.scheduled_entry_utc + timedelta(minutes=1)
    }
    assert outputs[0].entry_price < outputs[1].entry_price < outputs[2].entry_price
    assert outputs[0].exit_price > outputs[1].exit_price > outputs[2].exit_price


def test_invalid_stress_multiplier_fails_closed() -> None:
    occurrence = _occurrence("B4F1A_EUR")
    bid, ask = _decision_fixture(occurrence)
    with pytest.raises(Batch4ExecutionError, match="cost stress"):
        execute_scheduled_occurrences(
            [occurrence],
            bid_m1=bid,
            ask_m1=ask,
            cost_stress_multiplier=Decimal("1.2"),
        )


def test_overlapping_distinct_hypotheses_are_independent() -> None:
    original = _occurrence("B4F1A_GBP")
    second_spec = next(
        spec
        for spec in load_frozen_clock_specs()
        if spec.hypothesis_id == "B4F1B_london_fix_flow_reversal:PRE_60m:GBP/USD.OANDA"
    )
    second = schedule_occurrence(second_spec, original.local_date)
    timestamps = [
        pd.Timestamp(original.scheduled_entry_utc + timedelta(minutes=1)),
        pd.Timestamp(second.scheduled_entry_utc + timedelta(minutes=1)),
        pd.Timestamp(original.scheduled_exit_utc + timedelta(minutes=1)),
    ]
    bid, ask = _frames(
        timestamps,
        [1.2, 1.21, 1.22],
        [1.2002, 1.2102, 1.2202],
    )
    trades, skips = execute_scheduled_occurrences(
        [original, second], bid_m1=bid, ask_m1=ask
    )
    assert len(trades) == 2
    assert not skips


def test_runtime_direction_or_clock_override_is_rejected() -> None:
    occurrence = _occurrence("B4F1A_EUR")
    tampered = ScheduledOccurrence(
        hypothesis_id=occurrence.hypothesis_id,
        family=occurrence.family,
        instrument_id=occurrence.instrument_id,
        direction="BUY",
        timezone=occurrence.timezone,
        local_date=occurrence.local_date,
        scheduled_entry_utc=occurrence.scheduled_entry_utc,
        scheduled_exit_utc=occurrence.scheduled_exit_utc,
    )
    bid, ask = _decision_fixture(occurrence)
    with pytest.raises(Batch4ExecutionError, match="does not match"):
        execute_scheduled_occurrences([tampered], bid_m1=bid, ask_m1=ask)


def test_duplicate_same_hypothesis_occurrence_is_rejected() -> None:
    occurrence = _occurrence("B4F1A_EUR")
    bid, ask = _decision_fixture(occurrence)
    with pytest.raises(Batch4ExecutionError, match="duplicate"):
        execute_scheduled_occurrences([occurrence, occurrence], bid_m1=bid, ask_m1=ask)


def test_execution_is_deterministic() -> None:
    occurrence = _occurrence("B4F1A_EUR")
    bid, ask = _decision_fixture(occurrence)
    first = execute_scheduled_occurrences([occurrence], bid_m1=bid, ask_m1=ask)
    second = execute_scheduled_occurrences([occurrence], bid_m1=bid, ask_m1=ask)
    assert first == second


def test_malformed_or_unpaired_frames_fail_closed() -> None:
    occurrence = _occurrence("B4F1A_EUR")
    bid, ask = _decision_fixture(occurrence)
    with pytest.raises(Batch4ExecutionError, match="identical paired index"):
        execute_scheduled_occurrences([occurrence], bid_m1=bid.iloc[:-1], ask_m1=ask)


def test_execution_module_has_no_real_loader_or_partition_access() -> None:
    import ftmoquant.research.alpha_lab.batch4_execution as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert all("development" not in name for name in imported_modules)
    assert all("validation" not in name for name in imported_modules)
    for token in (
        "ParquetDataCatalog",
        "load_alpha_lab_dataset",
        "load_validation_dataset",
        "oanda_alpha_lab_development",
        "oanda_alpha_lab_validation",
        "holdout",
    ):
        assert token not in source.lower()
