import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from ftmoquant.research.stage_g import InstrumentBarObservation, SynchronizedClockFrame
from ftmoquant.research.usd_macro_surprise_momentum_development import (
    EventResult,
    MacroEvent,
    _fold_for,
    _summary,
    evaluate_macro_events,
    load_development_macro_events,
)
from ftmoquant.research.usd_macro_surprise_momentum_spec import (
    USD_MACRO_SURPRISE_MOMENTUM_CONFIG_SHA256,
)


def _event(at: datetime, actual: str, forecast: str = "0") -> MacroEvent:
    return MacroEvent(
        at, "US_NFP_HEADLINE_EMPLOYMENT_CHANGE", Decimal(actual), Decimal(forecast), 1
    )


def _frame(at: datetime, *, complete: bool = True) -> SynchronizedClockFrame:
    observations = tuple(
        InstrumentBarObservation(
            instrument, at, at, Decimal("1.0000"), Decimal("1.0010")
        )
        for instrument in ("EUR/USD.DUKASCOPY", "GBP/USD.DUKASCOPY")
    )
    return SynchronizedClockFrame(
        at, at, observations if complete else (observations[0], None), True
    )


def test_surprise_direction_timing_and_bid_ask_execution() -> None:
    event_time = datetime(2020, 5, 1, 10, tzinfo=UTC)
    entry = _frame(event_time + timedelta(minutes=5))
    exit = _frame(event_time + timedelta(minutes=60))
    positive, negative, zero = evaluate_macro_events(
        (
            _event(event_time, "1"),
            _event(event_time + timedelta(hours=2), "-1"),
            _event(event_time + timedelta(hours=4), "0"),
        ),
        (
            entry,
            exit,
            _frame(event_time + timedelta(hours=2, minutes=5)),
            _frame(event_time + timedelta(hours=2, minutes=60)),
        ),
    )
    assert positive.direction == "USD_positive"
    assert positive.status == "executable"
    assert negative.direction == "USD_negative"
    assert negative.status == "executable"
    assert zero.status == "no_trade"


def test_both_pairs_required_missing_and_overlap_are_deterministic() -> None:
    event_time = datetime(2020, 5, 1, 10, tzinfo=UTC)
    missing = evaluate_macro_events(
        (_event(event_time, "1"),),
        (
            _frame(event_time + timedelta(minutes=5), complete=False),
            _frame(event_time + timedelta(minutes=60), complete=False),
        ),
    )[0]
    assert missing.reason == "both_pairs_required_missing_observation"
    overlap = evaluate_macro_events(
        (_event(event_time, "1"), _event(event_time + timedelta(minutes=30), "1")),
        tuple(
            _frame(event_time + timedelta(minutes=offset)) for offset in (5, 35, 60, 90)
        ),
    )
    assert overlap[1].reason == "overlap_same_instrument"


def test_event_loader_excludes_warmup_and_retains_each_comparison_fold(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    rows = [
        _raw("2019-03-11T00:00:00Z"),
        _raw("2023-04-11T00:00:00Z"),
        _raw("2020-05-01T10:00:00Z"),
        _raw("2021-05-01T10:00:00Z", family="US_CPI_HEADLINE_M_M"),
        _raw("2022-05-01T10:00:00Z"),
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    events = load_development_macro_events(path)
    assert [item.event_family for item in events] == [
        "US_NFP_HEADLINE_EMPLOYMENT_CHANGE",
        "US_CPI_HEADLINE_M_M",
        "US_NFP_HEADLINE_EMPLOYMENT_CHANGE",
    ]
    assert [item.timestamp_utc.year for item in events] == [2020, 2021, 2022]
    assert [_fold_for(item.timestamp_utc).fold_id for item in events] == [
        "dev_fold_1",
        "dev_fold_2",
        "dev_fold_3",
    ]
    assert USD_MACRO_SURPRISE_MOMENTUM_CONFIG_SHA256 == (
        "ce997472bfd600d3411dd7c30a9d2df04bce353c1481a3d3eab0d5efb6d9df66"
    )


def test_event_level_summary_gate_and_bootstrap_are_deterministic() -> None:
    rows = tuple(
        EventResult(
            f"202{index}-05-01T10:00:00Z",
            "US_NFP_HEADLINE_EMPLOYMENT_CHANGE",
            f"dev_fold_{index + 1}",
            "USD_positive",
            0.01,
            0.009,
            "executable",
            None,
        )
        for index in range(3)
    )
    first = _summary(rows)
    assert first["decision"] == "PASS_DEVELOPMENT"
    assert first["event_level_bootstrap_95_ci"]["observation_count"] == 3
    assert _summary(rows) == first
    rejected = _summary(
        rows[:1]
        + (
            EventResult(
                "2021-05-01T10:00:00Z",
                "US_NFP_HEADLINE_EMPLOYMENT_CHANGE",
                "dev_fold_2",
                "USD_positive",
                -0.01,
                -0.011,
                "executable",
                None,
            ),
            EventResult(
                "2022-05-01T10:00:00Z",
                "US_NFP_HEADLINE_EMPLOYMENT_CHANGE",
                "dev_fold_3",
                "USD_positive",
                -0.01,
                -0.011,
                "executable",
                None,
            ),
        )
    )
    assert rejected["decision"] == "REJECT_RETIRE"


def _raw(
    timestamp: str, family: str = "US_NFP_HEADLINE_EMPLOYMENT_CHANGE"
) -> dict[str, object]:
    return {
        "timestamp_utc": timestamp,
        "event_family": family,
        "actual_parsed": {"status": "parsed", "value": "1"},
        "forecast_parsed": {"status": "parsed", "value": "0"},
        "source_row_number": 1,
    }
