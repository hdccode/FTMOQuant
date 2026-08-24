from __future__ import annotations

import ast
import json
from datetime import date, time
from pathlib import Path

import pytest

from ftmoquant.research.alpha_lab.batch4_clock_scheduler import (
    EXPECTED_PREREGISTRATION_SEMANTIC_SHA256,
    Batch4ClockError,
    FrozenClockSpec,
    generate_occurrences,
    load_frozen_clock_specs,
    schedule_occurrence,
)
from ftmoquant.research.alpha_lab.batch4_preregistration import _canonical_sha256


def _by_id() -> dict[str, FrozenClockSpec]:
    return {spec.hypothesis_id: spec for spec in load_frozen_clock_specs()}


def test_preregistration_identity_and_exact_91_spec_grid() -> None:
    specs = load_frozen_clock_specs()
    assert EXPECTED_PREREGISTRATION_SEMANTIC_SHA256 == (
        "4a019140ab798cdccc14ba3c6a0817dfca10e4d626de58b95d1c5c0d7c01dd98"
    )
    assert len(specs) == 91
    assert len({spec.hypothesis_id for spec in specs}) == 91
    assert sum(spec.family.startswith("B4F1A") for spec in specs) == 7
    assert sum(spec.family.startswith("B4F1B") for spec in specs) == 42
    assert sum(spec.family.startswith("B4F1C") for spec in specs) == 42


def test_identity_check_rejects_self_consistent_but_nonfrozen_artifact(
    tmp_path: Path,
) -> None:
    source = Path(
        "config/research/batch4_structural_intraday_flow_preregistration_v1.json"
    )
    document = json.loads(source.read_text(encoding="utf-8"))
    document["purpose"] = "tampered but rehashed"
    document["preregistration_semantic_sha256"] = _canonical_sha256(document)
    path = tmp_path / "rehashed.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(Batch4ClockError, match="identity mismatch"):
        load_frozen_clock_specs(path)


def test_local_sleeves_are_derived_exactly_from_artifact() -> None:
    specs = _by_id()
    observed = {
        key: (
            spec.instrument_id,
            spec.timezone,
            spec.local_start_time.isoformat(timespec="minutes"),
            spec.local_end_time.isoformat(timespec="minutes"),
            spec.direction,
        )
        for key, spec in specs.items()
        if key.startswith("B4F1A_")
    }
    assert observed == {
        "B4F1A_AUD": (
            "AUD/USD.OANDA",
            "Australia/Sydney",
            "09:00",
            "17:00",
            "SELL",
        ),
        "B4F1A_EUR": (
            "EUR/USD.OANDA",
            "Europe/Berlin",
            "08:00",
            "16:00",
            "SELL",
        ),
        "B4F1A_GBP": (
            "GBP/USD.OANDA",
            "Europe/London",
            "08:00",
            "16:00",
            "SELL",
        ),
        "B4F1A_NZD": (
            "NZD/USD.OANDA",
            "Pacific/Auckland",
            "09:00",
            "17:00",
            "SELL",
        ),
        "B4F1A_CAD": (
            "USD/CAD.OANDA",
            "America/Toronto",
            "08:00",
            "16:00",
            "BUY",
        ),
        "B4F1A_CHF": (
            "USD/CHF.OANDA",
            "Europe/Zurich",
            "08:00",
            "16:00",
            "BUY",
        ),
        "B4F1A_JPY": (
            "USD/JPY.OANDA",
            "Asia/Tokyo",
            "09:00",
            "17:00",
            "BUY",
        ),
    }


def test_fix_windows_and_signs_are_derived_without_overrides() -> None:
    specs = load_frozen_clock_specs()
    london_eur = {
        spec.hypothesis_id.split(":")[1]: spec
        for spec in specs
        if spec.family.startswith("B4F1B") and spec.instrument_id == "EUR/USD.OANDA"
    }
    assert {
        key: (
            spec.local_start_time.isoformat(timespec="minutes"),
            spec.local_end_time.isoformat(timespec="minutes"),
            spec.direction,
        )
        for key, spec in london_eur.items()
    } == {
        "PRE_15m": ("15:45", "16:00", "SELL"),
        "PRE_30m": ("15:30", "16:00", "SELL"),
        "PRE_60m": ("15:00", "16:00", "SELL"),
        "POST_15m": ("16:00", "16:15", "BUY"),
        "POST_30m": ("16:00", "16:30", "BUY"),
        "POST_60m": ("16:00", "17:00", "BUY"),
    }
    tokyo_jpy = {
        spec.hypothesis_id.split(":")[1]: spec
        for spec in specs
        if spec.family.startswith("B4F1C") and spec.instrument_id == "USD/JPY.OANDA"
    }
    assert {
        key: (
            spec.local_start_time.isoformat(timespec="minutes"),
            spec.local_end_time.isoformat(timespec="minutes"),
            spec.direction,
        )
        for key, spec in tokyo_jpy.items()
    } == {
        "PRE_15m": ("09:40", "09:55", "BUY"),
        "PRE_30m": ("09:25", "09:55", "BUY"),
        "PRE_60m": ("08:55", "09:55", "BUY"),
        "POST_15m": ("09:55", "10:10", "SELL"),
        "POST_30m": ("09:55", "10:25", "SELL"),
        "POST_60m": ("09:55", "10:55", "SELL"),
    }


@pytest.mark.parametrize(
    ("spec_id", "winter", "summer", "winter_hour", "summer_hour"),
    [
        ("B4F1A_GBP", date(2026, 1, 15), date(2026, 7, 15), 8, 7),
        ("B4F1A_EUR", date(2026, 1, 15), date(2026, 7, 15), 7, 6),
        ("B4F1A_CAD", date(2026, 1, 15), date(2026, 7, 15), 13, 12),
        ("B4F1A_AUD", date(2026, 1, 15), date(2026, 7, 15), 22, 23),
        ("B4F1A_NZD", date(2026, 1, 15), date(2026, 7, 15), 20, 21),
    ],
)
def test_dst_changes_utc_not_frozen_local_clock(
    spec_id: str,
    winter: date,
    summer: date,
    winter_hour: int,
    summer_hour: int,
) -> None:
    spec = _by_id()[spec_id]
    winter_occurrence = schedule_occurrence(spec, winter)
    summer_occurrence = schedule_occurrence(spec, summer)
    assert winter_occurrence.scheduled_entry_utc.hour == winter_hour
    assert summer_occurrence.scheduled_entry_utc.hour == summer_hour
    assert (
        winter_occurrence.scheduled_entry_utc.astimezone(
            __import__("zoneinfo").ZoneInfo(spec.timezone)
        ).time()
        == spec.local_start_time
    )
    assert (
        summer_occurrence.scheduled_entry_utc.astimezone(
            __import__("zoneinfo").ZoneInfo(spec.timezone)
        ).time()
        == spec.local_start_time
    )


@pytest.mark.parametrize(
    ("spec_id", "transition_start"),
    [
        ("B4F1A_GBP", date(2026, 3, 27)),
        ("B4F1A_GBP", date(2026, 10, 23)),
        ("B4F1A_EUR", date(2026, 3, 27)),
        ("B4F1A_CAD", date(2026, 3, 6)),
        ("B4F1A_AUD", date(2026, 4, 3)),
        ("B4F1A_NZD", date(2026, 4, 3)),
    ],
)
def test_transition_ranges_have_one_unique_occurrence_per_local_date(
    spec_id: str, transition_start: date
) -> None:
    spec = _by_id()[spec_id]
    end = date.fromordinal(transition_start.toordinal() + 5)
    occurrences = generate_occurrences(spec, transition_start, end)
    assert len(occurrences) == 5
    assert len({row.local_date for row in occurrences}) == 5
    assert len({row.scheduled_entry_utc for row in occurrences}) == 5
    assert all(row.scheduled_exit_utc > row.scheduled_entry_utc for row in occurrences)


def test_tokyo_0955_is_exact_and_has_no_dst_shift() -> None:
    spec = next(
        spec
        for spec in load_frozen_clock_specs()
        if spec.family.startswith("B4F1C")
        and ":POST_15m:" in spec.hypothesis_id
        and spec.instrument_id == "USD/JPY.OANDA"
    )
    winter = schedule_occurrence(spec, date(2026, 1, 15))
    summer = schedule_occurrence(spec, date(2026, 7, 15))
    assert (winter.scheduled_entry_utc.hour, winter.scheduled_entry_utc.minute) == (
        0,
        55,
    )
    assert (summer.scheduled_entry_utc.hour, summer.scheduled_entry_utc.minute) == (
        0,
        55,
    )


def test_weekends_are_scheduled_without_market_data() -> None:
    spec = _by_id()["B4F1A_GBP"]
    rows = generate_occurrences(spec, date(2026, 8, 22), date(2026, 8, 24))
    assert [row.local_date.weekday() for row in rows] == [5, 6]


def test_midnight_spanning_or_runtime_direction_construction_fails_closed() -> None:
    with pytest.raises(Batch4ClockError, match="same-day"):
        FrozenClockSpec(
            "bad",
            "bad",
            "EUR/USD.OANDA",
            "BUY",
            "Europe/London",
            time(23),
            time(1),
        )


def test_scheduler_has_no_loader_or_price_dependency() -> None:
    import ftmoquant.research.alpha_lab.batch4_clock_scheduler as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    roots = {
        (node.module or "").split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert "pandas" not in roots
    for token in (
        "ParquetDataCatalog",
        "load_alpha_lab_dataset",
        "load_validation_dataset",
        "oanda_alpha_lab_development",
        "oanda_alpha_lab_validation",
    ):
        assert token not in source
