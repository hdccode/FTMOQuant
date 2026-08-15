from __future__ import annotations

from copy import deepcopy
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import yaml

from ftmoquant.research.leo_gbpusd_spec import (
    LEO_GBPUSD_CONFIG_SHA256,
    LEO_GBPUSD_SPEC_PATH,
    LeoGbpUsdSpecValidationError,
    leo_gbpusd_config_sha256,
    load_leo_gbpusd_spec,
)

LONDON = ZoneInfo("Europe/London")


def test_frozen_spec_is_gbpusd_15m_dst_aware_and_unevaluated() -> None:
    spec = load_leo_gbpusd_spec()

    assert (
        spec.strategy_id,
        spec.status,
        spec.instrument_id,
        spec.bar_timeframe,
        spec.session_timezone,
    ) == (
        "leo_gbpusd_v1",
        "specified_not_run",
        "GBP/USD.DUKASCOPY",
        "15m",
        "Europe/London",
    )
    assert leo_gbpusd_config_sha256(spec) == LEO_GBPUSD_CONFIG_SHA256
    assert spec.canonical_document["research_boundary"] == {
        "validation": "locked",
        "final_holdout": "locked",
        "strategy_returns_accessed": False,
    }


def test_spec_rejects_any_session_or_filter_change(tmp_path: Path) -> None:
    document = _document()
    document["sessions"]["london_entry"]["window"] = (
        "08:00 <= London completed signal bar end < 11:00"
    )
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(LeoGbpUsdSpecValidationError, match="frozen v1"):
        load_leo_gbpusd_spec(path)


@pytest.mark.parametrize(
    ("day", "expected_utc_hour"),
    [(date(2020, 1, 6), 8), (date(2020, 7, 6), 7)],
)
def test_london_session_boundaries_use_iana_dst(
    day: date, expected_utc_hour: int
) -> None:
    asia_end = _local(day, time(8))
    assert asia_end.astimezone(UTC).hour == expected_utc_hour
    assert _session_at(asia_end - timedelta(minutes=15)) == "asia"
    assert _session_at(asia_end) == "london_reference"
    assert _entry_session_at(_local(day, time(9))) == "london"
    assert _entry_session_at(_local(day, time(11))) is None
    assert _entry_session_at(_local(day, time(14))) == "new_york"
    assert _entry_session_at(_local(day, time(16))) is None


def test_sweep_rejection_is_strict_and_entry_is_strictly_later() -> None:
    harness = _SyntheticSessionSemantics()
    signal_at = _local(date(2020, 7, 6), time(9, 15))

    # Upper sweep returns inside: short; equal high/close values are not signals.
    assert (
        harness.signal(
            "london",
            signal_at,
            high="1.3010",
            low="1.2990",
            close="1.2995",
            reference_high="1.3000",
            reference_low="1.2900",
        )
        == "short"
    )
    assert (
        harness.signal(
            "london",
            signal_at,
            high="1.3000",
            low="1.2990",
            close="1.2995",
            reference_high="1.3000",
            reference_low="1.2900",
        )
        is None
    )
    assert (
        harness.signal(
            "london",
            signal_at,
            high="1.3010",
            low="1.2990",
            close="1.3000",
            reference_high="1.3000",
            reference_low="1.2900",
        )
        is None
    )

    pending = harness.pending
    assert pending is not None
    assert harness.enter(signal_at, "1.2994") is None
    trade = harness.enter(signal_at + timedelta(seconds=1), "1.2994")
    assert trade is not None
    assert trade.direction == "short"
    assert trade.entry_time > pending.signal_time


def test_3r_stop_calculation_one_trade_per_session_and_forced_exit() -> None:
    harness = _SyntheticSessionSemantics()
    day = date(2020, 7, 6)
    signal_at = _local(day, time(9, 15))
    assert (
        harness.signal(
            "london",
            signal_at,
            high="1.3010",
            low="1.2990",
            close="1.2995",
            reference_high="1.3000",
            reference_low="1.2900",
        )
        == "short"
    )
    trade = harness.enter(signal_at + timedelta(seconds=1), "1.2990")
    assert trade is not None
    assert trade.stop == Decimal("1.3010")
    assert trade.target == Decimal("1.2930")

    # An overlapping signal is discarded while the position is open; after exit,
    # this named session's one permitted entry remains consumed.
    assert (
        harness.signal(
            "london",
            _local(day, time(9, 30)),
            high="1.3020",
            low="1.2990",
            close="1.2990",
            reference_high="1.3000",
            reference_low="1.2900",
        )
        is None
    )
    assert harness.force_exit(_local(day, time(11))) == "session_window_end"
    assert (
        harness.signal(
            "london",
            _local(day, time(10, 45)),
            high="1.3010",
            low="1.2990",
            close="1.2995",
            reference_high="1.3000",
            reference_low="1.2900",
        )
        is None
    )


def _document() -> dict[str, object]:
    value = yaml.safe_load(LEO_GBPUSD_SPEC_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return deepcopy(value)


def _local(day: date, value: time) -> datetime:
    return datetime.combine(day, value, tzinfo=LONDON)


def _session_at(bar_end: datetime) -> str | None:
    local = bar_end.astimezone(LONDON).time()
    if time(0) <= local < time(8):
        return "asia"
    if time(8) <= local < time(13):
        return "london_reference"
    return None


def _entry_session_at(bar_end: datetime) -> str | None:
    local = bar_end.astimezone(LONDON).time()
    if time(9) <= local < time(11):
        return "london"
    if time(14) <= local < time(16):
        return "new_york"
    return None


class _PendingSweep:
    def __init__(
        self, direction: str, signal_time: datetime, sweep_extreme: Decimal
    ) -> None:
        self.direction = direction
        self.signal_time = signal_time
        self.sweep_extreme = sweep_extreme


class _SyntheticTrade:
    def __init__(
        self, direction: str, entry_time: datetime, stop: Decimal, target: Decimal
    ) -> None:
        self.direction = direction
        self.entry_time = entry_time
        self.stop = stop
        self.target = target


class _SyntheticSessionSemantics:
    """Test-only fixture for the frozen semantics; not a trading strategy."""

    def __init__(self) -> None:
        self.pending: _PendingSweep | None = None
        self.trade: _SyntheticTrade | None = None
        self.consumed_sessions: set[tuple[date, str]] = set()

    def signal(
        self,
        session: str,
        at: datetime,
        *,
        high: str,
        low: str,
        close: str,
        reference_high: str,
        reference_low: str,
    ) -> str | None:
        if (
            self.pending is not None
            or self.trade is not None
            or (at.date(), session) in self.consumed_sessions
        ):
            return None
        hi, lo, end, ref_hi, ref_lo = map(
            Decimal, (high, low, close, reference_high, reference_low)
        )
        if hi > ref_hi and end < ref_hi:
            self.pending = _PendingSweep("short", at, hi)
        elif lo < ref_lo and end > ref_lo:
            self.pending = _PendingSweep("long", at, lo)
        else:
            return None
        return self.pending.direction

    def enter(self, at: datetime, price: str) -> _SyntheticTrade | None:
        if self.pending is None or at <= self.pending.signal_time:
            return None
        entry = Decimal(price)
        distance = abs(entry - self.pending.sweep_extreme)
        if distance == 0:
            return None
        target = (
            entry - Decimal(3) * distance
            if self.pending.direction == "short"
            else entry + Decimal(3) * distance
        )
        self.trade = _SyntheticTrade(
            self.pending.direction, at, self.pending.sweep_extreme, target
        )
        self.consumed_sessions.add(
            (
                self.pending.signal_time.date(),
                _entry_session_at(self.pending.signal_time) or "",
            )
        )
        self.pending = None
        return self.trade

    def force_exit(self, at: datetime) -> str | None:
        if self.trade is None or _entry_session_at(at) is not None:
            return None
        self.trade = None
        return "session_window_end"
