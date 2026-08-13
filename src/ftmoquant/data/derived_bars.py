"""Deterministic Nautilus composite EUR/USD higher-timeframe bars."""

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

from nautilus_trader.backtest import BacktestEngine, BacktestEngineConfig
from nautilus_trader.common import DataActor, DataActorConfig, LoggerConfig, LogLevel
from nautilus_trader.data import DataEngineConfig
from nautilus_trader.model import (
    AccountType,
    Bar,
    BarIntervalType,
    BarType,
    BookType,
    Currency,
    CurrencyPair,
    Money,
    OmsType,
    Venue,
)
from nautilus_trader.persistence import ParquetDataCatalog

from ftmoquant.data.dukascopy import INSTRUMENT_ID, NAUTILUS_VERSION

DERIVATION_VERSION = "g0.6-1"
CONFIG_VERSION = 1
DERIVED_MANIFEST_FILENAME = "ftmoquant_derived_provenance.json"
PARENT_MANIFEST_FILENAME = "ftmoquant_provenance.json"

_MINUTE_NS = 60_000_000_000
_BUILD_DELAY_MICROSECONDS = 1
_BUILD_DELAY_NS = _BUILD_DELAY_MICROSECONDS * 1_000
_TARGETS = ((1, 60), (4, 240))
_SIDES = ("BID", "ASK")


class DerivationValidationError(ValueError):
    """Raised before writes when higher-timeframe output is not trustworthy."""


@dataclass(frozen=True, slots=True)
class DroppedWindow:
    """One aligned window which was not eligible for derived research data."""

    start_utc: str
    end_utc: str
    expected_minutes: int
    observed_minutes: int
    reason: str


@dataclass(frozen=True, slots=True)
class SeriesResult:
    """QA and output for one target/price-side series."""

    source_bar_type: str
    composite_bar_type: str
    target_bar_type: str
    source_bar_count: int
    eligible_source_bar_count: int
    emitted_bars: tuple[Bar, ...]
    dropped_windows: tuple[DroppedWindow, ...]
    callback_timestamps_ns: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DerivationResult:
    """Stable locations and counts from one G0.6 derivation."""

    manifest_path: Path
    catalog_path: Path
    emitted_counts: dict[str, int]
    research_ready: bool


class _BarCollector(DataActor):
    def __init__(self, composite_bar_type: BarType) -> None:
        super().__init__(DataActorConfig(log_events=False, log_commands=False))
        self.composite_bar_type = composite_bar_type
        self.bars: list[Bar] = []
        self.callback_timestamps_ns: list[int] = []

    def on_start(self) -> None:
        self.subscribe_bars(self.composite_bar_type)

    def on_bar(self, bar: Bar) -> None:
        self.bars.append(bar)
        self.callback_timestamps_ns.append(self.clock.timestamp_ns())


def derive_eurusd_bars(output_root: Path) -> DerivationResult:
    """Derive complete UTC 1H/4H BID/ASK bars from the G0.5 catalog."""

    root = output_root.resolve()
    parent_manifest_path = root / PARENT_MANIFEST_FILENAME
    catalog_path = root / "catalog"
    _validate_runtime(parent_manifest_path, catalog_path)
    parent_manifest = _load_parent_manifest(parent_manifest_path)
    parent_sha256 = _sha256(parent_manifest_path)
    requested_start_ns, requested_end_ns = _requested_range_ns(parent_manifest)

    catalog = ParquetDataCatalog(str(catalog_path))
    instruments = catalog.instruments([INSTRUMENT_ID])
    if len(instruments) != 1 or not isinstance(instruments[0], CurrencyPair):
        raise DerivationValidationError(
            f"catalog must contain exactly one CurrencyPair for {INSTRUMENT_ID}"
        )
    instrument = instruments[0]

    sources = {
        side: tuple(catalog.query_bars([_source_bar_type(side)])) for side in _SIDES
    }
    _validate_sources(sources, parent_manifest)

    series: list[SeriesResult] = []
    for hours, expected_minutes in _TARGETS:
        for side in _SIDES:
            eligible, dropped = _eligible_source_bars(
                sources[side],
                hours,
                expected_minutes,
                range_start_ns=requested_start_ns,
                range_end_ns=requested_end_ns,
            )
            emitted, callbacks = _aggregate_native(
                instrument=instrument,
                source_bars=eligible,
                side=side,
                hours=hours,
            )
            _validate_native_output(
                emitted=emitted,
                callback_timestamps_ns=callbacks,
                source_bars=eligible,
                hours=hours,
            )
            series.append(
                SeriesResult(
                    source_bar_type=_source_bar_type(side),
                    composite_bar_type=_composite_bar_type(hours, side),
                    target_bar_type=_target_bar_type(hours, side),
                    source_bar_count=len(sources[side]),
                    eligible_source_bar_count=len(eligible),
                    emitted_bars=emitted,
                    dropped_windows=dropped,
                    callback_timestamps_ns=callbacks,
                )
            )

    _validate_bid_ask_coverage(series)
    coverage_status, research_ready, readiness_reasons = _research_readiness(series)
    manifest = _derived_manifest(
        parent_sha256,
        parent_manifest,
        series,
        catalog_path,
        root,
        coverage_status,
        research_ready,
        readiness_reasons,
    )
    manifest_path = root / DERIVED_MANIFEST_FILENAME
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    _preflight_idempotency(catalog, series, manifest_path, manifest_bytes)
    for item in series:
        if item.emitted_bars and not catalog.query_bars([item.target_bar_type]):
            catalog.write_bars(item.emitted_bars)
    if not manifest_path.exists():
        manifest_path.write_bytes(manifest_bytes)

    return DerivationResult(
        manifest_path=manifest_path,
        catalog_path=catalog_path,
        emitted_counts={
            item.target_bar_type: len(item.emitted_bars) for item in series
        },
        research_ready=research_ready,
    )


def _validate_runtime(parent_manifest_path: Path, catalog_path: Path) -> None:
    if version("nautilus-trader") != NAUTILUS_VERSION:
        raise RuntimeError(f"expected NautilusTrader {NAUTILUS_VERSION}")
    if not parent_manifest_path.is_file():
        raise DerivationValidationError(
            f"missing G0.5 parent manifest: {parent_manifest_path}"
        )
    if not catalog_path.is_dir():
        raise DerivationValidationError(f"missing G0.5 catalog: {catalog_path}")


def _load_parent_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DerivationValidationError(
            f"invalid G0.5 parent manifest: {error}"
        ) from error
    if not isinstance(value, dict):
        raise DerivationValidationError("G0.5 parent manifest must be an object")
    manifest = cast(dict[str, Any], value)
    expected = {
        "instrument_id": INSTRUMENT_ID,
        "source_granularity": "1-minute",
        "price_sides": ["BID", "ASK"],
        "nautilus_version": NAUTILUS_VERSION,
        "ingestion_version": "g0.5-1",
    }
    for field, expected_value in expected.items():
        if manifest.get(field) != expected_value:
            raise DerivationValidationError(
                f"G0.5 parent manifest has incompatible {field}"
            )
    return manifest


def _requested_range_ns(manifest: dict[str, Any]) -> tuple[int, int]:
    raw = manifest.get("requested_utc_range")
    if not isinstance(raw, dict):
        raise DerivationValidationError(
            "G0.5 parent manifest has invalid requested_utc_range"
        )
    requested = cast(dict[str, Any], raw)
    try:
        start_date = date.fromisoformat(str(requested["start_date"]))
        end_date = date.fromisoformat(str(requested["end_date_inclusive"]))
    except (KeyError, ValueError) as error:
        raise DerivationValidationError(
            "G0.5 parent manifest has invalid requested UTC dates"
        ) from error
    start = datetime.combine(start_date, datetime.min.time(), tzinfo=UTC)
    end = datetime.combine(
        end_date + timedelta(days=1), datetime.min.time(), tzinfo=UTC
    )
    if end <= start:
        raise DerivationValidationError("G0.5 requested UTC range is empty")
    return int(start.timestamp() * 1_000_000_000), int(
        end.timestamp() * 1_000_000_000
    )


def _validate_sources(
    sources: dict[str, tuple[Bar, ...]], parent_manifest: dict[str, Any]
) -> None:
    qa_raw = parent_manifest.get("qa")
    if not isinstance(qa_raw, dict):
        raise DerivationValidationError("G0.5 parent manifest has invalid qa")
    qa = cast(dict[str, Any], qa_raw)
    expected_counts = {
        "BID": qa.get("bid_bar_count"),
        "ASK": qa.get("ask_bar_count"),
    }
    for side, bars in sources.items():
        if expected_counts[side] != len(bars):
            raise DerivationValidationError(
                f"{side} catalog count does not match the G0.5 parent manifest"
            )
        if any(str(bar.bar_type) != _source_bar_type(side) for bar in bars):
            raise DerivationValidationError(
                f"catalog returned an invalid {side} bar type"
            )
        previous: int | None = None
        for bar in bars:
            if bar.ts_event % _MINUTE_NS != 0:
                raise DerivationValidationError(
                    f"{side} source bar is not UTC-minute aligned"
                )
            if bar.ts_init != bar.ts_event + _MINUTE_NS:
                raise DerivationValidationError(
                    f"{side} source bar does not use G0.5 open/close timestamps"
                )
            if previous is not None and bar.ts_event <= previous:
                raise DerivationValidationError(
                    f"{side} source bars are not strictly chronological"
                )
            previous = bar.ts_event
    bid_times = tuple(bar.ts_event for bar in sources["BID"])
    ask_times = tuple(bar.ts_event for bar in sources["ASK"])
    if bid_times != ask_times:
        raise DerivationValidationError(
            "BID/ASK one-minute source coverage mismatch; derived data is not "
            "research-ready"
        )


def _eligible_source_bars(
    bars: Sequence[Bar],
    hours: int,
    expected_minutes: int,
    *,
    range_start_ns: int | None = None,
    range_end_ns: int | None = None,
) -> tuple[tuple[Bar, ...], tuple[DroppedWindow, ...]]:
    if not bars:
        return (), ()
    interval_ns = hours * 60 * _MINUTE_NS
    first_observed_start = (bars[0].ts_event // interval_ns) * interval_ns
    last_observed_start = (bars[-1].ts_event // interval_ns) * interval_ns
    first_start = (
        first_observed_start if range_start_ns is None else range_start_ns
    )
    end_exclusive = (
        last_observed_start + interval_ns
        if range_end_ns is None
        else range_end_ns
    )
    if first_start % interval_ns != 0 or end_exclusive % interval_ns != 0:
        raise DerivationValidationError(
            f"requested range is not aligned for {hours}H derivation"
        )
    if bars[0].ts_event < first_start or bars[-1].ts_init > end_exclusive:
        raise DerivationValidationError("source bars fall outside the parent UTC range")
    by_window: dict[int, list[Bar]] = {}
    for bar in bars:
        start = (bar.ts_event // interval_ns) * interval_ns
        by_window.setdefault(start, []).append(bar)

    eligible: list[Bar] = []
    dropped: list[DroppedWindow] = []
    for start in range(first_start, end_exclusive, interval_ns):
        observed = by_window.get(start, [])
        expected_times = tuple(
            start + index * _MINUTE_NS for index in range(expected_minutes)
        )
        observed_times = tuple(bar.ts_event for bar in observed)
        if observed_times == expected_times:
            eligible.extend(observed)
            continue
        reason = (
            "no_source_updates"
            if not observed
            else "incomplete_or_nonconsecutive_minutes"
        )
        dropped.append(
            DroppedWindow(
                start_utc=_iso_ns(start),
                end_utc=_iso_ns(start + interval_ns),
                expected_minutes=expected_minutes,
                observed_minutes=len(observed),
                reason=reason,
            )
        )
    return tuple(eligible), tuple(dropped)


def _aggregate_native(
    *,
    instrument: CurrencyPair,
    source_bars: Sequence[Bar],
    side: str,
    hours: int,
) -> tuple[tuple[Bar, ...], tuple[int, ...]]:
    if not source_bars:
        return (), ()
    composite = BarType.from_str(_composite_bar_type(hours, side))
    collector = _BarCollector(composite)
    logger = LoggerConfig(
        stdout_level=LogLevel.OFF,
        print_config=False,
        bypass_logging=True,
    )
    data_config = DataEngineConfig(
        time_bars_timestamp_on_close=True,
        time_bars_skip_first_non_full_bar=True,
        time_bars_build_with_no_updates=False,
        time_bars_build_delay=_BUILD_DELAY_MICROSECONDS,
        time_bars_interval_type=BarIntervalType.LEFT_OPEN,
    )
    engine = BacktestEngine(
        BacktestEngineConfig(
            logging=logger,
            bypass_logging=True,
            run_analysis=False,
            data_engine=data_config,
        )
    )
    engine.add_venue(
        venue=Venue("DUKASCOPY"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money.from_str("100000 USD")],
        base_currency=Currency.from_str("USD"),
        book_type=BookType.L1_MBP,
        use_random_ids=False,
    )
    engine.add_instrument(instrument)
    engine.add_data(source_bars)
    engine.add_actor(collector)
    try:
        # Starting one nanosecond before an aligned window lets rc2 distinguish a
        # complete first interval while retaining skip_first_non_full_bar=True.
        engine.run(
            start=source_bars[0].ts_event - 1,
            end=source_bars[-1].ts_init + _BUILD_DELAY_NS,
        )
    finally:
        engine.dispose()

    canonical = tuple(_canonicalize_close_timestamp(bar) for bar in collector.bars)
    return canonical, tuple(collector.callback_timestamps_ns)


def _canonicalize_close_timestamp(bar: Bar) -> Bar:
    nominal_close = bar.ts_event - _BUILD_DELAY_NS
    return Bar(
        bar_type=bar.bar_type,
        open=bar.open,
        high=bar.high,
        low=bar.low,
        close=bar.close,
        volume=bar.volume,
        ts_event=nominal_close,
        ts_init=nominal_close,
    )


def _validate_native_output(
    *,
    emitted: Sequence[Bar],
    callback_timestamps_ns: Sequence[int],
    source_bars: Sequence[Bar],
    hours: int,
) -> None:
    interval_ns = hours * 60 * _MINUTE_NS
    if len(emitted) != len(source_bars) // (hours * 60):
        raise DerivationValidationError(
            f"Nautilus emitted an unexpected number of {hours}H bars"
        )
    if len(emitted) != len(callback_timestamps_ns):
        raise DerivationValidationError("missing Nautilus callback timing evidence")
    source_by_close = {bar.ts_init: bar for bar in source_bars}
    for bar, callback_ns in zip(emitted, callback_timestamps_ns, strict=True):
        if str(bar.bar_type) not in {
            _target_bar_type(hours, "BID"),
            _target_bar_type(hours, "ASK"),
        }:
            raise DerivationValidationError(
                "Nautilus emitted the wrong target bar type"
            )
        if bar.ts_event % interval_ns != 0 or bar.ts_init != bar.ts_event:
            raise DerivationValidationError(
                "derived bar is not timestamped on UTC close"
            )
        final_source = source_by_close.get(bar.ts_event)
        if final_source is None:
            raise DerivationValidationError(
                "derived bar close lacks its final underlying one-minute bar"
            )
        if callback_ns < final_source.ts_init or callback_ns < bar.ts_event:
            raise DerivationValidationError(
                "derived bar became visible before the final source minute closed"
            )


def _validate_bid_ask_coverage(series: Sequence[SeriesResult]) -> None:
    for hours, _ in _TARGETS:
        by_side = {
            item.target_bar_type.split("-")[-2]: tuple(
                bar.ts_event for bar in item.emitted_bars
            )
            for item in series
            if item.target_bar_type.startswith(f"{INSTRUMENT_ID}-{hours}-HOUR")
        }
        if by_side.get("BID") != by_side.get("ASK"):
            raise DerivationValidationError(
                f"{hours}H BID/ASK derived coverage mismatch; not research-ready"
            )


def _research_readiness(
    series: Sequence[SeriesResult],
) -> tuple[str, bool, list[str]]:
    has_incomplete = any(
        window.reason == "incomplete_or_nonconsecutive_minutes"
        for item in series
        for window in item.dropped_windows
    )
    has_unclassified = any(
        window.reason == "no_source_updates"
        for item in series
        for window in item.dropped_windows
    )
    all_series_nonempty = all(item.emitted_bars for item in series)

    if has_incomplete:
        coverage_status = "incomplete"
    elif has_unclassified:
        coverage_status = "unclassified"
    else:
        coverage_status = "complete"

    reasons: list[str] = []
    if has_incomplete:
        reasons.append("one or more aligned windows have missing source minutes")
    if has_unclassified:
        reasons.append(
            "one or more no-update windows cannot be classified without a "
            "session calendar"
        )
    if not all_series_nonempty:
        reasons.append("one or more required 1H/4H BID/ASK series contain no bars")

    research_ready = coverage_status == "complete" and all_series_nonempty
    return coverage_status, research_ready, reasons


def _preflight_idempotency(
    catalog: ParquetDataCatalog,
    series: Sequence[SeriesResult],
    manifest_path: Path,
    manifest_bytes: bytes,
) -> None:
    for item in series:
        existing = catalog.query_bars([item.target_bar_type])
        if existing and existing != list(item.emitted_bars):
            raise DerivationValidationError(
                f"catalog already contains conflicting data for {item.target_bar_type}"
            )
        if existing and not item.emitted_bars:
            raise DerivationValidationError(
                f"catalog contains unexpected data for {item.target_bar_type}"
            )
    if manifest_path.exists() and manifest_path.read_bytes() != manifest_bytes:
        raise DerivationValidationError(
            f"existing derived manifest conflicts with this derivation: {manifest_path}"
        )


def _derived_manifest(
    parent_sha256: str,
    parent_manifest: dict[str, Any],
    series: Sequence[SeriesResult],
    catalog_path: Path,
    root: Path,
    coverage_status: str,
    research_ready: bool,
    readiness_reasons: Sequence[str],
) -> dict[str, Any]:
    series_manifest: dict[str, Any] = {}
    emitted_counts: dict[str, dict[str, int]] = {"1H": {}, "4H": {}}
    for item in series:
        incomplete = sum(
            window.reason == "incomplete_or_nonconsecutive_minutes"
            for window in item.dropped_windows
        )
        no_updates = sum(
            window.reason == "no_source_updates" for window in item.dropped_windows
        )
        series_manifest[item.target_bar_type] = {
            "source_bar_type": item.source_bar_type,
            "composite_bar_type": item.composite_bar_type,
            "source_bar_count": item.source_bar_count,
            "eligible_source_bar_count": item.eligible_source_bar_count,
            "emitted_bar_count": len(item.emitted_bars),
            "content_sha256": _bars_sha256(item.emitted_bars),
            "dropped_window_count": len(item.dropped_windows),
            "dropped_incomplete_window_count": incomplete,
            "no_update_window_count": no_updates,
            "dropped_window_details": [
                asdict(window) for window in item.dropped_windows
            ],
        }
        timeframe = "4H" if "-4-HOUR-" in item.target_bar_type else "1H"
        side = "ASK" if "-ASK-" in item.target_bar_type else "BID"
        emitted_counts[timeframe][side] = len(item.emitted_bars)
    return {
        "derivation_version": DERIVATION_VERSION,
        "config_version": CONFIG_VERSION,
        "parent_g0_5_manifest": PARENT_MANIFEST_FILENAME,
        "parent_g0_5_manifest_sha256": parent_sha256,
        "parent_ingestion_version": parent_manifest["ingestion_version"],
        "nautilus_version": NAUTILUS_VERSION,
        "instrument_id": INSTRUMENT_ID,
        "catalog_location": catalog_path.relative_to(root).as_posix(),
        "counts": {
            "source_1_minute": {
                "BID": next(
                    item.source_bar_count
                    for item in series
                    if item.source_bar_type.endswith("-BID-EXTERNAL")
                ),
                "ASK": next(
                    item.source_bar_count
                    for item in series
                    if item.source_bar_type.endswith("-ASK-EXTERNAL")
                ),
            },
            "emitted": emitted_counts,
        },
        "source_policy": {
            "direct_from_original_one_minute": True,
            "price_sides": list(_SIDES),
            "fills_or_interpolation": False,
            "fx_session_calendar": False,
        },
        "aggregation_configuration": {
            "api": "Nautilus BacktestEngine composite BarType bar-to-bar aggregation",
            "time_bars_timestamp_on_close": True,
            "time_bars_skip_first_non_full_bar": True,
            "time_bars_build_with_no_updates": False,
            "time_bars_build_delay_microseconds": _BUILD_DELAY_MICROSECONDS,
            "time_bars_interval_type": "LEFT_OPEN",
            "time_bars_origin_offset": {},
        },
        "utc_alignment": {
            "1H": "00:00, 01:00, ..., 23:00 UTC closes",
            "4H": "00:00, 04:00, ..., 20:00 UTC closes",
            "interval_semantics": "left-open: start excluded, close included",
        },
        "timestamp_convention": {
            "stored_ts_event": "nominal interval close UTC",
            "stored_ts_init": "nominal interval close UTC",
            "availability": (
                "native callback occurs at interval close plus configured 1 "
                "microsecond build delay, never before final one-minute ts_init"
            ),
            "rc2_canonicalization": (
                "subtract the configured 1 microsecond scheduling delay from native "
                "ts_event/ts_init only; OHLCV is unmodified native output"
            ),
        },
        "derived_bar_integrity_valid": True,
        "coverage_status": coverage_status,
        "research_ready": research_ready,
        "research_readiness_reasons": list(readiness_reasons),
        "bid_ask_derived_coverage_matches": True,
        "series": series_manifest,
    }


def _bars_sha256(bars: Sequence[Bar]) -> str:
    digest = hashlib.sha256()
    for bar in bars:
        record = {
            "bar_type": str(bar.bar_type),
            "open": str(bar.open),
            "high": str(bar.high),
            "low": str(bar.low),
            "close": str(bar.close),
            "volume": str(bar.volume),
            "ts_event": bar.ts_event,
            "ts_init": bar.ts_init,
        }
        digest.update(
            (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
        )
    return digest.hexdigest()


def _source_bar_type(side: str) -> str:
    return f"{INSTRUMENT_ID}-1-MINUTE-{side}-EXTERNAL"


def _composite_bar_type(hours: int, side: str) -> str:
    return f"{INSTRUMENT_ID}-{hours}-HOUR-{side}-INTERNAL@1-MINUTE-EXTERNAL"


def _target_bar_type(hours: int, side: str) -> str:
    return f"{INSTRUMENT_ID}-{hours}-HOUR-{side}-INTERNAL"


def _iso_ns(timestamp_ns: int) -> str:
    seconds, nanos = divmod(timestamp_ns, 1_000_000_000)
    value = datetime.fromtimestamp(seconds, tz=UTC)
    if nanos:
        return value.strftime("%Y-%m-%dT%H:%M:%S") + f".{nanos:09d}Z"
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Derive complete UTC EUR/USD 1H/4H BID/ASK bars from G0.5"
    )
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the G0.6 derived-bar command."""

    args = _build_parser().parse_args(argv)
    result = derive_eurusd_bars(cast(Path, args.output_root))
    print(result.manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
