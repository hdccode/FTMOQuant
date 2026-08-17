"""Deterministic Nautilus composite EUR/USD higher-timeframe bars."""

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
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

from ftmoquant.data.canonical_source import (
    CanonicalSourceValidationError,
    iter_paired_source_chunks,
    validate_canonical_eurusd_source_manifest,
    validate_canonical_source_manifest,
)
from ftmoquant.data.dukascopy import INSTRUMENT_ID, NAUTILUS_VERSION
from ftmoquant.data.instruments import EURUSD_SPEC, InstrumentSpec

DERIVATION_VERSION = "g0.6-2"
CONFIG_VERSION = 1
DERIVED_MANIFEST_FILENAME = "ftmoquant_derived_provenance.json"
PARENT_MANIFEST_FILENAME = "ftmoquant_provenance.json"

_MINUTE_NS = 60_000_000_000
_BUILD_DELAY_MICROSECONDS = 1
_BUILD_DELAY_NS = _BUILD_DELAY_MICROSECONDS * 1_000
#: Each target is (window minutes, expected 1-minute source bars per window);
#: the two are always equal, kept as a pair for a smaller historical diff.
_TARGETS = ((30, 30), (60, 60), (240, 240))
_SIDES = ("BID", "ASK")
_SOURCE_QUERY_CHUNK_MINUTES = 10_000


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
    """Bounded-memory QA summary for one target/price-side series."""

    source_bar_type: str
    composite_bar_type: str
    target_bar_type: str
    source_bar_count: int
    eligible_source_bar_count: int
    emitted_bar_count: int
    content_sha256: str
    coverage_sha256: str
    dropped_windows: tuple[DroppedWindow, ...]


@dataclass(slots=True)
class _SeriesAccumulator:
    """Only cross-query state needed for one aligned derived series."""

    side: str
    instrument_id: str
    minutes: int
    expected_minutes: int
    range_end_ns: int
    window_start_ns: int
    observed_window: list[Bar] = field(default_factory=list)
    eligible_pending: list[Bar] = field(default_factory=list)
    source_bar_count: int = 0
    eligible_source_bar_count: int = 0
    emitted_bar_count: int = 0
    dropped_windows: list[DroppedWindow] = field(default_factory=list)
    content_digest: Any = field(default_factory=hashlib.sha256)
    coverage_digest: Any = field(default_factory=hashlib.sha256)


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
    """Derive complete UTC M30/1H/4H bars from a trusted canonical 1m catalog."""

    return derive_instrument_bars(output_root, EURUSD_SPEC)


def derive_instrument_bars(
    output_root: Path, instrument_spec: InstrumentSpec
) -> DerivationResult:
    """Derive native complete-window bars for one configured instrument."""

    instrument_spec.validate()

    root = output_root.resolve()
    parent_manifest_path = root / PARENT_MANIFEST_FILENAME
    catalog_path = root / "catalog"
    _validate_runtime(parent_manifest_path, catalog_path)
    parent_manifest = _load_parent_manifest(
        parent_manifest_path, instrument_spec=instrument_spec
    )
    parent_sha256 = _sha256(parent_manifest_path)
    requested_start_ns, requested_end_ns = _requested_range_ns(parent_manifest)

    catalog = ParquetDataCatalog(str(catalog_path))
    instruments = catalog.instruments([instrument_spec.instrument_id])
    if len(instruments) != 1 or not isinstance(instruments[0], CurrencyPair):
        raise DerivationValidationError(
            "catalog must contain exactly one CurrencyPair for "
            f"{instrument_spec.instrument_id}"
        )
    instrument = instruments[0]

    manifest_path = root / DERIVED_MANIFEST_FILENAME
    write_outputs = _preflight_output_mode(
        catalog, manifest_path, instrument_spec.instrument_id
    )
    _validate_streaming_source(
        catalog,
        parent_manifest,
        requested_start_ns,
        requested_end_ns,
        instrument_spec,
    )

    accumulators = _series_accumulators(
        requested_start_ns, requested_end_ns, instrument_spec.instrument_id
    )
    try:
        chunks = iter_paired_source_chunks(
            catalog,
            parent_manifest,
            instrument_spec,
            requested_start_ns,
            requested_end_ns,
            chunk_minutes=_SOURCE_QUERY_CHUNK_MINUTES,
        )
        for chunk in chunks:
            for side, bars in (("BID", chunk.bid_bars), ("ASK", chunk.ask_bars)):
                for bar in bars:
                    for minutes, _ in _TARGETS:
                        _consume_series_bar(accumulators[(minutes, side)], bar)
            _flush_eligible_pending(
                accumulators, instrument, catalog, write_outputs=write_outputs
            )
    except CanonicalSourceValidationError as error:
        raise DerivationValidationError(str(error)) from error
    for accumulator in accumulators.values():
        _finish_series_windows(accumulator)
    _flush_eligible_pending(
        accumulators, instrument, catalog, write_outputs=write_outputs
    )
    series = [_series_result(item) for item in accumulators.values()]

    _validate_bid_ask_coverage(series, instrument_spec.instrument_id)
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
        instrument_spec.instrument_id,
    )
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    _validate_stored_outputs(catalog, series, requested_start_ns, requested_end_ns)
    if manifest_path.exists():
        if manifest_path.read_bytes() != manifest_bytes:
            raise DerivationValidationError(
                "existing derived manifest conflicts with this derivation: "
                f"{manifest_path}"
            )
    else:
        manifest_path.write_bytes(manifest_bytes)

    return DerivationResult(
        manifest_path=manifest_path,
        catalog_path=catalog_path,
        emitted_counts={
            item.target_bar_type: item.emitted_bar_count for item in series
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


def _load_parent_manifest(
    path: Path, *, instrument_spec: InstrumentSpec = EURUSD_SPEC
) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DerivationValidationError(
            f"invalid G0.5 parent manifest: {error}"
        ) from error
    if not isinstance(value, dict):
        raise DerivationValidationError("G0.5 parent manifest must be an object")
    manifest = cast(dict[str, Any], value)
    try:
        if instrument_spec == EURUSD_SPEC:
            validate_canonical_eurusd_source_manifest(manifest)
        else:
            validate_canonical_source_manifest(manifest, instrument_spec)
    except CanonicalSourceValidationError as error:
        raise DerivationValidationError(str(error)) from error
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
    return int(start.timestamp() * 1_000_000_000), int(end.timestamp() * 1_000_000_000)


def _preflight_output_mode(
    catalog: ParquetDataCatalog,
    manifest_path: Path,
    instrument_id: str = INSTRUMENT_ID,
) -> bool:
    """Return whether output is new, rejecting orphaned derived catalog data."""

    target_files = [
        path
        for minutes, _ in _TARGETS
        for side in _SIDES
        for path in catalog.query_files(
            "bars", [_target_bar_type(minutes, side, instrument_id)]
        )
    ]
    if target_files and not manifest_path.exists():
        raise DerivationValidationError(
            "catalog already contains conflicting data without a derived manifest"
        )
    return not manifest_path.exists()


def _validate_streaming_source(
    catalog: ParquetDataCatalog,
    parent_manifest: dict[str, Any],
    start_ns: int,
    end_ns: int,
    instrument_spec: InstrumentSpec = EURUSD_SPEC,
) -> None:
    """Complete a read-only paired validation pass before derived writes."""

    try:
        for _ in iter_paired_source_chunks(
            catalog,
            parent_manifest,
            instrument_spec,
            start_ns,
            end_ns,
            chunk_minutes=_SOURCE_QUERY_CHUNK_MINUTES,
        ):
            pass
    except CanonicalSourceValidationError as error:
        raise DerivationValidationError(str(error)) from error


def _series_accumulators(
    start_ns: int, end_ns: int, instrument_id: str = INSTRUMENT_ID
) -> dict[tuple[int, str], _SeriesAccumulator]:
    accumulators: dict[tuple[int, str], _SeriesAccumulator] = {}
    for minutes, expected_minutes in _TARGETS:
        interval_ns = minutes * _MINUTE_NS
        if start_ns % interval_ns or end_ns % interval_ns:
            raise DerivationValidationError(
                f"requested range is not aligned for {_timeframe_label(minutes)} "
                "derivation"
            )
        for side in _SIDES:
            accumulators[(minutes, side)] = _SeriesAccumulator(
                side=side,
                instrument_id=instrument_id,
                minutes=minutes,
                expected_minutes=expected_minutes,
                range_end_ns=end_ns,
                window_start_ns=start_ns,
            )
    return accumulators


def _consume_series_bar(accumulator: _SeriesAccumulator, bar: Bar) -> None:
    interval_ns = accumulator.minutes * _MINUTE_NS
    while bar.ts_event >= accumulator.window_start_ns + interval_ns:
        _finalize_series_window(accumulator)
    if bar.ts_event < accumulator.window_start_ns:
        raise DerivationValidationError("source bar precedes active derived window")
    accumulator.observed_window.append(bar)
    accumulator.source_bar_count += 1


def _finish_series_windows(accumulator: _SeriesAccumulator) -> None:
    while accumulator.window_start_ns < accumulator.range_end_ns:
        _finalize_series_window(accumulator)


def _finalize_series_window(accumulator: _SeriesAccumulator) -> None:
    interval_ns = accumulator.minutes * _MINUTE_NS
    observed = accumulator.observed_window
    complete = len(observed) == accumulator.expected_minutes and all(
        bar.ts_event == accumulator.window_start_ns + index * _MINUTE_NS
        for index, bar in enumerate(observed)
    )
    if complete:
        accumulator.eligible_pending.extend(observed)
        accumulator.eligible_source_bar_count += len(observed)
    else:
        accumulator.dropped_windows.append(
            DroppedWindow(
                start_utc=_iso_ns(accumulator.window_start_ns),
                end_utc=_iso_ns(accumulator.window_start_ns + interval_ns),
                expected_minutes=accumulator.expected_minutes,
                observed_minutes=len(observed),
                reason=(
                    "no_source_updates"
                    if not observed
                    else "incomplete_or_nonconsecutive_minutes"
                ),
            )
        )
    observed.clear()
    accumulator.window_start_ns += interval_ns


def _flush_eligible_pending(
    accumulators: dict[tuple[int, str], _SeriesAccumulator],
    instrument: CurrencyPair,
    catalog: ParquetDataCatalog,
    *,
    write_outputs: bool,
) -> None:
    for minutes, _ in _TARGETS:
        bid_state = accumulators[(minutes, "BID")]
        ask_state = accumulators[(minutes, "ASK")]
        bid_source = tuple(bid_state.eligible_pending)
        ask_source = tuple(ask_state.eligible_pending)
        bid_state.eligible_pending.clear()
        ask_state.eligible_pending.clear()
        label = _timeframe_label(minutes)
        if tuple(bar.ts_event for bar in bid_source) != tuple(
            bar.ts_event for bar in ask_source
        ):
            raise DerivationValidationError(
                f"{label} BID/ASK eligible source coverage mismatch"
            )
        if not bid_source:
            continue
        emitted_by_side: dict[str, tuple[Bar, ...]] = {}
        for side, source_bars, state in (
            ("BID", bid_source, bid_state),
            ("ASK", ask_source, ask_state),
        ):
            if state.instrument_id == INSTRUMENT_ID:
                emitted, callbacks = _aggregate_native(
                    instrument=instrument,
                    source_bars=source_bars,
                    side=side,
                    minutes=minutes,
                )
            else:
                emitted, callbacks = _aggregate_native(
                    instrument=instrument,
                    source_bars=source_bars,
                    side=side,
                    minutes=minutes,
                    instrument_id=state.instrument_id,
                )
            _validate_native_output(
                emitted=emitted,
                callback_timestamps_ns=callbacks,
                source_bars=source_bars,
                minutes=minutes,
                instrument_id=state.instrument_id,
            )
            for bar in emitted:
                _update_bar_digest(state.content_digest, bar)
                state.coverage_digest.update(f"{bar.ts_event}\n".encode())
            state.emitted_bar_count += len(emitted)
            emitted_by_side[side] = emitted
        if tuple(bar.ts_event for bar in emitted_by_side["BID"]) != tuple(
            bar.ts_event for bar in emitted_by_side["ASK"]
        ):
            raise DerivationValidationError(
                f"{label} BID/ASK derived coverage mismatch; not research-ready"
            )
        if write_outputs:
            catalog.write_bars(emitted_by_side["BID"])
            catalog.write_bars(emitted_by_side["ASK"])


def _series_result(accumulator: _SeriesAccumulator) -> SeriesResult:
    return SeriesResult(
        source_bar_type=_source_bar_type(accumulator.side, accumulator.instrument_id),
        composite_bar_type=_composite_bar_type(
            accumulator.minutes, accumulator.side, accumulator.instrument_id
        ),
        target_bar_type=_target_bar_type(
            accumulator.minutes, accumulator.side, accumulator.instrument_id
        ),
        source_bar_count=accumulator.source_bar_count,
        eligible_source_bar_count=accumulator.eligible_source_bar_count,
        emitted_bar_count=accumulator.emitted_bar_count,
        content_sha256=accumulator.content_digest.hexdigest(),
        coverage_sha256=accumulator.coverage_digest.hexdigest(),
        dropped_windows=tuple(accumulator.dropped_windows),
    )


def _validate_stored_outputs(
    catalog: ParquetDataCatalog,
    series: Sequence[SeriesResult],
    start_ns: int,
    end_ns: int,
) -> None:
    """Boundedly prove incremental writes or a repeated run against summaries."""

    chunk_ns = _SOURCE_QUERY_CHUNK_MINUTES * _MINUTE_NS
    for item in series:
        files = catalog.query_files("bars", [item.target_bar_type])
        if not files:
            if item.emitted_bar_count:
                raise DerivationValidationError(
                    f"catalog is missing derived data for {item.target_bar_type}"
                )
            continue
        first = catalog.query_first_timestamp("bars", item.target_bar_type)
        last = catalog.query_last_timestamp("bars", item.target_bar_type)
        if first is None or last is None or first <= start_ns or last > end_ns:
            raise DerivationValidationError(
                f"catalog has out-of-range derived data for {item.target_bar_type}"
            )
        digest = hashlib.sha256()
        count = 0
        previous: int | None = None
        chunk_start = start_ns
        while chunk_start < end_ns:
            chunk_end = min(chunk_start + chunk_ns, end_ns)
            queried = catalog.query_bars(
                [item.target_bar_type], start=chunk_start, end=chunk_end
            )
            bars = tuple(
                bar for bar in queried if chunk_start < bar.ts_event <= chunk_end
            )
            for bar in bars:
                if str(bar.bar_type) != item.target_bar_type:
                    raise DerivationValidationError(
                        "catalog returned an invalid derived bar type"
                    )
                if previous is not None and bar.ts_event <= previous:
                    raise DerivationValidationError(
                        "stored derived bars are not strictly chronological"
                    )
                previous = bar.ts_event
                _update_bar_digest(digest, bar)
            count += len(bars)
            chunk_start = chunk_end
        if count != item.emitted_bar_count or digest.hexdigest() != item.content_sha256:
            raise DerivationValidationError(
                f"catalog contains conflicting data for {item.target_bar_type}"
            )


def _eligible_source_bars(
    bars: Sequence[Bar],
    minutes: int,
    expected_minutes: int,
    *,
    range_start_ns: int | None = None,
    range_end_ns: int | None = None,
) -> tuple[tuple[Bar, ...], tuple[DroppedWindow, ...]]:
    if not bars:
        return (), ()
    interval_ns = minutes * _MINUTE_NS
    first_observed_start = (bars[0].ts_event // interval_ns) * interval_ns
    last_observed_start = (bars[-1].ts_event // interval_ns) * interval_ns
    first_start = first_observed_start if range_start_ns is None else range_start_ns
    end_exclusive = (
        last_observed_start + interval_ns if range_end_ns is None else range_end_ns
    )
    if first_start % interval_ns != 0 or end_exclusive % interval_ns != 0:
        raise DerivationValidationError(
            f"requested range is not aligned for {_timeframe_label(minutes)} "
            "derivation"
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
    minutes: int,
    instrument_id: str = INSTRUMENT_ID,
) -> tuple[tuple[Bar, ...], tuple[int, ...]]:
    if not source_bars:
        return (), ()
    composite = BarType.from_str(_composite_bar_type(minutes, side, instrument_id))
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
    minutes: int,
    instrument_id: str = INSTRUMENT_ID,
) -> None:
    interval_ns = minutes * _MINUTE_NS
    if len(emitted) != len(source_bars) // minutes:
        raise DerivationValidationError(
            f"Nautilus emitted an unexpected number of {_timeframe_label(minutes)} bars"
        )
    if len(emitted) != len(callback_timestamps_ns):
        raise DerivationValidationError("missing Nautilus callback timing evidence")
    source_by_close = {bar.ts_init: bar for bar in source_bars}
    for bar, callback_ns in zip(emitted, callback_timestamps_ns, strict=True):
        if str(bar.bar_type) not in {
            _target_bar_type(minutes, "BID", instrument_id),
            _target_bar_type(minutes, "ASK", instrument_id),
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


def _validate_bid_ask_coverage(
    series: Sequence[SeriesResult], instrument_id: str = INSTRUMENT_ID
) -> None:
    for minutes, _ in _TARGETS:
        prefix = f"{instrument_id}-{_timeframe_segment(minutes)}-"
        by_side = {
            item.target_bar_type.split("-")[-2]: (
                item.emitted_bar_count,
                item.coverage_sha256,
            )
            for item in series
            if item.target_bar_type.startswith(prefix)
        }
        if by_side.get("BID") != by_side.get("ASK"):
            raise DerivationValidationError(
                f"{_timeframe_label(minutes)} BID/ASK derived coverage mismatch; "
                "not research-ready"
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
    all_series_nonempty = all(item.emitted_bar_count > 0 for item in series)

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
        labels = "/".join(_timeframe_label(minutes) for minutes, _ in _TARGETS)
        reasons.append(f"one or more required {labels} BID/ASK series contain no bars")

    research_ready = coverage_status == "complete" and all_series_nonempty
    return coverage_status, research_ready, reasons


def _derived_manifest(
    parent_sha256: str,
    parent_manifest: dict[str, Any],
    series: Sequence[SeriesResult],
    catalog_path: Path,
    root: Path,
    coverage_status: str,
    research_ready: bool,
    readiness_reasons: Sequence[str],
    instrument_id: str = INSTRUMENT_ID,
) -> dict[str, Any]:
    series_manifest: dict[str, Any] = {}
    emitted_counts: dict[str, dict[str, int]] = {
        _timeframe_label(minutes): {} for minutes, _ in _TARGETS
    }
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
            "emitted_bar_count": item.emitted_bar_count,
            "content_sha256": item.content_sha256,
            "dropped_window_count": len(item.dropped_windows),
            "dropped_incomplete_window_count": incomplete,
            "no_update_window_count": no_updates,
            "dropped_window_details": [
                asdict(window) for window in item.dropped_windows
            ],
        }
        timeframe = next(
            _timeframe_label(minutes)
            for minutes, _ in _TARGETS
            if f"-{_timeframe_segment(minutes)}-" in item.target_bar_type
        )
        side = "ASK" if "-ASK-" in item.target_bar_type else "BID"
        emitted_counts[timeframe][side] = item.emitted_bar_count
    return {
        "derivation_version": DERIVATION_VERSION,
        "config_version": CONFIG_VERSION,
        "parent_g0_5_manifest": PARENT_MANIFEST_FILENAME,
        "parent_g0_5_manifest_sha256": parent_sha256,
        "parent_ingestion_version": parent_manifest["ingestion_version"],
        "nautilus_version": NAUTILUS_VERSION,
        "instrument_id": instrument_id,
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
            **{
                _timeframe_label(minutes): _utc_alignment_description(minutes)
                for minutes, _ in _TARGETS
            },
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
        _update_bar_digest(digest, bar)
    return digest.hexdigest()


def _update_bar_digest(digest: Any, bar: Bar) -> None:
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


def _timeframe_segment(minutes: int) -> str:
    """Nautilus bar-type step/unit segment, e.g. ``30-MINUTE`` or ``4-HOUR``."""

    if minutes % 60 == 0:
        return f"{minutes // 60}-HOUR"
    return f"{minutes}-MINUTE"


def _timeframe_label(minutes: int) -> str:
    """Short manifest label, e.g. ``M30``, ``1H``, ``4H``."""

    if minutes % 60 == 0:
        return f"{minutes // 60}H"
    return f"M{minutes}"


def _utc_alignment_description(minutes: int) -> str:
    def hhmm(total_minutes: int) -> str:
        return f"{(total_minutes // 60) % 24:02d}:{total_minutes % 60:02d}"

    return f"00:00, {hhmm(minutes)}, ..., {hhmm(1440 - minutes)} UTC closes"


def _source_bar_type(side: str, instrument_id: str = INSTRUMENT_ID) -> str:
    return f"{instrument_id}-1-MINUTE-{side}-EXTERNAL"


def _composite_bar_type(
    minutes: int, side: str, instrument_id: str = INSTRUMENT_ID
) -> str:
    segment = _timeframe_segment(minutes)
    return f"{instrument_id}-{segment}-{side}-INTERNAL@1-MINUTE-EXTERNAL"


def _target_bar_type(
    minutes: int, side: str, instrument_id: str = INSTRUMENT_ID
) -> str:
    return f"{instrument_id}-{_timeframe_segment(minutes)}-{side}-INTERNAL"


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
        description="Derive complete UTC EUR/USD M30/1H/4H BID/ASK bars from G0.5"
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
