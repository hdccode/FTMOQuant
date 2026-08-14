"""Pinned Hugging Face Dukascopy tick to canonical EUR/USD minute adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from huggingface_hub import HfApi, hf_hub_download
from nautilus_trader.model import Bar
from nautilus_trader.persistence import ParquetDataCatalog

from ftmoquant.data.canonical_source import HF_INGESTION_VERSION
from ftmoquant.data.dukascopy import (
    INSTRUMENT_ID,
    NAUTILUS_VERSION,
    PRICE_PRECISION,
    SIZE_PRECISION,
    MissingInterval,
    SourceBar,
    _eurusd_instrument,
    _reject_git_worktree_root,
    _sha256,
    _to_nautilus_bars,
)
from ftmoquant.data.research_plan import (
    HF_REVISION_PATTERN,
    ResearchDataPlan,
    load_research_data_plan,
)

HF_REPO_TYPE = "dataset"
SOURCE_DIRECTORY = "data/EURUSD/"
MANIFEST_FILENAME = "ftmoquant_provenance.json"
DEFAULT_PLAN_PATH = Path("config/data/eurusd_research_v1.yaml")
REQUIRED_COLUMNS = (
    "timestamp",
    "askPrice",
    "bidPrice",
    "askVolume",
    "bidVolume",
)
_EXPECTED_SCHEMA = {
    "timestamp": pa.int64(),
    "askPrice": pa.float64(),
    "bidPrice": pa.float64(),
    "askVolume": pa.float64(),
    "bidVolume": pa.float64(),
}
_SHARD_PATTERN = re.compile(
    r"data/EURUSD/(?P<start>\d{4}-\d{2}-\d{2})_"
    r"(?P<end>\d{4}-\d{2}-\d{2})\.parquet"
)
_ONE_MINUTE = timedelta(minutes=1)
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_MILLISECOND = timedelta(milliseconds=1)
_MINUTE_MS = 60_000
_WRITE_CHUNK_BARS = 50_000


class HfDukascopyValidationError(ValueError):
    """Raised when remote provenance or tick data fails closed validation."""


@dataclass(frozen=True, slots=True)
class RemoteFileMetadata:
    """Stable metadata exposed by the pinned Hugging Face repository."""

    size: int | None
    lfs_sha256: str | None


@dataclass(frozen=True, slots=True)
class SourceFilePlan:
    """One ordered physical Parquet shard selected for a G1 request."""

    repo_path: str
    filename_start_date: date
    filename_end_date: date
    dataset_revision: str
    remote_size: int | None
    remote_lfs_sha256: str | None
    overlaps_g1_permitted_interval: bool
    straddles_research_split_boundary: bool


@dataclass(frozen=True, slots=True)
class SourceFileResult:
    """Deterministic admitted-row provenance for one downloaded shard."""

    repo_path: str
    filename_start_date: str
    filename_end_date: str
    dataset_revision: str
    remote_size: int | None
    remote_lfs_sha256: str | None
    downloaded_size: int
    downloaded_sha256: str
    admitted_tick_count: int
    admitted_min_timestamp_utc: str | None
    admitted_max_timestamp_utc: str | None
    overlaps_g1_permitted_interval: bool
    straddles_research_split_boundary: bool


@dataclass(frozen=True, slots=True)
class HfIngestionResult:
    """Stable paths and counts returned by one G1 Hugging Face ingestion."""

    manifest_path: Path
    catalog_path: Path
    dataset_revision: str
    dataset_plan_sha256: str
    admitted_tick_count: int
    bid_bar_count: int
    ask_bar_count: int
    missing_intervals: tuple[MissingInterval, ...]
    semantic_sha256: str


@dataclass(slots=True)
class _MinuteAggregate:
    minute_ms: int
    bid_open: Decimal
    bid_high: Decimal
    bid_low: Decimal
    bid_close: Decimal
    bid_volume: Decimal
    ask_open: Decimal
    ask_high: Decimal
    ask_low: Decimal
    ask_close: Decimal
    ask_volume: Decimal

    @classmethod
    def from_tick(
        cls,
        minute_ms: int,
        bid: Decimal,
        ask: Decimal,
        bid_volume: Decimal,
        ask_volume: Decimal,
    ) -> _MinuteAggregate:
        return cls(
            minute_ms=minute_ms,
            bid_open=bid,
            bid_high=bid,
            bid_low=bid,
            bid_close=bid,
            bid_volume=bid_volume,
            ask_open=ask,
            ask_high=ask,
            ask_low=ask,
            ask_close=ask,
            ask_volume=ask_volume,
        )

    def update(
        self,
        bid: Decimal,
        ask: Decimal,
        bid_volume: Decimal,
        ask_volume: Decimal,
    ) -> None:
        self.bid_high = max(self.bid_high, bid)
        self.bid_low = min(self.bid_low, bid)
        self.bid_close = bid
        self.bid_volume += bid_volume
        self.ask_high = max(self.ask_high, ask)
        self.ask_low = min(self.ask_low, ask)
        self.ask_close = ask
        self.ask_volume += ask_volume


@dataclass(slots=True)
class _StreamState:
    catalog: ParquetDataCatalog
    current: _MinuteAggregate | None
    previous_tick_ms: int | None
    previous_file_last_ms: int | None
    last_emitted_minute_ms: int | None
    bid_pending: list[SourceBar]
    ask_pending: list[SourceBar]
    bar_count: int
    next_expected_minute: datetime
    missing_intervals: list[MissingInterval]


def resolve_current_dataset_revision(api: HfApi | None = None) -> str:
    """Resolve the current repository head through the official metadata API."""

    info = (api or HfApi()).dataset_info("mito0o852/dukascopy-ticks")
    revision = info.sha
    if not isinstance(revision, str) or HF_REVISION_PATTERN.fullmatch(revision) is None:
        raise HfDukascopyValidationError(
            "Hugging Face did not resolve an immutable dataset revision"
        )
    return revision


def discover_source_file_plan(
    repo_files: Sequence[str],
    revision: str,
    requested_start: datetime,
    requested_end_exclusive: datetime,
    *,
    metadata: dict[str, RemoteFileMetadata] | None = None,
    split_boundaries: Sequence[date] = (date(2023, 4, 11), date(2024, 8, 21)),
) -> tuple[SourceFilePlan, ...]:
    """Validate the EUR/USD inventory and select only overlapping shards."""

    if HF_REVISION_PATTERN.fullmatch(revision) is None:
        raise HfDukascopyValidationError(
            "an exact pinned dataset revision is mandatory"
        )
    start = _require_utc_minute(requested_start, "requested_start")
    end = _require_utc_minute(requested_end_exclusive, "requested_end_exclusive")
    if end <= start:
        raise HfDukascopyValidationError("requested interval must be non-empty")
    parsed: list[tuple[date, date, str]] = []
    for path in repo_files:
        if not path.startswith(SOURCE_DIRECTORY):
            continue
        match = _SHARD_PATTERN.fullmatch(path)
        if match is None:
            raise HfDukascopyValidationError(f"malformed EURUSD shard name: {path}")
        try:
            file_start = date.fromisoformat(match.group("start"))
            file_end = date.fromisoformat(match.group("end"))
        except ValueError as error:
            raise HfDukascopyValidationError(
                f"malformed EURUSD shard dates: {path}"
            ) from error
        if file_end < file_start:
            raise HfDukascopyValidationError(f"reversed EURUSD shard range: {path}")
        parsed.append((file_start, file_end, path))
    if not parsed:
        raise HfDukascopyValidationError("pinned repository contains no EURUSD shards")
    parsed.sort()
    seen_ranges: set[tuple[date, date]] = set()
    previous_end: date | None = None
    for file_start, file_end, path in parsed:
        key = (file_start, file_end)
        if key in seen_ranges:
            raise HfDukascopyValidationError(f"duplicate EURUSD shard range: {path}")
        seen_ranges.add(key)
        if previous_end is not None:
            if file_start <= previous_end:
                raise HfDukascopyValidationError(
                    f"overlapping EURUSD shard inventory at {path}"
                )
            if file_start != previous_end + timedelta(days=1):
                gap_start = previous_end + timedelta(days=1)
                gap_end = file_start - timedelta(days=1)
                raise HfDukascopyValidationError(
                    "unexplained date gap in EURUSD shard inventory: "
                    f"{gap_start}..{gap_end}"
                )
        previous_end = file_end

    remote = metadata or {}
    selected: list[SourceFilePlan] = []
    for file_start, file_end, path in parsed:
        file_start_utc = datetime.combine(file_start, datetime.min.time(), tzinfo=UTC)
        file_end_exclusive = datetime.combine(
            file_end + timedelta(days=1), datetime.min.time(), tzinfo=UTC
        )
        overlaps = file_start_utc < end and file_end_exclusive > start
        if not overlaps:
            continue
        details = remote.get(path, RemoteFileMetadata(None, None))
        selected.append(
            SourceFilePlan(
                repo_path=path,
                filename_start_date=file_start,
                filename_end_date=file_end,
                dataset_revision=revision,
                remote_size=details.size,
                remote_lfs_sha256=details.lfs_sha256,
                overlaps_g1_permitted_interval=overlaps,
                straddles_research_split_boundary=any(
                    file_start < boundary <= file_end for boundary in split_boundaries
                ),
            )
        )
    if not selected:
        raise HfDukascopyValidationError(
            "EURUSD inventory does not cover the requested interval"
        )
    if selected[0].filename_start_date > start.date() or (
        selected[-1].filename_end_date < (end - _MILLISECOND).date()
    ):
        raise HfDukascopyValidationError(
            "EURUSD shard inventory does not span the requested interval"
        )
    return tuple(selected)


def ingest_hf_eurusd(
    plan_path: Path,
    output_root: Path,
    *,
    requested_start: datetime | None = None,
    requested_end_exclusive: datetime | None = None,
    cache_dir: Path | None = None,
    api: HfApi | None = None,
    downloader: Callable[..., str] = hf_hub_download,
    batch_size: int = 65_536,
    on_admitted_batch: Callable[[pa.RecordBatch], None] | None = None,
) -> HfIngestionResult:
    """Download selected pinned shards and stream admitted ticks into 1m bars."""

    plan = load_research_data_plan(plan_path)
    start = requested_start or datetime.combine(
        plan.full_start, datetime.min.time(), tzinfo=UTC
    )
    end = requested_end_exclusive or plan.g1_allowed_end_exclusive
    start = _require_utc_minute(start, "requested_start")
    end = _require_utc_minute(end, "requested_end_exclusive")
    _validate_g1_request(plan, start, end)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    root = output_root.resolve()
    _reject_git_worktree_root(root)
    manifest_path = root / MANIFEST_FILENAME
    catalog_path = root / "catalog"
    if manifest_path.exists() or catalog_path.exists():
        raise HfDukascopyValidationError(
            "output root already contains canonical ingestion artifacts"
        )
    root.mkdir(parents=True, exist_ok=True)

    hub = api or HfApi()
    repo_files, remote_metadata = _pinned_inventory(hub, plan)
    files = discover_source_file_plan(
        repo_files,
        plan.huggingface_revision,
        start,
        end,
        metadata=remote_metadata,
    )

    catalog_path.mkdir(parents=True)
    catalog = ParquetDataCatalog(str(catalog_path))
    catalog.write_instruments([_eurusd_instrument()])
    state = _StreamState(
        catalog=catalog,
        current=None,
        previous_tick_ms=None,
        previous_file_last_ms=None,
        last_emitted_minute_ms=None,
        bid_pending=[],
        ask_pending=[],
        bar_count=0,
        next_expected_minute=start,
        missing_intervals=[],
    )
    source_results: list[SourceFileResult] = []
    total_ticks = 0
    for source in files:
        local = Path(
            downloader(
                repo_id=plan.huggingface_repo,
                filename=source.repo_path,
                repo_type=HF_REPO_TYPE,
                revision=plan.huggingface_revision,
                cache_dir=str(cache_dir) if cache_dir is not None else None,
            )
        )
        source_result = _process_source_file(
            local,
            source,
            start,
            end,
            state,
            batch_size=batch_size,
            on_admitted_batch=on_admitted_batch,
        )
        source_results.append(source_result)
        total_ticks += source_result.admitted_tick_count
    if total_ticks == 0 or state.current is None:
        raise HfDukascopyValidationError(
            "requested interval contains no admitted ticks"
        )
    _emit_current(state)
    _write_pending(state)
    if state.bar_count <= 0:
        raise HfDukascopyValidationError("tick aggregation emitted no minute bars")

    if state.next_expected_minute < end:
        state.missing_intervals.append(
            _missing_range(state.next_expected_minute, end - _ONE_MINUTE)
        )
    missing = tuple(state.missing_intervals)
    payload = _manifest_payload(
        plan=plan,
        start=start,
        end=end,
        catalog_path=catalog_path,
        root=root,
        source_results=source_results,
        admitted_tick_count=total_ticks,
        bar_count=state.bar_count,
        missing=missing,
    )
    semantic_sha256 = _semantic_sha256(payload)
    manifest = {**payload, "semantic_sha256": semantic_sha256}
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return HfIngestionResult(
        manifest_path=manifest_path,
        catalog_path=catalog_path,
        dataset_revision=plan.huggingface_revision,
        dataset_plan_sha256=plan.semantic_sha256,
        admitted_tick_count=total_ticks,
        bid_bar_count=state.bar_count,
        ask_bar_count=state.bar_count,
        missing_intervals=missing,
        semantic_sha256=semantic_sha256,
    )


def _pinned_inventory(
    api: HfApi, plan: ResearchDataPlan
) -> tuple[list[str], dict[str, RemoteFileMetadata]]:
    info = api.dataset_info(
        plan.huggingface_repo,
        revision=plan.huggingface_revision,
        files_metadata=True,
    )
    if info.sha != plan.huggingface_revision:
        raise HfDukascopyValidationError(
            "Hugging Face resolved revision differs from the frozen dataset plan"
        )
    repo_files = api.list_repo_files(
        plan.huggingface_repo,
        revision=plan.huggingface_revision,
        repo_type=HF_REPO_TYPE,
    )
    metadata: dict[str, RemoteFileMetadata] = {}
    for sibling in info.siblings or ():
        path = sibling.rfilename
        lfs = sibling.lfs
        lfs_sha = None if lfs is None else lfs.sha256
        metadata[path] = RemoteFileMetadata(sibling.size, lfs_sha)
    return list(repo_files), metadata


def _process_source_file(
    path: Path,
    source: SourceFilePlan,
    start: datetime,
    end: datetime,
    state: _StreamState,
    *,
    batch_size: int,
    on_admitted_batch: Callable[[pa.RecordBatch], None] | None,
) -> SourceFileResult:
    if not path.is_file():
        raise HfDukascopyValidationError(f"downloaded shard is missing: {path}")
    downloaded_size = path.stat().st_size
    digest = _sha256(path)
    if source.remote_size is not None and downloaded_size != source.remote_size:
        raise HfDukascopyValidationError(
            f"downloaded size differs from pinned metadata for {source.repo_path}"
        )
    if source.remote_lfs_sha256 is not None and digest != source.remote_lfs_sha256:
        raise HfDukascopyValidationError(
            f"downloaded SHA-256 differs from pinned metadata for {source.repo_path}"
        )
    parquet = pq.ParquetFile(path)
    _validate_schema(parquet.schema_arrow, source.repo_path)
    start_ms = _datetime_ms(start)
    end_ms = _datetime_ms(end)
    admitted_count = 0
    first_ms: int | None = None
    last_ms: int | None = None
    first_in_file: int | None = None
    for batch in parquet.iter_batches(
        batch_size=batch_size, columns=list(REQUIRED_COLUMNS)
    ):
        timestamp_column = batch.column(batch.schema.get_field_index("timestamp"))
        mask = pc.and_(
            pc.greater_equal(timestamp_column, pa.scalar(start_ms, pa.int64())),
            pc.less(timestamp_column, pa.scalar(end_ms, pa.int64())),
        )
        admitted = batch.filter(mask)
        if admitted.num_rows == 0:
            continue
        _validate_admitted_batch(admitted, source)
        if on_admitted_batch is not None:
            on_admitted_batch(admitted)
        columns = {
            name: admitted.column(admitted.schema.get_field_index(name)).to_pylist()
            for name in REQUIRED_COLUMNS
        }
        for index in range(admitted.num_rows):
            timestamp_ms = cast(int, columns["timestamp"][index])
            if first_in_file is None:
                first_in_file = timestamp_ms
                if (
                    state.previous_file_last_ms is not None
                    and timestamp_ms <= state.previous_file_last_ms
                ):
                    raise HfDukascopyValidationError(
                        "source files contain backwards or overlapping tick regions"
                    )
            if (
                state.previous_tick_ms is not None
                and timestamp_ms < state.previous_tick_ms
            ):
                raise HfDukascopyValidationError(
                    f"ticks are not monotonic nondecreasing in {source.repo_path}"
                )
            state.previous_tick_ms = timestamp_ms
            first_ms = timestamp_ms if first_ms is None else first_ms
            last_ms = timestamp_ms
            _consume_tick(
                state,
                timestamp_ms,
                _decimal_float(columns["bidPrice"][index]),
                _decimal_float(columns["askPrice"][index]),
                _decimal_float(columns["bidVolume"][index]),
                _decimal_float(columns["askVolume"][index]),
            )
            admitted_count += 1
    if last_ms is not None:
        state.previous_file_last_ms = last_ms
    return SourceFileResult(
        repo_path=source.repo_path,
        filename_start_date=source.filename_start_date.isoformat(),
        filename_end_date=source.filename_end_date.isoformat(),
        dataset_revision=source.dataset_revision,
        remote_size=source.remote_size,
        remote_lfs_sha256=source.remote_lfs_sha256,
        downloaded_size=downloaded_size,
        downloaded_sha256=digest,
        admitted_tick_count=admitted_count,
        admitted_min_timestamp_utc=(None if first_ms is None else _format_ms(first_ms)),
        admitted_max_timestamp_utc=(None if last_ms is None else _format_ms(last_ms)),
        overlaps_g1_permitted_interval=source.overlaps_g1_permitted_interval,
        straddles_research_split_boundary=source.straddles_research_split_boundary,
    )


def _validate_schema(schema: pa.Schema, source: str) -> None:
    if set(schema.names) != set(REQUIRED_COLUMNS) or len(schema.names) != len(
        REQUIRED_COLUMNS
    ):
        raise HfDukascopyValidationError(
            f"{source} must contain exactly the required semantic tick columns"
        )
    for name, expected_type in _EXPECTED_SCHEMA.items():
        field = schema.field(name)
        if field.type != expected_type:
            raise HfDukascopyValidationError(
                f"{source} has incompatible primitive type for {name}: {field.type}"
            )


def _validate_admitted_batch(batch: pa.RecordBatch, source: SourceFilePlan) -> None:
    for name in REQUIRED_COLUMNS:
        if batch.column(batch.schema.get_field_index(name)).null_count:
            raise HfDukascopyValidationError(
                f"{source.repo_path} contains null required fields"
            )
    timestamp_index = batch.schema.get_field_index("timestamp")
    timestamps = cast(list[int], batch.column(timestamp_index).to_pylist())
    _prove_millisecond_timestamp_unit(timestamps[0], source)
    previous: int | None = None
    for value in timestamps:
        if previous is not None and value < previous:
            raise HfDukascopyValidationError(
                f"ticks are not monotonic nondecreasing in {source.repo_path}"
            )
        converted = _datetime_from_ms(value)
        if (
            not source.filename_start_date
            <= converted.date()
            <= (source.filename_end_date)
        ):
            raise HfDukascopyValidationError(
                f"timestamp falls outside filename range in {source.repo_path}"
            )
        previous = value
    values = {
        name: cast(
            list[float], batch.column(batch.schema.get_field_index(name)).to_pylist()
        )
        for name in REQUIRED_COLUMNS[1:]
    }
    for index in range(batch.num_rows):
        ask = values["askPrice"][index]
        bid = values["bidPrice"][index]
        ask_volume = values["askVolume"][index]
        bid_volume = values["bidVolume"][index]
        if not all(
            math.isfinite(value) for value in (ask, bid, ask_volume, bid_volume)
        ):
            raise HfDukascopyValidationError(
                f"{source.repo_path} contains non-finite tick values"
            )
        if ask <= 0 or bid <= 0:
            raise HfDukascopyValidationError(
                f"{source.repo_path} contains nonpositive prices"
            )
        if ask < bid:
            raise HfDukascopyValidationError(
                f"{source.repo_path} contains ASK below BID"
            )
        if ask_volume < 0 or bid_volume < 0:
            raise HfDukascopyValidationError(
                f"{source.repo_path} contains negative volume"
            )


def _prove_millisecond_timestamp_unit(value: int, source: SourceFilePlan) -> None:
    matching: list[str] = []
    divisors = {
        "seconds": 1,
        "milliseconds": 1_000,
        "microseconds": 1_000_000,
        "nanoseconds": 1_000_000_000,
    }
    for unit, divisor in divisors.items():
        try:
            candidate = _EPOCH + timedelta(seconds=value / divisor)
        except (OverflowError, OSError):
            continue
        if source.filename_start_date <= candidate.date() <= source.filename_end_date:
            matching.append(unit)
    if matching != ["milliseconds"]:
        raise HfDukascopyValidationError(
            "timestamp unit cannot be established unambiguously from Parquet value, "
            f"UTC conversion, and filename range for {source.repo_path}: {matching}"
        )


def _consume_tick(
    state: _StreamState,
    timestamp_ms: int,
    bid: Decimal,
    ask: Decimal,
    bid_volume: Decimal,
    ask_volume: Decimal,
) -> None:
    minute_ms = (timestamp_ms // _MINUTE_MS) * _MINUTE_MS
    if state.current is None:
        state.current = _MinuteAggregate.from_tick(
            minute_ms, bid, ask, bid_volume, ask_volume
        )
        return
    if minute_ms == state.current.minute_ms:
        state.current.update(bid, ask, bid_volume, ask_volume)
        return
    if minute_ms < state.current.minute_ms:
        raise HfDukascopyValidationError("minute aggregation received backwards ticks")
    _emit_current(state)
    state.current = _MinuteAggregate.from_tick(
        minute_ms, bid, ask, bid_volume, ask_volume
    )


def _emit_current(state: _StreamState) -> None:
    item = state.current
    if item is None:
        return
    if state.last_emitted_minute_ms is not None and item.minute_ms <= (
        state.last_emitted_minute_ms
    ):
        raise HfDukascopyValidationError("minute bars are not strictly chronological")
    timestamp = _datetime_from_ms(item.minute_ms)
    if timestamp > state.next_expected_minute:
        state.missing_intervals.append(
            _missing_range(state.next_expected_minute, timestamp - _ONE_MINUTE)
        )
    state.next_expected_minute = timestamp + _ONE_MINUTE
    bid = SourceBar(
        timestamp,
        item.bid_open,
        item.bid_high,
        item.bid_low,
        item.bid_close,
        item.bid_volume,
    )
    ask = SourceBar(
        timestamp,
        item.ask_open,
        item.ask_high,
        item.ask_low,
        item.ask_close,
        item.ask_volume,
    )
    for field in ("open", "high", "low", "close"):
        if getattr(ask, field) < getattr(bid, field):
            raise HfDukascopyValidationError(
                f"aggregated ASK {field} is below BID at {_format_utc(timestamp)}"
            )
    state.bid_pending.append(bid)
    state.ask_pending.append(ask)
    state.bar_count += 1
    state.last_emitted_minute_ms = item.minute_ms
    if len(state.bid_pending) >= _WRITE_CHUNK_BARS:
        _write_pending(state)


def _write_pending(state: _StreamState) -> None:
    if not state.bid_pending:
        return
    bid_bars = _to_nautilus_bars(state.bid_pending, "BID")
    ask_bars = _to_nautilus_bars(state.ask_pending, "ASK")
    _validate_paired_bars(bid_bars, ask_bars)
    state.catalog.write_bars(bid_bars)
    state.catalog.write_bars(ask_bars)
    state.bid_pending.clear()
    state.ask_pending.clear()


def _validate_paired_bars(bid: Sequence[Bar], ask: Sequence[Bar]) -> None:
    if len(bid) != len(ask):
        raise HfDukascopyValidationError("aggregated BID/ASK bar counts differ")
    previous: int | None = None
    for bid_bar, ask_bar in zip(bid, ask, strict=True):
        if bid_bar.ts_event != ask_bar.ts_event or bid_bar.ts_init != ask_bar.ts_init:
            raise HfDukascopyValidationError("aggregated BID/ASK timestamps differ")
        if bid_bar.ts_init != bid_bar.ts_event + 60_000_000_000:
            raise HfDukascopyValidationError("canonical bar timestamps are invalid")
        if previous is not None and bid_bar.ts_event <= previous:
            raise HfDukascopyValidationError(
                "canonical bars are not strictly chronological"
            )
        previous = bid_bar.ts_event


def _missing_range(start: datetime, end: datetime) -> MissingInterval:
    return MissingInterval(
        start_utc=_format_utc(start),
        end_utc=_format_utc(end),
        minutes=int((end - start) / _ONE_MINUTE) + 1,
    )


def _manifest_payload(
    *,
    plan: ResearchDataPlan,
    start: datetime,
    end: datetime,
    catalog_path: Path,
    root: Path,
    source_results: Sequence[SourceFileResult],
    admitted_tick_count: int,
    bar_count: int,
    missing: Sequence[MissingInterval],
) -> dict[str, Any]:
    return {
        "provider": "Dukascopy",
        "market_data_origin": "Dukascopy",
        "distribution_source": "Hugging Face",
        "dataset_repo": plan.huggingface_repo,
        "dataset_revision": plan.huggingface_revision,
        "dataset_id": plan.dataset_id,
        "dataset_plan_sha256": plan.semantic_sha256,
        "symbol": "EURUSD",
        "instrument_id": INSTRUMENT_ID,
        "requested_utc_range": {
            "start_date": start.date().isoformat(),
            "end_date_inclusive": (end.date() - timedelta(days=1)).isoformat(),
        },
        "g1_cutoff_end_exclusive_utc": _format_utc(plan.g1_allowed_end_exclusive),
        "source_granularity": "1-minute",
        "price_sides": ["BID", "ASK"],
        "source_timestamp_convention": ("raw_unix_epoch_milliseconds_to_bar_open_utc"),
        "timestamp_unit_proof": {
            "raw_primitive_type": "int64",
            "established_unit": "milliseconds",
            "method": (
                "for every admitted shard, raw values converted under seconds, "
                "milliseconds, microseconds, and nanoseconds; exactly milliseconds "
                "maps into the filename-declared UTC date range"
            ),
        },
        "aggregation": {
            "interval": "UTC minute containing at least one admitted tick",
            "open": "first source row in deterministic source order",
            "high": "maximum side price",
            "low": "minimum side price",
            "close": "last source row in deterministic source order",
            "volume": "sum of same-side source volume",
            "timestamp_ties": "original Parquet row order",
            "fills_or_interpolation": False,
        },
        "fills_or_interpolation": False,
        "source_files": [asdict(item) for item in source_results],
        "catalog_location": catalog_path.relative_to(root).as_posix(),
        "ingestion_version": HF_INGESTION_VERSION,
        "config_version": 1,
        "nautilus_version": NAUTILUS_VERSION,
        "nautilus_encoding": {
            "price_precision": PRICE_PRECISION,
            "size_precision": SIZE_PRECISION,
            "ts_event": "bar_open_utc",
            "ts_init": "bar_close_utc",
        },
        "qa": {
            "admitted_tick_count": admitted_tick_count,
            "holdout_rows_admitted": 0,
            "bid_bar_count": bar_count,
            "ask_bar_count": bar_count,
            "missing_interval_count": sum(item.minutes for item in missing),
            "missing_intervals_utc": [asdict(item) for item in missing],
            "gaps_filled": False,
        },
        "bounded_memory": {
            "parquet_processing": "one file and one record batch at a time",
            "cross_batch_file_state": "current minute plus previous timestamp",
            "catalog_write_chunk_bars": _WRITE_CHUNK_BARS,
        },
        "semantic_hash_contract": (
            "SHA-256 of canonical JSON for every other manifest field; no fetch "
            "or wall-clock timestamp is included"
        ),
    }


def _semantic_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validate_g1_request(
    plan: ResearchDataPlan, start: datetime, end: datetime
) -> None:
    full_start = datetime.combine(plan.full_start, datetime.min.time(), tzinfo=UTC)
    if start < full_start:
        raise HfDukascopyValidationError("G1 request starts before the frozen universe")
    if start >= plan.g1_allowed_end_exclusive:
        raise HfDukascopyValidationError(
            "G1 cannot request a range entirely inside the final holdout"
        )
    if end > plan.g1_allowed_end_exclusive:
        raise HfDukascopyValidationError("G1 request crosses the final-holdout cutoff")
    if end <= start:
        raise HfDukascopyValidationError("G1 requested interval must be non-empty")
    if start.time() != datetime.min.time() or end.time() != datetime.min.time():
        raise HfDukascopyValidationError("G1 requests must use whole UTC days")


def _require_utc_minute(value: datetime, name: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != timedelta(0)
        or value.second != 0
        or value.microsecond != 0
    ):
        raise HfDukascopyValidationError(f"{name} must be an exact UTC minute")
    return value.astimezone(UTC)


def _datetime_ms(value: datetime) -> int:
    delta = value - _EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000 + delta.microseconds // 1_000


def _datetime_from_ms(value: int) -> datetime:
    try:
        return _EPOCH + timedelta(milliseconds=value)
    except (OverflowError, OSError) as error:
        raise HfDukascopyValidationError(
            "timestamp is outside supported UTC range"
        ) from error


def _format_ms(value: int) -> str:
    timestamp = _datetime_from_ms(value)
    return timestamp.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _format_utc(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _decimal_float(value: object) -> Decimal:
    if not isinstance(value, float):
        raise HfDukascopyValidationError(
            "validated float column returned invalid value"
        )
    return Decimal(str(value))


def _date_argument(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must use YYYY-MM-DD") from error
    if parsed.isoformat() != value:
        raise argparse.ArgumentTypeError("must use YYYY-MM-DD")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import pinned Hugging Face Dukascopy EUR/USD ticks"
    )
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--start", type=_date_argument)
    parser.add_argument("--end-exclusive", type=_date_argument)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the revision-pinned Hugging Face ingestion command."""

    args = _build_parser().parse_args(argv)
    start = (
        None
        if args.start is None
        else datetime.combine(cast(date, args.start), datetime.min.time(), tzinfo=UTC)
    )
    end = (
        None
        if args.end_exclusive is None
        else datetime.combine(
            cast(date, args.end_exclusive), datetime.min.time(), tzinfo=UTC
        )
    )
    if (start is None) != (end is None):
        raise SystemExit("--start and --end-exclusive must be supplied together")
    result = ingest_hf_eurusd(
        cast(Path, args.plan),
        cast(Path, args.output_root),
        requested_start=start,
        requested_end_exclusive=end,
        cache_dir=cast(Path | None, args.cache_dir),
    )
    print(result.manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
