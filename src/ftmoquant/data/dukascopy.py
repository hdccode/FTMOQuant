"""Reproducible Dukascopy EUR/USD source-to-Nautilus catalog pipeline."""

import argparse
import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import pyarrow as pa  # type: ignore[import-untyped]
from nautilus_trader.model import (
    Bar,
    Currency,
    CurrencyPair,
    InstrumentId,
    Price,
    Quantity,
    Symbol,
)
from nautilus_trader.persistence import BarDataWrangler, ParquetDataCatalog
from tradedesk_dukascopy.cli import (  # type: ignore[import-untyped]
    main as dukascopy_export_main,
)

SYMBOL = "EURUSD"
INSTRUMENT_ID = "EUR/USD.DUKASCOPY"
SOURCE_GRANULARITY = "1-minute"
INGESTION_VERSION = "g0.5-1"
CONFIG_VERSION = 1
UPSTREAM_VERSION = "1.0.0"
UPSTREAM_COMMIT = "b8fb503c9291d6e265949d008e288b76b68fb852"
UPSTREAM_LICENSE = "Apache-2.0"
NAUTILUS_VERSION = "2.0.0rc2"
PRICE_PRECISION = 5
SIZE_PRECISION = 8

_ONE_MINUTE = timedelta(minutes=1)
_PRICE_QUANTUM = Decimal("0.00001")
_VOLUME_QUANTUM = Decimal("0.00000001")
_FLOAT_TEXT_TOLERANCE = Decimal("0.000000001")
_EURUSD_SANITY_MIN = Decimal("0.1")
_EURUSD_SANITY_MAX = Decimal("10")
_CSV_FIELDS = ("timestamp", "open", "high", "low", "close", "volume")


class IngestionValidationError(ValueError):
    """Raised before catalog writes when source data is not trustworthy."""


@dataclass(frozen=True, slots=True)
class SourceBar:
    """One validated upstream source row."""

    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


@dataclass(frozen=True, slots=True)
class MissingInterval:
    """An inclusive contiguous range of absent one-minute timestamps."""

    start_utc: str
    end_utc: str
    minutes: int


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Stable locations and QA summary from one ingestion."""

    manifest_path: Path
    catalog_path: Path
    bid_bar_count: int
    ask_bar_count: int
    missing_intervals: tuple[MissingInterval, ...]


def probe_eurusd_scale(
    start_date: date,
    end_date: date,
    output_root: Path,
    probe_ticks: int = 10,
) -> None:
    """Run the upstream BI5 format/scale probe without writing source bars."""

    _validate_dates(start_date, end_date)
    root = output_root.resolve()
    _reject_git_worktree_root(root)
    cache_path = root / "upstream_cache"
    exit_code = dukascopy_export_main(
        [
            "--symbols",
            SYMBOL,
            "--from",
            start_date.isoformat(),
            "--to",
            end_date.isoformat(),
            "--cache-dir",
            str(cache_path),
            "--workers",
            "1",
            "--probe",
            "--probe-ticks",
            str(probe_ticks),
        ]
    )
    if exit_code != 0:
        raise RuntimeError(
            f"upstream Dukascopy scale probe failed with code {exit_code}"
        )


def ingest_eurusd(
    start_date: date,
    end_date: date,
    output_root: Path,
    price_divisor: Decimal,
    *,
    acquire: bool = True,
) -> IngestionResult:
    """Acquire, validate, and catalog one inclusive UTC date range."""

    _validate_runtime(start_date, end_date, price_divisor)
    root = output_root.resolve()
    _reject_git_worktree_root(root)
    source_path = root / "source"
    catalog_path = root / "catalog"
    source_path.mkdir(parents=True, exist_ok=True)

    if acquire:
        _export_source(start_date, end_date, root, source_path, price_divisor)

    bid_path = source_path / "EURUSD_1MIN_bid.csv"
    ask_path = source_path / "EURUSD_1MIN_ask.csv"
    bid_sidecar = bid_path.with_suffix(".csv.meta.json")
    ask_sidecar = ask_path.with_suffix(".csv.meta.json")
    _require_files((bid_path, ask_path, bid_sidecar, ask_sidecar))
    _validate_sidecar(bid_sidecar, "bid", start_date, end_date, price_divisor)
    _validate_sidecar(ask_sidecar, "ask", start_date, end_date, price_divisor)

    start_utc = datetime.combine(start_date, datetime.min.time(), tzinfo=UTC)
    end_exclusive = datetime.combine(
        end_date + timedelta(days=1), datetime.min.time(), tzinfo=UTC
    )
    bid = _load_and_validate_side(bid_path, "bid", start_utc, end_exclusive)
    ask = _load_and_validate_side(ask_path, "ask", start_utc, end_exclusive)
    _validate_coverage_and_spread(bid, ask)
    missing = _missing_intervals(bid, start_utc, end_exclusive)

    instrument = _eurusd_instrument()
    bid_bars = _to_nautilus_bars(bid, "BID")
    ask_bars = _to_nautilus_bars(ask, "ASK")
    catalog_path.mkdir(parents=True, exist_ok=True)
    catalog = ParquetDataCatalog(str(catalog_path))
    _write_catalog(catalog, instrument, bid_bars, ask_bars)

    manifest_path = root / "ftmoquant_provenance.json"
    manifest = _manifest(
        root=root,
        catalog_path=catalog_path,
        start_date=start_date,
        end_date=end_date,
        price_divisor=price_divisor,
        source_files=(bid_path, ask_path),
        sidecars=(bid_sidecar, ask_sidecar),
        bid_count=len(bid_bars),
        ask_count=len(ask_bars),
        missing=missing,
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return IngestionResult(
        manifest_path=manifest_path,
        catalog_path=catalog_path,
        bid_bar_count=len(bid_bars),
        ask_bar_count=len(ask_bars),
        missing_intervals=missing,
    )


def _export_source(
    start_date: date,
    end_date: date,
    root: Path,
    source_path: Path,
    price_divisor: Decimal,
) -> None:
    exit_code = dukascopy_export_main(
        [
            "--symbols",
            SYMBOL,
            "--from",
            start_date.isoformat(),
            "--to",
            end_date.isoformat(),
            "--resample",
            "1min",
            "--out",
            str(source_path),
            "--cache-dir",
            str(root / "upstream_cache"),
            "--price-divisor",
            format(price_divisor, "f"),
            "--workers",
            "1",
        ]
    )
    if exit_code != 0:
        raise RuntimeError(f"upstream Dukascopy export failed with code {exit_code}")


def _validate_runtime(
    start_date: date,
    end_date: date,
    price_divisor: Decimal,
) -> None:
    _validate_dates(start_date, end_date)
    if (
        not isinstance(price_divisor, Decimal)
        or not price_divisor.is_finite()
        or price_divisor <= 0
    ):
        raise ValueError(
            "price_divisor must be an explicit positive Decimal selected "
            "after the upstream --probe workflow"
        )
    if version("tradedesk-dukascopy") != UPSTREAM_VERSION:
        raise RuntimeError(f"expected tradedesk-dukascopy {UPSTREAM_VERSION}")
    if version("nautilus-trader") != NAUTILUS_VERSION:
        raise RuntimeError(f"expected NautilusTrader {NAUTILUS_VERSION}")


def _validate_dates(start_date: date, end_date: date) -> None:
    if not isinstance(start_date, date) or not isinstance(end_date, date):
        raise ValueError("start and end must be explicit UTC dates")
    if end_date < start_date:
        raise ValueError("end date must be on or after start date")


def _reject_git_worktree_root(root: Path) -> None:
    if any((candidate / ".git").exists() for candidate in (root, *root.parents)):
        raise ValueError(
            "output_root must be outside a Git worktree so source data and the "
            "Nautilus catalog cannot be tracked"
        )


def _require_files(paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise IngestionValidationError(
            "upstream export is incomplete; missing: " + ", ".join(missing)
        )


def _validate_sidecar(
    path: Path,
    side: str,
    start_date: date,
    end_date: date,
    price_divisor: Decimal,
) -> None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IngestionValidationError(
            f"invalid metadata sidecar {path}: {error}"
        ) from error
    if not isinstance(raw, dict):
        raise IngestionValidationError(f"metadata sidecar {path} must be an object")
    metadata = cast(dict[str, Any], raw)
    params_raw = metadata.get("params")
    if not isinstance(params_raw, dict):
        raise IngestionValidationError(f"metadata sidecar {path} has invalid params")
    params = cast(dict[str, Any], params_raw)
    expected = {
        "source": "dukascopy",
        "symbol": SYMBOL,
        "data_type": "candles",
        "timestamp_format": "iso8601_utc",
        "schema_version": "1",
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise IngestionValidationError(
                f"metadata sidecar {path} has invalid {field}"
            )
    expected_params = {
        "date_from": start_date.isoformat(),
        "date_to": end_date.isoformat(),
        "resample": "1min",
        "price_side": side,
    }
    for field, value in expected_params.items():
        if params.get(field) != value:
            raise IngestionValidationError(
                f"metadata sidecar {path} has invalid params.{field}"
            )
    try:
        metadata_divisor = Decimal(str(metadata["price_divisor"]))
    except (KeyError, InvalidOperation) as error:
        raise IngestionValidationError(
            f"metadata sidecar {path} has invalid price_divisor"
        ) from error
    if metadata_divisor != price_divisor:
        raise IngestionValidationError(
            f"metadata sidecar {path} price_divisor does not match "
            "the requested divisor"
        )


def _load_and_validate_side(
    path: Path,
    side: str,
    start_utc: datetime,
    end_exclusive: datetime,
) -> tuple[SourceBar, ...]:
    bars: list[SourceBar] = []
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or tuple(reader.fieldnames) != _CSV_FIELDS:
                raise IngestionValidationError(
                    f"{path} must have columns {', '.join(_CSV_FIELDS)}"
                )
            for line_number, row in enumerate(reader, start=2):
                bars.append(_parse_row(path, line_number, row))
    except (OSError, csv.Error) as error:
        raise IngestionValidationError(f"could not read {path}: {error}") from error
    if not bars:
        raise IngestionValidationError(f"{side} source dataset is empty")

    timestamps = [bar.timestamp for bar in bars]
    if any(
        current <= previous
        for previous, current in zip(timestamps, timestamps[1:])
    ):
        if len(timestamps) != len(set(timestamps)):
            raise IngestionValidationError(f"{side} contains duplicate timestamps")
        raise IngestionValidationError(
            f"{side} timestamps are not strictly chronological"
        )
    if timestamps[0] < start_utc or timestamps[-1] >= end_exclusive:
        raise IngestionValidationError(
            f"{side} timestamps fall outside the requested UTC range"
        )
    _validate_price_scale(bars, side)
    return tuple(bars)


def _parse_row(path: Path, line_number: int, row: Mapping[str, str]) -> SourceBar:
    timestamp_text = row.get("timestamp")
    if not isinstance(timestamp_text, str):
        raise IngestionValidationError(
            f"{path}:{line_number} has a missing timestamp"
        )
    try:
        timestamp = datetime.fromisoformat(timestamp_text.replace("Z", "+00:00"))
    except ValueError as error:
        raise IngestionValidationError(
            f"{path}:{line_number} has an invalid timestamp"
        ) from error
    if (
        timestamp.tzinfo is None
        or timestamp.utcoffset() != timedelta(0)
        or timestamp.second != 0
        or timestamp.microsecond != 0
    ):
        raise IngestionValidationError(
            f"{path}:{line_number} timestamp must be an exact UTC minute"
        )
    timestamp = timestamp.astimezone(UTC)

    values: dict[str, Decimal] = {}
    for field in ("open", "high", "low", "close", "volume"):
        text = row.get(field)
        if not isinstance(text, str):
            raise IngestionValidationError(
                f"{path}:{line_number} has missing {field}"
            )
        try:
            value = Decimal(text)
        except InvalidOperation as error:
            raise IngestionValidationError(
                f"{path}:{line_number} has invalid {field}"
            ) from error
        if not value.is_finite():
            raise IngestionValidationError(
                f"{path}:{line_number} has non-finite {field}"
            )
        values[field] = value

    prices = [values[field] for field in ("open", "high", "low", "close")]
    if any(value <= 0 for value in prices):
        raise IngestionValidationError(
            f"{path}:{line_number} prices must be positive"
        )
    if values["volume"] < 0:
        raise IngestionValidationError(
            f"{path}:{line_number} volume must be non-negative"
        )
    if (
        values["high"] < max(values["open"], values["low"], values["close"])
        or values["low"] > min(values["open"], values["high"], values["close"])
    ):
        raise IngestionValidationError(
            f"{path}:{line_number} has invalid OHLC ordering"
        )
    return SourceBar(timestamp=timestamp, **values)


def _validate_price_scale(bars: Sequence[SourceBar], side: str) -> None:
    prices = [
        value
        for bar in bars
        for value in (bar.open, bar.high, bar.low, bar.close)
    ]
    if min(prices) < _EURUSD_SANITY_MIN or max(prices) > _EURUSD_SANITY_MAX:
        raise IngestionValidationError(
            f"{side} EUR/USD price scale is uncertain: observed range "
            f"{min(prices)}..{max(prices)} is outside the broad natural-rate "
            "envelope; rerun the upstream scale probe and select the divisor explicitly"
        )


def _validate_coverage_and_spread(
    bid: Sequence[SourceBar],
    ask: Sequence[SourceBar],
) -> None:
    bid_times = tuple(bar.timestamp for bar in bid)
    ask_times = tuple(bar.timestamp for bar in ask)
    if bid_times != ask_times:
        bid_only = sorted(set(bid_times) - set(ask_times))
        ask_only = sorted(set(ask_times) - set(bid_times))
        raise IngestionValidationError(
            "BID/ASK coverage is incompatible: "
            f"{len(bid_only)} BID-only and {len(ask_only)} ASK-only timestamps"
        )
    for bid_bar, ask_bar in zip(bid, ask, strict=True):
        for field in ("open", "high", "low", "close"):
            if getattr(ask_bar, field) < getattr(bid_bar, field):
                raise IngestionValidationError(
                    f"ASK {field} is below BID at {bid_bar.timestamp.isoformat()}"
                )


def _missing_intervals(
    bars: Sequence[SourceBar],
    start_utc: datetime,
    end_exclusive: datetime,
) -> tuple[MissingInterval, ...]:
    ranges: list[MissingInterval] = []
    expected = start_utc
    for bar in bars:
        if bar.timestamp > expected:
            ranges.append(_missing_range(expected, bar.timestamp - _ONE_MINUTE))
        expected = bar.timestamp + _ONE_MINUTE
    if expected < end_exclusive:
        ranges.append(_missing_range(expected, end_exclusive - _ONE_MINUTE))
    return tuple(ranges)


def _missing_range(start: datetime, end: datetime) -> MissingInterval:
    return MissingInterval(
        start_utc=start.isoformat().replace("+00:00", "Z"),
        end_utc=end.isoformat().replace("+00:00", "Z"),
        minutes=int((end - start) / _ONE_MINUTE) + 1,
    )


def _to_nautilus_bars(rows: Sequence[SourceBar], price_type: str) -> list[Bar]:
    bar_type = f"{INSTRUMENT_ID}-1-MINUTE-{price_type}-EXTERNAL"
    arrays = [
        pa.array(
            [_price_bytes(getattr(row, field)) for row in rows],
            type=pa.binary(16),
        )
        for field in ("open", "high", "low", "close")
    ]
    arrays.extend(
        [
            pa.array([_volume_bytes(row.volume) for row in rows], type=pa.binary(16)),
            pa.array([_timestamp_ns(row.timestamp) for row in rows], type=pa.uint64()),
            pa.array(
                [_timestamp_ns(row.timestamp + _ONE_MINUTE) for row in rows],
                type=pa.uint64(),
            ),
        ]
    )
    schema = pa.schema(
        [
            pa.field("open", pa.binary(16), nullable=False),
            pa.field("high", pa.binary(16), nullable=False),
            pa.field("low", pa.binary(16), nullable=False),
            pa.field("close", pa.binary(16), nullable=False),
            pa.field("volume", pa.binary(16), nullable=False),
            pa.field("ts_event", pa.uint64(), nullable=False),
            pa.field("ts_init", pa.uint64(), nullable=False),
        ]
    )
    batch = pa.RecordBatch.from_arrays(arrays, schema=schema)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, schema) as writer:
        writer.write_batch(batch)
    wrangler = BarDataWrangler(bar_type, PRICE_PRECISION, SIZE_PRECISION)
    return wrangler.process_record_batch_bytes(sink.getvalue().to_pybytes())


def _price_bytes(value: Decimal) -> bytes:
    quantized = value.quantize(_PRICE_QUANTUM)
    if abs(value - quantized) > _FLOAT_TEXT_TOLERANCE:
        raise IngestionValidationError(
            f"EUR/USD price {value} cannot be represented at "
            f"precision {PRICE_PRECISION}"
        )
    raw = Price.from_str(format(quantized, f".{PRICE_PRECISION}f")).raw
    return raw.to_bytes(16, byteorder="little", signed=True)


def _volume_bytes(value: Decimal) -> bytes:
    quantized = value.quantize(_VOLUME_QUANTUM)
    raw = Quantity.from_str(format(quantized, f".{SIZE_PRECISION}f")).raw
    return raw.to_bytes(16, byteorder="little", signed=False)


def _timestamp_ns(timestamp: datetime) -> int:
    delta = timestamp - datetime(1970, 1, 1, tzinfo=UTC)
    return (
        (delta.days * 86_400 + delta.seconds) * 1_000_000_000
        + delta.microseconds * 1_000
    )


def _eurusd_instrument() -> CurrencyPair:
    return CurrencyPair(
        instrument_id=InstrumentId.from_str(INSTRUMENT_ID),
        raw_symbol=Symbol(SYMBOL),
        base_currency=Currency.from_str("EUR"),
        quote_currency=Currency.from_str("USD"),
        price_precision=PRICE_PRECISION,
        size_precision=SIZE_PRECISION,
        price_increment=Price.from_str("0.00001"),
        size_increment=Quantity.from_str("0.00000001"),
        ts_event=0,
        ts_init=0,
        info={"provider": "dukascopy", "source_granularity": SOURCE_GRANULARITY},
    )


def _write_catalog(
    catalog: ParquetDataCatalog,
    instrument: CurrencyPair,
    bid_bars: Sequence[Bar],
    ask_bars: Sequence[Bar],
) -> None:
    existing_instruments = catalog.instruments([INSTRUMENT_ID])
    if not existing_instruments:
        catalog.write_instruments([instrument])
    for bars in (bid_bars, ask_bars):
        identifier = str(bars[0].bar_type)
        existing = catalog.query_bars([identifier])
        if existing:
            if existing != list(bars):
                raise IngestionValidationError(
                    f"catalog already contains conflicting data for {identifier}"
                )
            continue
        catalog.write_bars(bars)


def _manifest(
    *,
    root: Path,
    catalog_path: Path,
    start_date: date,
    end_date: date,
    price_divisor: Decimal,
    source_files: Sequence[Path],
    sidecars: Sequence[Path],
    bid_count: int,
    ask_count: int,
    missing: Sequence[MissingInterval],
) -> dict[str, Any]:
    return {
        "provider": "Dukascopy",
        "symbol": SYMBOL,
        "instrument_id": INSTRUMENT_ID,
        "requested_utc_range": {
            "start_date": start_date.isoformat(),
            "end_date_inclusive": end_date.isoformat(),
        },
        "source_granularity": SOURCE_GRANULARITY,
        "price_sides": ["BID", "ASK"],
        "source_timestamp_convention": "bar_open_utc",
        "upstream_downloader": {
            "package": "tradedesk-dukascopy",
            "version": UPSTREAM_VERSION,
            "repository": "https://github.com/radiusred/tradedesk-dukascopy",
            "commit": UPSTREAM_COMMIT,
            "license": UPSTREAM_LICENSE,
            "price_divisor": format(price_divisor, "f"),
        },
        "nautilus_version": NAUTILUS_VERSION,
        "source_filenames": [_relative(path, root) for path in source_files],
        "source_metadata_sidecar_filenames": [
            _relative(path, root) for path in sidecars
        ],
        "source_sha256": {
            _relative(path, root): _sha256(path) for path in source_files
        },
        "catalog_location": _relative(catalog_path, root),
        "ingestion_version": INGESTION_VERSION,
        "config_version": CONFIG_VERSION,
        "nautilus_encoding": {
            "price_precision": PRICE_PRECISION,
            "size_precision": SIZE_PRECISION,
            "ts_event": "bar_open_utc",
            "ts_init": "bar_close_utc",
        },
        "qa": {
            "bid_bar_count": bid_count,
            "ask_bar_count": ask_count,
            "missing_interval_count": sum(item.minutes for item in missing),
            "missing_intervals_utc": [asdict(item) for item in missing],
            "gaps_filled": False,
        },
    }


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _date_argument(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must use YYYY-MM-DD") from error
    if parsed.isoformat() != value:
        raise argparse.ArgumentTypeError("must use YYYY-MM-DD")
    return parsed


def _decimal_argument(value: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError("must be a decimal number") from error
    if not parsed.is_finite() or parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest UTC EUR/USD Dukascopy 1-minute BID/ASK bars"
    )
    parser.add_argument("--start", required=True, type=_date_argument)
    parser.add_argument("--end", required=True, type=_date_argument)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--price-divisor", type=_decimal_argument)
    parser.add_argument(
        "--probe-scale",
        action="store_true",
        help="run the upstream BI5 format/scale probe and exit",
    )
    parser.add_argument(
        "--use-existing-source",
        action="store_true",
        help="validate already-exported source files without network access",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the EUR/USD ingestion command."""

    args = _build_parser().parse_args(argv)
    start_date = cast(date, args.start)
    end_date = cast(date, args.end)
    output_root = cast(Path, args.output_root)
    if args.probe_scale:
        if args.price_divisor is not None or args.use_existing_source:
            raise SystemExit(
                "--probe-scale cannot be combined with --price-divisor or "
                "--use-existing-source"
            )
        probe_eurusd_scale(start_date, end_date, output_root)
        return 0
    if args.price_divisor is None:
        raise SystemExit(
            "--price-divisor is required; run --probe-scale first when the "
            "Dukascopy encoding scale is uncertain"
        )
    result = ingest_eurusd(
        start_date=start_date,
        end_date=end_date,
        output_root=output_root,
        price_divisor=cast(Decimal, args.price_divisor),
        acquire=not cast(bool, args.use_existing_source),
    )
    print(result.manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
