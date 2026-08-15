"""Immutable, GBP/USD-only completed-15m cache for frozen Leo research.

This deliberately reads no candidate returns.  The expensive catalog is used
only by :func:`prepare_leo_gbpusd_15m_cache`; evaluation verifies and reads the
small parquet file instead.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from time import perf_counter
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from nautilus_trader.model import Bar, CurrencyPair
from nautilus_trader.persistence import ParquetDataCatalog

from ftmoquant.backtest.execution_harness import _sha256_tree
from ftmoquant.data.dukascopy import SourceBar
from ftmoquant.data.instruments import GBPUSD_SPEC, to_nautilus_bars
from ftmoquant.research.leo_gbpusd_spec import LEO_GBPUSD_CONFIG_SHA256
from ftmoquant.research.stage_g import (
    DEVELOPMENT_END_EXCLUSIVE,
    DEVELOPMENT_START,
    DevelopmentResearchContext,
    ResearchPartition,
    open_development_context,
)
from ftmoquant.research.ts_momentum_development import TsMomentumEvaluationError
from ftmoquant.strategies.leo_gbpusd import LeoCompleted15mBar
from ftmoquant.strategies.trend_pullback import PriceBar

_GBPUSD = "GBP/USD.DUKASCOPY"
_MINUTE_NS = 60_000_000_000
_FIFTEEN = timedelta(minutes=15)
_DATA_FILE = "gbpusd_15m.parquet"
_MANIFEST_FILE = "manifest.json"
_SCHEMA = "ftmoquant.leo-gbpusd-15m-cache"
_VERSION = 1


def prepare_leo_gbpusd_15m_cache(
    *,
    universe_readiness_path: Path,
    development_roots: Mapping[str, Path],
    cache_dir: Path,
) -> dict[str, Any]:
    """Build once from exactly two GBP/USD 1m catalog queries, then seal it."""
    if cache_dir.exists():
        raise FileExistsError(f"cache path already exists: {cache_dir}")
    started = perf_counter()
    log = logging.getLogger(__name__)
    log.info("Leo cache: verifying frozen DEVELOPMENT provenance")
    context = open_development_context(universe_readiness_path, development_roots)
    context.require_range(
        DEVELOPMENT_START,
        DEVELOPMENT_END_EXCLUSIVE,
        partition=ResearchPartition.DEVELOPMENT,
    )
    source_tree = _validate_gbp_source(context, development_roots)
    log.info(
        "Leo cache: source verification complete in %.2fs", perf_counter() - started
    )
    root = development_roots[_GBPUSD]
    catalog = ParquetDataCatalog(str(root / "catalog"))
    found = catalog.instruments([_GBPUSD])
    if len(found) != 1 or not isinstance(found[0], CurrencyPair):
        raise TsMomentumEvaluationError("frozen GBP/USD CurrencyPair is missing")
    start_ns, end_ns = _ns(DEVELOPMENT_START), _ns(DEVELOPMENT_END_EXCLUSIVE)
    by_side: dict[str, dict[int, Bar]] = {}
    for side in ("BID", "ASK"):
        identity = f"{_GBPUSD}-1-MINUTE-{side}-EXTERNAL"
        # This is intentionally the only market-row access in cache preparation.
        queried = catalog.query_bars([identity], start=start_ns, end=end_ns - 1)
        selected = {
            bar.ts_event: bar
            for bar in queried
            if start_ns <= bar.ts_event
            and bar.ts_init < end_ns
            and str(bar.bar_type) == identity
        }
        if not selected or len(selected) != len(queried):
            raise TsMomentumEvaluationError(
                "GBP/USD source query is empty, duplicate, or out of range"
            )
        by_side[side] = selected
    rows = _aggregate_complete_15m(by_side["BID"], by_side["ASK"])
    if not rows:
        raise TsMomentumEvaluationError("GBP/USD source produced no complete 15m bars")
    cache_dir.mkdir(parents=True)
    data_path = cache_dir / _DATA_FILE
    _write_rows(data_path, rows)
    output_hash = _sha256_file(data_path)
    manifest = {
        "schema": _SCHEMA,
        "schema_version": _VERSION,
        "source_instrument": _GBPUSD,
        "development_boundary": {
            "start_utc": _format(DEVELOPMENT_START),
            "end_exclusive_utc": _format(DEVELOPMENT_END_EXCLUSIVE),
        },
        "source_readiness_sha256": context.universe.readiness_sha256,
        "source_catalog_tree_sha256": source_tree,
        "source_bar_types": [
            f"{_GBPUSD}-1-MINUTE-BID-EXTERNAL",
            f"{_GBPUSD}-1-MINUTE-ASK-EXTERNAL",
        ],
        "aggregation": {
            "interval_minutes": 15,
            "semantics": "exact contiguous completed 1m bid/ask OHLC; ts_init=end",
        },
        "timezone_convention": (
            "UTC storage; Europe/London strategy session interpretation"
        ),
        "row_count": len(rows),
        "first_timestamp_utc": _format(rows[0].start_time_utc),
        "last_timestamp_utc": _format(rows[-1].start_time_utc),
        "output_file": _DATA_FILE,
        "output_file_sha256": output_hash,
        "frozen_strategy_sha256": LEO_GBPUSD_CONFIG_SHA256,
        "validation_accessed": False,
        "final_holdout_accessed": False,
    }
    manifest["semantic_sha256"] = _semantic_hash(manifest)
    (cache_dir / _MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    log.info(
        "Leo cache: prepared %d completed 15m rows in %.2fs",
        len(rows),
        perf_counter() - started,
    )
    return manifest


def load_leo_gbpusd_15m_cache(
    *,
    cache_dir: Path,
    context: DevelopmentResearchContext,
    development_roots: Mapping[str, Path],
) -> tuple[LeoCompleted15mBar, ...]:
    """Verify every identity before reading compact rows; never opens a catalog."""
    started = perf_counter()
    manifest_path = cache_dir / _MANIFEST_FILE
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TsMomentumEvaluationError(
            f"Leo cache manifest is unreadable: {error}"
        ) from error
    _validate_manifest(manifest, context, development_roots, cache_dir)
    rows = _read_rows(cache_dir / _DATA_FILE)
    if len(rows) != manifest["row_count"]:
        raise TsMomentumEvaluationError("Leo cache row count drifted")
    if (
        not rows
        or _format(rows[0].start_time_utc) != manifest["first_timestamp_utc"]
        or _format(rows[-1].start_time_utc) != manifest["last_timestamp_utc"]
    ):
        raise TsMomentumEvaluationError("Leo cache timestamps drifted")
    logging.getLogger(__name__).info(
        "Leo cache: loaded %d rows in %.2fs", len(rows), perf_counter() - started
    )
    return rows


def cache_rows_to_native_bars(rows: Sequence[LeoCompleted15mBar]) -> tuple[Bar, ...]:
    """Convert only compact cached rows at the native execution boundary."""
    bid = [
        SourceBar(
            item.start_time_utc,
            item.bid.open,
            item.bid.high,
            item.bid.low,
            item.bid.close,
            Decimal(1),
        )
        for item in rows
    ]
    ask = [
        SourceBar(
            item.start_time_utc,
            item.ask.open,
            item.ask.high,
            item.ask.low,
            item.ask.close,
            Decimal(1),
        )
        for item in rows
    ]
    return tuple(
        to_nautilus_bars(bid, "BID", GBPUSD_SPEC, minutes=15)
        + to_nautilus_bars(ask, "ASK", GBPUSD_SPEC, minutes=15)
    )


def _aggregate_complete_15m(
    bids: Mapping[int, Bar], asks: Mapping[int, Bar]
) -> tuple[LeoCompleted15mBar, ...]:
    """Canonical 1m OHLC aggregation: only all-15 exact paired minutes qualify."""
    result: list[LeoCompleted15mBar] = []
    common = sorted(set(bids) & set(asks))
    buckets: dict[int, list[int]] = {}
    for timestamp in common:
        buckets.setdefault(timestamp - timestamp % (15 * _MINUTE_NS), []).append(
            timestamp
        )
    for start, timestamps in sorted(buckets.items()):
        expected = [start + index * _MINUTE_NS for index in range(15)]
        if timestamps != expected:
            continue
        paired = [(bids[item], asks[item]) for item in expected]
        if any(
            bid.ts_init != timestamp + _MINUTE_NS or ask.ts_init != bid.ts_init
            for timestamp, (bid, ask) in zip(expected, paired, strict=True)
        ):
            continue

        def aggregate(side: int) -> PriceBar:
            values = [pair[side] for pair in paired]
            return PriceBar(
                values[0].open.as_decimal(),
                max(item.high.as_decimal() for item in values),
                min(item.low.as_decimal() for item in values),
                values[-1].close.as_decimal(),
            )

        start_time = _datetime(start)
        end = start_time + _FIFTEEN
        result.append(
            LeoCompleted15mBar(
                _GBPUSD, start_time, end, end, aggregate(0), aggregate(1)
            )
        )
    return tuple(result)


def _validate_gbp_source(
    context: DevelopmentResearchContext, roots: Mapping[str, Path]
) -> str:
    context.require_range(
        DEVELOPMENT_START,
        DEVELOPMENT_END_EXCLUSIVE,
        partition=ResearchPartition.DEVELOPMENT,
    )
    status = next(item for item in context.artifacts if item.instrument_id == _GBPUSD)
    actual = _sha256_tree(roots[_GBPUSD] / "catalog")
    if actual != status.catalog_tree_sha256:
        raise TsMomentumEvaluationError("GBP/USD DEVELOPMENT catalog tree hash drifted")
    return actual


def _validate_manifest(
    manifest: Any,
    context: DevelopmentResearchContext,
    roots: Mapping[str, Path],
    cache_dir: Path,
) -> None:
    if not isinstance(manifest, dict) or manifest.get(
        "semantic_sha256"
    ) != _semantic_hash(
        {key: value for key, value in manifest.items() if key != "semantic_sha256"}
    ):
        raise TsMomentumEvaluationError("Leo cache manifest semantic hash drifted")
    required = {
        "schema",
        "schema_version",
        "source_instrument",
        "development_boundary",
        "source_readiness_sha256",
        "source_catalog_tree_sha256",
        "source_bar_types",
        "aggregation",
        "timezone_convention",
        "row_count",
        "first_timestamp_utc",
        "last_timestamp_utc",
        "output_file",
        "output_file_sha256",
        "frozen_strategy_sha256",
        "validation_accessed",
        "final_holdout_accessed",
        "semantic_sha256",
    }
    if (
        set(manifest) != required
        or manifest["schema"] != _SCHEMA
        or manifest["schema_version"] != _VERSION
        or manifest["source_instrument"] != _GBPUSD
        or manifest["frozen_strategy_sha256"] != LEO_GBPUSD_CONFIG_SHA256
    ):
        raise TsMomentumEvaluationError("Leo cache identity is incompatible")
    if (
        manifest["development_boundary"]
        != {
            "start_utc": _format(DEVELOPMENT_START),
            "end_exclusive_utc": _format(DEVELOPMENT_END_EXCLUSIVE),
        }
        or manifest["aggregation"]
        != {
            "interval_minutes": 15,
            "semantics": "exact contiguous completed 1m bid/ask OHLC; ts_init=end",
        }
        or manifest["validation_accessed"] is not False
        or manifest["final_holdout_accessed"] is not False
    ):
        raise TsMomentumEvaluationError(
            "Leo cache boundary or aggregation is incompatible"
        )
    if manifest[
        "source_readiness_sha256"
    ] != context.universe.readiness_sha256 or manifest[
        "source_catalog_tree_sha256"
    ] != _validate_gbp_source(context, roots):
        raise TsMomentumEvaluationError("Leo cache source provenance drifted")
    data_path = cache_dir / _DATA_FILE
    if manifest["output_file"] != _DATA_FILE or manifest[
        "output_file_sha256"
    ] != _sha256_file(data_path):
        raise TsMomentumEvaluationError("Leo cache data file hash drifted")


def _write_rows(path: Path, rows: Sequence[LeoCompleted15mBar]) -> None:
    columns: dict[str, list[Any]] = {
        "start_ns": [],
        "bid_open": [],
        "bid_high": [],
        "bid_low": [],
        "bid_close": [],
        "ask_open": [],
        "ask_high": [],
        "ask_low": [],
        "ask_close": [],
    }
    for row in rows:
        row.validate()
        columns["start_ns"].append(_ns(row.start_time_utc))
        for prefix, bar in (("bid", row.bid), ("ask", row.ask)):
            for name in ("open", "high", "low", "close"):
                columns[f"{prefix}_{name}"].append(str(getattr(bar, name)))
    pq.write_table(pa.table(columns), path, compression="zstd")


def _read_rows(path: Path) -> tuple[LeoCompleted15mBar, ...]:
    try:
        table = pq.read_table(path)
    except Exception as error:
        raise TsMomentumEvaluationError(
            f"Leo cache data is unreadable: {error}"
        ) from error
    names = (
        "start_ns",
        "bid_open",
        "bid_high",
        "bid_low",
        "bid_close",
        "ask_open",
        "ask_high",
        "ask_low",
        "ask_close",
    )
    if set(table.column_names) != set(names):
        raise TsMomentumEvaluationError("Leo cache columns are incompatible")
    data = table.to_pydict()
    result: list[LeoCompleted15mBar] = []
    for i in range(table.num_rows):
        start = _datetime(int(data["start_ns"][i]))
        end = start + _FIFTEEN

        def price(prefix: str) -> PriceBar:
            return PriceBar(
                *(
                    Decimal(str(data[f"{prefix}_{name}"][i]))
                    for name in ("open", "high", "low", "close")
                )
            )

        row = LeoCompleted15mBar(_GBPUSD, start, end, end, price("bid"), price("ask"))
        row.validate()
        result.append(row)
    return tuple(result)


def _semantic_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise TsMomentumEvaluationError(
            f"Leo cache file unavailable: {error}"
        ) from error


def _ns(value: datetime) -> int:
    return int(value.astimezone(UTC).timestamp() * 1_000_000_000)


def _datetime(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1_000_000_000, tz=UTC)


def _format(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
