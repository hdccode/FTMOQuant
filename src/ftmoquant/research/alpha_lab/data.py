"""Adapter from frozen DEVELOPMENT FX catalogs to aligned pandas matrices.

This module is a thin translation layer: canonical Stage G DEVELOPMENT data
in ``nautilus_trader`` ``ParquetDataCatalog`` form -> pandas matrices indexed
by UTC timestamp with one column per instrument. It performs no validation
or holdout access, no interpolation, and no future backfill.

Discovery of the DEVELOPMENT instrument universe is read from the frozen
``ftmoquant_universe_readiness.json`` document via
:func:`ftmoquant.research.stage_g.load_frozen_universe`; nothing here
hardcodes an assumed instrument list.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd  # type: ignore[import-untyped]
from nautilus_trader.persistence import ParquetDataCatalog

from ftmoquant.backtest.execution_harness import _sha256_tree
from ftmoquant.data.universe_readiness import SPLIT_VIEW_FILENAME
from ftmoquant.research.stage_g import (
    DEVELOPMENT_END_EXCLUSIVE,
    DEVELOPMENT_START,
    DevelopmentResearchContext,
    FrozenUniverse,
    ResearchPartition,
    StageGValidationError,
    load_frozen_universe,
    open_development_context,
)

Timeframe = Literal["H1", "H4"]

_TIMEFRAME_BAR_SPEC: dict[str, str] = {
    "H1": "1-HOUR",
    "H4": "4-HOUR",
}

#: Explicit alignment policy applied by :func:`load_alpha_lab_dataset`.
#: Only timestamps for which every discovered instrument has a native,
#: already-paired BID and ASK bar are kept (inner join). No forward-fill,
#: no interpolation, and no synthetic bar is ever introduced.
ALIGNMENT_POLICY = (
    "inner_join_native_bars_no_fill: matrices are restricted to UTC "
    "timestamps at which every instrument has a native BID and ASK bar in "
    "the frozen DEVELOPMENT catalog; timestamps missing for any instrument "
    "are dropped from all matrices rather than filled or interpolated."
)


class AlphaLabDataError(ValueError):
    """Raised when DEVELOPMENT data cannot be loaded or aligned safely."""


@dataclass(frozen=True, slots=True)
class AlphaLabDataset:
    """Aligned multi-instrument DEVELOPMENT matrices for one timeframe."""

    timeframe: Timeframe
    instrument_ids: tuple[str, ...]
    start_utc: datetime
    end_exclusive_utc: datetime
    alignment_policy: str
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    bid: pd.DataFrame
    ask: pd.DataFrame
    spread: pd.DataFrame


def discover_development_instrument_ids(
    readiness_path: Path,
    plan_path: Path,
) -> tuple[str, ...]:
    """Return the research-ready DEVELOPMENT instrument ids, discovered
    (never hardcoded) from the frozen universe readiness document."""

    universe = _load_frozen_universe_document(readiness_path, plan_path)
    return universe.ordered_instrument_ids


def resolve_development_roots(
    instrument_ids: tuple[str, ...],
    development_root_dir: Path,
) -> dict[str, Path]:
    """Map each ``instrument_id`` to its DEVELOPMENT split root directory.

    Roots are resolved by the existing on-disk convention
    ``<development_root_dir>/<DATASET_SYMBOL>`` (e.g. ``EURUSD``), matching
    the layout already produced by ``materialize_instrument_split_views``.
    """

    roots: dict[str, Path] = {}
    for instrument_id in instrument_ids:
        symbol = instrument_id.split("/")[0] + instrument_id.split("/")[1].split(".")[0]
        root = development_root_dir / symbol
        lowered = str(root).lower()
        if "validation" in lowered or "holdout" in lowered:
            raise AlphaLabDataError(
                f"development root path is forbidden: {root}"
            )
        if not (root / SPLIT_VIEW_FILENAME).is_file():
            raise AlphaLabDataError(
                f"no DEVELOPMENT split view found for {instrument_id} at {root}"
            )
        roots[instrument_id] = root
    return roots


def load_alpha_lab_dataset(
    *,
    readiness_path: Path,
    plan_path: Path,
    development_root_dir: Path,
    timeframe: Timeframe,
) -> AlphaLabDataset:
    """Load aligned DEVELOPMENT OHLC/BID/ASK/spread matrices for every
    discovered instrument at ``timeframe``.

    Raises :class:`AlphaLabDataError` on any attempt to read validation or
    holdout data, on catalog drift, or on an empty result.
    """

    if timeframe not in _TIMEFRAME_BAR_SPEC:
        raise AlphaLabDataError(f"unsupported timeframe: {timeframe}")

    instrument_ids = discover_development_instrument_ids(readiness_path, plan_path)
    if not instrument_ids:
        raise AlphaLabDataError("no research-ready DEVELOPMENT instruments discovered")
    development_roots = resolve_development_roots(instrument_ids, development_root_dir)

    try:
        context = open_development_context(readiness_path, development_roots, plan_path)
    except StageGValidationError as error:
        raise AlphaLabDataError(f"DEVELOPMENT context rejected: {error}") from error
    context.require_range(
        DEVELOPMENT_START,
        DEVELOPMENT_END_EXCLUSIVE,
        partition=ResearchPartition.DEVELOPMENT,
    )
    _validate_catalog_trees(context, development_roots)

    per_instrument = {
        instrument_id: _load_instrument_frame(
            instrument_id=instrument_id,
            root=development_roots[instrument_id],
            timeframe=timeframe,
        )
        for instrument_id in instrument_ids
    }

    _validate_catalog_trees(context, development_roots)

    return assemble_aligned_dataset(
        per_instrument=per_instrument,
        instrument_ids=instrument_ids,
        timeframe=timeframe,
    )


def assemble_aligned_dataset(
    *,
    per_instrument: Mapping[str, pd.DataFrame],
    instrument_ids: tuple[str, ...],
    timeframe: Timeframe,
) -> AlphaLabDataset:
    """Pure assembly step: per-instrument OHLC/bid/ask frames -> one aligned
    :class:`AlphaLabDataset`, applying :data:`ALIGNMENT_POLICY`.

    Each input frame must have a UTC ``DatetimeIndex`` and
    ``open``/``high``/``low``/``close``/``bid``/``ask`` columns. Carries no
    catalog or readiness dependency, so it is directly unit-testable with
    synthetic frames.
    """

    ordered_ids = tuple(sorted(instrument_ids))
    aligned_index = _aligned_index(per_instrument, ordered_ids)
    if aligned_index.empty:
        raise AlphaLabDataError(
            "no UTC timestamp has a native bar for every discovered instrument"
        )

    fields: dict[str, pd.DataFrame] = {}
    for field in ("open", "high", "low", "close", "bid", "ask"):
        fields[field] = pd.DataFrame(
            {
                instrument_id: per_instrument[instrument_id][field].reindex(
                    aligned_index
                )
                for instrument_id in ordered_ids
            },
            index=aligned_index,
        )
    fields["spread"] = fields["ask"] - fields["bid"]

    return AlphaLabDataset(
        timeframe=timeframe,
        instrument_ids=ordered_ids,
        start_utc=DEVELOPMENT_START,
        end_exclusive_utc=DEVELOPMENT_END_EXCLUSIVE,
        alignment_policy=ALIGNMENT_POLICY,
        open=fields["open"],
        high=fields["high"],
        low=fields["low"],
        close=fields["close"],
        bid=fields["bid"],
        ask=fields["ask"],
        spread=fields["spread"],
    )


def _aligned_index(
    per_instrument: Mapping[str, pd.DataFrame], ordered_ids: tuple[str, ...]
) -> pd.DatetimeIndex:
    common: set[pd.Timestamp] | None = None
    for instrument_id in ordered_ids:
        stamps = set(per_instrument[instrument_id].index)
        common = stamps if common is None else common & stamps
    return pd.DatetimeIndex(sorted(common or set()), tz=UTC)


def _load_instrument_frame(
    *, instrument_id: str, root: Path, timeframe: Timeframe
) -> pd.DataFrame:
    bar_spec = _TIMEFRAME_BAR_SPEC[timeframe]
    catalog = ParquetDataCatalog(str(root / "catalog"))
    start_ns = _ns(DEVELOPMENT_START)
    end_ns = _ns(DEVELOPMENT_END_EXCLUSIVE)

    by_side: dict[str, dict[int, tuple[float, float, float, float]]] = {}
    for side in ("BID", "ASK"):
        bar_type = f"{instrument_id}-{bar_spec}-{side}-INTERNAL"
        queried = catalog.query_bars([bar_type], start=start_ns, end=end_ns - 1)
        selected = {
            bar.ts_event: (
                float(bar.open.as_decimal()),
                float(bar.high.as_decimal()),
                float(bar.low.as_decimal()),
                float(bar.close.as_decimal()),
            )
            for bar in queried
            if start_ns <= bar.ts_event < end_ns and str(bar.bar_type) == bar_type
        }
        if not selected:
            raise AlphaLabDataError(f"DEVELOPMENT has no {bar_type} data")
        by_side[side] = selected

    shared_ts = sorted(set(by_side["BID"]) & set(by_side["ASK"]))
    if not shared_ts:
        raise AlphaLabDataError(
            f"DEVELOPMENT BID/ASK bars never overlap for {instrument_id} at {timeframe}"
        )

    rows = []
    for ts in shared_ts:
        bo, bh, bl, bc = by_side["BID"][ts]
        ao, ah, al, ac = by_side["ASK"][ts]
        rows.append(
            {
                "timestamp": pd.Timestamp(ts, unit="ns", tz=UTC),
                "open": (bo + ao) / 2.0,
                "high": (bh + ah) / 2.0,
                "low": (bl + al) / 2.0,
                "close": (bc + ac) / 2.0,
                "bid": bc,
                "ask": ac,
            }
        )
    frame = pd.DataFrame(rows).set_index("timestamp").sort_index()
    return frame


def _validate_catalog_trees(
    context: DevelopmentResearchContext, development_roots: Mapping[str, Path]
) -> None:
    statuses = {item.instrument_id: item for item in context.artifacts}
    for instrument_id, root in development_roots.items():
        actual = _sha256_tree(root / "catalog")
        expected = statuses[instrument_id].catalog_tree_sha256
        if actual != expected:
            raise AlphaLabDataError(
                f"DEVELOPMENT catalog tree hash drifted: {instrument_id}"
            )


def _ns(value: datetime) -> int:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise AlphaLabDataError("timestamp must be timezone-aware UTC")
    return int(value.timestamp() * 1_000_000_000)


def _load_frozen_universe_document(
    readiness_path: Path, plan_path: Path
) -> FrozenUniverse:
    try:
        return load_frozen_universe(readiness_path, plan_path)
    except StageGValidationError as error:
        raise AlphaLabDataError(f"universe readiness rejected: {error}") from error
