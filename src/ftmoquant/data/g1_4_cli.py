"""Generic G1.4A command entrypoints; legacy EURUSD commands remain unchanged."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from ftmoquant.data.derived_bars import derive_instrument_bars
from ftmoquant.data.hf_dukascopy import ingest_hf_instrument
from ftmoquant.data.session_coverage import run_instrument_session_coverage
from ftmoquant.data.session_reconciliation import run_instrument_reconciliation
from ftmoquant.data.universe_plan import load_research_universe_plan
from ftmoquant.data.universe_readiness import (
    freeze_instrument_readiness,
    freeze_universe_readiness,
    materialize_instrument_split_views,
)


def ingest_main(argv: Sequence[str] | None = None) -> int:
    """Run holdout-safe HF ingestion for a range not requiring a direct tail."""

    parser = _base("Ingest one G1.4 instrument from pinned Hugging Face shards")
    parser.add_argument("--start", type=_utc)
    parser.add_argument("--end-exclusive", type=_utc)
    parser.add_argument("--data-use-rights-evidence-sha256", required=True)
    parser.add_argument("--direct-cache-root", type=Path)
    args = parser.parse_args(argv)
    result = ingest_hf_instrument(
        cast(Path, args.plan),
        cast(str, args.instrument_id),
        cast(Path, args.output_root),
        requested_start=cast(datetime | None, args.start),
        requested_end_exclusive=cast(datetime | None, args.end_exclusive),
        data_use_rights_evidence_sha256=cast(str, args.data_use_rights_evidence_sha256),
        direct_cache_root=cast(Path | None, args.direct_cache_root),
    )
    print(result.manifest_path)
    return 0


def derive_main(argv: Sequence[str] | None = None) -> int:
    parser = _base("Derive one G1.4 instrument's 1H/4H bars")
    args = parser.parse_args(argv)
    plan = load_research_universe_plan(cast(Path, args.plan))
    result = derive_instrument_bars(
        cast(Path, args.output_root), plan.instrument(cast(str, args.instrument_id))
    )
    print(result.manifest_path)
    return 0


def coverage_main(argv: Sequence[str] | None = None) -> int:
    parser = _base("Run one G1.4 instrument's session coverage QA")
    args = parser.parse_args(argv)
    plan = load_research_universe_plan(cast(Path, args.plan))
    result = run_instrument_session_coverage(
        cast(Path, args.output_root), plan.instrument(cast(str, args.instrument_id))
    )
    print(result.manifest_path)
    return 0


def reconcile_main(argv: Sequence[str] | None = None) -> int:
    parser = _base("Reconcile one G1.4 instrument against direct Dukascopy BI5")
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--offline-evidence", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    result = run_instrument_reconciliation(
        cast(Path, args.plan),
        cast(str, args.instrument_id),
        cast(Path, args.output_root),
        cast(Path, args.cache_root),
        offline_evidence_path=cast(Path | None, args.offline_evidence),
        workers=cast(int, args.workers),
    )
    print(result.manifest_path)
    return 0


def split_main(argv: Sequence[str] | None = None) -> int:
    parser = _base("Materialize isolated G1.4 development/validation catalogs")
    parser.add_argument("--split-output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    results = materialize_instrument_split_views(
        cast(Path, args.plan),
        cast(str, args.instrument_id),
        cast(Path, args.output_root),
        cast(Path, args.split_output_root),
    )
    for result in results:
        print(result.manifest_path)
    return 0


def freeze_instrument_main(argv: Sequence[str] | None = None) -> int:
    parser = _base("Freeze one G1.4 instrument's readiness")
    parser.add_argument("--split-root", required=True, type=Path)
    args = parser.parse_args(argv)
    result = freeze_instrument_readiness(
        cast(Path, args.plan),
        cast(str, args.instrument_id),
        cast(Path, args.output_root),
        cast(Path, args.split_root),
    )
    print(result.readiness_sha256)
    return 0


def freeze_universe_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze exact G1.4 universe readiness")
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--instrument-root",
        required=True,
        action="append",
        type=_instrument_root,
        metavar="INSTRUMENT_ID=PATH",
    )
    args = parser.parse_args(argv)
    roots = dict(cast(list[tuple[str, Path]], args.instrument_root))
    result = freeze_universe_readiness(
        cast(Path, args.plan), roots, cast(Path, args.output_root)
    )
    print(result.manifest_path)
    return 0


def _base(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--instrument-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def _utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an ISO UTC timestamp") from error
    if not value.endswith("Z") or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise argparse.ArgumentTypeError("must be an ISO UTC timestamp")
    return parsed.astimezone(UTC)


def _instrument_root(value: str) -> tuple[str, Path]:
    instrument_id, separator, path = value.partition("=")
    if not separator or not instrument_id or not path:
        raise argparse.ArgumentTypeError("must use INSTRUMENT_ID=PATH")
    return instrument_id, Path(path)
