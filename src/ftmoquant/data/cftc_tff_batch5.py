"""Acquire and normalize the data-only Batch 5 CFTC TFF lineage.

This module deliberately contains no alpha construction, returns, prices, or
performance code.  Every normalized row is stamped by the separately frozen
causal-availability amendment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from ftmoquant.research.alpha_lab.batch5_cftc_availability import (
    EXPECTED_AMENDMENT_SEMANTIC_SHA256,
    availability_for_report_date,
    verify_amendment,
)
from ftmoquant.research.alpha_lab.batch5_preregistration import (
    EXPECTED_PREREGISTRATION_SEMANTIC_SHA256,
)

LINEAGE_ID = "batch5-cftc-tff-futures-only-dealer-v1"
PREREG_SHA = EXPECTED_PREREGISTRATION_SEMANTIC_SHA256
AMENDMENT_SHA = EXPECTED_AMENDMENT_SEMANTIC_SHA256
YEARS = tuple(range(2018, 2024))
SOURCE_URL = "https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip"
DEVELOPMENT_START = date(2019, 3, 11)
DEVELOPMENT_END_INCLUSIVE = date(2023, 4, 11)
WARMUP_START = date(2018, 3, 1)

# Stable CFTC contract-market codes, not name matching (the CFTC renamed GBP
# and NZD labels in 2022 without changing their codes).
CONTRACT_CURRENCIES = {
    "090741": "CAD",
    "092741": "CHF",
    "096742": "GBP",
    "097741": "JPY",
    "099741": "EUR",
    "112741": "NZD",
    "232741": "AUD",
}

SPOT_ORIENTATION = {
    "AUD": ("AUD/USD", "SAME_AS_FUTURES_LOCAL_CURRENCY"),
    "CAD": ("USD/CAD", "INVERSE_OF_FUTURES_LOCAL_CURRENCY"),
    "CHF": ("USD/CHF", "INVERSE_OF_FUTURES_LOCAL_CURRENCY"),
    "EUR": ("EUR/USD", "SAME_AS_FUTURES_LOCAL_CURRENCY"),
    "GBP": ("GBP/USD", "SAME_AS_FUTURES_LOCAL_CURRENCY"),
    "JPY": ("USD/JPY", "INVERSE_OF_FUTURES_LOCAL_CURRENCY"),
    "NZD": ("NZD/USD", "SAME_AS_FUTURES_LOCAL_CURRENCY"),
}

NORMALIZED_FIELDS = (
    "report_date",
    "currency",
    "cftc_contract_market_code",
    "market_and_exchange_name",
    "dealer_long_all",
    "dealer_short_all",
    "dealer_net_all",
    "open_interest_all",
    "spot_pair",
    "spot_pair_orientation_metadata",
    "availability_status",
    "availability_timestamp",
    "publication_date",
    "publication_timestamp_utc",
    "first_known_vintage",
    "source_year",
    "source_sha256",
    "original_preregistration_semantic_sha256",
    "availability_amendment_semantic_sha256",
)


class CftcTffBatch5Error(RuntimeError):
    """Raised when official lineage or normalization cannot be proven."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, document: dict[str, Any]) -> None:
    payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise CftcTffBatch5Error(f"refusing to replace differing artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def acquire(output_root: Path) -> Path:
    """Download immutable official annual ZIPs and freeze content hashes."""

    amendment = verify_amendment()
    root = output_root.resolve()
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for year in YEARS:
        url = SOURCE_URL.format(year=year)
        target = raw / f"fut_fin_txt_{year}.zip"
        if not target.exists():
            with tempfile.NamedTemporaryFile(dir=raw, delete=False) as temporary:
                temporary_path = Path(temporary.name)
            try:
                request = urllib.request.Request(  # noqa: S310
                    url,
                    headers={
                        "User-Agent": (
                            "FTMOQuant research data acquisition/1.0 "
                            "(official CFTC public files)"
                        )
                    },
                )
                with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
                    with temporary_path.open("wb") as handle:
                        shutil.copyfileobj(response, handle)
                if not zipfile.is_zipfile(temporary_path):
                    raise CftcTffBatch5Error(f"official response is not a ZIP: {url}")
                temporary_path.replace(target)
            finally:
                temporary_path.unlink(missing_ok=True)
        records.append(
            {
                "year": year,
                "url": url,
                "path": str(target.relative_to(root)),
                "sha256": _sha256(target),
                "bytes": target.stat().st_size,
                "download_timestamp_utc": datetime.fromtimestamp(
                    target.stat().st_mtime, tz=UTC
                ).isoformat(),
            }
        )
    manifest = root / "cftc_tff_raw_manifest.json"
    _write_json(
        manifest,
        {
            "lineage_id": LINEAGE_ID,
            "source": "CFTC Historical Compressed TFF Futures Only",
            "files": records,
            "original_preregistration_semantic_sha256": PREREG_SHA,
            "availability_amendment_semantic_sha256": amendment[
                "amendment_semantic_sha256"
            ],
        },
    )
    return manifest


def _field(row: dict[str, str], *names: str) -> str:
    normalized = {
        key.strip().lstrip("\ufeff"): value.strip() for key, value in row.items()
    }
    for name in names:
        if name in normalized:
            return normalized[name]
    raise CftcTffBatch5Error(f"required official field absent: {names[0]}")


def _source_rows(archive: Path) -> list[dict[str, str]]:
    with zipfile.ZipFile(archive) as bundle:
        names = [name for name in bundle.namelist() if name.lower().endswith(".txt")]
        if len(names) != 1:
            raise CftcTffBatch5Error(f"expected one text member in {archive}")
        with bundle.open(names[0]) as raw_handle:
            text = (line.decode("utf-8-sig") for line in raw_handle)
            return list(csv.DictReader(text))


def normalize(output_root: Path) -> tuple[Path, Path]:
    """Normalize the seven frozen currency contracts and attach availability."""

    amendment = verify_amendment()
    root = output_root.resolve()
    manifest_path = root / "cftc_tff_raw_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CftcTffBatch5Error("raw manifest missing or invalid") from error
    if manifest.get("lineage_id") != LINEAGE_ID:
        raise CftcTffBatch5Error("raw lineage identity mismatch")
    if (
        manifest.get("availability_amendment_semantic_sha256")
        != EXPECTED_AMENDMENT_SEMANTIC_SHA256
    ):
        raise CftcTffBatch5Error("raw manifest amendment identity mismatch")

    normalized: list[dict[str, Any]] = []
    seen: set[tuple[date, str]] = set()
    for source in manifest["files"]:
        archive = root / source["path"]
        if _sha256(archive) != source["sha256"]:
            raise CftcTffBatch5Error(f"raw source hash mismatch: {archive}")
        for row in _source_rows(archive):
            code = _field(row, "CFTC_Contract_Market_Code")
            currency = CONTRACT_CURRENCIES.get(code)
            if currency is None:
                continue
            report_date = date.fromisoformat(
                _field(row, "Report_Date_as_YYYY-MM-DD", "Report_Date_as_MM_DD_YYYY")
            )
            if report_date < WARMUP_START or report_date > DEVELOPMENT_END_INCLUSIVE:
                continue
            key = (report_date, currency)
            if key in seen:
                raise CftcTffBatch5Error(f"duplicate official contract/date: {key}")
            seen.add(key)
            status, timestamp = availability_for_report_date(
                report_date, amendment=amendment
            )
            normalized.append(
                {
                    "report_date": report_date.isoformat(),
                    "currency": currency,
                    "cftc_contract_market_code": code,
                    "market_and_exchange_name": _field(
                        row, "Market_and_Exchange_Names"
                    ),
                    "dealer_long_all": int(_field(row, "Dealer_Positions_Long_All")),
                    "dealer_short_all": int(_field(row, "Dealer_Positions_Short_All")),
                    "dealer_net_all": int(_field(row, "Dealer_Positions_Long_All"))
                    - int(_field(row, "Dealer_Positions_Short_All")),
                    "open_interest_all": int(_field(row, "Open_Interest_All")),
                    "spot_pair": SPOT_ORIENTATION[currency][0],
                    "spot_pair_orientation_metadata": SPOT_ORIENTATION[currency][1],
                    "availability_status": status,
                    "availability_timestamp": timestamp.isoformat()
                    if timestamp
                    else "",
                    "publication_date": timestamp.date().isoformat()
                    if timestamp
                    else "",
                    "publication_timestamp_utc": timestamp.astimezone(UTC).isoformat()
                    if timestamp
                    else "",
                    "first_known_vintage": (
                        f"CFTC_ANNUAL_FILE_SHA256:{source['sha256']}"
                    ),
                    "source_year": source["year"],
                    "source_sha256": source["sha256"],
                    "original_preregistration_semantic_sha256": PREREG_SHA,
                    "availability_amendment_semantic_sha256": AMENDMENT_SHA,
                }
            )
    normalized.sort(key=lambda row: (row["report_date"], row["currency"]))
    target = root / "normalized" / "cftc_tff_futures_only_dealer.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=target.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=NORMALIZED_FIELDS)
        writer.writeheader()
        writer.writerows(normalized)
    temporary.replace(target)

    development = [
        row for row in normalized if row["report_date"] >= DEVELOPMENT_START.isoformat()
    ]
    dates_by_currency = Counter(row["currency"] for row in development)
    statuses_by_date: dict[str, str] = {}
    for row in development:
        prior = statuses_by_date.setdefault(
            row["report_date"], row["availability_status"]
        )
        if prior != row["availability_status"]:
            raise CftcTffBatch5Error("availability differs within a report date")
    readiness = root / "readiness" / "cftc_tff_readiness.json"
    _write_json(
        readiness,
        {
            "lineage_id": LINEAGE_ID,
            "research_ready": (
                set(dates_by_currency) == set(CONTRACT_CURRENCIES.values())
                and len(set(dates_by_currency.values())) == 1
                and Counter(statuses_by_date.values()) == amendment["status_counts"]
            ),
            "normalized_path": str(target.relative_to(root)),
            "normalized_sha256": _sha256(target),
            "normalized_row_count": len(normalized),
            "development_row_count": len(development),
            "development_week_count": len(statuses_by_date),
            "rows_per_currency": dict(sorted(dates_by_currency.items())),
            "availability_status_week_counts": dict(
                sorted(Counter(statuses_by_date.values()).items())
            ),
            "unresolved_report_dates": sorted(
                report_date
                for report_date, status in statuses_by_date.items()
                if status == "UNRESOLVED"
            ),
            "warmup_start": WARMUP_START.isoformat(),
            "development_start": DEVELOPMENT_START.isoformat(),
            "development_end_inclusive": DEVELOPMENT_END_INCLUSIVE.isoformat(),
            "original_preregistration_semantic_sha256": PREREG_SHA,
            "availability_amendment_semantic_sha256": AMENDMENT_SHA,
            "data_firewall": {
                "fx_prices_or_returns_read": False,
                "signals_or_backtests_run": False,
                "validation_or_holdout_read": False,
            },
        },
    )
    return target, readiness


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare Batch 5 CFTC TFF data lineage"
    )
    parser.add_argument("command", choices=("acquire", "normalize", "all"))
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command in {"acquire", "all"}:
        print(acquire(args.output_root))
    if args.command in {"normalize", "all"}:
        for path in normalize(args.output_root):
            print(path)


if __name__ == "__main__":
    main()
