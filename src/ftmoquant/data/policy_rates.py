"""Immutable, provenance-checked ECB/Fed policy-rate loader.

Freezes two short-rate series (ECB Deposit Facility Rate, US Effective Fed
Funds Rate) from a committed, hash-verified raw source and derives a single
causal daily differential series under a conservative one-business-day
information lag. This module performs no strategy logic, sizing, execution,
or backtest -- it only produces an immutable, causally ordered rate history.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

POLICY_RATES_DIR = Path("config/data/policy_rates")
PROVENANCE_PATH = POLICY_RATES_DIR / "provenance.json"

ECBDFR_RAW_SHA256 = (
    "a9993333c5648a3ee0cfbc6ac4cc925cb6270d9ccecfa32d0db62ea808d57fe4"
)
EFFR_RAW_SHA256 = "1c53e1aa230fbe2d39275775206b64ad7e56d36c8cd63e572f492278a1c79a73"


class PolicyRateProvenanceError(ValueError):
    """Raised when frozen policy-rate source data drifts or is inconsistent."""


@dataclass(frozen=True, slots=True)
class DatedRate:
    observation_date: date
    value_percent: Decimal


@dataclass(frozen=True, slots=True)
class CausalDifferentialObservation:
    """One calendar day's causally-lagged ECBDFR-minus-EFFR differential.

    ``as_of_date`` is the calendar day this observation applies to;
    ``ecbdfr_observation_date``/``effr_observation_date`` are the dated rate
    observations actually used (each strictly at or before the prior eligible
    business day relative to ``as_of_date``, per the frozen one-business-day
    information lag).
    """

    as_of_date: date
    differential_percent: Decimal
    ecbdfr_observation_date: date
    effr_observation_date: date


@dataclass(frozen=True, slots=True)
class PolicyRateHistory:
    ecbdfr: tuple[DatedRate, ...]
    effr: tuple[DatedRate, ...]
    canonical_data_sha256: str


def load_policy_rate_history(
    directory: Path = POLICY_RATES_DIR,
) -> PolicyRateHistory:
    """Load and provenance-check the frozen ECBDFR/EFFR raw source files."""

    provenance = _json_object(directory / "provenance.json")
    series = {item["series_id"]: item for item in provenance["series"]}
    if set(series) != {"ECBDFR", "EFFR"}:
        raise PolicyRateProvenanceError("provenance must declare ECBDFR and EFFR only")

    ecbdfr_path = directory / cast(str, series["ECBDFR"]["raw_file"])
    effr_path = directory / cast(str, series["EFFR"]["raw_file"])
    _verify_sha256(ecbdfr_path, cast(str, series["ECBDFR"]["raw_file_sha256"]))
    _verify_sha256(effr_path, cast(str, series["EFFR"]["raw_file_sha256"]))
    if series["ECBDFR"]["raw_file_sha256"] != ECBDFR_RAW_SHA256:
        raise PolicyRateProvenanceError(
            "ECBDFR raw file SHA-256 drifted from frozen value"
        )
    if series["EFFR"]["raw_file_sha256"] != EFFR_RAW_SHA256:
        raise PolicyRateProvenanceError(
            "EFFR raw file SHA-256 drifted from frozen value"
        )

    ecbdfr = _parse_fred_csv(ecbdfr_path, "ECBDFR")
    effr = _parse_fred_csv(effr_path, "EFFR")
    canonical = _canonical_data_sha256(ecbdfr, effr)
    return PolicyRateHistory(ecbdfr=ecbdfr, effr=effr, canonical_data_sha256=canonical)


def causal_differential_series(
    history: PolicyRateHistory,
    *,
    start_inclusive: date,
    end_inclusive: date,
) -> tuple[CausalDifferentialObservation, ...]:
    """Build the ECBDFR-minus-EFFR differential under a strict 1-business-day lag.

    For calendar day ``D``, only rate observations dated on or before the
    prior eligible business day (Monday-Friday) are used, with the most
    recent causally-available observation forward-filled through weekends
    and gaps. No observation dated after that lagged cutoff is ever read --
    there is no backfill from a later-dated observation.
    """

    if start_inclusive > end_inclusive:
        raise PolicyRateProvenanceError(
            "start_inclusive must not be after end_inclusive"
        )
    ecbdfr_by_date = {
        item.observation_date: item.value_percent for item in history.ecbdfr
    }
    effr_by_date = {
        item.observation_date: item.value_percent for item in history.effr
    }
    ecbdfr_dates = sorted(ecbdfr_by_date)
    effr_dates = sorted(effr_by_date)

    result: list[CausalDifferentialObservation] = []
    current = start_inclusive
    while current <= end_inclusive:
        lag_cutoff = _prior_business_day(current)
        ecbdfr_date = _latest_on_or_before(ecbdfr_dates, lag_cutoff)
        effr_date = _latest_on_or_before(effr_dates, lag_cutoff)
        if ecbdfr_date is not None and effr_date is not None:
            differential = ecbdfr_by_date[ecbdfr_date] - effr_by_date[effr_date]
            result.append(
                CausalDifferentialObservation(
                    as_of_date=current,
                    differential_percent=differential,
                    ecbdfr_observation_date=ecbdfr_date,
                    effr_observation_date=effr_date,
                )
            )
        current += timedelta(days=1)
    return tuple(result)


def causal_differential_series_pre_development_only(
    history: PolicyRateHistory,
    *,
    start_inclusive: date,
    end_inclusive: date,
) -> tuple[CausalDifferentialObservation, ...]:
    """Post-DEVELOPMENT covariate seal around :func:`causal_differential_series`.

    This is the only sanctioned entry point for any pre-DEVELOPMENT-task
    code path in this repository that wants to summarize, inspect, count
    regimes in, or inspect signs of the causal ECBDFR-minus-EFFR
    differential. It mechanically raises :class:`PolicyRateProvenanceError`
    if ``end_inclusive`` reaches or crosses the frozen VALIDATION_START
    boundary (imported from :mod:`ftmoquant.research.stage_g`, never
    hardcoded here), so a pre-DEVELOPMENT task literally cannot construct a
    differential series that touches validation-or-later dates through this
    path.

    Deliberately does **not** guard :func:`load_policy_rate_history` or raw
    hash/schema/provenance verification -- those remain fully permitted
    even though the committed raw CSVs extend well past DEVELOPMENT, since
    mechanical provenance validation of already-committed source files is
    not the same thing as summarizing or inspecting post-DEVELOPMENT
    differential values.
    """

    from ftmoquant.research.stage_g import VALIDATION_START

    if end_inclusive >= VALIDATION_START.date():
        raise PolicyRateProvenanceError(
            "causal_differential_series_pre_development_only refuses a "
            f"range whose end_inclusive ({end_inclusive.isoformat()}) "
            f"reaches or crosses the frozen VALIDATION_START "
            f"({VALIDATION_START.date().isoformat()}); pre-DEVELOPMENT "
            "code paths must never summarize, inspect, count regimes in, "
            "or inspect signs of validation-or-later policy-rate "
            "differential observations"
        )
    return causal_differential_series(
        history, start_inclusive=start_inclusive, end_inclusive=end_inclusive
    )


def _prior_business_day(value: date) -> date:
    """The business day (Mon-Fri) strictly before ``value``."""

    cursor = value - timedelta(days=1)
    while cursor.weekday() >= 5:  # Saturday=5, Sunday=6
        cursor -= timedelta(days=1)
    return cursor


def _latest_on_or_before(sorted_dates: list[date], cutoff: date) -> date | None:
    import bisect

    index = bisect.bisect_right(sorted_dates, cutoff)
    if index == 0:
        return None
    return sorted_dates[index - 1]


def _parse_fred_csv(path: Path, series_id: str) -> tuple[DatedRate, ...]:
    rows: list[DatedRate] = []
    previous: date | None = None
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != ["observation_date", series_id]:
            raise PolicyRateProvenanceError(
                f"{path} has an unexpected header: {header}"
            )
        for row in reader:
            if len(row) != 2:
                raise PolicyRateProvenanceError(f"{path} has a malformed row: {row}")
            raw_date, raw_value = row
            try:
                observation_date = datetime.strptime(raw_date, "%Y-%m-%d").date()
            except ValueError as error:
                raise PolicyRateProvenanceError(
                    f"{path} has an invalid date: {raw_date}"
                ) from error
            if raw_value in ("", "."):
                # A dated row with no published value (typically a US federal
                # holiday for EFFR). This date simply has no observation; the
                # causal lookup forward-fills from the latest earlier date
                # that does have one. Not an error -- explicit, intentional
                # missing-value handling, not a silent drop.
                if previous is not None and observation_date <= previous:
                    raise PolicyRateProvenanceError(
                        f"{path} dates are not strictly increasing"
                    )
                previous = observation_date
                continue
            try:
                value = Decimal(raw_value)
            except Exception as error:  # noqa: BLE001 - surfaces as a provenance error
                raise PolicyRateProvenanceError(
                    f"{path} has a non-numeric value at {raw_date}: {raw_value}"
                ) from error
            if not value.is_finite():
                raise PolicyRateProvenanceError(
                    f"{path} has a non-finite value at {raw_date}"
                )
            if previous is not None and observation_date <= previous:
                raise PolicyRateProvenanceError(
                    f"{path} dates are not strictly increasing"
                )
            previous = observation_date
            rows.append(DatedRate(observation_date, value))
    if not rows:
        raise PolicyRateProvenanceError(f"{path} contains no observations")
    return tuple(rows)


def _verify_sha256(path: Path, expected: str) -> None:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise PolicyRateProvenanceError(
            f"{path} SHA-256 mismatch: expected {expected}, got {actual}"
        )


def _canonical_data_sha256(
    ecbdfr: tuple[DatedRate, ...], effr: tuple[DatedRate, ...]
) -> str:
    payload = {
        "ecbdfr": [
            [item.observation_date.isoformat(), str(item.value_percent)]
            for item in ecbdfr
        ],
        "effr": [
            [item.observation_date.isoformat(), str(item.value_percent)]
            for item in effr
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PolicyRateProvenanceError(f"could not load {path}: {error}") from error
    if not isinstance(value, dict):
        raise PolicyRateProvenanceError(f"{path} must be a JSON object")
    return cast(dict[str, Any], value)
