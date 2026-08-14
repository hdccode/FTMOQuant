"""Strict, deterministic loader for the frozen EUR/USD research-data plan."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import yaml

DATASET_ID = "eurusd_research_v1"
HF_REPO_ID = "mito0o852/dukascopy-ticks"
INSTRUMENT_ID = "EUR/USD.DUKASCOPY"
BOUNDARY_RULE = "cumulative_fraction_floor_whole_utc_days"
HF_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")


class ResearchPlanValidationError(ValueError):
    """Raised when the frozen dataset plan is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class ResearchSplit:
    """One inclusive whole-UTC-day dataset split."""

    start: date
    end: date
    days: int


@dataclass(frozen=True, slots=True)
class ResearchDataPlan:
    """Validated immutable plan used by the G1 Hugging Face importer."""

    dataset_id: str
    market_data_origin: str
    distribution_source: str
    huggingface_repo: str
    huggingface_revision: str
    instrument: str
    full_start: date
    full_end: date
    boundary_rule: str
    development: ResearchSplit
    validation: ResearchSplit
    final_holdout: ResearchSplit
    g1_allowed_end_exclusive: datetime
    semantic_sha256: str


def load_research_data_plan(path: Path) -> ResearchDataPlan:
    """Load the exact plan schema and compute its canonical semantic hash."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ResearchPlanValidationError(
            f"could not load dataset plan: {error}"
        ) from error
    if not isinstance(raw, dict):
        raise ResearchPlanValidationError("dataset plan must be a mapping")
    document = cast(dict[str, Any], raw)
    _require_keys(
        document,
        {
            "dataset_id",
            "market_data",
            "full_universe",
            "split",
            "g1_allowed_end_exclusive",
        },
        "dataset plan",
    )
    market = _mapping(document["market_data"], "market_data")
    _require_keys(
        market,
        {
            "market_data_origin",
            "distribution_source",
            "huggingface_repo",
            "huggingface_revision",
            "instrument",
        },
        "market_data",
    )
    universe = _mapping(document["full_universe"], "full_universe")
    _require_keys(universe, {"start_utc_date", "end_utc_date"}, "full_universe")
    split = _mapping(document["split"], "split")
    _require_keys(
        split,
        {"boundary_rule", "development", "validation", "final_holdout"},
        "split",
    )

    development = _load_split(split["development"], "development")
    validation = _load_split(split["validation"], "validation")
    holdout = _load_split(split["final_holdout"], "final_holdout")
    full_start = _date(universe["start_utc_date"], "full_universe.start_utc_date")
    full_end = _date(universe["end_utc_date"], "full_universe.end_utc_date")
    cutoff = _utc_datetime(
        document["g1_allowed_end_exclusive"], "g1_allowed_end_exclusive"
    )
    revision = _string(market["huggingface_revision"], "huggingface_revision")
    if HF_REVISION_PATTERN.fullmatch(revision) is None:
        raise ResearchPlanValidationError(
            "huggingface_revision must be an exact 40-character lowercase SHA"
        )
    expected_literals = {
        "dataset_id": (document["dataset_id"], DATASET_ID),
        "market_data_origin": (market["market_data_origin"], "Dukascopy"),
        "distribution_source": (market["distribution_source"], "Hugging Face"),
        "huggingface_repo": (market["huggingface_repo"], HF_REPO_ID),
        "instrument": (market["instrument"], INSTRUMENT_ID),
        "boundary_rule": (split["boundary_rule"], BOUNDARY_RULE),
    }
    for field, (actual, expected) in expected_literals.items():
        if actual != expected:
            raise ResearchPlanValidationError(f"dataset plan has invalid {field}")

    if (full_start, full_end) != (date(2019, 3, 11), date(2025, 12, 31)):
        raise ResearchPlanValidationError(
            "dataset plan has invalid full-universe dates"
        )
    expected_splits = (
        (development, date(2019, 3, 11), date(2023, 4, 10), 1492),
        (validation, date(2023, 4, 11), date(2024, 8, 20), 498),
        (holdout, date(2024, 8, 21), date(2025, 12, 31), 498),
    )
    for item, expected_start, expected_end, expected_days in expected_splits:
        if (item.start, item.end, item.days) != (
            expected_start,
            expected_end,
            expected_days,
        ):
            raise ResearchPlanValidationError("dataset plan split is not frozen")
        if item.days != (item.end - item.start).days + 1:
            raise ResearchPlanValidationError("dataset plan split day count is invalid")
    if development.start != full_start or holdout.end != full_end:
        raise ResearchPlanValidationError(
            "dataset splits do not span the full universe"
        )
    if validation.start != development.end + timedelta(days=1):
        raise ResearchPlanValidationError(
            "development/validation boundary is not contiguous"
        )
    if holdout.start != validation.end + timedelta(days=1):
        raise ResearchPlanValidationError(
            "validation/holdout boundary is not contiguous"
        )
    expected_cutoff = datetime(2024, 8, 21, tzinfo=UTC)
    if cutoff != expected_cutoff or cutoff.date() != holdout.start:
        raise ResearchPlanValidationError(
            "G1 cutoff does not match the holdout boundary"
        )

    semantic_sha256 = hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return ResearchDataPlan(
        dataset_id=DATASET_ID,
        market_data_origin="Dukascopy",
        distribution_source="Hugging Face",
        huggingface_repo=HF_REPO_ID,
        huggingface_revision=revision,
        instrument=INSTRUMENT_ID,
        full_start=full_start,
        full_end=full_end,
        boundary_rule=BOUNDARY_RULE,
        development=development,
        validation=validation,
        final_holdout=holdout,
        g1_allowed_end_exclusive=cutoff,
        semantic_sha256=semantic_sha256,
    )


def _load_split(value: object, name: str) -> ResearchSplit:
    raw = _mapping(value, f"split.{name}")
    _require_keys(raw, {"start", "end", "days"}, f"split.{name}")
    days = raw["days"]
    if not isinstance(days, int) or isinstance(days, bool) or days <= 0:
        raise ResearchPlanValidationError(f"split.{name}.days must be positive")
    return ResearchSplit(
        start=_date(raw["start"], f"split.{name}.start"),
        end=_date(raw["end"], f"split.{name}.end"),
        days=days,
    )


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ResearchPlanValidationError(f"{name} must be a string-keyed mapping")
    return cast(dict[str, Any], value)


def _require_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ResearchPlanValidationError(
            f"{name} must contain exactly: {', '.join(sorted(expected))}"
        )


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ResearchPlanValidationError(f"{name} must be a non-empty string")
    return value


def _date(value: object, name: str) -> date:
    text = _string(value, name)
    try:
        parsed = date.fromisoformat(text)
    except ValueError as error:
        raise ResearchPlanValidationError(f"{name} must use YYYY-MM-DD") from error
    if parsed.isoformat() != text:
        raise ResearchPlanValidationError(f"{name} must use YYYY-MM-DD")
    return parsed


def _utc_datetime(value: object, name: str) -> datetime:
    text = _string(value, name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ResearchPlanValidationError(
            f"{name} must be an ISO UTC timestamp"
        ) from error
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() != timedelta(0)
        or not text.endswith("Z")
    ):
        raise ResearchPlanValidationError(f"{name} must be an ISO UTC timestamp")
    return parsed.astimezone(UTC)
