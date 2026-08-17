"""Strict, independent loader for G1.4 multi-instrument research universes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any, cast

import yaml

from ftmoquant.data.instruments import InstrumentSpec, InstrumentSpecValidationError
from ftmoquant.data.research_plan import HF_REVISION_PATTERN, ResearchSplit

UNIVERSE_ID = "g1_4_fx_usd_liquid_v1"
UNIVERSE_VERSION = 1
HF_REPO_ID = "mito0o852/dukascopy-ticks"
PINNED_HF_REVISION = "bf19dbd89c732f010e20db7c148922ba02b2e33b"
HOLDOUT_CUTOFF = datetime(2024, 8, 21, tzinfo=UTC)

#: A second, additive G1.4 universe for majors not carried by the original
#: frozen ``g1_4_fx_usd_liquid_v1`` plan. Kept fully separate so nothing here
#: can change the original plan's semantic hash or the frozen artifacts that
#: already depend on it.
MAJORS_UNIVERSE_ID = "g1_4_fx_majors_v1"
MAJORS_UNIVERSE_VERSION = 1

_KNOWN_UNIVERSE_VERSIONS = {
    UNIVERSE_ID: UNIVERSE_VERSION,
    MAJORS_UNIVERSE_ID: MAJORS_UNIVERSE_VERSION,
}

#: Only the original v1 universe has its exact instrument order and
#: quote-currency restriction frozen; any other known universe id is
#: validated generically (well-formed, deduplicated instruments only).
_FROZEN_INSTRUMENT_ORDER = {
    UNIVERSE_ID: ("EUR/USD.DUKASCOPY", "GBP/USD.DUKASCOPY"),
}
_QUOTE_USD_ONLY_UNIVERSES = {UNIVERSE_ID}


class ResearchUniversePlanValidationError(ValueError):
    """Raised when a universe can access holdout data or has ambiguous identity."""


@dataclass(frozen=True, slots=True)
class ResearchUniversePlan:
    """Validated multi-instrument source and split contract."""

    universe_id: str
    version: int
    market_data_origin: str
    distribution_source: str
    huggingface_repo: str
    huggingface_revision: str
    data_use_rights_status: str
    permitted_start_utc: datetime
    permitted_end_exclusive_utc: datetime
    development: ResearchSplit
    validation: ResearchSplit
    holdout_cutoff_utc: datetime
    instruments: tuple[InstrumentSpec, ...]
    semantic_sha256: str

    @property
    def exact_currency_set(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    currency
                    for item in self.instruments
                    for currency in (item.base_currency, item.quote_currency)
                }
            )
        )

    def instrument(self, instrument_id: str) -> InstrumentSpec:
        matches = tuple(
            item for item in self.instruments if item.instrument_id == instrument_id
        )
        if len(matches) != 1:
            raise ResearchUniversePlanValidationError(
                f"universe does not contain exactly one {instrument_id}"
            )
        return matches[0]


def load_research_universe_plan(path: Path) -> ResearchUniversePlan:
    """Load the exact universe schema and bind ordering into its semantic hash."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ResearchUniversePlanValidationError(
            f"could not load universe plan: {error}"
        ) from error
    document = _mapping(raw, "universe plan")
    _keys(
        document,
        {
            "universe_id",
            "version",
            "source",
            "permitted_range",
            "splits",
            "instruments",
        },
        "universe plan",
    )
    universe_id = document["universe_id"]
    if (
        universe_id not in _KNOWN_UNIVERSE_VERSIONS
        or document["version"] != _KNOWN_UNIVERSE_VERSIONS[universe_id]
    ):
        raise ResearchUniversePlanValidationError(
            "universe identity/version is not frozen"
        )

    source = _mapping(document["source"], "source")
    _keys(
        source,
        {
            "market_data_origin",
            "distribution_source",
            "huggingface_repo",
            "huggingface_revision",
            "data_use_rights_status",
        },
        "source",
    )
    expected_source = {
        "market_data_origin": "Dukascopy",
        "distribution_source": "Hugging Face + direct Dukascopy cutoff completion",
        "huggingface_repo": HF_REPO_ID,
        "huggingface_revision": PINNED_HF_REVISION,
        "data_use_rights_status": "unconfirmed_release_blocker",
    }
    if source != expected_source:
        raise ResearchUniversePlanValidationError("source identity is not frozen")
    revision = cast(str, source["huggingface_revision"])
    if HF_REVISION_PATTERN.fullmatch(revision) is None:
        raise ResearchUniversePlanValidationError(
            "source revision must be an exact SHA"
        )

    permitted = _mapping(document["permitted_range"], "permitted_range")
    _keys(
        permitted,
        {"start_utc", "end_exclusive_utc", "holdout_cutoff_utc"},
        "permitted_range",
    )
    permitted_start = _utc(permitted["start_utc"], "permitted_range.start_utc")
    permitted_end = _utc(
        permitted["end_exclusive_utc"], "permitted_range.end_exclusive_utc"
    )
    cutoff = _utc(permitted["holdout_cutoff_utc"], "permitted_range.holdout_cutoff_utc")
    if (
        cutoff != HOLDOUT_CUTOFF
        or permitted_end > cutoff
        or permitted_end <= permitted_start
    ):
        raise ResearchUniversePlanValidationError(
            "permitted range crosses the frozen holdout cutoff"
        )

    splits = _mapping(document["splits"], "splits")
    _keys(splits, {"development", "validation"}, "splits")
    development = _split(splits["development"], "development")
    validation = _split(splits["validation"], "validation")
    if development.start != permitted_start.date():
        raise ResearchUniversePlanValidationError(
            "development must start at permitted range"
        )
    if validation.start != development.end + timedelta(days=1):
        raise ResearchUniversePlanValidationError(
            "development and validation must be contiguous"
        )
    if (
        datetime.combine(validation.end + timedelta(days=1), time.min, tzinfo=UTC)
        != permitted_end
    ):
        raise ResearchUniversePlanValidationError(
            "validation must end at permitted range"
        )

    raw_instruments = document["instruments"]
    if not isinstance(raw_instruments, list) or not raw_instruments:
        raise ResearchUniversePlanValidationError(
            "instruments must be a non-empty ordered list"
        )
    instruments = tuple(
        _instrument(value, index) for index, value in enumerate(raw_instruments)
    )
    ids = [item.instrument_id for item in instruments]
    symbols = [item.dataset_symbol for item in instruments]
    if len(ids) != len(set(ids)) or len(symbols) != len(set(symbols)):
        raise ResearchUniversePlanValidationError(
            "duplicate instrument IDs or source symbols"
        )
    frozen_order = _FROZEN_INSTRUMENT_ORDER.get(universe_id)
    if frozen_order is not None and tuple(ids) != frozen_order:
        raise ResearchUniversePlanValidationError(
            "initial universe order is not frozen"
        )
    if universe_id in _QUOTE_USD_ONLY_UNIVERSES and any(
        item.quote_currency != "USD" for item in instruments
    ):
        raise ResearchUniversePlanValidationError("v1 permits quote-USD pairs only")

    semantic_sha256 = hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return ResearchUniversePlan(
        universe_id=cast(str, universe_id),
        version=cast(int, document["version"]),
        market_data_origin="Dukascopy",
        distribution_source=cast(str, source["distribution_source"]),
        huggingface_repo=HF_REPO_ID,
        huggingface_revision=revision,
        data_use_rights_status=cast(str, source["data_use_rights_status"]),
        permitted_start_utc=permitted_start,
        permitted_end_exclusive_utc=permitted_end,
        development=development,
        validation=validation,
        holdout_cutoff_utc=cutoff,
        instruments=instruments,
        semantic_sha256=semantic_sha256,
    )


def _instrument(value: object, index: int) -> InstrumentSpec:
    raw = _mapping(value, f"instruments[{index}]")
    fields = {
        "dataset_symbol",
        "instrument_id",
        "base_currency",
        "quote_currency",
        "price_precision",
        "price_increment",
        "size_precision",
        "size_increment",
        "session_policy_id",
    }
    _keys(raw, fields, f"instruments[{index}]")
    try:
        item = InstrumentSpec(**raw)
        item.validate()
    except (TypeError, InstrumentSpecValidationError) as error:
        raise ResearchUniversePlanValidationError(
            f"invalid instruments[{index}]: {error}"
        ) from error
    return item


def _split(value: object, name: str) -> ResearchSplit:
    raw = _mapping(value, f"splits.{name}")
    _keys(raw, {"start", "end", "days"}, f"splits.{name}")
    try:
        start = date.fromisoformat(cast(str, raw["start"]))
        end = date.fromisoformat(cast(str, raw["end"]))
    except (TypeError, ValueError) as error:
        raise ResearchUniversePlanValidationError(f"invalid {name} dates") from error
    days = raw["days"]
    if (
        not isinstance(days, int)
        or isinstance(days, bool)
        or days != (end - start).days + 1
    ):
        raise ResearchUniversePlanValidationError(f"invalid {name} day count")
    return ResearchSplit(start=start, end=end, days=days)


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ResearchUniversePlanValidationError(
            f"{name} must be a string-keyed mapping"
        )
    return cast(dict[str, Any], value)


def _keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ResearchUniversePlanValidationError(
            f"{name} must contain exactly: {', '.join(sorted(expected))}"
        )


def _utc(value: object, name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ResearchUniversePlanValidationError(
            f"{name} must be an ISO UTC timestamp"
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ResearchUniversePlanValidationError(
            f"{name} must be an ISO UTC timestamp"
        ) from error
    if parsed.utcoffset() != timedelta(0) or parsed.second or parsed.microsecond:
        raise ResearchUniversePlanValidationError(f"{name} must be minute-aligned UTC")
    return parsed.astimezone(UTC)
