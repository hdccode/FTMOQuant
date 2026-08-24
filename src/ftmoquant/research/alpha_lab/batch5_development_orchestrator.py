"""Sealed DEVELOPMENT-only orchestration and CLI for frozen Batch 5.

Importing this module never opens a catalog or runs a screen.  The only public
runner first rejects forbidden roots, refuses overwrite, verifies every frozen
identity/readiness artifact, and only then loads DEVELOPMENT market data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ftmoquant.backtest.execution_harness import _sha256_tree
from ftmoquant.data.cftc_tff_batch5 import LINEAGE_ID as CFTC_LINEAGE_ID
from ftmoquant.data.instruments import (
    OANDA_ALPHA_LAB_SPECS,
    OANDA_BATCH5_CROSS_SPECS,
    oanda_symbol,
)
from ftmoquant.data.oanda_batch5_crosses import (
    LINEAGE_ID as CROSS_LINEAGE_ID,
)
from ftmoquant.data.oanda_batch5_crosses import (
    load_config as load_cross_config,
)
from ftmoquant.research.alpha_lab.batch4_execution import _validate_frames
from ftmoquant.research.alpha_lab.batch5_cftc_availability import (
    EXPECTED_AMENDMENT_SEMANTIC_SHA256,
    verify_amendment,
)
from ftmoquant.research.alpha_lab.batch5_daily import (
    CompletedFxDay,
    build_completed_fx_days,
)
from ftmoquant.research.alpha_lab.batch5_development_scorecard import (
    DevelopmentSleeveInput,
    build_diagnostics_summary,
    build_family_summary,
    build_selection_summary,
    evaluate_development_sleeve,
    write_batch5_artifacts,
)
from ftmoquant.research.alpha_lab.batch5_execution import (
    Batch5SkipRecord,
    Batch5TradeResult,
)
from ftmoquant.research.alpha_lab.batch5_preregistration import (
    B5C_INSTRUMENTS,
    EXPECTED_PREREGISTRATION_SEMANTIC_SHA256,
    FAMILY_B5A,
    FAMILY_B5B,
    FAMILY_B5C,
    PRIMARY_FAMILIES,
    verify_preregistration,
)
from ftmoquant.research.alpha_lab.batch5_screen import (
    FrequencyStats,
    _expected_sleeves,
    load_frozen_policy,
)
from ftmoquant.research.alpha_lab.batch5a_cftc_execution import execute_cohorts
from ftmoquant.research.alpha_lab.batch5a_cftc_signals import (
    B5ASignal,
    CftcDealerObservation,
    YearMonth,
    signal_for_month,
)
from ftmoquant.research.alpha_lab.batch5b_direct_mr_execution import (
    execute_positions,
)
from ftmoquant.research.alpha_lab.batch5b_direct_mr_signals import (
    B5BSignal,
    generate_signals,
)
from ftmoquant.research.alpha_lab.batch5c_daily_reversal_execution import (
    execute_events,
)
from ftmoquant.research.alpha_lab.batch5c_daily_reversal_signals import (
    B5CEvent,
    generate_events,
)
from ftmoquant.research.alpha_lab.cost_stress import widen_bid_ask_frame
from ftmoquant.research.alpha_lab.data import _discover_oanda_universe
from ftmoquant.research.alpha_lab.wick_fvg_squeeze_execution import load_m1_bidask
from ftmoquant.research.stage_g import DEVELOPMENT_END_EXCLUSIVE, DEVELOPMENT_START

ARTIFACT_ROOT = Path(".artifacts/alpha_lab/batch5_three_fx_alpha_families_v1")
COST_MULTIPLIERS = (Decimal("1.0"), Decimal("1.5"), Decimal("2.0"))
_FORBIDDEN_ROOT_TOKENS = (
    "validation",
    "holdout",
    "final_holdout",
    "final-test",
    "final_test",
)
_DEVELOPMENT_DAYS = (DEVELOPMENT_END_EXCLUSIVE - DEVELOPMENT_START).days
_FOLD_DAYS = _DEVELOPMENT_DAYS // 4
DEVELOPMENT_FOLD_BOUNDARIES = tuple(
    DEVELOPMENT_START + timedelta(days=index * _FOLD_DAYS) for index in range(5)
)
if DEVELOPMENT_FOLD_BOUNDARIES[-1] != DEVELOPMENT_END_EXCLUSIVE:
    raise RuntimeError("Batch 5 DEVELOPMENT no longer divides into four folds")


class Batch5DevelopmentOrchestratorError(ValueError):
    """Raised before unsafe access or on frozen orchestration drift."""


@dataclass(frozen=True, slots=True)
class PreflightBundle:
    methodology: Mapping[str, Any]
    cftc_readiness: Mapping[str, Any]
    cross_readiness: Mapping[str, Any]
    canonical_readiness: Mapping[str, Any]
    cftc_readiness_path: Path
    cross_readiness_path: Path
    canonical_readiness_path: Path
    cftc_normalized_path: Path


def reject_forbidden_root(path: Path, *, label: str) -> None:
    lowered = str(path).lower()
    for token in _FORBIDDEN_ROOT_TOKENS:
        if token in lowered:
            raise Batch5DevelopmentOrchestratorError(
                f"{label} path contains forbidden token {token!r}: {path}"
            )


def reserve_output_directory(output_dir: Path) -> None:
    if output_dir.exists():
        raise Batch5DevelopmentOrchestratorError(
            f"{output_dir} already exists; refusing to overwrite"
        )


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Batch5DevelopmentOrchestratorError(
            f"could not read {label}: {path}"
        ) from error
    if not isinstance(value, dict):
        raise Batch5DevelopmentOrchestratorError(f"{label} must be a mapping")
    return cast(dict[str, Any], value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _semantic_sha(document: Mapping[str, Any]) -> str:
    semantic = {
        key: value for key, value in document.items() if key != "semantic_sha256"
    }
    return hashlib.sha256(
        json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def verify_frozen_methodology() -> dict[str, Any]:
    """Pin complete hypotheses, universes, gates, breadth, and no-rescue policy."""

    methodology = verify_preregistration()
    amendment = verify_amendment()
    if (
        methodology["preregistration_semantic_sha256"]
        != EXPECTED_PREREGISTRATION_SEMANTIC_SHA256
        or amendment["amendment_semantic_sha256"] != EXPECTED_AMENDMENT_SEMANTIC_SHA256
    ):
        raise Batch5DevelopmentOrchestratorError("Batch 5 semantic identity drift")
    if tuple(methodology["family_scope"]["primary_exact"]) != PRIMARY_FAMILIES:
        raise Batch5DevelopmentOrchestratorError("frozen family identities drifted")
    if methodology["family_scope"]["exact_family_count"] != 3:
        raise Batch5DevelopmentOrchestratorError("Batch 5 must contain three families")
    expected = _expected_sleeves()
    counts = tuple(
        map(len, (expected[FAMILY_B5A], expected[FAMILY_B5B], expected[FAMILY_B5C]))
    )
    if counts != (
        7,
        1,
        5,
    ):
        raise Batch5DevelopmentOrchestratorError("frozen sleeve counts drifted")
    if (
        tuple(
            methodology["families"][FAMILY_B5C]["literature_anchored_rule"]["universe"]
        )
        != B5C_INSTRUMENTS
    ):
        raise Batch5DevelopmentOrchestratorError("B5C universe drifted")
    load_frozen_policy()
    breadth = methodology["breadth_rules"]
    expected_breadth = {
        FAMILY_B5A: (5, 4),
        FAMILY_B5B: (1, 1),
        FAMILY_B5C: (3, 2),
    }
    observed_breadth = {
        FAMILY_B5A: (
            breadth[FAMILY_B5A]["sleeves_positive_native_and_1_5x_expectancy_gte"],
            breadth[FAMILY_B5A]["sleeves_passing_all_sleeve_gates_gte"],
        ),
        FAMILY_B5B: (
            1,
            int(breadth[FAMILY_B5B]["single_sleeve_must_pass_all_gates"]),
        ),
        FAMILY_B5C: (
            breadth[FAMILY_B5C]["sleeves_positive_native_and_1_5x_expectancy_gte"],
            breadth[FAMILY_B5C]["sleeves_passing_all_sleeve_gates_gte"],
        ),
    }
    if observed_breadth != expected_breadth:
        raise Batch5DevelopmentOrchestratorError("frozen breadth drifted")
    promotion = methodology["development_to_validation"]
    required_rescues = {
        "sign inversion",
        "nearby parameter",
        "alternate pair",
        "alternate holding period",
        "threshold relaxation",
        "favorable seed retry",
        "alternate data vintage",
    }
    if (
        promotion["maximum_representatives_per_family"] != 1
        or promotion["validation_in_this_task"] is not False
        or set(promotion["rescue_forbidden"]) != required_rescues
        or set(promotion["representative_selection"])
        != {*PRIMARY_FAMILIES, "tie_break_if_schema_error_creates_duplicate"}
    ):
        raise Batch5DevelopmentOrchestratorError("no-rescue policy drifted")
    return methodology


def _verify_cftc_readiness(cftc_root: Path) -> tuple[dict[str, Any], Path, Path]:
    readiness_path = cftc_root / "readiness" / "cftc_tff_readiness.json"
    readiness = _read_json(readiness_path, label="CFTC readiness")
    normalized_path = (cftc_root / str(readiness.get("normalized_path", ""))).resolve()
    if not normalized_path.is_relative_to(cftc_root.resolve()):
        raise Batch5DevelopmentOrchestratorError("CFTC normalized path escaped root")
    reject_forbidden_root(normalized_path, label="CFTC normalized path")
    if (
        readiness.get("lineage_id") != CFTC_LINEAGE_ID
        or readiness.get("research_ready") is not True
        or readiness.get("original_preregistration_semantic_sha256")
        != EXPECTED_PREREGISTRATION_SEMANTIC_SHA256
        or readiness.get("availability_amendment_semantic_sha256")
        != EXPECTED_AMENDMENT_SEMANTIC_SHA256
        or readiness.get("warmup_start") != "2018-03-01"
        or readiness.get("development_start") != "2019-03-11"
        or readiness.get("development_end_inclusive") != "2023-04-11"
        or readiness.get("data_firewall", {}).get("validation_or_holdout_read")
        is not False
    ):
        raise Batch5DevelopmentOrchestratorError("CFTC readiness identity drift")
    if not normalized_path.is_file() or _sha256_file(normalized_path) != readiness.get(
        "normalized_sha256"
    ):
        raise Batch5DevelopmentOrchestratorError("CFTC normalized hash drift")
    if set(readiness.get("rows_per_currency", {})) != {
        "AUD",
        "CAD",
        "CHF",
        "EUR",
        "GBP",
        "JPY",
        "NZD",
    }:
        raise Batch5DevelopmentOrchestratorError("CFTC currency universe drift")
    return readiness, readiness_path, normalized_path


def _verify_cross_readiness(cross_root: Path) -> tuple[dict[str, Any], Path]:
    config = load_cross_config()
    readiness_path = (
        cross_root / "readiness" / "oanda_batch5_native_crosses_readiness.json"
    )
    readiness = _read_json(readiness_path, label="Batch 5 cross readiness")
    if (
        readiness.get("semantic_sha256") != _semantic_sha(readiness)
        or readiness.get("lineage_id") != CROSS_LINEAGE_ID
        or readiness.get("config_semantic_sha256") != config.semantic_sha256
        or readiness.get("batch5_preregistration_semantic_sha256")
        != EXPECTED_PREREGISTRATION_SEMANTIC_SHA256
        or readiness.get("research_ready") is not True
        or readiness.get("validation_accessed") is not False
        or readiness.get("holdout_accessed") is not False
        or readiness.get("performance_accessed") is not False
        or readiness.get("acquisition_start_utc")
        != config.acquisition_start_utc.isoformat()
        or readiness.get("development_start_utc")
        != config.development_start_utc.isoformat()
        or readiness.get("development_end_exclusive_utc")
        != config.development_end_exclusive_utc.isoformat()
    ):
        raise Batch5DevelopmentOrchestratorError("native-cross readiness drift")
    artifacts = readiness.get("instrument_artifacts")
    if not isinstance(artifacts, list) or {
        row.get("instrument_id") for row in artifacts if isinstance(row, dict)
    } != set(config.instruments):
        raise Batch5DevelopmentOrchestratorError("native-cross universe drift")
    for item in artifacts:
        row = cast(dict[str, Any], item)
        instrument_id = str(row["instrument_id"])
        if row.get("research_ready") is not True:
            raise Batch5DevelopmentOrchestratorError(
                f"native cross is not research-ready: {instrument_id}"
            )
        spec = next(
            spec
            for spec in OANDA_BATCH5_CROSS_SPECS
            if spec.instrument_id == instrument_id
        )
        root = cross_root / "canonical" / oanda_symbol(spec.dataset_symbol)
        if _sha256_tree(root / "catalog") != row.get("catalog_tree_sha256"):
            raise Batch5DevelopmentOrchestratorError(
                f"native-cross catalog hash drift: {instrument_id}"
            )
        qa_path = (
            cross_root
            / "qa"
            / f"{oanda_symbol(spec.dataset_symbol)}_M1_bid_ask_qa.json"
        )
        if not qa_path.is_file() or _sha256_file(qa_path) != row.get(
            "acquisition_qa_sha256"
        ):
            raise Batch5DevelopmentOrchestratorError(
                f"native-cross QA hash drift: {instrument_id}"
            )
    return readiness, readiness_path


def verify_preflight(
    *,
    development_root: Path,
    universe_readiness: Path,
    batch5_cross_root: Path,
    cftc_root: Path,
) -> PreflightBundle:
    """Verify every identity and catalog tree before any M1/CFTC row loading."""

    for label, path in (
        ("--development-root", development_root),
        ("--universe-readiness", universe_readiness),
        ("--batch5-cross-root", batch5_cross_root),
        ("--cftc-root", cftc_root),
    ):
        reject_forbidden_root(path, label=label)
    methodology = verify_frozen_methodology()
    cftc, cftc_path, normalized = _verify_cftc_readiness(cftc_root)
    crosses, crosses_path = _verify_cross_readiness(batch5_cross_root)
    canonical = _read_json(universe_readiness, label="canonical OANDA readiness")
    if (
        canonical.get("semantic_sha256") != _semantic_sha(canonical)
        or canonical.get("readiness_version") != "oanda-alpha-lab-readiness-1"
        or canonical.get("lineage_id") != "oanda_fx_alpha_lab_v1"
        or canonical.get("development_start_utc") != "2019-03-11T00:00:00Z"
        or canonical.get("development_end_exclusive_utc") != "2023-04-11T00:00:00Z"
        or canonical.get("research_ready") is not True
        or canonical.get("holdout_accessed") is not False
        or canonical.get("holdout_rows_admitted") != 0
    ):
        raise Batch5DevelopmentOrchestratorError("canonical OANDA readiness drift")
    instruments, _ = _discover_oanda_universe(universe_readiness, development_root)
    expected = tuple(sorted(spec.instrument_id for spec in OANDA_ALPHA_LAB_SPECS))
    if instruments != expected:
        raise Batch5DevelopmentOrchestratorError("canonical OANDA universe drift")
    return PreflightBundle(
        methodology,
        cftc,
        crosses,
        canonical,
        cftc_path,
        crosses_path,
        universe_readiness,
        normalized,
    )


FramePair = tuple[pd.DataFrame, pd.DataFrame]
M1Loader = Callable[..., FramePair]


class CostFrameCache:
    """Load each native M1 stream once and widen each requested state once."""

    def __init__(
        self,
        *,
        development_root: Path,
        batch5_cross_root: Path,
        loader: M1Loader = load_m1_bidask,
    ) -> None:
        self.development_root = development_root
        self.batch5_cross_root = batch5_cross_root
        self.loader = loader
        self._frames: dict[tuple[str, Decimal], FramePair] = {}
        self.native_load_count: dict[str, int] = defaultdict(int)
        self.widen_count: dict[tuple[str, Decimal], int] = defaultdict(int)
        self._canonical = {spec.instrument_id: spec for spec in OANDA_ALPHA_LAB_SPECS}
        self._crosses = {spec.instrument_id: spec for spec in OANDA_BATCH5_CROSS_SPECS}
        self._cross_start = load_cross_config().acquisition_start_utc

    def _load_native(self, instrument_id: str) -> FramePair:
        if instrument_id in self._crosses:
            spec = self._crosses[instrument_id]
            root = (
                self.batch5_cross_root / "canonical" / oanda_symbol(spec.dataset_symbol)
            )
            start = self._cross_start
        elif instrument_id in self._canonical:
            spec = self._canonical[instrument_id]
            root = self.development_root / oanda_symbol(spec.dataset_symbol)
            start = DEVELOPMENT_START
        else:
            raise Batch5DevelopmentOrchestratorError(
                f"instrument outside frozen data roots: {instrument_id}"
            )
        frames = self.loader(
            instrument_id=instrument_id,
            root=root,
            start_utc=start,
            end_exclusive_utc=DEVELOPMENT_END_EXCLUSIVE,
        )
        index = _validate_frames(*frames)
        if index[-1].to_pydatetime() >= DEVELOPMENT_END_EXCLUSIVE:
            raise Batch5DevelopmentOrchestratorError("M1 escaped DEVELOPMENT")
        self.native_load_count[instrument_id] += 1
        return frames

    def frames(self, instrument_id: str, multiplier: Decimal) -> FramePair:
        if multiplier not in COST_MULTIPLIERS:
            raise Batch5DevelopmentOrchestratorError("cost multiplier is not frozen")
        native_key = (instrument_id, Decimal("1.0"))
        if native_key not in self._frames:
            self._frames[native_key] = self._load_native(instrument_id)
        key = (instrument_id, multiplier)
        if key not in self._frames:
            native = self._frames[native_key]
            self._frames[key] = widen_bid_ask_frame(
                native[0], native[1], float(multiplier)
            )
            self.widen_count[key] += 1
        return self._frames[key]


def load_cftc_observations(
    bundle: PreflightBundle,
) -> tuple[CftcDealerObservation, ...]:
    observations: list[CftcDealerObservation] = []
    with bundle.cftc_normalized_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row["original_preregistration_semantic_sha256"]
                != EXPECTED_PREREGISTRATION_SEMANTIC_SHA256
                or row["availability_amendment_semantic_sha256"]
                != EXPECTED_AMENDMENT_SEMANTIC_SHA256
            ):
                raise Batch5DevelopmentOrchestratorError("CFTC row lineage drift")
            timestamp = (
                datetime.fromisoformat(row["availability_timestamp"])
                if row["availability_timestamp"]
                else None
            )
            observations.append(
                CftcDealerObservation(
                    currency=row["currency"],
                    report_date=date.fromisoformat(row["report_date"]),
                    dealer_long=Decimal(row["dealer_long_all"]),
                    dealer_short=Decimal(row["dealer_short_all"]),
                    open_interest=Decimal(row["open_interest_all"]),
                    availability_timestamp=timestamp,
                    availability_status=row["availability_status"],
                    source_vintage=row["first_known_vintage"],
                )
            )
    if len(observations) != bundle.cftc_readiness.get("normalized_row_count"):
        raise Batch5DevelopmentOrchestratorError("CFTC row count drift")
    return tuple(observations)


def _month_range(first: YearMonth, last: YearMonth) -> tuple[YearMonth, ...]:
    result: list[YearMonth] = []
    current = first
    while current <= last:
        result.append(current)
        current = current.add(1)
    return tuple(result)


def _formation_timestamp(
    observations: Sequence[CftcDealerObservation], currency: str, month: YearMonth
) -> datetime | None:
    current: list[datetime] = [
        row.availability_timestamp
        for row in observations
        if row.currency == currency
        and row.month == month
        and row.availability_timestamp is not None
    ]
    if not current:
        return None
    next_month = month.add(1)
    month_end = datetime(next_month.year, next_month.month, 1, tzinfo=UTC)
    return max(month_end, *current)


def _formation_available(
    observations: Sequence[CftcDealerObservation],
    currency: str,
    month: YearMonth,
    timestamp: datetime,
) -> bool:
    visible_months = {
        row.month
        for row in observations
        if row.currency == currency
        and row.availability_timestamp is not None
        and row.availability_timestamp <= timestamp
        and row.month <= month
    }
    return all(month.add(offset) in visible_months for offset in range(-12, 1))


def materialize_b5a_signals(
    observations: Sequence[CftcDealerObservation],
) -> tuple[tuple[B5ASignal, ...], dict[str, tuple[datetime, ...]]]:
    document = verify_preregistration()
    rows = document["families"][FAMILY_B5A]["sleeves"]
    signals: list[B5ASignal] = []
    formations: dict[str, list[datetime]] = defaultdict(list)
    for sleeve in rows:
        currency = str(sleeve["currency_k"])
        sleeve_id = str(sleeve["sleeve_id"])
        for month in _month_range(YearMonth(2018, 3), YearMonth(2023, 3)):
            timestamp = _formation_timestamp(observations, currency, month)
            if timestamp is None or not _formation_available(
                observations, currency, month, timestamp
            ):
                continue
            if DEVELOPMENT_START <= timestamp < DEVELOPMENT_END_EXCLUSIVE:
                formations[sleeve_id].append(timestamp)
                signal = signal_for_month(
                    observations,
                    currency=currency,
                    formation_month=month,
                    formation_timestamp=timestamp,
                )
                if signal is not None:
                    signals.append(signal)
    return (
        tuple(
            sorted(signals, key=lambda row: (row.formation_timestamp, row.sleeve_id))
        ),
        {key: tuple(value) for key, value in formations.items()},
    )


def _trades(
    results: Sequence[Batch5TradeResult | Batch5SkipRecord],
) -> tuple[Batch5TradeResult, ...]:
    return tuple(
        row
        for row in results
        if isinstance(row, Batch5TradeResult)
        and row.signal_timestamp >= DEVELOPMENT_START
        and row.actual_entry_timestamp >= DEVELOPMENT_START
        and row.actual_exit_timestamp < DEVELOPMENT_END_EXCLUSIVE
    )


def _group(
    rows: Sequence[Batch5TradeResult],
) -> dict[str, tuple[Batch5TradeResult, ...]]:
    grouped: dict[str, list[Batch5TradeResult]] = defaultdict(list)
    for row in rows:
        grouped[row.sleeve_id].append(row)
    return {
        key: tuple(sorted(value, key=lambda row: row.actual_exit_timestamp))
        for key, value in grouped.items()
    }


def _nonoverlapping_units(rows: Sequence[Batch5TradeResult]) -> int:
    """Maximum deterministic set of completed non-overlapping cohort intervals."""

    count = 0
    last_exit: datetime | None = None
    for row in sorted(
        rows, key=lambda item: (item.actual_exit_timestamp, item.actual_entry_timestamp)
    ):
        if last_exit is None or row.actual_entry_timestamp >= last_exit:
            count += 1
            last_exit = row.actual_exit_timestamp
    return count


def _execution_keys(rows: Sequence[Batch5TradeResult]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            row.sleeve_id,
            row.signal_timestamp,
            row.actual_entry_timestamp,
            row.actual_exit_timestamp,
            row.direction,
            row.cohort_id,
        )
        for row in sorted(
            rows,
            key=lambda item: (
                item.sleeve_id,
                item.signal_timestamp,
                item.actual_entry_timestamp,
            ),
        )
    )


def _require_stress_timing_identity(
    passes: Sequence[Sequence[Batch5TradeResult]],
) -> None:
    keys = tuple(_execution_keys(rows) for rows in passes)
    if len(keys) != 3 or len(set(keys)) != 1:
        raise Batch5DevelopmentOrchestratorError(
            "signal/fill timing changed across cost states"
        )


def run_b5a(
    observations: Sequence[CftcDealerObservation], cache: CostFrameCache
) -> tuple[DevelopmentSleeveInput, ...]:
    signals, formations = materialize_b5a_signals(observations)
    passes: list[tuple[Batch5TradeResult, ...]] = []
    instruments = sorted({signal.instrument_id for signal in signals})
    for multiplier in COST_MULTIPLIERS:
        frame_map = {
            instrument: cache.frames(instrument, multiplier)
            for instrument in instruments
        }
        passes.append(_trades(execute_cohorts(signals, native_frames=frame_map)))
    _require_stress_timing_identity(passes)
    grouped = tuple(_group(rows) for rows in passes)
    document = verify_preregistration()
    result = []
    for sleeve in document["families"][FAMILY_B5A]["sleeves"]:
        sleeve_id = str(sleeve["sleeve_id"])
        formation_rows = formations.get(sleeve_id, ())
        formation_count = len(formation_rows)
        independent_units = _nonoverlapping_units(grouped[0].get(sleeve_id, ()))
        result.append(
            DevelopmentSleeveInput(
                FAMILY_B5A,
                "B5A_FROZEN_CFTC_DEALER_DEMAND_SHOCK",
                sleeve_id,
                str(sleeve["spot_instrument"]),
                grouped[0].get(sleeve_id, ()),
                grouped[1].get(sleeve_id, ()),
                grouped[2].get(sleeve_id, ()),
                DEVELOPMENT_FOLD_BOUNDARIES,
                FrequencyStats(
                    monthly_formation_count=formation_count,
                    nonoverlapping_three_month_units=independent_units,
                    active_year_count=len({value.year for value in formation_rows}),
                ),
                independent_units,
                len({value.year for value in formation_rows}),
            )
        )
    return tuple(result)


def _b5b_frequency(signals: Sequence[B5BSignal]) -> tuple[int, int, int]:
    relevant = tuple(
        row
        for row in signals
        if DEVELOPMENT_START <= row.signal_timestamp < DEVELOPMENT_END_EXCLUSIVE
    )
    active: str | None = None
    holding_days = 0
    changes = 0
    for signal in relevant:
        if active is not None:
            holding_days += 1
        if signal.direction == "FLAT":
            active = None
        elif active is None:
            active = signal.direction
        elif signal.direction != active:
            changes += 1
            active = signal.direction
    return holding_days, changes, len({row.signal_timestamp.year for row in relevant})


def run_b5b(cache: CostFrameCache) -> tuple[DevelopmentSleeveInput, ...]:
    native_bid, native_ask = cache.frames("AUD/CAD.OANDA", Decimal("1.0"))
    days = build_completed_fx_days("AUD/CAD.OANDA", native_bid, native_ask)
    signals = generate_signals(days)
    passes: list[tuple[Batch5TradeResult, ...]] = []
    for multiplier in COST_MULTIPLIERS:
        bid, ask = cache.frames("AUD/CAD.OANDA", multiplier)
        conversion_bid, conversion_ask = cache.frames("USD/CAD.OANDA", multiplier)
        passes.append(
            _trades(
                execute_positions(
                    signals,
                    bid_m1=bid,
                    ask_m1=ask,
                    usdcad_bid_m1=conversion_bid,
                    usdcad_ask_m1=conversion_ask,
                )
            )
        )
    _require_stress_timing_identity(passes)
    holding_days, changes, years = _b5b_frequency(signals)
    return (
        DevelopmentSleeveInput(
            FAMILY_B5B,
            "B5B_FROZEN_DIRECT_AUDCAD_MR",
            "B5B_AUDCAD",
            "AUD/CAD.OANDA",
            passes[0],
            passes[1],
            passes[2],
            DEVELOPMENT_FOLD_BOUNDARIES,
            FrequencyStats(
                daily_holding_observation_count=holding_days,
                position_sign_change_count=changes,
                rollover_supported=False,
                active_year_count=years,
            ),
            holding_days,
            years,
        ),
    )


def run_b5c(cache: CostFrameCache) -> tuple[DevelopmentSleeveInput, ...]:
    days_by_instrument: dict[str, tuple[CompletedFxDay, ...]] = {}
    events_by_instrument: dict[str, tuple[B5CEvent, ...]] = {}
    for instrument in B5C_INSTRUMENTS:
        bid, ask = cache.frames(instrument, Decimal("1.0"))
        days = build_completed_fx_days(instrument, bid, ask)
        days_by_instrument[instrument] = days
        events_by_instrument[instrument] = tuple(
            event
            for event in generate_events(days)
            if DEVELOPMENT_START <= event.signal_timestamp < DEVELOPMENT_END_EXCLUSIVE
        )
    result: list[DevelopmentSleeveInput] = []
    for instrument in B5C_INSTRUMENTS:
        events = events_by_instrument[instrument]
        passes: list[tuple[Batch5TradeResult, ...]] = []
        for multiplier in COST_MULTIPLIERS:
            frames = cache.frames(instrument, multiplier)
            conversion = cache.frames("USD/JPY.OANDA", multiplier)
            passes.append(
                _trades(
                    execute_events(
                        events,
                        days_by_instrument=days_by_instrument,
                        native_frames={instrument: frames},
                        usdjpy_conversion_frames=conversion,
                    )
                )
            )
        _require_stress_timing_identity(passes)
        sleeve_id = f"B5C_{instrument.split('.')[0].replace('/', '')}"
        years = len({event.signal_timestamp.year for event in events})
        result.append(
            DevelopmentSleeveInput(
                FAMILY_B5C,
                "B5C_FROZEN_DAILY_OVERREACTION_REVERSAL",
                sleeve_id,
                instrument,
                passes[0],
                passes[1],
                passes[2],
                DEVELOPMENT_FOLD_BOUNDARIES,
                FrequencyStats(
                    event_count=len(events),
                    active_year_count=years,
                ),
                len(events),
                years,
            )
        )
    return tuple(result)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[4],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def build_metadata(
    *,
    bundle: PreflightBundle,
    inputs: Sequence[DevelopmentSleeveInput],
    cache: CostFrameCache,
) -> dict[str, Any]:
    sleeves = [item.sleeve_id for item in sorted(inputs, key=lambda row: row.sleeve_id)]
    cross_rows = {
        row["instrument_id"]: row
        for row in bundle.cross_readiness["instrument_artifacts"]
    }
    return {
        "stage": "B5.3_sealed_DEVELOPMENT_screen",
        "preregistration_semantic_sha256": (EXPECTED_PREREGISTRATION_SEMANTIC_SHA256),
        "cftc_availability_amendment_semantic_sha256": (
            EXPECTED_AMENDMENT_SEMANTIC_SHA256
        ),
        "cftc_readiness_file_sha256": _sha256_file(bundle.cftc_readiness_path),
        "cftc_normalized_sha256": bundle.cftc_readiness["normalized_sha256"],
        "cftc_lineage_id": bundle.cftc_readiness["lineage_id"],
        "audcad_qa_sha256": cross_rows["AUD/CAD.OANDA"]["acquisition_qa_sha256"],
        "audcad_catalog_tree_sha256": cross_rows["AUD/CAD.OANDA"][
            "catalog_tree_sha256"
        ],
        "eurjpy_qa_sha256": cross_rows["EUR/JPY.OANDA"]["acquisition_qa_sha256"],
        "eurjpy_catalog_tree_sha256": cross_rows["EUR/JPY.OANDA"][
            "catalog_tree_sha256"
        ],
        "batch5_cross_readiness_file_sha256": _sha256_file(bundle.cross_readiness_path),
        "batch5_cross_readiness_semantic_sha256": bundle.cross_readiness[
            "semantic_sha256"
        ],
        "canonical_oanda_readiness_file_sha256": _sha256_file(
            bundle.canonical_readiness_path
        ),
        "canonical_oanda_readiness_semantic_sha256": bundle.canonical_readiness.get(
            "semantic_sha256"
        ),
        "canonical_oanda_lineage_id": bundle.canonical_readiness.get("lineage_id"),
        "development_start_utc": _iso(DEVELOPMENT_START),
        "development_end_exclusive_utc": _iso(DEVELOPMENT_END_EXCLUSIVE),
        "development_fold_boundaries_utc": [
            _iso(value) for value in DEVELOPMENT_FOLD_BOUNDARIES
        ],
        "family_ids": list(PRIMARY_FAMILIES),
        "sleeve_ids": sleeves,
        "sleeve_count": 13,
        "cost_stress_multipliers": [str(value) for value in COST_MULTIPLIERS],
        "native_m1_load_counts": dict(sorted(cache.native_load_count.items())),
        "stress_widen_counts": {
            f"{instrument}@{multiplier}x": count
            for (instrument, multiplier), count in sorted(cache.widen_count.items())
        },
        "gate_definitions": bundle.methodology["common_development_gates"],
        "frequency_gate_definitions": {
            family: bundle.methodology["families"][family]["screening"]
            for family in PRIMARY_FAMILIES
        },
        "breadth_rules": bundle.methodology["breadth_rules"],
        "promotion_rule": bundle.methodology["development_to_validation"],
        "cache_design": (
            "one native paired-M1 load and at most one widening per instrument/cost "
            "state; cached frames are shared by every family and conversion"
        ),
        "b5b_rollover_supported": False,
        "b5b_rollover_policy": "fail_closed_no_financing_series_in_frozen_M1_dataset",
        "git_commit": _git_commit(),
        "dependency_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "nautilus_trader": importlib.metadata.version("nautilus_trader"),
        },
        "development_accessed": True,
        "validation_accessed": False,
        "holdout_accessed": False,
        "monte_carlo_run": False,
        "parameter_search_run": False,
        "fallback_or_rescue_used": False,
    }


def run_batch5_development_screen(
    *,
    development_root: Path,
    universe_readiness: Path,
    batch5_cross_root: Path,
    cftc_root: Path,
    output_dir: Path,
) -> None:
    """Run once only when explicitly invoked by the user in a terminal."""

    reject_forbidden_root(output_dir, label="--output")
    reserve_output_directory(output_dir)
    bundle = verify_preflight(
        development_root=development_root,
        universe_readiness=universe_readiness,
        batch5_cross_root=batch5_cross_root,
        cftc_root=cftc_root,
    )
    observations = load_cftc_observations(bundle)
    cache = CostFrameCache(
        development_root=development_root,
        batch5_cross_root=batch5_cross_root,
    )
    inputs = (*run_b5a(observations, cache), *run_b5b(cache), *run_b5c(cache))
    expected_sleeves = set().union(*_expected_sleeves().values())
    if len(inputs) != 13 or {item.sleeve_id for item in inputs} != expected_sleeves:
        raise Batch5DevelopmentOrchestratorError("orchestrator sleeve set drifted")
    ordered_inputs = tuple(sorted(inputs, key=lambda row: (row.family, row.sleeve_id)))
    scorecard = tuple(evaluate_development_sleeve(item) for item in ordered_inputs)
    families = build_family_summary(ordered_inputs)
    write_batch5_artifacts(
        sleeve_scorecard=scorecard,
        family_summary=families,
        selection_summary=build_selection_summary(families),
        diagnostics_summary=build_diagnostics_summary(scorecard),
        metadata=build_metadata(bundle=bundle, inputs=ordered_inputs, cache=cache),
        output_dir=output_dir,
    )


def benchmark_synthetic_runtime(row_count: int = 200_000) -> dict[str, float | int]:
    """Synthetic compute benchmark plus conservative nine-stream run estimate."""

    if row_count < 10_000:
        raise Batch5DevelopmentOrchestratorError("benchmark needs at least 10000 rows")
    index = pd.date_range(DEVELOPMENT_START, periods=row_count, freq="min", tz="UTC")
    base = np.linspace(1.0, 1.01, row_count)
    bid = pd.DataFrame(
        {"open": base, "high": base, "low": base, "close": base}, index=index
    )
    ask = pd.DataFrame(
        {
            "open": base + 0.0002,
            "high": base + 0.0002,
            "low": base + 0.0002,
            "close": base + 0.0002,
        },
        index=index,
    )
    started = time.perf_counter()
    widen_bid_ask_frame(bid, ask, 1.5)
    widen_bid_ask_frame(bid, ask, 2.0)
    widening = time.perf_counter() - started
    started = time.perf_counter()
    build_completed_fx_days("EUR/USD.OANDA", bid, ask)
    daily = time.perf_counter() - started
    expected_rows = _DEVELOPMENT_DAYS * 1440 * 5 // 7
    scale = expected_rows / row_count
    compute = (widening + daily) * scale * 9
    assumed_load = 9 * 30.0
    assumed_cftc_and_preflight = 120.0
    assumed_signal_execution_score_artifacts = 180.0
    total = (
        compute
        + assumed_load
        + assumed_cftc_and_preflight
        + assumed_signal_execution_score_artifacts
    )
    return {
        "synthetic_row_count": row_count,
        "expected_m1_rows_per_instrument": expected_rows,
        "two_widening_passes_seconds": widening,
        "daily_construction_seconds": daily,
        "linear_scale_factor": scale,
        "estimated_compute_seconds_nine_streams": compute,
        "assumed_m1_load_seconds_nine_streams": assumed_load,
        "assumed_cftc_and_preflight_seconds": assumed_cftc_and_preflight,
        "assumed_signal_execution_score_artifact_seconds": (
            assumed_signal_execution_score_artifacts
        ),
        "estimated_total_seconds": total,
        "estimated_total_minutes": total / 60,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the three frozen Batch 5 DEVELOPMENT families once."
    )
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--universe-readiness", type=Path, required=True)
    parser.add_argument("--batch5-cross-root", type=Path, required=True)
    parser.add_argument("--cftc-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_batch5_development_screen(
        development_root=args.development_root,
        universe_readiness=args.universe_readiness,
        batch5_cross_root=args.batch5_cross_root,
        cftc_root=args.cftc_root,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
