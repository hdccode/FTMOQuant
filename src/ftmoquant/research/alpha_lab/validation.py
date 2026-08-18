"""One-shot, preregistered VALIDATION stage for the three OANDA alpha-lab
families that survived DEVELOPMENT Stage 2 (Donchian H4, TSM H4, mean
reversion H1 -- see :mod:`ftmoquant.research.alpha_lab.screening_stage2`).

Reuses the exact same signal functions (:mod:`ftmoquant.research.alpha_lab.
families`), cost model, and vectorbt runner (:func:`~ftmoquant.research.
alpha_lab.screening_common.run_family_grid`) as Stage 1/2. The only new
logic here is: representative selection, the five frozen validation gates,
reporting-only temporal subperiod diagnostics, and a VALIDATION-partition
-scoped data loader that fails closed against DEVELOPMENT/HOLDOUT data.

No VALIDATION partition was ever frozen for the OANDA alpha-lab lineage
(``config/data/oanda_fx_alpha_lab_v1.yaml`` and its readiness manifest only
ever defined a DEVELOPMENT window). The one temporal boundary already frozen
anywhere in this repository is ``ftmoquant.research.stage_g.VALIDATION_START``
(== ``DEVELOPMENT_END_EXCLUSIVE``) / ``HOLDOUT_START`` -- and the OANDA
DEVELOPMENT window was itself deliberately set to the identical
``[2019-03-11, 2023-04-11)`` dates as that lineage. This module reuses that
one existing frozen boundary rather than inventing a new split: VALIDATION
here is ``[VALIDATION_START, HOLDOUT_START)`` == ``[2023-04-11, 2024-08-21)``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from nautilus_trader.persistence import ParquetDataCatalog

from ftmoquant.backtest.execution_harness import _sha256_tree
from ftmoquant.data.instruments import oanda_symbol
from ftmoquant.data.oanda_alpha_lab_development import (
    CONFIG_PATH as OANDA_ALPHA_LAB_CONFIG_PATH,
)
from ftmoquant.data.oanda_alpha_lab_development import (
    load_oanda_alpha_lab_config,
)
from ftmoquant.data.oanda_alpha_lab_validation import (
    VALIDATION_CONFIG_PATH as OANDA_ALPHA_LAB_VALIDATION_CONFIG_PATH,
)
from ftmoquant.data.oanda_alpha_lab_validation import (
    VALIDATION_READINESS_VERSION,
    load_oanda_alpha_lab_validation_config,
)
from ftmoquant.research.alpha_lab.data import (
    _TIMEFRAME_BAR_SPEC,
    AlphaLabDataset,
    Timeframe,
    assemble_aligned_dataset,
)
from ftmoquant.research.alpha_lab.screening_batch import GridConfig, _signals_for_config
from ftmoquant.research.alpha_lab.screening_common import (
    _ANNUALIZATION_FACTOR,
    _TIMEFRAME_FREQ,
    AGGREGATE_INSTRUMENT_LABEL,
    SCREENING_COST_LABEL,
    STRESSED_COST_MULTIPLIER,
    ScreeningResultRow,
    _build_aggregate_portfolio,
    run_family_grid,
    screening_cost_per_instrument,
)
from ftmoquant.research.alpha_lab.screening_stage2 import (
    DONCHIAN_FAMILY,
    DONCHIAN_H4_LOOKBACKS,
    MEAN_REVERSION_FAMILY,
    MEAN_REVERSION_LOOKBACKS,
    MEAN_REVERSION_Z_ENTRIES,
    MIN_POSITIVE_FOLDS,
    MIN_PROFITABLE_PAIRS,
    MIN_STRESSED_RETURN,
    TSM_FAMILY,
    TSM_H4_LOOKBACKS,
)
from ftmoquant.research.stage_g import (
    DEVELOPMENT_END_EXCLUSIVE,
    DEVELOPMENT_START,
    HOLDOUT_START,
    VALIDATION_START,
)

STAGE1_METADATA_PATH = Path(".artifacts/alpha_lab/screening_batch_v1/metadata.json")
STAGE2_SCORECARD_PATH = Path(".artifacts/alpha_lab/screening_stage2_v1/scorecard.csv")
STAGE2_FAMILY_ROBUSTNESS_PATH = Path(
    ".artifacts/alpha_lab/screening_stage2_v1/family_robustness.csv"
)
STAGE2_METADATA_PATH = Path(".artifacts/alpha_lab/screening_stage2_v1/metadata.json")
#: Default location for the VALIDATION readiness artifact (section 6/10 of
#: the data-lineage task) -- distinct from, and never satisfied by, the
#: DEVELOPMENT readiness at .../oanda_fx_alpha_lab_v1/readiness/.
DEFAULT_VALIDATION_READINESS_PATH = Path(
    "/Users/Shared/FTMOQuant-data/oanda_fx_alpha_lab_v1/validation_readiness/"
    "ftmoquant_oanda_alpha_lab_validation_readiness.json"
)
PREREGISTRATION_PATH = Path(
    "config/validation/oanda_alpha_lab_stage2_survivors_v1_preregistration.json"
)

PREREGISTRATION_VERSION = "oanda-alpha-lab-validation-preregistration-1"

#: Section 6 of the task spec, frozen before any validation result exists.
MIN_PAIR_GATE = 4
CONCENTRATION_MAX_SHARE = 0.70

#: Section 7: reporting-only diagnostic, not a gate. The ~498-day VALIDATION
#: window is too short for the 3-fold split used in DEVELOPMENT Stage 1/2, so
#: it is instead split into 2 equal non-overlapping calendar subperiods.
VALIDATION_SUBPERIOD_COUNT = 2

FAILURE_POLICY_STATEMENT = (
    "One-shot per family: exactly one representative configuration is "
    "evaluated per family. If a family's representative fails any of the "
    "five frozen gates, that family fails this validation stage outright -- "
    "no alternative parameter from the same family, pair exclusion, signal "
    "modification, stop/target modification, cost-model modification, "
    "timeframe change, or rerun is permitted for that family after "
    "validation results are observed."
)


class AlphaLabValidationError(ValueError):
    """Raised on any fail-closed guard: forbidden path, out-of-partition
    observation, or a lineage/provenance mismatch."""


class ValidationPreregistrationError(ValueError):
    """Raised when a preregistration document fails verification, or when
    execution is attempted without one."""


# ---------------------------------------------------------------------------
# Section 2: frozen, mechanical representative-selection rule.
# ---------------------------------------------------------------------------


def select_1d_representative(
    full_grid: Sequence[int], passing_values: Iterable[int]
) -> int:
    """Among ``passing_values``, choose the one minimizing the sum of
    absolute GRID-INDEX distances (position in ``full_grid``, not raw
    parameter value) to every other passing value. Ties -> smaller value."""

    sorted_grid = sorted(full_grid)
    index_of = {value: i for i, value in enumerate(sorted_grid)}
    passing_indices = sorted({index_of[value] for value in passing_values})
    if not passing_indices:
        raise ValueError("no passing values to select a representative from")

    def cost(i: int) -> int:
        return sum(abs(i - j) for j in passing_indices)

    best = min(passing_indices, key=lambda i: (cost(i), sorted_grid[i]))
    return sorted_grid[best]


def select_2d_representative(
    lookback_grid: Sequence[int],
    z_grid: Sequence[float],
    passing_cells: Iterable[tuple[int, float]],
) -> tuple[int, float]:
    """Among ``passing_cells``, choose the cell minimizing the sum of
    Manhattan GRID-INDEX distances to every other cell. Ties -> smaller
    lookback, then smaller z_entry."""

    sorted_lookbacks = sorted(lookback_grid)
    sorted_zs = sorted(z_grid)
    lookback_index = {value: i for i, value in enumerate(sorted_lookbacks)}
    z_index = {value: i for i, value in enumerate(sorted_zs)}
    cells = sorted({(lookback_index[lb], z_index[z]) for lb, z in passing_cells})
    if not cells:
        raise ValueError("no passing cells to select a representative from")

    def cost(cell: tuple[int, int]) -> int:
        return sum(abs(cell[0] - other[0]) + abs(cell[1] - other[1]) for other in cells)

    best = min(cells, key=lambda cell: (cost(cell), cell[0], cell[1]))
    return sorted_lookbacks[best[0]], sorted_zs[best[1]]


def _stage2_aggregate_rows(scorecard_path: Path) -> list[dict[str, str]]:
    with scorecard_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return [row for row in rows if row["instrument"] == AGGREGATE_INSTRUMENT_LABEL]


def _stage2_passes_gate(row: Mapping[str, str]) -> bool:
    return (
        float(row["stressed_cost_return"]) > MIN_STRESSED_RETURN
        and int(row["profitable_pair_count"]) >= MIN_PROFITABLE_PAIRS
        and int(row["positive_fold_count"]) >= MIN_POSITIVE_FOLDS
    )


def derive_representatives(
    scorecard_path: Path = STAGE2_SCORECARD_PATH,
) -> dict[str, dict[str, object]]:
    """Derive the exactly-3 representatives from the frozen Stage-2
    scorecard by applying the frozen Stage-2 survival gate (reused from
    :mod:`screening_stage2`) then the generic selection rule above. Reads
    only already-observed DEVELOPMENT evidence -- never validation data."""

    aggregate_rows = _stage2_aggregate_rows(scorecard_path)

    donchian_passing = {
        json.loads(row["parameters"])["lookback"]
        for row in aggregate_rows
        if row["family"] == DONCHIAN_FAMILY and _stage2_passes_gate(row)
    }
    tsm_passing = {
        json.loads(row["parameters"])["lookback"]
        for row in aggregate_rows
        if row["family"] == TSM_FAMILY and _stage2_passes_gate(row)
    }
    meanrev_passing = {
        (
            json.loads(row["parameters"])["lookback"],
            json.loads(row["parameters"])["z_entry"],
        )
        for row in aggregate_rows
        if row["family"] == MEAN_REVERSION_FAMILY and _stage2_passes_gate(row)
    }

    donchian_lookback = select_1d_representative(
        DONCHIAN_H4_LOOKBACKS, donchian_passing
    )
    tsm_lookback = select_1d_representative(TSM_H4_LOOKBACKS, tsm_passing)
    meanrev_lookback, meanrev_z = select_2d_representative(
        MEAN_REVERSION_LOOKBACKS, MEAN_REVERSION_Z_ENTRIES, meanrev_passing
    )

    return {
        DONCHIAN_FAMILY: {
            "strategy_id": f"validation_donchian_h4_lookback{donchian_lookback}_v1",
            "timeframe": "H4",
            "lookback": donchian_lookback,
        },
        TSM_FAMILY: {
            "strategy_id": f"validation_tsm_h4_lookback{tsm_lookback}_v1",
            "timeframe": "H4",
            "lookback": tsm_lookback,
        },
        MEAN_REVERSION_FAMILY: {
            "strategy_id": (
                f"validation_meanrev_h1_lookback{meanrev_lookback}_z{meanrev_z}_v1"
            ),
            "timeframe": "H1",
            "lookback": meanrev_lookback,
            "z_entry": meanrev_z,
        },
    }


# ---------------------------------------------------------------------------
# Preregistration: generated/verified WITHOUT ever loading validation data.
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(document: Mapping[str, object]) -> str:
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def build_preregistration(
    *,
    scorecard_path: Path = STAGE2_SCORECARD_PATH,
    family_robustness_path: Path = STAGE2_FAMILY_ROBUSTNESS_PATH,
    stage2_metadata_path: Path = STAGE2_METADATA_PATH,
    stage1_metadata_path: Path = STAGE1_METADATA_PATH,
    config_path: Path = OANDA_ALPHA_LAB_CONFIG_PATH,
) -> dict[str, Any]:
    """Build the full preregistration document. Reads only Stage-1/Stage-2
    DEVELOPMENT artifacts and the lineage config -- never validation data."""

    representatives = derive_representatives(scorecard_path)
    config = load_oanda_alpha_lab_config(config_path)

    document: dict[str, Any] = {
        "preregistration_version": PREREGISTRATION_VERSION,
        "created_at_utc": _iso(datetime.now(UTC)),
        "development_lineage": {
            "lineage_id": config.lineage_id,
            "alpha_lab_config_sha256": config.semantic_sha256,
            "development_start_utc": _iso(DEVELOPMENT_START),
            "development_end_exclusive_utc": _iso(DEVELOPMENT_END_EXCLUSIVE),
        },
        "stage2_evidence": {
            "scorecard_sha256": _sha256_file(scorecard_path),
            "family_robustness_sha256": _sha256_file(family_robustness_path),
            "metadata_sha256": _sha256_file(stage2_metadata_path),
            "stage1_metadata_sha256": (
                _sha256_file(stage1_metadata_path)
                if stage1_metadata_path.is_file()
                else None
            ),
        },
        "representative_selection_algorithm": {
            "one_dimensional": (
                "Among Stage-2-passing lookbacks, choose the value minimizing "
                "the sum of absolute GRID-INDEX distances to all other "
                "passing values (not numeric distance); tie -> smaller value."
            ),
            "two_dimensional": (
                "Among Stage-2-passing cells in the largest connected "
                "component, choose the cell minimizing the sum of Manhattan "
                "GRID-INDEX distances to all other cells in that component; "
                "tie -> smaller lookback, then smaller z_entry."
            ),
        },
        "selected_representatives": representatives,
        "validation_partition": {
            "start_utc": _iso(VALIDATION_START),
            "end_exclusive_utc": _iso(HOLDOUT_START),
            "source": (
                "No VALIDATION partition was ever frozen for the OANDA "
                "alpha-lab lineage. Reused from "
                "ftmoquant.research.stage_g.VALIDATION_START/HOLDOUT_START "
                "-- the only validation boundary already frozen anywhere in "
                "this repository, chained off the identical "
                "DEVELOPMENT_END_EXCLUSIVE date the OANDA lineage's own "
                "DEVELOPMENT window also uses."
            ),
        },
        "cost_model": {
            "label": SCREENING_COST_LABEL,
            "stress_multiplier": STRESSED_COST_MULTIPLIER,
            "recalibration_from_validation_outcomes": "forbidden",
        },
        "validation_gates": {
            "A_cross_pair_breadth": (
                f"profitable_pair_count >= {MIN_PAIR_GATE} of 7, base cost"
            ),
            "B_aggregate_base_profitability": (
                "aggregate_equal_weight_return > 0, base cost"
            ),
            "C_aggregate_cost_stress_survival": (
                f"aggregate_stressed_return > 0 at {STRESSED_COST_MULTIPLIER}x "
                "screening costs"
            ),
            "D_aggregate_risk_adjusted_sign": "aggregate_equal_weight_sharpe > 0",
            "E_catastrophic_concentration_guard": (
                "max single-pair share of total positive pair-level net "
                f"profit <= {CONCENTRATION_MAX_SHARE}; fails if total "
                "positive profit <= 0"
            ),
            "validation_passed": "AND of all five gates A-E; no discretionary ranking",
        },
        "temporal_diagnostics": {
            "status": "reporting_only_not_a_gate",
            "subperiod_count": VALIDATION_SUBPERIOD_COUNT,
            "fields": [
                "positive_subperiod_count",
                "worst_subperiod_return",
                "median_subperiod_sharpe",
            ],
        },
        "one_shot_policy": {
            "representatives_per_family": 1,
            "total_configurations": 3,
            "search_or_selection_after_this_point": "forbidden",
            "failure_policy": FAILURE_POLICY_STATEMENT,
        },
        "holdout_accessed": False,
    }
    document["preregistration_semantic_sha256"] = _canonical_sha256(document)
    return document


def load_preregistration(path: Path = PREREGISTRATION_PATH) -> dict[str, Any]:
    try:
        return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationPreregistrationError(
            f"could not read preregistration: {error}"
        ) from error


def verify_preregistration(
    document: Mapping[str, Any],
    *,
    scorecard_path: Path = STAGE2_SCORECARD_PATH,
    family_robustness_path: Path = STAGE2_FAMILY_ROBUSTNESS_PATH,
    stage2_metadata_path: Path = STAGE2_METADATA_PATH,
    stage1_metadata_path: Path = STAGE1_METADATA_PATH,
    config_path: Path = OANDA_ALPHA_LAB_CONFIG_PATH,
) -> None:
    """Fail closed unless the preregistration is self-consistent, still
    matches the current lineage config, and its representatives are exactly
    reproducible from the pinned Stage-2 evidence right now."""

    working = dict(document)
    stored_hash = working.pop("preregistration_semantic_sha256", None)
    if stored_hash is None or _canonical_sha256(working) != stored_hash:
        raise ValidationPreregistrationError(
            "preregistration content does not match its own self-hash"
        )
    if document.get("preregistration_version") != PREREGISTRATION_VERSION:
        raise ValidationPreregistrationError("unrecognized preregistration_version")
    if document.get("holdout_accessed") is not False:
        raise ValidationPreregistrationError(
            "preregistration must declare holdout_accessed: false"
        )

    config = load_oanda_alpha_lab_config(config_path)
    lineage = document.get("development_lineage", {})
    if lineage.get("alpha_lab_config_sha256") != config.semantic_sha256:
        raise ValidationPreregistrationError(
            "preregistration lineage config hash does not match the current repo state"
        )

    partition = document.get("validation_partition", {})
    if partition.get("start_utc") != _iso(VALIDATION_START) or partition.get(
        "end_exclusive_utc"
    ) != _iso(HOLDOUT_START):
        raise ValidationPreregistrationError(
            "preregistration validation partition does not match the frozen boundary"
        )

    fresh = build_preregistration(
        scorecard_path=scorecard_path,
        family_robustness_path=family_robustness_path,
        stage2_metadata_path=stage2_metadata_path,
        stage1_metadata_path=stage1_metadata_path,
        config_path=config_path,
    )
    if fresh["selected_representatives"] != document.get("selected_representatives"):
        raise ValidationPreregistrationError(
            "stored representatives do not match what the frozen selection "
            "algorithm currently reproduces from the pinned Stage-2 evidence"
        )


# ---------------------------------------------------------------------------
# Section 4/11: fail-closed VALIDATION-partition-scoped data loader.
# ---------------------------------------------------------------------------

_FORBIDDEN_PATH_SUBSTRINGS = (
    "development",
    "holdout",
    "final_holdout",
    "final-holdout",
    "finaltest",
    "final_test",
)


def _reject_forbidden_path(path: Path) -> None:
    lowered = str(path).lower()
    if any(token in lowered for token in _FORBIDDEN_PATH_SUBSTRINGS):
        raise AlphaLabValidationError(
            f"path is forbidden for VALIDATION access: {path}"
        )


def _ns(value: datetime) -> int:
    return int(value.timestamp() * 1_000_000_000)


def _load_validation_instrument_frame(
    *, instrument_id: str, root: Path, timeframe: Timeframe
) -> pd.DataFrame:
    """Deliberately separate from :func:`ftmoquant.research.alpha_lab.data.
    _load_instrument_frame`, which is hardwired to the DEVELOPMENT window by
    design -- the two must never be interchangeable."""

    bar_spec = _TIMEFRAME_BAR_SPEC[timeframe]
    catalog = ParquetDataCatalog(str(root / "catalog"))
    start_ns = _ns(VALIDATION_START)
    end_ns = _ns(HOLDOUT_START)

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
            raise AlphaLabValidationError(f"VALIDATION has no {bar_type} data")
        by_side[side] = selected

    shared_ts = sorted(set(by_side["BID"]) & set(by_side["ASK"]))
    if not shared_ts:
        raise AlphaLabValidationError(
            f"VALIDATION BID/ASK bars never overlap for {instrument_id} at {timeframe}"
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
    return pd.DataFrame(rows).set_index("timestamp").sort_index()


def _reject_out_of_partition_observations(dataset: AlphaLabDataset) -> None:
    index = dataset.close.index
    if index.empty:
        raise AlphaLabValidationError("VALIDATION dataset has no aligned observations")
    if index.min() < VALIDATION_START or index.max() >= HOLDOUT_START:
        raise AlphaLabValidationError(
            "an observation falls outside the frozen VALIDATION partition"
        )


def _verify_validation_readiness(
    document: Mapping[str, Any], *, config_path: Path
) -> Any:
    """Fail closed unless ``document`` is the dedicated VALIDATION readiness
    artifact (never the DEVELOPMENT one, which uses a different
    ``readiness_version`` and lacks a ``partition``/``validation_start_utc``
    field entirely) for the current VALIDATION lineage config, with an
    intact self-hash. Returns the loaded :class:`OandaAlphaLabValidationConfig`."""

    working = dict(document)
    stored_hash = working.pop("semantic_sha256", None)
    if stored_hash is None or _canonical_sha256(working) != stored_hash:
        raise AlphaLabValidationError(
            "VALIDATION readiness content does not match its own self-hash"
        )
    if document.get("readiness_version") != VALIDATION_READINESS_VERSION:
        raise AlphaLabValidationError(
            "readiness document is not the VALIDATION readiness artifact "
            "(wrong partition -- DEVELOPMENT readiness cannot authorize "
            "VALIDATION access)"
        )
    if document.get("partition") != "VALIDATION":
        raise AlphaLabValidationError(
            "readiness document does not declare partition=VALIDATION"
        )
    if (
        document.get("holdout_accessed") is not False
        or document.get("holdout_rows_admitted") != 0
    ):
        raise AlphaLabValidationError("OANDA alpha-lab readiness document is invalid")

    config = load_oanda_alpha_lab_validation_config(config_path)
    if (
        document.get("alpha_lab_config_sha256") != config.semantic_sha256
        or document.get("validation_start_utc") != _iso(VALIDATION_START)
        or document.get("validation_end_exclusive_utc") != _iso(HOLDOUT_START)
    ):
        raise AlphaLabValidationError(
            "readiness lineage/config identity or partition boundary does "
            "not match the frozen VALIDATION lineage"
        )
    return config


def load_validation_dataset(
    *,
    validation_root: Path,
    universe_readiness_path: Path,
    timeframe: Timeframe,
    config_path: Path = OANDA_ALPHA_LAB_VALIDATION_CONFIG_PATH,
) -> AlphaLabDataset:
    """Load aligned VALIDATION matrices for every research-ready OANDA
    alpha-lab instrument at ``timeframe``. Fails closed on: a DEVELOPMENT or
    HOLDOUT-looking root path, a readiness document that is not the
    dedicated VALIDATION readiness artifact (partition, semantic hash,
    lineage config identity, and exact boundary all checked), a catalog
    whose tree hash has drifted from what readiness recorded, fewer than
    all seven preregistered instruments, or any observation outside
    ``[VALIDATION_START, HOLDOUT_START)``. Never opens any path other than
    ``validation_root``."""

    _reject_forbidden_path(validation_root)
    if timeframe not in _TIMEFRAME_BAR_SPEC:
        raise AlphaLabValidationError(f"unsupported timeframe: {timeframe}")

    try:
        document = json.loads(universe_readiness_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AlphaLabValidationError(
            f"could not read OANDA alpha-lab VALIDATION readiness: {error}"
        ) from error
    if not isinstance(document, dict):
        raise AlphaLabValidationError("OANDA alpha-lab readiness document is invalid")

    config = _verify_validation_readiness(document, config_path=config_path)

    statuses = document.get("per_instrument_status")
    artifacts = document.get("instrument_artifacts")
    if not isinstance(statuses, dict) or not isinstance(artifacts, list):
        raise AlphaLabValidationError("OANDA alpha-lab readiness document is malformed")
    instrument_ids = tuple(
        sorted(
            instrument_id
            for instrument_id, status in statuses.items()
            if status == "research_ready"
        )
    )
    if len(instrument_ids) != 7:
        raise AlphaLabValidationError(
            "VALIDATION requires exactly the seven preregistered "
            f"research-ready instruments, found {len(instrument_ids)}"
        )
    artifact_by_id = {
        item["instrument_id"]: item for item in artifacts if isinstance(item, dict)
    }

    per_instrument: dict[str, pd.DataFrame] = {}
    for instrument_id in instrument_ids:
        dataset_symbol = config.instrument(instrument_id).dataset_symbol
        root = validation_root / oanda_symbol(dataset_symbol)
        _reject_forbidden_path(root)
        artifact = artifact_by_id.get(instrument_id, {})
        expected_tree_sha = artifact.get("catalog_tree_sha256")
        actual_tree_sha = _sha256_tree(root / "catalog")
        if not expected_tree_sha or actual_tree_sha != expected_tree_sha:
            raise AlphaLabValidationError(
                f"VALIDATION catalog tree hash drifted: {instrument_id}"
            )
        per_instrument[instrument_id] = _load_validation_instrument_frame(
            instrument_id=instrument_id, root=root, timeframe=timeframe
        )

    dataset = assemble_aligned_dataset(
        per_instrument=per_instrument,
        instrument_ids=instrument_ids,
        timeframe=timeframe,
    )
    # assemble_aligned_dataset unconditionally stamps DEVELOPMENT bounds;
    # correct them to the true VALIDATION partition the bars were read from.
    dataset = replace(
        dataset, start_utc=VALIDATION_START, end_exclusive_utc=HOLDOUT_START
    )
    _reject_out_of_partition_observations(dataset)
    return dataset


# ---------------------------------------------------------------------------
# Section 6/7: frozen validation gates, concentration guard, and the
# reporting-only temporal subperiod diagnostic.
# ---------------------------------------------------------------------------


def compute_concentration_share(pair_net_profits: Mapping[str, float]) -> float | None:
    """Gate E. Net profit is approximated by each pair's base-cost
    ``cumulative_return`` under the equal-initial-capital sleeve convention
    :func:`~ftmoquant.research.alpha_lab.screening_common._build_aggregate_portfolio`
    already uses (every instrument gets equal ``init_cash``, so relative
    return equals relative profit share). Returns ``None`` (gate fails) if
    total positive pair-level profit is <= 0, per the frozen rule."""

    positive = {pair: profit for pair, profit in pair_net_profits.items() if profit > 0}
    total_positive = sum(positive.values())
    if total_positive <= 0:
        return None
    return max(positive.values()) / total_positive


def _validation_subperiod_windows() -> tuple[tuple[datetime, datetime], ...]:
    span = HOLDOUT_START - VALIDATION_START
    step = span / VALIDATION_SUBPERIOD_COUNT
    bounds = [
        VALIDATION_START + step * i for i in range(VALIDATION_SUBPERIOD_COUNT + 1)
    ]
    return tuple(
        (bounds[i], bounds[i + 1]) for i in range(VALIDATION_SUBPERIOD_COUNT)
    )


def _subperiod_diagnostics(
    returns: pd.Series,
    timeframe: Timeframe,
    windows: Sequence[tuple[datetime, datetime]],
) -> tuple[int, float, float]:
    """Reporting-only, not a gate. Mirrors ``screening_common._fold_diagnostics``
    (a slice of an already-computed return series), duplicated in miniature
    rather than reused directly because that function is hardwired to the
    frozen Stage-1/2 DEVELOPMENT ``FOLD_WINDOWS`` constant and must stay that
    way. The annualization-factor table itself is reused, not redefined."""

    ann_factor = _ANNUALIZATION_FACTOR[timeframe]
    period_returns: list[float] = []
    period_sharpes: list[float] = []
    for start, end in windows:
        window = returns.loc[(returns.index >= start) & (returns.index < end)]
        if window.empty:
            period_returns.append(0.0)
            period_sharpes.append(0.0)
            continue
        cumulative = float((1.0 + window).prod() - 1.0)
        std = float(window.std(ddof=1))
        mean = float(window.mean())
        sharpe = (mean / std) * float(np.sqrt(ann_factor)) if std > 0 else 0.0
        period_returns.append(cumulative)
        period_sharpes.append(sharpe)
    positive_count = sum(1 for r in period_returns if r > 0)
    worst = min(period_returns)
    median_sharpe = float(np.median(period_sharpes))
    return positive_count, worst, median_sharpe


@dataclass(frozen=True, slots=True)
class ValidationScorecardRow:
    family: str
    strategy_id: str
    instrument: str
    timeframe: str
    parameters: str
    observation_count: int
    trade_count: int
    cumulative_return: float
    annualized_return: float
    annualized_sharpe: float
    maximum_drawdown: float
    win_rate: float
    turnover_or_trade_activity: float
    base_cost_return: float
    stressed_cost_return: float
    profitable_pair_count: int | None
    positive_sharpe_pair_count: int | None
    median_pair_sharpe: float | None
    worst_pair_sharpe: float | None
    aggregate_equal_weight_return: float | None
    aggregate_equal_weight_sharpe: float | None
    aggregate_stressed_return: float | None
    concentration_share: float | None
    positive_subperiod_count: int | None
    worst_subperiod_return: float | None
    median_subperiod_sharpe: float | None


@dataclass(frozen=True, slots=True)
class FamilyValidationResult:
    family: str
    strategy_id: str
    representative_parameters: str
    profitable_pair_count: int
    aggregate_equal_weight_return: float
    aggregate_stressed_return: float
    aggregate_equal_weight_sharpe: float
    concentration_share: float | None
    cross_pair_gate: bool
    aggregate_return_gate: bool
    stressed_return_gate: bool
    sharpe_sign_gate: bool
    concentration_gate: bool
    validation_passed: bool


def _scorecard_row_from_screening_row(
    row: ScreeningResultRow,
    *,
    concentration_share: float | None,
    subperiod_stats: tuple[int, float, float] | None,
) -> ValidationScorecardRow:
    is_aggregate = row.instrument == AGGREGATE_INSTRUMENT_LABEL

    def only_if_aggregate(value: Any) -> Any:
        return value if is_aggregate else None

    positive_subperiod_count, worst_subperiod_return, median_subperiod_sharpe = (
        only_if_aggregate(subperiod_stats) or (None, None, None)
    )
    return ValidationScorecardRow(
        family=row.family,
        strategy_id=row.strategy_id,
        instrument=row.instrument,
        timeframe=row.timeframe,
        parameters=row.parameters,
        observation_count=row.observation_count,
        trade_count=row.trade_count,
        cumulative_return=row.cumulative_return,
        annualized_return=row.annualized_return,
        annualized_sharpe=row.annualized_sharpe,
        maximum_drawdown=row.maximum_drawdown,
        win_rate=row.win_rate,
        turnover_or_trade_activity=row.turnover_or_trade_activity,
        base_cost_return=row.base_cost_return,
        stressed_cost_return=row.stressed_cost_return,
        profitable_pair_count=only_if_aggregate(row.profitable_pair_count),
        positive_sharpe_pair_count=only_if_aggregate(row.positive_sharpe_pair_count),
        median_pair_sharpe=only_if_aggregate(row.median_pair_sharpe),
        worst_pair_sharpe=only_if_aggregate(row.worst_pair_sharpe),
        aggregate_equal_weight_return=only_if_aggregate(row.aggregate_equal_weight_return),
        aggregate_equal_weight_sharpe=only_if_aggregate(row.aggregate_equal_weight_sharpe),
        aggregate_stressed_return=only_if_aggregate(row.stressed_cost_return),
        concentration_share=only_if_aggregate(concentration_share),
        positive_subperiod_count=positive_subperiod_count,
        worst_subperiod_return=worst_subperiod_return,
        median_subperiod_sharpe=median_subperiod_sharpe,
    )


def _evaluate_family(rows: list[ValidationScorecardRow]) -> FamilyValidationResult:
    aggregate = next(
        row for row in rows if row.instrument == AGGREGATE_INSTRUMENT_LABEL
    )
    cross_pair_gate = (aggregate.profitable_pair_count or 0) >= MIN_PAIR_GATE
    aggregate_return_gate = (
        aggregate.aggregate_equal_weight_return is not None
        and aggregate.aggregate_equal_weight_return > 0
    )
    stressed_return = aggregate.aggregate_stressed_return
    stressed_return_gate = stressed_return is not None and stressed_return > 0
    sharpe_sign_gate = (
        aggregate.aggregate_equal_weight_sharpe is not None
        and aggregate.aggregate_equal_weight_sharpe > 0
    )
    concentration_gate = (
        aggregate.concentration_share is not None
        and aggregate.concentration_share <= CONCENTRATION_MAX_SHARE
    )
    validation_passed = all(
        (
            cross_pair_gate,
            aggregate_return_gate,
            stressed_return_gate,
            sharpe_sign_gate,
            concentration_gate,
        )
    )
    return FamilyValidationResult(
        family=aggregate.family,
        strategy_id=aggregate.strategy_id,
        representative_parameters=aggregate.parameters,
        profitable_pair_count=aggregate.profitable_pair_count or 0,
        aggregate_equal_weight_return=aggregate.aggregate_equal_weight_return or 0.0,
        aggregate_stressed_return=aggregate.aggregate_stressed_return or 0.0,
        aggregate_equal_weight_sharpe=aggregate.aggregate_equal_weight_sharpe or 0.0,
        concentration_share=aggregate.concentration_share,
        cross_pair_gate=cross_pair_gate,
        aggregate_return_gate=aggregate_return_gate,
        stressed_return_gate=stressed_return_gate,
        sharpe_sign_gate=sharpe_sign_gate,
        concentration_gate=concentration_gate,
        validation_passed=validation_passed,
    )


# ---------------------------------------------------------------------------
# Section 3/9: exactly 3 one-shot representative runs, reusing
# run_family_grid (screening_common) and _signals_for_config (screening_batch)
# verbatim -- no second backtester, no duplicated signal mechanics.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ValidationBatchResult:
    rows: tuple[ValidationScorecardRow, ...]
    family_results: tuple[FamilyValidationResult, ...]
    metadata: dict[str, object]


def run_validation(
    *,
    preregistration: Mapping[str, Any],
    datasets_by_timeframe: Mapping[Timeframe, AlphaLabDataset],
) -> ValidationBatchResult:
    """Evaluate exactly the 3 preregistered representatives. Does not verify
    the preregistration itself (callers must call :func:`verify_preregistration`
    first) and does not load data (callers pass already-loaded, already
    partition-checked datasets) -- kept separate so each concern is
    independently testable."""

    representatives = preregistration["selected_representatives"]
    if len(representatives) != 3:
        raise ValidationPreregistrationError(
            "preregistration must select exactly 3 representatives"
        )

    all_rows: list[ValidationScorecardRow] = []
    family_results: list[FamilyValidationResult] = []
    for family, params in representatives.items():
        timeframe = cast(Timeframe, params["timeframe"])
        dataset = datasets_by_timeframe[timeframe]
        strategy_id = str(params["strategy_id"])
        excluded_keys = ("strategy_id", "timeframe")
        parameters = {k: v for k, v in params.items() if k not in excluded_keys}
        config = GridConfig(
            family=family,
            strategy_id=strategy_id,
            timeframe=timeframe,
            parameters=parameters,
        )
        entries, exits, short_entries, short_exits = _signals_for_config(
            config, dataset
        )
        base_rows = run_family_grid(
            dataset=dataset,
            family=family,
            strategy_id=strategy_id,
            parameters=parameters,
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
        )

        pair_profits = {
            row.instrument: row.cumulative_return
            for row in base_rows
            if row.instrument != AGGREGATE_INSTRUMENT_LABEL
        }
        concentration_share = compute_concentration_share(pair_profits)

        cost = screening_cost_per_instrument(dataset)
        per_column_fees = cost.reindex(dataset.close.columns).to_numpy().reshape(1, -1)
        aggregate_portfolio = _build_aggregate_portfolio(
            dataset.close,
            entries,
            exits,
            short_entries,
            short_exits,
            per_column_fees=per_column_fees,
            freq=_TIMEFRAME_FREQ[timeframe],
        )
        subperiod_stats = _subperiod_diagnostics(
            aggregate_portfolio.returns(), timeframe, _validation_subperiod_windows()
        )

        family_rows = [
            _scorecard_row_from_screening_row(
                row,
                concentration_share=concentration_share,
                subperiod_stats=subperiod_stats,
            )
            for row in base_rows
        ]
        all_rows.extend(family_rows)
        family_results.append(_evaluate_family(family_rows))

    metadata: dict[str, object] = {
        "representative_count": len(representatives),
        "validation_partition": {
            "start_utc": _iso(VALIDATION_START),
            "end_exclusive_utc": _iso(HOLDOUT_START),
        },
        "cost_model": SCREENING_COST_LABEL,
        "stress_multiplier": STRESSED_COST_MULTIPLIER,
        "failure_policy": FAILURE_POLICY_STATEMENT,
        "holdout_accessed": False,
    }
    return ValidationBatchResult(
        rows=tuple(all_rows), family_results=tuple(family_results), metadata=metadata
    )


def write_validation_results(
    result: ValidationBatchResult, preregistration: Mapping[str, Any], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)

    with (output_dir / "preregistration.json").open("w", encoding="utf-8") as stream:
        json.dump(dict(preregistration), stream, sort_keys=True, indent=2)
        stream.write("\n")

    scorecard_fieldnames = list(asdict(result.rows[0]).keys())
    with (output_dir / "validation_scorecard.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=scorecard_fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        for row in result.rows:
            record = asdict(row)
            for key, value in record.items():
                if value is None:
                    record[key] = ""
            writer.writerow(record)

    summary_fieldnames = list(asdict(result.family_results[0]).keys())
    with (output_dir / "family_validation_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=summary_fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        for family_result in result.family_results:
            record = asdict(family_result)
            for key, value in record.items():
                if value is None:
                    record[key] = ""
            writer.writerow(record)

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(result.metadata, stream, sort_keys=True, indent=2)
        stream.write("\n")


# ---------------------------------------------------------------------------
# CLIs: preregistration generation never touches validation data; the run
# CLI verifies the preregistration before opening any VALIDATION path.
# ---------------------------------------------------------------------------


def build_preregister_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and freeze the one-shot VALIDATION preregistration for "
            "the three Stage-2 survivor families. Reads only Stage-1/Stage-2 "
            "DEVELOPMENT artifacts -- never opens validation data."
        )
    )
    parser.add_argument("--scorecard", type=Path, default=STAGE2_SCORECARD_PATH)
    parser.add_argument(
        "--family-robustness", type=Path, default=STAGE2_FAMILY_ROBUSTNESS_PATH
    )
    parser.add_argument("--stage2-metadata", type=Path, default=STAGE2_METADATA_PATH)
    parser.add_argument("--stage1-metadata", type=Path, default=STAGE1_METADATA_PATH)
    parser.add_argument("--output", type=Path, default=PREREGISTRATION_PATH)
    return parser


def preregister_main(argv: list[str] | None = None) -> None:
    parser = build_preregister_parser()
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValidationPreregistrationError(
            f"refusing to overwrite an existing frozen preregistration: {args.output}"
        )
    document = build_preregistration(
        scorecard_path=args.scorecard,
        family_robustness_path=args.family_robustness,
        stage2_metadata_path=args.stage2_metadata,
        stage1_metadata_path=args.stage1_metadata,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps(document["selected_representatives"], sort_keys=True, indent=2))


def build_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the one-shot VALIDATION stage for the three preregistered "
            "representatives against the frozen VALIDATION partition."
        )
    )
    parser.add_argument("--preregistration", type=Path, default=PREREGISTRATION_PATH)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument(
        "--universe-readiness",
        type=Path,
        default=DEFAULT_VALIDATION_READINESS_PATH,
        help="the dedicated VALIDATION readiness artifact -- never the DEVELOPMENT one",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_run_parser()
    args = parser.parse_args(argv)

    preregistration = load_preregistration(args.preregistration)
    verify_preregistration(preregistration)

    representatives = preregistration["selected_representatives"]
    needed_timeframes: list[Timeframe] = sorted(
        {cast(Timeframe, params["timeframe"]) for params in representatives.values()}
    )
    datasets_by_timeframe: dict[Timeframe, AlphaLabDataset] = {
        timeframe: load_validation_dataset(
            validation_root=args.validation_root,
            universe_readiness_path=args.universe_readiness,
            timeframe=timeframe,
        )
        for timeframe in needed_timeframes
    }

    result = run_validation(
        preregistration=preregistration, datasets_by_timeframe=datasets_by_timeframe
    )
    write_validation_results(result, preregistration, args.output)
    print(
        json.dumps(
            [asdict(family_result) for family_result in result.family_results],
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
