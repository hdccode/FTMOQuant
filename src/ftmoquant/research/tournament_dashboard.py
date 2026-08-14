"""Read-only Stage G view model for the marimo research UI."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from ftmoquant.research.stage_g import (
    DEFAULT_DEVELOPMENT_ROOTS,
    DEFAULT_UNIVERSE_READINESS_PATH,
    DEVELOPMENT_END_EXCLUSIVE,
    DEVELOPMENT_START,
    FROZEN_INSTRUMENT_IDS,
    FROZEN_UNIVERSE_ID,
    FROZEN_UNIVERSE_PLAN_SHA256,
    FROZEN_UNIVERSE_READINESS_SHA256,
    StageGValidationError,
    frozen_development_folds,
    open_development_context,
)
from ftmoquant.research.tournament_registry import (
    candidate_registry,
    preregistered_selection_contract,
)


def build_dashboard_snapshot(
    readiness_path: Path = DEFAULT_UNIVERSE_READINESS_PATH,
) -> dict[str, Any]:
    """Build a DEVELOPMENT-only view; fixed roots prevent split/path switching."""

    registry = candidate_registry()
    folds = frozen_development_folds()
    selection = preregistered_selection_contract()
    base: dict[str, Any] = {
        "universe_id": FROZEN_UNIVERSE_ID,
        "universe_readiness_sha256": FROZEN_UNIVERSE_READINESS_SHA256,
        "universe_plan_sha256": FROZEN_UNIVERSE_PLAN_SHA256,
        "ordered_instruments": FROZEN_INSTRUMENT_IDS,
        "development_interval": (
            DEVELOPMENT_START.isoformat(),
            DEVELOPMENT_END_EXCLUSIVE.isoformat(),
        ),
        "currency_metadata": (
            {"instrument_id": "EUR/USD.DUKASCOPY", "base": "EUR", "quote": "USD"},
            {"instrument_id": "GBP/USD.DUKASCOPY", "base": "GBP", "quote": "USD"},
        ),
        "registry": tuple(asdict(item) for item in registry.ordered_entries),
        "registry_sha256": registry.semantic_sha256,
        "folds_sha256": folds.semantic_sha256,
        "selection_contract_sha256": selection.semantic_sha256,
        "validation": "LOCKED / unavailable",
        "holdout": "LOCKED / unavailable",
        "strategy_returns": "NOT ACCESSED / unavailable",
    }
    try:
        context = open_development_context(readiness_path, DEFAULT_DEVELOPMENT_ROOTS)
    except StageGValidationError as error:
        return {
            **base,
            "frozen_readiness": f"UNAVAILABLE (fail closed): {error}",
            "development_artifacts": (),
            "synchronization": "UNAVAILABLE until exact DEVELOPMENT artifacts mount",
        }
    return {
        **base,
        "frozen_readiness": "VERIFIED",
        "development_artifacts": tuple(asdict(item) for item in context.artifacts),
        "synchronization": (
            "DEVELOPMENT metadata ready; runtime policy requires complete timestamps "
            "or explicit nontradable gaps, with no fills"
            if context.synchronization_metadata_ready
            else "DEVELOPMENT catalog directories unavailable"
        ),
    }
