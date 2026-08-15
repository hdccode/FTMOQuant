"""DEVELOPMENT-only evaluator for the frozen USD macro-surprise candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]

from ftmoquant.research.stage_g import (
    DEVELOPMENT_END_EXCLUSIVE,
    DEVELOPMENT_START,
    DevelopmentFold,
    InstrumentBarObservation,
    SynchronizedClockFrame,
    frozen_development_folds,
    open_development_context,
)
from ftmoquant.research.statistics import (
    StationaryBootstrapConfig,
    result_as_dict,
    stationary_bootstrap_confidence_interval,
)
from ftmoquant.research.ts_momentum_development import (
    _load_development_market_data_once,
    _sha256_file,
    load_development_evaluation_config,
)
from ftmoquant.research.usd_macro_surprise_momentum_spec import (
    USD_MACRO_SURPRISE_MOMENTUM_CONFIG_SHA256,
    load_usd_macro_surprise_momentum_spec,
)

EVALUATOR_VERSION = "g1.4g-usd-macro-surprise-momentum-development-1"
_INSTRUMENTS = ("EUR/USD.DUKASCOPY", "GBP/USD.DUKASCOPY")
_FAMILIES = ("US_NFP_HEADLINE_EMPLOYMENT_CHANGE", "US_CPI_HEADLINE_M_M")
_BOOTSTRAP = StationaryBootstrapConfig(
    block_size=1,
    repetitions=10_000,
    seed=14_042_026,
    confidence_level=0.95,
    method="basic",
)


class UsdMacroSurpriseMomentumEvaluationError(ValueError):
    """Raised when DEVELOPMENT-only macro evaluation cannot prove its contract."""


@dataclass(frozen=True, slots=True)
class MacroEvent:
    timestamp_utc: datetime
    event_family: str
    actual: Decimal
    forecast: Decimal
    source_row_number: int


@dataclass(frozen=True, slots=True)
class EventResult:
    timestamp_utc: str
    event_family: str
    fold_id: str
    direction: str
    base_net_event_return: float | None
    cost_stress_1_5x_event_return: float | None
    status: str
    reason: str | None


def evaluate_usd_macro_surprise_momentum_development(
    *,
    spec_path: Path,
    macro_events_path: Path,
    universe_readiness_path: Path,
    development_roots: Mapping[str, Path],
    cost_models_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Evaluate only frozen DEVELOPMENT data and persist event-level results."""
    if output_dir.exists():
        raise FileExistsError(f"evaluation output already exists: {output_dir}")
    spec = load_usd_macro_surprise_momentum_spec(spec_path)
    if spec.semantic_sha256 != USD_MACRO_SURPRISE_MOMENTUM_CONFIG_SHA256:
        raise UsdMacroSurpriseMomentumEvaluationError(
            "macro strategy semantic SHA drifted"
        )
    if set(development_roots) != set(_INSTRUMENTS):
        raise UsdMacroSurpriseMomentumEvaluationError(
            "exactly EUR/USD and GBP/USD DEVELOPMENT roots are required"
        )
    archive_sha = _verify_macro_provenance(macro_events_path)
    events = load_development_macro_events(macro_events_path)
    context = open_development_context(universe_readiness_path, development_roots)
    prepared = _load_development_market_data_once(context, development_roots)
    config = load_development_evaluation_config(cost_models_path)
    results = evaluate_macro_events(events, prepared.frames)
    manifest = _manifest(
        spec.semantic_sha256,
        archive_sha,
        context.universe.readiness_sha256,
        cost_models_path,
        config.semantic_sha256,
        results,
    )
    _write(output_dir, results, manifest)
    return manifest


def load_development_macro_events(path: Path) -> tuple[MacroEvent, ...]:
    events: list[MacroEvent] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise UsdMacroSurpriseMomentumEvaluationError(
                    "macro JSONL row must be an object"
                )
            family = raw.get("event_family")
            timestamp = _timestamp(raw.get("timestamp_utc"))
            if family not in _FAMILIES or not (
                DEVELOPMENT_START <= timestamp < DEVELOPMENT_END_EXCLUSIVE
            ):
                continue
            actual = _parsed_number(raw.get("actual_parsed"), "actual")
            forecast = _parsed_number(raw.get("forecast_parsed"), "forecast")
            events.append(
                MacroEvent(
                    timestamp,
                    cast(str, family),
                    actual,
                    forecast,
                    int(raw.get("source_row_number", line_number)),
                )
            )
    return tuple(
        sorted(
            events,
            key=lambda item: (
                item.timestamp_utc,
                item.event_family,
                item.source_row_number,
            ),
        )
    )


def evaluate_macro_events(
    events: Sequence[MacroEvent], frames: Sequence[SynchronizedClockFrame]
) -> tuple[EventResult, ...]:
    """Pure event-level execution using observed bid/ask prices only."""
    ordered = tuple(sorted(frames, key=lambda item: item.available_at_utc))
    active_until: dict[str, datetime] = {
        instrument: datetime.min.replace(tzinfo=UTC) for instrument in _INSTRUMENTS
    }
    results: list[EventResult] = []
    for event in events:
        surprise = event.actual - event.forecast
        fold = _fold_for(event.timestamp_utc)
        if surprise == 0:
            results.append(
                EventResult(
                    _utc(event.timestamp_utc),
                    event.event_family,
                    fold.fold_id,
                    "no_trade",
                    None,
                    None,
                    "no_trade",
                    "zero_surprise",
                )
            )
            continue
        direction = "USD_positive" if surprise > 0 else "USD_negative"
        entry = _first_at_or_after(ordered, event.timestamp_utc + timedelta(minutes=5))
        exit = _first_at_or_after(ordered, event.timestamp_utc + timedelta(minutes=60))
        if entry is None or exit is None:
            results.append(
                EventResult(
                    _utc(event.timestamp_utc),
                    event.event_family,
                    fold.fold_id,
                    direction,
                    None,
                    None,
                    "non_executable",
                    "missing_entry_or_exit_observation",
                )
            )
            continue
        if any(
            entry.available_at_utc < active_until[instrument]
            for instrument in _INSTRUMENTS
        ):
            results.append(
                EventResult(
                    _utc(event.timestamp_utc),
                    event.event_family,
                    fold.fold_id,
                    direction,
                    None,
                    None,
                    "non_executable",
                    "overlap_same_instrument",
                )
            )
            continue
        legs = [
            _leg_return(instrument, direction, entry, exit)
            for instrument in _INSTRUMENTS
        ]
        if any(leg is None for leg in legs):
            results.append(
                EventResult(
                    _utc(event.timestamp_utc),
                    event.event_family,
                    fold.fold_id,
                    direction,
                    None,
                    None,
                    "non_executable",
                    "both_pairs_required_missing_observation",
                )
            )
            continue
        base = sum(
            cast(tuple[tuple[Decimal, Decimal], ...], tuple(legs))[index][0]
            for index in range(2)
        ) / Decimal(2)
        stress = sum(
            cast(tuple[tuple[Decimal, Decimal], ...], tuple(legs))[index][1]
            for index in range(2)
        ) / Decimal(2)
        for instrument in _INSTRUMENTS:
            active_until[instrument] = exit.available_at_utc
        results.append(
            EventResult(
                _utc(event.timestamp_utc),
                event.event_family,
                fold.fold_id,
                direction,
                float(base),
                float(stress),
                "executable",
                None,
            )
        )
    return tuple(results)


def _leg_return(
    instrument: str,
    direction: str,
    entry: SynchronizedClockFrame,
    exit: SynchronizedClockFrame,
) -> tuple[Decimal, Decimal] | None:
    entry_observation = _observation(entry, instrument)
    exit_observation = _observation(exit, instrument)
    if entry_observation is None or exit_observation is None:
        return None
    if direction == "USD_positive":
        entry_price, exit_price = entry_observation.bid, exit_observation.ask
        midpoint_gross = (
            (entry_observation.bid + entry_observation.ask) / 2
            - (exit_observation.bid + exit_observation.ask) / 2
        ) / ((entry_observation.bid + entry_observation.ask) / 2)
    else:
        entry_price, exit_price = entry_observation.ask, exit_observation.bid
        midpoint_gross = (
            (exit_observation.bid + exit_observation.ask) / 2
            - (entry_observation.bid + entry_observation.ask) / 2
        ) / ((entry_observation.bid + entry_observation.ask) / 2)
    base = (
        (entry_price - exit_price) / entry_price
        if direction == "USD_positive"
        else (exit_price - entry_price) / entry_price
    )
    stress = midpoint_gross - Decimal("1.5") * (midpoint_gross - base)
    return base, stress


def _first_at_or_after(
    frames: Sequence[SynchronizedClockFrame], threshold: datetime
) -> SynchronizedClockFrame | None:
    return next(
        (
            frame
            for frame in frames
            if frame.tradable and frame.available_at_utc >= threshold
        ),
        None,
    )


def _observation(
    frame: SynchronizedClockFrame, instrument: str
) -> InstrumentBarObservation | None:
    return next(
        (
            item
            for item in frame.observations
            if item is not None and item.instrument_id == instrument
        ),
        None,
    )


def _fold_for(timestamp: datetime) -> DevelopmentFold:
    matches = [
        fold
        for fold in frozen_development_folds().folds
        if fold.compare_start_utc <= timestamp < fold.compare_end_exclusive_utc
    ]
    if len(matches) != 1:
        raise UsdMacroSurpriseMomentumEvaluationError(
            "event timestamp is not in exactly one DEVELOPMENT fold"
        )
    return matches[0]


def _verify_macro_provenance(path: Path) -> str:
    manifest_path = path.parent / "provenance_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source = manifest["source"]
        archive_sha = source["files"][0]["sha256"]
    except (OSError, KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
        raise UsdMacroSurpriseMomentumEvaluationError(
            "macro provenance manifest is invalid"
        ) from error
    expected = load_usd_macro_surprise_momentum_spec().canonical_document[
        "event_source"
    ]["zip_sha256"]
    if archive_sha != expected:
        raise UsdMacroSurpriseMomentumEvaluationError(
            "macro archive SHA is incompatible with frozen spec"
        )
    return cast(str, archive_sha)


def _parsed_number(value: Any, label: str) -> Decimal:
    if (
        not isinstance(value, dict)
        or value.get("status") != "parsed"
        or not isinstance(value.get("value"), str)
    ):
        raise UsdMacroSurpriseMomentumEvaluationError(f"event has no parsed {label}")
    return Decimal(value["value"])


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise UsdMacroSurpriseMomentumEvaluationError("event timestamp is invalid")
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _summary(results: Sequence[EventResult]) -> dict[str, Any]:
    executable = [item for item in results if item.status == "executable"]
    values = [cast(float, item.base_net_event_return) for item in executable]
    stressed = [cast(float, item.cost_stress_1_5x_event_return) for item in executable]
    folds = {
        fold.fold_id: [
            cast(float, item.base_net_event_return)
            for item in executable
            if item.fold_id == fold.fold_id
        ]
        for fold in frozen_development_folds().folds
    }
    fold_means = {
        key: (sum(value) / len(value) if value else None)
        for key, value in folds.items()
    }
    ci = (
        None
        if len(values) < 2
        else result_as_dict(
            stationary_bootstrap_confidence_interval(
                pd.Series(values, name="event_portfolio_return"), _BOOTSTRAP
            )
        )
    )
    pooled = sum(values) / len(values) if values else None
    stressed_mean = sum(stressed) / len(stressed) if stressed else None
    means = [value for value in fold_means.values() if value is not None]
    integrity_failure = not values or any(
        item.status == "non_executable" for item in results
    )
    passed = bool(
        pooled is not None
        and pooled > 0
        and sum(value > 0 for value in means) >= 2
        and len(means) == 3
        and statistics.median(means) > 0
        and stressed_mean is not None
        and stressed_mean > 0
        and not integrity_failure
    )
    return {
        "event_count_by_type_and_fold": _counts(results),
        "executable_event_count": len(values),
        "mean_net_event_return": pooled,
        "median_net_event_return": statistics.median(values) if values else None,
        "hit_rate": sum(value > 0 for value in values) / len(values)
        if values
        else None,
        "fold_mean_net_event_return": fold_means,
        "pooled_mean_net_event_return": pooled,
        "pooled_mean_cost_stress_1_5x": stressed_mean,
        "event_level_bootstrap_95_ci": ci,
        "worst_fold_mean": min(means) if means else None,
        "median_fold_mean": statistics.median(means) if means else None,
        "positive_fold_count": sum(value > 0 for value in means),
        "sequential_event_portfolio_maximum_drawdown": _drawdown(values),
        "decision": "PASS_DEVELOPMENT" if passed else "REJECT_RETIRE",
        "implementation_or_data_integrity_failure": integrity_failure,
    }


def _counts(results: Sequence[EventResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in results:
        key = f"{item.event_family}:{item.fold_id}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _drawdown(values: Sequence[float]) -> float:
    wealth = peak = 1.0
    maximum = 0.0
    for value in values:
        wealth *= 1 + value
        peak = max(peak, wealth)
        maximum = max(maximum, (peak - wealth) / peak)
    return maximum


def _manifest(
    strategy_sha: str,
    archive_sha: str,
    readiness_sha: str,
    costs: Path,
    costs_sha: str,
    results: Sequence[EventResult],
) -> dict[str, Any]:
    payload = {
        "schema": "ftmoquant.usd-macro-surprise-momentum-development-results",
        "evaluator_version": EVALUATOR_VERSION,
        "strategy_semantic_sha256": strategy_sha,
        "macro_archive_sha256": archive_sha,
        "universe_readiness_semantic_sha256": readiness_sha,
        "cost_model_file_sha256": _sha256_file(costs),
        "cost_model_semantic_sha256": costs_sha,
        "development_boundary": {
            "start": _utc(DEVELOPMENT_START),
            "end_exclusive": _utc(DEVELOPMENT_END_EXCLUSIVE),
        },
        "development_folds_semantic_sha256": frozen_development_folds().semantic_sha256,
        "parameter_optimization": False,
        "validation_accessed": False,
        "final_holdout_accessed": False,
        "summary": _summary(results),
    }
    payload["semantic_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def _write(
    output: Path, results: Sequence[EventResult], manifest: dict[str, Any]
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    (output / "event_results.jsonl").write_text(
        "".join(json.dumps(asdict(item), sort_keys=True) + "\n" for item in results),
        encoding="utf-8",
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _root(value: str) -> tuple[str, Path]:
    instrument, separator, path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("development root must be INSTRUMENT=PATH")
    return instrument, Path(path)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen macro candidate on DEVELOPMENT only"
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--macro-events", type=Path, required=True)
    parser.add_argument("--universe-readiness", type=Path, required=True)
    parser.add_argument(
        "--development-root", action="append", type=_root, required=True
    )
    parser.add_argument("--cost-models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    evaluate_usd_macro_surprise_momentum_development(
        spec_path=args.spec,
        macro_events_path=args.macro_events,
        universe_readiness_path=args.universe_readiness,
        development_roots=dict(args.development_root),
        cost_models_path=args.cost_models,
        output_dir=args.output,
    )
