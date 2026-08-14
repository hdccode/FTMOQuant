from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from ftmoquant.backtest.execution_harness import (
    CalibrationStatus,
    ExecutionProfile,
    FeeModelConfig,
    FeeModelKind,
    RolloverConfig,
    RolloverMode,
)
from ftmoquant.research import stage_g
from ftmoquant.research.stage_g import (
    CurrencyExposureLimits,
    FxInstrumentCurrencies,
    FxPosition,
    InstrumentBarObservation,
    MissingBarPolicy,
    PerInstrumentCostModel,
    ResearchPartition,
    StageGValidationError,
    aggregate_currency_exposures,
    frozen_development_folds,
    liquidation_pnl_usd,
    load_frozen_universe,
    normalized_price_return,
    open_development_context,
    synchronize_universe_clock,
    validate_cost_models,
)

PLAN = Path("config/data/g1_4_fx_usd_liquid_v1.yaml")
INSTRUMENTS = ("EUR/USD.DUKASCOPY", "GBP/USD.DUKASCOPY")


def _hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_semantic(path: Path, payload: dict[str, Any]) -> str:
    semantic = _hash(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({**payload, "semantic_sha256": semantic}, sort_keys=True),
        encoding="utf-8",
    )
    return semantic


def _frozen_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, Path]]:
    roots: dict[str, Path] = {}
    split_hashes: dict[str, str] = {}
    for index, instrument_id in enumerate(INSTRUMENTS, start=1):
        symbol = instrument_id.replace("/", "").split(".", maxsplit=1)[0]
        root = tmp_path / "development" / symbol
        (root / "catalog").mkdir(parents=True)
        split_payload = {
            "split_view_version": "g1.4a-split-view-1",
            "universe_plan_sha256": stage_g.FROZEN_UNIVERSE_PLAN_SHA256,
            "instrument_id": instrument_id,
            "split": "development",
            "range": {
                "start_utc": "2019-03-11T00:00:00Z",
                "end_exclusive_utc": "2023-04-11T00:00:00Z",
            },
            "canonical_parent": {
                "file_sha256": str(index) * 64,
                "semantic_sha256": str(index + 2) * 64,
            },
            "catalog_tree_sha256": str(index + 4) * 64,
            "bar_counts": {
                f"{instrument_id}-1-MINUTE-BID-EXTERNAL": 10,
                f"{instrument_id}-1-MINUTE-ASK-EXTERNAL": 10,
            },
            "holdout_rows": 0,
            "network_access_required": False,
            "access_policy": "candidate_read_only",
        }
        split_hashes[instrument_id] = _write_semantic(
            root / "ftmoquant_split_view.json", split_payload
        )
        roots[instrument_id] = root

    artifacts = []
    for index, instrument_id in enumerate(INSTRUMENTS, start=1):
        artifacts.append(
            {
                "instrument_id": instrument_id,
                "canonical_sha256": "a" * 64,
                "derived_sha256": "b" * 64,
                "coverage_sha256": "c" * 64,
                "reconciliation_sha256": "d" * 64,
                "correction_sha256": None,
                "readiness_sha256": "e" * 64,
                "catalog_tree_sha256": str(index + 6) * 64,
                "development_split_sha256": split_hashes[instrument_id],
                "validation_split_sha256": "f" * 64,
            }
        )
    readiness_payload = {
        "readiness_version": "g1.4a-universe-readiness-1",
        "universe_id": stage_g.FROZEN_UNIVERSE_ID,
        "universe_plan_sha256": stage_g.FROZEN_UNIVERSE_PLAN_SHA256,
        "ordered_instrument_ids": list(INSTRUMENTS),
        "exact_currency_set": ["EUR", "GBP", "USD"],
        "permitted_utc_interval": {
            "start_utc": "2019-03-11T00:00:00Z",
            "end_exclusive_utc": "2024-08-21T00:00:00Z",
        },
        "instrument_artifacts": artifacts,
        "per_instrument_status": dict.fromkeys(INSTRUMENTS, "research_ready"),
        "cross_sectional_factor_status": "not_eligible_insufficient_universe",
        "minimum_cross_sectional_instruments": 4,
        "research_ready": True,
        "holdout_accessed": False,
        "holdout_rows_admitted": 0,
        "strategy_return_accessed": False,
    }
    readiness_path = tmp_path / "universe" / "ftmoquant_universe_readiness.json"
    readiness_hash = _write_semantic(readiness_path, readiness_payload)
    monkeypatch.setattr(stage_g, "FROZEN_UNIVERSE_READINESS_SHA256", readiness_hash)
    return readiness_path, roots


def test_exact_frozen_universe_and_development_range_are_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    readiness, roots = _frozen_fixture(tmp_path, monkeypatch)

    universe = load_frozen_universe(readiness, PLAN)
    context = open_development_context(readiness, roots, PLAN)

    assert universe.ordered_instrument_ids == INSTRUMENTS
    assert context.require_range(
        stage_g.DEVELOPMENT_START, stage_g.DEVELOPMENT_END_EXCLUSIVE
    ) == (stage_g.DEVELOPMENT_START, stage_g.DEVELOPMENT_END_EXCLUSIVE)
    assert context.synchronization_metadata_ready is True


def test_wrong_frozen_hash_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    readiness, _ = _frozen_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(stage_g, "FROZEN_UNIVERSE_READINESS_SHA256", "0" * 64)

    with pytest.raises(StageGValidationError, match="not the frozen hash"):
        load_frozen_universe(readiness, PLAN)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_missing_or_extra_instrument_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    readiness, roots = _frozen_fixture(tmp_path, monkeypatch)
    changed = dict(roots)
    if mutation == "missing":
        changed.pop(INSTRUMENTS[-1])
    else:
        changed["AUD/USD.DUKASCOPY"] = tmp_path / "extra"

    with pytest.raises(StageGValidationError, match="missing/extra"):
        open_development_context(readiness, changed, PLAN)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_readiness_manifest_missing_or_extra_instrument_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    readiness, _ = _frozen_fixture(tmp_path, monkeypatch)
    document = json.loads(readiness.read_text(encoding="utf-8"))
    document.pop("semantic_sha256")
    artifacts = document["instrument_artifacts"]
    if mutation == "missing":
        artifacts.pop()
    else:
        extra = dict(artifacts[-1])
        extra["instrument_id"] = "AUD/USD.DUKASCOPY"
        artifacts.append(extra)
    semantic = _write_semantic(readiness, document)
    monkeypatch.setattr(stage_g, "FROZEN_UNIVERSE_READINESS_SHA256", semantic)

    with pytest.raises(StageGValidationError, match="instrument artifact"):
        load_frozen_universe(readiness, PLAN)


def test_validation_and_holdout_are_rejected_from_development_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    readiness, roots = _frozen_fixture(tmp_path, monkeypatch)
    context = open_development_context(readiness, roots, PLAN)

    with pytest.raises(StageGValidationError, match="validation is LOCKED"):
        context.require_range(
            stage_g.VALIDATION_START,
            stage_g.VALIDATION_START + timedelta(days=1),
            partition=ResearchPartition.VALIDATION,
        )
    with pytest.raises(StageGValidationError, match="holdout is LOCKED"):
        context.require_range(
            stage_g.HOLDOUT_START,
            stage_g.HOLDOUT_START + timedelta(days=1),
            partition=ResearchPartition.HOLDOUT,
        )
    with pytest.raises(StageGValidationError, match="locked holdout"):
        context.require_range(
            stage_g.HOLDOUT_START, stage_g.HOLDOUT_START + timedelta(days=1)
        )


def _bar(instrument_id: str, minute: int) -> InstrumentBarObservation:
    timestamp = datetime(2021, 1, 4, 0, minute, tzinfo=UTC)
    return InstrumentBarObservation(
        instrument_id=instrument_id,
        timestamp_utc=timestamp,
        available_at_utc=timestamp + timedelta(minutes=1),
        bid=Decimal("1.2000"),
        ask=Decimal("1.2002"),
    )


def test_synchronized_clock_is_deterministic_and_never_synthesizes_prices() -> None:
    streams = {
        INSTRUMENTS[0]: (_bar(INSTRUMENTS[0], 0), _bar(INSTRUMENTS[0], 1)),
        INSTRUMENTS[1]: (_bar(INSTRUMENTS[1], 1),),
    }

    gaps = synchronize_universe_clock(
        streams, missing_policy=MissingBarPolicy.EMIT_NONTRADABLE_GAP
    )
    dropped = synchronize_universe_clock(
        streams, missing_policy=MissingBarPolicy.DROP_INCOMPLETE_TIMESTAMP
    )

    assert gaps == synchronize_universe_clock(
        streams, missing_policy=MissingBarPolicy.EMIT_NONTRADABLE_GAP
    )
    assert gaps[0].tradable is False
    assert gaps[0].observations == (streams[INSTRUMENTS[0]][0], None)
    assert dropped == (gaps[1],)
    assert all(item is not None for item in dropped[0].observations)


def test_currency_exposure_aggregation_limits_and_usd_liquidation() -> None:
    exposures = aggregate_currency_exposures(
        (
            FxPosition(INSTRUMENTS[0], Decimal("100"), Decimal("1.10")),
            FxPosition(INSTRUMENTS[1], Decimal("-50"), Decimal("1.30")),
        )
    )

    assert [(item.currency, item.amount) for item in exposures] == [
        ("EUR", Decimal("100")),
        ("GBP", Decimal("-50")),
        ("USD", Decimal("-45.00")),
    ]
    limits = CurrencyExposureLimits(
        {"EUR": Decimal("99"), "GBP": Decimal("60"), "USD": Decimal("50")}
    )
    assert limits.breaches(exposures) == (exposures[0],)
    assert liquidation_pnl_usd(
        INSTRUMENTS[0], Decimal("100000"), Decimal("1.10"), Decimal("1.11")
    ) == Decimal("1000.00")
    normalized = normalized_price_return(
        INSTRUMENTS[1], Decimal("1.20"), Decimal("1.26")
    )
    assert normalized.quote_currency == "USD"
    assert normalized.price_return == Decimal("0.05")

    future_pair = FxInstrumentCurrencies("EUR/GBP.FUTURE", "EUR", "GBP")
    assert liquidation_pnl_usd(
        "EUR/GBP.FUTURE",
        Decimal("100"),
        Decimal("0.80"),
        Decimal("0.81"),
        quote_to_usd=Decimal("1.25"),
        currencies=future_pair,
    ) == Decimal("1.2500")
    with pytest.raises(StageGValidationError, match="PIT USD rate"):
        liquidation_pnl_usd(
            "EUR/GBP.FUTURE",
            Decimal("100"),
            Decimal("0.80"),
            Decimal("0.81"),
            currencies=future_pair,
        )


def _profile(seed: int) -> ExecutionProfile:
    return ExecutionProfile(
        fill_on_limit_probability=Decimal(1),
        adverse_slippage_probability=Decimal(0),
        random_seed=seed,
        base_latency_ns=0,
        insert_latency_ns=0,
        update_latency_ns=0,
        cancel_latency_ns=0,
        fee=FeeModelConfig(
            kind=FeeModelKind.FIXED,
            commission=Decimal(0),
            currency="USD",
            charge_commission_once=False,
        ),
        rollover=RolloverConfig(mode=RolloverMode.DISABLED),
        adaptive_high_low_ordering=True,
        calibration_status=CalibrationStatus.UNCALIBRATED,
    )


def test_per_instrument_cost_configuration_reuses_g07_profiles() -> None:
    models = tuple(
        PerInstrumentCostModel(instrument_id, _profile(index))
        for index, instrument_id in enumerate(INSTRUMENTS)
    )

    validate_cost_models(models)

    wrong_spread = PerInstrumentCostModel(
        INSTRUMENTS[0], _profile(0), spread_source="fixed_bps"
    )
    with pytest.raises(StageGValidationError, match="observed paired"):
        wrong_spread.validate()


def test_development_fold_hash_is_deterministic_and_development_only() -> None:
    first = frozen_development_folds()
    second = frozen_development_folds()

    assert first == second
    assert len(first.semantic_sha256) == 64
    assert all(
        stage_g.DEVELOPMENT_START <= fold.train_start_utc
        and fold.compare_end_exclusive_utc <= stage_g.DEVELOPMENT_END_EXCLUSIVE
        for fold in first.folds
    )


def test_stage_g_exposes_no_candidate_return_accessor() -> None:
    assert not hasattr(stage_g, "load_strategy_returns")
    assert not hasattr(stage_g, "candidate_returns")
