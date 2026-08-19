from __future__ import annotations

import csv
import inspect
import json
from datetime import UTC, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ftmoquant.data.oanda_alpha_lab_validation import (
    VALIDATION_READINESS_VERSION,
    load_oanda_alpha_lab_validation_config,
)
from ftmoquant.research.alpha_lab.data import ALIGNMENT_POLICY, AlphaLabDataset
from ftmoquant.research.alpha_lab.screening_batch import build_grid
from ftmoquant.research.alpha_lab.screening_stage2 import (
    DONCHIAN_FAMILY,
    DONCHIAN_H4_LOOKBACKS,
    MEAN_REVERSION_FAMILY,
    MEAN_REVERSION_LOOKBACKS,
    MEAN_REVERSION_Z_ENTRIES,
    TSM_FAMILY,
    TSM_H4_LOOKBACKS,
    build_stage2_grid,
)
from ftmoquant.research.alpha_lab.validation import (
    CONCENTRATION_MAX_SHARE,
    PREREGISTRATION_VERSION,
    AlphaLabValidationError,
    ValidationPreregistrationError,
    _canonical_sha256,
    _evaluate_family,
    _iso,
    _reject_forbidden_path,
    _reject_out_of_partition_observations,
    _verify_validation_readiness,
    build_preregistration,
    compute_concentration_share,
    derive_representatives,
    run_validation,
    select_1d_representative,
    select_2d_representative,
    verify_preregistration,
    write_validation_results,
)
from ftmoquant.research.stage_g import HOLDOUT_START, VALIDATION_START

_VALIDATION_CONFIG_PATH = Path("config/data/oanda_fx_alpha_lab_validation_v1.yaml")

_PAIRS = (
    "AUD/USD.OANDA",
    "EUR/USD.OANDA",
    "GBP/USD.OANDA",
    "NZD/USD.OANDA",
    "USD/CAD.OANDA",
    "USD/CHF.OANDA",
    "USD/JPY.OANDA",
)


# ---------------------------------------------------------------------------
# Representative-selection algorithm (pure, synthetic).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lookbacks,passing,expected",
    [
        (DONCHIAN_H4_LOOKBACKS, {35, 40, 45, 50, 55}, 45),
        (TSM_H4_LOOKBACKS, {30, 40, 50, 60}, 40),
    ],
)
def test_select_1d_representative_reproduces_the_frozen_region(
    lookbacks: list[int], passing: set[int], expected: int
) -> None:
    assert select_1d_representative(lookbacks, passing) == expected


def test_select_2d_representative_reproduces_the_frozen_mean_reversion_region() -> None:
    passing = {
        (30, 2.0),
        (30, 2.5),
        (40, 1.5),
        (40, 2.0),
        (40, 2.5),
        (50, 1.5),
        (50, 2.0),
        (50, 2.5),
    }
    result = select_2d_representative(
        MEAN_REVERSION_LOOKBACKS, MEAN_REVERSION_Z_ENTRIES, passing
    )
    assert result == (40, 2.0)


def test_select_1d_representative_ties_break_to_the_smaller_value() -> None:
    # 4 evenly-tested values -> tie between the two middle-ranked (20, 30).
    assert select_1d_representative([10, 20, 30, 40], {10, 20, 30, 40}) == 20


def test_select_2d_representative_ties_break_lexicographically() -> None:
    # A symmetric 2x2 passing block: all four cells tie on cost: smaller
    # lookback wins first, then smaller z_entry.
    lookbacks = [10, 20, 30]
    zs = [1.0, 2.0, 3.0]
    passing = {(10, 1.0), (10, 2.0), (20, 1.0), (20, 2.0)}
    assert select_2d_representative(lookbacks, zs, passing) == (10, 1.0)


def test_select_1d_representative_depends_on_grid_index_not_magnitude() -> None:
    # Same relative rank structure (5 passing values), wildly different
    # absolute gaps -- the grid-INDEX median is rank position 2 of 5 either
    # way, unaffected by how far the outlier sits numerically.
    narrow = [1, 2, 3, 4, 5]
    wide = [1, 2, 3, 4, 5_000_000]
    assert select_1d_representative(narrow, {1, 2, 3, 4, 5}) == 3
    assert select_1d_representative(wide, {1, 2, 3, 4, 5_000_000}) == 3


def test_select_1d_representative_ignores_input_list_order() -> None:
    ordered = [35, 40, 45, 50, 55, 60, 70]
    shuffled = [70, 45, 35, 60, 50, 40, 55]
    passing = {35, 40, 45, 50, 55}
    assert select_1d_representative(ordered, passing) == select_1d_representative(
        shuffled, passing
    )


def test_select_representatives_reject_empty_passing_sets() -> None:
    with pytest.raises(ValueError):
        select_1d_representative(DONCHIAN_H4_LOOKBACKS, set())
    with pytest.raises(ValueError):
        select_2d_representative(
            MEAN_REVERSION_LOOKBACKS, MEAN_REVERSION_Z_ENTRIES, set()
        )


# ---------------------------------------------------------------------------
# Deriving representatives from a (synthetic) Stage-2 scorecard.
# ---------------------------------------------------------------------------


def _write_synthetic_stage2_scorecard(path: Path) -> None:
    fieldnames = [
        "family",
        "strategy_id",
        "instrument",
        "timeframe",
        "parameters",
        "stressed_cost_return",
        "profitable_pair_count",
        "positive_fold_count",
    ]
    donchian_passing = {35, 40, 45, 50, 55}
    tsm_passing = {30, 40, 50, 60}
    meanrev_passing = {
        (30, 2.0),
        (30, 2.5),
        (40, 1.5),
        (40, 2.0),
        (40, 2.5),
        (50, 1.5),
        (50, 2.0),
        (50, 2.5),
    }
    rows = []
    for lookback in DONCHIAN_H4_LOOKBACKS:
        passes = lookback in donchian_passing
        rows.append(
            {
                "family": DONCHIAN_FAMILY,
                "strategy_id": f"stage2_donchian_h4_lookback{lookback}_v1",
                "instrument": "AGGREGATE_EQUAL_WEIGHT",
                "timeframe": "H4",
                "parameters": json.dumps({"lookback": lookback}),
                "stressed_cost_return": 0.01 if passes else -0.01,
                "profitable_pair_count": 5 if passes else 1,
                "positive_fold_count": 3 if passes else 0,
            }
        )
    for lookback in TSM_H4_LOOKBACKS:
        passes = lookback in tsm_passing
        rows.append(
            {
                "family": TSM_FAMILY,
                "strategy_id": f"stage2_tsm_h4_lookback{lookback}_v1",
                "instrument": "AGGREGATE_EQUAL_WEIGHT",
                "timeframe": "H4",
                "parameters": json.dumps({"lookback": lookback}),
                "stressed_cost_return": 0.01 if passes else -0.01,
                "profitable_pair_count": 5 if passes else 1,
                "positive_fold_count": 3 if passes else 0,
            }
        )
    for lookback in MEAN_REVERSION_LOOKBACKS:
        for z_entry in MEAN_REVERSION_Z_ENTRIES:
            passes = (lookback, z_entry) in meanrev_passing
            rows.append(
                {
                    "family": MEAN_REVERSION_FAMILY,
                    "strategy_id": (
                        f"stage2_meanrev_h1_lookback{lookback}_z{z_entry}_v1"
                    ),
                    "instrument": "AGGREGATE_EQUAL_WEIGHT",
                    "timeframe": "H1",
                    "parameters": json.dumps(
                        {"lookback": lookback, "z_entry": z_entry}
                    ),
                    "stressed_cost_return": 0.01 if passes else -0.01,
                    "profitable_pair_count": 5 if passes else 1,
                    "positive_fold_count": 3 if passes else 0,
                }
            )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_derive_representatives_from_synthetic_stage2_scorecard_matches_frozen_answer(
    tmp_path: Path,
) -> None:
    scorecard = tmp_path / "scorecard.csv"
    _write_synthetic_stage2_scorecard(scorecard)
    representatives = derive_representatives(scorecard)
    assert representatives[DONCHIAN_FAMILY]["lookback"] == 45
    assert representatives[DONCHIAN_FAMILY]["timeframe"] == "H4"
    assert representatives[TSM_FAMILY]["lookback"] == 40
    assert representatives[TSM_FAMILY]["timeframe"] == "H4"
    assert representatives[MEAN_REVERSION_FAMILY]["lookback"] == 40
    assert representatives[MEAN_REVERSION_FAMILY]["z_entry"] == 2.0
    assert representatives[MEAN_REVERSION_FAMILY]["timeframe"] == "H1"
    assert len(representatives) == 3


def test_derive_representatives_is_deterministic(tmp_path: Path) -> None:
    scorecard = tmp_path / "scorecard.csv"
    _write_synthetic_stage2_scorecard(scorecard)
    first = derive_representatives(scorecard)
    second = derive_representatives(scorecard)
    assert first == second


# ---------------------------------------------------------------------------
# Preregistration: generation, self-consistency, and fail-closed rejection.
# ---------------------------------------------------------------------------


def _synthetic_dev_artifacts(tmp_path: Path) -> dict[str, Path]:
    scorecard = tmp_path / "scorecard.csv"
    _write_synthetic_stage2_scorecard(scorecard)
    family_robustness = tmp_path / "family_robustness.csv"
    family_robustness.write_text(
        "family,tested_configuration_count\nA,1\n", encoding="utf-8"
    )
    stage2_metadata = tmp_path / "stage2_metadata.json"
    stage2_metadata.write_text(
        json.dumps({"configuration_count": 23}), encoding="utf-8"
    )
    stage1_metadata = tmp_path / "stage1_metadata.json"
    stage1_metadata.write_text(
        json.dumps({"configuration_count": 49}), encoding="utf-8"
    )
    return {
        "scorecard_path": scorecard,
        "family_robustness_path": family_robustness,
        "stage2_metadata_path": stage2_metadata,
        "stage1_metadata_path": stage1_metadata,
    }


def test_preregistration_generation_has_no_validation_data_argument() -> None:
    # No parameter of build_preregistration names a validation path at all --
    # generation is structurally incapable of touching validation data.
    signature = inspect.signature(build_preregistration)
    for name in signature.parameters:
        assert "validation" not in name.lower()


def test_build_preregistration_uses_only_stage1_stage2_and_config_inputs(
    tmp_path: Path,
) -> None:
    paths = _synthetic_dev_artifacts(tmp_path)
    document = build_preregistration(**paths)
    assert document["preregistration_version"] == PREREGISTRATION_VERSION
    assert document["holdout_accessed"] is False
    assert len(document["selected_representatives"]) == 3
    assert document["cost_model"]["stress_multiplier"] == 1.5
    assert "preregistration_semantic_sha256" in document


def test_verify_preregistration_accepts_a_fresh_document(tmp_path: Path) -> None:
    paths = _synthetic_dev_artifacts(tmp_path)
    document = build_preregistration(**paths)
    verify_preregistration(document, **paths)


def test_verify_preregistration_rejects_a_tampered_self_hash(tmp_path: Path) -> None:
    paths = _synthetic_dev_artifacts(tmp_path)
    document = build_preregistration(**paths)
    tampered = dict(document)
    tampered["selected_representatives"] = {"forged": True}
    with pytest.raises(ValidationPreregistrationError):
        verify_preregistration(tampered, **paths)


def test_verify_preregistration_rejects_wrong_version(tmp_path: Path) -> None:
    paths = _synthetic_dev_artifacts(tmp_path)
    document = dict(build_preregistration(**paths))
    document["preregistration_version"] = "some-other-version"
    document["preregistration_semantic_sha256"] = _canonical_sha256(
        {k: v for k, v in document.items() if k != "preregistration_semantic_sha256"}
    )
    with pytest.raises(ValidationPreregistrationError):
        verify_preregistration(document, **paths)


def test_verify_preregistration_rejects_stale_representatives(tmp_path: Path) -> None:
    # Build against one Stage-2 scorecard, then verify against a DIFFERENT
    # one -- representatives no longer reproduce, so it must be rejected.
    paths = _synthetic_dev_artifacts(tmp_path)
    document = build_preregistration(**paths)

    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other_scorecard = other_dir / "scorecard.csv"
    _write_synthetic_stage2_scorecard(other_scorecard)
    # Mutate the region so the derived representative changes.
    rows = list(csv.DictReader(other_scorecard.open(encoding="utf-8")))
    for row in rows:
        if row["family"] == DONCHIAN_FAMILY and json.loads(row["parameters"])[
            "lookback"
        ] == 45:
            row["stressed_cost_return"] = "-0.01"
            row["profitable_pair_count"] = "1"
    with other_scorecard.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0].keys()), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    other_paths = dict(paths)
    other_paths["scorecard_path"] = other_scorecard
    with pytest.raises(ValidationPreregistrationError):
        verify_preregistration(document, **other_paths)


def test_validation_execution_cannot_occur_without_a_valid_preregistration() -> None:
    with pytest.raises(ValidationPreregistrationError):
        verify_preregistration({})


# ---------------------------------------------------------------------------
# Fail-closed guards: forbidden paths and out-of-partition observations.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path,should_raise",
    [
        (Path("/data/oanda_fx_alpha_lab_v1/development_root/EURUSD"), True),
        (Path("/data/oanda_fx_alpha_lab_v1/holdout_root/EURUSD"), True),
        (Path("/data/final_holdout/EURUSD"), True),
        (Path("/data/oanda_fx_alpha_lab_v1/validation_root/EURUSD"), False),
    ],
)
def test_reject_forbidden_path_accepts_only_the_validation_root(
    path: Path, should_raise: bool
) -> None:
    if should_raise:
        with pytest.raises(AlphaLabValidationError):
            _reject_forbidden_path(path)
    else:
        _reject_forbidden_path(path)  # must not raise


def test_out_of_partition_observations_are_rejected() -> None:
    index = pd.date_range(
        VALIDATION_START - timedelta(days=1), periods=5, freq="1D", tz=UTC
    )
    close = pd.DataFrame({"EUR/USD.OANDA": np.linspace(1.0, 1.1, 5)}, index=index)
    dataset = AlphaLabDataset(
        timeframe="H4",
        instrument_ids=("EUR/USD.OANDA",),
        start_utc=VALIDATION_START,
        end_exclusive_utc=HOLDOUT_START,
        alignment_policy=ALIGNMENT_POLICY,
        open=close,
        high=close,
        low=close,
        close=close,
        bid=close,
        ask=close,
        spread=close * 0.0,
    )
    with pytest.raises(AlphaLabValidationError):
        _reject_out_of_partition_observations(dataset)


def test_in_partition_observations_are_accepted() -> None:
    index = pd.date_range(VALIDATION_START, periods=5, freq="1D", tz=UTC)
    close = pd.DataFrame({"EUR/USD.OANDA": np.linspace(1.0, 1.1, 5)}, index=index)
    dataset = AlphaLabDataset(
        timeframe="H4",
        instrument_ids=("EUR/USD.OANDA",),
        start_utc=VALIDATION_START,
        end_exclusive_utc=HOLDOUT_START,
        alignment_policy=ALIGNMENT_POLICY,
        open=close,
        high=close,
        low=close,
        close=close,
        bid=close,
        ask=close,
        spread=close * 0.0,
    )
    _reject_out_of_partition_observations(dataset)


# ---------------------------------------------------------------------------
# Section 7: the VALIDATION readiness contract -- the runner must require
# the dedicated VALIDATION readiness artifact, never the DEVELOPMENT one.
# ---------------------------------------------------------------------------


def _valid_validation_readiness_document() -> dict:
    config = load_oanda_alpha_lab_validation_config()
    payload = {
        "readiness_version": VALIDATION_READINESS_VERSION,
        "lineage_id": config.lineage_id,
        "partition": "VALIDATION",
        "alpha_lab_config_sha256": config.semantic_sha256,
        "validation_start_utc": _iso(VALIDATION_START),
        "validation_end_exclusive_utc": _iso(HOLDOUT_START),
        "ordered_instrument_ids": [item.instrument_id for item in config.instruments],
        "per_instrument_status": {
            item.instrument_id: "research_ready" for item in config.instruments
        },
        "instrument_artifacts": [
            {"instrument_id": item.instrument_id, "catalog_tree_sha256": "a" * 64}
            for item in config.instruments
        ],
        "holdout_accessed": False,
        "holdout_rows_admitted": 0,
        "research_ready": True,
    }
    payload["semantic_sha256"] = _canonical_sha256(payload)
    return payload


def test_verify_validation_readiness_accepts_a_valid_document() -> None:
    document = _valid_validation_readiness_document()
    config = _verify_validation_readiness(document, config_path=_VALIDATION_CONFIG_PATH)
    assert config.partition == "VALIDATION"


def _readiness_wrong_partition_document() -> dict:
    # The real, frozen DEVELOPMENT readiness shape: no "partition" field,
    # and readiness_version is the DEVELOPMENT one -- must be rejected.
    payload = {
        "readiness_version": "oanda-alpha-lab-readiness-1",
        "lineage_id": "oanda_fx_alpha_lab_v1",
        "alpha_lab_config_sha256": "b" * 64,
        "development_start_utc": "2019-03-11T00:00:00Z",
        "development_end_exclusive_utc": "2023-04-11T00:00:00Z",
        "holdout_accessed": False,
        "holdout_rows_admitted": 0,
        "research_ready": True,
    }
    payload["semantic_sha256"] = _canonical_sha256(payload)
    return payload


def _readiness_missing_partition_field_document() -> dict:
    document = _valid_validation_readiness_document()
    document.pop("partition")
    return document


def _readiness_tampered_self_hash_document() -> dict:
    document = _valid_validation_readiness_document()
    document["research_ready"] = False  # mutate after hashing -> stale hash
    return document


def _readiness_wrong_boundary_document() -> dict:
    config = load_oanda_alpha_lab_validation_config()
    payload = {
        "readiness_version": VALIDATION_READINESS_VERSION,
        "lineage_id": config.lineage_id,
        "partition": "VALIDATION",
        "alpha_lab_config_sha256": config.semantic_sha256,
        "validation_start_utc": _iso(VALIDATION_START),
        "validation_end_exclusive_utc": "2024-12-31T00:00:00Z",  # wrong
        "holdout_accessed": False,
        "holdout_rows_admitted": 0,
        "research_ready": True,
    }
    payload["semantic_sha256"] = _canonical_sha256(payload)
    return payload


def _readiness_holdout_accessed_true_document() -> dict:
    document = _valid_validation_readiness_document()
    document["holdout_accessed"] = True
    document["semantic_sha256"] = _canonical_sha256(
        {k: v for k, v in document.items() if k != "semantic_sha256"}
    )
    return document


@pytest.mark.parametrize(
    "build_document,match",
    [
        (_readiness_wrong_partition_document, "partition"),
        (_readiness_missing_partition_field_document, None),
        (_readiness_tampered_self_hash_document, "self-hash"),
        (_readiness_wrong_boundary_document, None),
        (_readiness_holdout_accessed_true_document, None),
    ],
)
def test_verify_validation_readiness_rejects_malformed_documents(
    build_document, match
) -> None:
    document = build_document()
    with pytest.raises(AlphaLabValidationError, match=match):
        _verify_validation_readiness(document, config_path=_VALIDATION_CONFIG_PATH)


def test_load_validation_dataset_requires_all_seven_instruments(tmp_path: Path) -> None:
    from ftmoquant.research.alpha_lab.validation import load_validation_dataset

    document = _valid_validation_readiness_document()
    # Downgrade one instrument to not_ready -- only six remain research_ready.
    first_id = document["ordered_instrument_ids"][0]
    document["per_instrument_status"][first_id] = "not_ready"
    document["semantic_sha256"] = _canonical_sha256(
        {k: v for k, v in document.items() if k != "semantic_sha256"}
    )
    readiness_path = tmp_path / "validation_readiness.json"
    readiness_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(AlphaLabValidationError, match="seven"):
        load_validation_dataset(
            validation_root=tmp_path / "validation_root",
            universe_readiness_path=readiness_path,
            timeframe="H4",
        )


def test_load_validation_dataset_requires_a_readiness_document() -> None:
    from ftmoquant.research.alpha_lab.validation import load_validation_dataset

    with pytest.raises(AlphaLabValidationError):
        load_validation_dataset(
            validation_root=Path("/nonexistent/validation_root"),
            universe_readiness_path=Path("/nonexistent/readiness.json"),
            timeframe="H4",
        )


def test_load_validation_dataset_rejects_a_development_root_path() -> None:
    from ftmoquant.research.alpha_lab.validation import load_validation_dataset

    with pytest.raises(AlphaLabValidationError, match="forbidden"):
        load_validation_dataset(
            validation_root=Path(
                "/Users/Shared/FTMOQuant-data/oanda_fx_alpha_lab_v1/"
                "development_m1_2019_03_11_2023_04_11"
            ),
            universe_readiness_path=Path("/nonexistent/readiness.json"),
            timeframe="H4",
        )


# ---------------------------------------------------------------------------
# Concentration gate.
# ---------------------------------------------------------------------------


def test_concentration_share_is_the_max_pairs_share_of_positive_profit() -> None:
    profits = {"A": 0.10, "B": 0.05, "C": -0.20, "D": 0.05}
    share = compute_concentration_share(profits)
    assert share == pytest.approx(0.10 / 0.20)


def test_concentration_gate_fails_when_total_positive_profit_is_non_positive() -> None:
    assert compute_concentration_share({"A": -0.1, "B": -0.2}) is None
    assert compute_concentration_share({"A": 0.0, "B": 0.0}) is None


def test_concentration_gate_boundary_at_exactly_seventy_percent() -> None:
    share = compute_concentration_share({"A": 0.70, "B": 0.30})
    assert share == pytest.approx(0.70)
    assert share <= CONCENTRATION_MAX_SHARE


# ---------------------------------------------------------------------------
# The five gates, independently enforced, and validation_passed as their AND.
# ---------------------------------------------------------------------------


def _validation_row(
    *,
    instrument: str = "AGGREGATE_EQUAL_WEIGHT",
    profitable_pair_count: int = 5,
    aggregate_equal_weight_return: float = 0.05,
    aggregate_stressed_return: float = 0.04,
    aggregate_equal_weight_sharpe: float = 0.5,
    concentration_share: float | None = 0.3,
):
    from ftmoquant.research.alpha_lab.validation import ValidationScorecardRow

    return ValidationScorecardRow(
        family=DONCHIAN_FAMILY,
        strategy_id="validation_donchian_h4_lookback45_v1",
        instrument=instrument,
        timeframe="H4",
        parameters=json.dumps({"lookback": 45}),
        observation_count=1000,
        trade_count=20,
        cumulative_return=aggregate_equal_weight_return,
        annualized_return=0.1,
        annualized_sharpe=aggregate_equal_weight_sharpe,
        maximum_drawdown=-0.05,
        win_rate=0.5,
        turnover_or_trade_activity=0.02,
        base_cost_return=aggregate_equal_weight_return,
        stressed_cost_return=aggregate_stressed_return,
        profitable_pair_count=profitable_pair_count,
        positive_sharpe_pair_count=profitable_pair_count,
        median_pair_sharpe=0.2,
        worst_pair_sharpe=-0.1,
        aggregate_equal_weight_return=aggregate_equal_weight_return,
        aggregate_equal_weight_sharpe=aggregate_equal_weight_sharpe,
        aggregate_stressed_return=aggregate_stressed_return,
        concentration_share=concentration_share,
        positive_subperiod_count=2,
        worst_subperiod_return=0.01,
        median_subperiod_sharpe=0.3,
    )


@pytest.mark.parametrize(
    "field,passing_value,failing_value,gate_attr",
    [
        ("profitable_pair_count", 4, 3, "cross_pair_gate"),
        ("aggregate_equal_weight_return", 0.001, 0.0, "aggregate_return_gate"),
        ("aggregate_stressed_return", 0.001, -0.001, "stressed_return_gate"),
        ("aggregate_equal_weight_sharpe", 0.01, -0.01, "sharpe_sign_gate"),
    ],
)
def test_each_gate_is_enforced_independently(
    field: str, passing_value: float, failing_value: float, gate_attr: str
) -> None:
    passing = _evaluate_family([_validation_row(**{field: passing_value})])
    failing = _evaluate_family([_validation_row(**{field: failing_value})])
    assert getattr(passing, gate_attr) is True
    assert getattr(failing, gate_attr) is False
    # every other gate held constant -> only the named gate differs
    assert passing.validation_passed is True
    assert failing.validation_passed is False


def test_concentration_gate_enforced_independently() -> None:
    passing = _evaluate_family([_validation_row(concentration_share=0.69)])
    failing = _evaluate_family([_validation_row(concentration_share=0.71)])
    failing_none = _evaluate_family([_validation_row(concentration_share=None)])
    assert passing.concentration_gate is True
    assert failing.concentration_gate is False
    assert failing_none.concentration_gate is False
    assert failing.validation_passed is False
    assert failing_none.validation_passed is False


def test_validation_passed_is_exactly_the_and_of_all_five_gates() -> None:
    all_pass = _evaluate_family([_validation_row()])
    assert all_pass.validation_passed == all(
        (
            all_pass.cross_pair_gate,
            all_pass.aggregate_return_gate,
            all_pass.stressed_return_gate,
            all_pass.sharpe_sign_gate,
            all_pass.concentration_gate,
        )
    )
    assert all_pass.validation_passed is True

    one_fail = _evaluate_family([_validation_row(profitable_pair_count=2)])
    assert one_fail.validation_passed is False


# ---------------------------------------------------------------------------
# End-to-end synthetic run: exactly 3 representatives, 24 rows, 1.5x stress,
# determinism, and the failure-policy structural guarantee.
# ---------------------------------------------------------------------------


def _synth_ds(*, timeframe: str, periods: int, freq: str, seed: int):
    index = pd.date_range(VALIDATION_START, periods=periods, freq=freq, tz=UTC)
    rng = np.random.default_rng(seed)
    close = pd.DataFrame(
        {pair: 1.1 + np.cumsum(rng.normal(0, 0.0005, periods)) for pair in _PAIRS},
        index=index,
    )
    high = close + 0.0006
    low = close - 0.0006
    bid = close - 0.0001
    ask = close + 0.0001
    return AlphaLabDataset(
        timeframe=timeframe,  # type: ignore[arg-type]
        instrument_ids=_PAIRS,
        start_utc=VALIDATION_START,
        end_exclusive_utc=HOLDOUT_START,
        alignment_policy=ALIGNMENT_POLICY,
        open=close,
        high=high,
        low=low,
        close=close,
        bid=bid,
        ask=ask,
        spread=ask - bid,
    )


def _synthetic_preregistration() -> dict[str, object]:
    return {
        "selected_representatives": {
            DONCHIAN_FAMILY: {
                "strategy_id": "validation_donchian_h4_lookback45_v1",
                "timeframe": "H4",
                "lookback": 45,
            },
            TSM_FAMILY: {
                "strategy_id": "validation_tsm_h4_lookback40_v1",
                "timeframe": "H4",
                "lookback": 40,
            },
            MEAN_REVERSION_FAMILY: {
                "strategy_id": "validation_meanrev_h1_lookback40_z2.0_v1",
                "timeframe": "H1",
                "lookback": 40,
                "z_entry": 2.0,
            },
        }
    }


def test_run_validation_evaluates_three_representatives_with_full_scorecard() -> None:
    datasets = {
        "H4": _synth_ds(timeframe="H4", periods=1200, freq="4h", seed=1),
        "H1": _synth_ds(timeframe="H1", periods=3000, freq="1h", seed=2),
    }
    result = run_validation(
        preregistration=_synthetic_preregistration(), datasets_by_timeframe=datasets
    )
    assert len(result.family_results) == 3
    assert {r.family for r in result.family_results} == {
        DONCHIAN_FAMILY,
        TSM_FAMILY,
        MEAN_REVERSION_FAMILY,
    }

    assert len(result.rows) == 24
    representatives = _synthetic_preregistration()["selected_representatives"]
    for family_config in representatives.values():
        strategy_id = family_config["strategy_id"]
        strategy_rows = [r for r in result.rows if r.strategy_id == strategy_id]
        assert len(strategy_rows) == 8
        assert {r.instrument for r in strategy_rows} == set(_PAIRS) | {
            "AGGREGATE_EQUAL_WEIGHT"
        }


def test_stress_return_uses_exactly_one_point_five_x_the_base_fee() -> None:
    datasets = {
        "H4": _synth_ds(timeframe="H4", periods=1200, freq="4h", seed=1),
        "H1": _synth_ds(timeframe="H1", periods=3000, freq="1h", seed=2),
    }
    result = run_validation(
        preregistration=_synthetic_preregistration(), datasets_by_timeframe=datasets
    )
    traded = [row for row in result.rows if row.trade_count > 0]
    assert traded
    for row in traded:
        assert row.stressed_cost_return <= row.base_cost_return + 1e-9
    aggregate_rows = [
        row for row in result.rows if row.instrument == "AGGREGATE_EQUAL_WEIGHT"
    ]
    for row in aggregate_rows:
        assert row.aggregate_stressed_return == pytest.approx(row.stressed_cost_return)


def test_run_validation_is_deterministic() -> None:
    datasets = {
        "H4": _synth_ds(timeframe="H4", periods=800, freq="4h", seed=3),
        "H1": _synth_ds(timeframe="H1", periods=1500, freq="1h", seed=4),
    }
    prereg = _synthetic_preregistration()
    first = run_validation(preregistration=prereg, datasets_by_timeframe=datasets)
    second = run_validation(preregistration=prereg, datasets_by_timeframe=datasets)
    assert first.rows == second.rows
    assert first.family_results == second.family_results


def test_failed_representative_never_triggers_an_alternate_family_parameter() -> None:
    # Even when a representative's own numbers look poor, run_validation has
    # no branch/loop that could substitute a different lookback: exactly the
    # preregistered strategy_id is what appears in the output, always.
    datasets = {
        "H4": _synth_ds(timeframe="H4", periods=200, freq="4h", seed=99),
        "H1": _synth_ds(timeframe="H1", periods=400, freq="1h", seed=98),
    }
    result = run_validation(
        preregistration=_synthetic_preregistration(), datasets_by_timeframe=datasets
    )
    donchian_strategy_ids = {
        row.strategy_id for row in result.rows if row.family == DONCHIAN_FAMILY
    }
    assert donchian_strategy_ids == {"validation_donchian_h4_lookback45_v1"}
    tsm_strategy_ids = {
        row.strategy_id for row in result.rows if row.family == TSM_FAMILY
    }
    assert tsm_strategy_ids == {"validation_tsm_h4_lookback40_v1"}


def test_write_validation_results_produces_all_required_files(tmp_path: Path) -> None:
    datasets = {
        "H4": _synth_ds(timeframe="H4", periods=400, freq="4h", seed=5),
        "H1": _synth_ds(timeframe="H1", periods=800, freq="1h", seed=6),
    }
    prereg = _synthetic_preregistration()
    result = run_validation(preregistration=prereg, datasets_by_timeframe=datasets)
    output_dir = tmp_path / "validation_out"
    write_validation_results(result, prereg, output_dir)

    assert (output_dir / "preregistration.json").is_file()
    assert (output_dir / "validation_scorecard.csv").is_file()
    assert (output_dir / "family_validation_summary.csv").is_file()
    assert (output_dir / "metadata.json").is_file()

    summary_path = output_dir / "family_validation_summary.csv"
    with summary_path.open(encoding="utf-8") as stream:
        summary_rows = list(csv.DictReader(stream))
    assert len(summary_rows) == 3

    with (output_dir / "validation_scorecard.csv").open(encoding="utf-8") as stream:
        scorecard_rows = list(csv.DictReader(stream))
    assert len(scorecard_rows) == 24


def test_write_validation_results_refuses_to_overwrite(tmp_path: Path) -> None:
    datasets = {
        "H4": _synth_ds(timeframe="H4", periods=200, freq="4h", seed=7),
        "H1": _synth_ds(timeframe="H1", periods=400, freq="1h", seed=8),
    }
    prereg = _synthetic_preregistration()
    result = run_validation(preregistration=prereg, datasets_by_timeframe=datasets)
    output_dir = tmp_path / "validation_out"
    write_validation_results(result, prereg, output_dir)
    with pytest.raises(FileExistsError):
        write_validation_results(result, prereg, output_dir)


# ---------------------------------------------------------------------------
# Existing DEVELOPMENT Stage-1/Stage-2 behavior must remain unchanged.
# ---------------------------------------------------------------------------


def test_stage1_and_stage2_grids_are_unaffected_by_this_module() -> None:
    assert len(build_grid()) == 49
    assert len(build_stage2_grid()) == 23
