from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

import pandas as pd
import pytest

from ftmoquant.research.alpha_lab.data import (
    ALIGNMENT_POLICY,
    AlphaLabDataError,
    assemble_aligned_dataset,
    resolve_development_roots,
)
from ftmoquant.research.stage_g import (
    DEVELOPMENT_END_EXCLUSIVE,
    DEVELOPMENT_START,
    DevelopmentResearchContext,
    ResearchPartition,
    StageGValidationError,
)


def _frame(timestamps: list[str], base: float) -> pd.DataFrame:
    index = pd.DatetimeIndex(timestamps, tz=UTC)
    return pd.DataFrame(
        {
            "open": [base] * len(index),
            "high": [base + 0.01] * len(index),
            "low": [base - 0.01] * len(index),
            "close": [base + 0.001] * len(index),
            "bid": [base] * len(index),
            "ask": [base + 0.0002] * len(index),
        },
        index=index,
    )


def test_assembled_index_is_utc_and_sorted() -> None:
    per_instrument = {
        "B/USD.X": _frame(["2020-01-01T02:00:00Z", "2020-01-01T01:00:00Z"], 2.0),
        "A/USD.X": _frame(["2020-01-01T02:00:00Z", "2020-01-01T01:00:00Z"], 1.0),
    }
    dataset = assemble_aligned_dataset(
        per_instrument=per_instrument,
        instrument_ids=("B/USD.X", "A/USD.X"),
        timeframe="H1",
    )
    assert str(dataset.close.index.tz) == "UTC"
    assert list(dataset.close.index) == sorted(dataset.close.index)


def test_deterministic_instrument_ordering_is_alphabetical() -> None:
    per_instrument = {
        "GBP/USD.X": _frame(["2020-01-01T01:00:00Z"], 1.3),
        "EUR/USD.X": _frame(["2020-01-01T01:00:00Z"], 1.1),
    }
    dataset = assemble_aligned_dataset(
        per_instrument=per_instrument,
        instrument_ids=("GBP/USD.X", "EUR/USD.X"),
        timeframe="H1",
    )
    assert dataset.instrument_ids == ("EUR/USD.X", "GBP/USD.X")
    assert list(dataset.close.columns) == ["EUR/USD.X", "GBP/USD.X"]


def test_multi_pair_alignment_keeps_only_shared_timestamps() -> None:
    per_instrument = {
        "A/USD.X": _frame(
            ["2020-01-01T01:00:00Z", "2020-01-01T02:00:00Z", "2020-01-01T03:00:00Z"],
            1.0,
        ),
        "B/USD.X": _frame(
            ["2020-01-01T02:00:00Z", "2020-01-01T03:00:00Z", "2020-01-01T04:00:00Z"],
            2.0,
        ),
    }
    dataset = assemble_aligned_dataset(
        per_instrument=per_instrument,
        instrument_ids=("A/USD.X", "B/USD.X"),
        timeframe="H1",
    )
    expected = pd.DatetimeIndex(
        ["2020-01-01T02:00:00Z", "2020-01-01T03:00:00Z"], tz=UTC
    )
    assert list(dataset.close.index) == list(expected)


def test_disjoint_instruments_raise_instead_of_filling() -> None:
    per_instrument = {
        "A/USD.X": _frame(["2020-01-01T01:00:00Z"], 1.0),
        "B/USD.X": _frame(["2020-01-01T09:00:00Z"], 2.0),
    }
    with pytest.raises(AlphaLabDataError):
        assemble_aligned_dataset(
            per_instrument=per_instrument,
            instrument_ids=("A/USD.X", "B/USD.X"),
            timeframe="H1",
        )


def test_no_nan_forward_fill_after_alignment() -> None:
    per_instrument = {
        "A/USD.X": _frame(
            ["2020-01-01T01:00:00Z", "2020-01-01T02:00:00Z"],
            1.0,
        ),
        "B/USD.X": _frame(
            ["2020-01-01T01:00:00Z"],
            2.0,
        ),
    }
    dataset = assemble_aligned_dataset(
        per_instrument=per_instrument,
        instrument_ids=("A/USD.X", "B/USD.X"),
        timeframe="H1",
    )
    assert len(dataset.close.index) == 1
    assert not dataset.close.isna().any().any()


def test_alignment_policy_is_recorded_and_stable() -> None:
    per_instrument = {"A/USD.X": _frame(["2020-01-01T01:00:00Z"], 1.0)}
    dataset = assemble_aligned_dataset(
        per_instrument=per_instrument, instrument_ids=("A/USD.X",), timeframe="H4"
    )
    assert dataset.alignment_policy == ALIGNMENT_POLICY
    assert "no_fill" in ALIGNMENT_POLICY


def test_resolve_development_roots_rejects_validation_and_holdout_paths(
    tmp_path: Any,
) -> None:
    for forbidden in ("validation", "holdout"):
        forbidden_dir = tmp_path / forbidden
        forbidden_dir.mkdir()
        with pytest.raises(AlphaLabDataError):
            resolve_development_roots(("EUR/USD.DUKASCOPY",), forbidden_dir)


def test_resolve_development_roots_requires_split_view_manifest(tmp_path: Any) -> None:
    with pytest.raises(AlphaLabDataError):
        resolve_development_roots(("EUR/USD.DUKASCOPY",), tmp_path)


def test_development_boundary_is_enforced_by_reused_stage_g_context() -> None:
    context = DevelopmentResearchContext(universe=cast(Any, None), artifacts=())
    with pytest.raises(StageGValidationError):
        context.require_range(
            DEVELOPMENT_START,
            datetime(2024, 1, 1, tzinfo=UTC),
            partition=ResearchPartition.DEVELOPMENT,
        )
    # A within-bounds request is admitted.
    context.require_range(
        DEVELOPMENT_START,
        DEVELOPMENT_END_EXCLUSIVE,
        partition=ResearchPartition.DEVELOPMENT,
    )


@pytest.mark.parametrize(
    "partition", [ResearchPartition.VALIDATION, ResearchPartition.HOLDOUT]
)
def test_validation_and_holdout_partitions_are_rejected(
    partition: ResearchPartition,
) -> None:
    context = DevelopmentResearchContext(universe=cast(Any, None), artifacts=())
    with pytest.raises(StageGValidationError):
        context.require_range(
            DEVELOPMENT_START, DEVELOPMENT_END_EXCLUSIVE, partition=partition
        )
