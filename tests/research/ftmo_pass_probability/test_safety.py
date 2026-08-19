"""Proves the firewalls required by the task's hard research rule and final-
holdout constraint: sizing/Monte Carlo selection can only ever be driven by
the labelled DEVELOPMENT artifact, and no trade at or after the final
holdout boundary is reachable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ftmoquant.research.ftmo_pass_probability.path_extraction import (
    PathExtractionError,
    load_development_trade_path,
)
from ftmoquant.research.stage_g import HOLDOUT_START

from .conftest import write_fixture_execution_dir

REAL_VALIDATION_DIR = Path(
    ".artifacts/usdcad_sweep_bos_retest_v1/validation_execution"
).resolve()
REAL_DEVELOPMENT_DIR = Path(
    ".artifacts/usdcad_sweep_bos_retest_v1/development_execution"
).resolve()
_HOLDOUT_START_NS = int(HOLDOUT_START.timestamp() * 1_000_000_000)


def test_the_real_validation_directory_is_never_loadable_for_sizing() -> None:
    with pytest.raises(PathExtractionError):
        load_development_trade_path(REAL_VALIDATION_DIR)


def test_the_real_development_artifact_never_reaches_the_final_holdout() -> None:
    path = load_development_trade_path(REAL_DEVELOPMENT_DIR)
    assert all(trade.exit_ns < _HOLDOUT_START_NS for trade in path.trades)


def test_a_directory_merely_named_development_is_not_enough(tmp_path: Path) -> None:
    """The guard reads the artifact's own reported partition/access flags,
    not the directory name -- renaming a validation export cannot bypass it."""

    execution_dir = write_fixture_execution_dir(
        tmp_path / "development_execution", partition="validation"
    )
    with pytest.raises(PathExtractionError):
        load_development_trade_path(execution_dir)


def test_a_reported_validation_access_is_rejected_even_with_the_right_partition(
    tmp_path: Path,
) -> None:
    execution_dir = write_fixture_execution_dir(
        tmp_path / "development_execution", validation_accessed=True
    )
    with pytest.raises(PathExtractionError):
        load_development_trade_path(execution_dir)


def test_a_reported_final_holdout_access_is_rejected(tmp_path: Path) -> None:
    execution_dir = write_fixture_execution_dir(
        tmp_path / "development_execution", final_holdout_accessed=True
    )
    with pytest.raises(PathExtractionError):
        load_development_trade_path(execution_dir)


def test_any_single_trade_at_or_after_the_holdout_boundary_fails_the_whole_load(
    tmp_path: Path,
) -> None:
    rows = [
        "0,1,1000000000000,2000000000000,1.3,1.3,target,1000000000000,2000000000000,1.3,1.3,657.00,1.9",
        (
            f"1,1,{_HOLDOUT_START_NS - 1},{_HOLDOUT_START_NS},1.3,1.3,stop,"
            f"{_HOLDOUT_START_NS - 1},{_HOLDOUT_START_NS},1.3,1.3,-330.00,-1.0"
        ),
    ]
    execution_dir = write_fixture_execution_dir(
        tmp_path / "development_execution", rows=rows
    )
    with pytest.raises(PathExtractionError):
        load_development_trade_path(execution_dir)
