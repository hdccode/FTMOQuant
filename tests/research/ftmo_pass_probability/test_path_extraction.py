from __future__ import annotations

from pathlib import Path

import pytest

from ftmoquant.research.ftmo_pass_probability.path_extraction import (
    PathExtractionError,
    load_development_trade_path,
)

from .conftest import write_fixture_execution_dir

REAL_DEVELOPMENT_DIR = Path(
    ".artifacts/usdcad_sweep_bos_retest_v1/development_execution"
).resolve()
REAL_VALIDATION_DIR = Path(
    ".artifacts/usdcad_sweep_bos_retest_v1/validation_execution"
).resolve()


def test_loads_the_real_frozen_development_artifact() -> None:
    path = load_development_trade_path(REAL_DEVELOPMENT_DIR)
    assert len(path.trades) == 306
    for previous, current in zip(path.trades, path.trades[1:]):
        assert current.entry_ns > previous.entry_ns
        assert current.entry_ns >= previous.exit_ns


def test_rejects_the_real_validation_labelled_artifact() -> None:
    with pytest.raises(PathExtractionError):
        load_development_trade_path(REAL_VALIDATION_DIR)


def test_accepts_a_well_formed_development_fixture(tmp_path: Path) -> None:
    execution_dir = write_fixture_execution_dir(tmp_path / "development_execution")
    path = load_development_trade_path(execution_dir)
    assert len(path.trades) == 2
    assert path.trades[0].exit_reason == "target"
    assert path.trades[1].exit_reason == "stop"


def test_rejects_non_development_partition(tmp_path: Path) -> None:
    execution_dir = write_fixture_execution_dir(
        tmp_path / "validation_execution", partition="validation"
    )
    with pytest.raises(PathExtractionError, match="partition"):
        load_development_trade_path(execution_dir)


def test_rejects_validation_accessed_true(tmp_path: Path) -> None:
    execution_dir = write_fixture_execution_dir(
        tmp_path / "development_execution", validation_accessed=True
    )
    with pytest.raises(PathExtractionError, match="VALIDATION"):
        load_development_trade_path(execution_dir)


def test_rejects_final_holdout_accessed_true(tmp_path: Path) -> None:
    execution_dir = write_fixture_execution_dir(
        tmp_path / "development_execution", final_holdout_accessed=True
    )
    with pytest.raises(PathExtractionError, match="holdout"):
        load_development_trade_path(execution_dir)


def test_rejects_a_mismatched_frozen_candidate_identity(tmp_path: Path) -> None:
    execution_dir = write_fixture_execution_dir(
        tmp_path / "development_execution", identity={"family": "not_the_frozen_family"}
    )
    with pytest.raises(PathExtractionError, match="identity"):
        load_development_trade_path(execution_dir)


def test_rejects_a_trade_timestamp_at_the_final_holdout_boundary(
    tmp_path: Path,
) -> None:
    holdout_start_ns = 1_724_198_400_000_000_000  # 2024-08-21T00:00:00Z
    entry_ns = holdout_start_ns - 3_600_000_000_000
    rows = [
        (
            f"0,1,{entry_ns},{holdout_start_ns},1.3,1.3,target,"
            f"{entry_ns},{holdout_start_ns},1.3,1.3,657.00,"
            "1.972972972972972972972972973"
        )
    ]
    execution_dir = write_fixture_execution_dir(
        tmp_path / "development_execution", rows=rows
    )
    with pytest.raises(PathExtractionError, match="holdout"):
        load_development_trade_path(execution_dir)


def test_rejects_overlapping_trades(tmp_path: Path) -> None:
    rows = [
        "0,1,1000000000000,5000000000000,1.3,1.3,target,1000000000000,5000000000000,1.3,1.3,657.00,1.9",
        "1,1,3000000000000,6000000000000,1.3,1.3,stop,3000000000000,6000000000000,1.3,1.3,-330.00,-1.0",
    ]
    execution_dir = write_fixture_execution_dir(
        tmp_path / "development_execution", rows=rows
    )
    with pytest.raises(PathExtractionError, match="overlap"):
        load_development_trade_path(execution_dir)


def test_stop_distance_and_risk_budget_are_derived_from_net_r(tmp_path: Path) -> None:
    execution_dir = write_fixture_execution_dir(tmp_path / "development_execution")
    path = load_development_trade_path(execution_dir)
    trade = path.trades[0]
    # 657.00 / 1.972972972972972972972972973 -- the causal, pre-trade risk
    # budget implied by the frozen 100,000-unit notional and this trade's
    # own frozen stop, expressed in CAD (the pair's settlement currency) --
    # then converted to USD by the fixture's dummy entry_price of 1.3.
    assert abs(float(trade.original_risk_budget) - 333.0 / 1.3) < 1.0
