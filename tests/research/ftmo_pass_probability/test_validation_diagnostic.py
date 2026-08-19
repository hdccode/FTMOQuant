"""Tests for the read-only VALIDATION diagnostic of the frozen sizing policy.

Kept minimal/high-value: hard-freeze enforcement, DEVELOPMENT-not-VALIDATION
block length, correct-partition loading, holdout firewall, determinism, and
write-once artifact behaviour.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from ftmoquant.prop_rules import load_prop_rule_set
from ftmoquant.research.ftmo_pass_probability import validation_diagnostic as vd
from ftmoquant.research.ftmo_pass_probability.artifacts import ArtifactError
from ftmoquant.research.ftmo_pass_probability.cli import (
    _derive_seed,
    validation_diagnostic_main,
)
from ftmoquant.research.ftmo_pass_probability.path_extraction import (
    PathExtractionError,
    load_validation_trade_path,
)
from ftmoquant.research.ftmo_pass_probability.sizing import SizingFamily

from .conftest import _BASE_NS, _DAY_NS, _row, write_fixture_execution_dir

RULE_CONFIG = Path("config/prop/ftmo_2step_swing_2026-08.yaml").resolve()
REAL_VALIDATION_DIR = Path(
    ".artifacts/usdcad_sweep_bos_retest_v1/validation_execution"
).resolve()
REAL_DEVELOPMENT_DIR = Path(
    ".artifacts/usdcad_sweep_bos_retest_v1/development_execution"
).resolve()
_YEAR_NS = 365 * 24 * 60 * 60 * 1_000_000_000


def _many_rows(count: int) -> list[str]:
    """arch's ``optimal_block_length`` needs more than a couple of
    observations to avoid a degenerate internal matmul -- build a small but
    sufficient synthetic trade series."""

    rows = []
    for i in range(count):
        entry = _BASE_NS + i * 2 * _DAY_NS
        exit_ = entry + 3_600_000_000_000
        if i % 3 == 0:
            rows.append(_row(i, entry, exit_, "stop", "-330.00", "-1.0"))
        else:
            rows.append(
                _row(
                    i, entry, exit_, "target", "657.00", "1.972972972972972972972972973"
                )
            )
    return rows


def test_policy_and_method_are_hard_frozen_constants() -> None:
    assert vd.FROZEN_POLICY_ID == "fixed_notional_2_0x"
    assert vd.FROZEN_METHOD == "stationary"
    policy = vd.frozen_policy()
    assert policy.family is SizingFamily.FIXED_NOTIONAL_MULTIPLIER
    assert policy.notional_multiplier == Decimal("2.0")


def test_cli_exposes_no_policy_or_method_selection_flag() -> None:
    with pytest.raises(SystemExit):
        validation_diagnostic_main(["--policy-id", "fixed_notional_1_0x"])
    with pytest.raises(SystemExit):
        validation_diagnostic_main(["--method", "circular"])


def test_block_size_is_derived_from_development_trades_not_validation(
    monkeypatch, tmp_path: Path
) -> None:
    development_dir = write_fixture_execution_dir(
        tmp_path / "development_execution", rows=_many_rows(20)
    )
    validation_dir = write_fixture_execution_dir(
        tmp_path / "validation_execution", partition="validation", rows=_many_rows(15)
    )
    development_path = vd.load_development_trade_path(development_dir)
    validation_path = load_validation_trade_path(validation_dir)

    seen_trade_sets: list[tuple] = []
    real_derive = vd.derive_frozen_block_length

    def _spy(trades):
        seen_trade_sets.append(trades)
        return real_derive(trades)

    monkeypatch.setattr(vd, "derive_frozen_block_length", _spy)

    rules = load_prop_rule_set(RULE_CONFIG)
    vd.run_validation_diagnostic(
        development_execution_dir=development_dir,
        validation_execution_dir=validation_dir,
        rules=rules,
        initial_capital=Decimal("100000"),
        paths=1,
        seed=1,
        challenge_horizon_ns=_YEAR_NS,
        verification_horizon_ns=_YEAR_NS,
        derive_seed=_derive_seed,
    )

    assert len(seen_trade_sets) == 1
    assert seen_trade_sets[0] == development_path.trades
    assert seen_trade_sets[0] != validation_path.trades


def test_load_validation_trade_path_rejects_a_development_partition(
    tmp_path: Path,
) -> None:
    execution_dir = write_fixture_execution_dir(
        tmp_path / "development_execution", partition="development"
    )
    with pytest.raises(PathExtractionError, match="validation"):
        load_validation_trade_path(execution_dir)


def test_load_validation_trade_path_loads_the_real_validation_artifact() -> None:
    path = load_validation_trade_path(REAL_VALIDATION_DIR)
    assert len(path.trades) == 253


def test_real_validation_trades_never_reach_the_final_holdout() -> None:
    from ftmoquant.research.stage_g import HOLDOUT_START

    holdout_start_ns = int(HOLDOUT_START.timestamp() * 1_000_000_000)
    path = load_validation_trade_path(REAL_VALIDATION_DIR)
    assert all(trade.exit_ns < holdout_start_ns for trade in path.trades)


def test_run_validation_diagnostic_is_deterministic_on_a_tiny_fixture(
    tmp_path: Path,
) -> None:
    development_dir = write_fixture_execution_dir(
        tmp_path / "development_execution", rows=_many_rows(20)
    )
    validation_dir = write_fixture_execution_dir(
        tmp_path / "validation_execution", partition="validation", rows=_many_rows(15)
    )
    rules = load_prop_rule_set(RULE_CONFIG)

    def _run():
        return vd.run_validation_diagnostic(
            development_execution_dir=development_dir,
            validation_execution_dir=validation_dir,
            rules=rules,
            initial_capital=Decimal("100000"),
            paths=5,
            seed=42,
            challenge_horizon_ns=_YEAR_NS,
            verification_horizon_ns=_YEAR_NS,
            derive_seed=_derive_seed,
        )

    first = _run()
    second = _run()
    assert first.summary.pass_both == second.summary.pass_both
    assert first.block_length.frozen_block_size == second.block_length.frozen_block_size


def test_validation_diagnostic_artifact_is_write_once(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    common_args = [
        "--development-execution-dir",
        str(REAL_DEVELOPMENT_DIR),
        "--validation-execution-dir",
        str(REAL_VALIDATION_DIR),
        "--output-dir",
        str(output_dir),
        "--paths",
        "2",
        "--seed",
        "1",
    ]
    validation_diagnostic_main(common_args)
    written = output_dir / "frozen_policy_validation_diagnostic.json"
    assert written.is_file()

    with pytest.raises(ArtifactError):
        validation_diagnostic_main(common_args)
