"""Tests for the read-only DEVELOPMENT-vs-VALIDATION alpha diagnostic.

Kept minimal/high-value per the task's own list: holdout firewall,
partition acceptance, zero MC/bootstrap usage, zero sizing-policy
selection, subgroup eligibility, determinism, write-once behaviour, and
chronological order preservation.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from ftmoquant.prop_rules import load_prop_rule_set
from ftmoquant.research.ftmo_pass_probability import alpha_diagnostic as ad
from ftmoquant.research.ftmo_pass_probability.artifacts import ArtifactError
from ftmoquant.research.ftmo_pass_probability.cli import alpha_diagnostic_main
from ftmoquant.research.ftmo_pass_probability.path_extraction import (
    PathExtractionError,
    load_development_trade_path,
    load_validation_trade_path,
)
from ftmoquant.research.stage_g import HOLDOUT_START

from .conftest import _BASE_NS, _DAY_NS, _row, write_fixture_execution_dir

RULE_CONFIG = Path("config/prop/ftmo_2step_swing_2026-08.yaml").resolve()
REAL_DEVELOPMENT_DIR = Path(
    ".artifacts/usdcad_sweep_bos_retest_v1/development_execution"
).resolve()
REAL_VALIDATION_DIR = Path(
    ".artifacts/usdcad_sweep_bos_retest_v1/validation_execution"
).resolve()
_HOLDOUT_START_NS = int(HOLDOUT_START.timestamp() * 1_000_000_000)


def _many_rows(count: int) -> list[str]:
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


def _tiny_fixture_dirs(tmp_path: Path) -> tuple[Path, Path]:
    development = write_fixture_execution_dir(
        tmp_path / "development_execution", rows=_many_rows(12)
    )
    validation = write_fixture_execution_dir(
        tmp_path / "validation_execution", partition="validation", rows=_many_rows(9)
    )
    return development, validation


def test_final_holdout_trade_cannot_be_loaded(tmp_path: Path) -> None:
    entry_ns = _HOLDOUT_START_NS - 3_600_000_000_000
    rows = [_row(0, entry_ns, _HOLDOUT_START_NS, "target", "657.00", "1.9")]
    execution_dir = write_fixture_execution_dir(
        tmp_path / "validation_execution", partition="validation", rows=rows
    )
    with pytest.raises(PathExtractionError, match="holdout"):
        load_validation_trade_path(execution_dir)


def test_only_development_and_validation_partitions_are_accepted(
    tmp_path: Path,
) -> None:
    development, validation = _tiny_fixture_dirs(tmp_path)
    # a VALIDATION-labelled artifact is rejected by the DEVELOPMENT loader...
    with pytest.raises(PathExtractionError):
        load_development_trade_path(validation)
    # ...and a DEVELOPMENT-labelled artifact is rejected by the VALIDATION loader.
    with pytest.raises(PathExtractionError):
        load_validation_trade_path(development)
    other = write_fixture_execution_dir(
        tmp_path / "holdout_execution", partition="holdout", rows=_many_rows(9)
    )
    with pytest.raises(PathExtractionError):
        load_development_trade_path(other)
    with pytest.raises(PathExtractionError):
        load_validation_trade_path(other)


def test_no_bootstrap_function_is_ever_called(monkeypatch, tmp_path: Path) -> None:
    from ftmoquant.research.ftmo_pass_probability import bootstrap as bootstrap_module

    def _fail(*args: object, **kwargs: object) -> object:
        raise AssertionError("bootstrap.draw_index_path must never be called here")

    monkeypatch.setattr(bootstrap_module, "draw_index_path", _fail)

    development, validation = _tiny_fixture_dirs(tmp_path)
    dev_path = load_development_trade_path(development)
    val_path = load_validation_trade_path(validation)
    rules = load_prop_rule_set(RULE_CONFIG)

    dev_pnl = ad.frozen_policy_pnl_series(dev_path.trades)
    val_pnl = ad.frozen_policy_pnl_series(val_path.trades)
    ad.diagnostic_a_trade_distribution(dev_path.trades, dev_pnl)
    ad.diagnostic_a_trade_distribution(val_path.trades, val_pnl)
    ad.diagnostic_b_chronological_path(dev_path.trades, dev_pnl, Decimal("100000"))
    ad.diagnostic_c_temporal_stability(dev_path.trades, dev_pnl)
    ad.diagnostic_d_subgroups(development, dev_path.trades, dev_pnl)
    ad.diagnostic_e_chronological_replay(dev_path.trades, rules, Decimal("100000"))
    ad.diagnostic_e_chronological_replay(val_path.trades, rules, Decimal("100000"))
    # reaching here without the monkeypatched assertion firing is the proof.


def test_module_never_imports_sizing_selection_or_ranking_machinery() -> None:
    forbidden = ("SIZING_GRID", "NOTIONAL_REFINEMENT_GRID", "rank_policies")
    for name in forbidden:
        assert not hasattr(ad, name), (
            f"{name} must not be reachable from alpha_diagnostic"
        )
    # exactly one policy is ever used, regardless of trade content.
    assert ad.frozen_policy().policy_id == "fixed_notional_2_0x"


def test_subgroup_eligibility_lists_only_preregistered_dimensions() -> None:
    dimensions = {
        entry["name"]: entry["eligible"] for entry in ad.subgroup_eligibility()
    }
    assert dimensions == {
        "direction": True,
        "exit_reason": False,
        "session": False,
        "regime_state": False,
        "setup_category": False,
    }


def test_diagnostic_d_only_ever_produces_long_and_short_groups(tmp_path: Path) -> None:
    development, _ = _tiny_fixture_dirs(tmp_path)
    path = load_development_trade_path(development)
    pnl = ad.frozen_policy_pnl_series(path.trades)
    result = ad.diagnostic_d_subgroups(development, path.trades, pnl)
    assert result is not None
    assert set(result.keys()) == {"long", "short"}


def test_diagnostics_are_deterministic_on_a_tiny_fixture(tmp_path: Path) -> None:
    development, _ = _tiny_fixture_dirs(tmp_path)
    path = load_development_trade_path(development)
    rules = load_prop_rule_set(RULE_CONFIG)

    def _run():
        pnl = ad.frozen_policy_pnl_series(path.trades)
        distribution = ad.diagnostic_a_trade_distribution(path.trades, pnl)
        chronological, _series = ad.diagnostic_b_chronological_path(
            path.trades, pnl, Decimal("100000")
        )
        replay = ad.diagnostic_e_chronological_replay(
            path.trades, rules, Decimal("100000")
        )
        return distribution, chronological, replay

    first = _run()
    second = _run()
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert first[2].passed_both == second[2].passed_both
    assert first[2].challenge.status == second[2].challenge.status


def test_alpha_diagnostic_artifact_is_write_once(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    args = [
        "--development-execution-dir",
        str(REAL_DEVELOPMENT_DIR),
        "--validation-execution-dir",
        str(REAL_VALIDATION_DIR),
        "--output-dir",
        str(output_dir),
    ]
    alpha_diagnostic_main(args)
    written = output_dir / "development_validation_alpha_diagnostic.json"
    assert written.is_file()

    with pytest.raises(ArtifactError):
        alpha_diagnostic_main(args)


def test_chronological_replay_uses_real_trade_order_not_a_resampled_one(
    monkeypatch, tmp_path: Path
) -> None:
    development, _ = _tiny_fixture_dirs(tmp_path)
    path = load_development_trade_path(development)
    rules = load_prop_rule_set(RULE_CONFIG)

    captured: list[tuple] = []
    real_size_synthetic_path = ad.size_synthetic_path

    def _spy(policy, placements, initial_capital):
        captured.append(placements)
        return real_size_synthetic_path(policy, placements, initial_capital)

    monkeypatch.setattr(ad, "size_synthetic_path", _spy)

    ad.diagnostic_e_chronological_replay(path.trades, rules, Decimal("100000"))

    assert captured, "size_synthetic_path was never called"
    first_call_placements = captured[0]
    expected_order = tuple(
        (trade, trade.entry_ns, trade.exit_ns) for trade in path.trades
    )
    assert first_call_placements == expected_order
