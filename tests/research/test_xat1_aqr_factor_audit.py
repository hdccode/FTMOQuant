from __future__ import annotations

import ast
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from ftmoquant.research import xat1_aqr_factor_audit as audit


def _frame(start: str = "1985-01", periods: int = 24) -> pd.DataFrame:
    index = pd.period_range(start, periods=periods, freq="M")
    values = np.linspace(-0.02, 0.03, periods)
    return pd.DataFrame(
        {
            "all_assets": values,
            "equities": values + 0.001,
            "currencies": values - 0.001,
            "fixed_income": values + 0.002,
            "commodities": values - 0.002,
        },
        index=index,
    )


def test_percent_decimal_unit_detection_and_failure() -> None:
    decimals = np.array([[0.01, -0.04], [0.03, 0.02]])
    percentages = np.array([[1.0, -4.0], [3.0, 2.0]])
    assert audit.detect_return_units(decimals) == "decimal"
    assert audit.detect_return_units(percentages) == "percent"
    assert (
        audit.detect_return_units(percentages, number_formats=("0.00%", "0.000000%"))
        == "decimal"
    )
    with pytest.raises(audit.Xat1AqrAuditError, match="ambiguous"):
        audit.detect_return_units(np.array([[1.0, -0.75]]))
    assert audit.detect_return_units(decimals, number_formats=("General",)) == "decimal"
    with pytest.raises(audit.Xat1AqrAuditError, match="mixed unit evidence"):
        audit.detect_return_units(decimals, number_formats=("General", "0.00%"))


def test_percent_values_are_converted_once() -> None:
    frame = _frame(periods=3) * 100.0
    validated = audit.validate_factor_frame(frame, units="percent")
    np.testing.assert_allclose(validated, _frame(periods=3))


@pytest.mark.parametrize(
    "bad_index,message",
    [
        (["1985-01", "not-a-date"], "malformed date"),
        (["1985-01", "1985-01"], "duplicate month"),
        (["1985-01", "1985-03"], "missing month"),
    ],
)
def test_date_validation_fails_closed(bad_index: list[str], message: str) -> None:
    frame = _frame(periods=2)
    frame.index = pd.Index(bad_index)
    with pytest.raises(audit.Xat1AqrAuditError, match=message):
        audit.validate_factor_frame(frame, units="decimal")


def test_wrong_columns_and_missing_primary_fail_closed() -> None:
    wrong = _frame().rename(columns={"commodities": "rates"})
    with pytest.raises(audit.Xat1AqrAuditError, match="wrong factor columns"):
        audit.validate_factor_frame(wrong, units="decimal")
    missing_primary = _frame().drop(columns="all_assets")
    with pytest.raises(audit.Xat1AqrAuditError, match="missing primary factor"):
        audit.validate_factor_frame(missing_primary, units="decimal")


def test_annual_hurdle_conversion_is_frozen() -> None:
    assert audit.annual_drag_to_monthly(0.0) == 0.0
    assert audit.annual_drag_to_monthly(0.01) == pytest.approx(0.01 / 12.0)
    assert audit.annual_drag_to_monthly(0.02) == pytest.approx(0.02 / 12.0)
    assert audit.annual_drag_to_monthly(0.03) == pytest.approx(0.03 / 12.0)
    with pytest.raises(audit.Xat1AqrAuditError, match="frozen hurdles"):
        audit.annual_drag_to_monthly(0.025)


def test_exact_frozen_period_boundaries_and_no_endogenous_dates() -> None:
    assert audit.PERIODS == {
        "SOURCE_SAMPLE": ("1985-01", "2009-12"),
        "BRIDGE": ("2010-01", "2012-12"),
        "STRICT_POST_PUBLICATION": ("2013-01", None),
        "EARLY_POST_PUBLICATION": ("2013-01", "2019-12"),
        "RECENT": ("2020-01", None),
    }
    frame = _frame(periods=12 * 42)
    expected = {
        "SOURCE_SAMPLE": ("1985-01", "2009-12", 300),
        "BRIDGE": ("2010-01", "2012-12", 36),
        "STRICT_POST_PUBLICATION": ("2013-01", "2026-12", 168),
        "EARLY_POST_PUBLICATION": ("2013-01", "2019-12", 84),
        "RECENT": ("2020-01", "2026-12", 84),
    }
    for name, (start, end, count) in expected.items():
        sliced = audit.slice_frozen_period(frame, name)
        assert (str(sliced.index[0]), str(sliced.index[-1]), len(sliced)) == (
            start,
            end,
            count,
        )
    with pytest.raises(audit.Xat1AqrAuditError, match="not frozen"):
        audit.slice_frozen_period(frame, "BEST_LOOKING_PERIOD")


def _passing_gate_inputs() -> audit.GateInputs:
    return audit.GateInputs(
        strict_mean_2pct=0.01,
        early_mean_2pct=0.01,
        recent_mean_2pct=0.01,
        strict_mean_3pct=0.01,
        strict_asset_class_means_2pct=(0.01, 0.01, 0.01, 0.01),
        strict_bootstrap_lower_2pct=0.001,
    )


@pytest.mark.parametrize(
    "gate,inputs",
    [
        (1, replace(_passing_gate_inputs(), strict_mean_2pct=0.0)),
        (2, replace(_passing_gate_inputs(), early_mean_2pct=0.0)),
        (3, replace(_passing_gate_inputs(), recent_mean_2pct=0.0)),
        (4, replace(_passing_gate_inputs(), strict_mean_3pct=0.0)),
        (
            5,
            replace(
                _passing_gate_inputs(),
                strict_asset_class_means_2pct=(0.01, 0.01, 0.0, -0.01),
            ),
        ),
        (6, replace(_passing_gate_inputs(), strict_bootstrap_lower_2pct=0.0)),
    ],
)
def test_each_gate_fails_independently(gate: int, inputs: audit.GateInputs) -> None:
    results = audit.evaluate_gates(inputs)
    assert [result.gate for result in results if not result.passed] == [gate]


def test_all_six_gates_pass_together() -> None:
    assert all(result.passed for result in audit.evaluate_gates(_passing_gate_inputs()))


def test_asset_class_winner_cannot_replace_all_assets_primary() -> None:
    inputs = replace(
        _passing_gate_inputs(),
        strict_mean_2pct=-0.01,
        strict_asset_class_means_2pct=(0.20, 0.18, 0.16, 0.14),
    )
    results = audit.evaluate_gates(inputs)
    assert not results[0].passed
    assert results[4].passed
    assert not all(result.passed for result in results)


def test_stationary_bootstrap_is_deterministic() -> None:
    frame = _frame(start="2013-01", periods=48)
    first = audit._bootstrap_gross(frame, "STRICT_POST_PUBLICATION", "all_assets")
    second = audit._bootstrap_gross(frame, "STRICT_POST_PUBLICATION", "all_assets")
    assert first == second


def test_original_updated_reconciliation_lists_difference_months() -> None:
    original = _frame(periods=300)
    updated = original.copy()
    updated.loc[pd.Period("1985-03", "M"), "all_assets"] += 0.001
    updated.loc[pd.Period("2009-12", "M"), "equities"] -= 0.002
    result = audit.reconcile_original(updated, original)
    assert result["months"] == 300
    assert result["factors"]["all_assets"] == {
        "exact_equality": False,
        "exact_equal_months": 299,
        "months_with_differences": 1,
        "difference_months": ["1985-03"],
        "maximum_absolute_difference": pytest.approx(0.001),
        "correlation": pytest.approx(
            np.corrcoef(updated["all_assets"], original["all_assets"])[0, 1]
        ),
    }
    assert result["factors"]["currencies"]["exact_equality"] is True
    assert result["factors"]["currencies"]["difference_months"] == []


def test_result_artifact_is_write_once(tmp_path: Path) -> None:
    output_dir = tmp_path / "deterministic-audit"
    output = audit.write_once_artifact(output_dir, {"decision": "RETIRE_XAT1"})
    assert json.loads(output.read_text(encoding="utf-8")) == {"decision": "RETIRE_XAT1"}
    with pytest.raises(FileExistsError):
        audit.write_once_artifact(output_dir, {"decision": "changed"})


def test_raw_workbooks_are_external_and_never_written_by_module() -> None:
    repo = Path(__file__).resolve().parents[2]
    data_root = Path("/Users/Shared/FTMOQuant-data/xat1_aqr_tsmom_v1")
    assert not data_root.is_relative_to(repo)
    source = Path(audit.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    writes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"write_bytes", "write_text"}
    ]
    assert len(writes) == 1
    assert "output.write_text" in ast.get_source_segment(source, writes[0])


def test_module_cannot_access_ftmo_validation_or_holdout_artifacts() -> None:
    source = Path(audit.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert (
        imports
        & {
            "ftmoquant.data",
            "ftmoquant.strategies",
            "ftmoquant.research.alpha_lab.validation",
            "ftmoquant.research.ftmo_joint_frontier_validation",
        }
        == set()
    )
    assert audit.run_audit.__code__.co_argcount == 2
