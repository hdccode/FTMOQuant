from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from ftmoquant.data.instruments import OANDA_ALPHA_LAB_SPECS
from ftmoquant.research.alpha_lab import batch4_development_orchestrator as module
from ftmoquant.research.alpha_lab.batch4_clock_scheduler import load_frozen_clock_specs
from ftmoquant.research.alpha_lab.batch4_development_orchestrator import (
    FROZEN_PREREGISTRATION_SHA256,
    Batch4OrchestratorError,
    build_parser,
    reject_forbidden_root,
    reserve_output_directory,
    run_batch4_development_screen,
    run_instrument_job,
    verify_frozen_methodology,
)
from ftmoquant.research.alpha_lab.batch4_screen import (
    evaluate_hypothesis,
    load_frozen_screen_policy,
)


def _synthetic_m1(instrument_id: str, root: Path, start_utc, end_exclusive_utc):
    del instrument_id, root, end_exclusive_utc
    index = pd.date_range(start_utc, periods=2_000, freq="min", tz="UTC")
    values = 1.1 + np.linspace(0, 0.001, len(index))
    bid = pd.DataFrame(
        {"open": values, "high": values, "low": values, "close": values},
        index=index,
    )
    ask = pd.DataFrame(
        {
            "open": values + 0.0002,
            "high": values + 0.0002,
            "low": values + 0.0002,
            "close": values + 0.0002,
        },
        index=index,
    )
    return bid, ask


def test_frozen_methodology_verifies_91_and_exact_family_counts() -> None:
    specs, policy, document = verify_frozen_methodology()
    assert len(specs) == 91
    assert [
        sum(spec.family == family for spec in specs) for family in module.FAMILIES
    ] == [
        7,
        42,
        42,
    ]
    assert policy.min_trade_count == 250
    assert document["preregistration_semantic_sha256"] == FROZEN_PREREGISTRATION_SHA256


@pytest.mark.parametrize(
    "token", ["validation", "VALIDATION", "holdout", "final_holdout", "final_test"]
)
def test_forbidden_roots_are_rejected(token: str, tmp_path: Path) -> None:
    with pytest.raises(Batch4OrchestratorError, match="forbidden"):
        reject_forbidden_root(tmp_path / token, label="test")


def test_existing_output_refused() -> None:
    with pytest.raises(Batch4OrchestratorError, match="already exists"):
        reserve_output_directory(Path("."))


def test_output_refusal_happens_before_any_data_loader(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    readiness = tmp_path / "readiness.json"
    readiness.write_text("{}", encoding="utf-8")
    with patch.object(module, "load_alpha_lab_dataset") as loader:
        with pytest.raises(Batch4OrchestratorError, match="already exists"):
            run_batch4_development_screen(
                development_root=tmp_path / "canonical",
                universe_readiness=readiness,
                output_dir=output,
            )
        loader.assert_not_called()


def test_validation_root_refused_before_any_data_loader(tmp_path: Path) -> None:
    readiness = tmp_path / "readiness.json"
    readiness.write_text("{}", encoding="utf-8")
    with patch.object(module, "load_alpha_lab_dataset") as loader:
        with pytest.raises(Batch4OrchestratorError, match="forbidden"):
            run_batch4_development_screen(
                development_root=tmp_path / "validation" / "canonical",
                universe_readiness=readiness,
                output_dir=tmp_path / "fresh",
            )
        loader.assert_not_called()


def test_one_load_two_widens_three_identical_schedule_passes_per_instrument(
    tmp_path: Path,
) -> None:
    specs, policy, _ = verify_frozen_methodology()
    instrument = OANDA_ALPHA_LAB_SPECS[0]
    occurrence_ids: list[int] = []

    def execution_spy(occurrences, **kwargs):
        del kwargs
        occurrence_ids.append(id(occurrences))
        return (), ()

    with (
        patch.object(module, "load_m1_bidask", side_effect=_synthetic_m1) as loader,
        patch.object(
            module, "widen_bid_ask_frame", wraps=module.widen_bid_ask_frame
        ) as widen,
        patch.object(
            module, "execute_scheduled_occurrences", side_effect=execution_spy
        ) as execute,
    ):
        rows, trades = run_instrument_job(
            instrument_spec=instrument,
            frozen_specs=specs,
            development_root=tmp_path,
            policy=policy,
        )
    assert loader.call_count == 1
    assert widen.call_count == 2
    assert execute.call_count == 3
    assert len(set(occurrence_ids)) == 1
    assert len(rows) == 13
    assert trades == {}


def _zero_rows_for_instrument(instrument_id: str):
    specs = [
        spec
        for spec in load_frozen_clock_specs()
        if spec.instrument_id == instrument_id
    ]
    policy = load_frozen_screen_policy()
    return tuple(
        evaluate_hypothesis(
            spec=spec,
            native_trades=(),
            native_skip_count=0,
            stressed_1_5x_trades=(),
            stressed_2_0x_trades=(),
            fold_boundaries=module.DEVELOPMENT_FOLD_BOUNDARIES,
            policy=policy,
        )
        for spec in specs
    )


def test_synthetic_end_to_end_produces_91_ordered_rows_and_all_artifacts(
    tmp_path: Path,
) -> None:
    readiness = tmp_path / "readiness.json"
    readiness.write_text(
        json.dumps({"semantic_sha256": "synthetic", "lineage_id": "synthetic"}),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    ids = tuple(sorted(spec.instrument_id for spec in OANDA_ALPHA_LAB_SPECS))

    def fake_job(*, instrument_spec, frozen_specs, development_root, policy):
        del frozen_specs, development_root, policy
        return _zero_rows_for_instrument(instrument_spec.instrument_id), {}

    with (
        patch.object(
            module,
            "load_alpha_lab_dataset",
            return_value=SimpleNamespace(instrument_ids=ids),
        ),
        patch.object(module, "run_instrument_job", side_effect=fake_job),
    ):
        run_batch4_development_screen(
            development_root=tmp_path / "canonical",
            universe_readiness=readiness,
            output_dir=output,
        )
    scorecard = pd.read_csv(output / "scorecard.csv")
    assert len(scorecard) == 91
    assert list(scorecard["hypothesis_id"]) == sorted(scorecard["hypothesis_id"])
    assert len(pd.read_csv(output / "family_summary.csv")) == 3
    assert len(pd.read_csv(output / "family_robustness.csv")) == 13
    selection = json.loads((output / "selection_summary.json").read_text())
    assert selection["selected_representative"] is None
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["hypothesis_count"] == 91
    assert metadata["validation_accessed"] is False
    assert metadata["holdout_accessed"] is False


def test_cli_has_only_three_allowed_arguments() -> None:
    parser = build_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings  # noqa: SLF001
    }
    assert options == {
        "-h",
        "--help",
        "--development-root",
        "--universe-readiness",
        "--output",
    }


def test_orchestrator_has_no_validation_loader_import_or_holdout_access() -> None:
    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert all("validation" not in name for name in imported)
    for token in ("load_validation_dataset", "oanda_alpha_lab_validation"):
        assert token not in source
