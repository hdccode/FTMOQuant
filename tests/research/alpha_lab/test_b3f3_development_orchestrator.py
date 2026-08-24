from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import ftmoquant.research.alpha_lab.b3f3_development_orchestrator as orchestrator
from ftmoquant.research.alpha_lab.b3f1_spread_signals import PAIR_UNIVERSE
from ftmoquant.research.alpha_lab.b3f3_development_orchestrator import (
    FROZEN_PREREGISTRATION_SHA256,
    B3F3OrchestratorError,
    build_metadata,
    reject_forbidden_root,
    reserve_output_directory,
    run_b3f3_development_screen,
)
from ftmoquant.research.alpha_lab.b3f3_session_open_mr_signals import build_b3f3_grid


def _synthetic_m1(instrument_id: str, root: Path, start_utc, end_exclusive_utc):
    n = 2000  # spans well past several days of session-open windows
    idx = pd.date_range(start_utc, periods=n, freq="min", tz="UTC")
    rng = np.random.default_rng(abs(hash(instrument_id)) % (2**31))
    base = 1.1 + 0.01 * (abs(hash(instrument_id)) % 7)
    prices = base + np.cumsum(rng.normal(scale=0.00002, size=n))
    bid = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.0001,
            "low": prices - 0.0001,
            "close": prices,
        },
        index=idx,
    )
    ask = bid + 0.0002
    return bid, ask


# ---------------------------------------------------------------------------
# Root / output rejection (must happen before any loader access)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "token", ["validation", "VALIDATION", "holdout", "final_holdout", "final_test"]
)
def test_reject_forbidden_root_catches_every_forbidden_token(
    token: str, tmp_path: Path
) -> None:
    with pytest.raises(B3F3OrchestratorError, match="forbidden"):
        reject_forbidden_root(
            tmp_path / f"oanda_{token}_v1", label="--development-root"
        )


def test_reserve_output_directory_rejects_existing_path(tmp_path: Path) -> None:
    existing = tmp_path / "out"
    existing.mkdir()
    with pytest.raises(B3F3OrchestratorError, match="already exists"):
        reserve_output_directory(existing)


def test_output_refusal_happens_before_any_loader_access(tmp_path: Path) -> None:
    existing_output = tmp_path / "out"
    existing_output.mkdir()
    readiness = tmp_path / "readiness.json"
    readiness.write_text("{}", encoding="utf-8")

    with patch.object(orchestrator, "load_alpha_lab_dataset") as loader_mock:
        with pytest.raises(B3F3OrchestratorError, match="already exists"):
            run_b3f3_development_screen(
                development_root=tmp_path / "canonical",
                universe_readiness=readiness,
                output_dir=existing_output,
            )
        loader_mock.assert_not_called()


def test_forbidden_root_rejected_before_any_loader_access(tmp_path: Path) -> None:
    readiness = tmp_path / "readiness.json"
    readiness.write_text("{}", encoding="utf-8")
    with patch.object(orchestrator, "load_alpha_lab_dataset") as loader_mock:
        with pytest.raises(B3F3OrchestratorError, match="forbidden"):
            run_b3f3_development_screen(
                development_root=tmp_path / "oanda_validation_v1" / "canonical",
                universe_readiness=readiness,
                output_dir=tmp_path / "fresh_output",
            )
        loader_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_metadata_pins_the_exact_v2_preregistration_sha() -> None:
    metadata = build_metadata(
        readiness_document={}, tested_instrument_count=7, tested_config_count=84
    )
    assert metadata["preregistration_semantic_sha256"] == FROZEN_PREREGISTRATION_SHA256
    assert metadata["preregistration_semantic_sha256"] == (
        "e5cd74527004585cfd24bea55549d84e9cb66b05ffc6498360dac5007b651f7c"
    )


def test_metadata_validation_and_holdout_flags_false() -> None:
    metadata = build_metadata(
        readiness_document={}, tested_instrument_count=7, tested_config_count=84
    )
    assert metadata["validation_accessed"] is False
    assert metadata["holdout_accessed"] is False


def test_metadata_session_definitions_match_frozen_clock() -> None:
    metadata = build_metadata(
        readiness_document={}, tested_instrument_count=7, tested_config_count=84
    )
    assert metadata["session_definitions"]["session_open_local"] == "08:00:00"
    assert metadata["session_definitions"]["entry_window_local"] == [
        "08:05:00",
        "09:00:00",
    ]
    assert metadata["session_definitions"]["time_exit_local"] == "11:00:00"


# ---------------------------------------------------------------------------
# Synthetic end-to-end CLI smoke run (no real DEVELOPMENT data touched)
# ---------------------------------------------------------------------------


def test_synthetic_smoke_run_produces_exactly_84_scorecard_rows(tmp_path: Path) -> None:
    output_dir = tmp_path / "b3f3_run"
    readiness = tmp_path / "readiness.json"
    readiness.write_text(
        json.dumps(
            {
                "semantic_sha256": "fake",
                "lineage_id": "fake",
                "alpha_lab_config_sha256": "fake",
            }
        ),
        encoding="utf-8",
    )

    with (
        patch.object(orchestrator, "load_alpha_lab_dataset", return_value=None),
        patch.object(orchestrator, "load_m1_bidask", side_effect=_synthetic_m1),
    ):
        run_b3f3_development_screen(
            development_root=tmp_path / "canonical",
            universe_readiness=readiness,
            output_dir=output_dir,
        )

    scorecard = pd.read_csv(output_dir / "scorecard.csv")
    assert len(scorecard) == 7 * 12

    robustness = pd.read_csv(output_dir / "pair_robustness.csv")
    assert len(robustness) == 7

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["preregistration_semantic_sha256"] == FROZEN_PREREGISTRATION_SHA256
    assert metadata["tested_config_count"] == 84


def test_synthetic_run_output_ordering_matches_instrument_and_grid_order(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "b3f3_run"
    readiness = tmp_path / "readiness.json"
    readiness.write_text(json.dumps({}), encoding="utf-8")

    with (
        patch.object(orchestrator, "load_alpha_lab_dataset", return_value=None),
        patch.object(orchestrator, "load_m1_bidask", side_effect=_synthetic_m1),
    ):
        run_b3f3_development_screen(
            development_root=tmp_path / "canonical",
            universe_readiness=readiness,
            output_dir=output_dir,
        )

    scorecard = pd.read_csv(output_dir / "scorecard.csv")
    expected_order = [spec.instrument_id for spec in PAIR_UNIVERSE for _ in range(12)]
    assert list(scorecard["instrument_id"]) == expected_order


def test_no_redundant_m5_or_anchor_reconstruction_per_config(tmp_path: Path) -> None:
    from ftmoquant.research.alpha_lab import b3f3_development_orchestrator as mod

    spec = PAIR_UNIVERSE[0]
    with (
        patch.object(mod, "load_m1_bidask", side_effect=_synthetic_m1),
        patch.object(mod, "derive_m5_mid_ohlc", wraps=mod.derive_m5_mid_ohlc) as m5_spy,
        patch.object(
            mod, "compute_opening_anchors", wraps=mod.compute_opening_anchors
        ) as anchors_spy,
        patch.object(
            mod, "compute_lagged_atr", wraps=mod.compute_lagged_atr
        ) as atr_spy,
        patch.object(
            mod, "widen_bid_ask_frame", wraps=mod.widen_bid_ask_frame
        ) as widen_spy,
    ):
        rows = mod.run_instrument_job(spec=spec, development_root_dir=tmp_path)

    assert len(rows) == 12
    assert m5_spy.call_count == 1
    assert anchors_spy.call_count == 1
    assert atr_spy.call_count == 1
    # exactly 2 widen calls (1.5x, 2.0x), not one per config.
    assert widen_spy.call_count == 2


# ---------------------------------------------------------------------------
# CLI surface: no grid/gate/cost overrides exist
# ---------------------------------------------------------------------------


def test_cli_exposes_no_frozen_parameter_overrides() -> None:
    parser = orchestrator.build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings  # noqa: SLF001
    }
    assert option_strings == {
        "-h",
        "--help",
        "--development-root",
        "--universe-readiness",
        "--output",
    }


def test_cli_help_smoke() -> None:
    parser = orchestrator.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])


def test_grid_size_is_84_across_7_instruments() -> None:
    assert len(PAIR_UNIVERSE) == 7
    assert len(build_b3f3_grid()) == 12
    assert len(PAIR_UNIVERSE) * len(build_b3f3_grid()) == 84
