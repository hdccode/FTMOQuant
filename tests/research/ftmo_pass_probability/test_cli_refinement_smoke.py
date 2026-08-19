"""Tiny synthetic smoke check for the sizing-screen-refinement CLI command.

Uses a tiny --paths count against the real (but cheap-to-load) DEVELOPMENT
artifact -- never the real 20,000-path sweep -- and writes to an isolated
tmp_path output directory so the real, already-produced
``sizing_screen.csv``/``sizing_screen_refinement.csv`` benchmark artifacts
are never touched by tests.
"""

from __future__ import annotations

import csv
from pathlib import Path

from ftmoquant.research.ftmo_pass_probability.cli import sizing_screen_refinement_main
from ftmoquant.research.ftmo_pass_probability.sizing import NOTIONAL_REFINEMENT_GRID

REAL_EXECUTION_DIR = Path(
    ".artifacts/usdcad_sweep_bos_retest_v1/development_execution"
).resolve()
RULE_CONFIG = Path("config/prop/ftmo_2step_swing_2026-08.yaml").resolve()


def test_sizing_screen_refinement_writes_a_separate_csv(tmp_path: Path) -> None:
    output_dir = tmp_path / "ftmo_pass_probability_smoke"
    sizing_screen_refinement_main(
        [
            "--execution-dir",
            str(REAL_EXECUTION_DIR),
            "--rule-config",
            str(RULE_CONFIG),
            "--output-dir",
            str(output_dir),
            "--paths",
            "2",
            "--seed",
            "1",
        ]
    )

    written = output_dir / "sizing_screen_refinement.csv"
    assert written.is_file()
    assert not (output_dir / "sizing_screen.csv").exists()

    with written.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    # 9 refinement candidates x 2 resampling methods, 2 replications each.
    assert len(rows) == len(NOTIONAL_REFINEMENT_GRID) * 2
    for row in rows:
        assert row["replications"] == "2"
        assert row["policy_id"].startswith("notional_refine_")
        assert row["method"] in ("stationary", "circular")
