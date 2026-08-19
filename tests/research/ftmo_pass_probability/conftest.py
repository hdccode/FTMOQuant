from __future__ import annotations

import json
from pathlib import Path

import pytest

from ftmoquant.research.ftmo_pass_probability.path_extraction import (
    FROZEN_CANDIDATE_IDENTITY,
)

_CSV_HEADER = (
    "trade_index,direction,alpha_lab_entry_ns,alpha_lab_exit_ns,"
    "alpha_lab_entry_price,alpha_lab_exit_price,alpha_lab_exit_reason,"
    "nautilus_entry_ns,nautilus_exit_ns,nautilus_entry_price,nautilus_exit_price,"
    "nautilus_realized_pnl,nautilus_net_r"
)

_BASE_NS = 1_555_081_320_000_000_000
_DAY_NS = 86_400_000_000_000


def _row(
    index: int, entry_ns: int, exit_ns: int, exit_reason: str, pnl: str, net_r: str
) -> str:
    return (
        f"{index},1,{entry_ns},{exit_ns},1.3,1.3,{exit_reason},"
        f"{entry_ns},{exit_ns},1.3,1.3,{pnl},{net_r}"
    )


def write_fixture_execution_dir(
    root: Path,
    *,
    partition: str = "development",
    validation_accessed: bool = False,
    final_holdout_accessed: bool = False,
    identity: dict | None = None,
    rows: list[str] | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    if rows is None:
        rows = [
            _row(
                0,
                _BASE_NS,
                _BASE_NS + 3_600_000_000_000,
                "target",
                "657.00",
                "1.972972972972972972972972973",
            ),
            _row(
                1,
                _BASE_NS + 2 * _DAY_NS,
                _BASE_NS + 2 * _DAY_NS + 3_600_000_000_000,
                "stop",
                "-330.00",
                "-1.0",
            ),
        ]
    (root / "trades.csv").write_text(
        _CSV_HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8"
    )
    (root / "result.json").write_text(
        json.dumps({"partition": partition}), encoding="utf-8"
    )
    (root / "runtime_provenance.json").write_text(
        json.dumps(
            {
                "validation_accessed": validation_accessed,
                "final_holdout_accessed": final_holdout_accessed,
                "frozen_candidate_identity": identity
                if identity is not None
                else FROZEN_CANDIDATE_IDENTITY,
                "frozen_candidate_identity_sha256": "deadbeef",
            }
        ),
        encoding="utf-8",
    )
    return root


@pytest.fixture
def fixture_execution_dir(tmp_path: Path) -> Path:
    return write_fixture_execution_dir(tmp_path / "development_execution")
