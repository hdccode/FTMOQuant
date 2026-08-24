from __future__ import annotations

import ast
from datetime import date
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ftmoquant.research.alpha_lab.batch5_development_orchestrator as mod
from ftmoquant.research.alpha_lab.batch5_daily import (
    build_completed_fx_days,
    ny_fx_boundary,
)
from ftmoquant.research.alpha_lab.batch5_development_orchestrator import (
    Batch5DevelopmentOrchestratorError,
    CostFrameCache,
    benchmark_synthetic_runtime,
    build_parser,
    run_batch5_development_screen,
    verify_frozen_methodology,
    verify_preflight,
)
from ftmoquant.research.alpha_lab.batch5_preregistration import PRIMARY_FAMILIES


def _frames(index: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = np.linspace(1, 1.001, len(index))
    bid = pd.DataFrame(
        {"open": base, "high": base, "low": base, "close": base}, index=index
    )
    ask = bid + 0.0002
    return bid, ask


def test_frozen_methodology_and_cli_are_exact() -> None:
    document = verify_frozen_methodology()
    assert tuple(document["family_scope"]["primary_exact"]) == PRIMARY_FAMILIES
    parser = build_parser()
    options = {action.dest for action in parser._actions} - {"help"}
    assert options == {
        "development_root",
        "universe_readiness",
        "batch5_cross_root",
        "cftc_root",
        "output",
    }


def test_cost_cache_loads_once_and_builds_exact_three_states() -> None:
    calls: list[tuple[str, Decimal]] = []
    index = pd.date_range(mod.DEVELOPMENT_START, periods=20, freq="min", tz="UTC")

    def loader(**kwargs: object) -> tuple[pd.DataFrame, pd.DataFrame]:
        calls.append((str(kwargs["instrument_id"]), Decimal("1")))
        return _frames(index)

    cache = CostFrameCache(
        development_root=Path("canonical"),
        batch5_cross_root=Path("crosses"),
        loader=loader,
    )
    native = cache.frames("EUR/USD.OANDA", Decimal("1.0"))
    stress_1 = cache.frames("EUR/USD.OANDA", Decimal("1.5"))
    stress_2 = cache.frames("EUR/USD.OANDA", Decimal("2.0"))
    assert len(calls) == 1
    assert cache.frames("EUR/USD.OANDA", Decimal("1.5")) is stress_1
    assert set(cache.widen_count) == {
        ("EUR/USD.OANDA", Decimal("1.5")),
        ("EUR/USD.OANDA", Decimal("2.0")),
    }
    assert native[0].index.equals(stress_1[0].index)
    assert stress_1[0].index.equals(stress_2[0].index)
    assert stress_2[0]["close"].iloc[0] < stress_1[0]["close"].iloc[0]


def test_completed_day_builder_requires_exact_dst_safe_native_boundaries() -> None:
    first_date = date(2021, 3, 13)
    second_date = date(2021, 3, 14)
    index = pd.DatetimeIndex(
        [
            ny_fx_boundary(first_date) - pd.Timedelta(minutes=1),
            ny_fx_boundary(second_date) - pd.Timedelta(minutes=1),
        ]
    )
    bid, ask = _frames(index)
    days = build_completed_fx_days("AUD/CAD.OANDA", bid, ask)
    assert len(days) == 1
    assert days[0].start_utc == ny_fx_boundary(first_date)
    assert days[0].end_utc == ny_fx_boundary(second_date)
    assert (days[0].end_utc - days[0].start_utc).total_seconds() == 23 * 3600


def test_write_once_refusal_precedes_preflight_and_data_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    called = False

    def forbidden(**_: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("preflight must not run")

    monkeypatch.setattr(mod, "verify_preflight", forbidden)
    with pytest.raises(
        Batch5DevelopmentOrchestratorError, match="refusing to overwrite"
    ):
        run_batch5_development_screen(
            development_root=tmp_path / "canonical",
            universe_readiness=tmp_path / "readiness.json",
            batch5_cross_root=tmp_path / "crosses",
            cftc_root=tmp_path / "cftc",
            output_dir=output,
        )
    assert not called


@pytest.mark.parametrize("token", ["validation", "holdout", "final_holdout"])
def test_forbidden_roots_are_rejected_before_any_file_read(
    tmp_path: Path, token: str
) -> None:
    with pytest.raises(Batch5DevelopmentOrchestratorError, match="forbidden token"):
        verify_preflight(
            development_root=tmp_path / token / "canonical",
            universe_readiness=tmp_path / "missing.json",
            batch5_cross_root=tmp_path / "crosses",
            cftc_root=tmp_path / "cftc",
        )


def test_orchestrator_imports_no_validation_or_holdout_loader() -> None:
    source = Path(mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    assert not any("validation" in name or "holdout" in name for name in imported)


def test_metadata_only_runtime_benchmark_stays_below_two_hour_stop() -> None:
    result = benchmark_synthetic_runtime(10_000)
    assert result["estimated_total_minutes"] < 120
