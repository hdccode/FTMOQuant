from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import ftmoquant.research.alpha_lab.b3f1_development_orchestrator as orchestrator
from ftmoquant.research.alpha_lab.b3f1_development_orchestrator import (
    FROZEN_PREREGISTRATION_SHA256,
    B3F1OrchestratorError,
    build_metadata,
    compute_pair_robustness,
    reject_forbidden_root,
    reserve_output_directory,
    run_b3f1_development_screen,
)
from ftmoquant.research.alpha_lab.b3f1_spread_screen import B3F1ScorecardRow
from ftmoquant.research.alpha_lab.b3f1_spread_signals import (
    FORMATION_WINDOWS,
    PAIR_UNIVERSE,
    Z_ENTRY_GRID,
    Z_STOP_GRID,
    B3F1Config,
    build_b3f1_grid,
    enumerate_candidate_pairs,
)
from ftmoquant.research.alpha_lab.data import AlphaLabDataset
from ftmoquant.research.stage_g import DEVELOPMENT_END_EXCLUSIVE, DEVELOPMENT_START

# ---------------------------------------------------------------------------
# Root / output rejection (must happen before any loader access)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "token",
    ["validation", "VALIDATION", "holdout", "final_holdout", "final_test"],
)
def test_reject_forbidden_root_catches_every_forbidden_token(
    token: str, tmp_path: Path
) -> None:
    path = tmp_path / f"oanda_{token}_v1"
    with pytest.raises(B3F1OrchestratorError, match="forbidden"):
        reject_forbidden_root(path, label="--development-root")


def test_reject_forbidden_root_accepts_a_clean_development_path(tmp_path: Path) -> None:
    reject_forbidden_root(tmp_path / "oanda_fx_alpha_lab_v1" / "canonical", label="x")


def test_reserve_output_directory_rejects_existing_path(tmp_path: Path) -> None:
    existing = tmp_path / "out"
    existing.mkdir()
    with pytest.raises(B3F1OrchestratorError, match="already exists"):
        reserve_output_directory(existing)


def test_reserve_output_directory_accepts_a_fresh_path(tmp_path: Path) -> None:
    reserve_output_directory(tmp_path / "does_not_exist_yet")


def test_output_refusal_happens_before_any_loader_access(tmp_path: Path) -> None:
    existing_output = tmp_path / "out"
    existing_output.mkdir()
    readiness = tmp_path / "readiness.json"
    readiness.write_text("{}", encoding="utf-8")
    development_root = tmp_path / "canonical"

    with patch.object(orchestrator, "load_alpha_lab_dataset") as loader_mock:
        with pytest.raises(B3F1OrchestratorError, match="already exists"):
            run_b3f1_development_screen(
                development_root=development_root,
                universe_readiness=readiness,
                output_dir=existing_output,
            )
        loader_mock.assert_not_called()


def test_forbidden_root_rejected_before_any_loader_access(tmp_path: Path) -> None:
    output_dir = tmp_path / "fresh_output"
    readiness = tmp_path / "readiness.json"
    readiness.write_text("{}", encoding="utf-8")
    bad_root = tmp_path / "oanda_validation_v1" / "canonical"

    with patch.object(orchestrator, "load_alpha_lab_dataset") as loader_mock:
        with pytest.raises(B3F1OrchestratorError, match="forbidden"):
            run_b3f1_development_screen(
                development_root=bad_root,
                universe_readiness=readiness,
                output_dir=output_dir,
            )
        loader_mock.assert_not_called()
    assert not output_dir.exists()


# ---------------------------------------------------------------------------
# Pair robustness (independent per pair; no global ranking)
# ---------------------------------------------------------------------------


def _row(
    formation_window: int, z_entry: Decimal, z_stop: Decimal, passed: bool
) -> B3F1ScorecardRow:
    return B3F1ScorecardRow(
        sleeve_id="s",
        config=B3F1Config(formation_window, z_entry, z_stop),
        native_trade_count=60,
        native_expectancy=Decimal("10") if passed else Decimal("-10"),
        native_profit_factor=Decimal("1.5") if passed else Decimal("0.5"),
        fold_positive_count=3,
        best_5pct_removed_expectancy=Decimal("5") if passed else Decimal("-5"),
        quarter_max_share=Decimal("0.2"),
        stressed_1_5x_trade_count=60,
        stressed_1_5x_expectancy=Decimal("5") if passed else Decimal("-5"),
        stressed_1_5x_profit_factor=Decimal("1.2"),
        hard_gates_passed=passed,
        rolling_30_median_expectancy=None,
        rolling_30_fraction_positive=None,
        rolling_50_median_expectancy=None,
        rolling_50_fraction_positive=None,
        monthly_max_share=None,
        largest_trade_share=None,
        pnl_skewness=None,
        pnl_kurtosis=None,
        rich_trade_count=0,
        rich_expectancy=None,
        rich_profit_factor=None,
        cheap_trade_count=0,
        cheap_expectancy=None,
        cheap_profit_factor=None,
    )


def test_pair_robustness_computed_independently_per_pair() -> None:
    grid = build_b3f1_grid()
    # pair_a: two adjacent passing cells -> survives the >=2 connected
    # region rule. pair_b: identical grid, nothing passes.
    rows_a = [
        _row(c.formation_window, c.z_entry, c.z_stop, i < 2) for i, c in enumerate(grid)
    ]
    rows_b = [_row(c.formation_window, c.z_entry, c.z_stop, False) for c in grid]
    result_a = compute_pair_robustness("pair_a", rows_a)
    result_b = compute_pair_robustness("pair_b", rows_b)
    assert result_a["survival_rule_passed"] != result_b["survival_rule_passed"]
    # Neither result references the other pair's data.
    assert result_a["sleeve_id"] == "pair_a"
    assert result_b["sleeve_id"] == "pair_b"


def test_pair_robustness_no_global_ranking_field() -> None:
    rows = [_row(FORMATION_WINDOWS[0], Z_ENTRY_GRID[0], Z_STOP_GRID[0], True)] * 18
    result = compute_pair_robustness("s", rows)
    forbidden_keys = {"rank", "global_rank", "best_pair", "score"}
    assert forbidden_keys.isdisjoint(result.keys())


def test_pair_robustness_tested_configuration_count_is_18() -> None:
    grid = build_b3f1_grid()
    rows = [_row(c.formation_window, c.z_entry, c.z_stop, False) for c in grid]
    result = compute_pair_robustness("s", rows)
    assert result["tested_configuration_count"] == 18


def test_pair_robustness_strongest_region_is_within_the_largest_component() -> None:
    rows = [
        _row(FORMATION_WINDOWS[0], Z_ENTRY_GRID[0], Z_STOP_GRID[0], True),
        _row(FORMATION_WINDOWS[0], Z_ENTRY_GRID[1], Z_STOP_GRID[0], True),
        _row(FORMATION_WINDOWS[0], Z_ENTRY_GRID[2], Z_STOP_GRID[1], True),  # isolated
    ]
    result = compute_pair_robustness("s", rows)
    assert result["largest_connected_region_size"] == 2
    assert result["strongest_region_formation_window"] == FORMATION_WINDOWS[0]
    assert result["strongest_region_z_stop"] == str(Z_STOP_GRID[0])


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_metadata_pins_the_exact_v2_preregistration_sha() -> None:
    metadata = build_metadata(
        readiness_document={
            "semantic_sha256": "abc",
            "lineage_id": "x",
            "alpha_lab_config_sha256": "y",
        },
        h1_row_count=8597,
        tested_pair_count=21,
        tested_config_count=378,
    )
    assert metadata["preregistration_semantic_sha256"] == FROZEN_PREREGISTRATION_SHA256
    assert metadata["preregistration_semantic_sha256"] == (
        "e5cd74527004585cfd24bea55549d84e9cb66b05ffc6498360dac5007b651f7c"
    )


def test_metadata_includes_grid_universe_and_library_versions() -> None:
    metadata = build_metadata(
        readiness_document={},
        h1_row_count=1,
        tested_pair_count=21,
        tested_config_count=378,
    )
    assert len(metadata["ordered_instrument_universe"]) == 7
    assert len(metadata["pair_ids"]) == 21
    assert metadata["grid"]["configs_per_pair"] == 18
    assert metadata["adf_settings"] == {
        "regression": "c",
        "autolag": "AIC",
        "p_value_threshold": 0.05,
    }
    for key in ("statsmodels", "scipy", "numpy", "pandas", "python"):
        assert key in metadata["library_versions"]
    assert metadata["validation_accessed"] is False
    assert metadata["holdout_accessed"] is False


# ---------------------------------------------------------------------------
# Synthetic end-to-end CLI smoke run (no real DEVELOPMENT data touched)
# ---------------------------------------------------------------------------


def _synthetic_dataset() -> AlphaLabDataset:
    n = 300
    idx = pd.date_range(DEVELOPMENT_START, periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(7)
    columns = {}
    base = 4.0
    drift = np.cumsum(rng.normal(scale=0.001, size=n))
    for offset, spec in enumerate(PAIR_UNIVERSE):
        columns[spec.instrument_id] = np.exp(
            base + 0.1 * offset + drift + rng.normal(scale=0.0005, size=n)
        )
    close = pd.DataFrame(columns, index=idx)
    return AlphaLabDataset(
        timeframe="H1",
        instrument_ids=tuple(spec.instrument_id for spec in PAIR_UNIVERSE),
        start_utc=DEVELOPMENT_START,
        end_exclusive_utc=DEVELOPMENT_END_EXCLUSIVE,
        alignment_policy="synthetic_test_fixture",
        open=close,
        high=close,
        low=close,
        close=close,
        bid=close,
        ask=close,
        spread=close * 0,
    )


def _synthetic_m1(instrument_id: str, root: Path, start_utc, end_exclusive_utc):
    # Deliberately small (not full-span): cost_stress.widen_bid_ask_frame's
    # cost is O(len(frame)) per call and this fixture is invoked 756 times
    # across the full smoke test (21 pairs x 18 configs x 2 legs) -- a
    # short M1 window keeps the WIRING smoke test fast. Entries beyond
    # this window legitimately skip with "no_later_m1_observation", an
    # already-covered code path; this test proves plumbing, not fill rate.
    n = 200
    idx = pd.date_range(start_utc, periods=n, freq="min", tz="UTC")
    rng = np.random.default_rng(abs(hash(instrument_id)) % (2**31))
    base = 1.1 + 0.01 * (abs(hash(instrument_id)) % 7)
    prices = base + np.cumsum(rng.normal(scale=0.00001, size=n))
    bid = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.00005,
            "low": prices - 0.00005,
            "close": prices,
        },
        index=idx,
    )
    ask = bid + 0.0002
    return bid, ask


def test_synthetic_cli_smoke_run_produces_exactly_378_scorecard_rows(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "b3f1_run"
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
    development_root = tmp_path / "canonical"

    with (
        patch.object(
            orchestrator, "load_alpha_lab_dataset", return_value=_synthetic_dataset()
        ),
        patch.object(orchestrator, "load_m1_bidask", side_effect=_synthetic_m1),
    ):
        run_b3f1_development_screen(
            development_root=development_root,
            universe_readiness=readiness,
            output_dir=output_dir,
            workers=1,
        )

    scorecard = pd.read_csv(output_dir / "scorecard.csv")
    assert len(scorecard) == 21 * 18

    pair_robustness = pd.read_csv(output_dir / "pair_robustness.csv")
    assert len(pair_robustness) == 21

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["preregistration_semantic_sha256"] == FROZEN_PREREGISTRATION_SHA256
    assert metadata["tested_config_count"] == 378


def test_synthetic_run_output_ordering_matches_pair_and_grid_order(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "b3f1_run"
    readiness = tmp_path / "readiness.json"
    readiness.write_text(json.dumps({}), encoding="utf-8")
    development_root = tmp_path / "canonical"

    with (
        patch.object(
            orchestrator, "load_alpha_lab_dataset", return_value=_synthetic_dataset()
        ),
        patch.object(orchestrator, "load_m1_bidask", side_effect=_synthetic_m1),
    ):
        run_b3f1_development_screen(
            development_root=development_root,
            universe_readiness=readiness,
            output_dir=output_dir,
            workers=1,
        )

    scorecard = pd.read_csv(output_dir / "scorecard.csv")
    expected_sleeve_order = [
        pair.sleeve_id for pair in enumerate_candidate_pairs() for _ in range(18)
    ]
    assert list(scorecard["sleeve_id"]) == expected_sleeve_order


def test_formation_series_computed_exactly_once_per_pair_per_window(
    tmp_path: Path,
) -> None:
    from ftmoquant.research.alpha_lab import b3f1_development_orchestrator as mod

    dataset = _synthetic_dataset()
    log_close = np.log(dataset.close)
    pair = enumerate_candidate_pairs()[0]
    job = mod._PairJob(
        pair=pair,
        log_y=log_close[pair.y.instrument_id],
        log_x=log_close[pair.x.instrument_id],
        development_root_dir=tmp_path,
    )

    with (
        patch.object(mod, "load_m1_bidask", side_effect=_synthetic_m1),
        patch.object(
            mod, "compute_formation_series", wraps=mod.compute_formation_series
        ) as formation_spy,
        patch.object(
            mod, "generate_b3f1_decisions", wraps=mod.generate_b3f1_decisions
        ) as decisions_spy,
    ):
        rows = mod.run_pair_job(job)

    assert len(rows) == 18
    assert formation_spy.call_count == len(FORMATION_WINDOWS)
    assert decisions_spy.call_count == 18


def test_same_decisions_reused_for_native_and_stressed_execution(
    tmp_path: Path,
) -> None:
    from ftmoquant.research.alpha_lab import b3f1_development_orchestrator as mod

    dataset = _synthetic_dataset()
    log_close = np.log(dataset.close)
    pair = enumerate_candidate_pairs()[0]
    job = mod._PairJob(
        pair=pair,
        log_y=log_close[pair.y.instrument_id],
        log_x=log_close[pair.x.instrument_id],
        development_root_dir=tmp_path,
    )
    seen_decisions: list[object] = []

    real_execute = mod.simulate_b3f1_intents

    def _spy_execute(decisions, **kwargs):
        seen_decisions.append(decisions)
        return real_execute(decisions, **kwargs)

    with (
        patch.object(mod, "load_m1_bidask", side_effect=_synthetic_m1),
        patch.object(mod, "simulate_b3f1_intents", side_effect=_spy_execute),
    ):
        mod.run_pair_job(job)

    # Each config calls execution exactly twice (native, 1.5x) with the
    # IDENTICAL decisions object both times.
    assert len(seen_decisions) == 36
    for native, stressed in zip(
        seen_decisions[0::2], seen_decisions[1::2], strict=True
    ):
        assert native is stressed


# ---------------------------------------------------------------------------
# Instrument-level stress cache: no redundant re-widening, deterministic,
# cleared between separate runs.
# ---------------------------------------------------------------------------


def test_stress_cache_eliminates_redundant_widening_within_a_pair(
    tmp_path: Path,
) -> None:
    from ftmoquant.research.alpha_lab import b3f1_development_orchestrator as mod

    dataset = _synthetic_dataset()
    log_close = np.log(dataset.close)
    pair = enumerate_candidate_pairs()[0]
    job = mod._PairJob(
        pair=pair,
        log_y=log_close[pair.y.instrument_id],
        log_x=log_close[pair.x.instrument_id],
        development_root_dir=tmp_path,
    )
    mod._stress_cache.clear()

    with (
        patch.object(mod, "load_m1_bidask", side_effect=_synthetic_m1),
        patch.object(
            mod, "widen_bid_ask_frame", wraps=mod.widen_bid_ask_frame
        ) as widen_spy,
    ):
        rows = mod.run_pair_job(job)

    assert len(rows) == 18
    # Exactly 2 widen calls for this pair's 18 configs (one per leg), not 36.
    assert widen_spy.call_count == 2


def test_stress_cache_is_reused_across_pairs_sharing_an_instrument(
    tmp_path: Path,
) -> None:
    from ftmoquant.research.alpha_lab import b3f1_development_orchestrator as mod

    dataset = _synthetic_dataset()
    log_close = np.log(dataset.close)
    pairs = [
        p
        for p in enumerate_candidate_pairs()
        if p.y.instrument_id == PAIR_UNIVERSE[0].instrument_id
    ][:3]
    assert len(pairs) == 3  # 3 pairs sharing PAIR_UNIVERSE[0] as the Y leg
    mod._stress_cache.clear()

    with (
        patch.object(mod, "load_m1_bidask", side_effect=_synthetic_m1),
        patch.object(
            mod, "widen_bid_ask_frame", wraps=mod.widen_bid_ask_frame
        ) as widen_spy,
    ):
        for pair in pairs:
            job = mod._PairJob(
                pair=pair,
                log_y=log_close[pair.y.instrument_id],
                log_x=log_close[pair.x.instrument_id],
                development_root_dir=tmp_path,
            )
            mod.run_pair_job(job)

    # 3 pairs x 2 legs = 6 legs total, but the shared Y instrument's widened
    # frame is computed once and reused -> 3 (shared Y) + 3 (distinct X's) = 4.
    assert widen_spy.call_count == 4


def test_stress_cache_output_is_deterministic(tmp_path: Path) -> None:
    from ftmoquant.research.alpha_lab import b3f1_development_orchestrator as mod

    dataset = _synthetic_dataset()
    log_close = np.log(dataset.close)
    pair = enumerate_candidate_pairs()[0]
    job = mod._PairJob(
        pair=pair,
        log_y=log_close[pair.y.instrument_id],
        log_x=log_close[pair.x.instrument_id],
        development_root_dir=tmp_path,
    )

    with patch.object(mod, "load_m1_bidask", side_effect=_synthetic_m1):
        mod._stress_cache.clear()
        first = mod.run_pair_job(job)
        mod._stress_cache.clear()
        second = mod.run_pair_job(job)

    assert first == second


def test_stress_cache_is_cleared_between_separate_orchestrator_runs(
    tmp_path: Path,
) -> None:
    """A stale cache from an earlier run must never silently leak into a
    later run against DIFFERENT data -- proven by running twice with two
    different M1 fixtures for the same instrument_ids and confirming the
    second run's own cache entries actually come from ITS OWN data (a
    leaking cache would keep the FIRST run's frames and this equality
    would spuriously still pass, so we assert the caches differ)."""

    from ftmoquant.research.alpha_lab import b3f1_development_orchestrator as mod

    def _synthetic_m1_variant(
        instrument_id: str, root: Path, start_utc, end_exclusive_utc
    ):
        bid, ask = _synthetic_m1(instrument_id, root, start_utc, end_exclusive_utc)
        return bid + 10.0, ask + 10.0  # obviously different price level

    readiness = tmp_path / "readiness.json"
    readiness.write_text(json.dumps({}), encoding="utf-8")
    development_root = tmp_path / "canonical"

    with patch.object(mod, "load_alpha_lab_dataset", return_value=_synthetic_dataset()):
        with patch.object(mod, "load_m1_bidask", side_effect=_synthetic_m1):
            run_b3f1_development_screen(
                development_root=development_root,
                universe_readiness=readiness,
                output_dir=tmp_path / "run_one",
                workers=1,
            )
            cache_after_run_one = dict(mod._stress_cache)

        with patch.object(mod, "load_m1_bidask", side_effect=_synthetic_m1_variant):
            run_b3f1_development_screen(
                development_root=development_root,
                universe_readiness=readiness,
                output_dir=tmp_path / "run_two",
                workers=1,
            )
            cache_after_run_two = dict(mod._stress_cache)

    some_key = next(iter(cache_after_run_one))
    run_one_bid = cache_after_run_one[some_key][0]
    run_two_bid = cache_after_run_two[some_key][0]
    # If run_two's cache had leaked run_one's stale entries instead of
    # being cleared and recomputed from run_two's own (shifted) data, these
    # would be identical -- they must not be.
    assert not run_one_bid["close"].equals(run_two_bid["close"])
    # And run_two's cache must reflect run_two's OWN data: its widened
    # close should sit ~10.0 above run_one's (the fixed offset the variant
    # M1 generator adds), not merely "different".
    assert (run_two_bid["close"] - run_one_bid["close"]).sub(10.0).abs().max() < 1e-6


# ---------------------------------------------------------------------------
# CLI surface: no grid/ADF/gate/cost overrides exist
# ---------------------------------------------------------------------------


def test_cli_exposes_no_frozen_parameter_overrides() -> None:
    parser = orchestrator.build_parser()
    option_strings = {
        option
        for action in parser._actions  # noqa: SLF001
        for option in action.option_strings
    }
    assert option_strings == {
        "-h",
        "--help",
        "--development-root",
        "--universe-readiness",
        "--output",
        "--workers",
    }


def test_cli_help_smoke() -> None:
    parser = orchestrator.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])
