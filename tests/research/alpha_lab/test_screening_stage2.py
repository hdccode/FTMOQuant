from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ftmoquant.research.alpha_lab.data import ALIGNMENT_POLICY, AlphaLabDataset
from ftmoquant.research.alpha_lab.screening_batch import (
    ScreeningBatchResult,
    run_screening_batch,
)
from ftmoquant.research.alpha_lab.screening_common import (
    AGGREGATE_INSTRUMENT_LABEL,
    ScreeningResultRow,
)
from ftmoquant.research.alpha_lab.screening_stage2 import (
    DONCHIAN_FAMILY,
    DONCHIAN_H4_LOOKBACKS,
    MEAN_REVERSION_FAMILY,
    MEAN_REVERSION_LOOKBACKS,
    MEAN_REVERSION_Z_ENTRIES,
    TSM_FAMILY,
    TSM_H4_LOOKBACKS,
    analyze_family_robustness,
    build_stage2_grid,
    write_stage2_results,
)


def _row(
    *,
    family: str,
    strategy_id: str,
    parameters: dict[str, object],
    stressed_return: float = -0.01,
    profitable_pairs: int = 0,
    positive_folds: int = 0,
) -> ScreeningResultRow:
    return ScreeningResultRow(
        family=family,
        strategy_id=strategy_id,
        instrument=AGGREGATE_INSTRUMENT_LABEL,
        timeframe="H4",
        parameters=json.dumps(parameters, sort_keys=True),
        observation_count=1000,
        trade_count=10,
        cumulative_return=stressed_return,
        annualized_return=stressed_return,
        annualized_sharpe=0.1,
        maximum_drawdown=-0.05,
        win_rate=0.5,
        turnover_or_trade_activity=0.01,
        base_cost_return=stressed_return,
        stressed_cost_return=stressed_return,
        profitable_pair_count=profitable_pairs,
        positive_sharpe_pair_count=profitable_pairs,
        median_pair_sharpe=0.1,
        worst_pair_sharpe=-0.1,
        aggregate_equal_weight_return=stressed_return,
        aggregate_equal_weight_sharpe=0.1,
        positive_fold_count=positive_folds,
        worst_fold_return=-0.01,
        median_fold_sharpe=0.1,
        diagnostic_rank=None,
    )


def _all_1d_rows(
    family: str,
    param_name: str,
    values: tuple[int, ...],
    *,
    passing_values: set[int] = frozenset(),
) -> list[ScreeningResultRow]:
    rows = []
    for value in values:
        passing = value in passing_values
        rows.append(
            _row(
                family=family,
                strategy_id=f"{family}_{value}",
                parameters={param_name: value},
                stressed_return=0.01 if passing else -0.01,
                profitable_pairs=5 if passing else 1,
                positive_folds=3 if passing else 0,
            )
        )
    return rows


def _all_mean_reversion_rows(
    *, passing_cells: set[tuple[int, float]] = frozenset()
) -> list[ScreeningResultRow]:
    rows = []
    for lookback in MEAN_REVERSION_LOOKBACKS:
        for z_entry in MEAN_REVERSION_Z_ENTRIES:
            passing = (lookback, z_entry) in passing_cells
            rows.append(
                _row(
                    family=MEAN_REVERSION_FAMILY,
                    strategy_id=f"meanrev_{lookback}_{z_entry}",
                    parameters={"lookback": lookback, "z_entry": z_entry},
                    stressed_return=0.01 if passing else -0.01,
                    profitable_pairs=5 if passing else 1,
                    positive_folds=3 if passing else 0,
                )
            )
    return rows


def _neutral_donchian_and_tsm_rows() -> list[ScreeningResultRow]:
    return _all_1d_rows(
        DONCHIAN_FAMILY, "lookback", DONCHIAN_H4_LOOKBACKS
    ) + _all_1d_rows(TSM_FAMILY, "lookback", TSM_H4_LOOKBACKS)


def test_stage2_grid_matches_the_exact_specified_parameter_lists() -> None:
    grid = build_stage2_grid()
    assert len(grid) == 23
    assert len({config.strategy_id for config in grid}) == 23
    donchian = sorted(
        int(c.parameters["lookback"]) for c in grid if c.family == DONCHIAN_FAMILY
    )
    tsm = sorted(int(c.parameters["lookback"]) for c in grid if c.family == TSM_FAMILY)
    meanrev = sorted(
        (int(c.parameters["lookback"]), float(c.parameters["z_entry"]))
        for c in grid
        if c.family == MEAN_REVERSION_FAMILY
    )
    assert donchian == [35, 40, 45, 50, 55, 60, 70]
    assert tsm == [25, 30, 40, 50, 60, 75, 90]
    assert meanrev == sorted(
        (lookback, z)
        for lookback in (30, 40, 50)
        for z in (1.5, 2.0, 2.5)
    )
    non_meanrev_families = (DONCHIAN_FAMILY, TSM_FAMILY)
    assert all(c.timeframe == "H4" for c in grid if c.family in non_meanrev_families)
    assert all(c.timeframe == "H1" for c in grid if c.family == MEAN_REVERSION_FAMILY)


@pytest.mark.parametrize(
    "family,lookbacks,other_family,other_lookbacks,passing_values,"
    "expected_region_size,expected_region_count",
    [
        (
            DONCHIAN_FAMILY,
            DONCHIAN_H4_LOOKBACKS,
            TSM_FAMILY,
            TSM_H4_LOOKBACKS,
            {50, 55},
            2,
            1,
        ),
        (
            TSM_FAMILY,
            TSM_H4_LOOKBACKS,
            DONCHIAN_FAMILY,
            DONCHIAN_H4_LOOKBACKS,
            {25, 30, 40},
            3,
            1,
        ),
    ],
)
def test_1d_family_adjacency_rule_requires_two_adjacent_passing_lookbacks(
    family: str,
    lookbacks: Sequence[int],
    other_family: str,
    other_lookbacks: Sequence[int],
    passing_values: set[int],
    expected_region_size: int,
    expected_region_count: int,
) -> None:
    rows = _all_1d_rows(
        family, "lookback", lookbacks, passing_values=passing_values
    ) + _all_1d_rows(other_family, "lookback", other_lookbacks)
    summaries = {s.family: s for s in analyze_family_robustness(rows)}
    summary = summaries[family]
    assert summary.largest_contiguous_region_size == expected_region_size
    assert summary.contiguous_passing_region_count == expected_region_count
    assert summary.survival_rule_passed is True


@pytest.mark.parametrize(
    "family,lookbacks,other_family,other_lookbacks,passing_values",
    [
        (
            DONCHIAN_FAMILY,
            DONCHIAN_H4_LOOKBACKS,
            TSM_FAMILY,
            TSM_H4_LOOKBACKS,
            {50},
        ),
        (
            TSM_FAMILY,
            TSM_H4_LOOKBACKS,
            DONCHIAN_FAMILY,
            DONCHIAN_H4_LOOKBACKS,
            {60},
        ),
    ],
)
def test_isolated_1d_optimum_fails_family_survival(
    family: str,
    lookbacks: Sequence[int],
    other_family: str,
    other_lookbacks: Sequence[int],
    passing_values: set[int],
) -> None:
    rows = _all_1d_rows(
        family, "lookback", lookbacks, passing_values=passing_values
    ) + _all_1d_rows(other_family, "lookback", other_lookbacks)
    summaries = {s.family: s for s in analyze_family_robustness(rows)}
    summary = summaries[family]
    assert summary.passing_configuration_count == 1
    assert summary.largest_contiguous_region_size == 1
    assert summary.survival_rule_passed is False


def test_mean_reversion_connected_region_rule_requires_three_connected_cells() -> None:
    passing = {(30, 1.5), (40, 1.5), (40, 2.0)}  # rook-connected L-shape
    rows = _neutral_donchian_and_tsm_rows()
    rows += _all_mean_reversion_rows(passing_cells=passing)
    summaries = {s.family: s for s in analyze_family_robustness(rows)}
    meanrev = summaries[MEAN_REVERSION_FAMILY]
    assert meanrev.largest_contiguous_region_size == 3
    assert meanrev.survival_rule_passed is True


def test_mean_reversion_diagonal_only_cells_do_not_count_as_connected() -> None:
    # (30, 1.5) and (40, 2.0) differ in both parameters -- not neighbours.
    passing = {(30, 1.5), (40, 2.0), (50, 2.5)}
    rows = _neutral_donchian_and_tsm_rows()
    rows += _all_mean_reversion_rows(passing_cells=passing)
    summaries = {s.family: s for s in analyze_family_robustness(rows)}
    meanrev = summaries[MEAN_REVERSION_FAMILY]
    assert meanrev.largest_contiguous_region_size == 1
    assert meanrev.contiguous_passing_region_count == 3
    assert meanrev.survival_rule_passed is False


def test_stressed_return_gate_is_enforced() -> None:
    # Good pair/fold counts (5/7 profitable, 3/3 folds) at two adjacent
    # lookbacks, but a non-positive stressed return -- must still fail.
    rows = _all_1d_rows(DONCHIAN_FAMILY, "lookback", DONCHIAN_H4_LOOKBACKS)
    for i, lookback in enumerate(DONCHIAN_H4_LOOKBACKS):
        if lookback in (50, 55):
            rows[i] = _row(
                family=DONCHIAN_FAMILY,
                strategy_id=f"donchian_{lookback}",
                parameters={"lookback": lookback},
                stressed_return=-0.001,  # fails only this gate
                profitable_pairs=5,
                positive_folds=3,
            )
    rows += _all_1d_rows(TSM_FAMILY, "lookback", TSM_H4_LOOKBACKS)
    summaries = {s.family: s for s in analyze_family_robustness(rows)}
    assert summaries[DONCHIAN_FAMILY].passing_configuration_count == 0
    assert summaries[DONCHIAN_FAMILY].survival_rule_passed is False


def test_four_of_seven_pair_gate_is_enforced() -> None:
    rows = _all_1d_rows(TSM_FAMILY, "lookback", TSM_H4_LOOKBACKS)
    idx = [i for i, c in enumerate(TSM_H4_LOOKBACKS) if c in (30, 40)]
    for i in idx:
        rows[i] = _row(
            family=TSM_FAMILY,
            strategy_id=f"tsm_{TSM_H4_LOOKBACKS[i]}",
            parameters={"lookback": TSM_H4_LOOKBACKS[i]},
            stressed_return=0.01,
            profitable_pairs=3,  # below the 4/7 gate
            positive_folds=3,
        )
    rows += _all_1d_rows(DONCHIAN_FAMILY, "lookback", DONCHIAN_H4_LOOKBACKS)
    summaries = {s.family: s for s in analyze_family_robustness(rows)}
    assert summaries[TSM_FAMILY].passing_configuration_count == 0


def test_two_of_three_fold_gate_is_enforced() -> None:
    rows = _all_1d_rows(TSM_FAMILY, "lookback", TSM_H4_LOOKBACKS)
    idx = [i for i, c in enumerate(TSM_H4_LOOKBACKS) if c in (30, 40)]
    for i in idx:
        rows[i] = _row(
            family=TSM_FAMILY,
            strategy_id=f"tsm_{TSM_H4_LOOKBACKS[i]}",
            parameters={"lookback": TSM_H4_LOOKBACKS[i]},
            stressed_return=0.01,
            profitable_pairs=5,
            positive_folds=1,  # below the 2/3 gate
        )
    rows += _all_1d_rows(DONCHIAN_FAMILY, "lookback", DONCHIAN_H4_LOOKBACKS)
    summaries = {s.family: s for s in analyze_family_robustness(rows)}
    assert summaries[TSM_FAMILY].passing_configuration_count == 0


def test_family_robustness_analysis_is_deterministic() -> None:
    rows = _neutral_donchian_and_tsm_rows() + _all_mean_reversion_rows(
        passing_cells={(30, 1.5), (40, 1.5)}
    )
    first = analyze_family_robustness(rows)
    second = analyze_family_robustness(rows)
    assert first == second


def test_write_stage2_results_produces_scorecard_and_family_summary(
    tmp_path: Path,
) -> None:
    rows = tuple(
        _neutral_donchian_and_tsm_rows()
        + _all_mean_reversion_rows(passing_cells={(30, 1.5), (40, 1.5)})
    )
    result = ScreeningBatchResult(rows=rows, metadata={"configuration_count": 23})
    summaries = analyze_family_robustness(list(rows))
    output_dir = tmp_path / "stage2_out"
    write_stage2_results(result, summaries, output_dir)

    assert (output_dir / "scorecard.csv").is_file()
    assert (output_dir / "metadata.json").is_file()
    family_csv = (output_dir / "family_robustness.csv").read_text(encoding="utf-8")
    assert "donchian" in family_csv.lower() or DONCHIAN_FAMILY in family_csv
    assert MEAN_REVERSION_FAMILY in family_csv


def _synthetic_dataset(
    *, timeframe: str, periods: int, freq: str, seed: int
) -> AlphaLabDataset:
    index = pd.date_range("2019-06-01", periods=periods, freq=freq, tz=UTC)
    rng = np.random.default_rng(seed)
    columns = ("EUR/USD.OANDA", "GBP/USD.OANDA")
    close = pd.DataFrame(
        {c: 1.1 + np.cumsum(rng.normal(0, 0.0005, periods)) for c in columns},
        index=index,
    )
    high = close + 0.0006
    low = close - 0.0006
    bid = close - 0.0001
    ask = close + 0.0001
    return AlphaLabDataset(
        timeframe=timeframe,  # type: ignore[arg-type]
        instrument_ids=columns,
        start_utc=index[0].to_pydatetime(),
        end_exclusive_utc=index[-1].to_pydatetime(),
        alignment_policy=ALIGNMENT_POLICY,
        open=close,
        high=high,
        low=low,
        close=close,
        bid=bid,
        ask=ask,
        spread=ask - bid,
    )


def test_stage2_batch_runs_end_to_end_through_the_real_vectorbt_runner() -> None:
    datasets = {
        "H1": _synthetic_dataset(timeframe="H1", periods=1800, freq="1h", seed=2),
        "H4": _synthetic_dataset(timeframe="H4", periods=900, freq="4h", seed=3),
    }
    grid = build_stage2_grid()
    result = run_screening_batch(datasets_by_timeframe=datasets, grid=grid)
    assert result.metadata["configuration_count"] == 23
    n_instruments = len(datasets["H1"].instrument_ids)
    assert len(result.rows) == 23 * (n_instruments + 1)

    summaries = analyze_family_robustness(list(result.rows))
    expected_families = {DONCHIAN_FAMILY, TSM_FAMILY, MEAN_REVERSION_FAMILY}
    assert {s.family for s in summaries} == expected_families
    for summary in summaries:
        assert summary.tested_configuration_count in (7, 9)


def test_stage2_batch_is_deterministic() -> None:
    datasets = {
        "H1": _synthetic_dataset(timeframe="H1", periods=1200, freq="1h", seed=5),
        "H4": _synthetic_dataset(timeframe="H4", periods=600, freq="4h", seed=6),
    }
    grid = build_stage2_grid()
    first = run_screening_batch(datasets_by_timeframe=datasets, grid=grid)
    second = run_screening_batch(datasets_by_timeframe=datasets, grid=grid)
    assert first.rows == second.rows
    assert analyze_family_robustness(list(first.rows)) == analyze_family_robustness(
        list(second.rows)
    )


def test_stage2_module_never_accesses_validation_or_final_holdout_data() -> None:
    import ftmoquant.research.alpha_lab.screening_stage2 as stage2_module

    source = Path(stage2_module.__file__).read_text(encoding="utf-8")
    assert "validation" not in source.lower()
    assert "holdout" not in source.lower()
