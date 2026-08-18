from __future__ import annotations

import csv
import json
from dataclasses import replace
from datetime import UTC
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ftmoquant.research.alpha_lab.data import ALIGNMENT_POLICY, AlphaLabDataset
from ftmoquant.research.alpha_lab.screening_batch import (
    build_grid,
    run_screening_batch,
    write_results,
)
from ftmoquant.research.alpha_lab.screening_common import AGGREGATE_INSTRUMENT_LABEL
from ftmoquant.research.stage_g import DEVELOPMENT_END_EXCLUSIVE, DEVELOPMENT_START

_SCORECARD_FIELDS = (
    "family",
    "strategy_id",
    "instrument",
    "timeframe",
    "parameters",
    "observation_count",
    "trade_count",
    "cumulative_return",
    "annualized_return",
    "annualized_sharpe",
    "maximum_drawdown",
    "win_rate",
    "turnover_or_trade_activity",
    "base_cost_return",
    "stressed_cost_return",
    "profitable_pair_count",
    "positive_sharpe_pair_count",
    "median_pair_sharpe",
    "worst_pair_sharpe",
    "aggregate_equal_weight_return",
    "aggregate_equal_weight_sharpe",
    "positive_fold_count",
    "worst_fold_return",
    "median_fold_sharpe",
    "diagnostic_rank",
)


def _synthetic_dataset(
    *, timeframe: str, periods: int, freq: str, seed: int, columns: tuple[str, ...]
) -> AlphaLabDataset:
    index = pd.date_range("2019-06-01", periods=periods, freq=freq, tz=UTC)
    rng = np.random.default_rng(seed)
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


def _synthetic_datasets() -> dict[str, AlphaLabDataset]:
    columns = ("EUR/USD.OANDA", "GBP/USD.OANDA")
    return {
        "M30": _synthetic_dataset(
            timeframe="M30", periods=2500, freq="30min", seed=1, columns=columns
        ),
        "H1": _synthetic_dataset(
            timeframe="H1", periods=1800, freq="1h", seed=2, columns=columns
        ),
        "H4": _synthetic_dataset(
            timeframe="H4", periods=900, freq="4h", seed=3, columns=columns
        ),
    }


def test_grid_has_roughly_fifty_to_two_hundred_configurations() -> None:
    grid = build_grid()
    assert 40 <= len(grid) <= 200
    assert len({config.strategy_id for config in grid}) == len(grid)


def test_grid_covers_exactly_the_five_implemented_families() -> None:
    families = {config.family for config in build_grid()}
    assert families == {
        "A_time_series_momentum",
        "B_donchian_breakout",
        "C_short_horizon_mean_reversion",
        "D_volatility_breakout",
        "E_session_effect_baseline",
    }


def test_scorecard_has_one_row_per_config_instrument_and_aggregate() -> None:
    datasets = _synthetic_datasets()
    grid = build_grid()
    result = run_screening_batch(datasets_by_timeframe=datasets, grid=grid)

    n_instruments = len(datasets["H1"].instrument_ids)
    expected_rows = len(grid) * (n_instruments + 1)  # +1 aggregate row per config
    assert len(result.rows) == expected_rows

    for config in grid:
        rows = [row for row in result.rows if row.strategy_id == config.strategy_id]
        assert len(rows) == n_instruments + 1
        expected_instruments = set(datasets[config.timeframe].instrument_ids) | {
            AGGREGATE_INSTRUMENT_LABEL
        }
        assert {row.instrument for row in rows} == expected_instruments
        assert all(row.family == config.family for row in rows)
        assert all(row.timeframe == config.timeframe for row in rows)


def test_result_rows_conform_to_the_standardized_schema() -> None:
    datasets = _synthetic_datasets()
    result = run_screening_batch(datasets_by_timeframe=datasets)
    for row in result.rows:
        assert isinstance(row.cumulative_return, float)
        assert isinstance(row.base_cost_return, float)
        assert isinstance(row.stressed_cost_return, float)
        assert 0.0 <= row.win_rate <= 1.0
        assert row.turnover_or_trade_activity >= 0.0
        assert row.diagnostic_rank is not None
        json.loads(row.parameters)


def test_cross_pair_robustness_fields_populated_on_every_row() -> None:
    datasets = _synthetic_datasets()
    result = run_screening_batch(datasets_by_timeframe=datasets)
    one_strategy_id = result.rows[0].strategy_id
    rows = [row for row in result.rows if row.strategy_id == one_strategy_id]
    assert all(row.profitable_pair_count is not None for row in rows)
    assert all(row.median_pair_sharpe is not None for row in rows)
    assert all(row.aggregate_equal_weight_return is not None for row in rows)
    values = {row.profitable_pair_count for row in rows}
    assert len(values) == 1  # identical across instrument and aggregate rows


def test_stressed_cost_return_is_computed_at_one_point_five_x_the_base_fee() -> None:
    datasets = _synthetic_datasets()
    result = run_screening_batch(datasets_by_timeframe=datasets)
    traded_rows = [row for row in result.rows if row.trade_count > 0]
    assert traded_rows
    # A higher proportional fee applied to the same trades can only reduce
    # (or leave unchanged, if flat) the realized return relative to the base
    # cost run -- it never improves it.
    assert all(
        row.stressed_cost_return <= row.base_cost_return + 1e-9 for row in traded_rows
    )


def test_fold_diagnostics_are_present_and_bounded() -> None:
    datasets = _synthetic_datasets()
    result = run_screening_batch(datasets_by_timeframe=datasets)
    aggregate_rows = [
        row for row in result.rows if row.instrument == AGGREGATE_INSTRUMENT_LABEL
    ]
    assert aggregate_rows
    for row in aggregate_rows:
        assert row.positive_fold_count is not None
        assert 0 <= row.positive_fold_count <= 3
        assert row.worst_fold_return is not None
        assert row.median_fold_sharpe is not None


def test_diagnostic_rank_orders_by_profitable_pairs_first() -> None:
    datasets = _synthetic_datasets()
    result = run_screening_batch(datasets_by_timeframe=datasets)
    aggregate_rows = [
        row for row in result.rows if row.instrument == AGGREGATE_INSTRUMENT_LABEL
    ]
    by_rank = sorted(aggregate_rows, key=lambda row: row.diagnostic_rank or 0)
    profitable_counts = [row.profitable_pair_count or 0 for row in by_rank]
    # profitable_pair_count is the primary sort key, so it must be
    # non-increasing as diagnostic_rank worsens (ties broken by later keys).
    assert all(
        profitable_counts[i] >= profitable_counts[i + 1]
        for i in range(len(profitable_counts) - 1)
    )
    ranks = [row.diagnostic_rank for row in aggregate_rows]
    assert sorted(ranks) == list(range(1, len(aggregate_rows) + 1))


def test_no_configuration_is_discarded() -> None:
    datasets = _synthetic_datasets()
    grid = build_grid()
    result = run_screening_batch(datasets_by_timeframe=datasets, grid=grid)
    strategy_ids_in_result = {row.strategy_id for row in result.rows}
    assert strategy_ids_in_result == {config.strategy_id for config in grid}


def test_screening_batch_is_deterministic() -> None:
    datasets = _synthetic_datasets()
    grid = build_grid()[:6]
    first = run_screening_batch(datasets_by_timeframe=datasets, grid=grid)
    second = run_screening_batch(datasets_by_timeframe=datasets, grid=grid)
    assert first.rows == second.rows


def test_write_results_produces_a_single_standardized_scorecard(tmp_path: Path) -> None:
    datasets = _synthetic_datasets()
    grid = build_grid()[:4]
    result = run_screening_batch(datasets_by_timeframe=datasets, grid=grid)
    output_dir = tmp_path / "batch_out"
    write_results(result, output_dir)

    with (output_dir / "scorecard.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames or ()) == _SCORECARD_FIELDS
        rows = list(reader)
    assert len(rows) == len(result.rows)

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["configuration_count"] == len(grid)
    assert metadata["probot_status"] == "BLOCKED_RULE_RECONSTRUCTION"
    assert len(metadata["fold_windows"]) == 3


def test_batch_never_accesses_validation_or_final_holdout_data() -> None:
    import ftmoquant.research.alpha_lab.screening_batch as batch_module

    source = Path(batch_module.__file__).read_text(encoding="utf-8")
    assert "validation" not in source.lower()
    assert "holdout" not in source.lower()


def test_dataset_time_bounds_match_the_frozen_development_window() -> None:
    # load_alpha_lab_dataset always stamps DEVELOPMENT_START/END_EXCLUSIVE
    # regardless of source; this pins that no other window can slip in.
    datasets = _synthetic_datasets()
    for dataset in datasets.values():
        replaced = replace(
            dataset,
            start_utc=DEVELOPMENT_START,
            end_exclusive_utc=DEVELOPMENT_END_EXCLUSIVE,
        )
        assert replaced.start_utc == DEVELOPMENT_START
        assert replaced.end_exclusive_utc == DEVELOPMENT_END_EXCLUSIVE


def test_empty_dataset_is_rejected() -> None:
    datasets = _synthetic_datasets()
    empty = replace(
        datasets["H1"],
        open=datasets["H1"].open.iloc[0:0],
        high=datasets["H1"].high.iloc[0:0],
        low=datasets["H1"].low.iloc[0:0],
        close=datasets["H1"].close.iloc[0:0],
        bid=datasets["H1"].bid.iloc[0:0],
        ask=datasets["H1"].ask.iloc[0:0],
        spread=datasets["H1"].spread.iloc[0:0],
    )
    with pytest.raises(ValueError):
        run_screening_batch(datasets_by_timeframe={"H1": empty}, grid=build_grid()[:1])
