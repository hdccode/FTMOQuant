from __future__ import annotations

import csv
import json
from datetime import UTC
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import vectorbt as vbt

from ftmoquant.research.alpha_lab.data import ALIGNMENT_POLICY, AlphaLabDataset
from ftmoquant.research.alpha_lab.smoke import (
    AGGREGATE_INSTRUMENT_LABEL,
    SCREENING_COST_LABEL,
    STRATEGY_ID,
    _build_aggregate_portfolio,
    run_sma_crossover_smoke,
    write_results,
)

_RESULT_FIELDS = (
    "strategy_id",
    "instrument",
    "timeframe",
    "parameters",
    "start",
    "end",
    "observation_count",
    "trade_count",
    "cumulative_return",
    "annualized_return",
    "annualized_sharpe",
    "maximum_drawdown",
    "win_rate",
    "turnover_or_trade_activity",
    "screening_cost_assumption",
)


def _synthetic_dataset(*, periods: int = 400, seed: int = 7) -> AlphaLabDataset:
    index = pd.date_range("2020-01-01", periods=periods, freq="h", tz=UTC)
    rng = np.random.default_rng(seed)
    columns = ("EUR/USD.X", "GBP/USD.X")
    close = pd.DataFrame(
        {
            "EUR/USD.X": 1.10 + np.cumsum(rng.normal(0, 0.0006, periods)),
            "GBP/USD.X": 1.30 + np.cumsum(rng.normal(0, 0.0006, periods)),
        },
        index=index,
    )
    bid = close - 0.0001
    ask = close + 0.0001
    return AlphaLabDataset(
        timeframe="H1",
        instrument_ids=columns,
        start_utc=index[0].to_pydatetime(),
        end_exclusive_utc=index[-1].to_pydatetime(),
        alignment_policy=ALIGNMENT_POLICY,
        open=close,
        high=close + 0.0005,
        low=close - 0.0005,
        close=close,
        bid=bid,
        ask=ask,
        spread=ask - bid,
    )


def test_smoke_run_builds_real_vectorbt_portfolios_per_instrument() -> None:
    dataset = _synthetic_dataset()
    result = run_sma_crossover_smoke(dataset)
    instruments = {row.instrument for row in result.rows}
    assert set(dataset.instrument_ids) <= instruments
    assert AGGREGATE_INSTRUMENT_LABEL in instruments
    assert len(result.rows) == len(dataset.instrument_ids) + 1


def test_result_rows_conform_to_standard_schema() -> None:
    dataset = _synthetic_dataset()
    result = run_sma_crossover_smoke(dataset)
    for row in result.rows:
        assert row.strategy_id == STRATEGY_ID
        assert row.timeframe == "H1"
        assert row.observation_count == len(dataset.close.index)
        assert isinstance(row.trade_count, int)
        assert isinstance(row.cumulative_return, float)
        assert isinstance(row.annualized_return, float)
        assert isinstance(row.annualized_sharpe, float)
        assert isinstance(row.maximum_drawdown, float)
        assert 0.0 <= row.win_rate <= 1.0
        assert row.turnover_or_trade_activity >= 0.0
        assert "screening_cost" in row.screening_cost_assumption or "=" in (
            row.screening_cost_assumption
        )
        json.loads(row.parameters)


def test_smoke_run_is_deterministic() -> None:
    dataset = _synthetic_dataset()
    first = run_sma_crossover_smoke(dataset)
    second = run_sma_crossover_smoke(dataset)
    assert first.rows == second.rows


def test_single_instrument_dataset_has_no_aggregate_row() -> None:
    dataset = _synthetic_dataset()
    single = AlphaLabDataset(
        timeframe=dataset.timeframe,
        instrument_ids=dataset.instrument_ids[:1],
        start_utc=dataset.start_utc,
        end_exclusive_utc=dataset.end_exclusive_utc,
        alignment_policy=dataset.alignment_policy,
        open=dataset.open[[dataset.instrument_ids[0]]],
        high=dataset.high[[dataset.instrument_ids[0]]],
        low=dataset.low[[dataset.instrument_ids[0]]],
        close=dataset.close[[dataset.instrument_ids[0]]],
        bid=dataset.bid[[dataset.instrument_ids[0]]],
        ask=dataset.ask[[dataset.instrument_ids[0]]],
        spread=dataset.spread[[dataset.instrument_ids[0]]],
    )
    result = run_sma_crossover_smoke(single)
    assert len(result.rows) == 1
    assert result.rows[0].instrument == dataset.instrument_ids[0]


def test_empty_dataset_is_rejected() -> None:
    dataset = _synthetic_dataset(periods=1)
    empty = AlphaLabDataset(
        timeframe=dataset.timeframe,
        instrument_ids=dataset.instrument_ids,
        start_utc=dataset.start_utc,
        end_exclusive_utc=dataset.end_exclusive_utc,
        alignment_policy=dataset.alignment_policy,
        open=dataset.open.iloc[0:0],
        high=dataset.high.iloc[0:0],
        low=dataset.low.iloc[0:0],
        close=dataset.close.iloc[0:0],
        bid=dataset.bid.iloc[0:0],
        ask=dataset.ask.iloc[0:0],
        spread=dataset.spread.iloc[0:0],
    )
    with pytest.raises(ValueError):
        run_sma_crossover_smoke(empty)


def test_write_results_produces_csv_with_standard_header_and_metadata(
    tmp_path: Path,
) -> None:
    dataset = _synthetic_dataset()
    result = run_sma_crossover_smoke(dataset)
    output_dir = tmp_path / "smoke_out"
    write_results(result, output_dir)

    with (output_dir / "results.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames or ()) == _RESULT_FIELDS
        rows = list(reader)
    assert len(rows) == len(result.rows)

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["strategy_id"] == STRATEGY_ID
    assert metadata["instrument_ids"] == list(dataset.instrument_ids)


def test_screening_cost_label_is_named_truthfully() -> None:
    assert (
        SCREENING_COST_LABEL
        == "fixed_mean_half_spread_estimated_from_full_development"
    )
    assert "causal" not in SCREENING_COST_LABEL


def test_aggregate_equal_weight_sums_independent_equal_capital_sleeves() -> None:
    # Instrument A: buy-and-hold +10%. Instrument B: buy-and-hold -2%.
    # Each gets equal initial capital and trades with no competition for
    # cash, so the equal-weight aggregate return must be their average.
    index = pd.date_range("2020-01-01", periods=10, freq="h", tz=UTC)
    close = pd.DataFrame(
        {
            "A": np.linspace(100.0, 110.0, 10),
            "B": np.linspace(100.0, 98.0, 10),
        },
        index=index,
    )
    entries = pd.DataFrame(False, index=index, columns=close.columns)
    exits = pd.DataFrame(False, index=index, columns=close.columns)
    entries.iloc[0] = True

    individual_returns = {}
    for column in close.columns:
        pf = vbt.Portfolio.from_signals(
            close[column], entries[column], exits[column], fees=0.0, freq="1h"
        )
        individual_returns[column] = float(pf.total_return())
    assert individual_returns["A"] == pytest.approx(0.10, abs=1e-9)
    assert individual_returns["B"] == pytest.approx(-0.02, abs=1e-9)

    aggregate = _build_aggregate_portfolio(
        close, entries, exits, per_column_fees=0.0, freq="1h"
    )
    aggregate_return = float(aggregate.total_return())
    expected = (individual_returns["A"] + individual_returns["B"]) / 2.0
    assert aggregate_return == pytest.approx(expected, abs=1e-9)
    assert aggregate_return == pytest.approx(0.04, abs=1e-9)

    # Column order must not change the aggregate result.
    reordered_close = close[["B", "A"]]
    reordered_entries = entries[["B", "A"]]
    reordered_exits = exits[["B", "A"]]
    reordered_aggregate = _build_aggregate_portfolio(
        reordered_close,
        reordered_entries,
        reordered_exits,
        per_column_fees=0.0,
        freq="1h",
    )
    assert float(reordered_aggregate.total_return()) == pytest.approx(
        aggregate_return, abs=1e-9
    )


def test_write_results_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    dataset = _synthetic_dataset()
    result = run_sma_crossover_smoke(dataset)
    output_dir = tmp_path / "smoke_out"
    write_results(result, output_dir)
    with pytest.raises(FileExistsError):
        write_results(result, output_dir)
