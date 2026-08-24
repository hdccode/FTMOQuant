from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from ftmoquant.research.alpha_lab.b3f3_session_open_mr_execution import B3F3Trade
from ftmoquant.research.alpha_lab.b3f3_session_open_mr_screen import (
    MIN_CONNECTED_REGION_SIZE,
    MIN_TRADE_COUNT,
    PROFIT_FACTOR_GT,
    B3F3ScorecardRow,
    B3F3ScreenError,
    evaluate_b3f3_config,
    largest_connected_passing_region,
    write_b3f3_artifacts,
)
from ftmoquant.research.alpha_lab.b3f3_session_open_mr_signals import (
    DISPLACEMENT_THRESHOLD_GRID,
    EXIT_REASON_STOP,
    EXIT_REASON_TARGET,
    STOP_ATR_GRID,
    TARGET_MODE_GRID,
    B3F3Config,
    build_b3f3_grid,
)


def _build_trades(
    n: int, *, winners: int, start="2024-01-01T08:10:00Z", gap_minutes=5 * 24 * 60
):
    trades = []
    ts_ns = int(
        datetime.fromisoformat(start.replace("Z", "+00:00")).timestamp() * 1_000_000_000
    )
    gap_ns = gap_minutes * 60 * 1_000_000_000
    for i in range(n):
        win = ((i + 1) * winners) // n != (i * winners) // n
        move = Decimal("0.0050") if win else Decimal("-0.0020")
        entry_price = Decimal("1.1000")
        exit_price = entry_price + move
        exit_ns = ts_ns + 3600 * 1_000_000_000
        trades.append(
            B3F3Trade(
                instrument_id="EUR/USD.OANDA",
                config_id="disp2.0_satr0.75_OPEN_ANCHOR",
                local_date_iso="2024-01-01",
                direction=1 if i % 2 == 0 else -1,
                entry_ns=ts_ns,
                entry_price=entry_price,
                exit_ns=exit_ns,
                exit_price=exit_price,
                exit_reason=EXIT_REASON_TARGET if win else EXIT_REASON_STOP,
                quantity=Decimal("10000"),
                realized_pnl_usd=move * Decimal("10000"),
                displacement_at_entry=2.5,
            )
        )
        ts_ns += gap_ns
    return tuple(trades)


def _fold_boundaries_for(trades):
    exits = sorted(
        datetime.fromtimestamp(t.exit_ns / 1_000_000_000, tz=UTC) for t in trades
    )
    start = exits[0].replace(hour=0, minute=0, second=0, microsecond=0)
    end = exits[-1] + timedelta(days=1)
    step = (end - start) / 4
    return [start + step * i for i in range(5)]


# ---------------------------------------------------------------------------
# Grid constants
# ---------------------------------------------------------------------------


def test_min_trade_count_is_80_no_override_for_b3f3() -> None:
    assert MIN_TRADE_COUNT == 80


def test_profit_factor_threshold_matches_frozen_v2() -> None:
    assert PROFIT_FACTOR_GT == Decimal("1.1")


def test_min_connected_region_matches_frozen_v2() -> None:
    assert MIN_CONNECTED_REGION_SIZE == 2


# ---------------------------------------------------------------------------
# Gate wiring
# ---------------------------------------------------------------------------


def test_below_80_trades_fails_gate() -> None:
    trades = _build_trades(30, winners=25)
    row = evaluate_b3f3_config(
        instrument_id="EUR/USD.OANDA",
        config=B3F3Config(Decimal("2.0"), Decimal("0.75"), "OPEN_ANCHOR"),
        native_trades=trades,
        stressed_1_5x_trades=trades,
        stressed_2_0x_trades=trades,
        fold_boundaries=_fold_boundaries_for(trades),
    )
    assert row.native_trade_count == 30
    assert not row.hard_gates_passed


def test_healthy_sleeve_passes_all_hard_gates() -> None:
    trades = _build_trades(90, winners=68)
    row = evaluate_b3f3_config(
        instrument_id="EUR/USD.OANDA",
        config=B3F3Config(Decimal("2.0"), Decimal("0.75"), "OPEN_ANCHOR"),
        native_trades=trades,
        stressed_1_5x_trades=trades,
        stressed_2_0x_trades=trades,
        fold_boundaries=_fold_boundaries_for(trades),
    )
    assert row.native_trade_count >= MIN_TRADE_COUNT
    assert row.native_profit_factor > PROFIT_FACTOR_GT
    assert row.hard_gates_passed


def test_2_0x_stress_failure_fails_gate_even_if_1_5x_and_native_pass() -> None:
    trades = _build_trades(90, winners=68)
    losing_stress = _build_trades(90, winners=0)
    row = evaluate_b3f3_config(
        instrument_id="EUR/USD.OANDA",
        config=B3F3Config(Decimal("2.0"), Decimal("0.75"), "OPEN_ANCHOR"),
        native_trades=trades,
        stressed_1_5x_trades=trades,
        stressed_2_0x_trades=losing_stress,
        fold_boundaries=_fold_boundaries_for(trades),
    )
    assert row.native_profit_factor > PROFIT_FACTOR_GT
    assert row.stressed_2_0x_expectancy <= 0
    assert not row.hard_gates_passed


def test_diagnostics_never_flip_hard_gates_passed() -> None:
    import dataclasses

    trades = _build_trades(90, winners=68)
    row = evaluate_b3f3_config(
        instrument_id="EUR/USD.OANDA",
        config=B3F3Config(Decimal("2.0"), Decimal("0.75"), "OPEN_ANCHOR"),
        native_trades=trades,
        stressed_1_5x_trades=trades,
        stressed_2_0x_trades=trades,
        fold_boundaries=_fold_boundaries_for(trades),
    )
    corrupted = dataclasses.replace(
        row, monthly_max_share=Decimal("1.0"), time_exit_fraction=Decimal("1.0")
    )
    assert corrupted.hard_gates_passed == row.hard_gates_passed
    assert row.hard_gates_passed is True


def test_diagnostics_include_exit_reason_fractions_and_holding_time() -> None:
    trades = _build_trades(90, winners=68)
    row = evaluate_b3f3_config(
        instrument_id="EUR/USD.OANDA",
        config=B3F3Config(Decimal("2.0"), Decimal("0.75"), "OPEN_ANCHOR"),
        native_trades=trades,
        stressed_1_5x_trades=trades,
        stressed_2_0x_trades=trades,
        fold_boundaries=_fold_boundaries_for(trades),
    )
    assert row.stop_fraction is not None
    assert row.target_fraction is not None
    assert row.mean_holding_seconds == pytest.approx(3600.0)
    assert row.median_holding_seconds == pytest.approx(3600.0)


def test_diagnostics_include_displacement_and_entry_minute_stats() -> None:
    trades = _build_trades(90, winners=68)
    row = evaluate_b3f3_config(
        instrument_id="EUR/USD.OANDA",
        config=B3F3Config(Decimal("2.0"), Decimal("0.75"), "OPEN_ANCHOR"),
        native_trades=trades,
        stressed_1_5x_trades=trades,
        stressed_2_0x_trades=trades,
        fold_boundaries=_fold_boundaries_for(trades),
    )
    assert row.mean_displacement_at_entry == pytest.approx(2.5)
    assert row.median_displacement_at_entry == pytest.approx(2.5)
    assert row.mean_entry_minute_past_open is not None
    assert row.median_entry_minute_past_open is not None


# ---------------------------------------------------------------------------
# Parameter-neighborhood robustness
# ---------------------------------------------------------------------------


def _row(
    displacement_threshold: Decimal, stop_atr: Decimal, target_mode: str, passed: bool
) -> B3F3ScorecardRow:
    return B3F3ScorecardRow(
        instrument_id="EUR/USD.OANDA",
        config=B3F3Config(displacement_threshold, stop_atr, target_mode),
        native_trade_count=90,
        native_expectancy=Decimal("10") if passed else Decimal("-10"),
        native_profit_factor=Decimal("1.5") if passed else Decimal("0.5"),
        fold_positive_count=3,
        best_5pct_removed_expectancy=Decimal("5") if passed else Decimal("-5"),
        quarter_max_share=Decimal("0.2"),
        stressed_1_5x_trade_count=90,
        stressed_1_5x_expectancy=Decimal("5") if passed else Decimal("-5"),
        stressed_1_5x_profit_factor=Decimal("1.2"),
        stressed_2_0x_trade_count=90,
        stressed_2_0x_expectancy=Decimal("5") if passed else Decimal("-5"),
        stressed_2_0x_profit_factor=Decimal("1.1"),
        hard_gates_passed=passed,
        rolling_30_median_expectancy=None,
        rolling_30_fraction_positive=None,
        rolling_50_median_expectancy=None,
        rolling_50_fraction_positive=None,
        monthly_max_share=None,
        largest_trade_share=None,
        pnl_skewness=None,
        pnl_kurtosis=None,
        long_trade_count=0,
        long_expectancy=None,
        long_profit_factor=None,
        short_trade_count=0,
        short_expectancy=None,
        short_profit_factor=None,
        stop_fraction=None,
        target_fraction=None,
        time_exit_fraction=None,
        mean_holding_seconds=None,
        median_holding_seconds=None,
        mean_displacement_at_entry=None,
        median_displacement_at_entry=None,
        mean_entry_minute_past_open=None,
        median_entry_minute_past_open=None,
    )


def test_connected_region_of_size_2_is_detected() -> None:
    rows = [
        _row(
            DISPLACEMENT_THRESHOLD_GRID[0], STOP_ATR_GRID[0], TARGET_MODE_GRID[0], True
        ),
        _row(
            DISPLACEMENT_THRESHOLD_GRID[1], STOP_ATR_GRID[0], TARGET_MODE_GRID[0], True
        ),
    ]
    assert largest_connected_passing_region(rows) == 2


def test_isolated_cell_is_not_a_region_of_2() -> None:
    rows = [
        _row(
            DISPLACEMENT_THRESHOLD_GRID[0], STOP_ATR_GRID[0], TARGET_MODE_GRID[0], True
        )
    ]
    region = largest_connected_passing_region(rows)
    assert region == 1
    assert region < MIN_CONNECTED_REGION_SIZE


def test_instrument_identity_is_not_an_adjacency_dimension() -> None:
    import dataclasses

    row_a = _row(
        DISPLACEMENT_THRESHOLD_GRID[0], STOP_ATR_GRID[0], TARGET_MODE_GRID[0], True
    )
    row_b = dataclasses.replace(row_a, instrument_id="GBP/USD.OANDA")
    with pytest.raises(B3F3ScreenError):
        largest_connected_passing_region([row_a, row_b])


def test_grid_produces_exactly_12_rows() -> None:
    grid = build_b3f3_grid()
    assert len(grid) == 12


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def test_write_b3f3_artifacts_produces_expected_files(tmp_path: Path) -> None:
    trades = _build_trades(90, winners=68)
    row = evaluate_b3f3_config(
        instrument_id="EUR/USD.OANDA",
        config=B3F3Config(Decimal("2.0"), Decimal("0.75"), "OPEN_ANCHOR"),
        native_trades=trades,
        stressed_1_5x_trades=trades,
        stressed_2_0x_trades=trades,
        fold_boundaries=_fold_boundaries_for(trades),
    )
    output_dir = tmp_path / "b3f3_run"
    write_b3f3_artifacts(
        scorecard=[row],
        pair_robustness=[
            {"instrument_id": "EUR/USD.OANDA", "largest_connected_region": 1}
        ],
        metadata={"holdout_accessed": False, "validation_accessed": False},
        output_dir=output_dir,
    )
    assert (output_dir / "scorecard.csv").exists()
    assert (output_dir / "pair_robustness.csv").exists()
    assert (output_dir / "metadata.json").exists()


def test_write_b3f3_artifacts_refuses_to_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "b3f3_run"
    write_b3f3_artifacts(
        scorecard=[], pair_robustness=[], metadata={}, output_dir=output_dir
    )
    with pytest.raises(B3F3ScreenError):
        write_b3f3_artifacts(
            scorecard=[], pair_robustness=[], metadata={}, output_dir=output_dir
        )
