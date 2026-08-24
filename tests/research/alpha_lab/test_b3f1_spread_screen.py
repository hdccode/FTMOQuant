from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from ftmoquant.research.alpha_lab.b3f1_spread_screen import (
    MIN_CONNECTED_REGION_SIZE,
    MIN_TRADE_COUNT,
    PROFIT_FACTOR_GT,
    B3F1ScorecardRow,
    B3F1ScreenError,
    _TradeRow,
    best_5pct_removed_expectancy,
    evaluate_b3f1_config,
    expectancy_and_profit_factor,
    fold_positive_count,
    largest_connected_passing_region,
    quarter_max_share,
    write_b3f1_artifacts,
)
from ftmoquant.research.alpha_lab.b3f1_spread_signals import (
    EXIT_REASON_Z_MEAN_REVERSION,
    FORMATION_WINDOWS,
    Z_ENTRY_GRID,
    Z_STOP_GRID,
    B3F1Config,
)


def _build_episodes(
    n: int, *, winners: int, start="2024-01-01T00:00:00Z", gap_minutes=5 * 24 * 60
):
    """``n`` hand-built :class:`RelativeValueEpisode` objects with an
    exactly controlled winner/loser split and known-good account-currency
    P&L (execution-engine correctness is covered separately in
    ``test_b3f1_spread_execution.py`` -- these tests only need arbitrary,
    reliable P&L sequences to exercise the gate arithmetic)."""

    from ftmoquant.research.alpha_lab.relative_value_adapter import (
        LegMark,
        RelativeValueEpisode,
        RelativeValueLeg,
    )

    episodes = []
    ts = pd.Timestamp(start)
    for i in range(n):
        # Bresenham-style even interleaving of exactly `winners` True
        # values among `n` trades, so profit is spread evenly across the
        # whole period rather than concentrated in whichever quarter
        # happens to contain a contiguous block of winners.
        win = ((i + 1) * winners) // n != (i * winners) // n
        move = Decimal("0.0050") if win else Decimal("-0.0020")
        entry_ns = int(ts.value)
        exit_ns = int((ts + pd.Timedelta(minutes=10)).value)
        entry_price = Decimal("1.0800")
        exit_price = entry_price + move
        leg_a = RelativeValueLeg(
            instrument_id="EUR/USD.OANDA",
            direction=1,
            quantity=Decimal("10000"),
            base_currency="EUR",
            quote_currency="USD",
            entry_ns=entry_ns,
            entry_price=entry_price,
            exit_ns=exit_ns,
            exit_price=exit_price,
            marks=(LegMark(entry_ns, entry_price), LegMark(exit_ns, exit_price)),
        )
        x_entry, x_exit = Decimal("1.3500"), Decimal("1.3500")
        leg_b = RelativeValueLeg(
            instrument_id="USD/CAD.OANDA",
            direction=-1,
            quantity=Decimal("10000"),
            base_currency="USD",
            quote_currency="CAD",
            entry_ns=entry_ns,
            entry_price=x_entry,
            exit_ns=exit_ns,
            exit_price=x_exit,
            marks=(LegMark(entry_ns, x_entry), LegMark(exit_ns, x_exit)),
        )
        episodes.append(
            RelativeValueEpisode(
                logical_trade_id=f"s:{i}",
                leg_a=leg_a,
                leg_b=leg_b,
                exit_reason=EXIT_REASON_Z_MEAN_REVERSION,
            )
        )
        ts = ts + pd.Timedelta(minutes=gap_minutes)
    return tuple(episodes)


# ---------------------------------------------------------------------------
# Gate primitives
# ---------------------------------------------------------------------------


def test_min_trade_count_matches_frozen_v2_override() -> None:
    assert MIN_TRADE_COUNT == 50


def test_profit_factor_threshold_matches_frozen_v2() -> None:
    assert PROFIT_FACTOR_GT == Decimal("1.1")


def test_min_connected_region_matches_frozen_v2() -> None:
    assert MIN_CONNECTED_REGION_SIZE == 2


def test_expectancy_and_pf_basic() -> None:
    expectancy, pf = expectancy_and_profit_factor(
        [Decimal("100"), Decimal("-50"), Decimal("200")]
    )
    assert expectancy == Decimal("250") / 3
    assert pf == Decimal("300") / Decimal("50")


def test_pf_is_infinite_with_no_losses() -> None:
    _, pf = expectancy_and_profit_factor([Decimal("100"), Decimal("50")])
    assert pf == Decimal("Infinity")


def test_best_5pct_removes_ceiling_count_of_profitable_trades_only() -> None:
    rows = tuple(
        _TradeRow(exit_ts=datetime(2020, 1, i + 1, tzinfo=UTC), pnl=pnl, side="rich")
        for i, pnl in enumerate(
            [
                Decimal(v)
                for v in (
                    10,
                    20,
                    30,
                    40,
                    -5,
                    -5,
                    -5,
                    -5,
                    -5,
                    -5,
                    -5,
                    -5,
                    -5,
                    -5,
                    -5,
                    -5,
                    -5,
                    -5,
                    -5,
                    -5,
                )
            ]
        )
    )
    # 4 profitable trades -> ceil(0.05*4) = 1 removed (the single largest: 40).
    result = best_5pct_removed_expectancy(rows)
    remaining = [r.pnl for r in rows if r.pnl != Decimal(40)]
    assert result == sum(remaining, Decimal(0)) / len(remaining)


def test_best_5pct_tie_break_removes_earlier_exit_first() -> None:
    rows = (
        _TradeRow(
            exit_ts=datetime(2020, 1, 1, tzinfo=UTC), pnl=Decimal("100"), side="rich"
        ),
        _TradeRow(
            exit_ts=datetime(2020, 2, 1, tzinfo=UTC), pnl=Decimal("100"), side="rich"
        ),
    ) + tuple(
        _TradeRow(
            exit_ts=datetime(2020, 3, i + 1, tzinfo=UTC),
            pnl=Decimal("-1"),
            side="cheap",
        )
        for i in range(18)
    )
    # 2 profitable trades, tied at 100 -> ceil(0.05*2)=1 removed, earlier
    # exit (Jan) removed first.
    result = best_5pct_removed_expectancy(rows)
    remaining_pnls = [
        r.pnl for r in rows if r.exit_ts != datetime(2020, 1, 1, tzinfo=UTC)
    ]
    assert result == sum(remaining_pnls, Decimal(0)) / len(remaining_pnls)


def test_quarter_concentration_fails_closed_on_nonpositive_total() -> None:
    rows = (
        _TradeRow(
            exit_ts=datetime(2020, 1, 1, tzinfo=UTC), pnl=Decimal("-10"), side="rich"
        ),
        _TradeRow(
            exit_ts=datetime(2020, 2, 1, tzinfo=UTC), pnl=Decimal("-20"), side="cheap"
        ),
    )
    assert quarter_max_share(rows) is None


def test_quarter_concentration_boundary_at_exactly_40_percent() -> None:
    rows = (
        _TradeRow(
            exit_ts=datetime(2020, 1, 1, tzinfo=UTC), pnl=Decimal("40"), side="rich"
        ),
        _TradeRow(
            exit_ts=datetime(2020, 4, 1, tzinfo=UTC), pnl=Decimal("30"), side="rich"
        ),
        _TradeRow(
            exit_ts=datetime(2020, 7, 1, tzinfo=UTC), pnl=Decimal("30"), side="rich"
        ),
    )
    assert quarter_max_share(rows) == Decimal("0.4")


def test_fold_positive_count_counts_correctly() -> None:
    rows = (
        _TradeRow(
            exit_ts=datetime(2020, 1, 15, tzinfo=UTC), pnl=Decimal("10"), side="rich"
        ),
        _TradeRow(
            exit_ts=datetime(2020, 2, 15, tzinfo=UTC), pnl=Decimal("-10"), side="cheap"
        ),
        _TradeRow(
            exit_ts=datetime(2020, 3, 15, tzinfo=UTC), pnl=Decimal("5"), side="rich"
        ),
    )
    boundaries = [
        datetime(2020, 1, 1, tzinfo=UTC),
        datetime(2020, 2, 1, tzinfo=UTC),
        datetime(2020, 3, 1, tzinfo=UTC),
        datetime(2020, 4, 1, tzinfo=UTC),
    ]
    assert fold_positive_count(rows, boundaries) == 2


# ---------------------------------------------------------------------------
# Full config evaluation / gate wiring
# ---------------------------------------------------------------------------


def _fold_boundaries_for(episodes) -> list[datetime]:
    exits = sorted(
        datetime.fromtimestamp(e.exit_ns / 1_000_000_000, tz=UTC) for e in episodes
    )
    start = exits[0].replace(hour=0, minute=0, second=0, microsecond=0)
    end = exits[-1] + pd_timedelta_days(1)
    step = (end - start) / 4
    return [start + step * i for i in range(5)]


def pd_timedelta_days(days: int):
    from datetime import timedelta

    return timedelta(days=days)


def test_below_min_trade_count_fails_the_gate() -> None:
    episodes = _build_episodes(10, winners=10)
    row = evaluate_b3f1_config(
        sleeve_id="s",
        config=B3F1Config(240, Decimal("1.5"), Decimal("3.0")),
        native_episodes=episodes,
        stressed_1_5x_episodes=episodes,
        fold_boundaries=_fold_boundaries_for(episodes),
    )
    assert row.native_trade_count == 10
    assert not row.hard_gates_passed


def test_healthy_sleeve_passes_all_hard_gates() -> None:
    episodes = _build_episodes(60, winners=45)
    row = evaluate_b3f1_config(
        sleeve_id="s",
        config=B3F1Config(240, Decimal("1.5"), Decimal("3.0")),
        native_episodes=episodes,
        stressed_1_5x_episodes=episodes,
        fold_boundaries=_fold_boundaries_for(episodes),
    )
    assert row.native_trade_count >= MIN_TRADE_COUNT
    assert row.native_profit_factor > PROFIT_FACTOR_GT
    assert row.hard_gates_passed


def test_1_5x_stress_failure_fails_the_gate_even_if_native_passes() -> None:
    episodes = _build_episodes(60, winners=45)
    # Simulate a stress run with all-losing trades to force the stress
    # gate to fail while native still passes.
    losing_stress = _build_episodes(60, winners=0)
    row = evaluate_b3f1_config(
        sleeve_id="s",
        config=B3F1Config(240, Decimal("1.5"), Decimal("3.0")),
        native_episodes=episodes,
        stressed_1_5x_episodes=losing_stress,
        fold_boundaries=_fold_boundaries_for(episodes),
    )
    assert row.native_profit_factor > PROFIT_FACTOR_GT
    assert row.stressed_1_5x_expectancy <= 0
    assert not row.hard_gates_passed


def test_diagnostics_never_flip_hard_gates_passed() -> None:
    """A sleeve that fails a diagnostic-only property (e.g. extreme
    monthly concentration or poor rolling stability) but clears every
    frozen hard gate must still report hard_gates_passed=True."""

    episodes = _build_episodes(60, winners=45)
    row = evaluate_b3f1_config(
        sleeve_id="s",
        config=B3F1Config(240, Decimal("1.5"), Decimal("3.0")),
        native_episodes=episodes,
        stressed_1_5x_episodes=episodes,
        fold_boundaries=_fold_boundaries_for(episodes),
    )
    # Corrupt only the diagnostic fields and confirm hard_gates_passed is
    # untouched by construction (it was computed once, from the hard-gate
    # inputs only, before diagnostics were ever calculated).
    import dataclasses

    corrupted = dataclasses.replace(
        row,
        monthly_max_share=Decimal("1.0"),
        rolling_30_fraction_positive=Decimal("0.0"),
    )
    assert corrupted.hard_gates_passed == row.hard_gates_passed
    assert row.hard_gates_passed is True


# ---------------------------------------------------------------------------
# Parameter-neighborhood robustness (reused N-D adjacency)
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


def test_connected_region_of_size_2_is_detected() -> None:
    rows = [
        _row(FORMATION_WINDOWS[0], Z_ENTRY_GRID[0], Z_STOP_GRID[0], True),
        _row(FORMATION_WINDOWS[0], Z_ENTRY_GRID[1], Z_STOP_GRID[0], True),
    ]
    region = largest_connected_passing_region(
        rows,
        formation_windows=FORMATION_WINDOWS,
        z_entries=Z_ENTRY_GRID,
        z_stops=Z_STOP_GRID,
    )
    assert region == 2


def test_isolated_single_passing_cell_is_not_a_connected_region_of_2() -> None:
    rows = [_row(FORMATION_WINDOWS[0], Z_ENTRY_GRID[0], Z_STOP_GRID[0], True)]
    region = largest_connected_passing_region(
        rows,
        formation_windows=FORMATION_WINDOWS,
        z_entries=Z_ENTRY_GRID,
        z_stops=Z_STOP_GRID,
    )
    assert region == 1
    assert region < MIN_CONNECTED_REGION_SIZE


def test_pair_identity_is_not_an_adjacency_dimension() -> None:
    rows = [
        _row(FORMATION_WINDOWS[0], Z_ENTRY_GRID[0], Z_STOP_GRID[0], True),
    ]
    other_sleeve_row = dataclasses_replace_sleeve(rows[0], "other_sleeve")
    with pytest.raises(B3F1ScreenError):
        largest_connected_passing_region(
            [rows[0], other_sleeve_row],
            formation_windows=FORMATION_WINDOWS,
            z_entries=Z_ENTRY_GRID,
            z_stops=Z_STOP_GRID,
        )


def dataclasses_replace_sleeve(
    row: B3F1ScorecardRow, sleeve_id: str
) -> B3F1ScorecardRow:
    import dataclasses

    return dataclasses.replace(row, sleeve_id=sleeve_id)


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def test_write_b3f1_artifacts_produces_expected_files(tmp_path: Path) -> None:
    episodes = _build_episodes(60, winners=45)
    row = evaluate_b3f1_config(
        sleeve_id="s",
        config=B3F1Config(240, Decimal("1.5"), Decimal("3.0")),
        native_episodes=episodes,
        stressed_1_5x_episodes=episodes,
        fold_boundaries=_fold_boundaries_for(episodes),
    )
    output_dir = tmp_path / "b3f1_run"
    write_b3f1_artifacts(
        scorecard=[row],
        pair_robustness=[{"sleeve_id": "s", "largest_connected_region": 1}],
        metadata={"holdout_accessed": False, "validation_accessed": False},
        output_dir=output_dir,
    )
    assert (output_dir / "scorecard.csv").exists()
    assert (output_dir / "pair_robustness.csv").exists()
    assert (output_dir / "metadata.json").exists()


def test_write_b3f1_artifacts_refuses_to_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "b3f1_run"
    write_b3f1_artifacts(
        scorecard=[], pair_robustness=[], metadata={}, output_dir=output_dir
    )
    with pytest.raises(B3F1ScreenError):
        write_b3f1_artifacts(
            scorecard=[], pair_robustness=[], metadata={}, output_dir=output_dir
        )
