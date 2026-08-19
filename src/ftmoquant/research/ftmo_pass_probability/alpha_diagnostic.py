"""Read-only DEVELOPMENT-vs-VALIDATION alpha/distribution diagnostic.

Alpha first, FTMO optimization second: this module exists to describe *why*
the underlying trade stream deteriorated between DEVELOPMENT and VALIDATION
(see ``docs`` / the frozen-policy VALIDATION diagnostic that motivated it),
not to tune, select, rank, or re-derive anything. It is diagnostic-only --
every artifact it writes is labelled ``diagnostic_only: true``.

Reuse, not parallel logic: DEVELOPMENT/VALIDATION loading and the
holdout/partition firewall come unchanged from ``path_extraction.py``; the
frozen sizing policy comes unchanged from ``validation_diagnostic.
frozen_policy()``; the single chronological two-phase replay in Diagnostic E
reuses ``sizing.apply_sizing``, ``monte_carlo.size_synthetic_path``,
``monte_carlo._censor_if_active``, and ``state_machine.simulate_phase``
directly -- no bootstrap, no resampling, no new state machine. Percentiles
reuse ``reporting._percentile``.

No Monte Carlo, no bootstrap, no alternative sizing policy, no alternative
resampling method, and no new regime/subgroup definitions are introduced
anywhere in this module.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from statistics import median, pstdev
from typing import Any

from ftmoquant.prop_rules.models import EvaluationPhase, PropRuleSet
from ftmoquant.research.ftmo_pass_probability.monte_carlo import (
    TwoPhaseOutcome,
    _censor_if_active,
    size_synthetic_path,
)
from ftmoquant.research.ftmo_pass_probability.path_extraction import TradeRecord
from ftmoquant.research.ftmo_pass_probability.reporting import _percentile
from ftmoquant.research.ftmo_pass_probability.sizing import SizingPolicy, apply_sizing
from ftmoquant.research.ftmo_pass_probability.state_machine import simulate_phase
from ftmoquant.research.ftmo_pass_probability.validation_diagnostic import frozen_policy

#: Fixed-notional sizing (this diagnostic's frozen policy) does not depend
#: on account balance -- this constant only satisfies apply_sizing's
#: positive-balance precondition and never affects the sized result.
_NOMINAL_BALANCE = Decimal("100000")
_DAY_NS = 86_400_000_000_000
_YEAR_DAYS = Decimal("365.25")

#: Subgroup dimensions audited for pre-existing (not post-hoc) eligibility.
_SESSION_COLUMN_PRESENT = False
_REGIME_COLUMN_PRESENT = False
_SETUP_CATEGORY_COLUMN_PRESENT = False


class AlphaDiagnosticError(ValueError):
    """Raised when the diagnostic cannot be computed safely."""


def frozen_policy_pnl_series(trades: tuple[TradeRecord, ...]) -> tuple[Decimal, ...]:
    """Chronological, frozen-policy (fixed_notional_2_0x) dollar P/L per trade.

    Reuses ``sizing.apply_sizing`` unchanged. Fixed-notional sizing does not
    depend on running balance, so this is a pure per-trade function of the
    trade itself -- there is no path/order dependence to preserve here
    (unlike a fixed-fractional policy, which this diagnostic does not use).
    """

    policy = frozen_policy()
    return tuple(
        apply_sizing(policy, trade, _NOMINAL_BALANCE).realized_pnl for trade in trades
    )


def _compare(development: Any, validation: Any) -> dict[str, Any]:
    """Development/validation/absolute-difference/relative-difference table."""

    absolute_difference: float | None
    relative_difference: float | None
    if development is None or validation is None:
        absolute_difference = None
        relative_difference = None
    else:
        development_f = float(development)
        validation_f = float(validation)
        absolute_difference = validation_f - development_f
        relative_difference = (
            absolute_difference / abs(development_f) if development_f != 0 else None
        )
    return {
        "development": _plain(development),
        "validation": _plain(validation),
        "absolute_difference": absolute_difference,
        "relative_difference": relative_difference,
    }


def _plain(value: Any) -> Any:
    return float(value) if isinstance(value, Decimal) else value


# ---------------------------------------------------------------------------
# Diagnostic A -- aggregate trade distribution
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TradeDistribution:
    """One partition's aggregate frozen-policy trade-economics summary."""

    trade_count: int
    span_days: float
    trades_per_year: float
    mean_pnl: float
    median_pnl: float
    stdev_pnl: float
    win_rate: float
    average_win: float | None
    average_loss: float | None
    payoff_ratio: float | None
    profit_factor: float | None
    expectancy: float
    percentiles: dict[str, float]
    best_trade: float
    worst_trade: float
    max_consecutive_wins: int
    max_consecutive_losses: int


def diagnostic_a_trade_distribution(
    trades: tuple[TradeRecord, ...], pnls: tuple[Decimal, ...]
) -> TradeDistribution:
    if not trades or not pnls or len(trades) != len(pnls):
        raise AlphaDiagnosticError("trades and pnls must be equal-length, non-empty")

    span_ns = trades[-1].exit_ns - trades[0].entry_ns
    span_days = span_ns / Decimal(_DAY_NS)
    trades_per_year = (
        Decimal(len(trades)) / (span_days / _YEAR_DAYS) if span_days > 0 else Decimal(0)
    )

    floats = [float(pnl) for pnl in pnls]
    wins = [value for value in floats if value > 0]
    losses = [value for value in floats if value < 0]
    gross_profit = sum(wins)
    gross_loss = -sum(losses)
    average_win = (gross_profit / len(wins)) if wins else None
    average_loss = (gross_loss / len(losses)) if losses else None
    payoff_ratio = (
        average_win / average_loss
        if average_win is not None and average_loss not in (None, 0)
        else None
    )
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None

    max_wins, max_losses = _consecutive_streaks(floats)

    return TradeDistribution(
        trade_count=len(trades),
        span_days=float(span_days),
        trades_per_year=float(trades_per_year),
        mean_pnl=sum(floats) / len(floats),
        median_pnl=median(floats),
        stdev_pnl=pstdev(floats) if len(floats) > 1 else 0.0,
        win_rate=len(wins) / len(floats),
        average_win=average_win,
        average_loss=average_loss,
        payoff_ratio=payoff_ratio,
        profit_factor=profit_factor,
        expectancy=sum(floats) / len(floats),
        percentiles={
            label: _percentile(floats, quantile)
            for label, quantile in (
                ("p05", 0.05),
                ("p10", 0.10),
                ("p25", 0.25),
                ("p50", 0.50),
                ("p75", 0.75),
                ("p90", 0.90),
                ("p95", 0.95),
            )
        },
        best_trade=max(floats),
        worst_trade=min(floats),
        max_consecutive_wins=max_wins,
        max_consecutive_losses=max_losses,
    )


def _consecutive_streaks(pnls: Sequence[float]) -> tuple[int, int]:
    max_wins = current_wins = 0
    max_losses = current_losses = 0
    for value in pnls:
        if value > 0:
            current_wins += 1
            current_losses = 0
        elif value < 0:
            current_losses += 1
            current_wins = 0
        else:
            current_wins = current_losses = 0
        max_wins = max(max_wins, current_wins)
        max_losses = max(max_losses, current_losses)
    return max_wins, max_losses


def compare_trade_distributions(
    development: TradeDistribution, validation: TradeDistribution
) -> dict[str, Any]:
    comparison = {
        "trade_count": _compare(development.trade_count, validation.trade_count),
        "span_days": _compare(development.span_days, validation.span_days),
        "trades_per_year": _compare(
            development.trades_per_year, validation.trades_per_year
        ),
        "mean_pnl": _compare(development.mean_pnl, validation.mean_pnl),
        "median_pnl": _compare(development.median_pnl, validation.median_pnl),
        "stdev_pnl": _compare(development.stdev_pnl, validation.stdev_pnl),
        "win_rate": _compare(development.win_rate, validation.win_rate),
        "average_win": _compare(development.average_win, validation.average_win),
        "average_loss": _compare(development.average_loss, validation.average_loss),
        "payoff_ratio": _compare(development.payoff_ratio, validation.payoff_ratio),
        "profit_factor": _compare(development.profit_factor, validation.profit_factor),
        "expectancy": _compare(development.expectancy, validation.expectancy),
        "best_trade": _compare(development.best_trade, validation.best_trade),
        "worst_trade": _compare(development.worst_trade, validation.worst_trade),
        "max_consecutive_wins": _compare(
            development.max_consecutive_wins, validation.max_consecutive_wins
        ),
        "max_consecutive_losses": _compare(
            development.max_consecutive_losses, validation.max_consecutive_losses
        ),
    }
    for label in development.percentiles:
        comparison[f"percentile_{label}"] = _compare(
            development.percentiles[label], validation.percentiles[label]
        )
    return comparison


# ---------------------------------------------------------------------------
# Diagnostic B -- chronological path behaviour (no bootstrapping)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RollingWindowSummary:
    window_size: int
    observation_count: int
    minimum: float | None
    median_value: float | None
    maximum: float | None
    final_value: float | None


@dataclass(frozen=True, slots=True)
class ChronologicalPathSummary:
    terminal_cumulative_pnl: float
    maximum_drawdown_fraction_of_initial_capital: float
    maximum_drawdown_dollars: float
    longest_completed_underwater_days: float | None
    final_stretch_still_underwater: bool
    final_stretch_underwater_days: float | None
    rolling: dict[str, RollingWindowSummary]


def _rolling_expectancy(
    pnls: Sequence[Decimal], window: int
) -> tuple[RollingWindowSummary, list[float]]:
    if window <= 0:
        raise AlphaDiagnosticError("window must be positive")
    floats = [float(value) for value in pnls]
    series = [
        sum(floats[index - window + 1 : index + 1]) / window
        for index in range(window - 1, len(floats))
    ]
    if not series:
        return (
            RollingWindowSummary(window, 0, None, None, None, None),
            [],
        )
    return (
        RollingWindowSummary(
            window_size=window,
            observation_count=len(series),
            minimum=min(series),
            median_value=median(series),
            maximum=max(series),
            final_value=series[-1],
        ),
        series,
    )


def diagnostic_b_chronological_path(
    trades: tuple[TradeRecord, ...],
    pnls: tuple[Decimal, ...],
    initial_capital: Decimal,
) -> tuple[ChronologicalPathSummary, dict[int, list[float]]]:
    if not trades or len(trades) != len(pnls):
        raise AlphaDiagnosticError("trades and pnls must be equal-length, non-empty")

    equity = initial_capital
    peak = initial_capital
    peak_entry_ns = trades[0].entry_ns
    max_drawdown_fraction = Decimal(0)
    max_drawdown_dollars = Decimal(0)
    completed_underwater_ns: list[int] = []
    underwater = False

    for trade, pnl in zip(trades, pnls):
        equity += pnl
        if equity >= peak:
            if underwater:
                completed_underwater_ns.append(trade.entry_ns - peak_entry_ns)
                underwater = False
            peak = equity
            peak_entry_ns = trade.entry_ns
        else:
            underwater = True
            drawdown_dollars = peak - equity
            drawdown_fraction = drawdown_dollars / initial_capital
            max_drawdown_dollars = max(max_drawdown_dollars, drawdown_dollars)
            max_drawdown_fraction = max(max_drawdown_fraction, drawdown_fraction)

    final_stretch_underwater_ns = (
        (trades[-1].exit_ns - peak_entry_ns) if underwater else None
    )
    longest_completed_underwater_days = (
        max(completed_underwater_ns) / _DAY_NS if completed_underwater_ns else None
    )

    rolling: dict[str, RollingWindowSummary] = {}
    rolling_series: dict[int, list[float]] = {}
    for window in (30, 50):
        window_summary, series = _rolling_expectancy(pnls, window)
        rolling[str(window)] = window_summary
        rolling_series[window] = series

    summary = ChronologicalPathSummary(
        terminal_cumulative_pnl=float(sum(pnls, Decimal(0))),
        maximum_drawdown_fraction_of_initial_capital=float(max_drawdown_fraction),
        maximum_drawdown_dollars=float(max_drawdown_dollars),
        longest_completed_underwater_days=longest_completed_underwater_days,
        final_stretch_still_underwater=underwater,
        final_stretch_underwater_days=(
            None
            if final_stretch_underwater_ns is None
            else final_stretch_underwater_ns / _DAY_NS
        ),
        rolling=rolling,
    )
    return summary, rolling_series


# ---------------------------------------------------------------------------
# Diagnostic C -- temporal stability (natural calendar groupings only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CalendarStability:
    granularity: str
    eligible_periods: int
    positive_periods: int
    periods: dict[str, dict[str, float]]
    best_period: str | None
    worst_period: str | None


def _calendar_key(entry_ns: int, *, quarterly: bool) -> str:
    moment = datetime.fromtimestamp(entry_ns / 1_000_000_000, tz=UTC)
    if quarterly:
        quarter = (moment.month - 1) // 3 + 1
        return f"{moment.year}-Q{quarter}"
    return f"{moment.year}-{moment.month:02d}"


def _calendar_stability(
    trades: tuple[TradeRecord, ...], pnls: tuple[Decimal, ...], *, quarterly: bool
) -> CalendarStability:
    groups: dict[str, list[float]] = {}
    for trade, pnl in zip(trades, pnls):
        key = _calendar_key(trade.entry_ns, quarterly=quarterly)
        groups.setdefault(key, []).append(float(pnl))

    periods = {
        key: {
            "trade_count": len(values),
            "total_pnl": sum(values),
            "expectancy": sum(values) / len(values),
        }
        for key, values in sorted(groups.items())
    }
    positive_periods = sum(1 for value in periods.values() if value["total_pnl"] > 0)
    best_period = (
        max(periods, key=lambda key: periods[key]["total_pnl"]) if periods else None
    )
    worst_period = (
        min(periods, key=lambda key: periods[key]["total_pnl"]) if periods else None
    )
    return CalendarStability(
        granularity="quarterly" if quarterly else "monthly",
        eligible_periods=len(periods),
        positive_periods=positive_periods,
        periods=periods,
        best_period=best_period,
        worst_period=worst_period,
    )


def diagnostic_c_temporal_stability(
    trades: tuple[TradeRecord, ...], pnls: tuple[Decimal, ...]
) -> dict[str, CalendarStability]:
    return {
        "monthly": _calendar_stability(trades, pnls, quarterly=False),
        "quarterly": _calendar_stability(trades, pnls, quarterly=True),
    }


# ---------------------------------------------------------------------------
# Diagnostic D -- existing preregistered subgroups only
# ---------------------------------------------------------------------------


def subgroup_eligibility() -> list[dict[str, Any]]:
    """The complete, honest audit of candidate subgroup dimensions.

    Only ``direction`` (long/short) is a genuinely preregistered dimension:
    it is a raw ``trades.csv`` column recorded by the frozen execution
    itself, fixed before this diagnostic and before the VALIDATION result
    it explains. Every other candidate is explicitly rejected here rather
    than silently omitted, with the reason recorded.
    """

    return [
        {
            "name": "direction",
            "eligible": True,
            "reason": (
                "raw trades.csv column recorded by the frozen execution "
                "(long=1/short=-1); preregistered, not derived from this "
                "diagnostic's own results"
            ),
        },
        {
            "name": "exit_reason",
            "eligible": False,
            "reason": (
                "outcome-conditioned, not a preregistered setup: whether a "
                "trade exits via stop or target is a result of the trade, "
                "almost perfectly correlated with win/loss -- splitting on "
                "it would be tautological, not a genuine subgroup"
            ),
        },
        {
            "name": "session",
            "eligible": _SESSION_COLUMN_PRESENT,
            "reason": "no session column exists in trades.csv",
        },
        {
            "name": "regime_state",
            "eligible": _REGIME_COLUMN_PRESENT,
            "reason": "no preregistered regime/state column exists in trades.csv",
        },
        {
            "name": "setup_category",
            "eligible": _SETUP_CATEGORY_COLUMN_PRESENT,
            "reason": (
                "the frozen signal (B2F1_sweep_bos_retest) has a single "
                "setup; no preregistered setup/category column exists"
            ),
        },
    ]


def _read_directions(execution_dir: Path) -> dict[int, int]:
    """Re-read the already-loaded, already-validated trades.csv's own
    ``direction`` column, keyed by trade_index. Read-only; does not
    re-validate partition/holdout (the caller already did, via the
    standard loader) and adds no new trade-selection logic.
    """

    path = execution_dir / "trades.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            int(row["trade_index"]): int(row["direction"])
            for row in csv.DictReader(handle)
        }


def diagnostic_d_subgroups(
    execution_dir: Path, trades: tuple[TradeRecord, ...], pnls: tuple[Decimal, ...]
) -> dict[str, dict[str, Any]] | None:
    directions = _read_directions(execution_dir)
    groups: dict[str, list[float]] = {"long": [], "short": []}
    for trade, pnl in zip(trades, pnls):
        direction = directions.get(trade.trade_index)
        if direction is None:
            raise AlphaDiagnosticError(
                f"trade {trade.trade_index} is missing a direction value"
            )
        groups["long" if direction > 0 else "short"].append(float(pnl))

    result: dict[str, dict[str, Any]] = {}
    for label, values in groups.items():
        if not values:
            result[label] = {
                "trade_count": 0,
                "mean_pnl": None,
                "win_rate": None,
                "profit_factor": None,
            }
            continue
        wins = [value for value in values if value > 0]
        losses = [value for value in values if value < 0]
        gross_profit = sum(wins)
        gross_loss = -sum(losses)
        result[label] = {
            "trade_count": len(values),
            "mean_pnl": sum(values) / len(values),
            "win_rate": len(wins) / len(values),
            "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else None,
        }
    return result


# ---------------------------------------------------------------------------
# Diagnostic E -- descriptive chronological frozen-policy replay (no MC)
# ---------------------------------------------------------------------------


def diagnostic_e_chronological_replay(
    trades: tuple[TradeRecord, ...],
    rules: PropRuleSet,
    initial_capital: Decimal,
) -> TwoPhaseOutcome:
    """Replay one partition's real trades, in their real chronological
    order, exactly once, under the frozen fixed_notional_2_0x policy.

    Deliberately does not go through ``bootstrap.draw_index_path`` or
    ``monte_carlo.simulate_two_phase_path`` (both are resampling
    machinery); it builds the *identity* placement (each trade at its own
    real ``entry_ns``/``exit_ns``) and reuses ``monte_carlo.
    size_synthetic_path`` (causal sizing) and ``state_machine.
    simulate_phase`` (breach/pass semantics) directly -- the same reusable
    pieces the Monte Carlo layer is built from, with no resampling step
    inserted. Verification, if Challenge passes, continues from the same
    real trade sequence with fresh capital, mirroring ``simulate_two_phase_
    path``'s own convention.
    """

    if not trades:
        raise AlphaDiagnosticError("trades must not be empty")

    policy: SizingPolicy = frozen_policy()
    placements = tuple((trade, trade.entry_ns, trade.exit_ns) for trade in trades)
    total_span_ns = trades[-1].exit_ns - trades[0].entry_ns + 1

    challenge_events = size_synthetic_path(policy, placements, initial_capital)
    challenge_outcome = _censor_if_active(
        simulate_phase(
            challenge_events,
            rules=rules,
            phase=EvaluationPhase.CHALLENGE,
            initial_capital=initial_capital,
            horizon_ns=total_span_ns,
        )
    )
    if challenge_outcome.status.value != "passed":
        return TwoPhaseOutcome(
            challenge=challenge_outcome, verification=None, passed_both=False
        )

    remaining_placements = placements[challenge_outcome.trades_replayed :]
    verification_events = size_synthetic_path(
        policy, remaining_placements, initial_capital
    )
    remaining_span_ns = (
        (remaining_placements[-1][2] - remaining_placements[0][1] + 1)
        if remaining_placements
        else 1
    )
    verification_outcome = _censor_if_active(
        simulate_phase(
            verification_events,
            rules=rules,
            phase=EvaluationPhase.VERIFICATION,
            initial_capital=initial_capital,
            horizon_ns=remaining_span_ns,
        )
    )
    return TwoPhaseOutcome(
        challenge=challenge_outcome,
        verification=verification_outcome,
        passed_both=verification_outcome.status.value == "passed",
    )
