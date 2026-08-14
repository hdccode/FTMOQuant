from decimal import Decimal

from ftmoquant.research.trend_pullback_experiment import (
    DAY_NS,
    _gates,
    _metrics,
    canonical_execution_profile,
)
from ftmoquant.strategies.trend_pullback import (
    CompletedTrade,
    Direction,
    ExitReason,
)


def test_canonical_profile_is_explicit_uncalibrated_g07_baseline() -> None:
    profile = canonical_execution_profile()

    assert profile.random_seed == 7
    assert profile.adverse_slippage_probability == 0
    assert profile.base_latency_ns == 0
    assert profile.fee.commission == 0
    assert profile.rollover.records == ()
    assert not profile.adaptive_high_low_ordering


def test_metrics_use_native_net_pnl_over_actual_initial_risk() -> None:
    trades = (
        _trade(Direction.LONG, Decimal("2"), 1),
        _trade(Direction.SHORT, Decimal("-1"), 2),
        _trade(Direction.LONG, Decimal("0.5"), 3),
    )

    result = _metrics(trades, 0, 365 * DAY_NS)

    assert result["trade_count"] == 3
    assert result["winners"] == 2
    assert result["losers"] == 1
    assert result["mean_net_r"] == "0.5"
    assert result["median_net_r"] == "0.5"
    assert result["total_net_r"] == "1.5"
    assert result["profit_factor"] == "2.5"
    assert result["max_drawdown_r"] == "1"
    assert result["direction_counts"] == {"long": 2, "short": 1}


def test_insufficient_counts_are_unresolved_and_bad_total_fails_year_gate() -> None:
    dev = _metrics((_trade(Direction.LONG, Decimal("1"), 1),), 0, DAY_NS)
    val = _metrics((_trade(Direction.SHORT, Decimal("-1"), 2),), 0, DAY_NS)

    gates = _gates(dev, val, None)

    assert gates["development_trade_count_gte_100"] == "UNRESOLVED"
    assert gates["validation_trade_count_gte_50"] == "UNRESOLVED"
    assert gates["validation_bca_95_lower_bound_gt_0"] == "UNRESOLVED"
    assert gates["validation_calendar_concentration_lte_0_50"] == "FAIL"


def _trade(direction: Direction, net_r: Decimal, day: int) -> CompletedTrade:
    initial_risk = Decimal("2")
    return CompletedTrade(
        direction=direction,
        entry_time_ns=day * DAY_NS,
        exit_time_ns=day * DAY_NS + 1,
        entry_price=Decimal("1.1"),
        exit_price=Decimal("1.2"),
        quantity=Decimal("1000"),
        stop_distance=Decimal("0.002"),
        initial_risk=initial_risk,
        realized_pnl=net_r * initial_risk,
        net_r=net_r,
        commissions=Decimal(0),
        exit_reason=ExitReason.TAKE_PROFIT if net_r > 0 else ExitReason.STOP_LOSS,
    )
