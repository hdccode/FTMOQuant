from decimal import Decimal

from ftmoquant.research.trend_pullback_postmortem import (
    _cost_diagnostics,
    _session,
    development_diagnostics,
)


def test_development_diagnostics_predefined_groups_only() -> None:
    trades = (
        _trade("long", "2", "2020-01-06T02:00:00Z", "take_profit", "0.001"),
        _trade("short", "-1", "2020-01-07T09:00:00Z", "stop_loss", "0.002"),
        _trade("long", "-1", "2021-01-08T14:00:00Z", "stop_loss", "0.003"),
        _trade("short", "2", "2021-01-10T22:00:00Z", "take_profit", "0.004"),
    )

    result = development_diagnostics(trades)

    assert result["overall"]["trade_count"] == 4
    assert result["overall"]["mean_r"] == "0.5"
    assert [row["group"] for row in result["direction"]] == ["long", "short"]
    assert [row["group"] for row in result["calendar_year_by_exit"]] == [
        "2020",
        "2021",
    ]
    assert len(result["signal_state"]["stop_distance_quartiles"]) == 4


def test_session_boundaries_are_fixed_utc_convention() -> None:
    assert _session(_trade("long", "1", "2020-01-01T07:59:00Z")) == "Asia"
    assert _session(_trade("long", "1", "2020-01-01T08:00:00Z")) == "London"
    assert _session(_trade("long", "1", "2020-01-01T13:00:00Z")) == "New_York"
    assert _session(_trade("long", "1", "2020-01-01T22:00:00Z")) == "Rollover_off_hours"


def test_cost_diagnostic_does_not_claim_unobserved_pre_spread_gross() -> None:
    result = _cost_diagnostics((_trade("long", "-1", "2020-01-01T09:00:00Z"),))

    assert result["commission_total_currency"] == "0"
    assert "counterfactual pre-spread gross R" in result["unavailable"]
    assert "Spread is already embedded" in result["interpretation"]


def _trade(
    direction: str,
    net_r: str,
    entry: str,
    exit_reason: str = "time",
    stop_distance: str = "0.001",
) -> dict[str, object]:
    entry_price = Decimal("1.1000")
    price_change = Decimal(net_r) * Decimal(stop_distance)
    exit_price = (
        entry_price + price_change
        if direction == "long"
        else entry_price - price_change
    )
    return {
        "commissions": "0",
        "direction": direction,
        "entry_price": str(entry_price),
        "entry_time_utc": entry,
        "exit_price": str(exit_price),
        "exit_reason": exit_reason,
        "exit_time_utc": entry,
        "initial_risk": "1",
        "net_r": net_r,
        "quantity": str(Decimal(1) / Decimal(stop_distance)),
        "realized_pnl": net_r,
        "stop_distance": stop_distance,
    }
