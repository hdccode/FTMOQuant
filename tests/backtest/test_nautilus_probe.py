from decimal import Decimal

from ftmoquant.backtest.nautilus_probe import run_synthetic_eurusd_probe


def test_nautilus_v2_synthetic_eurusd_probe_exposes_overlay_data() -> None:
    result = run_synthetic_eurusd_probe()

    assert result.engine_version == "2.0.0rc2"
    assert len(result.market_timestamps_ns) == 3
    assert result.order_timestamps_ns == result.fill_timestamps_ns
    assert result.fill_timestamps_ns == result.market_timestamps_ns[:2]
    assert result.fill_prices == (Decimal("1.10002"), Decimal("1.10100"))
    assert len(set(result.position_ids)) == 1
    assert result.fees == (Decimal("2.20"), Decimal("2.20"))
    assert result.realized_pnl == Decimal("93.60")
    assert result.balance == Decimal("100093.60")
    assert result.equity == result.balance
    assert result.account_state_timestamps_ns[-2:] == result.fill_timestamps_ns
    assert result.report_rows == (
        ("orders", 2),
        ("fills", 2),
        ("positions", 1),
        ("account", 3),
    )


def test_nautilus_v2_synthetic_eurusd_probe_is_deterministic() -> None:
    assert run_synthetic_eurusd_probe() == run_synthetic_eurusd_probe()
