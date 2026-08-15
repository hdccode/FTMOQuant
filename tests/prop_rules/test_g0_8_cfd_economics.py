from datetime import UTC, datetime
from decimal import Decimal

import pytest

from ftmoquant.prop_rules.g0_8_cfd_economics import (
    G0_8_CFD_ECONOMICS_SHA256,
    G08CfdEconomicsError,
    load_g08_cfd_economics,
)


def test_contract_pnl_commissions_and_margin() -> None:
    economics = load_g08_cfd_economics()
    assert economics.semantic_sha256 == G0_8_CFD_ECONOMICS_SHA256
    assert economics.gross_pnl(
        "EUR/USD", 1, Decimal("1.1"), Decimal("1.101"), Decimal("1")
    ) == Decimal("100.000")
    assert economics.gross_pnl(
        "XAU/USD", -1, Decimal("2000"), Decimal("1990"), Decimal("2")
    ) == Decimal("2000")
    assert economics.gross_pnl(
        "US500.cash", 1, Decimal("5000"), Decimal("5010"), Decimal("3")
    ) == Decimal("30")
    assert economics.gross_pnl(
        "USOIL.cash", 1, Decimal("70"), Decimal("71"), Decimal("1")
    ) == Decimal("100")
    assert economics.gross_pnl(
        "SOYBEAN.c", -1, Decimal("1200"), Decimal("1190"), Decimal("1")
    ) == Decimal("10")
    assert economics.side_commission(
        "EUR/USD", Decimal("1.1"), Decimal("2")
    ) == Decimal("5")
    assert economics.side_commission(
        "XAU/USD", Decimal("2000"), Decimal("1")
    ) == Decimal("1.400000")
    assert economics.side_commission("US500.cash", Decimal("5000"), Decimal("1")) == 0
    assert economics.margin_requirement(
        "EUR/USD", Decimal("1.2"), Decimal("1")
    ) == Decimal("4000")
    assert economics.margin_requirement(
        "XAU/USD", Decimal("2000"), Decimal("1")
    ) == Decimal("13333.33333333333333333333333")


def test_server_dst_sessions_rollover_and_unknown_symbol() -> None:
    economics = load_g08_cfd_economics()
    assert economics.is_session_eligible(
        "EUR/USD", datetime(2026, 1, 5, 22, 5, tzinfo=UTC)
    )
    assert economics.is_session_eligible(
        "EUR/USD", datetime(2026, 7, 6, 21, 5, tzinfo=UTC)
    )
    assert not economics.is_session_eligible(
        "SOYBEAN.c", datetime(2026, 7, 6, 13, 0, tzinfo=UTC)
    )
    assert economics.is_session_eligible(
        "SOYBEAN.c", datetime(2026, 7, 6, 14, 40, tzinfo=UTC)
    )
    assert "not fully calibrated" in economics.rollover_warning
    with pytest.raises(G08CfdEconomicsError, match="unknown"):
        economics.contract("UNKNOWN")
