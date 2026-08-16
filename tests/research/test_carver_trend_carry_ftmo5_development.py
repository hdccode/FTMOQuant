from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pytest

import ftmoquant.research.carver_trend_carry_ftmo5_development as carver
from ftmoquant.prop_rules.g0_8_cfd_economics import load_g08_cfd_economics
from ftmoquant.research.carver_trend_carry_ftmo5_development import (
    CarverTrendCarryFtmo5EvaluationError,
    causal_carry,
    causal_ewmac,
    cfd_margin_requirement,
    cfd_net_pnl,
    combine_forecasts,
    comparison_fold,
    first_eligible_cfd_execution,
    first_strictly_later_execution,
    normalize_execution_price,
    stressed_cost,
    verify_reference_sources,
)
from ftmoquant.research.carver_trend_carry_ftmo5_spec import (
    CARVER_TREND_CARRY_FTMO5_CONFIG_SHA256,
    load_carver_trend_carry_ftmo5_spec,
)


def test_reference_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    spec = load_carver_trend_carry_ftmo5_spec()
    document = deepcopy(spec.canonical_document)
    source = next(iter(document["provenance"]["source_sha256"]))
    document["provenance"]["source_sha256"] = {source: "0" * 64}
    (tmp_path / source).parent.mkdir(parents=True)
    (tmp_path / source).write_text("synthetic", encoding="utf-8")
    with pytest.raises(CarverTrendCarryFtmo5EvaluationError, match="SHA mismatch"):
        verify_reference_sources(tmp_path, replace(spec, canonical_document=document))


def test_causal_ewmac_carry_caps_weights_and_fdm() -> None:
    index = pd.date_range("2020-01-01", periods=300, freq="D")
    price = pd.Series(range(100, 400), index=index, dtype=float)
    multiple = pd.DataFrame(
        {
            "PRICE": price,
            "CARRY": price + 2,
            "PRICE_CONTRACT": 202406,
            "CARRY_CONTRACT": 202409,
        },
        index=index,
    )
    trend = causal_ewmac(price, 16, 64, 3.75)
    carry = causal_carry(multiple)
    forecasts = combine_forecasts(price, multiple)
    assert trend.dropna().abs().max() <= 20
    assert carry.dropna().abs().max() <= 20
    expected = (
        forecasts.trend_16_64 * 0.21
        + forecasts.trend_32_128 * 0.08
        + forecasts.trend_64_256 * 0.21
        + forecasts.carry * 0.50
    ) * 1.31
    pd.testing.assert_series_equal(forecasts.combined, expected)


def test_strict_later_fold_warmup_and_cost_stress() -> None:
    signal = datetime(2020, 5, 1, tzinfo=UTC)
    assert first_strictly_later_execution(
        signal, (signal, signal + timedelta(minutes=1))
    ) == signal + timedelta(minutes=1)
    assert comparison_fold(datetime(2019, 5, 1, tzinfo=UTC)) is None
    assert comparison_fold(signal) == "dev_fold_1"
    assert comparison_fold(datetime(2021, 5, 1, tzinfo=UTC)) == "dev_fold_2"
    assert comparison_fold(datetime(2022, 5, 1, tzinfo=UTC)) == "dev_fold_3"
    assert stressed_cost(0.01) == (0.01, 0.015)
    assert (
        CARVER_TREND_CARRY_FTMO5_CONFIG_SHA256
        == "489b53abff19e041afda9cc5ba210a67252bf89dea87c8b9994f08c4422e210d"
    )


def test_all_five_g08_mappings_pnl_commission_margin_and_sessions() -> None:
    economics = load_g08_cfd_economics()
    mappings = {
        "EUR/USD.DUKASCOPY": "EUR/USD",
        "XAU_USD.OANDA": "XAU/USD",
        "SPX500_USD.OANDA": "US500.cash",
        "WTICO_USD.OANDA": "USOIL.cash",
        "SOYBN_USD.OANDA": "SOYBEAN.c",
    }
    assert set(mappings) == set(carver._MAPPING)
    assert cfd_net_pnl(
        economics, "EUR/USD.DUKASCOPY", 1, Decimal("1"), Decimal("1.01"), Decimal("1")
    ) == Decimal("995.000")
    assert cfd_net_pnl(
        economics,
        "XAU_USD.OANDA",
        1,
        Decimal("2000"),
        Decimal("2010"),
        Decimal("1"),
    ) == Decimal("997.193000")
    assert cfd_net_pnl(
        economics, "SPX500_USD.OANDA", 1, Decimal("5000"), Decimal("5010"), Decimal("1")
    ) == Decimal("10")
    assert cfd_net_pnl(
        economics,
        "WTICO_USD.OANDA",
        1,
        Decimal("70"),
        Decimal("71"),
        Decimal("1"),
    ) == Decimal("100")
    assert cfd_net_pnl(
        economics,
        "SOYBN_USD.OANDA",
        1,
        Decimal("12"),
        Decimal("12.1"),
        Decimal("1"),
    ) == Decimal("10")
    assert normalize_execution_price("SOYBN_USD.OANDA", Decimal("15.2")) == Decimal(
        "1520.0"
    )
    assert normalize_execution_price("XAU_USD.OANDA", Decimal("2000")) == Decimal(
        "2000"
    )
    assert cfd_margin_requirement(
        economics, "SOYBN_USD.OANDA", Decimal("12"), Decimal("1")
    ) == Decimal("1200")
    assert cfd_margin_requirement(
        economics, "SPX500_USD.OANDA", Decimal("5000"), Decimal("1")
    ) == Decimal("333.3333333333333333333333333")
    signal = datetime(2026, 7, 6, 13, 0, tzinfo=UTC)
    assert first_eligible_cfd_execution(
        signal,
        (signal, datetime(2026, 7, 6, 14, 40, tzinfo=UTC)),
        "SOYBN_USD.OANDA",
        economics,
    ) == datetime(2026, 7, 6, 14, 40, tzinfo=UTC)


def test_g08_sha_drift_fails_closed_and_rollover_warning_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    economics = load_g08_cfd_economics()
    monkeypatch.setattr(
        carver,
        "load_g08_cfd_economics",
        lambda: replace(economics, semantic_sha256="0" * 64),
    )
    with pytest.raises(
        CarverTrendCarryFtmo5EvaluationError, match="G0.8 economics SHA drifted"
    ):
        carver._frozen_g08_economics()
    assert "not fully calibrated" in economics.rollover_warning
