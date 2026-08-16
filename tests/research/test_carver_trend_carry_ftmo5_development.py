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
    DesiredPosition,
    ExecutionObservation,
    SyntheticFoldInput,
    build_desired_positions,
    causal_carry,
    causal_ewmac,
    cfd_margin_requirement,
    cfd_net_pnl,
    combine_forecasts,
    comparison_fold,
    evaluate_synthetic_fold,
    first_eligible_cfd_execution,
    first_strictly_later_execution,
    forecast_to_desired_lots,
    normalize_execution_price,
    pinned_mixed_daily_price_volatility,
    stressed_cost,
    summarize_development,
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
            "CARRY": price + 2,
            "CARRY_CONTRACT": 202409,
            "PRICE": price,
            "PRICE_CONTRACT": 202406,
            "FORWARD": price + 1,
            "FORWARD_CONTRACT": 202412,
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


def test_carry_requires_exact_pinned_six_column_schema_and_ignores_forward() -> None:
    index = pd.date_range("2020-01-01", periods=300, freq="D")
    price = pd.Series(range(100, 400), index=index, dtype=float)
    multiple = pd.DataFrame(
        {
            "CARRY": price + 2,
            "CARRY_CONTRACT": 202409,
            "PRICE": price,
            "PRICE_CONTRACT": 202406,
            "FORWARD": price + 1,
            "FORWARD_CONTRACT": 202412,
        },
        index=index,
    )
    expected = causal_carry(multiple)
    changed_forward = multiple.copy()
    changed_forward["FORWARD"] = price * 1000
    changed_forward["FORWARD_CONTRACT"] = 209912
    pd.testing.assert_series_equal(causal_carry(changed_forward), expected)

    with pytest.raises(
        CarverTrendCarryFtmo5EvaluationError,
        match="multiple-price columns are not exact",
    ):
        causal_carry(multiple.drop(columns="FORWARD"))

    unexpected = multiple.assign(UNEXPECTED=1)
    with pytest.raises(
        CarverTrendCarryFtmo5EvaluationError,
        match="multiple-price columns are not exact",
    ):
        causal_carry(unexpected)


def test_strict_later_fold_warmup_and_cost_stress() -> None:
    signal = datetime(2020, 5, 1, tzinfo=UTC)
    assert first_strictly_later_execution(
        signal, (signal, signal + timedelta(minutes=1))
    ) == signal + timedelta(minutes=1)
    assert first_strictly_later_execution(
        signal, (signal, signal + timedelta(minutes=15))
    ) == signal + timedelta(minutes=15)
    assert comparison_fold(datetime(2019, 5, 1, tzinfo=UTC)) is None
    assert comparison_fold(signal) == "dev_fold_1"
    assert comparison_fold(datetime(2021, 5, 1, tzinfo=UTC)) == "dev_fold_2"
    assert comparison_fold(datetime(2022, 5, 1, tzinfo=UTC)) == "dev_fold_3"
    assert stressed_cost(0.01) == (0.01, 0.015)
    assert (
        CARVER_TREND_CARRY_FTMO5_CONFIG_SHA256
        == "f1831cf1cdedeedfc21610054da8e542796c8b3c0cbc26a62074bdbf1ab39365"
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


def test_frozen_sizing_is_continuous_and_uses_carver_cash_volatility() -> None:
    spec = load_carver_trend_carry_ftmo5_spec()
    result = forecast_to_desired_lots(
        combined_forecast=Decimal("10"),
        daily_proxy_price_volatility=Decimal("10"),
        contract_size=Decimal("100"),
        instrument_weight=Decimal("0.2"),
        spec=spec,
    )
    assert result == Decimal("1.5625")
    assert result != result.to_integral_value()
    day = pd.Timestamp("2020-04-12T00:00:00Z")
    desired = build_desired_positions(
        "XAU_USD.OANDA",
        pd.Series([10.0], index=pd.DatetimeIndex([day])),
        pd.Series([10.0], index=pd.DatetimeIndex([day])),
        load_g08_cfd_economics(),
        spec,
    )
    assert desired[0].signal_timestamp_utc == datetime(2020, 4, 13, tzinfo=UTC)
    assert desired[0].desired_lots == Decimal("1.5625")


def test_signal_and_proxy_volatility_are_causal_after_warmup() -> None:
    index = pd.date_range("2010-01-01", periods=700, freq="D", tz="UTC")
    original = pd.Series([100 + item * 0.1 for item in range(700)], index=index)
    changed = original.copy()
    changed.iloc[650:] = changed.iloc[650:] * 10
    first = causal_ewmac(original, 16, 64, 3.75)
    second = causal_ewmac(changed, 16, 64, 3.75)
    pd.testing.assert_series_equal(first.iloc[200:650], second.iloc[200:650])
    pd.testing.assert_series_equal(
        pinned_mixed_daily_price_volatility(original).iloc[200:650],
        pinned_mixed_daily_price_volatility(changed).iloc[200:650],
    )


@pytest.mark.parametrize(
    "values",
    [
        [-2.0, -1.0, 0.0, 1.0],
        [0.0, 1.0, 2.0, 3.0],
        [-4.0, -3.0, -2.0, -1.0],
    ],
)
def test_adjusted_price_validator_accepts_negative_and_zero_values(
    values: list[float],
) -> None:
    series = pd.Series(
        values,
        index=pd.date_range("2020-01-01", periods=len(values), freq="D"),
    )
    carver._price_series(series)


def test_adjusted_price_validator_still_rejects_unordered_and_duplicate_indexes() -> (
    None
):
    unordered = pd.Series(
        [1.0, -1.0],
        index=pd.DatetimeIndex(["2020-01-02", "2020-01-01"]),
    )
    duplicate = pd.Series(
        [0.0, -1.0],
        index=pd.DatetimeIndex(["2020-01-01", "2020-01-01"]),
    )
    with pytest.raises(CarverTrendCarryFtmo5EvaluationError, match="not causal"):
        carver._price_series(unordered)
    with pytest.raises(CarverTrendCarryFtmo5EvaluationError, match="not causal"):
        carver._price_series(duplicate)


def test_historical_nans_remain_causal_without_backfill_or_future_leakage() -> None:
    index = pd.date_range("2010-01-01", periods=400, freq="D", tz="UTC")
    adjusted = pd.Series(
        [float(item - 200) for item in range(400)], index=index, dtype=float
    )
    adjusted.iloc[:8] = float("nan")
    adjusted.iloc[75:78] = float("nan")
    multiple = pd.DataFrame(
        {
            "CARRY": adjusted + 2.0,
            "CARRY_CONTRACT": 202409,
            "PRICE": adjusted,
            "PRICE_CONTRACT": 202406,
            "FORWARD": adjusted + 1.0,
            "FORWARD_CONTRACT": 202412,
        },
        index=index,
    )
    original = combine_forecasts(adjusted, multiple)

    changed_adjusted = adjusted.copy()
    changed_multiple = multiple.copy()
    changed_adjusted.iloc[300:] = changed_adjusted.iloc[300:] * 100.0
    for column in ("PRICE", "CARRY"):
        changed_multiple.loc[index[300] :, column] = (
            changed_multiple.loc[index[300] :, column] * 100.0
        )
    changed = combine_forecasts(changed_adjusted, changed_multiple)
    pd.testing.assert_series_equal(
        original.combined.loc[: index[299]], changed.combined.loc[: index[299]]
    )

    volatility = pinned_mixed_daily_price_volatility(adjusted)
    short = adjusted.diff().ewm(adjust=True, span=35, min_periods=10).std()
    slow = short.ewm(span=20 * 256, adjust=True).mean()
    expected = (
        (slow * 0.35 + short * 0.65)
        .where((slow * 0.35 + short * 0.65) >= 1e-10, 1e-10)
        .ffill()
    )
    pd.testing.assert_series_equal(volatility, expected)
    assert (volatility.iloc[:18] == 1e-10).all()


def test_complete_synthetic_fold_uses_strict_later_bid_ask_g08_and_sparse_rows() -> (
    None
):
    start = datetime(2020, 4, 11, tzinfo=UTC)
    end = datetime(2020, 4, 21, tzinfo=UTC)
    observations = _synthetic_observations(start, end)
    signal = datetime(2020, 4, 13, tzinfo=UTC)
    desired = tuple(
        DesiredPosition(
            instrument,
            signal,
            Decimal("10"),
            Decimal("1"),
            Decimal("-1") if instrument == "SPX500_USD.OANDA" else Decimal("1"),
        )
        for instrument in carver._MAPPING
    )
    result = evaluate_synthetic_fold(
        SyntheticFoldInput("dev_fold_1", desired, observations, start, end)
    )
    transitions = result["transitions"]
    assert len(transitions) == 5
    assert all(
        datetime.fromisoformat(item["execution_timestamp_utc"].replace("Z", "+00:00"))
        > signal
        for item in transitions
    )
    assert all(
        item["fill_price"]
        == (item["bid"] if item["instrument"] == "SPX500_USD.OANDA" else item["ask"])
        for item in transitions
    )
    soybean = next(
        item for item in transitions if item["instrument"] == "SOYBN_USD.OANDA"
    )
    assert Decimal(soybean["bid"]) > Decimal("900")
    commissions = {
        item["instrument"]: Decimal(item["commission"]) for item in transitions
    }
    assert commissions["EUR/USD.DUKASCOPY"] > 0
    assert commissions["XAU_USD.OANDA"] > 0
    assert commissions["SPX500_USD.OANDA"] == 0
    assert commissions["WTICO_USD.OANDA"] == 0
    assert commissions["SOYBN_USD.OANDA"] == 0
    rows = result["daily_rows"]
    assert len(rows) == 10
    assert all(row["cost_stress_1_5x_return"] <= row["net_return"] for row in rows)


def test_sparse_observation_is_not_filled_and_margin_breach_fails_closed() -> None:
    start = datetime(2020, 4, 11, tzinfo=UTC)
    end = datetime(2020, 4, 16, tzinfo=UTC)
    observations = _synthetic_observations(start, end)
    signal = datetime(2020, 4, 13, 14, 1, tzinfo=UTC)
    desired = DesiredPosition(
        "XAU_USD.OANDA", signal, Decimal("10"), Decimal("1"), Decimal("1")
    )
    result = evaluate_synthetic_fold(
        SyntheticFoldInput("dev_fold_1", (desired,), observations, start, end)
    )
    transition = result["transitions"][0]
    assert transition["execution_timestamp_utc"] == "2020-04-14T14:01:00Z"
    too_large = DesiredPosition(
        "SOYBN_USD.OANDA",
        datetime(2020, 4, 13, tzinfo=UTC),
        Decimal("10"),
        Decimal("1"),
        Decimal("1000"),
    )
    with pytest.raises(CarverTrendCarryFtmo5EvaluationError, match="swing margin"):
        evaluate_synthetic_fold(
            SyntheticFoldInput("dev_fold_1", (too_large,), observations, start, end)
        )


def test_fold_scores_only_comparison_after_independent_warmup() -> None:
    compare_start = datetime(2020, 4, 11, tzinfo=UTC)
    compare_end = datetime(2020, 4, 16, tzinfo=UTC)
    observations = _synthetic_observations(
        datetime(2020, 4, 8, tzinfo=UTC), compare_end
    )
    warmup = DesiredPosition(
        "EUR/USD.DUKASCOPY",
        datetime(2020, 4, 9, tzinfo=UTC),
        Decimal("10"),
        Decimal("1"),
        Decimal("1"),
    )
    result = evaluate_synthetic_fold(
        SyntheticFoldInput(
            "dev_fold_1", (warmup,), observations, compare_start, compare_end
        )
    )
    assert result["transitions"][0]["execution_timestamp_utc"] < _utc_text(
        compare_start
    )
    assert len(result["daily_rows"]) == 5
    assert result["daily_rows"][0]["session_date"] == "2020-04-11"


def test_bootstrap_gate_and_result_artifact_are_deterministic(tmp_path: Path) -> None:
    spec = load_carver_trend_carry_ftmo5_spec()
    fold_results = []
    for index, fold_id in enumerate(("dev_fold_1", "dev_fold_2", "dev_fold_3")):
        values = [0.001 + index * 0.0001] * 20
        rows = [
            {
                "fold_id": fold_id,
                "session_date": f"202{index}-01-{day + 1:02d}",
                "mark_boundary_utc": f"202{index}-01-{day + 2:02d}T00:00:00Z",
                "net_return": value,
                "realized_spread_and_commission_cost_return": 0.0001,
                "cost_stress_1_5x_return": value - 0.00005,
                "per_instrument_net_return": {
                    instrument: value / 5 for instrument in carver._MAPPING
                },
            }
            for day, value in enumerate(values)
        ]
        fold_results.append(
            {
                "fold_id": fold_id,
                "daily_rows": rows,
                "transitions": [],
                "summary": {
                    "fold_id": fold_id,
                    "mean_daily_net_return": sum(values) / len(values),
                    "hard_failures": [],
                },
            }
        )
    first = summarize_development(fold_results, spec=spec)
    second = summarize_development(fold_results, spec=spec)
    assert first == second
    assert first["outcome"] == "PASS_DEVELOPMENT"
    economics = load_g08_cfd_economics()
    carver._write_result_artifacts(
        output_dir=tmp_path / "one",
        spec=spec,
        economics=economics,
        inputs={"synthetic": True},
        fold_results=fold_results,
        summary=first,
        run_timestamp_utc=datetime(2026, 8, 16, tzinfo=UTC),
    )
    carver._write_result_artifacts(
        output_dir=tmp_path / "two",
        spec=spec,
        economics=economics,
        inputs={"synthetic": True},
        fold_results=fold_results,
        summary=second,
        run_timestamp_utc=datetime(2026, 8, 17, tzinfo=UTC),
    )
    assert (tmp_path / "one/result.json").read_bytes() == (
        tmp_path / "two/result.json"
    ).read_bytes()
    assert (tmp_path / "one/run_provenance.json").read_bytes() != (
        tmp_path / "two/run_provenance.json"
    ).read_bytes()


def test_cli_has_no_validation_or_holdout_argument() -> None:
    parser_source = Path(carver.__file__).read_text(encoding="utf-8")
    assert 'parser.add_argument("--validation' not in parser_source
    assert 'parser.add_argument("--holdout' not in parser_source


def _synthetic_observations(
    start: datetime, end: datetime
) -> dict[str, tuple[ExecutionObservation, ...]]:
    bases = {
        "EUR/USD.DUKASCOPY": Decimal("1.10"),
        "XAU_USD.OANDA": Decimal("1700"),
        "SPX500_USD.OANDA": Decimal("3000"),
        "WTICO_USD.OANDA": Decimal("30"),
        "SOYBN_USD.OANDA": Decimal("10"),
    }
    spreads = {
        "EUR/USD.DUKASCOPY": Decimal("0.0002"),
        "XAU_USD.OANDA": Decimal("0.2"),
        "SPX500_USD.OANDA": Decimal("1"),
        "WTICO_USD.OANDA": Decimal("0.02"),
        "SOYBN_USD.OANDA": Decimal("0.01"),
    }
    result: dict[str, tuple[ExecutionObservation, ...]] = {}
    for instrument, base in bases.items():
        rows = []
        current = start
        count = 0
        while current < end:
            if current.weekday() < 5:
                candle = current + timedelta(hours=14)
                bid = base + Decimal(count) / Decimal("10")
                rows.append(
                    ExecutionObservation(
                        candle,
                        candle + timedelta(minutes=1),
                        bid,
                        bid + spreads[instrument],
                    )
                )
                count += 1
            current += timedelta(days=1)
        result[instrument] = tuple(rows)
    return result


def _utc_text(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")
