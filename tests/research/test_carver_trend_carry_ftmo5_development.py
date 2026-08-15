from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pytest

from ftmoquant.research.carver_trend_carry_ftmo5_development import (
    CarverTrendCarryFtmo5EvaluationError,
    causal_carry,
    causal_ewmac,
    combine_forecasts,
    comparison_fold,
    first_strictly_later_execution,
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
        == "f2a1eacf7d3adb18938bc7013d9873906b2fc6d5e7f3bf6a68cfd3754e9daa40"
    )
