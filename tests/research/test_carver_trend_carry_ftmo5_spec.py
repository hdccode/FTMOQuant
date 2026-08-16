from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

from ftmoquant.research.carver_trend_carry_ftmo5_spec import (
    CARVER_TREND_CARRY_FTMO5_CONFIG_SHA256,
    CARVER_TREND_CARRY_FTMO5_SPEC_PATH,
    CarverTrendCarryFtmo5SpecValidationError,
    load_carver_trend_carry_ftmo5_spec,
)


def test_frozen_carver_spec_semantic_sha() -> None:
    spec = load_carver_trend_carry_ftmo5_spec()
    assert spec.candidate_id == "carver_trend_carry_ftmo5_v1"
    assert spec.semantic_sha256 == CARVER_TREND_CARRY_FTMO5_CONFIG_SHA256
    assert spec.canonical_document["universe"]["futures_to_execution"] == {
        "EUR": "EUR/USD.DUKASCOPY",
        "GOLD": "XAU_USD.OANDA",
        "SP500": "SPX500_USD.OANDA",
        "CRUDE_W": "WTICO_USD.OANDA",
        "SOYBEAN": "SOYBN_USD.OANDA",
    }
    assert spec.canonical_document["universe"]["execution_economics_role"] == (
        "frozen_FTMO_G0_8_only"
    )
    assert spec.canonical_document["universe"][
        "execution_price_proxy_provider_metadata"
    ] == {
        "OANDA_v20_practice_confirmed_instruments": [
            "XAU_USD",
            "SPX500_USD",
            "WTICO_USD",
            "SOYBN_USD",
        ]
    }
    assert spec.canonical_document["universe"]["execution_price_scale_to_ftmo"] == {
        "SOYBN_USD.OANDA": 100
    }


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("combination", "forecast_diversification_multiplier", 1.30),
        ("carry", "smoothing_ewm_span_days", 91),
        ("research_boundary", "validation", "unlocked"),
    ],
)
def test_carver_spec_rejects_frozen_semantic_changes(
    tmp_path: Path, section: str, key: str, value: object
) -> None:
    document = _document()
    document[section][key] = value
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(CarverTrendCarryFtmo5SpecValidationError):
        load_carver_trend_carry_ftmo5_spec(path)


def _document() -> dict[str, Any]:
    value = yaml.safe_load(CARVER_TREND_CARRY_FTMO5_SPEC_PATH.read_text())
    assert isinstance(value, dict)
    return deepcopy(value)
