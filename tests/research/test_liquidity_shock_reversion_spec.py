# ruff: noqa: E501

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

from ftmoquant.research.liquidity_shock_reversion_spec import (
    LIQUIDITY_SHOCK_REVERSION_CONFIG_SHA256,
    LIQUIDITY_SHOCK_REVERSION_SPEC_PATH,
    LiquidityShockReversionSpecValidationError,
    liquidity_shock_reversion_config_sha256,
    load_liquidity_shock_reversion_spec,
)


def test_frozen_spec_and_semantic_sha() -> None:
    spec = load_liquidity_shock_reversion_spec()
    assert (
        spec.baseline_prior_eligible_returns,
        spec.shock_multiple,
        spec.hold_eligible_completed_minutes,
    ) == (60, 5, 15)
    assert (
        liquidity_shock_reversion_config_sha256(spec)
        == LIQUIDITY_SHOCK_REVERSION_CONFIG_SHA256
    )
    assert (
        LIQUIDITY_SHOCK_REVERSION_CONFIG_SHA256
        == "55a2a1814a507ebf53f5713d9672cf5d4fda19790d9c808efc0a9773bed27acd"
    )


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("signal", "baseline_prior_eligible_returns", 59),
        ("signal", "shock_rule", "greater_or_equal"),
        ("execution", "hold_eligible_completed_minutes_after_entry", 14),
    ],
)
def test_spec_rejects_baseline_changes(
    tmp_path: Path, section: str, key: str, value: object
) -> None:
    document = _document()
    document[section][key] = value
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(
        LiquidityShockReversionSpecValidationError, match="frozen Phase 1"
    ):
        load_liquidity_shock_reversion_spec(path)


def _document() -> dict[str, Any]:
    value = yaml.safe_load(
        LIQUIDITY_SHOCK_REVERSION_SPEC_PATH.read_text(encoding="utf-8")
    )
    assert isinstance(value, dict)
    return deepcopy(value)
