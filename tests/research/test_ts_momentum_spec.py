from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

from ftmoquant.research.ts_momentum_spec import (
    TS_MOMENTUM_CONFIG_SHA256,
    TS_MOMENTUM_SPEC_PATH,
    TsMomentumSpecValidationError,
    load_ts_momentum_spec,
    ts_momentum_config_sha256,
)


def test_frozen_ts_momentum_spec_and_hash() -> None:
    spec = load_ts_momentum_spec()

    assert spec.strategy_id == "ts_momentum_v1"
    assert spec.status == "implemented_not_evaluated"
    assert spec.lookback_prior_eligible_observations == 252
    assert spec.native_period == 253
    assert ts_momentum_config_sha256(spec) == TS_MOMENTUM_CONFIG_SHA256
    assert TS_MOMENTUM_CONFIG_SHA256 == (
        "edcbe2e4afe631e5fde1223558122ecf4d796abd0610729313ebbb32a468ccd5"
    )


def test_config_hash_ignores_yaml_key_order(tmp_path: Path) -> None:
    path = tmp_path / "reordered.yaml"
    path.write_text(yaml.safe_dump(_document(), sort_keys=True), encoding="utf-8")

    reordered = load_ts_momentum_spec(path)

    assert ts_momentum_config_sha256(reordered) == TS_MOMENTUM_CONFIG_SHA256


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("signal", "lookback_prior_eligible_observations", 251),
        ("signal", "native_period", 252),
        ("signal", "positive_target", 2),
        ("data_semantics", "daily_session_close", "00:00"),
        ("execution", "strategy_sizing", True),
        ("research_boundary", "validation", "available"),
        ("parameter_family", "permitted_variants", ["lookback_126"]),
    ],
)
def test_spec_rejects_any_frozen_parameter_change(
    tmp_path: Path, section: str, key: str, value: object
) -> None:
    document = _document()
    document[section][key] = value
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(TsMomentumSpecValidationError, match="frozen Phase 1"):
        load_ts_momentum_spec(path)


def test_spec_rejects_feature_soup_field(tmp_path: Path) -> None:
    document = _document()
    document["signal"]["volatility_filter"] = True
    path = tmp_path / "feature-soup.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(TsMomentumSpecValidationError, match="fields are not exact"):
        load_ts_momentum_spec(path)


def _document() -> dict[str, Any]:
    value = yaml.safe_load(TS_MOMENTUM_SPEC_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return deepcopy(value)
