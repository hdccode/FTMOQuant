from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

from ftmoquant.research.usd_macro_surprise_momentum_spec import (
    USD_MACRO_SURPRISE_MOMENTUM_CONFIG_SHA256,
    USD_MACRO_SURPRISE_MOMENTUM_SPEC_PATH,
    UsdMacroSurpriseMomentumSpecValidationError,
    load_usd_macro_surprise_momentum_spec,
)


def test_frozen_macro_spec_and_semantic_sha() -> None:
    spec = load_usd_macro_surprise_momentum_spec()
    assert spec.candidate_id == "usd_macro_surprise_momentum_v1"
    assert spec.semantic_sha256 == USD_MACRO_SURPRISE_MOMENTUM_CONFIG_SHA256


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("signal", "surprise_threshold", "one_standard_deviation"),
        ("timing", "immediate_reaction_exclusion_minutes", 4),
        ("research_boundary", "validation", "unlocked"),
    ],
)
def test_frozen_macro_spec_rejects_semantic_changes(
    tmp_path: Path, section: str, key: str, value: object
) -> None:
    document = _document()
    document[section][key] = value
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(UsdMacroSurpriseMomentumSpecValidationError):
        load_usd_macro_surprise_momentum_spec(path)


def _document() -> dict[str, Any]:
    value = yaml.safe_load(USD_MACRO_SURPRISE_MOMENTUM_SPEC_PATH.read_text())
    assert isinstance(value, dict)
    return deepcopy(value)
