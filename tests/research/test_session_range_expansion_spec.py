from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ftmoquant.research.session_range_expansion_spec import (
    SESSION_RANGE_EXPANSION_CONFIG_SHA256,
    SESSION_RANGE_EXPANSION_SPEC_PATH,
    SessionRangeExpansionSpecValidationError,
    load_session_range_expansion_spec,
    session_range_expansion_config_sha256,
)


def test_frozen_session_range_spec_hash_is_stable() -> None:
    spec = load_session_range_expansion_spec()

    assert session_range_expansion_config_sha256(spec) == (
        SESSION_RANGE_EXPANSION_CONFIG_SHA256
    )
    assert spec.ordered_instruments == ("EUR/USD.DUKASCOPY", "GBP/USD.DUKASCOPY")
    assert spec.session_timezone == "Europe/London"


def test_spec_rejects_a_feature_or_window_change(tmp_path: Path) -> None:
    document = yaml.safe_load(SESSION_RANGE_EXPANSION_SPEC_PATH.read_text())
    document["data_semantics"]["breakout_window"] = "08:00 <= London time < 13:00"
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(SessionRangeExpansionSpecValidationError, match="frozen"):
        load_session_range_expansion_spec(path)
