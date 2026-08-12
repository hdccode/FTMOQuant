from datetime import date, time
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml

from ftmoquant.prop_rules import (
    AccountType,
    ConfigValidationError,
    EvaluationPhase,
    EvaluationProgram,
    MaximumLossType,
    Provider,
    load_prop_rule_set,
)

CONFIG_PATH = Path("config/prop/ftmo_2step_swing_2026-08.yaml")


def test_loads_ftmo_two_step_swing_rules() -> None:
    rules = load_prop_rule_set(CONFIG_PATH)

    assert rules.provider is Provider.FTMO
    assert rules.program is EvaluationProgram.TWO_STEP
    assert rules.account_type is AccountType.SWING
    assert rules.verified_on == date(2026, 8, 12)
    assert rules.loss_limits.maximum_daily_loss == Decimal("0.05")
    assert rules.loss_limits.maximum_loss == Decimal("0.10")
    assert rules.loss_limits.maximum_loss_type is MaximumLossType.STATIC
    assert rules.daily_reset.time == time(0, 0)
    assert rules.daily_reset.timezone.key == "Europe/Prague"
    assert rules.source_urls

    phases = {phase.phase: phase for phase in rules.phases}
    assert phases[EvaluationPhase.CHALLENGE].profit_target == Decimal("0.10")
    assert phases[EvaluationPhase.VERIFICATION].profit_target == Decimal("0.05")
    assert all(phase.minimum_trading_days == 4 for phase in rules.phases)
    assert all(phase.evaluation_period_days is None for phase in rules.phases)


@pytest.mark.parametrize(
    ("path", "value", "error"),
    [
        (("phases", "challenge", "profit_target"), "1.01", "no greater than 1"),
        (("phases", "verification", "minimum_trading_days"), 0, "positive integer"),
        (("daily_reset", "timezone"), "Not/A_Timezone", "valid IANA timezone"),
        (("daily_reset", "time"), "midnight", "HH:MM"),
        (("verified_on",), "2026-02-30", "valid YYYY-MM-DD date"),
        (("source_urls",), ["https://example.com/rules"], "official HTTPS FTMO URL"),
    ],
)
def test_rejects_invalid_fields(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
    error: str,
) -> None:
    config = _read_config()
    _set_nested(config, path, value)
    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ConfigValidationError, match=error):
        load_prop_rule_set(invalid_path)


def test_requires_both_evaluation_phases(tmp_path: Path) -> None:
    config = _read_config()
    del config["phases"]["verification"]
    invalid_path = tmp_path / "missing-phase.yaml"
    invalid_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ConfigValidationError, match="missing fields: verification"):
        load_prop_rule_set(invalid_path)


def _read_config() -> dict[str, Any]:
    loaded = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _set_nested(config: dict[str, Any], path: tuple[str, ...], value: object) -> None:
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
