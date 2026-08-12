"""Load and validate prop-firm rule configurations from YAML."""

import re
from collections.abc import Mapping
from datetime import date, time
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from typing import TypeVar
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

from ftmoquant.prop_rules.models import (
    AccountType,
    DailyReset,
    EvaluationPhase,
    EvaluationProgram,
    LossLimits,
    MaximumLossType,
    PhaseRules,
    PropRuleSet,
    Provider,
)

EnumType = TypeVar("EnumType", bound=Enum)
_TIME_PATTERN = re.compile(r"\d{2}:\d{2}")


class ConfigValidationError(ValueError):
    """Raised when a prop-rule configuration is invalid."""


def load_prop_rule_set(path: str | Path) -> PropRuleSet:
    """Load a prop-rule YAML file and validate its complete schema."""

    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigValidationError(f"could not load {config_path}: {error}") from error

    root = _mapping(raw, "configuration")
    _keys(
        root,
        {
            "schema_version",
            "rule_set_id",
            "version",
            "provider",
            "program",
            "account_type",
            "verified_on",
            "phases",
            "loss_limits",
            "daily_reset",
            "source_urls",
        },
        "configuration",
    )

    schema_version = _positive_integer(root["schema_version"], "schema_version")
    rule_set_id = _non_empty_string(root["rule_set_id"], "rule_set_id")
    version = _non_empty_string(root["version"], "version")

    phases = _load_phases(root["phases"])
    loss_limits = _load_loss_limits(root["loss_limits"])
    daily_reset = _load_daily_reset(root["daily_reset"])

    return PropRuleSet(
        schema_version=schema_version,
        rule_set_id=rule_set_id,
        version=version,
        provider=_enum(Provider, root["provider"], "provider"),
        program=_enum(EvaluationProgram, root["program"], "program"),
        account_type=_enum(AccountType, root["account_type"], "account_type"),
        verified_on=_date(root["verified_on"], "verified_on"),
        phases=phases,
        loss_limits=loss_limits,
        daily_reset=daily_reset,
        source_urls=_source_urls(root["source_urls"]),
    )


def _load_phases(value: object) -> tuple[PhaseRules, ...]:
    phases = _mapping(value, "phases")
    required = {phase.value for phase in EvaluationPhase}
    _keys(phases, required, "phases")

    loaded: list[PhaseRules] = []
    for phase in EvaluationPhase:
        name = f"phases.{phase.value}"
        values = _mapping(phases[phase.value], name)
        _keys(
            values,
            {"profit_target", "minimum_trading_days", "evaluation_period_days"},
            name,
        )
        loaded.append(
            PhaseRules(
                phase=phase,
                profit_target=_percentage(
                    values["profit_target"], f"{name}.profit_target"
                ),
                minimum_trading_days=_positive_integer(
                    values["minimum_trading_days"],
                    f"{name}.minimum_trading_days",
                ),
                evaluation_period_days=_optional_positive_integer(
                    values["evaluation_period_days"],
                    f"{name}.evaluation_period_days",
                ),
            )
        )
    return tuple(loaded)


def _load_loss_limits(value: object) -> LossLimits:
    limits = _mapping(value, "loss_limits")
    _keys(
        limits,
        {"maximum_daily_loss", "maximum_loss", "maximum_loss_type"},
        "loss_limits",
    )
    return LossLimits(
        maximum_daily_loss=_percentage(
            limits["maximum_daily_loss"], "loss_limits.maximum_daily_loss"
        ),
        maximum_loss=_percentage(limits["maximum_loss"], "loss_limits.maximum_loss"),
        maximum_loss_type=_enum(
            MaximumLossType,
            limits["maximum_loss_type"],
            "loss_limits.maximum_loss_type",
        ),
    )


def _load_daily_reset(value: object) -> DailyReset:
    reset = _mapping(value, "daily_reset")
    _keys(reset, {"time", "timezone"}, "daily_reset")
    return DailyReset(
        time=_time(reset["time"], "daily_reset.time"),
        timezone=_timezone(reset["timezone"], "daily_reset.timezone"),
    )


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ConfigValidationError(f"{field} must be a mapping with string keys")
    return value


def _keys(values: Mapping[str, object], expected: set[str], field: str) -> None:
    actual = set(values)
    missing = expected - actual
    unexpected = actual - expected
    if missing:
        fields = ", ".join(sorted(missing))
        raise ConfigValidationError(f"{field} is missing fields: {fields}")
    if unexpected:
        raise ConfigValidationError(
            f"{field} has unexpected fields: {', '.join(sorted(unexpected))}"
        )


def _non_empty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigValidationError(f"{field} must be a non-empty string")
    return value


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigValidationError(f"{field} must be a positive integer")
    return value


def _optional_positive_integer(value: object, field: str) -> int | None:
    if value is None:
        return None
    return _positive_integer(value, field)


def _percentage(value: object, field: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ConfigValidationError(f"{field} must be a decimal percentage")
    try:
        percentage = Decimal(str(value))
    except InvalidOperation as error:
        raise ConfigValidationError(f"{field} must be a decimal percentage") from error
    if not percentage.is_finite() or not Decimal("0") < percentage <= Decimal("1"):
        raise ConfigValidationError(
            f"{field} must be greater than 0 and no greater than 1"
        )
    return percentage


def _enum(enum_type: type[EnumType], value: object, field: str) -> EnumType:
    if not isinstance(value, str):
        raise ConfigValidationError(f"{field} must be a string")
    try:
        return enum_type(value)
    except ValueError as error:
        allowed = ", ".join(str(member.value) for member in enum_type)
        raise ConfigValidationError(f"{field} must be one of: {allowed}") from error


def _date(value: object, field: str) -> date:
    if not isinstance(value, str):
        raise ConfigValidationError(f"{field} must use YYYY-MM-DD format")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ConfigValidationError(
            f"{field} must be a valid YYYY-MM-DD date"
        ) from error
    if parsed.isoformat() != value:
        raise ConfigValidationError(f"{field} must use YYYY-MM-DD format")
    return parsed


def _time(value: object, field: str) -> time:
    if not isinstance(value, str) or _TIME_PATTERN.fullmatch(value) is None:
        raise ConfigValidationError(f"{field} must use HH:MM format")
    try:
        return time.fromisoformat(value)
    except ValueError as error:
        raise ConfigValidationError(f"{field} must be a valid time") from error


def _timezone(value: object, field: str) -> ZoneInfo:
    name = _non_empty_string(value, field)
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as error:
        raise ConfigValidationError(f"{field} must be a valid IANA timezone") from error


def _source_urls(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigValidationError("source_urls must be a non-empty list")

    urls: list[str] = []
    for index, item in enumerate(value):
        field = f"source_urls[{index}]"
        url = _non_empty_string(item, field)
        parsed = urlsplit(url)
        hostname = parsed.hostname
        if (
            parsed.scheme != "https"
            or hostname is None
            or (hostname != "ftmo.com" and not hostname.endswith(".ftmo.com"))
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ConfigValidationError(f"{field} must be an official HTTPS FTMO URL")
        urls.append(url)
    return tuple(urls)
