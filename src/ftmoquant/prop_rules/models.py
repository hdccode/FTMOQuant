"""Typed domain models for prop-firm rule configurations."""

from dataclasses import dataclass
from datetime import date, time
from decimal import Decimal
from enum import StrEnum
from zoneinfo import ZoneInfo


class Provider(StrEnum):
    """Supported prop-firm providers."""

    FTMO = "ftmo"


class EvaluationProgram(StrEnum):
    """Supported evaluation programs."""

    TWO_STEP = "two_step"


class AccountType(StrEnum):
    """Supported prop account types."""

    SWING = "swing"


class EvaluationPhase(StrEnum):
    """Phases in an evaluation program."""

    CHALLENGE = "challenge"
    VERIFICATION = "verification"


class MaximumLossType(StrEnum):
    """Ways a maximum loss limit may be anchored."""

    STATIC = "static"


@dataclass(frozen=True, slots=True)
class PhaseRules:
    """Rules that vary by evaluation phase."""

    phase: EvaluationPhase
    profit_target: Decimal
    minimum_trading_days: int
    evaluation_period_days: int | None


@dataclass(frozen=True, slots=True)
class LossLimits:
    """Loss limits shared by the evaluation phases."""

    maximum_daily_loss: Decimal
    maximum_loss: Decimal
    maximum_loss_type: MaximumLossType


@dataclass(frozen=True, slots=True)
class DailyReset:
    """Local time at which daily rules reset."""

    time: time
    timezone: ZoneInfo


@dataclass(frozen=True, slots=True)
class PropRuleSet:
    """A versioned, source-backed prop-firm rule set."""

    schema_version: int
    rule_set_id: str
    version: str
    provider: Provider
    program: EvaluationProgram
    account_type: AccountType
    verified_on: date
    phases: tuple[PhaseRules, ...]
    loss_limits: LossLimits
    daily_reset: DailyReset
    source_urls: tuple[str, ...]
