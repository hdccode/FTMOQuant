"""Small practical contract separating strategy families from the G1 engine."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ftmoquant.research.g1.parameter_space import (
    ParameterMapping,
    ParameterSpace,
    ParameterValue,
)
from ftmoquant.research.g1.sessions import SessionId

_ID = re.compile(r"[a-z][a-z0-9_]*")
_VERSION = re.compile(r"[1-9]\d*\.\d+\.\d+")


class FamilyContractError(ValueError):
    """Raised when a family definition is incomplete or ambiguous."""


@dataclass(frozen=True, slots=True)
class FamilyMetadata:
    family_id: str
    version: str
    economic_hypothesis: str
    supported_timeframes: tuple[str, ...]
    eligible_sessions: tuple[SessionId, ...] = (SessionId.ALL,)

    def __post_init__(self) -> None:
        if _ID.fullmatch(self.family_id) is None:
            raise FamilyContractError("family_id must be lower snake_case")
        if _VERSION.fullmatch(self.version) is None:
            raise FamilyContractError("family version must be semantic x.y.z")
        if not self.economic_hypothesis.strip():
            raise FamilyContractError("economic hypothesis must not be empty")
        if not self.supported_timeframes or any(
            not timeframe.strip() for timeframe in self.supported_timeframes
        ):
            raise FamilyContractError("supported timeframes must not be empty")
        if len(set(self.supported_timeframes)) != len(self.supported_timeframes):
            raise FamilyContractError("supported timeframes must be unique")
        if not self.eligible_sessions or len(set(self.eligible_sessions)) != len(
            self.eligible_sessions
        ):
            raise FamilyContractError("eligible sessions must be non-empty and unique")


class StrategyFamily[DataT, SignalT](ABC):
    """Only family-owned signal state, bounded parameters, and constraints."""

    @property
    @abstractmethod
    def metadata(self) -> FamilyMetadata: ...

    @property
    @abstractmethod
    def parameter_space(self) -> ParameterSpace: ...

    def validate_parameters(self, parameters: ParameterMapping) -> None:
        """Enforce generic bounds; subclasses may add economic constraints."""

        self.parameter_space.validate(parameters)

    @abstractmethod
    def build_signals(self, data: DataT, parameters: ParameterMapping) -> SignalT:
        """Build causal family state/signals; execution remains outside the family."""

    def neighbours(
        self, parameters: ParameterMapping
    ) -> tuple[Mapping[str, ParameterValue], ...]:
        """Return family-defined adjacent configurations, if the family has them."""

        return ()

    def contract_document(self) -> dict[str, object]:
        return {
            "family_id": self.metadata.family_id,
            "version": self.metadata.version,
            "economic_hypothesis": self.metadata.economic_hypothesis,
            "supported_timeframes": self.metadata.supported_timeframes,
            "eligible_sessions": tuple(
                session.value for session in self.metadata.eligible_sessions
            ),
            "parameter_space": self.parameter_space.as_dict(),
        }


class FamilyRegistry:
    """Deterministic strategy-function registry inspired by Forex Strategy Lab."""

    def __init__(self) -> None:
        self._families: dict[tuple[str, str], StrategyFamily[Any, Any]] = {}

    def register(self, family: StrategyFamily[Any, Any]) -> None:
        key = (family.metadata.family_id, family.metadata.version)
        if key in self._families:
            raise FamilyContractError(f"duplicate family registration: {key}")
        self._families[key] = family

    def get(self, family_id: str, version: str) -> StrategyFamily[Any, Any]:
        try:
            return self._families[(family_id, version)]
        except KeyError as error:
            raise FamilyContractError(
                f"unregistered family: {(family_id, version)}"
            ) from error

    def definitions(self) -> tuple[dict[str, object], ...]:
        return tuple(
            self._families[key].contract_document() for key in sorted(self._families)
        )
