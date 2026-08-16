"""Bounded parameter spaces and canonical configuration identities.

``suggest_parameters`` is adapted from Mohammed Abumtary's MIT-licensed
Forex Strategy Lab ``smart_backtester.py::suggest_params`` at commit
df34085afb615818038c0e8bedc5aad837883e5c (lines 618-628).  FTMOQuant replaces
the upstream tuple schema with typed, validated declarations and adds exact
grid enumeration, canonical JSON, duplicate-safe hashes, and stepped floats.

Copyright (c) 2026 Mohammed Abumtary.  See
``LICENSES/forex-strategy-lab-MIT.txt`` and ``docs/oss_adoption.md``.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

import optuna

type ParameterValue = str | int | float | bool | None
type ParameterMapping = Mapping[str, ParameterValue]


class ParameterSpaceError(ValueError):
    """Raised when a bounded search space or configuration is ambiguous."""


class Parameter(Protocol):
    @property
    def name(self) -> str: ...

    def contains(self, value: ParameterValue) -> bool: ...

    def grid_values(self) -> tuple[ParameterValue, ...] | None: ...

    def suggest(self, trial: optuna.Trial) -> ParameterValue: ...

    def as_dict(self) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class CategoricalParameter:
    name: str
    choices: tuple[ParameterValue, ...]

    def __post_init__(self) -> None:
        _validate_name(self.name)
        if not self.choices:
            raise ParameterSpaceError(f"{self.name} choices must not be empty")
        canonical = tuple(_canonical_scalar(value) for value in self.choices)
        if len(set(canonical)) != len(canonical):
            raise ParameterSpaceError(f"{self.name} choices must be unique")

    def contains(self, value: ParameterValue) -> bool:
        return _canonical_scalar(value) in {
            _canonical_scalar(choice) for choice in self.choices
        }

    def grid_values(self) -> tuple[ParameterValue, ...]:
        return self.choices

    def suggest(self, trial: optuna.Trial) -> ParameterValue:
        return trial.suggest_categorical(self.name, list(self.choices))

    def as_dict(self) -> dict[str, object]:
        return {"kind": "categorical", "name": self.name, "choices": self.choices}


@dataclass(frozen=True, slots=True)
class IntParameter:
    name: str
    low: int
    high: int
    step: int = 1
    log: bool = False

    def __post_init__(self) -> None:
        _validate_name(self.name)
        if any(isinstance(value, bool) for value in (self.low, self.high, self.step)):
            raise ParameterSpaceError(f"{self.name} integer bounds cannot be bool")
        if self.low > self.high or self.step <= 0:
            raise ParameterSpaceError(f"{self.name} integer bounds/step are invalid")
        if self.log and self.step != 1:
            raise ParameterSpaceError(f"{self.name} logarithmic integer step must be 1")
        if self.log and self.low <= 0:
            raise ParameterSpaceError(
                f"{self.name} logarithmic lower bound must be > 0"
            )

    def contains(self, value: ParameterValue) -> bool:
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
            and self.low <= value <= self.high
            and (value - self.low) % self.step == 0
        )

    def grid_values(self) -> tuple[ParameterValue, ...]:
        return tuple(range(self.low, self.high + 1, self.step))

    def suggest(self, trial: optuna.Trial) -> int:
        return trial.suggest_int(
            self.name, self.low, self.high, step=self.step, log=self.log
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "int",
            "name": self.name,
            "low": self.low,
            "high": self.high,
            "step": self.step,
            "log": self.log,
        }


@dataclass(frozen=True, slots=True)
class FloatParameter:
    name: str
    low: float
    high: float
    step: float | None = None
    log: bool = False

    def __post_init__(self) -> None:
        _validate_name(self.name)
        if not math.isfinite(self.low) or not math.isfinite(self.high):
            raise ParameterSpaceError(f"{self.name} float bounds must be finite")
        if self.low > self.high:
            raise ParameterSpaceError(f"{self.name} float bounds are reversed")
        if self.step is not None and (not math.isfinite(self.step) or self.step <= 0.0):
            raise ParameterSpaceError(f"{self.name} float step must be finite and > 0")
        if self.log and self.step is not None:
            raise ParameterSpaceError(
                f"{self.name} logarithmic float cannot use a step"
            )
        if self.log and self.low <= 0.0:
            raise ParameterSpaceError(
                f"{self.name} logarithmic lower bound must be > 0"
            )

    def contains(self, value: ParameterValue) -> bool:
        if not isinstance(value, float):
            return False
        numeric = float(value)
        if not math.isfinite(numeric) or not self.low <= numeric <= self.high:
            return False
        if self.step is None:
            return True
        offset = (Decimal(str(numeric)) - Decimal(str(self.low))) / Decimal(
            str(self.step)
        )
        return offset == offset.to_integral_value()

    def grid_values(self) -> tuple[ParameterValue, ...] | None:
        if self.step is None:
            return None
        low = Decimal(str(self.low))
        high = Decimal(str(self.high))
        step = Decimal(str(self.step))
        values: list[ParameterValue] = []
        current = low
        while current <= high:
            values.append(float(current))
            current += step
        return tuple(values)

    def suggest(self, trial: optuna.Trial) -> float:
        return trial.suggest_float(
            self.name, self.low, self.high, step=self.step, log=self.log
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "float",
            "name": self.name,
            "low": self.low,
            "high": self.high,
            "step": self.step,
            "log": self.log,
        }


@dataclass(frozen=True, slots=True)
class ParameterSpace:
    parameters: tuple[Parameter, ...]

    def __post_init__(self) -> None:
        names = tuple(parameter.name for parameter in self.parameters)
        if len(set(names)) != len(names):
            raise ParameterSpaceError("parameter names must be unique")

    def validate(self, parameters: ParameterMapping) -> None:
        expected = {parameter.name for parameter in self.parameters}
        if set(parameters) != expected:
            raise ParameterSpaceError(
                f"parameter keys must be exact; expected {sorted(expected)}"
            )
        for declaration in self.parameters:
            if not declaration.contains(parameters[declaration.name]):
                raise ParameterSpaceError(
                    f"{declaration.name} is outside its frozen bounded space"
                )

    @property
    def is_finite_grid(self) -> bool:
        return all(parameter.grid_values() is not None for parameter in self.parameters)

    def as_dict(self) -> dict[str, object]:
        return {"parameters": [parameter.as_dict() for parameter in self.parameters]}


def enumerate_grid(space: ParameterSpace) -> Iterator[dict[str, ParameterValue]]:
    """Enumerate a finite grid exactly once in declaration/value order."""

    values = tuple(parameter.grid_values() for parameter in space.parameters)
    if any(item is None for item in values):
        raise ParameterSpaceError("exact enumeration requires a fully finite grid")
    finite_values = tuple(item for item in values if item is not None)
    for combination in itertools.product(*finite_values):
        yield {
            parameter.name: value
            for parameter, value in zip(space.parameters, combination, strict=True)
        }


def suggest_parameters(
    trial: optuna.Trial, space: ParameterSpace
) -> dict[str, ParameterValue]:
    """Suggest one typed bounded configuration through Optuna.

    Adapted from Forex Strategy Lab's dispatch over categorical, integer, and
    floating declarations; FTMOQuant delegates validation to typed parameters.
    """

    parameters = {
        declaration.name: declaration.suggest(trial) for declaration in space.parameters
    }
    space.validate(parameters)
    return parameters


def canonical_parameter_json(parameters: ParameterMapping) -> str:
    """Return deterministic JSON without accepting non-finite or nested values."""

    normalized: dict[str, ParameterValue] = {}
    for name, value in parameters.items():
        _validate_name(name)
        _canonical_scalar(value)
        normalized[name] = 0.0 if isinstance(value, float) and value == 0.0 else value
    return json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def deterministic_trial_id(
    family_id: str, family_version: str, parameters: ParameterMapping
) -> str:
    payload = {
        "family_id": family_id,
        "family_version": family_version,
        "parameters": json.loads(canonical_parameter_json(parameters)),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _canonical_scalar(value: ParameterValue) -> tuple[str, str]:
    if value is None:
        return ("null", "")
    if isinstance(value, bool):
        return ("bool", str(value).lower())
    if isinstance(value, int):
        return ("int", str(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ParameterSpaceError("parameter floats must be finite")
        return ("float", repr(0.0 if value == 0.0 else value))
    if isinstance(value, str):
        return ("str", value)
    raise ParameterSpaceError(f"unsupported parameter type: {type(value).__name__}")


def _validate_name(name: str) -> None:
    if not isinstance(name, str) or not name or name.strip() != name:
        raise ParameterSpaceError("parameter names must be non-empty trimmed strings")
