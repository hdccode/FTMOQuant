from __future__ import annotations

import json

import optuna
import pytest

from ftmoquant.research.g1.family import FamilyContractError, FamilyRegistry
from ftmoquant.research.g1.parameter_space import (
    CategoricalParameter,
    FloatParameter,
    IntParameter,
    ParameterSpace,
    ParameterSpaceError,
    canonical_parameter_json,
    deterministic_trial_id,
    enumerate_grid,
    suggest_parameters,
)
from tests.research.g1.helpers import IntegerFamily


def test_finite_grid_enumeration_is_exact_ordered_and_deterministic() -> None:
    space = ParameterSpace(
        (
            IntParameter("lookback", 2, 4, step=2),
            CategoricalParameter("session", ("all", "london")),
            FloatParameter("threshold", 0.5, 1.0, step=0.5),
        )
    )

    first = tuple(enumerate_grid(space))
    second = tuple(enumerate_grid(space))

    assert first == second
    assert len(first) == 8
    assert first[0] == {"lookback": 2, "session": "all", "threshold": 0.5}
    assert len({canonical_parameter_json(item) for item in first}) == len(first)
    assert tuple(enumerate_grid(ParameterSpace(()))) == ({},)


def test_adapted_optuna_dispatch_preserves_typed_declaration_behavior() -> None:
    space = ParameterSpace(
        (
            CategoricalParameter("session", ("all", "london")),
            IntParameter("lookback", 10, 30, step=10),
            FloatParameter("threshold", 0.5, 1.5),
        )
    )
    trial = optuna.trial.FixedTrial(
        {"session": "london", "lookback": 20, "threshold": 0.75}
    )

    assert suggest_parameters(trial, space) == {
        "session": "london",
        "lookback": 20,
        "threshold": 0.75,
    }


def test_canonical_serialization_and_trial_hash_are_order_independent() -> None:
    left = {"period": 20, "session": "london", "enabled": True}
    right = {"enabled": True, "session": "london", "period": 20}

    assert canonical_parameter_json(left) == canonical_parameter_json(right)
    assert json.loads(canonical_parameter_json(left)) == left
    assert deterministic_trial_id("trend", "1.0.0", left) == (
        deterministic_trial_id("trend", "1.0.0", right)
    )
    assert len(deterministic_trial_id("trend", "1.0.0", left)) == 64
    with pytest.raises(ParameterSpaceError, match="finite"):
        canonical_parameter_json({"bad": float("nan")})


def test_family_contract_and_registry_are_deterministic_and_duplicate_safe() -> None:
    family = IntegerFamily()
    registry = FamilyRegistry()
    registry.register(family)

    assert family.contract_document() == family.contract_document()
    assert registry.definitions() == registry.definitions()
    assert registry.get("synthetic_integer", "1.0.0") is family
    assert family.build_signals((1.0, 2.0), {"lookback": 2}) == (2.0, 4.0)
    with pytest.raises(FamilyContractError, match="duplicate"):
        registry.register(family)
