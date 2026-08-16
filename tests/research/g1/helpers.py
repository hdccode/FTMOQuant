from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime

from ftmoquant.research.g1.family import FamilyMetadata, StrategyFamily
from ftmoquant.research.g1.guards import (
    DevelopmentAccessPolicy,
    DevelopmentSearchContext,
    WalkForwardWindow,
)
from ftmoquant.research.g1.parameter_space import (
    IntParameter,
    ParameterMapping,
    ParameterSpace,
    ParameterValue,
)


class IntegerFamily(StrategyFamily[tuple[float, ...], tuple[float, ...]]):
    def __init__(self, low: int = 1, high: int = 4, *, invalidate_two: bool = False):
        self._space = ParameterSpace((IntParameter("lookback", low, high),))
        self.invalidate_two = invalidate_two

    @property
    def metadata(self) -> FamilyMetadata:
        return FamilyMetadata(
            family_id="synthetic_integer",
            version="1.0.0",
            economic_hypothesis="Synthetic monotonic response for contract tests.",
            supported_timeframes=("synthetic",),
        )

    @property
    def parameter_space(self) -> ParameterSpace:
        return self._space

    def validate_parameters(self, parameters: ParameterMapping) -> None:
        super().validate_parameters(parameters)
        if self.invalidate_two and parameters["lookback"] == 2:
            raise ValueError("family-specific invalid lookback")

    def build_signals(
        self, data: tuple[float, ...], parameters: ParameterMapping
    ) -> tuple[float, ...]:
        return tuple(value * int(parameters["lookback"] or 0) for value in data)

    def neighbours(
        self, parameters: ParameterMapping
    ) -> tuple[Mapping[str, ParameterValue], ...]:
        current = int(parameters["lookback"] or 0)
        values = []
        for candidate in (current - 1, current + 1):
            if self._space.parameters[0].contains(candidate):
                values.append({"lookback": candidate})
        return tuple(values)


def context() -> DevelopmentSearchContext:
    policy = DevelopmentAccessPolicy(
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
    )
    return DevelopmentSearchContext(
        access_policy=policy,
        windows=(
            WalkForwardWindow(
                "fold_1",
                datetime(2020, 1, 1, tzinfo=UTC),
                datetime(2020, 3, 1, tzinfo=UTC),
                datetime(2020, 3, 1, tzinfo=UTC),
                datetime(2020, 4, 1, tzinfo=UTC),
            ),
            WalkForwardWindow(
                "fold_2",
                datetime(2020, 1, 1, tzinfo=UTC),
                datetime(2020, 6, 1, tzinfo=UTC),
                datetime(2020, 6, 1, tzinfo=UTC),
                datetime(2020, 7, 1, tzinfo=UTC),
            ),
        ),
    )
