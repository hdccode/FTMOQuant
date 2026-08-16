"""Causal G1 underlying-volatility estimation and pre-execution risk sizing."""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

import pandas as pd  # type: ignore[import-untyped]

G1_ANNUAL_VOLATILITY_TARGET = 0.01
G1_EWMA_CENTER_OF_MASS_DAYS = 60.0
G1_MINIMUM_COMPLETED_DAILY_RETURNS = 20
G1_ANNUALIZATION_DAYS = 252


class VolatilityNormalizationError(ValueError):
    """Raised when causal risk-normalization inputs are structurally invalid."""


@dataclass(frozen=True, slots=True)
class CompletedDailyMidpoint:
    """One authoritative midpoint observation known at ``information_time_ns``."""

    information_time_ns: int
    midpoint: float

    def __post_init__(self) -> None:
        _timestamp(self.information_time_ns, "daily midpoint information time")
        if not math.isfinite(self.midpoint) or self.midpoint <= 0.0:
            raise VolatilityNormalizationError(
                "completed daily midpoint must be finite and positive"
            )


@dataclass(frozen=True, slots=True)
class CompletedDailyLogReturn:
    """One daily log return available only at its completed endpoint."""

    endpoint_information_time_ns: int
    log_return: float

    def __post_init__(self) -> None:
        _timestamp(self.endpoint_information_time_ns, "daily return endpoint")
        if not math.isfinite(self.log_return):
            raise VolatilityNormalizationError("daily log return must be finite")


@dataclass(frozen=True, slots=True)
class CausalExposureDecision:
    """A dimensionless target exposure fixed before subsequent P&L is realized."""

    decision_time_ns: int
    directional_signal: float
    ex_ante_annualized_volatility: float | None
    desired_exposure: float

    def target_base_units(self, base_units_per_unit_exposure: Decimal) -> Decimal:
        """Convert exposure to signed units before native order construction."""

        if (
            not isinstance(base_units_per_unit_exposure, Decimal)
            or not base_units_per_unit_exposure.is_finite()
            or base_units_per_unit_exposure <= 0
        ):
            raise VolatilityNormalizationError(
                "base units per unit exposure must be a positive finite Decimal"
            )
        return Decimal(str(self.desired_exposure)) * base_units_per_unit_exposure


def completed_daily_log_returns(
    observations: Sequence[CompletedDailyMidpoint],
) -> tuple[CompletedDailyLogReturn, ...]:
    """Create causal midpoint log returns without fill, interpolation, or backfill."""

    result: list[CompletedDailyLogReturn] = []
    previous: CompletedDailyMidpoint | None = None
    for observation in observations:
        if (
            previous is not None
            and observation.information_time_ns <= previous.information_time_ns
        ):
            raise VolatilityNormalizationError(
                "completed daily midpoints must be strictly information ordered"
            )
        if previous is not None:
            result.append(
                CompletedDailyLogReturn(
                    endpoint_information_time_ns=observation.information_time_ns,
                    log_return=math.log(observation.midpoint / previous.midpoint),
                )
            )
        previous = observation
    return tuple(result)


class CausalEwmaDailyVolatility:
    """Query pandas' bias-corrected EWMA variance using strictly prior returns."""

    def __init__(self, returns: Sequence[CompletedDailyLogReturn]) -> None:
        self._returns = tuple(returns)
        endpoints = tuple(item.endpoint_information_time_ns for item in self._returns)
        if any(
            current <= previous
            for previous, current in zip(endpoints, endpoints[1:])
        ):
            raise VolatilityNormalizationError(
                "completed daily returns must be strictly endpoint ordered"
            )
        self._endpoints = endpoints

    @property
    def returns(self) -> tuple[CompletedDailyLogReturn, ...]:
        return self._returns

    def estimate_before(self, decision_time_ns: int) -> float | None:
        """Return annualized EWMA volatility from strictly prior endpoints."""

        decision = _timestamp(decision_time_ns, "decision time")
        prior_count = bisect_left(self._endpoints, decision)
        if prior_count < G1_MINIMUM_COMPLETED_DAILY_RETURNS:
            return None
        values = pd.Series(
            [item.log_return for item in self._returns[:prior_count]],
            dtype="float64",
        )
        variance = float(
            values.ewm(
                com=G1_EWMA_CENTER_OF_MASS_DAYS,
                adjust=True,
                min_periods=G1_MINIMUM_COMPLETED_DAILY_RETURNS,
                ignore_na=False,
            )
            .var(bias=False)
            .iloc[-1]
        )
        if not math.isfinite(variance) or variance <= 0.0:
            return None
        annualized = math.sqrt(variance * G1_ANNUALIZATION_DAYS)
        if not math.isfinite(annualized) or annualized <= 0.0:
            return None
        return annualized


@dataclass(frozen=True, slots=True)
class G1VolatilityNormalizer:
    """Convert a causal signal and ex-ante underlying risk into target exposure."""

    target_annualized_volatility: float = G1_ANNUAL_VOLATILITY_TARGET

    def __post_init__(self) -> None:
        if self.target_annualized_volatility != G1_ANNUAL_VOLATILITY_TARGET:
            raise VolatilityNormalizationError("G1 target is fixed at 1% annualized")

    def exposure_for_volatility(
        self,
        directional_signal: float,
        ex_ante_annualized_volatility: float | None,
    ) -> float:
        """Fail closed when ex-ante risk is unavailable or pathological."""

        signal = float(directional_signal)
        if not math.isfinite(signal) or abs(signal) > 1.0:
            raise VolatilityNormalizationError(
                "directional signal must be finite and within [-1, 1]"
            )
        if signal == 0.0:
            return 0.0
        if (
            ex_ante_annualized_volatility is None
            or not math.isfinite(ex_ante_annualized_volatility)
            or ex_ante_annualized_volatility <= 0.0
        ):
            return 0.0
        exposure = (
            signal
            * self.target_annualized_volatility
            / ex_ante_annualized_volatility
        )
        return exposure if math.isfinite(exposure) else 0.0

    def target_at(
        self,
        directional_signal: float,
        decision_time_ns: int,
        estimator: CausalEwmaDailyVolatility,
    ) -> CausalExposureDecision:
        """Size before execution from only daily-return endpoints before decision."""

        decision = _timestamp(decision_time_ns, "decision time")
        volatility = estimator.estimate_before(decision)
        return CausalExposureDecision(
            decision_time_ns=decision,
            directional_signal=float(directional_signal),
            ex_ante_annualized_volatility=volatility,
            desired_exposure=self.exposure_for_volatility(
                directional_signal, volatility
            ),
        )


def _timestamp(value: int, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise VolatilityNormalizationError(
            f"{description} must be a non-negative integer nanosecond timestamp"
        )
    return value
