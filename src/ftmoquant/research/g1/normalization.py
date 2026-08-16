"""Fixed alpha-comparison risk normalization, deliberately unrelated to G4."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

G1_ANNUAL_VOLATILITY_TARGET = 0.01


class VolatilityNormalizationError(ValueError):
    """Raised when normalization would be undefined or numerically unsafe."""


@dataclass(frozen=True, slots=True)
class G1VolatilityNormalizer:
    """Normalize comparable alpha returns to exactly 1% annualized volatility."""

    target_annualized_volatility: float = G1_ANNUAL_VOLATILITY_TARGET

    def __post_init__(self) -> None:
        if self.target_annualized_volatility != G1_ANNUAL_VOLATILITY_TARGET:
            raise VolatilityNormalizationError("G1 target is fixed at 1% annualized")

    def scale_for(self, observed_annualized_volatility: float) -> float:
        if (
            not math.isfinite(observed_annualized_volatility)
            or observed_annualized_volatility <= 0.0
        ):
            raise VolatilityNormalizationError(
                "observed annualized volatility must be finite and positive"
            )
        scale = self.target_annualized_volatility / observed_annualized_volatility
        if not math.isfinite(scale) or scale <= 0.0:
            raise VolatilityNormalizationError("volatility scale is pathological")
        return scale

    def normalize(
        self, returns: Sequence[float], *, periods_per_year: int
    ) -> tuple[float, ...]:
        if periods_per_year <= 0 or len(returns) < 2:
            raise VolatilityNormalizationError(
                "normalization needs positive periods_per_year and two returns"
            )
        values = np.asarray(returns, dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise VolatilityNormalizationError(
                "returns must be finite and one-dimensional"
            )
        annualized = float(values.std(ddof=1) * math.sqrt(periods_per_year))
        scale = self.scale_for(annualized)
        normalized = values * scale
        if not np.isfinite(normalized).all():
            raise VolatilityNormalizationError("normalized returns are non-finite")
        return tuple(float(value) for value in normalized)
