"""Frozen sign-only signal for ``eurusd_policy_rate_carry_proxy_v1``.

Consumes only the causal, one-business-day-lagged ECBDFR-minus-EFFR
differential (see :mod:`ftmoquant.data.policy_rates`) and emits a raw
directional target. No threshold, lookback, smoothing, or parameter exists:
this is deliberately a single, externally-defined, parameterless rule. This
module has no sizing, execution, cost, or carry-accrual logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from ftmoquant.strategies.ts_momentum import RawDirectionalTarget


class EurusdPolicyRateCarryProxyValidationError(ValueError):
    """Raised when causal candidate inputs violate the frozen contract."""


@dataclass(frozen=True, slots=True)
class DailyDirectionalTarget:
    as_of_date: date
    target: RawDirectionalTarget
    differential_percent: Decimal


class EurusdPolicyRateCarryProxyState:
    """Stateless-by-design sign(differential) mapping, one call per day."""

    def target_for(
        self, as_of_date: date, differential_percent: Decimal
    ) -> DailyDirectionalTarget:
        if not differential_percent.is_finite():
            raise EurusdPolicyRateCarryProxyValidationError(
                "differential must be finite"
            )
        if differential_percent > 0:
            target = RawDirectionalTarget.LONG
        elif differential_percent < 0:
            target = RawDirectionalTarget.SHORT
        else:
            target = RawDirectionalTarget.FLAT
        return DailyDirectionalTarget(as_of_date, target, differential_percent)
