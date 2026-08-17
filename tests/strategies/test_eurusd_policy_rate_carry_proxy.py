"""Synthetic tests for the sign-only ``eurusd_policy_rate_carry_proxy_v1`` signal."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from ftmoquant.strategies.eurusd_policy_rate_carry_proxy import (
    EurusdPolicyRateCarryProxyState,
    EurusdPolicyRateCarryProxyValidationError,
)
from ftmoquant.strategies.ts_momentum import RawDirectionalTarget


def test_positive_differential_maps_to_long() -> None:
    state = EurusdPolicyRateCarryProxyState()
    result = state.target_for(date(2024, 1, 1), Decimal("0.5"))
    assert result.target is RawDirectionalTarget.LONG


def test_negative_differential_maps_to_short() -> None:
    state = EurusdPolicyRateCarryProxyState()
    result = state.target_for(date(2024, 1, 1), Decimal("-0.5"))
    assert result.target is RawDirectionalTarget.SHORT


def test_zero_differential_maps_to_flat() -> None:
    state = EurusdPolicyRateCarryProxyState()
    result = state.target_for(date(2024, 1, 1), Decimal("0"))
    assert result.target is RawDirectionalTarget.FLAT


def test_rate_state_transitions_when_differential_sign_flips_mid_history() -> None:
    state = EurusdPolicyRateCarryProxyState()
    sequence = [Decimal("0.2"), Decimal("0.1"), Decimal("-0.1"), Decimal("-0.3")]
    targets = [
        state.target_for(date(2024, 1, index + 1), value)
        for index, value in enumerate(sequence)
    ]
    assert [item.target for item in targets] == [
        RawDirectionalTarget.LONG,
        RawDirectionalTarget.LONG,
        RawDirectionalTarget.SHORT,
        RawDirectionalTarget.SHORT,
    ]


def test_state_is_memoryless_across_calls() -> None:
    # The signal has no lookback/parameters: calling out of order or
    # repeatedly for the same date must not change the mapping.
    state = EurusdPolicyRateCarryProxyState()
    first = state.target_for(date(2024, 3, 1), Decimal("0.75"))
    second = state.target_for(date(2024, 1, 1), Decimal("-0.75"))
    third = state.target_for(date(2024, 3, 1), Decimal("0.75"))
    assert first.target is third.target is RawDirectionalTarget.LONG
    assert second.target is RawDirectionalTarget.SHORT


def test_non_finite_differential_is_rejected() -> None:
    state = EurusdPolicyRateCarryProxyState()
    with pytest.raises(EurusdPolicyRateCarryProxyValidationError):
        state.target_for(date(2024, 1, 1), Decimal("NaN"))
