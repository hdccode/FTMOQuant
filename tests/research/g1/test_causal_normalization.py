from __future__ import annotations

import math
from decimal import Decimal

import pandas as pd
import pytest

from ftmoquant.research.g1.normalization import (
    G1_ANNUALIZATION_DAYS,
    G1_EWMA_CENTER_OF_MASS_DAYS,
    G1_MINIMUM_COMPLETED_DAILY_RETURNS,
    CausalEwmaDailyVolatility,
    CompletedDailyLogReturn,
    CompletedDailyMidpoint,
    G1VolatilityNormalizer,
    VolatilityNormalizationError,
    completed_daily_log_returns,
)

DAY_NS = 86_400_000_000_000


def _returns(values: list[float]) -> tuple[CompletedDailyLogReturn, ...]:
    return tuple(
        CompletedDailyLogReturn((index + 1) * DAY_NS, value)
        for index, value in enumerate(values)
    )


def _history(count: int = 25) -> list[float]:
    return [0.008 if index % 2 else -0.006 for index in range(count)]


def test_future_append_cannot_change_earlier_volatility_or_exposure() -> None:
    prefix = _history()
    decision = len(prefix) * DAY_NS + 1
    prefix_estimator = CausalEwmaDailyVolatility(_returns(prefix))
    extended_estimator = CausalEwmaDailyVolatility(
        _returns([*prefix, 0.50, -0.50, 1.00])
    )
    normalizer = G1VolatilityNormalizer()

    prefix_vol = prefix_estimator.estimate_before(decision)
    extended_vol = extended_estimator.estimate_before(decision)
    prefix_target = normalizer.target_at(1.0, decision, prefix_estimator)
    extended_target = normalizer.target_at(1.0, decision, extended_estimator)

    assert prefix_vol == extended_vol
    assert prefix_target == extended_target


def test_exact_old_defect_example_is_prefix_invariant_under_causal_api() -> None:
    warmup = _history(20)
    exposed_prefix = [-0.001, 0.0, 0.001, 0.002]
    prefix = [*warmup, *exposed_prefix]
    decision = len(prefix) * DAY_NS + 1
    first = CausalEwmaDailyVolatility(_returns(prefix))
    with_future = CausalEwmaDailyVolatility(_returns([*prefix, 0.10, -0.10]))
    normalizer = G1VolatilityNormalizer()

    assert normalizer.target_at(1.0, decision, first) == normalizer.target_at(
        1.0, decision, with_future
    )
    assert not hasattr(normalizer, "normalize")


def test_strict_lag_and_warmup_remain_unavailable_with_later_history() -> None:
    values = _history(30)
    estimator = CausalEwmaDailyVolatility(_returns(values))
    twentieth_endpoint = G1_MINIMUM_COMPLETED_DAILY_RETURNS * DAY_NS

    assert estimator.estimate_before(twentieth_endpoint) is None
    assert estimator.estimate_before(twentieth_endpoint + 1) is not None
    assert G1VolatilityNormalizer().target_at(
        1.0, twentieth_endpoint, estimator
    ).desired_exposure == 0.0


def test_future_volatility_shock_cannot_change_pre_shock_h1_or_h4_sizing() -> None:
    stable = _history(25)
    shock_endpoint = (len(stable) + 1) * DAY_NS
    estimator = CausalEwmaDailyVolatility(_returns([*stable, 2.0]))
    decision = shock_endpoint
    normalizer = G1VolatilityNormalizer()

    h1 = normalizer.target_at(1.0, decision, estimator)
    h4 = normalizer.target_at(1.0, decision, estimator)
    prefix = normalizer.target_at(
        1.0, decision, CausalEwmaDailyVolatility(_returns(stable))
    )

    assert h1 == h4 == prefix
    after_shock = normalizer.target_at(1.0, shock_endpoint + 1, estimator)
    assert after_shock.desired_exposure < h1.desired_exposure


def test_pandas_bias_corrected_ewma_and_sqrt_252_annualization_are_exact() -> None:
    values = _history(25)
    estimator = CausalEwmaDailyVolatility(_returns(values))
    actual = estimator.estimate_before(len(values) * DAY_NS + 1)
    variance = float(
        pd.Series(values, dtype="float64")
        .ewm(
            com=G1_EWMA_CENTER_OF_MASS_DAYS,
            adjust=True,
            min_periods=G1_MINIMUM_COMPLETED_DAILY_RETURNS,
            ignore_na=False,
        )
        .var(bias=False)
        .iloc[-1]
    )

    assert actual == pytest.approx(math.sqrt(variance) * math.sqrt(252))
    assert G1_ANNUALIZATION_DAYS == 252


def test_sizing_sign_pathology_monotonicity_and_determinism() -> None:
    normalizer = G1VolatilityNormalizer()

    assert normalizer.exposure_for_volatility(1.0, 0.10) == pytest.approx(0.10)
    assert normalizer.exposure_for_volatility(-1.0, 0.10) == pytest.approx(-0.10)
    assert normalizer.exposure_for_volatility(0.0, None) == 0.0
    assert normalizer.exposure_for_volatility(1.0, 0.05) > (
        normalizer.exposure_for_volatility(1.0, 0.20)
    )
    for pathological in (None, 0.0, -1.0, math.nan, math.inf, -math.inf):
        assert normalizer.exposure_for_volatility(1.0, pathological) == 0.0
    assert normalizer.exposure_for_volatility(1.0, 0.10) == (
        normalizer.exposure_for_volatility(1.0, 0.10)
    )
    with pytest.raises(VolatilityNormalizationError, match="directional signal"):
        normalizer.exposure_for_volatility(2.0, 0.10)


def test_completed_midpoints_create_ordered_causal_daily_log_returns() -> None:
    midpoints = (
        CompletedDailyMidpoint(DAY_NS, 1.10),
        CompletedDailyMidpoint(2 * DAY_NS, 1.11),
        CompletedDailyMidpoint(3 * DAY_NS, 1.09),
    )

    returns = completed_daily_log_returns(midpoints)

    assert tuple(item.endpoint_information_time_ns for item in returns) == (
        2 * DAY_NS,
        3 * DAY_NS,
    )
    assert returns[0].log_return == pytest.approx(math.log(1.11 / 1.10))
    with pytest.raises(VolatilityNormalizationError, match="strictly information"):
        completed_daily_log_returns((midpoints[1], midpoints[0]))


def test_exposure_converts_to_scaled_base_units_before_execution() -> None:
    decision = G1VolatilityNormalizer().target_at(
        1.0,
        26 * DAY_NS,
        CausalEwmaDailyVolatility(_returns(_history(25))),
    )

    assert decision.target_base_units(Decimal("100000")) == (
        Decimal(str(decision.desired_exposure)) * Decimal("100000")
    )
