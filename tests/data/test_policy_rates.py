"""Synthetic/static-only tests for the committed ECBDFR/EFFR policy-rate loader.

No live network calls are made anywhere in this file. Real committed
provenance-checked CSVs are used for hash/lag/propagation assertions (they
are already frozen, hash-verified repository data, not DEVELOPMENT market
data); tamper and fixture-based cases build their own temporary directories.
"""

from __future__ import annotations

import hashlib
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from ftmoquant.data.policy_rates import (
    ECBDFR_RAW_SHA256,
    EFFR_RAW_SHA256,
    POLICY_RATES_DIR,
    DatedRate,
    PolicyRateHistory,
    PolicyRateProvenanceError,
    causal_differential_series,
    causal_differential_series_pre_development_only,
    load_policy_rate_history,
)
from ftmoquant.research.stage_g import VALIDATION_START


def test_committed_source_hashes_match_frozen_constants() -> None:
    history = load_policy_rate_history()
    assert (
        hashlib.sha256((POLICY_RATES_DIR / "ECBDFR_raw.csv").read_bytes()).hexdigest()
        == ECBDFR_RAW_SHA256
    )
    assert (
        hashlib.sha256((POLICY_RATES_DIR / "EFFR_raw.csv").read_bytes()).hexdigest()
        == EFFR_RAW_SHA256
    )
    assert history.canonical_data_sha256


def test_tampering_a_single_byte_is_rejected(tmp_path: Path) -> None:
    directory = _copy_policy_rates_dir(tmp_path)
    corrupted = directory / "ECBDFR_raw.csv"
    original = corrupted.read_bytes()
    corrupted.write_bytes(original[:-1] + bytes([original[-1] ^ 0x01]))

    with pytest.raises(PolicyRateProvenanceError, match="SHA-256 mismatch"):
        load_policy_rate_history(directory)


def test_rate_units_are_percent_per_annum_decimals() -> None:
    history = load_policy_rate_history()
    target = date(2018, 1, 1)
    sample = next(item for item in history.ecbdfr if item.observation_date == target)
    assert sample.value_percent == Decimal("-0.40")


def test_one_business_day_information_lag_uses_prior_business_day_only() -> None:
    history = load_policy_rate_history()
    # Friday 2018-01-05: lag cutoff is Thursday 2018-01-04.
    observations = causal_differential_series(
        history, start_inclusive=date(2018, 1, 5), end_inclusive=date(2018, 1, 5)
    )
    assert observations[0].effr_observation_date == date(2018, 1, 4)
    assert observations[0].ecbdfr_observation_date == date(2018, 1, 4)


def test_weekend_forward_fills_from_the_last_available_business_day() -> None:
    history = load_policy_rate_history()
    # Saturday/Sunday still resolve; their causal lag cutoff (Friday) is the
    # same as Monday's cutoff would use Friday too, but the calendar day
    # itself is a genuine weekend day, not a fabricated one.
    saturday, sunday = causal_differential_series(
        history, start_inclusive=date(2018, 1, 6), end_inclusive=date(2018, 1, 7)
    )
    assert saturday.effr_observation_date == date(2018, 1, 5)
    assert sunday.effr_observation_date == date(2018, 1, 5)
    assert saturday.differential_percent == sunday.differential_percent


def test_holiday_gap_forward_fills_using_the_committed_mlk_day_gap() -> None:
    # 2018-01-15 (MLK Day) has an explicit empty EFFR row in the committed
    # CSV; the lag cutoff for as_of_date 2018-01-17 (Wed) is 2018-01-16
    # (Tue), which does have a value, so no gap gets exercised here directly
    # -- exercise the gap itself: as_of_date 2018-01-16's lag cutoff is
    # 2018-01-15, the holiday itself, which has no EFFR row and must forward
    # fill from 2018-01-12 (Fri).
    history = load_policy_rate_history()
    (observation,) = causal_differential_series(
        history, start_inclusive=date(2018, 1, 16), end_inclusive=date(2018, 1, 16)
    )
    assert observation.effr_observation_date == date(2018, 1, 12)


def test_july_31_2019_fomc_cut_propagates_only_after_its_own_lag() -> None:
    history = load_policy_rate_history()
    before, on_cut, after = causal_differential_series(
        history, start_inclusive=date(2019, 7, 31), end_inclusive=date(2019, 8, 2)
    )
    # as_of 2019-07-31 (Wed): lag cutoff Tue 07-30, still pre-cut (2.39).
    assert before.effr_observation_date == date(2019, 7, 30)
    # as_of 2019-08-01 (Thu, the cut's own effective date): lag cutoff Wed
    # 07-31, still pre-cut (2.40) -- the cut is not visible same-day.
    assert on_cut.effr_observation_date == date(2019, 7, 31)
    # as_of 2019-08-02 (Fri): lag cutoff Thu 08-01, now reflects the cut
    # (2.14) -- exactly one business day after the cut took effect.
    assert after.effr_observation_date == date(2019, 8, 1)
    assert before.differential_percent != after.differential_percent


def test_equal_rates_yield_zero_differential() -> None:
    # causal_differential_series takes a PolicyRateHistory value directly, so
    # this synthetic fixture never needs to pass load_policy_rate_history's
    # frozen-hash provenance gate (which is intentionally locked to the one
    # committed real dataset).
    history = PolicyRateHistory(
        ecbdfr=(
            DatedRate(date(2024, 1, 1), Decimal("1.50")),
            DatedRate(date(2024, 1, 2), Decimal("1.50")),
        ),
        effr=(
            DatedRate(date(2024, 1, 1), Decimal("1.50")),
            DatedRate(date(2024, 1, 2), Decimal("1.50")),
        ),
        canonical_data_sha256="synthetic",
    )
    (observation,) = causal_differential_series(
        history, start_inclusive=date(2024, 1, 3), end_inclusive=date(2024, 1, 3)
    )
    assert observation.differential_percent == Decimal("0")


def test_no_future_backfill_a_later_dated_value_never_leaks_earlier() -> None:
    history = PolicyRateHistory(
        ecbdfr=(
            DatedRate(date(2024, 1, 1), Decimal("1.00")),
            DatedRate(date(2024, 1, 5), Decimal("9.99")),
        ),
        effr=(
            DatedRate(date(2024, 1, 1), Decimal("1.00")),
            DatedRate(date(2024, 1, 5), Decimal("1.00")),
        ),
        canonical_data_sha256="synthetic",
    )
    (observation,) = causal_differential_series(
        history, start_inclusive=date(2024, 1, 3), end_inclusive=date(2024, 1, 3)
    )
    assert observation.ecbdfr_observation_date == date(2024, 1, 1)
    assert observation.differential_percent == Decimal("0")


def test_development_window_differential_sign_never_flips() -> None:
    history = load_policy_rate_history()
    observations = causal_differential_series(
        history,
        start_inclusive=date(2019, 3, 11),
        end_inclusive=date(2023, 4, 11),
    )
    assert all(item.differential_percent < 0 for item in observations)


def test_canonical_representation_is_deterministic_across_repeated_loads() -> None:
    first = load_policy_rate_history()
    second = load_policy_rate_history()
    assert first.canonical_data_sha256 == second.canonical_data_sha256


def test_pre_development_seal_accepts_a_range_strictly_inside_development() -> None:
    history = load_policy_rate_history()
    observations = causal_differential_series_pre_development_only(
        history,
        start_inclusive=date(2019, 3, 11),
        end_inclusive=date(2023, 4, 10),  # strictly before VALIDATION_START's date
    )
    assert observations
    assert all(item.as_of_date < VALIDATION_START.date() for item in observations)


def test_pre_development_seal_rejects_a_range_touching_validation_start() -> None:
    history = load_policy_rate_history()
    with pytest.raises(PolicyRateProvenanceError):
        causal_differential_series_pre_development_only(
            history,
            start_inclusive=date(2019, 3, 11),
            end_inclusive=VALIDATION_START.date(),  # touches, must reject
        )


def test_pre_development_seal_rejects_a_range_exceeding_validation_start() -> None:
    history = load_policy_rate_history()
    with pytest.raises(PolicyRateProvenanceError):
        causal_differential_series_pre_development_only(
            history,
            start_inclusive=date(2019, 3, 11),
            end_inclusive=date(2024, 1, 1),  # deep into validation/holdout
        )


def _copy_policy_rates_dir(tmp_path: Path) -> Path:
    import shutil

    destination = tmp_path / "policy_rates"
    shutil.copytree(POLICY_RATES_DIR, destination)
    return destination


