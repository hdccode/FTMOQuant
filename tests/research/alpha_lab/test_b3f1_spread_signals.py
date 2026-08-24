from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm  # type: ignore[import-untyped]

from ftmoquant.research.alpha_lab.b3f1_spread_signals import (
    ADF_AUTOLAG,
    ADF_P_VALUE_THRESHOLD,
    ADF_REGRESSION,
    EXIT_REASON_Z_MEAN_REVERSION,
    EXIT_REASON_Z_STOP,
    PAIR_UNIVERSE,
    B3F1Config,
    B3F1SignalError,
    B3F1TradeIntent,
    OrderedPair,
    SpreadSide,
    build_b3f1_grid,
    compute_formation_series,
    enumerate_candidate_pairs,
    generate_b3f1_decisions,
)


def _series(index: pd.DatetimeIndex, values: np.ndarray) -> pd.Series:
    return pd.Series(values, index=index)


def _synthetic_log_prices(
    n: int, *, beta: float = 1.2, alpha: float = 0.1, seed: int = 0
):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    x = np.cumsum(rng.normal(scale=0.0005, size=n)) + 4.0
    residual = rng.normal(scale=0.0005, size=n)
    y = alpha + beta * x + residual
    return idx, _series(idx, y), _series(idx, x)


# ---------------------------------------------------------------------------
# Pair universe / grid
# ---------------------------------------------------------------------------


def test_enumerate_candidate_pairs_returns_exactly_21_combinations() -> None:
    pairs = enumerate_candidate_pairs()
    assert len(pairs) == 21
    assert len({p.sleeve_id for p in pairs}) == 21


def test_orientation_is_lexicographic_by_instrument_id() -> None:
    for pair in enumerate_candidate_pairs():
        assert pair.y.instrument_id < pair.x.instrument_id


def test_ordered_pair_rejects_reversed_orientation() -> None:
    specs = sorted(PAIR_UNIVERSE, key=lambda s: s.instrument_id)
    with pytest.raises(B3F1SignalError):
        OrderedPair(y=specs[1], x=specs[0])


def test_build_b3f1_grid_has_exactly_18_configs() -> None:
    grid = build_b3f1_grid()
    assert len(grid) == 18
    assert len({config.config_id for config in grid}) == 18


def test_all_frozen_configs_satisfy_z_stop_gt_z_entry() -> None:
    for config in build_b3f1_grid():
        assert config.z_stop > config.z_entry


def test_config_rejects_non_frozen_values() -> None:
    with pytest.raises(B3F1SignalError):
        B3F1Config(formation_window=100, z_entry=Decimal("1.5"), z_stop=Decimal("3.0"))
    with pytest.raises(B3F1SignalError):
        B3F1Config(formation_window=240, z_entry=Decimal("1.5"), z_stop=Decimal("1.0"))


# ---------------------------------------------------------------------------
# Causal formation statistics
# ---------------------------------------------------------------------------


def test_formation_excludes_the_current_bar() -> None:
    """A pathological current-bar outlier must not perturb alpha/beta at
    that same bar (they are fit on [t-window, t) only)."""

    idx, log_y, log_x = _synthetic_log_prices(80)
    formation_clean = compute_formation_series(log_y, log_x, window=20)

    log_y_perturbed = log_y.copy()
    log_y_perturbed.iloc[60] += 5.0  # huge outlier at bar 60 only
    formation_perturbed = compute_formation_series(log_y_perturbed, log_x, window=20)

    # alpha/beta AT bar 60 depend only on bars [40, 60) -- untouched by the
    # bar-60 outlier itself.
    assert formation_clean["alpha"].iloc[60] == pytest.approx(
        formation_perturbed["alpha"].iloc[60]
    )
    assert formation_clean["beta"].iloc[60] == pytest.approx(
        formation_perturbed["beta"].iloc[60]
    )


def test_future_append_cannot_alter_historical_formation_values() -> None:
    idx, log_y, log_x = _synthetic_log_prices(100)
    short_formation = compute_formation_series(
        log_y.iloc[:70], log_x.iloc[:70], window=20
    )
    full_formation = compute_formation_series(log_y, log_x, window=20)

    for column in ("alpha", "beta", "adf_pvalue", "z", "valid"):
        pd.testing.assert_series_equal(
            short_formation[column],
            full_formation[column].iloc[:70],
            check_names=False,
        )


def test_rolling_ols_matches_statsmodels_ols_exactly() -> None:
    idx, log_y, log_x = _synthetic_log_prices(60)
    formation = compute_formation_series(log_y, log_x, window=20)

    # Independently refit with statsmodels on the identical [t-window, t)
    # window and compare -- bar t's fit must use exactly the `window` bars
    # strictly before t, never bar t itself.
    for t in (25, 40, 59):
        window_x = log_x.iloc[t - 20 : t].to_numpy()
        window_y = log_y.iloc[t - 20 : t].to_numpy()
        model = sm.OLS(window_y, sm.add_constant(window_x)).fit()
        expected_alpha, expected_beta = model.params
        # rel=1e-6, not machine epsilon: the closed-form rolling-sum
        # formula and statsmodels' QR-decomposition solver are different
        # (both exact) algorithms for the same least-squares problem, so
        # they agree to numerical precision, not bit-for-bit.
        assert formation["alpha"].iloc[t] == pytest.approx(expected_alpha, rel=1e-6)
        assert formation["beta"].iloc[t] == pytest.approx(expected_beta, rel=1e-6)


def test_ols_recovers_a_known_synthetic_beta() -> None:
    idx, log_y, log_x = _synthetic_log_prices(80, beta=1.7, alpha=-0.3, seed=1)
    formation = compute_formation_series(log_y, log_x, window=40)
    valid = formation.dropna(subset=["beta"])
    assert valid["beta"].iloc[-1] == pytest.approx(1.7, abs=0.1)
    # alpha (the intercept) is estimated far less precisely than beta here
    # because log_x's window is centered well away from zero (~4.0), the
    # classic OLS intercept/slope collinearity effect -- a wider tolerance
    # is appropriate for alpha alone; the fitted LINE (alpha + beta*x) is
    # what actually matters for the spread, checked directly below.
    fitted_level = valid["alpha"].iloc[-1] + valid["beta"].iloc[-1] * log_x.iloc[-1]
    true_level = -0.3 + 1.7 * log_x.iloc[-1]
    assert fitted_level == pytest.approx(true_level, abs=0.05)


def test_adf_settings_are_frozen_exactly() -> None:
    assert ADF_REGRESSION == "c"
    assert ADF_AUTOLAG == "AIC"
    assert ADF_P_VALUE_THRESHOLD == 0.05


def test_adf_filter_rejects_a_clearly_nonstationary_pair() -> None:
    """Two independent random walks: no genuine cointegration, so ADF
    should fail to reject the unit-root null for most/all windows,
    leaving the formation invalid."""

    rng = np.random.default_rng(2)
    n = 80
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    log_x = _series(idx, np.cumsum(rng.normal(scale=0.01, size=n)) + 4.0)
    log_y = _series(idx, np.cumsum(rng.normal(scale=0.01, size=n)) + 4.5)
    formation = compute_formation_series(log_y, log_x, window=30)
    valid_fraction = formation["valid"].mean()
    assert valid_fraction < 0.5


def test_adf_filter_accepts_a_genuinely_stationary_residual() -> None:
    idx, log_y, log_x = _synthetic_log_prices(100, seed=3)
    formation = compute_formation_series(log_y, log_x, window=40)
    assert formation["valid"].iloc[70:].mean() > 0.5


def test_zero_std_fails_closed() -> None:
    idx = pd.date_range("2024-01-01", periods=30, freq="h", tz="UTC")
    log_x = _series(idx, np.linspace(4.0, 4.1, 30))
    # y is an EXACT linear function of x -> zero-residual window -> std==0.
    log_y = _series(idx, 0.5 + 2.0 * log_x.to_numpy())
    formation = compute_formation_series(log_y, log_x, window=15)
    assert not formation["valid"].iloc[20]
    assert pd.isna(formation["z"].iloc[20])


def test_beta_le_zero_fails_closed() -> None:
    idx = pd.date_range("2024-01-01", periods=30, freq="h", tz="UTC")
    log_x = _series(idx, np.linspace(4.0, 4.3, 30))
    log_y = _series(idx, 5.0 - 0.8 * log_x.to_numpy())  # negative beta by construction
    formation = compute_formation_series(log_y, log_x, window=15)
    assert not formation["valid"].iloc[20:].any()


def test_mismatched_index_is_rejected() -> None:
    idx_a = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    idx_b = pd.date_range("2024-01-02", periods=10, freq="h", tz="UTC")
    with pytest.raises(B3F1SignalError):
        compute_formation_series(
            _series(idx_a, np.ones(10)), _series(idx_b, np.ones(10)), window=5
        )


# ---------------------------------------------------------------------------
# H1-decision signal walker
# ---------------------------------------------------------------------------


def _manual_formation(
    z_values: list[float],
    *,
    alpha: float = 0.1,
    beta: float = 1.2,
    mean: float = 0.0,
    std: float = 1.0,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build a hand-controlled formation series + matching log price series
    so a specific z path can be driven exactly for signal-logic tests,
    without needing a real OLS/ADF fit."""

    idx = pd.date_range("2024-01-01", periods=len(z_values), freq="h", tz="UTC")
    spread = [mean + std * z for z in z_values]
    log_x = pd.Series(np.zeros(len(z_values)), index=idx)
    log_y = pd.Series(
        alpha + np.array(spread), index=idx
    )  # so spread = log_y - alpha - beta*log_x = alpha+spread-alpha=spread (beta*0=0)
    formation = pd.DataFrame(
        {
            "alpha": alpha,
            "beta": beta,
            "spread": spread,
            "spread_mean": mean,
            "spread_std": std,
            "adf_pvalue": 0.01,
            "valid": True,
            "z": z_values,
        },
        index=idx,
    )
    return formation, log_y, log_x


def test_positive_z_entry_gives_short_y_long_x() -> None:
    formation, log_y, log_x = _manual_formation([2.5, 2.5, 0.0])
    decisions = generate_b3f1_decisions(
        formation,
        log_y,
        log_x,
        sleeve_id="s",
        z_entry=Decimal("2.0"),
        z_stop=Decimal("3.5"),
    )
    assert len(decisions) == 1
    assert decisions[0].side is SpreadSide.RICH


def test_negative_z_entry_gives_long_y_short_x() -> None:
    formation, log_y, log_x = _manual_formation([-2.5, -2.5, 0.0])
    decisions = generate_b3f1_decisions(
        formation,
        log_y,
        log_x,
        sleeve_id="s",
        z_entry=Decimal("2.0"),
        z_stop=Decimal("3.5"),
    )
    assert len(decisions) == 1
    assert decisions[0].side is SpreadSide.CHEAP


def test_zero_cross_triggers_mean_reversion_exit() -> None:
    formation, log_y, log_x = _manual_formation([2.5, 1.0, -0.1])
    decisions = generate_b3f1_decisions(
        formation,
        log_y,
        log_x,
        sleeve_id="s",
        z_entry=Decimal("2.0"),
        z_stop=Decimal("3.5"),
    )
    assert len(decisions) == 1
    assert decisions[0].exit_reason == EXIT_REASON_Z_MEAN_REVERSION


def test_z_stop_triggers_stop_exit() -> None:
    formation, log_y, log_x = _manual_formation([2.5, 2.8, 3.6])
    decisions = generate_b3f1_decisions(
        formation,
        log_y,
        log_x,
        sleeve_id="s",
        z_entry=Decimal("2.0"),
        z_stop=Decimal("3.5"),
    )
    assert len(decisions) == 1
    assert decisions[0].exit_reason == EXIT_REASON_Z_STOP


def test_stop_wins_when_both_conditions_coincide() -> None:
    """A RICH position where the frozen z jumps straight past zero to
    below -z_stop in one bar: both "crossed zero" and "|z|>=z_stop" are
    true simultaneously -- stop must win."""

    formation, log_y, log_x = _manual_formation([2.5, -3.6])
    decisions = generate_b3f1_decisions(
        formation,
        log_y,
        log_x,
        sleeve_id="s",
        z_entry=Decimal("2.0"),
        z_stop=Decimal("3.5"),
    )
    assert len(decisions) == 1
    assert decisions[0].exit_reason == EXIT_REASON_Z_STOP


def test_no_pyramiding_a_second_entry_signal_while_open_is_ignored() -> None:
    # RICH entry at bar0 (z=2.5), a second RICH-eligible bar at bar1
    # (z=2.6) must be ignored while open, exit at bar2's zero-cross.
    formation, log_y, log_x = _manual_formation([2.5, 2.6, -0.1])
    decisions = generate_b3f1_decisions(
        formation,
        log_y,
        log_x,
        sleeve_id="s",
        z_entry=Decimal("2.0"),
        z_stop=Decimal("3.5"),
    )
    assert len(decisions) == 1
    assert decisions[0].entry_z == pytest.approx(2.5)


def test_frozen_entry_parameters_remain_unchanged_while_open() -> None:
    """The formation series' alpha/beta/mean/std can drift arbitrarily
    after entry -- the trade intent must still report the ENTRY bar's own
    frozen values, not anything from a later bar."""

    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    formation = pd.DataFrame(
        {
            "alpha": [0.1, 99.0, 99.0],
            "beta": [1.2, 5.0, 5.0],
            "spread": [2.5, 1.0, -0.1],
            "spread_mean": [0.0, 50.0, 50.0],
            "spread_std": [1.0, 9.0, 9.0],
            "adf_pvalue": [0.01, 0.01, 0.01],
            "valid": [True, True, True],
            "z": [2.5, 1.0, -0.1],
        },
        index=idx,
    )
    log_x = pd.Series(np.zeros(3), index=idx)
    log_y = pd.Series([0.1 + 2.5, 0.1 + 1.0, 0.1 - 0.1], index=idx)
    decisions = generate_b3f1_decisions(
        formation,
        log_y,
        log_x,
        sleeve_id="s",
        z_entry=Decimal("2.0"),
        z_stop=Decimal("3.5"),
    )
    assert len(decisions) == 1
    assert decisions[0].frozen_alpha == pytest.approx(0.1)
    assert decisions[0].frozen_beta == pytest.approx(1.2)
    assert decisions[0].frozen_spread_mean == pytest.approx(0.0)
    assert decisions[0].frozen_spread_std == pytest.approx(1.0)


def test_invalid_formation_bar_produces_no_signal() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    formation = pd.DataFrame(
        {
            "alpha": [0.1, 0.1],
            "beta": [1.2, 1.2],
            "spread": [2.5, 2.5],
            "spread_mean": [0.0, 0.0],
            "spread_std": [1.0, 1.0],
            "adf_pvalue": [0.5, 0.5],
            "valid": [False, False],
            "z": [np.nan, np.nan],
        },
        index=idx,
    )
    log_x = pd.Series(np.zeros(2), index=idx)
    log_y = pd.Series(np.zeros(2), index=idx)
    decisions = generate_b3f1_decisions(
        formation,
        log_y,
        log_x,
        sleeve_id="s",
        z_entry=Decimal("2.0"),
        z_stop=Decimal("3.5"),
    )
    assert decisions == ()


def test_trade_intent_rejects_non_frozen_exit_reason() -> None:
    with pytest.raises(B3F1SignalError):
        B3F1TradeIntent(
            sleeve_id="s",
            side=SpreadSide.RICH,
            entry_ts=pd.Timestamp("2024-01-01T00:00:00Z"),
            entry_z=2.0,
            frozen_alpha=0.1,
            frozen_beta=1.2,
            frozen_spread_mean=0.0,
            frozen_spread_std=1.0,
            exit_ts=pd.Timestamp("2024-01-01T01:00:00Z"),
            exit_reason="not_a_frozen_reason",
        )
