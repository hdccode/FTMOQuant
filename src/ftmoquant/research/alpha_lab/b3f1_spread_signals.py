"""B3F1 (cross-instrument spread mean reversion) -- causal formation
statistics and H1-decision signal generation.

Frozen by ``config/research/batch3_methodology_preregistration_v2.json``
(semantic hash
``e5cd74527004585cfd24bea55549d84e9cb66b05ffc6498360dac5007b651f7c``). This
module implements sections 2-11 and 16 of the B3.2 task brief: the causal
hedge-ratio/stationarity/normalization pipeline and the pure, M1-independent
entry/exit decision walker. Execution (M1 fills, sizing, cost stress) is a
separate concern -- see :mod:`ftmoquant.research.alpha_lab.
b3f1_spread_execution`.

No pair is selected here by performance -- :func:`enumerate_candidate_pairs`
returns all 21 unordered combinations of the frozen seven-pair OANDA
universe identically, and :func:`build_b3f1_grid` returns the full frozen
18-configuration grid identically for every pair.

Performance design (task section 16): rolling OLS is vectorized in closed
form (``pandas``/``numpy`` rolling sums, O(n) total, not O(n*window)) since
that is "mathematically identical" to per-bar ``statsmodels`` OLS -- proven
by a parity test against ``statsmodels.api.OLS`` on synthetic data. The ADF
test has no vectorized equivalent in ``statsmodels`` and none is
substituted; :func:`compute_formation_series` therefore calls
``statsmodels.tsa.stattools.adfuller`` once per eligible bar. This is
computed ONCE per (pair orientation, formation_window) and cached (as a
returned ``pandas.DataFrame``) so all six ``(z_entry, z_stop)`` combinations
sharing that formation window reuse the identical causal series -- an
explicit 6x reduction (18 configs -> 3 formation passes per pair), matching
the only caching strategy the task brief permits. Real-DEVELOPMENT-scale
cost (21 pairs x 3 windows x ~35,000 H1 bars, each requiring one
``adfuller`` call) is a known, documented, accepted cost of the frozen
methodology -- not something this module works around by reducing fitting
frequency or substituting a cheaper test.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from statsmodels.tsa.stattools import adfuller  # type: ignore[import-untyped]

from ftmoquant.data.instruments import (
    AUDUSD_OANDA_SPEC,
    EURUSD_OANDA_SPEC,
    GBPUSD_OANDA_SPEC,
    NZDUSD_OANDA_SPEC,
    USDCAD_OANDA_SPEC,
    USDCHF_OANDA_SPEC,
    USDJPY_OANDA_SPEC,
    InstrumentSpec,
)

# ---------------------------------------------------------------------------
# Frozen universe / grid (sections 3, 5, 9)
# ---------------------------------------------------------------------------

#: The frozen seven-pair OANDA alpha-lab universe, unchanged from Batch 1/2.
PAIR_UNIVERSE: tuple[InstrumentSpec, ...] = (
    AUDUSD_OANDA_SPEC,
    EURUSD_OANDA_SPEC,
    GBPUSD_OANDA_SPEC,
    NZDUSD_OANDA_SPEC,
    USDCAD_OANDA_SPEC,
    USDCHF_OANDA_SPEC,
    USDJPY_OANDA_SPEC,
)
assert len(PAIR_UNIVERSE) == 7
assert len({spec.instrument_id for spec in PAIR_UNIVERSE}) == 7

FORMATION_WINDOWS: tuple[int, ...] = (240, 480, 960)
Z_ENTRY_GRID: tuple[Decimal, ...] = (Decimal("1.5"), Decimal("2.0"), Decimal("2.5"))
Z_STOP_GRID: tuple[Decimal, ...] = (Decimal("3.0"), Decimal("3.5"))
ADF_P_VALUE_THRESHOLD = 0.05
ADF_REGRESSION = "c"
ADF_AUTOLAG = "AIC"


class B3F1SignalError(ValueError):
    """Raised on any violation of the frozen B3F1 formation/signal contract."""


@dataclass(frozen=True, slots=True)
class OrderedPair:
    """One unordered instrument-pair sleeve with its FROZEN, deterministic
    Y/X orientation: ``y`` is whichever instrument's ``instrument_id`` is
    lexicographically SMALLER; ``x`` is the other. This choice is arbitrary
    but fixed here, before any DEVELOPMENT result exists for any pair --
    never chosen or swapped after observing which orientation performs
    better."""

    y: InstrumentSpec
    x: InstrumentSpec

    def __post_init__(self) -> None:
        if self.y.instrument_id >= self.x.instrument_id:
            raise B3F1SignalError(
                "y.instrument_id must be strictly lexicographically smaller "
                "than x.instrument_id"
            )

    @property
    def sleeve_id(self) -> str:
        return f"{self.y.instrument_id}__{self.x.instrument_id}"


def enumerate_candidate_pairs(
    universe: Sequence[InstrumentSpec] = PAIR_UNIVERSE,
) -> tuple[OrderedPair, ...]:
    """All C(7,2) = 21 unordered pair combinations of ``universe``, each
    with its frozen lexicographic Y/X orientation. Identical treatment for
    every combination -- no pair is ever selected or excluded here."""

    specs = sorted(universe, key=lambda spec: spec.instrument_id)
    pairs = tuple(
        OrderedPair(y=specs[i], x=specs[j])
        for i in range(len(specs))
        for j in range(i + 1, len(specs))
    )
    expected = len(specs) * (len(specs) - 1) // 2
    if len(pairs) != expected:
        raise B3F1SignalError(f"expected {expected} unordered pairs, got {len(pairs)}")
    return pairs


@dataclass(frozen=True, slots=True)
class B3F1Config:
    """One frozen statistical configuration cell (section 9): exactly
    ``len(FORMATION_WINDOWS) * len(Z_ENTRY_GRID) * len(Z_STOP_GRID) == 18``
    of these exist, identical for every pair sleeve."""

    formation_window: int
    z_entry: Decimal
    z_stop: Decimal

    def __post_init__(self) -> None:
        if self.formation_window not in FORMATION_WINDOWS:
            raise B3F1SignalError(
                f"formation_window {self.formation_window} not frozen"
            )
        if self.z_entry not in Z_ENTRY_GRID:
            raise B3F1SignalError(f"z_entry {self.z_entry} not frozen")
        if self.z_stop not in Z_STOP_GRID:
            raise B3F1SignalError(f"z_stop {self.z_stop} not frozen")
        if not (self.z_stop > self.z_entry):
            raise B3F1SignalError("z_stop must be strictly greater than z_entry")

    @property
    def config_id(self) -> str:
        return f"fw{self.formation_window}_ze{self.z_entry}_zs{self.z_stop}"


def build_b3f1_grid() -> tuple[B3F1Config, ...]:
    """The full, frozen 18-configuration grid -- identical for every pair
    sleeve, built once and reused unmodified for the whole broad screen."""

    grid = tuple(
        B3F1Config(formation_window=window, z_entry=z_entry, z_stop=z_stop)
        for window in FORMATION_WINDOWS
        for z_entry in Z_ENTRY_GRID
        for z_stop in Z_STOP_GRID
        if z_stop > z_entry
    )
    if len(grid) != 18:
        raise B3F1SignalError(f"expected exactly 18 frozen configs, got {len(grid)}")
    return grid


# ---------------------------------------------------------------------------
# Causal formation statistics (sections 4, 6, 7, 8)
# ---------------------------------------------------------------------------

_FORMATION_COLUMNS = (
    "alpha",
    "beta",
    "spread",
    "spread_mean",
    "spread_std",
    "adf_pvalue",
    "valid",
    "z",
)


def _rolling_ols_alpha_beta(
    log_y: pd.Series, log_x: pd.Series, window: int
) -> tuple[pd.Series, pd.Series]:
    """Vectorized rolling simple-OLS (``log_y = alpha + beta*log_x``) fit
    on the ``window`` bars strictly BEFORE each timestamp (``shift(1)``
    before rolling, so bar ``t``'s fit uses exactly ``[t-window, t)``,
    never bar ``t`` itself). Closed-form least squares -- mathematically
    identical to ``statsmodels.api.OLS(..., add_constant(...)).fit()`` on
    the same window (proven in
    ``tests/research/alpha_lab/test_b3f1_spread_signals.py::
    test_rolling_ols_matches_statsmodels_ols_exactly``), used instead of
    calling ``statsmodels`` once per bar purely for speed -- the ADF test
    below has no such vectorized substitute and is not approximated."""

    lagged_x = log_x.shift(1)
    lagged_y = log_y.shift(1)
    sum_x = lagged_x.rolling(window).sum()
    sum_y = lagged_y.rolling(window).sum()
    sum_xx = (lagged_x * lagged_x).rolling(window).sum()
    sum_xy = (lagged_x * lagged_y).rolling(window).sum()

    denominator = window * sum_xx - sum_x * sum_x
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = (window * sum_xy - sum_x * sum_y) / denominator
        alpha = (sum_y - beta * sum_x) / window
    beta = beta.where(denominator != 0)
    alpha = alpha.where(denominator != 0)
    return alpha, beta


def compute_formation_series(
    log_y: pd.Series, log_x: pd.Series, window: int
) -> pd.DataFrame:
    """The full causal formation-statistics series for one (pair
    orientation, formation_window), computed ONCE and reused by every
    ``(z_entry, z_stop)`` config sharing this window (section 16 caching).

    ``log_y``/``log_x`` must be aligned, sorted, log-mid-close H1 series
    with an identical index. Returns a DataFrame indexed identically, with
    columns ``alpha, beta, spread, spread_mean, spread_std, adf_pvalue,
    valid, z``. ``valid`` is False (and ``z`` is NaN) wherever there is
    insufficient history, ``beta <= 0``, ``spread_std`` is not finite and
    positive, or the ADF p-value exceeds :data:`ADF_P_VALUE_THRESHOLD` --
    every such bar has "no signal / formation invalid" per section 6.

    ADF settings are frozen exactly as specified: ``regression="c"``,
    ``autolag="AIC"``, applied to the IN-SAMPLE residuals of the same
    lagged formation window used to fit ``alpha``/``beta`` (never the
    current, unlagged bar).
    """

    if not log_y.index.equals(log_x.index):
        raise B3F1SignalError("log_y and log_x must share an identical index")
    if not log_y.index.is_monotonic_increasing:
        raise B3F1SignalError("log_y/log_x index must be sorted ascending")
    if window <= 1:
        raise B3F1SignalError("window must be > 1")

    alpha, beta = _rolling_ols_alpha_beta(log_y, log_x, window)
    spread = log_y - alpha - beta * log_x

    lagged_x_values = log_x.shift(1).to_numpy()
    lagged_y_values = log_y.shift(1).to_numpy()
    alpha_values = alpha.to_numpy()
    beta_values = beta.to_numpy()
    n = len(log_y)

    spread_mean = np.full(n, np.nan)
    spread_std = np.full(n, np.nan)
    adf_pvalue = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)

    for t in range(window, n):
        a = alpha_values[t]
        b = beta_values[t]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if not (b > 0):
            continue
        window_x = lagged_x_values[t - window : t]
        window_y = lagged_y_values[t - window : t]
        residuals = window_y - a - b * window_x
        std = residuals.std(ddof=1)
        if not (np.isfinite(std) and std > 0):
            continue
        mean = residuals.mean()
        p_value = float(
            adfuller(residuals, regression=ADF_REGRESSION, autolag=ADF_AUTOLAG)[1]
        )
        spread_mean[t] = mean
        spread_std[t] = std
        adf_pvalue[t] = p_value
        valid[t] = p_value <= ADF_P_VALUE_THRESHOLD

    spread_mean_series = pd.Series(spread_mean, index=log_y.index)
    spread_std_series = pd.Series(spread_std, index=log_y.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (spread - spread_mean_series) / spread_std_series
    z = z.where(valid)

    result = pd.DataFrame(
        {
            "alpha": alpha,
            "beta": beta,
            "spread": spread,
            "spread_mean": spread_mean_series,
            "spread_std": spread_std_series,
            "adf_pvalue": adf_pvalue,
            "valid": valid,
            "z": z,
        },
        index=log_y.index,
    )
    return result[list(_FORMATION_COLUMNS)]


# ---------------------------------------------------------------------------
# H1-decision signal walker (sections 9-12, 14)
# ---------------------------------------------------------------------------


class SpreadSide(StrEnum):
    """Which side of the spread a logical position is on. ``RICH`` = short
    Y / long X (entered on ``z >= +z_entry``); ``CHEAP`` = long Y / short X
    (entered on ``z <= -z_entry``)."""

    RICH = "rich"
    CHEAP = "cheap"


EXIT_REASON_Z_STOP = "z_stop"
EXIT_REASON_Z_MEAN_REVERSION = "z_mean_reversion"


@dataclass(frozen=True, slots=True)
class B3F1TradeIntent:
    """One fully-causal, M1-independent logical spread-trade decision: an
    H1 entry bar, an H1 exit bar, and the ENTRY-FROZEN formation
    parameters (section 11 -- never rebalanced while open). Execution
    (finding actual M1 fills for both legs) is a separate, later step."""

    sleeve_id: str
    side: SpreadSide
    entry_ts: pd.Timestamp
    entry_z: float
    frozen_alpha: float
    frozen_beta: float
    frozen_spread_mean: float
    frozen_spread_std: float
    exit_ts: pd.Timestamp
    exit_reason: str

    def __post_init__(self) -> None:
        if self.exit_ts <= self.entry_ts:
            raise B3F1SignalError("exit_ts must be strictly after entry_ts")
        if self.exit_reason not in (EXIT_REASON_Z_STOP, EXIT_REASON_Z_MEAN_REVERSION):
            raise B3F1SignalError(f"unknown exit_reason {self.exit_reason!r}")


def generate_b3f1_decisions(
    formation: pd.DataFrame,
    log_y: pd.Series,
    log_x: pd.Series,
    *,
    sleeve_id: str,
    z_entry: Decimal,
    z_stop: Decimal,
) -> tuple[B3F1TradeIntent, ...]:
    """Walk ``formation`` bar-by-bar (chronological, H1 decision cadence
    only -- no M1 data, no intrabar fabrication) and emit the causal
    entry/exit decision sequence for one ``(z_entry, z_stop)`` config
    sharing this already-computed ``formation`` series.

    One logical position at a time (section 12): a fresh entry signal
    while a position is already open is silently ignored, never queued.
    On each open bar, the STOP condition is checked before the mean-exit
    condition, so a bar where both would fire exits as a stop (section
    14). The exit itself uses the frozen entry-time alpha/beta/mean/std
    (section 11), evaluated against that bar's own current, unlagged
    ``log_y``/``log_x`` (never a future bar).
    """

    if not (
        formation.index.equals(log_y.index) and formation.index.equals(log_x.index)
    ):
        raise B3F1SignalError("formation/log_y/log_x must share an identical index")

    z_entry_f = float(z_entry)
    z_stop_f = float(z_stop)
    intents: list[B3F1TradeIntent] = []

    open_side: SpreadSide | None = None
    open_entry_ts: pd.Timestamp | None = None
    open_entry_z: float | None = None
    open_alpha = open_beta = open_mean = open_std = None

    for ts in formation.index:
        row = formation.loc[ts]
        if open_side is None:
            if not bool(row["valid"]):
                continue
            z_value = float(row["z"])
            if z_value >= z_entry_f:
                open_side = SpreadSide.RICH
            elif z_value <= -z_entry_f:
                open_side = SpreadSide.CHEAP
            else:
                continue
            open_entry_ts = ts
            open_entry_z = z_value
            open_alpha = float(row["alpha"])
            open_beta = float(row["beta"])
            open_mean = float(row["spread_mean"])
            open_std = float(row["spread_std"])
            continue

        assert open_alpha is not None and open_beta is not None
        assert open_mean is not None and open_std is not None
        frozen_spread = (
            float(log_y.loc[ts]) - open_alpha - open_beta * float(log_x.loc[ts])
        )
        frozen_z = (frozen_spread - open_mean) / open_std

        stop_hit = abs(frozen_z) >= z_stop_f
        if open_side is SpreadSide.RICH:
            mean_hit = frozen_z <= 0.0
        else:
            mean_hit = frozen_z >= 0.0

        if stop_hit:
            exit_reason = EXIT_REASON_Z_STOP
        elif mean_hit:
            exit_reason = EXIT_REASON_Z_MEAN_REVERSION
        else:
            continue

        assert open_entry_ts is not None and open_entry_z is not None
        intents.append(
            B3F1TradeIntent(
                sleeve_id=sleeve_id,
                side=open_side,
                entry_ts=open_entry_ts,
                entry_z=open_entry_z,
                frozen_alpha=open_alpha,
                frozen_beta=open_beta,
                frozen_spread_mean=open_mean,
                frozen_spread_std=open_std,
                exit_ts=ts,
                exit_reason=exit_reason,
            )
        )
        open_side = None
        open_entry_ts = None
        open_entry_z = None
        open_alpha = open_beta = open_mean = open_std = None

    return tuple(intents)
