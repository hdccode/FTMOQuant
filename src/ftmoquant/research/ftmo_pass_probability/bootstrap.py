"""Deterministic block-bootstrap resampling of the DEVELOPMENT trade path.

Reuses ``arch.bootstrap.StationaryBootstrap`` / ``CircularBlockBootstrap``
for the actual resampling (never reimplements a block-bootstrap algorithm)
and :func:`ftmoquant.research.statistics.estimate_optimal_block_length` --
itself a thin, already-reviewed ``arch.bootstrap.optimal_block_length``
wrapper -- for the DEVELOPMENT-only block-length estimate that must be
frozen before any VALIDATION simulation (task §6).

Resampling unit: consecutive complete trade episodes, not calendar-day
vectors. This frozen strategy holds at most one position at a time (proven
in ``path_extraction._validate_chronology_and_holdout``) with holding times
ranging from minutes to ~62 days (``longest_holding_time_ns`` in
``development_execution/result.json``) -- collapsing that onto a fixed
calendar-day grid would either shatter a single 62-day trade into dozens of
synthetic day-rows with no intratrade information to fill them, or silently
merge distinct trades. The trade episode is this strategy's actual atomic
unit of local dependence (win/loss streaks are streaks of trades, not of
days), so resampling blocks of *trades* is the least-lossy choice available,
per task §7 -- it was not chosen because it produces a higher pass
probability than a day-vector unit; no day-vector alternative was scored
against pass probability, only against representational fidelity to the
single-position structure above.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from arch.bootstrap import CircularBlockBootstrap, StationaryBootstrap

from ftmoquant.research.ftmo_pass_probability.path_extraction import TradeRecord
from ftmoquant.research.statistics import estimate_optimal_block_length

ResamplingMethod = Literal["stationary", "circular", "iid_diagnostic"]

_PRIMARY_METHODS: frozenset[str] = frozenset({"stationary", "circular"})


class BootstrapValidationError(ValueError):
    """Raised when a resampling request is ambiguous or unsafe."""


@dataclass(frozen=True, slots=True)
class FrozenBlockLength:
    """The DEVELOPMENT-derived block length frozen before any VALIDATION use."""

    stationary_block_length: float
    circular_block_length: float
    frozen_block_size: int
    observation_count: int
    input_content_sha256: str


def derive_frozen_block_length(trades: tuple[TradeRecord, ...]) -> FrozenBlockLength:
    """Estimate and freeze the block length from DEVELOPMENT net-R only.

    The stationary-bootstrap estimate (``arch_optimal_block_length``'s
    ``stationary`` column) is used for both the primary Stationary Bootstrap
    and, mechanically related, the secondary Circular Block Bootstrap (task
    §6: "using a mechanically related/frozen block length") -- one frozen
    integer, not two independently tuned ones.
    """

    if len(trades) < 2:
        raise BootstrapValidationError(
            "block-length estimation requires at least two DEVELOPMENT trades"
        )
    series = pd.Series(
        [float(trade.net_r) for trade in trades],
        name="usdcad_sweep_bos_retest_v1_development_net_r",
    )
    result = estimate_optimal_block_length(series)
    frozen_block_size = max(1, round(result.stationary))
    return FrozenBlockLength(
        stationary_block_length=result.stationary,
        circular_block_length=result.circular,
        frozen_block_size=frozen_block_size,
        observation_count=result.observation_count,
        input_content_sha256=result.input_content_sha256,
    )


def draw_index_path(
    trade_count: int,
    *,
    method: ResamplingMethod,
    block_size: int,
    seed: int,
    min_length: int,
) -> tuple[int, ...]:
    """Draw a deterministic, seeded resampled index path of at least
    ``min_length`` trade indices into the original DEVELOPMENT sequence.

    One seeded bootstrap object is drawn from repeatedly (never
    re-instantiated mid-path) so consecutive draws advance the same
    deterministic RNG state; the concatenated path is truncated to exactly
    ``min_length``. ``method='iid_diagnostic'`` (single-trade shuffle with
    replacement, i.e. a stationary bootstrap with ``block_size=1``) may only
    be used as a diagnostic baseline, never as the primary probability
    estimate (task §6).
    """

    if trade_count <= 0:
        raise BootstrapValidationError("trade_count must be positive")
    if min_length <= 0:
        raise BootstrapValidationError("min_length must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise BootstrapValidationError("seed must be an explicit integer")

    index_array = np.arange(trade_count)
    bootstrap: StationaryBootstrap | CircularBlockBootstrap
    if method == "iid_diagnostic":
        bootstrap = StationaryBootstrap(1, index_array, seed=seed)
    elif method == "stationary":
        _require_positive_block_size(block_size)
        bootstrap = StationaryBootstrap(block_size, index_array, seed=seed)
    elif method == "circular":
        _require_positive_block_size(block_size)
        bootstrap = CircularBlockBootstrap(block_size, index_array, seed=seed)
    else:
        raise BootstrapValidationError(f"unsupported resampling method: {method!r}")

    drawn: list[int] = []
    reps_needed = -(-min_length // trade_count)  # ceil division
    for positional, _keyword in bootstrap.bootstrap(reps_needed):
        (resampled,) = positional
        drawn.extend(int(value) for value in resampled)
        if len(drawn) >= min_length:
            break
    if len(drawn) < min_length:
        raise RuntimeError("arch bootstrap did not yield enough resampled trades")
    return tuple(drawn[:min_length])


def _require_positive_block_size(block_size: int) -> None:
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size <= 0
    ):
        raise BootstrapValidationError("block_size must be a positive integer")
