"""B3F1 underpowered-candidate resolution protocol (Underpowered Candidate
Resolution v1).

Pure, deterministic diagnostics only -- no data loading, no VALIDATION or
holdout access, no rerun logic. The two exact candidates this protocol may
ever be applied to are frozen in :data:`FROZEN_CANDIDATES`; nothing else is
eligible. This module NEVER overwrites B3F1's original preregistered
family-survival result (which remains FAILED) -- it only computes a
separate, additional classification
(:data:`UNDERPOWERED_CONFIRMATION_ELIGIBLE`/:data:`UNDERPOWERED_REJECTED`)
for the two named candidates.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from arch.bootstrap import IIDBootstrap, StationaryBootstrap

UNDERPOWERED_CONFIRMATION_ELIGIBLE = "UNDERPOWERED_CONFIRMATION_ELIGIBLE"
UNDERPOWERED_REJECTED = "UNDERPOWERED_REJECTED"

#: The two, and only two, exact candidates this protocol may ever be
#: applied to (section 1 of the frozen task brief). No other B3F1
#: configuration is eligible.
FROZEN_CANDIDATES: dict[str, dict[str, object]] = {
    "U1": {
        "sleeve_id": "USD/CHF.OANDA__USD/JPY.OANDA",
        "formation_window": 240,
        "z_entry": Decimal("1.5"),
        "z_stop": Decimal("3.0"),
    },
    "U2": {
        "sleeve_id": "USD/CAD.OANDA__USD/CHF.OANDA",
        "formation_window": 240,
        "z_entry": Decimal("1.5"),
        "z_stop": Decimal("3.5"),
    },
}

#: Eligibility rule thresholds (section 9), frozen BEFORE computing any new
#: statistics -- never adjusted after seeing results.
ORIGINAL_EXPECTANCY_GT = Decimal(0)
ORIGINAL_PF_GT = Decimal("1.10")
ORIGINAL_FOLD_REQUIREMENT = 3
ORIGINAL_FOLD_TOTAL = 4
ORIGINAL_QUARTER_CONCENTRATION_LE = Decimal("0.40")
BOOTSTRAP_P_NATIVE_EXPECTANCY_GT_0_GE = 0.80
BOOTSTRAP_P_STRESSED_EXPECTANCY_GT_0_GE = 0.75
LEAVE_ONE_OUT_POSITIVE_FRACTION_GE = 0.90


#: Mechanically justified frozen block-size fallback (Politis & White's
#: standard n^(1/3) asymptotic rate for block bootstraps), used ONLY if
#: arch's own optimal-block-length estimate is unstable (non-finite,
#: non-positive, or implausibly large relative to n) at n<=40. Never tuned
#: to this candidate's own results.
def frozen_fallback_block_size(n: int) -> int:
    return max(1, int(round(n ** (1.0 / 3.0))))


BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 14042026  # matches the seed already used elsewhere in this repo


class B3F1ResolutionError(ValueError):
    """Raised on any violation of the frozen underpowered-resolution contract."""


def require_frozen_candidate(label: str) -> dict[str, object]:
    """Fail closed if ``label`` is not exactly ``U1`` or ``U2``."""

    if label not in FROZEN_CANDIDATES:
        frozen = tuple(FROZEN_CANDIDATES)
        raise B3F1ResolutionError(
            f"{label!r} is not one of the two frozen candidates {frozen} -- "
            "no other B3F1 configuration is eligible for this protocol"
        )
    return FROZEN_CANDIDATES[label]


# ---------------------------------------------------------------------------
# Leave-one-out influence analysis (section 6)
# ---------------------------------------------------------------------------


def expectancy_and_profit_factor(pnls: Sequence[Decimal]) -> tuple[Decimal, Decimal]:
    if not pnls:
        raise B3F1ResolutionError(
            "expectancy/profit-factor requires at least one trade"
        )
    expectancy = sum(pnls, Decimal(0)) / len(pnls)
    gross_profit = sum((p for p in pnls if p > 0), Decimal(0))
    gross_loss = sum((-p for p in pnls if p < 0), Decimal(0))
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = Decimal("Infinity")
    else:
        profit_factor = Decimal(0)
    return expectancy, profit_factor


@dataclass(frozen=True, slots=True)
class LeaveOneOutSummary:
    minimum_expectancy: Decimal
    minimum_profit_factor: Decimal
    fraction_expectancy_positive: float
    fraction_profit_factor_gt_1: float
    per_trade_expectancy: tuple[Decimal, ...]
    per_trade_profit_factor: tuple[Decimal, ...]


def leave_one_out(pnls: Sequence[Decimal]) -> LeaveOneOutSummary:
    """Remove exactly one trade at a time and recompute expectancy/PF on
    the remaining ``n-1`` trades. Diagnostic only -- never used to tune a
    parameter or exclude a trade from the frozen trade history itself."""

    if len(pnls) < 2:
        raise B3F1ResolutionError("leave-one-out requires at least two trades")
    expectancies: list[Decimal] = []
    profit_factors: list[Decimal] = []
    for i in range(len(pnls)):
        remaining = [p for j, p in enumerate(pnls) if j != i]
        expectancy, pf = expectancy_and_profit_factor(remaining)
        expectancies.append(expectancy)
        profit_factors.append(pf)
    finite_pfs = [pf for pf in profit_factors if pf.is_finite()]
    return LeaveOneOutSummary(
        minimum_expectancy=min(expectancies),
        minimum_profit_factor=min(finite_pfs) if finite_pfs else Decimal("Infinity"),
        fraction_expectancy_positive=sum(1 for e in expectancies if e > 0)
        / len(expectancies),
        fraction_profit_factor_gt_1=sum(1 for pf in profit_factors if pf > 1)
        / len(profit_factors),
        per_trade_expectancy=tuple(expectancies),
        per_trade_profit_factor=tuple(profit_factors),
    )


# ---------------------------------------------------------------------------
# Winner-dependence diagnostics (section 7) -- diagnostic only, no new gates
# ---------------------------------------------------------------------------


def remove_n_best(pnls: Sequence[Decimal], n: int) -> tuple[Decimal, ...]:
    ranked = sorted(pnls, reverse=True)
    return tuple(ranked[n:])


def remove_n_worst(pnls: Sequence[Decimal], n: int) -> tuple[Decimal, ...]:
    ranked = sorted(pnls)
    return tuple(ranked[n:])


def winner_dependence_diagnostics(pnls: Sequence[Decimal]) -> dict[str, Decimal]:
    result = {}
    for n in (1, 2):
        remaining = remove_n_best(pnls, n)
        expectancy, _ = (
            expectancy_and_profit_factor(remaining)
            if remaining
            else (
                Decimal(0),
                Decimal(0),
            )
        )
        result[f"remove_best_{n}_expectancy"] = expectancy
    for n in (1, 2):
        remaining = remove_n_worst(pnls, n)
        expectancy, _ = (
            expectancy_and_profit_factor(remaining)
            if remaining
            else (
                Decimal(0),
                Decimal(0),
            )
        )
        result[f"remove_worst_{n}_expectancy"] = expectancy
    return result


# ---------------------------------------------------------------------------
# Bootstrap uncertainty (section 5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BootstrapDiagnostics:
    bootstrap_kind: str
    block_size: int | None
    block_size_source: str
    observation_count: int
    mean_estimate: float
    ci_lower: float
    ci_upper: float
    p_expectancy_gt_0: float
    p_pf_gt_1: float
    p_stressed_expectancy_gt_0: float | None


def _pf_statistic(values: np.ndarray) -> np.ndarray:
    gross_profit = values[values > 0].sum()
    gross_loss = -values[values < 0].sum()
    if gross_loss > 0:
        pf = gross_profit / gross_loss
    elif gross_profit > 0:
        pf = np.inf
    else:
        pf = 0.0
    return np.asarray([pf])


def _mean_statistic(values: np.ndarray) -> np.ndarray:
    return np.asarray([values.mean()])


def compute_bootstrap_diagnostics(
    native_pnls: np.ndarray,
    stressed_pnls: np.ndarray | None,
    *,
    kind: str,
    block_size: int | None,
    block_size_source: str,
    reps: int = BOOTSTRAP_REPETITIONS,
    seed: int = BOOTSTRAP_SEED,
) -> BootstrapDiagnostics:
    """Compute bootstrap mean CI + empirical P(statistic > threshold) for
    one series, using the same ``arch.bootstrap`` engine already used
    elsewhere in this repo (:mod:`ftmoquant.research.statistics`).

    ``kind`` is ``"stationary"`` (primary) or ``"iid"`` (diagnostic only,
    per section 5). For ``"iid"``, ``block_size``/``block_size_source`` are
    ignored (IID resampling has no block structure).
    """

    n = len(native_pnls)
    bootstrap: StationaryBootstrap | IIDBootstrap
    if kind == "stationary":
        if block_size is None or block_size <= 0:
            raise B3F1ResolutionError(
                "stationary bootstrap requires a positive block_size"
            )
        bootstrap = StationaryBootstrap(block_size, native_pnls, seed=seed)
    elif kind == "iid":
        bootstrap = IIDBootstrap(native_pnls, seed=seed)
    else:
        raise B3F1ResolutionError(f"unknown bootstrap kind {kind!r}")

    mean_replicates = bootstrap.apply(_mean_statistic, reps).reshape(-1)
    pf_replicates = bootstrap.apply(_pf_statistic, reps).reshape(-1)

    ci_lower = float(np.quantile(mean_replicates, 0.025))
    ci_upper = float(np.quantile(mean_replicates, 0.975))
    p_expectancy_gt_0 = float(np.mean(mean_replicates > 0))
    p_pf_gt_1 = float(np.mean(pf_replicates > 1))

    p_stressed_gt_0: float | None = None
    if stressed_pnls is not None:
        stressed_bootstrap: StationaryBootstrap | IIDBootstrap
        if kind == "stationary":
            assert block_size is not None
            stressed_bootstrap = StationaryBootstrap(
                block_size, stressed_pnls, seed=seed
            )
        else:
            stressed_bootstrap = IIDBootstrap(stressed_pnls, seed=seed)
        stressed_mean_replicates = stressed_bootstrap.apply(
            _mean_statistic, reps
        ).reshape(-1)
        p_stressed_gt_0 = float(np.mean(stressed_mean_replicates > 0))

    return BootstrapDiagnostics(
        bootstrap_kind=kind,
        block_size=block_size if kind == "stationary" else None,
        block_size_source=block_size_source
        if kind == "stationary"
        else "not_applicable",
        observation_count=n,
        mean_estimate=float(native_pnls.mean()),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_expectancy_gt_0=p_expectancy_gt_0,
        p_pf_gt_1=p_pf_gt_1,
        p_stressed_expectancy_gt_0=p_stressed_gt_0,
    )


def resolve_block_size(estimated_stationary: float, n: int) -> tuple[int, str]:
    """Section 5: use arch's own ``optimal_block_length`` estimate if it is
    stable (finite, positive, and no larger than n/2); otherwise fall back
    to the frozen, mechanically justified ``n^(1/3)`` rule -- never a
    block length tuned by looking at this candidate's own bootstrap
    output."""

    if np.isfinite(estimated_stationary) and 0 < estimated_stationary <= n / 2:
        return max(1, round(estimated_stationary)), "arch_optimal_block_length"
    return frozen_fallback_block_size(n), "frozen_fallback_n_cbrt"


# ---------------------------------------------------------------------------
# Fold diagnostics (section 8) -- read-only, frozen fold boundaries
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FoldDiagnostic:
    fold_index: int
    trade_count: int
    expectancy: Decimal
    profit_factor: Decimal
    net_pnl: Decimal


def fold_diagnostics(
    exit_timestamps: Sequence[pd.Timestamp],
    pnls: Sequence[Decimal],
    fold_boundaries: Sequence[pd.Timestamp],
) -> tuple[FoldDiagnostic, ...]:
    if len(exit_timestamps) != len(pnls):
        raise B3F1ResolutionError("exit_timestamps and pnls must be the same length")
    if len(fold_boundaries) < 2:
        raise B3F1ResolutionError("fold_boundaries must contain at least 2 edges")
    results = []
    for index, (start, end) in enumerate(
        zip(fold_boundaries[:-1], fold_boundaries[1:], strict=False)
    ):
        fold_pnls = [
            pnl
            for ts, pnl in zip(exit_timestamps, pnls, strict=True)
            if start <= ts < end
        ]
        if fold_pnls:
            expectancy, pf = expectancy_and_profit_factor(fold_pnls)
            net_pnl = sum(fold_pnls, Decimal(0))
        else:
            expectancy, pf, net_pnl = Decimal(0), Decimal(0), Decimal(0)
        results.append(
            FoldDiagnostic(
                fold_index=index,
                trade_count=len(fold_pnls),
                expectancy=expectancy,
                profit_factor=pf,
                net_pnl=net_pnl,
            )
        )
    return tuple(results)


# ---------------------------------------------------------------------------
# Mechanical eligibility rule (section 9) -- frozen BEFORE computing results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EligibilityInputs:
    original_failure_reason_is_trade_count: bool
    native_expectancy: Decimal
    native_profit_factor: Decimal
    stressed_1_5x_expectancy: Decimal
    positive_fold_count: int
    best_5pct_removed_expectancy: Decimal
    quarter_concentration: Decimal | None
    bootstrap_p_native_expectancy_gt_0: float
    bootstrap_p_stressed_expectancy_gt_0: float
    leave_one_out_fraction_positive: float


@dataclass(frozen=True, slots=True)
class EligibilityVerdict:
    verdict: str
    failed_criteria: tuple[str, ...]


def evaluate_eligibility(inputs: EligibilityInputs) -> EligibilityVerdict:
    p_native = inputs.bootstrap_p_native_expectancy_gt_0
    p_stressed = inputs.bootstrap_p_stressed_expectancy_gt_0
    loo_fraction = inputs.leave_one_out_fraction_positive
    checks: dict[str, bool] = {
        "1_original_failure_is_trade_count": (
            inputs.original_failure_reason_is_trade_count
        ),
        "2_native_expectancy_gt_0": inputs.native_expectancy > ORIGINAL_EXPECTANCY_GT,
        "3_native_pf_gt_1_10": inputs.native_profit_factor > ORIGINAL_PF_GT,
        "4_stressed_expectancy_gt_0": (
            inputs.stressed_1_5x_expectancy > ORIGINAL_EXPECTANCY_GT
        ),
        "5_folds_ge_3_of_4": inputs.positive_fold_count >= ORIGINAL_FOLD_REQUIREMENT,
        "6_best5pct_removed_expectancy_gt_0": (
            inputs.best_5pct_removed_expectancy > ORIGINAL_EXPECTANCY_GT
        ),
        "7_quarter_concentration_le_0_40": (
            inputs.quarter_concentration is not None
            and inputs.quarter_concentration <= ORIGINAL_QUARTER_CONCENTRATION_LE
        ),
        "8_bootstrap_p_native_expectancy_ge_0_80": (
            p_native >= BOOTSTRAP_P_NATIVE_EXPECTANCY_GT_0_GE
        ),
        "9_bootstrap_p_stressed_expectancy_ge_0_75": (
            p_stressed >= BOOTSTRAP_P_STRESSED_EXPECTANCY_GT_0_GE
        ),
        "10_leave_one_out_positive_fraction_ge_0_90": (
            loo_fraction >= LEAVE_ONE_OUT_POSITIVE_FRACTION_GE
        ),
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    verdict = UNDERPOWERED_REJECTED if failed else UNDERPOWERED_CONFIRMATION_ELIGIBLE
    return EligibilityVerdict(verdict=verdict, failed_criteria=failed)


# ---------------------------------------------------------------------------
# Mechanical winner selection (section 10) -- at most ONE candidate
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WinnerSelectionInputs:
    candidate_id: str
    bootstrap_p_stressed_expectancy_gt_0: float
    bootstrap_p_native_expectancy_gt_0: float
    minimum_leave_one_out_expectancy: Decimal
    stressed_profit_factor: Decimal


def select_winner(
    eligible_candidates: Sequence[WinnerSelectionInputs],
) -> str | None:
    """Section 10's exact tie-break ladder, applied only to candidates that
    already passed :func:`evaluate_eligibility`. Returns ``None`` if zero
    candidates are eligible; never both."""

    if not eligible_candidates:
        return None
    if len(eligible_candidates) == 1:
        return eligible_candidates[0].candidate_id
    ranked = sorted(
        eligible_candidates,
        key=lambda c: (
            -c.bootstrap_p_stressed_expectancy_gt_0,
            -c.bootstrap_p_native_expectancy_gt_0,
            -c.minimum_leave_one_out_expectancy,
            -c.stressed_profit_factor
            if c.stressed_profit_factor.is_finite()
            else Decimal("-Infinity"),
            c.candidate_id,
        ),
    )
    return ranked[0].candidate_id
