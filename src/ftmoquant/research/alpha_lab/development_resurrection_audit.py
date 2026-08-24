"""Read-only forensic re-classification of historical DEVELOPMENT screens.

This module contains ONLY the pure, deterministic classification rule
(:func:`classify_forensic`) used by the repo-wide DEVELOPMENT resurrection
audit. It performs no data loading, no backtests, no Monte Carlo, and never
reads VALIDATION or holdout artifacts -- those responsibilities belong to
the separate, one-off audit script that calls this function per candidate.

The classification is purely additive: it never overrides a study's own
original preregistered pass/fail decision (``original_preregistered_pass``
is read, not recomputed), and it distinguishes ECONOMIC failure (dead on
arrival) from POWER/ROBUSTNESS failure (plausibly a false negative) per the
frozen forensic rules below.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

#: Forensic economic bar, independent of whatever PF threshold the original
#: study preregistered -- applied uniformly across all families being
#: re-audited so different studies are judged on a consistent economic bar.
FORENSIC_PROFIT_FACTOR_GT = 1.10
FORENSIC_PROFIT_FACTOR_DEAD_LE = 1.00

#: Non-economic "budget" gates a CREDIBLE_NEAR_MISS is allowed to fail (at
#: most two of these, per the frozen forensic rule). Concentration/winner-
#: dependency gates are NOT in this pool -- they are a separate hard gate.
_POOL_GATE_NAMES = ("trade_count", "fold_count", "connected_region")


class ForensicClass(StrEnum):
    ROBUST_SURVIVOR = "ROBUST_SURVIVOR"
    CREDIBLE_NEAR_MISS = "CREDIBLE_NEAR_MISS"
    UNDERPOWERED_NEAR_MISS = "UNDERPOWERED_NEAR_MISS"
    ECONOMICALLY_DEAD = "ECONOMICALLY_DEAD"
    NOT_AUDITABLE = "NOT_AUDITABLE_FROM_EXISTING_ARTIFACTS"
    VALIDATION_EXPOSED = "VALIDATION_EXPOSED"


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    """Whatever comparable metrics an existing DEVELOPMENT artifact
    happened to record for one candidate/config. Every field is Optional
    except the two exposure/decision flags -- absence means "not computed
    by that study", never "assumed zero/failing"."""

    validation_exposed: bool = False
    original_preregistered_pass: bool | None = None

    trade_count: int | None = None
    min_trade_count_requirement: int | None = None

    native_expectancy: float | None = None
    native_profit_factor: float | None = None
    native_net_return: float | None = None
    native_sharpe: float | None = None

    stress_expectancy: float | None = None
    stress_net_return: float | None = None
    stress_profit_factor: float | None = None
    stress_label: str | None = None

    positive_fold_count: int | None = None
    fold_requirement: int | None = None

    best_5pct_removed_expectancy: float | None = None
    quarter_concentration: float | None = None
    quarter_concentration_limit: float | None = None
    largest_trade_share: float | None = None

    connected_region_size: int | None = None
    connected_region_requirement: int | None = None


@dataclass(frozen=True, slots=True)
class ForensicResult:
    forensic_class: ForensicClass
    reason: str


def _economic_value(evidence: CandidateEvidence) -> float | None:
    if evidence.native_expectancy is not None:
        return evidence.native_expectancy
    return evidence.native_net_return


def _stress_value(evidence: CandidateEvidence) -> float | None:
    if evidence.stress_expectancy is not None:
        return evidence.stress_expectancy
    return evidence.stress_net_return


def classify_forensic(evidence: CandidateEvidence) -> ForensicResult:
    """Deterministic, pure classification. Never fabricates a value: any
    criterion whose inputs are absent is simply skipped ("where available"),
    never treated as a pass or a fail."""

    if evidence.validation_exposed:
        return ForensicResult(
            ForensicClass.VALIDATION_EXPOSED,
            "This exact candidate has already been evaluated on VALIDATION "
            "-- not resurrection-eligible under any circumstances.",
        )

    if evidence.original_preregistered_pass is True:
        return ForensicResult(
            ForensicClass.ROBUST_SURVIVOR,
            "Passed its original preregistered family-survival rule.",
        )

    econ_value = _economic_value(evidence)
    if econ_value is None:
        return ForensicResult(
            ForensicClass.NOT_AUDITABLE,
            "Neither native expectancy nor native net return is present in "
            "the existing DEVELOPMENT artifact -- no economic signal to "
            "classify from without a rerun.",
        )

    if econ_value <= 0:
        return ForensicResult(
            ForensicClass.ECONOMICALLY_DEAD,
            f"Native expectancy/return is non-positive ({econ_value!r}).",
        )

    if (
        evidence.native_profit_factor is not None
        and evidence.native_profit_factor <= FORENSIC_PROFIT_FACTOR_DEAD_LE
    ):
        return ForensicResult(
            ForensicClass.ECONOMICALLY_DEAD,
            f"Native profit factor {evidence.native_profit_factor!r} <= "
            f"{FORENSIC_PROFIT_FACTOR_DEAD_LE} -- broadly unprofitable.",
        )

    if (
        evidence.native_profit_factor is not None
        and evidence.native_profit_factor <= FORENSIC_PROFIT_FACTOR_GT
    ):
        # Positive but marginal (1.00 < PF <= 1.10): the forensic rule
        # requires PF > 1.10 as a hard economic criterion for near-miss
        # eligibility (rule 2) -- this is not the "broadly unprofitable"
        # case above, but it still fails to clear the near-miss economic
        # bar, so it is not eligible to be escalated as a false negative.
        return ForensicResult(
            ForensicClass.ECONOMICALLY_DEAD,
            f"Native profit factor {evidence.native_profit_factor!r} is "
            f"positive but does not clear the forensic near-miss bar of "
            f"PF > {FORENSIC_PROFIT_FACTOR_GT} -- marginal economics.",
        )

    stress_value = _stress_value(evidence)
    if stress_value is not None and stress_value <= 0:
        label = evidence.stress_label or "the strongest computed stress"
        return ForensicResult(
            ForensicClass.ECONOMICALLY_DEAD,
            f"Fails to survive {label}: stressed expectancy/return "
            f"{stress_value!r} <= 0.",
        )

    concentration_failures: list[str] = []
    if (
        evidence.best_5pct_removed_expectancy is not None
        and evidence.best_5pct_removed_expectancy <= 0
    ):
        concentration_failures.append(
            "best-5%-removed expectancy <= 0 (a handful of exceptional "
            "winners account for all apparent profitability)"
        )
    if (
        evidence.quarter_concentration is not None
        and evidence.quarter_concentration_limit is not None
        and evidence.quarter_concentration > evidence.quarter_concentration_limit
    ):
        concentration_failures.append(
            f"quarter concentration {evidence.quarter_concentration!r} "
            f"exceeds limit {evidence.quarter_concentration_limit!r}"
        )
    if concentration_failures:
        return ForensicResult(
            ForensicClass.ECONOMICALLY_DEAD,
            "Fails winner-dependency/concentration diagnostics: "
            + "; ".join(concentration_failures),
        )

    pool_gate_results: dict[str, bool] = {}
    if (
        evidence.trade_count is not None
        and evidence.min_trade_count_requirement is not None
    ):
        pool_gate_results["trade_count"] = (
            evidence.trade_count >= evidence.min_trade_count_requirement
        )
    if (
        evidence.positive_fold_count is not None
        and evidence.fold_requirement is not None
    ):
        pool_gate_results["fold_count"] = (
            evidence.positive_fold_count >= evidence.fold_requirement
        )
    if (
        evidence.connected_region_size is not None
        and evidence.connected_region_requirement is not None
    ):
        pool_gate_results["connected_region"] = (
            evidence.connected_region_size >= evidence.connected_region_requirement
        )

    failed_pool_gates = [
        name for name in _POOL_GATE_NAMES if pool_gate_results.get(name) is False
    ]

    if not failed_pool_gates:
        if evidence.original_preregistered_pass is False:
            return ForensicResult(
                ForensicClass.NOT_AUDITABLE,
                "Every economic/cost/concentration/power/robustness proxy "
                "available in the DEVELOPMENT artifact passes, yet the "
                "original screen recorded a fail -- the artifact does not "
                "capture the actual original failure reason, so this "
                "cannot be forensically resolved without a rerun.",
            )
        return ForensicResult(
            ForensicClass.NOT_AUDITABLE,
            "Economics and concentration diagnostics pass on every "
            "available metric, but no trade-count/fold-count/connected-"
            "region data is present to confirm a near-miss classification.",
        )

    if len(failed_pool_gates) > 2:
        return ForensicResult(
            ForensicClass.ECONOMICALLY_DEAD,
            "Clean economics, but exceeds the forensic near-miss budget: "
            f"fails {len(failed_pool_gates)} non-economic robustness/power "
            f"gates ({', '.join(failed_pool_gates)}) -- the original "
            "rejection stands as the more defensible read; likely curve-fit "
            "noise rather than a genuine false negative.",
        )

    if failed_pool_gates == ["trade_count"]:
        return ForensicResult(
            ForensicClass.UNDERPOWERED_NEAR_MISS,
            "Passes every available economic, cost-stress, and "
            "concentration gate; fails ONLY the minimum-trade-count "
            "requirement -- plausibly a statistical-power/sample-size "
            "false negative rather than a genuine economic failure.",
        )

    return ForensicResult(
        ForensicClass.CREDIBLE_NEAR_MISS,
        "Passes every available economic, cost-stress, and concentration "
        f"gate; fails at most two non-economic robustness/power gates "
        f"({', '.join(failed_pool_gates)}) -- plausibly power- or "
        "temporal-threshold-related rather than basic negative economics.",
    )
