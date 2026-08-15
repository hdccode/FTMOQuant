"""Preregistered G1.4B candidates and selection policy, with no result fields."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from ftmoquant.research.leo_gbpusd_spec import (
    LEO_GBPUSD_CONFIG_SHA256,
    LEO_GBPUSD_SPEC_PATH,
)
from ftmoquant.research.liquidity_shock_reversion_spec import (
    LIQUIDITY_SHOCK_REVERSION_CONFIG_SHA256,
    LIQUIDITY_SHOCK_REVERSION_SPEC_PATH,
)
from ftmoquant.research.session_range_expansion_spec import (
    SESSION_RANGE_EXPANSION_CONFIG_SHA256,
    SESSION_RANGE_EXPANSION_SPEC_PATH,
)
from ftmoquant.research.stage_g import FROZEN_INSTRUMENT_IDS
from ftmoquant.research.ts_momentum_spec import (
    TS_MOMENTUM_CONFIG_SHA256,
    TS_MOMENTUM_SPEC_PATH,
)

TOURNAMENT_REGISTRY_VERSION = "g1.4e-candidate-registry-5"
SELECTION_CONTRACT_VERSION = "g1.4b-selection-contract-1"


class CandidateEligibility(StrEnum):
    ELIGIBLE_FOR_IMPLEMENTATION = "eligible_for_implementation"
    BLOCKED_PREREQUISITES = "blocked_prerequisites"


class CandidateImplementationStatus(StrEnum):
    IMPLEMENTED_NOT_EVALUATED = "implemented_not_evaluated"
    DEVELOPMENT_FAILED_RETIRED = "development_failed_retired"
    SPECIFIED_NOT_IMPLEMENTED = "specified_not_implemented"
    BLOCKED_NOT_IMPLEMENTED = "blocked_not_implemented"


@dataclass(frozen=True, slots=True)
class EligibilityEnvironment:
    instrument_count: int
    pit_inputs: frozenset[str]


@dataclass(frozen=True, slots=True)
class CandidateDefinition:
    candidate_id: str
    family: str
    prerequisites: tuple[str, ...]
    requires_cross_sectional_ranking: bool
    minimum_cross_sectional_instruments: int | None


@dataclass(frozen=True, slots=True)
class CandidateRegistryEntry:
    candidate_id: str
    family: str
    eligibility: CandidateEligibility
    implementation_status: CandidateImplementationStatus
    prerequisites: tuple[str, ...]
    unmet_prerequisites: tuple[str, ...]
    spec_path: str | None
    strategy_config_sha256: str | None
    strategy_logic_present: bool = False
    results_accessed: bool = False


@dataclass(frozen=True, slots=True)
class CandidateRegistry:
    version: str
    ordered_entries: tuple[CandidateRegistryEntry, ...]
    semantic_sha256: str


_CANDIDATES = (
    CandidateDefinition(
        "ts_momentum_v1",
        "time_series",
        ("synchronized_development_clock", "cost_profile", "exposure_limits"),
        False,
        None,
    ),
    CandidateDefinition(
        "carry_momentum_v1",
        "time_series_carry",
        (
            "synchronized_development_clock",
            "cost_profile",
            "exposure_limits",
            "pit_carry_inputs",
        ),
        False,
        None,
    ),
    CandidateDefinition(
        "carry_momentum_value_v1",
        "cross_sectional_factor",
        (
            "synchronized_development_clock",
            "cost_profile",
            "exposure_limits",
            "pit_carry_inputs",
            "pit_value_inputs",
        ),
        True,
        4,
    ),
    CandidateDefinition(
        "session_range_expansion_v1",
        "session",
        ("synchronized_development_clock", "cost_profile", "exposure_limits"),
        False,
        None,
    ),
    CandidateDefinition(
        "liquidity_shock_reversion_v1",
        "time_series_liquidity",
        ("synchronized_development_clock", "cost_profile", "exposure_limits"),
        False,
        None,
    ),
    CandidateDefinition(
        "leo_gbpusd_v1",
        "session",
        ("synchronized_development_clock", "cost_profile", "exposure_limits"),
        False,
        None,
    ),
    CandidateDefinition(
        "session_regime_hybrid_v1",
        "session",
        ("synchronized_development_clock", "cost_profile", "exposure_limits"),
        False,
        None,
    ),
)

_INFRASTRUCTURE_PREREQUISITES = frozenset(
    {"synchronized_development_clock", "cost_profile", "exposure_limits"}
)


def candidate_registry(
    environment: EligibilityEnvironment | None = None,
) -> CandidateRegistry:
    """Evaluate only prerequisites; never import or inspect candidate results."""

    current = environment or EligibilityEnvironment(
        instrument_count=len(FROZEN_INSTRUMENT_IDS), pit_inputs=frozenset()
    )
    if current.instrument_count <= 0:
        raise ValueError("instrument_count must be positive")
    available = _INFRASTRUCTURE_PREREQUISITES | current.pit_inputs
    entries: list[CandidateRegistryEntry] = []
    for candidate in _CANDIDATES:
        unmet = [item for item in candidate.prerequisites if item not in available]
        if (
            candidate.requires_cross_sectional_ranking
            and candidate.minimum_cross_sectional_instruments is not None
            and current.instrument_count < candidate.minimum_cross_sectional_instruments
        ):
            unmet.append(
                "minimum_cross_sectional_instruments="
                f"{candidate.minimum_cross_sectional_instruments}"
            )
        eligibility = (
            CandidateEligibility.ELIGIBLE_FOR_IMPLEMENTATION
            if not unmet
            else CandidateEligibility.BLOCKED_PREREQUISITES
        )
        implemented = {
            "ts_momentum_v1": (
                TS_MOMENTUM_SPEC_PATH.as_posix(),
                TS_MOMENTUM_CONFIG_SHA256,
            ),
            "session_range_expansion_v1": (
                SESSION_RANGE_EXPANSION_SPEC_PATH.as_posix(),
                SESSION_RANGE_EXPANSION_CONFIG_SHA256,
            ),
            "liquidity_shock_reversion_v1": (
                LIQUIDITY_SHOCK_REVERSION_SPEC_PATH.as_posix(),
                LIQUIDITY_SHOCK_REVERSION_CONFIG_SHA256,
            ),
            "leo_gbpusd_v1": (
                LEO_GBPUSD_SPEC_PATH.as_posix(),
                LEO_GBPUSD_CONFIG_SHA256,
            ),
        }
        implementation = implemented.get(candidate.candidate_id)
        is_implemented = implementation is not None
        entries.append(
            CandidateRegistryEntry(
                candidate_id=candidate.candidate_id,
                family=candidate.family,
                eligibility=eligibility,
                implementation_status=(
                    CandidateImplementationStatus.DEVELOPMENT_FAILED_RETIRED
                    if candidate.candidate_id
                    in {
                        "session_range_expansion_v1",
                        "liquidity_shock_reversion_v1",
                    }
                    else CandidateImplementationStatus.IMPLEMENTED_NOT_EVALUATED
                    if is_implemented
                    else CandidateImplementationStatus.SPECIFIED_NOT_IMPLEMENTED
                    if eligibility is CandidateEligibility.ELIGIBLE_FOR_IMPLEMENTATION
                    else CandidateImplementationStatus.BLOCKED_NOT_IMPLEMENTED
                ),
                prerequisites=candidate.prerequisites,
                unmet_prerequisites=tuple(unmet),
                spec_path=(implementation[0] if implementation is not None else None),
                strategy_config_sha256=(
                    implementation[1] if implementation is not None else None
                ),
                strategy_logic_present=is_implemented,
                results_accessed=(
                    candidate.candidate_id
                    in {
                        "session_range_expansion_v1",
                        "liquidity_shock_reversion_v1",
                    }
                ),
            )
        )
    payload = {
        "version": TOURNAMENT_REGISTRY_VERSION,
        "ordered_entries": [_jsonable(asdict(item)) for item in entries],
    }
    return CandidateRegistry(
        version=TOURNAMENT_REGISTRY_VERSION,
        ordered_entries=tuple(entries),
        semantic_sha256=_hash(payload),
    )


@dataclass(frozen=True, slots=True)
class SelectionContract:
    """Frozen comparison policy to apply only after candidate implementation."""

    version: str
    comparison_unit: str
    primary_metrics: tuple[str, ...]
    robustness_metrics: tuple[str, ...]
    fold_aggregation: tuple[str, ...]
    exclusion_and_failure_semantics: tuple[str, ...]
    resampling_configuration: tuple[str, ...]
    multiple_testing_policy: tuple[str, ...]
    deterministic_tie_breaking: tuple[str, ...]
    advancement_rule: tuple[str, ...]
    composite_rule: tuple[str, ...]
    semantic_sha256: str


def preregistered_selection_contract() -> SelectionContract:
    """Return the immutable, result-independent G1.4B comparison contract."""

    values: dict[str, Any] = {
        "version": SELECTION_CONTRACT_VERSION,
        "comparison_unit": (
            "causally aligned Dukascopy session-day net return ending 17:00 "
            "America/New_York after observed spread and native commission, "
            "slippage, and rollover, normalized to fixed research exposure"
        ),
        "primary_metrics": (
            "median development-fold annualized net Sharpe using zero risk-free "
            "rate and sqrt(252)",
            "pooled development mean daily net return",
        ),
        "robustness_metrics": (
            "worst-fold annualized net Sharpe",
            "positive-fold count",
            "maximum drawdown",
            "turnover",
            "positive mean net return after a fixed 1.5x realized-cost stress",
            "stationary-bootstrap mean-return lower confidence bound",
            "currency-exposure-limit breach count",
        ),
        "fold_aggregation": (
            "compute every metric separately on each frozen comparison fold",
            "rank primary Sharpe by the median of the three fold values",
            "pool daily observations only for preregistered SPA/MCS inference",
            "do not reweight, drop, or redefine folds after inspecting results",
        ),
        "exclusion_and_failure_semantics": (
            "blocked candidates never enter execution or the multiplicity family",
            "an eligible candidate with any failed/missing fold is recorded failed "
            "and cannot advance",
            "non-finite metrics, causal violations, or exposure-limit breaches are "
            "hard failures and cannot be imputed",
            "every failure remains visible in the tournament report",
        ),
        "resampling_configuration": (
            "daily loss block_size=20 observations",
            "repetitions=10000",
            "seed=14042026",
            "significance_level=0.05",
            "SPA bootstrap=stationary, studentize=true, nested=false, "
            "pvalue_type=consistent",
            "MCS bootstrap=stationary, method=R, test_size=0.05",
            "stationary-bootstrap mean CI confidence_level=0.95, method=basic, "
            "tail=two, sampling=nonparametric",
        ),
        "multiple_testing_policy": (
            "freeze the implemented eligible candidate family before reading returns",
            "use arch 8.0.0 SPA with consistent p-value at alpha=0.05 against the "
            "zero-net-return benchmark on aligned daily losses",
            "use arch 8.0.0 MCS at size=0.05 as a robustness set, not a winner rule",
            "use the frozen stationary-bootstrap configuration for mean-return "
            "confidence bounds",
            "do not substitute an estimated block length without a new preregistration",
        ),
        "deterministic_tie_breaking": (
            "higher median fold net Sharpe",
            "higher worst-fold net Sharpe",
            "lower turnover",
            "lexicographically smaller candidate_id",
        ),
        "advancement_rule": (
            "advance at most one candidate that has all folds complete and no "
            "hard failure",
            "require positive median and worst-fold net Sharpe",
            "require stationary-bootstrap mean-return lower bound above zero",
            "require SPA consistent p-value <= 0.05 and MCS inclusion",
            "if none qualifies, advance no candidate to validation",
            "validation remains locked until an independent validation-runner handoff",
        ),
        "composite_rule": (
            "a composite is permitted only if its constituents and fixed weights were "
            "preregistered before constituent returns were inspected",
            "the composite must independently satisfy every advancement gate",
            "no post-hoc ensemble, weight fitting, or validation-informed composite",
        ),
    }
    payload = {key: _jsonable(value) for key, value in values.items()}
    return SelectionContract(
        **values,
        semantic_sha256=_hash(payload),
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
