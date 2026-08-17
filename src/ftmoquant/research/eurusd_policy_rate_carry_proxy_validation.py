"""One-shot, candidate-only validation adapter for
``eurusd_policy_rate_carry_proxy_v1``.

This family is baseline-only: exactly one configuration (``sign(ECBDFR -
EFFR)``, causal 1% annualized-volatility sizing, proxy FX rollover carry, the
existing G0.7 execution engine). It has never used ``run_search``,
``select_candidate``, or ``assess_plateau`` -- there is no parameter grid, no
``selected_candidate.json``, no ``trial_registry.json`` for this family (see
``eurusd_policy_rate_carry_proxy_v1``'s own preregistration:
``parameter_family.mode == "baseline_only"``). This module therefore never
imports :mod:`ftmoquant.research.g1.search`, :mod:`ftmoquant.research.g1.selector`,
:mod:`ftmoquant.research.g1.plateau`, or ``TrialRegistry``/``run_search``-adjacent
machinery -- "one candidate only" is enforced structurally by the module's own
import list, not by a runtime check (see
``tests/research/test_eurusd_policy_rate_carry_proxy_validation.py::
test_module_source_never_imports_search_or_selector_machinery``).

Importing this module never opens validation market/rate data. The one
authorized real-data entry point,
:func:`run_eurusd_policy_rate_carry_proxy_validation`, is never invoked
anywhere in this task -- see the module docstring on that function for the
frozen failure-after-exposure protocol governing any future real run.

Two DEVELOPMENT-module fixes are reused verbatim rather than reimplemented,
which is itself the enforcement mechanism for two of the frozen protocol
gates (see preflight checks 10 and 11 in
:func:`_run_preflight_checks`):

* Sample-window filtering: :func:`_decompositions_in_validation_window`
  delegates to
  :func:`ftmoquant.research.eurusd_policy_rate_carry_proxy_development._decompositions_in_evaluate_window`,
  which excludes any pre-validation warm-up decomposition from the sample.
* Carry isolation: this module never recomputes a daily P&L decomposition
  itself -- it calls
  :func:`ftmoquant.research.eurusd_policy_rate_carry_proxy_development.daily_decompositions_from_equity_marks`,
  which already isolates the native FX rollover cash accrual via
  ``post.balance - pre.balance`` (never an equity diff).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]
from nautilus_trader.model import Bar, CurrencyPair
from nautilus_trader.persistence import ParquetDataCatalog

from ftmoquant.backtest.execution_harness import _instrument_for_profile
from ftmoquant.data.policy_rates import (
    PolicyRateHistory,
    causal_differential_series,
    load_policy_rate_history,
)
from ftmoquant.data.universe_readiness import SPLIT_VIEW_FILENAME, _sha256_tree
from ftmoquant.research.eurusd_policy_rate_carry_proxy_development import (
    _FROZEN_ACCOUNT,
    BASE_RESEARCH_UNITS,
    EXECUTION_COST_STRESS_MULTIPLIER,
    FROZEN_POLICY_RATE_CANONICAL_DATA_SHA256,
    DailyPnlDecomposition,
    _daily_midpoints_from_minute_bars,
    _decompositions_in_evaluate_window,
    _minute_execution_frames,
    _ns,
    _run_native_fold,
    build_carry_proxy_instructions,
    build_monthly_interest_rate_inputs,
    carry_contribution_ratio,
    carry_proxy_execution_profile,
    component_contributions,
    daily_decompositions_from_equity_marks,
    rate_regime_attribution,
    stressed_daily_total_return,
    uncalibrated_baseline_execution_profile,
)
from ftmoquant.research.eurusd_policy_rate_carry_proxy_spec import (
    EURUSD_POLICY_RATE_CARRY_PROXY_SEMANTIC_SHA256,
    load_eurusd_policy_rate_carry_proxy_spec,
)
from ftmoquant.research.g1.trials import Attribution
from ftmoquant.research.stage_g import (
    DEVELOPMENT_START,
    FROZEN_UNIVERSE_PLAN_SHA256,
    FROZEN_UNIVERSE_READINESS_SHA256,
    HOLDOUT_START,
    VALIDATION_START,
)
from ftmoquant.research.statistics import (
    StationaryBootstrapConfig,
    stationary_bootstrap_confidence_interval,
)
from ftmoquant.research.ts_momentum_development import (
    _annualized_sharpe,
    _maximum_drawdown,
)
from ftmoquant.strategies.eurusd_policy_rate_carry_proxy import (
    EurusdPolicyRateCarryProxyState,
)
from ftmoquant.strategies.ts_momentum import RawDirectionalTarget

PROTOCOL_PATH = Path(
    "config/validation/eurusd_policy_rate_carry_proxy_v1_one_shot.json"
)
PROTOCOL_SEMANTIC_SHA256 = (
    "5133900618c9cbe8744dbb5d0ad5163c16521af30a94b9745d09fb05a5e298cc"
)
FAMILY_SEMANTIC_SHA256 = EURUSD_POLICY_RATE_CARRY_PROXY_SEMANTIC_SHA256
DEVELOPMENT_RESULT_PATH = Path(
    ".artifacts/g1_4h/eurusd_policy_rate_carry_proxy_v1/development_run/"
    "development_result.json"
)
DEVELOPMENT_RESULT_SHA256 = (
    "f44de295580517c9678b95d3c557e9f4a6c3b3df838df061656f2151029d5988"
)
INTEGRITY_AUDIT_PATH = Path(
    ".artifacts/g1_4h/eurusd_policy_rate_carry_proxy_v1/development_run/"
    "post_run_integrity_audit.json"
)
INTEGRITY_AUDIT_SHA256 = (
    "3835ec98f107d303f321740e65e3487deb10ad1d396db0440c41584ef944a723"
)
VALIDATION_OUTPUT_DIR = Path(
    ".artifacts/g1_4h/eurusd_policy_rate_carry_proxy_v1/validation_run"
)
VALIDATION_RESULT_SCHEMA = (
    "ftmoquant.eurusd-policy-rate-carry-proxy-v1-one-shot-validation-result"
)
MINIMUM_ELIGIBLE_OBSERVATIONS = 200
_EURUSD = "EUR/USD.DUKASCOPY"
_SEALED_PATH_MARKERS = ("holdout", "final_holdout", "final-holdout")

# Reused directly from the DEVELOPMENT gate rationale; this multiplier
# stresses only the execution-cost component, never the carry accrual --
# see EXECUTION_COST_STRESS_MULTIPLIER's own docstring in the DEVELOPMENT
# module for the identity this preserves.
_STRESS_MULTIPLIER = EXECUTION_COST_STRESS_MULTIPLIER
assert _STRESS_MULTIPLIER == Decimal("1.5")


class EurusdPolicyRateCarryProxyValidationError(ValueError):
    """Raised before or during the one authorized validation evaluation."""


# ---------------------------------------------------------------------------
# Frozen protocol loader
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ValidationProtocol:
    semantic_sha256: str
    family_semantic_sha256: str
    development_result_sha256: str
    integrity_audit_sha256: str
    start_utc: datetime
    end_exclusive_utc: datetime
    duration_days: int
    minimum_eligible_observations: int
    canonical_document: Mapping[str, Any]


def load_validation_protocol(path: Path = PROTOCOL_PATH) -> ValidationProtocol:
    """Load the frozen protocol document and prove it is exactly what was frozen."""

    document = _json_object(path, "validation protocol")
    computed = _semantic_sha(document)
    declared = document.get("semantic_sha256")
    if declared != computed or declared != PROTOCOL_SEMANTIC_SHA256:
        raise EurusdPolicyRateCarryProxyValidationError(
            "validation protocol semantic SHA mismatch"
        )

    family = _mapping(document.get("family"), "family")
    evidence = _mapping(document.get("development_evidence"), "development_evidence")
    partition = _mapping(document.get("validation_partition"), "validation_partition")
    holdout = _mapping(document.get("final_holdout"), "final_holdout")
    one_shot = _mapping(document.get("one_shot"), "one_shot")
    candidate = _mapping(document.get("candidate"), "candidate")
    sample_window = _mapping(document.get("sample_window"), "sample_window")
    carry_isolation = _mapping(document.get("carry_isolation"), "carry_isolation")
    sample_semantics = _mapping(document.get("sample_semantics"), "sample_semantics")
    acceptance = _mapping(document.get("acceptance"), "acceptance")
    seal = _mapping(
        document.get("post_development_rate_covariate_seal"),
        "post_development_rate_covariate_seal",
    )

    start = _utc(partition.get("start_utc"), "validation start")
    end = _utc(partition.get("end_exclusive_utc"), "validation end")
    expected_gates = [
        "deterministic_numerically_valid_completion",
        "eligible_daily_observations_gte_200",
        "validation_mean_daily_total_return_gt_zero",
        "cost_stressed_validation_mean_daily_total_return_gt_zero",
    ]
    if (
        document.get("schema")
        != "ftmoquant.eurusd-policy-rate-carry-proxy-v1-one-shot-validation-protocol"
        or document.get("schema_version") != 1
        or document.get("status") != "frozen_pre_validation_unobserved"
        or document.get("validation_returns_accessed") is not False
        or family
        != {
            "family_id": "eurusd_policy_rate_carry_proxy_v1",
            "family_version": "1.0.0",
            "family_semantic_sha256": FAMILY_SEMANTIC_SHA256,
        }
        or evidence.get("development_result_sha256") != DEVELOPMENT_RESULT_SHA256
        or evidence.get("integrity_audit_sha256") != INTEGRITY_AUDIT_SHA256
        or evidence.get("development_classification")
        != "DEVELOPMENT_CANDIDATE_SELECTED"
        or evidence.get("corrected_pooled_observation_count") != 777
        or evidence.get("corrected_positive_fold_count") != 2
        or evidence.get("corrected_fold_count") != 3
        or start != VALIDATION_START
        or end != HOLDOUT_START
        or partition.get("partition") != "validation"
        or partition.get("duration_days") != 498
        or (end - start).days != partition.get("duration_days")
        or holdout.get("locked") is not True
        or holdout.get("accessed") is not False
        or one_shot.get("maximum_configurations") != 1
        or one_shot.get("validation_search_or_selection") != "forbidden"
        or one_shot.get("alternate_candidate_after_rejection") != "forbidden"
        or one_shot.get("fixed_output_directory") != str(VALIDATION_OUTPUT_DIR)
        or one_shot.get("no_search_or_selector_imports") is not True
        or candidate.get("signal") != "sign_of_causal_ecbdfr_minus_effr_differential"
        or candidate.get("parameter_search") != "none"
        or candidate.get("cost_stress_multiplier") != 1.5
        or candidate.get("pending_target_policy")
        != "latest_causal_target_supersedes_unexecuted_older_target"
        or sample_window.get("pre_validation_pnl_or_sample_counted") is not False
        or carry_isolation.get("method") != "post_balance_minus_pre_balance"
        or carry_isolation.get("forbidden_method") != "post_equity_minus_pre_equity"
        or carry_isolation.get("fill_inside_bracket") != "fail_closed"
        or sample_semantics.get("is_not_a_trade_count") is not True
        or sample_semantics.get("minimum_eligible_observations")
        != MINIMUM_ELIGIBLE_OBSERVATIONS
        or acceptance.get("gates") != expected_gates
        or acceptance.get("all_required") is not True
        or seal.get("rate_data_consumed_only_during_the_one_shot_run_itself")
        is not True
    ):
        raise EurusdPolicyRateCarryProxyValidationError(
            "validation protocol semantics drifted"
        )

    return ValidationProtocol(
        semantic_sha256=cast(str, declared),
        family_semantic_sha256=FAMILY_SEMANTIC_SHA256,
        development_result_sha256=DEVELOPMENT_RESULT_SHA256,
        integrity_audit_sha256=INTEGRITY_AUDIT_SHA256,
        start_utc=start,
        end_exclusive_utc=end,
        duration_days=cast(int, partition["duration_days"]),
        minimum_eligible_observations=cast(
            int, sample_semantics["minimum_eligible_observations"]
        ),
        canonical_document=document,
    )


def verify_development_evidence(
    protocol: ValidationProtocol,
    *,
    development_result_path: Path = DEVELOPMENT_RESULT_PATH,
    integrity_audit_path: Path = INTEGRITY_AUDIT_PATH,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Hash-check and cross-verify the two DEVELOPMENT evidence artifacts.

    This family never ran ``select_candidate`` or ``run_search``, so there is
    no ``selected_candidate.json``/``trial_registry.json`` to verify (unlike
    other families' validation adapters) -- DEVELOPMENT evidence here is
    exactly these two artifacts.
    """

    for path, expected, label in (
        (
            development_result_path,
            protocol.development_result_sha256,
            "DEVELOPMENT result",
        ),
        (integrity_audit_path, protocol.integrity_audit_sha256, "integrity audit"),
    ):
        if _sha256(path) != expected:
            raise EurusdPolicyRateCarryProxyValidationError(f"{label} SHA mismatch")

    result = _json_object(development_result_path, "DEVELOPMENT result")
    audit = _json_object(integrity_audit_path, "integrity audit")
    if (
        result.get("family_semantic_sha256") != protocol.family_semantic_sha256
        or result.get("gate_passed") is not True
        or audit.get("family_semantic_sha256") != protocol.family_semantic_sha256
        or audit.get("final_classification") != "DEVELOPMENT_CANDIDATE_SELECTED"
    ):
        raise EurusdPolicyRateCarryProxyValidationError(
            "DEVELOPMENT evidence identity mismatch"
        )
    return result, audit


# ---------------------------------------------------------------------------
# Metrics and gate evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ValidationMetrics:
    observation_count: int
    cumulative_total_return: float
    mean_daily_total_return: float
    stressed_cumulative_total_return: float
    stressed_mean_daily_total_return: float
    sharpe: float
    maximum_drawdown: float
    stationary_bootstrap_ci: Mapping[str, Any] | None
    yearly_attribution: tuple[Attribution, ...]
    rate_regime_attribution: tuple[Attribution, ...]
    spot_pnl_contribution: float
    carry_accrual_contribution: float
    execution_cost_contribution: float
    carry_contribution_ratio: float | None


def validation_passes(metrics: ValidationMetrics, protocol: ValidationProtocol) -> bool:
    """The exact four hard gates -- nothing else is gated (Section 5)."""

    numerical = (
        metrics.cumulative_total_return,
        metrics.mean_daily_total_return,
        metrics.stressed_cumulative_total_return,
        metrics.stressed_mean_daily_total_return,
    )
    return (
        all(math.isfinite(item) for item in numerical)
        and metrics.observation_count >= protocol.minimum_eligible_observations
        and metrics.mean_daily_total_return > 0.0
        and metrics.stressed_mean_daily_total_return > 0.0
    )


def _decompositions_in_validation_window(
    decompositions: Sequence[DailyPnlDecomposition],
) -> tuple[DailyPnlDecomposition, ...]:
    """Validation-window analogue of the DEVELOPMENT module's own fold filter.

    Delegates to the exact same fixed function used by DEVELOPMENT, given a
    lightweight object exposing only the two attributes it reads
    (``compare_start_utc``, ``compare_end_exclusive_utc``) -- no filtering
    logic is reimplemented here.
    """

    window = SimpleNamespace(
        compare_start_utc=VALIDATION_START,
        compare_end_exclusive_utc=HOLDOUT_START,
    )
    return _decompositions_in_evaluate_window(decompositions, window)


def _bootstrap_ci(
    decompositions: Sequence[DailyPnlDecomposition],
) -> dict[str, Any] | None:
    """Diagnostic-only dependence-aware CI; never gates acceptance."""

    if len(decompositions) < 20:
        return None
    series = pd.Series(
        [float(item.total_return) for item in decompositions],
        name="eurusd_policy_rate_carry_proxy_v1_validation_daily_total_return",
    )
    config = StationaryBootstrapConfig(
        block_size=20,
        repetitions=10_000,
        seed=14_042_026,
        confidence_level=0.95,
        method="basic",
    )
    result = stationary_bootstrap_confidence_interval(series, config)
    return {
        "arch_version": result.arch_version,
        "procedure": result.procedure,
        "seed": result.seed,
        "observation_count": result.observation_count,
        "estimate": result.estimate,
        "lower_bound": result.lower_bound,
        "upper_bound": result.upper_bound,
    }


def _signal_by_date(
    decompositions: Sequence[DailyPnlDecomposition],
    differentials: Sequence[Any],
) -> dict[Any, RawDirectionalTarget]:
    state = EurusdPolicyRateCarryProxyState()
    by_date = {item.as_of_date: item.differential_percent for item in differentials}
    result: dict[Any, RawDirectionalTarget] = {}
    for item in decompositions:
        differential = by_date.get(item.as_of_date)
        if differential is None:
            continue
        result[item.as_of_date] = state.target_for(item.as_of_date, differential).target
    return result


def compute_validation_metrics(
    decompositions: Sequence[DailyPnlDecomposition],
    differentials: Sequence[Any],
) -> ValidationMetrics:
    windowed = _decompositions_in_validation_window(decompositions)
    if not windowed:
        raise EurusdPolicyRateCarryProxyValidationError(
            "validation window has no eligible daily return observations"
        )
    returns = tuple(float(item.total_return) for item in windowed)
    stressed_returns = tuple(
        float(stressed_daily_total_return(item) / BASE_RESEARCH_UNITS)
        for item in windowed
    )
    count = len(windowed)
    yearly: dict[str, float] = {}
    for item, value in zip(windowed, returns, strict=True):
        label = str(item.as_of_date.year)
        yearly[label] = yearly.get(label, 0.0) + value
    contributions = component_contributions(windowed)
    signal_by_date = _signal_by_date(windowed, differentials)
    return ValidationMetrics(
        observation_count=count,
        cumulative_total_return=sum(returns),
        mean_daily_total_return=sum(returns) / count,
        stressed_cumulative_total_return=sum(stressed_returns),
        stressed_mean_daily_total_return=sum(stressed_returns) / count,
        sharpe=_annualized_sharpe(returns) or 0.0,
        maximum_drawdown=_maximum_drawdown(returns),
        stationary_bootstrap_ci=_bootstrap_ci(windowed),
        yearly_attribution=tuple(
            Attribution(label, value) for label, value in sorted(yearly.items())
        ),
        rate_regime_attribution=rate_regime_attribution(windowed, signal_by_date),
        spot_pnl_contribution=contributions["spot_pnl_contribution"],
        carry_accrual_contribution=contributions["carry_accrual_contribution"],
        execution_cost_contribution=contributions["execution_cost_contribution"],
        carry_contribution_ratio=carry_contribution_ratio(windowed),
    )


# ---------------------------------------------------------------------------
# Evaluation (native engine, real data -- never invoked in this task)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ValidationPreparedData:
    instrument: CurrencyPair
    minute_bars: tuple[Bar, ...]
    history: PolicyRateHistory
    development_manifest_sha256: str
    validation_manifest_sha256: str
    validation_catalog_tree_sha256: str


def evaluate_prepared_validation(
    prepared: ValidationPreparedData,
) -> ValidationMetrics:
    """Run the single frozen candidate over the validation window.

    The engine itself is started at ``DEVELOPMENT_START`` (identical to
    every DEVELOPMENT fold's own ``train_start_utc``) purely to let causal
    EWMA volatility and rate state warm up before ``VALIDATION_START`` --
    exactly the pattern DEVELOPMENT's own folds already use. Warm-up days
    carry zero exposure by construction (no instruction exists before
    ``VALIDATION_START``) and are excluded from the sample by
    :func:`_decompositions_in_validation_window` regardless.
    """

    frames = _minute_execution_frames(prepared.minute_bars)
    daily_midpoints = _daily_midpoints_from_minute_bars(prepared.minute_bars)
    span_start = DEVELOPMENT_START.date()
    span_end = (HOLDOUT_START - timedelta(days=1)).date()
    differentials = causal_differential_series(
        prepared.history, start_inclusive=span_start, end_inclusive=span_end
    )
    records = build_monthly_interest_rate_inputs(
        prepared.history, span_start=span_start, span_end=span_end
    )
    profile = carry_proxy_execution_profile(
        uncalibrated_baseline_execution_profile(), records=records
    )
    instructions = build_carry_proxy_instructions(
        minute_execution_frames=frames,
        daily_midpoints=daily_midpoints,
        differentials=differentials,
        evaluate_start_ns=_ns(VALIDATION_START),
        evaluate_end_exclusive_ns=_ns(HOLDOUT_START),
    )
    decompositions = _run_native_fold(
        fold_id="validation_one_shot",
        instrument=prepared.instrument,
        minute_bars=prepared.minute_bars,
        instructions=instructions,
        account=_FROZEN_ACCOUNT,
        profile=profile,
        start_ns=_ns(DEVELOPMENT_START),
        end_exclusive_ns=_ns(HOLDOUT_START),
    )
    return compute_validation_metrics(decompositions, differentials)


def prepare_validation_market_data(
    *,
    protocol: ValidationProtocol,
    universe_readiness_path: Path,
    development_root: Path,
    validation_root: Path,
    policy_rates_dir: Path,
) -> ValidationPreparedData:
    """Open validation data only after every non-data preflight has passed."""

    _reject_final_holdout_path(validation_root.resolve())

    readiness = _json_object(universe_readiness_path, "universe readiness")
    if readiness.get("semantic_sha256") != FROZEN_UNIVERSE_READINESS_SHA256:
        raise EurusdPolicyRateCarryProxyValidationError(
            "universe readiness SHA mismatch"
        )
    artifacts = {
        item.get("instrument_id"): item
        for item in cast(
            list[dict[str, Any]], readiness.get("instrument_artifacts", [])
        )
    }
    eurusd_artifact = artifacts.get(_EURUSD)
    if not isinstance(eurusd_artifact, dict):
        raise EurusdPolicyRateCarryProxyValidationError(
            "universe readiness has no EUR/USD instrument artifact"
        )

    dev_root = development_root.resolve()
    _reject_final_holdout_path(dev_root)
    dev_manifest = _json_object(
        dev_root / SPLIT_VIEW_FILENAME, "EUR/USD development split"
    )
    dev_expected_range = {
        "start_utc": _format_utc(DEVELOPMENT_START),
        "end_exclusive_utc": _format_utc(protocol.start_utc),
    }
    if (
        dev_manifest.get("semantic_sha256") != _semantic_sha(dev_manifest)
        or dev_manifest.get("semantic_sha256")
        != eurusd_artifact.get("development_split_sha256")
        or dev_manifest.get("instrument_id") != _EURUSD
        or dev_manifest.get("universe_plan_sha256") != FROZEN_UNIVERSE_PLAN_SHA256
        or dev_manifest.get("split") != "development"
        or dev_manifest.get("range") != dev_expected_range
        or dev_manifest.get("holdout_rows") != 0
    ):
        raise EurusdPolicyRateCarryProxyValidationError(
            "sealed DEVELOPMENT manifest is incompatible"
        )

    root = validation_root.resolve()
    manifest_path = root / SPLIT_VIEW_FILENAME
    manifest = _json_object(manifest_path, "EUR/USD validation split")
    expected_range = {
        "start_utc": _format_utc(protocol.start_utc),
        "end_exclusive_utc": _format_utc(protocol.end_exclusive_utc),
    }
    if (
        manifest.get("semantic_sha256") != _semantic_sha(manifest)
        or manifest.get("semantic_sha256")
        != eurusd_artifact.get("validation_split_sha256")
        or manifest.get("instrument_id") != _EURUSD
        or manifest.get("universe_plan_sha256") != FROZEN_UNIVERSE_PLAN_SHA256
        or manifest.get("split") != "validation"
        or manifest.get("range") != expected_range
        or manifest.get("holdout_rows") != 0
        or manifest.get("network_access_required") is not False
    ):
        raise EurusdPolicyRateCarryProxyValidationError(
            "sealed validation manifest is incompatible"
        )
    declared_tree = manifest.get("catalog_tree_sha256")
    if (
        not isinstance(declared_tree, str)
        or _sha256_tree(root / "catalog") != declared_tree
    ):
        raise EurusdPolicyRateCarryProxyValidationError(
            "validation catalog tree SHA mismatch"
        )

    history = load_policy_rate_history(policy_rates_dir)
    if history.canonical_data_sha256 != FROZEN_POLICY_RATE_CANONICAL_DATA_SHA256:
        raise EurusdPolicyRateCarryProxyValidationError(
            "policy rate canonical_data_sha256 does not match the value "
            "pinned at audit time"
        )

    development_catalog = ParquetDataCatalog(
        str(development_root.resolve() / "catalog")
    )
    validation_catalog = ParquetDataCatalog(str(root / "catalog"))
    found = validation_catalog.instruments([_EURUSD])
    if len(found) != 1 or not isinstance(found[0], CurrencyPair):
        raise EurusdPolicyRateCarryProxyValidationError(
            "validation EUR/USD instrument is unavailable"
        )
    profile = uncalibrated_baseline_execution_profile()
    instrument = _instrument_for_profile(found[0], profile.fee)

    development_minute = _query_minute_bars(
        development_catalog, DEVELOPMENT_START, protocol.start_utc
    )
    validation_minute = _query_minute_bars(
        validation_catalog, protocol.start_utc, protocol.end_exclusive_utc
    )
    combined = tuple(
        sorted(
            (*development_minute, *validation_minute),
            key=lambda bar: (bar.ts_event, str(bar.bar_type)),
        )
    )
    if _sha256_tree(root / "catalog") != declared_tree:
        raise EurusdPolicyRateCarryProxyValidationError(
            "validation catalog changed during read"
        )
    development_manifest = _json_object(
        development_root.resolve() / SPLIT_VIEW_FILENAME, "EUR/USD development split"
    )
    return ValidationPreparedData(
        instrument=instrument,
        minute_bars=combined,
        history=history,
        development_manifest_sha256=cast(
            str, development_manifest.get("semantic_sha256")
        ),
        validation_manifest_sha256=cast(str, manifest["semantic_sha256"]),
        validation_catalog_tree_sha256=declared_tree,
    )


def run_eurusd_policy_rate_carry_proxy_validation(
    *,
    protocol_path: Path = PROTOCOL_PATH,
    development_result_path: Path = DEVELOPMENT_RESULT_PATH,
    integrity_audit_path: Path = INTEGRITY_AUDIT_PATH,
    universe_readiness_path: Path,
    development_root: Path,
    validation_root: Path,
    policy_rates_dir: Path,
    output_dir: Path = VALIDATION_OUTPUT_DIR,
) -> dict[str, Any]:
    """Run the single frozen validation candidate exactly once.

    THIS FUNCTION IS NOT INVOKED ANYWHERE IN THIS TASK.

    Failure-after-exposure protocol (Section 10, documented not automated):
    once this function has actually run against real validation data and
    validation observations/returns have been exposed to any human or
    process, no automatic repair-and-rerun of this function -- or any
    substitute for it -- is permitted, regardless of the nature of the
    failure. If a runtime, data, or implementation failure is discovered
    after such exposure, it must be reported explicitly: (1) the exact
    failure, (2) how much validation evidence was exposed (partial metrics?
    the full result payload? to whom?), (3) whether any economic result was
    actually generated, and (4) whether the suspected bug is economically
    material to that result. A second validation attempt requires explicit
    methodological review before any code runs again -- there is
    deliberately no retry, repair, or fallback path implemented in this
    module; its absence is the safety property, not an oversight.
    """

    spec, protocol, _evidence = _run_preflight_checks(
        protocol_path=protocol_path,
        development_result_path=development_result_path,
        integrity_audit_path=integrity_audit_path,
        output_dir=output_dir,
    )
    prepared = prepare_validation_market_data(
        protocol=protocol,
        universe_readiness_path=universe_readiness_path,
        development_root=development_root,
        validation_root=validation_root,
        policy_rates_dir=policy_rates_dir,
    )
    metrics = evaluate_prepared_validation(prepared)
    passed = validation_passes(metrics, protocol)
    payload: dict[str, Any] = {
        "schema": VALIDATION_RESULT_SCHEMA,
        "schema_version": 1,
        "protocol_semantic_sha256": protocol.semantic_sha256,
        "family_id": spec.family_id,
        "family_version": spec.version,
        "family_semantic_sha256": spec.semantic_sha256,
        "git_commit": _git_commit(),
        "validation_partition": {
            "start_utc": _format_utc(protocol.start_utc),
            "end_exclusive_utc": _format_utc(protocol.end_exclusive_utc),
            "duration_days": protocol.duration_days,
            "validation_manifest_sha256": prepared.validation_manifest_sha256,
            "validation_catalog_tree_sha256": prepared.validation_catalog_tree_sha256,
        },
        "metrics": {
            "observation_count": metrics.observation_count,
            "cumulative_total_return": metrics.cumulative_total_return,
            "mean_daily_total_return": metrics.mean_daily_total_return,
            "stressed_cumulative_total_return": (
                metrics.stressed_cumulative_total_return
            ),
            "stressed_mean_daily_total_return": (
                metrics.stressed_mean_daily_total_return
            ),
            "sharpe": metrics.sharpe,
            "maximum_drawdown": metrics.maximum_drawdown,
            "stationary_bootstrap_ci": metrics.stationary_bootstrap_ci,
            "yearly_attribution": [asdict(item) for item in metrics.yearly_attribution],
            "rate_regime_attribution": [
                asdict(item) for item in metrics.rate_regime_attribution
            ],
            "spot_pnl_contribution": metrics.spot_pnl_contribution,
            "carry_accrual_contribution": metrics.carry_accrual_contribution,
            "execution_cost_contribution": metrics.execution_cost_contribution,
            "carry_contribution_ratio": metrics.carry_contribution_ratio,
        },
        "acceptance": {
            "outcome": "VALIDATION_PASSED" if passed else "VALIDATION_REJECTED",
            "all_gates_passed": passed,
            "minimum_eligible_observations": protocol.minimum_eligible_observations,
        },
        "validation_returns_accessed": True,
        "final_holdout_accessed": False,
    }
    _write_validation_artifacts(output_dir, payload)
    return payload


def _query_minute_bars(
    catalog: ParquetDataCatalog, start: datetime, end: datetime
) -> tuple[Bar, ...]:
    start_ns = _ns(start)
    end_ns = _ns(end)
    bid_type = f"{_EURUSD}-1-MINUTE-BID-EXTERNAL"
    ask_type = f"{_EURUSD}-1-MINUTE-ASK-EXTERNAL"
    bid = tuple(catalog.query_bars([bid_type], start=start_ns, end=end_ns - 1))
    ask = tuple(catalog.query_bars([ask_type], start=start_ns, end=end_ns - 1))
    _validate_bar_stream(bid, bid_type, start_ns, end_ns)
    _validate_bar_stream(ask, ask_type, start_ns, end_ns)
    return tuple(
        sorted((*bid, *ask), key=lambda bar: (bar.ts_event, str(bar.bar_type)))
    )


def _validate_bar_stream(
    bars: Sequence[Bar], bar_type: str, start_ns: int, end_ns: int
) -> None:
    if not bars:
        raise EurusdPolicyRateCarryProxyValidationError(
            f"validation has no {bar_type} bars"
        )
    keys = tuple((bar.ts_event, bar.ts_init) for bar in bars)
    if len(set(keys)) != len(keys) or any(
        current <= previous for previous, current in zip(keys, keys[1:], strict=False)
    ):
        raise EurusdPolicyRateCarryProxyValidationError(
            f"{bar_type} timestamps are not monotonic"
        )
    if any(
        str(bar.bar_type) != bar_type
        or bar.ts_event < start_ns
        or bar.ts_init >= end_ns
        for bar in bars
    ):
        raise EurusdPolicyRateCarryProxyValidationError(
            f"{bar_type} query admitted invalid rows"
        )


# ---------------------------------------------------------------------------
# Preflight checks (Section 9). Each is independently unit-testable and
# runs, in order, before any market/rate data is opened.
# ---------------------------------------------------------------------------


def _check_family_semantic_sha() -> Any:
    """Check 1: the family spec loads and matches the frozen semantic SHA."""

    spec = load_eurusd_policy_rate_carry_proxy_spec()
    if spec.semantic_sha256 != FAMILY_SEMANTIC_SHA256:
        raise EurusdPolicyRateCarryProxyValidationError(
            "family semantic SHA does not match the frozen validation target"
        )
    return spec


def _check_development_result_sha(path: Path) -> None:
    """Check 2: the exact DEVELOPMENT result artifact SHA."""

    if _sha256(path) != DEVELOPMENT_RESULT_SHA256:
        raise EurusdPolicyRateCarryProxyValidationError(
            "DEVELOPMENT result SHA mismatch"
        )


def _check_integrity_audit_sha(path: Path) -> None:
    """Check 3: the exact post-run integrity audit artifact SHA."""

    if _sha256(path) != INTEGRITY_AUDIT_SHA256:
        raise EurusdPolicyRateCarryProxyValidationError("integrity audit SHA mismatch")


def _check_validation_boundaries(protocol: ValidationProtocol) -> None:
    """Check 4: protocol boundaries match ``stage_g`` imports, not duplicates."""

    if (
        protocol.start_utc != VALIDATION_START
        or protocol.end_exclusive_utc != HOLDOUT_START
    ):
        raise EurusdPolicyRateCarryProxyValidationError(
            "validation boundaries do not match the frozen stage_g partition"
        )


def _check_clean_worktree() -> None:
    """Check 5: refuse to run against any uncommitted change."""

    completed = subprocess.run(
        ["git", "status", "--porcelain=v1"], check=True, capture_output=True, text=True
    )
    if completed.stdout.strip():
        raise EurusdPolicyRateCarryProxyValidationError(
            "one-shot validation requires a clean git worktree"
        )


def _check_output_not_exists(output_dir: Path) -> None:
    """Check 6: refuse to silently overwrite an existing output directory."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise EurusdPolicyRateCarryProxyValidationError(
            f"output already exists and is non-empty: {output_dir}"
        )


def _check_one_candidate_only(document: Mapping[str, Any]) -> None:
    """Check 7: exactly one configuration; no selector/trial fields present."""

    one_shot = _mapping(document.get("one_shot"), "one_shot")
    if one_shot.get("maximum_configurations") != 1:
        raise EurusdPolicyRateCarryProxyValidationError(
            "protocol does not declare exactly one configuration"
        )
    forbidden_keys = {
        "selected_trial_id",
        "parameters",
        "parameter_grid",
        "trial_registry",
    }
    if forbidden_keys & set(document):
        raise EurusdPolicyRateCarryProxyValidationError(
            "protocol document contains a selector/trial-grid field, which "
            "this baseline-only family must never have"
        )


_FORBIDDEN_IMPORT_MODULE_PREFIXES = (
    "ftmoquant.research.g1.search",
    "ftmoquant.research.g1.selector",
    "ftmoquant.research.g1.plateau",
)
_FORBIDDEN_IMPORT_NAMES = (
    "TrialRegistry",
    "run_search",
    "select_candidate",
    "assess_plateau",
)


def _check_no_search_or_selector_imports() -> None:
    """Check 8: static code-hygiene property of this module's own import
    statements only (parsed via ``ast``, never a raw text/docstring grep --
    this module's own prose necessarily *names* the forbidden modules while
    explaining that it never imports them, so a text grep would false-fail
    on its own docstring)."""

    import ast

    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"), filename=__file__)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(
                    alias.name.startswith(prefix)
                    for prefix in _FORBIDDEN_IMPORT_MODULE_PREFIXES
                ):
                    raise EurusdPolicyRateCarryProxyValidationError(
                        f"forbidden search/selector import present: {alias.name}"
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if any(
                module.startswith(prefix)
                for prefix in _FORBIDDEN_IMPORT_MODULE_PREFIXES
            ):
                raise EurusdPolicyRateCarryProxyValidationError(
                    f"forbidden search/selector import present: {module}"
                )
            for alias in node.names:
                if alias.name in _FORBIDDEN_IMPORT_NAMES:
                    raise EurusdPolicyRateCarryProxyValidationError(
                        f"forbidden search/selector import present: {alias.name}"
                    )


def _reject_final_holdout_path(path: Path) -> None:
    """Check 9: refuse any caller-supplied path implying final-holdout access."""

    lowered = tuple(part.casefold() for part in path.parts)
    if any(marker in part for marker in _SEALED_PATH_MARKERS for part in lowered):
        raise EurusdPolicyRateCarryProxyValidationError(
            f"path implies final-holdout access: {path}"
        )


def _check_sample_window_and_carry_isolation_enforcement() -> None:
    """Checks 10 & 11: enforced by direct reuse of the DEVELOPMENT module's
    already-fixed functions, not reimplementation -- verified here by
    identity, so any future refactor that stops reusing them fails loudly."""

    from ftmoquant.research.eurusd_policy_rate_carry_proxy_development import (
        _decompositions_in_evaluate_window as _dev_filter,
    )
    from ftmoquant.research.eurusd_policy_rate_carry_proxy_development import (
        daily_decompositions_from_equity_marks as _dev_carry,
    )

    if daily_decompositions_from_equity_marks is not _dev_carry:
        raise EurusdPolicyRateCarryProxyValidationError(
            "carry isolation is not reusing the fixed DEVELOPMENT function"
        )
    if _decompositions_in_evaluate_window is not _dev_filter:
        raise EurusdPolicyRateCarryProxyValidationError(
            "sample-window filtering is not reusing the fixed DEVELOPMENT function"
        )


def _run_preflight_checks(
    *,
    protocol_path: Path,
    development_result_path: Path,
    integrity_audit_path: Path,
    output_dir: Path,
) -> tuple[Any, ValidationProtocol, tuple[dict[str, Any], dict[str, Any]]]:
    """Run every preflight check, in order, before any catalog is opened."""

    spec = _check_family_semantic_sha()
    _check_development_result_sha(development_result_path)
    _check_integrity_audit_sha(integrity_audit_path)
    protocol = load_validation_protocol(protocol_path)
    _check_validation_boundaries(protocol)
    _check_clean_worktree()
    _check_output_not_exists(output_dir)
    _check_one_candidate_only(protocol.canonical_document)
    _check_no_search_or_selector_imports()
    _check_sample_window_and_carry_isolation_enforcement()
    evidence = verify_development_evidence(
        protocol,
        development_result_path=development_result_path,
        integrity_audit_path=integrity_audit_path,
    )
    return spec, protocol, evidence


# ---------------------------------------------------------------------------
# Artifact writing
# ---------------------------------------------------------------------------


def _write_validation_artifacts(output_dir: Path, payload: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    normalized = json.loads(_canonical_bytes(payload))
    semantic = hashlib.sha256(_canonical_bytes(normalized)).hexdigest()
    result_bytes = _canonical_bytes({**normalized, "semantic_sha256": semantic}) + b"\n"
    result_path = output_dir / "validation_result.json"
    result_path.write_bytes(result_bytes)
    result_sha = hashlib.sha256(result_bytes).hexdigest()
    (output_dir / "artifact_hashes.json").write_bytes(
        _canonical_bytes({result_path.name: result_sha}) + b"\n"
    )
    (output_dir / "runtime_provenance.json").write_bytes(
        _canonical_bytes(
            {
                "git_commit": _git_commit(),
                "python_module": (
                    "ftmoquant.research.eurusd_policy_rate_carry_proxy_validation"
                ),
            }
        )
        + b"\n"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str
    ).encode()


def _semantic_sha(document: Mapping[str, Any]) -> str:
    payload = {
        key: value for key, value in document.items() if key != "semantic_sha256"
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EurusdPolicyRateCarryProxyValidationError(
            f"could not load {label}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise EurusdPolicyRateCarryProxyValidationError(
            f"{label} must be a JSON object"
        )
    return cast(dict[str, Any], value)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise EurusdPolicyRateCarryProxyValidationError(
            f"{label} must be a string-keyed object"
        )
    return cast(dict[str, Any], value)


def _utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise EurusdPolicyRateCarryProxyValidationError(f"{label} must be UTC text")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise EurusdPolicyRateCarryProxyValidationError(
            f"{label} is invalid"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise EurusdPolicyRateCarryProxyValidationError(f"{label} must be UTC")
    return parsed


def _format_utc(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    try:
        content = path.read_bytes()
    except OSError as error:
        raise EurusdPolicyRateCarryProxyValidationError(
            f"could not read {path} for hashing: {error}"
        ) from error
    return hashlib.sha256(content).hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _validation_root(value: str) -> Path:
    path = Path(value)
    try:
        _reject_final_holdout_path(path)
    except EurusdPolicyRateCarryProxyValidationError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the frozen one-shot eurusd_policy_rate_carry_proxy_v1 "
            "validation candidate (NOT authorized to run in this task)"
        )
    )
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument(
        "--development-result", type=Path, default=DEVELOPMENT_RESULT_PATH
    )
    parser.add_argument("--integrity-audit", type=Path, default=INTEGRITY_AUDIT_PATH)
    parser.add_argument("--universe-readiness", type=Path, required=True)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--validation-root", type=_validation_root, required=True)
    parser.add_argument("--policy-rates-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=VALIDATION_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_eurusd_policy_rate_carry_proxy_validation(
        protocol_path=cast(Path, args.protocol),
        development_result_path=cast(Path, args.development_result),
        integrity_audit_path=cast(Path, args.integrity_audit),
        universe_readiness_path=cast(Path, args.universe_readiness),
        development_root=cast(Path, args.development_root),
        validation_root=cast(Path, args.validation_root),
        policy_rates_dir=cast(Path, args.policy_rates_dir),
        output_dir=cast(Path, args.output),
    )
    print(_canonical_bytes(result).decode())


if __name__ == "__main__":
    main()
