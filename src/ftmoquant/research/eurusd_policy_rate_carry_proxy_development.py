"""Production DEVELOPMENT evaluator for ``eurusd_policy_rate_carry_proxy_v1``.

This is a baseline-only family: exactly one configuration, no parameter
grid, no selector, no plateau/neighbour analysis. ``select_candidate`` and
``assess_plateau`` are never imported or called anywhere in this module.

The module has no validation or holdout loader.  Real execution is opt-in
only through :func:`main`/:func:`run_eurusd_policy_rate_carry_proxy_development`;
importing and unit-testing this module never opens the frozen DEVELOPMENT
catalogs or reads real market/rate data beyond the already-committed,
hash-verified ``config/data/policy_rates`` source.

Reuses, rather than reimplements:

* :class:`ftmoquant.research.g1.normalization.G1VolatilityNormalizer` for
  causal 1% annualized-volatility position sizing (verbatim).
* :func:`ftmoquant.research.stage_g.frozen_development_folds` for the exact
  3 frozen DEVELOPMENT folds.
* :mod:`ftmoquant.research.g1.trials` (``FoldMetrics``, ``TrialRegistry``,
  ``make_trial_record``, ``summarize_trial``) for pooled-metric bookkeeping.
* :mod:`ftmoquant.research.g1.artifacts` for deterministic artifact bytes.
* The native G0.7 Nautilus execution engine and
  :class:`nautilus_trader.backtest.FXRolloverInterestModule` (wired through
  :class:`ftmoquant.backtest.execution_harness.RolloverConfig`) for the
  proxy carry accrual -- no second carry calculation is implemented here.

``FoldMetrics.trade_count`` is repurposed in this family to hold
``eligible_daily_return_observation_count`` -- the count of eligible causal
trading-day marked TOTAL return observations (spot P&L + proxy carry
accrual - execution costs). It is explicitly **not** a trade, order, fill,
or rebalance count for this family; see ``sample_count`` in the frozen
preregistration YAML.
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

from nautilus_trader.backtest import BacktestEngine
from nautilus_trader.execution import ProbabilisticFillModel, StaticLatencyModel
from nautilus_trader.model import (
    Bar,
    Currency,
    CurrencyPair,
    InstrumentId,
    OrderFilled,
    OrderSide,
    Position,
    Price,
    StrategyId,
)
from nautilus_trader.persistence import ParquetDataCatalog
from nautilus_trader.trading import Strategy, StrategyConfig

from ftmoquant.backtest.execution_harness import (
    AccountParameters,
    CalibrationStatus,
    ExecutionProfile,
    FeeModelConfig,
    FeeModelKind,
    InterestRateInput,
    RolloverConfig,
    RolloverMode,
    _add_venue,
    _engine_config,
    _fee_model,
    _instrument_for_profile,
    _rollover_modules,
)
from ftmoquant.data.dukascopy import NAUTILUS_VERSION
from ftmoquant.data.policy_rates import (
    DatedRate,
    PolicyRateHistory,
    PolicyRateProvenanceError,
    causal_differential_series,
    load_policy_rate_history,
)
from ftmoquant.research.eurusd_policy_rate_carry_proxy_spec import (
    EURUSD_POLICY_RATE_CARRY_PROXY_SEMANTIC_SHA256,
    EurusdPolicyRateCarryProxySpec,
    load_eurusd_policy_rate_carry_proxy_spec,
)
from ftmoquant.research.g1.family import FamilyMetadata, StrategyFamily
from ftmoquant.research.g1.normalization import (
    CausalEwmaDailyVolatility,
    CompletedDailyMidpoint,
    G1VolatilityNormalizer,
    completed_daily_log_returns,
)
from ftmoquant.research.g1.parameter_space import (
    CategoricalParameter,
    ParameterMapping,
    ParameterSpace,
)
from ftmoquant.research.g1.sessions import SessionId
from ftmoquant.research.g1.trials import (
    Attribution,
    FoldMetrics,
    TrialStatus,
    make_trial_record,
    summarize_trial,
)
from ftmoquant.research.stage_g import (
    DEVELOPMENT_END_EXCLUSIVE,
    DEVELOPMENT_START,
    HOLDOUT_START,
    VALIDATION_START,
    frozen_development_folds,
    load_frozen_universe,
)
from ftmoquant.research.ts_momentum_development import (
    _annualized_sharpe,
    _maximum_drawdown,
)
from ftmoquant.strategies.eurusd_policy_rate_carry_proxy import (
    EurusdPolicyRateCarryProxyState,
)
from ftmoquant.strategies.ts_momentum import RawDirectionalTarget

EVALUATOR_VERSION = "eurusd-policy-rate-carry-proxy-v1-development-evaluator-1"
RESULT_SCHEMA = "ftmoquant.g1-eurusd-policy-rate-carry-proxy-development-results"
BASE_RESEARCH_UNITS = Decimal("100000")
EXECUTION_COST_STRESS_MULTIPLIER = Decimal("1.5")
MINIMUM_POOLED_ELIGIBLE_OBSERVATIONS = 500
REQUIRED_FOLD_COUNT = 3
MINIMUM_POSITIVE_FOLDS = 2
_EURUSD = "EUR/USD.DUKASCOPY"
_SESSION_ZONE = ZoneInfo("America/New_York")
_EUR_LOCATION = "EA19"
_USD_LOCATION = "USA"

# Frozen at audit time from a clean load of the currently-committed raw
# ECBDFR/EFFR CSVs (config/data/policy_rates). If a future production run's
# `load_policy_rate_history()` call yields a different canonical hash, the
# committed raw source data changed since this audit and the run must not
# proceed -- see `_check_policy_rate_data_hashes`.
FROZEN_POLICY_RATE_CANONICAL_DATA_SHA256 = (
    "c8a5091311ee539d86f21c2467752ca685a98ed3144ef92effce5c67f572651d"
)

# Frozen research account for this family's production runs -- identical in
# shape to the account used by the family's own synthetic native-engine
# tests, so the (never-invoked-here) real run and the tested engine wiring
# share the exact same account assumptions.
_FROZEN_ACCOUNT = AccountParameters(
    initial_capital=Decimal("100000"), currency="USD", leverage=Decimal("20")
)


class EurusdPolicyRateCarryProxyEvaluationError(ValueError):
    """Raised when the frozen production evaluation cannot be honored exactly."""


# ---------------------------------------------------------------------------
# Pure, engine-free logic: proxy-carry record construction, P&L decomposition,
# cost stress, and gate evaluation. All of these are unit-testable with
# fabricated inputs and never touch real market data.
# ---------------------------------------------------------------------------


def build_monthly_interest_rate_inputs(
    history: PolicyRateHistory,
    *,
    span_start: date,
    span_end: date,
) -> tuple[InterestRateInput, ...]:
    """Build monthly EA19/USA InterestRateInput records for FXRolloverInterestModule.

    For calendar month M spanned by ``[span_start, span_end]``, the record
    value for each currency's ``"YYYY-MM"`` key is that currency's raw rate
    causally known as of the last business day of month M-1, under the same
    one-business-day information lag frozen in
    :func:`ftmoquant.data.policy_rates.causal_differential_series`. This
    preserves no-future-information at the module's coarser native monthly
    resolution: no value dated after the lagged cutoff for month M-1 is ever
    read to construct month M's record.

    Raw ECBDFR/EFFR rates are used directly (not a precomputed differential)
    so that the module itself economically signs the financing component; no
    second carry calculation is performed here.
    """

    if span_start > span_end:
        raise EurusdPolicyRateCarryProxyEvaluationError(
            "span_start must not be after span_end"
        )
    months = sorted(_months_spanned(span_start, span_end))
    result: list[InterestRateInput] = []
    for year, month in months:
        prior_year, prior_month = (year, month - 1) if month > 1 else (year - 1, 12)
        last_business_day = _last_business_day_of_month(prior_year, prior_month)
        lag_cutoff = _prior_business_day(last_business_day)
        eur_rate = _causal_rate_before(history.ecbdfr, lag_cutoff)
        usd_rate = _causal_rate_before(history.effr, lag_cutoff)
        key = f"{year:04d}-{month:02d}"
        result.append(
            InterestRateInput(location=_EUR_LOCATION, time=key, value=eur_rate)
        )
        result.append(
            InterestRateInput(location=_USD_LOCATION, time=key, value=usd_rate)
        )
    return tuple(result)


def carry_proxy_execution_profile(
    base_profile: ExecutionProfile, records: Sequence[InterestRateInput]
) -> ExecutionProfile:
    """Return a new proxy-labelled ExecutionProfile with FX rollover wired in.

    Never mutates ``canonical_execution_profile()``; builds a distinct
    profile constant. ``calibration_status`` is always ``PROXY``, never
    ``BROKER_CALIBRATED``.
    """

    if not records:
        raise EurusdPolicyRateCarryProxyEvaluationError(
            "proxy carry profile requires at least one interest-rate record"
        )
    return ExecutionProfile(
        fill_on_limit_probability=base_profile.fill_on_limit_probability,
        adverse_slippage_probability=base_profile.adverse_slippage_probability,
        random_seed=base_profile.random_seed,
        base_latency_ns=base_profile.base_latency_ns,
        insert_latency_ns=base_profile.insert_latency_ns,
        update_latency_ns=base_profile.update_latency_ns,
        cancel_latency_ns=base_profile.cancel_latency_ns,
        fee=base_profile.fee,
        rollover=RolloverConfig(mode=RolloverMode.FX_INTEREST, records=tuple(records)),
        adaptive_high_low_ordering=base_profile.adaptive_high_low_ordering,
        calibration_status=CalibrationStatus.PROXY,
        broker_evidence_sha256=None,
    )


def uncalibrated_baseline_execution_profile() -> ExecutionProfile:
    """A distinct, family-owned uncalibrated baseline (mirrors G0.7 defaults)."""

    return ExecutionProfile(
        fill_on_limit_probability=Decimal(1),
        adverse_slippage_probability=Decimal(0),
        random_seed=7,
        base_latency_ns=0,
        insert_latency_ns=0,
        update_latency_ns=0,
        cancel_latency_ns=0,
        fee=FeeModelConfig(
            kind=FeeModelKind.FIXED,
            commission=Decimal(0),
            currency="USD",
            charge_commission_once=True,
        ),
        rollover=RolloverConfig(mode=RolloverMode.DISABLED),
        adaptive_high_low_ordering=False,
        calibration_status=CalibrationStatus.UNCALIBRATED,
    )


@dataclass(frozen=True, slots=True)
class DailyPnlDecomposition:
    """One eligible day's total return, split into three explicit components.

    ``spot_pnl`` is always the residual of the identity
    ``total_equity_change = spot_pnl + carry_accrual - execution_cost``,
    i.e. ``spot_pnl = total_equity_change - carry_accrual + execution_cost``.
    The identity is therefore an algebraic invariant of this constructor,
    not an incidental property -- see
    :func:`decompose_daily_pnl` and its unit tests.
    """

    as_of_date: date
    total_equity_change: Decimal
    carry_accrual: Decimal
    execution_cost: Decimal
    spot_pnl: Decimal

    def __post_init__(self) -> None:
        values = (
            self.total_equity_change,
            self.carry_accrual,
            self.execution_cost,
            self.spot_pnl,
        )
        if any(not value.is_finite() for value in values):
            raise EurusdPolicyRateCarryProxyEvaluationError(
                "P&L decomposition components must be finite"
            )
        if self.execution_cost < 0:
            raise EurusdPolicyRateCarryProxyEvaluationError(
                "execution_cost must be non-negative"
            )
        reconstructed = self.spot_pnl + self.carry_accrual - self.execution_cost
        if reconstructed != self.total_equity_change:
            raise EurusdPolicyRateCarryProxyEvaluationError(
                "P&L decomposition identity violated: "
                "total != spot + carry - cost"
            )

    @property
    def total_return(self) -> Decimal:
        return self.total_equity_change / BASE_RESEARCH_UNITS


def decompose_daily_pnl(
    as_of_date: date,
    *,
    total_equity_change: Decimal,
    carry_accrual: Decimal,
    execution_cost: Decimal,
) -> DailyPnlDecomposition:
    """Construct one day's decomposition; spot P&L is always the residual."""

    spot_pnl = total_equity_change - carry_accrual + execution_cost
    return DailyPnlDecomposition(
        as_of_date=as_of_date,
        total_equity_change=total_equity_change,
        carry_accrual=carry_accrual,
        execution_cost=execution_cost,
        spot_pnl=spot_pnl,
    )


def stressed_daily_total_return(
    decomposition: DailyPnlDecomposition,
    *,
    multiplier: Decimal = EXECUTION_COST_STRESS_MULTIPLIER,
) -> Decimal:
    """Apply the frozen 1.5x stress to the execution-cost component only.

    The proxy carry accrual is passed through completely unchanged -- this
    function never multiplies ``carry_accrual``.
    """

    return (
        decomposition.spot_pnl
        + decomposition.carry_accrual
        - multiplier * decomposition.execution_cost
    )


# ---------------------------------------------------------------------------
# Gate evaluation (Section 9). Deliberately does not invoke run_search,
# select_candidate, or assess_plateau -- this is a single baseline
# configuration evaluated directly against manually-checked gates.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GateEvaluation:
    valid_completion: bool
    all_folds_present: bool
    minimum_observations_met: bool
    pooled_mean_positive: bool
    pooled_stressed_mean_positive: bool
    majority_folds_positive: bool
    numerics_valid: bool
    pooled_observation_count: int
    pooled_mean_daily_total_return: float
    pooled_stressed_mean_daily_total_return: float
    positive_fold_count: int
    fold_count: int

    @property
    def passed(self) -> bool:
        return (
            self.valid_completion
            and self.all_folds_present
            and self.minimum_observations_met
            and self.pooled_mean_positive
            and self.pooled_stressed_mean_positive
            and self.majority_folds_positive
            and self.numerics_valid
        )


def evaluate_development_gates(
    family_id: str, family_version: str, fold_metrics: Sequence[FoldMetrics]
) -> GateEvaluation:
    """Manually check the frozen Section 9 gates for the single baseline trial."""

    all_folds_present = len(fold_metrics) == REQUIRED_FOLD_COUNT and len(
        {fold.fold_id for fold in fold_metrics}
    ) == REQUIRED_FOLD_COUNT
    numerics_valid = all(
        math.isfinite(fold.net_expectancy)
        and math.isfinite(fold.cost_stressed_expectancy)
        for fold in fold_metrics
    )
    if not fold_metrics or not numerics_valid:
        return GateEvaluation(
            valid_completion=bool(fold_metrics),
            all_folds_present=all_folds_present,
            minimum_observations_met=False,
            pooled_mean_positive=False,
            pooled_stressed_mean_positive=False,
            majority_folds_positive=False,
            numerics_valid=numerics_valid,
            pooled_observation_count=0,
            pooled_mean_daily_total_return=0.0,
            pooled_stressed_mean_daily_total_return=0.0,
            positive_fold_count=0,
            fold_count=len(fold_metrics),
        )
    record = make_trial_record(
        family_id=family_id,
        family_version=family_version,
        parameters={"baseline": True},
        status=TrialStatus.COMPLETE,
        folds=fold_metrics,
    )
    summary = summarize_trial(record)
    return GateEvaluation(
        valid_completion=True,
        all_folds_present=all_folds_present,
        minimum_observations_met=(
            summary.trade_count >= MINIMUM_POOLED_ELIGIBLE_OBSERVATIONS
        ),
        pooled_mean_positive=summary.pooled_expectancy > 0.0,
        pooled_stressed_mean_positive=summary.cost_stressed_expectancy > 0.0,
        majority_folds_positive=summary.positive_fold_count >= MINIMUM_POSITIVE_FOLDS,
        numerics_valid=numerics_valid,
        pooled_observation_count=summary.trade_count,
        pooled_mean_daily_total_return=summary.pooled_expectancy,
        pooled_stressed_mean_daily_total_return=summary.cost_stressed_expectancy,
        positive_fold_count=summary.positive_fold_count,
        fold_count=summary.fold_count,
    )


def fold_metrics_from_daily_decompositions(
    fold_id: str, decompositions: Sequence[DailyPnlDecomposition]
) -> FoldMetrics:
    """Aggregate one fold's eligible daily decompositions into FoldMetrics.

    ``trade_count`` is the eligible daily return observation count -- not a
    trade count. See the module docstring.
    """

    if not decompositions:
        raise EurusdPolicyRateCarryProxyEvaluationError(
            f"{fold_id} has no eligible daily return observations"
        )
    returns = tuple(float(item.total_return) for item in decompositions)
    stressed_returns = tuple(
        float(stressed_daily_total_return(item) / BASE_RESEARCH_UNITS)
        for item in decompositions
    )
    count = len(decompositions)
    net_expectancy = sum(returns) / count
    stressed_expectancy = sum(stressed_returns) / count
    yearly: dict[str, float] = {}
    regime: dict[str, float] = {}
    for item, daily_return in zip(decompositions, returns, strict=True):
        year_label = str(item.as_of_date.year)
        yearly[year_label] = yearly.get(year_label, 0.0) + daily_return
    return FoldMetrics(
        fold_id=fold_id,
        trade_count=count,
        net_expectancy=net_expectancy,
        sharpe=_annualized_sharpe(returns) or 0.0,
        drawdown=_maximum_drawdown(returns),
        turnover=0.0,
        cost_stressed_expectancy=stressed_expectancy,
        net_daily_mean=net_expectancy,
        yearly_attribution=tuple(
            Attribution(label, value) for label, value in sorted(yearly.items())
        ),
        regime_attribution=tuple(
            Attribution(label, value) for label, value in sorted(regime.items())
        ),
        execution_perturbed_expectancy=None,
    )


def component_contributions(
    decompositions: Sequence[DailyPnlDecomposition],
) -> dict[str, float]:
    """Diagnostic pooled contribution of each P&L component, in return units."""

    if not decompositions:
        raise EurusdPolicyRateCarryProxyEvaluationError(
            "component contributions require at least one observation"
        )
    return {
        "spot_pnl_contribution": float(
            sum(item.spot_pnl for item in decompositions) / BASE_RESEARCH_UNITS
        ),
        "carry_accrual_contribution": float(
            sum(item.carry_accrual for item in decompositions) / BASE_RESEARCH_UNITS
        ),
        "execution_cost_contribution": float(
            -sum(item.execution_cost for item in decompositions) / BASE_RESEARCH_UNITS
        ),
    }


def carry_contribution_ratio(
    decompositions: Sequence[DailyPnlDecomposition],
) -> float | None:
    """Diagnostic-only pooled carry/abs-total attribution ratio.

    ``carry_contribution_ratio = pooled_carry_accrual / (abs(pooled_spot_pnl)
    + abs(pooled_carry_accrual) + abs(pooled_execution_cost))``.

    This is purely an additional reported number: it is never used in any
    pass/fail gate decision, never used to build a spot-only competing
    model, selector, or alternate ranking. Returns ``None`` (not NaN) when
    the denominator is exactly zero, following this family's existing
    ``FoldMetrics.execution_perturbed_expectancy: None`` convention for
    "not computable" rather than a NaN sentinel.
    """

    if not decompositions:
        raise EurusdPolicyRateCarryProxyEvaluationError(
            "carry contribution ratio requires at least one observation"
        )
    pooled_spot = sum((item.spot_pnl for item in decompositions), Decimal(0))
    pooled_carry = sum((item.carry_accrual for item in decompositions), Decimal(0))
    pooled_cost = sum((item.execution_cost for item in decompositions), Decimal(0))
    denominator = abs(pooled_spot) + abs(pooled_carry) + abs(pooled_cost)
    if denominator == 0:
        return None
    return float(pooled_carry / denominator)


def rate_regime_attribution(
    decompositions: Sequence[DailyPnlDecomposition],
    signal_by_date: Mapping[date, RawDirectionalTarget],
) -> tuple[Attribution, ...]:
    """Diagnostic net-return contribution grouped by differential-sign regime."""

    regime: dict[str, float] = {}
    for item in decompositions:
        target = signal_by_date.get(item.as_of_date, RawDirectionalTarget.FLAT)
        label = target.name
        regime[label] = regime.get(label, 0.0) + float(item.total_return)
    return tuple(Attribution(label, value) for label, value in sorted(regime.items()))


# ---------------------------------------------------------------------------
# Strategy family (single baseline configuration only).
# ---------------------------------------------------------------------------


class EurusdPolicyRateCarryProxyFamily(
    StrategyFamily[Sequence[tuple[int, Decimal]], tuple[Any, ...]]
):
    """The frozen single-configuration ``eurusd_policy_rate_carry_proxy_v1`` family."""

    def __init__(self, spec: EurusdPolicyRateCarryProxySpec | None = None) -> None:
        self.spec = (
            load_eurusd_policy_rate_carry_proxy_spec() if spec is None else spec
        )
        self._parameter_space = ParameterSpace(
            (CategoricalParameter("baseline", (True,)),)
        )

    @property
    def metadata(self) -> FamilyMetadata:
        return FamilyMetadata(
            family_id=self.spec.family_id,
            version=self.spec.version,
            economic_hypothesis=self.spec.economic_hypothesis,
            supported_timeframes=("1-DAY",),
            eligible_sessions=(SessionId.ALL,),
        )

    @property
    def parameter_space(self) -> ParameterSpace:
        return self._parameter_space

    def create_state(
        self, parameters: ParameterMapping
    ) -> EurusdPolicyRateCarryProxyState:
        self.validate_parameters(parameters)
        return EurusdPolicyRateCarryProxyState()

    def build_signals(
        self,
        data: Sequence[tuple[int, Decimal]],
        parameters: ParameterMapping,
    ) -> tuple[Any, ...]:
        state = self.create_state(parameters)
        result = []
        for _event_ns, differential in data:
            result.append(state.target_for(date.today(), differential))
        return tuple(result)


# ---------------------------------------------------------------------------
# Native Nautilus fold execution: signal + causal vol sizing outside the
# engine, execution + proxy carry accrual + equity marking inside it.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CarryProxyInstruction:
    """One day's decision, resolved to an executable native order instruction."""

    decision_information_ns: int
    execution_event_ns: int
    execution_information_ns: int
    raw_target: RawDirectionalTarget
    target_base_units: Decimal
    execution_midpoint: Decimal
    as_of_date: date


def build_carry_proxy_instructions(
    *,
    minute_execution_frames: Sequence[tuple[int, int, Decimal]],
    daily_midpoints: Sequence[CompletedDailyMidpoint],
    differentials: Sequence[Any],
    evaluate_start_ns: int,
    evaluate_end_exclusive_ns: int,
) -> tuple[CarryProxyInstruction, ...]:
    """Build the fold's execution instructions from a 00:00 UTC daily anchor.

    ``differentials`` is a sequence of
    :class:`ftmoquant.data.policy_rates.CausalDifferentialObservation`. Only
    a decision whose 00:00 UTC information time falls in
    ``[evaluate_start_ns, evaluate_end_exclusive_ns)`` produces an
    instruction, so no pre-fold P&L can ever enter the fold's evaluation
    bookkeeping: no instruction, no order, no exposure, no P&L.
    """

    normalizer = G1VolatilityNormalizer()
    returns = completed_daily_log_returns(daily_midpoints)
    estimator = CausalEwmaDailyVolatility(returns)
    signal_state = EurusdPolicyRateCarryProxyState()
    frames_by_ns = sorted(minute_execution_frames, key=lambda item: item[1])
    result: list[CarryProxyInstruction] = []
    for observation in sorted(differentials, key=lambda item: item.as_of_date):
        decision_ns = _midnight_utc_ns(observation.as_of_date)
        if not (evaluate_start_ns <= decision_ns < evaluate_end_exclusive_ns):
            continue
        directional = signal_state.target_for(
            observation.as_of_date, observation.differential_percent
        )
        executable = _first_frame_strictly_after(frames_by_ns, decision_ns)
        if executable is None:
            continue
        execution_event_ns, execution_information_ns, midpoint = executable
        decision = normalizer.target_at(
            float(directional.target), decision_ns, estimator
        )
        target_units = decision.target_base_units(BASE_RESEARCH_UNITS).quantize(
            Decimal("0.00000001")
        )
        result.append(
            CarryProxyInstruction(
                decision_information_ns=decision_ns,
                execution_event_ns=execution_event_ns,
                execution_information_ns=execution_information_ns,
                raw_target=directional.target,
                target_base_units=target_units,
                execution_midpoint=midpoint,
                as_of_date=observation.as_of_date,
            )
        )
    if len({item.execution_event_ns for item in result}) != len(result):
        raise EurusdPolicyRateCarryProxyEvaluationError(
            "multiple daily decisions mapped to one execution frame"
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class _EquityMark:
    information_time_ns: int
    equity: Decimal
    cumulative_cost: Decimal


class _PolicyRateCarryProxyExecutor(Strategy):
    """Executes precomputed daily instructions and brackets the 17:00 ET accrual.

    Decisions are anchored at 00:00 UTC (never 17:00 ET) specifically so that
    the strategy's own order flow never coincides with
    ``FXRolloverInterestModule``'s fixed 17:00 America/New_York accrual
    instant; this keeps the equity-diff bracketing below causally clean.
    """

    def __new__(
        cls,
        instructions: Sequence[CarryProxyInstruction],
        instrument: CurrencyPair,
        start_ns: int,
        end_exclusive_ns: int,
    ) -> _PolicyRateCarryProxyExecutor:
        del instructions, instrument, start_ns, end_exclusive_ns
        return super().__new__(cls)

    def __init__(
        self,
        instructions: Sequence[CarryProxyInstruction],
        instrument: CurrencyPair,
        start_ns: int,
        end_exclusive_ns: int,
    ) -> None:
        super().__init__(
            StrategyConfig(
                strategy_id=StrategyId("EURUSD-POLICY-RATE-CARRY-PROXY-V1-DEV-001"),
                log_events=False,
                log_commands=False,
            )
        )
        self._instructions = {item.execution_event_ns: item for item in instructions}
        if len(self._instructions) != len(instructions):
            raise EurusdPolicyRateCarryProxyEvaluationError(
                "duplicate execution instruction"
            )
        self._instrument = instrument
        self._start_ns = start_ns
        self._end_exclusive_ns = end_exclusive_ns
        self._target_units = Decimal(0)
        self.realized_variable_cost = Decimal(0)
        self._bars: dict[int, dict[str, Bar]] = {}
        self._last_pair: tuple[int, Price, Price] | None = None
        self._pre_mark: _EquityMark | None = None
        self._order_context: dict[str, CarryProxyInstruction] = {}
        self.equity_marks: list[_EquityMark] = []

    def on_start(self) -> None:
        for side in ("BID", "ASK"):
            self.subscribe_bars(_bar_type(f"{_EURUSD}-1-MINUTE-{side}-EXTERNAL"))

    def on_bar(self, bar: Bar) -> None:
        if bar.ts_init < self._start_ns or bar.ts_init >= self._end_exclusive_ns:
            return
        side = bar.bar_type.spec.price_type.name
        frame = self._bars.setdefault(bar.ts_event, {})
        if side in frame:
            raise RuntimeError("duplicate native side bar")
        frame[side] = bar
        if set(frame) != {"BID", "ASK"}:
            return
        bid = frame["BID"].close
        ask = frame["ASK"].close
        info = max(frame["BID"].ts_init, frame["ASK"].ts_init)
        self._last_pair = (info, bid, ask)
        instruction = self._instructions.get(bar.ts_event)
        if instruction is not None:
            if info != instruction.execution_information_ns:
                raise RuntimeError("execution information timestamp drifted")
            self._submit_target(instruction)
        local = _datetime(info).astimezone(_SESSION_ZONE)
        if local.weekday() <= 4 and local.hour == 16 and local.minute == 59:
            self.clock.set_time_alert_ns(
                name=f"eurusd-carry-proxy-pre-{bar.ts_event}",
                alert_time_ns=info + 1,
                callback=self._on_pre_mark,
            )
        if local.weekday() <= 4 and local.hour == 17 and local.minute == 0:
            self.clock.set_time_alert_ns(
                name=f"eurusd-carry-proxy-post-{bar.ts_event}",
                alert_time_ns=info + 1,
                callback=self._on_post_mark,
            )
        del self._bars[bar.ts_event]

    def on_order_filled(self, event: OrderFilled) -> None:
        instruction = self._order_context.get(event.client_order_id.value)
        if instruction is None:
            raise RuntimeError("native fill lacks scaled instruction context")
        quantity = event.last_qty.as_decimal()
        fill = event.last_px.as_decimal()
        spread = (
            quantity * (fill - instruction.execution_midpoint)
            if event.order_side is OrderSide.BUY
            else quantity * (instruction.execution_midpoint - fill)
        )
        commission = (
            Decimal(0)
            if event.commission is None
            else abs(event.commission.as_decimal())
        )
        cost = spread + commission
        if cost < 0:
            raise RuntimeError("native fill improved beyond execution midpoint")
        self.realized_variable_cost += cost

    def _submit_target(self, instruction: CarryProxyInstruction) -> None:
        delta = instruction.target_base_units - self._target_units
        if delta == 0:
            return
        if abs(instruction.target_base_units) > Decimal("3000000"):
            raise RuntimeError("frozen EUR currency-exposure limit breached")
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        quantity = self._instrument.make_qty(float(abs(delta)))
        order = self.order_factory.market(
            instrument_id=InstrumentId.from_str(_EURUSD),
            order_side=side,
            quantity=quantity,
            tags=(
                "eurusd_policy_rate_carry_proxy_v1",
                f"decision_information_ns={instruction.decision_information_ns}",
                f"raw_target={int(instruction.raw_target)}",
            ),
        )
        self._order_context[order.client_order_id.value] = instruction
        self._target_units = instruction.target_base_units
        self.submit_order(order)

    def _on_pre_mark(self, _: Any) -> None:
        if self._last_pair is not None:
            self._pre_mark = self._equity_mark(*self._last_pair)

    def _on_post_mark(self, _: Any) -> None:
        if self._last_pair is not None and self._pre_mark is not None:
            post = self._equity_mark(*self._last_pair)
            self.equity_marks.append(self._pre_mark)
            self.equity_marks.append(post)
            self._pre_mark = None

    def _equity_mark(self, info: int, bid: Price, ask: Price) -> _EquityMark:
        account = self.cache.account_for_venue(self._instrument.id.venue)
        if account is None:
            raise RuntimeError("native account unavailable")
        balance = account.balance_total(Currency.from_str("USD"))
        if balance is None:
            raise RuntimeError("native USD balance unavailable")
        equity = balance.as_decimal()
        for position in cast(list[Position], self.cache.positions_open()):
            mark = bid if position.is_long else ask
            equity += position.unrealized_pnl(mark).as_decimal()
        return _EquityMark(info, equity, self.realized_variable_cost)

    def on_stop(self) -> None:
        return


def daily_decompositions_from_equity_marks(
    marks: Sequence[_EquityMark],
) -> tuple[DailyPnlDecomposition, ...]:
    """Turn (pre, post) equity-mark pairs into daily P&L decompositions.

    ``marks`` must be an even-length sequence of alternating (pre, post)
    marks in chronological order, one pair per weekday. Consecutive *post*
    marks form the daily total-equity-change boundary
    (existing 17:00 America/New_York completed-pair semantics); the gap
    between each day's own pre/post pair isolates that day's carry accrual.
    """

    if len(marks) % 2 != 0:
        raise EurusdPolicyRateCarryProxyEvaluationError(
            "equity marks must be paired (pre, post) per day"
        )
    pairs = [(marks[i], marks[i + 1]) for i in range(0, len(marks), 2)]
    result: list[DailyPnlDecomposition] = []
    for index in range(1, len(pairs)):
        previous_post = pairs[index - 1][1]
        pre, post = pairs[index]
        carry_accrual = post.equity - pre.equity
        total_equity_change = post.equity - previous_post.equity
        execution_cost = post.cumulative_cost - previous_post.cumulative_cost
        result.append(
            decompose_daily_pnl(
                _datetime(post.information_time_ns).astimezone(_SESSION_ZONE).date(),
                total_equity_change=total_equity_change,
                carry_accrual=carry_accrual,
                execution_cost=execution_cost,
            )
        )
    return tuple(result)


def _run_native_fold(
    *,
    fold_id: str,
    instrument: CurrencyPair,
    minute_bars: Sequence[Bar],
    instructions: Sequence[CarryProxyInstruction],
    account: Any,
    profile: ExecutionProfile,
    start_ns: int,
    end_exclusive_ns: int,
) -> tuple[DailyPnlDecomposition, ...]:
    if version("nautilus_trader") != NAUTILUS_VERSION:
        raise RuntimeError("NautilusTrader version drifted")
    identity = _sha256_json(
        {
            "family_semantic_sha256": EURUSD_POLICY_RATE_CARRY_PROXY_SEMANTIC_SHA256,
            "fold_id": fold_id,
            "instructions": [asdict(item) for item in instructions],
        }
    )
    engine = BacktestEngine(_engine_config(identity))
    executor = _PolicyRateCarryProxyExecutor(
        instructions, instrument, start_ns, end_exclusive_ns
    )
    try:
        _add_venue(
            engine,
            account,
            profile,
            ProbabilisticFillModel(
                prob_fill_on_limit=float(profile.fill_on_limit_probability),
                prob_slippage=float(profile.adverse_slippage_probability),
                random_seed=profile.random_seed,
            ),
            StaticLatencyModel(
                base_latency_nanos=profile.base_latency_ns,
                insert_latency_nanos=profile.insert_latency_ns,
                update_latency_nanos=profile.update_latency_ns,
                cancel_latency_nanos=profile.cancel_latency_ns,
            ),
            _fee_model(profile.fee),
            _rollover_modules(profile.rollover),
        )
        engine.add_instrument(instrument)
        engine.add_data(list(minute_bars), validate=True, sort=True)
        engine.add_strategy(executor)
        engine.run(
            start=start_ns,
            end=end_exclusive_ns - 1,
            run_config_id=f"G1CARRY-{identity[:20]}",
        )
        return daily_decompositions_from_equity_marks(tuple(executor.equity_marks))
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _months_spanned(start: date, end: date) -> set[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        result.add((cursor.year, cursor.month))
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    return result


def _last_business_day_of_month(year: int, month: int) -> date:
    last_day = calendar.monthrange(year, month)[1]
    cursor = date(year, month, last_day)
    while cursor.weekday() >= 5:
        cursor -= timedelta(days=1)
    return cursor


def _prior_business_day(value: date) -> date:
    cursor = value - timedelta(days=1)
    while cursor.weekday() >= 5:
        cursor -= timedelta(days=1)
    return cursor


def _causal_rate_before(rates: Sequence[DatedRate], cutoff: date) -> Decimal:
    ordered = sorted(rates, key=lambda item: item.observation_date)
    result: Decimal | None = None
    for item in ordered:
        if item.observation_date <= cutoff:
            result = item.value_percent
        else:
            break
    if result is None:
        raise PolicyRateProvenanceError(
            f"no causal rate observation is available on or before {cutoff}"
        )
    return result


def _midnight_utc_ns(value: date) -> int:
    combined = datetime.combine(value, datetime.min.time(), tzinfo=UTC)
    return int(combined.timestamp() * 1_000_000_000)


def _first_frame_strictly_after(
    frames: Sequence[tuple[int, int, Decimal]], decision_ns: int
) -> tuple[int, int, Decimal] | None:
    for event_ns, information_ns, midpoint in frames:
        if information_ns > decision_ns:
            return event_ns, information_ns, midpoint
    return None


def _bar_type(value: str) -> Any:
    from nautilus_trader.model import BarType

    return BarType.from_str(value)


def _datetime(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1_000_000_000, tz=UTC)


def _sha256_json(value: Any) -> str:
    import json

    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


# ---------------------------------------------------------------------------
# Real-data orchestrator. Deliberately never invoked in this task.
#
# Every check below is independently unit-testable in isolation (with
# mocked/patched inputs, never real DEVELOPMENT/validation/holdout data) and
# is called, in order, by `_run_preflight_checks` before any catalog is
# opened. `run_eurusd_policy_rate_carry_proxy_development` itself is never
# invoked anywhere in this task -- see tests/research/
# test_eurusd_policy_rate_carry_proxy_development.py for the preflight-only
# unit tests and the explicit "runner refuses on a dirty worktree / stale
# hash / etc." coverage.
# ---------------------------------------------------------------------------


def _check_semantic_sha(spec_path: Path) -> EurusdPolicyRateCarryProxySpec:
    """Check 1: load the spec; the loader itself enforces the frozen SHA."""

    return load_eurusd_policy_rate_carry_proxy_spec(spec_path)


def _check_clean_worktree() -> None:
    """Check 2: refuse to run against any uncommitted change, staged or not."""

    completed = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout.strip():
        raise EurusdPolicyRateCarryProxyEvaluationError(
            "production runner requires a clean git worktree before "
            "executing -- results must be attributable to one exact commit"
        )


def _check_family_identity(spec: EurusdPolicyRateCarryProxySpec) -> None:
    """Check 3: confirm the loaded spec is this family, not another one."""

    if spec.family_id != "eurusd_policy_rate_carry_proxy_v1":
        raise EurusdPolicyRateCarryProxyEvaluationError(
            f"loaded spec family_id {spec.family_id!r} is not "
            "eurusd_policy_rate_carry_proxy_v1"
        )


def _check_development_folds(spec: EurusdPolicyRateCarryProxySpec) -> None:
    """Check 4: defense-in-depth fold-identity check independent of the loader."""

    folds = frozen_development_folds()
    declared = _mapping(spec.canonical_document["development_folds"])
    if (
        declared.get("version") != folds.version
        or declared.get("semantic_sha256") != folds.semantic_sha256
    ):
        raise EurusdPolicyRateCarryProxyEvaluationError(
            "spec DEVELOPMENT fold identity does not match "
            "frozen_development_folds(); refusing to run against the wrong folds"
        )


def _check_policy_rate_data_hashes(policy_rates_dir: Path) -> PolicyRateHistory:
    """Check 5: load+hash-verify rate history and pin against an audited value."""

    history = load_policy_rate_history(policy_rates_dir)
    if history.canonical_data_sha256 != FROZEN_POLICY_RATE_CANONICAL_DATA_SHA256:
        raise EurusdPolicyRateCarryProxyEvaluationError(
            "policy rate canonical_data_sha256 does not match the value "
            "pinned at audit time; the committed raw CSVs changed since "
            "that audit and this run must not proceed"
        )
    return history


def _check_execution_profile_config() -> None:
    """Check 6: the family's execution profile is proxy carry, never disabled/broker."""

    profile = carry_proxy_execution_profile(
        uncalibrated_baseline_execution_profile(),
        records=(
            InterestRateInput(_EUR_LOCATION, "2000-01", Decimal(0)),
            InterestRateInput(_USD_LOCATION, "2000-01", Decimal(0)),
        ),
    )
    if (
        profile.calibration_status is not CalibrationStatus.PROXY
        or profile.rollover.mode is not RolloverMode.FX_INTEREST
    ):
        raise EurusdPolicyRateCarryProxyEvaluationError(
            "carry_proxy_execution_profile calibration/rollover "
            "configuration drifted from the frozen proxy contract"
        )


def _check_output_not_exists(output_dir: Path) -> None:
    """Check 7: refuse to silently overwrite an existing output directory."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise EurusdPolicyRateCarryProxyEvaluationError(
            f"output already exists and is non-empty: {output_dir}"
        )


_SEALED_PATH_MARKERS = ("validation", "holdout", "final_holdout", "final-holdout")


def _check_partition_bounds(
    spec: EurusdPolicyRateCarryProxySpec,
    *,
    universe_readiness_path: Path,
    development_root: Path,
    policy_rates_dir: Path,
    output_dir: Path,
) -> None:
    """Check 8: refuse anything implying validation/final-holdout access."""

    dataset = _mapping(spec.canonical_document["dataset"])
    dev_end = _parse_utc(cast(str, dataset["development_end_exclusive_utc"]))
    if dev_end > VALIDATION_START or dev_end > HOLDOUT_START:
        raise EurusdPolicyRateCarryProxyEvaluationError(
            "spec DEVELOPMENT boundary reaches into validation/final-holdout"
        )
    for path in (
        universe_readiness_path,
        development_root,
        policy_rates_dir,
        output_dir,
    ):
        lowered = tuple(part.casefold() for part in path.parts)
        if any(marker in part for marker in _SEALED_PATH_MARKERS for part in lowered):
            raise EurusdPolicyRateCarryProxyEvaluationError(
                f"path implies validation/final-holdout access: {path}"
            )


def _run_preflight_checks(
    *,
    spec_path: Path,
    universe_readiness_path: Path,
    development_root: Path,
    policy_rates_dir: Path,
    output_dir: Path,
) -> tuple[EurusdPolicyRateCarryProxySpec, PolicyRateHistory]:
    """Run every preflight check, in order, before any catalog is opened."""

    spec = _check_semantic_sha(spec_path)
    _check_clean_worktree()
    _check_family_identity(spec)
    _check_development_folds(spec)
    history = _check_policy_rate_data_hashes(policy_rates_dir)
    _check_execution_profile_config()
    _check_output_not_exists(output_dir)
    _check_partition_bounds(
        spec,
        universe_readiness_path=universe_readiness_path,
        development_root=development_root,
        policy_rates_dir=policy_rates_dir,
        output_dir=output_dir,
    )
    return spec, history


def run_eurusd_policy_rate_carry_proxy_development(
    *,
    spec_path: Path,
    universe_readiness_path: Path,
    development_root: Path,
    policy_rates_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Evaluate the single frozen baseline against the 3 real DEVELOPMENT folds.

    THIS FUNCTION IS NOT INVOKED ANYWHERE IN THIS TASK. Every preflight
    check above runs, in order, before any catalog is opened; the first
    failure raises and no market/rate data beyond the already-committed,
    hash-verified ``config/data/policy_rates`` source is ever touched.
    Calling it for real is explicitly out of scope for this preregistration
    task and has not happened here.
    """

    spec, history = _run_preflight_checks(
        spec_path=spec_path,
        universe_readiness_path=universe_readiness_path,
        development_root=development_root,
        policy_rates_dir=policy_rates_dir,
        output_dir=output_dir,
    )

    universe = load_frozen_universe(universe_readiness_path)
    dataset = _mapping(spec.canonical_document["dataset"])
    if universe.readiness_sha256 != dataset["universe_readiness_sha256"]:
        raise EurusdPolicyRateCarryProxyEvaluationError(
            "loaded universe readiness SHA-256 does not match the spec's "
            "declared universe_readiness_sha256"
        )

    span_start = DEVELOPMENT_START.date()
    span_end = (DEVELOPMENT_END_EXCLUSIVE - timedelta(days=1)).date()
    instrument, minute_bars = _load_eurusd_development_minute_bars(
        development_root, DEVELOPMENT_START, DEVELOPMENT_END_EXCLUSIVE
    )
    records = build_monthly_interest_rate_inputs(
        history, span_start=span_start, span_end=span_end
    )
    profile = carry_proxy_execution_profile(
        uncalibrated_baseline_execution_profile(), records=records
    )
    differentials = causal_differential_series(
        history, start_inclusive=span_start, end_inclusive=span_end
    )
    frames = _minute_execution_frames(minute_bars)
    daily_midpoints = _daily_midpoints_from_minute_bars(minute_bars)

    folds = frozen_development_folds()
    fold_metrics: list[FoldMetrics] = []
    all_decompositions: list[DailyPnlDecomposition] = []
    for fold in folds.folds:
        instructions = build_carry_proxy_instructions(
            minute_execution_frames=frames,
            daily_midpoints=daily_midpoints,
            differentials=differentials,
            evaluate_start_ns=_ns(fold.compare_start_utc),
            evaluate_end_exclusive_ns=_ns(fold.compare_end_exclusive_utc),
        )
        decompositions = _run_native_fold(
            fold_id=fold.fold_id,
            instrument=instrument,
            minute_bars=minute_bars,
            instructions=instructions,
            account=_FROZEN_ACCOUNT,
            profile=profile,
            start_ns=_ns(fold.train_start_utc),
            end_exclusive_ns=_ns(fold.compare_end_exclusive_utc),
        )
        fold_metrics.append(
            fold_metrics_from_daily_decompositions(fold.fold_id, decompositions)
        )
        all_decompositions.extend(decompositions)

    gate_evaluation = evaluate_development_gates(
        spec.family_id, spec.version, fold_metrics
    )
    contributions = component_contributions(all_decompositions)
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "family_id": spec.family_id,
        "family_version": spec.version,
        "family_semantic_sha256": spec.semantic_sha256,
        "git_commit": _git_commit(),
        "gate_passed": gate_evaluation.passed,
        "pooled_observation_count": gate_evaluation.pooled_observation_count,
        "pooled_mean_daily_total_return": (
            gate_evaluation.pooled_mean_daily_total_return
        ),
        "pooled_stressed_mean_daily_total_return": (
            gate_evaluation.pooled_stressed_mean_daily_total_return
        ),
        "positive_fold_count": gate_evaluation.positive_fold_count,
        "fold_count": gate_evaluation.fold_count,
        **contributions,
        "carry_contribution_ratio": carry_contribution_ratio(all_decompositions),
        "validation_accessed": False,
        "final_holdout_accessed": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        result, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    (output_dir / "development_result.json").write_bytes(encoded + b"\n")
    return result


def _load_eurusd_development_minute_bars(
    development_root: Path, start: datetime, end_exclusive: datetime
) -> tuple[CurrencyPair, tuple[Bar, ...]]:
    """Query the frozen EUR/USD one-minute BID/ASK DEVELOPMENT stream once."""

    catalog = ParquetDataCatalog(str(development_root / "catalog"))
    found = catalog.instruments([_EURUSD])
    if len(found) != 1 or not isinstance(found[0], CurrencyPair):
        raise EurusdPolicyRateCarryProxyEvaluationError(
            "frozen EUR/USD CurrencyPair is unavailable in the DEVELOPMENT catalog"
        )
    profile = uncalibrated_baseline_execution_profile()
    instrument = _instrument_for_profile(found[0], profile.fee)
    start_ns = _ns(start)
    end_ns = _ns(end_exclusive)
    bars: list[Bar] = []
    for side in ("BID", "ASK"):
        bar_type = f"{_EURUSD}-1-MINUTE-{side}-EXTERNAL"
        queried = tuple(catalog.query_bars([bar_type], start=start_ns, end=end_ns - 1))
        if not queried:
            raise EurusdPolicyRateCarryProxyEvaluationError(
                f"DEVELOPMENT has no {bar_type} bars"
            )
        bars.extend(queried)
    ordered = tuple(
        sorted(bars, key=lambda bar: (bar.ts_event, str(bar.bar_type)))
    )
    return instrument, ordered


def _minute_execution_frames(
    minute_bars: Sequence[Bar],
) -> tuple[tuple[int, int, Decimal], ...]:
    """Pair BID/ASK one-minute bars by event time into (event, info, mid) frames."""

    by_event: dict[int, dict[str, Bar]] = {}
    for bar in minute_bars:
        side = bar.bar_type.spec.price_type.name
        if side not in {"BID", "ASK"}:
            raise EurusdPolicyRateCarryProxyEvaluationError(
                "minute execution bar side is invalid"
            )
        sides = by_event.setdefault(bar.ts_event, {})
        if side in sides:
            raise EurusdPolicyRateCarryProxyEvaluationError(
                "duplicate minute execution side"
            )
        sides[side] = bar
    result: list[tuple[int, int, Decimal]] = []
    for event in sorted(by_event):
        sides = by_event[event]
        if set(sides) != {"BID", "ASK"}:
            continue
        info = max(sides["BID"].ts_init, sides["ASK"].ts_init)
        midpoint = (
            sides["BID"].close.as_decimal() + sides["ASK"].close.as_decimal()
        ) / Decimal(2)
        result.append((event, info, midpoint))
    return tuple(result)


def _daily_midpoints_from_minute_bars(
    minute_bars: Sequence[Bar],
) -> tuple[CompletedDailyMidpoint, ...]:
    """Extract the existing 17:00 America/New_York completed-pair daily midpoint."""

    result: list[CompletedDailyMidpoint] = []
    for _, information, midpoint in _minute_execution_frames(minute_bars):
        local = _datetime(information).astimezone(_SESSION_ZONE)
        if local.weekday() <= 4 and local.hour == 17 and local.minute == 0:
            result.append(CompletedDailyMidpoint(information, float(midpoint)))
    return tuple(result)


def _mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EurusdPolicyRateCarryProxyEvaluationError("expected a mapping")
    return cast(dict[str, Any], value)


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _ns(value: datetime) -> int:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise EurusdPolicyRateCarryProxyEvaluationError(
            "timestamp must be timezone-aware UTC"
        )
    return int(value.timestamp() * 1_000_000_000)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the frozen eurusd_policy_rate_carry_proxy_v1 single "
            "DEVELOPMENT baseline (NOT authorized to run in this task)"
        )
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--universe-readiness", type=Path, required=True)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--policy-rates-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_eurusd_policy_rate_carry_proxy_development(
        spec_path=cast(Path, args.spec),
        universe_readiness_path=cast(Path, args.universe_readiness),
        development_root=cast(Path, args.development_root),
        policy_rates_dir=cast(Path, args.policy_rates_dir),
        output_dir=cast(Path, args.output),
    )


if __name__ == "__main__":
    main()
