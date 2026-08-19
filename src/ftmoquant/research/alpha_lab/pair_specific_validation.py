"""One-shot, preregistered, PAIR-SPECIFIC VALIDATION stage for the three
frozen DEVELOPMENT candidates that survived their own screens:

- Candidate A: F2 fresh-FVG/order-block mitigation, USD/CAD.OANDA, H4,
  ``backward_search_n=10`` (see :mod:`~ftmoquant.research.alpha_lab.
  wick_fvg_squeeze_screen`).
- Candidate B: F3 volatility-squeeze breakout, USD/JPY.OANDA, H1,
  ``bb_width_threshold=0.004`` (same module).
- Candidate C: B2F1 sweep -> BOS -> retest, USD/CAD.OANDA, M30,
  ``swing_lookback=40``, ``rr=2.0`` (see :mod:`~ftmoquant.research.alpha_lab.
  liquidity_structure_screen`).

Nothing here re-derives signal or execution mechanics. The three signal
functions (:func:`~ftmoquant.research.alpha_lab.wick_fvg_squeeze_signals.
f2_fresh_fvg_mitigation_signals`, ``f3_volatility_squeeze_breakout_signals``,
:func:`~ftmoquant.research.alpha_lab.liquidity_structure_signals.
b2f1_sweep_bos_retest_signals``) and the M1 BID/ASK trade-lifecycle engine
(:func:`~ftmoquant.research.alpha_lab.wick_fvg_squeeze_execution.
load_m1_bidask`, ``simulate_trades``) are imported and used unchanged -- this
module only wires them to the frozen VALIDATION partition and computes the
two frozen gates. The VALIDATION-partition readiness contract
(:func:`~ftmoquant.research.alpha_lab.validation._verify_validation_readiness`)
and the per-instrument mid-OHLC loader
(:func:`~ftmoquant.research.alpha_lab.validation._load_validation_instrument_frame`)
are reused from the existing (aggregate, multi-family) VALIDATION module
rather than re-implemented, since both are already generic over instrument
and timeframe.

Unlike :mod:`~ftmoquant.research.alpha_lab.validation` (which validates
broad-FX, aggregate-across-seven-pairs representatives with five gates), each
candidate here is a single, named pair -- so there is no cross-pair breadth
gate, no aggregate portfolio, and no concentration guard. Only two gates are
frozen (section 6 of the task spec): native-spread net_return > 0 and
native-spread annualized_sharpe > 0. A third, cost-stress gate was
considered and explicitly NOT implemented -- see :data:`STRESS_GATE_REASON`.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]

from ftmoquant.data.instruments import oanda_symbol
from ftmoquant.data.oanda_alpha_lab_validation import (
    VALIDATION_CONFIG_PATH as OANDA_ALPHA_LAB_VALIDATION_CONFIG_PATH,
)
from ftmoquant.research.alpha_lab.data import Timeframe
from ftmoquant.research.alpha_lab.liquidity_structure_signals import (
    b2f1_sweep_bos_retest_signals,
)
from ftmoquant.research.alpha_lab.validation import (
    AlphaLabValidationError,
    ValidationPreregistrationError,
    _canonical_sha256,
    _iso,
    _load_validation_instrument_frame,
    _reject_forbidden_path,
    _verify_validation_readiness,
)
from ftmoquant.research.alpha_lab.wick_fvg_squeeze_execution import (
    SkipRecord,
    Trade,
    load_m1_bidask,
    simulate_trades,
)
from ftmoquant.research.alpha_lab.wick_fvg_squeeze_screen import (
    F2_FAMILY,
    F3_FAMILY,
    PairStats,
    _annualized_sharpe,
    _pair_stats,
)
from ftmoquant.research.alpha_lab.wick_fvg_squeeze_signals import (
    SignalEvent,
    f2_fresh_fvg_mitigation_signals,
    f3_volatility_squeeze_breakout_signals,
)
from ftmoquant.research.stage_g import HOLDOUT_START, VALIDATION_START

B2F1_FAMILY = "B2F1_sweep_bos_retest"

#: The exact DEVELOPMENT scorecards this module's preregistration is allowed
#: to read (never validation prices/results) to verify the three frozen
#: candidate identities still exist and still match their stated parameters.
F2_F3_DEVELOPMENT_SCORECARD_PATH = Path(
    ".artifacts/alpha_lab/wick_fvg_squeeze_screen_v1/scorecard.csv"
)
B2F1_DEVELOPMENT_SCORECARD_PATH = Path(
    ".artifacts/alpha_lab/liquidity_structure_screen_v1/scorecard.csv"
)

PREREGISTRATION_PATH = Path(
    "config/validation/oanda_pair_specific_alpha_v1_preregistration.json"
)
PREREGISTRATION_VERSION = "oanda-pair-specific-alpha-validation-preregistration-1"

DEFAULT_VALIDATION_READINESS_PATH = Path(
    "/Users/Shared/FTMOQuant-data/oanda_fx_alpha_lab_v1/validation_readiness/"
    "ftmoquant_oanda_alpha_lab_validation_readiness.json"
)
DEFAULT_OUTPUT_DIR = Path(".artifacts/alpha_lab/pair_specific_validation_v1")

#: Section 6: frozen BEFORE any real VALIDATION run. Each candidate is
#: judged independently; no minimum trade count, no pair-breadth gate (these
#: are explicitly pair-specific hypotheses, not broad-FX families), no
#: validation subperiod gate.
GATE_A_DESCRIPTION = "native-spread VALIDATION net_return > 0"
GATE_B_DESCRIPTION = "native-spread VALIDATION annualized_sharpe > 0"

#: Section 6's cost-stress diagnostic (C) was evaluated and NOT implemented.
#: The M1 BID/ASK trade-lifecycle engine
#: (wick_fvg_squeeze_execution.simulate_trades) has no separate cost
#: parameter to scale -- the native OANDA spread is embedded directly in
#: which side (BID/ASK) each entry/exit crosses, not applied as an
#: independent fee on top of a mid price. Producing a "1.5x realized spread"
#: stress would require synthetically widening the BID/ASK spread and
#: re-executing every trade against those fabricated prices: exactly the
#: "inventing a new cost model" / "materially changing the execution engine"
#: the task explicitly forbids ("No: ... slippage invented here"). The gate
#: is therefore frozen as A AND B only.
STRESS_GATE_STATUS = "not_available"
STRESS_GATE_REASON = (
    "The M1 BID/ASK trade-lifecycle engine "
    "(wick_fvg_squeeze_execution.simulate_trades) has no separate cost "
    "parameter to scale by 1.5x: native spread cost is embedded directly in "
    "which side (BID for sells/long-exits, ASK for buys/short-exits) each "
    "entry and exit crosses, not applied as an independent fee on top of a "
    "mid price. Computing a cost-stress diagnostic would require inventing "
    "a synthetic spread-widening mechanism and re-executing every trade "
    "against fabricated prices -- exactly what the task's own instructions "
    "forbid ('No ... slippage invented here', 'If implementing C would "
    "require inventing a new cost model ... DO NOT invent it'). The frozen "
    "gate is therefore A AND B only; C is not computed."
)

FAILURE_POLICY_STATEMENT = (
    "One shot per candidate, independently judged. If a candidate fails "
    "gates A/B, that exact (family, pair, timeframe, parameter) "
    "configuration fails VALIDATION outright -- no adjacent parameter value "
    "from the same family (e.g. F2 backward_search_n=5/20, F3 "
    "bb_width_threshold=0.006, B2F1 rr=1.5), no pair substitution, no "
    "timeframe change, no stop/target/filter modification, no cost-model "
    "change, and no rerun is permitted for that candidate after VALIDATION "
    "results are observed."
)

#: Section 7's reporting-only temporal diagnostic: the ~498-day VALIDATION
#: window split into 2 equal non-overlapping calendar subperiods (mirrors
#: ftmoquant.research.alpha_lab.validation.VALIDATION_SUBPERIOD_COUNT).
#: Never a gate.
VALIDATION_SUBPERIOD_COUNT = 2


class PairSpecificValidationError(ValueError):
    """Raised on any fail-closed guard specific to this module."""


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    candidate_id: str
    family: str
    strategy_id: str
    instrument_id: str
    dataset_symbol: str
    timeframe: Timeframe
    parameters: dict[str, int | float]


#: Exactly the three FROZEN candidates from the task spec. Parameters here
#: can never drift silently: every one of the fields below is checked, byte
#: for byte, against a real row in the pinned DEVELOPMENT scorecard by
#: :func:`build_preregistration`/:func:`verify_preregistration`.
CANDIDATE_A = CandidateSpec(
    candidate_id="A",
    family=F2_FAMILY,
    strategy_id="f2_h4_n10_v1",
    instrument_id="USD/CAD.OANDA",
    dataset_symbol="USDCAD",
    timeframe="H4",
    parameters={"backward_search_n": 10},
)
CANDIDATE_B = CandidateSpec(
    candidate_id="B",
    family=F3_FAMILY,
    strategy_id="f3_h1_thr0.004_v1",
    instrument_id="USD/JPY.OANDA",
    dataset_symbol="USDJPY",
    timeframe="H1",
    parameters={"bb_width_threshold": 0.004},
)
CANDIDATE_C = CandidateSpec(
    candidate_id="C",
    family=B2F1_FAMILY,
    strategy_id="b2f1_m30_sw40_rr2.0_v1",
    instrument_id="USD/CAD.OANDA",
    dataset_symbol="USDCAD",
    timeframe="M30",
    parameters={"swing_lookback": 40, "rr": 2.0},
)
FROZEN_CANDIDATES: tuple[CandidateSpec, ...] = (CANDIDATE_A, CANDIDATE_B, CANDIDATE_C)

if len(FROZEN_CANDIDATES) != 3:
    raise PairSpecificValidationError("exactly 3 candidates must be frozen")


# ---------------------------------------------------------------------------
# Preregistration: generated/verified WITHOUT ever loading validation data.
# Reads only the pinned DEVELOPMENT scorecards named above.
# ---------------------------------------------------------------------------


def build_preregistration(
    *,
    f2_f3_scorecard_path: Path = F2_F3_DEVELOPMENT_SCORECARD_PATH,
    b2f1_scorecard_path: Path = B2F1_DEVELOPMENT_SCORECARD_PATH,
) -> dict[str, Any]:
    """Build the full preregistration document. Reads only the two pinned
    DEVELOPMENT scorecards named by the arguments above -- structurally
    incapable of accepting a validation-data argument (no parameter of this
    function names one)."""

    def _evidence(candidate: CandidateSpec) -> dict[str, Any]:
        path = (
            f2_f3_scorecard_path
            if candidate.family in (F2_FAMILY, F3_FAMILY)
            else b2f1_scorecard_path
        )
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        matches = [
            row
            for row in rows
            if row["family"] == candidate.family
            and row["strategy_id"] == candidate.strategy_id
            and row["instrument"] == candidate.instrument_id
            and row["timeframe"] == candidate.timeframe
            and json.loads(row["parameters"]) == candidate.parameters
        ]
        if len(matches) != 1:
            raise PairSpecificValidationError(
                f"candidate {candidate.candidate_id} does not match exactly "
                f"one row in {path}"
            )
        row = matches[0]
        return {
            "trade_count": int(row["trade_count"]),
            "net_return": float(row["net_return"]),
            "annualized_sharpe": float(row["annualized_sharpe"]),
            "maximum_drawdown": float(row["maximum_drawdown"]),
        }

    candidates_doc = [
        {
            "candidate_id": candidate.candidate_id,
            "family": candidate.family,
            "strategy_id": candidate.strategy_id,
            "instrument": candidate.instrument_id,
            "timeframe": candidate.timeframe,
            "parameters": candidate.parameters,
            "development_evidence": _evidence(candidate),
        }
        for candidate in FROZEN_CANDIDATES
    ]

    document: dict[str, Any] = {
        "preregistration_version": PREREGISTRATION_VERSION,
        "created_at_utc": _iso(datetime.now(UTC)),
        "candidates": candidates_doc,
        "development_lineage": {
            "f2_f3_scorecard_sha256": _sha256_file(f2_f3_scorecard_path),
            "b2f1_scorecard_sha256": _sha256_file(b2f1_scorecard_path),
        },
        "validation_partition": {
            "start_utc": _iso(VALIDATION_START),
            "end_exclusive_utc": _iso(HOLDOUT_START),
        },
        "execution_semantics": {
            "signals": "completed M30/H1/H4 bars only",
            "entry": (
                "first paired M1 BID/ASK observation strictly after "
                "signal-information time"
            ),
            "buy_crosses": "ASK",
            "sell_crosses": "BID",
            "long_liquidation_crosses": "BID",
            "short_liquidation_crosses": "ASK",
            "interpolation": "none",
            "synthetic_prices": "none",
            "stops_targets": "exactly as each frozen family already implements",
            "same_bar_collision": "stop first",
            "sleeve": "independent $100,000 notional sleeve per candidate",
            "cost_model": (
                "native OANDA spread only, included intrinsically by which "
                "side (BID/ASK) each entry/exit crosses; no commissions, "
                "slippage, or volatility/leverage/FTMO-risk normalization "
                "invented here"
            ),
        },
        "validation_gates": {
            "A_native_spread_positive_return": GATE_A_DESCRIPTION,
            "B_native_spread_positive_sharpe": GATE_B_DESCRIPTION,
            "C_cost_stress": {
                "status": STRESS_GATE_STATUS,
                "reason": STRESS_GATE_REASON,
            },
            "validation_passed": "A AND B (C not computed; see C_cost_stress.reason)",
            "no_minimum_trade_count_gate": True,
            "no_pair_breadth_gate": (
                "not applicable -- each candidate is an explicitly "
                "pair-specific hypothesis, not a broad-FX family"
            ),
            "no_validation_subperiod_gate": True,
        },
        "temporal_diagnostics": {
            "status": "reporting_only_not_a_gate",
            "subperiod_count": VALIDATION_SUBPERIOD_COUNT,
            "fields": [
                "positive_subperiod_count",
                "worst_subperiod_return",
                "median_subperiod_sharpe",
            ],
        },
        "failure_policy": FAILURE_POLICY_STATEMENT,
        "one_shot_policy": {
            "candidates_per_family": 1,
            "total_configurations": 3,
            "parameter_changes_after_this_point": "forbidden",
            "pair_substitutions_after_this_point": "forbidden",
            "reruns_after_observing_results": "forbidden",
        },
        "holdout_accessed": False,
    }
    document["preregistration_semantic_sha256"] = _canonical_sha256(document)
    return document


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_preregistration(path: Path = PREREGISTRATION_PATH) -> dict[str, Any]:
    try:
        return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationPreregistrationError(
            f"could not read preregistration: {error}"
        ) from error


def verify_preregistration(
    document: dict[str, Any],
    *,
    f2_f3_scorecard_path: Path = F2_F3_DEVELOPMENT_SCORECARD_PATH,
    b2f1_scorecard_path: Path = B2F1_DEVELOPMENT_SCORECARD_PATH,
) -> None:
    """Fail closed unless ``document`` is self-consistent, declares exactly
    the 3 frozen candidates with their exact frozen parameters, and still
    reproduces byte-for-byte from the pinned DEVELOPMENT scorecards."""

    working = dict(document)
    stored_hash = working.pop("preregistration_semantic_sha256", None)
    if stored_hash is None or _canonical_sha256(working) != stored_hash:
        raise ValidationPreregistrationError(
            "preregistration content does not match its own self-hash"
        )
    if document.get("preregistration_version") != PREREGISTRATION_VERSION:
        raise ValidationPreregistrationError("unrecognized preregistration_version")
    if document.get("holdout_accessed") is not False:
        raise ValidationPreregistrationError(
            "preregistration must declare holdout_accessed: false"
        )

    partition = document.get("validation_partition", {})
    if partition.get("start_utc") != _iso(VALIDATION_START) or partition.get(
        "end_exclusive_utc"
    ) != _iso(HOLDOUT_START):
        raise ValidationPreregistrationError(
            "preregistration validation partition does not match the frozen boundary"
        )

    candidates = document.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 3:
        raise ValidationPreregistrationError(
            "preregistration must declare exactly 3 candidates"
        )
    expected_by_id = {
        candidate.candidate_id: candidate for candidate in FROZEN_CANDIDATES
    }
    for entry in candidates:
        expected = expected_by_id.get(entry.get("candidate_id"))
        if (
            expected is None
            or entry.get("family") != expected.family
            or entry.get("strategy_id") != expected.strategy_id
            or entry.get("instrument") != expected.instrument_id
            or entry.get("timeframe") != expected.timeframe
            or entry.get("parameters") != expected.parameters
        ):
            raise ValidationPreregistrationError(
                "preregistration candidates do not match the frozen "
                "candidate identities"
            )

    fresh = build_preregistration(
        f2_f3_scorecard_path=f2_f3_scorecard_path,
        b2f1_scorecard_path=b2f1_scorecard_path,
    )
    if fresh["candidates"] != document.get("candidates"):
        raise ValidationPreregistrationError(
            "stored candidate development_evidence no longer reproduces from "
            "the pinned DEVELOPMENT scorecards"
        )


# ---------------------------------------------------------------------------
# VALIDATION-partition-scoped data loading. Reuses the existing readiness
# verifier and per-instrument mid-OHLC loader from
# ftmoquant.research.alpha_lab.validation unchanged; reuses load_m1_bidask
# unchanged from wick_fvg_squeeze_execution (already generic over
# instrument/root/start/end, not DEVELOPMENT-hardcoded).
# ---------------------------------------------------------------------------


def _reject_holdout_timestamps(index: pd.DatetimeIndex, *, context: str) -> None:
    """Defensive belt-and-suspenders check (section 9): even though the
    loaders below already restrict their own catalog queries to
    ``[VALIDATION_START, HOLDOUT_START)``, explicitly reject any observation
    at or after ``HOLDOUT_START`` that might otherwise slip through."""

    if index.empty:
        raise AlphaLabValidationError(f"{context}: no observations")
    if index.min() < VALIDATION_START or index.max() >= HOLDOUT_START:
        raise AlphaLabValidationError(
            f"{context}: an observation falls outside the frozen VALIDATION partition"
        )


def _verify_candidate_catalog_tree(
    *, candidate: CandidateSpec, readiness_document: dict[str, Any], root: Path
) -> None:
    from ftmoquant.backtest.execution_harness import _sha256_tree

    statuses = readiness_document.get("per_instrument_status", {})
    if statuses.get(candidate.instrument_id) != "research_ready":
        raise AlphaLabValidationError(
            f"VALIDATION instrument is not research_ready: {candidate.instrument_id}"
        )
    artifacts = readiness_document.get("instrument_artifacts", [])
    artifact_by_id = {
        item["instrument_id"]: item for item in artifacts if isinstance(item, dict)
    }
    artifact = artifact_by_id.get(candidate.instrument_id, {})
    expected_tree_sha = artifact.get("catalog_tree_sha256")
    actual_tree_sha = _sha256_tree(root / "catalog")
    if not expected_tree_sha or actual_tree_sha != expected_tree_sha:
        raise AlphaLabValidationError(
            f"VALIDATION catalog tree hash drifted: {candidate.instrument_id}"
        )


def load_candidate_data(
    *,
    candidate: CandidateSpec,
    validation_root: Path,
    universe_readiness_path: Path,
    config_path: Path = OANDA_ALPHA_LAB_VALIDATION_CONFIG_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load ``candidate``'s VALIDATION-partition mid-OHLC decision frame and
    M1 BID/ASK execution streams. Fails closed on: a DEVELOPMENT/HOLDOUT
    -looking root path, a readiness document that is not the dedicated
    VALIDATION readiness artifact, a catalog tree hash mismatch, or any
    observation outside ``[VALIDATION_START, HOLDOUT_START)``."""

    _reject_forbidden_path(validation_root)
    _reject_forbidden_path(universe_readiness_path)

    try:
        document = json.loads(universe_readiness_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AlphaLabValidationError(
            f"could not read OANDA alpha-lab VALIDATION readiness: {error}"
        ) from error
    if not isinstance(document, dict):
        raise AlphaLabValidationError("OANDA alpha-lab readiness document is invalid")

    _verify_validation_readiness(document, config_path=config_path)

    root = validation_root / oanda_symbol(candidate.dataset_symbol)
    _reject_forbidden_path(root)
    _verify_candidate_catalog_tree(
        candidate=candidate, readiness_document=document, root=root
    )

    ohlc = _load_validation_instrument_frame(
        instrument_id=candidate.instrument_id, root=root, timeframe=candidate.timeframe
    )
    _reject_holdout_timestamps(ohlc.index, context=f"{candidate.candidate_id} OHLC")

    bid_m1, ask_m1 = load_m1_bidask(
        instrument_id=candidate.instrument_id,
        root=root,
        start_utc=VALIDATION_START,
        end_exclusive_utc=HOLDOUT_START,
    )
    _reject_holdout_timestamps(bid_m1.index, context=f"{candidate.candidate_id} M1 BID")
    _reject_holdout_timestamps(ask_m1.index, context=f"{candidate.candidate_id} M1 ASK")

    return ohlc, bid_m1, ask_m1


# ---------------------------------------------------------------------------
# Signal dispatch -- imports the exact, unchanged, frozen signal functions.
# ---------------------------------------------------------------------------


def _events_for_candidate(
    candidate: CandidateSpec, ohlc: pd.DataFrame
) -> tuple[SignalEvent, ...]:
    o, h, low, c = ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"]
    if candidate.family == F2_FAMILY:
        return f2_fresh_fvg_mitigation_signals(
            o,
            h,
            low,
            c,
            backward_search_n=int(candidate.parameters["backward_search_n"]),
        )
    if candidate.family == F3_FAMILY:
        return f3_volatility_squeeze_breakout_signals(
            h,
            low,
            c,
            bb_width_threshold=float(candidate.parameters["bb_width_threshold"]),
        )
    if candidate.family == B2F1_FAMILY:
        return b2f1_sweep_bos_retest_signals(
            h,
            low,
            c,
            swing_lookback=int(candidate.parameters["swing_lookback"]),
            rr=float(candidate.parameters["rr"]),
        )
    raise PairSpecificValidationError(f"unknown family: {candidate.family}")


# ---------------------------------------------------------------------------
# Section 7: reporting-only temporal subperiod diagnostic (never a gate).
# Deliberately duplicated in miniature from
# ftmoquant.research.alpha_lab.wick_fvg_squeeze_screen._pair_fold_returns
# rather than reused directly -- that function is hardwired to the frozen
# DEVELOPMENT FOLD_WINDOWS constant (3 folds) and must stay that way; this
# one windows over the 2 VALIDATION subperiods instead. Same windowing/
# compounding arithmetic, different window source.
# ---------------------------------------------------------------------------


def _validation_subperiod_windows() -> tuple[tuple[datetime, datetime], ...]:
    span = HOLDOUT_START - VALIDATION_START
    step = span / VALIDATION_SUBPERIOD_COUNT
    bounds = [
        VALIDATION_START + step * i for i in range(VALIDATION_SUBPERIOD_COUNT + 1)
    ]
    return tuple((bounds[i], bounds[i + 1]) for i in range(VALIDATION_SUBPERIOD_COUNT))


def _subperiod_diagnostics(
    daily_returns: tuple[tuple[date, float], ...],
) -> tuple[int, float, float]:
    windows = _validation_subperiod_windows()
    period_returns: list[float] = []
    period_sharpes: list[float] = []
    for start, end in windows:
        start_date, end_date = start.date(), end.date()
        window_values = [
            value for day, value in daily_returns if start_date <= day < end_date
        ]
        cumulative = 1.0
        for value in window_values:
            cumulative *= 1.0 + value
        period_returns.append(cumulative - 1.0)
        period_sharpes.append(_annualized_sharpe(window_values) or 0.0)
    positive_count = sum(1 for r in period_returns if r > 0)
    worst = min(period_returns) if period_returns else 0.0
    import statistics

    median_sharpe = statistics.median(period_sharpes) if period_sharpes else 0.0
    return positive_count, worst, median_sharpe


# ---------------------------------------------------------------------------
# Section 6: the two frozen gates, evaluated independently per candidate.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CandidateValidationResult:
    candidate_id: str
    family: str
    strategy_id: str
    instrument: str
    timeframe: str
    parameters: str
    trade_count: int
    skip_count: int
    net_return: float
    annualized_sharpe: float | None
    maximum_drawdown: float
    win_rate: float
    stressed_return: float | None
    gate_a_positive_return: bool
    gate_b_positive_sharpe: bool
    validation_passed: bool
    positive_subperiod_count: int
    worst_subperiod_return: float
    median_subperiod_sharpe: float


def _evaluate_candidate(
    candidate: CandidateSpec, stats: PairStats
) -> CandidateValidationResult:
    gate_a = stats.net_return > 0.0
    gate_b = stats.annualized_sharpe is not None and stats.annualized_sharpe > 0.0
    positive_subperiod_count, worst_subperiod_return, median_subperiod_sharpe = (
        _subperiod_diagnostics(stats.daily_returns)
    )
    return CandidateValidationResult(
        candidate_id=candidate.candidate_id,
        family=candidate.family,
        strategy_id=candidate.strategy_id,
        instrument=candidate.instrument_id,
        timeframe=candidate.timeframe,
        parameters=json.dumps(candidate.parameters, sort_keys=True),
        trade_count=stats.trade_count,
        skip_count=stats.skip_count,
        net_return=stats.net_return,
        annualized_sharpe=stats.annualized_sharpe,
        maximum_drawdown=stats.maximum_drawdown,
        win_rate=stats.win_rate,
        stressed_return=None,
        gate_a_positive_return=gate_a,
        gate_b_positive_sharpe=gate_b,
        validation_passed=bool(gate_a and gate_b),
        positive_subperiod_count=positive_subperiod_count,
        worst_subperiod_return=worst_subperiod_return,
        median_subperiod_sharpe=median_subperiod_sharpe,
    )


def run_candidate(
    candidate: CandidateSpec,
    ohlc: pd.DataFrame,
    bid_m1: pd.DataFrame,
    ask_m1: pd.DataFrame,
) -> tuple[CandidateValidationResult, tuple[Trade, ...], tuple[SkipRecord, ...]]:
    events = _events_for_candidate(candidate, ohlc)
    trades, skips = simulate_trades(events, bid_m1, ask_m1)
    stats = _pair_stats(candidate.instrument_id, trades, skips)
    result = _evaluate_candidate(candidate, stats)
    return result, trades, skips


@dataclass(frozen=True, slots=True)
class PairSpecificValidationBatchResult:
    results: tuple[CandidateValidationResult, ...]
    metadata: dict[str, object]


def run_validation(
    *,
    preregistration: dict[str, Any],
    data_by_candidate_id: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
) -> PairSpecificValidationBatchResult:
    """Evaluate exactly the 3 preregistered candidates. Does not verify the
    preregistration itself (callers must call :func:`verify_preregistration`
    first) and does not load data (callers pass already-loaded,
    already-partition-checked data) -- kept separate so each concern is
    independently testable."""

    candidate_docs = preregistration["candidates"]
    if len(candidate_docs) != 3:
        raise ValidationPreregistrationError(
            "preregistration must declare exactly 3 candidates"
        )

    results: list[CandidateValidationResult] = []
    for candidate in FROZEN_CANDIDATES:
        ohlc, bid_m1, ask_m1 = data_by_candidate_id[candidate.candidate_id]
        result, _trades, _skips = run_candidate(candidate, ohlc, bid_m1, ask_m1)
        results.append(result)

    metadata: dict[str, object] = {
        "candidate_count": len(results),
        "validation_partition": {
            "start_utc": _iso(VALIDATION_START),
            "end_exclusive_utc": _iso(HOLDOUT_START),
        },
        "stress_gate_status": STRESS_GATE_STATUS,
        "stress_gate_reason": STRESS_GATE_REASON,
        "failure_policy": FAILURE_POLICY_STATEMENT,
        "holdout_accessed": False,
        "holdout_rows_admitted": 0,
    }
    return PairSpecificValidationBatchResult(results=tuple(results), metadata=metadata)


def write_validation_results(
    result: PairSpecificValidationBatchResult,
    preregistration: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)

    with (output_dir / "preregistration.json").open("w", encoding="utf-8") as stream:
        json.dump(dict(preregistration), stream, sort_keys=True, indent=2)
        stream.write("\n")

    fieldnames = list(asdict(result.results[0]).keys())

    def _write_csv(path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for row in result.results:
                record = asdict(row)
                for key, value in record.items():
                    if value is None:
                        record[key] = ""
                writer.writerow(record)

    _write_csv(output_dir / "validation_scorecard.csv")
    _write_csv(output_dir / "candidate_validation_summary.csv")

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(result.metadata, stream, sort_keys=True, indent=2)
        stream.write("\n")


# ---------------------------------------------------------------------------
# CLIs: preregistration generation never touches validation data; the run
# CLI verifies the preregistration before opening any VALIDATION path.
# ---------------------------------------------------------------------------


def build_preregister_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and freeze the one-shot pair-specific VALIDATION "
            "preregistration for the three frozen candidates. Reads only "
            "the pinned DEVELOPMENT scorecards -- never opens validation data."
        )
    )
    parser.add_argument(
        "--f2-f3-scorecard", type=Path, default=F2_F3_DEVELOPMENT_SCORECARD_PATH
    )
    parser.add_argument(
        "--b2f1-scorecard", type=Path, default=B2F1_DEVELOPMENT_SCORECARD_PATH
    )
    parser.add_argument("--output", type=Path, default=PREREGISTRATION_PATH)
    return parser


def preregister_main(argv: list[str] | None = None) -> None:
    parser = build_preregister_parser()
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValidationPreregistrationError(
            f"refusing to overwrite an existing frozen preregistration: {args.output}"
        )
    document = build_preregistration(
        f2_f3_scorecard_path=args.f2_f3_scorecard,
        b2f1_scorecard_path=args.b2f1_scorecard,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(
        f"preregistration_semantic_sha256={document['preregistration_semantic_sha256']}"
    )


def build_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the one-shot pair-specific VALIDATION stage for the three "
            "preregistered candidates against the frozen VALIDATION partition."
        )
    )
    parser.add_argument("--preregistration", type=Path, default=PREREGISTRATION_PATH)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument(
        "--universe-readiness",
        type=Path,
        default=DEFAULT_VALIDATION_READINESS_PATH,
        help="the dedicated VALIDATION readiness artifact -- never the DEVELOPMENT one",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_run_parser()
    args = parser.parse_args(argv)

    preregistration = load_preregistration(args.preregistration)
    verify_preregistration(preregistration)

    data_by_candidate_id = {
        candidate.candidate_id: load_candidate_data(
            candidate=candidate,
            validation_root=args.validation_root,
            universe_readiness_path=args.universe_readiness,
        )
        for candidate in FROZEN_CANDIDATES
    }

    result = run_validation(
        preregistration=preregistration, data_by_candidate_id=data_by_candidate_id
    )
    write_validation_results(result, preregistration, args.output)
    print(
        json.dumps(
            [asdict(candidate_result) for candidate_result in result.results],
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
