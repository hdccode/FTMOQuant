"""B3F1 underpowered candidate U2 -- one-shot VALIDATION run.

Evaluates EXACTLY ONE frozen candidate:

    sleeve:            USD/CAD.OANDA__USD/CHF.OANDA
    formation_window:  240 H1 bars
    z_entry:           1.5
    z_stop:            3.5

against the real VALIDATION partition. This is confirmation of a
post-screen exploratory underpowered candidate, NOT a new search stage --
there is no loop anywhere in this module over pairs, formation windows,
z_entry/z_stop grids, or alternative U candidates. Candidate U1
(``USD/CHF.OANDA__USD/JPY.OANDA``) is never named, imported, or referenced
here; it is structurally impossible to run through this module.

Reuses, rather than re-derives:

- :mod:`ftmoquant.research.alpha_lab.b3f1_spread_signals` (formation/signal
  walker) and :mod:`ftmoquant.research.alpha_lab.b3f1_spread_execution`
  (two-leg execution, cost-stress-aware) -- completely unchanged.
- :mod:`ftmoquant.research.alpha_lab.b3f1_spread_screen`'s pure statistics
  helpers (``expectancy_and_profit_factor``, ``best_5pct_removed_expectancy``,
  ``_rows_from_episodes``) for report-only diagnostics -- never its
  DEVELOPMENT-specific hard-gate constants (``MIN_TRADE_COUNT``,
  ``PROFIT_FACTOR_GT``, etc.), which do not apply to VALIDATION.
- :func:`ftmoquant.research.alpha_lab.validation.load_validation_dataset`
  for the aligned VALIDATION H1 dataset (readiness verification, partition
  boundary verification, catalog-tree hash checks, and the DEVELOPMENT/
  HOLDOUT path firewall all come from this call, unmodified) and
  :func:`ftmoquant.research.alpha_lab.wick_fvg_squeeze_execution.
  load_m1_bidask` for the M1 BID/ASK execution streams (already generic
  over instrument/root/start/end).
- :func:`ftmoquant.research.alpha_lab.wick_fvg_squeeze_screen._annualized_sharpe`
  and the subperiod-diagnostic pair
  (:func:`ftmoquant.research.alpha_lab.pair_specific_validation.
  _validation_subperiod_windows`/``_subperiod_diagnostics``) -- reused
  directly rather than reimplemented, exactly as
  :mod:`ftmoquant.research.alpha_lab.pair_specific_validation` already
  reuses them for its own one-shot candidate VALIDATION runs.
- The exact A/B gate DEFINITIONS from
  :data:`ftmoquant.research.alpha_lab.batch3_preregistration_v2.
  VALIDATION_POLICY` -- read programmatically, never retyped. No gate this
  module does not find there (trade-count floor, fold gate, concentration
  gate, bootstrap-significance gate, parameter-neighborhood gate) is ever
  applied; those were DEVELOPMENT/underpowered-resolution selection
  evidence, not VALIDATION pass criteria.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import scipy  # type: ignore[import-untyped]
import statsmodels  # type: ignore[import-untyped]

from ftmoquant.backtest.execution_harness import _sha256_tree
from ftmoquant.data.instruments import (
    USDCAD_OANDA_SPEC,
    USDCHF_OANDA_SPEC,
    oanda_symbol,
)
from ftmoquant.data.oanda_alpha_lab_validation import (
    VALIDATION_CONFIG_PATH as OANDA_ALPHA_LAB_VALIDATION_CONFIG_PATH,
)
from ftmoquant.research.alpha_lab.b3f1_spread_execution import (
    GROSS_NOTIONAL_USD,
    simulate_b3f1_intents,
)
from ftmoquant.research.alpha_lab.b3f1_spread_screen import (
    _rows_from_episodes,
    best_5pct_removed_expectancy,
    expectancy_and_profit_factor,
)
from ftmoquant.research.alpha_lab.b3f1_spread_signals import (
    compute_formation_series,
    generate_b3f1_decisions,
)
from ftmoquant.research.alpha_lab.batch3_preregistration_v2 import VALIDATION_POLICY
from ftmoquant.research.alpha_lab.pair_specific_validation import (
    _subperiod_diagnostics,
)
from ftmoquant.research.alpha_lab.relative_value_adapter import RelativeValueEpisode
from ftmoquant.research.alpha_lab.validation import (
    AlphaLabValidationError,
    _reject_forbidden_path,
    _verify_validation_readiness,
    load_validation_dataset,
)
from ftmoquant.research.alpha_lab.wick_fvg_squeeze_execution import load_m1_bidask
from ftmoquant.research.alpha_lab.wick_fvg_squeeze_screen import _annualized_sharpe
from ftmoquant.research.stage_g import HOLDOUT_START, VALIDATION_START

# ---------------------------------------------------------------------------
# Frozen, immutable candidate identity (section 0). Nothing below this block
# may ever be parameterized by a CLI flag, environment variable, or loop.
# ---------------------------------------------------------------------------

CANDIDATE_ID = "U2"
FAMILY_ID = "B3F1_spread_mean_reversion"
SLEEVE_ID = "USD/CAD.OANDA__USD/CHF.OANDA"
Y_SPEC = USDCAD_OANDA_SPEC
X_SPEC = USDCHF_OANDA_SPEC
FORMATION_WINDOW = 240
Z_ENTRY = Decimal("1.5")
Z_STOP = Decimal("3.5")

assert Y_SPEC.instrument_id < X_SPEC.instrument_id, (
    "Y must be the lexicographically smaller instrument_id, matching the "
    "frozen B3F1 orientation rule this candidate was screened under"
)
assert f"{Y_SPEC.instrument_id}__{X_SPEC.instrument_id}" == SLEEVE_ID

PREREGISTRATION_PATH = Path(
    "config/validation/b3f1_underpowered_candidate_v1_preregistration.json"
)
FROZEN_PREREGISTRATION_SHA256 = (
    "a6403e507296991524ef0ba751ca5b34fb94ff71af04eb677cc54baa89f61dc0"
)

RESOLUTION_SUMMARY_PATH = Path(
    ".artifacts/research_audits/b3f1_underpowered_resolution_v1/resolution_summary.json"
)

DEFAULT_VALIDATION_READINESS_PATH = Path(
    "/Users/Shared/FTMOQuant-data/oanda_fx_alpha_lab_v1/validation_readiness/"
    "ftmoquant_oanda_alpha_lab_validation_readiness.json"
)
DEFAULT_OUTPUT_DIR = Path(".artifacts/alpha_lab/b3f1_underpowered_u2_validation_v1")

STRESS_MULTIPLIER = Decimal("1.5")
VALIDATION_SUBPERIOD_COUNT_FOR_U2 = 2  # matches pair_specific_validation's convention


class B3F1UnderpoweredValidationError(ValueError):
    """Raised on any violation of this one-shot VALIDATION contract."""


# ---------------------------------------------------------------------------
# Preregistration verification -- byte-exact hash check, no rebuilding, no
# rederiving from DEVELOPMENT data (that already happened once, frozen).
# ---------------------------------------------------------------------------


def verify_preregistration(path: Path = PREREGISTRATION_PATH) -> dict[str, Any]:
    """Fail closed unless ``path`` byte-for-byte hashes to
    :data:`FROZEN_PREREGISTRATION_SHA256` and declares exactly the frozen U2
    identity with ``validation_accessed``/``holdout_accessed`` both false."""

    try:
        raw = path.read_bytes()
    except OSError as error:
        raise B3F1UnderpoweredValidationError(
            f"could not read preregistration: {error}"
        ) from error
    actual_hash = hashlib.sha256(raw).hexdigest()
    if actual_hash != FROZEN_PREREGISTRATION_SHA256:
        raise B3F1UnderpoweredValidationError(
            "preregistration content does not match its frozen SHA256 -- "
            f"expected {FROZEN_PREREGISTRATION_SHA256}, got {actual_hash}"
        )
    document = cast(dict[str, Any], json.loads(raw.decode("utf-8")))

    candidate = document.get("selected_candidate", {})
    if (
        candidate.get("candidate_id") != CANDIDATE_ID
        or candidate.get("family_id") != FAMILY_ID
        or candidate.get("sleeve_id") != SLEEVE_ID
        or candidate.get("formation_window") != FORMATION_WINDOW
        or candidate.get("z_entry") != str(Z_ENTRY)
        or candidate.get("z_stop") != str(Z_STOP)
    ):
        raise B3F1UnderpoweredValidationError(
            "preregistration selected_candidate does not match the frozen U2 identity"
        )
    if document.get("validation_accessed") is not False:
        raise B3F1UnderpoweredValidationError(
            "preregistration must declare validation_accessed: false prior to this run"
        )
    if document.get("holdout_accessed") is not False:
        raise B3F1UnderpoweredValidationError(
            "preregistration must declare holdout_accessed: false"
        )
    return document


# ---------------------------------------------------------------------------
# Output reservation -- BEFORE any market-data access.
# ---------------------------------------------------------------------------


def reserve_output_directory(output_dir: Path) -> None:
    if output_dir.exists():
        raise B3F1UnderpoweredValidationError(
            f"{output_dir} already exists; refusing to overwrite"
        )


# ---------------------------------------------------------------------------
# Data loading -- exactly the two legs' VALIDATION-partition H1 (formation)
# and M1 BID/ASK (execution) series. Reuses the existing readiness/catalog-
# hash/partition-boundary machinery unchanged.
# ---------------------------------------------------------------------------


def _verify_leg_catalog_tree(
    *, instrument_id: str, readiness_document: dict[str, Any], root: Path
) -> None:
    statuses = readiness_document.get("per_instrument_status", {})
    if statuses.get(instrument_id) != "research_ready":
        raise AlphaLabValidationError(
            f"VALIDATION instrument is not research_ready: {instrument_id}"
        )
    artifacts = readiness_document.get("instrument_artifacts", [])
    artifact_by_id = {
        item["instrument_id"]: item for item in artifacts if isinstance(item, dict)
    }
    artifact = artifact_by_id.get(instrument_id, {})
    expected_tree_sha = artifact.get("catalog_tree_sha256")
    actual_tree_sha = _sha256_tree(root / "catalog")
    if not expected_tree_sha or actual_tree_sha != expected_tree_sha:
        raise AlphaLabValidationError(
            f"VALIDATION catalog tree hash drifted: {instrument_id}"
        )


@dataclass(frozen=True, slots=True)
class U2ValidationData:
    log_y: pd.Series
    log_x: pd.Series
    y_bid_m1: pd.DataFrame
    y_ask_m1: pd.DataFrame
    x_bid_m1: pd.DataFrame
    x_ask_m1: pd.DataFrame
    readiness_document: dict[str, Any]


def load_u2_validation_data(
    *,
    validation_root: Path,
    universe_readiness_path: Path,
    config_path: Path = OANDA_ALPHA_LAB_VALIDATION_CONFIG_PATH,
) -> U2ValidationData:
    """Load U2's two legs' H1 log-mid-close (formation) and M1 BID/ASK
    (execution) VALIDATION-partition series. Fails closed on a DEVELOPMENT/
    HOLDOUT-looking path, wrong readiness partition/lineage/config hash, a
    catalog-tree hash mismatch, or any observation outside
    ``[VALIDATION_START, HOLDOUT_START)``."""

    _reject_forbidden_path(validation_root)
    _reject_forbidden_path(universe_readiness_path)

    dataset = load_validation_dataset(
        validation_root=validation_root,
        universe_readiness_path=universe_readiness_path,
        timeframe="H1",
        config_path=config_path,
    )
    if Y_SPEC.instrument_id not in dataset.close.columns:
        raise B3F1UnderpoweredValidationError(
            f"VALIDATION H1 dataset is missing U2's Y leg: {Y_SPEC.instrument_id}"
        )
    if X_SPEC.instrument_id not in dataset.close.columns:
        raise B3F1UnderpoweredValidationError(
            f"VALIDATION H1 dataset is missing U2's X leg: {X_SPEC.instrument_id}"
        )
    log_close = np.log(dataset.close)
    log_y = log_close[Y_SPEC.instrument_id]
    log_x = log_close[X_SPEC.instrument_id]

    try:
        readiness_document = json.loads(
            universe_readiness_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise AlphaLabValidationError(
            f"could not read VALIDATION readiness: {error}"
        ) from error
    _verify_validation_readiness(readiness_document, config_path=config_path)

    y_root = validation_root / oanda_symbol(Y_SPEC.dataset_symbol)
    x_root = validation_root / oanda_symbol(X_SPEC.dataset_symbol)
    _reject_forbidden_path(y_root)
    _reject_forbidden_path(x_root)
    _verify_leg_catalog_tree(
        instrument_id=Y_SPEC.instrument_id,
        readiness_document=readiness_document,
        root=y_root,
    )
    _verify_leg_catalog_tree(
        instrument_id=X_SPEC.instrument_id,
        readiness_document=readiness_document,
        root=x_root,
    )

    y_bid, y_ask = load_m1_bidask(
        instrument_id=Y_SPEC.instrument_id,
        root=y_root,
        start_utc=VALIDATION_START,
        end_exclusive_utc=HOLDOUT_START,
    )
    x_bid, x_ask = load_m1_bidask(
        instrument_id=X_SPEC.instrument_id,
        root=x_root,
        start_utc=VALIDATION_START,
        end_exclusive_utc=HOLDOUT_START,
    )
    _reject_out_of_partition(y_bid.index, "Y BID")
    _reject_out_of_partition(y_ask.index, "Y ASK")
    _reject_out_of_partition(x_bid.index, "X BID")
    _reject_out_of_partition(x_ask.index, "X ASK")

    return U2ValidationData(
        log_y=log_y,
        log_x=log_x,
        y_bid_m1=y_bid,
        y_ask_m1=y_ask,
        x_bid_m1=x_bid,
        x_ask_m1=x_ask,
        readiness_document=readiness_document,
    )


def _reject_out_of_partition(index: pd.DatetimeIndex, label: str) -> None:
    if index.empty:
        raise AlphaLabValidationError(f"{label}: no observations")
    if index.min() < VALIDATION_START or index.max() >= HOLDOUT_START:
        raise AlphaLabValidationError(
            f"{label}: an observation falls outside the frozen VALIDATION partition"
        )


# ---------------------------------------------------------------------------
# Report-only diagnostics -- computed the same way for native and stressed
# execution, never turned into an additional pass/fail gate.
# ---------------------------------------------------------------------------


def _daily_return_curve(
    episodes: tuple[RelativeValueEpisode, ...],
) -> tuple[tuple[tuple[Any, float], ...], float, float, float]:
    """One $100k-notional equity curve over ``episodes`` (identical
    equal-capital-sleeve convention to
    ``wick_fvg_squeeze_screen._pair_stats``, adapted from single-leg
    ``Trade.return_frac`` to two-leg ``RelativeValueEpisode.realized_pnl()
    / GROSS_NOTIONAL_USD``). Returns (daily_returns, net_return,
    max_drawdown, win_rate)."""

    if not episodes:
        return (), 0.0, 0.0, 0.0
    ordered = sorted(episodes, key=lambda ep: ep.exit_ns)
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    wins = 0
    points: list[tuple[pd.Timestamp, float]] = [
        (pd.Timestamp(ordered[0].entry_ns, unit="ns", tz="UTC"), equity)
    ]
    for episode in ordered:
        return_frac = float(episode.realized_pnl() / GROSS_NOTIONAL_USD)
        equity *= 1.0 + return_frac
        if return_frac > 0:
            wins += 1
        peak = max(peak, equity)
        max_dd = min(max_dd, equity / peak - 1.0)
        points.append((pd.Timestamp(episode.exit_ns, unit="ns", tz="UTC"), equity))
    series = pd.Series(
        [value for _, value in points],
        index=pd.DatetimeIndex([ts for ts, _ in points]),
    )
    daily = series.resample("1D").last().ffill()
    daily_returns_series = daily.pct_change().dropna()
    daily_returns = tuple(
        (ts.date(), float(value)) for ts, value in daily_returns_series.items()
    )
    return daily_returns, equity - 1.0, max_dd, wins / len(ordered)


@dataclass(frozen=True, slots=True)
class U2ValidationResult:
    candidate_id: str
    family: str
    sleeve_id: str
    formation_window: int
    z_entry: str
    z_stop: str
    trade_count: int
    skip_count: int
    native_expectancy_usd: str
    native_profit_factor: str
    native_net_return: float
    native_annualized_sharpe: float | None
    native_maximum_drawdown: float
    native_win_rate: float
    best_5pct_removed_expectancy_usd: str
    stressed_1_5x_trade_count: int
    stressed_1_5x_skip_count: int
    stressed_1_5x_expectancy_usd: str
    stressed_1_5x_profit_factor: str
    stressed_1_5x_net_return: float
    positive_subperiod_count: int
    worst_subperiod_return: float
    median_subperiod_sharpe: float
    rich_trade_count: int
    rich_expectancy_usd: str
    rich_profit_factor: str
    cheap_trade_count: int
    cheap_expectancy_usd: str
    cheap_profit_factor: str
    mean_holding_seconds: float | None
    median_holding_seconds: float | None
    adf_valid_bar_count: int
    gate_a_native_positive_return: bool
    gate_b_native_positive_sharpe: bool
    validation_passed: bool


def evaluate_u2(
    *,
    native_episodes: tuple[RelativeValueEpisode, ...],
    native_skips: tuple[Any, ...],
    stressed_episodes: tuple[RelativeValueEpisode, ...],
    stressed_skips: tuple[Any, ...],
    formation_valid_count: int,
) -> U2ValidationResult:
    """Pure function: no data access. Computes exactly gates A/B (read from
    :data:`VALIDATION_POLICY`) plus every report-only diagnostic section 8
    asks for -- none of which ever feeds back into ``validation_passed``."""

    native_rows = _rows_from_episodes(native_episodes)
    trade_count = len(native_rows)
    if trade_count == 0:
        native_expectancy = Decimal(0)
        native_pf = Decimal(0)
        best5 = Decimal(0)
    else:
        native_expectancy, native_pf = expectancy_and_profit_factor(
            [row.pnl for row in native_rows]
        )
        best5 = best_5pct_removed_expectancy(native_rows)

    daily_returns, net_return, max_dd, win_rate = _daily_return_curve(native_episodes)
    annualized_sharpe = _annualized_sharpe([value for _, value in daily_returns])
    positive_subperiod_count, worst_subperiod_return, median_subperiod_sharpe = (
        _subperiod_diagnostics(daily_returns)
    )

    stressed_rows = _rows_from_episodes(stressed_episodes)
    if stressed_rows:
        stressed_expectancy, stressed_pf = expectancy_and_profit_factor(
            [row.pnl for row in stressed_rows]
        )
    else:
        stressed_expectancy, stressed_pf = Decimal(0), Decimal(0)
    _, stressed_net_return, _, _ = _daily_return_curve(stressed_episodes)

    rich_rows = [row for row in native_rows if row.side == "rich"]
    cheap_rows = [row for row in native_rows if row.side == "cheap"]
    if rich_rows:
        rich_expectancy, rich_pf = expectancy_and_profit_factor(
            [row.pnl for row in rich_rows]
        )
    else:
        rich_expectancy, rich_pf = Decimal(0), Decimal(0)
    if cheap_rows:
        cheap_expectancy, cheap_pf = expectancy_and_profit_factor(
            [row.pnl for row in cheap_rows]
        )
    else:
        cheap_expectancy, cheap_pf = Decimal(0), Decimal(0)

    holding_seconds = [
        (episode.exit_ns - episode.entry_ns) / 1_000_000_000
        for episode in native_episodes
    ]
    mean_holding: float | None
    median_holding: float | None
    if holding_seconds:
        import statistics

        mean_holding = statistics.mean(holding_seconds)
        median_holding = statistics.median(holding_seconds)
    else:
        mean_holding = median_holding = None

    gate_a = net_return > 0.0
    gate_b = annualized_sharpe is not None and annualized_sharpe > 0.0

    return U2ValidationResult(
        candidate_id=CANDIDATE_ID,
        family=FAMILY_ID,
        sleeve_id=SLEEVE_ID,
        formation_window=FORMATION_WINDOW,
        z_entry=str(Z_ENTRY),
        z_stop=str(Z_STOP),
        trade_count=trade_count,
        skip_count=len(native_skips),
        native_expectancy_usd=str(native_expectancy),
        native_profit_factor=str(native_pf),
        native_net_return=net_return,
        native_annualized_sharpe=annualized_sharpe,
        native_maximum_drawdown=max_dd,
        native_win_rate=win_rate,
        best_5pct_removed_expectancy_usd=str(best5),
        stressed_1_5x_trade_count=len(stressed_rows),
        stressed_1_5x_skip_count=len(stressed_skips),
        stressed_1_5x_expectancy_usd=str(stressed_expectancy),
        stressed_1_5x_profit_factor=str(stressed_pf),
        stressed_1_5x_net_return=stressed_net_return,
        positive_subperiod_count=positive_subperiod_count,
        worst_subperiod_return=worst_subperiod_return,
        median_subperiod_sharpe=median_subperiod_sharpe,
        rich_trade_count=len(rich_rows),
        rich_expectancy_usd=str(rich_expectancy),
        rich_profit_factor=str(rich_pf),
        cheap_trade_count=len(cheap_rows),
        cheap_expectancy_usd=str(cheap_expectancy),
        cheap_profit_factor=str(cheap_pf),
        mean_holding_seconds=mean_holding,
        median_holding_seconds=median_holding,
        adf_valid_bar_count=formation_valid_count,
        gate_a_native_positive_return=gate_a,
        gate_b_native_positive_sharpe=gate_b,
        validation_passed=bool(gate_a and gate_b),
    )


# ---------------------------------------------------------------------------
# DEVELOPMENT comparison (section 9) -- read from the frozen resolution
# audit, never hardcoded/approximated.
# ---------------------------------------------------------------------------


def load_development_comparison(
    resolution_summary_path: Path = RESOLUTION_SUMMARY_PATH,
) -> dict[str, Any]:
    try:
        document = json.loads(resolution_summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise B3F1UnderpoweredValidationError(
            f"could not read frozen DEVELOPMENT resolution summary: {error}"
        ) from error
    candidate = document.get("candidates", {}).get(CANDIDATE_ID)
    if candidate is None:
        raise B3F1UnderpoweredValidationError(
            f"resolution summary does not contain candidate {CANDIDATE_ID}"
        )
    return cast(dict[str, Any], candidate)


# ---------------------------------------------------------------------------
# Metadata / artifacts
# ---------------------------------------------------------------------------


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[4],
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, OSError, FileNotFoundError):
        return "unknown"


def build_metadata(
    *,
    preregistration: dict[str, Any],
    readiness_document: dict[str, Any],
    result: U2ValidationResult,
) -> dict[str, Any]:
    gates = VALIDATION_POLICY["precommitted_gates"]
    return {
        "protocol": "b3f1_underpowered_u2_validation_v1",
        "preregistration_path": str(PREREGISTRATION_PATH),
        "preregistration_sha256": FROZEN_PREREGISTRATION_SHA256,
        "candidate_identity": {
            "candidate_id": CANDIDATE_ID,
            "family_id": FAMILY_ID,
            "sleeve_id": SLEEVE_ID,
            "formation_window": FORMATION_WINDOW,
            "z_entry": str(Z_ENTRY),
            "z_stop": str(Z_STOP),
        },
        "disclosure": preregistration["disclosure"],
        "source_partition": "VALIDATION",
        "validation_partition": {
            "start_utc": VALIDATION_START.astimezone(UTC)
            .isoformat()
            .replace("+00:00", "Z"),
            "end_exclusive_utc": HOLDOUT_START.astimezone(UTC)
            .isoformat()
            .replace("+00:00", "Z"),
        },
        "readiness_identity": {
            "readiness_version": readiness_document.get("readiness_version"),
            "semantic_sha256": readiness_document.get("semantic_sha256"),
            "alpha_lab_config_sha256": readiness_document.get(
                "alpha_lab_config_sha256"
            ),
            "partition": readiness_document.get("partition"),
        },
        "one_shot": True,
        "alternative_candidates_evaluated": 0,
        "policy_retune_permitted": False,
        "validation_gates": {
            "A": gates["A_native_spread_positive_return"],
            "B": gates["B_native_spread_positive_sharpe"],
            "validation_passed_rule": gates["validation_passed"],
        },
        "cost_stress_multipliers_computed": [str(STRESS_MULTIPLIER)],
        "validation_passed": result.validation_passed,
        "on_failure": VALIDATION_POLICY["on_failure"],
        "library_versions": {
            "statsmodels": statsmodels.__version__,
            "scipy": scipy.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "python": sys.version.split()[0],
        },
        "git_commit": _git_commit(),
        "validation_accessed": True,
        "final_holdout_accessed": False,
    }


def write_validation_results(
    *,
    result: U2ValidationResult,
    preregistration: dict[str, Any],
    development_comparison: dict[str, Any],
    metadata: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)

    (output_dir / "preregistration.json").write_text(
        json.dumps(preregistration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    scorecard_row = asdict(result)
    scorecard_path = output_dir / "validation_scorecard.csv"
    import csv

    with scorecard_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scorecard_row.keys()))
        writer.writeheader()
        writer.writerow({k: ("" if v is None else v) for k, v in scorecard_row.items()})

    degradation = {}
    try:
        dev_native_expectancy = float(development_comparison["native_expectancy"])
        degradation["native_expectancy_delta_usd"] = (
            float(result.native_expectancy_usd) - dev_native_expectancy
        )
    except (KeyError, ValueError, TypeError):
        pass
    try:
        dev_stressed_expectancy = float(
            development_comparison["stressed_1_5x_expectancy"]
        )
        degradation["stressed_1_5x_expectancy_delta_usd"] = (
            float(result.stressed_1_5x_expectancy_usd) - dev_stressed_expectancy
        )
    except (KeyError, ValueError, TypeError):
        pass

    summary = {
        "candidate": CANDIDATE_ID,
        "result": scorecard_row,
        "development_evidence_historical_selection_only": development_comparison,
        "mechanical_degradation_or_improvement": degradation,
        "note": (
            "development_evidence_historical_selection_only is reported for "
            "context ONLY -- it is the evidence that led to U2's selection "
            "under the exploratory underpowered-resolution protocol, not a "
            "VALIDATION criterion. Only gates A and B (read from "
            "VALIDATION_POLICY) determine validation_passed."
        ),
    }
    (output_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output_dir.iterdir())
    }
    (output_dir / "artifact_hashes.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Orchestration -- exactly ONE strategy configuration, no loop, no ranking,
# no optimization, no fallback.
# ---------------------------------------------------------------------------


def run_u2_validation(
    *,
    validation_root: Path,
    universe_readiness: Path = DEFAULT_VALIDATION_READINESS_PATH,
    preregistration_path: Path = PREREGISTRATION_PATH,
    resolution_summary_path: Path = RESOLUTION_SUMMARY_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> U2ValidationResult:
    """The full one-shot U2 VALIDATION run. Write-once check happens FIRST,
    before the preregistration is even read; preregistration verification
    happens before any market-data path is opened."""

    reserve_output_directory(output_dir)
    preregistration = verify_preregistration(preregistration_path)
    development_comparison = load_development_comparison(resolution_summary_path)

    data = load_u2_validation_data(
        validation_root=validation_root, universe_readiness_path=universe_readiness
    )

    formation = compute_formation_series(data.log_y, data.log_x, FORMATION_WINDOW)
    decisions = generate_b3f1_decisions(
        formation,
        data.log_y,
        data.log_x,
        sleeve_id=SLEEVE_ID,
        z_entry=Z_ENTRY,
        z_stop=Z_STOP,
    )

    native_episodes, native_skips = simulate_b3f1_intents(
        decisions,
        y_spec=Y_SPEC,
        x_spec=X_SPEC,
        y_bid_m1=data.y_bid_m1,
        y_ask_m1=data.y_ask_m1,
        x_bid_m1=data.x_bid_m1,
        x_ask_m1=data.x_ask_m1,
        gross_notional_usd=GROSS_NOTIONAL_USD,
        cost_stress_multiplier=Decimal("1"),
    )
    stressed_episodes, stressed_skips = simulate_b3f1_intents(
        decisions,
        y_spec=Y_SPEC,
        x_spec=X_SPEC,
        y_bid_m1=data.y_bid_m1,
        y_ask_m1=data.y_ask_m1,
        x_bid_m1=data.x_bid_m1,
        x_ask_m1=data.x_ask_m1,
        gross_notional_usd=GROSS_NOTIONAL_USD,
        cost_stress_multiplier=STRESS_MULTIPLIER,
    )

    result = evaluate_u2(
        native_episodes=native_episodes,
        native_skips=native_skips,
        stressed_episodes=stressed_episodes,
        stressed_skips=stressed_skips,
        formation_valid_count=int(formation["valid"].sum()),
    )

    metadata = build_metadata(
        preregistration=preregistration,
        readiness_document=data.readiness_document,
        result=result,
    )

    write_validation_results(
        result=result,
        preregistration=preregistration,
        development_comparison=development_comparison,
        metadata=metadata,
        output_dir=output_dir,
    )

    if not result.validation_passed:
        print(
            f"U2 FAILED VALIDATION (gate_a={result.gate_a_native_positive_return}, "
            f"gate_b={result.gate_b_native_positive_sharpe}) -- retired permanently. "
            "No rescue candidate may be substituted."
        )
    else:
        print("U2 PASSED VALIDATION.")

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the one-shot VALIDATION evaluation for B3F1 underpowered "
            "candidate U2 ONLY. No parameter, pair, or threshold override "
            "flags exist -- the candidate identity is frozen in this module."
        )
    )
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument(
        "--universe-readiness", type=Path, default=DEFAULT_VALIDATION_READINESS_PATH
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_u2_validation(
        validation_root=args.validation_root,
        universe_readiness=args.universe_readiness,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
