"""Nautilus execution-promotion wiring for the frozen, VALIDATED
``B2F1_sweep_bos_retest`` / USD/CAD.OANDA / M30 signal
(:mod:`ftmoquant.strategies.usdcad_sweep_bos_retest`).

Mirrors ``mean_reversion_h1_development.py``'s wiring pattern (engine
construction via the shared ``execution_harness`` helpers, provenance via
``g1/artifacts.py``) but is smaller and single-instrument. Reused, not
reinvented:

- :class:`~ftmoquant.research.mean_reversion_h1_development.Partition`,
  :func:`~ftmoquant.research.mean_reversion_h1_development.parse_partition`,
  :func:`~ftmoquant.research.mean_reversion_h1_development.partition_bounds`,
  :func:`~ftmoquant.research.mean_reversion_h1_development._reject_sealed_path`,
  :func:`~ftmoquant.research.mean_reversion_h1_development._validate_bar_stream`,
  :func:`~ftmoquant.research.mean_reversion_h1_development._add_oanda_venue`,
  :func:`~ftmoquant.research.mean_reversion_h1_development._convert_to_account_currency`
  -- all already fully generic (parameterized by explicit dates/paths, not
  hardwired to mean-reversion), reused unchanged rather than copied.
- ``execution_harness.py``'s private ``_engine_config``/``_fee_model``/
  ``_instrument_for_profile``/``_rollover_modules``/``canonical_execution_
  profile``/``_money_text``/``_sha256_json``/``_git_commit`` -- unchanged.
- ``g1/artifacts.py``'s ``write_runtime_provenance`` -- unchanged.
- ``eurusd_tsm_development.EquityPoint`` and
  ``ts_momentum_development._annualized_sharpe``/``_maximum_drawdown`` --
  unchanged.
- DEVELOPMENT data: ``ftmoquant.research.alpha_lab.data.load_alpha_lab_
  dataset`` (source="oanda") for the causal M30 mid-OHLC decision input
  (the exact same computation the Alpha Lab screen used), and
  ``wick_fvg_squeeze_execution.load_m1_bidask`` for the M1 BID/ASK
  execution stream -- both unchanged.
- VALIDATION data: :mod:`ftmoquant.research.alpha_lab.pair_specific_
  validation`'s ``CANDIDATE_C`` (this exact candidate identity, already
  frozen there) and its ``load_candidate_data`` -- unchanged; this is the
  same readiness-verified, holdout-firewalled VALIDATION loader the
  pair-specific validation stage already used for this candidate.

No second backtester is built: all fills, spread crossing, cost, latency,
and account/currency mechanics remain entirely owned by the real Nautilus
``BacktestEngine`` and ``Portfolio``/``Account``. The only new logic in this
module is: precomputed-trade -> engine wiring, single-instrument
partitioned data resolution, and performance measurement/reporting.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path
from typing import Any

from nautilus_trader.backtest import BacktestEngine
from nautilus_trader.execution import ProbabilisticFillModel, StaticLatencyModel
from nautilus_trader.model import Bar, CurrencyPair
from nautilus_trader.persistence import ParquetDataCatalog

from ftmoquant.backtest.execution_harness import (
    AccountParameters,
    _engine_config,
    _fee_model,
    _git_commit,
    _instrument_for_profile,
    _rollover_modules,
    _sha256_json,
    canonical_execution_profile,
)
from ftmoquant.data.dukascopy import NAUTILUS_VERSION
from ftmoquant.data.instruments import oanda_symbol
from ftmoquant.data.oanda_alpha_lab_development import load_oanda_alpha_lab_config
from ftmoquant.research import mean_reversion_h1_development as _mrh1
from ftmoquant.research.alpha_lab import data as alpha_lab_data
from ftmoquant.research.alpha_lab.data import load_alpha_lab_dataset
from ftmoquant.research.alpha_lab.liquidity_structure_signals import (
    b2f1_sweep_bos_retest_signals,
)
from ftmoquant.research.alpha_lab.pair_specific_validation import (
    CANDIDATE_C,
    DEFAULT_VALIDATION_READINESS_PATH,
    load_candidate_data,
)
from ftmoquant.research.alpha_lab.wick_fvg_squeeze_execution import (
    SkipRecord,
    Trade,
    load_m1_bidask,
)
from ftmoquant.research.eurusd_tsm_development import EquityPoint
from ftmoquant.research.g1.artifacts import write_runtime_provenance
from ftmoquant.research.stage_g import (
    DEVELOPMENT_END_EXCLUSIVE,
    DEVELOPMENT_START,
    HOLDOUT_START,
    VALIDATION_START,
)
from ftmoquant.research.ts_momentum_development import (
    _annualized_sharpe,
    _maximum_drawdown,
)
from ftmoquant.strategies.usdcad_sweep_bos_retest import (
    BASE_RESEARCH_UNITS,
    FROZEN_FAMILY,
    FROZEN_INSTRUMENT_ID,
    FROZEN_RR,
    FROZEN_SWING_LOOKBACK,
    FROZEN_TIMEFRAME,
    STRATEGY_SEMANTIC_ID,
    CompletedNautilusTrade,
    FillRecord,
    OrderSubmissionRecord,
    TradeInstruction,
    UsdCadSweepBosRetestExecutor,
    precompute_alpha_lab_trades,
    trade_instructions_from_alpha_lab_trades,
)

RUN_MODULE_NAME = "ftmoquant.research.usdcad_sweep_bos_retest_development"
#: Truthful label (Section 7): genuine native OANDA BID/ASK spread crossing
#: is the only cost/friction actually modeled. Reused unchanged from
#: canonical_execution_profile(): zero added latency, zero commission, zero
#: adverse-slippage probability, rollover disabled. This is NOT a claim of
#: full realism -- see EXECUTION_PROFILE_CAVEAT below and the final report.
EXECUTION_PROFILE_LABEL = "native_spread_nautilus_execution"
EXECUTION_PROFILE_CAVEAT = (
    "canonical_execution_profile() is UNCALIBRATED: zero modeled commission, "
    "zero modeled slippage (adverse_slippage_probability=0), zero added "
    "latency, rollover DISABLED. Only genuine native OANDA BID/ASK spread "
    "crossing is modeled. This is not 'fully realistic' -- no calibrated "
    "commission/slippage/latency/rollover evidence exists yet for any "
    "FTMOQuant OANDA account; that sensitivity study is deferred to a later "
    "stage, not invented here."
)
SIZING_CONVENTION_LABEL = (
    "fixed_100000_reference_notional_1x_no_vol_normalization_no_dynamic_resizing"
)
DEFAULT_OUTPUT_ROOT = Path(".artifacts/usdcad_sweep_bos_retest_v1")
_M1_BAR_TYPE = "1-MINUTE"

Partition = _mrh1.Partition
parse_partition = _mrh1.parse_partition
partition_bounds = _mrh1.partition_bounds
_reject_sealed_path = _mrh1._reject_sealed_path
_validate_bar_stream = _mrh1._validate_bar_stream
_add_oanda_venue = _mrh1._add_oanda_venue
_convert_to_account_currency = _mrh1._convert_to_account_currency


class UsdCadSweepBosRetestDevelopmentError(ValueError):
    """Raised on any fail-closed guard: forbidden path, wrong readiness
    document, catalog drift, or a malformed engine run."""


def _ns(value: datetime) -> int:
    return int(value.timestamp() * 1_000_000_000)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Partition-scoped data resolution. DEVELOPMENT reuses the OANDA alpha-lab
# DEVELOPMENT loaders (data.py, wick_fvg_squeeze_execution.py); VALIDATION
# reuses pair_specific_validation's already-frozen CANDIDATE_C loader.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResolvedPartitionData:
    instrument: CurrencyPair
    ohlc_m30: Any  # pandas.DataFrame with open/high/low/close columns
    m1_bid_pd: Any  # pandas.DataFrame
    m1_ask_pd: Any  # pandas.DataFrame
    m1_bid_bars: tuple[Bar, ...]
    m1_ask_bars: tuple[Bar, ...]
    readiness_identity_sha256: str


def _instrument_from_catalog(root: Path, instrument_id: str) -> CurrencyPair:
    catalog = ParquetDataCatalog(str(root / "catalog"))
    found = catalog.instruments([instrument_id])
    if len(found) != 1 or not isinstance(found[0], CurrencyPair):
        raise UsdCadSweepBosRetestDevelopmentError(
            f"frozen CurrencyPair is unavailable: {instrument_id}"
        )
    return found[0]


def _query_m1_bars(
    root: Path, instrument_id: str, side: str, start_ns: int, end_ns: int
) -> tuple[Bar, ...]:
    catalog = ParquetDataCatalog(str(root / "catalog"))
    bar_type = f"{instrument_id}-{_M1_BAR_TYPE}-{side}-EXTERNAL"
    bars = tuple(catalog.query_bars([bar_type], start=start_ns, end=end_ns - 1))
    _validate_bar_stream(bars, bar_type, start_ns, end_ns)
    return bars


def resolve_development_data(
    *, readiness_path: Path, development_root_dir: Path
) -> ResolvedPartitionData:
    """Resolve DEVELOPMENT-partition data for the frozen candidate. Fails
    closed via the same firewall ``wick_fvg_squeeze_screen.py`` already
    uses: ``data._discover_oanda_universe`` rejects any validation/holdout
    -looking root and verifies the OANDA DEVELOPMENT readiness manifest and
    catalog-tree hash before any bar is read."""

    instrument_ids, development_roots = alpha_lab_data._discover_oanda_universe(
        readiness_path, development_root_dir
    )
    if FROZEN_INSTRUMENT_ID not in instrument_ids:
        raise UsdCadSweepBosRetestDevelopmentError(
            f"{FROZEN_INSTRUMENT_ID} is not research_ready in DEVELOPMENT readiness"
        )
    root = development_roots[FROZEN_INSTRUMENT_ID]

    dataset = load_alpha_lab_dataset(
        readiness_path=readiness_path,
        development_root_dir=development_root_dir,
        timeframe=FROZEN_TIMEFRAME,  # type: ignore[arg-type]
        source="oanda",
    )
    import pandas as pd  # type: ignore[import-untyped]  # local: else pandas-free

    ohlc_m30 = pd.DataFrame(
        {
            "open": dataset.open[FROZEN_INSTRUMENT_ID],
            "high": dataset.high[FROZEN_INSTRUMENT_ID],
            "low": dataset.low[FROZEN_INSTRUMENT_ID],
            "close": dataset.close[FROZEN_INSTRUMENT_ID],
        }
    )

    start_ns, end_ns = _ns(DEVELOPMENT_START), _ns(DEVELOPMENT_END_EXCLUSIVE)
    m1_bid_pd, m1_ask_pd = load_m1_bidask(
        instrument_id=FROZEN_INSTRUMENT_ID,
        root=root,
        start_utc=DEVELOPMENT_START,
        end_exclusive_utc=DEVELOPMENT_END_EXCLUSIVE,
    )
    m1_bid_bars = _query_m1_bars(root, FROZEN_INSTRUMENT_ID, "BID", start_ns, end_ns)
    m1_ask_bars = _query_m1_bars(root, FROZEN_INSTRUMENT_ID, "ASK", start_ns, end_ns)
    instrument = _instrument_from_catalog(root, FROZEN_INSTRUMENT_ID)

    config = load_oanda_alpha_lab_config()
    return ResolvedPartitionData(
        instrument=instrument,
        ohlc_m30=ohlc_m30,
        m1_bid_pd=m1_bid_pd,
        m1_ask_pd=m1_ask_pd,
        m1_bid_bars=m1_bid_bars,
        m1_ask_bars=m1_ask_bars,
        readiness_identity_sha256=config.semantic_sha256,
    )


def resolve_validation_data(
    *,
    validation_root: Path,
    universe_readiness_path: Path = DEFAULT_VALIDATION_READINESS_PATH,
) -> ResolvedPartitionData:
    """Resolve VALIDATION-partition data for the frozen candidate by reusing
    :func:`~ftmoquant.research.alpha_lab.pair_specific_validation.
    load_candidate_data` against ``CANDIDATE_C`` unchanged -- the exact same
    readiness-verified, holdout-firewalled loader the pair-specific
    VALIDATION stage already used for this candidate."""

    ohlc_m30, m1_bid_pd, m1_ask_pd = load_candidate_data(
        candidate=CANDIDATE_C,
        validation_root=validation_root,
        universe_readiness_path=universe_readiness_path,
    )
    root = validation_root / oanda_symbol(CANDIDATE_C.dataset_symbol)
    start_ns, end_ns = _ns(VALIDATION_START), _ns(HOLDOUT_START)
    m1_bid_bars = _query_m1_bars(root, FROZEN_INSTRUMENT_ID, "BID", start_ns, end_ns)
    m1_ask_bars = _query_m1_bars(root, FROZEN_INSTRUMENT_ID, "ASK", start_ns, end_ns)
    instrument = _instrument_from_catalog(root, FROZEN_INSTRUMENT_ID)

    readiness_document = json.loads(universe_readiness_path.read_text(encoding="utf-8"))
    return ResolvedPartitionData(
        instrument=instrument,
        ohlc_m30=ohlc_m30,
        m1_bid_pd=m1_bid_pd,
        m1_ask_pd=m1_ask_pd,
        m1_bid_bars=m1_bid_bars,
        m1_ask_bars=m1_ask_bars,
        readiness_identity_sha256=readiness_document.get("semantic_sha256", ""),
    )


# ---------------------------------------------------------------------------
# Pure engine core -- no partition concept, no catalog access; directly
# usable from synthetic parity tests.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EngineRunOutcome:
    alpha_lab_trades: tuple[Trade, ...]
    alpha_lab_skips: tuple[SkipRecord, ...]
    instructions: tuple[TradeInstruction, ...]
    submissions: tuple[OrderSubmissionRecord, ...]
    fills: tuple[FillRecord, ...]
    completed_trades: tuple[CompletedNautilusTrade, ...]
    equity_points: tuple[EquityPoint, ...]
    order_report_rows: int
    fill_report_rows: int
    position_report_rows: int
    initial_capital: Decimal


def run_frozen_signal_backtest(
    *,
    start: datetime,
    end_exclusive: datetime,
    instrument: CurrencyPair,
    ohlc_m30: Any,
    m1_bid_pd: Any,
    m1_ask_pd: Any,
    m1_bid_bars: tuple[Bar, ...],
    m1_ask_bars: tuple[Bar, ...],
) -> EngineRunOutcome:
    """Precompute the frozen Alpha Lab trade lifecycle (unchanged, see
    :func:`~ftmoquant.strategies.usdcad_sweep_bos_retest.
    precompute_alpha_lab_trades`), then mechanically replay it into one real
    Nautilus ``BacktestEngine`` over ``[start, end_exclusive)``. Must never
    be given real catalog data with caller-chosen dates from the CLI -- the
    only production entry points are :func:`run_development` and
    :func:`run_validation`, which always resolve dates via the frozen
    :func:`partition_bounds`."""

    start_ns, end_ns = _ns(start), _ns(end_exclusive)
    _validate_bar_stream(
        m1_bid_bars,
        f"{FROZEN_INSTRUMENT_ID}-{_M1_BAR_TYPE}-BID-EXTERNAL",
        start_ns,
        end_ns,
    )
    _validate_bar_stream(
        m1_ask_bars,
        f"{FROZEN_INSTRUMENT_ID}-{_M1_BAR_TYPE}-ASK-EXTERNAL",
        start_ns,
        end_ns,
    )

    trades, skips = precompute_alpha_lab_trades(
        high=ohlc_m30["high"],
        low=ohlc_m30["low"],
        close=ohlc_m30["close"],
        bid_m1=m1_bid_pd,
        ask_m1=m1_ask_pd,
    )
    instructions = trade_instructions_from_alpha_lab_trades(trades)

    profile = canonical_execution_profile()
    account = AccountParameters(
        initial_capital=Decimal("100000"), currency="USD", leverage=Decimal(1)
    )
    identity = _sha256_json(
        {
            "module": RUN_MODULE_NAME,
            "instrument": FROZEN_INSTRUMENT_ID,
            "family": FROZEN_FAMILY,
            "timeframe": FROZEN_TIMEFRAME,
            "swing_lookback": FROZEN_SWING_LOOKBACK,
            "rr": FROZEN_RR,
            "start_utc": _iso(start),
            "end_exclusive_utc": _iso(end_exclusive),
            "execution_profile": EXECUTION_PROFILE_LABEL,
        }
    )
    engine = BacktestEngine(_engine_config(identity))
    executor = UsdCadSweepBosRetestExecutor(
        instrument=_instrument_for_profile(instrument, profile.fee),
        instructions=instructions,
        start_ns=start_ns,
        end_exclusive_ns=end_ns,
        initial_capital=account.initial_capital,
    )
    try:
        fill_model = ProbabilisticFillModel(
            prob_fill_on_limit=float(profile.fill_on_limit_probability),
            prob_slippage=float(profile.adverse_slippage_probability),
            random_seed=profile.random_seed,
        )
        latency_model = StaticLatencyModel(
            base_latency_nanos=profile.base_latency_ns,
            insert_latency_nanos=profile.insert_latency_ns,
            update_latency_nanos=profile.update_latency_ns,
            cancel_latency_nanos=profile.cancel_latency_ns,
        )
        _add_oanda_venue(
            engine,
            account,
            profile,
            fill_model,
            latency_model,
            _fee_model(profile.fee),
            _rollover_modules(profile.rollover),
        )
        engine.add_instrument(_instrument_for_profile(instrument, profile.fee))
        all_bars = sorted(
            (*m1_bid_bars, *m1_ask_bars),
            key=lambda bar: (bar.ts_event, str(bar.bar_type)),
        )
        engine.add_data(all_bars, validate=True, sort=True)
        engine.add_strategy(executor)
        engine.run(
            start=start_ns, end=end_ns - 1, run_config_id=f"USDCAD-SBR-{identity[:20]}"
        )
        order_report = engine.generate_orders_report()
        fill_report = engine.generate_fills_report()
        position_report = engine.generate_positions_report()
    finally:
        engine.dispose()

    return EngineRunOutcome(
        alpha_lab_trades=trades,
        alpha_lab_skips=skips,
        instructions=instructions,
        submissions=tuple(executor.submissions),
        fills=tuple(executor.fills),
        completed_trades=tuple(executor.completed_trades),
        equity_points=tuple(executor.equity_points),
        order_report_rows=len(order_report),
        fill_report_rows=len(fill_report),
        position_report_rows=len(position_report),
        initial_capital=account.initial_capital,
    )


# ---------------------------------------------------------------------------
# Performance measurement (reporting only). No promotion verdict computed
# by these functions; run_development/run_validation only report numbers.
# ---------------------------------------------------------------------------


def _daily_returns(
    equity_points: tuple[EquityPoint, ...], initial_capital: Decimal
) -> tuple[tuple[date, float], ...]:
    return tuple(
        (
            datetime.fromtimestamp(
                current.information_time_ns / 1_000_000_000, tz=UTC
            ).date(),
            float((current.equity - previous.equity) / initial_capital),
        )
        for previous, current in zip(equity_points, equity_points[1:])
    )


def _holding_times_ns(entries_exits: Sequence[tuple[int, int]]) -> tuple[int, ...]:
    return tuple(exit_ns - entry_ns for entry_ns, exit_ns in entries_exits)


@dataclass(frozen=True, slots=True)
class PerformanceSummary:
    signal_count: int
    trade_count: int
    skipped_trade_count: int
    net_return: float
    annualized_sharpe: float | None
    maximum_drawdown: float
    win_rate: float
    gross_profit: str
    gross_loss: str
    commission_total: str
    rollover_total: str
    slippage_total: str
    average_r_outcome: float | None
    median_holding_time_ns: int | None
    longest_holding_time_ns: int | None


def _nautilus_performance(outcome: EngineRunOutcome) -> PerformanceSummary:
    daily = _daily_returns(outcome.equity_points, outcome.initial_capital)
    values = [value for _, value in daily]
    net_return = float(
        (outcome.equity_points[-1].equity - outcome.equity_points[0].equity)
        / outcome.initial_capital
    )
    completed = outcome.completed_trades
    wins = [t for t in completed if t.realized_pnl > 0]
    losses = [t for t in completed if t.realized_pnl < 0]
    gross_profit = sum((t.realized_pnl for t in wins), Decimal(0))
    gross_loss = sum((t.realized_pnl for t in losses), Decimal(0))
    commission_total = sum((t.commission for t in completed), Decimal(0))
    r_values = [float(t.net_r) for t in completed if t.net_r is not None]
    holding = _holding_times_ns([(t.entry_time_ns, t.exit_time_ns) for t in completed])
    return PerformanceSummary(
        signal_count=0,  # patched by the caller (_run_and_write), which knows it
        trade_count=len(completed),
        skipped_trade_count=len(outcome.alpha_lab_skips),
        net_return=net_return,
        annualized_sharpe=_annualized_sharpe(values),
        maximum_drawdown=_maximum_drawdown(values),
        win_rate=(len(wins) / len(completed)) if completed else 0.0,
        gross_profit=str(gross_profit),
        gross_loss=str(gross_loss),
        commission_total=str(commission_total),
        rollover_total="0",
        slippage_total="0",
        average_r_outcome=(sum(r_values) / len(r_values)) if r_values else None,
        median_holding_time_ns=int(statistics.median(holding)) if holding else None,
        longest_holding_time_ns=max(holding) if holding else None,
    )


def _alpha_lab_performance(
    trades: tuple[Trade, ...], skips: tuple[SkipRecord, ...], signal_count: int
) -> PerformanceSummary:
    if not trades:
        return PerformanceSummary(
            signal_count=signal_count,
            trade_count=0,
            skipped_trade_count=len(skips),
            net_return=0.0,
            annualized_sharpe=None,
            maximum_drawdown=0.0,
            win_rate=0.0,
            gross_profit="0",
            gross_loss="0",
            commission_total="0",
            rollover_total="0",
            slippage_total="0",
            average_r_outcome=None,
            median_holding_time_ns=None,
            longest_holding_time_ns=None,
        )
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    wins = 0
    gross_profit = 0.0
    gross_loss = 0.0
    r_values: list[float] = []
    for trade in trades:
        equity *= 1.0 + trade.return_frac
        peak = max(peak, equity)
        max_dd = min(max_dd, equity / peak - 1.0)
        if trade.return_frac > 0:
            wins += 1
            gross_profit += trade.return_frac
        elif trade.return_frac < 0:
            gross_loss += trade.return_frac
        risk = abs(trade.entry_price - trade.stop_price)
        if risk > 0:
            r_values.append(
                trade.direction * (trade.exit_price - trade.entry_price) / risk
            )
    holding = _holding_times_ns(
        [(int(t.entry_ts.value), int(t.exit_ts.value)) for t in trades]
    )
    return PerformanceSummary(
        signal_count=signal_count,
        trade_count=len(trades),
        skipped_trade_count=len(skips),
        net_return=equity - 1.0,
        annualized_sharpe=None,
        maximum_drawdown=max_dd,
        win_rate=wins / len(trades),
        gross_profit=str(gross_profit),
        gross_loss=str(gross_loss),
        commission_total="0",
        rollover_total="0",
        slippage_total="0",
        average_r_outcome=(sum(r_values) / len(r_values)) if r_values else None,
        median_holding_time_ns=int(statistics.median(holding)) if holding else None,
        longest_holding_time_ns=max(holding) if holding else None,
    )


# ---------------------------------------------------------------------------
# Orchestration + provenance.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UsdCadSweepBosRetestRunResult:
    partition: str
    start_utc: str
    end_exclusive_utc: str
    instrument_id: str
    execution_profile: str
    sizing_convention: str
    alpha_lab_performance: PerformanceSummary
    nautilus_performance: PerformanceSummary
    order_report_rows: int
    fill_report_rows: int
    position_report_rows: int
    submission_count: int
    fill_count: int


def run_development(
    *, readiness_path: Path, development_root_dir: Path, output_dir: Path
) -> UsdCadSweepBosRetestRunResult:
    data = resolve_development_data(
        readiness_path=readiness_path, development_root_dir=development_root_dir
    )
    return _run_and_write(
        partition=Partition.DEVELOPMENT,
        data=data,
        output_dir=output_dir,
    )


def run_validation(
    *,
    validation_root: Path,
    universe_readiness_path: Path,
    output_dir: Path,
) -> UsdCadSweepBosRetestRunResult:
    data = resolve_validation_data(
        validation_root=validation_root, universe_readiness_path=universe_readiness_path
    )
    return _run_and_write(
        partition=Partition.VALIDATION,
        data=data,
        output_dir=output_dir,
    )


def _run_and_write(
    *, partition: Partition, data: ResolvedPartitionData, output_dir: Path
) -> UsdCadSweepBosRetestRunResult:
    if output_dir.exists():
        raise UsdCadSweepBosRetestDevelopmentError(
            f"output directory already exists: {output_dir}"
        )
    start, end_exclusive = partition_bounds(partition)

    events = b2f1_sweep_bos_retest_signals(
        data.ohlc_m30["high"],
        data.ohlc_m30["low"],
        data.ohlc_m30["close"],
        swing_lookback=FROZEN_SWING_LOOKBACK,
        rr=FROZEN_RR,
    )
    signal_count = len(events)

    outcome = run_frozen_signal_backtest(
        start=start,
        end_exclusive=end_exclusive,
        instrument=data.instrument,
        ohlc_m30=data.ohlc_m30,
        m1_bid_pd=data.m1_bid_pd,
        m1_ask_pd=data.m1_ask_pd,
        m1_bid_bars=data.m1_bid_bars,
        m1_ask_bars=data.m1_ask_bars,
    )

    alpha_lab_perf = _alpha_lab_performance(
        outcome.alpha_lab_trades, outcome.alpha_lab_skips, signal_count
    )
    nautilus_perf = _nautilus_performance(outcome)
    nautilus_perf = PerformanceSummary(
        **{**asdict(nautilus_perf), "signal_count": signal_count}
    )

    result = UsdCadSweepBosRetestRunResult(
        partition=partition.value,
        start_utc=_iso(start),
        end_exclusive_utc=_iso(end_exclusive),
        instrument_id=FROZEN_INSTRUMENT_ID,
        execution_profile=EXECUTION_PROFILE_LABEL,
        sizing_convention=SIZING_CONVENTION_LABEL,
        alpha_lab_performance=alpha_lab_perf,
        nautilus_performance=nautilus_perf,
        order_report_rows=outcome.order_report_rows,
        fill_report_rows=outcome.fill_report_rows,
        position_report_rows=outcome.position_report_rows,
        submission_count=len(outcome.submissions),
        fill_count=len(outcome.fills),
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_result_artifacts(output_dir, result, outcome, data)
    return result


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _write_result_artifacts(
    output_dir: Path,
    result: UsdCadSweepBosRetestRunResult,
    outcome: EngineRunOutcome,
    data: ResolvedPartitionData,
) -> None:
    result_bytes = _canonical_bytes(asdict(result)) + b"\n"
    result_path = output_dir / "result.json"
    result_path.write_bytes(result_bytes)

    trades_path = output_dir / "trades.csv"
    with trades_path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "trade_index",
            "direction",
            "alpha_lab_entry_ns",
            "alpha_lab_exit_ns",
            "alpha_lab_entry_price",
            "alpha_lab_exit_price",
            "alpha_lab_exit_reason",
            "nautilus_entry_ns",
            "nautilus_exit_ns",
            "nautilus_entry_price",
            "nautilus_exit_price",
            "nautilus_realized_pnl",
            "nautilus_net_r",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        nautilus_by_index = {t.trade_index: t for t in outcome.completed_trades}
        for index, trade in enumerate(outcome.alpha_lab_trades):
            nautilus_trade = nautilus_by_index.get(index)
            writer.writerow(
                {
                    "trade_index": index,
                    "direction": trade.direction,
                    "alpha_lab_entry_ns": int(trade.entry_ts.value),
                    "alpha_lab_exit_ns": int(trade.exit_ts.value),
                    "alpha_lab_entry_price": trade.entry_price,
                    "alpha_lab_exit_price": trade.exit_price,
                    "alpha_lab_exit_reason": trade.exit_reason,
                    "nautilus_entry_ns": nautilus_trade.entry_time_ns
                    if nautilus_trade
                    else "",
                    "nautilus_exit_ns": nautilus_trade.exit_time_ns
                    if nautilus_trade
                    else "",
                    "nautilus_entry_price": nautilus_trade.entry_price
                    if nautilus_trade
                    else "",
                    "nautilus_exit_price": nautilus_trade.exit_price
                    if nautilus_trade
                    else "",
                    "nautilus_realized_pnl": nautilus_trade.realized_pnl
                    if nautilus_trade
                    else "",
                    "nautilus_net_r": nautilus_trade.net_r if nautilus_trade else "",
                }
            )

    frozen_identity = {
        "family": FROZEN_FAMILY,
        "instrument_id": FROZEN_INSTRUMENT_ID,
        "timeframe": FROZEN_TIMEFRAME,
        "swing_lookback": FROZEN_SWING_LOOKBACK,
        "rr": FROZEN_RR,
        "strategy_semantic_id": STRATEGY_SEMANTIC_ID,
    }
    write_runtime_provenance(
        output_dir / "runtime_provenance.json",
        {
            "git_commit": _git_commit(),
            "python_module": RUN_MODULE_NAME,
            "frozen_candidate_identity": frozen_identity,
            "frozen_candidate_identity_sha256": _sha256_json(frozen_identity),
            "readiness_identity_sha256": data.readiness_identity_sha256,
            "execution_profile": EXECUTION_PROFILE_LABEL,
            "execution_profile_caveat": EXECUTION_PROFILE_CAVEAT,
            "sizing_convention": SIZING_CONVENTION_LABEL,
            "base_reference_notional": str(BASE_RESEARCH_UNITS),
            "nautilus_trader_version": NAUTILUS_VERSION,
            "nautilus_trader_installed_version": version("nautilus_trader"),
            "promotion_gates_evaluated": False,
            "validation_accessed": result.partition == Partition.VALIDATION.value,
            "final_holdout_accessed": False,
        },
    )

    manifest = {
        result_path.name: __import__("hashlib").sha256(result_bytes).hexdigest()
    }
    (output_dir / "artifact_hashes.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the frozen usdcad_sweep_bos_retest_v1 signal "
            "(B2F1_sweep_bos_retest, USD/CAD.OANDA, M30, swing_lookback=40, "
            "rr=2.0) through the real Nautilus execution harness for one "
            "already-frozen partition."
        )
    )
    parser.add_argument(
        "--partition", choices=("development", "validation"), required=True
    )
    parser.add_argument("--catalog-root", type=Path, required=True)
    parser.add_argument("--universe-readiness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    partition = parse_partition(args.partition)
    if partition is Partition.DEVELOPMENT:
        result = run_development(
            readiness_path=args.universe_readiness,
            development_root_dir=args.catalog_root,
            output_dir=args.output,
        )
    else:
        result = run_validation(
            validation_root=args.catalog_root,
            universe_readiness_path=args.universe_readiness,
            output_dir=args.output,
        )
    print(json.dumps(asdict(result), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
