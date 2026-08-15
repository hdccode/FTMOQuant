"""One-shot G1.3 runner for the frozen ``trend_pullback_v1`` baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]
from nautilus_trader.backtest import BacktestEngine
from nautilus_trader.execution import ProbabilisticFillModel, StaticLatencyModel
from nautilus_trader.model import Bar, CurrencyPair
from nautilus_trader.persistence import ParquetDataCatalog

from ftmoquant.backtest.execution_harness import (
    AccountParameters,
    ExecutionProfile,
    _add_venue,
    _engine_config,
    _fee_model,
    _instrument_for_profile,
    _profile_dict,
    _sha256_json,
    _validate_profile,
    canonical_execution_profile,
)
from ftmoquant.data.dukascopy import NAUTILUS_VERSION
from ftmoquant.research.statistics import (
    StationaryBootstrapConfig,
    stationary_bootstrap_confidence_interval,
)
from ftmoquant.strategies.trend_pullback import (
    FROZEN_CONFIG_SHA256,
    CompletedTrade,
    Timeframe,
    TrendPullbackStrategy,
)

EXPERIMENT_ID = "g1.3-trend_pullback_v1-first-frozen-baseline"
EXPERIMENT_VERSION = "1.0.0"
EXPECTED_PARENT_COMMIT = "50df7945f85b941754684b5c5da3398a53282e98"
EXPECTED_DATASET_IDENTITY = "g1-dukascopy-corrected-1"
EXPECTED_READINESS_SHA256 = (
    "7c59ff86f428f9d1cf120cd54134de7c739ba5bf1e2a0a772abf53bc5b80cc0f"
)
READINESS_FILENAME = "ftmoquant_research_readiness.json"
DEVELOPMENT_START_NS = 1_552_262_400_000_000_000
VALIDATION_START_NS = 1_681_171_200_000_000_000
HOLDOUT_START_NS = 1_724_198_400_000_000_000
DAY_NS = 86_400_000_000_000


def run_experiment(dataset_root: Path, output_path: Path) -> dict[str, Any]:
    """Validate all gates, execute both frozen splits once, and write one artifact."""

    if output_path.exists():
        raise FileExistsError(f"one-shot artifact already exists: {output_path}")
    git_commit = _validate_preconditions(dataset_root)
    readiness_path = dataset_root / READINESS_FILENAME
    readiness = _load_json(readiness_path)
    profile = canonical_execution_profile()
    _validate_profile(profile)
    profile_dict = _profile_dict(profile)
    profile_sha256 = _sha256_json(profile_dict)
    account = AccountParameters(
        initial_capital=Decimal("100000"), currency="USD", leverage=Decimal("30")
    )

    development = _run_split(
        dataset_root=dataset_root,
        split_name="development",
        load_start_ns=DEVELOPMENT_START_NS,
        trading_start_ns=DEVELOPMENT_START_NS,
        end_ns=VALIDATION_START_NS,
        profile=profile,
        account=account,
        identity_material=f"{git_commit}:development:{profile_sha256}",
    )
    validation = _run_split(
        dataset_root=dataset_root,
        split_name="validation",
        load_start_ns=DEVELOPMENT_START_NS,
        trading_start_ns=VALIDATION_START_NS,
        end_ns=HOLDOUT_START_NS,
        profile=profile,
        account=account,
        identity_material=f"{git_commit}:validation:{profile_sha256}",
    )

    dev_metrics = _metrics(
        development["trades"], DEVELOPMENT_START_NS, VALIDATION_START_NS
    )
    val_metrics = _metrics(validation["trades"], VALIDATION_START_NS, HOLDOUT_START_NS)
    bootstrap = _bootstrap(validation["trades"])
    gates = _gates(dev_metrics, val_metrics, bootstrap)
    artifact: dict[str, Any] = {
        "schema": "ftmoquant.g1.3-frozen-baseline",
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "experiment_version": EXPERIMENT_VERSION,
        "strategy": {
            "name": "trend_pullback_v1",
            "version": "1.0.0",
            "config_semantic_sha256": FROZEN_CONFIG_SHA256,
            "parameter_variants": [],
        },
        "git_commit": git_commit,
        "required_parent_commit": EXPECTED_PARENT_COMMIT,
        "dataset": {
            "identity": readiness["dataset_identity"],
            "readiness_filename": READINESS_FILENAME,
            "readiness_file_sha256": _sha256_file(readiness_path),
            "readiness_semantic_sha256": readiness["semantic_sha256"],
            "canonical": readiness["canonical"],
            "derived": readiness["derived"],
        },
        "boundaries": {
            "development": {
                "start_inclusive": _iso(DEVELOPMENT_START_NS),
                "end_exclusive": _iso(VALIDATION_START_NS),
            },
            "validation": {
                "start_inclusive": _iso(VALIDATION_START_NS),
                "end_exclusive": _iso(HOLDOUT_START_NS),
                "warmup_start": _iso(DEVELOPMENT_START_NS),
                "warmup_trading_enabled": False,
            },
            "sealed_holdout_start_inclusive": _iso(HOLDOUT_START_NS),
            "boundary_enforcement": (
                "catalog query and local ts_init filter before engine.add_data"
            ),
        },
        "execution": {
            "identity": "canonical-g0.7-uncalibrated-baseline",
            "profile": profile_dict,
            "profile_sha256": profile_sha256,
            "account": {key: str(value) for key, value in asdict(account).items()},
            "engine": f"nautilus-trader=={NAUTILUS_VERSION}",
            "cost_note": (
                "native BID/ASK spread; zero added fee/slippage/latency; "
                "rollover disabled"
            ),
        },
        "seeds": {"execution": profile.random_seed, "bootstrap": 1729},
        "results": {
            "development": {
                "metrics": dev_metrics,
                "diagnostics": development["diagnostics"],
                "trades": [_trade_dict(item) for item in development["trades"]],
            },
            "validation": {
                "metrics": val_metrics,
                "diagnostics": validation["diagnostics"],
                "trades": [_trade_dict(item) for item in validation["trades"]],
            },
        },
        "bootstrap": bootstrap,
        "gates": gates,
        "overall_verdict": _overall_verdict(gates),
        "holdout_accessed": False,
        "holdout_rows_admitted": 0,
        "parameter_optimization_occurred": False,
        "baseline_run_count": 1,
    }
    artifact["semantic_sha256"] = _sha256_json(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return artifact


def _run_split(
    *,
    dataset_root: Path,
    split_name: str,
    load_start_ns: int,
    trading_start_ns: int,
    end_ns: int,
    profile: ExecutionProfile,
    account: AccountParameters,
    identity_material: str,
) -> dict[str, Any]:
    catalog = ParquetDataCatalog(str(dataset_root / "catalog"))
    instruments = catalog.instruments(["EUR/USD.DUKASCOPY"])
    if len(instruments) != 1 or not isinstance(instruments[0], CurrencyPair):
        raise RuntimeError("corrected catalog lacks the frozen EUR/USD instrument")
    instrument = _instrument_for_profile(instruments[0], profile.fee)
    strategy = TrendPullbackStrategy(
        research_ready=True,
        trading_start_ns=trading_start_ns,
        verified_zero_unexplained_omissions=True,
    )
    bar_types = strategy.required_bar_types
    bars: list[Bar] = []
    counts: dict[str, int] = {}
    for bar_type in bar_types:
        queried = catalog.query_bars([bar_type], start=load_start_ns, end=end_ns - 1)
        selected = [bar for bar in queried if load_start_ns <= bar.ts_init < end_ns]
        if any(bar.ts_init >= HOLDOUT_START_NS for bar in selected):
            raise RuntimeError("holdout bar reached the G1.3 loader")
        counts[bar_type] = len(selected)
        bars.extend(selected)
    if any(count == 0 for count in counts.values()):
        raise RuntimeError(f"{split_name} has an empty required bar stream")

    identity = hashlib.sha256(identity_material.encode()).hexdigest()
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
    engine = BacktestEngine(_engine_config(identity))
    try:
        _add_venue(
            engine,
            account,
            profile,
            fill_model,
            latency_model,
            _fee_model(profile.fee),
            (),
        )
        engine.add_instrument(instrument)
        engine.add_data(bars, validate=True, sort=True)
        engine.add_strategy(strategy)
        engine.run(
            start=load_start_ns, end=end_ns - 1, run_config_id=f"G13-{identity[:20]}"
        )
        reports = {
            "orders": engine.generate_orders_report(),
            "fills": engine.generate_fills_report(),
            "positions": engine.generate_positions_report(),
        }
        open_positions = len(engine.cache.positions_open())
    finally:
        engine.dispose()
    diagnostics = {
        "source_bar_counts": counts,
        "paired_bar_counts": {
            timeframe.value: strategy.pair_counts[timeframe] for timeframe in Timeframe
        },
        "entry_intents": strategy.entry_intent_count,
        "exit_intents": strategy.exit_intent_count,
        "entry_rejections": strategy.entry_rejection_count,
        "native_order_rows": len(reports["orders"]),
        "native_fill_rows": len(reports["fills"]),
        "native_position_rows": len(reports["positions"]),
        "completed_trade_records": len(strategy.completed_trades),
        "open_positions_at_boundary": open_positions,
        "indicators_initialized": strategy.state_machine.indicators_initialized,
    }
    return {"trades": tuple(strategy.completed_trades), "diagnostics": diagnostics}


def _metrics(
    trades: Sequence[CompletedTrade], start_ns: int, end_ns: int
) -> dict[str, Any]:
    values = [trade.net_r for trade in trades]
    winners = [value for value in values if value > 0]
    losers = [value for value in values if value < 0]
    total = sum(values, Decimal(0))
    mean = total / len(values) if values else None
    sorted_values = sorted(values)
    median: Decimal | None = None
    if sorted_values:
        middle = len(sorted_values) // 2
        median = (
            sorted_values[middle]
            if len(sorted_values) % 2
            else (sorted_values[middle - 1] + sorted_values[middle]) / 2
        )
    gross_profit = sum(winners, Decimal(0))
    gross_loss = -sum(losers, Decimal(0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
    equity = Decimal(0)
    peak = Decimal(0)
    max_drawdown = Decimal(0)
    for value in values:
        equity += value
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    years: defaultdict[str, Decimal] = defaultdict(Decimal)
    for trade in trades:
        years[
            datetime.fromtimestamp(trade.exit_time_ns / 1e9, tz=UTC).strftime("%Y")
        ] += trade.net_r
    directions = Counter(trade.direction.value for trade in trades)
    exits = Counter(trade.exit_reason.value for trade in trades)
    duration_days = Decimal(end_ns - start_ns) / Decimal(DAY_NS)
    return {
        "trade_count": len(trades),
        "winners": len(winners),
        "losers": len(losers),
        "breakeven": len(values) - len(winners) - len(losers),
        "win_rate": _decimal(Decimal(len(winners)) / len(values) if values else None),
        "mean_net_r": _decimal(mean),
        "median_net_r": _decimal(median),
        "total_net_r": _decimal(total),
        "profit_factor": _decimal(profit_factor),
        "max_drawdown_r": _decimal(max_drawdown),
        "average_win_r": _decimal(gross_profit / len(winners) if winners else None),
        "average_loss_r": _decimal(
            sum(losers, Decimal(0)) / len(losers) if losers else None
        ),
        "trade_frequency_per_year": _decimal(
            Decimal(len(trades)) * Decimal("365.2425") / duration_days
        ),
        "direction_counts": {key: directions.get(key, 0) for key in ("long", "short")},
        "calendar_year_net_r": {key: _decimal(years[key]) for key in sorted(years)},
        "exit_reason_counts": {
            key: exits.get(key, 0) for key in ("stop_loss", "take_profit", "time")
        },
    }


def _bootstrap(trades: Sequence[CompletedTrade]) -> dict[str, Any] | None:
    if not trades:
        return None
    config = StationaryBootstrapConfig(
        block_size=5,
        repetitions=10_000,
        seed=1729,
        confidence_level=0.95,
        method="bca",
    )
    series = pd.Series(
        [float(trade.net_r) for trade in trades], name="validation_net_r"
    )
    return cast(
        dict[str, Any],
        _jsonable(asdict(stationary_bootstrap_confidence_interval(series, config))),
    )


def _gates(
    dev: dict[str, Any], val: dict[str, Any], bootstrap: dict[str, Any] | None
) -> dict[str, Any]:
    dev_count = cast(int, dev["trade_count"])
    val_count = cast(int, val["trade_count"])
    dev_mean = _as_decimal(dev["mean_net_r"])
    val_mean = _as_decimal(val["mean_net_r"])
    profit_factor = _as_decimal(val["profit_factor"])
    total = _as_decimal(val["total_net_r"])
    years = cast(dict[str, str], val["calendar_year_net_r"])
    concentration_share: Decimal | None = None
    if total is not None and total > 0 and years:
        concentration_share = max(Decimal(value) for value in years.values()) / total
    return {
        "development_mean_net_r_gt_0": _status(dev_mean is not None and dev_mean > 0),
        "development_trade_count_gte_100": _count_status(dev_count, 100),
        "validation_mean_net_r_gt_0": _status(val_mean is not None and val_mean > 0),
        "validation_trade_count_gte_50": _count_status(val_count, 50),
        "validation_profit_factor_gt_1": _status(
            profit_factor is not None and profit_factor > 1
        ),
        "validation_bca_95_lower_bound_gt_0": (
            "UNRESOLVED"
            if bootstrap is None
            else _status(Decimal(str(bootstrap["lower_bound"])) > 0)
        ),
        "validation_calendar_concentration_lte_0_50": (
            "FAIL"
            if total is None or total <= 0
            else _status(
                concentration_share is not None
                and concentration_share <= Decimal("0.50")
            )
        ),
        "validation_max_single_year_net_r_share": _decimal(concentration_share),
    }


def _count_status(count: int, minimum: int) -> str:
    return "PASS" if count >= minimum else "UNRESOLVED"


def _status(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _overall_verdict(gates: dict[str, Any]) -> str:
    statuses = [value for key, value in gates.items() if not key.endswith("_share")]
    if "UNRESOLVED" in statuses:
        return "UNRESOLVED"
    return "PASS" if all(value == "PASS" for value in statuses) else "FAIL"


def _validate_preconditions(dataset_root: Path) -> str:
    if version("nautilus-trader") != NAUTILUS_VERSION:
        raise RuntimeError(f"expected NautilusTrader {NAUTILUS_VERSION}")
    status = subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout
    if status:
        raise RuntimeError("G1.3 requires a clean worktree")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", EXPECTED_PARENT_COMMIT, commit]
    )
    if ancestry.returncode != 0:
        raise RuntimeError("corrected-data commit is not incorporated")
    readiness = _load_json(dataset_root / READINESS_FILENAME)
    if readiness.get("dataset_identity") != EXPECTED_DATASET_IDENTITY:
        raise RuntimeError("corrected dataset identity mismatch")
    if readiness.get("semantic_sha256") != EXPECTED_READINESS_SHA256:
        raise RuntimeError("research-readiness semantic hash mismatch")
    if readiness.get("research_ready") is not True:
        raise RuntimeError("dataset is not research ready")
    gates = readiness.get("gates", {})
    if gates.get("zero_unexplained_omissions") is not True:
        raise RuntimeError("zero unexplained omissions is not proven")
    if (
        readiness.get("holdout_accessed") is not False
        or readiness.get("holdout_rows_admitted") != 0
    ):
        raise RuntimeError("sealed holdout precondition failed")
    if (
        readiness.get("strategy_return_accessed") is not False
        or readiness.get("g1_3_run") is not False
    ):
        raise RuntimeError("the first frozen baseline was already accessed")
    return commit


def _trade_dict(trade: CompletedTrade) -> dict[str, Any]:
    value = cast(dict[str, Any], _jsonable(asdict(trade)))
    value["entry_time_utc"] = _iso(trade.entry_time_ns)
    value["exit_time_utc"] = _iso(trade.exit_time_ns)
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iso(value_ns: int) -> str:
    return (
        datetime.fromtimestamp(value_ns / 1e9, tz=UTC)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _decimal(value: Decimal | None) -> str | None:
    return None if value is None else format(value, "f")


def _as_decimal(value: Any) -> Decimal | None:
    return None if value is None else Decimal(cast(str, value))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    artifact = run_experiment(args.dataset_root, args.output_path)
    print(
        json.dumps(
            {
                "artifact": str(args.output_path),
                "semantic_sha256": artifact["semantic_sha256"],
                "verdict": artifact["overall_verdict"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
