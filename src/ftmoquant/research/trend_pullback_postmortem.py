"""Development-only diagnostic post-mortem for the frozen G1.3 baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

SOURCE_EXPERIMENT_ID = "g1.3-trend_pullback_v1-first-frozen-baseline"
SOURCE_EXPERIMENT_SHA256 = (
    "db5c372bdbb8d549ed0baef1ba5f00ca96d98c7a7528226274270966d51afaa5"
)
SOURCE_COMMIT = "4bb4ec507cfd5bf4cb1a47cd8c6972eeafb6c248"
DEVELOPMENT_START = "2019-03-11T00:00:00Z"
DEVELOPMENT_END_EXCLUSIVE = "2023-04-11T00:00:00Z"
HOLDOUT_START = "2024-08-21T00:00:00Z"
SESSION_CONVENTION = {
    "Asia": "00:00-07:59 UTC",
    "London": "08:00-12:59 UTC",
    "New_York": "13:00-21:59 UTC",
    "Rollover_off_hours": "22:00-23:59 UTC",
}
WEEKDAYS = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)

Trade = Mapping[str, Any]


def build_postmortem(source_path: Path, output_path: Path) -> dict[str, Any]:
    """Build one deterministic report without running or reconstructing strategy."""

    if output_path.exists():
        raise FileExistsError(f"post-mortem artifact already exists: {output_path}")
    source = _load_json(source_path)
    _validate_source(source)
    development = cast(dict[str, Any], source["results"]["development"])
    trades = cast(list[Trade], development["trades"])
    diagnostics = development_diagnostics(trades)
    report: dict[str, Any] = {
        "schema": "ftmoquant.g1.3-development-postmortem",
        "schema_version": 1,
        "label": "EXPLORATORY_NON_CONFIRMATORY",
        "source_experiment": {
            "identity": SOURCE_EXPERIMENT_ID,
            "semantic_sha256": SOURCE_EXPERIMENT_SHA256,
            "artifact_file_sha256": _sha256_file(source_path),
            "git_commit": SOURCE_COMMIT,
            "strategy": source["strategy"],
            "dataset_identity": source["dataset"]["identity"],
            "frozen_verdict": source["overall_verdict"],
        },
        "scope": {
            "development_start_inclusive": DEVELOPMENT_START,
            "development_end_exclusive": DEVELOPMENT_END_EXCLUSIVE,
            "development_trade_count": len(trades),
            "validation_subgroup_data_analyzed": False,
            "validation_note": (
                "Only the already-recorded aggregate FAIL is acknowledged; no "
                "validation trade, feature, time, direction, exit, or subgroup was "
                "used in this post-mortem."
            ),
            "holdout_start_inclusive": HOLDOUT_START,
            "holdout_accessed": False,
            "strategy_rerun": False,
            "strategy_variant_run": False,
            "parameter_optimization": False,
        },
        "diagnostics": diagnostics,
        "failure_assessment": _failure_assessment(diagnostics),
        "candidate_hypotheses": _candidate_hypotheses(),
        "family_recommendation": {
            "classification": "RETAINED_ONLY_AS_RESEARCH_REFERENCE",
            "rationale": (
                "The baseline is broadly negative, with no robust descriptive "
                "support for incremental EMA/pullback tuning. Preserve it as a "
                "falsified reference; do not create trend_pullback_v1.1."
            ),
        },
        "future_evaluation_protocol": _future_protocol(),
    }
    report["semantic_sha256"] = _sha256_json(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def development_diagnostics(trades: Sequence[Trade]) -> dict[str, Any]:
    """Compute only the predefined descriptive development diagnostics."""

    if not trades:
        raise ValueError("development post-mortem requires persisted trades")
    required = {
        "commissions",
        "direction",
        "entry_price",
        "entry_time_utc",
        "exit_price",
        "exit_reason",
        "exit_time_utc",
        "initial_risk",
        "net_r",
        "quantity",
        "realized_pnl",
        "stop_distance",
    }
    for trade in trades:
        missing = required - trade.keys()
        if missing:
            raise ValueError(f"development trade lacks fields: {sorted(missing)}")

    overall = _summary(trades)
    average_win = Decimal(cast(str, overall["average_win_r"]))
    average_loss = abs(Decimal(cast(str, overall["average_loss_r"])))
    break_even = average_loss / (average_win + average_loss)
    observed_win_rate = Decimal(cast(str, overall["win_rate"]))
    return {
        "overall": overall,
        "direction": _group_table(trades, lambda trade: str(trade["direction"])),
        "calendar_year_by_exit": _group_table(
            trades, lambda trade: str(trade["exit_time_utc"])[:4]
        ),
        "exit_mechanics": {
            "by_exit_reason": _group_table(
                trades, lambda trade: str(trade["exit_reason"])
            ),
            "realized_win_distribution_r": _distribution(
                [_r(trade) for trade in trades if _r(trade) > 0]
            ),
            "realized_loss_distribution_r": _distribution(
                [_r(trade) for trade in trades if _r(trade) < 0]
            ),
            "observed_win_rate": _decimal(observed_win_rate),
            "break_even_win_rate_at_observed_average_payoffs": _decimal(break_even),
            "win_rate_shortfall_percentage_points": _decimal(
                (break_even - observed_win_rate) * 100
            ),
            "intended_geometry_assessment": (
                "Target exits cluster near +2R and stop exits near -1R, but "
                "market execution produces tails beyond both levels. The main "
                "deficit is win probability, not collapse of the 2R/1R geometry."
            ),
        },
        "entry_timing": {
            "timestamp_basis": "persisted native entry fill time",
            "session_convention": SESSION_CONVENTION,
            "hour_utc": _group_table(
                trades, lambda trade: str(trade["entry_time_utc"])[11:13]
            ),
            "session": _group_table(trades, _session),
            "weekday": _ordered_group_table(trades, _weekday, WEEKDAYS),
        },
        "signal_state": {
            "volatility_proxy": (
                "Persisted stop_distance, equal to frozen 1.5 * trigger ATR; "
                "quartiles contain 115 trades each and are descriptive only."
            ),
            "stop_distance_quartiles": _volatility_quartiles(trades),
            "available_trend_state": "entry direction only",
            "unavailable": [
                "pullback depth relative to EMA or ATR",
                "signal-to-fill delay",
                "armed duration before trigger",
                (
                    "signal-time EMA and ATR values other than ATR implied by "
                    "stop distance"
                ),
            ],
        },
        "costs_execution": _cost_diagnostics(trades),
    }


def _summary(trades: Sequence[Trade]) -> dict[str, Any]:
    values = [_r(trade) for trade in trades]
    winners = [value for value in values if value > 0]
    losers = [value for value in values if value < 0]
    total = sum(values, Decimal(0))
    gross_profit = sum(winners, Decimal(0))
    gross_loss = -sum(losers, Decimal(0))
    return {
        "trade_count": len(values),
        "winners": len(winners),
        "losers": len(losers),
        "win_rate": _decimal(Decimal(len(winners)) / len(values)),
        "mean_r": _decimal(total / len(values)),
        "median_r": _decimal(_quantile(values, Decimal("0.5"))),
        "total_r": _decimal(total),
        "profit_factor": _decimal(
            gross_profit / gross_loss if gross_loss > 0 else None
        ),
        "average_win_r": _decimal(gross_profit / len(winners) if winners else None),
        "average_loss_r": _decimal(
            sum(losers, Decimal(0)) / len(losers) if losers else None
        ),
    }


def _group_table(
    trades: Sequence[Trade], key: Callable[[Trade], str]
) -> list[dict[str, Any]]:
    grouped: defaultdict[str, list[Trade]] = defaultdict(list)
    for trade in trades:
        grouped[key(trade)].append(trade)
    return [{"group": group, **_summary(grouped[group])} for group in sorted(grouped)]


def _ordered_group_table(
    trades: Sequence[Trade],
    key: Callable[[Trade], str],
    order: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: defaultdict[str, list[Trade]] = defaultdict(list)
    for trade in trades:
        grouped[key(trade)].append(trade)
    return [
        {"group": group, **_summary(grouped[group])}
        for group in order
        if grouped[group]
    ]


def _volatility_quartiles(trades: Sequence[Trade]) -> list[dict[str, Any]]:
    ordered = sorted(
        enumerate(trades),
        key=lambda item: (Decimal(str(item[1]["stop_distance"])), item[0]),
    )
    grouped: list[list[Trade]] = [[], [], [], []]
    for rank, (_, trade) in enumerate(ordered):
        grouped[min(rank * 4 // len(ordered), 3)].append(trade)
    result: list[dict[str, Any]] = []
    for index, group in enumerate(grouped, start=1):
        distances = [Decimal(str(trade["stop_distance"])) for trade in group]
        result.append(
            {
                "group": f"Q{index}",
                "minimum_stop_distance": _decimal(min(distances)),
                "maximum_stop_distance": _decimal(max(distances)),
                **_summary(group),
            }
        )
    return result


def _distribution(values: Sequence[Decimal]) -> dict[str, Any]:
    return {
        "count": len(values),
        "minimum": _decimal(min(values)),
        "q25": _decimal(_quantile(values, Decimal("0.25"))),
        "median": _decimal(_quantile(values, Decimal("0.5"))),
        "q75": _decimal(_quantile(values, Decimal("0.75"))),
        "maximum": _decimal(max(values)),
        "mean": _decimal(sum(values, Decimal(0)) / len(values)),
    }


def _quantile(values: Sequence[Decimal], probability: Decimal) -> Decimal:
    """Deterministic nearest-rank-on-index descriptive quantile."""

    ordered = sorted(values)
    index = int((Decimal(len(ordered) - 1) * probability).to_integral_value())
    return ordered[index]


def _cost_diagnostics(trades: Sequence[Trade]) -> dict[str, Any]:
    executed_price_r: list[Decimal] = []
    for trade in trades:
        quantity = Decimal(str(trade["quantity"]))
        entry = Decimal(str(trade["entry_price"]))
        exit_price = Decimal(str(trade["exit_price"]))
        risk = Decimal(str(trade["initial_risk"]))
        price_pnl = (
            quantity * (exit_price - entry)
            if trade["direction"] == "long"
            else quantity * (entry - exit_price)
        )
        executed_price_r.append(price_pnl / risk)
    native_net = [_r(trade) for trade in trades]
    commissions = sum(
        (Decimal(str(trade["commissions"])) for trade in trades), Decimal(0)
    )
    return {
        "executed_price_pnl_after_spread_before_commission": {
            "mean_r": _decimal(
                sum(executed_price_r, Decimal(0)) / len(executed_price_r)
            ),
            "total_r": _decimal(sum(executed_price_r, Decimal(0))),
        },
        "native_net": {
            "mean_r": _decimal(sum(native_net, Decimal(0)) / len(native_net)),
            "total_r": _decimal(sum(native_net, Decimal(0))),
        },
        "commission_total_currency": _decimal(commissions),
        "interpretation": (
            "The observed executed-price expectancy is negative before commission, "
            "and commission was zero. Spread is already embedded in BID/ASK fills. "
            "The ledger lacks counterfactual mid fills and intended pre-fill prices, "
            "so raw pre-spread signal expectancy, spread cost, slippage, and entry "
            "delay cannot be isolated. The evidence rules out commission as the "
            "cause but cannot cleanly separate raw signal from spread."
        ),
        "unavailable": [
            "counterfactual pre-spread gross R",
            "spread cost per trade",
            "slippage relative to intended price",
            "signal-to-fill entry delay",
        ],
    }


def _failure_assessment(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    mechanics = cast(Mapping[str, Any], diagnostics["exit_mechanics"])
    return {
        "primary_mechanism": "INSUFFICIENT_WIN_PROBABILITY",
        "evidence": (
            f"Observed development win rate was {mechanics['observed_win_rate']} "
            "against a break-even rate of "
            f"{mechanics['break_even_win_rate_at_observed_average_payoffs']}; "
            "stops outnumbered targets while their realized sizes remained near "
            "the intended -1R/+2R geometry."
        ),
        "breadth": (
            "Broad rather than attributable to one obvious subset: both directions, "
            "four of five exit-calendar years, all three major session buckets, "
            "and all four volatility quartiles had negative mean R."
        ),
        "secondary_observations": [
            "Short trades were materially worse than long trades, but longs also lost.",
            "The highest stop-distance quartile was worst, but every quartile lost.",
            "A few hours and weekdays were positive post hoc; they are fragmented "
            "exploratory cells and are not evidence for filters.",
            "Target and stop execution tails show some gap/market-exit degradation, "
            "but payoff geometry did not collapse.",
        ],
        "premise_support": (
            "No robust descriptive support. The EMA-trend plus pullback-continuation "
            "premise failed broadly in development and its already-recorded aggregate "
            "validation result was also negative."
        ),
    }


def _candidate_hypotheses() -> list[dict[str, Any]]:
    return [
        {
            "name": "session_opening_range_expansion",
            "classification": "GENUINELY_NEW_VOLATILITY_BREAKOUT_FAMILY",
            "mechanism": (
                "Liquidity and information arrival at regional session transitions "
                "can release overnight compression into persistent range expansion."
            ),
            "signal_concept": (
                "Preregister a session-clock-defined compression and opening-range "
                "breakout using externally justified, fixed definitions—not values "
                "selected from the post-mortem cells."
            ),
            "why_it_addresses_failure": (
                "It demands observable expansion before entry instead of assuming "
                "an EMA retrigger has continuation probability."
            ),
            "required_data": (
                "Research-ready BID/ASK intraday bars, DST-safe session calendar, "
                "and the same explicit execution model."
            ),
            "main_falsification_criterion": (
                "Positive preregistered net expectancy and uncertainty gate on new "
                "prospective validation data; otherwise reject the family."
            ),
            "overfitting_risk": (
                "High if session windows, compression measures, or breakout buffers "
                "are selected from historical return rankings."
            ),
        },
        {
            "name": "macro_rate_differential_direction",
            "classification": "GENUINELY_NEW_MACRO_FAMILY",
            "mechanism": (
                "Persistent EUR/USD pressure may arise from expected monetary-policy "
                "and real-yield differentials rather than price-only EMA state."
            ),
            "signal_concept": (
                "Use a preregistered low-frequency directional state derived from "
                "vintage-safe policy-rate, yield, and inflation-expectation inputs."
            ),
            "why_it_addresses_failure": (
                "It supplies an economic source of direction and operates at a lower "
                "turnover than the failed frequent price-only pullback entries."
            ),
            "required_data": (
                "Point-in-time ECB/Fed policy expectations, EUR/USD BID/ASK data, "
                "release timestamps, and revision/vintage metadata."
            ),
            "main_falsification_criterion": (
                "No positive net expectancy across a preregistered prospective period "
                "with directionally symmetric rules."
            ),
            "overfitting_risk": (
                "High from macro-series selection, revised data, lag choices, and "
                "event-window flexibility."
            ),
        },
        {
            "name": "liquidity_shock_mean_reversion",
            "classification": "GENUINELY_NEW_MICROSTRUCTURE_FAMILY",
            "mechanism": (
                "Temporary spread/range dislocations can reflect short-lived liquidity "
                "imbalance that reverts once quotes normalize."
            ),
            "signal_concept": (
                "Preregister an abnormal spread-and-range shock followed by quote "
                "normalization, with conservative marketable execution."
            ),
            "why_it_addresses_failure": (
                "It tests reversion after observable liquidity stress instead of "
                "continuation after a common moving-average crossing."
            ),
            "required_data": (
                "Tick or quote-level BID/ASK observations with sizes, provider-quality "
                "flags, and realistic latency/slippage evidence."
            ),
            "main_falsification_criterion": (
                "Net reversion payoff must remain positive after preregistered spread, "
                "latency, and adverse-selection costs on prospective data."
            ),
            "overfitting_risk": (
                "Very high from shock percentile, normalization window, venue quality, "
                "and stale-quote filtering choices."
            ),
        },
    ]


def _future_protocol() -> dict[str, Any]:
    return {
        "status": "PROPOSED_NOT_EXECUTED",
        "steps": [
            (
                "Use the old development period only for mechanism documentation "
                "and engineering tests; never present its post-mortem subgroups as "
                "confirmation."
            ),
            "Select at most one candidate and preregister its complete signal, costs, "
            "primary statistic, minimum count, uncertainty method, and failure gates.",
            (
                "Do not reuse the already-observed 2023-04-11 through 2024-08-20 "
                "validation period as clean evidence for an invented hypothesis."
            ),
            "Collect and seal genuinely prospective EUR/USD observations beginning "
            "after preregistration (recommended no earlier than 2026-08-17 UTC) and "
            "evaluate exactly once after a fixed horizon and minimum trade count.",
            "Keep the original >=2024-08-21 final holdout sealed during this process; "
            "consider its governed use only after an independent prospective pass.",
        ],
        "clean_validation_source": (
            "Prospectively collected post-preregistration data, not any portion of the "
            "observed original validation period."
        ),
        "original_final_holdout_remains_sealed": True,
    }


def _session(trade: Trade) -> str:
    hour = int(str(trade["entry_time_utc"])[11:13])
    if hour < 8:
        return "Asia"
    if hour < 13:
        return "London"
    if hour < 22:
        return "New_York"
    return "Rollover_off_hours"


def _weekday(trade: Trade) -> str:
    timestamp = datetime.fromisoformat(
        str(trade["entry_time_utc"]).replace("Z", "+00:00")
    )
    if timestamp.tzinfo != UTC:
        raise ValueError("entry timestamp must be UTC")
    return timestamp.strftime("%A")


def _r(trade: Trade) -> Decimal:
    return Decimal(str(trade["net_r"]))


def _validate_source(source: Mapping[str, Any]) -> None:
    claimed = source.get("semantic_sha256")
    material = dict(source)
    material.pop("semantic_sha256", None)
    if claimed != SOURCE_EXPERIMENT_SHA256 or _sha256_json(material) != claimed:
        raise ValueError("source G1.3 artifact semantic hash is invalid")
    if source.get("experiment_id") != SOURCE_EXPERIMENT_ID:
        raise ValueError("unexpected source experiment")
    if source.get("git_commit") != SOURCE_COMMIT:
        raise ValueError("unexpected source experiment commit")
    if source.get("holdout_accessed") is not False:
        raise ValueError("source experiment reports holdout access")
    boundaries = cast(Mapping[str, Any], source["boundaries"])
    development = cast(Mapping[str, Any], boundaries["development"])
    if (
        development.get("start_inclusive") != DEVELOPMENT_START
        or development.get("end_exclusive") != DEVELOPMENT_END_EXCLUSIVE
    ):
        raise ValueError("unexpected development boundary")


def _decimal(value: Decimal | None) -> str | None:
    return None if value is None else format(value, "f")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_artifact", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    report = build_postmortem(args.source_artifact, args.output_path)
    print(
        json.dumps(
            {
                "artifact": str(args.output_path),
                "semantic_sha256": report["semantic_sha256"],
                "label": report["label"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
