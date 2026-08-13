import hashlib
import json
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from nautilus_trader.model import Bar
from nautilus_trader.persistence import ParquetDataCatalog

from ftmoquant.backtest.execution_harness import (
    AccountParameters,
    CalibrationStatus,
    ExecutionProfile,
    ExecutionRunRequest,
    ExecutionValidationError,
    FeeModelConfig,
    FeeModelKind,
    InterestRateInput,
    ProbeOrderKind,
    ProbePlan,
    ProbeSide,
    RolloverConfig,
    RolloverMode,
    RunMode,
    _validate_profile,
    run_eurusd_execution,
)
from ftmoquant.data.dukascopy import SourceBar, _eurusd_instrument, _to_nautilus_bars

START = datetime(2024, 1, 2, tzinfo=UTC)
RULE_CONFIG = Path("config/prop/ftmo_2step_swing_2026-08.yaml").resolve()


def test_paired_bars_execute_buy_at_ask_sell_at_bid_and_spread_once(
    tmp_path: Path,
) -> None:
    root = _catalog_root(tmp_path / "data", _flat_minutes(6))

    buy = _run(root, tmp_path / "buy", side=ProbeSide.BUY)
    sell = _run(root, tmp_path / "sell", side=ProbeSide.SELL)

    buy_fill = buy.result_summary["fills"][0]
    sell_fill = sell.result_summary["fills"][0]
    assert buy_fill["last_px"] == "1.10020"
    assert sell_fill["last_px"] == "1.10000"
    assert Decimal(buy_fill["last_px"]) - Decimal(sell_fill["last_px"]) == Decimal(
        "0.00020"
    )


def test_mismatched_bid_ask_timestamps_are_rejected(tmp_path: Path) -> None:
    bid = _flat_minutes(4)
    ask = [*bid[:2], bid[2] + timedelta(minutes=1), bid[3] + timedelta(minutes=1)]
    root = _catalog_root(tmp_path / "data", bid, ask)

    with pytest.raises(ExecutionValidationError, match="matching timestamps"):
        _run(root, tmp_path / "run")


def test_slippage_probability_boundary_is_one_adverse_tick(tmp_path: Path) -> None:
    root = _catalog_root(tmp_path / "data", _flat_minutes(6))

    no_slip = _run(root, tmp_path / "no-slip", slippage=Decimal(0))
    slipped = _run(root, tmp_path / "slipped", slippage=Decimal(1))

    no_slip_px = Decimal(no_slip.result_summary["fills"][0]["last_px"])
    slipped_px = Decimal(slipped.result_summary["fills"][0]["last_px"])
    assert slipped_px - no_slip_px == Decimal("0.00001")


@pytest.mark.parametrize(("probability", "expected_fills"), [("0", 0), ("1", 1)])
def test_limit_fill_probability_boundaries(
    tmp_path: Path, probability: str, expected_fills: int
) -> None:
    root = _catalog_root(
        tmp_path / "data",
        _flat_minutes(6),
        ask_spread=Decimal(0),
        base_prices=(Decimal("1.10020"), *(Decimal("1.10010"),) * 5),
        bar_range=Decimal("0.00010"),
    )
    plan = ProbePlan(
        order_kind=ProbeOrderKind.LIMIT,
        side=ProbeSide.BUY,
        quantity=Decimal("1"),
        entry_bar_index=0,
        limit_price=Decimal("1.10000"),
    )

    result = _run(
        root,
        tmp_path / f"p-{probability}",
        limit_fill=Decimal(probability),
        plan=plan,
    )

    assert result.result_summary["fill_count"] == expected_fills


def test_static_latency_delays_execution(tmp_path: Path) -> None:
    root = _catalog_root(tmp_path / "data", _flat_minutes(8))

    immediate = _run(root, tmp_path / "immediate")
    delayed = _run(root, tmp_path / "delayed", insert_latency_ns=60_000_000_000)

    delayed_ts = datetime.fromisoformat(delayed.result_summary["fills"][0]["ts_event"])
    immediate_ts = datetime.fromisoformat(
        immediate.result_summary["fills"][0]["ts_event"]
    )
    assert delayed_ts > immediate_ts


def test_fixed_fee_changes_native_account_balance(tmp_path: Path) -> None:
    root = _catalog_root(tmp_path / "data", _flat_minutes(8))
    plan = ProbePlan(
        order_kind=ProbeOrderKind.MARKET,
        side=ProbeSide.BUY,
        quantity=Decimal("1"),
        entry_bar_index=0,
        exit_bar_index=3,
    )

    free = _run(root, tmp_path / "free", plan=plan)
    charged = _run(root, tmp_path / "charged", plan=plan, fixed_fee=Decimal("1"))

    assert Decimal(free.result_summary["ending_balance"]) - Decimal(
        charged.result_summary["ending_balance"]
    ) == Decimal("2.00")


def test_deterministic_seed_and_identical_inputs_reproduce_native_state(
    tmp_path: Path,
) -> None:
    root = _catalog_root(tmp_path / "data", _flat_minutes(8))

    first = _run(root, tmp_path / "first")
    second = _run(root, tmp_path / "second")

    assert first.run_config_id == second.run_config_id
    assert first.profile_sha256 == second.profile_sha256
    assert first.result_summary == second.result_summary
    first_manifest = _manifest(first.manifest_path)
    second_manifest = _manifest(second.manifest_path)
    assert first_manifest["report_hashes"] == second_manifest["report_hashes"]


def test_adaptive_high_low_ordering_is_deterministic(tmp_path: Path) -> None:
    root = _catalog_root(tmp_path / "data", _swing_minutes(6))
    plan = ProbePlan(
        order_kind=ProbeOrderKind.DUAL_LIMIT,
        side=ProbeSide.BUY,
        quantity=Decimal("1"),
        entry_bar_index=0,
        limit_price=Decimal("1.09990"),
        second_limit_price=Decimal("1.10030"),
    )

    first = _run(root, tmp_path / "first", plan=plan, adaptive=True)
    second = _run(root, tmp_path / "second", plan=plan, adaptive=True)

    assert first.result_summary == second.result_summary
    assert first.result_summary["fill_count"] == 2


def test_research_mode_rejects_false_manifest_and_accepts_true_one(
    tmp_path: Path,
) -> None:
    root = _catalog_root(tmp_path / "data", _flat_minutes(6))
    derived = root / "ftmoquant_derived_provenance.json"
    parent_hash = hashlib.sha256(
        (root / "ftmoquant_provenance.json").read_bytes()
    ).hexdigest()
    derived.write_text(
        json.dumps(
            {
                "parent_g0_5_manifest_sha256": parent_hash,
                "derived_bar_integrity_valid": True,
                "research_ready": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ExecutionValidationError, match="research_ready=true"):
        _run(root, tmp_path / "rejected", mode=RunMode.RESEARCH)

    derived.write_text(
        json.dumps(
            {
                "parent_g0_5_manifest_sha256": parent_hash,
                "derived_bar_integrity_valid": True,
                "research_ready": True,
            }
        ),
        encoding="utf-8",
    )
    accepted = _run(root, tmp_path / "accepted", mode=RunMode.RESEARCH)
    assert _manifest(accepted.manifest_path)["g0_6_manifest_sha256"] == hashlib.sha256(
        derived.read_bytes()
    ).hexdigest()


def test_profile_variants_never_mutate_source_catalog(tmp_path: Path) -> None:
    root = _catalog_root(tmp_path / "data", _flat_minutes(6))
    before = _tree_hash(root / "catalog")

    _run(root, tmp_path / "first", adaptive=False)
    _run(root, tmp_path / "second", adaptive=True, slippage=Decimal(1))

    assert _tree_hash(root / "catalog") == before


def test_rollover_module_is_proxy_and_disabled_mode_has_no_rollover_cost(
    tmp_path: Path,
) -> None:
    # Thursday 21:58 UTC crosses the native 17:00 US/Eastern rollover boundary.
    start = datetime(2024, 1, 4, 21, 58, tzinfo=UTC)
    root = _catalog_root(
        tmp_path / "data",
        [start + timedelta(minutes=index) for index in range(7)],
    )
    plan = ProbePlan(
        order_kind=ProbeOrderKind.MARKET,
        side=ProbeSide.BUY,
        quantity=Decimal("100000"),
        entry_bar_index=0,
        exit_bar_index=5,
    )

    disabled = _run(root, tmp_path / "disabled", plan=plan)
    enabled = _run(root, tmp_path / "enabled", plan=plan, rollover=True)

    assert disabled.result_summary["ending_balance"] != enabled.result_summary[
        "ending_balance"
    ]
    manifest = _manifest(enabled.manifest_path)
    assert manifest["calibration_status"] == "proxy"
    assert manifest["execution_profile"]["rollover"]["mode"] == (
        "fx_rollover_interest"
    )


def test_manifest_contains_required_provenance_and_reports(tmp_path: Path) -> None:
    root = _catalog_root(tmp_path / "data", _flat_minutes(6))

    result = _run(root, tmp_path / "run")

    manifest = _manifest(result.manifest_path)
    assert manifest["nautilus_version"] == "2.0.0rc2"
    assert manifest["ftmo_rule_config_version"] == "2026-08"
    assert manifest["g0_5_manifest_sha256"]
    assert manifest["execution_profile_sha256"] == result.profile_sha256
    assert manifest["run_config"]["spread_adjustment"] == "none"
    assert set(manifest["report_hashes"]) == {
        "account",
        "orders",
        "order_fills",
        "fills",
        "positions",
    }
    assert all(path.is_file() for path in result.report_paths.values())
    with pytest.raises(FileExistsError, match="immutable"):
        _run(root, tmp_path / "run")


@pytest.mark.parametrize(
    "evidence",
    [None, "a" * 63, "a" * 65, "g" * 64],
)
def test_broker_calibrated_profile_rejects_missing_or_malformed_evidence(
    evidence: str | None,
) -> None:
    profile = replace(
        _profile(),
        calibration_status=CalibrationStatus.BROKER_CALIBRATED,
        broker_evidence_sha256=evidence,
    )

    with pytest.raises(ExecutionValidationError, match="evidence SHA-256"):
        _validate_profile(profile)


def test_broker_calibrated_profile_accepts_valid_sha256() -> None:
    profile = replace(
        _profile(),
        calibration_status=CalibrationStatus.BROKER_CALIBRATED,
        broker_evidence_sha256="0123456789abcdef" * 4,
    )

    _validate_profile(profile)


@pytest.mark.parametrize(
    "status",
    [CalibrationStatus.UNCALIBRATED, CalibrationStatus.PROXY],
)
def test_non_broker_calibrated_profile_rejects_broker_evidence(
    status: CalibrationStatus,
) -> None:
    profile = replace(
        _profile(),
        calibration_status=status,
        broker_evidence_sha256="0123456789abcdef" * 4,
    )

    with pytest.raises(ExecutionValidationError, match="only valid"):
        _validate_profile(profile)


def _run(
    root: Path,
    output: Path,
    *,
    side: ProbeSide = ProbeSide.BUY,
    slippage: Decimal = Decimal(0),
    limit_fill: Decimal = Decimal(1),
    insert_latency_ns: int = 0,
    fixed_fee: Decimal = Decimal(0),
    adaptive: bool = False,
    rollover: bool = False,
    plan: ProbePlan | None = None,
    mode: RunMode = RunMode.TEST,
):
    bars = ParquetDataCatalog(str(root / "catalog")).query_bars(
        ["EUR/USD.DUKASCOPY-1-MINUTE-ASK-EXTERNAL"]
    )
    rollover_config = RolloverConfig(mode=RolloverMode.DISABLED)
    calibration = CalibrationStatus.UNCALIBRATED
    if rollover:
        rollover_config = RolloverConfig(
            mode=RolloverMode.FX_INTEREST,
            records=(
                InterestRateInput("EA19", "2024-01", Decimal("4")),
                InterestRateInput("USA", "2024-01", Decimal("1")),
            ),
        )
        calibration = CalibrationStatus.PROXY
    profile = replace(
        _profile(),
        fill_on_limit_probability=limit_fill,
        adverse_slippage_probability=slippage,
        insert_latency_ns=insert_latency_ns,
        fee=FeeModelConfig(
            kind=FeeModelKind.FIXED,
            commission=fixed_fee,
            currency="USD",
            charge_commission_once=True,
        ),
        rollover=rollover_config,
        adaptive_high_low_ordering=adaptive,
        calibration_status=calibration,
    )
    default_plan = ProbePlan(
        order_kind=ProbeOrderKind.MARKET,
        side=side,
        quantity=Decimal("1"),
        entry_bar_index=0,
    )
    request = ExecutionRunRequest(
        dataset_root=root,
        output_dir=output,
        prop_rule_config_path=RULE_CONFIG,
        profile=profile,
        account=AccountParameters(
            initial_capital=Decimal("100000"),
            currency="USD",
            leverage=Decimal("30"),
        ),
        probe=default_plan if plan is None else plan,
        requested_start_ns=bars[0].ts_init,
        requested_end_ns=bars[-1].ts_init,
        mode=mode,
    )
    return run_eurusd_execution(request)


def _profile() -> ExecutionProfile:
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


def _catalog_root(
    root: Path,
    bid_timestamps: Sequence[datetime],
    ask_timestamps: Sequence[datetime] | None = None,
    *,
    ask_spread: Decimal = Decimal("0.00020"),
    flat_ohlc: bool = False,
    base_prices: Sequence[Decimal] | None = None,
    bar_range: Decimal = Decimal("0.00030"),
) -> Path:
    ask_timestamps = bid_timestamps if ask_timestamps is None else ask_timestamps
    catalog_path = root / "catalog"
    catalog_path.mkdir(parents=True)
    catalog = ParquetDataCatalog(str(catalog_path))
    catalog.write_instruments([_eurusd_instrument()])
    bid_bases = base_prices or (Decimal("1.10000"),) * len(bid_timestamps)
    ask_bases = base_prices or (Decimal("1.10000"),) * len(ask_timestamps)
    catalog.write_bars(
        _bars(bid_timestamps, "BID", Decimal(0), flat_ohlc, bid_bases, bar_range)
    )
    catalog.write_bars(
        _bars(ask_timestamps, "ASK", ask_spread, flat_ohlc, ask_bases, bar_range)
    )
    manifest = {
        "instrument_id": "EUR/USD.DUKASCOPY",
        "source_granularity": "1-minute",
        "price_sides": ["BID", "ASK"],
        "nautilus_version": "2.0.0rc2",
        "ingestion_version": "g0.5-1",
        "qa": {
            "bid_bar_count": len(bid_timestamps),
            "ask_bar_count": len(ask_timestamps),
        },
    }
    (root / "ftmoquant_provenance.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root


def _bars(
    timestamps: Sequence[datetime],
    side: str,
    spread: Decimal,
    flat_ohlc: bool,
    base_prices: Sequence[Decimal],
    bar_range: Decimal,
) -> list[Bar]:
    high_offset = Decimal(0) if flat_ohlc else bar_range
    low_offset = Decimal(0) if flat_ohlc else -bar_range
    rows = [
        SourceBar(
            timestamp=timestamp,
            open=base + spread,
            high=base + high_offset + spread,
            low=base + low_offset + spread,
            close=base + spread,
            volume=Decimal("1000000"),
        )
        for timestamp, base in zip(timestamps, base_prices, strict=True)
    ]
    return _to_nautilus_bars(rows, side)


def _flat_minutes(count: int) -> list[datetime]:
    return [START + timedelta(minutes=index) for index in range(count)]


def _swing_minutes(count: int) -> list[datetime]:
    return _flat_minutes(count)


def _manifest(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        candidate for candidate in path.rglob("*") if candidate.is_file()
    )
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(item.read_bytes())
    return digest.hexdigest()
