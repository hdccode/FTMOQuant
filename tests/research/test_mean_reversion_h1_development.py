from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import vectorbt as vbt
from nautilus_trader.model import Bar, BarType, Price, Quantity

import ftmoquant.research.mean_reversion_h1_development as dev
from ftmoquant.data.instruments import OANDA_ALPHA_LAB_SPECS
from ftmoquant.data.oanda_alpha_lab_validation import (
    VALIDATION_READINESS_VERSION,
    load_oanda_alpha_lab_validation_config,
)
from ftmoquant.research.alpha_lab.families import mean_reversion_signals
from ftmoquant.research.g1.normalization import G1_ANNUAL_VOLATILITY_TARGET
from ftmoquant.research.stage_g import (
    DEVELOPMENT_END_EXCLUSIVE,
    DEVELOPMENT_START,
    HOLDOUT_START,
    VALIDATION_START,
)
from ftmoquant.strategies.mean_reversion_h1 import (
    FROZEN_LOOKBACK,
    FROZEN_UNIVERSE,
    FROZEN_Z_ENTRY,
    causal_signal_stream,
)

_DUMMY_ROOTS: dict[str, Path] = {
    instrument_id: Path(f"/tmp/fake_catalog_root/{i}")
    for i, instrument_id in enumerate(FROZEN_UNIVERSE)
}


# ---------------------------------------------------------------------------
# Partitions: accepted, fail-closed on anything else.
# ---------------------------------------------------------------------------


def test_development_and_validation_partitions_are_accepted() -> None:
    assert dev.parse_partition("development") is dev.Partition.DEVELOPMENT
    assert dev.parse_partition("validation") is dev.Partition.VALIDATION


@pytest.mark.parametrize(
    "value", ["holdout", "final_holdout", "DEVELOPMENT", "", "development ", "prod"]
)
def test_invalid_partitions_fail_closed(value: str) -> None:
    with pytest.raises(dev.MeanReversionH1DevelopmentError):
        dev.parse_partition(value)


def test_partition_bounds_match_the_frozen_stage_g_boundaries() -> None:
    assert dev.partition_bounds(dev.Partition.DEVELOPMENT) == (
        DEVELOPMENT_START,
        DEVELOPMENT_END_EXCLUSIVE,
    )
    assert dev.partition_bounds(dev.Partition.VALIDATION) == (
        VALIDATION_START,
        HOLDOUT_START,
    )


def test_cli_declares_exactly_the_two_frozen_partition_choices() -> None:
    parser = dev.build_parser()
    partition_action = next(
        action for action in parser._actions if action.dest == "partition"
    )
    assert set(partition_action.choices) == {"development", "validation"}


# ---------------------------------------------------------------------------
# No final holdout access, ever -- path guard and boundary invariant.
# ---------------------------------------------------------------------------


def test_no_partition_bound_ever_reaches_the_final_holdout() -> None:
    for partition in dev.Partition:
        _, end_exclusive = dev.partition_bounds(partition)
        assert end_exclusive <= HOLDOUT_START


def test_holdout_labeled_paths_are_always_rejected() -> None:
    for partition in dev.Partition:
        with pytest.raises(dev.MeanReversionH1DevelopmentError, match="holdout"):
            dev._reject_sealed_path(
                Path("/data/oanda/holdout_root/EUR_USD"), partition=partition
            )
        with pytest.raises(dev.MeanReversionH1DevelopmentError):
            dev._reject_sealed_path(
                Path("/data/final_holdout/EUR_USD"), partition=partition
            )


def test_development_run_rejects_a_validation_labeled_path() -> None:
    with pytest.raises(dev.MeanReversionH1DevelopmentError, match="DEVELOPMENT"):
        dev._reject_sealed_path(
            Path("/data/oanda/validation_root/EUR_USD"),
            partition=dev.Partition.DEVELOPMENT,
        )


def test_validation_run_accepts_a_validation_labeled_path() -> None:
    dev._reject_sealed_path(
        Path("/data/oanda/validation_root/EUR_USD"), partition=dev.Partition.VALIDATION
    )  # must not raise


def test_reject_final_holdout_is_wired_into_strategy_module() -> None:
    source = Path(dev.__file__).read_text(encoding="utf-8")
    assert "reject_final_holdout" in source


# ---------------------------------------------------------------------------
# Frozen seven-pair universe enforcement -- fails before any catalog access.
# ---------------------------------------------------------------------------


def test_run_rejects_fewer_than_seven_pairs(tmp_path: Path) -> None:
    incomplete = dict(list(_DUMMY_ROOTS.items())[:-1])
    with pytest.raises(dev.MeanReversionH1DevelopmentError, match="seven"):
        dev.run_mean_reversion_h1_development(
            partition=dev.Partition.DEVELOPMENT,
            catalog_roots=incomplete,
            output_dir=tmp_path / "out",
        )


def test_run_rejects_an_eighth_extra_pair(tmp_path: Path) -> None:
    extra = dict(_DUMMY_ROOTS)
    extra["XAU/USD.OANDA"] = tmp_path / "extra"
    with pytest.raises(dev.MeanReversionH1DevelopmentError, match="seven"):
        dev.run_mean_reversion_h1_development(
            partition=dev.Partition.DEVELOPMENT,
            catalog_roots=extra,
            output_dir=tmp_path / "out",
        )


def test_run_rejects_a_sealed_path_before_touching_any_catalog(tmp_path: Path) -> None:
    sealed = dict(_DUMMY_ROOTS)
    sealed[FROZEN_UNIVERSE[0]] = tmp_path / "holdout_root"
    with pytest.raises(dev.MeanReversionH1DevelopmentError, match="holdout"):
        dev.run_mean_reversion_h1_development(
            partition=dev.Partition.DEVELOPMENT,
            catalog_roots=sealed,
            output_dir=tmp_path / "out",
        )


def test_run_refuses_to_overwrite_an_existing_output_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    with pytest.raises(dev.MeanReversionH1DevelopmentError, match="already exists"):
        dev.run_mean_reversion_h1_development(
            partition=dev.Partition.DEVELOPMENT,
            catalog_roots=_DUMMY_ROOTS,
            output_dir=output_dir,
        )


def test_frozen_universe_is_exactly_seven_pairs_no_exclusions() -> None:
    assert len(FROZEN_UNIVERSE) == 7
    assert "USD/JPY.OANDA" in FROZEN_UNIVERSE


# ---------------------------------------------------------------------------
# Section 5: capital semantics. Each pair must get its own independent
# equal-capital sleeve -- one dedicated engine/account per instrument,
# never one shared margin account competing across all seven pairs --
# matching screening_common._build_aggregate_portfolio's own convention
# (cash_sharing=False) exactly.
# ---------------------------------------------------------------------------


def test_run_calls_the_engine_core_once_per_pair_with_independent_capital(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict] = []

    def fake_run_frozen_signal_backtest(**kwargs):
        calls.append(kwargs)
        (instrument_id,) = kwargs["instruments"]
        initial_capital = Decimal("100000")
        return dev.EngineRunOutcome(
            transition_counts={instrument_id: 0},
            submissions=(),
            fills=(),
            order_report_rows=0,
            fill_report_rows=0,
            position_report_rows=0,
            equity_points={
                instrument_id: (dev.EquityPoint(0, initial_capital),)
            },
            realized_variable_cost={instrument_id: Decimal(0)},
            initial_capital=initial_capital,
        )

    monkeypatch.setattr(
        dev, "run_frozen_signal_backtest", fake_run_frozen_signal_backtest
    )
    monkeypatch.setattr(dev, "_validate_bar_stream", lambda *a, **k: None)

    spec_by_id = {spec.instrument_id: spec for spec in OANDA_ALPHA_LAB_SPECS}

    class _FakeCatalog:
        def __init__(self, instrument_id: str) -> None:
            self._instrument_id = instrument_id

        def instruments(self, ids):
            return [spec_by_id[self._instrument_id].nautilus_instrument()]

        def query_bars(self, bar_types, start, end):
            return ()

    instrument_id_by_catalog_path = {
        str(root / "catalog"): instrument_id
        for instrument_id, root in _DUMMY_ROOTS.items()
    }

    def fake_parquet_catalog(path):
        instrument_id = instrument_id_by_catalog_path[str(path)]
        return _FakeCatalog(instrument_id)

    monkeypatch.setattr(dev, "ParquetDataCatalog", fake_parquet_catalog)

    dev.run_mean_reversion_h1_development(
        partition=dev.Partition.DEVELOPMENT,
        catalog_roots=_DUMMY_ROOTS,
        output_dir=tmp_path / "out",
    )

    assert len(calls) == 7
    for call in calls:
        assert len(call["instruments"]) == 1
        assert len(call["h1_bars_by_instrument"]) == 1
        assert len(call["m1_bars_by_instrument"]) == 1
    called_instrument_ids = {next(iter(call["instruments"])) for call in calls}
    assert called_instrument_ids == set(FROZEN_UNIVERSE)


# ---------------------------------------------------------------------------
# Execution profile remains canonical_execution_profile(), unmodified.
# ---------------------------------------------------------------------------


def test_module_imports_canonical_execution_profile_and_nothing_else() -> None:
    source = Path(dev.__file__).read_text(encoding="utf-8")
    assert "canonical_execution_profile" in source
    # No alternate/ad-hoc ExecutionProfile construction anywhere.
    assert "ExecutionProfile(" not in source


def test_canonical_execution_profile_used_is_the_real_zero_cost_baseline() -> None:
    from ftmoquant.backtest.execution_harness import (
        CalibrationStatus,
        FeeModelKind,
        RolloverMode,
        canonical_execution_profile,
    )

    profile = canonical_execution_profile()
    assert profile.fee.kind is FeeModelKind.FIXED
    assert profile.fee.commission == 0
    assert profile.rollover.mode is RolloverMode.DISABLED
    assert profile.base_latency_ns == 0
    assert profile.adverse_slippage_probability == 0
    assert profile.calibration_status is CalibrationStatus.UNCALIBRATED


def _dummy_pair_performance() -> dict[str, dev.PairPerformance]:
    return {
        pair: dev.PairPerformance(
            instrument_id=pair,
            initial_capital="100000",
            net_return=0.0,
            realized_variable_cost="0",
            cost_stress_1_5x_return=0.0,
            annualized_sharpe=None,
            daily_return_count=0,
        )
        for pair in FROZEN_UNIVERSE
    }


def _dummy_aggregate_performance() -> dev.AggregatePerformance:
    return dev.AggregatePerformance(
        pair_count=7,
        equal_weight_net_return=0.0,
        equal_weight_cost_stress_1_5x_return=0.0,
        profitable_pair_count=0,
        annualized_sharpe=None,
        daily_return_count=0,
        cost_stress_methodology=dev.COST_STRESS_METHODOLOGY_LABEL,
    )


def test_result_records_canonical_execution_profile_label() -> None:
    result = dev.MeanReversionH1RunResult(
        partition="development",
        start_utc="2019-03-11T00:00:00Z",
        end_exclusive_utc="2023-04-11T00:00:00Z",
        instrument_ids=FROZEN_UNIVERSE,
        transition_counts={pair: 0 for pair in FROZEN_UNIVERSE},
        order_report_rows=0,
        fill_report_rows=0,
        position_report_rows=0,
        execution_profile="canonical_execution_profile",
        execution_timing=dev.EXECUTION_TIMING_LABEL,
        risk_normalization="G1VolatilityNormalizer_1pct_annualized_ewma60",
        pair_performance=_dummy_pair_performance(),
        aggregate_performance=_dummy_aggregate_performance(),
    )
    assert result.execution_profile == "canonical_execution_profile"


# ---------------------------------------------------------------------------
# G1 volatility normalization remains unchanged (1% annualized, EWMA-60).
# ---------------------------------------------------------------------------


def test_g1_volatility_target_is_frozen_at_one_percent() -> None:
    assert G1_ANNUAL_VOLATILITY_TARGET == 0.01


def test_executor_uses_the_default_unmodified_normalizer() -> None:
    source = Path(dev.__file__).read_text(encoding="utf-8")
    # The normalizer must be constructed with no override of its target.
    assert "G1VolatilityNormalizer()" in source
    assert "target_annualized_volatility=" not in source


# ---------------------------------------------------------------------------
# Daily-midpoint aggregation and bar-stream validation (pure logic, no
# Nautilus engine required).
# ---------------------------------------------------------------------------


def _fake_bar(ts_event: int, close: float) -> SimpleNamespace:
    return SimpleNamespace(ts_event=ts_event, close=close)


def test_daily_midpoints_take_the_last_paired_bar_of_each_day() -> None:
    day1_start = int(datetime(2023, 4, 11, tzinfo=UTC).timestamp() * 1_000_000_000)
    hour_ns = 3_600_000_000_000
    bid_bars = (
        _fake_bar(day1_start, 1.10),
        _fake_bar(day1_start + hour_ns, 1.11),
        _fake_bar(day1_start + 25 * hour_ns, 1.20),  # next day
    )
    ask_bars = (
        _fake_bar(day1_start, 1.12),
        _fake_bar(day1_start + hour_ns, 1.13),
        _fake_bar(day1_start + 25 * hour_ns, 1.22),
    )
    midpoints = dev._daily_midpoints_from_h1_bars(bid_bars, ask_bars)
    assert len(midpoints) == 2
    assert midpoints[0].midpoint == pytest.approx((1.11 + 1.13) / 2)
    assert midpoints[1].midpoint == pytest.approx((1.20 + 1.22) / 2)
    assert midpoints[0].information_time_ns < midpoints[1].information_time_ns


def test_daily_midpoints_skip_unpaired_bars() -> None:
    ts = 0
    bid_bars = (_fake_bar(ts, 1.10),)
    ask_bars = ()  # no matching ASK bar at all
    assert dev._daily_midpoints_from_h1_bars(bid_bars, ask_bars) == ()


@pytest.mark.parametrize(
    "bars",
    [
        (),
        (
            SimpleNamespace(bar_type="EUR/USD.OANDA-1-HOUR-BID-INTERNAL", ts_event=100),
            SimpleNamespace(bar_type="EUR/USD.OANDA-1-HOUR-BID-INTERNAL", ts_event=100),
        ),
        (SimpleNamespace(bar_type="EUR/USD.OANDA-1-HOUR-BID-INTERNAL", ts_event=5000),),
    ],
    ids=["empty", "non_monotonic", "out_of_range"],
)
def test_validate_bar_stream_rejects_malformed_streams(bars) -> None:
    bar_type = "EUR/USD.OANDA-1-HOUR-BID-INTERNAL"
    with pytest.raises(dev.MeanReversionH1DevelopmentError):
        dev._validate_bar_stream(bars, bar_type, 0, 1000)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Readiness/universe resolution (synthetic readiness documents only).
# ---------------------------------------------------------------------------


def _development_readiness_document() -> dict:
    from ftmoquant.data.oanda_alpha_lab_development import load_oanda_alpha_lab_config

    config = load_oanda_alpha_lab_config()
    return {
        "readiness_version": "oanda-alpha-lab-readiness-1",
        "alpha_lab_config_sha256": config.semantic_sha256,
        "holdout_accessed": False,
        "holdout_rows_admitted": 0,
        "per_instrument_status": {
            instrument_id: "research_ready" for instrument_id in FROZEN_UNIVERSE
        },
    }


def _validation_readiness_document() -> dict:
    config = load_oanda_alpha_lab_validation_config()
    return {
        "readiness_version": VALIDATION_READINESS_VERSION,
        "partition": "VALIDATION",
        "alpha_lab_config_sha256": config.semantic_sha256,
        "holdout_accessed": False,
        "holdout_rows_admitted": 0,
        "per_instrument_status": {
            instrument_id: "research_ready" for instrument_id in FROZEN_UNIVERSE
        },
    }


@pytest.mark.parametrize(
    "partition,document_fn,catalog_dir",
    [
        (dev.Partition.DEVELOPMENT, _development_readiness_document, "canonical"),
        (
            dev.Partition.VALIDATION,
            _validation_readiness_document,
            "validation_canonical",
        ),
    ],
)
def test_resolve_catalog_roots_accepts_matching_readiness(
    tmp_path: Path, partition: dev.Partition, document_fn, catalog_dir: str
) -> None:
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text(json.dumps(document_fn()), encoding="utf-8")
    roots = dev._resolve_catalog_roots(
        partition=partition,
        catalog_root=tmp_path / catalog_dir,
        universe_readiness_path=readiness_path,
    )
    assert set(roots) == set(FROZEN_UNIVERSE)


def test_resolve_catalog_roots_rejects_development_readiness_for_validation(
    tmp_path: Path,
) -> None:
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text(
        json.dumps(_development_readiness_document()), encoding="utf-8"
    )
    with pytest.raises(dev.MeanReversionH1DevelopmentError):
        dev._resolve_catalog_roots(
            partition=dev.Partition.VALIDATION,
            catalog_root=tmp_path / "canonical",
            universe_readiness_path=readiness_path,
        )


def test_resolve_catalog_roots_rejects_validation_readiness_for_development(
    tmp_path: Path,
) -> None:
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text(
        json.dumps(_validation_readiness_document()), encoding="utf-8"
    )
    with pytest.raises(dev.MeanReversionH1DevelopmentError):
        dev._resolve_catalog_roots(
            partition=dev.Partition.DEVELOPMENT,
            catalog_root=tmp_path / "canonical",
            universe_readiness_path=readiness_path,
        )


def test_resolve_catalog_roots_requires_all_seven_research_ready(
    tmp_path: Path,
) -> None:
    document = _development_readiness_document()
    document["per_instrument_status"][FROZEN_UNIVERSE[0]] = "not_ready"
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(dev.MeanReversionH1DevelopmentError, match="seven"):
        dev._resolve_catalog_roots(
            partition=dev.Partition.DEVELOPMENT,
            catalog_root=tmp_path / "canonical",
            universe_readiness_path=readiness_path,
        )


def test_resolve_catalog_roots_rejects_holdout_accessed_true(tmp_path: Path) -> None:
    document = _development_readiness_document()
    document["holdout_accessed"] = True
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(dev.MeanReversionH1DevelopmentError):
        dev._resolve_catalog_roots(
            partition=dev.Partition.DEVELOPMENT,
            catalog_root=tmp_path / "canonical",
            universe_readiness_path=readiness_path,
        )


# ---------------------------------------------------------------------------
# Provenance artifacts.
# ---------------------------------------------------------------------------


def test_provenance_artifacts_are_written_correctly(tmp_path: Path) -> None:
    result = dev.MeanReversionH1RunResult(
        partition="validation",
        start_utc="2023-04-11T00:00:00Z",
        end_exclusive_utc="2024-08-21T00:00:00Z",
        instrument_ids=FROZEN_UNIVERSE,
        transition_counts={pair: 3 for pair in FROZEN_UNIVERSE},
        order_report_rows=21,
        fill_report_rows=21,
        position_report_rows=7,
        execution_profile="canonical_execution_profile",
        execution_timing=dev.EXECUTION_TIMING_LABEL,
        risk_normalization="G1VolatilityNormalizer_1pct_annualized_ewma60",
        pair_performance=_dummy_pair_performance(),
        aggregate_performance=_dummy_aggregate_performance(),
    )
    output_dir = tmp_path / "run_out"
    output_dir.mkdir()
    dev._write_result_artifacts(output_dir, result)

    result_path = output_dir / "mean_reversion_h1_result.json"
    provenance_path = output_dir / "runtime_provenance.json"
    manifest_path = output_dir / "artifact_hashes.json"
    assert result_path.is_file()
    assert provenance_path.is_file()
    assert manifest_path.is_file()

    written_result = json.loads(result_path.read_text(encoding="utf-8"))
    assert written_result["partition"] == "validation"
    assert written_result["instrument_ids"] == list(FROZEN_UNIVERSE)
    assert written_result["execution_profile"] == "canonical_execution_profile"

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["validation_accessed"] is True
    assert provenance["final_holdout_accessed"] is False
    assert provenance["python_module"] == dev.RUN_MODULE_NAME

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    import hashlib

    assert manifest[result_path.name] == hashlib.sha256(
        result_path.read_bytes()
    ).hexdigest()


def test_provenance_marks_development_runs_as_validation_not_accessed(
    tmp_path: Path,
) -> None:
    result = dev.MeanReversionH1RunResult(
        partition="development",
        start_utc="2019-03-11T00:00:00Z",
        end_exclusive_utc="2023-04-11T00:00:00Z",
        instrument_ids=FROZEN_UNIVERSE,
        transition_counts={pair: 0 for pair in FROZEN_UNIVERSE},
        order_report_rows=0,
        fill_report_rows=0,
        position_report_rows=0,
        execution_profile="canonical_execution_profile",
        execution_timing=dev.EXECUTION_TIMING_LABEL,
        risk_normalization="G1VolatilityNormalizer_1pct_annualized_ewma60",
        pair_performance=_dummy_pair_performance(),
        aggregate_performance=_dummy_aggregate_performance(),
    )
    output_dir = tmp_path / "run_out"
    output_dir.mkdir()
    dev._write_result_artifacts(output_dir, result)
    provenance = json.loads(
        (output_dir / "runtime_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["validation_accessed"] is False
    assert provenance["final_holdout_accessed"] is False


def test_provenance_artifacts_are_deterministic(tmp_path: Path) -> None:
    result = dev.MeanReversionH1RunResult(
        partition="development",
        start_utc="2019-03-11T00:00:00Z",
        end_exclusive_utc="2023-04-11T00:00:00Z",
        instrument_ids=FROZEN_UNIVERSE,
        transition_counts={pair: 1 for pair in FROZEN_UNIVERSE},
        order_report_rows=7,
        fill_report_rows=7,
        position_report_rows=7,
        execution_profile="canonical_execution_profile",
        execution_timing=dev.EXECUTION_TIMING_LABEL,
        risk_normalization="G1VolatilityNormalizer_1pct_annualized_ewma60",
        pair_performance=_dummy_pair_performance(),
        aggregate_performance=_dummy_aggregate_performance(),
    )
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    dev._write_result_artifacts(first_dir, result)
    dev._write_result_artifacts(second_dir, result)
    first_bytes = (first_dir / "mean_reversion_h1_result.json").read_bytes()
    second_bytes = (second_dir / "mean_reversion_h1_result.json").read_bytes()
    assert first_bytes == second_bytes


# ---------------------------------------------------------------------------
# Genuine end-to-end synthetic parity: a real Nautilus BacktestEngine run
# (via run_frozen_signal_backtest -- the pure, partition-free core; see its
# docstring for why extracting this does not weaken production partition
# guards) compared against the frozen VectorBT screening implementation and
# the causal signal, on deterministic fabricated H1 BID/ASK bars.
# ---------------------------------------------------------------------------

_PARITY_INSTRUMENT_ID = "EUR/USD.OANDA"


def _parity_price_sequence() -> list[float]:
    """30 days of tiny, non-constant daily variation (so the causal
    volatility estimator has >=20 prior completed daily returns and sizes
    non-zero) followed by the exact reversal pattern already proven in
    tests/strategies/test_mean_reversion_h1.py::test_reversal_across_two_bars:
    flat warm-up -> long entry -> mean-cross exit -> short entry."""

    values = []
    for day in range(30):
        for hour in range(24):
            idx = day * 24 + hour
            values.append(1.10000 + 0.00010 * ((idx % 7) - 3))
    values += [1.10000] * (FROZEN_LOOKBACK - 1) + [1.07000, 1.10000, 1.13000]
    return values


def _build_parity_bars(
    values: list[float], start: datetime
) -> tuple[Bar, ...]:
    """H1 decision bars -- read only offline (never engine-injected), tagged
    INTERNAL exactly as the real canonical catalog stores them."""

    spec = next(
        s for s in OANDA_ALPHA_LAB_SPECS if s.instrument_id == _PARITY_INSTRUMENT_ID
    )
    bid_type = BarType.from_str(f"{_PARITY_INSTRUMENT_ID}-1-HOUR-BID-INTERNAL")
    ask_type = BarType.from_str(f"{_PARITY_INSTRUMENT_ID}-1-HOUR-ASK-INTERNAL")
    volume = Quantity.from_str(f"{1:.{spec.size_precision}f}")
    bars = []
    for i, mid in enumerate(values):
        ts = int((start + timedelta(hours=i)).timestamp() * 1_000_000_000)
        bid_price = Price.from_str(f"{mid - 0.00010:.5f}")
        ask_price = Price.from_str(f"{mid + 0.00010:.5f}")
        bars.append(
            Bar(
                bar_type=bid_type,
                open=bid_price,
                high=bid_price,
                low=bid_price,
                close=bid_price,
                volume=volume,
                ts_event=ts,
                ts_init=ts,
            )
        )
        bars.append(
            Bar(
                bar_type=ask_type,
                open=ask_price,
                high=ask_price,
                low=ask_price,
                close=ask_price,
                volume=volume,
                ts_event=ts,
                ts_init=ts,
            )
        )
    return tuple(bars)


def _build_m1_bars(pairs: list[tuple[int, float]]) -> tuple[Bar, ...]:
    """Genuine paired M1 BID/ASK execution observations at the given
    ``(ts_event_ns, mid)`` points -- the sole engine-injected stream."""

    spec = next(
        s for s in OANDA_ALPHA_LAB_SPECS if s.instrument_id == _PARITY_INSTRUMENT_ID
    )
    bid_type = BarType.from_str(f"{_PARITY_INSTRUMENT_ID}-1-MINUTE-BID-EXTERNAL")
    ask_type = BarType.from_str(f"{_PARITY_INSTRUMENT_ID}-1-MINUTE-ASK-EXTERNAL")
    volume = Quantity.from_str(f"{1:.{spec.size_precision}f}")
    bars = []
    for ts, mid in pairs:
        bid_price = Price.from_str(f"{mid - 0.00010:.5f}")
        ask_price = Price.from_str(f"{mid + 0.00010:.5f}")
        bars.append(
            Bar(
                bar_type=bid_type,
                open=bid_price,
                high=bid_price,
                low=bid_price,
                close=bid_price,
                volume=volume,
                ts_event=ts,
                ts_init=ts,
            )
        )
        bars.append(
            Bar(
                bar_type=ask_type,
                open=ask_price,
                high=ask_price,
                low=ask_price,
                close=ask_price,
                volume=volume,
                ts_event=ts,
                ts_init=ts,
            )
        )
    return tuple(bars)


#: Every H1 decision's strictly-later execution observation sits exactly
#: one minute after the decision -- close enough to keep the fixture small,
#: far enough that decision_ns != execution_ns is unambiguous in assertions.
_M1_EXECUTION_OFFSET = timedelta(minutes=1)


def _build_parity_m1_bars(values: list[float], start: datetime) -> tuple[Bar, ...]:
    pairs = [
        (
            int(
                (start + timedelta(hours=i) + _M1_EXECUTION_OFFSET).timestamp()
                * 1_000_000_000
            ),
            mid,
        )
        for i, mid in enumerate(values)
    ]
    return _build_m1_bars(pairs)


def _split_by_side(bars: tuple[Bar, ...]) -> dict[str, tuple[Bar, ...]]:
    return {
        side: tuple(b for b in bars if b.bar_type.spec.price_type.name == side)
        for side in ("BID", "ASK")
    }


def _run_parity_backtest(values: list[float], start: datetime) -> dev.EngineRunOutcome:
    spec = next(
        s for s in OANDA_ALPHA_LAB_SPECS if s.instrument_id == _PARITY_INSTRUMENT_ID
    )
    instrument = spec.nautilus_instrument()
    h1_by_side = _split_by_side(_build_parity_bars(values, start))
    m1_by_side = _split_by_side(_build_parity_m1_bars(values, start))
    end_exclusive = start + timedelta(hours=len(values))
    return dev.run_frozen_signal_backtest(
        start=start,
        end_exclusive=end_exclusive,
        instruments={_PARITY_INSTRUMENT_ID: instrument},
        h1_bars_by_instrument={_PARITY_INSTRUMENT_ID: h1_by_side},
        m1_bars_by_instrument={_PARITY_INSTRUMENT_ID: m1_by_side},
    )


def _vectorbt_target_sequence(values: list[float]) -> list[int]:
    index = pd.date_range("2023-01-01", periods=len(values), freq="1h", tz="UTC")
    close = pd.DataFrame({"X": values}, index=index)
    entries, exits, short_entries, short_exits = mean_reversion_signals(
        close, FROZEN_LOOKBACK, FROZEN_Z_ENTRY
    )
    portfolio = vbt.Portfolio.from_signals(
        close,
        entries,
        exits,
        short_entries=short_entries,
        short_exits=short_exits,
        fees=0.0,
        freq="1h",
    )
    flow = portfolio.asset_flow(direction="both")["X"].cumsum()
    return [int(np.sign(v)) for v in flow]


def test_synthetic_engine_parity_target_sequence_and_trade_count() -> None:
    """The required synthetic parity evidence: raw target sequence, long
    entry, short entry, exit, reversal, execution ordering, and completed
    trade count, all compared across VectorBT, the causal signal, and a
    real Nautilus BacktestEngine run -- at zero added latency/commission/
    slippage (canonical_execution_profile())."""

    values = _parity_price_sequence()
    start = datetime(2023, 1, 2, tzinfo=UTC)

    outcome = _run_parity_backtest(values, start)
    causal_positions = [
        bar.position
        for bar in causal_signal_stream(
            (
                int((start + timedelta(hours=i)).timestamp() * 1_000_000_000),
                value,
            )
            for i, value in enumerate(values)
        )
    ]
    vectorbt_positions = _vectorbt_target_sequence(values)

    # 1. Exactly three transitions: long entry, mean-cross exit, short entry.
    assert len(outcome.submissions) == 3
    assert [s.side for s in outcome.submissions] == ["BUY", "SELL", "SELL"]
    assert outcome.transition_counts[_PARITY_INSTRUMENT_ID] == 3

    # 2. Decision timestamps match the causal signal's own transition bars
    #    exactly (same information, same decision instant).
    causal_transition_indices = [
        i
        for i in range(1, len(causal_positions))
        if causal_positions[i] != causal_positions[i - 1]
    ]
    assert len(causal_transition_indices) == 3
    expected_decision_ns = [
        int((start + timedelta(hours=i)).timestamp() * 1_000_000_000)
        for i in causal_transition_indices
    ]
    assert [s.decision_ns for s in outcome.submissions] == expected_decision_ns

    # 3. VectorBT's own target sequence transitions at the identical bars
    #    (same decision timing as the causal/Nautilus path).
    vbt_transition_indices = [
        i
        for i in range(1, len(vectorbt_positions))
        if vectorbt_positions[i] != vectorbt_positions[i - 1]
    ]
    assert vbt_transition_indices == causal_transition_indices

    # 4. Strictly-later execution (Section 2, this amendment): the order is
    #    submitted/filled on the M1 execution timestamp, never on the H1
    #    decision bar itself. This explicitly demonstrates the difference
    #    from the superseded same-H1-bar-fill fixture this promotion
    #    previously accepted (fill_ns == decision_ns) -- that assumption is
    #    no longer valid or tested; every fill here is strictly later.
    assert len(outcome.fills) == 3
    start_ns = int(start.timestamp() * 1_000_000_000)
    hour_ns = 3_600_000_000_000
    offset_ns = int(_M1_EXECUTION_OFFSET.total_seconds() * 1_000_000_000)
    for submission, fill in zip(outcome.submissions, outcome.fills, strict=True):
        assert fill.fill_ns > submission.decision_ns
        assert fill.fill_ns == submission.execution_ns
        assert submission.execution_ns == submission.decision_ns + offset_ns
        assert fill.instrument_id == submission.instrument_id

        # 5. Spread side is correct: BUY fills at ASK, SELL fills at BID --
        #    genuine spread crossing, not a fee approximation.
        hour_index = (submission.decision_ns - start_ns) // hour_ns
        mid = values[hour_index]
        expected_price = mid + 0.00010 if submission.side == "BUY" else mid - 0.00010
        assert float(fill.last_px) == pytest.approx(expected_price, abs=1e-9)


def test_synthetic_engine_parity_is_deterministic() -> None:
    values = _parity_price_sequence()
    start = datetime(2023, 1, 2, tzinfo=UTC)
    first = _run_parity_backtest(values, start)
    second = _run_parity_backtest(values, start)
    assert first.submissions == second.submissions
    assert first.fills == second.fills
    assert first.transition_counts == second.transition_counts


def test_full_engine_run_populates_equity_points_and_realized_cost() -> None:
    """End-to-end proof (real Nautilus BacktestEngine, not a fixture-level
    unit test) that the new performance-measurement wiring actually
    produces usable output: a growing daily equity series seeded at the
    sleeve's starting capital, and a non-negative realized spread cost
    exactly on the three trades the parity fixture is known to produce."""

    values = _parity_price_sequence()
    start = datetime(2023, 1, 2, tzinfo=UTC)
    outcome = _run_parity_backtest(values, start)

    assert outcome.initial_capital == Decimal("100000")
    points = outcome.equity_points[_PARITY_INSTRUMENT_ID]
    assert len(points) >= 2
    assert points[0].equity == Decimal("100000")
    assert points[0].information_time_ns == int(start.timestamp() * 1_000_000_000)
    for previous, current in zip(points, points[1:], strict=False):
        assert current.information_time_ns > previous.information_time_ns

    cost = outcome.realized_variable_cost[_PARITY_INSTRUMENT_ID]
    assert cost >= 0
    assert cost > 0  # three real spread-crossing fills occurred

    daily = dev._pair_daily_returns(points, outcome.initial_capital)
    performance = dev._pair_performance(
        _PARITY_INSTRUMENT_ID, points, daily, cost, outcome.initial_capital
    )
    assert performance.net_return == pytest.approx(
        float((points[-1].equity - points[0].equity) / outcome.initial_capital)
    )
    assert performance.cost_stress_1_5x_return < performance.net_return


def test_synthetic_engine_parity_uses_zero_friction_canonical_profile() -> None:
    from ftmoquant.backtest.execution_harness import canonical_execution_profile

    profile = canonical_execution_profile()
    assert profile.base_latency_ns == 0
    assert profile.fee.commission == 0
    assert profile.adverse_slippage_probability == 0


# ---------------------------------------------------------------------------
# Section 2/6: strictly-later execution semantics, tested directly against
# the offline precomputation (no engine required) -- proves the "first
# strictly later" resolution, real-gap skip-forward with no interpolation,
# and the partition-end fail-closed behavior explicitly.
# ---------------------------------------------------------------------------


def _causal_estimator_with_history() -> dev.CausalEwmaDailyVolatility:
    """A volatility estimator with >=20 completed prior daily returns, so
    G1VolatilityNormalizer sizes non-zero exposure -- otherwise every
    decision is flat and no instruction would ever be precomputed."""

    from ftmoquant.research.g1.normalization import (
        CausalEwmaDailyVolatility,
        CompletedDailyMidpoint,
    )

    day_ns = 24 * 3_600_000_000_000
    midpoints = [
        CompletedDailyMidpoint(
            information_time_ns=i * day_ns, midpoint=1.10000 + 0.0002 * ((i % 5) - 2)
        )
        for i in range(1, 31)
    ]
    from ftmoquant.research.g1.normalization import completed_daily_log_returns

    return CausalEwmaDailyVolatility(completed_daily_log_returns(tuple(midpoints)))


#: H1 decision fixtures below must start after the volatility estimator's
#: 30-day history window (see _causal_estimator_with_history) so sizing is
#: non-zero -- otherwise every decision is flat and no instruction is ever
#: precomputed regardless of the execution-timing logic under test.
_AFTER_VOLATILITY_HISTORY_NS = 40 * 24 * 3_600_000_000_000


def _h1_transition_bars(start_ns: int) -> tuple[tuple[Bar, ...], tuple[Bar, ...]]:
    """A minimal H1 stream: flat warm-up, then a long entry -- exactly one
    transition, at the bar immediately after the (FROZEN_LOOKBACK - 1)-bar
    warm-up."""

    hour_ns = 3_600_000_000_000
    values = [1.10000] * (FROZEN_LOOKBACK - 1) + [1.07000]
    bid_type = BarType.from_str(f"{_PARITY_INSTRUMENT_ID}-1-HOUR-BID-INTERNAL")
    ask_type = BarType.from_str(f"{_PARITY_INSTRUMENT_ID}-1-HOUR-ASK-INTERNAL")
    spec = next(
        s for s in OANDA_ALPHA_LAB_SPECS if s.instrument_id == _PARITY_INSTRUMENT_ID
    )
    volume = Quantity.from_str(f"{1:.{spec.size_precision}f}")
    bid_bars = []
    ask_bars = []
    for i, mid in enumerate(values):
        ts = start_ns + i * hour_ns
        bid_price = Price.from_str(f"{mid - 0.00010:.5f}")
        ask_price = Price.from_str(f"{mid + 0.00010:.5f}")
        bid_bars.append(
            Bar(
                bar_type=bid_type,
                open=bid_price,
                high=bid_price,
                low=bid_price,
                close=bid_price,
                volume=volume,
                ts_event=ts,
                ts_init=ts,
            )
        )
        ask_bars.append(
            Bar(
                bar_type=ask_type,
                open=ask_price,
                high=ask_price,
                low=ask_price,
                close=ask_price,
                volume=volume,
                ts_event=ts,
                ts_init=ts,
            )
        )
    return tuple(bid_bars), tuple(ask_bars)


def test_execution_skips_forward_over_a_real_m1_gap_with_no_interpolation() -> None:
    """The M1 stream has a real multi-minute gap right after the decision;
    the resolved execution timestamp must be the first actual observation
    after the gap, never an interpolated/synthesized point in between."""

    hour_ns = 3_600_000_000_000
    start_ns = _AFTER_VOLATILITY_HISTORY_NS
    h1_bid_bars, h1_ask_bars = _h1_transition_bars(start_ns)
    decision_ns = h1_bid_bars[-1].ts_event

    # A real gap: no M1 observation for the first 10 minutes after the
    # decision, then a genuine observation at +11 minutes.
    gap_free_ns = decision_ns + 11 * 60_000_000_000
    m1_by_side = _split_by_side(_build_m1_bars([(gap_free_ns, 1.07000)]))

    instructions = dev._precompute_h1_decisions(
        h1_bid_bars=h1_bid_bars,
        h1_ask_bars=h1_ask_bars,
        m1_bid_bars=m1_by_side["BID"],
        m1_ask_bars=m1_by_side["ASK"],
        start_ns=start_ns,
        end_exclusive_ns=decision_ns + hour_ns,
        estimator=_causal_estimator_with_history(),
    )

    assert len(instructions) == 1
    assert instructions[0].decision_ns == decision_ns
    assert instructions[0].execution_ns == gap_free_ns
    assert instructions[0].execution_ns > instructions[0].decision_ns


def test_partition_end_pending_signal_cannot_execute_into_next_partition() -> None:
    """A decision fires with no later M1 observation before the partition
    boundary -- Section 2 requires this instruction be dropped (fail
    closed), never executed using a next-partition observation."""

    hour_ns = 3_600_000_000_000
    start_ns = _AFTER_VOLATILITY_HISTORY_NS
    h1_bid_bars, h1_ask_bars = _h1_transition_bars(start_ns)
    decision_ns = h1_bid_bars[-1].ts_event
    end_exclusive_ns = decision_ns + hour_ns

    # The only M1 observation strictly after the decision falls exactly at
    # (i.e. on or after) the partition boundary -- must not be used.
    boundary_pairs = [(end_exclusive_ns, 1.07000)]
    boundary_bars = _split_by_side(_build_m1_bars(boundary_pairs))

    instructions = dev._precompute_h1_decisions(
        h1_bid_bars=h1_bid_bars,
        h1_ask_bars=h1_ask_bars,
        m1_bid_bars=boundary_bars["BID"],
        m1_ask_bars=boundary_bars["ASK"],
        start_ns=start_ns,
        end_exclusive_ns=end_exclusive_ns,
        estimator=_causal_estimator_with_history(),
    )

    assert instructions == ()


def test_precompute_rejects_multiple_decisions_mapped_onto_one_frame() -> None:
    """If an M1 gap is so wide it spans more than one H1 decision, both
    decisions would resolve to the identical execution frame -- this must
    fail closed (raise) rather than silently collapse two decisions onto
    one order, matching eurusd_tsm_development.py's own established
    duplicate-execution-frame guard."""

    hour_ns = 3_600_000_000_000
    start_ns = _AFTER_VOLATILITY_HISTORY_NS
    values = [1.10000] * (FROZEN_LOOKBACK - 1) + [1.07000, 1.10000, 1.13000]
    bid_type = BarType.from_str(f"{_PARITY_INSTRUMENT_ID}-1-HOUR-BID-INTERNAL")
    ask_type = BarType.from_str(f"{_PARITY_INSTRUMENT_ID}-1-HOUR-ASK-INTERNAL")
    spec = next(
        s for s in OANDA_ALPHA_LAB_SPECS if s.instrument_id == _PARITY_INSTRUMENT_ID
    )
    volume = Quantity.from_str(f"{1:.{spec.size_precision}f}")
    h1_bid_bars = []
    h1_ask_bars = []
    for i, mid in enumerate(values):
        ts = start_ns + i * hour_ns
        bid_price = Price.from_str(f"{mid - 0.00010:.5f}")
        ask_price = Price.from_str(f"{mid + 0.00010:.5f}")
        h1_bid_bars.append(
            Bar(
                bar_type=bid_type,
                open=bid_price,
                high=bid_price,
                low=bid_price,
                close=bid_price,
                volume=volume,
                ts_event=ts,
                ts_init=ts,
            )
        )
        h1_ask_bars.append(
            Bar(
                bar_type=ask_type,
                open=ask_price,
                high=ask_price,
                low=ask_price,
                close=ask_price,
                volume=volume,
                ts_event=ts,
                ts_init=ts,
            )
        )
    last_decision_ns = h1_bid_bars[-1].ts_event
    end_exclusive_ns = last_decision_ns + hour_ns

    # A single M1 observation strictly after every H1 decision in this
    # fixture -- the entry, the mean-cross exit, and the short reversal all
    # resolve to the same one, which must be rejected rather than silently
    # collapsed.
    shared_execution_ns = last_decision_ns + 60_000_000_000
    shared_bars = _split_by_side(_build_m1_bars([(shared_execution_ns, 1.13000)]))

    with pytest.raises(dev.MeanReversionH1DevelopmentError, match="same M1 execution"):
        dev._precompute_h1_decisions(
            h1_bid_bars=tuple(h1_bid_bars),
            h1_ask_bars=tuple(h1_ask_bars),
            m1_bid_bars=shared_bars["BID"],
            m1_ask_bars=shared_bars["ASK"],
            start_ns=start_ns,
            end_exclusive_ns=end_exclusive_ns,
            estimator=_causal_estimator_with_history(),
        )


def test_first_strictly_later_paired_ns_matches_the_carver_helper_semantics() -> None:
    """Cross-check the production bisect-based implementation against the
    existing, already cross-family-reused
    ``first_strictly_later_execution`` helper (Section 3: prefer direct
    reuse of the established strictly-later pattern) for representative
    sequences, including gaps and boundary cases."""

    from datetime import timedelta as _timedelta

    from ftmoquant.research.carver_trend_carry_ftmo5_development import (
        first_strictly_later_execution,
    )

    # Microsecond-granular throughout (datetime cannot represent finer than
    # a microsecond, so sub-microsecond boundary cases are deliberately not
    # exercised here -- they would only measure datetime's own precision
    # loss, not a real semantic disagreement).
    epoch = datetime(2023, 1, 1, tzinfo=UTC)
    candidate_us = [0, 60_000_000, 600_000_000, 3_600_000_000]
    candidate_ns = [us * 1_000 for us in candidate_us]
    candidate_dt = [epoch + _timedelta(microseconds=us) for us in candidate_us]

    for decision_us in [
        -60_000_000,
        0,
        30_000_000,
        60_000_000,
        599_000_000,
        3_600_000_000,
        3_700_000_000,
    ]:
        decision_ns = decision_us * 1_000
        decision_dt = epoch + _timedelta(microseconds=decision_us)
        expected = first_strictly_later_execution(decision_dt, candidate_dt)
        expected_ns = (
            None if expected is None else int((expected - epoch).total_seconds() * 1e9)
        )
        actual_ns = dev._first_strictly_later_paired_ns(candidate_ns, decision_ns)
        assert actual_ns == expected_ns


# ---------------------------------------------------------------------------
# Performance measurement (new): per-pair return extraction, 7-sleeve
# equal-weight aggregation, profitable-pair count, aggregate daily-return
# Sharpe, zero/empty edge cases, and the 1.5x cost-stress mapping's
# equivalence to the existing repo precedent -- measurement only, no
# promotion verdict, no change to execution timing semantics.
# ---------------------------------------------------------------------------

_DAY_NS = 24 * 3_600_000_000_000


def test_pair_daily_returns_from_equity_points() -> None:
    initial = Decimal("100000")
    points = (
        dev.EquityPoint(0, initial),
        dev.EquityPoint(_DAY_NS, initial + Decimal("1000")),
        dev.EquityPoint(2 * _DAY_NS, initial + Decimal("500")),
    )
    daily = dev._pair_daily_returns(points, initial)
    assert len(daily) == 2
    assert daily[0][1] == pytest.approx(0.01)
    assert daily[1][1] == pytest.approx(-0.005)
    assert daily[0][0] < daily[1][0]


def test_pair_daily_returns_rejects_non_positive_capital() -> None:
    points = (
        dev.EquityPoint(0, Decimal("0")),
        dev.EquityPoint(_DAY_NS, Decimal("100")),
    )
    with pytest.raises(dev.MeanReversionH1DevelopmentError):
        dev._pair_daily_returns(points, Decimal("0"))


def test_pair_performance_net_return_and_stress_return() -> None:
    initial = Decimal("100000")
    points = (
        dev.EquityPoint(0, initial),
        dev.EquityPoint(_DAY_NS, initial + Decimal("500")),
    )
    daily = dev._pair_daily_returns(points, initial)
    realized_cost = Decimal("50")
    perf = dev._pair_performance(
        "EUR/USD.OANDA", points, daily, realized_cost, initial
    )
    assert perf.instrument_id == "EUR/USD.OANDA"
    assert perf.net_return == pytest.approx(0.005)
    assert perf.realized_variable_cost == "50"
    expected_stress = 0.005 - 0.5 * float(realized_cost / initial)
    assert perf.cost_stress_1_5x_return == pytest.approx(expected_stress)
    assert perf.daily_return_count == 1


def test_pair_performance_requires_at_least_the_seed_equity_point() -> None:
    with pytest.raises(dev.MeanReversionH1DevelopmentError):
        dev._pair_performance("EUR/USD.OANDA", (), (), Decimal(0), Decimal("100000"))


def test_pair_performance_zero_activity_gives_zero_return_and_no_sharpe() -> None:
    """Zero/empty edge case: a pair with only the seed equity point (no
    trading activity, no day transitions observed) must report zero net
    return and an unevaluated (None) Sharpe, never a crash or a fabricated
    number."""

    initial = Decimal("100000")
    points = (dev.EquityPoint(0, initial),)
    daily = dev._pair_daily_returns(points, initial)
    perf = dev._pair_performance("EUR/USD.OANDA", points, daily, Decimal(0), initial)
    assert perf.net_return == 0.0
    assert perf.cost_stress_1_5x_return == 0.0
    assert perf.annualized_sharpe is None
    assert perf.daily_return_count == 0


def _pair_performance_fixture(
    net_returns: dict[str, float],
) -> dict[str, dev.PairPerformance]:
    return {
        instrument_id: dev.PairPerformance(
            instrument_id=instrument_id,
            initial_capital="100000",
            net_return=net_return,
            realized_variable_cost="0",
            cost_stress_1_5x_return=net_return,
            annualized_sharpe=None,
            daily_return_count=0,
        )
        for instrument_id, net_return in net_returns.items()
    }


def test_aggregate_equal_weight_net_return_is_the_simple_average() -> None:
    net_returns = {pair: 0.01 * (i + 1) for i, pair in enumerate(FROZEN_UNIVERSE)}
    performance = _pair_performance_fixture(net_returns)
    daily_returns_by_pair = {pair: () for pair in FROZEN_UNIVERSE}
    aggregate = dev._aggregate_performance(performance, daily_returns_by_pair)
    assert aggregate.pair_count == 7
    assert aggregate.equal_weight_net_return == pytest.approx(
        sum(net_returns.values()) / 7
    )
    assert aggregate.equal_weight_cost_stress_1_5x_return == pytest.approx(
        aggregate.equal_weight_net_return
    )
    assert aggregate.cost_stress_methodology == dev.COST_STRESS_METHODOLOGY_LABEL


def test_aggregate_profitable_pair_count() -> None:
    net_returns = dict(
        zip(FROZEN_UNIVERSE, [0.02, -0.01, 0.03, 0.0, -0.05, 0.01, -0.02], strict=True)
    )
    performance = _pair_performance_fixture(net_returns)
    daily_returns_by_pair = {pair: () for pair in FROZEN_UNIVERSE}
    aggregate = dev._aggregate_performance(performance, daily_returns_by_pair)
    # Strictly positive only: 0.02, 0.03, 0.01 -> 3 (0.0 does not count).
    assert aggregate.profitable_pair_count == 3


def test_aggregate_performance_rejects_mismatched_pair_sets() -> None:
    performance = _pair_performance_fixture({pair: 0.0 for pair in FROZEN_UNIVERSE})
    daily_returns_by_pair = {pair: () for pair in FROZEN_UNIVERSE[:-1]}
    with pytest.raises(dev.MeanReversionH1DevelopmentError):
        dev._aggregate_performance(performance, daily_returns_by_pair)


def test_aggregate_performance_rejects_zero_pairs() -> None:
    with pytest.raises(dev.MeanReversionH1DevelopmentError):
        dev._aggregate_performance({}, {})


def test_aggregate_daily_returns_causal_equal_weight_union_of_days() -> None:
    """Days a pair has no mark contribute 0.0 for that pair (no observed
    sleeve movement that day), and the aggregate is still divided by the
    full pair count -- proving the "sum of equal-capital dollar P&L / sum
    of equal capital == simple average" identity behind
    screening_common._build_aggregate_portfolio's own convention."""

    day0 = datetime(2023, 1, 1, tzinfo=UTC).date()
    day1 = datetime(2023, 1, 2, tzinfo=UTC).date()
    daily_returns_by_pair = {
        "EUR/USD.OANDA": ((day0, 0.02), (day1, 0.01)),
        "GBP/USD.OANDA": ((day0, -0.02),),  # no day1 mark -> 0.0 that day
    }
    aggregate = dev._aggregate_daily_returns(daily_returns_by_pair)
    assert aggregate == pytest.approx((0.0, 0.005))


def test_aggregate_daily_sharpe_matches_annualized_sharpe_of_the_union_series() -> None:
    pairs = FROZEN_UNIVERSE
    day0 = datetime(2023, 1, 1, tzinfo=UTC).date()
    day1 = datetime(2023, 1, 2, tzinfo=UTC).date()
    day2 = datetime(2023, 1, 3, tzinfo=UTC).date()
    per_pair_series = {
        pair: (
            (day0, 0.001 * (i + 1)),
            (day1, -0.0005 * (i + 1)),
            (day2, 0.0007 * (i + 1)),
        )
        for i, pair in enumerate(pairs)
    }
    performance = _pair_performance_fixture({pair: 0.0 for pair in pairs})
    aggregate = dev._aggregate_performance(performance, per_pair_series)

    union_series = dev._aggregate_daily_returns(per_pair_series)
    assert aggregate.annualized_sharpe == dev._annualized_sharpe(union_series)
    assert aggregate.daily_return_count == len(union_series) == 3


def test_aggregate_sharpe_is_none_when_fewer_than_two_daily_observations() -> None:
    """Zero/empty edge case at the aggregate level: with at most one
    unioned day across all seven pairs, Sharpe is unevaluated (None), not
    fabricated."""

    performance = _pair_performance_fixture({pair: 0.0 for pair in FROZEN_UNIVERSE})
    daily_returns_by_pair = {pair: () for pair in FROZEN_UNIVERSE}
    aggregate = dev._aggregate_performance(performance, daily_returns_by_pair)
    assert aggregate.annualized_sharpe is None
    assert aggregate.daily_return_count == 0
    assert aggregate.profitable_pair_count == 0
    assert aggregate.equal_weight_net_return == 0.0


# ---------------------------------------------------------------------------
# 1.5x cost-stress mapping: faithful reuse of the existing, already-
# established (not invented here) native-spread-crossing decomposition,
# cross-checked against its source precedent and tied numerically back to
# screening_common.STRESSED_COST_MULTIPLIER.
# ---------------------------------------------------------------------------


def test_cost_stress_half_coefficient_derives_from_screening_1_5x_multiplier() -> None:
    """The screening-stage 1.5x stress means TOTAL cost is scaled by 1.5x --
    i.e. an ADDITIONAL 0.5x of the base realized cost is subtracted on top
    of the 1.0x already embedded in net_return. This ties our reused
    formula's "0.5" coefficient directly back to
    screening_common.STRESSED_COST_MULTIPLIER=1.5, rather than being an
    independently invented number."""

    from ftmoquant.research.alpha_lab.screening_common import (
        STRESSED_COST_MULTIPLIER,
    )

    assert STRESSED_COST_MULTIPLIER == 1.5
    extra_stress_coefficient = STRESSED_COST_MULTIPLIER - 1.0

    initial = Decimal("100000")
    points = (
        dev.EquityPoint(0, initial),
        dev.EquityPoint(_DAY_NS, initial + Decimal("1000")),
    )
    realized_cost = Decimal("200")
    daily = dev._pair_daily_returns(points, initial)
    perf = dev._pair_performance("EUR/USD.OANDA", points, daily, realized_cost, initial)

    net_return = float(Decimal("1000") / initial)
    cost_return = float(realized_cost / initial)
    expected = net_return - extra_stress_coefficient * cost_return
    assert perf.cost_stress_1_5x_return == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Execution-timing semantics are unchanged by this performance-measurement
# addition: the new execution_midpoint field on _ScaledInstruction is
# audit-only and must never influence WHEN an order executes.
# ---------------------------------------------------------------------------


def test_execution_midpoint_is_audit_only_and_does_not_alter_timing() -> None:
    start_ns = _AFTER_VOLATILITY_HISTORY_NS
    hour_ns = 3_600_000_000_000
    h1_bid_bars, h1_ask_bars = _h1_transition_bars(start_ns)
    decision_ns = h1_bid_bars[-1].ts_event
    execution_ns = decision_ns + 60_000_000_000
    m1_by_side = _split_by_side(_build_m1_bars([(execution_ns, 1.07000)]))

    instructions = dev._precompute_h1_decisions(
        h1_bid_bars=h1_bid_bars,
        h1_ask_bars=h1_ask_bars,
        m1_bid_bars=m1_by_side["BID"],
        m1_ask_bars=m1_by_side["ASK"],
        start_ns=start_ns,
        end_exclusive_ns=decision_ns + hour_ns,
        estimator=_causal_estimator_with_history(),
    )

    assert len(instructions) == 1
    instruction = instructions[0]
    # Timing semantics (Section 2 of the prior amendment) are exactly as
    # before: strictly later, resolved to the genuine M1 observation.
    assert instruction.execution_ns == execution_ns
    assert instruction.execution_ns > instruction.decision_ns

    # The new field is a plain mid price of that same M1 observation --
    # audit-only, does not participate in the timing decision above. The
    # M1 fixture prices round to 5dp exactly as _build_m1_bars constructs
    # them ("1.06990"/"1.07010"), so the expected mid is exact Decimal math.
    expected_mid = (Decimal("1.06990") + Decimal("1.07010")) / 2
    assert instruction.execution_midpoint == expected_mid


# ---------------------------------------------------------------------------
# Currency-denomination audit: realized_variable_cost and net_return must
# be expressed in the USD account/reporting currency for every frozen
# pair -- not silently left in a pair's own quote currency (JPY for
# USD/JPY, CAD for USD/CAD, CHF for USD/CHF). These tests are written to
# FAIL if a 100 JPY (or CAD, or CHF) amount were ever treated as 100 USD.
# ---------------------------------------------------------------------------


def test_convert_to_account_currency_is_a_no_op_when_already_usd() -> None:
    assert dev._convert_to_account_currency(
        Decimal("123.45"),
        "USD",
        base_currency="EUR",
        quote_currency="USD",
        conversion_price=Decimal("1.10000"),
    ) == Decimal("123.45")


@pytest.mark.parametrize(
    "quote_currency,conversion_price",
    [
        ("JPY", Decimal("150.000")),
        ("CAD", Decimal("1.35000")),
        ("CHF", Decimal("0.88000")),
    ],
)
def test_convert_to_account_currency_divides_quote_amount_when_base_is_usd(
    quote_currency: str, conversion_price: Decimal
) -> None:
    """The literal "100 JPY/CAD/CHF treated as 100 USD" failure mode: 100
    units of the quote currency must convert to 100/price USD, not stay
    100."""

    converted = dev._convert_to_account_currency(
        Decimal("100"),
        quote_currency,
        base_currency="USD",
        quote_currency=quote_currency,
        conversion_price=conversion_price,
    )
    assert converted == Decimal("100") / conversion_price
    assert converted != Decimal("100")


def test_convert_to_account_currency_multiplies_base_amount_when_quote_is_usd() -> None:
    """The mirror direction: an amount in the pair's BASE currency (e.g. a
    commission denominated in EUR on a EUR/USD trade) converts to USD by
    multiplying by the quote/base price."""

    converted = dev._convert_to_account_currency(
        Decimal("10"),
        "EUR",
        base_currency="EUR",
        quote_currency="USD",
        conversion_price=Decimal("1.10000"),
    )
    assert converted == Decimal("10") * Decimal("1.10000")


def test_convert_to_account_currency_rejects_unsupported_currency_combination() -> None:
    """Fails closed rather than guessing when the amount's currency is
    neither leg of the pair -- e.g. a GBP amount on a EUR/USD trade, which
    would need a cross rate this repo's frozen universe never requires."""

    with pytest.raises(dev.MeanReversionH1DevelopmentError):
        dev._convert_to_account_currency(
            Decimal("10"),
            "GBP",
            base_currency="EUR",
            quote_currency="USD",
            conversion_price=Decimal("1.10000"),
        )


def _instrument_h1_bars(
    instrument_id: str, price_precision: int, values: list[Decimal], start_ns: int
) -> tuple[tuple[Bar, ...], tuple[Bar, ...]]:
    hour_ns = 3_600_000_000_000
    bid_type = BarType.from_str(f"{instrument_id}-1-HOUR-BID-INTERNAL")
    ask_type = BarType.from_str(f"{instrument_id}-1-HOUR-ASK-INTERNAL")
    spec = next(s for s in OANDA_ALPHA_LAB_SPECS if s.instrument_id == instrument_id)
    volume = Quantity.from_str(f"{1:.{spec.size_precision}f}")
    half_spread = Decimal(2) * Decimal(10) ** (-price_precision)
    bid_bars = []
    ask_bars = []
    for i, mid in enumerate(values):
        ts = start_ns + i * hour_ns
        bid_price = Price.from_str(f"{mid - half_spread:.{price_precision}f}")
        ask_price = Price.from_str(f"{mid + half_spread:.{price_precision}f}")
        bid_bars.append(
            Bar(
                bar_type=bid_type,
                open=bid_price,
                high=bid_price,
                low=bid_price,
                close=bid_price,
                volume=volume,
                ts_event=ts,
                ts_init=ts,
            )
        )
        ask_bars.append(
            Bar(
                bar_type=ask_type,
                open=ask_price,
                high=ask_price,
                low=ask_price,
                close=ask_price,
                volume=volume,
                ts_event=ts,
                ts_init=ts,
            )
        )
    return tuple(bid_bars), tuple(ask_bars)


def _instrument_m1_bars(
    instrument_id: str, price_precision: int, pairs: list[tuple[int, Decimal]]
) -> tuple[Bar, ...]:
    bid_type = BarType.from_str(f"{instrument_id}-1-MINUTE-BID-EXTERNAL")
    ask_type = BarType.from_str(f"{instrument_id}-1-MINUTE-ASK-EXTERNAL")
    spec = next(s for s in OANDA_ALPHA_LAB_SPECS if s.instrument_id == instrument_id)
    volume = Quantity.from_str(f"{1:.{spec.size_precision}f}")
    half_spread = Decimal(2) * Decimal(10) ** (-price_precision)
    bars = []
    for ts, mid in pairs:
        bid_price = Price.from_str(f"{mid - half_spread:.{price_precision}f}")
        ask_price = Price.from_str(f"{mid + half_spread:.{price_precision}f}")
        bars.append(
            Bar(
                bar_type=bid_type,
                open=bid_price,
                high=bid_price,
                low=bid_price,
                close=bid_price,
                volume=volume,
                ts_event=ts,
                ts_init=ts,
            )
        )
        bars.append(
            Bar(
                bar_type=ask_type,
                open=ask_price,
                high=ask_price,
                low=ask_price,
                close=ask_price,
                volume=volume,
                ts_event=ts,
                ts_init=ts,
            )
        )
    return tuple(bars)


#: Same relative drop _h1_transition_bars uses (1.07000 / 1.10000), applied
#: proportionally so the causal z-score path -- and therefore the resulting
#: long-entry transition -- is IDENTICAL regardless of a pair's own price
#: level (z-scores are scale-invariant under uniform multiplication).
_TRANSITION_RATIO = Decimal("1.07000") / Decimal("1.10000")

_CURRENCY_AUDIT_CASES = [
    ("EUR/USD.OANDA", 5, Decimal("1.10000"), "USD"),
    ("USD/JPY.OANDA", 3, Decimal("150.000"), "JPY"),
    ("USD/CAD.OANDA", 5, Decimal("1.35000"), "CAD"),
    ("USD/CHF.OANDA", 5, Decimal("0.88000"), "CHF"),
]

_CURRENCY_PROBE_START = datetime(2023, 2, 20, tzinfo=UTC)

#: Index of the first (long-entry) transition bar within
#: _parity_price_sequence()'s own 30-day-warm-up + 39-flat-bar + transition
#: layout (720 + 39 == 759, zero-indexed): the same bar
#: _h1_transition_bars uses, just not yet followed by the mean-cross-exit/
#: short-entry bars that come after it in the full parity fixture.
_FIRST_TRANSITION_INDEX = 30 * 24 + (FROZEN_LOOKBACK - 1)


def _scaled_price_sequence(base_price: Decimal) -> list[Decimal]:
    """_parity_price_sequence()'s own proven fixture (30 days of daily
    variation -- >=20 prior completed daily returns so the causal
    volatility estimator sizes non-zero -- then a flat lookback window and
    a one-bar entry drop), rescaled to an arbitrary price level. Both the
    causal z-score (scale-invariant under uniform multiplication) and the
    EWMA daily log-return volatility estimate (scale-invariant: log(a*x2 /
    (a*x1)) == log(x2/x1)) are unaffected by this rescaling, so the exact
    same signal transition and comparable sizing occur at any price level
    -- this is why USD/JPY (~150), USD/CAD (~1.35), and EUR/USD (~1.10)
    can all be driven by the identical relative fixture."""

    scale = base_price / Decimal("1.10000")
    return [
        Decimal(str(v)) * scale
        for v in _parity_price_sequence()[: _FIRST_TRANSITION_INDEX + 1]
    ]


def _rounded_execution_mid(mid: Decimal, price_precision: int) -> Decimal:
    """The exact execution midpoint _precompute_h1_decisions will attach to
    an instruction, given how _instrument_m1_bars rounds BID/ASK to
    price_precision decimal places -- ((bid+ask)/2) of the ROUNDED prices,
    not the unrounded fixture value, matching production bar construction
    exactly so expected-value assertions aren't polluted by rounding
    noise unrelated to the currency-conversion behavior under test."""

    half_spread = Decimal(2) * Decimal(10) ** (-price_precision)
    bid = Decimal(f"{mid - half_spread:.{price_precision}f}")
    ask = Decimal(f"{mid + half_spread:.{price_precision}f}")
    return (bid + ask) / 2


def _run_currency_probe(
    instrument_id: str, price_precision: int, base_price: Decimal
) -> dev.EngineRunOutcome:
    """The proven 30-day-warm-up + one long-entry-transition fixture,
    rescaled to this pair's own price level, executed through a real
    Nautilus BacktestEngine end to end."""

    hour_ns = 3_600_000_000_000
    start_ns = int(_CURRENCY_PROBE_START.timestamp() * 1_000_000_000)
    values = _scaled_price_sequence(base_price)
    h1_bid_bars, h1_ask_bars = _instrument_h1_bars(
        instrument_id, price_precision, values, start_ns
    )
    decision_ns = h1_bid_bars[-1].ts_event
    execution_mid = values[-1]
    execution_ns = decision_ns + 60_000_000_000
    m1_by_side = _split_by_side(
        _instrument_m1_bars(
            instrument_id, price_precision, [(execution_ns, execution_mid)]
        )
    )

    spec = next(s for s in OANDA_ALPHA_LAB_SPECS if s.instrument_id == instrument_id)
    instrument = spec.nautilus_instrument()
    end_exclusive = _CURRENCY_PROBE_START + timedelta(
        seconds=(decision_ns + hour_ns - start_ns) / 1_000_000_000
    )

    return dev.run_frozen_signal_backtest(
        start=_CURRENCY_PROBE_START,
        end_exclusive=end_exclusive,
        instruments={instrument_id: instrument},
        h1_bars_by_instrument={instrument_id: {"BID": h1_bid_bars, "ASK": h1_ask_bars}},
        m1_bars_by_instrument={instrument_id: m1_by_side},
    )


@pytest.mark.parametrize(
    "instrument_id,price_precision,base_price,quote_currency", _CURRENCY_AUDIT_CASES
)
def test_realized_cost_is_usd_denominated_for_every_frozen_leg_convention(
    instrument_id: str,
    price_precision: int,
    base_price: Decimal,
    quote_currency: str,
) -> None:
    """A single long-entry fill's realized spread cost, run through a real
    Nautilus BacktestEngine, must equal the same-magnitude USD figure the
    executor's own currency conversion produces -- proven independently
    here via _convert_to_account_currency on the fill's own recorded
    quantity/spread, not merely re-asserting whatever the executor
    happened to compute. For USD/JPY specifically this is the literal
    reproduction of the reported bug: without conversion the recorded
    "cost" would be ~150x too large (JPY quote amount treated as USD)."""

    outcome = _run_currency_probe(instrument_id, price_precision, base_price)
    execution_mid = _rounded_execution_mid(
        _scaled_price_sequence(base_price)[-1], price_precision
    )
    spec = next(s for s in OANDA_ALPHA_LAB_SPECS if s.instrument_id == instrument_id)
    instrument = spec.nautilus_instrument()

    assert len(outcome.submissions) == 1
    submission = outcome.submissions[0]
    assert submission.side == "BUY"  # matches _h1_transition_bars' long entry
    quantity = Decimal(submission.target_units)

    half_spread = Decimal(2) * Decimal(10) ** (-price_precision)
    # BUY fills at ASK = execution_mid + half_spread; the realized spread
    # cost relative to the execution midpoint is quantity * half_spread,
    # in the pair's own QUOTE currency (see on_order_filled).
    expected_cost_quote = quantity * half_spread
    expected_cost_usd = dev._convert_to_account_currency(
        expected_cost_quote,
        quote_currency,
        base_currency=instrument.base_currency.code,
        quote_currency=instrument.quote_currency.code,
        conversion_price=execution_mid,
    )

    actual_cost = outcome.realized_variable_cost[instrument_id]
    assert float(actual_cost) == pytest.approx(float(expected_cost_usd), rel=1e-6)

    # The literal "100 JPY/CAD/CHF treated as 100 USD" failure mode: if the
    # bug were reintroduced, actual_cost would equal expected_cost_quote
    # (the raw, unconverted quote-currency figure) instead.
    if quote_currency != "USD":
        assert float(actual_cost) != pytest.approx(float(expected_cost_quote), rel=1e-6)

    # Sanity bound: a ~2-pip-equivalent spread on a G1-normalized ~1%
    # annualized-vol position must cost a handful of USD, never hundreds.
    assert 0 <= actual_cost < Decimal("50")

    if instrument_id == "USD/JPY.OANDA":
        # Directly reproduces the originally reported defect: the
        # unconverted (buggy) cost figure is on the order of the USD/JPY
        # price level (~150x) larger than the correctly converted figure --
        # exactly matching the originally observed 82876.78 vs. an
        # expected sub-$100 result.
        ratio = expected_cost_quote / actual_cost
        assert float(ratio) == pytest.approx(float(execution_mid), rel=1e-6)
        assert ratio > 100  # ~150 for USD/JPY -- the originally observed scale


@pytest.mark.parametrize(
    "instrument_id,price_precision,base_price,quote_currency", _CURRENCY_AUDIT_CASES
)
def test_pair_net_return_is_usd_denominated_for_every_frozen_leg_convention(
    instrument_id: str,
    price_precision: int,
    base_price: Decimal,
    quote_currency: str,
) -> None:
    """End-to-end: net_return computed from the resulting equity series
    must be a small, sane fraction (position sized to ~1% annualized
    vol against a $100k sleeve) for every pair, including USD/JPY --
    never inflated by a stray ~100x quote-currency conversion error."""

    outcome = _run_currency_probe(instrument_id, price_precision, base_price)

    points = outcome.equity_points[instrument_id]
    net_return = float((points[-1].equity - points[0].equity) / outcome.initial_capital)
    # A single ~2.7% one-bar mean-reversion entry sized to ~1% annualized
    # vol must move a $100k sleeve by a small fraction, not tens/hundreds
    # of percent (the magnitude a ~100x-mispriced JPY/CAD/CHF PnL would
    # produce if summed unconverted into equity).
    assert abs(net_return) < 0.05
