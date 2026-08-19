from __future__ import annotations

import json
from datetime import UTC, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ftmoquant.research.alpha_lab.liquidity_structure_signals import (
    b2f1_sweep_bos_retest_signals,
)
from ftmoquant.research.alpha_lab.pair_specific_validation import (
    B2F1_FAMILY,
    CANDIDATE_A,
    CANDIDATE_B,
    CANDIDATE_C,
    FROZEN_CANDIDATES,
    PREREGISTRATION_VERSION,
    PairSpecificValidationError,
    _canonical_sha256,
    _evaluate_candidate,
    _events_for_candidate,
    build_preregistration,
    load_candidate_data,
    run_validation,
    verify_preregistration,
    write_validation_results,
)
from ftmoquant.research.alpha_lab.validation import (
    AlphaLabValidationError,
    ValidationPreregistrationError,
    _reject_forbidden_path,
)
from ftmoquant.research.alpha_lab.wick_fvg_squeeze_execution import simulate_trades
from ftmoquant.research.alpha_lab.wick_fvg_squeeze_screen import (
    F2_FAMILY,
    F3_FAMILY,
    PairStats,
)
from ftmoquant.research.alpha_lab.wick_fvg_squeeze_signals import (
    f2_fresh_fvg_mitigation_signals,
    f3_volatility_squeeze_breakout_signals,
)
from ftmoquant.research.stage_g import HOLDOUT_START, VALIDATION_START

_F2_F3_SCORECARD = Path(".artifacts/alpha_lab/wick_fvg_squeeze_screen_v1/scorecard.csv")
_B2F1_SCORECARD = Path(
    ".artifacts/alpha_lab/liquidity_structure_screen_v1/scorecard.csv"
)


# ---------------------------------------------------------------------------
# 1. exactly the 3 frozen candidates, with exact frozen parameters.
# ---------------------------------------------------------------------------


def test_exactly_three_frozen_candidates_with_exact_parameters() -> None:
    assert len(FROZEN_CANDIDATES) == 3
    assert CANDIDATE_A.family == F2_FAMILY
    assert CANDIDATE_A.instrument_id == "USD/CAD.OANDA"
    assert CANDIDATE_A.timeframe == "H4"
    assert CANDIDATE_A.parameters == {"backward_search_n": 10}
    assert CANDIDATE_B.family == F3_FAMILY
    assert CANDIDATE_B.instrument_id == "USD/JPY.OANDA"
    assert CANDIDATE_B.timeframe == "H1"
    assert CANDIDATE_B.parameters == {"bb_width_threshold": 0.004}
    assert CANDIDATE_C.family == B2F1_FAMILY
    assert CANDIDATE_C.instrument_id == "USD/CAD.OANDA"
    assert CANDIDATE_C.timeframe == "M30"
    assert CANDIDATE_C.parameters == {"swing_lookback": 40, "rr": 2.0}


def test_candidate_signal_functions_are_the_existing_frozen_implementations() -> None:
    # No local reimplementation exists: the module dispatches straight to
    # the same functions used by the DEVELOPMENT screens.
    from ftmoquant.research.alpha_lab import pair_specific_validation as mod

    assert mod.f2_fresh_fvg_mitigation_signals is f2_fresh_fvg_mitigation_signals
    assert (
        mod.f3_volatility_squeeze_breakout_signals
        is f3_volatility_squeeze_breakout_signals
    )
    assert mod.b2f1_sweep_bos_retest_signals is b2f1_sweep_bos_retest_signals


# ---------------------------------------------------------------------------
# 2. preregistration: generation, self-consistency, tamper/overwrite rejection.
# ---------------------------------------------------------------------------


def test_preregistration_has_no_validation_data_argument() -> None:
    import inspect

    from ftmoquant.research.alpha_lab.pair_specific_validation import (
        build_preregistration,
    )

    signature = inspect.signature(build_preregistration)
    for name in signature.parameters:
        assert "validation" not in name.lower()


def test_build_preregistration_reproduces_the_real_frozen_evidence() -> None:
    document = build_preregistration()
    assert document["preregistration_version"] == PREREGISTRATION_VERSION
    assert document["holdout_accessed"] is False
    assert len(document["candidates"]) == 3
    by_id = {c["candidate_id"]: c for c in document["candidates"]}
    assert by_id["A"]["development_evidence"]["trade_count"] == 83
    assert by_id["A"]["development_evidence"]["net_return"] == pytest.approx(
        0.10413525925370948
    )
    assert by_id["B"]["development_evidence"]["trade_count"] == 72
    assert by_id["C"]["development_evidence"]["trade_count"] == 306


def test_verify_preregistration_accepts_a_fresh_document() -> None:
    document = build_preregistration()
    verify_preregistration(document)


def test_verify_preregistration_rejects_a_tampered_self_hash() -> None:
    document = build_preregistration()
    tampered = dict(document)
    tampered["candidates"] = []
    with pytest.raises(ValidationPreregistrationError):
        verify_preregistration(tampered)


def test_verify_preregistration_rejects_parameter_drift() -> None:
    document = build_preregistration()
    working = dict(document)
    candidates = [dict(c) for c in working["candidates"]]
    candidates[0]["parameters"] = {"backward_search_n": 5}  # drifted from 10
    working["candidates"] = candidates
    working["preregistration_semantic_sha256"] = _canonical_sha256(
        {k: v for k, v in working.items() if k != "preregistration_semantic_sha256"}
    )
    with pytest.raises(ValidationPreregistrationError):
        verify_preregistration(working)


def test_verify_preregistration_rejects_wrong_validation_bounds() -> None:
    document = build_preregistration()
    working = dict(document)
    working["validation_partition"] = {
        "start_utc": "2020-01-01T00:00:00Z",
        "end_exclusive_utc": "2021-01-01T00:00:00Z",
    }
    working["preregistration_semantic_sha256"] = _canonical_sha256(
        {k: v for k, v in working.items() if k != "preregistration_semantic_sha256"}
    )
    with pytest.raises(ValidationPreregistrationError):
        verify_preregistration(working)


def test_verify_preregistration_rejects_lineage_hash_mismatch(tmp_path: Path) -> None:
    # Point verification at a scorecard whose file bytes differ (even if the
    # matching row is identical) -- the pinned scorecard_sha256 in the
    # stored document must no longer match, and evidence must still
    # reproduce identically for it to pass; forging a different-but-same-
    # looking scorecard file must be rejected via the evidence-reproduction
    # check when the row is altered.
    document = build_preregistration()
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other_f2f3 = other_dir / "scorecard.csv"
    lines = _F2_F3_SCORECARD.read_text(encoding="utf-8").splitlines()
    mutated = [
        line.replace('""backward_search_n"": 10', '""backward_search_n"": 999')
        if "f2_h4_n10_v1" in line
        else line
        for line in lines
    ]
    other_f2f3.write_text("\n".join(mutated) + "\n", encoding="utf-8")
    with pytest.raises(PairSpecificValidationError):
        verify_preregistration(
            document,
            f2_f3_scorecard_path=other_f2f3,
            b2f1_scorecard_path=_B2F1_SCORECARD,
        )


def test_preregistration_file_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "prereg.json"
    document = build_preregistration()
    output.write_text(json.dumps(document), encoding="utf-8")
    from ftmoquant.research.alpha_lab.pair_specific_validation import preregister_main

    with pytest.raises(ValidationPreregistrationError):
        preregister_main(["--output", str(output)])


def test_frozen_preregistration_artifact_on_disk_is_self_consistent() -> None:
    real_path = Path(
        "config/validation/oanda_pair_specific_alpha_v1_preregistration.json"
    )
    if not real_path.is_file():
        pytest.skip("preregistration artifact not generated in this checkout")
    document = json.loads(real_path.read_text(encoding="utf-8"))
    verify_preregistration(document)


# ---------------------------------------------------------------------------
# 3. readiness must be VALIDATION, never DEVELOPMENT; forbidden paths.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path,should_raise",
    [
        (Path("/data/oanda_fx_alpha_lab_v1/development_root/USDCAD"), True),
        (Path("/data/oanda_fx_alpha_lab_v1/holdout_root/USDCAD"), True),
        (Path("/data/final_holdout/USDCAD"), True),
        (Path("/data/oanda_fx_alpha_lab_v1/validation_canonical/USDCAD"), False),
    ],
)
def test_development_and_holdout_paths_rejected_validation_accepted(
    path: Path, should_raise: bool
) -> None:
    if should_raise:
        with pytest.raises(AlphaLabValidationError):
            _reject_forbidden_path(path)
    else:
        _reject_forbidden_path(path)


def test_load_candidate_data_rejects_development_readiness_document(
    tmp_path: Path,
) -> None:
    development_shaped = {
        "readiness_version": "oanda-alpha-lab-readiness-1",  # DEVELOPMENT version
        "holdout_accessed": False,
        "holdout_rows_admitted": 0,
        "research_ready": True,
    }
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text(json.dumps(development_shaped), encoding="utf-8")
    with pytest.raises(AlphaLabValidationError):
        load_candidate_data(
            candidate=CANDIDATE_A,
            validation_root=tmp_path / "validation_canonical",
            universe_readiness_path=readiness_path,
        )


def test_load_candidate_data_rejects_a_development_root_path() -> None:
    with pytest.raises(AlphaLabValidationError):
        load_candidate_data(
            candidate=CANDIDATE_A,
            validation_root=Path(
                "/Users/Shared/FTMOQuant-data/oanda_fx_alpha_lab_v1/"
                "development_m1_2019_03_11_2023_04_11"
            ),
            universe_readiness_path=Path("/nonexistent/readiness.json"),
        )


def test_bar_at_or_after_holdout_start_is_rejected() -> None:
    from ftmoquant.research.alpha_lab.pair_specific_validation import (
        _reject_holdout_timestamps,
    )

    bad_index = pd.date_range(HOLDOUT_START, periods=3, freq="1D", tz=UTC)
    with pytest.raises(AlphaLabValidationError):
        _reject_holdout_timestamps(bad_index, context="test")

    good_index = pd.date_range(VALIDATION_START, periods=3, freq="1D", tz=UTC)
    _reject_holdout_timestamps(good_index, context="test")  # must not raise

    boundary_index = pd.DatetimeIndex(
        [VALIDATION_START, HOLDOUT_START - timedelta(seconds=1)]
    )
    _reject_holdout_timestamps(boundary_index, context="test")  # must not raise


# ---------------------------------------------------------------------------
# 4. execution semantics unchanged (BID/ASK, strictly-later entry, no
# interpolation, stop-first) -- exercised through this module's own
# dispatch, reusing wick_fvg_squeeze_execution.simulate_trades directly.
# ---------------------------------------------------------------------------


def _m1_index(periods: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01T00:00:00Z", periods=periods, freq="1min", tz=UTC)


def test_events_for_candidate_dispatches_to_the_correct_family_function() -> None:
    n = 30
    index = pd.date_range("2020-01-01", periods=n, freq="4h", tz=UTC)
    ohlc = pd.DataFrame(
        {
            "open": np.full(n, 1.30),
            "high": np.full(n, 1.301),
            "low": np.full(n, 1.299),
            "close": np.full(n, 1.30),
        },
        index=index,
    )
    events_a = _events_for_candidate(CANDIDATE_A, ohlc)
    events_b = _events_for_candidate(CANDIDATE_B, ohlc)
    events_c = _events_for_candidate(CANDIDATE_C, ohlc)
    assert events_a == f2_fresh_fvg_mitigation_signals(
        ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"], backward_search_n=10
    )
    assert events_b == f3_volatility_squeeze_breakout_signals(
        ohlc["high"], ohlc["low"], ohlc["close"], bb_width_threshold=0.004
    )
    assert events_c == b2f1_sweep_bos_retest_signals(
        ohlc["high"], ohlc["low"], ohlc["close"], swing_lookback=40, rr=2.0
    )


def test_no_forbidden_family_reaches_events_dispatch() -> None:
    from dataclasses import replace

    bogus = replace(CANDIDATE_A, family="not_a_real_family")
    ohlc = pd.DataFrame(
        {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]},
        index=pd.date_range("2020-01-01", periods=1, freq="4h", tz=UTC),
    )
    with pytest.raises(PairSpecificValidationError):
        _events_for_candidate(bogus, ohlc)


def test_execution_uses_the_shared_stop_first_m1_bidask_engine() -> None:
    # Same collision convention proven in the DEVELOPMENT execution tests;
    # exercised here via the exact function this module imports unchanged.
    from ftmoquant.research.alpha_lab.wick_fvg_squeeze_signals import (
        DIRECTION_LONG,
        SignalEvent,
    )

    periods = 6
    index = _m1_index(periods)
    entry_price = 1.1000
    flat = {
        "open": entry_price,
        "high": entry_price,
        "low": entry_price,
        "close": entry_price,
    }
    bid = pd.DataFrame(flat, index=index)
    ask = pd.DataFrame(flat, index=index)
    bid.loc[index[2], "low"] = entry_price - 0.02
    bid.loc[index[2], "high"] = entry_price + 0.02
    event = SignalEvent(
        signal_bar_ts=index[0],
        direction=DIRECTION_LONG,
        stop_distance=0.005,
        target_distance=0.005,
    )
    trades, _ = simulate_trades([event], bid, ask)
    assert len(trades) == 1
    assert trades[0].exit_reason == "stop"
    assert trades[0].entry_ts > index[0]


# ---------------------------------------------------------------------------
# 5. gates A/B independently enforced; no stress gate invented.
# ---------------------------------------------------------------------------


def _stats(*, net_return: float, sharpe: float | None) -> PairStats:
    return PairStats(
        instrument_id="USD/CAD.OANDA",
        trade_count=10,
        skip_count=0,
        net_return=net_return,
        annualized_sharpe=sharpe,
        maximum_drawdown=-0.02,
        win_rate=0.5,
        daily_returns=(),
    )


@pytest.mark.parametrize(
    "net_return,sharpe,expected_a,expected_b,expected_pass",
    [
        (0.05, 0.5, True, True, True),
        (0.0, 0.5, False, True, False),  # strictly greater than zero required
        (-0.01, 0.5, False, True, False),
        (0.05, 0.0, True, False, False),
        (0.05, -0.1, True, False, False),
        (0.05, None, True, False, False),
    ],
)
def test_gates_a_and_b_are_independently_enforced(
    net_return: float,
    sharpe: float | None,
    expected_a: bool,
    expected_b: bool,
    expected_pass: bool,
) -> None:
    result = _evaluate_candidate(
        CANDIDATE_A, _stats(net_return=net_return, sharpe=sharpe)
    )
    assert result.gate_a_positive_return is expected_a
    assert result.gate_b_positive_sharpe is expected_b
    assert result.validation_passed is expected_pass
    assert result.stressed_return is None  # C is not computed


def test_stress_gate_status_is_documented_as_not_available() -> None:
    from ftmoquant.research.alpha_lab.pair_specific_validation import (
        STRESS_GATE_REASON,
        STRESS_GATE_STATUS,
    )

    assert STRESS_GATE_STATUS == "not_available"
    assert (
        "cost model" in STRESS_GATE_REASON.lower()
        or "cost-stress" in STRESS_GATE_REASON.lower()
    )


# ---------------------------------------------------------------------------
# 6. one failed candidate never triggers an alternate/adjacent parameter.
# ---------------------------------------------------------------------------


def test_failed_candidate_never_triggers_an_alternate_parameter() -> None:
    prereg = build_preregistration()
    data_by_id = {
        "A": _synthetic_data("H4", seed=1),
        "B": _synthetic_data("H1", seed=2),
        "C": _synthetic_data("M30", seed=3),
    }
    result = run_validation(preregistration=prereg, data_by_candidate_id=data_by_id)
    strategy_ids = {r.strategy_id for r in result.results}
    # exactly the 3 frozen strategy ids, regardless of whether they pass.
    assert strategy_ids == {
        CANDIDATE_A.strategy_id,
        CANDIDATE_B.strategy_id,
        CANDIDATE_C.strategy_id,
    }


def _synthetic_data(
    timeframe: str, *, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    freq = {"M30": "30min", "H1": "1h", "H4": "4h"}[timeframe]
    periods = 400
    index = pd.date_range(VALIDATION_START, periods=periods, freq=freq, tz=UTC)
    mid = 1.10 + np.cumsum(rng.normal(0, 0.0005, periods))
    ohlc = pd.DataFrame(
        {
            "open": mid,
            "high": mid + 0.001,
            "low": mid - 0.001,
            "close": mid,
        },
        index=index,
    )
    m1_index = pd.date_range(
        VALIDATION_START, periods=periods * 20, freq="1min", tz=UTC
    )
    m1_mid = 1.10 + np.cumsum(rng.normal(0, 0.0001, periods * 20))
    bid = pd.DataFrame(
        {
            "open": m1_mid - 0.0001,
            "high": m1_mid - 0.00005,
            "low": m1_mid - 0.00015,
            "close": m1_mid - 0.0001,
        },
        index=m1_index,
    )
    ask = pd.DataFrame(
        {
            "open": m1_mid + 0.0001,
            "high": m1_mid + 0.00015,
            "low": m1_mid + 0.00005,
            "close": m1_mid + 0.0001,
        },
        index=m1_index,
    )
    return ohlc, bid, ask


# ---------------------------------------------------------------------------
# 7. determinism + output-file contract (exactly 3 rows, refuses overwrite).
# ---------------------------------------------------------------------------


def test_run_validation_is_deterministic() -> None:
    prereg = build_preregistration()
    data_by_id = {
        "A": _synthetic_data("H4", seed=11),
        "B": _synthetic_data("H1", seed=12),
        "C": _synthetic_data("M30", seed=13),
    }
    first = run_validation(preregistration=prereg, data_by_candidate_id=data_by_id)
    second = run_validation(preregistration=prereg, data_by_candidate_id=data_by_id)
    assert first.results == second.results


def test_write_validation_results_produces_exactly_three_rows_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    prereg = build_preregistration()
    data_by_id = {
        "A": _synthetic_data("H4", seed=21),
        "B": _synthetic_data("H1", seed=22),
        "C": _synthetic_data("M30", seed=23),
    }
    result = run_validation(preregistration=prereg, data_by_candidate_id=data_by_id)
    output_dir = tmp_path / "pair_specific_validation_v1"
    write_validation_results(result, prereg, output_dir)

    assert (output_dir / "preregistration.json").is_file()
    assert (output_dir / "validation_scorecard.csv").is_file()
    assert (output_dir / "candidate_validation_summary.csv").is_file()
    assert (output_dir / "metadata.json").is_file()

    import csv

    with (output_dir / "candidate_validation_summary.csv").open(
        encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert {row["candidate_id"] for row in rows} == {"A", "B", "C"}
    required_fields = {
        "candidate_id",
        "family",
        "instrument",
        "timeframe",
        "parameters",
        "trade_count",
        "net_return",
        "annualized_sharpe",
        "maximum_drawdown",
        "win_rate",
        "stressed_return",
        "validation_passed",
    }
    assert required_fields.issubset(rows[0].keys())

    with pytest.raises(FileExistsError):
        write_validation_results(result, prereg, output_dir)

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["holdout_accessed"] is False
    assert metadata["holdout_rows_admitted"] == 0
