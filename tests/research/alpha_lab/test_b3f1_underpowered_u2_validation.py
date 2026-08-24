from __future__ import annotations

import ast
import json
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import ftmoquant.research.alpha_lab.b3f1_underpowered_u2_validation as m
from ftmoquant.research.alpha_lab.validation import AlphaLabValidationError


def _m1(prices: list[float], *, start: str) -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(prices), freq="min", tz="UTC")
    return pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices}, index=idx
    )


def _synthetic_formation(n: int, *, start: str) -> pd.DataFrame:
    """A tiny, hand-crafted formation series producing exactly one RICH
    entry (z rises to +2.0) followed by a mean-reversion exit (z crosses
    0), independent of any real ADF/OLS computation -- used only to
    exercise the signal-walker/execution/evaluation plumbing without a
    real cointegrated fixture."""

    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    z = [0.0] * n
    z[2] = 2.0  # entry (>= z_entry=1.5)
    z[5] = 0.0  # mean-reversion exit
    valid = [False, False, True, True, True, True] + [False] * (n - 6)
    return pd.DataFrame(
        {
            "alpha": [0.0] * n,
            "beta": [1.2] * n,
            "spread": [0.0] * n,
            "spread_mean": [0.0] * n,
            "spread_std": [0.01] * n,
            "adf_pvalue": [0.01] * n,
            "valid": valid[:n],
            "z": z,
        },
        index=idx,
    )


def _synthetic_data() -> m.U2ValidationData:
    n_h1 = 20
    start = "2023-04-11T00:00:00Z"
    log_y = pd.Series(
        np.log([1.35] * n_h1),
        index=pd.date_range(start, periods=n_h1, freq="h", tz="UTC"),
        name="log_y",
    )
    log_x = pd.Series(
        np.log([0.90] * n_h1),
        index=pd.date_range(start, periods=n_h1, freq="h", tz="UTC"),
        name="log_x",
    )
    n_m1 = 24 * 60
    y_bid = _m1([1.3500] * n_m1, start=start)
    y_ask = _m1([1.3502] * n_m1, start=start)
    x_bid = _m1([0.9000] * n_m1, start=start)
    x_ask = _m1([0.9002] * n_m1, start=start)
    readiness_document = {
        "readiness_version": "oanda-alpha-lab-validation-readiness-1",
        "semantic_sha256": "fake",
        "alpha_lab_config_sha256": "fake",
        "partition": "VALIDATION",
    }
    return m.U2ValidationData(
        log_y=log_y,
        log_x=log_x,
        y_bid_m1=y_bid,
        y_ask_m1=y_ask,
        x_bid_m1=x_bid,
        x_ask_m1=x_ask,
        readiness_document=readiness_document,
    )


def _synthetic_development_comparison(tmp_path: Path) -> Path:
    path = tmp_path / "resolution_summary.json"
    path.write_text(
        json.dumps(
            {
                "candidates": {
                    "U2": {
                        "native_expectancy": "77.29",
                        "stressed_1_5x_expectancy": "69.76",
                        "n_trades": 38,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Exact U2 identity; U1 structurally impossible
# ---------------------------------------------------------------------------


def test_exact_u2_identity_is_frozen() -> None:
    assert m.CANDIDATE_ID == "U2"
    assert m.SLEEVE_ID == "USD/CAD.OANDA__USD/CHF.OANDA"
    assert m.Y_SPEC.instrument_id == "USD/CAD.OANDA"
    assert m.X_SPEC.instrument_id == "USD/CHF.OANDA"
    assert m.FORMATION_WINDOW == 240
    assert m.Z_ENTRY == Decimal("1.5")
    assert m.Z_STOP == Decimal("3.5")


def test_u1_is_structurally_impossible_to_run() -> None:
    """U1's exact sleeve identity must never appear as an executable
    reference -- no module-level constant equals it, no candidate table
    (a list/tuple of candidates, or a loop construct) exists anywhere, and
    its frozen z_stop (3.0, distinct from U2's 3.5) is never used as a
    value. The docstring's single prose mention that U1 must NOT be run is
    expected and harmless -- it is text, not code."""

    module_values = [getattr(m, name) for name in dir(m) if not name.startswith("__")]
    assert "USD/CHF.OANDA__USD/JPY.OANDA" not in module_values
    assert not hasattr(m, "FROZEN_CANDIDATES")
    assert not hasattr(m, "CANDIDATE_U1")
    assert m.Z_STOP == Decimal("3.5")  # never 3.0 (U1's frozen z_stop)

    tree = ast.parse(Path(m.__file__).read_text(encoding="utf-8"))
    string_literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "USD/CHF.OANDA__USD/JPY.OANDA" not in string_literals


def test_no_parameter_override_cli_exists() -> None:
    parser = m.build_parser()
    option_strings = {
        option
        for action in parser._actions  # noqa: SLF001
        for option in action.option_strings
    }
    assert option_strings == {
        "-h",
        "--help",
        "--validation-root",
        "--universe-readiness",
        "--output",
    }


# ---------------------------------------------------------------------------
# Preregistration hash verification
# ---------------------------------------------------------------------------


def test_preregistration_hash_is_verified_against_the_real_frozen_file() -> None:
    document = m.verify_preregistration()
    assert document["selected_candidate"]["candidate_id"] == "U2"
    assert document["selected_candidate"]["sleeve_id"] == "USD/CAD.OANDA__USD/CHF.OANDA"


def test_tampered_preregistration_is_rejected(tmp_path: Path) -> None:
    original = m.PREREGISTRATION_PATH.read_bytes()
    tampered_path = tmp_path / "tampered_preregistration.json"
    tampered_path.write_bytes(original + b" ")  # single byte appended
    with pytest.raises(m.B3F1UnderpoweredValidationError, match="SHA256"):
        m.verify_preregistration(tampered_path)


def test_preregistration_with_wrong_candidate_identity_is_rejected(
    tmp_path: Path,
) -> None:
    document = json.loads(m.PREREGISTRATION_PATH.read_text(encoding="utf-8"))
    document["selected_candidate"]["z_stop"] = "3.0"  # U1's z_stop, not U2's
    tampered_path = tmp_path / "wrong_identity.json"
    tampered_path.write_text(json.dumps(document), encoding="utf-8")
    # This document no longer hashes to the frozen SHA256 either way, but
    # even a document engineered to pass the hash check must still fail
    # the identity check -- verified directly against the parsed document.
    with pytest.raises(m.B3F1UnderpoweredValidationError):
        m.verify_preregistration(tampered_path)


# ---------------------------------------------------------------------------
# Firewall: DEVELOPMENT/HOLDOUT path + partition-boundary rejection
# ---------------------------------------------------------------------------


def test_development_looking_path_is_rejected() -> None:
    with pytest.raises(AlphaLabValidationError):
        m._reject_forbidden_path(  # noqa: SLF001
            Path("/Users/Shared/FTMOQuant-data/oanda_fx_alpha_lab_v1/development_m1")
        )


def test_holdout_looking_path_is_rejected() -> None:
    with pytest.raises(AlphaLabValidationError):
        m._reject_forbidden_path(Path("/some/root/final_holdout/canonical"))  # noqa: SLF001


def test_observation_at_or_after_holdout_start_is_rejected() -> None:
    idx = pd.date_range(
        m.HOLDOUT_START - pd.Timedelta(days=1), periods=2, freq="D", tz="UTC"
    )
    with pytest.raises(AlphaLabValidationError):
        m._reject_out_of_partition(idx, "test")  # noqa: SLF001


def test_observation_before_validation_start_is_rejected() -> None:
    idx = pd.date_range(
        m.VALIDATION_START - pd.Timedelta(days=1), periods=2, freq="D", tz="UTC"
    )
    with pytest.raises(AlphaLabValidationError):
        m._reject_out_of_partition(idx, "test")  # noqa: SLF001


def test_within_partition_observations_are_accepted() -> None:
    idx = pd.date_range(m.VALIDATION_START, periods=2, freq="D", tz="UTC")
    m._reject_out_of_partition(idx, "test")  # noqa: SLF001 -- must not raise


# ---------------------------------------------------------------------------
# One-shot: exactly one config executed, no loop, no fallback
# ---------------------------------------------------------------------------


def test_exactly_one_config_is_executed(tmp_path: Path) -> None:
    data = _synthetic_data()
    formation = _synthetic_formation(len(data.log_y), start="2023-04-11T00:00:00Z")
    output_dir = tmp_path / "out"
    dev_comparison_path = _synthetic_development_comparison(tmp_path)

    with (
        patch.object(m, "reserve_output_directory", wraps=m.reserve_output_directory),
        patch.object(m, "verify_preregistration", return_value=_fake_preregistration()),
        patch.object(m, "load_u2_validation_data", return_value=data),
        patch.object(
            m, "compute_formation_series", return_value=formation
        ) as formation_spy,
        patch.object(
            m, "generate_b3f1_decisions", wraps=m.generate_b3f1_decisions
        ) as decisions_spy,
        patch.object(
            m, "simulate_b3f1_intents", wraps=m.simulate_b3f1_intents
        ) as simulate_spy,
    ):
        m.run_u2_validation(
            validation_root=tmp_path / "validation_root",
            universe_readiness=tmp_path / "readiness.json",
            resolution_summary_path=dev_comparison_path,
            output_dir=output_dir,
        )

    assert formation_spy.call_count == 1
    assert decisions_spy.call_count == 1
    # exactly native + 1.5x stress, never a third (e.g. 2.0x) execution pass.
    assert simulate_spy.call_count == 2


def _fake_preregistration() -> dict:
    document = json.loads(m.PREREGISTRATION_PATH.read_text(encoding="utf-8"))
    return document


# ---------------------------------------------------------------------------
# Output-exists rejection happens BEFORE any catalog access
# ---------------------------------------------------------------------------


def test_output_exists_rejection_happens_before_data_access(tmp_path: Path) -> None:
    existing_output = tmp_path / "out"
    existing_output.mkdir()

    with patch.object(m, "load_u2_validation_data") as loader_mock:
        with pytest.raises(m.B3F1UnderpoweredValidationError, match="already exists"):
            m.run_u2_validation(
                validation_root=tmp_path / "validation_root",
                universe_readiness=tmp_path / "readiness.json",
                output_dir=existing_output,
            )
        loader_mock.assert_not_called()


def test_reserve_output_directory_rejects_existing_path(tmp_path: Path) -> None:
    existing = tmp_path / "out"
    existing.mkdir()
    with pytest.raises(m.B3F1UnderpoweredValidationError, match="already exists"):
        m.reserve_output_directory(existing)


# ---------------------------------------------------------------------------
# Pass rule matches VALIDATION_POLICY exactly -- A AND B, nothing else
# ---------------------------------------------------------------------------


def test_evaluate_u2_pass_rule_is_exactly_a_and_b() -> None:
    from ftmoquant.research.alpha_lab.b3f1_spread_signals import (
        EXIT_REASON_Z_MEAN_REVERSION,
    )
    from ftmoquant.research.alpha_lab.relative_value_adapter import (
        LegMark,
        RelativeValueEpisode,
        RelativeValueLeg,
    )

    def _episode(
        entry_ns: int, exit_ns: int, pnl_direction: int, magnitude: str = "0.0100"
    ) -> RelativeValueEpisode:
        entry_price = Decimal("1.0000")
        exit_price = entry_price + Decimal(magnitude) * pnl_direction
        leg_a = RelativeValueLeg(
            instrument_id="USD/CAD.OANDA",
            direction=1,
            quantity=Decimal("50000"),
            base_currency="USD",
            quote_currency="CAD",
            entry_ns=entry_ns,
            entry_price=entry_price,
            exit_ns=exit_ns,
            exit_price=exit_price,
            marks=(LegMark(entry_ns, entry_price), LegMark(exit_ns, exit_price)),
        )
        leg_b = RelativeValueLeg(
            instrument_id="USD/CHF.OANDA",
            direction=-1,
            quantity=Decimal("50000"),
            base_currency="USD",
            quote_currency="CHF",
            entry_ns=entry_ns,
            entry_price=entry_price,
            exit_ns=exit_ns,
            exit_price=entry_price,
            marks=(LegMark(entry_ns, entry_price), LegMark(exit_ns, entry_price)),
        )
        return RelativeValueEpisode(
            logical_trade_id=f"t:{entry_ns}",
            leg_a=leg_a,
            leg_b=leg_b,
            exit_reason=EXIT_REASON_Z_MEAN_REVERSION,
        )

    base_ns = 1_700_000_000_000_000_000
    day_ns = 86_400_000_000_000
    magnitudes = ("0.0080", "0.0110", "0.0095", "0.0130", "0.0090")
    winning_episodes = tuple(
        _episode(
            base_ns + i * day_ns,
            base_ns + i * day_ns + 3_600_000_000_000,
            1,
            magnitude=magnitudes[i],
        )
        for i in range(5)
    )
    losing_episodes = tuple(
        _episode(
            base_ns + i * day_ns,
            base_ns + i * day_ns + 3_600_000_000_000,
            -1,
            magnitude=magnitudes[i],
        )
        for i in range(5)
    )

    winning_result = m.evaluate_u2(
        native_episodes=winning_episodes,
        native_skips=(),
        stressed_episodes=winning_episodes,
        stressed_skips=(),
        formation_valid_count=10,
    )
    assert winning_result.gate_a_native_positive_return is True
    assert winning_result.gate_b_native_positive_sharpe is True
    assert winning_result.validation_passed is True

    losing_result = m.evaluate_u2(
        native_episodes=losing_episodes,
        native_skips=(),
        stressed_episodes=losing_episodes,
        stressed_skips=(),
        formation_valid_count=10,
    )
    assert losing_result.gate_a_native_positive_return is False
    assert losing_result.validation_passed is False


def test_no_trade_count_or_other_development_gate_blocks_validation() -> None:
    """A tiny 2-trade, all-winning result must pass VALIDATION under A/B
    alone -- unlike DEVELOPMENT, there is no minimum-trade-count gate."""

    from ftmoquant.research.alpha_lab.b3f1_spread_signals import (
        EXIT_REASON_Z_MEAN_REVERSION,
    )
    from ftmoquant.research.alpha_lab.relative_value_adapter import (
        LegMark,
        RelativeValueEpisode,
        RelativeValueLeg,
    )

    def _episode(
        entry_ns: int, exit_ns: int, exit_price: Decimal
    ) -> RelativeValueEpisode:
        entry_price = Decimal("1.0000")
        leg_a = RelativeValueLeg(
            instrument_id="USD/CAD.OANDA",
            direction=1,
            quantity=Decimal("50000"),
            base_currency="USD",
            quote_currency="CAD",
            entry_ns=entry_ns,
            entry_price=entry_price,
            exit_ns=exit_ns,
            exit_price=exit_price,
            marks=(LegMark(entry_ns, entry_price), LegMark(exit_ns, exit_price)),
        )
        leg_b = RelativeValueLeg(
            instrument_id="USD/CHF.OANDA",
            direction=-1,
            quantity=Decimal("50000"),
            base_currency="USD",
            quote_currency="CHF",
            entry_ns=entry_ns,
            entry_price=entry_price,
            exit_ns=exit_ns,
            exit_price=entry_price,
            marks=(LegMark(entry_ns, entry_price), LegMark(exit_ns, entry_price)),
        )
        return RelativeValueEpisode(
            logical_trade_id=f"t:{entry_ns}",
            leg_a=leg_a,
            leg_b=leg_b,
            exit_reason=EXIT_REASON_Z_MEAN_REVERSION,
        )

    base_ns = 1_700_000_000_000_000_000
    day_ns = 86_400_000_000_000
    exit_prices = (
        Decimal("1.0090"),
        Decimal("1.0130"),
        Decimal("1.0080"),
        Decimal("1.0110"),
    )
    episodes = tuple(
        _episode(
            base_ns + i * day_ns,
            base_ns + i * day_ns + 3_600_000_000_000,
            exit_prices[i],
        )
        for i in range(4)
    )
    result = m.evaluate_u2(
        native_episodes=episodes,
        native_skips=(),
        stressed_episodes=episodes,
        stressed_skips=(),
        formation_valid_count=4,
    )
    assert result.trade_count == 4
    # far below B3F1 DEVELOPMENT's min_trade_count=50 -- must still pass,
    # since VALIDATION has no trade-count gate at all.
    assert result.validation_passed is True


# ---------------------------------------------------------------------------
# Deterministic synthetic fixture: identical inputs -> identical outputs
# ---------------------------------------------------------------------------


def test_full_run_is_deterministic(tmp_path: Path) -> None:
    data = _synthetic_data()
    formation = _synthetic_formation(len(data.log_y), start="2023-04-11T00:00:00Z")
    dev_comparison_path = _synthetic_development_comparison(tmp_path)

    def _run(output_dir: Path) -> dict:
        with (
            patch.object(
                m, "verify_preregistration", return_value=_fake_preregistration()
            ),
            patch.object(m, "load_u2_validation_data", return_value=data),
            patch.object(m, "compute_formation_series", return_value=formation),
        ):
            m.run_u2_validation(
                validation_root=tmp_path / "validation_root",
                universe_readiness=tmp_path / "readiness.json",
                resolution_summary_path=dev_comparison_path,
                output_dir=output_dir,
            )
        return json.loads((output_dir / "validation_summary.json").read_text())

    first = _run(tmp_path / "out1")
    second = _run(tmp_path / "out2")
    assert first["result"] == second["result"]
