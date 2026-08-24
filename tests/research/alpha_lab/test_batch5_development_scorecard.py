from __future__ import annotations

import json
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from ftmoquant.research.alpha_lab.batch5_development_orchestrator import (
    DEVELOPMENT_FOLD_BOUNDARIES,
)
from ftmoquant.research.alpha_lab.batch5_development_scorecard import (
    Batch5DevelopmentScorecardError,
    DevelopmentSleeveInput,
    build_diagnostics_summary,
    build_family_summary,
    build_selection_summary,
    evaluate_development_sleeve,
    write_batch5_artifacts,
)
from ftmoquant.research.alpha_lab.batch5_execution import Batch5TradeResult
from ftmoquant.research.alpha_lab.batch5_preregistration import (
    FAMILY_B5A,
    FAMILY_B5B,
    FAMILY_B5C,
    verify_preregistration,
)
from ftmoquant.research.alpha_lab.batch5_screen import FrequencyStats


def _trades(
    family: str, strategy: str, sleeve: str, instrument: str
) -> tuple[Batch5TradeResult, ...]:
    result = []
    for start, end in zip(
        DEVELOPMENT_FOLD_BOUNDARIES[:-1],
        DEVELOPMENT_FOLD_BOUNDARIES[1:],
        strict=True,
    ):
        exit_timestamp = start + (end - start) / 2
        entry_timestamp = exit_timestamp - timedelta(hours=1)
        result.append(
            Batch5TradeResult(
                family=family,
                strategy_id=strategy,
                sleeve_id=sleeve,
                instrument_id=instrument,
                signal_timestamp=entry_timestamp,
                actual_entry_timestamp=entry_timestamp,
                actual_exit_timestamp=exit_timestamp,
                direction="BUY",
                quantity=Decimal("100000"),
                entry_price=Decimal("1"),
                exit_price=Decimal("1.001"),
                pnl_usd=Decimal("100"),
                return_on_reference_notional=Decimal("0.001"),
                holding_seconds=3600,
                cohort_id=None,
            )
        )
    return tuple(result)


def _inputs() -> tuple[DevelopmentSleeveInput, ...]:
    document = verify_preregistration()
    result: list[DevelopmentSleeveInput] = []
    for row in document["families"][FAMILY_B5A]["sleeves"]:
        trades = _trades(
            FAMILY_B5A,
            "B5A_FROZEN_CFTC_DEALER_DEMAND_SHOCK",
            row["sleeve_id"],
            row["spot_instrument"],
        )
        result.append(
            DevelopmentSleeveInput(
                FAMILY_B5A,
                "B5A_FROZEN_CFTC_DEALER_DEMAND_SHOCK",
                row["sleeve_id"],
                row["spot_instrument"],
                trades,
                trades,
                trades,
                DEVELOPMENT_FOLD_BOUNDARIES,
                FrequencyStats(36, 12, active_year_count=4),
                12,
                4,
            )
        )
    trades = _trades(
        FAMILY_B5B,
        "B5B_FROZEN_DIRECT_AUDCAD_MR",
        "B5B_AUDCAD",
        "AUD/CAD.OANDA",
    )
    result.append(
        DevelopmentSleeveInput(
            FAMILY_B5B,
            "B5B_FROZEN_DIRECT_AUDCAD_MR",
            "B5B_AUDCAD",
            "AUD/CAD.OANDA",
            trades,
            trades,
            trades,
            DEVELOPMENT_FOLD_BOUNDARIES,
            FrequencyStats(
                daily_holding_observation_count=500,
                position_sign_change_count=20,
                rollover_supported=True,
                active_year_count=4,
            ),
            500,
            4,
        )
    )
    for instrument in document["families"][FAMILY_B5C]["literature_anchored_rule"][
        "universe"
    ]:
        sleeve = f"B5C_{instrument.split('.')[0].replace('/', '')}"
        trades = _trades(
            FAMILY_B5C,
            "B5C_FROZEN_DAILY_OVERREACTION_REVERSAL",
            sleeve,
            instrument,
        )
        result.append(
            DevelopmentSleeveInput(
                FAMILY_B5C,
                "B5C_FROZEN_DAILY_OVERREACTION_REVERSAL",
                sleeve,
                instrument,
                trades,
                trades,
                trades,
                DEVELOPMENT_FOLD_BOUNDARIES,
                FrequencyStats(event_count=15, active_year_count=4),
                15,
                4,
            )
        )
    return tuple(result)


def test_exact_13_row_schema_gates_and_per_family_binary_eligibility() -> None:
    inputs = _inputs()
    rows = tuple(evaluate_development_sleeve(item) for item in inputs)
    assert len(rows) == 13
    assert all(row.sleeve_hard_gates_passed for row in rows)
    assert all(row.stressed_1_5x_profit_factor == Decimal("Infinity") for row in rows)
    families = build_family_summary(inputs)
    assert [row["tested_sleeve_count"] for row in families] == [7, 1, 5]
    assert all(row["eligible_for_future_validation"] for row in families)
    selection = build_selection_summary(families)
    assert selection["selection_scope"] == "binary_per_family_no_global_ranking"
    assert selection["eligible_family_count"] == 3


def test_family_aggregate_rejects_pair_removal() -> None:
    with pytest.raises(ValueError, match="all and only frozen"):
        build_family_summary(_inputs()[:-1])


def test_artifact_contract_hashes_ordering_and_write_once(tmp_path: Path) -> None:
    inputs = tuple(reversed(_inputs()))
    rows = tuple(evaluate_development_sleeve(item) for item in inputs)
    families = build_family_summary(inputs)
    output = tmp_path / "batch5"
    write_batch5_artifacts(
        sleeve_scorecard=rows,
        family_summary=families,
        selection_summary=build_selection_summary(families),
        diagnostics_summary=build_diagnostics_summary(rows),
        metadata={"validation_accessed": False, "holdout_accessed": False},
        output_dir=output,
    )
    assert {path.name for path in output.iterdir()} == {
        "sleeve_scorecard.csv",
        "family_summary.csv",
        "selection_summary.json",
        "diagnostics_summary.json",
        "metadata.json",
        "artifact_hashes.json",
    }
    scorecard = pd.read_csv(output / "sleeve_scorecard.csv")
    assert len(scorecard) == 13
    assert list(zip(scorecard.family, scorecard.sleeve_id, strict=True)) == sorted(
        zip(scorecard.family, scorecard.sleeve_id, strict=True)
    )
    hashes = json.loads((output / "artifact_hashes.json").read_text())
    assert len(hashes) == 5
    with pytest.raises(Batch5DevelopmentScorecardError, match="refusing to overwrite"):
        write_batch5_artifacts(
            sleeve_scorecard=rows,
            family_summary=families,
            selection_summary={},
            diagnostics_summary={},
            metadata={},
            output_dir=output,
        )
