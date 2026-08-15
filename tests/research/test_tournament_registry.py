from __future__ import annotations

import importlib.util
from pathlib import Path

from ftmoquant.research.tournament_dashboard import build_dashboard_snapshot
from ftmoquant.research.tournament_registry import (
    CandidateEligibility,
    CandidateImplementationStatus,
    EligibilityEnvironment,
    candidate_registry,
    preregistered_selection_contract,
)


def test_current_candidate_eligibility_blocks_pit_and_cross_sectional_work() -> None:
    registry = candidate_registry()
    entries = {item.candidate_id: item for item in registry.ordered_entries}

    assert entries["ts_momentum_v1"].eligibility is (
        CandidateEligibility.ELIGIBLE_FOR_IMPLEMENTATION
    )
    assert entries["session_range_expansion_v1"].eligibility is (
        CandidateEligibility.ELIGIBLE_FOR_IMPLEMENTATION
    )
    assert "pit_carry_inputs" in entries["carry_momentum_v1"].unmet_prerequisites
    carry_value = entries["carry_momentum_value_v1"]
    assert "pit_carry_inputs" in carry_value.unmet_prerequisites
    assert "pit_value_inputs" in carry_value.unmet_prerequisites
    assert "minimum_cross_sectional_instruments=4" in (carry_value.unmet_prerequisites)
    assert entries["session_range_expansion_v1"].results_accessed is True
    assert entries["liquidity_shock_reversion_v1"].implementation_status is (
        CandidateImplementationStatus.DEVELOPMENT_FAILED_RETIRED
    )
    assert entries["liquidity_shock_reversion_v1"].results_accessed is True
    assert all(
        item.results_accessed is False
        for item in registry.ordered_entries
        if item.candidate_id
        not in {"session_range_expansion_v1", "liquidity_shock_reversion_v1"}
    )
    assert entries["ts_momentum_v1"].implementation_status is (
        CandidateImplementationStatus.IMPLEMENTED_NOT_EVALUATED
    )
    assert entries["leo_gbpusd_v1"].implementation_status is (
        CandidateImplementationStatus.IMPLEMENTED_NOT_EVALUATED
    )
    assert entries["leo_gbpusd_v1"].results_accessed is False
    assert entries["ts_momentum_v1"].strategy_logic_present is True
    assert entries["ts_momentum_v1"].strategy_config_sha256 == (
        "edcbe2e4afe631e5fde1223558122ecf4d796abd0610729313ebbb32a468ccd5"
    )
    assert entries["liquidity_shock_reversion_v1"].strategy_logic_present is True
    assert entries["liquidity_shock_reversion_v1"].strategy_config_sha256 == (
        "55a2a1814a507ebf53f5713d9672cf5d4fda19790d9c808efc0a9773bed27acd"
    )
    assert entries["session_range_expansion_v1"].implementation_status is (
        CandidateImplementationStatus.DEVELOPMENT_FAILED_RETIRED
    )
    assert entries["session_range_expansion_v1"].strategy_logic_present is True
    assert entries["session_range_expansion_v1"].strategy_config_sha256 == (
        "5303094db45b3d9164d1854787c39d9ce0e69974875689dc51742e4495b9e472"
    )
    assert all(
        item.strategy_logic_present is False
        for item in registry.ordered_entries
        if item.candidate_id
        not in {
            "ts_momentum_v1",
            "session_range_expansion_v1",
            "liquidity_shock_reversion_v1",
            "leo_gbpusd_v1",
        }
    )


def test_cross_sectional_candidate_needs_both_universe_and_pit_prerequisites() -> None:
    registry = candidate_registry(
        EligibilityEnvironment(
            instrument_count=4,
            pit_inputs=frozenset({"pit_carry_inputs", "pit_value_inputs"}),
        )
    )
    entry = next(
        item
        for item in registry.ordered_entries
        if item.candidate_id == "carry_momentum_value_v1"
    )

    assert entry.eligibility is CandidateEligibility.ELIGIBLE_FOR_IMPLEMENTATION
    assert entry.unmet_prerequisites == ()


def test_selection_contract_is_deterministic_and_does_not_choose_a_winner() -> None:
    contract = preregistered_selection_contract()

    assert contract == preregistered_selection_contract()
    assert len(contract.semantic_sha256) == 64
    assert any("SPA" in item for item in contract.multiple_testing_policy)
    assert any("MCS" in item for item in contract.multiple_testing_policy)
    assert "repetitions=10000" in contract.resampling_configuration
    assert "seed=14042026" in contract.resampling_configuration
    assert not hasattr(contract, "winner")


def test_marimo_notebook_imports_and_remains_development_only(tmp_path: Path) -> None:
    path = Path("research/g1_4_tournament.py")
    spec = importlib.util.spec_from_file_location("g1_4_tournament_notebook", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    snapshot = build_dashboard_snapshot(tmp_path / "missing-readiness.json")
    assert module.__generated_with == "0.23.15"
    assert snapshot["validation"] == "LOCKED / unavailable"
    assert snapshot["holdout"] == "LOCKED / unavailable"
    assert snapshot["strategy_returns"] == "NOT ACCESSED / unavailable"
    assert snapshot["development_artifacts"] == ()
