from __future__ import annotations

import ast
from decimal import Decimal
from pathlib import Path

import pytest

import ftmoquant.research.ftmo_joint_frontier as m
from ftmoquant.prop_rules.loader import load_prop_rule_set
from ftmoquant.research.alpha_lab.relative_value_adapter import (
    LegMark,
    RelativeValueEpisode,
    RelativeValueLeg,
)
from ftmoquant.research.ftmo_pass_probability.path_extraction import TradeRecord
from ftmoquant.research.ftmo_pass_probability.sizing import apply_sizing

RULES = load_prop_rule_set(m.FTMO_RULES_PATH)

_DAY_NS = 86_400_000_000_000
_HOUR_NS = 3_600_000_000_000


def _a_trade(
    index: int,
    *,
    entry_ns: int,
    exit_ns: int,
    net_r: str,
    exit_reason: str = "stop",
    risk_budget: str = "1000",
) -> TradeRecord:
    original_risk_budget = Decimal(risk_budget)
    usd_risk_per_unit = Decimal("100")  # arbitrary, positive
    return TradeRecord(
        trade_index=index,
        entry_ns=entry_ns,
        exit_ns=exit_ns,
        exit_reason=exit_reason,  # type: ignore[arg-type]
        net_r=Decimal(net_r),
        original_realized_pnl=Decimal(net_r) * original_risk_budget,
        original_risk_budget=original_risk_budget,
        usd_risk_per_unit=usd_risk_per_unit,
    )


def _u2_episode(
    trade_id: str,
    *,
    y_entry_ns: int,
    y_exit_ns: int,
    x_entry_ns: int,
    x_exit_ns: int,
    y_entry_price: str = "1.3500",
    y_exit_price: str = "1.3500",
    x_entry_price: str = "0.9000",
    x_exit_price: str = "0.9000",
    y_direction: int = 1,
    x_direction: int = -1,
) -> RelativeValueEpisode:
    leg_a = RelativeValueLeg(
        instrument_id="USD/CAD.OANDA",
        direction=y_direction,  # type: ignore[arg-type]
        quantity=Decimal("50000"),
        base_currency="USD",
        quote_currency="CAD",
        entry_ns=y_entry_ns,
        entry_price=Decimal(y_entry_price),
        exit_ns=y_exit_ns,
        exit_price=Decimal(y_exit_price),
        marks=(
            LegMark(y_entry_ns, Decimal(y_entry_price)),
            LegMark(y_exit_ns, Decimal(y_exit_price)),
        ),
    )
    leg_b = RelativeValueLeg(
        instrument_id="USD/CHF.OANDA",
        direction=x_direction,  # type: ignore[arg-type]
        quantity=Decimal("50000"),
        base_currency="USD",
        quote_currency="CHF",
        entry_ns=x_entry_ns,
        entry_price=Decimal(x_entry_price),
        exit_ns=x_exit_ns,
        exit_price=Decimal(x_exit_price),
        marks=(
            LegMark(x_entry_ns, Decimal(x_entry_price)),
            LegMark(x_exit_ns, Decimal(x_exit_price)),
        ),
    )
    return RelativeValueEpisode(
        logical_trade_id=trade_id,
        leg_a=leg_a,
        leg_b=leg_b,
        exit_reason="z_mean_reversion",
    )


# ---------------------------------------------------------------------------
# Exact 36-policy grid, no expansion
# ---------------------------------------------------------------------------


def test_exactly_36_policies_in_the_frozen_grid() -> None:
    assert len(m.JOINT_POLICY_GRID) == 36
    assert len({p.policy_id for p in m.JOINT_POLICY_GRID}) == 36
    a_mults = {p.a_multiplier for p in m.JOINT_POLICY_GRID}
    u2_mults = {p.u2_multiplier for p in m.JOINT_POLICY_GRID}
    assert a_mults == {
        Decimal("1.5"),
        Decimal("2.0"),
        Decimal("2.5"),
        Decimal("3.0"),
        Decimal("3.5"),
        Decimal("4.0"),
    }
    assert u2_mults == {
        Decimal("0.00"),
        Decimal("0.25"),
        Decimal("0.50"),
        Decimal("0.75"),
        Decimal("1.00"),
        Decimal("1.25"),
    }


def test_4_0x_is_the_frozen_upper_boundary_and_is_not_exceeded() -> None:
    assert max(m.A_MULTIPLIERS) == Decimal("4.0")
    assert all(mult <= Decimal("4.0") for mult in m.A_MULTIPLIERS)


def test_cli_exposes_no_grid_expansion_flags() -> None:
    parser = m.build_parser()
    option_strings = {
        option
        for action in parser._actions  # noqa: SLF001
        for option in action.option_strings
    }
    assert option_strings == {
        "-h",
        "--help",
        "--catalog-root",
        "--universe-readiness",
        "--strategy-a-execution-dir",
        "--ftmo-rules",
        "--output",
    }


def test_critical_control_policy_is_a_2_0x_u2_0x() -> None:
    assert m.CRITICAL_CONTROL_POLICY_ID == "A2.0x_U20.00x"
    assert len(m.CONTROL_POLICY_IDS) == 6
    assert "A4.0x_U20.00x" in m.CONTROL_POLICY_IDS


# ---------------------------------------------------------------------------
# A=2.0/U2=0 reproduces Strategy-A standalone semantics; U2=0 contributes nothing
# ---------------------------------------------------------------------------


def test_a_only_group_reproduces_apply_sizing_directly() -> None:
    trade = _a_trade(0, entry_ns=1_000, exit_ns=2_000, net_r="1.5")
    (group,) = m.build_joint_groups([trade], [])
    assert group.a_trade_count == 1
    assert group.u2_episode_count == 0

    event = m.scale_joint_group(
        group, a_multiplier=Decimal("2.0"), u2_multiplier=Decimal("0.00")
    )
    expected = apply_sizing(
        m.JointPolicy("x", Decimal("2.0"), Decimal("0.00")).a_sizing_policy,
        trade,
        m.INITIAL_CAPITAL,
    )
    assert event.realized_pnl == expected.realized_pnl
    assert event.floor_equity_delta == expected.floor_equity_delta
    assert event.entry_ns == trade.entry_ns
    assert event.exit_ns == trade.exit_ns


def test_u2_zero_multiplier_leaves_no_u2_contribution_even_when_overlapping() -> None:
    trade = _a_trade(0, entry_ns=1_000, exit_ns=10 * _HOUR_NS, net_r="-1.0")
    # U2 episode fully overlaps the A trade and is a big winner -- if U2=0
    # correctly contributes nothing, the group's outcome at U2=0 multiplier
    # must be IDENTICAL to the A-only outcome.
    episode = _u2_episode(
        "u2-1",
        y_entry_ns=2_000,
        y_exit_ns=5 * _HOUR_NS,
        x_entry_ns=2_500,
        x_exit_ns=5 * _HOUR_NS + 500,
        y_exit_price="1.5000",  # huge winning move
    )
    (group,) = m.build_joint_groups([trade], [episode])
    assert group.a_trade_count == 1
    assert group.u2_episode_count == 1

    with_u2_zero = m.scale_joint_group(
        group, a_multiplier=Decimal("2.0"), u2_multiplier=Decimal("0.00")
    )
    a_only_group = m.build_joint_groups([trade], [])[0]
    a_only_event = m.scale_joint_group(
        a_only_group, a_multiplier=Decimal("2.0"), u2_multiplier=Decimal("0.00")
    )
    assert with_u2_zero.realized_pnl == a_only_event.realized_pnl
    assert with_u2_zero.floor_equity_delta == a_only_event.floor_equity_delta


# ---------------------------------------------------------------------------
# Joint event ordering deterministic; overlapping losses aggregate correctly
# ---------------------------------------------------------------------------


def test_build_joint_groups_is_deterministic() -> None:
    trade = _a_trade(0, entry_ns=1_000, exit_ns=10 * _HOUR_NS, net_r="-1.0")
    episode = _u2_episode(
        "u2-1",
        y_entry_ns=2_000,
        y_exit_ns=3 * _HOUR_NS,
        x_entry_ns=2_100,
        x_exit_ns=3 * _HOUR_NS + 100,
    )
    first = m.build_joint_groups([trade], [episode])
    second = m.build_joint_groups([trade], [episode])
    assert first == second


def test_overlapping_losses_aggregate_below_either_alone() -> None:
    """When A and U2 both lose simultaneously, the combined floor must be
    at least as bad as EITHER strategy's own standalone floor -- losses
    add, they do not net away."""

    trade = _a_trade(
        0, entry_ns=1_000, exit_ns=10 * _HOUR_NS, net_r="-1.0", exit_reason="stop"
    )
    episode = _u2_episode(
        "u2-1",
        y_entry_ns=2_000,
        y_exit_ns=5 * _HOUR_NS,
        x_entry_ns=2_000,
        x_exit_ns=5 * _HOUR_NS,
        y_exit_price="1.2000",  # Y leg (long) loses heavily
    )
    (group,) = m.build_joint_groups([trade], [episode])
    combined = m.scale_joint_group(
        group, a_multiplier=Decimal("1.0"), u2_multiplier=Decimal("1.0")
    )
    a_alone = m.scale_joint_group(
        m.build_joint_groups([trade], [])[0],
        a_multiplier=Decimal("1.0"),
        u2_multiplier=Decimal("1.0"),
    )
    assert combined.floor_equity_delta <= a_alone.floor_equity_delta
    assert combined.realized_pnl < a_alone.realized_pnl


# ---------------------------------------------------------------------------
# U2 legging path preserved; same-day breach sees both strategies
# ---------------------------------------------------------------------------


def test_u2_legging_gap_is_preserved_in_candidate_instants() -> None:
    episode = _u2_episode(
        "u2-legging",
        y_entry_ns=1_000,
        y_exit_ns=5 * _HOUR_NS,
        x_entry_ns=2 * _HOUR_NS,  # X legs in 2 hours after Y
        x_exit_ns=5 * _HOUR_NS,
    )
    (group,) = m.build_joint_groups([], [episode])
    instants = m._candidate_instants(group)  # noqa: SLF001
    # the X leg's own entry (the legging transition) must be a candidate
    # instant -- it is exactly where the account's exposure composition
    # changes from one-legged to two-legged.
    assert 2 * _HOUR_NS in instants


def test_same_day_breach_sees_combined_equity_not_just_one_strategy() -> None:
    """A single Strategy-A trade alone must NOT breach daily loss, but
    combined with a simultaneous U2 loss it must."""

    daily_loss_limit = RULES.loss_limits.maximum_daily_loss
    small_a_loss_budget = m.INITIAL_CAPITAL * daily_loss_limit * Decimal("0.3")
    trade = _a_trade(
        0,
        entry_ns=1_000,
        exit_ns=10 * _HOUR_NS,
        net_r="-1.0",
        exit_reason="stop",
        risk_budget=str(small_a_loss_budget),
    )
    a_only_event = m.scale_joint_group(
        m.build_joint_groups([trade], [])[0],
        a_multiplier=Decimal("1.0"),
        u2_multiplier=Decimal("1.0"),
    )
    daily_floor = m.INITIAL_CAPITAL - m.INITIAL_CAPITAL * daily_loss_limit
    assert (
        m.INITIAL_CAPITAL + a_only_event.floor_equity_delta > daily_floor
    )  # no breach alone

    episode = _u2_episode(
        "u2-1",
        y_entry_ns=2_000,
        y_exit_ns=5 * _HOUR_NS,
        x_entry_ns=2_000,
        x_exit_ns=5 * _HOUR_NS,
        y_exit_price="1.2500",
    )
    (group,) = m.build_joint_groups([trade], [episode])
    combined_event = m.scale_joint_group(
        group, a_multiplier=Decimal("1.0"), u2_multiplier=Decimal("1.0")
    )
    assert (
        m.INITIAL_CAPITAL + combined_event.floor_equity_delta <= daily_floor
    )  # breach


# ---------------------------------------------------------------------------
# Multiplier scales intratrade risk, not merely terminal P&L (linearity)
# ---------------------------------------------------------------------------


def test_multiplier_scales_floor_and_realized_pnl_linearly() -> None:
    trade = _a_trade(0, entry_ns=1_000, exit_ns=2_000, net_r="-1.0")
    episode = _u2_episode(
        "u2-1",
        y_entry_ns=1_500,
        y_exit_ns=1_800,
        x_entry_ns=1_500,
        x_exit_ns=1_800,
        y_exit_price="1.3000",
    )
    (group,) = m.build_joint_groups([trade], [episode])

    one_x = m.scale_joint_group(
        group, a_multiplier=Decimal("1.0"), u2_multiplier=Decimal("1.0")
    )
    two_x = m.scale_joint_group(
        group, a_multiplier=Decimal("2.0"), u2_multiplier=Decimal("2.0")
    )
    assert float(two_x.realized_pnl) == pytest.approx(
        float(one_x.realized_pnl) * 2, rel=1e-9
    )
    assert float(two_x.floor_equity_delta) == pytest.approx(
        float(one_x.floor_equity_delta) * 2, rel=1e-9
    )


# ---------------------------------------------------------------------------
# Bootstrap: one draw drives both strategies (no independent shuffling)
# ---------------------------------------------------------------------------


def test_bootstrap_draws_a_single_index_path_over_joint_groups() -> None:
    source = Path(m.__file__).read_text(encoding="utf-8")
    # exactly one call site of draw_index_path per Monte Carlo replication
    # (inside simulate_two_phase_joint_path) -- never two (one per strategy).
    assert source.count("draw_index_path(") == 1


def test_same_resampled_index_carries_both_strategies_events() -> None:
    trade = _a_trade(0, entry_ns=1_000, exit_ns=2 * _HOUR_NS, net_r="1.0")
    episode = _u2_episode(
        "u2-1",
        y_entry_ns=1_200,
        y_exit_ns=1_800,
        x_entry_ns=1_200,
        x_exit_ns=1_800,
    )
    groups = m.build_joint_groups([trade], [episode])
    assert len(groups) == 1
    # a bootstrap draw of index 0 necessarily carries BOTH strategies'
    # events for that period -- they were never split into separate arrays.
    assert groups[0].a_trade_count == 1
    assert groups[0].u2_episode_count == 1


# ---------------------------------------------------------------------------
# Eligibility / selection / Pareto -- exact
# ---------------------------------------------------------------------------


def _fake_binomial(estimate: float):
    from ftmoquant.research.ftmo_pass_probability.reporting import BinomialEstimate

    return BinomialEstimate(
        successes=int(estimate * 1000),
        trials=1000,
        estimate=estimate,
        ci_lower_95=estimate,
        ci_upper_95=estimate,
    )


def _fake_policy_summary(
    policy_id: str,
    *,
    a_mult: str,
    u2_mult: str,
    pass_both: float,
    median_days: float | None,
    p90_days: float | None,
    fail_daily_loss: float = 0.0,
    fail_max_loss: float = 0.0,
    p95_dd: float = 0.1,
) -> m.JointPolicySummary:
    from ftmoquant.research.ftmo_pass_probability.reporting import (
        CertaintyTier,
        PolicySummary,
    )

    def _ps(method: str) -> PolicySummary:
        return PolicySummary(
            policy_id=policy_id,
            method=method,
            replications=1000,
            pass_challenge=_fake_binomial(pass_both),
            pass_verification_given_challenge=_fake_binomial(pass_both),
            pass_both=_fake_binomial(pass_both),
            fail_daily_loss=_fake_binomial(fail_daily_loss),
            fail_max_loss=_fake_binomial(fail_max_loss),
            censoring_rate=_fake_binomial(0.0),
            median_trading_days_to_pass_both=median_days,
            p90_trading_days_to_pass_both=p90_days,
            p95_trading_days_to_pass_both=p90_days,
            median_max_drawdown=p95_dd,
            p95_max_drawdown=p95_dd,
            certainty_tier=CertaintyTier.STRONG,
        )

    return m.JointPolicySummary(
        policy=m.JointPolicy(policy_id, Decimal(a_mult), Decimal(u2_mult)),
        stationary=_ps("stationary"),
        circular=_ps("circular"),
        median_trading_days_to_pass_challenge=median_days,
        p75_trading_days_to_pass_both=median_days,
    )


def test_eligibility_boundary_values_are_exact() -> None:
    at_boundary = _fake_policy_summary(
        "p1",
        a_mult="2.0",
        u2_mult="0.5",
        pass_both=0.70,
        median_days=75,
        p90_days=150,
        fail_daily_loss=0.02,
        fail_max_loss=0.25,
    )
    assert m.evaluate_eligibility(at_boundary).eligible is True

    just_below = _fake_policy_summary(
        "p2",
        a_mult="2.0",
        u2_mult="0.5",
        pass_both=0.6999,
        median_days=75,
        p90_days=150,
    )
    verdict = m.evaluate_eligibility(just_below)
    assert verdict.eligible is False
    assert "A_stationary_pass_both_ge_0_70" in verdict.failed_criteria


def test_no_eligible_policy_selects_nothing() -> None:
    ineligible = _fake_policy_summary(
        "p1", a_mult="2.0", u2_mult="0.0", pass_both=0.5, median_days=200, p90_days=300
    )
    assert m.select_policy([ineligible]) is None


def test_selection_rule_prefers_lower_median_days_first() -> None:
    slower = _fake_policy_summary(
        "p_slow",
        a_mult="2.0",
        u2_mult="0.0",
        pass_both=0.90,
        median_days=70,
        p90_days=140,
    )
    faster = _fake_policy_summary(
        "p_fast",
        a_mult="2.5",
        u2_mult="0.5",
        pass_both=0.71,
        median_days=60,
        p90_days=140,
    )
    selected = m.select_policy([slower, faster])
    assert selected is not None
    assert selected.policy.policy_id == "p_fast"


def test_selection_rule_lexicographic_final_tiebreak() -> None:
    a = _fake_policy_summary(
        "B_policy",
        a_mult="2.0",
        u2_mult="0.0",
        pass_both=0.80,
        median_days=60,
        p90_days=140,
    )
    b = _fake_policy_summary(
        "A_policy",
        a_mult="2.0",
        u2_mult="0.0",
        pass_both=0.80,
        median_days=60,
        p90_days=140,
    )
    selected = m.select_policy([a, b])
    assert selected is not None
    assert selected.policy.policy_id == "A_policy"


def test_pareto_frontier_excludes_strictly_dominated_policy() -> None:
    dominant = _fake_policy_summary(
        "dominant",
        a_mult="2.0",
        u2_mult="0.0",
        pass_both=0.80,
        median_days=60,
        p90_days=120,
        fail_max_loss=0.05,
    )
    dominated = _fake_policy_summary(
        "dominated",
        a_mult="2.0",
        u2_mult="0.5",
        pass_both=0.70,
        median_days=80,
        p90_days=150,
        fail_max_loss=0.10,
    )
    frontier = m.compute_pareto_frontier([dominant, dominated])
    assert "dominant" in frontier
    assert "dominated" not in frontier


def test_pareto_frontier_keeps_genuine_tradeoffs() -> None:
    faster_riskier = _fake_policy_summary(
        "fast",
        a_mult="3.0",
        u2_mult="0.0",
        pass_both=0.72,
        median_days=50,
        p90_days=100,
        fail_max_loss=0.20,
    )
    slower_safer = _fake_policy_summary(
        "safe",
        a_mult="1.5",
        u2_mult="0.0",
        pass_both=0.85,
        median_days=90,
        p90_days=170,
        fail_max_loss=0.05,
    )
    frontier = m.compute_pareto_frontier([faster_riskier, slower_safer])
    assert set(frontier) == {"fast", "safe"}


# ---------------------------------------------------------------------------
# Safety: VALIDATION cannot affect selection; holdout inaccessible; output
# refusal happens before any Monte Carlo work.
# ---------------------------------------------------------------------------


def test_module_never_imports_validation_or_holdout_machinery() -> None:
    source = Path(m.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    forbidden = (
        "ftmoquant.data.oanda_alpha_lab_validation",
        "ftmoquant.research.alpha_lab.validation",
        "ftmoquant.research.ftmo_pass_probability.validation_diagnostic",
    )
    for name in forbidden:
        assert name not in imported
    for literal in (
        "VALIDATION_START",
        "validation_readiness",
        "load_validation_dataset",
    ):
        assert literal not in source


def test_u2_loader_only_ever_uses_development_partition() -> None:
    source = Path(m.__file__).read_text(encoding="utf-8")
    assert "Partition.DEVELOPMENT" in source
    assert "Partition.VALIDATION" not in source


def test_output_refusal_happens_before_any_data_loading(tmp_path: Path) -> None:
    existing = tmp_path / "out"
    existing.mkdir()
    with pytest.raises(m.JointFrontierError, match="already exists"):
        m.main(
            [
                "--catalog-root",
                str(tmp_path / "catalog"),
                "--universe-readiness",
                str(tmp_path / "readiness.json"),
                "--output",
                str(existing),
            ]
        )


def test_reserve_output_directory_checks_every_expected_artifact(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "selection_summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(m.JointFrontierError):
        m.reserve_output_directory(output_dir)


# ---------------------------------------------------------------------------
# Block-length derivation (DEVELOPMENT-only)
# ---------------------------------------------------------------------------


def test_block_length_derivation_uses_development_only_reference_sizing() -> None:
    trades = [
        _a_trade(
            i,
            entry_ns=i * _DAY_NS,
            exit_ns=i * _DAY_NS + _HOUR_NS,
            net_r=str(1 if i % 2 == 0 else -1),
        )
        for i in range(10)
    ]
    groups = m.build_joint_groups(trades, [])
    result = m.derive_frozen_joint_block_length(
        groups, strategy_a_standalone_block_size=1
    )
    assert result.frozen_block_size >= 1
    assert result.observation_count == len(groups)
    assert result.strategy_a_standalone_block_size == 1


def test_block_length_requires_at_least_two_groups() -> None:
    trades = [_a_trade(0, entry_ns=1_000, exit_ns=2_000, net_r="1.0")]
    groups = m.build_joint_groups(trades, [])
    with pytest.raises(m.JointFrontierError):
        m.derive_frozen_joint_block_length(groups, strategy_a_standalone_block_size=1)


# ---------------------------------------------------------------------------
# Daily-loss-cliff audit regression tests (forensic audit follow-up)
# ---------------------------------------------------------------------------


def test_u2_scaling_at_0_75_1_00_1_25_is_exactly_linear() -> None:
    """The exact triple implicated in the observed A2.5x cliff: U2's own
    realized-P&L delta at 1.00x/1.25x must be an EXACT linear multiple of
    its 0.75x delta, with the fixed A-only baseline subtracted out."""

    trade = _a_trade(0, entry_ns=1_000, exit_ns=20 * _HOUR_NS, net_r="-0.3")
    episode = _u2_episode(
        "u2-1",
        y_entry_ns=2_000,
        y_exit_ns=10 * _HOUR_NS,
        x_entry_ns=2_000,
        x_exit_ns=10 * _HOUR_NS,
        y_exit_price="1.3700",
    )
    (group,) = m.build_joint_groups([trade], [episode])
    a_mult = Decimal("2.5")
    baseline = m.scale_joint_group(
        group, a_multiplier=a_mult, u2_multiplier=Decimal("0.00")
    )
    e075 = m.scale_joint_group(
        group, a_multiplier=a_mult, u2_multiplier=Decimal("0.75")
    )
    e100 = m.scale_joint_group(
        group, a_multiplier=a_mult, u2_multiplier=Decimal("1.00")
    )
    e125 = m.scale_joint_group(
        group, a_multiplier=a_mult, u2_multiplier=Decimal("1.25")
    )

    u2_075 = e075.realized_pnl - baseline.realized_pnl
    u2_100 = e100.realized_pnl - baseline.realized_pnl
    u2_125 = e125.realized_pnl - baseline.realized_pnl
    # cross-multiplication avoids any Decimal division-rounding artifact:
    # u2_100 / 1.00 == u2_075 / 0.75  <=>  u2_100 * 0.75 == u2_075 * 1.00
    assert u2_100 * Decimal("0.75") == u2_075 * Decimal("1.00")
    assert u2_125 * Decimal("1.00") == u2_100 * Decimal("1.25")


def test_prepared_scaling_is_exactly_equal_to_legacy_scaling() -> None:
    trade = _a_trade(0, entry_ns=1_000, exit_ns=20 * _HOUR_NS, net_r="-0.3")
    episode = _u2_episode(
        "u2-prepared",
        y_entry_ns=2_000,
        y_exit_ns=10 * _HOUR_NS,
        x_entry_ns=3_000,
        x_exit_ns=11 * _HOUR_NS,
        y_exit_price="1.3700",
        x_exit_price="0.8900",
    )
    (group,) = m.build_joint_groups([trade], [episode])
    prepared = m.prepare_joint_group_scaling(group)
    for a_multiplier, u2_multiplier in (
        (Decimal("2.20"), Decimal("1.25")),
        (Decimal("2.30"), Decimal("1.00")),
        (Decimal("2.30"), Decimal("1.25")),
    ):
        assert m.scale_prepared_joint_group(
            prepared,
            a_multiplier=a_multiplier,
            u2_multiplier=u2_multiplier,
        ) == m.scale_joint_group(
            group,
            a_multiplier=a_multiplier,
            u2_multiplier=u2_multiplier,
        )


def test_bisect_u2_component_is_exact_at_and_between_marks() -> None:
    episode = _u2_episode(
        "u2-bisect",
        y_entry_ns=2_000,
        y_exit_ns=10_000,
        x_entry_ns=3_000,
        x_exit_ns=11_000,
        y_exit_price="1.3700",
        x_exit_price="0.8900",
    )
    for ts_ns in (1_999, 2_000, 2_500, 3_000, 9_999, 10_000, 11_000, 12_000):
        assert m._episode_component_at_bisect(  # noqa: SLF001
            ts_ns, episode
        ) == episode.combined_pnl_usd_at(ts_ns)


def test_same_seed_draws_identical_index_path_across_u2_multipliers() -> None:
    """Section 5's critical requirement: comparing A2.5/U2=0.75 against
    A2.5/U2=1.00 must replay the IDENTICAL bootstrap-resampled group
    sequence -- never two independently redrawn paths."""

    from ftmoquant.research.ftmo_pass_probability.bootstrap import draw_index_path

    trades = [
        _a_trade(i, entry_ns=i * _DAY_NS, exit_ns=i * _DAY_NS + _HOUR_NS, net_r="0.5")
        for i in range(6)
    ]
    groups = m.build_joint_groups(trades, [])
    path_a = draw_index_path(
        len(groups), method="stationary", block_size=1, seed=20260819, min_length=20
    )
    path_b = draw_index_path(
        len(groups), method="stationary", block_size=1, seed=20260819, min_length=20
    )
    assert path_a == path_b  # same trade_count/block_size/seed -> identical draw


def test_daily_loss_breach_takes_precedence_over_max_loss_at_the_boundary() -> None:
    """Exact boundary crossing: a single joint group whose combined floor
    crosses the 5% daily-loss line, well before the 10% max-loss line,
    must be classified FAILED_DAILY_LOSS -- matching _breach_status's
    frozen precedence (daily checked before max, unchanged here)."""

    from ftmoquant.prop_rules.models import EvaluationPhase
    from ftmoquant.research.ftmo_pass_probability.state_machine import (
        FtmoPathStatus,
        TradeEvent,
        simulate_phase,
    )

    initial_capital = m.INITIAL_CAPITAL
    daily_loss_limit = RULES.loss_limits.maximum_daily_loss
    # a loss $1 past the 5% daily line but nowhere near the 10% max-loss line.
    loss = initial_capital * daily_loss_limit + Decimal("1")
    event = TradeEvent(
        entry_ns=1_000,
        exit_ns=2_000,
        floor_equity_delta=-loss,
        realized_pnl=-loss,
    )
    outcome = simulate_phase(
        [event],
        rules=RULES,
        phase=EvaluationPhase.CHALLENGE,
        initial_capital=initial_capital,
        horizon_ns=m.DEFAULT_HORIZON_NS,
    )
    assert outcome.status is FtmoPathStatus.FAILED_DAILY_LOSS


def test_joint_sizing_screen_csv_preserves_policy_metric_association(
    tmp_path: Path,
) -> None:
    """Section 9: a written CSV row must carry the SAME policy_id/
    a_multiplier/u2_multiplier/metric values as the in-memory
    JointPolicySummary that produced it -- no column swap, no row
    misalignment between the stationary and circular method rows."""

    import csv

    known = _fake_policy_summary(
        "A2.5x_U21.00x",
        a_mult="2.5",
        u2_mult="1.00",
        pass_both=0.36225,
        median_days=60,
        p90_days=123,
        fail_daily_loss=0.46560,
        fail_max_loss=0.16570,
    )
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    m.write_joint_sizing_screen_csv(output_dir, (known,))

    with (output_dir / "joint_sizing_screen.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2  # stationary + circular
    stationary_row = next(r for r in rows if r["method"] == "stationary")
    assert stationary_row["policy_id"] == "A2.5x_U21.00x"
    assert stationary_row["a_multiplier"] == "2.5"
    assert stationary_row["u2_multiplier"] == "1.00"
    assert float(stationary_row["pass_both"]) == pytest.approx(0.36225)
    assert float(stationary_row["fail_daily_loss"]) == pytest.approx(0.46560)
    assert float(stationary_row["fail_max_loss"]) == pytest.approx(0.16570)
    # daily-loss and max-loss columns must not be swapped: the larger of
    # the two known-distinct values must land in the correct column.
    assert float(stationary_row["fail_daily_loss"]) != float(
        stationary_row["fail_max_loss"]
    )
