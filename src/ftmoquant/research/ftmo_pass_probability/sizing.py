"""Causal sizing policies applied on top of the frozen, immutable alpha.

Exactly eight candidates, frozen before any Monte Carlo screen is run (task
§8): four fixed-fractional-risk levels and four fixed-notional multipliers
of the frozen research notional (``BASE_RESEARCH_UNITS = 100,000``, see
``path_extraction.py``). No Kelly, no martingale, no loss-recovery/streak
sizing, no post-hoc grid expansion.

Every function here uses only information available at trade entry: the
current simulated account balance and the frozen, pre-computed stop
distance for that trade (``TradeRecord.usd_risk_per_unit``). Neither the
eventual realized R-multiple nor any future price is used to decide size —
it is used only afterward, to scale the already-realized-at-1x outcome,
which is mathematically identical to sizing before the fact (see
``docs/research/ftmo_pass_probability_v1_design.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from enum import StrEnum

from ftmoquant.data.instruments import USDCAD_OANDA_SPEC
from ftmoquant.research.ftmo_pass_probability.path_extraction import TradeRecord

ZERO = Decimal("0")
_SIZE_INCREMENT = Decimal(USDCAD_OANDA_SPEC.size_increment)


class SizingFamily(StrEnum):
    """The two permitted causal sizing families."""

    FIXED_FRACTIONAL_RISK = "fixed_fractional_risk"
    FIXED_NOTIONAL_MULTIPLIER = "fixed_notional_multiplier"


@dataclass(frozen=True, slots=True)
class SizingPolicy:
    """One frozen, named sizing candidate."""

    policy_id: str
    family: SizingFamily
    risk_fraction: Decimal | None = None
    notional_multiplier: Decimal | None = None

    def __post_init__(self) -> None:
        if self.family is SizingFamily.FIXED_FRACTIONAL_RISK:
            if self.risk_fraction is None or self.notional_multiplier is not None:
                raise ValueError(
                    "fixed_fractional_risk policies require risk_fraction only"
                )
            if not (Decimal("0") < self.risk_fraction < Decimal("1")):
                raise ValueError("risk_fraction must be strictly between 0 and 1")
        else:
            if self.notional_multiplier is None or self.risk_fraction is not None:
                raise ValueError(
                    "fixed_notional_multiplier policies require "
                    "notional_multiplier only"
                )
            if self.notional_multiplier <= ZERO:
                raise ValueError("notional_multiplier must be positive")


@dataclass(frozen=True, slots=True)
class SizedTrade:
    """The result of applying one sizing policy to one trade, causally."""

    quantity: Decimal
    risk_budget: Decimal
    realized_pnl: Decimal
    floor_equity_delta: Decimal


#: Frozen before any Monte Carlo screen is run. Do not expand after results.
SIZING_GRID: tuple[SizingPolicy, ...] = (
    SizingPolicy(
        "fixed_fractional_0_25pct",
        SizingFamily.FIXED_FRACTIONAL_RISK,
        risk_fraction=Decimal("0.0025"),
    ),
    SizingPolicy(
        "fixed_fractional_0_50pct",
        SizingFamily.FIXED_FRACTIONAL_RISK,
        risk_fraction=Decimal("0.0050"),
    ),
    SizingPolicy(
        "fixed_fractional_0_75pct",
        SizingFamily.FIXED_FRACTIONAL_RISK,
        risk_fraction=Decimal("0.0075"),
    ),
    SizingPolicy(
        "fixed_fractional_1_00pct",
        SizingFamily.FIXED_FRACTIONAL_RISK,
        risk_fraction=Decimal("0.0100"),
    ),
    SizingPolicy(
        "fixed_notional_0_5x",
        SizingFamily.FIXED_NOTIONAL_MULTIPLIER,
        notional_multiplier=Decimal("0.5"),
    ),
    SizingPolicy(
        "fixed_notional_1_0x",
        SizingFamily.FIXED_NOTIONAL_MULTIPLIER,
        notional_multiplier=Decimal("1.0"),
    ),
    SizingPolicy(
        "fixed_notional_1_5x",
        SizingFamily.FIXED_NOTIONAL_MULTIPLIER,
        notional_multiplier=Decimal("1.5"),
    ),
    SizingPolicy(
        "fixed_notional_2_0x",
        SizingFamily.FIXED_NOTIONAL_MULTIPLIER,
        notional_multiplier=Decimal("2.0"),
    ),
)
assert len(SIZING_GRID) == 8
assert len({policy.policy_id for policy in SIZING_GRID}) == 8


#: A separate, explicitly-labelled *local refinement* of the fixed-notional
#: family only, around the apparent optimum observed in the frozen
#: ``SIZING_GRID`` 20,000-path screen (stationary/circular
#: fixed_notional_2_0x ~0.75, fixed_notional_1_5x ~0.73). This is a distinct,
#: authorized follow-up grid -- it does not modify, extend, or replace
#: ``SIZING_GRID`` above (still exactly 8, still untouched), and it contains
#: no fixed-fractional-risk candidates and no new sizing family. Because it
#: was chosen after observing SIZING_GRID's own pass probabilities, it is a
#: form of post-hoc refinement around an empirically-observed peak: it must
#: be read as confirmatory/local-search evidence, not as an independent,
#: pre-registered candidate set -- the same statistical caveat the original
#: task's "do not expand this grid after seeing results" rule exists to
#: flag. Frozen once, before any refinement Monte Carlo run.
NOTIONAL_REFINEMENT_GRID: tuple[SizingPolicy, ...] = (
    SizingPolicy(
        "notional_refine_1_50x",
        SizingFamily.FIXED_NOTIONAL_MULTIPLIER,
        notional_multiplier=Decimal("1.50"),
    ),
    SizingPolicy(
        "notional_refine_1_65x",
        SizingFamily.FIXED_NOTIONAL_MULTIPLIER,
        notional_multiplier=Decimal("1.65"),
    ),
    SizingPolicy(
        "notional_refine_1_75x",
        SizingFamily.FIXED_NOTIONAL_MULTIPLIER,
        notional_multiplier=Decimal("1.75"),
    ),
    SizingPolicy(
        "notional_refine_1_85x",
        SizingFamily.FIXED_NOTIONAL_MULTIPLIER,
        notional_multiplier=Decimal("1.85"),
    ),
    SizingPolicy(
        "notional_refine_1_95x",
        SizingFamily.FIXED_NOTIONAL_MULTIPLIER,
        notional_multiplier=Decimal("1.95"),
    ),
    SizingPolicy(
        "notional_refine_2_05x",
        SizingFamily.FIXED_NOTIONAL_MULTIPLIER,
        notional_multiplier=Decimal("2.05"),
    ),
    SizingPolicy(
        "notional_refine_2_15x",
        SizingFamily.FIXED_NOTIONAL_MULTIPLIER,
        notional_multiplier=Decimal("2.15"),
    ),
    SizingPolicy(
        "notional_refine_2_25x",
        SizingFamily.FIXED_NOTIONAL_MULTIPLIER,
        notional_multiplier=Decimal("2.25"),
    ),
    SizingPolicy(
        "notional_refine_2_50x",
        SizingFamily.FIXED_NOTIONAL_MULTIPLIER,
        notional_multiplier=Decimal("2.50"),
    ),
)
assert len(NOTIONAL_REFINEMENT_GRID) == 9
assert len({policy.policy_id for policy in NOTIONAL_REFINEMENT_GRID}) == 9
assert all(
    policy.family is SizingFamily.FIXED_NOTIONAL_MULTIPLIER
    for policy in NOTIONAL_REFINEMENT_GRID
)
assert {policy.policy_id for policy in NOTIONAL_REFINEMENT_GRID}.isdisjoint(
    {policy.policy_id for policy in SIZING_GRID}
)


def apply_sizing(
    policy: SizingPolicy, trade: TradeRecord, balance_at_entry: Decimal
) -> SizedTrade:
    """Causally size one trade under one frozen policy.

    Only ``balance_at_entry`` (known before this trade opens) and the
    trade's own frozen, pre-computed stop distance are used to choose size.
    The eventual R-multiple then determines the dollar outcome, exactly as
    it would have live: no future information is used to decide quantity.
    """

    if not balance_at_entry.is_finite() or balance_at_entry <= ZERO:
        raise ValueError("balance_at_entry must be a positive finite Decimal")

    if policy.family is SizingFamily.FIXED_FRACTIONAL_RISK:
        assert policy.risk_fraction is not None
        target_risk_budget = balance_at_entry * policy.risk_fraction
        quantity = _quantize_quantity(target_risk_budget / trade.usd_risk_per_unit)
    else:
        assert policy.notional_multiplier is not None
        quantity = _quantize_quantity(
            policy.notional_multiplier
            * (trade.original_risk_budget / trade.usd_risk_per_unit)
        )

    risk_budget = quantity * trade.usd_risk_per_unit
    realized_pnl = trade.net_r * risk_budget
    floor_equity_delta = (
        min(realized_pnl, ZERO) if trade.exit_reason == "stop" else -risk_budget
    )
    return SizedTrade(
        quantity=quantity,
        risk_budget=risk_budget,
        realized_pnl=realized_pnl,
        floor_equity_delta=floor_equity_delta,
    )


def _quantize_quantity(raw_quantity: Decimal) -> Decimal:
    if raw_quantity <= ZERO:
        raise ValueError("computed quantity must be positive")
    quantized = raw_quantity.quantize(_SIZE_INCREMENT, rounding=ROUND_DOWN)
    if quantized <= ZERO:
        raise ValueError(
            "risk budget rounds down to zero at this instrument's minimum "
            "size increment"
        )
    return quantized
