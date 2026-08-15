"""DEVELOPMENT-only adapter for frozen ``liquidity_shock_reversion_v1``.

The shared Stage G evaluator owns catalog access, execution, costs, portfolio
limits, statistics, and provenance.  This module only turns minute targets into
its native instruction shape.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from ftmoquant.research.liquidity_shock_reversion_spec import (
    LIQUIDITY_SHOCK_REVERSION_CONFIG_SHA256,
    liquidity_shock_reversion_config_sha256,
    load_liquidity_shock_reversion_spec,
)
from ftmoquant.research.stage_g import (
    FROZEN_INSTRUMENT_IDS,
    DevelopmentFold,
    DevelopmentResearchContext,
    ResearchPartition,
    StageGValidationError,
    SynchronizedClockFrame,
)
from ftmoquant.research.ts_momentum_development import (
    TargetInstruction,
    TsMomentumEvaluationError,
    _development_root,
    _target_instructions_at_frame,
    evaluate_frozen_development_candidate,
)
from ftmoquant.strategies.liquidity_shock_reversion import (
    LiquidityShockReversionDevelopmentFold,
    derive_minute_midpoint_closes,
)

EVALUATOR_VERSION = "g1.4e-liquidity-shock-reversion-development-evaluator-1"
RESULT_SCHEMA = "ftmoquant.liquidity-shock-reversion-development-results"


def evaluate_liquidity_shock_reversion_development(
    *,
    spec_path: Path,
    universe_readiness_path: Path,
    development_roots: Mapping[str, Path],
    cost_models_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Evaluate only the frozen liquidity-shock candidate on DEVELOPMENT."""
    spec = load_liquidity_shock_reversion_spec(spec_path)
    if (
        liquidity_shock_reversion_config_sha256(spec)
        != LIQUIDITY_SHOCK_REVERSION_CONFIG_SHA256
    ):
        raise TsMomentumEvaluationError(
            "liquidity_shock_reversion_v1 semantic SHA drifted"
        )
    return evaluate_frozen_development_candidate(
        strategy_id=spec.strategy_id,
        strategy_config_sha256=LIQUIDITY_SHOCK_REVERSION_CONFIG_SHA256,
        spec_path=spec_path,
        universe_readiness_path=universe_readiness_path,
        development_roots=development_roots,
        cost_models_path=cost_models_path,
        output_dir=output_dir,
        target_instruction_builder=_build_target_instructions,
        evaluator_version=EVALUATOR_VERSION,
        result_schema=RESULT_SCHEMA,
        implementation_paths=(
            Path("src/ftmoquant/strategies/liquidity_shock_reversion.py"),
            Path("src/ftmoquant/research/liquidity_shock_reversion_development.py"),
        ),
    )


def _build_target_instructions(
    context: DevelopmentResearchContext,
    fold: DevelopmentFold,
    frames: tuple[SynchronizedClockFrame, ...],
    spec_path: Path,
) -> tuple[TargetInstruction, ...]:
    """Feed complete frames to the candidate; retain pending targets over gaps."""
    candidate = LiquidityShockReversionDevelopmentFold(
        context, fold, load_liquidity_shock_reversion_spec(spec_path)
    )
    instructions: list[TargetInstruction] = []
    for frame in frames:
        if not _has_synchronized_execution_prices(frame):
            continue
        for close in derive_minute_midpoint_closes((frame,)):
            candidate.on_minute_close(close, partition=ResearchPartition.DEVELOPMENT)
        executable = candidate.on_execution_frame(
            frame, partition=ResearchPartition.DEVELOPMENT
        )
        if executable:
            instructions.extend(_target_instructions_at_frame(executable, frame))
    return tuple(instructions)


def _has_synchronized_execution_prices(frame: SynchronizedClockFrame) -> bool:
    """Use only complete valid Stage G frames; gaps never clear pending targets."""
    if not frame.tradable or len(frame.observations) != len(FROZEN_INSTRUMENT_IDS):
        return False
    for instrument_id, observation in zip(
        FROZEN_INSTRUMENT_IDS, frame.observations, strict=True
    ):
        if observation is None or observation.instrument_id != instrument_id:
            return False
        try:
            observation.validate()
        except StageGValidationError:
            return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen liquidity_shock_reversion_v1 on Stage G DEVELOPMENT only"
        )
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--universe-readiness", type=Path, required=True)
    parser.add_argument(
        "--development-root", type=_development_root, action="append", required=True
    )
    parser.add_argument("--cost-models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    roots = dict(cast(list[tuple[str, Path]], args.development_root))
    if len(roots) != len(cast(list[tuple[str, Path]], args.development_root)):
        raise TsMomentumEvaluationError("duplicate development root instrument")
    evaluate_liquidity_shock_reversion_development(
        spec_path=cast(Path, args.spec),
        universe_readiness_path=cast(Path, args.universe_readiness),
        development_roots=roots,
        cost_models_path=cast(Path, args.cost_models),
        output_dir=cast(Path, args.output),
    )


if __name__ == "__main__":
    main()
