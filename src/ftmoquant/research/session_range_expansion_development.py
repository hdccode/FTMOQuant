"""G1.4D DEVELOPMENT-only wiring for ``session_range_expansion_v1``.

This is intentionally a thin candidate adapter over the frozen Stage G
evaluator. Native execution, costs, portfolio limits, statistics, and
provenance are owned by :mod:`ts_momentum_development`.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from ftmoquant.research.session_range_expansion_spec import (
    SESSION_RANGE_EXPANSION_CONFIG_SHA256,
    load_session_range_expansion_spec,
    session_range_expansion_config_sha256,
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
from ftmoquant.strategies.session_range_expansion import (
    SessionRangeExpansionDevelopmentFold,
)

EVALUATOR_VERSION = "g1.4d-session-range-expansion-development-evaluator-1"
RESULT_SCHEMA = "ftmoquant.session-range-expansion-development-results"


def evaluate_session_range_expansion_development(
    *,
    spec_path: Path,
    universe_readiness_path: Path,
    development_roots: Mapping[str, Path],
    cost_models_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Evaluate exactly one frozen session-range candidate on DEVELOPMENT."""

    spec = load_session_range_expansion_spec(spec_path)
    if (
        session_range_expansion_config_sha256(spec)
        != SESSION_RANGE_EXPANSION_CONFIG_SHA256
    ):
        raise TsMomentumEvaluationError(
            "session_range_expansion_v1 semantic SHA drifted"
        )
    return evaluate_frozen_development_candidate(
        strategy_id=spec.strategy_id,
        strategy_config_sha256=SESSION_RANGE_EXPANSION_CONFIG_SHA256,
        spec_path=spec_path,
        universe_readiness_path=universe_readiness_path,
        development_roots=development_roots,
        cost_models_path=cost_models_path,
        output_dir=output_dir,
        target_instruction_builder=_build_target_instructions,
        evaluator_version=EVALUATOR_VERSION,
        result_schema=RESULT_SCHEMA,
        implementation_paths=(
            Path("src/ftmoquant/strategies/session_range_expansion.py"),
            Path("src/ftmoquant/research/session_range_expansion_development.py"),
        ),
    )


def _build_target_instructions(
    context: DevelopmentResearchContext,
    fold: DevelopmentFold,
    frames: tuple[SynchronizedClockFrame, ...],
    spec_path: Path,
) -> tuple[TargetInstruction, ...]:
    """Adapt only session raw targets; execution remains in the shared runner."""

    candidate = SessionRangeExpansionDevelopmentFold(
        context, fold, load_session_range_expansion_spec(spec_path)
    )
    instructions: list[TargetInstruction] = []
    for frame in frames:
        candidate.on_frame(frame, partition=ResearchPartition.DEVELOPMENT)
        executable = (
            candidate.on_execution_frame(frame, partition=ResearchPartition.DEVELOPMENT)
            if _has_synchronized_execution_prices(frame)
            else ()
        )
        if executable:
            instructions.extend(_target_instructions_at_frame(executable, frame))
    return tuple(instructions)


def _has_synchronized_execution_prices(frame: SynchronizedClockFrame) -> bool:
    """Admit only a complete, valid Stage G frame to native target release."""

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
            "Evaluate frozen session_range_expansion_v1 on Stage G DEVELOPMENT only"
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
    evaluate_session_range_expansion_development(
        spec_path=cast(Path, args.spec),
        universe_readiness_path=cast(Path, args.universe_readiness),
        development_roots=roots,
        cost_models_path=cast(Path, args.cost_models),
        output_dir=cast(Path, args.output),
    )


if __name__ == "__main__":
    main()
