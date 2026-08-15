"""DEVELOPMENT-only Stage G adapter for frozen ``leo_gbpusd_v1``.

The common evaluator owns catalog access, G0.7 execution/costs, folds,
statistics, and result artifacts. This adapter only builds complete GBP/USD
15-minute bid/ask bars and converts frozen state transitions to its target API.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from ftmoquant.backtest.execution_harness import (
    _instrument_for_profile,
    canonical_execution_profile,
)
from ftmoquant.data.instruments import GBPUSD_SPEC
from ftmoquant.research.leo_gbpusd_cache import (
    cache_rows_to_native_bars,
    load_leo_gbpusd_15m_cache,
    prepare_leo_gbpusd_15m_cache,
)
from ftmoquant.research.leo_gbpusd_spec import (
    LEO_GBPUSD_CONFIG_SHA256,
    leo_gbpusd_config_sha256,
    load_leo_gbpusd_spec,
)
from ftmoquant.research.stage_g import DevelopmentFold, DevelopmentResearchContext
from ftmoquant.research.ts_momentum_development import (
    PreparedDevelopmentMarketData,
    TargetInstruction,
    TargetInstructionBuilder,
    TsMomentumEvaluationError,
    _development_root,
    _semantic_sha256,
    evaluate_frozen_development_candidate,
)
from ftmoquant.strategies.leo_gbpusd import (
    LeoAction,
    LeoCompleted15mBar,
    LeoEntry,
    LeoExit,
    LeoExitReason,
    LeoGbpUsdStateMachine,
)
from ftmoquant.strategies.trend_pullback import Direction
from ftmoquant.strategies.ts_momentum import RawDirectionalTarget

EVALUATOR_VERSION = "g1.4f-leo-gbpusd-development-evaluator-1"
RESULT_SCHEMA = "ftmoquant.leo-gbpusd-development-results"
_GBPUSD = "GBP/USD.DUKASCOPY"


def evaluate_leo_gbpusd_development(
    *,
    spec_path: Path,
    universe_readiness_path: Path,
    development_roots: Mapping[str, Path],
    cost_models_path: Path,
    output_dir: Path,
    cache_dir: Path,
) -> dict[str, Any]:
    """Evaluate the exact frozen Leo candidate through the shared G0.7 runner."""
    spec = load_leo_gbpusd_spec(spec_path)
    if leo_gbpusd_config_sha256(spec) != LEO_GBPUSD_CONFIG_SHA256:
        raise TsMomentumEvaluationError("leo_gbpusd_v1 semantic SHA drifted")
    state_diagnostics: dict[str, dict[str, int]] = {}

    def build(
        context: DevelopmentResearchContext,
        fold: DevelopmentFold,
        frames: tuple[LeoCompleted15mBar, ...],
        path: Path,
    ) -> tuple[TargetInstruction, ...]:
        return _build_target_instructions(
            context, fold, frames, path, state_diagnostics
        )

    def load_cache(
        context: DevelopmentResearchContext, roots: Mapping[str, Path]
    ) -> PreparedDevelopmentMarketData:
        rows = load_leo_gbpusd_15m_cache(
            cache_dir=cache_dir, context=context, development_roots=roots
        )
        return PreparedDevelopmentMarketData(
            (
                _instrument_for_profile(
                    GBPUSD_SPEC.nautilus_instrument(), canonical_execution_profile().fee
                ),
            ),
            cache_rows_to_native_bars(rows),
            cast(tuple[Any, ...], rows),
            0,
        )

    manifest = evaluate_frozen_development_candidate(
        strategy_id=spec.strategy_id,
        strategy_config_sha256=LEO_GBPUSD_CONFIG_SHA256,
        spec_path=spec_path,
        universe_readiness_path=universe_readiness_path,
        development_roots=development_roots,
        cost_models_path=cost_models_path,
        output_dir=output_dir,
        target_instruction_builder=cast(TargetInstructionBuilder, build),
        evaluator_version=EVALUATOR_VERSION,
        result_schema=RESULT_SCHEMA,
        implementation_paths=(
            Path("src/ftmoquant/strategies/leo_gbpusd.py"),
            Path("src/ftmoquant/research/leo_gbpusd_development.py"),
            Path("src/ftmoquant/research/leo_gbpusd_cache.py"),
        ),
        prepared_market_data_loader=load_cache,
        bar_interval_minutes=15,
        instruction_timestamp="init",
    )
    if output_dir.is_dir():
        manifest["leo_state_diagnostics"] = state_diagnostics
        manifest["semantic_sha256"] = _semantic_sha256(
            {key: value for key, value in manifest.items() if key != "semantic_sha256"}
        )
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return manifest


def _build_target_instructions(
    _context: DevelopmentResearchContext,
    fold: DevelopmentFold,
    frames: tuple[LeoCompleted15mBar, ...],
    spec_path: Path,
    diagnostics: dict[str, dict[str, int]] | None = None,
) -> tuple[TargetInstruction, ...]:
    """Construct only causal 15m bars; a fresh state is used for each fold."""
    machine = LeoGbpUsdStateMachine(load_leo_gbpusd_spec(spec_path))
    instructions: list[TargetInstruction] = []
    entry_counts = {"london": 0, "new_york": 0}
    exit_count = 0
    for bar in frames:
        if bar.end_time_utc < fold.compare_start_utc:
            continue
        for action in machine.on_bar(bar):
            if isinstance(action, LeoEntry):
                entry_counts[action.named_session.value] += 1
            elif isinstance(action, LeoExit):
                exit_count += 1
            instruction = _instruction_from_action(action, bar)
            if instruction is not None:
                instructions.append(instruction)
    if diagnostics is not None:
        diagnostics[fold.fold_id] = {
            "london_entry_count": entry_counts["london"],
            "new_york_entry_count": entry_counts["new_york"],
            "exit_count": exit_count,
            "invalid_stop_distance_skipped_trade_count": (
                machine.invalid_stop_distance_skip_count
            ),
        }
    return tuple(instructions)


def _instruction_from_action(
    action: LeoAction, bar: LeoCompleted15mBar
) -> TargetInstruction | None:
    if not isinstance(action, (LeoEntry, LeoExit)):
        return None
    target = (
        RawDirectionalTarget.LONG
        if isinstance(action, LeoEntry) and action.direction is Direction.LONG
        else RawDirectionalTarget.SHORT
        if isinstance(action, LeoEntry)
        else RawDirectionalTarget.FLAT
    )
    return TargetInstruction(
        instrument_id=_GBPUSD,
        target=target,
        event_time_ns=_ns(bar.available_at_utc),
        information_time_ns=_ns(bar.available_at_utc),
        midpoint=bar.midpoint.close,
        frame_midpoints=((_GBPUSD, bar.midpoint.close),),
        execution_price=(
            action.exit_price
            if isinstance(action, LeoExit)
            and action.reason in {LeoExitReason.STOP_LOSS, LeoExitReason.TAKE_PROFIT}
            else None
        ),
        exit_reason=action.reason.value if isinstance(action, LeoExit) else None,
    )


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise TsMomentumEvaluationError("Stage G timestamps must be UTC")
    return value.astimezone(UTC)


def _ns(value: datetime) -> int:
    return int(_utc(value).timestamp() * 1_000_000_000)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen leo_gbpusd_v1 on Stage G DEVELOPMENT only"
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--universe-readiness", type=Path, required=True)
    parser.add_argument(
        "--development-root", type=_development_root, action="append", required=True
    )
    parser.add_argument("--cost-models", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def build_cache_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare immutable GBP/USD 15m cache for frozen Leo DEVELOPMENT"
    )
    parser.add_argument("--universe-readiness", type=Path, required=True)
    parser.add_argument(
        "--development-root", type=_development_root, action="append", required=True
    )
    parser.add_argument("--cache", type=Path, required=True)
    return parser


def prepare_cache_main(argv: Sequence[str] | None = None) -> None:
    args = build_cache_parser().parse_args(argv)
    supplied = cast(list[tuple[str, Path]], args.development_root)
    roots = dict(supplied)
    if len(roots) != len(supplied):
        raise TsMomentumEvaluationError("duplicate development root instrument")
    prepare_leo_gbpusd_15m_cache(
        universe_readiness_path=cast(Path, args.universe_readiness),
        development_roots=roots,
        cache_dir=cast(Path, args.cache),
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    supplied = cast(list[tuple[str, Path]], args.development_root)
    roots = dict(supplied)
    if len(roots) != len(supplied):
        raise TsMomentumEvaluationError("duplicate development root instrument")
    evaluate_leo_gbpusd_development(
        spec_path=cast(Path, args.spec),
        universe_readiness_path=cast(Path, args.universe_readiness),
        development_roots=roots,
        cost_models_path=cast(Path, args.cost_models),
        output_dir=cast(Path, args.output),
        cache_dir=cast(Path, args.cache),
    )


if __name__ == "__main__":
    main()
