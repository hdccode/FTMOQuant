"""One-shot provider-aware DEVELOPMENT runner for frozen Batch 5B/B5C only."""

from __future__ import annotations

import argparse
from pathlib import Path

from ftmoquant.research.alpha_lab.batch5_bc_corrected_development_orchestrator import (
    Batch5BCCorrectedOrchestratorError,
    _sha256_file,
    run_corrected_development,
)
from ftmoquant.research.alpha_lab.batch5_bc_fx_day_amendment import (
    PRIOR_EXACT_BOUNDARY_ARTIFACT_HASH_MANIFEST_SHA256,
    verify_fx_day_amendment,
)
from ftmoquant.research.alpha_lab.batch5_daily import (
    build_provider_aware_fx_days_with_diagnostics,
)

ARTIFACT_ROOT = Path(".artifacts/alpha_lab/batch5_bc_provider_aware_development_v1")
PRIOR_ARTIFACT_HASH_MANIFEST = Path(
    ".artifacts/alpha_lab/batch5_bc_corrected_development_v1/artifact_hashes.json"
)
STAGE = "B5_BC_provider_aware_corrected_DEVELOPMENT_v1"


def run_provider_aware_development(
    *,
    development_root: Path,
    universe_readiness: Path,
    batch5_cross_root: Path,
    output_dir: Path,
) -> None:
    """Run the single amended B5B/B5C DEVELOPMENT evaluation."""

    if output_dir.resolve() != ARTIFACT_ROOT.resolve():
        raise Batch5BCCorrectedOrchestratorError(
            f"provider-aware output is frozen at {ARTIFACT_ROOT}"
        )
    if (
        not PRIOR_ARTIFACT_HASH_MANIFEST.is_file()
        or _sha256_file(PRIOR_ARTIFACT_HASH_MANIFEST)
        != PRIOR_EXACT_BOUNDARY_ARTIFACT_HASH_MANIFEST_SHA256
    ):
        raise Batch5BCCorrectedOrchestratorError(
            "prior exact-boundary corrected artifact identity drift"
        )
    amendment = verify_fx_day_amendment()
    run_corrected_development(
        development_root=development_root,
        universe_readiness=universe_readiness,
        batch5_cross_root=batch5_cross_root,
        output_dir=output_dir,
        daily_builder=build_provider_aware_fx_days_with_diagnostics,
        stage=STAGE,
        methodology_amendment=amendment,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the one-shot provider-aware Batch 5B/B5C DEVELOPMENT."
    )
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--universe-readiness", type=Path, required=True)
    parser.add_argument("--batch5-cross-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ARTIFACT_ROOT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_provider_aware_development(
        development_root=args.development_root,
        universe_readiness=args.universe_readiness,
        batch5_cross_root=args.batch5_cross_root,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
