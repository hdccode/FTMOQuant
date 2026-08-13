"""Isolated research support with no strategy-selection policy."""

from ftmoquant.research.experiment_spec import (
    REGISTRY_COLUMNS,
    ExperimentRegistryEntry,
    ExperimentRegistryValidationError,
    StrategySpecValidationError,
    TrendPullbackSpec,
    canonical_spec_json,
    load_experiment_registry,
    load_trend_pullback_spec,
    strategy_config_sha256,
)

__all__ = [
    "REGISTRY_COLUMNS",
    "ExperimentRegistryEntry",
    "ExperimentRegistryValidationError",
    "StrategySpecValidationError",
    "TrendPullbackSpec",
    "canonical_spec_json",
    "load_experiment_registry",
    "load_trend_pullback_spec",
    "strategy_config_sha256",
]
