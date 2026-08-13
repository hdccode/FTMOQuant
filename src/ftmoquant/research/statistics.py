"""Deterministic thin wrappers around arch 8.0 statistical procedures.

This module contains no strategy, ranking, selection, annualization, or
holdout policy. Inputs are validated without sorting, aligning, or dropping.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from importlib.metadata import version
from typing import Any, Literal, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from arch.bootstrap import MCS, SPA, RealityCheck, StationaryBootstrap
from arch.bootstrap import optimal_block_length as arch_optimal_block_length

ARCH_VERSION = "8.0.0"
LOSS_ORIENTATION = "lower_loss_is_better"

ConfidenceIntervalMethod = Literal[
    "basic", "percentile", "studentized", "norm", "bc", "bca"
]
ConfidenceIntervalTail = Literal["two", "upper", "lower"]
BootstrapSampling = Literal[
    "nonparametric", "semi-parametric", "semi", "parametric", "semiparametric"
]
BootstrapType = Literal[
    "stationary", "sb", "circular", "cbb", "moving block", "mbb"
]
SPAPValueType = Literal["lower", "consistent", "upper"]
MCSMethod = Literal["R", "max"]
MultipleComparisonProcedure = Literal["SPA", "Reality Check"]


class ResearchStatisticsValidationError(ValueError):
    """Raised when inputs would make a statistical result ambiguous."""


@dataclass(frozen=True, slots=True)
class StationaryBootstrapConfig:
    """Complete configuration for an arch stationary-bootstrap mean CI."""

    block_size: int
    repetitions: int
    seed: int
    confidence_level: float
    method: ConfidenceIntervalMethod
    tail: ConfidenceIntervalTail = "two"
    sampling: BootstrapSampling = "nonparametric"
    studentize_repetitions: int = 1000


@dataclass(frozen=True, slots=True)
class BlockLengthConfig:
    """Explicit marker that native estimates are returned, not selected."""

    apply_estimate: Literal[False] = False


@dataclass(frozen=True, slots=True)
class SPAConfig:
    """Complete native SPA or Reality Check configuration."""

    procedure: MultipleComparisonProcedure
    block_size: int
    repetitions: int
    seed: int
    bootstrap: BootstrapType
    studentize: bool
    nested: bool
    significance_level: float
    pvalue_type: SPAPValueType


@dataclass(frozen=True, slots=True)
class MCSConfig:
    """Complete native Model Confidence Set configuration."""

    test_size: float
    block_size: int
    repetitions: int
    seed: int
    bootstrap: BootstrapType
    method: MCSMethod


@dataclass(frozen=True, slots=True)
class StationaryBootstrapResult:
    """Immutable sample-mean confidence interval and provenance."""

    arch_version: str
    procedure: str
    config: StationaryBootstrapConfig
    seed: int
    observation_count: int
    series_label: str
    input_content_sha256: str
    estimate: float
    lower_bound: float
    upper_bound: float


@dataclass(frozen=True, slots=True)
class BlockLengthResult:
    """Native arch block-length estimates; never applied automatically."""

    arch_version: str
    procedure: str
    config: BlockLengthConfig
    seed: None
    observation_count: int
    series_label: str
    input_content_sha256: str
    stationary: float
    circular: float


@dataclass(frozen=True, slots=True)
class SPAResult:
    """Native SPA/Reality Check output with original model identities."""

    arch_version: str
    procedure: MultipleComparisonProcedure
    config: SPAConfig
    seed: int
    observation_count: int
    loss_orientation: str
    benchmark_label: str
    model_labels: tuple[str, ...]
    input_content_sha256: str
    pvalues: tuple[tuple[str, float], ...]
    superior_models: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MCSResult:
    """Native MCS membership and p-values without winner selection."""

    arch_version: str
    procedure: str
    config: MCSConfig
    seed: int
    observation_count: int
    loss_orientation: str
    model_labels: tuple[str, ...]
    input_content_sha256: str
    included_models: tuple[str, ...]
    excluded_models: tuple[str, ...]
    pvalues: tuple[tuple[str, float], ...]


def stationary_bootstrap_confidence_interval(
    observations: pd.Series, config: StationaryBootstrapConfig
) -> StationaryBootstrapResult:
    """Compute an arch stationary-bootstrap confidence interval for the mean."""

    _require_arch_version()
    series = _validate_series(observations, "observations")
    _validate_resampling(
        config.block_size, config.repetitions, config.seed, len(series)
    )
    if not 0.0 < config.confidence_level < 1.0:
        raise ResearchStatisticsValidationError(
            "confidence_level must be strictly between 0 and 1"
        )
    if config.studentize_repetitions <= 0:
        raise ResearchStatisticsValidationError(
            "studentize_repetitions must be positive"
        )
    bootstrap = StationaryBootstrap(config.block_size, series, seed=config.seed)
    interval = bootstrap.conf_int(
        _sample_mean,
        reps=config.repetitions,
        method=config.method,
        size=config.confidence_level,
        tail=config.tail,
        sampling=config.sampling,
        studentize_reps=config.studentize_repetitions,
    )
    return StationaryBootstrapResult(
        arch_version=ARCH_VERSION,
        procedure="StationaryBootstrap mean confidence interval",
        config=config,
        seed=config.seed,
        observation_count=len(series),
        series_label=cast(str, series.name),
        input_content_sha256=_hash_series(series),
        estimate=float(series.mean()),
        lower_bound=float(interval[0, 0]),
        upper_bound=float(interval[1, 0]),
    )


def estimate_optimal_block_length(observations: pd.Series) -> BlockLengthResult:
    """Return arch block-length estimates without using either estimate."""

    _require_arch_version()
    series = _validate_series(observations, "observations")
    if len(series) < 2:
        raise ResearchStatisticsValidationError(
            "block-length estimation requires at least two observations"
        )
    estimates = arch_optimal_block_length(series)
    row = estimates.iloc[0]
    return BlockLengthResult(
        arch_version=ARCH_VERSION,
        procedure="optimal_block_length",
        config=BlockLengthConfig(),
        seed=None,
        observation_count=len(series),
        series_label=cast(str, series.name),
        input_content_sha256=_hash_series(series),
        stationary=float(row["stationary"]),
        circular=float(row["circular"]),
    )


def spa_reality_check(
    benchmark_losses: pd.Series,
    model_losses: pd.DataFrame,
    config: SPAConfig,
) -> SPAResult:
    """Run native arch SPA or Reality Check on explicit aligned losses."""

    _require_arch_version()
    benchmark = _validate_series(benchmark_losses, "benchmark losses")
    models = _validate_loss_frame(model_losses)
    _require_aligned_index(benchmark.index, models.index)
    _validate_resampling(
        config.block_size, config.repetitions, config.seed, len(models)
    )
    if not 0.0 < config.significance_level < 1.0:
        raise ResearchStatisticsValidationError(
            "significance_level must be strictly between 0 and 1"
        )
    procedure_type = SPA if config.procedure == "SPA" else RealityCheck
    procedure = procedure_type(
        benchmark,
        models,
        block_size=config.block_size,
        reps=config.repetitions,
        bootstrap=config.bootstrap,
        studentize=config.studentize,
        nested=config.nested,
        seed=config.seed,
    )
    procedure.compute()
    pvalues = tuple(
        (str(label), float(value)) for label, value in procedure.pvalues.items()
    )
    superior_positions = procedure.better_models(
        pvalue=config.significance_level,
        pvalue_type=config.pvalue_type,
    )
    labels = tuple(cast(str, label) for label in models.columns)
    superior = tuple(labels[int(position)] for position in superior_positions)
    return SPAResult(
        arch_version=ARCH_VERSION,
        procedure=config.procedure,
        config=config,
        seed=config.seed,
        observation_count=len(models),
        loss_orientation=LOSS_ORIENTATION,
        benchmark_label=cast(str, benchmark.name),
        model_labels=labels,
        input_content_sha256=_hash_spa_inputs(benchmark, models),
        pvalues=pvalues,
        superior_models=superior,
    )


def model_confidence_set(losses: pd.DataFrame, config: MCSConfig) -> MCSResult:
    """Run native arch MCS and return membership without model selection."""

    _require_arch_version()
    frame = _validate_loss_frame(losses)
    _validate_resampling(config.block_size, config.repetitions, config.seed, len(frame))
    if not 0.0 < config.test_size < 1.0:
        raise ResearchStatisticsValidationError(
            "test_size must be strictly between 0 and 1"
        )
    procedure = MCS(
        frame,
        size=config.test_size,
        reps=config.repetitions,
        block_size=config.block_size,
        method=config.method,
        bootstrap=config.bootstrap,
        seed=config.seed,
    )
    procedure.compute()
    pvalues = tuple(
        (cast(str, label), float(row["Pvalue"]))
        for label, row in procedure.pvalues.iterrows()
    )
    labels = tuple(cast(str, label) for label in frame.columns)
    return MCSResult(
        arch_version=ARCH_VERSION,
        procedure="MCS",
        config=config,
        seed=config.seed,
        observation_count=len(frame),
        loss_orientation=LOSS_ORIENTATION,
        model_labels=labels,
        input_content_sha256=_hash_frame(frame),
        included_models=tuple(cast(str, label) for label in procedure.included),
        excluded_models=tuple(cast(str, label) for label in procedure.excluded),
        pvalues=pvalues,
    )


def result_as_dict(result: Any) -> dict[str, Any]:
    """Convert an immutable result to stable JSON-compatible metadata."""

    if not hasattr(result, "__dataclass_fields__"):
        raise TypeError("result must be a G0.8 result dataclass")
    return asdict(result)


def _sample_mean(values: pd.Series) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    return np.asarray([values.mean()], dtype=np.float64)


def _validate_series(value: pd.Series, label: str) -> pd.Series:
    if not isinstance(value, pd.Series):
        raise ResearchStatisticsValidationError(f"{label} must be a pandas Series")
    if not isinstance(value.name, str) or not value.name:
        raise ResearchStatisticsValidationError(f"{label} must have a string label")
    if not value.index.is_unique:
        raise ResearchStatisticsValidationError(f"{label} index must be unique")
    try:
        numeric = value.to_numpy(dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise ResearchStatisticsValidationError(f"{label} must be numeric") from error
    if numeric.ndim != 1 or len(numeric) == 0:
        raise ResearchStatisticsValidationError(f"{label} must not be empty")
    if not np.isfinite(numeric).all():
        raise ResearchStatisticsValidationError(
            f"{label} contains NaN or infinite values"
        )
    return pd.Series(numeric, index=value.index.copy(), name=value.name, copy=False)


def _validate_loss_frame(value: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise ResearchStatisticsValidationError(
            "model losses must be a pandas DataFrame"
        )
    if value.empty or len(value.columns) == 0:
        raise ResearchStatisticsValidationError("model losses must not be empty")
    if not value.index.is_unique:
        raise ResearchStatisticsValidationError("model-loss index must be unique")
    if not value.columns.is_unique:
        raise ResearchStatisticsValidationError("duplicate model labels are invalid")
    if any(not isinstance(label, str) or not label for label in value.columns):
        raise ResearchStatisticsValidationError(
            "all model labels must be non-empty strings"
        )
    try:
        numeric = value.to_numpy(dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise ResearchStatisticsValidationError(
            "model losses must be numeric"
        ) from error
    if not np.isfinite(numeric).all():
        raise ResearchStatisticsValidationError(
            "model losses contain NaN or infinite values"
        )
    return pd.DataFrame(
        numeric,
        index=value.index.copy(),
        columns=value.columns.copy(),
        copy=False,
    )


def _require_aligned_index(left: pd.Index, right: pd.Index) -> None:
    if not left.equals(right):
        raise ResearchStatisticsValidationError(
            "benchmark and model losses must have identical aligned indexes"
        )


def _validate_resampling(
    block_size: int, repetitions: int, seed: int, observation_count: int
) -> None:
    if isinstance(block_size, bool) or not isinstance(block_size, int):
        raise ResearchStatisticsValidationError("block_size must be an integer")
    if block_size <= 0 or block_size > observation_count:
        raise ResearchStatisticsValidationError(
            "block_size must be positive and no larger than observation count"
        )
    if isinstance(repetitions, bool) or not isinstance(repetitions, int):
        raise ResearchStatisticsValidationError("repetitions must be an integer")
    if repetitions <= 0:
        raise ResearchStatisticsValidationError("repetitions must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ResearchStatisticsValidationError("seed must be an explicit integer")


def _hash_series(series: pd.Series) -> str:
    return _sha256_json(
        {
            "kind": "series",
            "label": series.name,
            "index": [_canonical_scalar(value) for value in series.index],
            "values": [float(value).hex() for value in series.to_numpy()],
        }
    )


def _hash_frame(frame: pd.DataFrame) -> str:
    return _sha256_json(
        {
            "kind": "frame",
            "labels": list(frame.columns),
            "index": [_canonical_scalar(value) for value in frame.index],
            "values": [
                [float(value).hex() for value in row]
                for row in frame.to_numpy(dtype=np.float64)
            ],
        }
    )


def _hash_spa_inputs(benchmark: pd.Series, models: pd.DataFrame) -> str:
    return _sha256_json(
        {
            "benchmark": _hash_payload_series(benchmark),
            "models": _hash_payload_frame(models),
        }
    )


def _hash_payload_series(series: pd.Series) -> dict[str, Any]:
    return {
        "label": series.name,
        "index": [_canonical_scalar(value) for value in series.index],
        "values": [float(value).hex() for value in series.to_numpy()],
    }


def _hash_payload_frame(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "labels": list(frame.columns),
        "index": [_canonical_scalar(value) for value in frame.index],
        "values": [
            [float(value).hex() for value in row]
            for row in frame.to_numpy(dtype=np.float64)
        ],
    }


def _canonical_scalar(value: Any) -> dict[str, str]:
    if isinstance(value, float):
        text = value.hex()
    elif isinstance(value, pd.Timestamp):
        text = value.isoformat()
    else:
        text = str(value)
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "value": text,
    }


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_arch_version() -> None:
    installed = version("arch")
    if installed != ARCH_VERSION:
        raise RuntimeError(f"expected arch {ARCH_VERSION}, found {installed}")
