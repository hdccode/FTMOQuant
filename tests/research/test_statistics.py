import json
from dataclasses import asdict, replace
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
from arch.bootstrap import (
    MCS,
    SPA,
    RealityCheck,
    StationaryBootstrap,
    optimal_block_length,
)

import ftmoquant.research.statistics as statistics_module
from ftmoquant.research.statistics import (
    LOSS_ORIENTATION,
    MCSConfig,
    ResearchStatisticsValidationError,
    SPAConfig,
    StationaryBootstrapConfig,
    estimate_optimal_block_length,
    model_confidence_set,
    result_as_dict,
    spa_reality_check,
    stationary_bootstrap_confidence_interval,
)


def test_stationary_bootstrap_is_deterministic_and_matches_arch() -> None:
    observations = _observations()
    config = _ci_config()

    first = stationary_bootstrap_confidence_interval(observations, config)
    second = stationary_bootstrap_confidence_interval(observations, config)
    direct = StationaryBootstrap(
        config.block_size, observations.astype(float), seed=config.seed
    ).conf_int(
        _direct_mean,
        reps=config.repetitions,
        method=config.method,
        size=config.confidence_level,
        tail=config.tail,
        sampling=config.sampling,
        studentize_reps=config.studentize_repetitions,
    )

    assert first == second
    assert first.estimate == float(observations.mean())
    assert first.lower_bound == float(direct[0, 0])
    assert first.upper_bound == float(direct[1, 0])


@pytest.mark.parametrize(
    "sampling",
    ["semi-parametric", "semi", "semiparametric", "parametric"],
)
def test_stationary_mean_ci_rejects_non_nonparametric_sampling(
    sampling: str,
) -> None:
    config = replace(_ci_config(), sampling=cast(Any, sampling))

    with pytest.raises(ResearchStatisticsValidationError, match="nonparametric"):
        stationary_bootstrap_confidence_interval(_observations(), config)


@pytest.mark.parametrize("tail", ["upper", "lower"])
def test_stationary_mean_ci_rejects_one_sided_tails(tail: str) -> None:
    config = replace(_ci_config(), tail=cast(Any, tail))

    with pytest.raises(ResearchStatisticsValidationError, match="two-sided"):
        stationary_bootstrap_confidence_interval(_observations(), config)


def test_optimal_block_length_matches_arch_without_applying_estimate() -> None:
    observations = _observations()

    result = estimate_optimal_block_length(observations)
    direct = optimal_block_length(observations.astype(float)).iloc[0]

    assert result.stationary == float(direct["stationary"])
    assert result.circular == float(direct["circular"])
    assert not hasattr(result, "selected_block_size")


@pytest.mark.parametrize(
    ("stationary", "circular", "message"),
    [
        (float("nan"), 2.0, "finite"),
        (2.0, float("inf"), "finite"),
        (0.0, 2.0, "positive"),
        (2.0, -1.0, "positive"),
    ],
)
def test_unusable_native_block_length_estimates_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    stationary: float,
    circular: float,
    message: str,
) -> None:
    estimates = pd.DataFrame(
        {"stationary": [stationary], "circular": [circular]},
        index=["returns"],
    )
    monkeypatch.setattr(
        statistics_module,
        "arch_optimal_block_length",
        lambda observations: estimates,
    )

    with pytest.raises(ResearchStatisticsValidationError, match=message):
        estimate_optimal_block_length(_observations())


def test_spa_matches_direct_arch_and_preserves_loss_orientation() -> None:
    benchmark, models = _losses()
    config = _spa_config()

    result = spa_reality_check(benchmark, models, config)
    direct = SPA(
        benchmark,
        models,
        block_size=config.block_size,
        reps=config.repetitions,
        bootstrap=config.bootstrap,
        studentize=config.studentize,
        nested=config.nested,
        seed=config.seed,
    )
    direct.compute()
    positions = direct.better_models(
        config.significance_level, config.pvalue_type
    )

    assert result.pvalues == tuple(
        (str(label), float(value)) for label, value in direct.pvalues.items()
    )
    assert result.superior_models == tuple(
        str(models.columns[int(position)]) for position in positions
    )
    assert result.loss_orientation == LOSS_ORIENTATION
    assert result.model_labels == tuple(models.columns)


def test_reality_check_uses_explicit_procedure() -> None:
    benchmark, models = _losses()
    config = SPAConfig(
        procedure="Reality Check",
        block_size=5,
        repetitions=99,
        seed=91,
        bootstrap="stationary",
        studentize=False,
        nested=False,
        significance_level=0.05,
        pvalue_type="consistent",
    )

    result = spa_reality_check(benchmark, models, config)
    direct = RealityCheck(
        benchmark,
        models,
        block_size=config.block_size,
        reps=config.repetitions,
        bootstrap=config.bootstrap,
        studentize=config.studentize,
        nested=config.nested,
        seed=config.seed,
    )
    direct.compute()

    assert result.procedure == "Reality Check"
    assert result.config == config
    assert result.pvalues == tuple(
        (str(label), float(value)) for label, value in direct.pvalues.items()
    )
    positions = direct.better_models(
        config.significance_level, config.pvalue_type
    )
    assert result.superior_models == tuple(
        str(models.columns[int(position)]) for position in positions
    )


def test_invalid_spa_procedure_is_rejected_instead_of_falling_through() -> None:
    benchmark, models = _losses()
    config = replace(_spa_config(), procedure=cast(Any, "invalid"))

    with pytest.raises(ResearchStatisticsValidationError, match="procedure"):
        spa_reality_check(benchmark, models, config)


@pytest.mark.parametrize("method", ["R", "max"])
def test_mcs_matches_direct_arch_without_selecting_winner(method: str) -> None:
    _, losses = _losses()
    config = replace(_mcs_config(), method=cast(Any, method))

    result = model_confidence_set(losses, config)
    direct = MCS(
        losses,
        size=config.test_size,
        reps=config.repetitions,
        block_size=config.block_size,
        method=config.method,
        bootstrap=config.bootstrap,
        seed=config.seed,
    )
    direct.compute()

    assert result.included_models == tuple(direct.included)
    assert result.excluded_models == tuple(direct.excluded)
    assert result.pvalues == tuple(
        (str(label), float(row["Pvalue"]))
        for label, row in direct.pvalues.iterrows()
    )
    assert not hasattr(result, "winner")


def test_invalid_mcs_method_is_rejected() -> None:
    _, losses = _losses()
    config = replace(_mcs_config(), method=cast(Any, "invalid"))

    with pytest.raises(ResearchStatisticsValidationError, match="method"):
        model_confidence_set(losses, config)


def test_one_model_mcs_is_rejected() -> None:
    _, losses = _losses()

    with pytest.raises(ResearchStatisticsValidationError, match="at least two"):
        model_confidence_set(losses[["model-a"]], _mcs_config())


def test_input_order_changes_content_hash() -> None:
    observations = _observations()

    ordered = stationary_bootstrap_confidence_interval(observations, _ci_config())
    reversed_result = stationary_bootstrap_confidence_interval(
        observations.iloc[::-1], _ci_config()
    )

    assert ordered.input_content_sha256 != reversed_result.input_content_sha256


def test_mismatched_spa_indexes_are_rejected() -> None:
    benchmark, models = _losses()
    models.index = models.index + 1

    with pytest.raises(ResearchStatisticsValidationError, match="aligned indexes"):
        spa_reality_check(benchmark, models, _spa_config())


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_values_are_rejected_without_dropping(invalid: float) -> None:
    observations = _observations()
    observations.iloc[3] = invalid
    benchmark, models = _losses()
    models.iloc[3, 0] = invalid

    with pytest.raises(ResearchStatisticsValidationError, match="NaN or infinite"):
        stationary_bootstrap_confidence_interval(observations, _ci_config())
    with pytest.raises(ResearchStatisticsValidationError, match="NaN or infinite"):
        model_confidence_set(models, _mcs_config())
    with pytest.raises(ResearchStatisticsValidationError, match="NaN or infinite"):
        spa_reality_check(benchmark, models, _spa_config())


@pytest.mark.parametrize("block_size", [0, 41])
def test_invalid_or_insufficient_block_size_is_rejected(block_size: int) -> None:
    config = StationaryBootstrapConfig(
        block_size=block_size,
        repetitions=99,
        seed=7,
        confidence_level=0.9,
        method="percentile",
    )

    with pytest.raises(ResearchStatisticsValidationError, match="block_size"):
        stationary_bootstrap_confidence_interval(_observations(), config)


def test_duplicate_model_labels_are_rejected() -> None:
    _, losses = _losses()
    losses.columns = ["duplicate", "duplicate", "model-c"]

    with pytest.raises(ResearchStatisticsValidationError, match="duplicate"):
        model_confidence_set(losses, _mcs_config())


def test_results_are_serialization_stable() -> None:
    benchmark, losses = _losses()
    results = (
        stationary_bootstrap_confidence_interval(_observations(), _ci_config()),
        estimate_optimal_block_length(_observations()),
        spa_reality_check(benchmark, losses, _spa_config()),
        model_confidence_set(losses, _mcs_config()),
    )

    for result in results:
        first = json.dumps(result_as_dict(result), sort_keys=True, allow_nan=False)
        second = json.dumps(asdict(result), sort_keys=True, allow_nan=False)
        assert first == second
        assert json.loads(first)["arch_version"] == "8.0.0"


def test_quantstats_and_empyrical_are_not_project_dependencies() -> None:
    dependencies = (
        __import__("pathlib").Path("pyproject.toml").read_text(encoding="utf-8").lower()
    )

    assert "quantstats" not in dependencies
    assert "empyrical" not in dependencies
    assert '"arch==8.0.0"' in dependencies


def _direct_mean(values: pd.Series) -> np.ndarray:
    return np.asarray([values.mean()], dtype=np.float64)


def _observations() -> pd.Series:
    values = np.sin(np.arange(40, dtype=float) / 4.0) / 100.0
    return pd.Series(values, index=pd.RangeIndex(100, 140), name="returns")


def _losses() -> tuple[pd.Series, pd.DataFrame]:
    index = pd.RangeIndex(200, 240)
    base = np.cos(np.arange(40, dtype=float) / 5.0) / 100.0
    benchmark = pd.Series(base, index=index, name="benchmark")
    models = pd.DataFrame(
        {
            "model-a": base - 0.003,
            "model-b": base + np.sin(np.arange(40) / 3.0) / 500.0,
            "model-c": base + 0.003,
        },
        index=index,
    )
    return benchmark, models


def _ci_config() -> StationaryBootstrapConfig:
    return StationaryBootstrapConfig(
        block_size=5,
        repetitions=199,
        seed=17,
        confidence_level=0.9,
        method="percentile",
        tail="two",
        sampling="nonparametric",
        studentize_repetitions=101,
    )


def _spa_config() -> SPAConfig:
    return SPAConfig(
        procedure="SPA",
        block_size=5,
        repetitions=199,
        seed=23,
        bootstrap="stationary",
        studentize=True,
        nested=False,
        significance_level=0.05,
        pvalue_type="consistent",
    )


def _mcs_config() -> MCSConfig:
    return MCSConfig(
        test_size=0.1,
        block_size=5,
        repetitions=199,
        seed=31,
        bootstrap="stationary",
        method="R",
    )
