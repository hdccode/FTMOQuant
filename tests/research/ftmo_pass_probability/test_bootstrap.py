from __future__ import annotations

from decimal import Decimal

import pytest

from ftmoquant.research.ftmo_pass_probability.bootstrap import (
    BootstrapValidationError,
    derive_frozen_block_length,
    draw_index_path,
)
from ftmoquant.research.ftmo_pass_probability.path_extraction import TradeRecord


def _synthetic_trades(count: int) -> tuple[TradeRecord, ...]:
    trades = []
    for i in range(count):
        net_r = 2.0 if i % 3 else -1.0
        trades.append(
            TradeRecord(
                trade_index=i,
                entry_ns=1_000_000_000 * i * 10,
                exit_ns=1_000_000_000 * i * 10 + 500_000_000,
                exit_reason="target" if net_r > 0 else "stop",
                net_r=Decimal(str(net_r)),
                original_realized_pnl=Decimal(str(net_r * 300)),
                original_risk_budget=Decimal("300"),
                usd_risk_per_unit=Decimal("0.003"),
            )
        )
    return tuple(trades)


def test_block_length_is_derived_only_from_the_given_trades() -> None:
    trades = _synthetic_trades(60)
    result = derive_frozen_block_length(trades)
    assert result.frozen_block_size >= 1
    assert result.observation_count == 60
    assert result.stationary_block_length > 0
    assert result.circular_block_length > 0


def test_block_length_requires_at_least_two_trades() -> None:
    with pytest.raises(Exception):
        derive_frozen_block_length(_synthetic_trades(1))


def test_seeded_resampling_is_deterministic() -> None:
    first = draw_index_path(
        60, method="stationary", block_size=5, seed=42, min_length=60
    )
    second = draw_index_path(
        60, method="stationary", block_size=5, seed=42, min_length=60
    )
    assert first == second


def test_different_seeds_generally_produce_different_paths() -> None:
    first = draw_index_path(
        60, method="stationary", block_size=5, seed=1, min_length=60
    )
    second = draw_index_path(
        60, method="stationary", block_size=5, seed=2, min_length=60
    )
    assert first != second


def test_stationary_bootstrap_preserves_local_block_order_dependence() -> None:
    path = draw_index_path(
        200, method="stationary", block_size=8, seed=7, min_length=200
    )
    # within resampled blocks, consecutive indices should mostly increase by
    # exactly 1 (proving local order, i.e. dependence, is preserved rather
    # than every position being drawn independently).
    consecutive_increments = sum(
        1 for a, b in zip(path, path[1:]) if b == a + 1 or (a == 199 and b == 0)
    )
    assert consecutive_increments / (len(path) - 1) > 0.5


def test_circular_bootstrap_is_used_for_the_secondary_method() -> None:
    path = draw_index_path(60, method="circular", block_size=5, seed=3, min_length=60)
    assert len(path) == 60
    assert all(0 <= index < 60 for index in path)


def test_iid_diagnostic_uses_block_size_one() -> None:
    path = draw_index_path(
        500, method="iid_diagnostic", block_size=999, seed=9, min_length=500
    )
    # with block_size effectively 1, consecutive-increment runs should be
    # short and rare compared to a real block bootstrap with a large block.
    consecutive_increments = sum(1 for a, b in zip(path, path[1:]) if b == a + 1)
    assert consecutive_increments / (len(path) - 1) < 0.1


def test_draw_index_path_can_extend_beyond_the_original_trade_count() -> None:
    path = draw_index_path(
        60, method="stationary", block_size=5, seed=11, min_length=500
    )
    assert len(path) == 500
    assert all(0 <= index < 60 for index in path)


@pytest.mark.parametrize("block_size", [0, -1])
def test_non_positive_block_size_is_rejected(block_size) -> None:
    with pytest.raises(BootstrapValidationError):
        draw_index_path(
            60, method="stationary", block_size=block_size, seed=1, min_length=60
        )


def test_seed_must_be_an_explicit_integer() -> None:
    with pytest.raises(BootstrapValidationError):
        draw_index_path(60, method="stationary", block_size=5, seed=True, min_length=60)  # type: ignore[arg-type]
