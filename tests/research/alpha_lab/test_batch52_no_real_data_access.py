from __future__ import annotations

import inspect

from ftmoquant.research.alpha_lab import (
    batch5_execution,
    batch5_screen,
    batch5a_cftc_execution,
    batch5a_cftc_signals,
    batch5b_direct_mr_execution,
    batch5b_direct_mr_signals,
    batch5c_daily_reversal_execution,
    batch5c_daily_reversal_signals,
)


def test_batch52_modules_have_no_autonomous_partition_loaders_or_real_paths() -> None:
    modules = (
        batch5_execution,
        batch5_screen,
        batch5a_cftc_execution,
        batch5a_cftc_signals,
        batch5b_direct_mr_execution,
        batch5b_direct_mr_signals,
        batch5c_daily_reversal_execution,
        batch5c_daily_reversal_signals,
    )
    forbidden = (
        "load_alpha_lab_dataset",
        ".artifacts/batch5_cftc_tff_v1",
        "ftmoquant-data/batch5_oanda_native_crosses_v1",
        "VALIDATION",
        "holdout",
        "Monte Carlo",
    )
    for module in modules:
        source = inspect.getsource(module)
        for token in forbidden:
            assert token not in source
