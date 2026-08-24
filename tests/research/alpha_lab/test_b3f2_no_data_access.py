"""Structural firewall check for the B3F2 modules -- mirrors
``test_b32_no_data_access.py``. None of the four new modules
(``b3f2_asian_range_fade_signals``, ``b3f2_asian_range_fade_execution``,
``b3f2_asian_range_fade_screen``, ``b3f2_development_orchestrator``) may
import real VALIDATION/HOLDOUT data-loading machinery.
"""

from __future__ import annotations

import ast
from pathlib import Path

import ftmoquant.research.alpha_lab.b3f2_asian_range_fade_execution as b3f2_execution_module  # noqa: E501
import ftmoquant.research.alpha_lab.b3f2_asian_range_fade_screen as b3f2_screen_module
import ftmoquant.research.alpha_lab.b3f2_asian_range_fade_signals as b3f2_signals_module
import ftmoquant.research.alpha_lab.b3f2_development_orchestrator as b3f2_orchestrator_module  # noqa: E501

_FORBIDDEN_IMPORT_MODULES = (
    "nautilus_trader.persistence",
    "ftmoquant.research.alpha_lab.data",
    "ftmoquant.research.alpha_lab.wick_fvg_squeeze_execution",
    "ftmoquant.data.oanda_alpha_lab_validation",
    "ftmoquant.data.canonical_source",
    "ftmoquant.research.ftmo_pass_probability.path_extraction",
    "ftmoquant.research.ftmo_pass_probability.validation_diagnostic",
)
#: The orchestrator is allowed to import the real DEVELOPMENT loader and
#: the real M1 loader -- that is its entire job -- so it is checked
#: separately, only for VALIDATION/HOLDOUT-specific machinery.
_ORCHESTRATOR_FORBIDDEN_IMPORT_MODULES = (
    "ftmoquant.data.oanda_alpha_lab_validation",
    "ftmoquant.data.canonical_source",
    "ftmoquant.research.ftmo_pass_probability.path_extraction",
    "ftmoquant.research.ftmo_pass_probability.validation_diagnostic",
)

_SIGNAL_EXECUTION_SCREEN_MODULES = (
    b3f2_signals_module,
    b3f2_execution_module,
    b3f2_screen_module,
)


def _imported_module_names(source: str) -> set[str]:
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return names


def test_signal_execution_screen_modules_import_no_data_loading_machinery() -> None:
    for module in _SIGNAL_EXECUTION_SCREEN_MODULES:
        source = Path(module.__file__).read_text(encoding="utf-8")
        imported = _imported_module_names(source)
        for forbidden in _FORBIDDEN_IMPORT_MODULES:
            assert forbidden not in imported, (
                f"{module.__name__} unexpectedly imports {forbidden!r}"
            )


def test_orchestrator_imports_no_validation_or_holdout_machinery() -> None:
    source = Path(b3f2_orchestrator_module.__file__).read_text(encoding="utf-8")
    imported = _imported_module_names(source)
    for forbidden in _ORCHESTRATOR_FORBIDDEN_IMPORT_MODULES:
        assert forbidden not in imported, (
            f"orchestrator unexpectedly imports {forbidden!r}"
        )


def test_no_module_references_validation_or_holdout_partition_constants() -> None:
    for module in (*_SIGNAL_EXECUTION_SCREEN_MODULES, b3f2_orchestrator_module):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "VALIDATION_START" not in source
        assert "HOLDOUT_START" not in source
        assert "load_validation_trade_path" not in source


def test_no_module_hardcodes_a_news_volatility_or_trend_filter() -> None:
    """Hard rule (section 0): no news filters, no volatility filters, no
    trend filters, no discretionary ICT/liquidity terminology."""

    forbidden_literals = (
        "news_filter",
        "volatility_filter",
        "trend_filter",
        "liquidity_sweep",
        "order_block",
        "fair_value_gap",
        "smart_money",
    )
    for module in (*_SIGNAL_EXECUTION_SCREEN_MODULES, b3f2_orchestrator_module):
        source = Path(module.__file__).read_text(encoding="utf-8").lower()
        for literal in forbidden_literals:
            assert literal not in source, f"{module.__name__} references {literal!r}"


def test_no_module_cherry_picks_a_pair() -> None:
    for module in (*_SIGNAL_EXECUTION_SCREEN_MODULES, b3f2_orchestrator_module):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "AUD/NZD" not in source
