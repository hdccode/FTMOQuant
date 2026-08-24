"""Structural firewall check for the B3.2 B3F1 modules -- mirrors
``test_b31_no_data_access.py``. None of the three new modules
(``b3f1_spread_signals``, ``b3f1_spread_execution``, ``b3f1_spread_screen``)
may import real DEVELOPMENT/VALIDATION/HOLDOUT data-loading machinery; the
B3.2 task brief is explicit that no real DEVELOPMENT screen may run inside
this task.
"""

from __future__ import annotations

import ast
from pathlib import Path

import ftmoquant.research.alpha_lab.b3f1_spread_execution as b3f1_execution_module
import ftmoquant.research.alpha_lab.b3f1_spread_screen as b3f1_screen_module
import ftmoquant.research.alpha_lab.b3f1_spread_signals as b3f1_signals_module

_FORBIDDEN_IMPORT_MODULES = (
    "nautilus_trader.persistence",
    "ftmoquant.research.alpha_lab.data",
    "ftmoquant.research.alpha_lab.wick_fvg_squeeze_execution",
    "ftmoquant.data.oanda_alpha_lab_development",
    "ftmoquant.data.oanda_alpha_lab_validation",
    "ftmoquant.data.canonical_source",
    "ftmoquant.research.stage_g",
    "ftmoquant.research.ftmo_pass_probability.path_extraction",
    "ftmoquant.research.ftmo_pass_probability.validation_diagnostic",
)

_B32_MODULES = (b3f1_signals_module, b3f1_execution_module, b3f1_screen_module)


def _imported_module_names(source: str) -> set[str]:
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return names


def test_b32_modules_import_no_data_loading_machinery() -> None:
    for module in _B32_MODULES:
        source = Path(module.__file__).read_text(encoding="utf-8")
        imported = _imported_module_names(source)
        for forbidden in _FORBIDDEN_IMPORT_MODULES:
            assert forbidden not in imported, (
                f"{module.__name__} unexpectedly imports {forbidden!r} -- "
                "B3.2 must contain no real Batch-3 data path"
            )


def test_b32_modules_do_not_import_nautilus_persistence_catalog() -> None:
    for module in _B32_MODULES:
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "nautilus_trader.persistence" not in source


def test_b32_modules_do_not_reference_validation_or_holdout() -> None:
    for module in _B32_MODULES:
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "VALIDATION_START" not in source
        assert "HOLDOUT_START" not in source
        assert "load_validation_trade_path" not in source


def test_b32_modules_never_hardcode_a_single_pair_selection() -> None:
    """No B3.2 module may special-case one instrument pair -- the whole
    point of the broad screen is identical treatment of all 21 pairs."""

    for module in _B32_MODULES:
        source = Path(module.__file__).read_text(encoding="utf-8")
        for forbidden_literal in ("AUD/NZD", "AUD_NZD", "cointegrat"):
            assert forbidden_literal.lower() not in source.lower()
