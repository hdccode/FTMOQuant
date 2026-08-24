"""Structural firewall check for the B3.1 shared-infrastructure modules.

Both ``ftmoquant.research.alpha_lab.cost_stress`` and
``ftmoquant.research.alpha_lab.relative_value_adapter`` are pure plumbing:
they must never open a real DEVELOPMENT/VALIDATION/HOLDOUT data path, and
must never import anything that could reach one (``ParquetDataCatalog``,
the alpha-lab dataset loaders, OANDA acquisition modules, or the
canonical/validation catalog directory names themselves). This is a static
source-text check, not a runtime one -- it exists so a future edit that
accidentally wires either module to real data fails CI immediately rather
than only being caught by convention.
"""

from __future__ import annotations

import ast
from pathlib import Path

import ftmoquant.research.alpha_lab.cost_stress as cost_stress_module
import ftmoquant.research.alpha_lab.relative_value_adapter as relative_value_module

#: Checked against actual `import X` / `from X import Y` statements only
#: (via `ast`), never raw substring search -- both modules' docstrings
#: legitimately CITE some of these names in prose (e.g. explaining why
#: ``load_m1_bidask``'s BID/ASK bars motivate the cost-stress transform's
#: conservative design) without importing them.
_FORBIDDEN_IMPORT_MODULES = (
    "nautilus_trader.persistence",
    "ftmoquant.research.alpha_lab.data",
    "ftmoquant.research.alpha_lab.wick_fvg_squeeze_execution",
    "ftmoquant.research.alpha_lab.liquidity_structure_screen",
    "ftmoquant.data.oanda_alpha_lab_development",
    "ftmoquant.data.oanda_alpha_lab_validation",
    "ftmoquant.data.canonical_source",
    "ftmoquant.research.stage_g",
    "ftmoquant.research.ftmo_pass_probability.path_extraction",
)

_B31_MODULES = (cost_stress_module, relative_value_module)


def _imported_module_names(source: str) -> set[str]:
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return names


def test_b31_modules_import_no_data_loading_machinery() -> None:
    for module in _B31_MODULES:
        source = Path(module.__file__).read_text(encoding="utf-8")
        imported = _imported_module_names(source)
        for forbidden in _FORBIDDEN_IMPORT_MODULES:
            assert forbidden not in imported, (
                f"{module.__name__} unexpectedly imports {forbidden!r} -- "
                "B3.1 infrastructure must contain no real Batch-3 data path"
            )


def test_b31_modules_do_not_import_nautilus_persistence_catalog() -> None:
    for module in _B31_MODULES:
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "nautilus_trader.persistence" not in source


def test_b31_modules_have_no_filesystem_read_calls() -> None:
    for module in _B31_MODULES:
        source = Path(module.__file__).read_text(encoding="utf-8")
        for token in ("open(", "read_text(", "read_bytes(", "Path(__file__)"):
            assert token not in source, (
                f"{module.__name__} unexpectedly performs filesystem I/O: {token!r}"
            )
