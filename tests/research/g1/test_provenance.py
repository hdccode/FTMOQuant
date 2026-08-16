from pathlib import Path


def test_adapted_oss_code_has_exact_provenance_license_and_tests() -> None:
    documentation = Path("docs/oss_adoption.md").read_text(encoding="utf-8")
    module = Path("src/ftmoquant/research/g1/parameter_space.py").read_text(
        encoding="utf-8"
    )
    license_text = Path("LICENSES/forex-strategy-lab-MIT.txt").read_text(
        encoding="utf-8"
    )

    assert "df34085afb615818038c0e8bedc5aad837883e5c" in documentation
    assert "smart_backtester.py" in documentation
    assert "suggest_params" in documentation
    assert "tests/research/g1/test_parameter_space.py" in documentation
    assert "df34085afb615818038c0e8bedc5aad837883e5c" in module
    assert "Copyright (c) 2026 Mohammed Abumtary" in module
    assert "MIT License" in license_text
