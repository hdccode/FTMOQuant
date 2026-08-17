"""Real-path regression: OANDA readiness CLI -> alpha-lab smoke CLI
(--source oanda) -> seven instruments discovered -> VectorBT results, using
the exact real canonical layout (canonical/EUR_USD, .../GBP_USD, ...).

Synthetic tmp_path fixtures only; no real data is read or written.
"""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from ftmoquant.data.derived_bars import derive_instrument_bars
from ftmoquant.data.instruments import (
    OANDA_ALPHA_LAB_SPECS,
    InstrumentSpec,
    oanda_symbol,
)
from ftmoquant.data.oanda_alpha_lab_development import (
    OandaAlphaLabReadinessError,
    canonicalize_oanda_instrument,
    load_oanda_alpha_lab_config,
    readiness_main,
)
from ftmoquant.research.alpha_lab import smoke

START = datetime(2019, 3, 11, tzinfo=UTC)
_CSV_FIELDS = (
    "timestamp_utc",
    "bid_open",
    "bid_high",
    "bid_low",
    "bid_close",
    "ask_open",
    "ask_high",
    "ask_low",
    "ask_close",
    "volume",
)


def _write_processed_csv(path: Path, spec: InstrumentSpec, minutes: int) -> None:
    quantum = Decimal(1).scaleb(-spec.price_precision)
    base = Decimal("110") if spec.price_precision == 3 else Decimal("1.1")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for index in range(minutes):
            timestamp = START + timedelta(minutes=index)

            def fmt(value: Decimal) -> str:
                return str(value.quantize(quantum))

            writer.writerow(
                {
                    "timestamp_utc": timestamp.isoformat().replace("+00:00", "Z"),
                    "bid_open": fmt(base),
                    "bid_high": fmt(base + quantum * 10),
                    "bid_low": fmt(base - quantum * 10),
                    "bid_close": fmt(base + quantum * 5),
                    "ask_open": fmt(base + quantum * 20),
                    "ask_high": fmt(base + quantum * 30),
                    "ask_low": fmt(base + quantum * 10),
                    "ask_close": fmt(base + quantum * 25),
                    "volume": "1",
                }
            )


def _build_all_seven_canonical_roots(
    canonical_root: Path, whole_days: int, *, symbol_style: str = "oanda"
) -> None:
    """``symbol_style="oanda"`` matches the real observed CLI convention
    (``EUR_USD``, ...); ``symbol_style="dataset"`` deliberately builds the
    wrong (``EURUSD``, ...) layout to prove it is NOT silently accepted."""

    canonical_root.mkdir(parents=True, exist_ok=True)
    config = load_oanda_alpha_lab_config()
    minutes = whole_days * 24 * 60
    for spec in OANDA_ALPHA_LAB_SPECS:
        csv_path = canonical_root / f"{spec.dataset_symbol}_source.csv"
        _write_processed_csv(csv_path, spec, minutes)
        symbol = (
            oanda_symbol(spec.dataset_symbol)
            if symbol_style == "oanda"
            else spec.dataset_symbol
        )
        root = canonical_root / symbol
        canonicalize_oanda_instrument(
            processed_csv_path=csv_path,
            instrument_spec=spec,
            output_root=root,
            alpha_lab_config_sha256=config.semantic_sha256,
            start_utc=START,
            end_exclusive_utc=START + timedelta(minutes=minutes),
        )
        derive_instrument_bars(root, spec)


def test_readiness_cli_succeeds_with_real_oanda_symbol_layout(
    tmp_path: Path,
) -> None:
    """Regression for the real observed layout: canonical/EUR_USD, .../
    GBP_USD, etc. (oanda_symbol(dataset_symbol)), not canonical/EURUSD."""

    canonical_root = tmp_path / "canonical"
    _build_all_seven_canonical_roots(
        canonical_root, whole_days=1, symbol_style="oanda"
    )
    for spec in OANDA_ALPHA_LAB_SPECS:
        assert (canonical_root / oanda_symbol(spec.dataset_symbol)).is_dir()
        assert not (canonical_root / spec.dataset_symbol).is_dir()

    readiness_main(
        [
            "--canonical-root",
            str(canonical_root),
            "--output",
            str(tmp_path / "readiness"),
        ]
    )

    readiness_path = (
        tmp_path / "readiness" / "ftmoquant_oanda_alpha_lab_readiness.json"
    )
    assert readiness_path.exists()
    document = json.loads(readiness_path.read_text(encoding="utf-8"))
    assert document["research_ready"] is True
    assert len(document["ordered_instrument_ids"]) == 7
    assert all(
        status == "research_ready"
        for status in document["per_instrument_status"].values()
    )


def test_readiness_cli_rejects_dataset_symbol_style_layout(tmp_path: Path) -> None:
    """canonical/EURUSD (the bare dataset_symbol, no underscore) must NOT
    be silently accepted as the OANDA alpha-lab canonical layout."""

    canonical_root = tmp_path / "canonical"
    _build_all_seven_canonical_roots(
        canonical_root, whole_days=1, symbol_style="dataset"
    )
    for spec in OANDA_ALPHA_LAB_SPECS:
        assert (canonical_root / spec.dataset_symbol).is_dir()
        assert not (canonical_root / oanda_symbol(spec.dataset_symbol)).is_dir()

    with pytest.raises(OandaAlphaLabReadinessError):
        readiness_main(
            [
                "--canonical-root",
                str(canonical_root),
                "--output",
                str(tmp_path / "readiness"),
            ]
        )


def test_readiness_cli_then_smoke_cli_oanda_end_to_end(tmp_path: Path) -> None:
    """Full real-layout round trip: freeze OANDA readiness -> alpha-lab
    smoke CLI (--source oanda) -> all seven instruments discovered ->
    VectorBT portfolio results emitted."""

    canonical_root = tmp_path / "canonical"
    _build_all_seven_canonical_roots(
        canonical_root, whole_days=2, symbol_style="oanda"
    )
    for spec in OANDA_ALPHA_LAB_SPECS:
        assert (canonical_root / oanda_symbol(spec.dataset_symbol)).is_dir()

    readiness_main(
        [
            "--canonical-root",
            str(canonical_root),
            "--output",
            str(tmp_path / "readiness"),
        ]
    )
    readiness_path = (
        tmp_path / "readiness" / "ftmoquant_oanda_alpha_lab_readiness.json"
    )
    assert readiness_path.exists()
    document = json.loads(readiness_path.read_text(encoding="utf-8"))
    assert document["research_ready"] is True
    assert len(document["ordered_instrument_ids"]) == 7

    output_dir = tmp_path / "smoke_out"
    smoke.main(
        [
            "--source",
            "oanda",
            "--development-root",
            str(canonical_root),
            "--universe-readiness",
            str(readiness_path),
            "--timeframe",
            "M30",
            "--output",
            str(output_dir),
        ]
    )

    results_csv = output_dir / "results.csv"
    assert results_csv.exists()
    with results_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    instruments = {row["instrument"] for row in rows}
    assert instruments == {
        "AUD/USD.OANDA",
        "EUR/USD.OANDA",
        "GBP/USD.OANDA",
        "NZD/USD.OANDA",
        "USD/CAD.OANDA",
        "USD/CHF.OANDA",
        "USD/JPY.OANDA",
        "AGGREGATE_EQUAL_WEIGHT",
    }
    assert len(rows) == 8
    for row in rows:
        assert row["strategy_id"] == smoke.STRATEGY_ID
        assert row["timeframe"] == "M30"

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert len(metadata["instrument_ids"]) == 7


def test_smoke_cli_oanda_discovery_rejects_dataset_symbol_style_layout(
    tmp_path: Path,
) -> None:
    """A readiness artifact built over the real EUR_USD-style layout must
    NOT be loadable against a mismatched canonical/EURUSD-style root."""

    canonical_root = tmp_path / "canonical"
    _build_all_seven_canonical_roots(
        canonical_root, whole_days=1, symbol_style="oanda"
    )
    readiness_main(
        [
            "--canonical-root",
            str(canonical_root),
            "--output",
            str(tmp_path / "readiness"),
        ]
    )
    readiness_path = (
        tmp_path / "readiness" / "ftmoquant_oanda_alpha_lab_readiness.json"
    )

    from ftmoquant.research.alpha_lab.data import AlphaLabDataError

    wrong_layout_root = tmp_path / "wrong_canonical"
    wrong_layout_root.mkdir()
    for spec in OANDA_ALPHA_LAB_SPECS:
        (wrong_layout_root / spec.dataset_symbol).mkdir()

    with pytest.raises(AlphaLabDataError, match="catalog tree hash drifted"):
        smoke.main(
            [
                "--source",
                "oanda",
                "--development-root",
                str(wrong_layout_root),
                "--universe-readiness",
                str(readiness_path),
                "--timeframe",
                "M30",
                "--output",
                str(tmp_path / "smoke_out_wrong"),
            ]
        )


def test_dukascopy_source_path_unchanged(tmp_path: Path) -> None:
    """--source dukascopy (default) must still require plan_path and never
    touch the OANDA discovery path added by this fix."""

    from ftmoquant.research.alpha_lab.data import (
        AlphaLabDataError,
        load_alpha_lab_dataset,
    )

    with pytest.raises(AlphaLabDataError, match="plan_path"):
        load_alpha_lab_dataset(
            readiness_path=tmp_path / "readiness.json",
            development_root_dir=tmp_path / "development",
            timeframe="H1",
            source="dukascopy",
        )


def test_smoke_cli_default_source_is_dukascopy() -> None:
    parser = smoke.build_parser()
    args = parser.parse_args(
        [
            "--development-root",
            "dev",
            "--universe-readiness",
            "readiness.json",
            "--output",
            "out",
        ]
    )
    assert args.source == "dukascopy"


def test_smoke_cli_unknown_source_fails() -> None:
    parser = smoke.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--source",
                "bogus",
                "--development-root",
                "dev",
                "--universe-readiness",
                "readiness.json",
                "--output",
                "out",
            ]
        )


def test_readiness_cli_still_requires_all_seven_instruments(tmp_path: Path) -> None:
    canonical_root = tmp_path / "canonical"
    config = load_oanda_alpha_lab_config()
    spec = OANDA_ALPHA_LAB_SPECS[0]
    minutes = 24 * 60
    csv_path = canonical_root / f"{spec.dataset_symbol}_source.csv"
    canonical_root.mkdir(parents=True)
    _write_processed_csv(csv_path, spec, minutes)
    root = canonical_root / spec.dataset_symbol
    canonicalize_oanda_instrument(
        processed_csv_path=csv_path,
        instrument_spec=spec,
        output_root=root,
        alpha_lab_config_sha256=config.semantic_sha256,
        start_utc=START,
        end_exclusive_utc=START + timedelta(minutes=minutes),
    )
    derive_instrument_bars(root, spec)

    with pytest.raises(OandaAlphaLabReadinessError):
        readiness_main(
            [
                "--canonical-root",
                str(canonical_root),
                "--output",
                str(tmp_path / "readiness"),
            ]
        )


def test_smoke_cli_oanda_source_rejects_validation_holdout_named_root(
    tmp_path: Path,
) -> None:
    canonical_root = tmp_path / "canonical"
    _build_all_seven_canonical_roots(canonical_root, whole_days=1)
    readiness_main(
        [
            "--canonical-root",
            str(canonical_root),
            "--output",
            str(tmp_path / "readiness"),
        ]
    )
    readiness_path = (
        tmp_path / "readiness" / "ftmoquant_oanda_alpha_lab_readiness.json"
    )

    from ftmoquant.research.alpha_lab.data import AlphaLabDataError

    forbidden_root = tmp_path / "validation_canonical"
    forbidden_root.mkdir()
    for spec in OANDA_ALPHA_LAB_SPECS:
        (forbidden_root / oanda_symbol(spec.dataset_symbol)).mkdir()

    with pytest.raises(AlphaLabDataError):
        smoke.main(
            [
                "--source",
                "oanda",
                "--development-root",
                str(forbidden_root),
                "--universe-readiness",
                str(readiness_path),
                "--timeframe",
                "M30",
                "--output",
                str(tmp_path / "smoke_out_forbidden"),
            ]
        )
