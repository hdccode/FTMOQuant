import json
import zipfile
from pathlib import Path

import pytest

from ftmoquant.data.macro_events import import_historical_calendar


def _write_fixture(path: Path, rows: list[str]) -> None:
    path.write_text(
        (
            "Date,Time,Currency,Impact,News Description,Actual,Forecast,"
            "Previous,Revised From,FF Event ID\n"
        )
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )


def test_import_preserves_raw_values_parses_numbers_and_quarantines_duplicate(
    tmp_path: Path,
) -> None:
    source = tmp_path / "hanover.csv"
    _write_fixture(
        source,
        [
            (
                "2020-01-10,8:30am,USD,High,Non-Farm Employment Change,145K,"
                "160K,256K,241K,100"
            ),
            (
                "2020-01-10,8:30am,USD,High,Non-Farm Employment Change,145K,"
                "160K,256K,241K,100"
            ),
            "2020-01-10,8:30am,USD,High,Unemployment Rate,3.5%,3.5%,3.5%,,101",
            "2020-02-13,8:30am,USD,High,CPI m/m,0.1%,0.2%,oops,,102",
            "2020-02-13,00:00,USD,High,Core CPI y/y,2.3%,2.3%,2.3%,,103",
            "2020-02-13,All Day,USD,High,CPI y/y,2.5%,2.4%,2.3%,,104",
        ],
    )
    output = tmp_path / "output"

    result = import_historical_calendar(
        [source],
        output,
        source_name="fixture",
        source_url="https://example.test/hanover",
        access_date="2026-08-15",
    )

    assert (result.normalized_rows, result.quarantined_rows) == (3, 2)
    events = [
        json.loads(line)
        for line in (output / "normalized_events.jsonl").read_text().splitlines()
    ]
    assert events[0]["timestamp_utc"] == "2020-01-10T13:30:00Z"
    assert events[0]["actual_raw"] == "145K"
    assert events[0]["actual_parsed"] == {
        "status": "parsed",
        "unit": "k",
        "value": "145",
    }
    assert (
        next(row for row in events if row["event_family"] == "US_CPI_HEADLINE_M_M")[
            "previous_parsed"
        ]["status"]
        == "malformed"
    )
    qa = json.loads((output / "qa_report.json").read_text())
    assert qa["counts_by_event_family"] == {
        "US_CPI_CORE_Y_Y": 1,
        "US_CPI_HEADLINE_M_M": 1,
        "US_NFP_HEADLINE_EMPLOYMENT_CHANGE": 1,
    }
    assert qa["exact_duplicate_count"] == 1
    assert qa["timestamp_anomalies"]["missing_or_nonexact_timestamp"] == 1
    assert qa["timestamp_anomalies"]["suspicious_midnight_count"] == 1
    assert qa["revision_count"] == 1
    manifest = json.loads((output / "provenance_manifest.json").read_text())
    assert manifest["source"]["files"][0]["sha256"]
    assert manifest["outputs"]["normalized_events_sha256"] == qa["output_sha256"]


def test_secondary_cross_check_is_non_authoritative_and_fixed(tmp_path: Path) -> None:
    primary = tmp_path / "primary.csv"
    secondary = tmp_path / "secondary.csv"
    rows = [
        "2020-01-10,8:30am,USD,High,Nonfarm Payrolls,145K,160K,256K,,100",
        "2020-02-13,8:30am,USD,High,CPI y/y,2.5%,2.4%,2.3%,,102",
    ]
    _write_fixture(primary, rows)
    _write_fixture(secondary, rows)

    import_historical_calendar(
        [primary],
        tmp_path / "output",
        source_name="fixture",
        source_url="https://example.test/primary",
        access_date="2026-08-15",
        secondary_input=secondary,
    )

    qa = json.loads((tmp_path / "output" / "qa_report.json").read_text())
    assert qa["secondary_cross_check"]["status"] == "completed_non_authoritative"
    assert qa["secondary_cross_check"]["fixed_sample_size"] == 2
    assert qa["secondary_cross_check"]["matching_rows"] == 2


_HANOVER_HEADERLESS = (
    "2024-03-08,13:30,USD,High,Non-Farm Employment Change, 275K ,200K,"
    "353K,290K,100,900, better ,worse\n"
    "2024-03-12,12:30,USD,High,CPI y/y,3.2%,3.1%,3.1%,,101,901,better,\n"
    "2024-03-12,12:30,USD,High,CPI y/y,3.2%,3.1%,3.1%,,101,901,better,\n"
    "2024-03-13,00:00,EUR,Low,Other,foo,,, ,102,901,,\n"
)


def _strict_import(source: Path, output: Path) -> None:
    import_historical_calendar(
        [source],
        output,
        source_name="Hanover / Forex Factory consolidated historical calendar",
        source_url="https://example.test/post",
        source_reference="https://example.test/post#archive",
        stated_coverage="2007-01-01 through 2024-03-31",
        stated_timestamp_semantics="UTC",
        timezone_name="UTC",
        access_date="2026-08-15",
        strict_hanover_utc_schema=True,
    )


def test_strict_2024_utc_headerless_schema_records_field_mapping_and_provenance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "hanover_2007_2024_03.csv"
    source.write_text(_HANOVER_HEADERLESS, encoding="utf-8")
    _strict_import(source, tmp_path / "output")

    output = tmp_path / "output"
    events = [
        json.loads(line)
        for line in (output / "normalized_events.jsonl").read_text().splitlines()
    ]
    assert len(events) == 2
    assert events[0]["timestamp_utc"] == "2024-03-08T13:30:00Z"
    assert events[0]["actual_raw"] == " 275K "
    assert events[0]["actual_better_worse_raw"] == " better "
    assert events[0]["previous_better_worse_raw"] == "worse"
    assert events[0]["raw_source_fields"]["Group ID"] == "900"
    qa = json.loads((output / "qa_report.json").read_text())
    assert qa["total_source_rows"] == 4
    assert qa["usd_event_count"] == 3
    assert qa["nfp_family_count"] == 1
    assert qa["cpi_family_counts"]["US_CPI_HEADLINE_Y_Y"] == 1
    assert qa["duplicate_event_id_count"] == 1
    assert qa["duplicate_timestamp_currency_event_count"] == 1
    assert qa["timestamp_anomalies"]["suspicious_midnight_count"] == 1
    assert qa["group_id_description_change_count"] == 1
    manifest = json.loads((output / "provenance_manifest.json").read_text())
    assert manifest["source"]["files"][0]["filename"] == source.name
    assert manifest["source"]["stated_timestamp_semantics"] == "UTC"
    assert manifest["input_format"] == {
        "header_mode": "headerless_positional_13",
        "positional_schema": [
            "Date",
            "Time",
            "Currency",
            "Impact",
            "Description",
            "Actual",
            "Forecast",
            "Previous",
            "Revised from",
            "FF event ID",
            "Group ID",
            "Actual better/worse",
            "Revised from better/worse",
        ],
    }


def test_strict_headerless_csv_inside_zip_records_member_hash(tmp_path: Path) -> None:
    archive = tmp_path / "hanover_2007_2024_03.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("hanover_2007_2024_03.csv", _HANOVER_HEADERLESS)

    _strict_import(archive, tmp_path / "output")

    manifest = json.loads(
        (tmp_path / "output" / "provenance_manifest.json").read_text()
    )
    source = manifest["source"]["files"][0]
    assert source["filename"] == archive.name
    assert source["csv_member"] == "hanover_2007_2024_03.csv"
    assert len(source["sha256"]) == len(source["csv_member_sha256"]) == 64


def test_strict_headerless_schema_rejects_wrong_column_count(tmp_path: Path) -> None:
    source = tmp_path / "invalid.csv"
    source.write_text(
        "2024-03-08,13:30,USD,High,Non-Farm Employment Change,275K,200K,"
        "353K,290K,100,900,better\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly 13 columns"):
        _strict_import(source, tmp_path / "output")


def test_strict_headerless_repeated_imports_are_identical(tmp_path: Path) -> None:
    source = tmp_path / "hanover.csv"
    source.write_text(_HANOVER_HEADERLESS, encoding="utf-8")
    first = tmp_path / "first"
    second = tmp_path / "second"

    _strict_import(source, first)
    _strict_import(source, second)

    for filename in (
        "normalized_events.jsonl",
        "quarantined_events.jsonl",
        "qa_report.json",
    ):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
