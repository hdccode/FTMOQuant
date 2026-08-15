import json
from pathlib import Path

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
