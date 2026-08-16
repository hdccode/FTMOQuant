import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import ftmoquant.data.oanda_development as oanda

START = datetime(2019, 3, 11, tzinfo=UTC)


def _row(timestamp: datetime, bid: str = "15.2", ask: str = "15.3") -> dict[str, str]:
    return {
        "timestamp_utc": timestamp.isoformat().replace("+00:00", "Z"),
        "bid_open": bid,
        "bid_high": bid,
        "bid_low": bid,
        "bid_close": bid,
        "ask_open": ask,
        "ask_high": ask,
        "ask_low": ask,
        "ask_close": ask,
        "volume": "1",
    }


def _candle(timestamp: datetime, *, complete: bool = True) -> dict[str, object]:
    return {
        "complete": complete,
        "time": timestamp.isoformat().replace("+00:00", ".000000000Z"),
        "volume": 7,
        "bid": {"o": "15.2", "h": "15.2", "l": "15.2", "c": "15.2"},
        "ask": {"o": "15.3", "h": "15.3", "l": "15.3", "c": "15.3"},
    }


def _cached_instrument(
    tmp_path: Path,
    timestamps: list[datetime],
) -> tuple[Path, datetime, dict[str, str]]:
    end = START + timedelta(minutes=4)
    instrument = "SOYBN_USD"
    raw_root = tmp_path / "raw" / instrument
    raw_root.mkdir(parents=True)
    stem = oanda._segment_stem(0, START, end)
    request = {
        "provider": "OANDA_v20_practice",
        "instrument": instrument,
        "url": oanda._candles_url(instrument, START, end),
        "requested_start_utc": oanda._utc(START),
        "requested_end_exclusive_utc": oanda._utc(end),
        "granularity": "M1",
        "price_components": "BA",
        "include_first": True,
        "authorization": "redacted",
    }
    (raw_root / f"{stem}.request.json").write_text(
        json.dumps(request), encoding="utf-8"
    )
    candles = [_candle(timestamp) for timestamp in timestamps]
    response = {
        "instrument": instrument,
        "granularity": "M1",
        "candles": candles,
    }
    response_path = raw_root / f"{stem}.json"
    response_path.write_text(json.dumps(response), encoding="utf-8")
    raw_hashes = {response_path.name: oanda._sha256(response_path)}
    processed = tmp_path / "processed" / f"{instrument}_M1_bid_ask.csv"
    qa = tmp_path / "qa" / f"{instrument}_M1_bid_ask_qa.json"
    rows = [oanda._candle_row(candle) for candle in candles]
    oanda._write_processed_and_qa(
        processed, qa, instrument, rows, START, end, raw_hashes
    )
    return (
        tmp_path,
        end,
        {
            "instrument": instrument,
            "processed_sha256": oanda._sha256(processed),
        },
    )


def test_processed_oanda_prices_are_raw_unscaled_and_idempotent(tmp_path: Path) -> None:
    processed = tmp_path / "processed.csv"
    qa = tmp_path / "qa.json"
    rows = [_row(START), _row(START + timedelta(minutes=2))]
    count, missing = oanda._write_processed_and_qa(
        processed,
        qa,
        "SOYBN_USD",
        rows,
        START,
        START + timedelta(minutes=3),
        {"a.json": "a" * 64},
    )
    assert count == 2
    assert missing[0].start_utc == "2019-03-11T00:01:00Z"
    assert "15.2" in processed.read_text(encoding="utf-8")
    report = json.loads(qa.read_text(encoding="utf-8"))
    assert report["raw_prices_preserved_unscaled"] is True
    assert report["soybean_scale_transform"] == "not_applied_in_acquisition"
    assert report["research_ready"] is False
    assert (
        oanda._write_processed_and_qa(
            processed,
            qa,
            "SOYBN_USD",
            rows,
            START,
            START + timedelta(minutes=3),
            {"a.json": "a" * 64},
        )[0]
        == 2
    )


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([_row(START), _row(START)], "duplicate or unordered"),
        (
            [_row(START + timedelta(minutes=1)), _row(START)],
            "duplicate or unordered",
        ),
        ([_row(START, bid="15.4", ask="15.3")], "ASK is below BID"),
        ([_row(START, bid="bad")], "malformed numeric"),
    ],
)
def test_oanda_qa_rejects_duplicates_and_invalid_bid_ask(
    tmp_path: Path, rows: list[dict[str, str]], message: str
) -> None:
    with pytest.raises(oanda.OandaDevelopmentDataError, match=message):
        oanda._write_processed_and_qa(
            tmp_path / "processed.csv",
            tmp_path / "qa.json",
            "XAU_USD",
            rows,
            START,
            START + timedelta(minutes=2),
            {},
        )


def test_candle_raw_mapping_and_request_are_bid_ask_m1_only() -> None:
    candle = {
        "complete": True,
        "time": "2019-03-11T00:00:00.000000000Z",
        "volume": 7,
        "bid": {"o": "1", "h": "2", "l": "1", "c": "2"},
        "ask": {"o": "2", "h": "3", "l": "2", "c": "3"},
    }
    assert oanda._candle_row(candle)["bid_close"] == "2"
    url = oanda._candles_url("SOYBN_USD", START, START + timedelta(minutes=1))
    assert "granularity=M1" in url and "price=BA" in url and "includeFirst=true" in url


def test_incomplete_candle_fails_closed() -> None:
    with pytest.raises(oanda.OandaDevelopmentDataError, match="complete"):
        oanda._candle_row(_candle(START, complete=False))


def test_request_partition_rejects_skips_and_overlaps() -> None:
    end = START + timedelta(minutes=3)
    with pytest.raises(oanda.OandaDevelopmentDataError, match="skipped window"):
        oanda._validate_request_partition(
            (
                (START, START + timedelta(minutes=1)),
                (START + timedelta(minutes=2), end),
            ),
            START,
            end,
        )
    with pytest.raises(oanda.OandaDevelopmentDataError, match="overlapping"):
        oanda._validate_request_partition(
            (
                (START, START + timedelta(minutes=2)),
                (START + timedelta(minutes=1), end),
            ),
            START,
            end,
        )


def test_sparse_complete_cache_is_ready_without_filling(tmp_path: Path) -> None:
    root, end, record = _cached_instrument(
        tmp_path, [START, START + timedelta(minutes=2)]
    )
    result = oanda._audit_cached_instrument(root, "SOYBN_USD", START, end, record)
    report = json.loads(result.qa_path.read_text(encoding="utf-8"))
    assert result.acquisition_complete is True
    assert result.research_ready is True
    assert result.returned_candle_count == 2
    assert result.missing_minutes == 2
    assert (
        report["provider_observation_availability"][
            "calendar_minutes_without_returned_candle"
        ]
        == 2
    )
    assert (
        report["provider_observation_availability"]["filled_or_interpolated_minutes"]
        == 0
    )
    assert "15.2" in (root / "processed/SOYBN_USD_M1_bid_ask.csv").read_text()
    assert "1520" not in (root / "processed/SOYBN_USD_M1_bid_ask.csv").read_text()


def test_processing_drop_is_detected(tmp_path: Path) -> None:
    root, end, record = _cached_instrument(
        tmp_path, [START, START + timedelta(minutes=1)]
    )
    processed = root / "processed/SOYBN_USD_M1_bid_ask.csv"
    lines = processed.read_text(encoding="utf-8").splitlines(keepends=True)
    processed.write_text("".join(lines[:-1]), encoding="utf-8")
    with pytest.raises(oanda.OandaDevelopmentDataError, match="dropped"):
        oanda._audit_cached_instrument(root, "SOYBN_USD", START, end, record)


def test_range_boundary_is_half_open() -> None:
    end = START + timedelta(minutes=1)
    with pytest.raises(oanda.OandaDevelopmentDataError, match="outside"):
        oanda._validate_processed_row(_row(end), end, START, end, None)


def test_frozen_boundaries_and_instruments_are_not_expandable() -> None:
    assert tuple(oanda._segments(START, START + timedelta(minutes=5_001))) == (
        (START, START + timedelta(minutes=5_000)),
        (START + timedelta(minutes=5_000), START + timedelta(minutes=5_001)),
    )
    assert oanda._INSTRUMENTS == ("XAU_USD", "SPX500_USD", "WTICO_USD", "SOYBN_USD")


def test_existing_oanda_token_environment_aliases_are_never_logged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OANDA_API_TOKEN", "synthetic-secret")
    assert oanda._access_token_from_environment() == "synthetic-secret"
