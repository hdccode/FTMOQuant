"""Resume/retry robustness for OANDA acquisition -- the exact failure shape
observed in the real VALIDATION run: N chunks already cached, the next
chunk raises a transient HTTP error, and acquisition must resume cleanly
without re-fetching or corrupting anything already on disk."""

from __future__ import annotations

import email.message
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

import ftmoquant.data.oanda_development as oanda

START = datetime(2023, 4, 11, tzinfo=UTC)
_INSTRUMENT = "EUR_USD"


def _no_sleep(_seconds: float) -> None:
    return None


def _http_error(code: int, *, retry_after: str | None = None) -> HTTPError:
    headers = email.message.Message()
    if retry_after is not None:
        headers.add_header("Retry-After", retry_after)
    return HTTPError("https://example.invalid", code, "error", headers, None)


# ---------------------------------------------------------------------------
# Section 2: _fetch_response retry behavior.
# ---------------------------------------------------------------------------


def test_transient_504_is_retried_and_eventually_succeeds(monkeypatch) -> None:
    pending_errors = [_http_error(504), _http_error(504)]
    attempts = {"n": 0}

    class _Response:
        def read(self) -> bytes:
            return b'{"instrument": "EUR_USD", "granularity": "M1", "candles": []}'

        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    def fake_urlopen(request, timeout):  # noqa: ANN001
        attempts["n"] += 1
        if pending_errors:
            raise pending_errors.pop(0)
        return _Response()

    monkeypatch.setattr(oanda, "urlopen", fake_urlopen)
    sleeps: list[float] = []
    payload = oanda._fetch_response(
        "https://example.invalid", "Bearer x", sleep=sleeps.append
    )
    assert json.loads(payload)["instrument"] == "EUR_USD"
    assert attempts["n"] == 3
    assert len(sleeps) == 2
    assert sleeps == sorted(sleeps)  # non-decreasing (exponential backoff)


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_each_retryable_status_is_retried(monkeypatch, status: int) -> None:
    attempts = {"n": 0}

    def fake_urlopen(request, timeout):  # noqa: ANN001
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _http_error(status)

        class _Response:
            def read(self) -> bytes:
                return b'{"instrument": "EUR_USD", "granularity": "M1", "candles": []}'

            def __enter__(self) -> _Response:
                return self

            def __exit__(self, *exc: object) -> None:
                return None

        return _Response()

    monkeypatch.setattr(oanda, "urlopen", fake_urlopen)
    oanda._fetch_response("https://example.invalid", "Bearer x", sleep=_no_sleep)
    assert attempts["n"] == 2


def test_repeated_504_exhausts_retry_budget_and_fails_cleanly(monkeypatch) -> None:
    attempts = {"n": 0}

    def fake_urlopen(request, timeout):  # noqa: ANN001
        attempts["n"] += 1
        raise _http_error(504)

    monkeypatch.setattr(oanda, "urlopen", fake_urlopen)
    with pytest.raises(oanda.OandaDevelopmentDataError, match="failed after"):
        oanda._fetch_response("https://example.invalid", "Bearer x", sleep=_no_sleep)
    assert attempts["n"] == oanda._MAX_FETCH_ATTEMPTS


def test_404_does_not_retry(monkeypatch) -> None:
    attempts = {"n": 0}

    def fake_urlopen(request, timeout):  # noqa: ANN001
        attempts["n"] += 1
        raise _http_error(404)

    monkeypatch.setattr(oanda, "urlopen", fake_urlopen)
    with pytest.raises(oanda.OandaDevelopmentDataError):
        oanda._fetch_response("https://example.invalid", "Bearer x", sleep=_no_sleep)
    assert attempts["n"] == 1


def test_network_error_is_retried(monkeypatch) -> None:
    attempts = {"n": 0}

    def fake_urlopen(request, timeout):  # noqa: ANN001
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise URLError("connection reset")

        class _Response:
            def read(self) -> bytes:
                return b'{"instrument": "EUR_USD", "granularity": "M1", "candles": []}'

            def __enter__(self) -> _Response:
                return self

            def __exit__(self, *exc: object) -> None:
                return None

        return _Response()

    monkeypatch.setattr(oanda, "urlopen", fake_urlopen)
    oanda._fetch_response("https://example.invalid", "Bearer x", sleep=_no_sleep)
    assert attempts["n"] == 3


def test_429_respects_retry_after_header(monkeypatch) -> None:
    delay = oanda._retry_delay_seconds(_http_error(429, retry_after="12.5"), 1)
    assert delay == 12.5


def test_429_without_retry_after_falls_back_to_backoff() -> None:
    delay = oanda._retry_delay_seconds(_http_error(429), 1)
    assert delay == oanda._RETRY_BASE_DELAY_SECONDS


def test_backoff_is_exponential_and_capped() -> None:
    delays = [
        oanda._retry_delay_seconds(_http_error(504), attempt) for attempt in range(1, 8)
    ]
    assert delays[0] == oanda._RETRY_BASE_DELAY_SECONDS
    assert delays[1] == oanda._RETRY_BASE_DELAY_SECONDS * 2
    assert delays[-1] == oanda._RETRY_MAX_DELAY_SECONDS
    assert delays == sorted(delays)


def test_final_failure_raises_oanda_development_data_error_not_swallowed(
    monkeypatch,
) -> None:
    def fake_urlopen(request, timeout):  # noqa: ANN001
        raise _http_error(503)

    monkeypatch.setattr(oanda, "urlopen", fake_urlopen)
    with pytest.raises(oanda.OandaDevelopmentDataError):
        oanda._fetch_response("https://example.invalid", "Bearer x", sleep=_no_sleep)


# ---------------------------------------------------------------------------
# Section 4: the exact observed failure shape, at _acquire_instrument level.
# ---------------------------------------------------------------------------


def _write_cached_chunk(
    raw_root: Path, instrument: str, index: int, seg_start: datetime, seg_end: datetime
) -> None:
    stem = oanda._segment_stem(index, seg_start, seg_end)
    request = {
        "provider": "OANDA_v20_practice",
        "instrument": instrument,
        "url": oanda._candles_url(instrument, seg_start, seg_end),
        "requested_start_utc": oanda._utc(seg_start),
        "requested_end_exclusive_utc": oanda._utc(seg_end),
        "granularity": "M1",
        "price_components": "BA",
        "include_first": True,
        "authorization": "redacted",
    }
    (raw_root / f"{stem}.request.json").write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    candles = [
        {
            "complete": True,
            "time": seg_start.isoformat().replace("+00:00", ".000000000Z"),
            "volume": 1,
            "bid": {"o": "1.1", "h": "1.1", "l": "1.1", "c": "1.1"},
            "ask": {"o": "1.2", "h": "1.2", "l": "1.2", "c": "1.2"},
        }
    ]
    response = {"instrument": instrument, "granularity": "M1", "candles": candles}
    (raw_root / f"{stem}.json").write_text(json.dumps(response), encoding="utf-8")


def _total_chunk_count(chunks: int) -> datetime:
    return START + timedelta(minutes=oanda._CHUNK_MINUTES * chunks)


def test_acquisition_resumes_from_first_missing_chunk_after_transient_failure(
    tmp_path: Path,
) -> None:
    total_chunks = 5
    cached_chunks = 3
    end = _total_chunk_count(total_chunks)
    segments = list(oanda._segments(START, end))
    assert len(segments) == total_chunks

    raw_root = tmp_path / "raw" / _INSTRUMENT
    raw_root.mkdir(parents=True)
    for index in range(cached_chunks):
        seg_start, seg_end = segments[index]
        _write_cached_chunk(raw_root, _INSTRUMENT, index, seg_start, seg_end)
    cached_files_before = sorted(p.name for p in raw_root.glob("*.json"))

    fetch_calls: list[str] = []

    def fetcher(url: str, authorization: str) -> bytes:
        # The chunk immediately after the cache: a transient failure that
        # exhausts _fetch_response's own retry budget (tested in isolation
        # above) and propagates up -- exactly what the real 504 did.
        fetch_calls.append(url)
        raise oanda.OandaDevelopmentDataError("simulated transient failure")

    # First attempt: the network call for chunk `cached_chunks` fails.
    with pytest.raises(oanda.OandaDevelopmentDataError):
        oanda._acquire_instrument(tmp_path, _INSTRUMENT, START, end, "token", fetcher)
    assert len(fetch_calls) == 1
    # Nothing was written for the failed chunk.
    assert sorted(p.name for p in raw_root.glob("*.json")) == cached_files_before

    # Resume: the same call retried, this time succeeding for every
    # remaining chunk. Cached chunks 0..cached_chunks-1 must not be
    # re-requested.
    fetch_calls.clear()

    def succeeding_fetcher(url: str, authorization: str) -> bytes:
        fetch_calls.append(url)
        return json.dumps(
            {"instrument": _INSTRUMENT, "granularity": "M1", "candles": []}
        ).encode()

    result = oanda._acquire_instrument(
        tmp_path, _INSTRUMENT, START, end, "token", succeeding_fetcher
    )
    assert len(fetch_calls) == total_chunks - cached_chunks
    for seg_start, _ in segments[:cached_chunks]:
        assert not any(seg_start.isoformat() in url for url in fetch_calls)
    assert result.raw_response_count == total_chunks
    all_response_files = sorted(
        p.name for p in raw_root.glob("*.json") if not p.name.endswith(".request.json")
    )
    assert len(all_response_files) == total_chunks
    assert len(set(all_response_files)) == total_chunks  # no duplicates

    # Rerunning again is fully deterministic: nothing left to fetch.
    fetch_calls.clear()
    second_result = oanda._acquire_instrument(
        tmp_path, _INSTRUMENT, START, end, "token", succeeding_fetcher
    )
    assert fetch_calls == []
    assert second_result.processed_sha256 == result.processed_sha256
    assert second_result.raw_response_sha256 == result.raw_response_sha256


def test_completed_cached_instrument_is_reused_without_any_network_call(
    tmp_path: Path,
) -> None:
    total_chunks = 3
    end = _total_chunk_count(total_chunks)
    segments = list(oanda._segments(START, end))
    raw_root = tmp_path / "raw" / _INSTRUMENT
    raw_root.mkdir(parents=True)
    for index, (seg_start, seg_end) in enumerate(segments):
        _write_cached_chunk(raw_root, _INSTRUMENT, index, seg_start, seg_end)

    def failing_fetcher(url: str, authorization: str) -> bytes:
        raise AssertionError("network fetch attempted for a fully-cached instrument")

    result = oanda._acquire_instrument(
        tmp_path, _INSTRUMENT, START, end, "token", failing_fetcher
    )
    assert result.raw_response_count == total_chunks
    assert result.row_count == total_chunks


def test_malformed_cached_response_fails_closed_and_is_not_silently_trusted(
    tmp_path: Path,
) -> None:
    total_chunks = 2
    end = _total_chunk_count(total_chunks)
    segments = list(oanda._segments(START, end))
    raw_root = tmp_path / "raw" / _INSTRUMENT
    raw_root.mkdir(parents=True)
    seg_start, seg_end = segments[0]
    _write_cached_chunk(raw_root, _INSTRUMENT, 0, seg_start, seg_end)
    # Corrupt the cached response: wrong instrument recorded inside it.
    stem = oanda._segment_stem(0, seg_start, seg_end)
    response_path = raw_root / f"{stem}.json"
    document = json.loads(response_path.read_text(encoding="utf-8"))
    document["instrument"] = "GBP_USD"
    response_path.write_text(json.dumps(document), encoding="utf-8")

    def failing_fetcher(url: str, authorization: str) -> bytes:
        raise AssertionError("must fail before reaching the network")

    with pytest.raises(oanda.OandaDevelopmentDataError, match="instrument"):
        oanda._acquire_instrument(
            tmp_path, _INSTRUMENT, START, end, "token", failing_fetcher
        )


def test_malformed_json_cached_response_fails_closed(tmp_path: Path) -> None:
    total_chunks = 1
    end = _total_chunk_count(total_chunks)
    segments = list(oanda._segments(START, end))
    raw_root = tmp_path / "raw" / _INSTRUMENT
    raw_root.mkdir(parents=True)
    seg_start, seg_end = segments[0]
    stem = oanda._segment_stem(0, seg_start, seg_end)
    (raw_root / f"{stem}.json").write_text("{not valid json", encoding="utf-8")

    def failing_fetcher(url: str, authorization: str) -> bytes:
        raise AssertionError("must fail before reaching the network")

    with pytest.raises(oanda.OandaDevelopmentDataError):
        oanda._acquire_instrument(
            tmp_path, _INSTRUMENT, START, end, "token", failing_fetcher
        )


def test_invalid_fresh_response_is_never_persisted(tmp_path: Path) -> None:
    """A fresh fetch that returns garbage must not leave a cached artifact
    behind -- otherwise the corruption would "stick" and permanently block
    resumption without manual intervention."""

    total_chunks = 1
    end = _total_chunk_count(total_chunks)

    def bad_fetcher(url: str, authorization: str) -> bytes:
        return json.dumps(
            {"instrument": "WRONG_PAIR", "granularity": "M1", "candles": []}
        ).encode()

    with pytest.raises(oanda.OandaDevelopmentDataError):
        oanda._acquire_instrument(
            tmp_path, _INSTRUMENT, START, end, "token", bad_fetcher
        )

    raw_root = tmp_path / "raw" / _INSTRUMENT
    response_files = list(raw_root.glob("*.json")) if raw_root.exists() else []
    response_files = [p for p in response_files if not p.name.endswith(".request.json")]
    assert response_files == []
