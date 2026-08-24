from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from ftmoquant.data.causal_fx_conversion import (
    CausalFxQuoteError,
    PairedConversionQuote,
    latest_quote_at_or_before,
)
from ftmoquant.data.instruments import OANDA_BATCH5_CROSS_SPECS
from ftmoquant.data.oanda_batch5_crosses import load_config


def test_config_is_exactly_two_native_crosses_and_seals_partitions() -> None:
    config = load_config()
    assert config.instruments == ("AUD/CAD.OANDA", "EUR/JPY.OANDA")
    assert tuple(spec.dataset_symbol for spec in OANDA_BATCH5_CROSS_SPECS) == (
        "AUDCAD",
        "EURJPY",
    )
    assert config.acquisition_start_utc < config.development_start_utc
    assert config.development_end_exclusive_utc == datetime(2023, 4, 11, tzinfo=UTC)


def test_causal_conversion_lookup_never_uses_future_quote() -> None:
    start = datetime(2020, 1, 1, tzinfo=UTC)
    quotes = [
        PairedConversionQuote(start, Decimal("1.30000"), Decimal("1.30002")),
        PairedConversionQuote(
            start + timedelta(minutes=1), Decimal("1.30001"), Decimal("1.30003")
        ),
    ]
    assert latest_quote_at_or_before(quotes, start).timestamp == start
    assert (
        latest_quote_at_or_before(quotes, start + timedelta(seconds=59)).timestamp
        == start
    )
    with pytest.raises(CausalFxQuoteError, match="causally available"):
        latest_quote_at_or_before(quotes, start - timedelta(microseconds=1))


def test_conversion_lookup_rejects_bad_or_duplicate_pairs() -> None:
    timestamp = datetime(2020, 1, 1, tzinfo=UTC)
    with pytest.raises(CausalFxQuoteError, match="ask >= bid"):
        latest_quote_at_or_before(
            [PairedConversionQuote(timestamp, Decimal("2"), Decimal("1"))], timestamp
        )
