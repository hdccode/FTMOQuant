"""Future-safe native quote lookup for later non-USD P&L conversion."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal


class CausalFxQuoteError(ValueError):
    """Raised when a causal, paired executable conversion quote is unavailable."""


@dataclass(frozen=True, slots=True)
class PairedConversionQuote:
    timestamp: datetime
    bid: Decimal
    ask: Decimal


def latest_quote_at_or_before(
    quotes: Sequence[PairedConversionQuote], timestamp: datetime
) -> PairedConversionQuote:
    """Return the latest genuine paired quote no later than ``timestamp``."""

    if timestamp.tzinfo is None:
        raise CausalFxQuoteError("lookup timestamp must be timezone-aware")
    prior: datetime | None = None
    for quote in quotes:
        if quote.timestamp.tzinfo is None:
            raise CausalFxQuoteError("quote timestamps must be timezone-aware")
        if prior is not None and quote.timestamp <= prior:
            raise CausalFxQuoteError(
                "quotes must be strictly increasing without duplicates"
            )
        if quote.bid <= 0 or quote.ask < quote.bid:
            raise CausalFxQuoteError(
                "conversion quote must have positive bid and ask >= bid"
            )
        prior = quote.timestamp
    index = bisect_right([quote.timestamp for quote in quotes], timestamp) - 1
    if index < 0:
        raise CausalFxQuoteError("no conversion quote is causally available")
    return quotes[index]
