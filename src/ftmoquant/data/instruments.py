"""Immutable FX instrument identity used by the multi-instrument data pipeline."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation

import pyarrow as pa  # type: ignore[import-untyped]
from nautilus_trader.model import (
    Bar,
    Currency,
    CurrencyPair,
    InstrumentId,
    Price,
    Quantity,
    Symbol,
)
from nautilus_trader.persistence import BarDataWrangler

from ftmoquant.data.dukascopy import SourceBar

_DATASET_SYMBOL = re.compile(r"[A-Z]{6}")
_CURRENCY = re.compile(r"[A-Z]{3}")
_INSTRUMENT_ID = re.compile(r"[A-Z]{3}/[A-Z]{3}\.DUKASCOPY")
_SIDES = frozenset({"BID", "ASK"})


class InstrumentSpecValidationError(ValueError):
    """Raised when an instrument cannot be encoded without ambiguity."""


@dataclass(frozen=True, slots=True)
class InstrumentSpec:
    """Complete source and Nautilus identity for one spot-FX instrument."""

    dataset_symbol: str
    instrument_id: str
    base_currency: str
    quote_currency: str
    price_precision: int
    price_increment: str
    size_precision: int
    size_increment: str
    session_policy_id: str

    def validate(self) -> None:
        if _DATASET_SYMBOL.fullmatch(self.dataset_symbol) is None:
            raise InstrumentSpecValidationError(
                "dataset_symbol must be exactly six uppercase currency letters"
            )
        if _INSTRUMENT_ID.fullmatch(self.instrument_id) is None:
            raise InstrumentSpecValidationError("instrument_id is not canonical")
        if (
            _CURRENCY.fullmatch(self.base_currency) is None
            or _CURRENCY.fullmatch(self.quote_currency) is None
        ):
            raise InstrumentSpecValidationError("currency codes must be uppercase ISO")
        if self.base_currency == self.quote_currency:
            raise InstrumentSpecValidationError("base and quote currencies must differ")
        if self.dataset_symbol != self.base_currency + self.quote_currency:
            raise InstrumentSpecValidationError(
                "dataset_symbol must equal base_currency + quote_currency"
            )
        expected_id = f"{self.base_currency}/{self.quote_currency}.DUKASCOPY"
        if self.instrument_id != expected_id:
            raise InstrumentSpecValidationError(
                "instrument_id must match the declared base and quote currencies"
            )
        if not 0 < self.price_precision <= 9 or not 0 < self.size_precision <= 9:
            raise InstrumentSpecValidationError("precisions must be in the range 1..9")
        self._validate_increment(
            self.price_increment, self.price_precision, "price_increment"
        )
        self._validate_increment(
            self.size_increment, self.size_precision, "size_increment"
        )
        if not self.session_policy_id:
            raise InstrumentSpecValidationError("session_policy_id must not be empty")

    @staticmethod
    def _validate_increment(value: str, precision: int, field: str) -> None:
        try:
            parsed = Decimal(value)
        except InvalidOperation as error:
            raise InstrumentSpecValidationError(f"{field} must be decimal") from error
        expected = Decimal(1).scaleb(-precision)
        if not parsed.is_finite() or parsed != expected:
            raise InstrumentSpecValidationError(
                f"{field} must be exactly one unit at its declared precision"
            )

    def bar_type(self, *, minutes: int, side: str, aggregation: str) -> str:
        """Build an exact Nautilus bar type from this specification."""

        self.validate()
        normalized_side = side.upper()
        normalized_aggregation = aggregation.upper()
        if normalized_side not in _SIDES:
            raise InstrumentSpecValidationError("bar side must be BID or ASK")
        if minutes <= 0:
            raise InstrumentSpecValidationError("bar interval must be positive")
        if normalized_aggregation not in {"EXTERNAL", "INTERNAL"}:
            raise InstrumentSpecValidationError(
                "bar aggregation must be EXTERNAL or INTERNAL"
            )
        unit, step = (
            ("HOUR", minutes // 60) if minutes % 60 == 0 else ("MINUTE", minutes)
        )
        return (
            f"{self.instrument_id}-{step}-{unit}-{normalized_side}-"
            f"{normalized_aggregation}"
        )

    def composite_bar_type(self, *, hours: int, side: str) -> str:
        target = self.bar_type(minutes=hours * 60, side=side, aggregation="INTERNAL")
        return f"{target}@1-MINUTE-EXTERNAL"

    def nautilus_instrument(self) -> CurrencyPair:
        """Construct the pinned-runtime native instrument without string rewriting."""

        self.validate()
        return CurrencyPair(
            instrument_id=InstrumentId.from_str(self.instrument_id),
            raw_symbol=Symbol(self.dataset_symbol),
            base_currency=Currency.from_str(self.base_currency),
            quote_currency=Currency.from_str(self.quote_currency),
            price_precision=self.price_precision,
            size_precision=self.size_precision,
            price_increment=Price.from_str(self.price_increment),
            size_increment=Quantity.from_str(self.size_increment),
            ts_event=0,
            ts_init=0,
            info={
                "provider": "dukascopy",
                "source_granularity": "1-minute",
                "session_policy_id": self.session_policy_id,
            },
        )


EURUSD_SPEC = InstrumentSpec(
    dataset_symbol="EURUSD",
    instrument_id="EUR/USD.DUKASCOPY",
    base_currency="EUR",
    quote_currency="USD",
    price_precision=5,
    price_increment="0.00001",
    size_precision=8,
    size_increment="0.00000001",
    session_policy_id="dukascopy-fx-ny-close-v1",
)

GBPUSD_SPEC = InstrumentSpec(
    dataset_symbol="GBPUSD",
    instrument_id="GBP/USD.DUKASCOPY",
    base_currency="GBP",
    quote_currency="USD",
    price_precision=5,
    price_increment="0.00001",
    size_precision=8,
    size_increment="0.00000001",
    session_policy_id="dukascopy-fx-ny-close-v1",
)

AUDUSD_SPEC = InstrumentSpec(
    dataset_symbol="AUDUSD",
    instrument_id="AUD/USD.DUKASCOPY",
    base_currency="AUD",
    quote_currency="USD",
    price_precision=5,
    price_increment="0.00001",
    size_precision=8,
    size_increment="0.00000001",
    session_policy_id="dukascopy-fx-ny-close-v1",
)

#: JPY-quoted pairs conventionally carry 3 decimal places (2 major + 1
#: fractional pip), unlike the 5-decimal quoting used by the other majors.
USDJPY_SPEC = InstrumentSpec(
    dataset_symbol="USDJPY",
    instrument_id="USD/JPY.DUKASCOPY",
    base_currency="USD",
    quote_currency="JPY",
    price_precision=3,
    price_increment="0.001",
    size_precision=8,
    size_increment="0.00000001",
    session_policy_id="dukascopy-fx-ny-close-v1",
)

USDCHF_SPEC = InstrumentSpec(
    dataset_symbol="USDCHF",
    instrument_id="USD/CHF.DUKASCOPY",
    base_currency="USD",
    quote_currency="CHF",
    price_precision=5,
    price_increment="0.00001",
    size_precision=8,
    size_increment="0.00000001",
    session_policy_id="dukascopy-fx-ny-close-v1",
)


def to_nautilus_bars(
    rows: Sequence[SourceBar],
    side: str,
    instrument: InstrumentSpec,
    *,
    minutes: int = 1,
) -> list[Bar]:
    """Encode source bars using only precision and identity from the spec."""

    instrument.validate()
    if minutes <= 0:
        raise InstrumentSpecValidationError("bar interval minutes must be positive")
    bar_type = instrument.bar_type(minutes=minutes, side=side, aggregation="EXTERNAL")
    price_quantum = Decimal(1).scaleb(-instrument.price_precision)
    size_quantum = Decimal(1).scaleb(-instrument.size_precision)

    def price_bytes(value: Decimal) -> bytes:
        quantized = value.quantize(price_quantum)
        if value != quantized:
            raise InstrumentSpecValidationError(
                f"{instrument.instrument_id} price cannot be represented at "
                f"precision {instrument.price_precision}"
            )
        raw = Price.from_str(format(quantized, f".{instrument.price_precision}f")).raw
        return raw.to_bytes(16, byteorder="little", signed=True)

    def size_bytes(value: Decimal) -> bytes:
        quantized = value.quantize(size_quantum)
        raw = Quantity.from_str(format(quantized, f".{instrument.size_precision}f")).raw
        return raw.to_bytes(16, byteorder="little", signed=False)

    epoch = datetime(1970, 1, 1, tzinfo=UTC)

    def timestamp_ns(value: datetime) -> int:
        delta = value - epoch
        return (
            delta.days * 86_400 + delta.seconds
        ) * 1_000_000_000 + delta.microseconds * 1_000

    arrays = [
        pa.array([price_bytes(getattr(row, field)) for row in rows], type=pa.binary(16))
        for field in ("open", "high", "low", "close")
    ]
    arrays.extend(
        [
            pa.array([size_bytes(row.volume) for row in rows], type=pa.binary(16)),
            pa.array([timestamp_ns(row.timestamp) for row in rows], type=pa.uint64()),
            pa.array(
                [
                    timestamp_ns(row.timestamp + timedelta(minutes=minutes))
                    for row in rows
                ],
                type=pa.uint64(),
            ),
        ]
    )
    schema = pa.schema(
        [
            pa.field("open", pa.binary(16), nullable=False),
            pa.field("high", pa.binary(16), nullable=False),
            pa.field("low", pa.binary(16), nullable=False),
            pa.field("close", pa.binary(16), nullable=False),
            pa.field("volume", pa.binary(16), nullable=False),
            pa.field("ts_event", pa.uint64(), nullable=False),
            pa.field("ts_init", pa.uint64(), nullable=False),
        ]
    )
    batch = pa.RecordBatch.from_arrays(arrays, schema=schema)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, schema) as writer:
        writer.write_batch(batch)
    return BarDataWrangler(
        bar_type, instrument.price_precision, instrument.size_precision
    ).process_record_batch_bytes(sink.getvalue().to_pybytes())
