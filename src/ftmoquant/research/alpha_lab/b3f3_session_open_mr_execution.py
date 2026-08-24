"""B3F3 execution: turns causal :class:`B3F3TradeIntent` decisions into
genuine, single-instrument, native-BID/ASK-executed trades.

Structurally identical to
:mod:`ftmoquant.research.alpha_lab.b3f2_asian_range_fade_execution` (same
reuse precedents: :func:`ftmoquant.research.alpha_lab.b3f1_spread_execution.
usd_gross_to_quantity` for sizing, ``ftmoquant.research.
mean_reversion_h1_development._convert_to_account_currency`` for USD P&L
conversion, :mod:`ftmoquant.research.alpha_lab.cost_stress` for
pre-execution widening, the same bisect-based "first strictly later"
convention and stop-before-target collision rule) -- reimplemented here
(not imported) only because it operates on :class:`B3F3TradeIntent` and
its own frozen 11:00 London time-exit boundary rather than B3F2's 16:00
one; the underlying execution semantics are unchanged.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

import pandas as pd  # type: ignore[import-untyped]

from ftmoquant.data.instruments import InstrumentSpec
from ftmoquant.research.alpha_lab.b3f1_spread_execution import (
    GROSS_NOTIONAL_USD,
    usd_gross_to_quantity,
)
from ftmoquant.research.alpha_lab.b3f3_session_open_mr_signals import (
    EXIT_REASON_STOP,
    EXIT_REASON_TARGET,
    EXIT_REASON_TIME,
    B3F3TradeIntent,
)
from ftmoquant.research.alpha_lab.cost_stress import widen_bid_ask_frame
from ftmoquant.research.mean_reversion_h1_development import (
    _convert_to_account_currency as convert_to_account_currency,
)

_LONG = 1
_SHORT = -1


class B3F3ExecutionError(ValueError):
    """Raised on any violation of the frozen B3F3 execution contract."""


@dataclass(frozen=True, slots=True)
class B3F3Trade:
    instrument_id: str
    config_id: str
    local_date_iso: str
    direction: int
    entry_ns: int
    entry_price: Decimal
    exit_ns: int
    exit_price: Decimal
    exit_reason: str
    quantity: Decimal
    realized_pnl_usd: Decimal
    displacement_at_entry: float

    def __post_init__(self) -> None:
        if self.exit_ns <= self.entry_ns:
            raise B3F3ExecutionError("exit must be strictly after entry")
        if self.exit_reason not in (
            EXIT_REASON_STOP,
            EXIT_REASON_TARGET,
            EXIT_REASON_TIME,
        ):
            raise B3F3ExecutionError(f"unknown exit_reason {self.exit_reason!r}")


@dataclass(frozen=True, slots=True)
class B3F3SkipRecord:
    instrument_id: str
    signal_ts: pd.Timestamp
    reason: str


def simulate_b3f3_intents(
    intents: Sequence[B3F3TradeIntent],
    *,
    instrument_spec: InstrumentSpec,
    bid_m1: pd.DataFrame,
    ask_m1: pd.DataFrame,
    gross_notional_usd: Decimal = GROSS_NOTIONAL_USD,
    cost_stress_multiplier: Decimal = Decimal("1"),
) -> tuple[tuple[B3F3Trade, ...], tuple[B3F3SkipRecord, ...]]:
    """Execute a chronological sequence of causal :class:`B3F3TradeIntent`
    decisions against genuine M1 BID/ASK data, one :class:`B3F3Trade` per
    filled intent.

    Section 14: ``cost_stress_multiplier`` is applied to the raw M1
    BID/ASK frames BEFORE any fill price is read -- signal formation (M5,
    upstream of this function) is untouched by this parameter.

    Section 8/11: one position at a time (a signal during an open trade
    is skipped, never queued); entry at the first strictly-later paired
    M1 observation; LONG buys ASK / sells BID, SHORT sells BID / buys ASK;
    stop wins a same-bar collision with target; the 11:00 London time
    exit is checked only once neither stop nor target has touched on that
    bar, using that bar's own liquidation-side close as the first valid
    M1 liquidation quote at or after the frozen exit time (section 12) --
    never a bar from the next trading day.
    """

    if not bid_m1.index.equals(ask_m1.index):
        raise B3F3ExecutionError(
            "bid_m1 and ask_m1 must share an identical paired index"
        )

    if cost_stress_multiplier != Decimal("1"):
        bid_m1, ask_m1 = widen_bid_ask_frame(
            bid_m1, ask_m1, float(cost_stress_multiplier)
        )

    paired_index = bid_m1.index
    paired_ns = paired_index.as_unit("ns").asi8.tolist()

    trades: list[B3F3Trade] = []
    skips: list[B3F3SkipRecord] = []
    busy_until_ns: int | None = None

    for intent in intents:
        decision_ns = int(intent.signal_ts.value)
        if busy_until_ns is not None and decision_ns <= busy_until_ns:
            skips.append(
                B3F3SkipRecord(
                    intent.instrument_id, intent.signal_ts, "signal_during_open_trade"
                )
            )
            continue

        entry_pos = bisect.bisect_right(paired_ns, decision_ns)
        if entry_pos >= len(paired_ns):
            skips.append(
                B3F3SkipRecord(
                    intent.instrument_id, intent.signal_ts, "no_later_m1_observation"
                )
            )
            continue

        entry_ts = paired_index[entry_pos]
        entry_price = (
            Decimal(str(ask_m1["close"].iloc[entry_pos]))
            if intent.direction == _LONG
            else Decimal(str(bid_m1["close"].iloc[entry_pos]))
        )
        stop_price = Decimal(str(intent.stop_price))
        target_price = Decimal(str(intent.target_price))
        time_exit_ns = int(intent.time_exit_boundary_utc.value)

        exit_found = False
        for i in range(entry_pos + 1, len(paired_index)):
            liquidation = (
                bid_m1.iloc[i] if intent.direction == _LONG else ask_m1.iloc[i]
            )
            stop_touched = (
                liquidation["low"] <= float(stop_price)
                if intent.direction == _LONG
                else liquidation["high"] >= float(stop_price)
            )
            target_touched = (
                liquidation["high"] >= float(target_price)
                if intent.direction == _LONG
                else liquidation["low"] <= float(target_price)
            )
            if stop_touched:
                exit_reason, exit_price = EXIT_REASON_STOP, stop_price
            elif target_touched:
                exit_reason, exit_price = EXIT_REASON_TARGET, target_price
            elif paired_ns[i] >= time_exit_ns:
                exit_reason = EXIT_REASON_TIME
                exit_price = Decimal(str(liquidation["close"]))
            else:
                continue

            exit_ts = paired_index[i]
            quantity = usd_gross_to_quantity(
                gross_notional_usd,
                entry_price,
                base_currency=instrument_spec.base_currency,
                quote_currency=instrument_spec.quote_currency,
            )
            pnl_quote = intent.direction * quantity * (exit_price - entry_price)
            pnl_usd = convert_to_account_currency(
                pnl_quote,
                instrument_spec.quote_currency,
                base_currency=instrument_spec.base_currency,
                quote_currency=instrument_spec.quote_currency,
                conversion_price=entry_price,
            )
            trades.append(
                B3F3Trade(
                    instrument_id=intent.instrument_id,
                    config_id=intent.config.config_id,
                    local_date_iso=intent.local_date.isoformat(),
                    direction=intent.direction,
                    entry_ns=int(entry_ts.value),
                    entry_price=entry_price,
                    exit_ns=int(exit_ts.value),
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    quantity=quantity,
                    realized_pnl_usd=pnl_usd,
                    displacement_at_entry=intent.displacement_at_entry,
                )
            )
            busy_until_ns = int(exit_ts.value)
            exit_found = True
            break

        if not exit_found:
            skips.append(
                B3F3SkipRecord(
                    intent.instrument_id, intent.signal_ts, "no_m1_exit_before_data_end"
                )
            )
            busy_until_ns = paired_ns[-1]

    return tuple(trades), tuple(skips)
