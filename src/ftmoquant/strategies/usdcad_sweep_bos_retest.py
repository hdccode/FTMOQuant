"""Execution-promotion Nautilus adapter for the frozen, validated
``B2F1_sweep_bos_retest`` / USD/CAD.OANDA / M30 / ``swing_lookback=40,
rr=2.0`` Alpha Lab candidate.

This module is deliberately thin, mirroring
:mod:`ftmoquant.strategies.trend_pullback`'s separation of a pure causal
core from a "thin Nautilus strategy adapter", and
:mod:`ftmoquant.research.mean_reversion_h1_development`'s "precompute
offline, mechanically replay into Nautilus" pattern:

- The signal (sweep -> BOS -> retest detection) is never reimplemented here
  -- :func:`~ftmoquant.research.alpha_lab.liquidity_structure_signals.
  b2f1_sweep_bos_retest_signals` is imported and called unchanged.
- The trade lifecycle (first-strictly-later M1 entry, BUY=ASK/SELL=BID,
  stop-first same-M1-observation collision, one-position-at-a-time) is never
  reimplemented here either -- :func:`~ftmoquant.research.alpha_lab.
  wick_fvg_squeeze_execution.simulate_trades` is imported and called
  unchanged. Its output (:class:`~ftmoquant.research.alpha_lab.
  wick_fvg_squeeze_execution.Trade`) is the single source of truth for
  *when* and *in which direction* this module tells Nautilus to trade.
- :class:`UsdCadSweepBosRetestExecutor` (a real ``nautilus_trader.trading.
  Strategy``) does not decide anything live: it mechanically submits one
  market entry order at each precomputed trade's ``entry_ts`` and one
  market close at its ``exit_ts``, then lets Nautilus's own native
  BID/ASK-crossing bar-execution mechanics, portfolio, and account produce
  the actual fill prices, currency-converted equity, and realized P&L. This
  is intentionally NOT a second implementation of stop-first collision
  logic inside Nautilus -- that logic lives in exactly one place
  (``simulate_trades``), proven once, reused here unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal

import pandas as pd  # type: ignore[import-untyped]
from nautilus_trader.model import (
    Bar,
    BarType,
    Currency,
    CurrencyPair,
    InstrumentId,
    OrderFilled,
    OrderSide,
    PositionClosed,
    StrategyId,
)
from nautilus_trader.trading import Strategy, StrategyConfig

from ftmoquant.research.alpha_lab.liquidity_structure_signals import (
    b2f1_sweep_bos_retest_signals,
)
from ftmoquant.research.alpha_lab.wick_fvg_squeeze_execution import (
    SkipRecord,
    Trade,
    simulate_trades,
)
from ftmoquant.research.alpha_lab.wick_fvg_squeeze_signals import (
    DIRECTION_LONG,
)
from ftmoquant.research.eurusd_tsm_development import EquityPoint
from ftmoquant.research.stage_g import HOLDOUT_START

# ---------------------------------------------------------------------------
# Frozen candidate identity -- never changed by this module.
# ---------------------------------------------------------------------------

FROZEN_FAMILY = "B2F1_sweep_bos_retest"
FROZEN_INSTRUMENT_ID = "USD/CAD.OANDA"
FROZEN_TIMEFRAME = "M30"
FROZEN_SWING_LOOKBACK = 40
FROZEN_RR = 2.0
STRATEGY_SEMANTIC_ID = "usdcad_sweep_bos_retest_v1"

#: Section 8: "$100,000 sleeve, 1x notional/reference sizing, no dynamic
#: resizing" -- reused, unchanged in spirit, from mean_reversion_h1_
#: development.py's own ``BASE_RESEARCH_UNITS`` convention. Every trade is
#: sized at this FIXED reference notional (base-currency units of the
#: pair); nothing here compounds equity into position size or normalizes by
#: volatility/risk.
BASE_RESEARCH_UNITS = Decimal("100000")

_M1_BAR_TIMEFRAME = "1-MINUTE"


class UsdCadSweepBosRetestError(ValueError):
    """Raised on any fail-closed guard specific to this module: a final
    -holdout timestamp, an overlapping precomputed trade, or an
    out-of-sequence Nautilus event."""


def reject_final_holdout(timestamp_ns: int) -> None:
    """Fail closed on any timestamp at or after the frozen final-holdout
    boundary. This promotion stage is authorized only through the
    already-observed VALIDATION period, never final holdout."""

    holdout_start_ns = int(HOLDOUT_START.timestamp() * 1_000_000_000)
    if timestamp_ns >= holdout_start_ns:
        raise UsdCadSweepBosRetestError(
            "final holdout is not accessible during execution promotion"
        )


# ---------------------------------------------------------------------------
# Precompute: the Alpha Lab ground truth, unchanged.
# ---------------------------------------------------------------------------


def precompute_alpha_lab_trades(
    *,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    bid_m1: pd.DataFrame,
    ask_m1: pd.DataFrame,
) -> tuple[tuple[Trade, ...], tuple[SkipRecord, ...]]:
    """The exact, unchanged Alpha Lab signal + execution computation for the
    frozen candidate: :func:`b2f1_sweep_bos_retest_signals` with the frozen
    ``swing_lookback``/``rr``, executed by :func:`simulate_trades` against
    the M1 BID/ASK stream. Nothing here decides anything -- it only pins
    the frozen parameters and forwards to the two reused functions."""

    events = b2f1_sweep_bos_retest_signals(
        high, low, close, swing_lookback=FROZEN_SWING_LOOKBACK, rr=FROZEN_RR
    )
    return simulate_trades(events, bid_m1, ask_m1)


@dataclass(frozen=True, slots=True)
class TradeInstruction:
    """One precomputed Alpha Lab :class:`Trade`, reduced to exactly what a
    mechanical Nautilus executor needs: when to submit an entry, when to
    submit the matching close, and the frozen reference prices to measure
    Nautilus's own native fills against (never used to decide the fill
    price itself -- that is entirely owned by Nautilus)."""

    trade_index: int
    direction: int
    entry_ns: int
    exit_ns: int
    stop_price: Decimal
    target_price: Decimal
    exit_reason: str
    alpha_lab_entry_price: Decimal
    alpha_lab_exit_price: Decimal


def trade_instructions_from_alpha_lab_trades(
    trades: tuple[Trade, ...],
) -> tuple[TradeInstruction, ...]:
    """Translate the Alpha Lab ``Trade`` tuple (already non-overlapping by
    construction -- ``simulate_trades`` never opens a new trade while one is
    open) into submission instants. Fails closed if that invariant is ever
    violated, rather than silently reordering or dropping a trade."""

    instructions: list[TradeInstruction] = []
    for index, trade in enumerate(trades):
        entry_ns = int(trade.entry_ts.value)
        exit_ns = int(trade.exit_ts.value)
        if exit_ns <= entry_ns:
            raise UsdCadSweepBosRetestError(
                f"trade {index} has a non-positive holding duration"
            )
        if instructions and entry_ns <= instructions[-1].exit_ns:
            raise UsdCadSweepBosRetestError(
                "Alpha Lab trades overlap; one-position-at-a-time invariant violated"
            )
        instructions.append(
            TradeInstruction(
                trade_index=index,
                direction=trade.direction,
                entry_ns=entry_ns,
                exit_ns=exit_ns,
                stop_price=Decimal(str(trade.stop_price)),
                target_price=Decimal(str(trade.target_price)),
                exit_reason=trade.exit_reason,
                alpha_lab_entry_price=Decimal(str(trade.entry_price)),
                alpha_lab_exit_price=Decimal(str(trade.exit_price)),
            )
        )
    return tuple(instructions)


# ---------------------------------------------------------------------------
# Records emitted by the executor, for parity assertions and reporting.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OrderSubmissionRecord:
    trade_index: int
    kind: str  # "entry" | "exit"
    ts_event: int
    side: str


@dataclass(frozen=True, slots=True)
class FillRecord:
    trade_index: int
    kind: str  # "entry" | "exit"
    fill_ns: int
    side: str
    last_px: str


@dataclass(frozen=True, slots=True)
class CompletedNautilusTrade:
    """One Nautilus-native realized round trip, for direct comparison
    against the Alpha Lab ``Trade`` at the same ``trade_index``."""

    trade_index: int
    direction: int
    entry_time_ns: int
    exit_time_ns: int
    entry_price: Decimal
    exit_price: Decimal
    quantity: Decimal
    realized_pnl: Decimal
    commission: Decimal
    initial_risk_quote: Decimal
    net_r: Decimal | None
    exit_reason: str


class UsdCadSweepBosRetestExecutor(Strategy):
    """Thin Nautilus strategy adapter. Decides nothing: mechanically
    submits the precomputed :class:`TradeInstruction` sequence's entries and
    exits against the live M1 BID/ASK stream, at 1x fixed reference-notional
    sizing (Section 8), and records every native submission/fill/closed
    -position event for parity checking and reporting."""

    def __new__(
        cls,
        *,
        instrument: CurrencyPair,
        instructions: tuple[TradeInstruction, ...],
        start_ns: int,
        end_exclusive_ns: int,
        initial_capital: Decimal,
    ) -> UsdCadSweepBosRetestExecutor:
        del instrument, instructions, start_ns, end_exclusive_ns, initial_capital
        return super().__new__(cls)

    def __init__(
        self,
        *,
        instrument: CurrencyPair,
        instructions: tuple[TradeInstruction, ...],
        start_ns: int,
        end_exclusive_ns: int,
        initial_capital: Decimal,
    ) -> None:
        super().__init__(
            StrategyConfig(
                strategy_id=StrategyId("USDCAD-SWEEP-BOS-RETEST-V1-001"),
                log_events=False,
                log_commands=False,
            )
        )
        self._instrument = instrument
        self._instrument_id = str(instrument.id)
        self._instructions = list(instructions)
        self._entry_index_by_ns = {
            instruction.entry_ns: idx for idx, instruction in enumerate(instructions)
        }
        self._exit_index_by_ns = {
            instruction.exit_ns: idx for idx, instruction in enumerate(instructions)
        }
        if len(self._entry_index_by_ns) != len(instructions) or len(
            self._exit_index_by_ns
        ) != len(instructions):
            raise UsdCadSweepBosRetestError(
                "two precomputed trades resolve onto the identical M1 observation"
            )
        self._start_ns = start_ns
        self._end_exclusive_ns = end_exclusive_ns
        self._pending_bar: dict[str, Bar] = {}
        self._open_trade_index: int | None = None
        self._entry_order_client_id: str | None = None
        self._entry_context: dict[str, TradeInstruction] = {}
        self._entry_fill_price: Decimal | None = None
        self._entry_fill_ns: int | None = None
        self._entry_quantity: Decimal | None = None
        self._trade_commissions = Decimal(0)

        self.submissions: list[OrderSubmissionRecord] = []
        self.fills: list[FillRecord] = []
        self.completed_trades: list[CompletedNautilusTrade] = []
        self.equity_points: list[EquityPoint] = [EquityPoint(start_ns, initial_capital)]
        self._current_day: date | None = None
        self._last_bar_ns: int | None = None

    def on_start(self) -> None:
        for side in ("BID", "ASK"):
            self.subscribe_bars(
                BarType.from_str(
                    f"{self._instrument_id}-{_M1_BAR_TIMEFRAME}-{side}-EXTERNAL"
                )
            )

    def on_stop(self) -> None:
        if self._last_bar_ns is not None:
            self._append_equity_mark(self._last_bar_ns)

    def on_bar(self, bar: Bar) -> None:
        if bar.ts_event < self._start_ns or bar.ts_event >= self._end_exclusive_ns:
            return
        reject_final_holdout(bar.ts_event)
        side = bar.bar_type.spec.price_type.name
        self._pending_bar[side] = bar
        if "BID" not in self._pending_bar or "ASK" not in self._pending_bar:
            return
        bid_bar = self._pending_bar["BID"]
        ask_bar = self._pending_bar["ASK"]
        if bid_bar.ts_event != ask_bar.ts_event:
            return
        self._pending_bar = {}
        ts_event = bid_bar.ts_event

        day = datetime.fromtimestamp(ts_event / 1_000_000_000, tz=UTC).date()
        if self._current_day is not None and day != self._current_day:
            assert self._last_bar_ns is not None
            self._append_equity_mark(self._last_bar_ns)
        self._current_day = day

        entry_index = self._entry_index_by_ns.get(ts_event)
        if entry_index is not None:
            self._submit_entry(self._instructions[entry_index])

        exit_index = self._exit_index_by_ns.get(ts_event)
        if exit_index is not None and self._open_trade_index == exit_index:
            self._submit_exit(self._instructions[exit_index])

        self._last_bar_ns = ts_event

    def _submit_entry(self, instruction: TradeInstruction) -> None:
        if self._open_trade_index is not None:
            raise UsdCadSweepBosRetestError(
                "cannot open a new trade while one is already open"
            )
        side = (
            OrderSide.BUY if instruction.direction == DIRECTION_LONG else OrderSide.SELL
        )
        quantity = self._instrument.make_qty(float(BASE_RESEARCH_UNITS))
        order = self.order_factory.market(
            instrument_id=InstrumentId.from_str(self._instrument_id),
            order_side=side,
            quantity=quantity,
            tags=(
                STRATEGY_SEMANTIC_ID,
                f"trade_index={instruction.trade_index}",
                f"stop_price={instruction.stop_price}",
                f"target_price={instruction.target_price}",
            ),
        )
        self._entry_order_client_id = order.client_order_id.value
        self._entry_context[order.client_order_id.value] = instruction
        self._open_trade_index = instruction.trade_index
        self.submissions.append(
            OrderSubmissionRecord(
                trade_index=instruction.trade_index,
                kind="entry",
                ts_event=instruction.entry_ns,
                side=side.name,
            )
        )
        self.submit_order(order)

    def _submit_exit(self, instruction: TradeInstruction) -> None:
        self.submissions.append(
            OrderSubmissionRecord(
                trade_index=instruction.trade_index,
                kind="exit",
                ts_event=instruction.exit_ns,
                side=(
                    OrderSide.SELL.name
                    if instruction.direction == DIRECTION_LONG
                    else OrderSide.BUY.name
                ),
            )
        )
        self.close_all_positions(InstrumentId.from_str(self._instrument_id))

    def on_order_filled(self, event: OrderFilled) -> None:
        client_id = event.client_order_id.value
        instruction = self._entry_context.get(client_id)
        if instruction is not None and client_id == self._entry_order_client_id:
            kind = "entry"
            self._entry_fill_price = event.last_px.as_decimal()
            self._entry_fill_ns = event.ts_event
            self._entry_quantity = event.last_qty.as_decimal()
            self._trade_commissions = Decimal(0)
            self._entry_order_client_id = None
        else:
            kind = "exit"
            active = (
                self._instructions[self._open_trade_index]
                if self._open_trade_index is not None
                else None
            )
            instruction = active
        if event.commission is not None:
            self._trade_commissions += event.commission.as_decimal()
        if instruction is None:
            raise UsdCadSweepBosRetestError(
                "fill event lacks a precomputed trade context"
            )
        self.fills.append(
            FillRecord(
                trade_index=instruction.trade_index,
                kind=kind,
                fill_ns=event.ts_event,
                side=event.order_side.name,
                last_px=str(event.last_px),
            )
        )

    def on_position_closed(self, event: PositionClosed) -> None:
        if event.instrument_id != InstrumentId.from_str(self._instrument_id):
            return
        if self._open_trade_index is None:
            raise UsdCadSweepBosRetestError(
                "position closed with no open trade recorded"
            )
        instruction = self._instructions[self._open_trade_index]
        if (
            self._entry_fill_price is None
            or self._entry_fill_ns is None
            or self._entry_quantity is None
            or event.realized_pnl is None
            or event.ts_closed is None
        ):
            raise UsdCadSweepBosRetestError(
                "closed position has incomplete fill evidence"
            )
        risk_per_unit = abs(instruction.alpha_lab_entry_price - instruction.stop_price)
        initial_risk_quote = self._entry_quantity * risk_per_unit
        realized_pnl = event.realized_pnl.as_decimal()
        net_r = realized_pnl / initial_risk_quote if initial_risk_quote > 0 else None
        self.completed_trades.append(
            CompletedNautilusTrade(
                trade_index=instruction.trade_index,
                direction=instruction.direction,
                entry_time_ns=self._entry_fill_ns,
                exit_time_ns=event.ts_closed,
                entry_price=self._entry_fill_price,
                exit_price=Decimal(str(event.avg_px_close)),
                quantity=self._entry_quantity,
                realized_pnl=realized_pnl,
                commission=self._trade_commissions,
                initial_risk_quote=initial_risk_quote,
                net_r=net_r,
                exit_reason=instruction.exit_reason,
            )
        )
        self._open_trade_index = None
        self._entry_fill_price = None
        self._entry_fill_ns = None
        self._entry_quantity = None
        self._trade_commissions = Decimal(0)

    def _append_equity_mark(self, ts_event: int) -> None:
        # Reuse Nautilus's OWN currency-converted total account equity
        # (nautilus_trader.portfolio.Portfolio.equity), exactly the pattern
        # already established and regression-tested in
        # mean_reversion_h1_development.py's own
        # ``_MeanReversionH1Executor._append_equity_mark`` -- never a manual
        # quote-currency P&L sum (Section 12).
        account_currency = "USD"
        equity_by_currency = self.portfolio.equity(venue=self._instrument.id.venue)
        equity_money = equity_by_currency.get(Currency.from_str(account_currency))
        if equity_money is None:
            raise UsdCadSweepBosRetestError(
                f"native portfolio equity unavailable in {account_currency}"
            )
        equity = equity_money.as_decimal()
        if ts_event > self.equity_points[-1].information_time_ns:
            self.equity_points.append(EquityPoint(ts_event, equity))
