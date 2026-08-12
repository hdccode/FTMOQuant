"""Deterministic NautilusTrader v2 adoption probe.

This module exercises engine plumbing only. It contains no trading signal or
FTMO overlay logic.
"""

from dataclasses import dataclass
from decimal import Decimal
from importlib.metadata import version
from typing import cast

from nautilus_trader.backtest import BacktestEngine, BacktestEngineConfig
from nautilus_trader.common import LoggerConfig, LogLevel
from nautilus_trader.model import (
    AccountType,
    BookType,
    Currency,
    CurrencyPair,
    InstrumentId,
    MarginAccount,
    Money,
    OmsType,
    OrderFilled,
    OrderSide,
    Position,
    Price,
    Quantity,
    QuoteTick,
    StrategyId,
    Symbol,
    Venue,
)
from nautilus_trader.trading import Strategy, StrategyConfig

NAUTILUS_VERSION = "2.0.0rc2"
INSTALLED_NAUTILUS_VERSION = version("nautilus_trader")
EURUSD = InstrumentId.from_str("EUR/USD.SIM")
SIM = Venue("SIM")
USD = Currency.from_str("USD")
EUR = Currency.from_str("EUR")
START_NS = 1_786_512_000_000_000_000


@dataclass(frozen=True, slots=True)
class NautilusProbeResult:
    """Stable, comparable evidence produced by one synthetic engine run."""

    engine_version: str
    market_timestamps_ns: tuple[int, ...]
    order_timestamps_ns: tuple[int, ...]
    fill_timestamps_ns: tuple[int, ...]
    fill_prices: tuple[Decimal, ...]
    position_ids: tuple[str, ...]
    fees: tuple[Decimal, ...]
    realized_pnl: Decimal
    balance: Decimal
    equity: Decimal
    account_state_timestamps_ns: tuple[int, ...]
    report_rows: tuple[tuple[str, int], ...]


class _RoundTripProbe(Strategy):
    """Scripted fixture: open on quote one and close on quote two."""

    def __init__(self) -> None:
        super().__init__(
            StrategyConfig(
                strategy_id=StrategyId("G0-PROBE-001"),
                log_events=False,
                log_commands=False,
            )
        )
        self._quote_count = 0

    def on_start(self) -> None:
        self.subscribe_quotes(EURUSD)

    def on_quote(self, quote: QuoteTick) -> None:
        self._quote_count += 1
        if self._quote_count == 1:
            order = self.order_factory.market(
                instrument_id=EURUSD,
                order_side=OrderSide.BUY,
                quantity=Quantity.from_int(100_000),
            )
            self.submit_order(order)
        elif self._quote_count == 2:
            self.close_all_positions(EURUSD)


def run_synthetic_eurusd_probe() -> NautilusProbeResult:
    """Run one deterministic EUR/USD round trip and return normalized evidence."""

    if INSTALLED_NAUTILUS_VERSION != NAUTILUS_VERSION:
        raise RuntimeError(
            f"expected NautilusTrader {NAUTILUS_VERSION}, "
            f"found {INSTALLED_NAUTILUS_VERSION}"
        )

    quotes = _quotes()
    engine = _build_engine(quotes)
    try:
        engine.run()
        return _collect_result(engine, quotes)
    finally:
        engine.dispose()


def _build_engine(quotes: tuple[QuoteTick, ...]) -> BacktestEngine:
    logger = LoggerConfig(
        stdout_level=LogLevel.OFF,
        print_config=False,
        bypass_logging=True,
    )
    engine = BacktestEngine(
        BacktestEngineConfig(
            logging=logger,
            bypass_logging=True,
            run_analysis=False,
        )
    )
    engine.add_venue(
        venue=SIM,
        oms_type=OmsType.HEDGING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money.from_str("100000 USD")],
        base_currency=USD,
        default_leverage=Decimal("30"),
        book_type=BookType.L1_MBP,
        use_random_ids=False,
    )
    engine.add_instrument(_instrument())
    engine.add_data(quotes)
    engine.add_strategy(_RoundTripProbe())
    return engine


def _instrument() -> CurrencyPair:
    return CurrencyPair(
        instrument_id=EURUSD,
        raw_symbol=Symbol("EUR/USD"),
        base_currency=EUR,
        quote_currency=USD,
        price_precision=5,
        size_precision=0,
        price_increment=Price.from_str("0.00001"),
        size_increment=Quantity.from_int(1),
        ts_event=0,
        ts_init=0,
        margin_init=Decimal("0.03"),
        margin_maint=Decimal("0.03"),
        maker_fee=Decimal("0.00002"),
        taker_fee=Decimal("0.00002"),
    )


def _quotes() -> tuple[QuoteTick, ...]:
    prices = (
        ("1.10000", "1.10002"),
        ("1.10100", "1.10102"),
        ("1.10200", "1.10202"),
    )
    return tuple(
        QuoteTick(
            instrument_id=EURUSD,
            bid_price=Price.from_str(bid),
            ask_price=Price.from_str(ask),
            bid_size=Quantity.from_int(1_000_000),
            ask_size=Quantity.from_int(1_000_000),
            ts_event=START_NS + index * 1_000_000_000,
            ts_init=START_NS + index * 1_000_000_000,
        )
        for index, (bid, ask) in enumerate(prices)
    )


def _collect_result(
    engine: BacktestEngine,
    quotes: tuple[QuoteTick, ...],
) -> NautilusProbeResult:
    orders_report = engine.generate_orders_report()
    fills_report = engine.generate_fills_report()
    positions_report = engine.generate_positions_report()
    account_report = engine.generate_account_report(venue=SIM)
    _validate_report_columns(
        orders_report,
        fills_report,
        positions_report,
        account_report,
    )

    positions = cast(list[Position], engine.cache.positions())
    if len(positions) != 1:
        raise RuntimeError(f"expected one position, found {len(positions)}")
    position = positions[0]
    fills = position.events()

    account = cast(MarginAccount | None, engine.cache.account_for_venue(SIM))
    if account is None:
        raise RuntimeError("Nautilus account state is unavailable")
    balance = account.balance_total(USD)
    if balance is None:
        raise RuntimeError("Nautilus USD balance is unavailable")

    equity_by_currency = cast(dict[Currency, Money], engine.portfolio.equity(SIM))
    equity = equity_by_currency.get(USD)
    if equity is None:
        raise RuntimeError("Nautilus USD equity is unavailable")

    realized_pnl = position.realized_pnl
    if realized_pnl is None:
        raise RuntimeError("closed position has no realized P/L")

    return NautilusProbeResult(
        engine_version=INSTALLED_NAUTILUS_VERSION,
        market_timestamps_ns=tuple(quote.ts_event for quote in quotes),
        order_timestamps_ns=tuple(fill.ts_init for fill in fills),
        fill_timestamps_ns=tuple(fill.ts_event for fill in fills),
        fill_prices=tuple(fill.last_px.as_decimal() for fill in fills),
        position_ids=tuple(_position_id(fill) for fill in fills),
        fees=tuple(_commission(fill) for fill in fills),
        realized_pnl=realized_pnl.as_decimal(),
        balance=balance.as_decimal(),
        equity=equity.as_decimal(),
        account_state_timestamps_ns=tuple(event.ts_event for event in account.events),
        report_rows=(
            ("orders", len(orders_report)),
            ("fills", len(fills_report)),
            ("positions", len(positions_report)),
            ("account", len(account_report)),
        ),
    )


def _position_id(fill: OrderFilled) -> str:
    if fill.position_id is None:
        raise RuntimeError("fill has no position ID")
    return fill.position_id.value


def _commission(fill: OrderFilled) -> Decimal:
    if fill.commission is None:
        raise RuntimeError("fill has no commission")
    return fill.commission.as_decimal()


def _validate_report_columns(
    orders_report: object,
    fills_report: object,
    positions_report: object,
    account_report: object,
) -> None:
    required = (
        (orders_report, {"ts_init", "status", "filled_qty", "commissions"}),
        (fills_report, {"ts_event", "position_id", "commission"}),
        (positions_report, {"ts_opened", "ts_closed", "realized_pnl"}),
        (account_report, {"total", "locked", "free", "base_currency"}),
    )
    for report, columns in required:
        actual = set(cast(list[str], getattr(report, "columns")))
        if not columns <= actual:
            missing = ", ".join(sorted(columns - actual))
            raise RuntimeError(f"Nautilus report is missing columns: {missing}")
