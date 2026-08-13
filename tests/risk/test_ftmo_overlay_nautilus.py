from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from nautilus_trader.backtest import BacktestEngine, BacktestEngineConfig
from nautilus_trader.common import LoggerConfig, LogLevel
from nautilus_trader.model import (
    AccountType,
    BookType,
    Money,
    OmsType,
    OrderFilled,
    OrderSide,
    PositionOpened,
    Price,
    Quantity,
    QuoteTick,
    StrategyId,
)
from nautilus_trader.trading import Strategy, StrategyConfig

from ftmoquant.backtest.nautilus_probe import EURUSD, SIM, USD, _instrument
from ftmoquant.prop_rules import EvaluationPhase, load_prop_rule_set
from ftmoquant.risk import (
    FtmoRuntimeConfig,
    NautilusAccountSnapshotSource,
    NautilusFtmoOverlay,
)

CONFIG_PATH = Path("config/prop/ftmo_2step_swing_2026-08.yaml")


class _OverlayProbe(Strategy):
    def __init__(self, trade: bool) -> None:
        super().__init__(
            StrategyConfig(
                strategy_id=StrategyId("G04-OVERLAY-001"),
                log_events=False,
                log_commands=False,
            )
        )
        self.trade = trade
        self.quote_count = 0
        self.overlay: NautilusFtmoOverlay | None = None

    def on_start(self) -> None:
        source = NautilusAccountSnapshotSource(
            cache=self.cache,
            portfolio=self.portfolio,
            venue=SIM,
            currency=USD,
        )
        self.overlay = NautilusFtmoOverlay(
            runtime=FtmoRuntimeConfig(
                initial_capital=Decimal("100000"),
                currency="USD",
                active_phase=EvaluationPhase.CHALLENGE,
            ),
            rules=load_prop_rule_set(CONFIG_PATH),
            clock=self.clock,
            account_source=source,
        )
        self.subscribe_quotes(EURUSD)

    def on_quote(self, quote: QuoteTick) -> None:
        self.quote_count += 1
        if not self.trade:
            return
        if self.quote_count == 1:
            self.submit_order(
                self.order_factory.market(
                    instrument_id=EURUSD,
                    order_side=OrderSide.BUY,
                    quantity=Quantity.from_int(100_000),
                )
            )
        elif self.quote_count == 2:
            self.close_all_positions(EURUSD)

    def on_order_filled(self, event: OrderFilled) -> None:
        assert self.overlay is not None
        self.overlay.on_order_filled(event)

    def on_position_opened(self, event: PositionOpened) -> None:
        assert self.overlay is not None
        self.overlay.on_position_opened(event)


def test_nautilus_clock_fires_midnight_reset_without_trade() -> None:
    strategy, engine = _run_engine(trade=False)
    try:
        assert strategy.overlay is not None
        assert strategy.overlay.state.current_trading_day.isoformat() == "2026-01-05"
        assert strategy.overlay.state.midnight_reset_balance == Decimal("100000.00")
        assert strategy.overlay.state.trading_days == frozenset()
    finally:
        engine.dispose()


def test_native_position_and_fill_events_feed_overlay() -> None:
    strategy, engine = _run_engine(trade=True)
    try:
        assert strategy.overlay is not None
        assert len(strategy.overlay.state.trading_days) == 1
        source = NautilusAccountSnapshotSource(
            cache=engine.cache,
            portfolio=engine.portfolio,
            venue=SIM,
            currency=USD,
        )
        snapshot = source.snapshot()
        assert snapshot.open_position_ids == frozenset()
        assert snapshot.balance == snapshot.equity
        assert snapshot.balance != Decimal("100000")
    finally:
        engine.dispose()


def _run_engine(trade: bool) -> tuple[_OverlayProbe, BacktestEngine]:
    strategy = _OverlayProbe(trade)
    engine = BacktestEngine(
        BacktestEngineConfig(
            logging=LoggerConfig(
                stdout_level=LogLevel.OFF,
                print_config=False,
                bypass_logging=True,
            ),
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
    engine.add_data(_quotes_across_prague_midnight())
    engine.add_strategy(strategy)
    engine.run()
    return strategy, engine


def _quotes_across_prague_midnight() -> list[QuoteTick]:
    timestamps = (
        _to_ns(datetime(2026, 1, 4, 22, 59, 30, tzinfo=UTC)),
        _to_ns(datetime(2026, 1, 4, 23, 0, 30, tzinfo=UTC)),
        _to_ns(datetime(2026, 1, 4, 23, 1, tzinfo=UTC)),
    )
    prices = (
        ("1.10000", "1.10002"),
        ("1.10100", "1.10102"),
        ("1.10200", "1.10202"),
    )
    return [
        QuoteTick(
            instrument_id=EURUSD,
            bid_price=Price.from_str(bid),
            ask_price=Price.from_str(ask),
            bid_size=Quantity.from_int(1_000_000),
            ask_size=Quantity.from_int(1_000_000),
            ts_event=timestamp,
            ts_init=timestamp,
        )
        for timestamp, (bid, ask) in zip(timestamps, prices, strict=True)
    ]


def _to_ns(timestamp: datetime) -> int:
    return int(timestamp.timestamp() * 1_000_000_000)
