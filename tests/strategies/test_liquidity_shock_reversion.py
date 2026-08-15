# ruff: noqa: E501

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from ftmoquant.research.liquidity_shock_reversion_spec import (
    load_liquidity_shock_reversion_spec,
)
from ftmoquant.research.stage_g import (
    DEVELOPMENT_END_EXCLUSIVE,
    DEVELOPMENT_START,
    FROZEN_INSTRUMENT_IDS,
    DevelopmentResearchContext,
    InstrumentBarObservation,
    StageGValidationError,
    SynchronizedClockFrame,
    frozen_development_folds,
)
from ftmoquant.strategies.liquidity_shock_reversion import (
    LiquidityShockReversionDevelopmentFold,
    LiquidityShockReversionStateMachine,
    MinuteMidpointClose,
    RawDirectionalTarget,
)

EURUSD, GBPUSD = FROZEN_INSTRUMENT_IDS
START = datetime(2020, 1, 6, tzinfo=UTC)


def test_exact_prior_window_strict_shock_and_inversion() -> None:
    machine = _machine()
    positive = _seed_and_shock(machine, EURUSD, Decimal("0.004"))
    assert positive is None  # the prior 60-return scale excludes the current return

    machine = _machine()
    positive = _seed_and_shock(machine, EURUSD, Decimal("0.006"))
    assert positive is not None and positive.target is RawDirectionalTarget.SHORT
    machine = _machine()
    negative = _seed_and_shock(machine, EURUSD, Decimal("-0.006"))
    assert negative is not None and negative.target is RawDirectionalTarget.LONG


def test_insufficient_and_zero_scale_are_no_signal() -> None:
    machine = _machine()
    assert _close(machine, EURUSD, 0, Decimal("1")) is None
    for index in range(1, 60):
        assert _close(machine, EURUSD, index, Decimal("0.001")) is None
    assert _close(machine, EURUSD, 60, Decimal("0.1")) is None

    machine = _machine()
    assert _fixed_close(machine, EURUSD, 0, Decimal("1")) is None
    for index in range(1, 61):
        assert _fixed_close(machine, EURUSD, index, Decimal("1")) is None
    assert _fixed_close(machine, EURUSD, 61, Decimal("1.1")) is None


def test_strict_later_execution_exact_hold_and_overlapping_shocks_ignored() -> None:
    machine = _machine()
    signal = _seed_and_shock(machine, EURUSD, Decimal("0.006"))
    assert signal is not None
    assert machine.on_execution_frame(_frame(signal.signal_information_time_utc)) == ()
    entry = machine.on_execution_frame(
        _frame(signal.signal_information_time_utc + timedelta(minutes=1))
    )
    assert [item.target for item in entry] == [RawDirectionalTarget.SHORT]
    assert _close(machine, EURUSD, 62, Decimal("0.1")) is None
    for minute in range(2, 17):
        assert (
            machine.on_execution_frame(
                _frame(signal.signal_information_time_utc + timedelta(minutes=minute))
            )
            == ()
        )
    assert machine.active_targets == ((EURUSD, RawDirectionalTarget.SHORT),)
    exit_ = machine.on_execution_frame(
        _frame(signal.signal_information_time_utc + timedelta(minutes=17))
    )
    assert [item.target for item in exit_] == [RawDirectionalTarget.FLAT]
    assert machine.active_targets == ()


def test_instruments_are_independent_and_deterministic() -> None:
    def run() -> tuple[object, ...]:
        machine = _machine()
        eur = _seed_and_shock(machine, EURUSD, Decimal("0.006"))
        gbp = _seed_and_shock(machine, GBPUSD, Decimal("-0.006"))
        assert eur is not None and gbp is not None
        return (
            eur,
            gbp,
            machine.on_execution_frame(_frame(START + timedelta(minutes=63))),
        )

    first, second = run(), run()
    assert first == second
    assert first[0].instrument_id == EURUSD and first[1].instrument_id == GBPUSD
    assert {item.target for item in first[2]} == {
        RawDirectionalTarget.LONG,
        RawDirectionalTarget.SHORT,
    }


def test_fold_boundary_resets_and_never_admits_outside_fold() -> None:
    fold = frozen_development_folds().folds[0]
    context = DevelopmentResearchContext(
        universe=None,
        artifacts=(),
        start_utc=DEVELOPMENT_START,
        end_exclusive_utc=DEVELOPMENT_END_EXCLUSIVE,
    )  # type: ignore[arg-type]
    candidate = LiquidityShockReversionDevelopmentFold(
        context, fold, load_liquidity_shock_reversion_spec()
    )
    candidate.state._active[EURUSD] = (
        RawDirectionalTarget.LONG
    )  # test observable boundary reset
    with pytest.raises(StageGValidationError, match="outside the frozen fold horizon"):
        candidate.on_minute_close(
            MinuteMidpointClose(
                EURUSD,
                fold.compare_end_exclusive_utc - timedelta(minutes=1),
                fold.compare_end_exclusive_utc,
                Decimal("1"),
            )
        )
    assert candidate.state.active_targets == ()


def _machine() -> LiquidityShockReversionStateMachine:
    return LiquidityShockReversionStateMachine(load_liquidity_shock_reversion_spec())


def _seed_and_shock(
    machine: LiquidityShockReversionStateMachine, instrument: str, shock: Decimal
):
    assert _close(machine, instrument, 0, Decimal("1")) is None
    for index in range(1, 61):
        assert _close(machine, instrument, index, Decimal("0.001")) is None
    return _close(machine, instrument, 61, shock)


def _close(
    machine: LiquidityShockReversionStateMachine,
    instrument: str,
    index: int,
    log_return: Decimal,
):
    price = log_return.exp() if index == 0 else _price_at(index, log_return)
    information = START + timedelta(minutes=index + 1)
    return machine.on_minute_close(
        MinuteMidpointClose(
            instrument, information - timedelta(minutes=1), information, price
        )
    )


def _fixed_close(
    machine: LiquidityShockReversionStateMachine,
    instrument: str,
    index: int,
    price: Decimal,
):
    information = START + timedelta(minutes=index + 1)
    return machine.on_minute_close(
        MinuteMidpointClose(
            instrument, information - timedelta(minutes=1), information, price
        )
    )


def _price_at(index: int, current_return: Decimal) -> Decimal:
    if index <= 60:
        return (Decimal("0.001") * Decimal(index)).exp()
    return (Decimal("0.060") + current_return).exp()


def _frame(information: datetime) -> SynchronizedClockFrame:
    event = information - timedelta(minutes=1)
    observations = tuple(
        InstrumentBarObservation(
            item, event, information, Decimal("1"), Decimal("1.0002")
        )
        for item in FROZEN_INSTRUMENT_IDS
    )
    return SynchronizedClockFrame(event, information, observations, True)
