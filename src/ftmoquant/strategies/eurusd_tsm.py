"""Causal H1/H4 EUR/USD time-series-momentum family for generic G1 research.

The family reuses FTMOQuant's completed BID/ASK pair, raw directional target,
and hold-until-changed conventions. It deliberately contains no execution,
cost, P&L, historical-data loading, or FTMO account logic.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ftmoquant.research.eurusd_tsm_spec import (
    EurusdTsmSpec,
    TsmTimeframeGrid,
    load_eurusd_tsm_spec,
)
from ftmoquant.research.g1.family import (
    FamilyContractError,
    FamilyMetadata,
    StrategyFamily,
)
from ftmoquant.research.g1.parameter_space import (
    CategoricalParameter,
    ParameterMapping,
    ParameterSpace,
    ParameterValue,
    canonical_parameter_json,
)
from ftmoquant.research.g1.sessions import SessionId
from ftmoquant.strategies.trend_pullback import CompletedPair, Timeframe
from ftmoquant.strategies.ts_momentum import RawDirectionalTarget


class EurusdTsmValidationError(ValueError):
    """Raised when a signal input or conditional parameter cell is invalid."""


@dataclass(frozen=True, slots=True)
class EurusdTsmParameters:
    timeframe: str
    lookback_bars: int
    deadband: float
    refresh_interval_bars: int


@dataclass(frozen=True, slots=True)
class EurusdTsmSignal:
    """One changed raw target available only at a completed-bar information time."""

    target: RawDirectionalTarget
    trailing_log_return: float
    trailing_return_volatility: float
    normalized_trailing_return: float
    signal_event_ns: int
    signal_information_ns: int
    target_changed: bool


class EurusdTsmState:
    """Incremental signal state with causal warm-up and refresh scheduling."""

    def __init__(self, parameters: EurusdTsmParameters) -> None:
        self.parameters = parameters
        self._timeframe = _timeframe(parameters.timeframe)
        self._volatility_window = max(20, parameters.lookback_bars)
        self._closes: deque[float] = deque(maxlen=self._volatility_window + 1)
        self._last_event_ns: int | None = None
        self._last_information_ns: int | None = None
        self._bars_since_refresh: int | None = None
        self._latest_target: RawDirectionalTarget | None = None

    @property
    def required_close_count(self) -> int:
        return self._volatility_window + 1

    @property
    def latest_target(self) -> RawDirectionalTarget | None:
        return self._latest_target

    def on_bar(self, pair: CompletedPair) -> EurusdTsmSignal | None:
        """Consume one completed pair and possibly emit one changed raw target."""

        refresh = self.on_target_refresh(pair)
        return refresh if refresh is not None and refresh.target_changed else None

    def on_target_refresh(self, pair: CompletedPair) -> EurusdTsmSignal | None:
        """Consume a bar and expose every valid scheduled target refresh."""

        if pair.timeframe is not self._timeframe:
            raise EurusdTsmValidationError("completed bar timeframe is not configured")
        if self._last_event_ns is not None and pair.ts_event <= self._last_event_ns:
            raise EurusdTsmValidationError(
                "completed bars must be strictly event ordered"
            )
        if (
            self._last_information_ns is not None
            and pair.info_time_ns <= self._last_information_ns
        ):
            raise EurusdTsmValidationError(
                "completed bars must be strictly information ordered"
            )
        self._last_event_ns = pair.ts_event
        self._last_information_ns = pair.info_time_ns
        if not pair.contiguous:
            self._closes.clear()
            self._bars_since_refresh = None
            return None

        close = float(pair.midpoint.close)
        if not math.isfinite(close) or close <= 0.0:
            raise EurusdTsmValidationError("midpoint close must be finite and positive")
        self._closes.append(close)
        if len(self._closes) < self.required_close_count:
            return None
        if self._bars_since_refresh is None:
            self._bars_since_refresh = 0
        else:
            self._bars_since_refresh += 1
            if self._bars_since_refresh < self.parameters.refresh_interval_bars:
                return None
            self._bars_since_refresh = 0

        closes = np.asarray(self._closes, dtype=np.float64)
        one_bar_returns = np.diff(np.log(closes))
        volatility = float(one_bar_returns.std(ddof=1))
        if not math.isfinite(volatility) or volatility <= 0.0:
            return None
        lookback = self.parameters.lookback_bars
        trailing_return = math.log(closes[-1] / closes[-(lookback + 1)])
        denominator = volatility * math.sqrt(lookback)
        normalized = trailing_return / denominator
        if not all(
            math.isfinite(item) for item in (trailing_return, denominator, normalized)
        ):
            return None
        target = (
            RawDirectionalTarget.LONG
            if normalized > self.parameters.deadband
            else RawDirectionalTarget.SHORT
            if normalized < -self.parameters.deadband
            else RawDirectionalTarget.FLAT
        )
        changed = target is not self._latest_target
        self._latest_target = target
        return EurusdTsmSignal(
            target=target,
            trailing_log_return=trailing_return,
            trailing_return_volatility=denominator,
            normalized_trailing_return=normalized,
            signal_event_ns=pair.ts_event,
            signal_information_ns=pair.info_time_ns,
            target_changed=changed,
        )


class EurusdTsmFamily(
    StrategyFamily[Sequence[CompletedPair], tuple[EurusdTsmSignal, ...]]
):
    """The frozen conditional 90-cell ``eurusd_tsm_v1`` family adapter."""

    def __init__(self, spec: EurusdTsmSpec | None = None) -> None:
        self.spec = load_eurusd_tsm_spec() if spec is None else spec
        lookbacks = tuple(
            dict.fromkeys(
                lookback
                for grid in self.spec.timeframe_grids
                for lookback in grid.lookback_bars
            )
        )
        refresh = tuple(
            dict.fromkeys(
                interval
                for grid in self.spec.timeframe_grids
                for interval in grid.refresh_interval_bars
            )
        )
        self._parameter_space = ParameterSpace(
            (
                CategoricalParameter(
                    "timeframe",
                    tuple(grid.timeframe for grid in self.spec.timeframe_grids),
                ),
                CategoricalParameter("lookback_bars", lookbacks),
                CategoricalParameter("deadband", self.spec.deadbands),
                CategoricalParameter("refresh_interval_bars", refresh),
            )
        )

    @property
    def metadata(self) -> FamilyMetadata:
        return FamilyMetadata(
            family_id=self.spec.family_id,
            version=self.spec.version,
            economic_hypothesis=self.spec.economic_hypothesis,
            supported_timeframes=tuple(
                item.timeframe for item in self.spec.timeframe_grids
            ),
            eligible_sessions=(SessionId.ALL,),
        )

    @property
    def parameter_space(self) -> ParameterSpace:
        return self._parameter_space

    def validate_parameters(self, parameters: ParameterMapping) -> None:
        super().validate_parameters(parameters)
        parsed = _parameters(parameters)
        grid = self._grid(parsed.timeframe)
        if parsed.lookback_bars not in grid.lookback_bars:
            raise EurusdTsmValidationError("lookback is invalid for timeframe")
        if parsed.refresh_interval_bars not in grid.refresh_interval_bars:
            raise EurusdTsmValidationError("refresh interval is invalid for timeframe")

    def enumerate_parameters(self) -> tuple[dict[str, ParameterValue], ...]:
        configurations_list: list[dict[str, ParameterValue]] = []
        for grid in self.spec.timeframe_grids:
            for lookback in grid.lookback_bars:
                for deadband in self.spec.deadbands:
                    for refresh in grid.refresh_interval_bars:
                        configurations_list.append(
                            {
                                "timeframe": grid.timeframe,
                                "lookback_bars": lookback,
                                "deadband": deadband,
                                "refresh_interval_bars": refresh,
                            }
                        )
        configurations = tuple(configurations_list)
        for parameters in configurations:
            self.validate_parameters(parameters)
        identities = tuple(canonical_parameter_json(item) for item in configurations)
        if len(configurations) != self.spec.expected_unique_trial_count:
            raise FamilyContractError("family grid count differs from preregistration")
        if len(set(identities)) != len(identities):
            raise FamilyContractError("family grid contains duplicate configurations")
        return configurations

    def create_state(self, parameters: ParameterMapping) -> EurusdTsmState:
        self.validate_parameters(parameters)
        return EurusdTsmState(_parameters(parameters))

    def build_signals(
        self, data: Sequence[CompletedPair], parameters: ParameterMapping
    ) -> tuple[EurusdTsmSignal, ...]:
        state = self.create_state(parameters)
        return tuple(
            signal for pair in data if (signal := state.on_bar(pair)) is not None
        )

    def build_target_refreshes(
        self, data: Sequence[CompletedPair], parameters: ParameterMapping
    ) -> tuple[EurusdTsmSignal, ...]:
        """Return all risk-sizing refreshes while preserving changed-alpha output."""

        state = self.create_state(parameters)
        return tuple(
            refresh
            for pair in data
            if (refresh := state.on_target_refresh(pair)) is not None
        )

    def neighbours(
        self, parameters: ParameterMapping
    ) -> tuple[Mapping[str, ParameterValue], ...]:
        self.validate_parameters(parameters)
        parsed = _parameters(parameters)
        grid = self._grid(parsed.timeframe)
        dimensions: tuple[tuple[str, tuple[ParameterValue, ...]], ...] = (
            ("lookback_bars", grid.lookback_bars),
            ("deadband", self.spec.deadbands),
            ("refresh_interval_bars", grid.refresh_interval_bars),
        )
        result: list[Mapping[str, ParameterValue]] = []
        base = dict(parameters)
        for name, ordered in dimensions:
            current = base[name]
            index = ordered.index(current)
            for adjacent in (index - 1, index + 1):
                if 0 <= adjacent < len(ordered):
                    result.append({**base, name: ordered[adjacent]})
        return tuple(result)

    def _grid(self, timeframe: str) -> TsmTimeframeGrid:
        for grid in self.spec.timeframe_grids:
            if grid.timeframe == timeframe:
                return grid
        raise EurusdTsmValidationError("unknown timeframe")


def _parameters(parameters: ParameterMapping) -> EurusdTsmParameters:
    timeframe = parameters.get("timeframe")
    lookback = parameters.get("lookback_bars")
    deadband = parameters.get("deadband")
    refresh = parameters.get("refresh_interval_bars")
    if (
        not isinstance(timeframe, str)
        or isinstance(lookback, bool)
        or not isinstance(lookback, int)
        or not isinstance(deadband, float)
        or isinstance(refresh, bool)
        or not isinstance(refresh, int)
    ):
        raise EurusdTsmValidationError("parameter types are not exact")
    return EurusdTsmParameters(timeframe, lookback, deadband, refresh)


def _timeframe(value: str) -> Timeframe:
    if value == "H1":
        return Timeframe.HOUR
    if value == "H4":
        return Timeframe.FOUR_HOURS
    raise EurusdTsmValidationError("timeframe must be H1 or H4")
