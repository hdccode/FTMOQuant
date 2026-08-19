"""Path extraction for the frozen usdcad_sweep_bos_retest_v1 execution
artifacts (DEVELOPMENT and, read-only/diagnostic-only, VALIDATION).

Builds the smallest sufficient trade-level path representation that still
lets :mod:`state_machine` replay FTMO daily-loss/max-loss semantics exactly
(for stop-exits) or with a proven conservative bound (for target-exits) —
see the module docstring in ``state_machine.py`` and
``docs/research/ftmo_pass_probability_v1_design.md`` for the derivation.

No M1/M30 bar re-derivation is required: the frozen execution's own
``trades.csv`` already records, per trade, the realized dollar P/L and the
realized R-multiple (``nautilus_net_r``) at the fixed 100,000-unit research
notional. Because at most one position is open at a time (structurally
verified below, not merely assumed) and the strategy's stop is a hard order
in the same zero-slippage-modeled execution profile used throughout this
repo, this is sufficient to derive, causally, the frozen per-trade stop
distance and therefore the worst intratrade mark under any causal sizing
policy -- without reading a single further price bar.

``load_development_trade_path`` and ``load_validation_trade_path`` each
refuse to load anything but their own labelled partition (a VALIDATION
artifact can never be loaded as DEVELOPMENT or vice versa), and both refuse
any trade timestamp at or after the final-holdout boundary
(``ftmoquant.research.stage_g.HOLDOUT_START``), matching the firewall used
throughout the rest of this repository's research code. VALIDATION access
is read-only/diagnostic-only: nothing in this module lets VALIDATION data
influence sizing-policy or bootstrap-method selection -- see
``validation_diagnostic.py``.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal

from ftmoquant.research.stage_g import HOLDOUT_START

BASE_RESEARCH_UNITS = Decimal("100000")
FROZEN_CANDIDATE_IDENTITY = {
    "family": "B2F1_sweep_bos_retest",
    "instrument_id": "USD/CAD.OANDA",
    "rr": 2.0,
    "strategy_semantic_id": "usdcad_sweep_bos_retest_v1",
    "swing_lookback": 40,
    "timeframe": "M30",
}
_HOLDOUT_START_NS = int(HOLDOUT_START.timestamp() * 1_000_000_000)

ExitReason = Literal["stop", "target"]


class PathExtractionError(ValueError):
    """Raised when an execution artifact is unsafe or malformed to load."""


@dataclass(frozen=True, slots=True)
class TradeRecord:
    """One causal, sizing-independent record of a frozen strategy trade.

    ``original_realized_pnl``/``original_risk_budget``/``usd_risk_per_unit``
    are all in the FTMO account currency (USD), not the pair's native quote
    currency (CAD). ``trades.csv``'s own ``nautilus_realized_pnl`` is
    genuinely CAD-denominated -- it is Nautilus's native
    ``PositionClosed.realized_pnl``, which for a CurrencyPair position is
    always settled in the pair's quote currency, independent of the venue
    account's own base currency (USD here). This module converts once,
    causally, using each trade's own entry price as the USD/CAD rate at
    that instant (1 USD = ``entry_price`` CAD, since USD is USD/CAD's base
    leg) -- the same single-conversion-price convention already used by
    ``mean_reversion_h1_development._convert_to_account_currency``
    elsewhere in this repo, not a new invented method. ``net_r`` itself
    needs no conversion: it is a same-currency ratio (CAD P/L over CAD
    risk), so it is currency-invariant by construction.
    """

    trade_index: int
    entry_ns: int
    exit_ns: int
    exit_reason: ExitReason
    net_r: Decimal
    original_realized_pnl: Decimal
    original_risk_budget: Decimal
    usd_risk_per_unit: Decimal

    def __post_init__(self) -> None:
        if self.exit_ns <= self.entry_ns:
            raise PathExtractionError("trade exit must be strictly after entry")
        if self.net_r == 0:
            raise PathExtractionError("trade net_r must be non-zero")
        if not self.original_risk_budget.is_finite() or self.original_risk_budget <= 0:
            raise PathExtractionError("original_risk_budget must be positive")
        if not self.usd_risk_per_unit.is_finite() or self.usd_risk_per_unit <= 0:
            raise PathExtractionError("usd_risk_per_unit must be positive")


@dataclass(frozen=True, slots=True)
class DevelopmentTradePath:
    """The complete, chronological, DEVELOPMENT-only trade sequence."""

    trades: tuple[TradeRecord, ...]
    source_dir: Path
    trades_csv_sha256: str
    frozen_candidate_identity_sha256: str


@dataclass(frozen=True, slots=True)
class ValidationTradePath:
    """The complete, chronological, VALIDATION-only trade sequence.

    Structurally identical to :class:`DevelopmentTradePath` but kept as a
    distinct type so a VALIDATION path can never be silently substituted
    where sizing/bootstrap *selection* code expects a DEVELOPMENT path.
    Nothing in this package accepts a ``ValidationTradePath`` except
    ``validation_diagnostic.py``'s read-only, already-frozen-policy
    evaluation -- see that module's docstring for the anti-tuning rule this
    type boundary supports.
    """

    trades: tuple[TradeRecord, ...]
    source_dir: Path
    trades_csv_sha256: str
    frozen_candidate_identity_sha256: str


def load_development_trade_path(execution_dir: Path) -> DevelopmentTradePath:
    """Load and validate the frozen DEVELOPMENT execution artifact.

    Raises :class:`PathExtractionError` if the directory is not labelled
    ``development``, if VALIDATION/holdout was reportedly accessed to
    produce it, if the frozen candidate identity does not match, or if any
    trade timestamp reaches the final-holdout boundary.
    """

    execution_dir = Path(execution_dir)
    provenance = _read_json(execution_dir / "runtime_provenance.json")
    if provenance.get("validation_accessed") is not False:
        raise PathExtractionError(
            "refusing an artifact that reports VALIDATION was accessed"
        )
    trades, digest, identity_sha256 = _load_trade_path_core(
        execution_dir, expected_partition="development"
    )
    return DevelopmentTradePath(
        trades=trades,
        source_dir=execution_dir,
        trades_csv_sha256=digest,
        frozen_candidate_identity_sha256=identity_sha256,
    )


def load_validation_trade_path(execution_dir: Path) -> ValidationTradePath:
    """Load and validate the frozen VALIDATION execution artifact.

    This function exists only for read-only diagnostic evaluation of an
    already-frozen, DEVELOPMENT-selected sizing policy/bootstrap method
    (see ``validation_diagnostic.py``). It must never be used to select,
    rank, or tune a sizing policy, bootstrap method, or block length.

    Raises :class:`PathExtractionError` if the directory is not labelled
    ``validation``, if the frozen candidate identity does not match, or if
    any trade timestamp reaches the final-holdout boundary -- the same
    holdout firewall used by :func:`load_development_trade_path`.
    """

    execution_dir = Path(execution_dir)
    trades, digest, identity_sha256 = _load_trade_path_core(
        execution_dir, expected_partition="validation"
    )
    return ValidationTradePath(
        trades=trades,
        source_dir=execution_dir,
        trades_csv_sha256=digest,
        frozen_candidate_identity_sha256=identity_sha256,
    )


def _load_trade_path_core(
    execution_dir: Path, *, expected_partition: str
) -> tuple[tuple[TradeRecord, ...], str, str]:
    """Shared loading/validation core for both DEVELOPMENT and VALIDATION.

    Every check here applies identically to both partitions: the reported
    partition label must match exactly what the caller expects, the final
    holdout must never have been accessed, the frozen candidate identity
    must match, and the resulting trade sequence must be chronological,
    non-overlapping, and entirely before the final-holdout boundary.
    """

    result = _read_json(execution_dir / "result.json")
    provenance = _read_json(execution_dir / "runtime_provenance.json")

    if result.get("partition") != expected_partition:
        raise PathExtractionError(
            f"this loader may only read the {expected_partition!r} "
            f"partition, found {result.get('partition')!r}"
        )
    if provenance.get("final_holdout_accessed") is not False:
        raise PathExtractionError(
            "refusing an artifact that reports the final holdout was accessed"
        )
    identity = provenance.get("frozen_candidate_identity")
    if identity != FROZEN_CANDIDATE_IDENTITY:
        raise PathExtractionError(
            f"frozen candidate identity mismatch: {identity!r} != "
            f"{FROZEN_CANDIDATE_IDENTITY!r}"
        )

    trades_path = execution_dir / "trades.csv"
    raw_rows, digest = _read_trades_csv(trades_path)
    trades = tuple(_build_trade(row) for row in raw_rows)
    _validate_chronology_and_holdout(trades)

    return (
        trades,
        digest,
        str(provenance.get("frozen_candidate_identity_sha256", "")),
    )


def _build_trade(row: dict[str, str]) -> TradeRecord:
    trade_index = int(row["trade_index"])
    entry_ns = int(row["nautilus_entry_ns"])
    exit_ns = int(row["nautilus_exit_ns"])
    exit_reason_raw = row["alpha_lab_exit_reason"].strip().lower()
    if exit_reason_raw not in ("stop", "target"):
        raise PathExtractionError(
            f"trade {trade_index} has unsupported exit_reason {exit_reason_raw!r}"
        )
    exit_reason: ExitReason = exit_reason_raw  # type: ignore[assignment]
    net_r = _decimal(row["nautilus_net_r"], f"trade {trade_index} nautilus_net_r")
    realized_pnl_cad = _decimal(
        row["nautilus_realized_pnl"], f"trade {trade_index} nautilus_realized_pnl"
    )
    entry_price = _decimal(
        row["nautilus_entry_price"], f"trade {trade_index} nautilus_entry_price"
    )
    if entry_price <= 0:
        raise PathExtractionError(f"trade {trade_index} entry price must be positive")
    original_risk_budget_cad = abs(realized_pnl_cad / net_r)
    usd_risk_per_unit = (original_risk_budget_cad / BASE_RESEARCH_UNITS) / entry_price
    original_risk_budget_usd = BASE_RESEARCH_UNITS * usd_risk_per_unit
    original_realized_pnl_usd = net_r * original_risk_budget_usd
    return TradeRecord(
        trade_index=trade_index,
        entry_ns=entry_ns,
        exit_ns=exit_ns,
        exit_reason=exit_reason,
        net_r=net_r,
        original_realized_pnl=original_realized_pnl_usd,
        original_risk_budget=original_risk_budget_usd,
        usd_risk_per_unit=usd_risk_per_unit,
    )


def _validate_chronology_and_holdout(trades: tuple[TradeRecord, ...]) -> None:
    if not trades:
        raise PathExtractionError("DEVELOPMENT trade path must not be empty")
    previous_exit_ns: int | None = None
    previous_entry_ns: int | None = None
    for trade in trades:
        if trade.exit_ns >= _HOLDOUT_START_NS or trade.entry_ns >= _HOLDOUT_START_NS:
            raise PathExtractionError(
                f"trade {trade.trade_index} reaches the final-holdout boundary"
            )
        if previous_entry_ns is not None and trade.entry_ns <= previous_entry_ns:
            raise PathExtractionError("trades.csv is not strictly chronological")
        if previous_exit_ns is not None and trade.entry_ns < previous_exit_ns:
            raise PathExtractionError(
                "trades.csv contains overlapping positions; the "
                "conservative-floor derivation requires at most one open "
                "position at a time"
            )
        previous_entry_ns = trade.entry_ns
        previous_exit_ns = trade.exit_ns


def _read_trades_csv(path: Path) -> tuple[tuple[dict[str, str], ...], str]:
    import hashlib

    if not path.is_file():
        raise PathExtractionError(f"missing trades.csv: {path}")
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    rows = tuple(csv.DictReader(content.decode("utf-8").splitlines()))
    if not rows:
        raise PathExtractionError(f"trades.csv has no rows: {path}")
    return rows, digest


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise PathExtractionError(f"missing artifact file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise PathExtractionError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise PathExtractionError(f"{path} must contain a JSON object")
    return value


def _decimal(value: str, field: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise PathExtractionError(
            f"{field} is not a valid decimal: {value!r}"
        ) from error
    if not parsed.is_finite():
        raise PathExtractionError(f"{field} must be finite")
    return parsed
