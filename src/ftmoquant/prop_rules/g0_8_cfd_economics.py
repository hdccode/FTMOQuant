"""Frozen G0.8 FTMO CFD deployment economics, separate from strategy semantics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

G0_8_CFD_ECONOMICS_PATH = Path("config/prop/g0_8_ftmo_cfd_economics_2026-08-15.yaml")
G0_8_CFD_ECONOMICS_SHA256 = (
    "a5178afb97606c6d8f8887c59323d94ffdd1c32b0301a862bd8e992b890b1c99"
)


class G08CfdEconomicsError(ValueError):
    """Raised when an economics snapshot cannot prove a requested calculation."""


@dataclass(frozen=True, slots=True)
class Commission:
    kind: str
    value: Decimal


@dataclass(frozen=True, slots=True)
class CfdContract:
    symbol: str
    contract_size: Decimal
    swing_leverage: Decimal
    commission: Commission
    sessions: tuple[tuple[time, time], ...]
    deployment_lot_metadata: str | dict[str, str]


@dataclass(frozen=True, slots=True)
class G08CfdEconomics:
    snapshot_id: str
    semantic_sha256: str
    server_timezone: ZoneInfo
    rollover_warning: str
    contracts: tuple[CfdContract, ...]
    canonical_document: dict[str, Any]

    def contract(self, symbol: str) -> CfdContract:
        found = [item for item in self.contracts if item.symbol == symbol]
        if len(found) != 1:
            raise G08CfdEconomicsError(f"unknown G0.8 CFD symbol: {symbol}")
        return found[0]

    def gross_pnl(
        self, symbol: str, direction: int, entry: Decimal, exit: Decimal, lots: Decimal
    ) -> Decimal:
        contract = self.contract(symbol)
        _positive(entry, "entry price")
        _positive(exit, "exit price")
        _positive(lots, "lots")
        if direction not in {-1, 1}:
            raise G08CfdEconomicsError("direction must be -1 or 1")
        return Decimal(direction) * (exit - entry) * contract.contract_size * lots

    def margin_requirement(self, symbol: str, price: Decimal, lots: Decimal) -> Decimal:
        contract = self.contract(symbol)
        _positive(price, "price")
        _positive(lots, "lots")
        return abs(price * contract.contract_size * lots) / contract.swing_leverage

    def side_commission(self, symbol: str, price: Decimal, lots: Decimal) -> Decimal:
        contract = self.contract(symbol)
        _positive(price, "price")
        _positive(lots, "lots")
        if contract.commission.kind == "none":
            return Decimal(0)
        if contract.commission.kind == "fixed_usd_per_lot_per_side":
            return contract.commission.value * lots
        if contract.commission.kind == "percent_of_traded_notional_per_side":
            return (
                abs(price * contract.contract_size * lots) * contract.commission.value
            )
        raise G08CfdEconomicsError("unknown commission model")

    def is_session_eligible(self, symbol: str, timestamp_utc: datetime) -> bool:
        if timestamp_utc.tzinfo is None:
            raise G08CfdEconomicsError("timestamp must be timezone-aware")
        contract = self.contract(symbol)
        local = timestamp_utc.astimezone(self.server_timezone)
        if local.weekday() > 4:
            return False
        local_time = local.timetz().replace(tzinfo=None)
        return any(start <= local_time < end for start, end in contract.sessions)


def load_g08_cfd_economics(path: Path = G0_8_CFD_ECONOMICS_PATH) -> G08CfdEconomics:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise G08CfdEconomicsError(f"invalid G0.8 snapshot: {error}") from error
    if not isinstance(document, dict):
        raise G08CfdEconomicsError("snapshot must be a mapping")
    _validate(document)
    contracts = tuple(
        _contract(symbol, raw) for symbol, raw in document["symbols"].items()
    )
    return G08CfdEconomics(
        document["snapshot_id"],
        _hash(document),
        ZoneInfo(document["server_timezone"]),
        document["rollover"]["warning"],
        contracts,
        document,
    )


def _contract(symbol: str, raw: Any) -> CfdContract:
    assert isinstance(raw, dict)
    return CfdContract(
        symbol,
        Decimal(str(raw["contract_size"])),
        Decimal(str(raw["swing_leverage"])),
        Commission(raw["commission"]["kind"], Decimal(str(raw["commission"]["value"]))),
        tuple((_time(start), _time(end)) for start, end in raw["sessions"]),
        raw["deployment_lot_metadata"],
    )


def _validate(document: dict[str, Any]) -> None:
    if set(document) != {
        "schema_version",
        "snapshot_id",
        "version",
        "verified_on",
        "server_timezone",
        "rollover",
        "research_lot_granularity",
        "symbols",
    }:
        raise G08CfdEconomicsError("snapshot keys are not exact")
    if (
        document["schema_version"],
        document["snapshot_id"],
        document["version"],
        document["server_timezone"],
    ) != (
        1,
        "g0_8_ftmo_cfd_deployment_economics_2026_08_15",
        "2026-08-15",
        "Europe/Helsinki",
    ):
        raise G08CfdEconomicsError("snapshot identity/timezone drifted")
    if (
        document["rollover"].get("status") != "UNMODELLED"
        or document["research_lot_granularity"] != "unquantized_continuous_targets"
    ):
        raise G08CfdEconomicsError("rollover or research sizing drifted")
    expected = {
        "EUR/USD": (100000, 30, "fixed_usd_per_lot_per_side", "2.50"),
        "XAU/USD": (100, 15, "percent_of_traded_notional_per_side", "0.000007"),
        "US500.cash": (1, 15, "none", "0"),
        "USOIL.cash": (100, 15, "none", "0"),
        "SOYBEAN.c": (1, 1, "none", "0"),
    }
    if set(document["symbols"]) != set(expected):
        raise G08CfdEconomicsError("symbols drifted")
    for symbol, values in expected.items():
        item = document["symbols"][symbol]
        commission = item["commission"]
        if (
            item["contract_size"],
            item["swing_leverage"],
            commission["kind"],
            str(commission["value"]),
        ) != values:
            raise G08CfdEconomicsError(f"economics drifted: {symbol}")


def _time(value: str) -> time:
    return time.fromisoformat(value)


def _positive(value: Decimal, label: str) -> None:
    if not value.is_finite() or value <= 0:
        raise G08CfdEconomicsError(f"{label} must be positive")


def _hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
