"""Core domain objects: Signal, Order, Fill and supporting types.

All objects are immutable and validate themselves on construction.
All times must be timezone-aware and are normalized to UTC.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum


class Direction(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        return 1 if self is Direction.BUY else -1


def _require_utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _check_stop_side(direction: Direction, reference: float, stop_loss: float | None) -> None:
    if stop_loss is None:
        raise ValueError("stop_loss is required")
    if stop_loss <= 0:
        raise ValueError("stop_loss must be > 0")
    if direction is Direction.BUY and stop_loss >= reference:
        raise ValueError("BUY stop_loss must be below entry")
    if direction is Direction.SELL and stop_loss <= reference:
        raise ValueError("SELL stop_loss must be above entry")


def _check_tp_side(direction: Direction, reference: float, take_profit: float | None) -> None:
    if take_profit is None:
        return
    if direction is Direction.BUY and take_profit <= reference:
        raise ValueError("BUY take_profit must be above entry")
    if direction is Direction.SELL and take_profit >= reference:
        raise ValueError("SELL take_profit must be below entry")


@dataclass(frozen=True)
class Signal:
    """A strategy's trade idea. Not an order — it must pass the risk manager first."""

    symbol: str
    direction: Direction
    entry: float
    stop_loss: float
    take_profit: float | None
    reason: str
    time: datetime

    def __post_init__(self) -> None:
        if self.entry <= 0:
            raise ValueError("entry must be > 0")
        _check_stop_side(self.direction, self.entry, self.stop_loss)
        _check_tp_side(self.direction, self.entry, self.take_profit)
        object.__setattr__(self, "time", _require_utc(self.time, "time"))

    @property
    def risk_per_unit(self) -> float:
        return abs(self.entry - self.stop_loss)

    @property
    def risk_reward(self) -> float | None:
        if self.take_profit is None:
            return None
        return abs(self.take_profit - self.entry) / self.risk_per_unit


@dataclass(frozen=True)
class Order:
    """A market order approved by the risk manager. Always carries a stop loss."""

    symbol: str
    direction: Direction
    volume: float
    stop_loss: float
    take_profit: float | None = None
    comment: str = ""

    def __post_init__(self) -> None:
        if self.volume <= 0:
            raise ValueError("volume must be > 0")
        if self.stop_loss is None or self.stop_loss <= 0:
            raise ValueError("stop_loss is required and must be > 0")
        if self.take_profit is not None and self.take_profit <= 0:
            raise ValueError("take_profit must be > 0")


@dataclass(frozen=True)
class Fill:
    """Result of an executed order."""

    ticket: int
    order: Order
    price: float
    volume: float
    time: datetime

    def __post_init__(self) -> None:
        if self.price <= 0 or self.volume <= 0:
            raise ValueError("price and volume must be > 0")
        object.__setattr__(self, "time", _require_utc(self.time, "time"))


@dataclass(frozen=True)
class Tick:
    symbol: str
    bid: float
    ask: float
    time: datetime

    def __post_init__(self) -> None:
        if self.ask < self.bid:
            raise ValueError("ask must be >= bid")
        object.__setattr__(self, "time", _require_utc(self.time, "time"))

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass(frozen=True)
class SymbolSpec:
    """Trading properties of a symbol, as reported by the broker (never hardcoded)."""

    symbol: str
    contract_size: float          # units per 1.0 lot (100 oz for XAUUSD)
    tick_size: float              # minimum price increment
    tick_value: float             # account-currency value of one tick for 1.0 lot
    volume_min: float
    volume_step: float
    volume_max: float
    min_stop_distance: float = 0.0  # broker minimum SL distance from market, price units
    commission_per_lot: float = 0.0  # round turn, account currency

    def __post_init__(self) -> None:
        for name in ("contract_size", "tick_size", "tick_value", "volume_min", "volume_step", "volume_max"):
            v = getattr(self, name)
            if not v > 0:  # also rejects NaN
                raise ValueError(f"{name} must be > 0")
        if self.volume_max < self.volume_min:
            raise ValueError("volume_max must be >= volume_min")
        if self.min_stop_distance < 0 or self.commission_per_lot < 0:
            raise ValueError("min_stop_distance and commission_per_lot must be >= 0")

    @property
    def value_per_price_unit(self) -> float:
        """Account-currency P&L of a 1.0-lot position per 1.0 price move."""
        return self.tick_value / self.tick_size


@dataclass(frozen=True)
class AccountInfo:
    balance: float
    equity: float
    currency: str
    is_demo: bool


@dataclass
class Position:
    """An open position held by a broker."""

    ticket: int
    symbol: str
    direction: Direction
    volume: float
    open_price: float
    stop_loss: float
    take_profit: float | None
    open_time: datetime
    comment: str = ""


@dataclass(frozen=True)
class ClosedTrade:
    """A position that has been closed, with its realized P&L."""

    ticket: int
    symbol: str
    direction: Direction
    volume: float
    open_price: float
    close_price: float
    open_time: datetime
    close_time: datetime
    profit: float
    close_reason: str  # "stop_loss", "take_profit", "manual"
