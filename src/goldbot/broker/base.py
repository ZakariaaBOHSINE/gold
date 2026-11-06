"""Abstract broker interface.

Only ``goldbot.execution`` may call ``place_order`` / ``close_position``,
and only for orders approved by ``goldbot.risk``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from goldbot.models import AccountInfo, ClosedTrade, Fill, Order, Position, SymbolSpec, Tick


class BrokerError(RuntimeError):
    """Raised when the broker cannot perform an operation."""


class OrderRejected(BrokerError):
    """Raised when the broker refuses an order."""


class Broker(ABC):
    @abstractmethod
    def connect(self) -> None:
        """Open the connection. Must refuse non-demo accounts."""

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def account_info(self) -> AccountInfo: ...

    @abstractmethod
    def get_tick(self, symbol: str) -> Tick: ...

    @abstractmethod
    def symbol_info(self, symbol: str) -> SymbolSpec:
        """Contract size, tick size/value, volume limits, stop level for ``symbol``."""

    @abstractmethod
    def place_order(self, order: Order) -> Fill:
        """Send a market order. Raises OrderRejected on failure."""

    @abstractmethod
    def open_positions(self, symbol: str | None = None) -> list[Position]: ...

    @abstractmethod
    def close_position(self, ticket: int) -> ClosedTrade: ...
