"""MT5Broker: real MetaTrader 5 connection (Windows only). Phase 1 stub.

``MetaTrader5`` is imported lazily inside ``connect()`` so this module imports
cleanly on macOS. Order methods are implemented in Phase 4.
"""

from __future__ import annotations

from typing import Any

from goldbot.broker.base import Broker, BrokerError
from goldbot.config import MT5Credentials
from goldbot.models import AccountInfo, ClosedTrade, Fill, Order, Position, SymbolSpec, Tick

_PHASE4 = "MT5Broker order handling is implemented in Phase 4"


def _import_mt5() -> Any:
    try:
        import MetaTrader5 as mt5  # noqa: N813  (Windows-only package)
    except ImportError as exc:
        raise BrokerError(
            "MetaTrader5 package not available. MT5Broker only runs on Windows "
            "with `pip install -e \".[mt5]\"`. Use FakeBroker on macOS."
        ) from exc
    return mt5


class MT5Broker(Broker):
    def __init__(self, credentials: MT5Credentials) -> None:
        self._credentials = credentials
        self._mt5: Any = None

    def connect(self) -> None:
        mt5 = _import_mt5()
        c = self._credentials
        kwargs: dict[str, Any] = {"login": c.login, "password": c.password, "server": c.server}
        if c.path:
            kwargs["path"] = c.path
        if not mt5.initialize(**kwargs):
            raise BrokerError(f"MT5 initialize failed: {mt5.last_error()}")

        info = mt5.account_info()
        if info is None:
            mt5.shutdown()
            raise BrokerError(f"could not read MT5 account info: {mt5.last_error()}")
        # Hard rule: demo accounts only.
        if info.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO:
            mt5.shutdown()
            raise BrokerError(
                "Refusing to start: connected MT5 account is NOT a demo account."
            )
        self._mt5 = mt5

    def disconnect(self) -> None:
        if self._mt5 is not None:
            self._mt5.shutdown()
            self._mt5 = None

    def account_info(self) -> AccountInfo:
        raise NotImplementedError(_PHASE4)

    def get_tick(self, symbol: str) -> Tick:
        raise NotImplementedError(_PHASE4)

    def symbol_info(self, symbol: str) -> SymbolSpec:
        # Phase 4: map mt5.symbol_info(): trade_contract_size, trade_tick_size,
        # trade_tick_value, volume_min/step/max, trade_stops_level * point.
        raise NotImplementedError(_PHASE4)

    def place_order(self, order: Order) -> Fill:
        raise NotImplementedError(_PHASE4)

    def open_positions(self, symbol: str | None = None) -> list[Position]:
        raise NotImplementedError(_PHASE4)

    def close_position(self, ticket: int) -> ClosedTrade:
        raise NotImplementedError(_PHASE4)
