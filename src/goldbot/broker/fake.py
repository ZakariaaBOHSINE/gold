"""FakeBroker: an in-memory simulated broker for development and tests (macOS-safe).

Price model: you feed it the bid (via ``update_price`` or ``update_bar``);
ask = bid + spread. Buys fill at the ask and close at the bid; sells fill at
the bid and close at the ask. Stop loss / take profit are checked against
each bar's range.
"""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import count

from goldbot.broker.base import Broker, BrokerError, OrderRejected
from goldbot.models import (
    AccountInfo,
    ClosedTrade,
    Direction,
    Fill,
    Order,
    Position,
    SymbolSpec,
    Tick,
)


class FakeBroker(Broker):
    def __init__(
        self,
        symbol: str,
        spread: float = 0.30,
        starting_balance: float = 10_000.0,
        contract_size: float = 100.0,
        currency: str = "USD",
        tick_size: float = 0.01,
        volume_min: float = 0.01,
        volume_step: float = 0.01,
        volume_max: float = 100.0,
        min_stop_distance: float = 0.0,
        commission_per_lot: float = 0.0,
    ) -> None:
        if spread < 0:
            raise ValueError("spread must be >= 0")
        self.symbol = symbol
        self.spread = spread
        self.contract_size = contract_size
        self.currency = currency
        self.balance = starting_balance
        self._connected = False
        self._bid: float | None = None
        self._time: datetime | None = None
        self._tickets = count(1)
        self._positions: dict[int, Position] = {}
        self.closed_trades: list[ClosedTrade] = []
        self._spec = SymbolSpec(
            symbol=symbol,
            contract_size=contract_size,
            tick_size=tick_size,
            tick_value=contract_size * tick_size,  # account currency == quote currency (USD)
            volume_min=volume_min,
            volume_step=volume_step,
            volume_max=volume_max,
            min_stop_distance=min_stop_distance,
            commission_per_lot=commission_per_lot,
        )

    # ---- connection -------------------------------------------------------
    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def _require_connected(self) -> None:
        if not self._connected:
            raise BrokerError("FakeBroker is not connected")

    # ---- prices -----------------------------------------------------------
    def update_price(self, bid: float, time: datetime) -> list[ClosedTrade]:
        """Set the current bid. Checks SL/TP against this single price."""
        return self.update_bar(high=bid, low=bid, close=bid, time=time)

    def update_bar(
        self, high: float, low: float, close: float, time: datetime
    ) -> list[ClosedTrade]:
        """Advance one bar (bid prices). Returns trades closed by SL/TP in this bar.

        If both SL and TP fall inside the same bar, the stop loss is assumed to
        have been hit first (the conservative assumption).
        """
        if time.tzinfo is None:
            raise ValueError("time must be timezone-aware")
        if low > high:
            raise ValueError("low must be <= high")
        time = time.astimezone(timezone.utc)
        closed: list[ClosedTrade] = []
        for pos in list(self._positions.values()):
            hit = self._check_exit(pos, high, low)
            if hit is not None:
                price, reason = hit
                closed.append(self._close(pos, price, time, reason))
        self._bid = close
        self._time = time
        return closed

    def _check_exit(
        self, pos: Position, high_bid: float, low_bid: float
    ) -> tuple[float, str] | None:
        if pos.direction is Direction.BUY:
            # long positions close at the bid
            if low_bid <= pos.stop_loss:
                return pos.stop_loss, "stop_loss"
            if pos.take_profit is not None and high_bid >= pos.take_profit:
                return pos.take_profit, "take_profit"
        else:
            # short positions close at the ask
            high_ask, low_ask = high_bid + self.spread, low_bid + self.spread
            if high_ask >= pos.stop_loss:
                return pos.stop_loss, "stop_loss"
            if pos.take_profit is not None and low_ask <= pos.take_profit:
                return pos.take_profit, "take_profit"
        return None

    def get_tick(self, symbol: str) -> Tick:
        self._require_connected()
        self._check_symbol(symbol)
        if self._bid is None or self._time is None:
            raise BrokerError("no price yet — call update_price/update_bar first")
        return Tick(symbol=symbol, bid=self._bid, ask=self._bid + self.spread, time=self._time)

    def symbol_info(self, symbol: str) -> SymbolSpec:
        self._check_symbol(symbol)
        return self._spec

    def _check_symbol(self, symbol: str) -> None:
        if symbol != self.symbol:
            raise BrokerError(f"unknown symbol {symbol!r} (FakeBroker trades {self.symbol!r})")

    # ---- account ----------------------------------------------------------
    def _floating_pnl(self) -> float:
        if self._bid is None:
            return 0.0
        total = 0.0
        for pos in self._positions.values():
            exit_price = self._bid if pos.direction is Direction.BUY else self._bid + self.spread
            total += self._pnl(pos, exit_price)
        return total

    def _pnl(self, pos: Position, exit_price: float) -> float:
        return (exit_price - pos.open_price) * pos.direction.sign * pos.volume * self.contract_size

    def account_info(self) -> AccountInfo:
        self._require_connected()
        return AccountInfo(
            balance=self.balance,
            equity=self.balance + self._floating_pnl(),
            currency=self.currency,
            is_demo=True,
        )

    # ---- orders -----------------------------------------------------------
    def place_order(self, order: Order) -> Fill:
        self._require_connected()
        if order.symbol != self.symbol:
            raise OrderRejected(f"unknown symbol {order.symbol!r}")
        if order.stop_loss is None:  # defence in depth; Order already enforces this
            raise OrderRejected("order has no stop loss")
        tick = self.get_tick(order.symbol)
        price = tick.ask if order.direction is Direction.BUY else tick.bid

        if order.direction is Direction.BUY:
            if order.stop_loss >= tick.bid:
                raise OrderRejected("BUY stop loss must be below the current bid")
            if order.take_profit is not None and order.take_profit <= price:
                raise OrderRejected("BUY take profit must be above the fill price")
        else:
            if order.stop_loss <= tick.ask:
                raise OrderRejected("SELL stop loss must be above the current ask")
            if order.take_profit is not None and order.take_profit >= price:
                raise OrderRejected("SELL take profit must be below the fill price")

        ticket = next(self._tickets)
        self._positions[ticket] = Position(
            ticket=ticket,
            symbol=order.symbol,
            direction=order.direction,
            volume=order.volume,
            open_price=price,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            open_time=tick.time,
            comment=order.comment,
        )
        return Fill(ticket=ticket, order=order, price=price, volume=order.volume, time=tick.time)

    def open_positions(self, symbol: str | None = None) -> list[Position]:
        self._require_connected()
        return [p for p in self._positions.values() if symbol is None or p.symbol == symbol]

    def close_position(self, ticket: int) -> ClosedTrade:
        self._require_connected()
        pos = self._positions.get(ticket)
        if pos is None:
            raise BrokerError(f"no open position with ticket {ticket}")
        tick = self.get_tick(pos.symbol)
        price = tick.bid if pos.direction is Direction.BUY else tick.ask
        return self._close(pos, price, tick.time, "manual")

    def _close(self, pos: Position, price: float, time: datetime, reason: str) -> ClosedTrade:
        profit = self._pnl(pos, price)
        self.balance += profit
        del self._positions[pos.ticket]
        trade = ClosedTrade(
            ticket=pos.ticket,
            symbol=pos.symbol,
            direction=pos.direction,
            volume=pos.volume,
            open_price=pos.open_price,
            close_price=price,
            open_time=pos.open_time,
            close_time=time,
            profit=profit,
            close_reason=reason,
        )
        self.closed_trades.append(trade)
        return trade
