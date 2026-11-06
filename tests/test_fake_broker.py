from datetime import datetime, timedelta, timezone

import pytest

from goldbot.broker import BrokerError, FakeBroker, OrderRejected
from goldbot.models import Direction, Order

T0 = datetime(2024, 1, 2, 13, 0, tzinfo=timezone.utc)
SYMBOL = "XAUUSD"


@pytest.fixture
def broker() -> FakeBroker:
    b = FakeBroker(SYMBOL, spread=0.30, starting_balance=10_000, contract_size=100)
    b.connect()
    b.update_price(2000.00, T0)
    return b


def buy(volume=0.10, sl=1990.0, tp=2020.0) -> Order:
    return Order(SYMBOL, Direction.BUY, volume, stop_loss=sl, take_profit=tp)


def sell(volume=0.10, sl=2010.0, tp=1980.0) -> Order:
    return Order(SYMBOL, Direction.SELL, volume, stop_loss=sl, take_profit=tp)


def test_tick_includes_spread(broker):
    tick = broker.get_tick(SYMBOL)
    assert tick.bid == 2000.00
    assert tick.ask == pytest.approx(2000.30)
    assert tick.spread == pytest.approx(0.30)


def test_configurable_spread():
    b = FakeBroker(SYMBOL, spread=1.0)
    b.connect()
    b.update_price(2000, T0)
    assert b.get_tick(SYMBOL).ask == 2001.0


def test_buy_fills_at_ask_sell_at_bid(broker):
    f1 = broker.place_order(buy())
    f2 = broker.place_order(sell())
    assert f1.price == pytest.approx(2000.30)
    assert f2.price == pytest.approx(2000.00)
    assert f1.ticket != f2.ticket
    assert f1.time == T0
    assert len(broker.open_positions()) == 2


def test_spread_is_an_immediate_cost(broker):
    broker.place_order(buy(volume=1.0))
    # 0.30 spread * 1 lot * 100 oz
    assert broker.account_info().equity == pytest.approx(10_000 - 30)
    assert broker.account_info().balance == 10_000


def test_order_without_stop_loss_cannot_exist():
    with pytest.raises(ValueError, match="stop_loss"):
        Order(SYMBOL, Direction.BUY, 0.1, stop_loss=None)  # type: ignore[arg-type]


def test_rejects_stop_on_wrong_side(broker):
    with pytest.raises(OrderRejected):
        broker.place_order(buy(sl=2005.0))
    with pytest.raises(OrderRejected):
        broker.place_order(sell(sl=1995.0))
    assert broker.open_positions() == []


def test_rejects_take_profit_on_wrong_side(broker):
    with pytest.raises(OrderRejected):
        broker.place_order(buy(tp=1999.0))
    with pytest.raises(OrderRejected):
        broker.place_order(sell(tp=2001.0))


def test_rejects_unknown_symbol(broker):
    with pytest.raises(OrderRejected):
        broker.place_order(Order("EURUSD", Direction.BUY, 0.1, stop_loss=1.0))


def test_requires_connection_and_price():
    b = FakeBroker(SYMBOL)
    with pytest.raises(BrokerError, match="not connected"):
        b.account_info()
    b.connect()
    with pytest.raises(BrokerError, match="no price"):
        b.get_tick(SYMBOL)


def test_reports_demo_account(broker):
    assert broker.account_info().is_demo is True


def test_buy_take_profit_hit(broker):
    broker.place_order(buy(volume=0.10))  # fill 2000.30
    closed = broker.update_bar(high=2021.0, low=2001.0, close=2015.0, time=T0 + timedelta(minutes=15))
    assert len(closed) == 1
    t = closed[0]
    assert t.close_reason == "take_profit"
    assert t.close_price == 2020.0
    assert t.profit == pytest.approx((2020.0 - 2000.30) * 0.10 * 100)
    assert broker.balance == pytest.approx(10_000 + t.profit)
    assert broker.open_positions() == []


def test_buy_stop_loss_hit(broker):
    broker.place_order(buy(volume=0.10))
    closed = broker.update_bar(high=2001.0, low=1989.0, close=1992.0, time=T0 + timedelta(minutes=15))
    assert closed[0].close_reason == "stop_loss"
    assert closed[0].profit == pytest.approx((1990.0 - 2000.30) * 10)


def test_sell_sl_uses_ask(broker):
    broker.place_order(sell(volume=0.10, sl=2010.0))  # fill 2000.00
    # bid high 2009.80 -> ask high 2010.10 >= SL
    closed = broker.update_bar(high=2009.80, low=2000.0, close=2005.0, time=T0 + timedelta(minutes=15))
    assert closed[0].close_reason == "stop_loss"
    assert closed[0].profit == pytest.approx((2000.0 - 2010.0) * 10)


def test_sell_tp_uses_ask(broker):
    broker.place_order(sell(volume=0.10, tp=1980.0))
    # bid low 1979.80 -> ask low 1980.10: TP not reached yet
    assert broker.update_bar(high=1990, low=1979.80, close=1985, time=T0 + timedelta(minutes=15)) == []
    closed = broker.update_bar(high=1990, low=1979.50, close=1985, time=T0 + timedelta(minutes=30))
    assert closed[0].close_reason == "take_profit"
    assert closed[0].profit == pytest.approx((2000.0 - 1980.0) * 10)


def test_sl_assumed_first_when_both_in_bar(broker):
    broker.place_order(buy())
    closed = broker.update_bar(high=2030, low=1980, close=2000, time=T0 + timedelta(minutes=15))
    assert closed[0].close_reason == "stop_loss"


def test_floating_pnl_and_manual_close(broker):
    broker.place_order(buy(volume=0.50))  # fill 2000.30
    broker.update_price(2005.30, T0 + timedelta(minutes=5))
    info = broker.account_info()
    assert info.equity == pytest.approx(10_000 + 5.0 * 0.5 * 100)
    ticket = broker.open_positions()[0].ticket
    trade = broker.close_position(ticket)
    assert trade.close_reason == "manual"
    assert trade.close_price == 2005.30
    assert broker.balance == pytest.approx(10_250)
    assert broker.account_info().equity == pytest.approx(10_250)
    with pytest.raises(BrokerError):
        broker.close_position(ticket)


def test_sell_manual_close_at_ask(broker):
    broker.place_order(sell(volume=1.0))  # fill 2000.00
    broker.update_price(1995.00, T0 + timedelta(minutes=5))
    trade = broker.close_position(broker.open_positions()[0].ticket)
    assert trade.close_price == pytest.approx(1995.30)
    assert trade.profit == pytest.approx(4.70 * 100)


def test_naive_time_rejected(broker):
    with pytest.raises(ValueError):
        broker.update_price(2000, datetime(2024, 1, 1))


def test_mt5_module_imports_on_macos():
    from goldbot.broker.mt5 import MT5Broker  # noqa: F401  must not import MetaTrader5


def test_symbol_info(broker):
    spec = broker.symbol_info(SYMBOL)
    assert spec.contract_size == 100
    assert spec.tick_size == 0.01
    assert spec.tick_value == pytest.approx(1.0)       # 0.01 * 100 oz
    assert spec.value_per_price_unit == pytest.approx(100.0)
    assert spec.volume_min == spec.volume_step == 0.01
    with pytest.raises(BrokerError):
        broker.symbol_info("EURUSD")
