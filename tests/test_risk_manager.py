"""Risk manager: every rule in isolation, sizing edge cases, daily reset, independence."""

import ast
import json
import math
from dataclasses import replace
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from goldbot.logging import TradeLogger
from goldbot.models import Direction, Signal, SymbolSpec, Tick
from goldbot.risk import AccountState, DailyLossTracker, RiskManager, position_size, round_volume_down

UTC = timezone.utc
NY = ZoneInfo("America/New_York")
T = datetime(2024, 1, 3, 15, 0, tzinfo=UTC)  # Wednesday, NY 10:00


@pytest.fixture
def limits(settings):
    # project defaults: 0.5% risk, 2% daily, 1 position, 2R, 3 bps, 3x spread, 17:00 NY
    return settings.risk


@pytest.fixture
def spec():
    return SymbolSpec("XAUUSD", contract_size=100, tick_size=0.01, tick_value=1.0,
                      volume_min=0.01, volume_step=0.01, volume_max=50, commission_per_lot=0.0)


def tick(bid=2000.0, ask=2000.3, ts=T):
    return Tick("XAUUSD", bid, ask, ts)


def long_signal(sl=1990.0, tp=None, entry=2000.3, ts=T):
    tp = tp if tp is not None else entry + 2 * (entry - sl)
    return Signal("XAUUSD", Direction.BUY, entry, sl, tp, "test long", ts)


def short_signal(sl=2010.0, tp=None, entry=2000.0, ts=T):
    tp = tp if tp is not None else entry - 2 * (sl - entry)
    return Signal("XAUUSD", Direction.SELL, entry, sl, tp, "test short", ts)


ACCOUNT = AccountState(equity=10_000.0, balance=10_000.0, open_positions=0)


def rm(limits, kill=False, logger=None):
    return RiskManager(limits, kill_switch=lambda: kill, logger=logger)


# ---- approval and sizing ----------------------------------------------------
def test_approves_long_and_sizes_from_broker_spec(limits, spec):
    d = rm(limits).evaluate(long_signal(), ACCOUNT, tick(), spec)
    assert d.approved and d.code == "approved"
    # 0.5% of 10000 = 50; stop 10.3 from ask fill * 100 per lot = 1030 -> 0.0485 -> 0.04
    assert d.order.volume == pytest.approx(0.04)
    assert d.order.stop_loss == 1990.0
    assert d.details["expected_fill"] == 2000.3
    assert d.details["rr"] == pytest.approx(2.0)
    assert "approved 0.04 lots" in d.reason


def test_approves_short_using_bid_fill(limits, spec):
    d = rm(limits).evaluate(short_signal(), ACCOUNT, tick(), spec)
    assert d.approved
    assert d.details["expected_fill"] == 2000.0
    assert d.order.volume == pytest.approx(0.05)  # 50 / (10 * 100)


def test_sizing_uses_tick_value_not_hardcoded(limits):
    # A broker quoting 1 lot = 10 oz: tick value 0.1 -> 10x the volume for the same risk
    small = SymbolSpec("XAUUSD", contract_size=10, tick_size=0.01, tick_value=0.1,
                       volume_min=0.01, volume_step=0.01, volume_max=100)
    d = rm(limits).evaluate(long_signal(), ACCOUNT, tick(), small)
    assert d.order.volume == pytest.approx(0.48)  # 50 / (10.3 * 10) = 0.485


def test_sizing_includes_commission(limits, spec):
    s = replace(spec, commission_per_lot=7.0)
    d = rm(limits).evaluate(short_signal(), ACCOUNT, tick(), s)
    # 50 / (10*100 + 7) = 0.0496 -> 0.04 (without commission it would be 0.05)
    assert d.order.volume == pytest.approx(0.04)
    assert d.details["risk_money"] <= 50


@pytest.mark.parametrize(
    "raw, step, expected",
    [(0.0485, 0.01, 0.04), (0.07, 0.01, 0.07), (0.07000000001, 0.01, 0.07),
     (0.06999999999999, 0.01, 0.07), (0.999, 0.1, 0.9), (1.25, 0.5, 1.0),
     (0.0, 0.01, 0.0), (-1, 0.01, 0.0), (math.nan, 0.01, 0.0), (math.inf, 0.01, 0.0)],
)
def test_round_volume_down(raw, step, expected):
    assert round_volume_down(raw, step) == pytest.approx(expected)


def test_volume_step_rounding_never_exceeds_risk(spec):
    s = replace(spec, volume_min=0.1, volume_step=0.1)
    # 50 / (10.3*100) = 0.0485 -> below 0.1 minimum -> 0
    assert position_size(10_000, 0.5, 10.3, s) == 0.0
    # 500 / 1030 = 0.485 -> 0.4
    assert position_size(100_000, 0.5, 10.3, s) == pytest.approx(0.4)


def test_volume_capped_at_max(limits, spec):
    big = AccountState(equity=50_000_000, balance=50_000_000, open_positions=0)
    d = rm(limits).evaluate(long_signal(), big, tick(), spec)
    assert d.approved and d.order.volume == 50


def test_volume_max_off_grid_is_rounded_down(spec):
    s = replace(spec, volume_max=10.005)
    assert position_size(1e9, 0.5, 10, s) == pytest.approx(10.0)


# ---- each rejection rule ----------------------------------------------------
def test_kill_switch(limits, spec):
    d = rm(limits, kill=True).evaluate(long_signal(), ACCOUNT, tick(), spec)
    assert not d.approved and d.code == "kill_switch" and "STOP" in d.reason


def test_kill_switch_file_appearing_between_calls(limits, spec, tmp_path):
    stop = tmp_path / "STOP"
    r = RiskManager(limits, kill_switch=stop)
    assert r.evaluate(long_signal(), ACCOUNT, tick(), spec).approved
    stop.touch()
    assert r.evaluate(long_signal(), ACCOUNT, tick(), spec).code == "kill_switch"
    stop.unlink()
    assert r.evaluate(long_signal(), ACCOUNT, tick(), spec).approved


def test_max_open_positions(limits, spec):
    acc = replace(ACCOUNT, open_positions=1)
    d = rm(limits).evaluate(long_signal(), acc, tick(), spec)
    assert d.code == "max_open_positions" and "1/1" in d.reason


@pytest.mark.parametrize("sl", [None, 0.0, -5.0, math.nan])
def test_missing_or_invalid_stop_loss(limits, spec, sl):
    # Bypass Signal's own validation with a look-alike object
    bad = SimpleNamespace(symbol="XAUUSD", direction=Direction.BUY, entry=2000.3,
                          stop_loss=sl, take_profit=2030.0, reason="x", time=T)
    d = rm(limits).evaluate(bad, ACCOUNT, tick(), spec)
    assert d.code == "missing_stop_loss"


def test_stop_on_wrong_side_of_market(limits, spec):
    # Valid vs. the signal's entry, but the market has since moved below the stop
    d = rm(limits).evaluate(long_signal(sl=1995), ACCOUNT, tick(bid=1994.0, ask=1994.3), spec)
    assert d.code == "stop_wrong_side"
    d = rm(limits).evaluate(short_signal(sl=2005), ACCOUNT, tick(bid=2005.0, ask=2005.3), spec)
    assert d.code == "stop_wrong_side"


def test_tiny_stop_rejected_by_spread_multiple(limits, spec):
    # stop 0.8 from ask fill < 3 x 0.3 spread = 0.9
    d = rm(limits).evaluate(long_signal(sl=1999.5), ACCOUNT, tick(), spec)
    assert d.code == "stop_too_close_spread" and "3 x spread" in d.reason


def test_stop_exactly_at_spread_multiple_is_allowed(limits, spec):
    d = rm(limits).evaluate(long_signal(sl=2000.3 - 0.9), ACCOUNT, tick(), spec)
    assert d.approved


def test_spread_multiple_zero_disables_check(limits, spec):
    lim = replace(limits, min_stop_spread_multiple=0.0)
    d = rm(lim).evaluate(long_signal(sl=1999.9), ACCOUNT, tick(), spec)
    assert d.code != "stop_too_close_spread"


def test_tiny_stop_with_huge_size_is_capped_not_unbounded(limits, spec):
    lim = replace(limits, min_stop_spread_multiple=0.0)
    d = rm(lim).evaluate(long_signal(sl=1999.99), ACCOUNT, tick(bid=2000.0, ask=2000.0), spec)
    assert d.approved and d.order.volume == spec.volume_max


def test_broker_minimum_stop_distance(limits, spec):
    s = replace(spec, min_stop_distance=15.0)
    d = rm(limits).evaluate(long_signal(sl=1990), ACCOUNT, tick(), s)
    assert d.code == "stop_too_close_broker"


def test_spread_above_max(limits, spec):
    # 0.70 at 2000 = 3.5 bps > 3
    d = rm(limits).evaluate(long_signal(entry=2000.7, sl=1980), ACCOUNT, tick(ask=2000.7), spec)
    assert d.code == "spread_above_max" and "3.50 bps" in d.reason


def test_spread_is_relative_to_price(limits, spec):
    # 1.20 spread is fine at 4500 (2.67 bps)
    t = tick(bid=4500.0, ask=4501.2)
    d = rm(limits).evaluate(long_signal(entry=4501.2, sl=4470), ACCOUNT, t, spec)
    assert d.approved


def test_risk_reward_measured_from_expected_fill(limits, spec):
    # 2R from the bid (2000) is only 1.94R from the ask fill (2000.3)
    sig = long_signal(entry=2000.0, sl=1990, tp=2020.0)
    d = rm(limits).evaluate(sig, ACCOUNT, tick(), spec)
    assert d.code == "risk_reward_below_min" and "expected fill 2000.30" in d.reason
    assert d.details["rr"] == pytest.approx(19.7 / 10.3, rel=1e-4)


def test_short_risk_reward_from_bid(limits, spec):
    sig = short_signal(sl=2010, tp=1981)  # 19/10 = 1.9R from bid
    assert rm(limits).evaluate(sig, ACCOUNT, tick(), spec).code == "risk_reward_below_min"


def test_missing_take_profit(limits, spec):
    sig = Signal("XAUUSD", Direction.BUY, 2000.3, 1990, None, "no tp", T)
    assert rm(limits).evaluate(sig, ACCOUNT, tick(), spec).code == "missing_take_profit"


@pytest.mark.parametrize("equity", [0.0, -100.0, math.nan])
def test_zero_or_negative_equity(limits, spec, equity):
    acc = AccountState(equity=equity, balance=equity, open_positions=0)
    d = rm(limits).evaluate(long_signal(), acc, tick(), spec)
    assert not d.approved
    assert d.code in ("no_equity", "daily_loss_limit")


def test_size_below_minimum_volume(limits, spec):
    tiny = AccountState(equity=100.0, balance=100.0, open_positions=0)  # 0.5 USD risk
    d = rm(limits).evaluate(long_signal(), tiny, tick(), spec)
    assert d.code == "size_below_minimum" and "minimum 0.01" in d.reason


def test_invalid_prices(limits, spec):
    d = rm(limits).evaluate(long_signal(), ACCOUNT, tick(bid=math.nan, ask=math.nan), spec)
    assert d.code == "invalid_price"


def test_symbol_mismatch(limits, spec):
    d = rm(limits).evaluate(long_signal(), ACCOUNT, tick(), replace(spec, symbol="GOLD"))
    assert d.code == "symbol_mismatch"


def test_check_order_kill_switch_first(limits, spec):
    acc = AccountState(equity=0, balance=0, open_positions=5)
    assert rm(limits, kill=True).evaluate(long_signal(), acc, tick(), spec).code == "kill_switch"


# ---- daily loss limit -------------------------------------------------------
def tracker(limit=2.0):
    return DailyLossTracker(limit, time(17, 0), NY)


def test_trading_day_boundary_follows_new_york_dst():
    tr = tracker()
    # Winter (EST, UTC-5): 17:00 NY = 22:00 UTC
    assert tr.trading_day(datetime(2024, 1, 3, 21, 59, tzinfo=UTC)) != tr.trading_day(
        datetime(2024, 1, 3, 22, 0, tzinfo=UTC))
    assert tr.trading_day(datetime(2024, 1, 3, 22, 0, tzinfo=UTC)) == tr.trading_day(
        datetime(2024, 1, 4, 21, 59, tzinfo=UTC))
    # Summer (EDT, UTC-4): 17:00 NY = 21:00 UTC
    assert tr.trading_day(datetime(2024, 7, 3, 20, 59, tzinfo=UTC)) != tr.trading_day(
        datetime(2024, 7, 3, 21, 0, tzinfo=UTC))
    assert tr.trading_day(datetime(2024, 7, 3, 21, 30, tzinfo=UTC)) == tr.trading_day(
        datetime(2024, 7, 3, 23, 59, tzinfo=UTC))


def test_trading_day_across_dst_switch_dates():
    tr = tracker()
    # US DST starts 2024-03-10, ends 2024-11-03 (both Sundays)
    for d in (datetime(2024, 3, 8, tzinfo=UTC), datetime(2024, 3, 11, tzinfo=UTC),
              datetime(2024, 11, 1, tzinfo=UTC), datetime(2024, 11, 4, tzinfo=UTC)):
        reset_utc = datetime.combine(d.date(), time(17), NY).astimezone(UTC)
        assert tr.trading_day(reset_utc - timedelta(seconds=1)) != tr.trading_day(reset_utc)


def test_naive_time_rejected():
    with pytest.raises(ValueError):
        tracker().trading_day(datetime(2024, 1, 1))


def test_daily_limit_uses_equity_and_latches_until_reset(limits, spec):
    r = rm(limits)
    day1 = datetime(2024, 1, 3, 22, 30, tzinfo=UTC)  # just after reset: snapshot
    r.observe_equity(day1, 10_000)
    # open loss counts: equity 9,810 = -1.9% -> still allowed
    acc = AccountState(equity=9_810, balance=10_000, open_positions=0)
    assert r.evaluate(long_signal(ts=day1), acc, tick(ts=day1 + timedelta(hours=1)), spec).approved
    # equity 9,795 = -2.05% -> blocked, even though balance is untouched
    r.observe_equity(day1 + timedelta(hours=2), 9_795)
    acc = AccountState(equity=9_795, balance=10_000, open_positions=1)
    d = r.evaluate(long_signal(), replace(acc, open_positions=0), tick(ts=day1 + timedelta(hours=2)), spec)
    assert d.code == "daily_loss_limit" and "2.05%" in d.reason
    # equity recovers the same day -> still blocked (rest of the day)
    later = day1 + timedelta(hours=10)
    d = r.evaluate(long_signal(), AccountState(10_100, 10_100, 0), tick(ts=later), spec)
    assert d.code == "daily_loss_limit"
    # next trading day (after 17:00 NY = 22:00 UTC) -> new snapshot, allowed again
    next_day = datetime(2024, 1, 4, 22, 5, tzinfo=UTC)
    assert r.evaluate(long_signal(), AccountState(9_700, 9_700, 0), tick(ts=next_day), spec).approved
    assert r.daily.start_equity == 9_700


def test_daily_limit_does_not_close_positions(limits, spec):
    """The risk manager only vetoes new orders; it has no ability to close positions."""
    assert not any(hasattr(RiskManager, n) for n in ("close_position", "close_all", "flatten"))


# ---- logging ------------------------------------------------------------------
def test_every_decision_is_logged_with_reason(limits, spec, tmp_path):
    log = TradeLogger(tmp_path, filename="risk.jsonl")
    r = rm(limits, logger=log)
    r.evaluate(long_signal(), ACCOUNT, tick(), spec)
    r.evaluate(long_signal(), replace(ACCOUNT, open_positions=1), tick(), spec)
    lines = [json.loads(x) for x in (tmp_path / "risk.jsonl").read_text().splitlines()]
    assert [x["event"] for x in lines] == ["risk_approved", "rejection"]
    assert lines[0]["order"]["volume"] == 0.04
    assert lines[1]["code"] == "max_open_positions"
    assert "max open positions" in lines[1]["reason"]


# ---- independence ---------------------------------------------------------------
RISK_DIR = Path(__file__).resolve().parents[1] / "src" / "goldbot" / "risk"
FORBIDDEN = ("goldbot.strategy", "goldbot.backtest", "goldbot.execution", "goldbot.broker",
             "anthropic", "openai", "langchain", "llm", "goldbot.llm")


def test_risk_package_does_not_import_strategy_or_llm():
    files = list(RISK_DIR.glob("*.py"))
    assert files
    for f in files:
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for n in names:
                assert not any(n == b or n.startswith(b + ".") for b in FORBIDDEN), f"{f.name} imports {n}"
