"""Engine arithmetic with hand-crafted candles and injected signals (risk manager in the loop)."""

from dataclasses import replace

import pandas as pd
import pytest

from goldbot.backtest.engine import BacktestConfig, run_backtest
from goldbot.backtest.metrics import compute_metrics, max_drawdown
from goldbot.models import Direction, Signal, SymbolSpec
from goldbot.risk import RiskManager
from goldbot.strategy.trend_pullback import reprice_to_fill

SPREAD = 0.5


def spec(**kw):
    base = dict(symbol="XAUUSD", contract_size=100.0, tick_size=0.01, tick_value=1.0,
                volume_min=0.01, volume_step=0.01, volume_max=100.0, commission_per_lot=7.0)
    base.update(kw)
    return SymbolSpec(**base)


def cfg(**kw):
    base = dict(symbol="XAUUSD", initial_balance=10_000.0, spec=spec())
    base.update(kw)
    return BacktestConfig(**base)


@pytest.fixture
def risk_settings(settings):
    return replace(settings.risk, risk_per_trade_pct=1.0, max_spread_bps=5.0)


def run(df, signal_fn, risk_settings, c=None, reprice=True):
    rm = RiskManager(risk_settings, kill_switch=lambda: False)
    rp = (lambda s, f: reprice_to_fill(s, f, 2.0)) if reprice else None
    return run_backtest(df, signal_fn, c or cfg(), rm, rp)


def bars(rows, start="2024-01-02 10:00", spread=SPREAD):
    idx = pd.date_range(start, periods=len(rows), freq="15min", tz="UTC", name="time")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)
    for c in ("open", "high", "low", "close"):
        df[f"ask_{c}"] = df[c] + spread
    return df


def sig(df, i, direction, sl, tp, entry=None):
    return Signal("XAUUSD", direction, entry or df["close"].iloc[i], sl, tp, "test",
                  df.index[i].to_pydatetime())


def once(df, i, direction, sl, tp):
    s = sig(df, i, direction, sl, tp)
    return lambda j: s if j == i else None


def test_long_take_profit_repriced_from_ask_fill(risk_settings):
    df = bars([
        [2000, 2001, 1999, 2000],   # 0: signal at close
        [2000, 2002, 1999, 2001],   # 1: enter at ask open 2000.5 -> TP = 2000.5 + 2*10.5 = 2021.5
        [2001, 2025, 2000, 2020],   # 2: bid high 2025 >= 2021.5
        [2020, 2021, 2019, 2020],
    ])
    r = run(df, once(df, 0, Direction.BUY, 1990, 2020), risk_settings)
    (t,) = r.trades
    assert t.entry_price == 2000.5
    assert t.take_profit == pytest.approx(2021.5)
    assert t.exit_reason == "take_profit" and t.exit_price == pytest.approx(2021.5)
    # 1% of 10000 = 100; per lot = 10.5*100 + 7 commission = 1057 -> 0.0946 -> 0.09
    assert t.volume == pytest.approx(0.09)
    assert t.gross_pnl == pytest.approx(21.0 * 0.09 * 100)
    assert t.commission == pytest.approx(7 * 0.09)
    assert t.risk_amount == pytest.approx(0.09 * 1057)
    assert r.equity.iloc[-1] == pytest.approx(10_000 + t.pnl)


def test_without_reprice_long_rr_is_below_2_and_rejected(risk_settings):
    """The old behaviour: TP 2R from the bid close is < 2R from the ask fill."""
    df = bars([[2000, 2001, 1999, 2000]] * 3)
    r = run(df, once(df, 0, Direction.BUY, 1990, 2020), risk_settings, reprice=False)
    assert r.trades == [] and r.skipped == {"risk_reward_below_min": 1}


def test_long_tp_needs_bid_not_ask(risk_settings):
    df = bars([
        [2000, 2001, 1999, 2000],
        [2000, 2001, 1999, 2000],   # entry at 2000.5, TP 2021.5
        [2010, 2021.4, 2009, 2015], # ask high 2021.9 > TP but bid high 2021.4 < TP -> no exit
        [2015, 2021.5, 2014, 2016], # bid reaches 2021.5 -> exit
    ])
    (t,) = run(df, once(df, 0, Direction.BUY, 1990, 2020), risk_settings).trades
    assert t.exit_reason == "take_profit" and t.exit_time == df.index[3]


def test_long_stop_on_bid_and_sl_first_when_both_in_bar(risk_settings):
    df = bars([
        [2000, 2001, 1999, 2000],
        [2000, 2030, 1980, 2000],   # entry bar: both SL and TP inside -> SL
        [2000, 2001, 1999, 2000],
    ])
    (t,) = run(df, once(df, 0, Direction.BUY, 1990, 2020), risk_settings).trades
    assert t.exit_reason == "stop_loss" and t.exit_price == 1990
    assert t.pnl == pytest.approx(-t.risk_amount)
    assert t.r_multiple == pytest.approx(-1.0)


def test_long_stop_not_hit_by_ask_low(risk_settings):
    df = bars([
        [2000, 2001, 1999, 2000],
        [2000, 2001, 1999, 2000],
        [1995, 1996, 1990.01, 1995],  # bid low just above SL
        [1995, 1996, 1994, 1995],
    ])
    (t,) = run(df, once(df, 0, Direction.BUY, 1990, 2020), risk_settings).trades
    assert t.exit_reason == "end_of_data"


def test_short_stop_uses_ask(risk_settings):
    df = bars([
        [2000, 2001, 1999, 2000],
        [2000, 2001, 1999, 2000],    # enter short at bid open 2000
        [2000, 2009.6, 1995, 1996],  # ask high 2010.1 >= SL 2010 -> stopped
        [1996, 1997, 1995, 1996],
    ])
    (t,) = run(df, once(df, 0, Direction.SELL, 2010, 1980), risk_settings).trades
    assert t.entry_price == 2000
    assert t.exit_reason == "stop_loss" and t.exit_price == 2010


def test_short_take_profit_uses_ask(risk_settings):
    df = bars([
        [2000, 2001, 1999, 2000],
        [2000, 2001, 1999, 2000],     # TP = 2000 - 2*10 = 1980
        [2000, 2001, 1979.6, 1990],   # ask low 1980.1 > TP -> not hit
        [1990, 1991, 1979.4, 1985],   # ask low 1979.9 -> hit
    ])
    (t,) = run(df, once(df, 0, Direction.SELL, 2010, 1980), risk_settings).trades
    assert t.exit_reason == "take_profit" and t.exit_time == df.index[3]
    assert t.exit_price == pytest.approx(1980)


def test_gap_through_stop_exits_at_open(risk_settings):
    df = bars([
        [2000, 2001, 1999, 2000],
        [2000, 2001, 1999, 2000],
        [1985, 1986, 1980, 1984],    # opens below SL 1990
    ])
    (t,) = run(df, once(df, 0, Direction.BUY, 1990, 2020), risk_settings).trades
    assert t.exit_reason == "stop_loss_gap" and t.exit_price == 1985
    assert t.r_multiple < -1


def test_no_entry_across_market_gap(risk_settings):
    df = bars([[2000, 2001, 1999, 2000]] * 3)
    df.index = df.index[:1].append(df.index[1:] + pd.Timedelta("2D"))
    r = run(df, once(df, 0, Direction.BUY, 1990, 2020), risk_settings)
    assert r.trades == [] and r.skipped == {"market_gap": 1}


def test_risk_manager_spread_rejection(risk_settings):
    df = bars([[2000, 2001, 1999, 2000]] * 3, spread=2.0)  # 10 bps > 5
    r = run(df, once(df, 0, Direction.BUY, 1990, 2020), risk_settings)
    assert r.trades == [] and r.skipped == {"spread_above_max": 1}


def test_risk_manager_min_stop_vs_spread(risk_settings):
    df = bars([[2000, 2001, 1999, 2000]] * 3, spread=0.5)
    r = run(df, once(df, 0, Direction.BUY, 1999.2, 2010), risk_settings)  # stop 1.3 < 3 x 0.5
    assert r.skipped == {"stop_too_close_spread": 1}


def test_entry_skipped_if_open_beyond_stop(risk_settings):
    df = bars([[2000, 2001, 1999, 2000], [1989, 1990, 1988, 1989], [1989, 1990, 1988, 1989]])
    r = run(df, once(df, 0, Direction.BUY, 1990, 2020), risk_settings)
    assert r.trades == [] and r.skipped == {"gapped_past_stop": 1}


def test_max_open_positions_via_risk_manager(risk_settings):
    df = bars([[2000, 2001, 1999, 2000]] * 6)
    s0, s2 = sig(df, 0, Direction.BUY, 1990, 2020), sig(df, 2, Direction.BUY, 1990, 2020)
    r = run(df, lambda j: {0: s0, 2: s2}.get(j), risk_settings)
    assert r.skipped == {"max_open_positions": 1}
    assert len(r.trades) == 1 and r.trades[0].exit_reason == "end_of_data"


def test_two_positions_allowed_when_limit_is_two(risk_settings):
    df = bars([[2000, 2001, 1999, 2000]] * 6)
    s0, s2 = sig(df, 0, Direction.BUY, 1990, 2020), sig(df, 2, Direction.BUY, 1990, 2020)
    r = run(df, lambda j: {0: s0, 2: s2}.get(j), replace(risk_settings, max_open_positions=2))
    assert len(r.trades) == 2 and not r.skipped


def test_daily_loss_limit_blocks_until_ny_reset(risk_settings):
    # 2024-01-02 is winter: 17:00 New York = 22:00 UTC
    rs = replace(risk_settings, risk_per_trade_pct=2.0, daily_loss_limit_pct=2.0)
    df = bars(
        [[2000, 2001, 1999, 2000]] * 2
        + [[1985, 1986, 1984, 1986]]          # 20:30 gaps through SL 1990 -> ~-2.9%
        + [[1986, 1987, 1985, 1986]] * 10,
        start="2024-01-02 20:00",
    )
    s0 = sig(df, 0, Direction.BUY, 1990, 2020)
    s_same_day = sig(df, 3, Direction.BUY, 1975, 2100)   # entry 21:15 UTC: same trading day
    s_next_day = sig(df, 8, Direction.BUY, 1975, 2100)   # entry 22:15 UTC: after the reset
    r = run(df, lambda j: {0: s0, 3: s_same_day, 8: s_next_day}.get(j), rs)
    assert r.skipped == {"daily_loss_limit": 1}
    assert len(r.trades) == 2
    assert r.trades[1].entry_time == df.index[9].to_pydatetime()


def test_volume_step_from_broker_spec(risk_settings):
    df = bars([[2000, 2001, 1999, 2000]] * 3)
    c = cfg(spec=spec(volume_min=0.1, volume_step=0.1),
            initial_balance=100_000.0)
    # 1% of 100k = 1000 / 1057 per lot = 0.946 -> 0.9 with step 0.1
    (t,) = run(df, once(df, 0, Direction.BUY, 1990, 2020), risk_settings, c).trades
    assert t.volume == pytest.approx(0.9)


def test_fixed_spread_without_ask_columns(risk_settings):
    df = bars([[2000, 2001, 1999, 2000]] * 3)[["open", "high", "low", "close"]]
    r = run(df, once(df, 0, Direction.BUY, 1990, 2020), risk_settings, cfg(fixed_spread=0.3))
    assert r.trades[0].entry_price == pytest.approx(2000.3)


def test_max_drawdown():
    eq = pd.Series([100, 120, 90, 130, 117], index=pd.date_range("2024", periods=5, tz="UTC"))
    money, pct, _ = max_drawdown(eq)
    assert money == 30 and pct == pytest.approx(25.0)


def test_metrics_basic(risk_settings):
    df = bars([
        [2000, 2001, 1999, 2000], [2000, 2002, 1999, 2001], [2001, 2025, 2000, 2020],
        [2020, 2021, 2019, 2020], [2020, 2021, 2019, 2020], [2020, 2021, 2005, 2010],
    ])
    s0 = sig(df, 0, Direction.BUY, 1990, 2020)
    s3 = sig(df, 3, Direction.BUY, 2010, 2040)
    m = compute_metrics(run(df, lambda j: {0: s0, 3: s3}.get(j), risk_settings))
    assert m["trades"] == 2
    assert m["win_rate_pct"] == 50.0
    assert m["profit_factor"] > 1
    assert m["by_year"][2024]["trades"] == 2
