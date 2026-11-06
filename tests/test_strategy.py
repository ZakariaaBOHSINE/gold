import numpy as np
import pandas as pd
import pytest

from goldbot.models import Direction
from goldbot.strategy.indicators import atr, ema, resample_ohlc, rolling_any, rsi
from goldbot.strategy.trend_pullback import compute_features, evaluate

T = pd.Timestamp("2024-01-02 13:00", tz="UTC").to_pydatetime()


# ---- indicators ---------------------------------------------------------
def test_ema_matches_recursive_definition():
    s = pd.Series([1.0, 2, 3, 4, 5, 6])
    out = ema(s, 3)
    assert out.iloc[:2].isna().all()
    a = 2 / 4
    expected = [1.0]
    for x in s.iloc[1:]:
        expected.append(a * x + (1 - a) * expected[-1])
    assert np.allclose(out.iloc[2:], expected[2:])


def test_rsi_extremes_and_known_value():
    up = pd.Series(np.arange(1.0, 40.0))
    assert rsi(up, 14).iloc[-1] == pytest.approx(100.0)
    assert rsi(-up, 14).iloc[-1] == pytest.approx(0.0)
    flat = pd.Series([5.0] * 30)
    assert rsi(flat, 14).iloc[-1] == 50.0
    # alternating +1/-1 => avg gain ~= avg loss => RSI ~ 50
    alt = pd.Series(np.cumsum([1.0, -1.0] * 50))
    assert 40 < rsi(alt, 14).iloc[-1] < 60
    assert rsi(up, 14).iloc[:14].isna().all()


def test_atr_constant_range():
    df = pd.DataFrame({"high": [11.0] * 30, "low": [9.0] * 30, "close": [10.0] * 30})
    assert atr(df, 14).iloc[-1] == pytest.approx(2.0)


def test_resample_ohlc():
    idx = pd.date_range("2024-01-01 10:00", periods=8, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {"open": range(8), "high": [x + 10 for x in range(8)], "low": range(8), "close": [x + 1 for x in range(8)]},
        index=idx, dtype=float,
    )
    h = resample_ohlc(df, "1h")
    assert list(h.index.hour) == [10, 11]
    assert h.iloc[0].tolist() == [0, 13, 0, 4]


def test_rolling_any_excludes_current():
    m = pd.Series([False, False, True, False, False, False])
    assert rolling_any(m, 2, include_current=False).tolist() == [False, False, False, True, True, False]
    assert rolling_any(m, 2, include_current=True).tolist() == [False, False, True, True, False, False]


# ---- rules --------------------------------------------------------------
def _row(**kw):
    base = dict(
        open=2000.0, high=2004.0, low=1999.0, close=2003.0, entry_long=2003.0,
        zone_fast=2001.0, zone_slow=1998.0, zone_top=2001.0, zone_bot=1998.0,
        rsi=55.0, atr=2.0, swing_low=1995.0, swing_high=2010.0,
        htf_close=2005.0, htf_ema_fast=2000.0, htf_ema_slow=1990.0, bias=1,
        pullback_long=True, broke_long=False, pullback_short=False, broke_short=False,
    )
    base.update(kw)
    return pd.Series(base)


def test_long_signal(settings):
    s = evaluate(_row(), T, "XAUUSD", settings.strategy)
    assert s is not None and s.direction is Direction.BUY
    assert s.entry == 2003.0
    assert s.stop_loss == pytest.approx(1995.0 - 0.1 * 2.0)
    assert s.take_profit == pytest.approx(2003.0 + 2 * (2003.0 - s.stop_loss))
    assert s.risk_reward == pytest.approx(2.0)
    assert "LONG" in s.reason and "RSI 55.0" in s.reason and "swing low" in s.reason


@pytest.mark.parametrize(
    "change",
    [
        dict(bias=0),                        # no trend
        dict(bias=-1),                       # opposite trend
        dict(pullback_long=False),           # no pullback
        dict(broke_long=True),               # pullback went through the zone
        dict(close=1999.5),                  # bearish candle
        dict(open=2000.5, close=2000.8),     # bullish but still inside the zone
        dict(rsi=70.0),                      # overbought
        dict(rsi=float("nan")),              # warm-up
        dict(swing_low=2004.0),              # stop would be above entry
    ],
)
def test_long_rule_rejections(settings, change):
    assert evaluate(_row(**change), T, "XAUUSD", settings.strategy) is None


def _short_row(**kw):
    base = dict(
        open=2000.0, high=2001.0, low=1996.0, close=1997.0, entry_long=1997.3,
        zone_fast=1999.0, zone_slow=2002.0, zone_top=2002.0, zone_bot=1999.0,
        rsi=45.0, atr=2.0, swing_low=1990.0, swing_high=2005.0,
        htf_close=1995.0, htf_ema_fast=2000.0, htf_ema_slow=2010.0, bias=-1,
        pullback_long=False, broke_long=False, pullback_short=True, broke_short=False,
    )
    base.update(kw)
    return pd.Series(base)


def test_short_signal(settings):
    s = evaluate(_short_row(), T, "XAUUSD", settings.strategy)
    assert s is not None and s.direction is Direction.SELL
    assert s.stop_loss == pytest.approx(2005.0 + 0.2)
    assert s.take_profit == pytest.approx(1997.0 - 2 * (s.stop_loss - 1997.0))
    assert "SHORT" in s.reason


@pytest.mark.parametrize(
    "change",
    [dict(bias=1), dict(pullback_short=False), dict(broke_short=True), dict(close=2000.5),
     dict(rsi=30.0), dict(swing_high=1996.0)],
)
def test_short_rule_rejections(settings, change):
    assert evaluate(_short_row(**change), T, "XAUUSD", settings.strategy) is None


def test_bias_from_htf(candles, settings):
    f = compute_features(candles, settings.strategy)
    up = f[f["bias"] == 1]
    assert (up["htf_close"] > up["htf_ema_slow"]).all()
    assert (up["htf_ema_fast"] > up["htf_ema_slow"]).all()
    down = f[f["bias"] == -1]
    assert (down["htf_close"] < down["htf_ema_slow"]).all()
    assert len(up) and len(down)


def test_all_generated_signals_valid(candles, settings):
    from goldbot.strategy.trend_pullback import signal_at

    f = compute_features(candles, settings.strategy)
    sigs = [s for i in range(len(f)) if (s := signal_at(f, i, "XAUUSD", settings.strategy))]
    assert sigs
    for s in sigs:
        assert s.reason
        assert s.risk_reward >= settings.risk.min_risk_reward - 1e-9


def test_long_tp_measured_from_ask(settings):
    s = evaluate(_row(entry_long=2003.5), T, "XAUUSD", settings.strategy)
    assert s.entry == 2003.5
    assert s.take_profit == pytest.approx(2003.5 + 2 * (2003.5 - s.stop_loss))
    assert "expected fill (ask) 2003.50" in s.reason


def test_short_tp_measured_from_bid(settings):
    s = evaluate(_short_row(), T, "XAUUSD", settings.strategy)
    assert s.entry == 1997.0  # bid close; ask (entry_long) is ignored for shorts


def test_reprice_to_fill_keeps_2r():
    from goldbot.strategy.trend_pullback import reprice_to_fill
    from goldbot.models import Signal

    long = Signal("XAUUSD", Direction.BUY, 2003.0, 1995.0, 2019.0, "x", T)
    r = reprice_to_fill(long, 2004.0, 2.0)
    assert r.entry == 2004.0 and r.stop_loss == 1995.0
    assert r.take_profit == pytest.approx(2004.0 + 2 * 9.0)
    assert r.risk_reward == pytest.approx(2.0)
    assert "repriced" in r.reason
    assert reprice_to_fill(long, 1995.0, 2.0) is None          # fill at stop
    assert reprice_to_fill(long, 2003.0, 2.0) is long           # unchanged

    short = Signal("XAUUSD", Direction.SELL, 1997.0, 2005.0, 1981.0, "x", T)
    r = reprice_to_fill(short, 1996.0, 2.0)
    assert r.take_profit == pytest.approx(1996.0 - 2 * 9.0)
    assert reprice_to_fill(short, 2006.0, 2.0) is None
