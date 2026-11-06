"""Trend-pullback strategy v1 (pure functions, no I/O, no broker calls).

Rules (long; short is the mirror image):
1. Trend filter (higher timeframe, e.g. H1): close > EMA(slow) and EMA(fast) > EMA(slow).
2. Pullback: in the previous ``pullback_lookback`` signal-timeframe bars, a low
   touched the EMA zone (low <= upper of EMA20/EMA50) and none of those bars
   closed below the lower EMA (a deeper break invalidates the setup).
3. Trigger: the current (closed) candle is bullish and closes above the zone.
4. RSI(14) < overbought.
5. Stop loss: lowest low of the last ``swing_lookback`` bars minus
   ``stop_buffer_atr`` * ATR. Take profit: entry + ``take_profit_r`` * risk.

Entry and take profit are measured from the EXPECTED FILL price: the ask for
longs, the bid for shorts. At signal time that is the trigger candle's ask
close (bid close for shorts). When the real fill price is known (next tick /
next bar open), ``reprice_to_fill`` re-anchors entry and take profit to it so
the trade still targets exactly ``take_profit_r`` R. The stop is unchanged.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import pandas as pd

from goldbot.config import TIMEFRAME_RULES, StrategySettings
from goldbot.models import Direction, Signal
from goldbot.strategy.indicators import (
    align_to_lower,
    atr,
    ema,
    resample_ohlc,
    rolling_any,
    rsi,
)

FEATURE_COLUMNS = [
    "open", "high", "low", "close", "entry_long",
    "zone_fast", "zone_slow", "zone_top", "zone_bot",
    "rsi", "atr", "swing_low", "swing_high",
    "htf_close", "htf_ema_fast", "htf_ema_slow", "bias",
    "pullback_long", "broke_long", "pullback_short", "broke_short",
]


def compute_features(
    candles: pd.DataFrame,
    params: StrategySettings,
    signal_tf: str = "M15",
    trend_tf: str = "H1",
) -> pd.DataFrame:
    """Compute all strategy inputs. Row ``t`` uses only candles ``<= t`` (closed bars)."""
    if candles.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS, index=candles.index)
    p = params
    f = candles[["open", "high", "low", "close"]].copy()
    # expected long fill = ask; without ask data fall back to bid (execution reprices anyway)
    f["entry_long"] = candles["ask_close"] if "ask_close" in candles.columns else candles["close"]

    # Signal timeframe indicators
    f["zone_fast"] = ema(f["close"], p.ema_zone_fast)
    f["zone_slow"] = ema(f["close"], p.ema_zone_slow)
    f["zone_top"] = f[["zone_fast", "zone_slow"]].max(axis=1, skipna=False)
    f["zone_bot"] = f[["zone_fast", "zone_slow"]].min(axis=1, skipna=False)
    f["rsi"] = rsi(f["close"], p.rsi_period)
    f["atr"] = atr(f, p.atr_period)
    f["swing_low"] = f["low"].rolling(p.swing_lookback, min_periods=p.swing_lookback).min()
    f["swing_high"] = f["high"].rolling(p.swing_lookback, min_periods=p.swing_lookback).max()

    # Trend timeframe, aligned so only fully closed higher bars are visible
    htf_period = pd.Timedelta(TIMEFRAME_RULES[trend_tf])
    ltf_period = pd.Timedelta(TIMEFRAME_RULES[signal_tf])
    htf = resample_ohlc(candles, TIMEFRAME_RULES[trend_tf])
    htf_feat = pd.DataFrame(
        {
            "htf_close": htf["close"],
            "htf_ema_fast": ema(htf["close"], p.ema_trend_fast),
            "htf_ema_slow": ema(htf["close"], p.ema_trend_slow),
        }
    )
    f = f.join(align_to_lower(htf_feat, htf_period, f.index, ltf_period))

    up = (f["htf_close"] > f["htf_ema_slow"]) & (f["htf_ema_fast"] > f["htf_ema_slow"])
    down = (f["htf_close"] < f["htf_ema_slow"]) & (f["htf_ema_fast"] < f["htf_ema_slow"])
    f["bias"] = 0
    f.loc[up, "bias"] = 1
    f.loc[down, "bias"] = -1

    # Pullback history over the previous N bars (excluding the trigger bar)
    n = p.pullback_lookback
    f["pullback_long"] = rolling_any(f["low"] <= f["zone_top"], n, include_current=False)
    f["broke_long"] = rolling_any(f["close"] < f["zone_bot"], n, include_current=False)
    f["pullback_short"] = rolling_any(f["high"] >= f["zone_bot"], n, include_current=False)
    f["broke_short"] = rolling_any(f["close"] > f["zone_top"], n, include_current=False)
    return f[FEATURE_COLUMNS]


def _fmt(x: float) -> str:
    return f"{x:.2f}"


def evaluate(row: pd.Series, time: datetime, symbol: str, params: StrategySettings) -> Signal | None:
    """Apply the entry rules to one feature row (the just-closed candle)."""
    p = params
    needed = ["zone_top", "zone_bot", "rsi", "atr", "swing_low", "swing_high", "htf_ema_slow"]
    if row[needed].isna().any():
        return None

    close, open_ = row["close"], row["open"]
    trend_txt = (
        f"HTF close {_fmt(row['htf_close'])} vs EMA{p.ema_trend_slow} {_fmt(row['htf_ema_slow'])}, "
        f"EMA{p.ema_trend_fast} {_fmt(row['htf_ema_fast'])}"
    )
    zone_txt = (
        f"EMA{p.ema_zone_fast}/{p.ema_zone_slow} zone [{_fmt(row['zone_bot'])}-{_fmt(row['zone_top'])}]"
    )

    if (
        row["bias"] == 1
        and row["pullback_long"]
        and not row["broke_long"]
        and close > open_
        and close > row["zone_top"]
        and row["rsi"] < p.rsi_overbought
    ):
        stop = row["swing_low"] - p.stop_buffer_atr * row["atr"]
        entry = row["entry_long"] if pd.notna(row["entry_long"]) else close
        if stop >= close:
            return None
        risk = entry - stop
        tp = entry + p.take_profit_r * risk
        reason = (
            f"LONG: uptrend ({trend_txt}); pulled back into {zone_txt} within last "
            f"{p.pullback_lookback} bars; bullish close {_fmt(close)} back above zone; "
            f"RSI {row['rsi']:.1f} < {p.rsi_overbought:g}; SL {_fmt(stop)} = {p.swing_lookback}-bar "
            f"swing low {_fmt(row['swing_low'])} - {p.stop_buffer_atr:g} ATR; TP {_fmt(tp)} = "
            f"{p.take_profit_r:g}R from expected fill (ask) {_fmt(entry)}"
        )
        return Signal(symbol, Direction.BUY, entry, stop, tp, reason, time)

    if (
        row["bias"] == -1
        and row["pullback_short"]
        and not row["broke_short"]
        and close < open_
        and close < row["zone_bot"]
        and row["rsi"] > p.rsi_oversold
    ):
        stop = row["swing_high"] + p.stop_buffer_atr * row["atr"]
        if stop <= close:
            return None
        risk = stop - close
        tp = close - p.take_profit_r * risk
        if tp <= 0:
            return None
        reason = (
            f"SHORT: downtrend ({trend_txt}); pulled back into {zone_txt} within last "
            f"{p.pullback_lookback} bars; bearish close {_fmt(close)} back below zone; "
            f"RSI {row['rsi']:.1f} > {p.rsi_oversold:g}; SL {_fmt(stop)} = {p.swing_lookback}-bar "
            f"swing high {_fmt(row['swing_high'])} + {p.stop_buffer_atr:g} ATR; TP {_fmt(tp)} = "
            f"{p.take_profit_r:g}R from expected fill (bid) {_fmt(close)}"
        )
        return Signal(symbol, Direction.SELL, close, stop, tp, reason, time)

    return None


def reprice_to_fill(signal: Signal, fill_price: float, take_profit_r: float) -> Signal | None:
    """Re-anchor entry and take profit to the actual expected fill price.

    Keeps the stop. Returns None if the fill is already at/beyond the stop.
    This is part of the strategy's TP rule: TP = fill +/- take_profit_r * |fill - SL|.
    """
    if signal.direction is Direction.BUY:
        if fill_price <= signal.stop_loss:
            return None
        tp = fill_price + take_profit_r * (fill_price - signal.stop_loss)
    else:
        if fill_price >= signal.stop_loss:
            return None
        tp = fill_price - take_profit_r * (signal.stop_loss - fill_price)
        if tp <= 0:
            return None
    if fill_price == signal.entry and tp == signal.take_profit:
        return signal
    return replace(
        signal,
        entry=fill_price,
        take_profit=tp,
        reason=f"{signal.reason}; repriced to fill {_fmt(fill_price)} -> TP {_fmt(tp)}",
    )


def signal_at(
    features: pd.DataFrame, i: int, symbol: str, params: StrategySettings
) -> Signal | None:
    """Signal for the candle at position ``i``, evaluated at that candle's close."""
    return evaluate(features.iloc[i], features.index[i].to_pydatetime(), symbol, params)


def generate_signal(
    candles: pd.DataFrame,
    params: StrategySettings,
    symbol: str,
    signal_tf: str = "M15",
    trend_tf: str = "H1",
) -> Signal | None:
    """Live entry point: signal for the LAST row of ``candles``.

    The caller must pass only closed candles (drop the still-forming one).
    """
    if candles.empty:
        return None
    feats = compute_features(candles, params, signal_tf, trend_tf)
    return signal_at(feats, len(feats) - 1, symbol, params)
