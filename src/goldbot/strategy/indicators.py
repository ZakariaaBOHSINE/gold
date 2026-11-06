"""Causal technical indicators and lookahead-free multi-timeframe alignment.

Every function here is causal: the value at row ``t`` depends only on rows
``<= t``. This is verified by tests/test_lookahead.py.
"""

from __future__ import annotations

import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average (alpha = 2/(period+1)), NaN during warm-up."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI. 100 when there are no losses in the window, NaN during warm-up."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    out = 100.0 - 100.0 / (1.0 + rs)
    # avg_loss == 0 -> rs = inf -> 100; both zero -> NaN -> treat as neutral 50
    out = out.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return out.where(avg_gain.notna())


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's Average True Range."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate candles to a higher timeframe. Index = bar OPEN time (left label).

    Empty bins (weekends, breaks) are dropped. The last bar may be incomplete;
    ``align_to_lower`` makes sure an incomplete bar is never used.
    """
    agg = df[["open", "high", "low", "close"]].resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    )
    return agg.dropna(subset=["close"])


def align_to_lower(
    higher: pd.DataFrame,
    higher_period: pd.Timedelta,
    lower_index: pd.DatetimeIndex,
    lower_period: pd.Timedelta,
) -> pd.DataFrame:
    """Attach higher-timeframe rows to lower-timeframe bars without lookahead.

    A higher bar opening at T is only *known* at T + higher_period (its close).
    A lower bar opening at t is evaluated at its close, t + lower_period.
    Each lower bar gets the most recent higher bar with
    ``T + higher_period <= t + lower_period``.
    """
    right = higher.copy()
    right["_available_at"] = right.index + higher_period
    right = right.reset_index(drop=True).sort_values("_available_at")
    left = pd.DataFrame({"_decision_at": lower_index + lower_period})
    merged = pd.merge_asof(
        left, right, left_on="_decision_at", right_on="_available_at", direction="backward"
    )
    merged.index = lower_index
    return merged.drop(columns=["_decision_at", "_available_at"])


def rolling_any(mask: pd.Series, window: int, include_current: bool) -> pd.Series:
    """True where ``mask`` was True in the last ``window`` rows (optionally excluding the current)."""
    m = mask.astype(float)
    if not include_current:
        m = m.shift(1)
    return m.rolling(window, min_periods=window).max().fillna(0.0).astype(bool)


__all__ = ["align_to_lower", "atr", "ema", "resample_ohlc", "rolling_any", "rsi"]
