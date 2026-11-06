"""Lookahead-bias tests: nothing computed for bar t may depend on bars after t."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from goldbot.backtest.runner import run_strategy_backtest
from goldbot.strategy.trend_pullback import (
    FEATURE_COLUMNS,
    compute_features,
    generate_signal,
    signal_at,
)

REAL_DATA = Path(__file__).resolve().parents[1] / "data" / "xauusd_m15_2021_2026.csv"


def _signal_bars(feats, settings):
    return [
        i for i in range(len(feats))
        if signal_at(feats, i, settings.symbol, settings.strategy) is not None
    ]


def test_synthetic_data_produces_signals(candles, settings):
    feats = compute_features(candles, settings.strategy)
    assert len(_signal_bars(feats, settings)) >= 10, "test data must exercise the strategy"


def _assert_truncation_invariant(candles, settings, points):
    full = compute_features(candles, settings.strategy)
    for t in points:
        prefix = candles.iloc[: t + 1]
        part = compute_features(prefix, settings.strategy)
        pd.testing.assert_series_equal(
            part.iloc[-1][FEATURE_COLUMNS], full.iloc[t][FEATURE_COLUMNS],
            check_names=False, rtol=1e-9, obj=f"features at bar {t}",
        )
        s_full = signal_at(full, t, settings.symbol, settings.strategy)
        s_live = generate_signal(prefix, settings.strategy, settings.symbol)
        assert s_full == s_live, f"signal differs at bar {t}"


def test_truncation_invariance(candles, settings):
    """Features/signals at bar t are identical whether or not future bars exist."""
    full = compute_features(candles, settings.strategy)
    sig_bars = _signal_bars(full, settings)
    rng = np.random.default_rng(1)
    random_bars = rng.choice(np.arange(900, len(candles)), size=40, replace=False)
    points = sorted(set(sig_bars[:30]) | set(random_bars.tolist()) | {899, 900, len(candles) - 1})
    _assert_truncation_invariant(candles, settings, points)


def test_future_scramble_does_not_change_past(candles, settings):
    """Rewriting every bar after t must not change any feature or signal up to t."""
    t = 3500
    scrambled = candles.copy()
    rng = np.random.default_rng(99)
    future = scrambled.iloc[t + 1:]
    factor = rng.uniform(0.8, 1.2, size=len(future))
    for c in ("open", "high", "low", "close", "ask_open", "ask_high", "ask_low", "ask_close"):
        scrambled.iloc[t + 1:, scrambled.columns.get_loc(c)] = future[c].to_numpy() * factor
    a = compute_features(candles, settings.strategy).iloc[: t + 1]
    b = compute_features(scrambled, settings.strategy).iloc[: t + 1]
    pd.testing.assert_frame_equal(a, b)
    assert _signal_bars(a, settings) == _signal_bars(b, settings)


def test_higher_timeframe_only_visible_after_close(settings):
    """The H1 bar 13:00-14:00 may only affect M15 bars that close at/after 14:00.

    We spike the close of the 13:45 M15 bar, which is the close of the 13:00 H1 bar.
    M15 bars opening 13:00/13:15/13:30 close before 14:00 and must not see it;
    the 13:45 bar closes at 14:00 and must.
    """
    idx = pd.date_range("2024-01-01", periods=1200, freq="15min", tz="UTC")
    base = pd.DataFrame({"open": 2000.0, "high": 2001.0, "low": 1999.0, "close": 2000.0}, index=idx)
    spiked = base.copy()
    spiked.loc[pd.Timestamp("2024-01-10 13:45", tz="UTC"), ["high", "close"]] = [2500.0, 2500.0]

    fa = compute_features(base, settings.strategy)
    fb = compute_features(spiked, settings.strategy)
    before = fb.index < pd.Timestamp("2024-01-10 13:45", tz="UTC")
    htf_cols = ["htf_close", "htf_ema_fast", "htf_ema_slow", "bias"]
    pd.testing.assert_frame_equal(fa.loc[before, htf_cols], fb.loc[before, htf_cols])
    assert fb.loc[pd.Timestamp("2024-01-10 13:45", tz="UTC"), "htf_close"] == 2500.0
    assert fb.loc[pd.Timestamp("2024-01-10 13:30", tz="UTC"), "htf_close"] == 2000.0


def test_incomplete_last_htf_bar_is_never_used(candles, settings):
    """With data ending mid-hour, the partial H1 bar must not be used."""
    prefix = candles.iloc[: 2000 + 2]  # ends at :15 or :30 into an hour
    f = compute_features(prefix, settings.strategy)
    last = prefix.index[-1]
    assert last.minute != 45, "prefix must end mid-hour"
    hour_open = last.floor("1h")
    prev_hour_close = candles.loc[: hour_open - pd.Timedelta("1ns"), "close"].iloc[-1]
    assert f.iloc[-1]["htf_close"] == prev_hour_close


def test_engine_enters_on_next_bar_open(candles, settings):
    result = run_strategy_backtest(candles, settings)
    assert result.trades, "need trades to test"
    bar = pd.Timedelta("15min")
    for t in result.trades:
        entry = pd.Timestamp(t.entry_time)
        assert entry == pd.Timestamp(t.signal_time) + bar
        expected = candles.loc[entry, "ask_open" if t.direction == "buy" else "open"]
        assert t.entry_price == expected
        assert t.exit_time >= t.entry_time


def test_engine_results_unchanged_by_future_data(candles, settings):
    """Trades closed before t are identical when the data after t is scrambled."""
    t = 4000
    cut = candles.index[t]
    scrambled = candles.copy()
    rng = np.random.default_rng(5)
    factor = rng.uniform(0.9, 1.1, size=len(candles) - t - 1)
    for c in ("open", "high", "low", "close", "ask_open", "ask_high", "ask_low", "ask_close"):
        scrambled.iloc[t + 1:, scrambled.columns.get_loc(c)] *= factor
    a = [x for x in run_strategy_backtest(candles, settings).trades if x.exit_time <= cut]
    b = [x for x in run_strategy_backtest(scrambled, settings).trades if x.exit_time <= cut]
    assert a and a == b


@pytest.mark.skipif(not REAL_DATA.exists(), reason="real data not downloaded")
def test_truncation_invariance_on_real_data(settings):
    from goldbot.data import load_candles

    df = load_candles(REAL_DATA).iloc[:20000]  # first ~10 months keeps the test fast
    full = compute_features(df, settings.strategy)
    sig_bars = _signal_bars(full, settings)
    rng = np.random.default_rng(3)
    points = sorted(set(sig_bars[:25]) | set(rng.choice(np.arange(2000, len(df)), 25, replace=False).tolist()))
    _assert_truncation_invariant(df, settings, points)
