import numpy as np
import pandas as pd
import pytest

from goldbot.config import load_settings


def make_candles(n: int = 6000, seed: int = 7, spread: float = 0.30) -> pd.DataFrame:
    """Synthetic M15 gold-like candles with alternating trend regimes (bid + ask)."""
    rng = np.random.default_rng(seed)
    regime_len = 700
    drift = np.repeat(rng.choice([-1, 1], size=n // regime_len + 1), regime_len)[:n] * 0.00025
    rets = drift + rng.normal(0, 0.0016, n)
    close = 2000 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[2000.0], close[:-1]])
    wick = np.abs(rng.normal(0, 0.0008, (2, n))) * close
    high = np.maximum(open_, close) + wick[0]
    low = np.minimum(open_, close) - wick[1]
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC", name="time")
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1.0}, index=idx
    )
    for c in ("open", "high", "low", "close"):
        df[f"ask_{c}"] = df[c] + spread
    return df


@pytest.fixture(scope="session")
def settings():
    return load_settings()


@pytest.fixture(scope="session")
def candles():
    return make_candles()
