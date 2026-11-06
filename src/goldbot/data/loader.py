"""Historical candle loader (CSV / Parquet).

Output format: a DataFrame indexed by a tz-aware UTC ``DatetimeIndex`` named
``time``, with float columns ``open, high, low, close, volume`` (bid prices),
sorted ascending with duplicate timestamps removed. If the file also has
``ask_open, ask_high, ask_low, ask_close`` they are kept (and validated), so a
backtest can use the real spread.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

OHLC = ["open", "high", "low", "close"]
COLUMNS = OHLC + ["volume"]
ASK_OHLC = ["ask_open", "ask_high", "ask_low", "ask_close"]

_ALIASES: dict[str, str] = {
    "time": "time", "timestamp": "time", "datetime": "time", "date_time": "time",
    "date": "date",
    "open": "open", "o": "open",
    "high": "high", "h": "high",
    "low": "low", "l": "low",
    "close": "close", "c": "close",
    "volume": "volume", "vol": "volume", "v": "volume",
    "tick_volume": "volume", "tickvol": "volume",
    "real_volume": "real_volume",
    "spread": "spread",
}


class DataError(ValueError):
    """Raised when candle data is missing columns or is invalid."""


def _clean_name(name: object) -> str:
    # "<TICKVOL>" -> "tickvol", "Tick Volume" -> "tick_volume"
    s = str(name).strip().strip("<>").strip().lower()
    return re.sub(r"[\s\-]+", "_", s)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known column aliases to canonical names."""
    renamed = {}
    for col in df.columns:
        clean = _clean_name(col)
        renamed[col] = _ALIASES.get(clean, clean)
    # MT5 exports have both <TICKVOL> and <VOL>: tick volume wins, the other is real volume
    vol_sources = [c for c, n in renamed.items() if n == "volume"]
    if len(vol_sources) > 1:
        tick = [c for c in vol_sources if _clean_name(c) in ("tick_volume", "tickvol")]
        if len(tick) == 1:
            for c in vol_sources:
                if c != tick[0]:
                    renamed[c] = "real_volume"
    out = df.rename(columns=renamed)
    # Prefer tick_volume over real_volume (MT5 real volume is usually 0 for gold CFDs)
    if "volume" not in out.columns and "real_volume" in out.columns:
        out = out.rename(columns={"real_volume": "volume"})
    if out.columns.duplicated().any():
        dupes = sorted(set(out.columns[out.columns.duplicated()]))
        raise DataError(f"ambiguous columns after normalization: {dupes}")
    return out


def _read_csv(path: Path) -> pd.DataFrame:
    # sep=None sniffs comma / tab / semicolon (MT5 exports are tab-separated)
    return pd.read_csv(path, sep=None, engine="python")


def _build_time(df: pd.DataFrame) -> pd.Series:
    if "time" in df.columns and "date" in df.columns:
        raw = df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip()
    elif "time" in df.columns:
        raw = df["time"]
    elif "date" in df.columns:
        raw = df["date"]
    else:
        raise DataError("no time/date column found")

    if pd.api.types.is_numeric_dtype(raw):
        # Unix epoch: seconds or milliseconds
        unit = "ms" if raw.abs().max() > 10**11 else "s"
        return pd.to_datetime(raw, unit=unit, utc=True)
    if pd.api.types.is_datetime64_any_dtype(raw):
        return pd.Series(raw)
    # MT5 uses "2024.01.02 13:00"
    text = raw.astype(str).str.strip().str.replace(
        r"^(\d{4})\.(\d{2})\.(\d{2})", r"\1-\2-\3", regex=True
    )
    return pd.to_datetime(text, format="mixed")


def to_utc(times: pd.Series, source_tz: str) -> pd.Series:
    """Localize naive timestamps to ``source_tz`` and convert to UTC.

    Aware timestamps are converted to UTC directly; ``source_tz`` is ignored.
    """
    if times.dt.tz is None:
        times = times.dt.tz_localize(source_tz, ambiguous="raise", nonexistent="raise")
    return times.dt.tz_convert("UTC")


def load_candles(path: str | Path, source_tz: str = "UTC") -> pd.DataFrame:
    """Load OHLC candles from a CSV or Parquet file.

    ``source_tz`` is the timezone of naive timestamps in the file (MT5 exports
    use broker server time, e.g. ``"Etc/GMT-2"`` for UTC+2 or ``"EET"``).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix in (".csv", ".txt", ".tsv"):
        df = _read_csv(path)
    elif suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path)
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index(names="time")
    else:
        raise DataError(f"unsupported file type: {suffix}")

    df = normalize_columns(df)
    missing = [c for c in OHLC if c not in df.columns]
    if missing:
        raise DataError(f"missing required columns: {missing}")

    times = to_utc(_build_time(df), source_tz)

    out = pd.DataFrame(
        {c: pd.to_numeric(df[c], errors="raise").astype(float) for c in OHLC},
    )
    out["volume"] = (
        pd.to_numeric(df["volume"], errors="raise").astype(float) if "volume" in df.columns else 0.0
    )
    has_ask = all(c in df.columns for c in ASK_OHLC)
    if has_ask:
        for c in ASK_OHLC:
            out[c] = pd.to_numeric(df[c], errors="raise").astype(float)
    out.index = pd.DatetimeIndex(times, name="time")

    if out[OHLC].isna().any().any():
        raise DataError("NaN values in OHLC columns")
    bad = (out["high"] < out["low"]) | (out["high"] < out[["open", "close"]].max(axis=1)) | (
        out["low"] > out[["open", "close"]].min(axis=1)
    )
    if bad.any():
        raise DataError(f"{int(bad.sum())} candles with inconsistent OHLC, first at {out.index[bad][0]}")

    if has_ask:
        if out[ASK_OHLC].isna().any().any():
            raise DataError("NaN values in ask columns")
        if (out["ask_close"] < out["close"]).any() or (out["ask_open"] < out["open"]).any():
            raise DataError("ask below bid")

    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out[COLUMNS + (ASK_OHLC if has_ask else [])]
