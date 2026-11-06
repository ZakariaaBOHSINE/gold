from pathlib import Path

import pandas as pd
import pytest

from goldbot.data import DataError, load_candles

FIXTURES = Path(__file__).parent / "fixtures"


def _check_canonical(df: pd.DataFrame) -> None:
    assert list(df.columns)[:5] == ["open", "high", "low", "close", "volume"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"
    assert df.index.name == "time"
    assert df.index.is_monotonic_increasing
    assert not df.index.has_duplicates


def test_generic_csv(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text(
        "Timestamp,Open,High,Low,Close,Volume\n"
        "2024-01-02 13:00:00,2060.1,2062.0,2059.5,2061.2,100\n"
        "2024-01-02 13:15:00,2061.2,2063.0,2060.0,2062.5,120\n"
    )
    df = load_candles(p)
    _check_canonical(df)
    assert df.index[0] == pd.Timestamp("2024-01-02 13:00", tz="UTC")
    assert df.iloc[1]["close"] == 2062.5


def test_mt5_export_format_with_server_timezone():
    # MT5 export: tab-separated, <DATE> <TIME> columns, server time UTC+2
    df = load_candles(FIXTURES / "mt5_export.csv", source_tz="Etc/GMT-2")
    _check_canonical(df)
    assert len(df) == 3
    assert df.index[0] == pd.Timestamp("2024-01-02 11:00", tz="UTC")
    assert df.iloc[0]["volume"] == 1500  # TICKVOL, not VOL


def test_short_aliases_and_iso_offsets(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text(
        "time,o,h,l,c,tick_volume\n"
        "2024-01-02T15:00:00+02:00,1,2,0.5,1.5,7\n"
    )
    df = load_candles(p, source_tz="America/New_York")  # ignored: times already aware
    assert df.index[0] == pd.Timestamp("2024-01-02 13:00", tz="UTC")
    assert df.iloc[0]["volume"] == 7


def test_unix_epoch_seconds_and_ms(tmp_path):
    ts = int(pd.Timestamp("2024-01-02 13:00", tz="UTC").timestamp())
    for value in (ts, ts * 1000):
        p = tmp_path / f"e{value}.csv"
        p.write_text(f"time,open,high,low,close\n{value},1,2,0.5,1.5\n")
        df = load_candles(p)
        assert df.index[0] == pd.Timestamp("2024-01-02 13:00", tz="UTC")
        assert df.iloc[0]["volume"] == 0.0


def test_sorts_and_drops_duplicates(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text(
        "time,open,high,low,close,volume\n"
        "2024-01-02 13:30,3,4,2,3,1\n"
        "2024-01-02 13:00,1,2,0.5,1.5,1\n"
        "2024-01-02 13:30,3,5,2,4,2\n"
    )
    df = load_candles(p)
    _check_canonical(df)
    assert len(df) == 2
    assert df.iloc[-1]["close"] == 4  # last duplicate kept


def test_dst_aware_source_tz(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text(
        "time,open,high,low,close\n"
        "2024-01-15 10:00,1,2,0.5,1.5\n"  # winter: EET = UTC+2
        "2024-07-15 10:00,1,2,0.5,1.5\n"  # summer: EEST = UTC+3
    )
    df = load_candles(p, source_tz="EET")
    assert df.index[0].hour == 8
    assert df.index[1].hour == 7


def test_parquet_with_datetime_index(tmp_path):
    idx = pd.date_range("2024-01-02 13:00", periods=3, freq="15min", tz="UTC", name="Date")
    src = pd.DataFrame(
        {"Open": [1.0, 2, 3], "High": [2.0, 3, 4], "Low": [0.5, 1, 2], "Close": [1.5, 2.5, 3.5]},
        index=idx,
    )
    p = tmp_path / "c.parquet"
    src.to_parquet(p)
    df = load_candles(p)
    _check_canonical(df)
    assert len(df) == 3
    assert df.index[0] == pd.Timestamp("2024-01-02 13:00", tz="UTC")


def test_parquet_naive_column(tmp_path):
    src = pd.DataFrame(
        {
            "time": pd.to_datetime(["2024-01-02 15:00"]),
            "open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "tick_volume": [9],
        }
    )
    p = tmp_path / "c.parquet"
    src.to_parquet(p, index=False)
    df = load_candles(p, source_tz="Etc/GMT-2")
    assert df.index[0] == pd.Timestamp("2024-01-02 13:00", tz="UTC")


def test_missing_columns(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("time,open,high,close\n2024-01-02 13:00,1,2,1.5\n")
    with pytest.raises(DataError, match="low"):
        load_candles(p)


def test_missing_time(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("open,high,low,close\n1,2,0.5,1.5\n")
    with pytest.raises(DataError, match="time"):
        load_candles(p)


def test_inconsistent_ohlc(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("time,open,high,low,close\n2024-01-02 13:00,1,0.5,2,1.5\n")
    with pytest.raises(DataError, match="inconsistent"):
        load_candles(p)


def test_unsupported_and_missing_file(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("{}")
    with pytest.raises(DataError, match="unsupported"):
        load_candles(p)
    with pytest.raises(FileNotFoundError):
        load_candles(tmp_path / "none.csv")


def test_keeps_ask_columns(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text(
        "timestamp,open,high,low,close,volume,ask_open,ask_high,ask_low,ask_close,spread\n"
        "2024-01-02T13:00:00Z,1,2,0.5,1.5,3,1.1,2.1,0.6,1.6,0.1\n"
    )
    df = load_candles(p)
    assert list(df.columns) == [
        "open", "high", "low", "close", "volume", "ask_open", "ask_high", "ask_low", "ask_close"
    ]
    assert df.iloc[0]["ask_close"] == 1.6


def test_rejects_ask_below_bid(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text(
        "timestamp,open,high,low,close,ask_open,ask_high,ask_low,ask_close\n"
        "2024-01-02T13:00:00Z,1,2,0.5,1.5,1.1,2.1,0.6,1.4\n"
    )
    with pytest.raises(DataError, match="ask below bid"):
        load_candles(p)
