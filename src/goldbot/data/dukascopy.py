"""Merge yearly Dukascopy bid/ask CSV chunks into one candle file.

Chunks are produced by ``scripts/fetch_data.sh`` (dukascopy-node) and named
``<prefix>_bid_<year>.csv`` / ``<prefix>_ask_<year>.csv`` with columns
``timestamp`` (Unix ms, UTC), ``open, high, low, close, volume``.

Usage:
    python -m goldbot.data.dukascopy data/raw xauusd_m15 data/xauusd_m15_2021_2026.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from goldbot.data.loader import DataError


def _load_side(raw_dir: Path, prefix: str, side: str) -> pd.DataFrame:
    files = sorted(raw_dir.glob(f"{prefix}_{side}_*.csv"))
    if not files:
        raise DataError(f"no {side} chunks matching {prefix}_{side}_*.csv in {raw_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop(columns="timestamp").drop_duplicates("time", keep="last")
    return df.set_index("time").sort_index()


def merge_bid_ask(raw_dir: str | Path, prefix: str) -> pd.DataFrame:
    """Return bid OHLCV + ask OHLC on identical UTC timestamps (inner join)."""
    raw_dir = Path(raw_dir)
    bid = _load_side(raw_dir, prefix, "bid")
    ask = _load_side(raw_dir, prefix, "ask")
    merged = bid.join(ask[["open", "high", "low", "close"]].add_prefix("ask_"), how="inner")
    if (merged["ask_close"] < merged["close"]).any():
        raise DataError("ask below bid in merged data")
    merged["spread"] = merged["ask_close"] - merged["close"]
    return merged


def write_merged(df: pd.DataFrame, out: str | Path) -> None:
    out_df = df.reset_index()
    out_df.insert(0, "timestamp", out_df.pop("time").dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
    out_df.to_csv(out, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_dir")
    ap.add_argument("prefix")
    ap.add_argument("out")
    args = ap.parse_args()
    df = merge_bid_ask(args.raw_dir, args.prefix)
    write_merged(df, args.out)
    print(f"wrote {len(df)} rows, {df.index[0]} -> {df.index[-1]}, to {args.out}")


if __name__ == "__main__":
    main()
