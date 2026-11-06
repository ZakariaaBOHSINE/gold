"""Run a backtest: ``python -m goldbot.backtest [--data FILE] [--config FILE] [--out DIR]``."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from goldbot.backtest.runner import run_from_settings
from goldbot.config import DEFAULT_SETTINGS_PATH, load_settings


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the strategy backtest on historical data.")
    ap.add_argument("--config", default=str(DEFAULT_SETTINGS_PATH))
    ap.add_argument("--data", help="candle file (default: backtest.data_file in settings)")
    ap.add_argument("--out", help="output dir (default: logs/backtest-<UTC timestamp>)")
    args = ap.parse_args()

    settings = load_settings(args.config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out) if args.out else settings.paths.logs / f"backtest-{stamp}"
    metrics, paths = run_from_settings(settings, out_dir, data_path=args.data)

    summary = {k: v for k, v in metrics.items() if k not in ("by_year", "by_direction")}
    print(json.dumps(summary, indent=2, default=str))
    print("\nBy year:")
    for y, s in metrics.get("by_year", {}).items():
        print(f"  {y}: {s}")
    print("By direction:")
    for d, s in metrics.get("by_direction", {}).items():
        print(f"  {d}: {s}")
    print("\nOutputs:")
    for name, p in paths.items():
        print(f"  {name}: {p}")


if __name__ == "__main__":
    main()
