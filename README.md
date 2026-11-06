# Gold Trading Bot (XAUUSD)

Rule-based gold trading bot for MetaTrader 5. **Demo accounts only.**
See [CLAUDE.md](CLAUDE.md) for the architecture, hard rules and build phases.

**Status:** Phases 1–3 done (foundation, strategy + backtest, risk manager).
Strategy v1 is **not profitable** after costs in backtests — see CLAUDE.md decisions.

## Setup (macOS / development)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env      # only needed later for MT5 (Windows)
pytest
```

On Windows with MT5 installed, also install the MT5 extra: `pip install -e ".[dev,mt5]"`.

## Layout

| Path | Purpose |
|---|---|
| `config/settings.yaml` | symbol, timeframes, risk limits, FakeBroker settings |
| `src/goldbot/config.py` | typed settings loader + MT5 credentials from `.env` |
| `src/goldbot/models.py` | `Signal`, `Order`, `Fill`, `Position`, `Tick`, ... |
| `src/goldbot/broker/` | `Broker` interface, `FakeBroker`, `MT5Broker` (stub) |
| `src/goldbot/logging/` | JSON-lines trade log → `logs/trades-YYYY-MM-DD.jsonl` |
| `src/goldbot/data/` | candle loader for CSV / Parquet in `data/` |

## Loading historical data

```python
from goldbot.data import load_candles

# Generic CSV with UTC timestamps
df = load_candles("data/xauusd_m15.csv")

# MT5 "Export bars" file; naive timestamps are in broker server time
df = load_candles("data/XAUUSD_M15.csv", source_tz="Etc/GMT-2")  # fixed UTC+2
df = load_candles("data/XAUUSD_M15.csv", source_tz="EET")        # UTC+2/+3 with DST
```

Result: UTC `DatetimeIndex` named `time`, columns `open, high, low, close, volume`,
sorted, no duplicate timestamps. Note: `Etc/GMT-2` means UTC**+**2 (POSIX sign convention).

## Historical data

```bash
scripts/fetch_data.sh          # Dukascopy XAUUSD M15 bid+ask, yearly chunks -> data/raw, merged -> data/
```

## Backtest

```bash
python -m goldbot.backtest                     # uses config/settings.yaml
python -m goldbot.backtest --out logs/my-run   # custom output dir
```

Writes `trades.csv`, `metrics.json`, `equity.png` and `decisions.jsonl` (every
signal, rejection, fill and closed trade) to `logs/backtest-<timestamp>/`.
Signals use closed candles only and are filled at the next bar's open (ask for
longs, bid for shorts) with the real Dukascopy spread and the configured commission.

## Kill switch

Create a file named `STOP` in the project root to halt all new orders
(enforced by the risk manager; backtests ignore it).
