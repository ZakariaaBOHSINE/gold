# CLAUDE.md — Gold Trading Bot

## Project overview

A local, rule-based trading bot for gold (XAUUSD) that reads live market data from MetaTrader 5, generates trade signals from coded strategy rules, passes every signal through an independent risk manager, and executes approved trades on MT5. An LLM layer may be added later as an analyst/filter, never as the trigger.

Current stage: **Phases 1–3 complete; next is Phase 4 (see Build phases below).** Do not jump ahead to later phases unless explicitly asked.

## Hard rules (never break these)

1. **Demo account only.** Never write code that connects to a live/real account. The broker module must refuse to start if the connected account is not a demo account (check `account_info().trade_mode`).
2. **Every order goes through the risk manager.** No module other than `execution/` may call the broker's order functions, and `execution/` only acts on orders approved by `risk/`.
3. **The risk manager has no dependency on the LLM or the strategy.** It is plain, deterministic code that can veto anything.
4. **Every order has a stop loss.** Orders without a stop loss are rejected.
5. **No secrets in code or git.** MT5 login, password, server and any API keys live in `.env` only. `.env` is in `.gitignore`. Use `.env.example` with placeholder values.
6. **Never place, modify or close orders during development or tests** unless the user explicitly asks in that session. Tests use the fake broker.
7. **Log every decision**: signals generated, signals rejected (with the reason), orders sent, fills, errors.
8. Ask before adding a new dependency, changing a risk limit, or changing strategy logic.

## Architecture

```
data source (MT5 / CSV history) → strategy → risk manager → execution → MT5
                                      ↓            ↓            ↓
                                           trade log (every decision)
```

- **Data**: live candles and ticks from MT5 via the `MetaTrader5` Python package. Historical data from CSV/Parquet for backtesting. TradingView webhooks are optional and come later.
- **Strategy**: pure functions. Input: a DataFrame of candles. Output: a `Signal` (direction, entry, stop loss, take profit, reason) or `None`. No I/O, no broker calls.
- **Risk manager**: takes a `Signal` plus account state, returns an approved `Order` or a rejection with a reason.
- **Execution**: sends approved orders to the broker, handles retries and errors, records fills.
- **Broker interface**: an abstract `Broker` class with two implementations:
  - `FakeBroker` — simulated fills, works on macOS, used for development and tests
  - `MT5Broker` — real MT5 connection, Windows only

## Directory structure

```
gold/                        # project root
├── CLAUDE.md
├── README.md
├── .env.example
├── .gitignore
├── pyproject.toml
├── config/
│   └── settings.yaml        # symbol, timeframes, risk limits, strategy params
├── src/goldbot/
│   ├── broker/
│   │   ├── base.py          # abstract Broker interface
│   │   ├── fake.py          # FakeBroker (macOS-safe)
│   │   └── mt5.py           # MT5Broker (Windows only, import lazily)
│   ├── data/                # candle loading, live feed, resampling
│   ├── strategy/            # signal generation (pure functions)
│   ├── risk/                # position sizing, limits, kill switch
│   ├── execution/           # order placement and tracking
│   ├── backtest/            # backtest engine and metrics
│   ├── logging/             # structured trade log
│   └── main.py              # entry point / run loop
├── data/                    # historical data (gitignored)
├── logs/                    # trade logs (gitignored)
└── tests/
```

## Risk limits (defaults, configurable in settings.yaml)

- Risk per trade: 0.5% of account equity (sizing includes commission; volume rounded **down** to the broker's volume step)
- Daily loss limit: 2% of equity → bot stops trading for the rest of the day (details in Decisions below)
- Max open positions: 1
- Minimum risk/reward: 1:2, measured from the **expected fill price**
- Max spread allowed to enter: `max_spread_bps: 3.0` — relative to price, in basis points (gold spreads scale with price and widen around news)
- Minimum stop distance: `min_stop_spread_multiple: 3` — stop must be at least 3 × the current spread from the expected fill, and never inside the broker's minimum stop distance
- Kill switch: a file named `STOP` in the project root halts all new orders immediately
- No new trades in a configurable window around high-impact news (added in Phase 5)

## Decisions (agreed with the user, 2026-11-02)

- **Spread limit is relative** (`max_spread_bps`), not a fixed USD amount. A fixed 0.50 USD would have blocked 82–92% of bars in 2025–26 because gold's spread grows with its price (~1.5–2 bps median throughout).
- **R:R is measured from the expected fill**: ask for longs, bid for shorts — never from the signal candle's bid close.
- **Take profit is anchored to the fill.** The strategy sets TP at `take_profit_r` × risk from the expected fill (ask close for longs at signal time); when the real fill price is known, `reprice_to_fill` re-anchors entry and TP to it (stop unchanged), so valid setups still reach exactly 2R.
- **Exits use the correct side of the book:** longs exit on the bid (SL and TP), shorts on the ask.
- **Minimum stop distance:** `min_stop_spread_multiple: 3` (0 disables). Backtest comparison of 0/2/3/4 showed a small improvement with larger values (drawdown 40% → 35%), not enough to make strategy v1 profitable.
- **Daily loss limit:** the trading day resets at **17:00 America/New_York** (gold's daily break), computed with `zoneinfo` so DST is automatic — never a hardcoded UTC hour. At the first observation after the reset an equity snapshot is taken; the limit uses **equity (realized + open P&L)** against that snapshot. Once hit, it blocks new trades until the next reset even if equity recovers. It **does not close open positions**.
- **Backtests ignore the `STOP` file** (injected kill-switch check) so a stray file cannot change results; live code must always use the real file check.
- **Historical data:** Dukascopy via `dukascopy-node` (`scripts/fetch_data.sh`), M15 bid + ask in yearly chunks, merged to `data/xauusd_m15_2021_2026.csv` (UTC). Backtests use the real bid/ask spread plus `backtest.commission_per_lot` (7 USD round turn).

## Environment

- **Development**: macOS. MT5 is not available here. Always use `FakeBroker` and historical data.
- **Live demo trading**: Windows PC with the MT5 terminal installed, open and logged into a **demo** account.
- `src/goldbot/broker/mt5.py` must import `MetaTrader5` lazily so the rest of the codebase imports cleanly on macOS.
- Broker symbol names vary (`XAUUSD`, `XAUUSDm`, `GOLD`). The symbol comes from config, never hardcoded.
- Times: store everything in UTC. Note that the MT5 server time zone depends on the broker.

## Build phases

1. **Foundation** — project skeleton, config loading, `Broker` interface, `FakeBroker`, structured logging, historical data loader.
2. **Strategy + backtest** — first strategy as pure functions, backtest engine including spread and commission, metrics (win rate, profit factor, max drawdown, expectancy).
3. **Risk manager** — position sizing and all limits above, fully unit-tested.
4. **MT5 integration (Windows)** — `MT5Broker`, demo-account check, live candle feed, order execution on demo.
5. **Live demo run** — run loop, news filter, monitoring, at least several weeks on demo before anything else.
6. **LLM analyst layer (optional)** — the LLM can only veto or annotate signals, never create orders on its own.

## Conventions

- Python 3.11+, type hints everywhere, dataclasses or pydantic for `Signal`, `Order`, `Fill`.
- pandas for candle data.
- Tests with pytest. Strategy and risk modules must have unit tests before being used by execution.
- Small, focused modules. Prefer clarity over cleverness.
- Run tests before declaring a task done.

## Commands

```bash
pip install -e ".[dev]"        # install
pytest                         # run tests
python -m goldbot.backtest     # run a backtest
python -m goldbot.main         # run the bot (Windows + demo account only)
```
