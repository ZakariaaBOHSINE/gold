"""Candle-by-candle backtest engine.

Timeline for each bar ``i`` (all prices from the data, bid + ask):
1. ENTRY at bar ``i``'s OPEN for a signal produced at the close of bar ``i-1``
   (only if bar ``i`` directly follows it, i.e. no market gap):
   the open's bid/ask is the "current tick". The signal is re-anchored to the
   expected fill (``reprice_fn``: ask for longs, bid for shorts) and then sent
   to the RISK MANAGER, which sizes it or rejects it. Approved orders fill at
   that price.
2. EXITS inside bar ``i``. Longs exit on BID prices, shorts on ASK prices, for
   both stop loss and take profit. If both levels are inside the same bar the
   stop is assumed first. If the bar opens beyond the stop (gap), the exit is
   at the open (worse than the stop).
3. SIGNAL at bar ``i``'s CLOSE from ``signal_fn(i)``, which may only use bars
   ``<= i``.
4. Equity is marked to market at the close (longs at bid, shorts at ask) and
   fed to the risk manager's daily-loss tracker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd

from goldbot.models import Direction, Signal, SymbolSpec, Tick
from goldbot.risk import AccountState, RiskManager


@dataclass(frozen=True)
class BacktestConfig:
    symbol: str
    initial_balance: float
    spec: SymbolSpec                   # contract size, tick value, volume limits, commission
    bar_period: pd.Timedelta = pd.Timedelta("15min")
    fixed_spread: float | None = None  # used when the data has no ask columns

    @property
    def contract_size(self) -> float:
        return self.spec.contract_size


@dataclass
class Trade:
    direction: str
    signal_time: datetime
    entry_time: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    volume: float
    risk_amount: float                 # money lost if the stop is hit exactly (incl. commission)
    entry_spread: float
    exit_time: datetime | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    gross_pnl: float = 0.0
    commission: float = 0.0
    pnl: float = 0.0
    r_multiple: float = 0.0
    reason: str = ""


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity: pd.Series                  # mark-to-market equity at each bar close
    skipped: dict[str, int] = field(default_factory=dict)
    config: BacktestConfig | None = None


Decision = Callable[[str, dict], None]
Reprice = Callable[[Signal, float], "Signal | None"]


def _ask_arrays(candles: pd.DataFrame, cfg: BacktestConfig) -> tuple[np.ndarray, ...]:
    cols = ("ask_open", "ask_high", "ask_low", "ask_close")
    if all(c in candles.columns for c in cols):
        return tuple(candles[c].to_numpy(float) for c in cols)
    if cfg.fixed_spread is None:
        raise ValueError("data has no ask columns and no fixed_spread configured")
    s = cfg.fixed_spread
    return tuple(candles[c].to_numpy(float) + s for c in ("open", "high", "low", "close"))


def run_backtest(
    candles: pd.DataFrame,
    signal_fn: Callable[[int], Signal | None],
    cfg: BacktestConfig,
    risk: RiskManager,
    reprice_fn: Reprice | None = None,
    on_decision: Decision | None = None,
) -> BacktestResult:
    log = on_decision or (lambda event, data: None)
    times = candles.index
    close_times = times + cfg.bar_period
    bo, bh, bl, bc = (candles[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    ao, ah, al, ac = _ask_arrays(candles, cfg)
    n = len(candles)
    unit_value = cfg.spec.value_per_price_unit

    balance = cfg.initial_balance
    equity = np.empty(n)
    trades: list[Trade] = []
    skipped: dict[str, int] = {}
    open_pos: list[Trade] = []
    pending: tuple[int, Signal] | None = None

    def skip(code: str, sig: Signal, **extra) -> None:
        skipped[code] = skipped.get(code, 0) + 1
        log("rejection", {"code": code, "signal": sig, **extra})

    def mtm(bid: float, ask: float) -> float:
        floating = 0.0
        for t in open_pos:
            if t.direction == Direction.BUY.value:
                floating += (bid - t.entry_price) * t.volume * unit_value
            else:
                floating += (t.entry_price - ask) * t.volume * unit_value
        return balance + floating

    def close(t: Trade, i: int, price: float, reason: str) -> None:
        nonlocal balance
        sign = 1 if t.direction == Direction.BUY.value else -1
        t.exit_time = times[i].to_pydatetime()
        t.exit_price = price
        t.exit_reason = reason
        t.gross_pnl = (price - t.entry_price) * sign * t.volume * unit_value
        t.commission = cfg.spec.commission_per_lot * t.volume
        t.pnl = t.gross_pnl - t.commission
        t.r_multiple = t.pnl / t.risk_amount if t.risk_amount > 0 else 0.0
        balance += t.pnl
        trades.append(t)
        log("trade_closed", {"trade": t})

    for i in range(n):
        # 1) entry at this bar's open
        if pending is not None:
            sig_i, sig = pending
            pending = None
            if times[i] - times[sig_i] != cfg.bar_period:
                skip("market_gap", sig, entry_bar=times[i])
            else:
                is_long = sig.direction is Direction.BUY
                fill = ao[i] if is_long else bo[i]
                priced = reprice_fn(sig, fill) if reprice_fn else sig
                if priced is None:
                    skip("gapped_past_stop", sig, fill=fill)
                else:
                    tick = Tick(cfg.symbol, bid=bo[i], ask=ao[i], time=times[i].to_pydatetime())
                    account = AccountState(
                        equity=mtm(bo[i], ao[i]), balance=balance, open_positions=len(open_pos)
                    )
                    decision = risk.evaluate(priced, account, tick, cfg.spec)
                    if not decision.approved:
                        skipped[decision.code] = skipped.get(decision.code, 0) + 1
                    else:
                        order = decision.order
                        stop_dist = abs(fill - order.stop_loss)
                        t = Trade(
                            direction=order.direction.value,
                            signal_time=sig.time,
                            entry_time=times[i].to_pydatetime(),
                            entry_price=fill,
                            stop_loss=order.stop_loss,
                            take_profit=order.take_profit,
                            volume=order.volume,
                            risk_amount=order.volume
                            * (stop_dist * unit_value + cfg.spec.commission_per_lot),
                            entry_spread=ao[i] - bo[i],
                            reason=priced.reason,
                        )
                        open_pos.append(t)
                        log("fill", {"trade": t})

        # 2) exits inside this bar: longs on bid, shorts on ask
        still_open: list[Trade] = []
        for t in open_pos:
            if t.direction == Direction.BUY.value:
                if bo[i] <= t.stop_loss:
                    close(t, i, bo[i], "stop_loss_gap")
                elif bl[i] <= t.stop_loss:
                    close(t, i, t.stop_loss, "stop_loss")
                elif bh[i] >= t.take_profit:
                    close(t, i, t.take_profit, "take_profit")
                else:
                    still_open.append(t)
            else:
                if ao[i] >= t.stop_loss:
                    close(t, i, ao[i], "stop_loss_gap")
                elif ah[i] >= t.stop_loss:
                    close(t, i, t.stop_loss, "stop_loss")
                elif al[i] <= t.take_profit:
                    close(t, i, t.take_profit, "take_profit")
                else:
                    still_open.append(t)
        open_pos = still_open

        # 3) signal at this bar's close (uses bars <= i only)
        sig = signal_fn(i)
        if sig is not None:
            log("signal", {"signal": sig})
            pending = (i, sig)

        # 4) mark to market at the close and feed the daily-loss tracker
        equity[i] = mtm(bc[i], ac[i])
        risk.observe_equity(close_times[i].to_pydatetime(), equity[i])

    last = n - 1
    for t in open_pos:  # close anything still open at the last bar
        close(t, last, bc[last] if t.direction == Direction.BUY.value else ac[last], "end_of_data")
    if n:
        equity[last] = balance

    return BacktestResult(
        trades=trades,
        equity=pd.Series(equity, index=times, name="equity"),
        skipped=skipped,
        config=cfg,
    )
