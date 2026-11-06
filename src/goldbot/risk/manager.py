"""Risk manager: the deterministic gate between a Signal and an Order.

Plain code with no dependency on strategy, backtest or LLM modules. It can
veto anything. Checks run in a fixed order; the first failure is the reason.

R:R and stop distances are measured from the EXPECTED FILL price
(ask for longs, bid for shorts), not from the signal's reference price.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from goldbot.config import RiskSettings
from goldbot.logging import TradeLogger
from goldbot.models import Direction, Order, Signal, SymbolSpec, Tick
from goldbot.risk.daily import DailyLossTracker
from goldbot.risk.sizing import position_size, risk_per_lot

ORDER_COMMENT = "goldbot"
_EPS = 1e-9


@dataclass(frozen=True)
class AccountState:
    equity: float                 # balance + open P&L
    balance: float
    open_positions: int


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    code: str                     # "approved" or a stable rejection code
    reason: str                   # human-readable explanation
    order: Order | None = None
    details: dict[str, Any] = field(default_factory=dict)


class RiskManager:
    def __init__(
        self,
        limits: RiskSettings,
        kill_switch: Path | Callable[[], bool],
        logger: TradeLogger | None = None,
    ) -> None:
        self.limits = limits
        self._kill_switch = kill_switch if callable(kill_switch) else Path(kill_switch).exists
        self.logger = logger
        self.daily = DailyLossTracker(
            limits.daily_loss_limit_pct, limits.reset_time, limits.reset_zone
        )

    # ------------------------------------------------------------------
    def observe_equity(self, ts, equity: float) -> None:
        """Call regularly (each bar / tick) so the daily limit latches intraday."""
        self.daily.observe(ts, equity)

    def evaluate(
        self, signal: Signal, account: AccountState, tick: Tick, spec: SymbolSpec
    ) -> RiskDecision:
        decision = self._evaluate(signal, account, tick, spec)
        if self.logger is not None:
            if decision.approved:
                self.logger.log(
                    "risk_approved", signal=signal, order=decision.order, details=decision.details
                )
            else:
                self.logger.log(
                    "rejection", signal=signal, code=decision.code, reason=decision.reason,
                    details=decision.details,
                )
        return decision

    # ------------------------------------------------------------------
    def _evaluate(
        self, signal: Signal, account: AccountState, tick: Tick, spec: SymbolSpec
    ) -> RiskDecision:
        L = self.limits

        def reject(code: str, reason: str, **details: Any) -> RiskDecision:
            return RiskDecision(False, code, reason, None, details)

        # 1. kill switch
        if self._kill_switch():
            return reject("kill_switch", "kill switch active: STOP file present")

        # 2. daily loss limit (equity-based, latched until the next reset)
        if self.daily.is_blocked(tick.time, account.equity):
            return reject(
                "daily_loss_limit",
                f"daily loss limit hit: equity {account.equity:.2f} is "
                f"{self.daily.loss_pct(account.equity):.2f}% below the day-start snapshot "
                f"{self.daily.start_equity or 0:.2f} (limit {L.daily_loss_limit_pct}%)",
                trading_day=self.daily.day,
            )

        # 3. max open positions
        if account.open_positions >= L.max_open_positions:
            return reject(
                "max_open_positions",
                f"max open positions reached ({account.open_positions}/{L.max_open_positions})",
            )

        # basic sanity of inputs
        if signal.symbol != spec.symbol or signal.symbol != tick.symbol:
            return reject("symbol_mismatch", f"symbol mismatch: signal {signal.symbol}, "
                          f"tick {tick.symbol}, spec {spec.symbol}")
        if not (_finite_pos(tick.bid) and _finite_pos(tick.ask)) or tick.ask < tick.bid:
            return reject("invalid_price", f"invalid market price bid={tick.bid} ask={tick.ask}")
        if not math.isfinite(account.equity) or account.equity <= 0:
            return reject("no_equity", f"equity must be > 0 (is {account.equity})")

        is_long = signal.direction is Direction.BUY
        fill = tick.ask if is_long else tick.bid
        exit_side = tick.bid if is_long else tick.ask  # price the stop is triggered on
        sl = getattr(signal, "stop_loss", None)
        tp = getattr(signal, "take_profit", None)

        # 4. stop loss present and on the correct side of the market
        if sl is None or not _finite_pos(sl):
            return reject("missing_stop_loss", "signal has no valid stop loss")
        if (is_long and sl >= exit_side) or (not is_long and sl <= exit_side):
            return reject(
                "stop_wrong_side",
                f"{signal.direction.value} stop loss {sl} is on the wrong side of the market "
                f"(bid {tick.bid}, ask {tick.ask})",
            )
        stop_distance = abs(fill - sl)

        # 5. stop distance vs. broker minimum and vs. current spread
        broker_distance = abs(exit_side - sl)
        if broker_distance < spec.min_stop_distance - _EPS:
            return reject(
                "stop_too_close_broker",
                f"stop {broker_distance:.2f} from market is inside the broker minimum "
                f"{spec.min_stop_distance:.2f}",
            )
        spread = tick.ask - tick.bid
        min_by_spread = L.min_stop_spread_multiple * spread
        if stop_distance < min_by_spread - _EPS:
            return reject(
                "stop_too_close_spread",
                f"stop distance {stop_distance:.2f} < {L.min_stop_spread_multiple:g} x spread "
                f"{spread:.2f} = {min_by_spread:.2f}",
                stop_distance=stop_distance, spread=spread,
            )

        # 6. spread
        spread_bps = spread / tick.bid * 1e4
        if spread_bps > L.max_spread_bps + _EPS:
            return reject(
                "spread_above_max",
                f"spread {spread:.2f} = {spread_bps:.2f} bps > max {L.max_spread_bps:g} bps",
                spread_bps=round(spread_bps, 3),
            )

        # 7. risk/reward from the expected fill
        if tp is None or not _finite_pos(tp):
            return reject("missing_take_profit", "signal has no take profit; R:R cannot be verified")
        reward = (tp - fill) if is_long else (fill - tp)
        rr = reward / stop_distance
        if rr < L.min_risk_reward - _EPS:
            return reject(
                "risk_reward_below_min",
                f"R:R {rr:.2f} from expected fill {fill:.2f} < minimum {L.min_risk_reward:g}",
                rr=round(rr, 4), fill=fill,
            )

        # 8. position size
        volume = position_size(account.equity, L.risk_per_trade_pct, stop_distance, spec)
        if volume <= 0:
            return reject(
                "size_below_minimum",
                f"position size below broker minimum {spec.volume_min}: risking "
                f"{L.risk_per_trade_pct}% of {account.equity:.2f} with a {stop_distance:.2f} stop "
                f"needs {risk_per_lot(stop_distance, spec) * spec.volume_min:.2f} per min lot",
                stop_distance=stop_distance,
            )

        order = Order(
            symbol=signal.symbol,
            direction=signal.direction,
            volume=volume,
            stop_loss=sl,
            take_profit=tp,
            comment=ORDER_COMMENT,
        )
        risk_money = volume * risk_per_lot(stop_distance, spec)
        return RiskDecision(
            True,
            "approved",
            f"approved {volume} lots, risk {risk_money:.2f} "
            f"({risk_money / account.equity * 100:.3f}% of equity), R:R {rr:.2f}",
            order,
            {
                "expected_fill": fill,
                "stop_distance": stop_distance,
                "risk_money": round(risk_money, 2),
                "rr": round(rr, 4),
                "spread_bps": round(spread_bps, 3),
            },
        )


def _finite_pos(x: Any) -> bool:
    try:
        return math.isfinite(x) and x > 0
    except TypeError:
        return False
