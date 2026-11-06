"""Daily loss limit, measured on equity (realized + open P&L).

The trading day starts at a configurable local time (default 17:00
America/New_York, gold's daily break). ``zoneinfo`` handles DST, so the reset
is 21:00 UTC in summer and 22:00 UTC in winter without hardcoding either.

At the first equity observation of a new trading day the tracker takes an
equity snapshot. Once equity falls ``limit_pct`` below that snapshot, new
trades are blocked until the next reset — even if equity later recovers.
Open positions are not touched.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo


class DailyLossTracker:
    def __init__(self, limit_pct: float, reset_time: time, reset_tz: ZoneInfo) -> None:
        if limit_pct <= 0:
            raise ValueError("limit_pct must be > 0")
        self.limit_pct = limit_pct
        self.reset_time = reset_time
        self.reset_tz = reset_tz
        self.day: date | None = None
        self.start_equity: float | None = None
        self.blocked = False

    def trading_day(self, ts: datetime) -> date:
        """Trading-day label: times at/after the reset belong to the next calendar day."""
        if ts.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        local = ts.astimezone(self.reset_tz)
        shift = timedelta(hours=self.reset_time.hour, minutes=self.reset_time.minute)
        # local 17:00 -> 00:00 of the next day
        return (local - shift + timedelta(days=1)).date()

    def observe(self, ts: datetime, equity: float) -> None:
        """Feed the current equity. Rolls the day and latches the block when the limit is hit."""
        day = self.trading_day(ts)
        if day != self.day:
            self.day = day
            self.start_equity = equity
            self.blocked = False
        if self.start_equity is None or self.start_equity <= 0:
            self.blocked = True
            return
        if self.loss_pct(equity) >= self.limit_pct:
            self.blocked = True

    def loss_pct(self, equity: float) -> float:
        if not self.start_equity:
            return 0.0
        return (self.start_equity - equity) / self.start_equity * 100.0

    def is_blocked(self, ts: datetime, equity: float) -> bool:
        self.observe(ts, equity)
        return self.blocked
