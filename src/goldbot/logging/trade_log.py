"""Structured trade log: one JSON object per line in ``logs/trades-YYYY-MM-DD.jsonl``.

Every decision is logged: signals, rejections (with reason), orders sent,
fills, closed trades and errors. Timestamps are UTC ISO-8601.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from goldbot.models import ClosedTrade, Fill, Order, Signal


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, datetime):
        return obj.astimezone(timezone.utc).isoformat() if obj.tzinfo else obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return obj


class TradeLogger:
    def __init__(
        self,
        log_dir: str | Path,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        filename: str | None = None,
    ) -> None:
        """``filename`` writes everything to one fixed file (used by backtests);
        otherwise one file per UTC day."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._filename = filename

    def path_for(self, day: date) -> Path:
        if self._filename:
            return self.log_dir / self._filename
        return self.log_dir / f"trades-{day.isoformat()}.jsonl"

    def log(self, event: str, **data: Any) -> dict[str, Any]:
        now = self._clock().astimezone(timezone.utc)
        record = {"ts": now.isoformat(), "event": event, **_to_jsonable(data)}
        with self.path_for(now.date()).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        return record

    # Typed helpers -------------------------------------------------------
    def signal(self, signal: Signal) -> dict[str, Any]:
        return self.log("signal", signal=signal)

    def rejection(self, signal: Signal, reason: str) -> dict[str, Any]:
        return self.log("rejection", signal=signal, reason=reason)

    def order_sent(self, order: Order) -> dict[str, Any]:
        return self.log("order_sent", order=order)

    def fill(self, fill: Fill) -> dict[str, Any]:
        return self.log("fill", fill=fill)

    def trade_closed(self, trade: ClosedTrade) -> dict[str, Any]:
        return self.log("trade_closed", trade=trade)

    def error(self, message: str, **context: Any) -> dict[str, Any]:
        return self.log("error", message=message, **context)
