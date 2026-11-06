import json
from datetime import datetime, timezone

from goldbot.logging import TradeLogger
from goldbot.models import Direction, Signal

NOW = datetime(2024, 1, 2, 23, 59, tzinfo=timezone.utc)


def test_writes_json_lines_per_utc_day(tmp_path):
    log = TradeLogger(tmp_path, clock=lambda: NOW)
    sig = Signal("XAUUSD", Direction.BUY, 2000.0, 1990.0, 2020.0, "test", NOW)
    log.signal(sig)
    log.rejection(sig, "spread too wide")
    log.error("boom", code=42)

    path = tmp_path / "trades-2024-01-02.jsonl"
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["event"] for r in lines] == ["signal", "rejection", "error"]
    assert lines[0]["signal"]["direction"] == "buy"
    assert lines[0]["ts"] == "2024-01-02T23:59:00+00:00"
    assert lines[1]["reason"] == "spread too wide"
    assert lines[2]["code"] == 42
