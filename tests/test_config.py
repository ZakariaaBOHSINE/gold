from pathlib import Path

import pytest
import yaml

from goldbot.config import (
    DEFAULT_SETTINGS_PATH,
    PROJECT_ROOT,
    ConfigError,
    load_mt5_credentials,
    load_settings,
)


def _raw() -> dict:
    return yaml.safe_load(DEFAULT_SETTINGS_PATH.read_text())


def _write(tmp_path: Path, raw: dict) -> Path:
    p = tmp_path / "settings.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


def test_loads_project_settings_with_claude_md_defaults():
    s = load_settings()
    assert s.symbol == "XAUUSD"
    assert s.timeframes.signal == "M15"
    assert s.timeframes.trend == "H1"
    assert s.risk.risk_per_trade_pct == 0.5
    assert s.risk.daily_loss_limit_pct == 2.0
    assert s.risk.max_open_positions == 1
    assert s.risk.min_risk_reward == 2.0
    assert s.risk.max_spread_bps == 3.0
    assert s.risk.min_stop_spread_multiple == 3.0
    assert s.risk.reset_time.hour == 17 and s.risk.reset_time.minute == 0
    assert str(s.risk.reset_zone) == "America/New_York"
    assert s.strategy.ema_trend_slow == 200
    assert s.strategy.take_profit_r >= s.risk.min_risk_reward
    assert s.backtest.commission_per_lot == 7.0
    assert s.backtest.use_data_spread is True
    assert s.kill_switch_path == PROJECT_ROOT / "STOP"
    assert s.fake_broker.spread == 0.30
    assert s.paths.logs == PROJECT_ROOT / "logs"


def test_settings_are_immutable():
    s = load_settings()
    with pytest.raises(AttributeError):
        s.risk.risk_per_trade_pct = 10  # type: ignore[misc]


def test_ints_are_coerced_to_float(tmp_path):
    raw = _raw()
    raw["risk"]["min_risk_reward"] = 1
    s = load_settings(_write(tmp_path, raw))
    assert isinstance(s.risk.min_risk_reward, float)


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_settings(tmp_path / "nope.yaml")


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda r: r.pop("symbol"), "symbol"),
        (lambda r: r.pop("risk"), "risk"),
        (lambda r: r["risk"].pop("max_spread_bps"), "missing keys"),
        (lambda r: r["risk"].update(foo=1), "unknown keys"),
        (lambda r: r.update(extra=1), "unknown top-level"),
        (lambda r: r["risk"].update(risk_per_trade_pct=0), "risk_per_trade_pct"),
        (lambda r: r["risk"].update(risk_per_trade_pct=50), "risk_per_trade_pct"),
        (lambda r: r["risk"].update(daily_loss_limit_pct=0.1), "daily_loss_limit_pct"),
        (lambda r: r["risk"].update(max_open_positions=0), "max_open_positions"),
        (lambda r: r["risk"].update(max_open_positions=1.5), "max_open_positions"),
        (lambda r: r["risk"].update(min_risk_reward="two"), "min_risk_reward"),
        (lambda r: r["risk"].update(max_spread_bps=True), "max_spread_bps"),
        (lambda r: r["risk"].update(max_spread_bps=0), "max_spread_bps"),
        (lambda r: r["risk"].update(min_stop_spread_multiple=-1), "min_stop_spread_multiple"),
        (lambda r: r["risk"].update(daily_reset_time="5pm"), "daily_reset_time"),
        (lambda r: r["risk"].update(daily_reset_time="24:00"), "daily_reset_time"),
        (lambda r: r["risk"].update(daily_reset_tz="Mars/Olympus"), "daily_reset_tz"),
        (lambda r: r["fake_broker"].update(volume_step=0), "volume_step"),
        (lambda r: r["fake_broker"].update(volume_max=0.001), "volume_max"),
        (lambda r: r.pop("strategy"), "strategy"),
        (lambda r: r["strategy"].update(ema_trend_fast=300), "ema_trend_fast"),
        (lambda r: r["strategy"].update(ema_zone_fast=60), "ema_zone_fast"),
        (lambda r: r["strategy"].update(rsi_oversold=80), "rsi_oversold"),
        (lambda r: r["strategy"].update(take_profit_r=1.5), "take_profit_r"),
        (lambda r: r["strategy"].update(swing_lookback=0), "swing_lookback"),
        (lambda r: r["backtest"].update(use_data_spread="yes"), "use_data_spread"),
        (lambda r: r["backtest"].update(commission_per_lot=-1), "commission"),
        (lambda r: r["timeframes"].update(signal="M7"), "timeframes.signal"),
        (lambda r: r["fake_broker"].update(spread=-1), "spread"),
        (lambda r: r["paths"].pop("logs"), "logs"),
    ],
)
def test_invalid_settings_rejected(tmp_path, mutate, match):
    raw = _raw()
    mutate(raw)
    with pytest.raises(ConfigError, match=match):
        load_settings(_write(tmp_path, raw))


def test_credentials_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MT5_LOGIN", "123456")
    monkeypatch.setenv("MT5_PASSWORD", "secret")
    monkeypatch.setenv("MT5_SERVER", "Broker-Demo")
    monkeypatch.delenv("MT5_PATH", raising=False)
    creds = load_mt5_credentials(env_file=tmp_path / "missing.env")
    assert creds.login == 123456
    assert creds.server == "Broker-Demo"
    assert creds.path is None
    assert "secret" not in repr(creds)


def test_credentials_from_env_file(monkeypatch, tmp_path):
    for k in ("MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER", "MT5_PATH"):
        monkeypatch.delenv(k, raising=False)
    env = tmp_path / ".env"
    env.write_text("MT5_LOGIN=42\nMT5_PASSWORD=pw\nMT5_SERVER=X-Demo\n")
    creds = load_mt5_credentials(env_file=env)
    assert creds.login == 42
    for k in ("MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER"):
        monkeypatch.delenv(k, raising=False)  # don't leak into other tests


def test_credentials_missing(monkeypatch, tmp_path):
    for k in ("MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(ConfigError, match="missing environment"):
        load_mt5_credentials(env_file=tmp_path / "missing.env")


def test_credentials_bad_login(monkeypatch, tmp_path):
    monkeypatch.setenv("MT5_LOGIN", "abc")
    monkeypatch.setenv("MT5_PASSWORD", "x")
    monkeypatch.setenv("MT5_SERVER", "y")
    with pytest.raises(ConfigError, match="integer"):
        load_mt5_credentials(env_file=tmp_path / "missing.env")
