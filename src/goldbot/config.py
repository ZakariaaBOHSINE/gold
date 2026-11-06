"""Typed configuration loading.

Non-secret settings come from ``config/settings.yaml``.
Secrets (MT5 credentials) come from environment variables / ``.env`` only.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, fields
from datetime import time as dtime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"

# MT5-style timeframe name -> pandas offset alias
TIMEFRAME_RULES = {
    "M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min",
    "H1": "1h", "H4": "4h", "D1": "1D",
}
VALID_TIMEFRAMES = frozenset(TIMEFRAME_RULES)


class ConfigError(ValueError):
    """Raised when settings are missing or invalid."""


@dataclass(frozen=True)
class TimeframeSettings:
    signal: str
    trend: str

    def __post_init__(self) -> None:
        for name in ("signal", "trend"):
            value = getattr(self, name)
            if value not in VALID_TIMEFRAMES:
                raise ConfigError(
                    f"timeframes.{name}: {value!r} is not one of {sorted(VALID_TIMEFRAMES)}"
                )


@dataclass(frozen=True)
class RiskSettings:
    risk_per_trade_pct: float
    daily_loss_limit_pct: float
    max_open_positions: int
    min_risk_reward: float
    max_spread_bps: float
    min_stop_spread_multiple: float
    daily_reset_time: str
    daily_reset_tz: str
    kill_switch_file: str

    def __post_init__(self) -> None:
        if not 0 < self.risk_per_trade_pct <= 5:
            raise ConfigError("risk.risk_per_trade_pct must be in (0, 5]")
        if not 0 < self.daily_loss_limit_pct <= 20:
            raise ConfigError("risk.daily_loss_limit_pct must be in (0, 20]")
        if self.daily_loss_limit_pct < self.risk_per_trade_pct:
            raise ConfigError("risk.daily_loss_limit_pct must be >= risk_per_trade_pct")
        if not isinstance(self.max_open_positions, int) or self.max_open_positions < 1:
            raise ConfigError("risk.max_open_positions must be an integer >= 1")
        if self.min_risk_reward <= 0:
            raise ConfigError("risk.min_risk_reward must be > 0")
        if self.max_spread_bps <= 0:
            raise ConfigError("risk.max_spread_bps must be > 0")
        if self.min_stop_spread_multiple < 0:
            raise ConfigError("risk.min_stop_spread_multiple must be >= 0")
        if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", self.daily_reset_time):
            raise ConfigError("risk.daily_reset_time must be HH:MM (24h)")
        try:
            ZoneInfo(self.daily_reset_tz)
        except (ZoneInfoNotFoundError, ValueError):
            raise ConfigError(f"risk.daily_reset_tz: unknown time zone {self.daily_reset_tz!r}") from None

    @property
    def reset_time(self) -> dtime:
        h, m = self.daily_reset_time.split(":")
        return dtime(int(h), int(m))

    @property
    def reset_zone(self) -> ZoneInfo:
        return ZoneInfo(self.daily_reset_tz)
        if not self.kill_switch_file:
            raise ConfigError("risk.kill_switch_file must not be empty")


@dataclass(frozen=True)
class StrategySettings:
    """Parameters of the trend-pullback strategy (v1)."""

    ema_trend_fast: int
    ema_trend_slow: int
    ema_zone_fast: int
    ema_zone_slow: int
    rsi_period: int
    rsi_overbought: float
    rsi_oversold: float
    atr_period: int
    pullback_lookback: int
    swing_lookback: int
    stop_buffer_atr: float
    take_profit_r: float

    def __post_init__(self) -> None:
        for name in (
            "ema_trend_fast", "ema_trend_slow", "ema_zone_fast", "ema_zone_slow",
            "rsi_period", "atr_period", "pullback_lookback", "swing_lookback",
        ):
            if getattr(self, name) < 1:
                raise ConfigError(f"strategy.{name} must be >= 1")
        if self.ema_trend_fast >= self.ema_trend_slow:
            raise ConfigError("strategy.ema_trend_fast must be < ema_trend_slow")
        if self.ema_zone_fast >= self.ema_zone_slow:
            raise ConfigError("strategy.ema_zone_fast must be < ema_zone_slow")
        if not 0 < self.rsi_oversold < self.rsi_overbought < 100:
            raise ConfigError("strategy: need 0 < rsi_oversold < rsi_overbought < 100")
        if self.stop_buffer_atr < 0:
            raise ConfigError("strategy.stop_buffer_atr must be >= 0")
        if self.take_profit_r <= 0:
            raise ConfigError("strategy.take_profit_r must be > 0")


@dataclass(frozen=True)
class BacktestSettings:
    data_file: str
    initial_balance: float
    commission_per_lot: float
    use_data_spread: bool

    def __post_init__(self) -> None:
        if self.initial_balance <= 0:
            raise ConfigError("backtest.initial_balance must be > 0")
        if self.commission_per_lot < 0:
            raise ConfigError("backtest.commission_per_lot must be >= 0")


@dataclass(frozen=True)
class FakeBrokerSettings:
    spread: float
    starting_balance: float
    contract_size: float
    tick_size: float
    volume_min: float
    volume_step: float
    volume_max: float
    min_stop_distance: float

    def __post_init__(self) -> None:
        if self.spread < 0:
            raise ConfigError("fake_broker.spread must be >= 0")
        if self.starting_balance <= 0:
            raise ConfigError("fake_broker.starting_balance must be > 0")
        for name in ("contract_size", "tick_size", "volume_min", "volume_step", "volume_max"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"fake_broker.{name} must be > 0")
        if self.volume_max < self.volume_min:
            raise ConfigError("fake_broker.volume_max must be >= volume_min")
        if self.min_stop_distance < 0:
            raise ConfigError("fake_broker.min_stop_distance must be >= 0")


@dataclass(frozen=True)
class PathSettings:
    data: Path
    logs: Path


@dataclass(frozen=True)
class Settings:
    symbol: str
    timeframes: TimeframeSettings
    risk: RiskSettings
    strategy: StrategySettings
    backtest: BacktestSettings
    fake_broker: FakeBrokerSettings
    paths: PathSettings

    def __post_init__(self) -> None:
        if self.strategy.take_profit_r < self.risk.min_risk_reward:
            raise ConfigError("strategy.take_profit_r must be >= risk.min_risk_reward")

    @property
    def kill_switch_path(self) -> Path:
        return PROJECT_ROOT / self.risk.kill_switch_file


@dataclass(frozen=True)
class MT5Credentials:
    login: int
    password: str
    server: str
    path: str | None = None

    def __repr__(self) -> str:  # never leak the password into logs
        return f"MT5Credentials(login={self.login}, server={self.server!r}, password='***')"


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"missing or invalid section: {key!r}")
    return value


def _build(cls: type, data: dict[str, Any], section: str) -> Any:
    """Construct a settings dataclass, rejecting missing/unknown keys and coercing numbers."""
    expected = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(expected)
    if unknown:
        raise ConfigError(f"{section}: unknown keys {sorted(unknown)}")
    missing = set(expected) - set(data)
    if missing:
        raise ConfigError(f"{section}: missing keys {sorted(missing)}")

    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        ftype = expected[name].type
        try:
            if ftype == "float":
                if isinstance(value, bool):
                    raise TypeError
                value = float(value)
            elif ftype == "int":
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError
            elif ftype == "str":
                if not isinstance(value, str):
                    raise TypeError
            elif ftype == "bool":
                if not isinstance(value, bool):
                    raise TypeError
        except (TypeError, ValueError):
            raise ConfigError(f"{section}.{name}: expected {ftype}, got {value!r}") from None
        kwargs[name] = value
    return cls(**kwargs)


def load_settings(path: str | Path = DEFAULT_SETTINGS_PATH) -> Settings:
    """Load and validate settings from a YAML file."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"settings file not found: {path}")
    with path.open() as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ConfigError("settings file must contain a mapping")

    allowed = {"symbol", "timeframes", "risk", "strategy", "backtest", "fake_broker", "paths"}
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigError(f"unknown top-level keys {sorted(unknown)}")

    symbol = raw.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        raise ConfigError("symbol must be a non-empty string")

    paths_raw = _section(raw, "paths")
    for key in ("data", "logs"):
        if key not in paths_raw:
            raise ConfigError(f"paths: missing key {key!r}")

    def _resolve(p: str) -> Path:
        p_ = Path(p)
        return p_ if p_.is_absolute() else PROJECT_ROOT / p_

    return Settings(
        symbol=symbol.strip(),
        timeframes=_build(TimeframeSettings, _section(raw, "timeframes"), "timeframes"),
        risk=_build(RiskSettings, _section(raw, "risk"), "risk"),
        strategy=_build(StrategySettings, _section(raw, "strategy"), "strategy"),
        backtest=_build(BacktestSettings, _section(raw, "backtest"), "backtest"),
        fake_broker=_build(FakeBrokerSettings, _section(raw, "fake_broker"), "fake_broker"),
        paths=PathSettings(data=_resolve(paths_raw["data"]), logs=_resolve(paths_raw["logs"])),
    )


def load_mt5_credentials(env_file: str | Path | None = None) -> MT5Credentials:
    """Read MT5 credentials from the environment (optionally loading a .env file first)."""
    load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)
    missing = [k for k in ("MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER") if not os.environ.get(k)]
    if missing:
        raise ConfigError(f"missing environment variables: {missing}")
    try:
        login = int(os.environ["MT5_LOGIN"])
    except ValueError:
        raise ConfigError("MT5_LOGIN must be an integer") from None
    return MT5Credentials(
        login=login,
        password=os.environ["MT5_PASSWORD"],
        server=os.environ["MT5_SERVER"],
        path=os.environ.get("MT5_PATH") or None,
    )
