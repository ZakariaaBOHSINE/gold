"""Glue: settings + data + strategy -> engine -> metrics + report files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from goldbot.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from goldbot.backtest.metrics import compute_metrics
from goldbot.backtest.report import write_report
from goldbot.broker import FakeBroker
from goldbot.config import TIMEFRAME_RULES, Settings
from goldbot.data import load_candles
from goldbot.logging import TradeLogger
from goldbot.risk import RiskManager
from goldbot.strategy.trend_pullback import compute_features, reprice_to_fill, signal_at


def fake_broker_from_settings(settings: Settings) -> FakeBroker:
    fb = settings.fake_broker
    return FakeBroker(
        settings.symbol,
        spread=fb.spread,
        starting_balance=settings.backtest.initial_balance,
        contract_size=fb.contract_size,
        tick_size=fb.tick_size,
        volume_min=fb.volume_min,
        volume_step=fb.volume_step,
        volume_max=fb.volume_max,
        min_stop_distance=fb.min_stop_distance,
        commission_per_lot=settings.backtest.commission_per_lot,
    )


def config_from_settings(settings: Settings) -> BacktestConfig:
    # symbol properties come from the (fake) broker, exactly as live code will get them
    spec = fake_broker_from_settings(settings).symbol_info(settings.symbol)
    return BacktestConfig(
        symbol=settings.symbol,
        initial_balance=settings.backtest.initial_balance,
        spec=spec,
        bar_period=pd.Timedelta(TIMEFRAME_RULES[settings.timeframes.signal]),
        fixed_spread=None if settings.backtest.use_data_spread else settings.fake_broker.spread,
    )


def run_strategy_backtest(
    candles: pd.DataFrame, settings: Settings, logger: TradeLogger | None = None
) -> BacktestResult:
    if not settings.backtest.use_data_spread:
        candles = candles.drop(columns=[c for c in candles.columns if c.startswith("ask_")])
    feats = compute_features(
        candles, settings.strategy, settings.timeframes.signal, settings.timeframes.trend
    )

    def signal_fn(i: int):
        return signal_at(feats, i, settings.symbol, settings.strategy)

    def reprice(sig, fill):
        return reprice_to_fill(sig, fill, settings.strategy.take_profit_r)

    # Backtests ignore the STOP file so a stray file cannot change results.
    risk = RiskManager(settings.risk, kill_switch=lambda: False, logger=logger)
    on_decision = (lambda event, data: logger.log(event, **data)) if logger else None
    return run_backtest(
        candles, signal_fn, config_from_settings(settings), risk, reprice, on_decision
    )


def run_from_settings(
    settings: Settings, out_dir: Path, data_path: str | Path | None = None
) -> tuple[dict, dict[str, Path]]:
    path = Path(data_path) if data_path else settings.paths.data / settings.backtest.data_file
    candles = load_candles(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = TradeLogger(out_dir, filename="decisions.jsonl")
    result = run_strategy_backtest(candles, settings, logger)
    metrics = compute_metrics(result)
    paths = write_report(result, metrics, out_dir)
    paths["decisions"] = out_dir / "decisions.jsonl"
    return metrics, paths
