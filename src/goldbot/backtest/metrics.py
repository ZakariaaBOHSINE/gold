"""Backtest performance metrics."""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from goldbot.backtest.engine import BacktestResult, Trade


def trades_frame(trades: list[Trade], contract_size: float = 100.0) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=list(Trade.__dataclass_fields__))
    df = pd.DataFrame([asdict(t) for t in trades])
    for c in ("signal_time", "entry_time", "exit_time"):
        df[c] = pd.to_datetime(df[c], utc=True)
    df["duration_hours"] = (df["exit_time"] - df["entry_time"]).dt.total_seconds() / 3600
    # entry-bar spread x size: approximate (shorts actually pay the spread at exit)
    df["spread_cost_est"] = df["entry_spread"] * df["volume"] * contract_size
    return df


def _group_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"trades": 0}
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]
    gross_win, gross_loss = wins["pnl"].sum(), -losses["pnl"].sum()
    return {
        "trades": int(len(df)),
        "win_rate_pct": round(100 * len(wins) / len(df), 2),
        "net_pnl": round(float(df["pnl"].sum()), 2),
        "profit_factor": round(float(gross_win / gross_loss), 3) if gross_loss > 0 else None,
        "expectancy_r": round(float(df["r_multiple"].mean()), 3),
    }


def max_drawdown(equity: pd.Series) -> tuple[float, float, pd.Timestamp | None]:
    """Return (max drawdown in money, in % of peak, time of the trough)."""
    if equity.empty:
        return 0.0, 0.0, None
    peak = equity.cummax()
    dd = equity - peak
    dd_pct = dd / peak * 100
    trough = dd_pct.idxmin()
    return float(-dd.min()), float(-dd_pct.min()), trough


def compute_metrics(result: BacktestResult) -> dict:
    df = trades_frame(result.trades, result.config.contract_size if result.config else 100.0)
    eq = result.equity
    initial = result.config.initial_balance if result.config else float(eq.iloc[0])
    dd_money, dd_pct, dd_time = max_drawdown(eq)
    final = float(eq.iloc[-1]) if not eq.empty else initial
    years = (eq.index[-1] - eq.index[0]).days / 365.25 if len(eq) > 1 else 0
    cagr = ((final / initial) ** (1 / years) - 1) * 100 if years > 0 and final > 0 else None

    out: dict = {
        "period": f"{eq.index[0]} -> {eq.index[-1]}" if not eq.empty else None,
        "initial_balance": initial,
        "final_equity": round(final, 2),
        "return_pct": round((final / initial - 1) * 100, 2),
        "cagr_pct": round(cagr, 2) if cagr is not None else None,
        "max_drawdown": round(dd_money, 2),
        "max_drawdown_pct": round(dd_pct, 2),
        "max_drawdown_at": str(dd_time) if dd_time is not None else None,
        **_group_stats(df),
        "skipped_signals": dict(sorted(result.skipped.items())),
    }
    if df.empty:
        return out

    out.update(
        {
            "avg_win_r": round(float(df.loc[df["pnl"] > 0, "r_multiple"].mean()), 3)
            if (df["pnl"] > 0).any() else None,
            "avg_loss_r": round(float(df.loc[df["pnl"] <= 0, "r_multiple"].mean()), 3)
            if (df["pnl"] <= 0).any() else None,
            "avg_duration_hours": round(float(df["duration_hours"].mean()), 2),
            "median_duration_hours": round(float(df["duration_hours"].median()), 2),
            "max_consecutive_losses": _max_streak(df["pnl"] <= 0),
            "total_commission": round(float(df["commission"].sum()), 2),
            "est_total_spread_cost": round(float(df["spread_cost_est"].sum()), 2),
            "exit_reasons": df["exit_reason"].value_counts().to_dict(),
            "by_direction": {d: _group_stats(g) for d, g in df.groupby("direction")},
            "by_year": {int(y): _group_stats(g) for y, g in df.groupby(df["entry_time"].dt.year)},
        }
    )
    return out


def _max_streak(mask: pd.Series) -> int:
    best = cur = 0
    for v in mask.to_numpy():
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return int(best)


__all__ = ["compute_metrics", "max_drawdown", "trades_frame"]
