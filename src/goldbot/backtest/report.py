"""Write backtest outputs: trades CSV, metrics JSON, equity curve chart."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

from goldbot.backtest.engine import BacktestResult  # noqa: E402
from goldbot.backtest.metrics import trades_frame  # noqa: E402


def write_report(result: BacktestResult, metrics: dict, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cs = result.config.contract_size if result.config else 100.0

    trades_path = out / "trades.csv"
    trades_frame(result.trades, cs).to_csv(trades_path, index=False)

    metrics_path = out / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))

    chart_path = out / "equity.png"
    eq = result.equity
    dd_pct = (eq / eq.cummax() - 1) * 100
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax1.plot(eq.index, eq.to_numpy(), lw=1.0, color="#1f5fa8")
    ax1.axhline(eq.iloc[0], color="grey", lw=0.6, ls="--")
    ax1.set_ylabel("Equity (USD)")
    ax1.set_title(
        f"Backtest equity — {metrics.get('trades', 0)} trades, "
        f"PF {metrics.get('profit_factor')}, exp {metrics.get('expectancy_r')}R, "
        f"max DD {metrics.get('max_drawdown_pct')}%"
    )
    ax1.grid(alpha=0.3)
    ax2.fill_between(dd_pct.index, dd_pct.to_numpy(), 0, color="#c0392b", alpha=0.5, lw=0)
    ax2.set_ylabel("Drawdown %")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=120)
    plt.close(fig)

    return {"trades": trades_path, "metrics": metrics_path, "chart": chart_path}
