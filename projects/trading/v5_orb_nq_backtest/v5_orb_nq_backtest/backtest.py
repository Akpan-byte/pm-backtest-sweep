#!/usr/bin/env python3
"""
Bar-by-bar v5 ORB backtest on NQ 1-minute CSV data.

This is a clean re-implementation of the v5 live-bot backtest path that:
  * loads NQ 1-minute bars from CSV,
  * drives StrategyProcessor bar-by-bar,
  * records daily PnL and closed trades,
  * computes summary metrics.

It intentionally avoids broker SDK imports so it can run in CI and locally.
"""

from __future__ import annotations

import gzip
import json
import os
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DEFAULT_STRATEGY_CONFIG, TIMEFRAMES, TICK_VALUE
from .models import StrategyConfig, TFParams
from .strategy_engine import StrategyProcessor


def _make_config(params: dict[str, Any] | None = None) -> StrategyConfig:
    merged = dict(DEFAULT_STRATEGY_CONFIG)
    if params:
        merged.update(params)
    return StrategyConfig(**merged)


def load_csv(filepath: str | Path) -> pd.DataFrame:
    """Load NQ 1-minute CSV and prepare ET session columns."""
    filepath = Path(filepath)
    if filepath.suffix == ".gz":
        df = pd.read_csv(filepath, compression="gzip")
    else:
        df = pd.read_csv(filepath)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df.tz_convert("America/New_York")
    df["date"] = df.index.date
    df["time"] = df.index.time
    df["time_min"] = df.index.hour * 60 + df.index.minute
    # Cash session only (09:30–16:00 ET); v5 engine filters 570–958 itself.
    df = df[(df["time_min"] >= 570) & (df["time_min"] <= 960)]
    return df


def compute_metrics(
    trades: list[dict[str, Any]], day_results: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compute backtest summary metrics."""
    metrics: dict[str, Any] = {"total_trades": len(trades)}
    if not trades:
        return metrics

    pnls = np.array([t["net"] for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    n = len(trades)

    metrics["win_rate"] = round(len(wins) / n * 100, 2) if n else 0.0
    metrics["profit_factor"] = round(wins.sum() / abs(losses.sum()), 4) if losses.sum() < 0 else 999.0
    metrics["net_pnl"] = round(float(pnls.sum()), 2)
    metrics["avg_pnl"] = round(float(pnls.mean()), 2)
    metrics["std_pnl"] = round(float(pnls.std()), 2)
    metrics["avg_win"] = round(float(wins.mean()), 2) if len(wins) else 0.0
    metrics["avg_loss"] = round(float(losses.mean()), 2) if len(losses) else 0.0
    metrics["best_trade"] = round(float(pnls.max()), 2)
    metrics["worst_trade"] = round(float(pnls.min()), 2)

    max_cw = max_cl = cur_w = cur_l = 0
    for p in pnls:
        if p > 0:
            cur_w += 1
            cur_l = 0
        else:
            cur_l += 1
            cur_w = 0
        max_cw = max(max_cw, cur_w)
        max_cl = max(max_cl, cur_l)
    metrics["max_consec_wins"] = max_cw
    metrics["max_consec_losses"] = max_cl

    equities = [r["equity"] for r in day_results]
    if equities:
        metrics["start_equity"] = round(equities[0], 2)
        metrics["final_equity"] = round(equities[-1], 2)
        metrics["peak_equity"] = round(max(equities), 2)
        peak = equities[0]
        max_dd = 0.0
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > max_dd:
                max_dd = dd
        metrics["max_drawdown_dollars"] = round(max_dd, 2)
        metrics["max_drawdown_pct"] = round(max_dd / metrics["peak_equity"] * 100, 4) if metrics["peak_equity"] else 0.0
    else:
        metrics["start_equity"] = 0.0
        metrics["final_equity"] = 0.0
        metrics["peak_equity"] = 0.0
        metrics["max_drawdown_dollars"] = 0.0
        metrics["max_drawdown_pct"] = 0.0

    return metrics


def run_backtest(
    df: pd.DataFrame,
    strategy_params: dict[str, Any] | None = None,
    tfs: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Run the v5 ORB backtest on the prepared DataFrame."""
    config = _make_config(strategy_params)
    tfs = tfs or TIMEFRAMES
    processor = StrategyProcessor(config, tfs)

    dates = sorted(df["date"].unique())
    day_results: list[dict[str, Any]] = []
    all_trades: list[dict[str, Any]] = []
    initial_capital = config.initial_capital

    for d in dates:
        daily = df[df["date"] == d]
        if len(daily) < 10:
            continue

        pre_counts = {name: len(eng.trade_history) for name, eng in processor.engines.items()}
        processor.reset_daily(d)

        for ts, row in daily.iterrows():
            processor.process_bar(
                ts,
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                int(row["time_min"]),
            )

        day_trades = []
        for name, eng in processor.engines.items():
            new_trades = eng.trade_history[pre_counts[name] :]
            for rec in new_trades:
                dct = asdict(rec)
                dct["tf"] = name
                day_trades.append(dct)
                all_trades.append(dct)

        day_pnl = sum(t["net"] for t in day_trades)
        cum = sum(t["net"] for t in all_trades)
        day_results.append(
            {
                "date": str(d),
                "trades": len(day_trades),
                "day_pnl": round(day_pnl, 2),
                "equity": round(initial_capital + cum, 2),
            }
        )

    metrics = compute_metrics(all_trades, day_results)
    return {
        "parameters": asdict(config),
        "timeframes": tfs,
        "metrics": metrics,
        "day_results": day_results,
        "trades": all_trades,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="v5 ORB NQ backtest")
    parser.add_argument("--input", required=True, help="Path to NQ_1min.csv(.gz)")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--max-entries", type=int, default=2, help="Max entries per timeframe")
    parser.add_argument("--max-contracts", type=int, default=5, help="Global max contracts")
    parser.add_argument("--baseline-index", type=float, default=None, help="Override baseline index")
    parser.add_argument("--tick-value", type=float, default=None, help="Override tick value (e.g., 5 for YM, 20 for NQ)")
    args = parser.parse_args()

    df = load_csv(args.input)
    params = {"max_entries": args.max_entries, "max_contracts": args.max_contracts}
    if args.baseline_index is not None:
        params["baseline_index"] = args.baseline_index
    if args.tick_value is not None:
        params["tick_value"] = args.tick_value

    result = run_backtest(df, strategy_params=params)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    m = result["metrics"]
    print(f"Days: {len(result['day_results'])}, Trades: {m['total_trades']}")
    print(f"Net PnL: ${m['net_pnl']:+.2f}, Win Rate: {m['win_rate']:.1f}%, PF: {m['profit_factor']:.2f}")
    print(f"Max DD: ${m['max_drawdown_dollars']:.2f} ({m['max_drawdown_pct']:.2f}%)")
    print(f"Saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
