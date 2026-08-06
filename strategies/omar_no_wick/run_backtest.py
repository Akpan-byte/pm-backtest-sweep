#!/usr/bin/env python3
"""Backtest Omar Nowick No Wick strategy on NQ 5m data."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import sharpe_ratio, sortino_ratio, max_drawdown, profit_factor, probabilistic_sharpe_ratio, deflated_sharpe_ratio
from data import load_ohlcv
from no_wick import detect_structure, detect_a_setups, detect_no_wick, simulate_no_wick


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output", default="results/no_wick_nq.json")
    parser.add_argument("--min-sl-pips", type=float, default=5.0)
    parser.add_argument("--breather-pct", type=float, default=0.10)
    parser.add_argument("--omar-offset-pips", type=float, default=2.0)
    parser.add_argument("--omar-threshold-pips", type=float, default=10.0)
    parser.add_argument("--pip-size", type=float, default=1.0)
    args = parser.parse_args()

    df = load_ohlcv(args.data_path)
    print(f"Loaded {len(df)} rows from {args.data_path}")

    df = detect_structure(df)
    df = detect_a_setups(df)
    df = detect_no_wick(df, pip_tol=0.0)

    eq, trades = simulate_no_wick(
        df,
        min_sl_pips=args.min_sl_pips,
        breather_pct=args.breather_pct,
        omar_offset_pips=args.omar_offset_pips,
        omar_threshold_pips=args.omar_threshold_pips,
        pip_size=args.pip_size,
    )

    total_return = (eq.iloc[-1] / 100000.0) - 1.0
    trade_pnls = [t["pnl"] for t in trades]
    wins = sum(1 for p in trade_pnls if p > 0)
    periods_per_year = 252 * 78  # ~78 5m bars per trading day
    rets = eq.pct_change().dropna()

    metrics = {
        "total_return": float(total_return),
        "net_profit": float(eq.iloc[-1] - 100000.0),
        "final_equity": float(eq.iloc[-1]),
        "sharpe": float(sharpe_ratio(rets, periods_per_year)),
        "sortino": float(sortino_ratio(rets, periods_per_year)),
        "max_drawdown": float(max_drawdown(eq)),
        "trades": len(trades),
        "win_rate": wins / len(trades) if trades else 0.0,
        "profit_factor": float(profit_factor(trade_pnls)) if trade_pnls else 0.0,
        "psr": float(probabilistic_sharpe_ratio(rets, 0.0, periods_per_year)),
        "dsr": float(deflated_sharpe_ratio(rets, 1, periods_per_year)),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"metrics": metrics, "trades": trades}, indent=2, default=str) + "\n")
    print(out)
    print("metrics:", metrics)


if __name__ == "__main__":
    main()
