#!/usr/bin/env python3
"""Payout analysis for ORB strategies under Topstep-style rules."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data import load_ohlcv
from orb_clean import backtest_orb_clean
from orb_clean_dollar import backtest_orb_clean_dollar
from prop_payout_sim import simulate_payouts


def calculate_payouts(daily_pnl: dict, account_start: float, payout_rate: float = 0.40,
                      min_active_days: int = 5, consistency_max_day_pct: float = 0.50,
                      window_weeks: int = 2) -> dict:
    """Calculate prop-firm payouts over rolling windows."""
    if not daily_pnl:
        return {}

    series = pd.Series(daily_pnl).sort_index()
    # Group into windows of approximately window_weeks * 5 trading days
    window_size = window_weeks * 5

    eligible_windows = []
    total_payout = 0.0
    total_growth = 0.0

    for i in range(0, len(series) - window_size + 1, window_size):
        window = series.iloc[i:i + window_size]
        active_days = (window != 0).sum()
        if active_days < min_active_days:
            continue
        window_profit = window.sum()
        if window_profit <= 0:
            continue
        # Consistency: no single day > 50% of total window profit
        max_day_pct = window.max() / window_profit if window_profit > 0 else 1.0
        if max_day_pct > consistency_max_day_pct:
            continue
        payout = window_profit * payout_rate
        eligible_windows.append({
            "start": str(window.index[0]),
            "end": str(window.index[-1]),
            "active_days": int(active_days),
            "window_profit": float(window_profit),
            "payout": float(payout),
            "max_day_pct": float(max_day_pct),
        })
        total_payout += payout
        total_growth += window_profit

    avg_payout = total_payout / len(eligible_windows) if eligible_windows else 0.0
    return {
        "account_start": account_start,
        "payout_rate": payout_rate,
        "min_active_days": min_active_days,
        "consistency_max_day_pct": consistency_max_day_pct,
        "window_weeks": window_weeks,
        "eligible_windows": len(eligible_windows),
        "total_window_profit": float(total_growth),
        "total_payout": float(total_payout),
        "avg_payout_per_window": float(avg_payout),
        "windows_per_year": float(len(eligible_windows) / (len(series) / 252)) if len(series) > 0 else 0.0,
        "estimated_annual_payout": float(avg_payout * (len(eligible_windows) / (len(series) / 252))) if len(series) > 0 else 0.0,
        "first_few_windows": eligible_windows[:5],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["dollar", "clean"], required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--dollar-stop", type=float, default=0)
    parser.add_argument("--dollar-target", type=float, default=0)
    parser.add_argument("--or-tf")
    parser.add_argument("--stop-mult", type=float, default=0)
    parser.add_argument("--data-dir", default="market_data")
    parser.add_argument("--account-start", type=float, default=100000.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.type == "dollar":
        # Use pre-built 5min data (matches original saved dollar sweep).
        df_5min = load_ohlcv(f"{args.data_dir}/NQ_5min.csv")
        result = backtest_orb_clean_dollar(
            df_5min,
            or_minutes=240,
            stop_multiplier=0.5,
            dollar_stop=args.dollar_stop,
            dollar_target=args.dollar_target,
            contract_value=20.0,
        )
        daily_pnl = {pd.to_datetime(t["date"]).date(): t["pnl_dollars"]
                     for t in result["trades"]}
        metrics = {k: v for k, v in result["metrics"].items() if k != "daily_pnl"}
    else:
        # Clean ORB: the saved sweep used 1h bars for both 1h OR and 4h OR.
        # 4h OR means the opening range spans 240 minutes, executed on 1h bars.
        df_tf = load_ohlcv(f"{args.data_dir}/NQ_1h.csv")
        tf_minutes = {"1h": 60, "4h": 240}[args.or_tf]
        # orb_clean returns pnl in MNQ ($5/point); scale to full NQ ($20/point) for payout analysis.
        result = backtest_orb_clean(df_tf, tf_minutes, args.stop_mult)
        scaled_trades = []
        daily_pnl: dict = {}
        for t in result["trades"]:
            pnl = t["pnl_dollars"] * 4.0
            d = pd.to_datetime(t["date"]).date()
            daily_pnl[d] = daily_pnl.get(d, 0.0) + pnl
            scaled_trades.append({"pnl_dollars": pnl})
        wins = sum(1 for t in scaled_trades if t["pnl_dollars"] > 0)
        total = sum(t["pnl_dollars"] for t in scaled_trades)
        metrics = {
            "total_trades": len(scaled_trades),
            "win_rate": wins / len(scaled_trades) if scaled_trades else 0.0,
            "total_dollars": total,
        }

    payout_5day = calculate_payouts(daily_pnl, args.account_start, min_active_days=5)
    payout_3day = calculate_payouts(daily_pnl, args.account_start, min_active_days=3)

    prop_sims = {}
    for label, account_start, profit_target, daily_loss_limit in [
        ("topstep_50k", 50000.0, 3000.0, 900.0),
        ("topstep_150k", 150000.0, 10000.0, 3000.0),
    ]:
        prop_sims[label] = simulate_payouts(
            daily_pnl,
            account_start=account_start,
            profit_target=profit_target,
            daily_loss_limit=daily_loss_limit,
            payout_rate=0.40,
            min_active_days=5,
            consistency_max_day_pct=0.50,
            max_payout=profit_target * 0.40,
        )

    out = {
        "name": args.name,
        "type": args.type,
        "metrics": metrics,
        "payout_5day_min": payout_5day,
        "payout_3day_min": payout_3day,
        "prop_sims": prop_sims,
        "daily_pnl": {str(k): float(v) for k, v in daily_pnl.items()},
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
