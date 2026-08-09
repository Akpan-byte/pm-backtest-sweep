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
from orb_1h_5min_dollar import backtest_1h_5min_dollar


def resample_ohlcv(df_1min: pd.DataFrame, freq: str) -> pd.DataFrame:
    df = df_1min.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")
    else:
        df.index = df.index.tz_convert("America/New_York")
    resampled = df.resample(freq, label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()
    return resampled


def backtest_clean_orb(df_or: pd.DataFrame, df_exec: pd.DataFrame, or_minutes: int, stop_mult: float,
                       session_start: str = "09:30", session_end: str = "16:00",
                       contract_value: float = 20.0) -> dict:
    """Clean ORB: OR built from df_or, breakout and stop execution checked on df_exec (e.g. 1min)."""
    df_or = df_or.copy()
    df_or.index = df_or.index.tz_localize("America/New_York") if df_or.index.tz is None else df_or.index.tz_convert("America/New_York")
    df_or["date"] = df_or.index.date
    df_or["time"] = df_or.index.time

    df_exec = df_exec.copy()
    df_exec.index = df_exec.index.tz_localize("America/New_York") if df_exec.index.tz is None else df_exec.index.tz_convert("America/New_York")
    df_exec["date"] = df_exec.index.date
    df_exec["time"] = df_exec.index.time

    daily_pnl = {}
    trades = []

    for date, day_or in df_or.groupby("date"):
        start_bar = day_or[day_or["time"] >= pd.Timestamp(session_start).time()]
        if start_bar.empty:
            continue
        start_idx = start_bar.index[0]
        or_end = start_idx + pd.Timedelta(minutes=or_minutes)
        or_df = day_or[(day_or.index >= start_idx) & (day_or.index < or_end)]
        if or_df.empty:
            continue
        or_high = or_df["high"].max()
        or_low = or_df["low"].min()
        or_range = or_high - or_low
        if or_range <= 0:
            continue

        # Execution bars for this day
        day_exec = df_exec[df_exec["date"] == date]
        trade_exec = day_exec[(day_exec.index >= or_end) & (day_exec["time"] < pd.Timestamp(session_end).time())]
        if trade_exec.empty:
            continue

        entry = None
        for ts, bar in trade_exec.iterrows():
            if bar["high"] >= or_high:
                entry = (ts, 1, max(bar["open"], or_high))
                break
            elif bar["low"] <= or_low:
                entry = (ts, -1, min(bar["open"], or_low))
                break
        if entry is None:
            continue

        ts, direction, entry_price = entry
        stop_price = entry_price - direction * or_range * stop_mult

        exit_price = trade_exec["close"].iloc[-1]
        exit_reason = "session_close"
        for ts2, bar2 in trade_exec.loc[ts:].iloc[1:].iterrows():
            if direction == 1:
                if bar2["low"] <= stop_price:
                    exit_price = stop_price
                    exit_reason = "stop"
                    break
            else:
                if bar2["high"] >= stop_price:
                    exit_price = stop_price
                    exit_reason = "stop"
                    break

        pnl_points = (exit_price - entry_price) * direction
        pnl_dollars = pnl_points * contract_value
        daily_pnl[date] = daily_pnl.get(date, 0.0) + pnl_dollars
        trades.append({"pnl_dollars": pnl_dollars, "exit_reason": exit_reason})

    return {"daily_pnl": daily_pnl, "trades": trades}


def backtest_dollar_orb(df_5min: pd.DataFrame, dollar_stop: float, dollar_target: float,
                        contract_value: float = 20.0) -> dict:
    """Dollar stop/target ORB using 5min bars (240-min OR, same logic as saved dollar sweep)."""
    # Use the 1h-5min function with 5min signal and 5min execution by passing same df
    df_5min = df_5min.copy()
    if df_5min.index.tz is None:
        df_5min.index = df_5min.index.tz_localize("America/New_York")
    else:
        df_5min.index = df_5min.index.tz_convert("America/New_York")

    # The saved dollar sweep used 5min signal and 5min execution.
    # Reuse backtest_1h_5min_dollar with df_5min as both signal and execution bars.
    return backtest_1h_5min_dollar(
        df_5min, df_5min, or_minutes=240,
        dollar_stop=dollar_stop, dollar_target=dollar_target,
        contract_value=contract_value
    )


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
    parser.add_argument("--dollar-stop", type=float)
    parser.add_argument("--dollar-target", type=float)
    parser.add_argument("--or-tf")
    parser.add_argument("--stop-mult", type=float)
    parser.add_argument("--account-start", type=float, default=100000.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df_1min = load_ohlcv("market_data/NQ_1min.csv")

    if args.type == "dollar":
        df_5min = resample_ohlcv(df_1min, "5min")
        result = backtest_dollar_orb(df_5min, args.dollar_stop, args.dollar_target)
        daily_pnl = result.get("daily_pnl", {})
        metrics = {k: v for k, v in result.items() if k != "daily_pnl"}
    else:
        tf_minutes = {"1h": 60, "4h": 240}[args.or_tf]
        df_tf = resample_ohlcv(df_1min, args.or_tf)
        result = backtest_clean_orb(df_tf, df_1min, tf_minutes, args.stop_mult)
        daily_pnl = result["daily_pnl"]
        wins = sum(1 for t in result["trades"] if t["pnl_dollars"] > 0)
        total = sum(t["pnl_dollars"] for t in result["trades"])
        metrics = {
            "total_trades": len(result["trades"]),
            "win_rate": wins / len(result["trades"]) if result["trades"] else 0.0,
            "total_dollars": total,
        }

    payout_5day = calculate_payouts(daily_pnl, args.account_start, min_active_days=5)
    payout_3day = calculate_payouts(daily_pnl, args.account_start, min_active_days=3)

    out = {
        "name": args.name,
        "type": args.type,
        "metrics": metrics,
        "payout_5day_min": payout_5day,
        "payout_3day_min": payout_3day,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
