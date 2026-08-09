#!/usr/bin/env python3
"""Clean ORB variant: no far target, stop at N * OR range, close at 16:00."""
from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min()) if len(dd) else 0.0


def profit_factor(pnls: list) -> float:
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    return gross_profit / gross_loss if gross_loss > 0 else 0.0


def backtest_orb_clean(df: pd.DataFrame,
                       or_minutes: int,
                       stop_multiplier: float = 1.0,
                       session_start: str = "09:30",
                       session_end: str = "16:00",
                       commission_per_side: float = 0.0) -> dict:
    df = df.copy()
    df.index = df.index.tz_localize("America/New_York") if df.index.tz is None else df.index.tz_convert("America/New_York")

    df["date"] = df.index.date
    df["time"] = df.index.time

    trades = []
    daily_pnl: dict = {}

    for date, day_df in df.groupby("date"):
        start_bar = day_df[day_df["time"] >= pd.Timestamp(session_start).time()]
        if start_bar.empty:
            continue

        start_idx = start_bar.index[0]
        or_end = start_idx + pd.Timedelta(minutes=or_minutes)
        or_df = day_df[(day_df.index >= start_idx) & (day_df.index < or_end)]
        if or_df.empty:
            continue

        or_high = or_df["high"].max()
        or_low = or_df["low"].min()
        or_range = or_high - or_low
        if or_range <= 0:
            continue

        trade_df = day_df[day_df.index >= or_end]
        trade_df = trade_df[trade_df["time"] < pd.Timestamp(session_end).time()]
        if trade_df.empty:
            continue

        entered = False
        for ts, bar in trade_df.iterrows():
            if entered:
                break

            direction = None
            entry_price = None
            stop_price = None

            if bar["high"] >= or_high:
                direction = 1
                entry_price = max(bar["open"], or_high)
                stop_price = entry_price - or_range * stop_multiplier
            elif bar["low"] <= or_low:
                direction = -1
                entry_price = min(bar["open"], or_low)
                stop_price = entry_price + or_range * stop_multiplier
            else:
                continue

            entered = True
            pnl = 0.0
            exit_price = entry_price
            exit_reason = "session_close"
            exit_idx = None

            # Walk forward; no target, only stop or session close
            for ts2, bar2 in trade_df.loc[ts:].iloc[1:].iterrows():
                if direction == 1:
                    if bar2["low"] <= stop_price:
                        exit_price = stop_price
                        exit_reason = "stop"
                        exit_idx = ts2
                        break
                else:
                    if bar2["high"] >= stop_price:
                        exit_price = stop_price
                        exit_reason = "stop"
                        exit_idx = ts2
                        break

            if exit_reason == "session_close":
                last_bar = trade_df.iloc[-1]
                exit_idx = trade_df.index[-1]
                exit_price = last_bar["close"]

            pnl = (exit_price - entry_price) * direction - 2 * commission_per_side
            pnl_dollars = pnl * 5.0

            trades.append({
                "date": str(date),
                "entry_time": str(ts),
                "direction": direction,
                "entry_price": float(entry_price),
                "stop_price": float(stop_price),
                "exit_time": str(exit_idx) if exit_idx else str(trade_df.index[-1]),
                "exit_price": float(exit_price),
                "exit_reason": exit_reason,
                "or_high": float(or_high),
                "or_low": float(or_low),
                "or_range": float(or_range),
                "pnl_points": float(pnl),
                "pnl_dollars": float(pnl_dollars),
            })

            daily_pnl[date] = daily_pnl.get(date, 0.0) + pnl_dollars

    total_points = sum(t["pnl_points"] for t in trades)
    wins = sum(1 for t in trades if t["pnl_points"] > 0)

    if daily_pnl:
        daily_series = pd.Series(daily_pnl).sort_index()
        equity = 100000.0 + (daily_series.cumsum())
    else:
        equity = pd.Series([100000.0])

    rets = equity.pct_change().dropna()
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if len(rets) > 1 and rets.std() > 0 else 0.0

    return {
        "trades": trades,
        "metrics": {
            "total_trades": len(trades),
            "win_rate": wins / len(trades) if trades else 0.0,
            "total_return": float((equity.iloc[-1] / 100000.0) - 1.0),
            "total_points": float(total_points),
            "total_dollars": float(total_points * 5.0),
            "avg_trade_points": float(total_points / len(trades)) if trades else 0.0,
            "sharpe": sharpe,
            "max_drawdown": float(max_drawdown(equity)),
            "profit_factor": float(profit_factor([t["pnl_points"] for t in trades])) if trades else 0.0,
        }
    }
