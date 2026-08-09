#!/usr/bin/env python3
"""ORB with dollar-based stop loss and profit target for prop firms."""
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


def backtest_orb_prop(df_signal: pd.DataFrame,
                      df_exec: pd.DataFrame,
                      or_minutes: int,
                      dollar_stop: float,
                      dollar_target: float,
                      contract_value: float = 20.0,
                      session_start: str = "09:30",
                      session_end: str = "16:00",
                      commission_per_side: float = 0.0) -> dict:
    """
    df_signal: bars for OR and entry signal (e.g., 1h)
    df_exec: bars for stop/target execution (e.g., 5min or 1min)
    dollar_stop: max loss in dollars (positive number)
    dollar_target: profit target in dollars
    """
    df_signal = df_signal.copy()
    df_signal.index = df_signal.index.tz_localize("America/New_York") if df_signal.index.tz is None else df_signal.index.tz_convert("America/New_York")
    df_signal["date"] = df_signal.index.date
    df_signal["time"] = df_signal.index.time

    df_exec = df_exec.copy()
    df_exec.index = df_exec.index.tz_localize("America/New_York") if df_exec.index.tz is None else df_exec.index.tz_convert("America/New_York")

    trades = []
    daily_pnl: dict = {}

    for date, day_signal in df_signal.groupby("date"):
        start_bar = day_signal[day_signal["time"] >= pd.Timestamp(session_start).time()]
        if start_bar.empty:
            continue

        start_idx = start_bar.index[0]
        or_end = start_idx + pd.Timedelta(minutes=or_minutes)
        or_df = day_signal[(day_signal.index >= start_idx) & (day_signal.index < or_end)]
        if or_df.empty:
            continue

        or_high = or_df["high"].max()
        or_low = or_df["low"].min()
        or_range = or_high - or_low
        if or_range <= 0:
            continue

        trade_df = day_signal[day_signal.index >= or_end]
        trade_df = trade_df[trade_df["time"] < pd.Timestamp(session_end).time()]
        if trade_df.empty:
            continue

        entered = False
        for ts, bar in trade_df.iterrows():
            if entered:
                break

            direction = None
            entry_price = None
            or_stop = None

            if bar["high"] >= or_high:
                direction = 1
                entry_price = max(bar["open"], or_high)
                or_stop = entry_price - or_range * 0.5  # base 0.5x stop
            elif bar["low"] <= or_low:
                direction = -1
                entry_price = min(bar["open"], or_low)
                or_stop = entry_price + or_range * 0.5
            else:
                continue

            entered = True
            exit_price = entry_price
            exit_reason = "session_close"
            exit_idx = None

            # Price-based stops converted from dollars
            stop_price = entry_price - (dollar_stop / contract_value) * direction
            target_price = entry_price + (dollar_target / contract_value) * direction

            exec_bars = df_exec[(df_exec.index >= ts)]
            exec_bars = exec_bars[exec_bars.index.time < pd.Timestamp(session_end).time()]
            exec_bars = exec_bars[exec_bars.index > ts]

            for ts2, bar2 in exec_bars.iterrows():
                if direction == 1:
                    # Dollar stop hit
                    if bar2["low"] <= stop_price:
                        exit_price = stop_price
                        exit_reason = "dollar_stop"
                        exit_idx = ts2
                        break
                    # Dollar target hit
                    if bar2["high"] >= target_price:
                        exit_price = target_price
                        exit_reason = "dollar_target"
                        exit_idx = ts2
                        break
                    # OR 0.5 stop hit
                    if bar2["low"] <= or_stop:
                        exit_price = or_stop
                        exit_reason = "or_stop"
                        exit_idx = ts2
                        break
                else:
                    if bar2["high"] >= stop_price:
                        exit_price = stop_price
                        exit_reason = "dollar_stop"
                        exit_idx = ts2
                        break
                    if bar2["low"] <= target_price:
                        exit_price = target_price
                        exit_reason = "dollar_target"
                        exit_idx = ts2
                        break
                    if bar2["high"] >= or_stop:
                        exit_price = or_stop
                        exit_reason = "or_stop"
                        exit_idx = ts2
                        break

            if exit_reason == "session_close":
                last_bar = trade_df.iloc[-1]
                exit_idx = trade_df.index[-1]
                exit_price = last_bar["close"]

            pnl = (exit_price - entry_price) * direction - 2 * commission_per_side
            pnl_dollars = pnl * contract_value

            trades.append({
                "date": str(date),
                "entry_time": str(ts),
                "direction": direction,
                "entry_price": float(entry_price),
                "or_stop": float(or_stop),
                "dollar_stop_price": float(stop_price),
                "dollar_target_price": float(target_price),
                "exit_time": str(exit_idx) if exit_idx else str(trade_df.index[-1]),
                "exit_price": float(exit_price),
                "exit_reason": exit_reason,
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

    # Days exceeding daily loss limit
    exceed_900 = sum(1 for v in daily_pnl.values() if v < -900)
    exceed_3000 = sum(1 for v in daily_pnl.values() if v < -3000)

    return {
        "trades": trades,
        "metrics": {
            "total_trades": len(trades),
            "win_rate": wins / len(trades) if trades else 0.0,
            "total_return": float((equity.iloc[-1] / 100000.0) - 1.0),
            "total_points": float(total_points),
            "total_dollars": float(total_points * contract_value),
            "avg_trade_points": float(total_points / len(trades)) if trades else 0.0,
            "sharpe": sharpe,
            "max_drawdown": float(max_drawdown(equity)),
            "profit_factor": float(profit_factor([t["pnl_points"] for t in trades])) if trades else 0.0,
            "days_exceed_900": exceed_900,
            "days_exceed_3000": exceed_3000,
        }
    }
