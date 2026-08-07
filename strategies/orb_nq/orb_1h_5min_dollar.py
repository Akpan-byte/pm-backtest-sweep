#!/usr/bin/env python3
"""1h signal + 5min execution ORB with dollar stop/target for NQ ($20/point)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def backtest_1h_5min_dollar(df_1h: pd.DataFrame,
                             df_5min: pd.DataFrame,
                             or_minutes: int,
                             dollar_stop: float,
                             dollar_target: float,
                             contract_value: float = 20.0,
                             session_start: str = "09:30",
                             session_end: str = "16:00") -> dict:
    df_1h = df_1h.copy()
    df_1h.index = df_1h.index.tz_localize("America/New_York") if df_1h.index.tz is None else df_1h.index.tz_convert("America/New_York")
    df_1h["date"] = df_1h.index.date
    df_1h["time"] = df_1h.index.time

    df_5min = df_5min.copy()
    df_5min.index = df_5min.index.tz_localize("America/New_York") if df_5min.index.tz is None else df_5min.index.tz_convert("America/New_York")

    trades = []
    daily_pnl = {}

    for date, day_1h in df_1h.groupby("date"):
        mask = day_1h["time"] >= pd.Timestamp(session_start).time()
        if not mask.any():
            continue
        start_idx = day_1h[mask].index[0]
        or_end = start_idx + pd.Timedelta(minutes=or_minutes)
        or_df = day_1h[(day_1h.index >= start_idx) & (day_1h.index < or_end)]
        if or_df.empty:
            continue
        or_high = or_df["high"].max()
        or_low = or_df["low"].min()
        or_range = or_high - or_low
        if or_range <= 0:
            continue

        trade_1h = day_1h[(day_1h.index >= or_end) & (day_1h["time"] < pd.Timestamp(session_end).time())]
        if trade_1h.empty:
            continue

        entry = None
        for ts, bar in trade_1h.iterrows():
            if bar["high"] >= or_high:
                entry = (ts, 1, max(bar["open"], or_high))
                break
            elif bar["low"] <= or_low:
                entry = (ts, -1, min(bar["open"], or_low))
                break
        if entry is None:
            continue

        ts, direction, entry_price = entry
        or_stop = entry_price - or_range * 0.5 * direction
        stop_price = entry_price - (dollar_stop / contract_value) * direction
        target_price = entry_price + (dollar_target / contract_value) * direction
        eff_stop = max(or_stop, stop_price) if direction == 1 else min(or_stop, stop_price)

        exec_bars = df_5min[(df_5min.index > ts) & (df_5min.index.time < pd.Timestamp(session_end).time())]
        if len(exec_bars) == 0:
            continue

        exit_price = exec_bars["close"].iloc[-1]
        exit_reason = "session_close"

        for ts2, bar2 in exec_bars.iterrows():
            if direction == 1:
                if bar2["low"] <= eff_stop:
                    exit_price = eff_stop
                    exit_reason = "stop"
                    break
                if bar2["high"] >= target_price:
                    exit_price = target_price
                    exit_reason = "target"
                    break
            else:
                if bar2["high"] >= eff_stop:
                    exit_price = eff_stop
                    exit_reason = "stop"
                    break
                if bar2["low"] <= target_price:
                    exit_price = target_price
                    exit_reason = "target"
                    break

        pnl = (exit_price - entry_price) * direction
        pnl_dollars = pnl * contract_value
        trades.append({"date": str(date), "pnl_dollars": pnl_dollars, "exit_reason": exit_reason})
        daily_pnl[date] = daily_pnl.get(date, 0.0) + pnl_dollars

    wins = sum(1 for t in trades if t["pnl_dollars"] > 0)
    total = sum(t["pnl_dollars"] for t in trades)
    series = pd.Series(daily_pnl).sort_index() if daily_pnl else pd.Series([0.0])
    equity = 100000.0 + series.cumsum()
    rets = equity.pct_change().dropna()
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if len(rets) > 1 and rets.std() > 0 else 0.0
    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = float(dd.min())

    return {
        "total_trades": len(trades),
        "win_rate": wins / len(trades) if trades else 0.0,
        "total_return": float((equity.iloc[-1] / 100000.0) - 1.0),
        "total_dollars": float(total),
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "days_exceed_900": sum(1 for v in daily_pnl.values() if v < -900),
        "days_exceed_3000": sum(1 for v in daily_pnl.values() if v < -3000),
    }
