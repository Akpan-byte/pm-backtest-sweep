#!/usr/bin/env python3
"""Opening Range Breakout (ORB) strategy for NQ futures.

Rules:
- Define opening range as first N minutes after 09:30 ET.
- Wait for the OR period to close.
- First breakout of OR high -> long; stop = OR low; target = OR high + range * rr.
- First breakout of OR low -> short; stop = OR high; target = OR low - range * rr.
- Only one trade per day.
- Close at 16:00 ET if not already exited.
- 1 contract sizing -> P&L reported in NQ points and dollars ($5/point).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def backtest_orb(df: pd.DataFrame,
                 or_minutes: int,
                 rr_ratio: float,
                 session_start: str = "09:30",
                 session_end: str = "16:00",
                 one_trade_per_day: bool = True,
                 commission_per_side: float = 0.0) -> dict:
    """Run ORB backtest on a DataFrame indexed by timestamp.

    Parameters
    ----------
    df : OHLCV DataFrame with DatetimeIndex in America/New_York.
    or_minutes : length of opening range in minutes (15, 30, 60).
    rr_ratio : risk/reward multiplier for target distance (target = entry ± range * rr).
    """
    df = df.copy()
    df.index = df.index.tz_localize("America/New_York") if df.index.tz is None else df.index.tz_convert("America/New_York")

    # Filter to trading days with a 09:30 bar
    df["date"] = df.index.date
    df["time"] = df.index.time

    trades = []
    daily_pnl: dict = {}

    for date, day_df in df.groupby("date"):
        # Find 09:30 bar
        start_bar = day_df[day_df["time"] >= pd.Timestamp(session_start).time()]
        if start_bar.empty:
            continue

        start_idx = start_bar.index[0]
        # Opening range covers first or_minutes after 09:30
        or_end = start_idx + pd.Timedelta(minutes=or_minutes)
        or_df = day_df[(day_df.index >= start_idx) & (day_df.index < or_end)]
        if or_df.empty:
            continue

        or_high = or_df["high"].max()
        or_low = or_df["low"].min()
        or_range = or_high - or_low
        if or_range <= 0:
            continue

        # Trading bars after OR
        trade_df = day_df[day_df.index >= or_end]
        trade_df = trade_df[trade_df["time"] < pd.Timestamp(session_end).time()]
        if trade_df.empty:
            continue

        entered = False
        for ts, bar in trade_df.iterrows():
            if entered and one_trade_per_day:
                break

            direction = None
            entry_price = None

            # Breakout above OR high
            if bar["high"] >= or_high:
                direction = 1
                entry_price = max(bar["open"], or_high)  # assume fill at breakout level

            # Breakout below OR low
            elif bar["low"] <= or_low:
                direction = -1
                entry_price = min(bar["open"], or_low)

            if direction is None:
                continue

            stop_price = or_low if direction == 1 else or_high
            risk = abs(entry_price - stop_price)
            if risk <= 0:
                continue

            target_price = entry_price + direction * risk * rr_ratio

            # Simulate from next bar onwards
            exit_idx = None
            exit_price = None
            exit_reason = None
            for ts2, bar2 in trade_df.loc[ts:].iloc[1:].iterrows():
                if direction == 1:
                    if bar2["low"] <= stop_price:
                        exit_price = stop_price
                        exit_reason = "stop"
                        break
                    elif bar2["high"] >= target_price:
                        exit_price = target_price
                        exit_reason = "target"
                        break
                else:
                    if bar2["high"] >= stop_price:
                        exit_price = stop_price
                        exit_reason = "stop"
                        break
                    elif bar2["low"] <= target_price:
                        exit_price = target_price
                        exit_reason = "target"
                        break

            # If not exited by session end, close at session end close
            if exit_price is None:
                last_bar = trade_df.iloc[-1]
                exit_price = last_bar["close"]
                exit_reason = "session_close"
                exit_idx = trade_df.index[-1]

            raw_pnl = direction * (exit_price - entry_price)
            pnl = raw_pnl - 2 * commission_per_side

            trades.append({
                "date": str(date),
                "entry_time": str(ts),
                "direction": direction,
                "entry_price": float(entry_price),
                "stop_price": float(stop_price),
                "target_price": float(target_price),
                "exit_time": str(exit_idx) if exit_idx else str(trade_df.index[-1]),
                "exit_price": float(exit_price),
                "exit_reason": exit_reason,
                "or_high": float(or_high),
                "or_low": float(or_low),
                "or_range": float(or_range),
                "pnl_points": float(pnl),
                "pnl_dollars": float(pnl * 20.0),
            })

            daily_pnl[date] = daily_pnl.get(date, 0.0) + pnl
            entered = True

    total_points = sum(t["pnl_points"] for t in trades)
    wins = sum(1 for t in trades if t["pnl_points"] > 0)

    # Build equity curve from daily P&L
    if daily_pnl:
        daily_series = pd.Series({str(k): v for k, v in daily_pnl.items()}).sort_index()
        equity = 100000.0 + (daily_series.cumsum() * 5.0)
    else:
        daily_series = pd.Series(dtype=float)
        equity = pd.Series([100000.0])

    rets = equity.pct_change().dropna()
    periods_per_year = 252

    from common import sharpe_ratio, max_drawdown, profit_factor

    return {
        "trades": trades,
        "daily_pnl": {str(k): v for k, v in daily_pnl.items()},
        "equity": equity,
        "metrics": {
            "total_trades": len(trades),
            "win_rate": wins / len(trades) if trades else 0.0,
            "total_return": float((equity.iloc[-1] / 100000.0) - 1.0),
            "total_points": float(total_points),
            "total_dollars": float(total_points * 5.0),
            "avg_trade_points": float(total_points / len(trades)) if trades else 0.0,
            "sharpe": float(sharpe_ratio(rets, periods_per_year)) if len(rets) > 1 else 0.0,
            "max_drawdown": float(max_drawdown(equity)),
            "profit_factor": float(profit_factor([t["pnl_points"] for t in trades])) if trades else 0.0,
        }
    }
