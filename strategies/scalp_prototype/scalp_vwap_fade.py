#!/usr/bin/env python3
"""VWAP fade scalper prototype for NQ/ES/YM."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_ohlcv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['timestamp'] if 'timestamp' in pd.read_csv(path, nrows=0).columns else [0])
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    else:
        df = df.set_index(df.columns[0])
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize('America/New_York')
    else:
        df.index = df.index.tz_convert('America/New_York')
    return df


def backtest_vwap_fade(df: pd.DataFrame,
                       threshold_points: float,
                       target_points: float,
                       stop_points: float,
                       warmup_minutes: int = 15,
                       session_start: str = "09:30",
                       session_end: str = "11:30",
                       contract_value: float = 20.0,
                       max_trades_per_day: int = 2) -> dict:
    df = df.copy()
    df['date'] = df.index.date
    df['time'] = df.index.time
    df['typical'] = (df['high'] + df['low'] + df['close']) / 3.0

    trades = []
    daily_pnl: dict = {}

    for date, day_df in df.groupby('date'):
        session = day_df[day_df['time'] >= pd.Timestamp(session_start).time()]
        session = session[session['time'] < pd.Timestamp(session_end).time()]
        if len(session) <= warmup_minutes:
            continue

        warmup = session.iloc[:warmup_minutes]
        trade_df = session.iloc[warmup_minutes:]

        # VWAP from market open through current bar, computed iteratively
        cum_pv = (warmup['typical'] * warmup['volume']).sum()
        cum_v = warmup['volume'].sum()

        trades_today = 0
        for ts, bar in trade_df.iterrows():
            if trades_today >= max_trades_per_day:
                break

            cum_pv += bar['typical'] * bar['volume']
            cum_v += bar['volume']
            if cum_v <= 0:
                continue
            vwap = cum_pv / cum_v

            direction = None
            if bar['high'] >= vwap + threshold_points:
                direction = -1  # fade short
                entry = max(bar['open'], vwap + threshold_points)
            elif bar['low'] <= vwap - threshold_points:
                direction = 1   # fade long
                entry = min(bar['open'], vwap - threshold_points)
            else:
                continue

            target = entry + direction * target_points
            stop = entry - direction * stop_points

            exit_price = entry
            exit_reason = "session_close"
            for ts2, bar2 in trade_df.loc[ts:].iloc[1:].iterrows():
                if direction == 1:
                    if bar2['high'] >= target:
                        exit_price = target
                        exit_reason = "target"
                        break
                    if bar2['low'] <= stop:
                        exit_price = stop
                        exit_reason = "stop"
                        break
                else:
                    if bar2['low'] <= target:
                        exit_price = target
                        exit_reason = "target"
                        break
                    if bar2['high'] >= stop:
                        exit_price = stop
                        exit_reason = "stop"
                        break

            pnl_points = (exit_price - entry) * direction
            pnl_dollars = pnl_points * contract_value
            trades.append({
                'date': str(date),
                'entry_time': str(ts),
                'direction': direction,
                'entry': float(entry),
                'vwap': float(vwap),
                'exit': float(exit_price),
                'exit_reason': exit_reason,
                'pnl_points': float(pnl_points),
                'pnl_dollars': float(pnl_dollars),
            })
            daily_pnl[date] = daily_pnl.get(date, 0.0) + pnl_dollars
            trades_today += 1

    wins = sum(1 for t in trades if t['pnl_dollars'] > 0)
    total_points = sum(t['pnl_points'] for t in trades)
    total_dollars = sum(t['pnl_dollars'] for t in trades)

    if daily_pnl:
        series = pd.Series(daily_pnl).sort_index()
        equity = 100000.0 + series.cumsum()
        rets = equity.pct_change().dropna()
        sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if len(rets) > 1 and rets.std() > 0 else 0.0
        peak = equity.cummax()
        max_dd = float((equity - peak).min() / peak.max())
    else:
        sharpe = 0.0
        max_dd = 0.0

    return {
        'trades': trades,
        'metrics': {
            'total_trades': len(trades),
            'win_rate': wins / len(trades) if trades else 0.0,
            'total_points': float(total_points),
            'total_dollars': float(total_dollars),
            'avg_trade_points': float(total_points / len(trades)) if trades else 0.0,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
        },
        'daily_pnl': {str(k): float(v) for k, v in daily_pnl.items()},
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--contract-value', type=float, default=20.0)
    parser.add_argument('--sample-years', type=int)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    df = load_ohlcv(args.data)
    if args.sample_years:
        cutoff = df.index.max() - pd.DateOffset(years=args.sample_years)
        df = df[df.index >= cutoff]

    results = []
    for thresh in [5, 10, 15, 20, 25]:
        for target in [10, 15, 20, 30, 50]:
            for stop in [5, 10, 15, 20]:
                if target <= stop:
                    continue
                r = backtest_vwap_fade(df, thresh, target, stop, contract_value=args.contract_value)
                m = r['metrics']
                results.append({
                    'threshold': thresh,
                    'target': target,
                    'stop': stop,
                    **m,
                })
                print(f"thresh={thresh} target={target} stop={stop}: trades={m['total_trades']} wr={m['win_rate']:.1%} total=${m['total_dollars']:,.0f} sharpe={m['sharpe']:.2f} dd={m['max_drawdown']:.1%}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({'results': results}, indent=2) + '\n')
    print(f'\nSaved {args.output}')
