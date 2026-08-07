#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from data import load_ohlcv
from orb import backtest_orb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--tf', required=True, choices=['15m','30m','1h'])
    parser.add_argument('--rr', type=float, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    tf_map = {'15m': 15, '30m': 30, '1h': 60}
    df = load_ohlcv(args.data_path)
    result = backtest_orb(df, or_minutes=tf_map[args.tf], rr_ratio=args.rr)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str) + '\n')
    m = result['metrics']
    print(f"TF={args.tf} RR={args.rr}: ret={m['total_return']:.2%} sharpe={m['sharpe']:.3f} wr={m['win_rate']:.1%} trades={m['total_trades']} dd={m['max_drawdown']:.2%} pts={m['total_points']:.1f}")

if __name__ == '__main__':
    main()
