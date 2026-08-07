#!/usr/bin/env python3
import json
from pathlib import Path
from data import load_ohlcv
from orb import backtest_orb

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', default='results')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tfs = {'15m': 15, '30m': 30, '1h': 60}
    rrs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    results = []
    for tf, minutes in tfs.items():
        df = load_ohlcv(f"{args.data_dir}/NQ_{tf}.csv")
        for rr in rrs:
            result = backtest_orb(df, or_minutes=minutes, rr_ratio=rr)
            m = result['metrics']
            res = {
                'tf': tf,
                'rr': rr,
                **m,
            }
            results.append(res)
            (out_dir / f"orb_{tf}_rr{rr:.0f}.json").write_text(json.dumps(res, indent=2) + '\n')
            print(f"{tf} RR={rr:.0f}: ret={m['total_return']:.2%} sharpe={m['sharpe']:.3f} wr={m['win_rate']:.1%} trades={m['total_trades']} dd={m['max_drawdown']:.2%} pts={m['total_points']:.1f}")

    (out_dir / 'summary.json').write_text(json.dumps({'results': results}, indent=2) + '\n')

if __name__ == '__main__':
    main()
