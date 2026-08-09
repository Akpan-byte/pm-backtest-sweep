#!/usr/bin/env python3
"""Full VWAP fade sweep for NQ/ES/YM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'orb_nq'))
from scalp_vwap_fade import load_ohlcv, backtest_vwap_fade
from prop_payout_sim import simulate_payouts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--contract-value', type=float, required=True)
    parser.add_argument('--symbol', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    df = load_ohlcv(args.data)

    configs = []
    for thresh in [10, 15, 20]:
        for target in [10, 15, 20]:
            for stop in [5, 10]:
                if target <= stop:
                    continue
                for max_trades in [5, 10, 20]:
                    for session_end in ['11:30', '16:00']:
                        configs.append((thresh, target, stop, max_trades, session_end))

    results = []
    for thresh, target, stop, max_trades, session_end in configs:
        r = backtest_vwap_fade(df, thresh, target, stop, session_end=session_end,
                               max_trades_per_day=max_trades, contract_value=args.contract_value)
        daily_pnl = {k: float(v) for k, v in r['daily_pnl'].items()}

        sim_50k = simulate_payouts(daily_pnl, 50000.0, 3000.0, 900.0, max_payout=1200.0)
        sim_150k = simulate_payouts(daily_pnl, 150000.0, 10000.0, 3000.0, max_payout=4000.0)

        m = r['metrics']
        results.append({
            'symbol': args.symbol,
            'threshold': thresh,
            'target': target,
            'stop': stop,
            'max_trades': max_trades,
            'session_end': session_end,
            'total_trades': m['total_trades'],
            'win_rate': m['win_rate'],
            'total_dollars': m['total_dollars'],
            'avg_trade_points': m['avg_trade_points'],
            'sharpe': m['sharpe'],
            'max_drawdown': m['max_drawdown'],
            'sim_50k': {k: v for k, v in sim_50k.items() if k != 'first_few_payouts'},
            'sim_150k': {k: v for k, v in sim_150k.items() if k != 'first_few_payouts'},
        })
        print(f"{args.symbol} t={thresh} tgt={target} s={stop} mt={max_trades} end={session_end}: "
              f"trades={m['total_trades']} wr={m['win_rate']:.1%} total=${m['total_dollars']:,.0f} "
              f"50k_first={sim_50k['first_payout_days']} 150k_first={sim_150k['first_payout_days']}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({'results': results}, indent=2) + '\n')
    print(f'\nSaved {args.output}')


if __name__ == '__main__':
    main()
