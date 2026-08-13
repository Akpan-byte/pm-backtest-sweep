#!/usr/bin/env python3
"""Run the existing v5 Topstep XFA payout simulator on combined main+counter daily PnL."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/config/projects/trading")
sys.path.insert(0, "/config/projects/trading/v5_orb_nq_backtest")

from scripts.xfa_simulator import XFARules, simulate_xfa


def _daily_df(daily_pnl: dict) -> pd.DataFrame:
    rows = [{"date": pd.to_datetime(d).date(), "day_pnl": v} for d, v in daily_pnl.items()]
    df = pd.DataFrame(rows).set_index("date").sort_index()
    return df


def _rules(account_size: str, path: str) -> XFARules:
    if account_size == "50K":
        mll_offset = 2_000.0
        dll = 1_000.0
        max_contracts = 5
        payout_cap = 4_000.0 if path == "standard" else 6_000.0
    else:
        mll_offset = 4_500.0
        dll = 3_000.0
        max_contracts = 15
        payout_cap = 10_000.0 if path == "standard" else 12_000.0
    return XFARules(
        name=f"topstep_{account_size.lower()}_{path}",
        account_size_label=account_size,
        mll_offset=mll_offset,
        dll=dll,
        max_contracts=max_contracts,
        payout_cap=payout_cap,
        path=path,
    )


def analyze(input_path: str, output_path: str) -> dict:
    data = json.loads(Path(input_path).read_text())
    summary = {}
    for sym, result in data.items():
        summary[sym] = {}
        df = _daily_df(result["daily_pnl"])
        for account_size in ["50K", "150K"]:
            for path in ["standard", "consistency"]:
                rules = _rules(account_size, path)
                res = simulate_xfa(df, rules)
                s = res.summary()
                # add simple per-month rate over the ~10yr sample
                n_days = len(df)
                s["payouts_per_month"] = round(s["n_payouts"] / (n_days / 365.25 * 12), 2)
                summary[sym][f"{account_size}_{path}"] = s
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/tmp/combined_full_20k.json")
    parser.add_argument("--output", default="/tmp/combined_xfa_summary.json")
    args = parser.parse_args()
    summary = analyze(args.input, args.output)
    print(f"Saved XFA summary to {args.output}")
    for sym, scenarios in summary.items():
        print(f"\n{sym}:")
        for scenario, stats in scenarios.items():
            print(f"  {scenario}: {stats['n_payouts']} payouts, "
                  f"{stats['n_blows']} blows, {stats['payouts_per_month']}/mo, "
                  f"first {stats['days_to_first_payout']}d, "
                  f"avg between {stats['avg_days_between_payouts']}d")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
