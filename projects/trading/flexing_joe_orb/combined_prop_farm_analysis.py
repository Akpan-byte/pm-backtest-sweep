#!/usr/bin/env python3
"""Run the existing v4 prop-farm simulator on combined main+counter daily PnL."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/config/projects/trading")
sys.path.insert(0, "/config/projects/trading/v5_orb_nq_backtest")

from scripts.prop_farm_simulator_v4 import (
    CostAssumptions,
    TopstepRules,
    make_rules,
    simulate_prop_farm,
)


def _daily_df(daily_pnl: dict) -> pd.DataFrame:
    rows = [{"date": pd.to_datetime(d).date(), "net_pnl": v} for d, v in daily_pnl.items()]
    df = pd.DataFrame(rows).set_index("date").sort_index()
    return df


def analyze(
    input_path: str,
    output_path: str,
    desired_contracts: int = 1,
    daily_cap: float = 0.0,
) -> dict:
    data = json.loads(Path(input_path).read_text())
    costs = CostAssumptions()
    summary = {}
    rows = []
    for sym, result in data.items():
        summary[sym] = {}
        df = _daily_df(result["daily_pnl"])
        if desired_contracts > 1:
            df["net_pnl"] = df["net_pnl"] * desired_contracts
        if daily_cap and daily_cap > 0:
            df["net_pnl"] = df["net_pnl"].clip(upper=daily_cap)
        for account_size in ["50K", "150K"]:
            for path in ["standard", "consistency"]:
                rules = make_rules(account_size)
                res = simulate_prop_farm(
                    df,
                    variant=f"{sym}_combined",
                    rules=rules,
                    costs=costs,
                    path=path,
                    reset_after_payout=False,
                    desired_contracts=desired_contracts,
                )
                s = res.summary()
                n_payouts = sum(len(a.payouts) for a in res.attempts)
                s["n_payouts"] = n_payouts
                n_days = len(df)
                s["payouts_per_month"] = round(n_payouts / (n_days / 365.25 * 12), 2)
                summary[sym][f"{account_size}_{path}"] = s
                rows.append({
                    "symbol": sym,
                    "account_size": account_size,
                    "path": path,
                    "n_payouts": n_payouts,
                    "payouts_per_month": s["payouts_per_month"],
                    "net_profit": s["net_profit"],
                    "n_xfa_blows": s["n_xfa_blows"],
                    "n_combine_blows": s["n_combine_blows"],
                    "first_payout_day": s["first_payout_day"],
                })
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if output_path.endswith(".csv"):
        pd.DataFrame(rows).to_csv(output_path, index=False)
    else:
        Path(output_path).write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/tmp/combined_full_20k.json")
    parser.add_argument("--output", default="/tmp/combined_prop_farm_summary.json")
    parser.add_argument("--desired-contracts", type=int, default=1)
    parser.add_argument("--daily-cap", type=float, default=0.0, help="Cap daily PnL at this value before prop-farm simulation")
    args = parser.parse_args()
    summary = analyze(args.input, args.output, args.desired_contracts, args.daily_cap)
    print(f"Saved prop-farm summary to {args.output}")
    for sym, scenarios in summary.items():
        print(f"\n{sym} (desired {args.desired_contracts} contracts):")
        for scenario, stats in scenarios.items():
            print(f"  {scenario}: {stats['n_payouts']} payouts, "
                  f"{stats['n_xfa_blows']} xfa blows, {stats['n_combine_blows']} combine blows, "
                  f"{stats['payouts_per_month']}/mo, first {stats['first_payout_day']}d, "
                  f"net ${stats['net_profit']:,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
