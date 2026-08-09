#!/usr/bin/env python3
"""Sweep dollar stop/target to find configs that best hit Topstep profit targets."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data import load_ohlcv
from orb_clean_dollar import backtest_orb_clean_dollar
from prop_payout_sim import simulate_payouts


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="market_data")
    parser.add_argument("--output", default="results/target_opt_sweep.json")
    args = parser.parse_args()

    df_5min = load_ohlcv(f"{args.data_dir}/NQ_5min.csv")

    configs = []
    for dollar_stop in [500, 750, 900, 1000, 1250, 1500, 2000, 2500, 3000]:
        for dollar_target in [500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 5000]:
            if dollar_target < dollar_stop * 0.5:
                continue
            configs.append((dollar_stop, dollar_target))

    results = []
    for dollar_stop, dollar_target in configs:
        result = backtest_orb_clean_dollar(
            df_5min, or_minutes=240, stop_multiplier=0.5,
            dollar_stop=dollar_stop, dollar_target=dollar_target,
            contract_value=20.0,
        )
        daily_pnl = {}
        for t in result["trades"]:
            d = pd.to_datetime(t["date"]).date()
            daily_pnl[d] = daily_pnl.get(d, 0.0) + t["pnl_dollars"]

        sim_50k = simulate_payouts(
            daily_pnl, account_start=50000.0, profit_target=3000.0,
            daily_loss_limit=900.0, payout_rate=0.40,
            min_active_days=5, consistency_max_day_pct=0.50,
            max_payout=1200.0,
        )
        sim_150k = simulate_payouts(
            daily_pnl, account_start=150000.0, profit_target=10000.0,
            daily_loss_limit=3000.0, payout_rate=0.40,
            min_active_days=5, consistency_max_day_pct=0.50,
            max_payout=4000.0,
        )

        m = result["metrics"]
        results.append({
            "dollar_stop": dollar_stop,
            "dollar_target": dollar_target,
            "total_trades": m["total_trades"],
            "win_rate": m["win_rate"],
            "total_dollars": m["total_dollars"],
            "sharpe": m["sharpe"],
            "max_drawdown": m["max_drawdown"],
            "profit_factor": m["profit_factor"],
            "sim_50k": sim_50k,
            "sim_150k": sim_150k,
        })
        print(f"ds={dollar_stop} dt={dollar_target}: trades={m['total_trades']} wr={m['win_rate']:.1%} "
              f"total=${m['total_dollars']:,.0f} "
              f"50k_payouts={sim_50k['total_payouts']} 150k_payouts={sim_150k['total_payouts']}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"results": results}, indent=2) + "\n")
    print(f"Saved {args.output}")

    # Best by total payout dollars
    best_50k = max(results, key=lambda r: r["sim_50k"]["total_payout_dollars"])
    best_150k = max(results, key=lambda r: r["sim_150k"]["total_payout_dollars"])
    fastest_50k = min([r for r in results if r["sim_50k"]["first_payout_days"] is not None],
                      key=lambda r: r["sim_50k"]["first_payout_days"])
    fastest_150k = min([r for r in results if r["sim_150k"]["first_payout_days"] is not None],
                       key=lambda r: r["sim_150k"]["first_payout_days"])
    print("\nBest total payout 50k:", best_50k["dollar_stop"], best_50k["dollar_target"],
          f"${best_50k['sim_50k']['total_payout_dollars']:,.0f}")
    print("Best total payout 150k:", best_150k["dollar_stop"], best_150k["dollar_target"],
          f"${best_150k['sim_150k']['total_payout_dollars']:,.0f}")
    print("Fastest first payout 50k:", fastest_50k["dollar_stop"], fastest_50k["dollar_target"],
          f"{fastest_50k['sim_50k']['first_payout_days']} days")
    print("Fastest first payout 150k:", fastest_150k["dollar_stop"], fastest_150k["dollar_target"],
          f"{fastest_150k['sim_150k']['first_payout_days']} days")


if __name__ == "__main__":
    main()
