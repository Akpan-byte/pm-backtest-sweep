#!/usr/bin/env python3
"""Sweep reentry caps + daily profit caps on combined main+counter strategy."""

import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/config/projects/trading")
sys.path.insert(0, "/config/projects/trading/v5_orb_nq_backtest")

from scripts.prop_farm_simulator_v4 import CostAssumptions, make_rules, simulate_prop_farm


DATA_ROOT = Path("/config/projects/trading/v5_orb_nq_backtest/market_data")
OUT_DIR = Path("/tmp/combined_reentry_sweep")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_backtest_task(task: tuple) -> Path:
    return run_backtest(*task)


def run_backtest(symbol: str, max_entries: int, contracts: int) -> Path:
    out = OUT_DIR / f"{symbol}_me{max_entries}_c{contracts}.json"
    cmd = [
        "python3", "/config/projects/trading/flexing_joe_orb/combined_main_counter_runner.py",
        "--symbols", symbol,
        "--data-root", str(DATA_ROOT),
        "--variant", "reentries",
        "--max-entries-per-day", str(max_entries),
        "--contracts-per-trade", str(contracts),
        "--prop-mc-runs", "0",
        "--prop-bootstrap-samples", "0",
        "--workers", "1",
        "--output", str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


def simulate(path: Path, daily_cap: float) -> dict:
    data = json.loads(path.read_text())
    symbol = list(data.keys())[0]
    result = data[symbol]
    df = pd.DataFrame(
        [{"date": pd.to_datetime(d).date(), "net_pnl": v} for d, v in result["daily_pnl"].items()]
    ).set_index("date").sort_index()
    if daily_cap is not None and daily_cap > 0:
        df["net_pnl"] = df["net_pnl"].clip(upper=daily_cap)

    costs = CostAssumptions()
    summary = {}
    for account_size in ["50K", "150K"]:
        for path_name in ["standard", "consistency"]:
            rules = make_rules(account_size)
            res = simulate_prop_farm(
                df, f"{symbol}_combined", rules, costs, path_name,
                reset_after_payout=False, desired_contracts=1,
            )
            s = res.summary()
            n_payouts = sum(len(a.payouts) for a in res.attempts)
            s["n_payouts"] = n_payouts
            s["payouts_per_month"] = round(n_payouts / (len(df) / 365.25 * 12), 2)
            summary[f"{account_size}_{path_name}"] = s
    return summary


def main() -> int:
    symbols = ["NQ"]  # focus on NQ; extend as needed
    max_entries_list = [2, 3, 4, 5, 6]
    contracts_list = [1, 2, 3]
    caps = [1000, 1500, 2000, 2500, 3000, 4000, 5000]

    tasks = [(s, me, c) for s in symbols for me in max_entries_list for c in contracts_list]

    print(f"Running {len(tasks)} backtest configurations...")
    with ProcessPoolExecutor(max_workers=3) as executor:
        paths = list(executor.map(run_backtest_task, tasks))

    print(f"Running {len(paths) * len(caps)} prop-farm simulations...")
    rows = []
    for path in paths:
        parts = path.stem.split("_")
        symbol = parts[0]
        max_entries = int(parts[1][2:])
        contracts = int(parts[2][1:])
        for cap in caps:
            sim = simulate(path, cap)
            row = {
                "symbol": symbol,
                "max_entries": max_entries,
                "contracts": contracts,
                "daily_cap": cap,
            }
            for key, s in sim.items():
                row[f"{key}_payouts_per_month"] = s["payouts_per_month"]
                row[f"{key}_net_profit"] = s["net_profit"]
                row[f"{key}_n_payouts"] = s["n_payouts"]
                row[f"{key}_n_xfa_blows"] = s["n_xfa_blows"]
                row[f"{key}_n_combine_blows"] = s["n_combine_blows"]
            rows.append(row)

    out_json = OUT_DIR / "sweep_summary.json"
    out_json.write_text(json.dumps(rows, indent=2, default=str))
    print(f"Saved sweep summary to {out_json}")

    # Print configs closest to target averages
    target_std = 3.1
    target_con = 3.98
    best = min(rows, key=lambda r: abs(r["50K_standard_payouts_per_month"] - target_std))
    best_con = min(rows, key=lambda r: abs(r["50K_consistency_payouts_per_month"] - target_con))
    print("\nClosest to 50K standard target (3.1/mo):")
    print(json.dumps(best, indent=2))
    print("\nClosest to 50K consistency target (3.98/mo):")
    print(json.dumps(best_con, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
