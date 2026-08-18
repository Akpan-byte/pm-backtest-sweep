# CHANGE_SUMMARY
# 2026-08-18  kilo
#   - Created strategies/backtest/dmc_merge_sweep.py: merges per-worker DMC sweep
#     artifacts into a single summary CSV and prints top configs by win rate and
#     by a prop-payout score.
# WHY: Final reporting step for the distributed DMC parameter sweep.

"""Merge DMC sweep worker artifacts into a summary report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="directory containing worker result files")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--top-n", type=int, default=20)
    args = ap.parse_args()

    results_dir = Path(args.results)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for metrics_path in sorted(results_dir.glob("*_metrics.json")):
        tag = metrics_path.stem.replace("_metrics", "")
        params_path = metrics_path.with_name(f"{tag}_params.json")
        if not params_path.exists():
            continue
        with open(metrics_path) as fh:
            metrics = json.load(fh)
        with open(params_path) as fh:
            params = json.load(fh)
        basic = metrics["basic"]
        symbol = tag.split("_")[0]
        rows.append({
            "tag": tag,
            "symbol": symbol,
            "config_id": tag.split("_")[-1],
            "params": params,
            "n_trades": basic["n_trades"],
            "win_rate": basic["win_rate"],
            "cagr": basic["cagr"],
            "sharpe": basic["sharpe_ratio"],
            "maxDD": basic["max_drawdown"],
            "total_ret": basic["total_return"],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No results found.")
        return

    df["score"] = (
        df["win_rate"].clip(lower=0) * 2.0
        + df["total_ret"].clip(lower=-1, upper=2)
        - df["maxDD"].abs() * 2.0
        + df["sharpe"].clip(lower=-2, upper=2) * 0.5
    )

    summary_path = outdir / "summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"Merged {len(df)} results -> {summary_path}")

    # Per-symbol top by score.
    for sym in sorted(df["symbol"].unique()):
        sub = df[df["symbol"] == sym].sort_values("score", ascending=False).head(args.top_n)
        print(f"\n=== {sym} top {args.top_n} by prop-payout score ===")
        print(sub[["tag", "n_trades", "win_rate", "cagr", "sharpe", "maxDD", "total_ret", "score"]].to_string(index=False))

    # Overall top by win rate.
    print(f"\n=== Top {args.top_n} by win rate ===")
    top_wr = df.sort_values("win_rate", ascending=False).head(args.top_n)
    print(top_wr[["tag", "n_trades", "win_rate", "cagr", "sharpe", "maxDD", "total_ret", "score"]].to_string(index=False))

    # Configs with >= 90% win rate and positive return.
    high_wr = df[(df["win_rate"] >= 0.90) & (df["total_ret"] > 0)].sort_values("total_ret", ascending=False)
    if not high_wr.empty:
        print(f"\n=== Configs with WR >= 90% and positive return ({len(high_wr)}) ===")
        print(high_wr[["tag", "n_trades", "win_rate", "cagr", "sharpe", "maxDD", "total_ret"]].to_string(index=False))
    else:
        print("\nNo configs achieved >= 90% win rate with positive return on IS.")


if __name__ == "__main__":
    main()
