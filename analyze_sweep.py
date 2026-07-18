#!/usr/bin/env python3
# CHANGE_SUMMARY
# 2026-07-18  kilo
#   - Aggregates entry timing sweep results into comparison tables.
# WHY: Quick analysis of which opening_window_sec improves PnL.
"""Analyze sweep results. Usage:
  python3 analyze_sweep.py sweep_tf_dema_*.json sweep_tf_vwap_*.json sweep_tf_holt_*.json
"""
from __future__ import annotations

import json
import os
import sys


def load_results(paths: list[str]) -> list[dict]:
    """Load all result files."""
    results = []
    for path in paths:
        if os.path.exists(path):
            with open(path) as fh:
                data = json.load(fh)
                if isinstance(data, list):
                    results.extend(data)
                else:
                    results.append(data)
    return results


def main():
    if len(sys.argv) < 2:
        # Try to find sweep files in current directory
        import glob
        paths = sorted(glob.glob("sweep_*.json") + glob.glob("fullres_*.json"))
        if not paths:
            print("Usage: python3 analyze_sweep.py <result_files...>")
            print("Or run from directory with sweep_*.json files")
            sys.exit(1)
    else:
        paths = sys.argv[1:]

    results = load_results(paths)
    if not results:
        print("No results found")
        sys.exit(1)

    # Group by strategy
    by_strategy = {}
    for r in results:
        strat = r["strategy"]
        by_strategy.setdefault(strat, []).append(r)

    # Print per-strategy tables
    for strat, runs in sorted(by_strategy.items()):
        print(f"\n{'='*70}")
        print(f"Strategy: {strat}")
        print(f"{'='*70}")
        print(f"{'Window':>8} {'PnL':>10} {'PnL%':>8} {'WinRate':>8} {'MaxDD':>8} "
              f"{'Trades':>7} {'Entry$':>8} {'Signals':>8} {'Time':>6}")
        print("-" * 75)

        baseline = None
        for r in sorted(runs, key=lambda x: x["opening_window_sec"]):
            w = r["opening_window_sec"]
            pnl = r["total_pnl"]
            pnl_pct = r.get("pnl_pct", 0)
            wr = r["win_rate"] * 100
            dd = r["max_dd_pct"]
            trades = r["n_trades"]
            entry = r["avg_entry_price"]
            signals = r.get("n_triggered", 0)
            t = r["runtime_s"]

            if w == 0:
                baseline = r

            # Delta from baseline
            delta = ""
            if baseline and w > 0:
                d_pnl = pnl - baseline["total_pnl"]
                delta = f" ({d_pnl:+.2f})"

            print(f"{w:>7.0f}s ${pnl:>9.2f}{delta:>10} {wr:>7.1f}% {dd:>7.1f}% "
                  f"{trades:>7} {entry:>8.4f} {signals:>8} {t:>5.0f}s")

        # Find best window
        if baseline:
            best = max(runs, key=lambda x: x["total_pnl"])
            if best["opening_window_sec"] != 0:
                print(f"\n  Best: {best['opening_window_sec']:.0f}s window "
                      f"(PnL ${best['total_pnl']:.2f}, "
                      f"delta {best['total_pnl'] - baseline['total_pnl']:+.2f})")
            else:
                print(f"\n  Best: baseline (0s) — no improvement from waiting")


if __name__ == "__main__":
    main()
