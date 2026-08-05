#!/usr/bin/env python3
"""
Aggregate v5 ORB NQ backtest chunk JSONs into a final report.

Combines trades, daily PnL, and metrics across all date chunks.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_chunk(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def aggregate(chunks: list[dict]) -> dict:
    all_trades = []
    day_results = []
    for c in sorted(chunks, key=lambda x: x.get("chunk", {}).get("start", "")):
        all_trades.extend(c.get("trades", []))
        day_results.extend(c.get("day_results", []))

    # Sort by date
    day_results = sorted(day_results, key=lambda x: x["date"])

    pnls = np.array([t["net"] for t in all_trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    n = len(all_trades)

    metrics = {
        "total_trades": int(n),
        "win_rate": round(len(wins) / n * 100, 2) if n else 0.0,
        "profit_factor": round(wins.sum() / abs(losses.sum()), 4) if losses.sum() < 0 else 999.0,
        "net_pnl": round(float(pnls.sum()), 2),
        "avg_pnl": round(float(pnls.mean()), 2) if n else 0.0,
        "std_pnl": round(float(pnls.std()), 2) if n else 0.0,
        "avg_win": round(float(wins.mean()), 2) if len(wins) else 0.0,
        "avg_loss": round(float(losses.mean()), 2) if len(losses) else 0.0,
        "best_trade": round(float(pnls.max()), 2) if n else 0.0,
        "worst_trade": round(float(pnls.min()), 2) if n else 0.0,
    }

    if day_results:
        metrics["start_equity"] = day_results[0]["equity"]
        metrics["final_equity"] = day_results[-1]["equity"]
        peak = day_results[0]["equity"]
        max_dd = 0.0
        for r in day_results:
            eq = r["equity"]
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > max_dd:
                max_dd = dd
        metrics["peak_equity"] = peak
        metrics["max_drawdown_dollars"] = round(max_dd, 2)
        metrics["max_drawdown_pct"] = round(max_dd / peak * 100, 4) if peak else 0.0
    else:
        metrics["start_equity"] = 0.0
        metrics["final_equity"] = 0.0
        metrics["peak_equity"] = 0.0
        metrics["max_drawdown_dollars"] = 0.0
        metrics["max_drawdown_pct"] = 0.0

    return {
        "metrics": metrics,
        "day_results": day_results,
        "trades": all_trades,
        "n_chunks": len(chunks),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate v5 ORB NQ backtest chunks")
    parser.add_argument("--glob", required=True, help="Glob pattern for chunk JSONs")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        print(f"No chunk files found for pattern: {args.glob}")
        return 1

    chunks = [load_chunk(f) for f in files]
    result = aggregate(chunks)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    m = result["metrics"]
    print(f"Aggregated {len(files)} chunks, {m['total_trades']} trades")
    print(f"Net PnL: ${m['net_pnl']:+.2f}, Win Rate: {m['win_rate']:.1f}%, PF: {m['profit_factor']:.2f}")
    print(f"Max DD: ${m['max_drawdown_dollars']:.2f} ({m['max_drawdown_pct']:.2f}%)")
    print(f"Saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
