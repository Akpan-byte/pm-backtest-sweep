#!/usr/bin/env python3
"""
Run one date-chunk of the v5 ORB NQ backtest.

Used by the 20-worker GitHub Actions pipeline. Each worker receives a
[start_date, end_date] range, loads the NQ CSV, runs the backtest, and writes a
JSON result file.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import sys
from pathlib import Path

# Add project root to path for standalone script execution.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from v5_orb_nq_backtest.backtest import load_csv, run_backtest


def parse_date(s: str) -> date:
    return date.fromisoformat(s)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one v5 ORB NQ backtest chunk")
    parser.add_argument("--input", required=True, help="Path to NQ_1min.csv(.gz)")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--max-entries", type=int, default=2)
    parser.add_argument("--max-contracts", type=int, default=5)
    parser.add_argument("--baseline-index", type=float, default=None)
    args = parser.parse_args()

    start = parse_date(args.start_date)
    end = parse_date(args.end_date)

    df = load_csv(args.input)
    df = df[(df.index.date >= start) & (df.index.date <= end)]

    params = {"max_entries": args.max_entries, "max_contracts": args.max_contracts}
    if args.baseline_index is not None:
        params["baseline_index"] = args.baseline_index

    result = run_backtest(df, strategy_params=params)
    result["chunk"] = {"start": str(start), "end": str(end)}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"Chunk {start} to {end}: {result['metrics']['total_trades']} trades, "
          f"PnL ${result['metrics']['net_pnl']:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
