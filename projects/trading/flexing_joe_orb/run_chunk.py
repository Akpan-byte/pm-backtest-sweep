#!/usr/bin/env python3
"""Run one date-range chunk of the Flexing Joe ORB backtest.

Used by the GitHub Actions matrix to split the 10-year dataset into
``total_chunks`` parallel jobs.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

from flexing_joe_orb.backtest import run_backtest
from flexing_joe_orb.models import StrategyConfig


INSTRUMENTS = {
    "NQ": {"point_value": 20.0, "tick_size": 0.25},
    "ES": {"point_value": 50.0, "tick_size": 0.25},
    "YM": {"point_value": 5.0, "tick_size": 1.0},
}

VARIANTS = {
    "one_trade_per_day": {"one_trade_per_day": True, "one_trade_per_direction": False},
    "one_per_direction": {"one_trade_per_day": False, "one_trade_per_direction": True},
    "reentries": {"one_trade_per_day": False, "one_trade_per_direction": False},
}


def chunk_date_range(
    full_start: str, full_end: str, chunk_id: int, total_chunks: int
) -> Tuple[str, str]:
    """Return inclusive (start, end) date strings for a chunk."""
    start_dt = datetime.strptime(full_start, "%Y-%m-%d").date()
    end_dt = datetime.strptime(full_end, "%Y-%m-%d").date()
    total_days = (end_dt - start_dt).days + 1
    chunk_size = max(1, total_days // total_chunks)

    chunk_start = start_dt + timedelta(days=chunk_id * chunk_size)
    if chunk_id == total_chunks - 1:
        chunk_end = end_dt
    else:
        chunk_end = chunk_start + timedelta(days=chunk_size - 1)

    return chunk_start.isoformat(), chunk_end.isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(description="Flexing Joe ORB chunk runner")
    parser.add_argument("--instrument", required=True, choices=list(INSTRUMENTS))
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--total-chunks", type=int, default=20)
    parser.add_argument("--data-root", default="/config/projects/trading/v5_orb_nq_backtest/market_data")
    parser.add_argument("--full-start", default="2016-06-01")
    parser.add_argument("--full-end", default="2026-05-29")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    start_date, end_date = chunk_date_range(
        args.full_start, args.full_end, args.chunk_id, args.total_chunks
    )

    data_path = Path(args.data_root) / f"{args.instrument}_1min.csv"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inst = INSTRUMENTS[args.instrument]

    for variant_name, flags in VARIANTS.items():
        cfg = StrategyConfig(
            symbol=args.instrument,
            data_path=str(data_path),
            start_date=start_date,
            end_date=end_date,
            point_value=inst["point_value"],
            tick_size=inst["tick_size"],
            # Chunks skip expensive MC/bootstrap; the aggregator recomputes them
            # on the combined 10-year trade series.
            mc_runs=0,
            bootstrap_samples=0,
            **flags,
        )
        result = run_backtest(cfg)
        result["parameters"]["variant"] = variant_name

        out_path = out_dir / f"{args.instrument}_chunk{args.chunk_id}_{variant_name}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        m = result["metrics"]
        print(
            f"{args.instrument} chunk {args.chunk_id} {variant_name}: "
            f"trades={m['total_trades']} pnl={m['net_pnl']:.0f} wr={m['win_rate']:.1f}%"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
