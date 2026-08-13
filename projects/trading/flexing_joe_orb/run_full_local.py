#!/usr/bin/env python3
"""Run the full 10-year Flexing Joe ORB backtest locally in parallel.

This single-pass runner loads each instrument once per variant, runs the full
backtest, then splits the trade list into chunk JSONs matching the format the
aggregator expects.  It avoids the heavy I/O of having every chunk reload the
entire 10-year CSV.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

FULL_START = "2016-06-01"
FULL_END = "2026-05-29"


def _chunk_date_range(full_start: str, full_end: str, chunk_id: int, total_chunks: int) -> Tuple[str, str]:
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


def _split_into_chunks(
    full_result: Dict[str, Any],
    instrument: str,
    variant: str,
    total_chunks: int,
    output_dir: Path,
) -> None:
    """Write chunk JSONs from a full backtest result."""
    parameters = full_result["parameters"]
    trades = full_result["trades"]
    signals = full_result.get("signals", [])

    # Group trades/signals by chunk date range.
    chunks: Dict[int, Dict[str, Any]] = {}
    for cid in range(total_chunks):
        cstart, cend = _chunk_date_range(FULL_START, FULL_END, cid, total_chunks)
        chunks[cid] = {
            "parameters": {**parameters, "variant": variant, "chunk_start": cstart, "chunk_end": cend},
            "trades": [],
            "signals": [],
            "daily_pnl": {},
            "execution_summary": full_result.get("execution_summary", {}),
            "metrics": {},
        }

    for t in trades:
        entry_date = t["entry_time"][:10]
        for cid in range(total_chunks):
            cstart, cend = _chunk_date_range(FULL_START, FULL_END, cid, total_chunks)
            if cstart <= entry_date <= cend:
                chunks[cid]["trades"].append(t)
                break

    for s in signals:
        ts_date = s["timestamp"][:10]
        for cid in range(total_chunks):
            cstart, cend = _chunk_date_range(FULL_START, FULL_END, cid, total_chunks)
            if cstart <= ts_date <= cend:
                chunks[cid]["signals"].append(s)
                break

    for cid, chunk in chunks.items():
        out_path = output_dir / f"{instrument}_chunk{cid}_{variant}.json"
        with open(out_path, "w") as f:
            json.dump(chunk, f, indent=2, default=str)


def _run_full_instrument_variant(
    args: Tuple[str, str, Path, Path, int]
) -> Tuple[str, str, str]:
    instrument, variant, data_root, output_dir, total_chunks = args
    flags = VARIANTS[variant]
    inst = INSTRUMENTS[instrument]

    cfg = StrategyConfig(
        symbol=instrument,
        data_path=str(data_root / f"{instrument}_1min.csv"),
        start_date=FULL_START,
        end_date=FULL_END,
        point_value=inst["point_value"],
        tick_size=inst["tick_size"],
        # Full run skips MC/bootstrap here; aggregator recomputes on combined series.
        mc_runs=0,
        bootstrap_samples=0,
        prop_mc_runs=0,
        prop_bootstrap_samples=0,
        **flags,
    )
    result = run_backtest(cfg)
    result["parameters"]["variant"] = variant

    _split_into_chunks(result, instrument, variant, total_chunks, output_dir)

    m = result["metrics"]
    return (
        instrument,
        variant,
        f"trades={m['total_trades']} pnl={m['net_pnl']:.0f} wr={m['win_rate']:.1f}%",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full Flexing Joe ORB backtest locally (single-pass)")
    parser.add_argument("--instruments", default=",".join(INSTRUMENTS), help="Comma-separated symbols")
    parser.add_argument("--total-chunks", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8, help="Parallel processes")
    parser.add_argument("--data-root", default="/config/projects/trading/v5_orb_nq_backtest/market_data")
    parser.add_argument("--output-dir", default="/tmp/fjo_full_results")
    parser.add_argument("--mc-runs", type=int, default=50_000)
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    instruments = [s.strip() for s in args.instruments.split(",") if s.strip()]
    tasks: List[Tuple[str, str, Path, Path, int]] = [
        (inst, variant, data_root, output_dir, args.total_chunks)
        for inst in instruments
        for variant in VARIANTS
    ]

    print(f"Running {len(tasks)} full backtests with {args.workers} workers...")
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_full_instrument_variant, t): t for t in tasks}
        for future in as_completed(futures):
            instrument, variant, summary = future.result()
            completed += 1
            print(f"[{completed}/{len(tasks)}] {instrument} {variant}: {summary}")

    print("\nAggregating results...")
    final_dir = output_dir / "final"
    agg_cmd = [
        sys.executable,
        "-m",
        "flexing_joe_orb.aggregate_results",
        "--results-dir",
        str(output_dir),
        "--output-dir",
        str(final_dir),
        "--mc-runs",
        str(args.mc_runs),
        "--bootstrap-samples",
        str(args.bootstrap_samples),
    ]
    subprocess.run(
        agg_cmd,
        cwd="/config/projects/trading",
        env={**dict(subprocess.os.environ), "PYTHONPATH": "/config/projects/trading"},
        check=True,
    )

    print(f"\nDone. Final reports in {final_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
