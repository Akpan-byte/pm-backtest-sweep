#!/usr/bin/env python3
"""
Run the full 5 ORB NQ backtest locally using at most 3 cores.

Splits the 10-year NQ dataset into chunks and runs them in parallel with
multiprocessing.Pool(max_workers=3), then aggregates the results.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


def date_range_chunks(start: date, end: date, n_chunks: int) -> list[tuple[date, date]]:
    total_days = (end - start).days + 1
    chunk_size = max(1, total_days // n_chunks)
    chunks = []
    current = start
    for i in range(n_chunks):
        if i == n_chunks - 1:
            chunk_end = end
        else:
            chunk_end = min(current + timedelta(days=chunk_size - 1), end)
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)
        if current > end:
            break
    return chunks


def run_chunk(args: tuple[int, tuple[date, date], str, int, int, Path]) -> str:
    chunk_id, (start, end), input_path, max_entries, max_contracts, results_dir = args
    output_path = results_dir / f"chunk_{chunk_id:02d}.json"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_chunk.py"),
        "--input", input_path,
        "--start-date", str(start),
        "--end-date", str(end),
        "--output", str(output_path),
        "--max-entries", str(max_entries),
        "--max-contracts", str(max_contracts),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    subprocess.run(cmd, env=env, check=True)
    return str(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Local 3-core v5 ORB NQ backtest")
    parser.add_argument("--input", required=True, help="Path to NQ_1min.csv(.gz)")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--max-entries", type=int, default=2)
    parser.add_argument("--max-contracts", type=int, default=5)
    parser.add_argument("--n-chunks", type=int, default=20)
    parser.add_argument("--workers", type=int, default=3, help="Max local workers (default 3)")
    args = parser.parse_args()

    results_dir = Path(args.output).parent / "chunks"
    results_dir.mkdir(parents=True, exist_ok=True)

    chunks = date_range_chunks(date(2016, 6, 1), date(2026, 5, 29), args.n_chunks)
    tasks = [
        (i, chunk, args.input, args.max_entries, args.max_contracts, results_dir)
        for i, chunk in enumerate(chunks)
    ]

    print(f"Running {len(chunks)} chunks with {args.workers} local workers...")
    with multiprocessing.Pool(processes=args.workers) as pool:
        pool.map(run_chunk, tasks)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "aggregate.py"),
        "--glob", str(results_dir / "chunk_*.json"),
        "--output", args.output,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    subprocess.run(cmd, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
